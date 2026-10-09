#!/usr/bin/env python3
"""Exactly-once SD302g hull scoring for an authorized cross-domain seed.

All masks, probability maps, timings, selectors and source identities are
authenticated before the score-once journal is created.  Only then is
``evaluate_hull.py`` invoked; it is the sole step permitted to open the expert
minutiae/hull artifacts.  A retained journal without a completed bundle blocks
automatic retry so a crash cannot silently score the held-out endpoint twice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

import _path  # noqa: F401
from segbench.artifact_identity import canonical_sha256, sha256_file
from segbench.b1_crossdomain import (
    AUTHORIZATION_KIND,
    AUTHORIZATION_SCHEMA_VERSION,
    PROTOCOL,
)


HEX40 = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_METRIC_FRAGMENTS = ("dice", "jaccard", "fdr", "miou", "mdr")
REFERENCE_METHODS = {"all_background", "all_foreground", "hull_as_prediction"}


class FinalizationError(ValueError):
    """A B1 artifact changed or an exactly-once gate failed."""


def read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON object")
    return value


def read_csv(path: Path, label: str) -> tuple[tuple[str, ...], list[Dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise FinalizationError(f"{label} is not readable: {path}") from exc
    if not columns or not rows:
        raise FinalizationError(f"{label} is empty")
    return columns, rows


def outside_repository(path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        path.resolve().relative_to(repository)
    except ValueError:
        return
    raise FinalizationError(f"restricted B1 output must be outside repository: {path}")


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def artifact_paths(authorization: Mapping[str, Any]) -> Dict[str, Path]:
    artifacts = authorization.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FinalizationError("authorization lacks artifact identities")
    resolved: Dict[str, Path] = {}
    for name, identity in artifacts.items():
        if not isinstance(identity, dict):
            raise FinalizationError(f"authorization artifact {name!r} is malformed")
        path = Path(str(identity.get("path", ""))).resolve()
        if not path.is_file() or path.is_symlink():
            raise FinalizationError(f"authorized artifact is missing or a symlink: {name}")
        if sha256_file(path) != identity.get("sha256"):
            raise FinalizationError(f"authorized artifact bytes changed: {name}")
        resolved[str(name)] = path
    return resolved


def expected_test_rows(
    verified_manifest: Path, expected_count: int
) -> tuple[list[Dict[str, str]], list[str]]:
    _, rows = read_csv(verified_manifest, "verified B1 manifest")
    test = [row for row in rows if row.get("split") == "test"]
    ids = sorted(row.get("image_id", "") for row in test)
    if len(ids) != expected_count or len(set(ids)) != expected_count or any(not item for item in ids):
        raise FinalizationError("verified B1 test identity universe changed")
    return test, ids


def validate_complete_inference(
    authorization_path: Path,
    authorization: Mapping[str, Any],
    artifacts: Mapping[str, Path],
) -> Mapping[str, Any]:
    model = authorization.get("model")
    target = authorization.get("target_domain")
    outputs = authorization.get("outputs")
    execution = authorization.get("execution")
    if not all(isinstance(item, dict) for item in (model, target, outputs, execution)):
        raise FinalizationError("authorization lacks model/target/output/execution bindings")
    assert isinstance(model, dict) and isinstance(target, dict)
    assert isinstance(outputs, dict) and isinstance(execution, dict)
    card_path = Path(str(outputs["inference_card"])).resolve()
    prediction_root = Path(str(outputs["predictions"])).resolve()
    probability_root = Path(str(outputs["probabilities"])).resolve()
    if not card_path.is_file() or card_path.is_symlink():
        raise FinalizationError("complete inference card is missing or a symlink")
    if not prediction_root.is_dir() or not probability_root.is_dir():
        raise FinalizationError("retained prediction/probability directory is missing")
    card = read_json(card_path, "complete B1 inference card")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "complete": True,
        "kind": "complete_probability_map_card",
        "method": model["method"],
        "task": model["task"],
        "seed": model["seed"],
        "subset": "finger_exemplar",
        "split": "test",
        "selected_validation_threshold": model["selected_threshold"],
        "threshold_receipt_sha256": model["selector_receipt_sha256"],
        "threshold_amendment_sha256": model["threshold_amendment_sha256"],
        "cross_domain_authorization_sha256": sha256_file(authorization_path),
        "source_config_sha256": model["source_config_sha256"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "manifest_sha256": target["manifest_sha256"],
        "split_sha256": target["split_sha256"],
        "labels_opened": False,
        "probabilities_are_pre_threshold": True,
        "derived_from_binary_mask": False,
        "binary_mask_reinterpreted": False,
        "probability_maps_required": True,
        "debug_limited": False,
        "preempted": False,
        "target_ppi": 500.0,
    }
    disagreements = {
        key: (card.get(key), wanted)
        for key, wanted in expected.items()
        if card.get(key) != wanted
    }
    if disagreements:
        raise FinalizationError(f"complete B1 inference card changed: {disagreements}")
    provenance = card.get("provenance")
    expected_commit = execution.get("expected_inference_commit")
    if (
        not isinstance(provenance, dict)
        or not isinstance(expected_commit, str)
        or not HEX40.fullmatch(expected_commit)
        or provenance.get("capture_timing") != "process_start"
        or provenance.get("git_commit") != expected_commit
        or provenance.get("git_dirty") is not False
    ):
        raise FinalizationError("B1 inference lacks clean pinned process-start provenance")

    test_rows, ids = expected_test_rows(
        artifacts["b1_verified_manifest"], int(target["test_records"])
    )
    if canonical_sha256(ids) != target.get("test_image_ids_sha256"):
        raise FinalizationError("authorized B1 test identity digest changed")
    masks = sorted(prediction_root.glob("*.png"))
    probabilities = sorted(probability_root.glob("*.npy"))
    if any(path.is_symlink() for path in (*masks, *probabilities)):
        raise FinalizationError("B1 inference output contains a symlink")
    if [path.stem for path in masks] != ids or [path.stem for path in probabilities] != ids:
        raise FinalizationError("B1 retained output universe differs from test subjects")
    timing = card.get("per_image")
    if not isinstance(timing, list) or len(timing) != len(ids):
        raise FinalizationError("B1 timing records are incomplete")
    timing_by_id = {
        str(item.get("image_id")): item for item in timing if isinstance(item, dict)
    }
    if sorted(timing_by_id) != ids:
        raise FinalizationError("B1 timing identity universe differs")
    _, inference_rows = read_csv(
        artifacts["b1_inference_manifest"], "B1 inference manifest"
    )
    source_by_id = {
        row["image_id"]: row.get("source_sha256", "")
        for row in inference_rows
        if row.get("image_id") in set(ids)
    }
    if sorted(source_by_id) != ids:
        raise FinalizationError("B1 inference manifest omits a test source identity")
    inference_by_id = {row["image_id"]: row for row in inference_rows}
    for image_id in ids:
        source = Path(inference_by_id[image_id].get("image_path", "")).resolve()
        if not source.is_file() or source.is_symlink():
            raise FinalizationError(f"B1 source image is missing or a symlink: {image_id}")
        if sha256_file(source) != source_by_id[image_id]:
            raise FinalizationError(f"B1 source image bytes changed: {image_id}")
    probability_records = []
    mask_records = []
    for image_id, mask_path, probability_path in zip(ids, masks, probabilities):
        native = next(row for row in test_rows if row["image_id"] == image_id)
        expected_shape = (
            max(1, round(int(native["height"]) * 500.0 / float(native["ppi"]))),
            max(1, round(int(native["width"]) * 500.0 / float(native["ppi"]))),
        )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if (
            mask is None or mask.ndim != 2 or mask.dtype != np.uint8
            or mask.shape != expected_shape
            or not set(np.unique(mask).tolist()).issubset({0, 255})
        ):
            raise FinalizationError(f"invalid retained B1 mask: {image_id}")
        try:
            probability = np.load(probability_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise FinalizationError(f"invalid retained B1 probability: {image_id}") from exc
        if probability.dtype != np.float16 or probability.ndim != 2 or not np.isfinite(probability).all():
            raise FinalizationError(f"invalid retained B1 probability: {image_id}")
        timing_row = timing_by_id[image_id]
        mask_hash = sha256_file(mask_path)
        probability_hash = sha256_file(probability_path)
        if timing_row.get("mask_sha256") != mask_hash or timing_row.get("probability_sha256") != probability_hash:
            raise FinalizationError(f"B1 retained output changed after timing: {image_id}")
        elapsed = timing_row.get("ms_per_img")
        if type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
            raise FinalizationError(f"invalid B1 timing: {image_id}")
        mask_records.append({"image_id": image_id, "sha256": mask_hash, "bytes": mask_path.stat().st_size})
        probability_records.append(
            {
                "image_id": image_id,
                "sha256": probability_hash,
                "bytes": probability_path.stat().st_size,
                "source_sha256": source_by_id[image_id],
            }
        )
    probability_set = card.get("probability_set")
    if (
        not isinstance(probability_set, dict)
        or probability_set.get("complete") is not True
        or probability_set.get("count") != len(ids)
        or probability_set.get("sha256")
        != canonical_sha256(
            [
                {key: record[key] for key in ("image_id", "sha256", "bytes")}
                for record in probability_records
            ]
        )
    ):
        raise FinalizationError("B1 probability-set identity changed")
    if probability_set.get("ordered_source_images_sha256") != target.get(
        "ordered_source_images_sha256"
    ):
        raise FinalizationError("B1 probability sources differ from authorized test inputs")
    derivation = card.get("mask_derivation_verification")
    if (
        not isinstance(derivation, dict)
        or derivation.get("complete") is not True
        or derivation.get("count") != len(ids)
        or derivation.get("all_masks_rederived_from_retained_probabilities") is not True
    ):
        raise FinalizationError("B1 masks are not authenticated probability derivations")
    mean_ms = card.get("mean_ms_per_img")
    if type(mean_ms) not in (int, float) or not math.isfinite(float(mean_ms)) or float(mean_ms) < 0:
        raise FinalizationError("B1 mean timing is invalid")
    if card.get("device") not in {"CPU", "GPU", "CUDA"}:
        raise FinalizationError("B1 timing card has an invalid device label")
    return {
        "card": card,
        "card_sha256": sha256_file(card_path),
        "prediction_count": len(ids),
        "prediction_set_sha256": canonical_sha256(mask_records),
        "prediction_records_sha256": canonical_sha256(mask_records),
        "probability_set_sha256": probability_set["sha256"],
        "mean_ms_per_img": float(mean_ms),
        "device": card["device"],
    }


def invoke_evaluator(
    python: Path,
    evaluator: Path,
    manifest: Path,
    predictions: Path,
    method_key: str,
    display_name: str,
    staging: Path,
) -> Mapping[str, str]:
    command = [
        str(python), str(evaluator), "--manifest", str(manifest),
        "--pred-template", str(predictions / "{image_id}.png"),
        "--method", method_key, "--display-name", display_name,
        "--out", str(staging), "--split", "test", "--pred-ppi", "500",
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        raise FinalizationError(
            f"evaluate_hull failed ({process.returncode}): {process.stderr[-1000:]}"
        )
    return {"stdout": process.stdout, "stderr": process.stderr, "argv": command}


def validate_evaluation(staging: Path, method_key: str, expected_count: int) -> Mapping[str, Any]:
    per_path = staging / f"per_image_{method_key}_test.csv"
    summary_path = staging / f"summary_{method_key}_test.csv"
    per_columns, per_rows = read_csv(per_path, "B1 per-image evaluation")
    summary_columns, summary_rows = read_csv(summary_path, "B1 summary evaluation")
    if any(fragment in column.lower() for fragment in FORBIDDEN_METRIC_FRAGMENTS for column in (*per_columns, *summary_columns)):
        raise FinalizationError("B1 evaluator emitted a forbidden full-mask metric")
    methods = {method_key, *REFERENCE_METHODS}
    if len(per_rows) != expected_count * len(methods) or {row["method"] for row in per_rows} != methods:
        raise FinalizationError("B1 evaluator output universe is incomplete")
    method_rows = [row for row in per_rows if row["method"] == method_key]
    if len(method_rows) != expected_count or any(int(row["missing"]) for row in method_rows):
        raise FinalizationError("B1 evaluator scored a missing prediction")
    hull_rows = [row for row in per_rows if row["method"] == "hull_as_prediction"]
    if len(hull_rows) != expected_count or any(
        float(row["mc"]) != 1.0 or float(row["hr"]) != 1.0 for row in hull_rows
    ):
        raise FinalizationError("B1 hull-as-prediction 100/100 gate failed")
    overall = [
        row for row in summary_rows
        if row["method"] == method_key and row["aggregate"] == "overall" and row["value"] == "all"
    ]
    if len(overall) != 1:
        raise FinalizationError("B1 overall summary is missing")
    breakdowns = {row["aggregate"] for row in summary_rows if row["method"] == method_key}
    if not {"overall", "group", "device", "capture"}.issubset(breakdowns):
        raise FinalizationError("B1 per-device/per-capture aggregation is incomplete")
    row = overall[0]
    for metric in ("mc", "hr", "par", "ohf"):
        value = float(row[f"{metric}_mean"])
        if not math.isfinite(value) or value < 0 or (metric != "par" and value > 1):
            raise FinalizationError(f"B1 summary contains invalid {metric}")
    return {
        "per_image_sha256": sha256_file(per_path),
        "summary_sha256": sha256_file(summary_path),
        "n_images": int(row["n_images"]),
        "n_subjects": int(row["n_subjects"]),
        "n_missing": int(row["n_missing"]),
        **{
            key: float(row[key])
            for key in (
                "mc_mean", "mc_ci_low", "mc_ci_high", "hr_mean", "hr_ci_low",
                "hr_ci_high", "par_mean", "ohf_mean",
            )
        },
        "breakdowns": sorted(breakdowns),
        "hull_reference_gate": {"mc": 1.0, "hr": 1.0, "passed": True},
    }


def parameter_total(value: object) -> float:
    total = value.get("total") if isinstance(value, dict) else value
    if type(total) not in (int, float) or not math.isfinite(float(total)) or float(total) < 0:
        raise FinalizationError("authorized training run lacks a valid parameter count")
    return float(total)


def validate_existing_bundle(
    out: Path,
    journal: Path,
    authorization_path: Path,
    authorization: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate an already completed transaction without rescoring it."""
    receipt = read_json(out / "adoption_receipt.json", "existing B1 adoption receipt")
    if (
        receipt.get("kind") != "sd302g_b1_reference_verified_adoption"
        or receipt.get("authorization_sha256") != sha256_file(authorization_path)
        or receipt.get("score_once_journal_sha256")
        != (sha256_file(journal) if journal.is_file() else None)
        or receipt.get("method") != authorization.get("method_key")
        or receipt.get("frozen_prediction", {}).get("run_card_sha256")
        != frozen.get("card_sha256")
        or receipt.get("frozen_prediction", {}).get("prediction_set_sha256")
        != frozen.get("prediction_set_sha256")
        or receipt.get("frozen_prediction", {}).get("probability_set", {}).get("sha256")
        != frozen.get("probability_set_sha256")
    ):
        raise FinalizationError("existing B1 bundle identity changed")
    target = authorization["target_domain"]
    observed = validate_evaluation(
        out, str(authorization["method_key"]), int(target["test_records"])
    )
    recorded = receipt.get("evaluation")
    if not isinstance(recorded, dict) or any(
        recorded.get(key) != value for key, value in observed.items()
    ):
        raise FinalizationError("existing B1 evaluation bytes or summary changed")
    return receipt


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    authorization_path = args.authorization.resolve()
    authorization = read_json(authorization_path, "B1 cross-domain authorization")
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION
        or authorization.get("kind") != AUTHORIZATION_KIND
        or authorization.get("protocol") != PROTOCOL
        or authorization.get("status") != "AUTHORIZED"
    ):
        raise FinalizationError("B1 cross-domain authorization is unsupported")
    artifacts = artifact_paths(authorization)
    outputs = authorization.get("outputs")
    target = authorization.get("target_domain")
    model = authorization.get("model")
    if not all(isinstance(item, dict) for item in (outputs, target, model)):
        raise FinalizationError("B1 authorization is incomplete")
    assert isinstance(outputs, dict) and isinstance(target, dict) and isinstance(model, dict)
    out = Path(str(outputs["score_out"])).resolve()
    journal = Path(str(outputs["score_journal"])).resolve()
    outside_repository(out)
    outside_repository(journal)
    frozen = validate_complete_inference(authorization_path, authorization, artifacts)
    if out.exists():
        receipt = validate_existing_bundle(
            out, journal, authorization_path, authorization, frozen
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return receipt
    if journal.exists():
        raise FinalizationError(
            "score-once journal already exists without a completed bundle; automatic retry is forbidden"
        )
    predictions = Path(str(outputs["predictions"])).resolve()
    journal_payload = {
        "schema_version": 1,
        "kind": "sd302g_b1_score_once_journal",
        "status": "ARMED",
        "authorization_sha256": sha256_file(authorization_path),
        "inference_card_sha256": frozen["card_sha256"],
        "prediction_set_sha256": frozen["prediction_set_sha256"],
        "probability_set_sha256": frozen["probability_set_sha256"],
        "verified_manifest_sha256": target["verified_manifest_sha256"],
        "split_sha256": target["split_sha256"],
        "evaluator_sha256": sha256_file(artifacts["evaluator"]),
        "method_key": authorization["method_key"],
        "metrics": ["mc", "hr", "par", "ohf"],
        "forbidden_metrics": ["dice", "jaccard", "fdr"],
        "test_labels_opened_before_journal": False,
    }
    write_json_exclusive(journal, journal_payload)
    journal_hash = sha256_file(journal)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.scoring-", dir=out.parent))
    # Deliberately retain staging and the journal on failure for audit/remediation.
    log = invoke_evaluator(
        Path(sys.executable), artifacts["evaluator"], artifacts["b1_verified_manifest"],
        predictions, str(authorization["method_key"]), str(authorization["display_name"]), staging,
    )
    evaluation = validate_evaluation(
        staging, str(authorization["method_key"]), int(target["test_records"])
    )
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "sd302g_b1_reference_verified_adoption",
        "protocol": PROTOCOL,
        "status": target["verification_status"],
        "method": authorization["method_key"],
        "display_name": authorization["display_name"],
        "seed": model["seed"],
        "table_method": model["method"],
        "authorization_sha256": sha256_file(authorization_path),
        "score_once_journal_sha256": journal_hash,
        "no_prediction_regeneration": True,
        "test_labels_opened_only_by_evaluate_hull": True,
        "no_supervised_sd302g_fitting": True,
        "identities": {
            "reference_gate_sha256": sha256_file(artifacts["reference_gate"]),
            "reference_report_sha256": sha256_file(artifacts["reference_report"]),
            "verified_provenance_sha256": target["verified_provenance_sha256"],
            "verified_manifest_sha256": target["verified_manifest_sha256"],
            "legacy_manifest_sha256": target["verified_manifest_sha256"],
            "split_sha256": target["split_sha256"],
            "inference_manifest_sha256": target["manifest_sha256"],
            "inference_receipt_sha256": target["manifest_receipt_sha256"],
            "run_card_sha256": frozen["card_sha256"],
            "evaluate_hull_sha256": sha256_file(artifacts["evaluator"]),
        },
        "inference": {
            "rows": target["records"],
            "test_rows": target["test_records"],
            "test_subjects": target["test_subjects"],
            "mask_paths_empty": True,
            "retained_masks": True,
            "retained_probabilities": True,
            "retained_timing": True,
        },
        "frozen_prediction": {
            "method_key": authorization["method_key"],
            "run_method": model["method"],
            "run_card_sha256": frozen["card_sha256"],
            "prediction_count": frozen["prediction_count"],
            "prediction_set_sha256": frozen["prediction_set_sha256"],
            "prediction_records_sha256": frozen["prediction_records_sha256"],
            "probability_set": {
                "count": frozen["prediction_count"],
                "sha256": frozen["probability_set_sha256"],
            },
        },
        "run": {
            "process_start_commit": authorization["execution"]["expected_inference_commit"],
            "process_start_clean": True,
            "labels_opened": False,
            "params_m": parameter_total(model["parameter_count"]) / 1_000_000.0,
            "ms_per_img": frozen["mean_ms_per_img"],
            "device": "GPU" if frozen["device"] == "CUDA" else frozen["device"],
        },
        "evaluation": evaluation,
        "producer": {
            "script_sha256": sha256_file(Path(__file__)),
            "evaluator_stdout_sha256": hashlib.sha256(log["stdout"].encode()).hexdigest(),
            "evaluator_stderr_sha256": hashlib.sha256(log["stderr"].encode()).hexdigest(),
            "evaluator_argv_sha256": canonical_sha256(log["argv"]),
        },
    }
    write_json_exclusive(staging / "adoption_receipt.json", receipt)
    os.replace(staging, out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        run(parse_args(argv))
    except (FinalizationError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
