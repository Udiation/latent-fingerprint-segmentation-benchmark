#!/usr/bin/env python3
"""Authorize one final Dataset-1 seed for ground-truth-blind SD302g inference.

This metadata-only barrier binds the frozen training run and Dataset-1
validation selector to a separately authenticated exemplar manifest.  It does
not decode an exemplar, open a hull/minutiae artifact, launch inference, or
compute a metric.  The emitted plan contains exact inference and score-once
commands for a later guarded queue.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import _path  # noqa: F401
from segbench.artifact_identity import canonical_sha256, sha256_file
from segbench import probability_artifacts
from segbench.b1_crossdomain import (
    AUTHORIZATION_KIND,
    AUTHORIZATION_SCHEMA_VERSION,
    PROTOCOL,
)


PRODUCTION_TOTAL = 2_380
PRODUCTION_TEST = 475
PRODUCTION_TEST_SUBJECTS = 40
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_METHOD = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class PreparationError(ValueError):
    """A frozen selection or target-domain identity failed authentication."""


def read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{label} must be a JSON object")
    return value


def read_csv(path: Path, label: str) -> tuple[tuple[str, ...], list[Dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise PreparationError(f"{label} is not readable: {path}") from exc
    if not columns or not rows:
        raise PreparationError(f"{label} is empty")
    return columns, rows


def outside_repository(path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        path.resolve().relative_to(repository)
    except ValueError:
        return
    raise PreparationError(f"restricted B1 artifact must be outside the repository: {path}")


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Publish immutable JSON, accepting only a byte-semantic replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = read_json(path, "existing immutable output")
        if canonical_sha256(observed) != canonical_sha256(value):
            raise PreparationError(f"refusing to replace changed output: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def identity(path: Path) -> Dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def validate_process_provenance(value: object, expected_commit: str, label: str) -> None:
    if not isinstance(value, dict):
        raise PreparationError(f"{label} lacks process-start provenance")
    expected = {
        "capture_timing": "process_start",
        "git_commit": expected_commit,
        "git_dirty": False,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise PreparationError(
                f"{label} provenance {key}={value.get(key)!r}; expected {wanted!r}"
            )


def validate_split(path: Path) -> Mapping[str, str]:
    payload = read_json(path, "subject split")
    mapping = payload.get("splits")
    if not isinstance(mapping, dict) or not mapping:
        raise PreparationError("subject split lacks a non-empty mapping")
    normalized = {str(key): str(value) for key, value in mapping.items()}
    if set(normalized.values()) - {"train", "val", "test"}:
        raise PreparationError("subject split contains an invalid assignment")
    if len(normalized) != 200 or Counter(normalized.values()) != {
        "train": 136,
        "val": 24,
        "test": 40,
    }:
        raise PreparationError("production subject split is not 136/24/40 over 200 subjects")
    return normalized


def validate_training_and_selector(args: argparse.Namespace) -> Mapping[str, Any]:
    run = read_json(args.training_run_card, "training run card")
    selector = read_json(args.selector_receipt, "validation selector receipt")
    config_hash = sha256_file(args.config)
    checkpoint_hash = sha256_file(args.checkpoint)
    source_manifest_hash = sha256_file(args.dataset1_manifest)
    split_hash = sha256_file(args.splits)
    amendment_hash = sha256_file(args.threshold_amendment)
    method = str(run.get("method", ""))
    task = str(run.get("task", ""))
    seed = run.get("seed")
    if not SAFE_METHOD.fullmatch(method) or task not in {"pixel", "patch"}:
        raise PreparationError("training run method/task is unsupported")
    if type(seed) is not int or seed not in {0, 1, 2}:
        raise PreparationError("training run seed is not one of 0, 1, 2")
    if run.get("debug_limited") is not False:
        raise PreparationError("debug/limited training runs are not B1 eligible")
    expected_run = {
        "source_config_sha256": config_hash,
        "manifest_sha256": source_manifest_hash,
        "split_sha256": split_hash,
        "threshold_amendment_sha256": amendment_hash,
    }
    for key, wanted in expected_run.items():
        if run.get(key) != wanted:
            raise PreparationError(f"training run {key} does not match its frozen input")
    checkpoints = run.get("checkpoint_sha256")
    if not isinstance(checkpoints, dict) or checkpoints.get("best.pt") != checkpoint_hash:
        raise PreparationError("training run does not bind the supplied best checkpoint")
    if any(key in run for key in ("target_manifest_sha256", "target_subsets")):
        raise PreparationError(
            "cross-domain B1 row must not come from supervised or aligned SD302g fitting"
        )
    provenance = run.get("provenance")
    if not isinstance(provenance, dict):
        raise PreparationError("training run lacks process-start provenance")
    training_commit = str(provenance.get("git_commit", ""))
    if not HEX40.fullmatch(training_commit):
        raise PreparationError("training run commit is not immutable")
    validate_process_provenance(provenance, training_commit, "training run")

    expected_selector = {
        "schema_version": 1,
        "kind": "expanded_validation_threshold_selector_receipt",
        "method": method,
        "task": task,
        "seed": seed,
        "selection_split": "val",
        "checkpoint_sha256": checkpoint_hash,
        "source_config_sha256": config_hash,
        "manifest_sha256": source_manifest_hash,
        "split_sha256": split_hash,
        "threshold_amendment_sha256": amendment_hash,
        "amendment_sha256": amendment_hash,
        "selection_source": "validation_only",
        "validation_only": True,
        "test_or_heldout_opened": False,
        "probabilities_are_pre_threshold": True,
        "derived_from_binary_mask": False,
        "debug_limited": False,
    }
    disagreements = {
        key: (selector.get(key), wanted)
        for key, wanted in expected_selector.items()
        if selector.get(key) != wanted
    }
    if disagreements:
        raise PreparationError(f"validation selector binding changed: {disagreements}")
    threshold = selector.get("selected_threshold")
    allowed = [float(value) for value in selector.get("allowed_grid", [])]
    if allowed != [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        raise PreparationError("validation selector does not use the frozen expanded grid")
    if type(threshold) not in (int, float) or float(threshold) not in allowed:
        raise PreparationError("validation selector chose an invalid threshold")
    scores = selector.get("threshold_scores")
    if not isinstance(scores, dict) or len(scores) != len(allowed):
        raise PreparationError("validation selector lacks the full score grid")
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in scores.values()):
        raise PreparationError("validation selector contains an invalid Dice score")
    return {
        "method": method,
        "task": task,
        "seed": seed,
        "selected_threshold": float(threshold),
        "selector_source_manifest_sha256": source_manifest_hash,
        "source_config_sha256": config_hash,
        "checkpoint_sha256": checkpoint_hash,
        "training_run_card_sha256": sha256_file(args.training_run_card),
        "selector_receipt_sha256": sha256_file(args.selector_receipt),
        "threshold_amendment_sha256": amendment_hash,
        "training_commit": training_commit,
        "parameter_count": run.get("parameter_count"),
    }


def validate_target(
    args: argparse.Namespace, mapping: Mapping[str, str]
) -> Mapping[str, Any]:
    provenance = read_json(
        args.b1_verified_manifest.parent / "provenance.json", "B1 build provenance"
    )
    status = str(provenance.get("verification_status", ""))
    expected_status = "SYNTHETIC_TEST_ONLY" if args.synthetic_test_only else "REFERENCE_VERIFIED"
    if provenance.get("schema") != "cureblock.sd302g.build-provenance.v1":
        raise PreparationError("B1 build provenance schema is unsupported")
    if provenance.get("protocol") != PROTOCOL or status != expected_status:
        raise PreparationError("B1 coordinate build is not eligible for this mode")
    if provenance.get("manifest_sha256") != sha256_file(args.b1_verified_manifest):
        raise PreparationError("B1 build provenance does not bind the manifest")
    if provenance.get("split_sha256") != sha256_file(args.splits):
        raise PreparationError("B1 build provenance does not bind the subject split")

    verified_columns, verified_rows = read_csv(args.b1_verified_manifest, "verified B1 manifest")
    required = {
        "image_id", "image_path", "hull_path", "minutiae_path", "subject_id",
        "split", "group", "device", "capture", "width", "height", "ppi",
    }
    if required - set(verified_columns):
        raise PreparationError("verified B1 manifest lacks protocol columns")
    inference_columns, inference_rows = read_csv(
        args.b1_inference_manifest, "ground-truth-blind B1 manifest"
    )
    required_inference = {
        "image_id", "image_path", "mask_path", "subject_id", "subset", "ppi",
        "group", "device", "capture", "source_sha256",
    }
    if required_inference - set(inference_columns):
        raise PreparationError("B1 inference manifest lacks authenticated source columns")
    if {row["image_id"] for row in verified_rows} != {row["image_id"] for row in inference_rows}:
        raise PreparationError("B1 inference identity universe differs from verified build")
    verified_by_id = {row["image_id"]: row for row in verified_rows}
    if len(verified_by_id) != len(verified_rows):
        raise PreparationError("verified B1 manifest contains duplicate identities")
    source_records = []
    test_ids = []
    test_subjects = set()
    for row in inference_rows:
        image_id = row["image_id"]
        verified = verified_by_id[image_id]
        subject = row["subject_id"]
        assignment = mapping.get(subject)
        if assignment is None or assignment != verified["split"]:
            raise PreparationError("B1 row disagrees with the authoritative subject split")
        if any(
            row.get(key) != expected
            for key, expected in (
                ("subject_id", verified["subject_id"]),
                ("subset", "finger_exemplar"),
                ("group", verified["group"]),
                ("device", verified["device"]),
                ("capture", verified["capture"]),
            )
        ):
            raise PreparationError("B1 inference metadata differs from verified build")
        if row.get("mask_path"):
            raise PreparationError("B1 inference manifest exposes a label path")
        digest = row.get("source_sha256", "")
        if not HEX64.fullmatch(digest):
            raise PreparationError("B1 inference manifest lacks a valid source digest")
        source_records.append({"image_id": image_id, "source_sha256": digest})
        if assignment == "test":
            test_ids.append(image_id)
            test_subjects.add(subject)

    receipt = read_json(args.b1_inference_receipt, "B1 inference-manifest receipt")
    expected_receipt = {
        "schema_version": 1,
        "kind": "sd302g_gt_blind_inference_manifest",
        "source_manifest_sha256": sha256_file(args.b1_verified_manifest),
        "split_sha256": sha256_file(args.splits),
        "output_sha256": sha256_file(args.b1_inference_manifest),
        "rows": len(verified_rows),
        "mask_paths_empty": True,
        "ordered_source_images_sha256": canonical_sha256(
            sorted(source_records, key=lambda item: item["image_id"])
        ),
    }
    disagreements = {
        key: (receipt.get(key), wanted)
        for key, wanted in expected_receipt.items()
        if receipt.get(key) != wanted
    }
    if disagreements:
        raise PreparationError(f"B1 inference receipt changed: {disagreements}")

    if args.synthetic_test_only:
        if len(verified_rows) < 1 or len(test_ids) < 1 or len(test_subjects) < 1:
            raise PreparationError("synthetic B1 fixture is empty")
    elif (
        len(verified_rows) != PRODUCTION_TOTAL
        or len(test_ids) != PRODUCTION_TEST
        or len(test_subjects) != PRODUCTION_TEST_SUBJECTS
    ):
        raise PreparationError("production B1 universe is not 2380/475/40")

    gate = read_json(args.reference_gate, "B1 reference gate")
    report = read_json(args.reference_report, "B1 independent reference report")
    if not args.synthetic_test_only:
        if (
            gate.get("schema") != "cureblock.sd302g.reference-gate.v1"
            or gate.get("protocol") != PROTOCOL
            or gate.get("trust_level") != "reference_verified"
        ):
            raise PreparationError("B1 reference gate is not reference verified")
        if (
            report.get("schema") != "cureblock.sd302g.independent-reference-report.v1"
            or report.get("status") != "REFERENCE_VERIFIED"
            or report.get("gate_sha256") != sha256_file(args.reference_gate)
            or report.get("comparison", {}).get("mismatch_count") != 0
        ):
            raise PreparationError("B1 independent reference verification did not pass")
    return {
        "manifest_sha256": sha256_file(args.b1_inference_manifest),
        "manifest_receipt_sha256": sha256_file(args.b1_inference_receipt),
        "verified_manifest_sha256": sha256_file(args.b1_verified_manifest),
        "verified_provenance_sha256": sha256_file(
            args.b1_verified_manifest.parent / "provenance.json"
        ),
        "split_sha256": sha256_file(args.splits),
        "subset": "finger_exemplar",
        "split": "test",
        "records": len(verified_rows),
        "test_records": len(test_ids),
        "test_subjects": len(test_subjects),
        "test_image_ids_sha256": canonical_sha256(sorted(test_ids)),
        "ordered_source_images_sha256": canonical_sha256(
            sorted(
                [item for item in source_records if verified_by_id[item["image_id"]]["split"] == "test"],
                key=lambda item: item["image_id"],
            )
        ),
        "mask_paths_empty": True,
        "verification_status": status,
    }


def prepare(args: argparse.Namespace) -> Mapping[str, Any]:
    if not SAFE_METHOD.fullmatch(args.method_key):
        raise PreparationError("method key is not filesystem safe")
    inputs = (
        args.config, args.checkpoint, args.training_run_card, args.selector_receipt,
        args.threshold_amendment, args.dataset1_manifest, args.b1_verified_manifest,
        args.b1_inference_manifest, args.b1_inference_receipt, args.reference_gate,
        args.reference_report, args.splits, args.inference_script,
        args.finalizer_script, args.evaluator,
    )
    if any(not path.is_file() for path in inputs):
        raise PreparationError("one or more required B1 preparation inputs are missing")
    for path in (
        args.authorization_out, args.plan_out, args.predictions, args.probabilities,
        args.inference_card, args.score_out, args.score_journal,
    ):
        outside_repository(path)
    if args.predictions.exists() or args.probabilities.exists() or args.inference_card.exists():
        raise PreparationError("fresh B1 inference outputs already exist")
    if args.score_out.exists() or args.score_journal.exists():
        raise PreparationError("B1 score destination or score-once journal already exists")
    if not args.synthetic_test_only:
        if probability_artifacts.validate_threshold_amendment(
            args.threshold_amendment
        ) != sha256_file(args.threshold_amendment):
            raise PreparationError("threshold amendment validation returned another identity")
        repository = args.inference_script.resolve().parents[2]
        try:
            observed_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repository,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PreparationError("cannot authenticate inference repository") from exc
        if observed_commit != args.expected_inference_commit or dirty:
            raise PreparationError("inference code is not the requested clean pinned commit")
    mapping = validate_split(args.splits) if not args.synthetic_test_only else {
        str(k): str(v)
        for k, v in read_json(args.splits, "synthetic split")["splits"].items()
    }
    model = dict(validate_training_and_selector(args))
    if args.method_key != f"{model['method']}_seed{model['seed']}":
        raise PreparationError("method key must be '<training-method>_seed<seed>'")
    target = dict(validate_target(args, mapping))
    authorization: Dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "kind": AUTHORIZATION_KIND,
        "protocol": PROTOCOL,
        "status": "AUTHORIZED",
        "method_key": args.method_key,
        "display_name": args.display_name,
        "model": model,
        "source_domain": {
            "name": "SD302-Latent",
            "manifest_sha256": sha256_file(args.dataset1_manifest),
            "split_sha256": sha256_file(args.splits),
            "selection_split": "val",
            "training_labels": "sd302h_roi",
        },
        "target_domain": target,
        "constraints": {
            "validation_selector_reused_without_change": True,
            "target_labels_opened": False,
            "sd302g_supervised_fitting": False,
            "hulls_used_for_inference": False,
            "minutiae_used_for_inference": False,
            "probability_maps_required": True,
            "score_once_required": True,
            "reported_metrics": ["mc", "hr", "par", "ohf"],
            "forbidden_metrics": ["dice", "jaccard", "fdr"],
        },
        "artifacts": {
            "config": identity(args.config),
            "checkpoint": identity(args.checkpoint),
            "training_run_card": identity(args.training_run_card),
            "selector_receipt": identity(args.selector_receipt),
            "threshold_amendment": identity(args.threshold_amendment),
            "dataset1_manifest": identity(args.dataset1_manifest),
            "b1_verified_manifest": identity(args.b1_verified_manifest),
            "b1_verified_provenance": identity(
                args.b1_verified_manifest.parent / "provenance.json"
            ),
            "b1_inference_manifest": identity(args.b1_inference_manifest),
            "b1_inference_receipt": identity(args.b1_inference_receipt),
            "reference_gate": identity(args.reference_gate),
            "reference_report": identity(args.reference_report),
            "splits": identity(args.splits),
            "inference_script": identity(args.inference_script),
            "finalizer_script": identity(args.finalizer_script),
            "evaluator": identity(args.evaluator),
        },
        "execution": {
            "expected_inference_commit": args.expected_inference_commit,
            "device": args.device,
            "memory_fraction": args.memory_fraction,
            "warmup": args.warmup,
        },
        "outputs": {
            "predictions": str(args.predictions.resolve()),
            "probabilities": str(args.probabilities.resolve()),
            "inference_card": str(args.inference_card.resolve()),
            "score_out": str(args.score_out.resolve()),
            "score_journal": str(args.score_journal.resolve()),
        },
        "synthetic_test_only": bool(args.synthetic_test_only),
    }
    if not HEX40.fullmatch(args.expected_inference_commit):
        raise PreparationError("expected inference commit must be an immutable 40-hex SHA")
    write_json_once(args.authorization_out, authorization)
    authorization_hash = sha256_file(args.authorization_out)
    inference_argv = [
        str(args.python.resolve()), str(args.inference_script.resolve()),
        "--config", str(args.config.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
        "--manifest", str(args.b1_inference_manifest.resolve()), "--splits", str(args.splits.resolve()),
        "--subset", "finger_exemplar", "--split", "test",
        "--out", str(args.predictions.resolve()),
        "--probability-out", str(args.probabilities.resolve()),
        "--threshold-receipt", str(args.selector_receipt.resolve()),
        "--threshold-amendment", str(args.threshold_amendment.resolve()),
        "--cross-domain-authorization", str(args.authorization_out.resolve()),
        "--timing-out", str(args.inference_card.resolve()),
        "--device", args.device, "--memory-fraction", f"{args.memory_fraction:g}",
        "--warmup", str(args.warmup), "--skip-valid",
    ]
    score_argv = [
        str(args.python.resolve()), str(args.finalizer_script.resolve()),
        "--authorization", str(args.authorization_out.resolve()),
    ]
    plan = {
        "schema_version": 1,
        "kind": "sd302g_b1_crossdomain_execution_plan",
        "authorization_sha256": authorization_hash,
        "method_key": args.method_key,
        "inference": {
            "argv": inference_argv,
            "reads_target_labels": False,
            "retains_masks": True,
            "retains_probabilities": True,
            "retains_timing": True,
        },
        "score_once": {
            "argv": score_argv,
            "journal": str(args.score_journal.resolve()),
            "metrics": ["mc", "hr", "par", "ohf"],
            "per_device_and_capture": True,
            "hull_reference_gate": {"mc": 1.0, "hr": 1.0},
        },
        "launch_performed": False,
    }
    write_json_once(args.plan_out, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if supplied[:1] == ["--spec"]:
        if len(supplied) != 2:
            raise PreparationError("--spec must be the only preparation argument")
        spec = read_json(Path(supplied[1]), "B1 cross-domain specification")
        allowed = {
            "method_key", "display_name", "config", "checkpoint",
            "training_run_card", "selector_receipt", "threshold_amendment",
            "dataset1_manifest", "b1_verified_manifest", "b1_inference_manifest",
            "b1_inference_receipt", "reference_gate", "reference_report", "splits",
            "inference_script", "finalizer_script", "evaluator", "python",
            "expected_inference_commit", "device", "memory_fraction", "warmup",
            "predictions", "probabilities", "inference_card", "score_out",
            "score_journal", "authorization_out", "plan_out", "synthetic_test_only",
        }
        if set(spec) - allowed:
            raise PreparationError(
                f"B1 specification contains unknown keys: {sorted(set(spec) - allowed)}"
            )
        supplied = []
        for key, value in spec.items():
            option = "--" + key.replace("_", "-")
            if key == "synthetic_test_only":
                if value is True:
                    supplied.append(option)
                elif value is not False:
                    raise PreparationError("synthetic_test_only must be boolean")
            else:
                supplied.extend((option, str(value)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        help="JSON specification; when used it must be the only CLI option",
    )
    parser.add_argument("--method-key", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-run-card", required=True, type=Path)
    parser.add_argument("--selector-receipt", required=True, type=Path)
    parser.add_argument("--threshold-amendment", required=True, type=Path)
    parser.add_argument("--dataset1-manifest", required=True, type=Path)
    parser.add_argument("--b1-verified-manifest", required=True, type=Path)
    parser.add_argument("--b1-inference-manifest", required=True, type=Path)
    parser.add_argument("--b1-inference-receipt", required=True, type=Path)
    parser.add_argument("--reference-gate", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--inference-script", required=True, type=Path)
    parser.add_argument("--finalizer-script", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--python", default=Path(sys.executable), type=Path)
    parser.add_argument("--expected-inference-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--probabilities", required=True, type=Path)
    parser.add_argument("--inference-card", required=True, type=Path)
    parser.add_argument("--score-out", required=True, type=Path)
    parser.add_argument("--score-journal", required=True, type=Path)
    parser.add_argument("--authorization-out", required=True, type=Path)
    parser.add_argument("--plan-out", required=True, type=Path)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args(supplied)
    if not 0 < args.memory_fraction <= 0.3:
        parser.error("--memory-fraction must be in (0, 0.3]")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    try:
        prepare(parse_args(argv))
    except (PreparationError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
