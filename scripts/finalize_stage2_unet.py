#!/usr/bin/env python3
"""One-time held-out U-Net inference, common scoring, curves, and private QA.

The validation-only Stage 2 gate authorizes exactly the checkpoint selected
before this program opens any test image or ground truth. Predictions are made
once through ``infer_masks_repro.py``; both GT definitions are then scored only
through ``evaluate_masks.py``. QA sampling is identifier-hash based and never
uses test performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import _path  # noqa: F401
from segbench.artifact_identity import (
    atomic_json,
    atomic_json_once,
    canonical_sha256,
    prediction_set_identity,
    require_clean_process_start,
    sha256_file,
)
from segbench.deep_models import build_model
from segbench.deep_utils import (
    examiner_exclusions_for_split,
    load_config,
    load_split_rows,
    run_provenance,
)
from segbench.io import (
    attach_split,
    load_manifest,
    manifest_target_shape,
    mask_ppi_for,
    match_shape,
    read_gray,
    read_mask,
    to_target_ppi,
)
from segbench.metrics import METRIC_KEYS, summarise
import infer_masks


GROUP_QUOTAS = {"VALUE": 4, "LIMITED": 3, "NOVALUE": 3}
RECOVERY_ACTION = "reuse_existing_inference_and_score_once"
RECOVERY_REASON_CODE = "cuda_device_token_validation_bug"
RECOVERY_AUTHORIZATION_BASIS = "v2_auto_advance_with_append_only_rerun_log"
EXPECTED_FAILED_QUEUE_SHA256 = "f09b45418a20f843706f1732e797fd32cc7874dd03c7ced7d954d8d6cd46c5e4"
EXPECTED_FAILED_JOB_IDENTITY_SHA256 = "003d90f56daefdacfd87e958a919784b03e7cf9ef60f9113911128537f7a680b"
POST_SCORE_RECOVERY_ACTION = "reuse_existing_inference_and_rescore_after_validator_bug"
POST_SCORE_RECOVERY_REASON_CODE = "examiner_group_validator_omitted_mixed"
POST_SCORE_RECOVERY_AUTHORIZATION_BASIS = "v2_bug_forced_rerun_logged"
POST_SCORE_RECOVERY_KIND = "stage2_unet_failed_scoring_validation_recovery"
EXPECTED_POST_SCORE_FAILED_JOURNAL_SHA256 = "99f8f3bc6bf82d1f9de20ab280ed206cc4bebcec21481648932c55da07d067eb"
EXPECTED_POST_SCORE_FAILED_LEDGER_SHA256 = "638ae1dedf56fd6a118b7197d817ef2d13068553d18377fb3415ee28a0ec9f71"
EXPECTED_POST_SCORE_FAILED_QUEUE_SHA256 = "0760948aef56e33408270c8d5e14dab3333032e3f102d8f235acce354eda0813"
EXPECTED_POST_SCORE_FAILED_QUEUE_LOG_SHA256 = "0877bb9abccf7e37685e6d1114655394e63e8615780e52f30ee66ee9b9dd3fdd"
EXPECTED_POST_SCORE_FAILED_JOB_IDENTITY_SHA256 = "3d78af825c64594a205b0e4a368105a7e01c3d42f9032bc70d2047dc84093fb3"
EXPECTED_FIRST_RECOVERY_AUTHORIZATION_SHA256 = "aaf466a12256654e1576f9c87daffc3dbce1818ae06f113c61b43da242319459"
EXPECTED_POST_SCORE_RERUN_LOG_SHA256 = "6ae10827802236b502e23ca6e53c35c57eb595f20d2cb34d4fd9ff7c056a6cba"
EXPECTED_INFERENCE_CARD_SHA256 = "f80c0ae27a51314e30500eed45f8acc4c9a13c58ca701e1ef98177d7468bdae8"
EXPECTED_PREDICTION_SET_SHA256 = "ab16459041691487d0daae4bbfcfc36acd1bac8d2b0e55dc3d196d9879170ec9"
EXPECTED_STAGE2_GATE_SHA256 = "08a6095f30cc790ee6526ea525a40cc0f1d80f7904b2657b7bc7a6bd4fdca25d"
EXPECTED_STAGE2_CHECKPOINT_SHA256 = "81ba56bb3c92d2830dbdee99c34e6125c9b94a23e6d2ea5d4e74466c709be79a"
EXPECTED_STAGE2_CONFIG_SHA256 = "d8b636de058f76f38b03e6d667cd73b04aaf2e14f1929437fe0133ae65d9404b"
EXPECTED_MANIFEST_AUDIT_SHA256 = "208983589a4ff709a092d217448e65995cbcd74e8a0dfc2fbdfe96c44fc48a44"
EXPECTED_ORDERED_SOURCE_GT_SHA256 = "ab913df5646d11e3ecf144ae5c2db17823dc551db8d847fd98691dc9ffb00002"
EXPECTED_EVALUATOR_SHA256 = "9eb327c3c1767669f775d73038ac63855c89809757c7db2e63ced520c5c3d5c9"
EXPECTED_METRICS_SHA256 = "d6277b5bec1dbcc3cb62d912b379a9473150c821f2556d0706d99813a9c5fb44"
EXPECTED_IO_SHA256 = "c9be15e4f1688a5524ae5df8cbd4f6ba167cf90106843d71572ea788625f28e5"
EXPECTED_SUMMARY_SHA256 = "43bab3710ffb9140e3272e26e02b1da7258e459adf3a3ef48027f6bfe6cee94a"
EXPECTED_FAILED_FINALIZER_COMMIT = "4f1d417351c62293557c90ce0689b43d000f264e"
EXPECTED_FIRST_FAILED_JOURNAL_SHA256 = "c54051323efe1d794d7dc831e3bfb0ee9ae9cec305057b9469620d061c119d20"
POST_SCORE_FAILURE_ERROR = "ValueError: Stage 2 summary does not contain exactly both registered tables"
EXPECTED_PREDICTION_COUNT = 2010
EXPECTED_TRAINING_COMMIT = "ee6acf4e3648ce7c98dea2ce30615d5bd55c0e1a"
EXPECTED_MANIFEST_SHA256 = "2b2eee4be7fc75284569450465c2579765fd6efb33fddccb3f7500994190dff6"
EXPECTED_SPLIT_SHA256 = "1d3af693767192f03c16ed8ac79e93029ee130f03cc68a09e755acdf8b018be8"
EXPECTED_GATE_SPEC_SHA256 = "04795e42e44b4d03dda584c176468f9932f6a80ad94c76cdac6fa1a2f6e9a261"
EXPECTED_GATE_EVALUATOR_SHA256 = "518876415ad2de2864d9e355c8f36982636653b4cd73f6d074b104d5ef65fc77"


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a JSON object")
    return value


def _same_hash(path: Path, expected: object, description: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{description} SHA-256 differs from gate evidence")
    return observed


def _canonical_device_label(value: object) -> str:
    """Canonicalize exact inference-card CPU/CUDA labels without aliases."""
    if not isinstance(value, str):
        raise ValueError("device label is not a string")
    normalized = value.upper()
    if normalized == "CUDA":
        return "CUDA"
    if normalized == "CPU":
        return "CPU"
    raise ValueError(f"unsupported device label: {value!r}")


def validate_gate(args: argparse.Namespace) -> Dict[str, Any]:
    """Prove the immutable validation gate authorizes this exact best checkpoint."""
    gate = _read_json(args.gate, "Stage 2 validation gate")
    if gate.get("schema_version") != 1 or gate.get("gate") != "stage2_unet_seed0_validation_health":
        raise ValueError("Stage 2 gate schema or identity changed")
    if gate.get("status") != "PASS" or gate.get("failures") != []:
        raise ValueError("Stage 2 validation gate is not an unqualified PASS")
    policy = gate.get("policy")
    if not isinstance(policy, dict) or gate.get("policy_sha256") != canonical_sha256(policy):
        raise ValueError("Stage 2 gate policy identity is invalid")
    if policy.get("expected_git_commit") != EXPECTED_TRAINING_COMMIT:
        raise ValueError("Stage 2 gate training commit changed")
    if policy.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256 or policy.get("splits_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("Stage 2 gate data identities changed")
    if tuple(policy.get("thresholds", ())) != (0.3, 0.4, 0.5, 0.6, 0.7):
        raise ValueError("Stage 2 gate threshold budget changed")
    specification = gate.get("specification")
    if not isinstance(specification, dict):
        raise ValueError("Stage 2 gate lacks specification identity")
    expected_spec = args.gate.parent / "STAGE2_GATE_SPEC.md"
    if Path(str(specification.get("path", ""))) != expected_spec or specification.get("sha256") != EXPECTED_GATE_SPEC_SHA256:
        raise ValueError("Stage 2 gate specification path or pin changed")
    _same_hash(expected_spec, EXPECTED_GATE_SPEC_SHA256, "gate specification")
    evaluator_path = args.gate.parent.parent / "jobs" / "evaluate_stage2_unet_gate.py"
    if gate.get("evaluator_sha256") != EXPECTED_GATE_EVALUATOR_SHA256:
        raise ValueError("Stage 2 gate evaluator pin changed")
    _same_hash(evaluator_path, EXPECTED_GATE_EVALUATOR_SHA256, "gate evaluator")
    evidence = gate.get("evidence", {})
    if evidence.get("validation_only") is not True or evidence.get("test_evidence_detected") is not False or evidence.get("test_data_opened") is not False:
        raise ValueError("Stage 2 gate does not attest validation-only evaluation")
    inputs = gate.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Stage 2 gate lacks input identities")
    for key, path in (
        ("manifest", args.manifest),
        ("splits", args.splits),
        ("full_config", args.config),
        ("full_run_card", args.run_dir / "run.json"),
        ("full_history", args.run_dir / "history.json"),
    ):
        record = inputs.get(key)
        if not isinstance(record, dict) or Path(str(record.get("path", ""))) != path:
            raise ValueError(f"gate {key} path differs from requested artifact")
        _same_hash(path, record.get("sha256"), f"gate {key}")
    checkpoints = inputs.get("full_checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"best.pt", "last.pt"}:
        raise ValueError("gate checkpoint universe is not exactly best.pt/last.pt")
    checkpoint = args.run_dir / "best.pt"
    checkpoint_sha = _same_hash(checkpoint, checkpoints["best.pt"], "best checkpoint")
    run = _read_json(args.run_dir / "run.json", "full U-Net run card")
    if run.get("checkpoint_sha256", {}).get("best.pt") != checkpoint_sha:
        raise ValueError("run card and gate disagree on best checkpoint")
    if run.get("method") != "unet" or run.get("seed") != 0 or run.get("debug_limited") is not False:
        raise ValueError("authorized run is not the non-debug U-Net seed-0 pilot")
    threshold = float(gate.get("evidence", {}).get("selected_validation_threshold"))
    if threshold != float(run.get("selected_validation_threshold")):
        raise ValueError("gate and run card disagree on validation-selected threshold")
    return {
        "gate": gate,
        "gate_sha256": sha256_file(args.gate),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "run": run,
        "threshold": threshold,
    }


def _run(command: Sequence[str]) -> None:
    completed = subprocess.run(list(command), check=False)
    if completed.returncode == 75:
        raise SystemExit(75)
    completed.check_returncode()


def _validate_inference_card(
    card: Mapping[str, Any], args: argparse.Namespace, authorization: Mapping[str, Any]
) -> None:
    require_clean_process_start(
        card, args.expected_inference_git_commit, description="immutable inference card"
    )
    if card.get("method") != "unet" or card.get("subset") != "finger_latent" or card.get("split") != "test":
        raise ValueError("inference card is not held-out U-Net inference")
    if card.get("schema_version") != 1 or card.get("kind") != "complete_unbiased_inference_card" or card.get("task") != "pixel":
        raise ValueError("inference card schema, kind, or task changed")
    if card.get("debug_limited") is not False:
        raise ValueError("held-out inference is debug-limited")
    expected_device = "CUDA" if args.device == "cuda" else "CPU"
    if _canonical_device_label(card.get("device")) != expected_device:
        raise ValueError("inference card device differs from requested device")
    timing = card.get("timing")
    if not isinstance(timing, dict) or timing.get("warmup_images") != 10:
        raise ValueError("inference timing did not use the registered 10-image warm-up")
    timing_rows = timing.get("per_image")
    if not isinstance(timing_rows, list) or not timing_rows:
        raise ValueError("inference card lacks full per-image timing")
    timing_values = np.asarray([item.get("ms_per_img") for item in timing_rows], dtype=float)
    if not np.isfinite(timing_values).all() or (timing_values < 0).any() or not np.isclose(timing_values.mean(), float(timing.get("mean_ms_per_img"))):
        raise ValueError("inference timing mean does not reproduce finite per-image timing")
    if float(card.get("selected_validation_threshold")) != float(authorization["threshold"]):
        raise ValueError("inference threshold differs from gate-selected validation threshold")
    identities = card.get("identities")
    if not isinstance(identities, dict):
        raise ValueError("inference card lacks identities")
    expected = {
        "source_config_sha256": sha256_file(args.config),
        "checkpoint_sha256": authorization["checkpoint_sha256"],
        "authorized_checkpoint_sha256": authorization["checkpoint_sha256"],
        "manifest_sha256": sha256_file(args.manifest),
        "split_sha256": sha256_file(args.splits),
    }
    for key, value in expected.items():
        if identities.get(key) != value:
            raise ValueError(f"inference card {key} mismatch")
    prediction_set = card.get("prediction_set")
    config = load_config(args.config)
    data = config["data"]
    frame = load_split_rows(
        args.manifest,
        args.splits,
        subset="finger_latent",
        split="test",
        mask_column=str(data["mask_column"]),
        require_mask=False,
        exclude_examiner_values=examiner_exclusions_for_split(data, "test"),
        limit=0,
    )
    expected_ids = frame.image_id.astype(str).tolist()
    if [str(item.get("image_id")) for item in timing_rows] != expected_ids:
        raise ValueError("timing IDs/order differ from shared inference loader")
    if not isinstance(prediction_set, dict) or prediction_set.get("count") != len(expected_ids) or identities.get("prediction_count") != len(expected_ids):
        raise ValueError("inference card prediction universe is inconsistent")
    records = prediction_set.get("records")
    if not isinstance(records, list) or [str(item.get("image_id")) for item in records] != sorted(expected_ids):
        raise ValueError("inference card IDs differ from exact held-out universe")
    items = []
    for record in records:
        image_id = str(record.get("image_id", ""))
        path = infer_masks._safe_output(args.masks_out, image_id)
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"prediction changed after inference card: {image_id}")
        items.append((image_id, path))
    recomputed = prediction_set_identity(items)
    if recomputed != prediction_set or recomputed["sha256"] != identities.get("prediction_set_sha256"):
        raise ValueError("prediction-set digest does not recompute exactly")


def ensure_inference(
    args: argparse.Namespace, authorization: Mapping[str, Any]
) -> Dict[str, Any]:
    if not args.inference_card.exists():
        _run(
            [
                sys.executable,
                str(Path(__file__).with_name("infer_masks_repro.py")),
                "--config", str(args.config),
                "--checkpoint", str(authorization["checkpoint"]),
                "--expected-checkpoint-sha256", str(authorization["checkpoint_sha256"]),
                "--manifest", str(args.manifest),
                "--splits", str(args.splits),
                "--subset", "finger_latent",
                "--split", "test",
                "--out", str(args.masks_out),
                "--card-out", str(args.inference_card),
                "--work-dir", str(args.inference_work_dir),
                "--expected-git-commit", args.expected_inference_git_commit,
                "--device", args.device,
                "--memory-fraction", "0.3",
                "--warmup", "10",
                "--seed", "0",
            ]
        )
    card = _read_json(args.inference_card, "immutable inference card")
    _validate_inference_card(card, args, authorization)
    return card


def _score_command(
    args: argparse.Namespace,
    stage: Path,
    card: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    table: str,
    gt_column: str,
) -> list[str]:
    config = load_config(args.config)
    model = build_model("unet", config["model"])
    params_m = sum(int(parameter.numel()) for parameter in model.parameters()) / 1_000_000.0
    timing = card.get("timing", {})
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_masks.py")),
        "--manifest", str(args.manifest),
        "--splits", str(args.splits),
        "--subsets", "finger_latent",
        "--method", "unet",
        "--display-name", "U-Net",
        "--pred-template", str(args.masks_out / "{image_id}.png"),
        "--pred-ppi", "500",
        "--target-ppi", "500",
        "--split", "test",
        "--gt-column", gt_column,
        "--exclude", "examiner_value=NONPRINT",
        "--seed", "0",
        "--year", "2022",
        "--family", "Encoder-decoder",
        "--training-data", "SD302-Latent train split",
        "--comment", "LR and threshold selected on validation only",
        "--params-m", str(params_m),
        "--device", "GPU" if args.device == "cuda" else "CPU",
        "--threshold", str(authorization["threshold"]),
        "--run-id", str(authorization["checkpoint_sha256"]),
        "--table", table,
        "--ms-per-img", str(timing["mean_ms_per_img"]),
        "--out", str(stage),
    ]
    command.extend(("--group-by", "examiner_value"))
    return command


def _aggregate_nan_expected(
    per_image: pd.DataFrame, metric: str, suffix: str
) -> bool:
    """Mirror the common evaluator's legitimate undefined aggregate cases."""
    if metric not in {"mdr", "fdr"}:
        return False
    if suffix == "_mean":
        return bool(pd.to_numeric(per_image[metric], errors="coerce").isna().all())
    if suffix != "_pooled":
        raise ValueError(f"unknown aggregate suffix {suffix!r}")
    denominator = (
        per_image.tp.astype(np.int64) + per_image.fn.astype(np.int64)
        if metric == "mdr"
        else per_image.fp.astype(np.int64) + per_image.tn.astype(np.int64)
    )
    return int(denominator.sum()) == 0


def validate_scoring(stage: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Reject missing predictions, wrong IDs, non-finite metrics, or row drift."""
    split_payload = _read_json(args.splits, "subject split")["splits"]
    frame = attach_split(load_manifest(args.manifest), split_payload)
    frame = frame[
        (frame.subset == "finger_latent")
        & (frame.split == "test")
        & (frame.examiner_value.astype(str) != "NONPRINT")
    ].copy()
    table_specs = {
        "A": "mask_path",
        "A_prime": "quality_mask_path",
    }
    universes: Dict[str, pd.DataFrame] = {}
    for table, gt_column in table_specs.items():
        paths = frame[gt_column].fillna("").astype(str)
        universe = frame[paths.str.len() > 0].copy()
        if universe.empty:
            raise ValueError(f"{table}: exact GT universe is empty")
        groups = universe.examiner_value.fillna("").astype(str)
        if (groups == "").any():
            raise ValueError(f"{table}: exact GT universe has an empty examiner value")
        universes[table] = universe

    summary_path = stage / "summary.csv"
    summary = pd.read_csv(summary_path)
    if set(summary.table.astype(str)) != set(table_specs):
        raise ValueError("Stage 2 summary does not contain exactly both registered tables")
    if summary.duplicated(["table", "group_value"], keep=False).any():
        raise ValueError("Stage 2 summary has duplicate table/group rows")
    expected_summary_rows = 0
    checkpoint_sha = sha256_file(args.run_dir / "best.pt")
    for table, gt_column in table_specs.items():
        universe = universes[table]
        expected_groups = {"overall", *universe.examiner_value.astype(str).unique()}
        rows = summary[summary.table.astype(str) == table]
        expected_summary_rows += len(expected_groups)
        if set(rows.group_value.astype(str)) != expected_groups:
            raise ValueError(
                f"{table}: examiner-value rows differ from the exact manifest universe"
            )
        identity = {
            "method": "unet",
            "subset": "finger_latent",
            "split": "test",
            "ground_truth": gt_column,
            "group_by": "examiner_value",
            "run_id": checkpoint_sha,
            "training_data": "SD302-Latent train split",
            "device": "GPU" if args.device == "cuda" else "CPU",
        }
        for column, expected_value in identity.items():
            if column not in rows or set(rows[column].astype(str)) != {expected_value}:
                raise ValueError(f"{table}: summary {column} identity changed")
        seeds = pd.to_numeric(rows.seed, errors="coerce")
        if seeds.isna().any() or not (seeds == 0).all():
            raise ValueError(f"{table}: summary seed identity changed")
        if not (pd.to_numeric(rows.n_missing, errors="coerce") == 0).all():
            raise ValueError(f"{table}: common evaluator reports missing predictions")
        expected_counts = {
            "overall": len(universe),
            **universe.examiner_value.astype(str).value_counts().to_dict(),
        }
        observed_counts = {
            str(row.group_value): int(row.n_images) for _, row in rows.iterrows()
        }
        if observed_counts != expected_counts:
            raise ValueError(f"{table}: summary image counts differ from the manifest")
        for key in METRIC_KEYS:
            for suffix in ("_mean", "_pooled"):
                values = pd.to_numeric(rows[f"{key}{suffix}"], errors="coerce")
                finite = values.dropna()
                if key not in {"mdr", "fdr"} and values.isna().any():
                    raise ValueError(f"{table}: {key}{suffix} is unexpectedly undefined")
                if not ((finite >= 0.0) & (finite <= 1.0)).all():
                    raise ValueError(f"{table}: {key}{suffix} is not finite in [0,1]")
    if len(summary) != expected_summary_rows:
        raise ValueError("Stage 2 summary row count differs from exact group universes")

    evidence: Dict[str, Any] = {"summary_sha256": sha256_file(summary_path)}
    for table, universe in universes.items():
        ids = set(universe.image_id.astype(str))
        matches = list(stage.glob(f"per_image_unet_finger_latent_{table}_seed0_identity-*.csv"))
        if len(matches) != 1:
            raise ValueError(f"{table}: expected exactly one common-evaluator per-image file")
        per = pd.read_csv(matches[0])
        observed = per.image_id.astype(str).tolist()
        if len(observed) != len(ids) or len(observed) != len(set(observed)) or set(observed) != ids:
            raise ValueError(f"{table}: per-image IDs differ from exact GT universe")
        if per.missing.astype(bool).any():
            raise ValueError(f"{table}: per-image evaluator reports missing predictions")
        for key in METRIC_KEYS:
            values = pd.to_numeric(per[key], errors="coerce")
            if key == "mdr":
                expected_nan = (per.tp + per.fn) == 0
            elif key == "fdr":
                expected_nan = (per.fp + per.tn) == 0
            else:
                expected_nan = pd.Series(False, index=per.index)
            if not values.isna().equals(expected_nan):
                raise ValueError(f"{table}: per-image {key} undefinedness is inconsistent")
            finite = values[~expected_nan]
            if not ((finite >= 0.0) & (finite <= 1.0)).all():
                raise ValueError(f"{table}: per-image {key} is not finite in [0,1]")
        summary_rows = summary[summary.table.astype(str) == table]
        for _, aggregate in summary_rows.iterrows():
            group_value = str(aggregate.group_value)
            aggregate_per = (
                per
                if group_value == "overall"
                else per[per.examiner_value.astype(str) == group_value]
            )
            if aggregate_per.empty:
                raise ValueError(f"{table}: empty aggregate group {group_value!r}")
            reproduced = summarise(aggregate_per)
            if int(aggregate.n_images) != int(reproduced["n_images"]):
                raise ValueError(
                    f"{table}/{group_value}: n_images does not reproduce"
                )
            for key in METRIC_KEYS:
                for suffix in ("_mean", "_std", "_pooled"):
                    value = pd.to_numeric(
                        pd.Series([aggregate[f"{key}{suffix}"]]), errors="coerce"
                    ).iloc[0]
                    expected_value = float(reproduced[f"{key}{suffix}"])
                    expected_nan = bool(pd.isna(expected_value))
                    if suffix in {"_mean", "_pooled"} and expected_nan != _aggregate_nan_expected(
                        aggregate_per, key, suffix
                    ):
                        raise ValueError(
                            f"{table}/{group_value}: internal aggregate undefinedness changed"
                        )
                    if bool(pd.isna(value)) != expected_nan:
                        raise ValueError(
                            f"{table}/{group_value}: {key}{suffix} undefinedness "
                            "does not match the common evaluator output"
                        )
                    if not expected_nan and not np.isclose(
                        float(value), expected_value, rtol=1e-12, atol=1e-12
                    ):
                        raise ValueError(
                            f"{table}/{group_value}: {key}{suffix} does not reproduce"
                        )
        evidence[table] = {
            "n_images": len(observed),
            "groups": {
                "overall": len(universe),
                **universe.examiner_value.astype(str).value_counts().to_dict(),
            },
            "per_image_file": matches[0].name,
            "per_image_sha256": sha256_file(matches[0]),
        }
    return evidence


def deterministic_overlay_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Select a preregistered stratified QA sample without reading scores."""
    chosen = []
    for group, count in GROUP_QUOTAS.items():
        part = df[df.examiner_value.astype(str) == group].copy()
        if len(part) < count:
            raise ValueError(f"not enough {group} test rows for QA sample")
        part["_qa_order"] = part.image_id.astype(str).map(
            lambda value: hashlib.sha256(
                f"stage2-unet-overlay-v1\\0{value}".encode("utf-8")
            ).hexdigest()
        )
        chosen.append(part.sort_values(["_qa_order", "image_id"]).head(count))
    return pd.concat(chosen, ignore_index=True)


def _write_qa(
    args: argparse.Namespace, stage: Path, authorization: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads((args.run_dir / "history.json").read_text(encoding="utf-8"))
    epochs = [int(item["epoch"]) + 1 for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="U-Net loss")
    axes[0].legend()
    axes[1].plot(epochs, [item["validation_dice"] for item in history])
    axes[1].set(xlabel="Epoch", ylabel="Dice", title="Validation Dice", ylim=(0, 1))
    figure.tight_layout()
    curve = stage / "training_curves.png"
    figure.savefig(curve, dpi=180)
    plt.close(figure)

    split_payload = _read_json(args.splits, "subject split")["splits"]
    frame = attach_split(load_manifest(args.manifest), split_payload)
    frame = frame[
        (frame.subset == "finger_latent")
        & (frame.split == "test")
        & (frame.mask_path.str.len() > 0)
        & (frame.examiner_value.astype(str) != "NONPRINT")
    ]
    selected = deterministic_overlay_rows(frame)
    overlay_dir = stage / "overlays"
    overlay_dir.mkdir(parents=True)
    records = []
    for index, row in enumerate(selected.to_dict("records"), 1):
        image = to_target_ppi(read_gray(row["image_path"]), float(row["ppi"]), 500.0)
        target_shape = manifest_target_shape(row, 500.0) or image.shape
        if image.shape != tuple(target_shape):
            image = cv2.resize(
                image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA
            )
        gt = to_target_ppi(
            read_mask(row["mask_path"]),
            mask_ppi_for(row, "mask_path"),
            500.0,
            is_mask=True,
        )
        gt = match_shape(gt, target_shape)
        prediction_path = infer_masks._safe_output(args.masks_out, str(row["image_id"]))
        pred = match_shape(read_mask(prediction_path), image.shape)
        canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        gt_contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, gt_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(canvas, pred_contours, -1, (255, 0, 255), 2)
        cv2.putText(canvas, f"GT green | U-Net magenta | {row['examiner_value']}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"GT green | U-Net magenta | {row['examiner_value']}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        destination = overlay_dir / f"overlay_{index:02d}.png"
        if not cv2.imwrite(str(destination), canvas):
            raise RuntimeError(f"could not write QA overlay {destination}")
        records.append({
            "ordinal": index,
            "image_id": str(row["image_id"]),
            "examiner_value": str(row["examiner_value"]),
            "sha256": sha256_file(destination),
        })
    return [{"curve_sha256": sha256_file(curve)}, *records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--masks-out", required=True, type=Path)
    parser.add_argument("--inference-card", required=True, type=Path)
    parser.add_argument("--inference-work-dir", required=True, type=Path)
    parser.add_argument("--bundle-out", required=True, type=Path)
    parser.add_argument("--expected-inference-git-commit", required=True)
    parser.add_argument("--expected-finalizer-git-commit", required=True)
    parser.add_argument("--recovery-authorization", type=Path)
    parser.add_argument("--prior-recovery-authorization", type=Path)
    parser.add_argument("--recovery-ledger", type=Path)
    parser.add_argument("--recovery-log", type=Path)
    parser.add_argument("--rerun-log", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _journal_path(bundle: Path) -> Path:
    return bundle.parent / f".{bundle.name}.journal.json"


def _lock_path(bundle: Path) -> Path:
    return bundle.parent / f".{bundle.name}.lock"


def _publish_journal(path: Path, status: str, **extra: Any) -> None:
    prior: Dict[str, Any] = {}
    if path.exists():
        prior = _read_json(path, "Stage 2 transaction journal")
    prior.update({"schema_version": 1, "status": status, **extra})
    atomic_json(path, prior)


_RECOVERY_AUTHORIZATION_KEYS = {
    "schema_version",
    "kind",
    "action",
    "reason_code",
    "authorization_basis",
    "approved",
    "failed_journal_sha256",
    "failed_ledger_sha256",
    "failed_queue_sha256",
    "failed_queue_log_sha256",
    "rerun_log_sha256",
    "inference_card_sha256",
    "gate_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "manifest_sha256",
    "split_sha256",
    "prediction_set_sha256",
    "prediction_count",
    "old_inference_git_commit",
    "new_finalizer_git_commit",
    "test_scoring_started",
    "rerun_inference",
    "bundle_absent",
}


def _require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} is not a lowercase SHA-256")
    return value


def _require_commit(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} is not an immutable lowercase Git SHA")
    return value


def _archive_failed_journal(journal: Path, expected_sha256: str) -> Path:
    """Preserve the exact failed journal bytes before publishing recovery state."""
    raw = journal.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("failed journal changed before write-once archival")
    archive = journal.with_name(
        f"{journal.name}.failed-{expected_sha256}.archive"
    )
    try:
        descriptor = os.open(
            archive, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
    except FileExistsError:
        if archive.read_bytes() != raw:
            raise FileExistsError(
                f"write-once failed-journal archive differs: {archive}"
            )
        return archive
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return archive


def _authorize_failed_recovery(
    args: argparse.Namespace,
    journal: Path,
    authorization: Mapping[str, Any],
    finalizer_provenance: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Authorize scoring from an already-complete immutable inference card.

    This path never creates or reruns inference.  It is available only for the
    specifically hash-bound failed transaction authorized by an external JSON
    object, and only while held-out scoring is still untouched.
    """
    if args.recovery_authorization is None:
        raise RuntimeError(
            "failed Stage 2 journal requires an explicit recovery authorization"
        )
    if (
        args.recovery_ledger is None
        or args.recovery_log is None
        or args.rerun_log is None
    ):
        raise RuntimeError(
            "failed Stage 2 recovery requires the failed ledger, queue log, and rerun log"
        )
    recovery = _read_json(
        args.recovery_authorization, "Stage 2 recovery authorization"
    )
    if set(recovery) != _RECOVERY_AUTHORIZATION_KEYS:
        raise ValueError("recovery authorization has an unexpected schema")
    if (
        type(recovery.get("schema_version")) is not int
        or recovery.get("schema_version") != 1
        or recovery.get("kind") != "stage2_unet_failed_delivery_recovery"
        or recovery.get("action") != RECOVERY_ACTION
        or recovery.get("reason_code") != RECOVERY_REASON_CODE
        or recovery.get("authorization_basis") != RECOVERY_AUTHORIZATION_BASIS
        or recovery.get("approved") is not True
    ):
        raise ValueError("recovery authorization identity or action changed")
    if recovery.get("test_scoring_started") is not False:
        raise ValueError("recovery authorization does not attest untouched scoring")
    if recovery.get("rerun_inference") is not False:
        raise ValueError("recovery authorization permits an inference rerun")
    if recovery.get("bundle_absent") is not True or args.bundle_out.exists():
        raise ValueError("recovery authorization does not bind an absent delivery bundle")

    failed = _read_json(journal, "failed Stage 2 transaction journal")
    if failed.get("status") != "failed":
        raise ValueError("recovery journal is not in the failed state")
    if failed.get("test_scoring_started") is not False:
        raise ValueError("failed journal shows that held-out scoring already started")
    journal_sha = _require_sha256(
        recovery.get("failed_journal_sha256"), "authorized failed journal digest"
    )
    if sha256_file(journal) != journal_sha:
        raise ValueError("failed journal differs from recovery authorization")

    ledger_sha = _require_sha256(
        recovery.get("failed_ledger_sha256"), "authorized failed ledger digest"
    )
    if sha256_file(args.recovery_ledger) != ledger_sha:
        raise ValueError("failed ledger differs from recovery authorization")
    ledger = _read_json(args.recovery_ledger, "failed Stage 2 queue ledger")
    queue_sha = _require_sha256(
        recovery.get("failed_queue_sha256"), "authorized failed queue digest"
    )
    if (
        queue_sha != EXPECTED_FAILED_QUEUE_SHA256
        or ledger.get("queue_sha256") != queue_sha
    ):
        raise ValueError("failed queue identity differs from recovery authorization")
    jobs = ledger.get("jobs")
    expected_job = {
        "attempts": 2,
        "exit_code": 1,
        "identity_sha256": EXPECTED_FAILED_JOB_IDENTITY_SHA256,
        "oom_retries": 0,
        "ordinary_retries": 1,
        "preemptions": 0,
        "status": "failed",
    }
    if (
        type(ledger.get("schema_version")) is not int
        or ledger.get("schema_version") != 1
        or set(ledger) != {"schema_version", "queue_sha256", "jobs"}
        or not isinstance(jobs, dict)
        or set(jobs) != {"stage2.unet.delivery"}
        or canonical_sha256(jobs.get("stage2.unet.delivery"))
        != canonical_sha256(expected_job)
    ):
        raise ValueError("Stage 2 queue ledger differs from the exact failed attempt")

    queue_log_sha = _require_sha256(
        recovery.get("failed_queue_log_sha256"),
        "authorized append-only queue-log digest",
    )
    if sha256_file(args.recovery_log) != queue_log_sha:
        raise ValueError("append-only queue log differs from recovery authorization")
    rerun_log_sha = _require_sha256(
        recovery.get("rerun_log_sha256"),
        "authorized append-only rerun-log digest",
    )
    if sha256_file(args.rerun_log) != rerun_log_sha:
        raise ValueError("append-only rerun log differs from recovery authorization")

    inference_card_sha = _require_sha256(
        recovery.get("inference_card_sha256"),
        "authorized inference-card digest",
    )
    if sha256_file(args.inference_card) != inference_card_sha:
        raise ValueError("inference card differs from recovery authorization")
    for key, path in (
        ("config_sha256", args.config),
        ("manifest_sha256", args.manifest),
        ("split_sha256", args.splits),
    ):
        expected = _require_sha256(
            recovery.get(key), f"authorized {key.removesuffix('_sha256')} digest"
        )
        if sha256_file(path) != expected:
            raise ValueError(f"{key.removesuffix('_sha256')} differs from recovery authorization")
    gate_sha = _require_sha256(
        recovery.get("gate_sha256"), "authorized gate digest"
    )
    if gate_sha != authorization.get("gate_sha256"):
        raise ValueError("gate differs from recovery authorization")
    checkpoint_sha = _require_sha256(
        recovery.get("checkpoint_sha256"), "authorized checkpoint digest"
    )
    if (
        checkpoint_sha != authorization.get("checkpoint_sha256")
        or sha256_file(authorization["checkpoint"]) != checkpoint_sha
    ):
        raise ValueError("checkpoint differs from recovery authorization")

    old_commit = _require_commit(
        recovery.get("old_inference_git_commit"),
        "authorized inference commit",
    )
    new_commit = _require_commit(
        recovery.get("new_finalizer_git_commit"),
        "authorized finalizer commit",
    )
    if old_commit != args.expected_inference_git_commit:
        raise ValueError("inference commit differs from recovery authorization")
    if new_commit != args.expected_finalizer_git_commit:
        raise ValueError("finalizer commit differs from recovery authorization")
    require_clean_process_start(
        {"provenance": finalizer_provenance},
        new_commit,
        description="Stage 2 recovery finalizer",
    )

    card = _read_json(args.inference_card, "immutable inference card")
    _validate_inference_card(card, args, authorization)
    prediction_set = card.get("prediction_set")
    identities = card.get("identities")
    prediction_sha = _require_sha256(
        recovery.get("prediction_set_sha256"),
        "authorized prediction-set digest",
    )
    if (
        not isinstance(prediction_set, dict)
        or not isinstance(identities, dict)
        or prediction_set.get("sha256") != prediction_sha
        or identities.get("prediction_set_sha256") != prediction_sha
        or recovery.get("prediction_count") != EXPECTED_PREDICTION_COUNT
        or prediction_set.get("count") != EXPECTED_PREDICTION_COUNT
        or identities.get("prediction_count") != EXPECTED_PREDICTION_COUNT
    ):
        raise ValueError("prediction set differs from recovery authorization")

    evidence = {
        "action": RECOVERY_ACTION,
        "reason_code": RECOVERY_REASON_CODE,
        "authorization_basis": RECOVERY_AUTHORIZATION_BASIS,
        "authorization_sha256": sha256_file(args.recovery_authorization),
        "failed_journal_sha256": journal_sha,
        "failed_ledger_sha256": ledger_sha,
        "failed_queue_sha256": queue_sha,
        "failed_queue_log_sha256": queue_log_sha,
        "rerun_log_sha256": rerun_log_sha,
        "inference_card_sha256": inference_card_sha,
        "gate_sha256": gate_sha,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": recovery["config_sha256"],
        "manifest_sha256": recovery["manifest_sha256"],
        "split_sha256": recovery["split_sha256"],
        "prediction_set_sha256": prediction_sha,
        "prediction_count": EXPECTED_PREDICTION_COUNT,
        "old_inference_git_commit": old_commit,
        "new_finalizer_git_commit": new_commit,
        "test_scoring_started": False,
        "rerun_inference": False,
        "bundle_absent": True,
    }
    return card, evidence


_POST_SCORE_RECOVERY_AUTHORIZATION_KEYS = {
    "schema_version",
    "kind",
    "action",
    "reason_code",
    "authorization_basis",
    "approved",
    "failed_journal_sha256",
    "failed_ledger_sha256",
    "failed_queue_sha256",
    "failed_queue_log_sha256",
    "first_recovery_authorization_sha256",
    "first_failed_journal_sha256",
    "rerun_log_sha256",
    "inference_card_sha256",
    "gate_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "manifest_sha256",
    "split_sha256",
    "manifest_audit_sha256",
    "ordered_source_gt_sha256",
    "evaluator_sha256",
    "metrics_sha256",
    "io_sha256",
    "summary_sha256",
    "prediction_set_sha256",
    "prediction_count",
    "old_inference_git_commit",
    "failed_finalizer_git_commit",
    "new_finalizer_git_commit",
    "prior_scoring_started",
    "prior_score_attempts",
    "prior_evaluator_invocations",
    "prior_outputs_persisted",
    "rerun_inference",
    "rerun_scoring",
    "bundle_absent",
    "surviving_score_artifacts_absent",
    "score_attempt_ordinal",
}


def _score_staging_artifacts(bundle: Path) -> list[Path]:
    """Return unpublished transaction directories for this exact bundle."""
    return sorted(
        bundle.parent.glob(f".{bundle.name}-*"), key=lambda path: path.name
    )


def _rehash_ordered_source_gt(manifest: Path) -> str:
    """Rehash every current union GT and reproduce the frozen v1 digest.

    The manifest and its audit are only maps of expected content.  A recovery
    that is about to score must also prove the mask bytes referenced by that
    map have not changed since the manifest was frozen.
    """
    frame = load_manifest(manifest)
    required = {
        "image_id",
        "mask_path",
        "quality_mask_path",
        "roi_union_sha256",
        "quality_union_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest lacks GT identity columns: {missing}")
    digest = hashlib.sha256(b"segbench-source-union-gt-v1\n")
    for row in frame.sort_values("image_id").to_dict("records"):
        image_id = str(row["image_id"])
        roi_expected = _require_sha256(
            row["roi_union_sha256"], f"{image_id} ROI GT digest"
        )
        if sha256_file(Path(str(row["mask_path"]))) != roi_expected:
            raise ValueError(f"{image_id}: ROI GT changed after manifest freeze")
        quality_path = (
            "" if pd.isna(row["quality_mask_path"]) else str(row["quality_mask_path"])
        )
        quality_expected = (
            ""
            if pd.isna(row["quality_union_sha256"])
            else str(row["quality_union_sha256"])
        )
        if quality_path:
            quality_expected = _require_sha256(
                quality_expected, f"{image_id} quality GT digest"
            )
            if sha256_file(Path(quality_path)) != quality_expected:
                raise ValueError(
                    f"{image_id}: quality GT changed after manifest freeze"
                )
        elif quality_expected:
            raise ValueError(f"{image_id}: quality GT digest exists without a path")
        item = {
            "image_id": image_id,
            "quality_union_sha256": quality_expected,
            "roi_union_sha256": roi_expected,
        }
        digest.update(
            (
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _authorize_failed_scoring_recovery(
    args: argparse.Namespace,
    journal: Path,
    authorization: Mapping[str, Any],
    finalizer_provenance: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Authorize exactly one logged rescore after the MIXED-group validator bug.

    The prior common-evaluator invocations did run, but their unpublished
    transaction directory was deleted by the old finalizer.  This path is
    deliberately distinct from the untouched-score recovery above.  It binds
    the exact failed attempt and may reuse only the immutable inference card;
    it can never invoke inference.
    """
    required_paths = {
        "post-score recovery authorization": args.recovery_authorization,
        "first recovery authorization": args.prior_recovery_authorization,
        "failed recovery ledger": args.recovery_ledger,
        "failed recovery queue log": args.recovery_log,
        "bug-forced rerun log snapshot": args.rerun_log,
    }
    missing = [name for name, path in required_paths.items() if path is None]
    if missing:
        raise RuntimeError(
            "post-score recovery requires " + ", ".join(sorted(missing))
        )
    recovery = _read_json(
        args.recovery_authorization, "Stage 2 post-score recovery authorization"
    )
    if set(recovery) != _POST_SCORE_RECOVERY_AUTHORIZATION_KEYS:
        raise ValueError("post-score recovery authorization has an unexpected schema")
    if (
        type(recovery.get("schema_version")) is not int
        or recovery.get("schema_version") != 1
        or recovery.get("kind") != POST_SCORE_RECOVERY_KIND
        or recovery.get("action") != POST_SCORE_RECOVERY_ACTION
        or recovery.get("reason_code") != POST_SCORE_RECOVERY_REASON_CODE
        or recovery.get("authorization_basis")
        != POST_SCORE_RECOVERY_AUTHORIZATION_BASIS
        or recovery.get("approved") is not True
    ):
        raise ValueError("post-score recovery identity or action changed")
    expected_facts = {
        "prior_scoring_started": True,
        "prior_score_attempts": 1,
        "prior_evaluator_invocations": 2,
        "prior_outputs_persisted": False,
        "rerun_inference": False,
        "rerun_scoring": True,
        "bundle_absent": True,
        "surviving_score_artifacts_absent": True,
        "score_attempt_ordinal": 2,
    }
    for key, expected in expected_facts.items():
        if type(recovery.get(key)) is not type(expected) or recovery.get(key) != expected:
            raise ValueError(f"post-score recovery fact changed: {key}")
    if args.bundle_out.exists():
        raise ValueError("post-score recovery requires an absent delivery bundle")
    staging = _score_staging_artifacts(args.bundle_out)
    if staging:
        raise ValueError(
            "post-score recovery found surviving score artifacts: "
            + ", ".join(path.name for path in staging)
        )

    failed = _read_json(journal, "failed Stage 2 scoring journal")
    if (
        failed.get("status") != "failed"
        or failed.get("phase") != "scoring"
        or failed.get("test_scoring_started") is not True
        or failed.get("error") != POST_SCORE_FAILURE_ERROR
        or failed.get("rerun_inference") is not False
    ):
        raise ValueError("failed journal is not the exact scoring-validator failure")
    journal_sha = _require_sha256(
        recovery.get("failed_journal_sha256"), "authorized scoring-failure journal"
    )
    if (
        journal_sha != EXPECTED_POST_SCORE_FAILED_JOURNAL_SHA256
        or sha256_file(journal) != journal_sha
    ):
        raise ValueError("scoring-failure journal differs from authorization")
    first = failed.get("recovery")
    if not isinstance(first, dict):
        raise ValueError("failed journal lacks first recovery evidence")
    first_auth_sha = _require_sha256(
        recovery.get("first_recovery_authorization_sha256"),
        "first recovery authorization digest",
    )
    if (
        first_auth_sha != EXPECTED_FIRST_RECOVERY_AUTHORIZATION_SHA256
        or first.get("authorization_sha256") != first_auth_sha
        or sha256_file(args.prior_recovery_authorization) != first_auth_sha
    ):
        raise ValueError("first recovery authorization differs")
    first_failed_sha = _require_sha256(
        recovery.get("first_failed_journal_sha256"),
        "first failed-journal digest",
    )
    if (
        first_failed_sha != EXPECTED_FIRST_FAILED_JOURNAL_SHA256
        or first.get("failed_journal_sha256") != first_failed_sha
    ):
        raise ValueError("first failed-journal identity differs")
    first_archive_name = first.get("failed_journal_archive")
    if not isinstance(first_archive_name, str) or Path(first_archive_name).name != first_archive_name:
        raise ValueError("first failed-journal archive path is unsafe")
    if sha256_file(journal.parent / first_archive_name) != first_failed_sha:
        raise ValueError("first failed-journal archive changed")

    ledger_sha = _require_sha256(
        recovery.get("failed_ledger_sha256"), "failed recovery-ledger digest"
    )
    if (
        ledger_sha != EXPECTED_POST_SCORE_FAILED_LEDGER_SHA256
        or sha256_file(args.recovery_ledger) != ledger_sha
    ):
        raise ValueError("failed recovery ledger differs")
    ledger = _read_json(args.recovery_ledger, "failed recovery queue ledger")
    queue_sha = _require_sha256(
        recovery.get("failed_queue_sha256"), "failed recovery queue digest"
    )
    expected_job = {
        "attempts": 2,
        "exit_code": 1,
        "identity_sha256": EXPECTED_POST_SCORE_FAILED_JOB_IDENTITY_SHA256,
        "oom_retries": 0,
        "ordinary_retries": 1,
        "preemptions": 0,
        "status": "failed",
    }
    if (
        queue_sha != EXPECTED_POST_SCORE_FAILED_QUEUE_SHA256
        or ledger.get("queue_sha256") != queue_sha
        or type(ledger.get("schema_version")) is not int
        or ledger.get("schema_version") != 1
        or set(ledger) != {"schema_version", "queue_sha256", "jobs"}
        or set(ledger.get("jobs", {})) != {"stage2.unet.delivery.recovery"}
        or canonical_sha256(
            ledger["jobs"].get("stage2.unet.delivery.recovery")
        )
        != canonical_sha256(expected_job)
    ):
        raise ValueError("failed recovery queue identity or ledger state differs")
    queue_log_sha = _require_sha256(
        recovery.get("failed_queue_log_sha256"), "failed recovery queue-log digest"
    )
    if (
        queue_log_sha != EXPECTED_POST_SCORE_FAILED_QUEUE_LOG_SHA256
        or sha256_file(args.recovery_log) != queue_log_sha
    ):
        raise ValueError("failed recovery queue log differs")
    log_text = args.recovery_log.read_text(encoding="utf-8")
    if (
        log_text.count("unet finger_latent: n=") != 2
        or "unet finger_latent: n=2010 missing=0" not in log_text
        or "unet finger_latent: n=2009 missing=0" not in log_text
        or log_text.count(POST_SCORE_FAILURE_ERROR) != 1
        or "failed journal shows that held-out scoring already started" not in log_text
    ):
        raise ValueError("failed recovery log does not prove exactly two evaluator completions")
    rerun_log_sha = _require_sha256(
        recovery.get("rerun_log_sha256"), "bug-forced rerun-log digest"
    )
    if (
        rerun_log_sha != EXPECTED_POST_SCORE_RERUN_LOG_SHA256
        or sha256_file(args.rerun_log) != rerun_log_sha
    ):
        raise ValueError("bug-forced rerun log differs from authorization")

    inference_card_sha = _require_sha256(
        recovery.get("inference_card_sha256"), "inference-card digest"
    )
    if (
        inference_card_sha != EXPECTED_INFERENCE_CARD_SHA256
        or sha256_file(args.inference_card) != inference_card_sha
    ):
        raise ValueError("inference card differs from post-score authorization")
    for key, path, pinned in (
        ("config_sha256", args.config, EXPECTED_STAGE2_CONFIG_SHA256),
        ("manifest_sha256", args.manifest, EXPECTED_MANIFEST_SHA256),
        ("split_sha256", args.splits, EXPECTED_SPLIT_SHA256),
    ):
        expected_sha = _require_sha256(recovery.get(key), f"authorized {key}")
        if expected_sha != pinned or sha256_file(path) != expected_sha:
            raise ValueError(f"{key} differs from post-score authorization")
    manifest_audit = args.manifest.with_suffix(".audit.json")
    manifest_audit_sha = _require_sha256(
        recovery.get("manifest_audit_sha256"), "manifest-audit digest"
    )
    if (
        manifest_audit_sha != EXPECTED_MANIFEST_AUDIT_SHA256
        or sha256_file(manifest_audit) != manifest_audit_sha
    ):
        raise ValueError("manifest audit differs from post-score authorization")
    audit = _read_json(manifest_audit, "latent manifest audit")
    ordered_gt_sha = _require_sha256(
        recovery.get("ordered_source_gt_sha256"), "ordered source-GT digest"
    )
    if (
        ordered_gt_sha != EXPECTED_ORDERED_SOURCE_GT_SHA256
        or audit.get("ordered_source_gt_sha256") != ordered_gt_sha
        or audit.get("manifest_sha256") != recovery.get("manifest_sha256")
        or _rehash_ordered_source_gt(args.manifest) != ordered_gt_sha
    ):
        raise ValueError("ordered source-GT identity differs")
    code_paths = {
        "evaluator_sha256": (
            Path(__file__).with_name("evaluate_masks.py"),
            EXPECTED_EVALUATOR_SHA256,
        ),
        "metrics_sha256": (
            Path(__file__).parents[1] / "segbench" / "metrics.py",
            EXPECTED_METRICS_SHA256,
        ),
        "io_sha256": (
            Path(__file__).parents[1] / "segbench" / "io.py",
            EXPECTED_IO_SHA256,
        ),
        "summary_sha256": (
            Path(__file__).parents[1] / "segbench" / "summary.py",
            EXPECTED_SUMMARY_SHA256,
        ),
    }
    code_hashes: Dict[str, str] = {}
    for key, (path, pinned) in code_paths.items():
        authorized = _require_sha256(recovery.get(key), f"authorized {key}")
        if authorized != pinned or sha256_file(path) != authorized:
            raise ValueError(f"{key} differs from post-score authorization")
        code_hashes[key] = authorized
    gate_sha = _require_sha256(recovery.get("gate_sha256"), "gate digest")
    checkpoint_sha = _require_sha256(
        recovery.get("checkpoint_sha256"), "checkpoint digest"
    )
    if (
        gate_sha != EXPECTED_STAGE2_GATE_SHA256
        or gate_sha != authorization.get("gate_sha256")
        or checkpoint_sha != EXPECTED_STAGE2_CHECKPOINT_SHA256
        or checkpoint_sha != authorization.get("checkpoint_sha256")
        or sha256_file(authorization["checkpoint"]) != checkpoint_sha
    ):
        raise ValueError("gate or checkpoint differs from post-score authorization")

    old_commit = _require_commit(
        recovery.get("old_inference_git_commit"), "inference commit"
    )
    failed_commit = _require_commit(
        recovery.get("failed_finalizer_git_commit"), "failed finalizer commit"
    )
    new_commit = _require_commit(
        recovery.get("new_finalizer_git_commit"), "new finalizer commit"
    )
    if old_commit != args.expected_inference_git_commit:
        raise ValueError("inference commit differs from post-score authorization")
    if (
        failed_commit != EXPECTED_FAILED_FINALIZER_COMMIT
        or failed.get("recovery_provenance", {}).get("git_commit") != failed_commit
    ):
        raise ValueError("failed finalizer commit differs")
    if new_commit != args.expected_finalizer_git_commit:
        raise ValueError("new finalizer commit differs from authorization")
    require_clean_process_start(
        {"provenance": finalizer_provenance},
        new_commit,
        description="Stage 2 post-score recovery finalizer",
    )

    card = _read_json(args.inference_card, "immutable inference card")
    _validate_inference_card(card, args, authorization)
    prediction_set = card.get("prediction_set")
    identities = card.get("identities")
    prediction_sha = _require_sha256(
        recovery.get("prediction_set_sha256"), "prediction-set digest"
    )
    if (
        not isinstance(prediction_set, dict)
        or not isinstance(identities, dict)
        or prediction_sha != EXPECTED_PREDICTION_SET_SHA256
        or prediction_set.get("sha256") != prediction_sha
        or identities.get("prediction_set_sha256") != prediction_sha
        or recovery.get("prediction_count") != EXPECTED_PREDICTION_COUNT
        or prediction_set.get("count") != EXPECTED_PREDICTION_COUNT
        or identities.get("prediction_count") != EXPECTED_PREDICTION_COUNT
    ):
        raise ValueError("prediction set differs from post-score authorization")

    evidence = {
        "kind": POST_SCORE_RECOVERY_KIND,
        "action": POST_SCORE_RECOVERY_ACTION,
        "reason_code": POST_SCORE_RECOVERY_REASON_CODE,
        "authorization_basis": POST_SCORE_RECOVERY_AUTHORIZATION_BASIS,
        "authorization_sha256": sha256_file(args.recovery_authorization),
        "failed_journal_sha256": journal_sha,
        "failed_ledger_sha256": ledger_sha,
        "failed_queue_sha256": queue_sha,
        "failed_queue_log_sha256": queue_log_sha,
        "first_recovery_authorization_sha256": first_auth_sha,
        "first_failed_journal_sha256": first_failed_sha,
        "rerun_log_sha256": rerun_log_sha,
        "inference_card_sha256": inference_card_sha,
        "gate_sha256": gate_sha,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": recovery["config_sha256"],
        "manifest_sha256": recovery["manifest_sha256"],
        "split_sha256": recovery["split_sha256"],
        "manifest_audit_sha256": manifest_audit_sha,
        "ordered_source_gt_sha256": ordered_gt_sha,
        **code_hashes,
        "prediction_set_sha256": prediction_sha,
        "prediction_count": EXPECTED_PREDICTION_COUNT,
        "old_inference_git_commit": old_commit,
        "failed_finalizer_git_commit": failed_commit,
        "new_finalizer_git_commit": new_commit,
        **expected_facts,
        "score_attempts": 2,
    }
    return card, evidence


def _validate_per_image_output_hashes(
    bundle: Path, outputs: Mapping[str, Any]
) -> None:
    scoring = outputs.get("scoring_evidence")
    if not isinstance(scoring, dict):
        raise ValueError("published Stage 2 bundle lacks scoring identities")
    for table in ("A", "A_prime"):
        record = scoring.get(table)
        if not isinstance(record, dict):
            raise ValueError(f"published Stage 2 bundle lacks {table} evidence")
        name = record.get("per_image_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"published Stage 2 {table} per-image path is unsafe")
        per_image = bundle / "results" / name
        if sha256_file(per_image) != record.get("per_image_sha256"):
            raise ValueError(f"published Stage 2 {table} per-image evidence changed")


def _recover_published_bundle(args: argparse.Namespace, journal: Path) -> bool:
    if not args.bundle_out.exists():
        return False
    delivery = _read_json(args.bundle_out / "delivery.json", "published Stage 2 delivery")
    if delivery.get("kind") != "stage2_unet_heldout_delivery":
        raise ValueError("existing Stage 2 bundle has the wrong identity")
    if delivery.get("gate_sha256") != sha256_file(args.gate) or delivery.get("inference_card_sha256") != sha256_file(args.inference_card):
        raise ValueError("published Stage 2 bundle input identities changed")
    outputs = delivery.get("results")
    if not isinstance(outputs, dict):
        raise ValueError("published Stage 2 bundle lacks output identities")
    for relative, key in (
        ("results/summary.csv", "summary_sha256"),
        ("results/tables.md", "tables_md_sha256"),
        ("results/tables.tex", "tables_tex_sha256"),
    ):
        if sha256_file(args.bundle_out / relative) != outputs.get(key):
            raise ValueError(f"published Stage 2 output changed: {relative}")
    qa = outputs.get("qa")
    if not isinstance(qa, list) or len(qa) != 11:
        raise ValueError("published Stage 2 QA universe changed")
    if sha256_file(args.bundle_out / "qa/training_curves.png") != qa[0].get("curve_sha256"):
        raise ValueError("published Stage 2 curve changed")
    for record in qa[1:]:
        ordinal = int(record.get("ordinal", -1))
        path = args.bundle_out / "qa/overlays" / f"overlay_{ordinal:02d}.png"
        if not 1 <= ordinal <= 10 or sha256_file(path) != record.get("sha256"):
            raise ValueError("published Stage 2 overlay changed")
    _validate_per_image_output_hashes(args.bundle_out, outputs)
    _publish_journal(
        journal,
        "complete",
        delivery_sha256=sha256_file(args.bundle_out / "delivery.json"),
        recovered_after_atomic_publish=True,
    )
    print(json.dumps(delivery, sort_keys=True))
    return True


def main() -> None:
    args = parse_args()
    command = list(sys.argv)
    provenance = run_provenance(command, capture_timing="process_start")
    require_clean_process_start(
        {"provenance": provenance},
        args.expected_finalizer_git_commit,
        description="Stage 2 finalizer",
    )
    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    journal = _journal_path(args.bundle_out)
    lock = _lock_path(args.bundle_out)
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Stage 2 finalizer lock exists; audit before retry: {lock}") from exc
    journal_owned = False
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        prior = (
            _read_json(journal, "Stage 2 transaction journal")
            if journal.exists()
            else None
        )
        recovery_arguments = (
            args.recovery_authorization,
            args.prior_recovery_authorization,
            args.recovery_ledger,
            args.recovery_log,
            args.rerun_log,
        )
        if any(value is not None for value in recovery_arguments) and (
            prior is None
            or prior.get("status") != "failed"
            or args.bundle_out.exists()
        ):
            raise RuntimeError(
                "recovery arguments require the matching failed journal and absent bundle"
            )
        if _recover_published_bundle(args, journal):
            return
        authorization = None
        recovered_inference = None
        recovery_evidence = None
        prior_recovery_evidence = None
        post_score_recovery = False
        if prior is not None:
            if prior.get("status") == "failed":
                authorization = validate_gate(args)
                if args.recovery_authorization is None:
                    raise RuntimeError(
                        "failed Stage 2 journal requires an explicit recovery authorization"
                    )
                recovery_kind = _read_json(
                    args.recovery_authorization, "Stage 2 recovery authorization"
                ).get("kind")
                if recovery_kind == POST_SCORE_RECOVERY_KIND:
                    post_score_recovery = True
                    prior_recovery_evidence = prior.get("recovery")
                    recovered_inference, recovery_evidence = (
                        _authorize_failed_scoring_recovery(
                            args, journal, authorization, provenance
                        )
                    )
                else:
                    recovered_inference, recovery_evidence = _authorize_failed_recovery(
                        args, journal, authorization, provenance
                    )
                archive = _archive_failed_journal(
                    journal, recovery_evidence["failed_journal_sha256"]
                )
                recovery_evidence["failed_journal_archive"] = archive.name
                if post_score_recovery:
                    _publish_journal(
                        journal,
                        "postscore_recovery_started",
                        postscore_recovery_command=command,
                        postscore_recovery_provenance=provenance,
                        postscore_recovery=recovery_evidence,
                        test_scoring_started=True,
                        score_attempts=1,
                        rerun_inference=False,
                    )
                else:
                    _publish_journal(
                        journal,
                        "started",
                        recovery_command=command,
                        recovery_provenance=provenance,
                        recovery=recovery_evidence,
                        test_scoring_started=False,
                    )
                journal_owned = True
            elif prior.get("status") != "preempted_before_scoring":
                raise RuntimeError(
                    "prior Stage 2 attempt is not automatically retryable; record an "
                    "approved bug-forced rerun before changing the journal"
                )
        if recovered_inference is None:
            _publish_journal(
                journal,
                "started",
                command=command,
                test_scoring_started=False,
                provenance=provenance,
            )
            journal_owned = True
        phase = "before_scoring"
        if authorization is None:
            authorization = validate_gate(args)
        _publish_journal(
            journal,
            "inference_revalidated" if post_score_recovery else "inference_started",
            gate_sha256=authorization["gate_sha256"],
            checkpoint_sha256=authorization["checkpoint_sha256"],
            test_inference_started=True,
            test_scoring_started=post_score_recovery,
            rerun_inference=recovered_inference is None,
            score_attempts=1 if post_score_recovery else 0,
        )
        inference = (
            recovered_inference
            if recovered_inference is not None
            else ensure_inference(args, authorization)
        )
        phase = "scoring"
        _publish_journal(
            journal,
            "scoring_started",
            gate_sha256=authorization["gate_sha256"],
            inference_card_sha256=sha256_file(args.inference_card),
            test_scoring_started=True,
            score_attempts=2 if post_score_recovery else 1,
            rerun_scoring=post_score_recovery,
        )
        bundle_stage = Path(
            tempfile.mkdtemp(prefix=f".{args.bundle_out.name}-", dir=args.bundle_out.parent)
        )
        results_stage = bundle_stage / "results"
        qa_stage = bundle_stage / "qa"
        results_stage.mkdir()
        qa_stage.mkdir()
        retained_staging_path = None
        try:
            _run(_score_command(args, results_stage, inference, authorization, table="A", gt_column="mask_path"))
            _run(_score_command(args, results_stage, inference, authorization, table="A_prime", gt_column="quality_mask_path"))
            scoring_evidence = validate_scoring(results_stage, args)
            _run([sys.executable, str(Path(__file__).with_name("make_tables.py")), "--summary", str(results_stage / "summary.csv"), "--out", str(results_stage)])
            qa_records = _write_qa(args, qa_stage, authorization)
            outputs = {
                "summary_sha256": sha256_file(results_stage / "summary.csv"),
                "tables_md_sha256": sha256_file(results_stage / "tables.md"),
                "tables_tex_sha256": sha256_file(results_stage / "tables.tex"),
                "scoring_evidence": scoring_evidence,
                "qa": qa_records,
            }
            final = {
                "schema_version": 1,
                "kind": "stage2_unet_heldout_delivery",
                "gate_sha256": authorization["gate_sha256"],
                "inference_card_sha256": sha256_file(args.inference_card),
                "checkpoint_sha256": authorization["checkpoint_sha256"],
                "best_epoch": authorization["run"]["best_epoch"],
                "wall_time_seconds": authorization["run"]["elapsed_seconds"],
                "selected_validation_threshold": authorization["threshold"],
                "score_attempts": 2 if post_score_recovery else 1,
                "results": outputs,
                "provenance": provenance,
            }
            if recovery_evidence is not None:
                final["recovery"] = dict(recovery_evidence)
            if post_score_recovery:
                final["prior_recovery"] = prior_recovery_evidence
                final["postscore_recovery"] = dict(recovery_evidence)
            atomic_json_once(bundle_stage / "delivery.json", final)
            os.replace(bundle_stage, args.bundle_out)
        except BaseException:
            if post_score_recovery:
                retained_staging_path = str(bundle_stage)
            else:
                shutil.rmtree(bundle_stage, ignore_errors=True)
            raise
        _publish_journal(
            journal,
            "complete",
            delivery_sha256=sha256_file(args.bundle_out / "delivery.json"),
            test_scoring_started=True,
            score_attempts=2 if post_score_recovery else 1,
        )
        print(json.dumps(final, sort_keys=True))
    except SystemExit as exc:
        if journal_owned:
            if int(exc.code or 0) == 75 and locals().get("phase") == "before_scoring":
                _publish_journal(journal, "preempted_before_scoring", test_scoring_started=False)
            else:
                _publish_journal(
                    journal,
                    "failed",
                    phase=locals().get("phase", "startup"),
                    test_scoring_started=(
                        post_score_recovery
                        or locals().get("phase") == "scoring"
                    ),
                    score_attempts=2 if post_score_recovery else 1,
                    retained_staging_path=locals().get("retained_staging_path"),
                )
        raise
    except BaseException as exc:
        if journal_owned:
            _publish_journal(
                journal,
                "failed",
                phase=locals().get("phase", "startup"),
                test_scoring_started=(
                    post_score_recovery or locals().get("phase") == "scoring"
                ),
                score_attempts=2 if post_score_recovery else 1,
                retained_staging_path=locals().get("retained_staging_path"),
                error=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
