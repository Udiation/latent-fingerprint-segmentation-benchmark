"""Synthetic-only tests for the public one-command reproduction contract."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHON"] = sys.executable
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=check,
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image_id", "image_path", "mask_path", "quality_mask_path",
                "subject_id", "subset", "ppi", "examiner_value",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "syn_case",
                "image_path": str(tmp_path / "image.png"),
                "mask_path": str(tmp_path / "mask.png"),
                "quality_mask_path": str(tmp_path / "quality.png"),
                "subject_id": "syn_subject",
                "subset": "finger_latent",
                "ppi": "500",
                "examiner_value": "VALUE",
            }
        )
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"seed": 42, "splits": {"syn_subject": "test"}}),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    return manifest, splits, predictions


def _plan(
    tmp_path: Path,
    table: str,
    manifest: Path,
    splits: Path,
    predictions: Path,
) -> Path:
    payload: dict[str, object] = {
        "schema": "segbench.reproduction-plan.v1",
        "table": table,
        "source_manifest": str(manifest),
        "splits": str(splits),
        "subset": "finger_latent",
        "split": "test",
        "entries": [
            {
                "run_key": "synthetic",
                "method": "synthetic_method",
                "display_name": "Synthetic method",
                "runner": {
                    "kind": "existing_predictions",
                    "pred_template": str(predictions / "{image_id}.png"),
                    "pred_ppi": 500,
                },
            }
        ],
    }
    if table == "B1":
        payload.pop("splits")
        payload["subset"] = "finger_exemplar"
        payload["evaluation_manifest"] = str(manifest)
    if table == "A_prime":
        receipt = tmp_path / "table_A_receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "segbench.reproduction-receipt.v1",
                    "status": "PASS",
                    "table": "A",
                    "entries": [
                        {
                            "run_key": "synthetic",
                            "method": "synthetic_method",
                            "prediction_template": str(
                                (predictions / "{image_id}.png").resolve()
                            ),
                            "pred_ppi": 500.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        payload["table_a_receipt"] = str(receipt)
        payload["table_a_receipt_sha256"] = hashlib.sha256(
            receipt.read_bytes()
        ).hexdigest()
    if table == "D":
        payload["evaluation"] = {"kind": "mask", "gt_column": "mask_path"}
    path = tmp_path / f"plan_{table}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("table", "script", "evaluator"),
    (
        ("A", "reproduce_table_A.sh", "evaluate_masks.py"),
        ("A_prime", "reproduce_table_A_prime.sh", "evaluate_masks.py"),
        ("B1", "reproduce_table_B1.sh", "evaluate_hull.py"),
        ("D", "reproduce_table_D.sh", "evaluate_masks.py"),
    ),
)
def test_all_entrypoints_dispatch_real_evaluators_in_dry_run(
    tmp_path: Path, table: str, script: str, evaluator: str
) -> None:
    manifest, splits, predictions = _inputs(tmp_path)
    plan = _plan(tmp_path, table, manifest, splits, predictions)
    completed = _run(
        script,
        "--plan", str(plan),
        "--output", str(tmp_path / "out"),
        "--dry-run",
    )
    receipt = json.loads(completed.stdout)
    command = receipt["entries"][0]["commands"][-1]["argv"]
    assert receipt["status"] == "DRY_RUN"
    assert any(str(part).endswith(evaluator) for part in command)
    assert "--help" not in command


def test_config_file_supplies_plan_and_output(tmp_path: Path) -> None:
    manifest, splits, predictions = _inputs(tmp_path)
    plan = _plan(tmp_path, "A", manifest, splits, predictions)
    config = tmp_path / "reproduction.env"
    config.write_text(
        f"REPRO_PLAN={plan}\nREPRO_OUTPUT={tmp_path / 'out'}\n",
        encoding="utf-8",
    )
    completed = _run("reproduce_table_A.sh", "--config", str(config), "--dry-run")
    assert json.loads(completed.stdout)["status"] == "DRY_RUN"


def test_deep_checkpoint_dry_run_is_hash_bound_and_uses_real_inference(
    tmp_path: Path,
) -> None:
    manifest, splits, _ = _inputs(tmp_path)
    checkpoint = tmp_path / "frozen.bin"
    checkpoint.write_bytes(b"synthetic checkpoint placeholder")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = tmp_path / "config.yaml"
    selector = tmp_path / "selector.json"
    amendment = tmp_path / "amendment.json"
    for path in (config, selector, amendment):
        path.write_text("{}\n", encoding="utf-8")
    plan = tmp_path / "deep.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "segbench.reproduction-plan.v1",
                "table": "A",
                "source_manifest": str(manifest),
                "splits": str(splits),
                "subset": "finger_latent",
                "entries": [
                    {
                        "run_key": "deep_seed0",
                        "method": "unet",
                        "display_name": "U-Net",
                        "runner": {
                            "kind": "deep_checkpoint",
                            "config": str(config),
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": digest,
                            "threshold_receipt": str(selector),
                            "threshold_amendment": str(amendment),
                            "device": "cpu",
                            "seed": 0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    completed = _run(
        "reproduce_table_A.sh",
        "--plan", str(plan),
        "--output", str(tmp_path / "out"),
        "--dry-run",
    )
    receipt = json.loads(completed.stdout)
    inference = receipt["entries"][0]["commands"][0]["argv"]
    assert any(str(part).endswith("infer_masks.py") for part in inference)
    assert "--threshold-receipt" in inference
    assert "--threshold-amendment" in inference

    bad = json.loads(plan.read_text(encoding="utf-8"))
    bad["entries"][0]["runner"]["checkpoint_sha256"] = "0" * 64
    plan.write_text(json.dumps(bad), encoding="utf-8")
    rejected = _run(
        "reproduce_table_A.sh",
        "--plan", str(plan),
        "--output", str(tmp_path / "out"),
        "--dry-run",
        check=False,
    )
    assert rejected.returncode != 0
    assert "SHA-256 mismatch" in rejected.stderr


def test_synthetic_table_a_transaction_scores_through_common_evaluator(
    tmp_path: Path,
) -> None:
    manifest, splits, predictions = _inputs(tmp_path)
    image = np.full((32, 32), 180, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 9:23] = 255
    assert cv2.imwrite(str(tmp_path / "image.png"), image)
    assert cv2.imwrite(str(tmp_path / "mask.png"), mask)
    assert cv2.imwrite(str(tmp_path / "quality.png"), mask)
    assert cv2.imwrite(str(predictions / "syn_case.png"), mask)
    plan = _plan(tmp_path, "A", manifest, splits, predictions)
    output = tmp_path / "out"
    completed = _run(
        "reproduce_table_A.sh", "--plan", str(plan), "--output", str(output)
    )
    assert "reproduction PASS" in completed.stdout
    summary = pd.read_csv(output / "results" / "A" / "synthetic" / "summary.csv")
    overall = summary[summary["group_value"] == "overall"].iloc[0]
    assert overall["dice_mean"] == pytest.approx(1.0)
    assert int(overall["n_missing"]) == 0
    receipt = json.loads((output / "reproduce_A.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"


def test_execution_rejects_incomplete_prediction_coverage(tmp_path: Path) -> None:
    manifest, splits, predictions = _inputs(tmp_path)
    image = np.zeros((8, 8), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "image.png"), image)
    assert cv2.imwrite(str(tmp_path / "mask.png"), image)
    assert cv2.imwrite(str(tmp_path / "quality.png"), image)
    plan = _plan(tmp_path, "A", manifest, splits, predictions)
    rejected = _run(
        "reproduce_table_A.sh",
        "--plan", str(plan),
        "--output", str(tmp_path / "out"),
        check=False,
    )
    assert rejected.returncode != 0
    assert "prediction coverage is incomplete" in rejected.stderr


def test_table_a_rejects_ground_truth_or_exclusion_override(tmp_path: Path) -> None:
    manifest, splits, predictions = _inputs(tmp_path)
    plan = _plan(tmp_path, "A", manifest, splits, predictions)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["evaluation"] = {
        "kind": "mask",
        "gt_column": "quality_mask_path",
        "group_by": "examiner_value",
        "exclude": [],
    }
    plan.write_text(json.dumps(payload), encoding="utf-8")
    rejected = _run(
        "reproduce_table_A.sh",
        "--plan", str(plan),
        "--output", str(tmp_path / "out"),
        "--dry-run",
        check=False,
    )
    assert rejected.returncode != 0
    assert "ground truth, grouping, and exclusions are frozen" in rejected.stderr
