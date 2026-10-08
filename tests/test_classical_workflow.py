"""Synthetic train/validation-only tests for the classical workflow."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from segbench.models import fitted_scalar_count


ROOT = Path(__file__).resolve().parents[1]


def _row(root: Path, subject: str, split: str) -> dict[str, object]:
    yy, xx = np.mgrid[:64, :64]
    mask = (xx > 12) & (xx < 51) & (yy > 10) & (yy < 53)
    image = np.where(mask, 45, 205).astype(np.uint8)
    image_path = root / f"{subject}.png"
    mask_path = root / f"{subject}_mask.png"
    assert cv2.imwrite(str(image_path), image)
    if split != "test":
        assert cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    return {
        "image_id": f"syn_{subject}",
        "image_path": str(image_path),
        # The held-out path deliberately does not exist. Development must not
        # try to open it, and prediction must be annotation-blind.
        "mask_path": str(mask_path),
        "subject_id": subject,
        "subset": "finger_latent",
        "ppi": 500,
        "split": split,
    }


def test_fitted_scalar_count_is_explicit_and_deterministic() -> None:
    model = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=4)
    values = np.arange(64, dtype=float).reshape(8, 8)
    labels = np.array([False, False, False, False, True, True, True, True])
    model.fit(values, labels)
    expected = 0
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        expected += sum(
            state["nodes"][name].size for name in state["nodes"].dtype.names
        )
        expected += state["values"].size
    assert fitted_scalar_count(model) == expected
    assert fitted_scalar_count(model) == expected  # no mutable/double-count state


def test_classical_training_never_opens_test_gt_and_resumes(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, "0001", "train"),
        _row(tmp_path, "0002", "train"),
        _row(tmp_path, "0003", "val"),
        _row(tmp_path, "0004", "test"),
    ]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "split"} for row in rows]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "seed": 42,
                "splits": {row["subject_id"]: row["split"] for row in rows},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "classical"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_classical.py"),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--train-subset",
        "finger_latent",
        "--methods",
        "rf",
        "--out",
        str(out),
        "--blocks-per-class",
        "4",
        "--max-train-blocks",
        "16",
        "--max-val-blocks",
        "8",
        "--tune-blocks",
        "8",
        "--n-jobs",
        "1",
    ]
    first = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    second = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert "no test rows were opened" in first.stdout
    assert "resuming verified frozen model" in second.stdout
    model_path = out / "models" / "rf_finger_latent.joblib"
    card_path = out / "cards" / "rf_finger_latent.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["train_images"] == 2
    assert card["validation_images"] == 1
    assert card["debug_limited"] is False
    assert card["parameter_semantics"].startswith("numeric scalars")
    assert card["fitted_scalar_count"] == fitted_scalar_count(joblib.load(model_path))
    assert not list(out.glob("per_image*.csv"))
    assert not (out / "summary.csv").exists()

    prediction_root = tmp_path / "predictions"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "predict_classical.py"),
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--model",
            str(model_path),
            "--training-card",
            str(card_path),
            "--method",
            "rf",
            "--display-name",
            "Synthetic RF",
            "--seed",
            "0",
            "--output-root",
            str(prediction_root),
            "--run-card",
            str(tmp_path / "predict.json"),
            "--warmup",
            "0",
            "--progress-every",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (prediction_root / "syn_0004.png").is_file()

    unsafe = subprocess.run(
        [*command, "--eval-subsets", "finger_latent"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert unsafe.returncode != 0
    assert "never opens test data" in unsafe.stderr
