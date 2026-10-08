"""Synthetic tests for ground-truth-blind classical mask generation."""
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


def test_predict_classical_never_requires_test_ground_truth(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), np.full((32, 32), 127, dtype=np.uint8))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_id": "synthetic_test",
                "image_path": str(image_path),
                "mask_path": "",
                "subject_id": "S001",
                "subset": "finger_latent",
                "ppi": 500,
            }
        ]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"seed": 7, "splits": {"S001": "test"}}), encoding="utf-8"
    )
    model = RandomForestClassifier(n_estimators=2, max_depth=1, random_state=0)
    model.fit(
        np.vstack([np.zeros((2, 8)), np.ones((2, 8))]),
        np.array([False, False, True, True]),
    )
    model_path = tmp_path / "rf.joblib"
    joblib.dump(model, model_path)
    import hashlib

    model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    scalar_count = fitted_scalar_count(model)
    training_card = tmp_path / "rf.json"
    training_card.write_text(
        json.dumps(
            {
                "method": "rf",
                "seed": 0,
                "model_sha256": model_digest,
                "debug_limited": False,
                "fitted_scalar_count": scalar_count,
                "parameter_semantics": "synthetic fitted inference scalars",
                "selected_parameters": {"max_depth": 1},
                "selected_validation_balanced_accuracy": 1.0,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "predictions"
    run_card = tmp_path / "run.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "predict_classical.py"),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--model",
        str(model_path),
        "--training-card",
        str(training_card),
        "--method",
        "rf",
        "--display-name",
        "Synthetic RF",
        "--seed",
        "0",
        "--output-root",
        str(output),
        "--run-card",
        str(run_card),
        "--warmup",
        "0",
        "--progress-every",
        "1",
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    # A second invocation validates and resumes the existing binary mask.
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    prediction = cv2.imread(
        str(output / "synthetic_test.png"), cv2.IMREAD_GRAYSCALE
    )
    assert prediction is not None and prediction.shape == (32, 32)
    assert set(np.unique(prediction).tolist()).issubset({0, 255})
    payload = json.loads(run_card.read_text(encoding="utf-8"))
    assert payload["prediction_count"] == 1
    assert payload["missing_count"] == 0
    assert payload["device"] == "CPU"
    assert payload["fitted_scalar_count"] == scalar_count
    assert payload["params_m"] == scalar_count / 1_000_000.0
