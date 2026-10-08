"""Regression tests for distinct SD302e image and SD302h annotation PPIs."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from segbench.io import load_manifest, manifest_target_shape  # noqa: E402


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG fixture and require success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_evaluator_uses_mask_ppi_and_canonical_image_grid(tmp_path: Path) -> None:
    """A perfect 500-PPI prediction remains perfect with a 1,200/1,000 pair."""
    image = np.full((120, 240), 127, dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:80, 40:160] = 255
    expected = cv2.resize(mask, (100, 50), interpolation=cv2.INTER_NEAREST)

    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    pred_path = tmp_path / "pred" / "sample.png"
    write_png(image_path, image)
    write_png(mask_path, mask)
    write_png(pred_path, expected)

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_id": "sample",
                "image_path": image_path,
                "mask_path": mask_path,
                "subject_id": "subject",
                "subset": "finger_latent",
                "ppi": 1200,
                "mask_ppi": 1000,
                "target_ppi": 500,
                "target_height": 50,
                "target_width": 100,
            }
        ]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"seed": 1, "splits": {"subject": "test"}}))
    out = tmp_path / "results"

    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_masks.py"),
        "--manifest", str(manifest),
        "--splits", str(splits),
        "--subsets", "finger_latent",
        "--method", "perfect",
        "--display-name", "Perfect",
        "--pred-template", str(tmp_path / "pred" / "{image_id}.png"),
        "--pred-ppi", "500",
        "--out", str(out),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    row = pd.read_csv(out / "summary.csv").iloc[0]
    assert row["dice_mean"] == 1.0
    assert row["mdr_mean"] == 0.0
    assert row["fdr_mean"] == 0.0


def test_trivial_references_are_scored_without_missing_predictions(tmp_path: Path) -> None:
    image = np.full((20, 20), 127, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    write_png(image_path, image)
    write_png(mask_path, mask)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{
            "image_id": "sample",
            "image_path": image_path,
            "mask_path": mask_path,
            "subject_id": "subject",
            "subset": "finger_latent",
            "ppi": 500,
        }]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"seed": 1, "splits": {"subject": "test"}}))
    for reference, expected_dice in (("all-background", 0.0), ("all-foreground", 0.4)):
        out = tmp_path / reference
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_masks.py"),
                "--manifest", str(manifest),
                "--splits", str(splits),
                "--subsets", "finger_latent",
                "--method", reference,
                "--display-name", reference,
                "--reference", reference,
                "--out", str(out),
            ],
            check=True,
            cwd=ROOT,
        )
        row = pd.read_csv(out / "summary.csv").iloc[0]
        assert int(row["n_missing"]) == 0
        assert row["dice_mean"] == pytest.approx(expected_dice)


def test_manifest_preserves_leading_zero_subject_ids(tmp_path: Path) -> None:
    """CSV type inference must not corrupt anonymous NIST subject labels."""
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{
            "image_path": "image.png",
            "mask_path": "mask.png",
            "subject_id": "00002302",
            "subset": "finger_latent",
            "ppi": 1000,
        }]
    ).to_csv(path, index=False)
    assert load_manifest(path).iloc[0]["subject_id"] == "00002302"


def test_declared_target_grid_must_be_complete_and_match_requested_ppi() -> None:
    with pytest.raises(ValueError, match="declared together"):
        manifest_target_shape(
            {"image_id": "sample", "target_ppi": 500, "target_height": 50},
            500,
        )
    with pytest.raises(ValueError, match="differs from requested"):
        manifest_target_shape(
            {
                "image_id": "sample",
                "target_ppi": 500,
                "target_height": 50,
                "target_width": 100,
            },
            600,
        )


def test_declared_target_dimensions_must_be_positive_integers() -> None:
    with pytest.raises(ValueError, match="invalid target shape"):
        manifest_target_shape(
            {
                "image_id": "sample",
                "target_ppi": 500,
                "target_height": 50.5,
                "target_width": 100,
            },
            500,
        )


def test_evaluator_keeps_distinct_table_per_image_outputs(tmp_path: Path) -> None:
    """Scoring A-prime must never overwrite Table A's per-image evidence."""
    image = np.full((20, 20), 127, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    image_path = tmp_path / "image.png"
    roi_path = tmp_path / "roi.png"
    quality_path = tmp_path / "quality.png"
    pred_path = tmp_path / "pred" / "sample.png"
    write_png(image_path, image)
    write_png(roi_path, mask)
    write_png(quality_path, mask)
    write_png(pred_path, mask)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{
            "image_id": "sample",
            "image_path": image_path,
            "mask_path": roi_path,
            "quality_mask_path": quality_path,
            "subject_id": "subject",
            "subset": "finger_latent",
            "ppi": 500,
            "mask_ppi": 500,
            "quality_mask_ppi": 500,
        }]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"seed": 1, "splits": {"subject": "test"}}))
    out = tmp_path / "results"
    common = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_masks.py"),
        "--manifest", str(manifest),
        "--splits", str(splits),
        "--subsets", "finger_latent",
        "--method", "model",
        "--display-name", "Model",
        "--pred-template", str(tmp_path / "pred" / "{image_id}.png"),
        "--pred-ppi", "500",
        "--seed", "0",
        "--out", str(out),
    ]
    subprocess.run(
        common + [
            "--gt-column", "mask_path", "--table", "A",
            "--run-id", "latent-seed0", "--training-data", "latent",
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        common + [
            "--gt-column", "quality_mask_path", "--table", "A_prime",
            "--run-id", "latent-seed0", "--training-data", "latent",
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        common + [
            "--gt-column", "mask_path", "--table", "A",
            "--run-id", "fvc-seed0", "--training-data", "FVC set B",
        ],
        check=True,
        cwd=ROOT,
    )
    assert len(list(out.glob("per_image_model_finger_latent_A_seed0_identity-*.csv"))) == 2
    assert len(list(out.glob("per_image_model_finger_latent_A_prime_seed0_identity-*.csv"))) == 1
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 3
