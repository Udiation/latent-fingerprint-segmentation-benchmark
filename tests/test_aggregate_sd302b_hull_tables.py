from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_sd302b_hull_tables.py"
SPEC = importlib.util.spec_from_file_location("aggregate_sd302b_hull_tables", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TARGET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TARGET
SPEC.loader.exec_module(TARGET)


def _write_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value)


def test_consensus_policy_is_write_once_and_requires_independent_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TARGET, "outside_repository", lambda _path: None)
    path = tmp_path / "policy.json"
    written = TARGET.write_policy(path)
    assert written["status"] == "PREREGISTERED"
    assert TARGET.authenticate_policy(path, written["sha256"])["selection_signal"] is False
    with pytest.raises(TARGET.AggregateError, match="SHA-256 differs"):
        TARGET.authenticate_policy(path, "0" * 64)
    with pytest.raises(TARGET.AggregateError, match="overwrite"):
        TARGET.write_policy(path)


def test_retained_masks_must_match_both_receipt_identities(tmp_path: Path) -> None:
    score = tmp_path / "method" / "score"
    masks = tmp_path / "method" / "masks"
    score.mkdir(parents=True)
    values = {
        "image-a": np.array([[0, 255], [0, 255]], dtype=np.uint8),
        "image-b": np.array([[255, 255], [0, 0]], dtype=np.uint8),
    }
    for image_id, value in values.items():
        _write_png(masks / f"{image_id}.png", value)
    records = [
        {
            "image_id": image_id,
            "sha256": TARGET.sha256_file(masks / f"{image_id}.png"),
            "bytes": (masks / f"{image_id}.png").stat().st_size,
        }
        for image_id in sorted(values)
    ]
    receipt_path = score / "adoption_receipt.json"
    receipt = {
        "frozen_prediction": {
            "prediction_set_sha256": TARGET._legacy_prediction_digest(records),
            "prediction_records_sha256": TARGET.canonical_sha256(records),
        }
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    found, status = TARGET.authenticate_mask_sets(
        {"method": receipt}, {"method": receipt_path}, set(values), None
    )
    assert found is not None and status["status"] == "AUTHENTICATED"
    receipt["frozen_prediction"]["prediction_set_sha256"] = TARGET.canonical_sha256(records)
    found, status = TARGET.authenticate_mask_sets(
        {"method": receipt}, {"method": receipt_path}, set(values), None
    )
    assert found is not None and status["status"] == "AUTHENTICATED"
    changed = values["image-a"].copy()
    changed[0, 0] = 255
    _write_png(masks / "image-a.png", changed)
    with pytest.raises(TARGET.AggregateError, match="none authenticate"):
        TARGET.authenticate_mask_sets(
            {"method": receipt}, {"method": receipt_path}, set(values), None
        )


def test_reference_rows_are_lower_bound_metrics_with_physical_dilation(tmp_path: Path) -> None:
    root = tmp_path / "verified"
    images = root / "images"
    hulls = root / "hulls"
    minutiae = root / "minutiae"
    images.mkdir(parents=True)
    hulls.mkdir()
    minutiae.mkdir()
    rows = []
    mapped = []
    for image_id, ppi, size, capture, label, device in (
        ("plain", 500, 21, "slap", "Plain (segmented slap)", "S"),
        ("rolled", 1000, 42, "roll", "Rolled", "V"),
    ):
        image = np.zeros((size, size), dtype=np.uint8)
        hull = np.zeros_like(image)
        start, stop = size // 3, 2 * size // 3 + 1
        hull[start:stop, start:stop] = 255
        _write_png(images / f"{image_id}.png", image)
        _write_png(hulls / f"{image_id}.png", hull)
        with (minutiae / f"{image_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["x_px", "y_px"])
            writer.writeheader()
            writer.writerow({"x_px": size // 2, "y_px": size // 2})
        rows.append({
            "image_id": image_id, "image_path": f"images/{image_id}.png",
            "hull_path": f"hulls/{image_id}.png",
            "minutiae_path": f"minutiae/{image_id}.csv", "subject_id": image_id,
            "split": "test", "group": "baseline", "device": device,
            "capture": capture, "width": size, "height": size, "ppi": ppi,
        })
        mapped.append({
            "image_id": image_id, "subject_id": image_id, "device": device,
            "source_capture": capture, "report_capture": label,
        })
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    references, evidence_hash = TARGET.build_reference_frames(manifest, mapped)
    assert len(evidence_hash) == 64
    foreground = references["all_foreground"]
    assert np.allclose(foreground.mc, 1.0) and np.allclose(foreground.hr, 1.0)
    hull = references["hull_as_prediction"]
    assert np.allclose(hull.par, 1.0) and np.allclose(hull.ohf, 0.0)
    dilated = references["hull_dilated_10px_500ppi"]
    assert (dilated.par > 1.0).all()
    assert (dilated.predicted_area_fraction > hull.predicted_area_fraction).all()


def test_table_builder_separates_segmented_slaps_and_rolls() -> None:
    records = [
        {
            "image_id": "plain", "subject_id": "s1", "group": "baseline",
            "device": "S", "capture": "slap", "report_capture": "Plain (segmented slap)",
            "n_minutiae": 2, "hull_pixels": 10, "image_pixels": 20,
            "mc": 0.5, "hr": 0.6, "par": 1.2, "ohf": 0.5,
            "predicted_area_fraction": 0.6, "method": "left_zero",
        },
        {
            "image_id": "rolled", "subject_id": "s2", "group": "baseline",
            "device": "V", "capture": "roll", "report_capture": "Rolled",
            "n_minutiae": 2, "hull_pixels": 10, "image_pixels": 20,
            "mc": 1.0, "hr": 0.8, "par": 1.4, "ohf": 0.4,
            "predicted_area_fraction": 0.7, "method": "left_zero",
        },
    ]
    frame = pd.DataFrame(records)
    frames = {"left": [(None, frame)]}
    universe = {
        "left": {
            "display_name": "Left", "family": "Synthetic", "training_data": "none",
            "status": "score", "reason": "", "seed_regime": "unseeded",
        },
        "pending": {
            "display_name": "Pending", "family": "Synthetic", "training_data": "none",
            "status": "N/A", "reason": "not available", "seed_regime": "seeded",
        },
    }
    references = {}
    for method in TARGET.REFERENCE_METHODS:
        value = frame.copy()
        value["method"] = method
        value["mc"] = 1.0
        value["hr"] = 1.0
        value["par"] = 1.0
        value["ohf"] = 0.0
        value["predicted_area_fraction"] = 0.5
        references[method] = value
    plain, rolled, devices, _ = TARGET.build_tables(universe, frames, references)
    assert {row["capture"] for row in plain} == {"Plain (segmented slap)"}
    assert {row["capture"] for row in rolled} == {"Rolled"}
    assert {row["device"] for row in devices} == {"S", "V"}
    assert next(row for row in plain if row["method"] == "pending")["status"] == "N/A"


def test_aligned_ra_runet_pending_reason_cannot_claim_missing_target_data() -> None:
    row = TARGET._na_row(
        "ra_runet_aligned",
        {
            "display_name": "RA-RUnet (aligned)",
            "family": "Encoder-decoder",
            "training_data": "latent to live-scan unlabelled adaptation",
            "reason": "stale upstream reason saying target data were absent",
        },
        capture="Rolled",
        device="all",
        n_images=430,
        n_subjects=40,
    )

    assert row["reason"] == TARGET.RA_RUNET_ALIGNED_PENDING_REASON
    assert "no authenticated aligned RA-RUnet checkpoint" in row["reason"]
    assert "never substitute the scored no-alignment RUnet" in row["reason"]
    assert "target data" not in row["reason"]


def test_efficiency_metadata_requires_hash_and_complete_scored_rows(tmp_path: Path) -> None:
    source = tmp_path / "table_e.csv"
    pd.DataFrame([
        {
            "method_key": "scored", "ms_per_img_mean": 12.5,
            "ms_per_img_seed_std": 0.2, "params_m_mean": 3.0,
            "params_m_seed_std": 0.0,
        },
        {
            "method_key": "all_background", "ms_per_img_mean": np.nan,
            "ms_per_img_seed_std": np.nan, "params_m_mean": 0.0,
            "params_m_seed_std": np.nan,
        },
    ]).to_csv(source, index=False)
    universe = {
        "scored": {"status": "score"},
        "pending": {"status": "N/A"},
    }

    metadata, identity = TARGET.authenticate_efficiency_table(
        source, TARGET.sha256_file(source), universe
    )

    assert metadata["scored"]["params_m_mean"] == 3.0
    assert identity["sha256"] == TARGET.sha256_file(source)
    with pytest.raises(TARGET.AggregateError, match="hash differs"):
        TARGET.authenticate_efficiency_table(source, "0" * 64, universe)
