from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_sd302b_hull_manifest as target  # noqa: E402


def _save(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, dpi=(500, 500))


def _case(root: Path) -> tuple[Path, Path, Path, target.UniverseExpectations]:
    verified = root / "verified"
    (verified / "images").mkdir(parents=True)
    b_root = root / "images_b" / "baseline"
    direct = np.arange(12, dtype=np.uint8).reshape(3, 4)
    source = np.arange(64, dtype=np.uint8).reshape(8, 8)
    resized = np.asarray(
        Image.fromarray(source).resize((4, 4), Image.Resampling.LANCZOS)
    )
    _save(verified / "images" / "subject_S_500_slap_01.png", direct)
    _save(verified / "images" / "subject_V_1000_roll_01.png", source)
    _save(
        b_root / "S" / "500" / "slap-segmented" / "png"
        / "subject_S_500_slap_01.png",
        direct,
    )
    _save(
        b_root / "V" / "1000" / "roll" / "png"
        / "subject_V_1000_roll_01.png",
        resized,
    )
    manifest = verified / "manifest.csv"
    columns = [
        "image_id", "image_path", "hull_path", "minutiae_path", "subject_id",
        "split", "group", "device", "capture", "finger_position", "width",
        "height", "ppi",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "image_id": "subject_S_500_slap_01",
            "image_path": "images/subject_S_500_slap_01.png",
            "hull_path": "private/not-opened.png",
            "minutiae_path": "private/not-opened.csv",
            "subject_id": "subject-a", "split": "test", "group": "baseline",
            "device": "S", "capture": "slap", "finger_position": "01",
            "width": 4, "height": 3, "ppi": 500,
        })
        writer.writerow({
            "image_id": "subject_V_1000_roll_01",
            "image_path": "images/subject_V_1000_roll_01.png",
            "hull_path": "private/not-opened.png",
            "minutiae_path": "private/not-opened.csv",
            "subject_id": "subject-b", "split": "train", "group": "baseline",
            "device": "V", "capture": "roll", "finger_position": "01",
            "width": 8, "height": 8, "ppi": 1000,
        })
    splits = root / "splits.json"
    splits.write_text(
        json.dumps({
            "seed": 17,
            "splits": {"subject-a": "test", "subject-b": "train"},
        }),
        encoding="utf-8",
    )
    (verified / "provenance.json").write_text(
        json.dumps({
            "schema": "cureblock.sd302g.build-provenance.v1",
            "verification_status": "REFERENCE_VERIFIED",
            "manifest_sha256": target.sha256_file(manifest),
            "split_sha256": target.sha256_file(splits),
        }),
        encoding="utf-8",
    )
    expected = target.UniverseExpectations(
        records=2, test_records=1, native_exact=1, split_seed=17,
        subject_split_counts={"train": 1, "test": 1},
        split_counts={"train": 1, "test": 1},
        capture_counts={"roll": 1, "slap": 1},
        test_capture_counts={"slap": 1},
        device_counts={"S": 1, "V": 1},
        test_device_counts={"S": 1},
    )
    return manifest, splits, b_root, expected


def test_builds_exact_private_mapping_and_gt_blind_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, splits, b_root, expected = _case(tmp_path)
    monkeypatch.setattr(target, "outside_repository", lambda _path: None)
    out = tmp_path / "out"
    receipt = target.build(
        manifest, splits, b_root, out, expectations=expected,
        expected_manifest_sha256=None, expected_split_sha256=None,
        expected_archive_sha256=None,
    )
    assert receipt["mapping"]["records"] == 2
    assert receipt["mapping"]["native_exact_records"] == 1
    assert receipt["mapping"]["canonical_archive_500_exact_records"] == 2
    with (out / "mapping_test_private.csv").open(newline="", encoding="utf-8") as handle:
        test_rows = list(csv.DictReader(handle))
    assert [row["report_capture"] for row in test_rows] == ["Plain (segmented slap)"]
    with (out / "inference_manifest.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        inference = list(reader)
        assert "hull_path" not in (reader.fieldnames or [])
        assert "minutiae_path" not in (reader.fieldnames or [])
    assert all(row["mask_path"] == "" and row["ppi"] == "500" for row in inference)


def test_rejects_canonical_pixel_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, splits, b_root, expected = _case(tmp_path)
    bad = b_root / "V" / "1000" / "roll" / "png" / "subject_V_1000_roll_01.png"
    pixels = np.asarray(Image.open(bad)).copy()
    pixels[0, 0] ^= np.uint8(1)
    _save(bad, pixels)
    monkeypatch.setattr(target, "outside_repository", lambda _path: None)
    with pytest.raises(target.MappingError, match="canonical archive-500 pixels differ"):
        target.build(
            manifest, splits, b_root, tmp_path / "out", expectations=expected,
            expected_manifest_sha256=None, expected_split_sha256=None,
            expected_archive_sha256=None,
        )


def test_rejects_frozen_split_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, splits, b_root, expected = _case(tmp_path)
    payload = json.loads(splits.read_text(encoding="utf-8"))
    payload["splits"]["subject-a"] = "train"
    splits.write_text(json.dumps(payload), encoding="utf-8")
    verified = manifest.parent
    provenance = json.loads((verified / "provenance.json").read_text(encoding="utf-8"))
    provenance["split_sha256"] = target.sha256_file(splits)
    (verified / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    monkeypatch.setattr(target, "outside_repository", lambda _path: None)
    with pytest.raises(target.MappingError, match="split disagreement|split policy changed"):
        target.build(
            manifest, splits, b_root, tmp_path / "out", expectations=expected,
            expected_manifest_sha256=None, expected_split_sha256=None,
            expected_archive_sha256=None,
        )
