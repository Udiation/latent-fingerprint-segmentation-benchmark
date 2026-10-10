from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_sd302b_annotation_package.py"
SPEC = importlib.util.spec_from_file_location("build_sd302b_annotation_package", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def mapping(tmp_path: Path, *, bad_capture: bool = False) -> Path:
    rows = []
    for index in range(8):
        image = tmp_path / f"source_{index}.png"
        image.write_bytes(f"image-{index}".encode())
        capture = "Plain (segmented slap)" if index == 0 else "Rolled"
        if bad_capture and index == 0:
            capture = "unknown"
        rows.append(
            {
                "image_id": f"id-{index}",
                "subject_id": f"subject-{index % 4}",
                "split": "test",
                "device": "V",
                "report_capture": capture,
                "finger_position": "01",
                "actual_ppi": "500",
                "sd302b_image_path": str(image),
            }
        )
    path = tmp_path / "mapping.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_package_is_blind_subject_stratified_and_repeat_complete(tmp_path: Path) -> None:
    out = tmp_path / "package"
    result = MODULE.build(
        argparse.Namespace(
            mapping=mapping(tmp_path),
            out=out,
            primary_count=4,
            repeat_count=2,
            seed=17,
            dry_run=False,
        )
    )
    assert result["status"] == "PASS"
    assert result["subjects"] == 4
    assert len(list((out / "images").glob("*.png"))) == 6
    assert not list(out.rglob("*mask*.png"))
    receipt = json.loads((out / "package_receipt.json").read_text())
    assert receipt["selection_uses_predictions"] is False
    assert receipt["contains_prefilled_annotations"] is False
    with (out / "manifest_private.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    assert sum(row["is_repeat"] == "True" for row in tasks) == 2
    assert any(row["capture"] == "Plain (segmented slap)" for row in tasks)


def test_unknown_capture_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(MODULE.PackageError, match="unsupported capture"):
        MODULE.read_mapping(mapping(tmp_path, bad_capture=True))
