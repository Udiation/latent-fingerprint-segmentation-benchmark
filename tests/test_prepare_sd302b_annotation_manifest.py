from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_sd302b_annotation_manifest as target  # noqa: E402


def test_builds_mapping_only_primary_and_repeat_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mapping"
    root.mkdir()
    source = root / "mapping_test_private.csv"
    rows = []
    for index, (capture, device) in enumerate(
        (("slap", "R"), ("slap", "S"), ("roll", "U"), ("roll", "V")),
        start=1,
    ):
        rows.append({
            "image_id": f"image-{index}", "subject_id": f"subject-{index}",
            "split": "test", "group": "baseline", "device": device,
            "source_capture": capture,
            "report_capture": "Rolled" if capture == "roll" else "Plain (segmented slap)",
            "canonical_pixels_equal": "True", "sd302b_image_path": f"/private/{index}.png",
            "sd302b_width": "10", "sd302b_height": "11", "actual_ppi": "500",
        })
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = root / "mapping_receipt.json"
    receipt.write_text(json.dumps({
        "kind": "sd302b_sd302g_exact_mapping_receipt_v1",
        "status": "REFERENCE_VERIFIED_MAPPING",
        "outputs": {"mapping_test_private.csv": {"sha256": target.sha256_file(source)}},
    }), encoding="utf-8")
    monkeypatch.setattr(target, "outside_repository", lambda _path: None)
    output = tmp_path / "annotation.csv"
    result = target.build(
        receipt, output,
        primary_quotas={("slap", "R"): 1, ("slap", "S"): 1,
                        ("roll", "U"): 1, ("roll", "V"): 1},
        repeat_quotas={("slap", "R"): 1, ("roll", "V"): 1},
        expected_test_records=4,
    )
    assert result["selection"] == {
        **result["selection"], "primary_images": 4, "repeat_tasks": 2, "total_tasks": 6,
    }
    with output.open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))
    assert len(tasks) == 6
    assert sum(row["role"] == "repeat" for row in tasks) == 2
    assert all(row["source_image_path"].startswith("/private/") for row in tasks)
    assert not (tmp_path / "images").exists()
