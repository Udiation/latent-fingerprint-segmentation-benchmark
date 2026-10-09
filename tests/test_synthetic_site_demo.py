from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from segbench.synthetic_demo import METHOD_KEYS, validate_synthetic_demo_receipt


ROOT = Path(__file__).resolve().parents[1]


def _run(*parts: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["KERAS_BACKEND"] = "torch"
    return subprocess.run(
        [sys.executable, *parts],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )


def _build(root: Path) -> dict:
    _run(
        str(ROOT / "scripts" / "make_synthetic_demo.py"),
        "--out",
        str(root),
        "--subjects",
        "4",
        "--per-subject",
        "1",
        "--seed",
        "7",
    )
    result = _run(
        str(ROOT / "scripts" / "build_synthetic_site_demo.py"),
        "--synthetic-root",
        str(root),
        "--methods",
        "rf",
        "--steps",
        "1",
        "--threads",
        "1",
    )
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary == {
        "benchmark_evidence": False,
        "complete": 1,
        "not_built": 13,
        "predictions": 3,
        "status": "PASS",
        "synthetic_only": True,
        "unavailable": 2,
    }
    return json.loads((root / "synthetic_receipt.json").read_text(encoding="utf-8"))


def test_core_only_limited_build_is_complete_and_deterministic(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _build(first_root)
    second = _build(second_root)
    first_index = validate_synthetic_demo_receipt(first_root, first)
    second_index = validate_synthetic_demo_receipt(second_root, second)
    assert tuple(first_index["methods"]) == METHOD_KEYS
    assert first_index["methods"]["rf"]["status"] == "complete"
    assert first_index["methods"]["fingernet"]["status"] == "unavailable"
    assert first_index["methods"]["segformer_sub"]["status"] == "unavailable"
    assert first_index["methods"]["unet"]["status"] == "not_built"
    assert len(first_index["predictions"]["rf"]) == 3
    assert [item["sha256"] for item in first_index["predictions"]["rf"]] == [
        item["sha256"] for item in second_index["predictions"]["rf"]
    ]
    assert first_index["methods"]["rf"] == second_index["methods"]["rf"]


def test_prediction_byte_mutation_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "synthetic"
    receipt = _build(root)
    prediction = next(
        item for item in receipt["assets"] if item.get("role") == "method_prediction"
    )
    path = root / prediction["path"]
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="bytes differ"):
        validate_synthetic_demo_receipt(root, receipt)


def test_method_record_and_source_binding_mutations_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "synthetic"
    receipt = _build(root)
    mutated = copy.deepcopy(receipt)
    rf = next(row for row in mutated["synthetic_demo"]["methods"] if row["method"] == "rf")
    rf["execution"]["seed"] = 99
    with pytest.raises(ValueError, match="record fingerprint"):
        validate_synthetic_demo_receipt(root, mutated, verify_files=False)

    mutated = copy.deepcopy(receipt)
    prediction = next(
        item for item in mutated["assets"] if item.get("role") == "method_prediction"
    )
    prediction["source_stem"] = "synthetic_unapproved"
    with pytest.raises(ValueError, match="unapproved source"):
        validate_synthetic_demo_receipt(root, mutated, verify_files=False)


def test_non_complete_method_cannot_carry_a_prediction(tmp_path: Path) -> None:
    root = tmp_path / "synthetic"
    receipt = _build(root)
    mutated = copy.deepcopy(receipt)
    rf = next(row for row in mutated["synthetic_demo"]["methods"] if row["method"] == "rf")
    rf["status"] = "unavailable"
    rf["reason"] = "invented unavailability"
    # Re-fingerprinting cannot make a contradictory status/output combination valid.
    from segbench.synthetic_demo import method_record_fingerprint

    rf["fingerprint_sha256"] = method_record_fingerprint(rf)
    with pytest.raises(ValueError, match="has fabricated predictions"):
        validate_synthetic_demo_receipt(root, mutated, verify_files=False)
