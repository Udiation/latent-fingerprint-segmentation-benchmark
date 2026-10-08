"""Synthetic tests for immutable prediction identity and unbiased timing."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from segbench.artifact_identity import prediction_set_identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts" / "infer_masks_repro.py"
    spec = importlib.util.spec_from_file_location("infer_repro_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_full_timing_rejects_resume_tail_bias() -> None:
    module = _module()
    good = {
        "written": 3,
        "skipped_valid": 0,
        "errors": [],
        "per_image": [
            {"image_id": "a", "ms_per_img": 1.0},
            {"image_id": "b", "ms_per_img": 2.0},
            {"image_id": "c", "ms_per_img": 3.0},
        ],
        "debug_limited": False,
    }
    module.validate_full_timing(good, ["a", "b", "c"])
    bad = dict(good, written=1, skipped_valid=2, per_image=good["per_image"][-1:])
    with pytest.raises(ValueError, match="skipped existing"):
        module.validate_full_timing(bad, ["a", "b", "c"])


def test_prediction_set_hash_binds_ids_bytes_and_universe(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"synthetic-a")
    b.write_bytes(b"synthetic-b")
    first = prediction_set_identity([("b", b), ("a", a)])
    second = prediction_set_identity([("a", a), ("b", b)])
    assert first["sha256"] == second["sha256"]
    assert first["count"] == 2
    b.write_bytes(b"changed")
    assert prediction_set_identity([("a", a), ("b", b)])["sha256"] != first["sha256"]


def test_prediction_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    with pytest.raises(ValueError, match="duplicate"):
        prediction_set_identity([("same", a), ("same", b)])


def test_checkpoint_requires_exact_authorized_digest(tmp_path: Path) -> None:
    import hashlib

    module = _module()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert module.require_expected_checkpoint(checkpoint, digest) == digest
    with pytest.raises(ValueError, match="authorized identity"):
        module.require_expected_checkpoint(checkpoint, "0" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        module.require_expected_checkpoint(checkpoint, "not-a-digest")


def test_child_command_propagates_selector_amendment_and_cache(tmp_path: Path) -> None:
    module = _module()
    args = SimpleNamespace(
        config=tmp_path / "config.yaml",
        checkpoint=tmp_path / "best.pt",
        manifest=tmp_path / "manifest.csv",
        splits=tmp_path / "splits.json",
        subset="finger_latent",
        split="test",
        out=tmp_path / "masks",
        probability_out=tmp_path / "probabilities",
        device="cuda",
        memory_fraction=0.3,
        warmup=10,
        threshold_receipt=tmp_path / "selector.json",
        threshold_amendment=tmp_path / "amendment.json",
        cache_dir=tmp_path / "cache",
        seed=2,
        limit=0,
    )
    command = module._base_command(args, tmp_path / "timing.json")
    for option, expected in (
        ("--threshold-receipt", args.threshold_receipt),
        ("--threshold-amendment", args.threshold_amendment),
        ("--cache-dir", args.cache_dir),
    ):
        assert command[command.index(option) + 1] == str(expected)
