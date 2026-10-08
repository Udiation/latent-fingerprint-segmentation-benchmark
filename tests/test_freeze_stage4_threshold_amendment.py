"""Synthetic-only tests for the write-once threshold-policy amendment."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_stage4_threshold_amendment.py"
SPEC = importlib.util.spec_from_file_location("threshold_amendment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base() -> bytes:
    return (
        '{"heldout_payloads_opened_during_creation":false,'
        '"kind":"stage4_preregistered_analysis_plan",'
        '"results_opened_during_creation":false,'
        '"threshold_policy":{"A_and_A_prime_reuse_identical_predictions":true,'
        '"allowed_trainable_grid":[0.3,0.4,0.5,0.6,0.7],'
        '"source":"frozen validation-selected run cards",'
        '"stage4_inference_or_rescoring_permitted":false,'
        '"stage4_reselection_permitted":false}}\n'
    ).encode()


def test_build_is_results_blind_and_exactly_scoped() -> None:
    base = _base()
    payload = MODULE.build_amendment(
        base_plan_bytes=base,
        base_plan_sha256=hashlib.sha256(base).hexdigest(),
        generator_sha256="a" * 64,
        git_commit="b" * 40,
        recorded_at_utc="2026-10-08T14:02:00Z",
    )
    assert payload["supersedes_exactly"] == ["threshold_policy"]
    assert payload["replacement_policy"]["allowed_trainable_grid"] == [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
    ]
    assert payload["required_attestations"]["validation_only"] is True
    assert not any(payload["creation_scope"].values())


def test_wrong_base_hash_or_grid_fails() -> None:
    base = _base()
    with pytest.raises(ValueError, match="SHA-256"):
        MODULE.build_amendment(
            base_plan_bytes=base,
            base_plan_sha256="0" * 64,
            generator_sha256="a" * 64,
            git_commit="b" * 40,
            recorded_at_utc="2026-10-08T14:02:00Z",
        )
    changed = base.replace(b"[0.3,0.4,0.5,0.6,0.7]", b"[0.2,0.3]")
    with pytest.raises(ValueError, match="superseded grid"):
        MODULE.build_amendment(
            base_plan_bytes=changed,
            base_plan_sha256=hashlib.sha256(changed).hexdigest(),
            generator_sha256="a" * 64,
            git_commit="b" * 40,
            recorded_at_utc="2026-10-08T14:02:00Z",
        )


def test_publication_is_write_once(tmp_path: Path) -> None:
    out = tmp_path / "amendment.json"
    assert MODULE._publish_write_once(out, b"one\n") == "created"
    assert MODULE._publish_write_once(out, b"one\n") == "verified_identical"
    with pytest.raises(FileExistsError):
        MODULE._publish_write_once(out, b"two\n")
