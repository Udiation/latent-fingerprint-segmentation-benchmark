"""Synthetic fail-closed tests for the post-pilot breadth-first barrier."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from segbench.artifact_identity import canonical_sha256, require_clean_process_start

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts" / "authorize_deep_jobs.py"
    spec = importlib.util.spec_from_file_location("stage3_authorize_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_selector_provenance_requires_clean_process_start() -> None:
    commit = "a" * 40
    valid = {
        "provenance": {
            "capture_timing": "process_start",
            "git_commit": commit,
            "git_dirty": False,
        }
    }
    require_clean_process_start(valid, commit, description="synthetic")
    for mutation in (
        {"capture_timing": "function_entry"},
        {"git_commit": "b" * 40},
        {"git_dirty": True},
    ):
        bad = json.loads(json.dumps(valid))
        bad["provenance"].update(mutation)
        with pytest.raises(ValueError, match="provenance"):
            require_clean_process_start(bad, commit, description="synthetic")


def test_locked_seed12_plan_is_not_a_run_queue(tmp_path: Path) -> None:
    module = _module()
    locked = {
        "launch_authorized": False,
        "jobs_locked": [{"name": name} for name in module.EXPECTED_SEED12],
    }
    path = tmp_path / "locked.yaml"
    import yaml
    path.write_text(yaml.safe_dump(locked), encoding="utf-8")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "jobs" not in payload
    assert payload["launch_authorized"] is False
    assert len(payload["jobs_locked"]) == 16


def test_ram_reservations_must_cover_exact_job_universe(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "ram.json"
    path.write_text(json.dumps({module.EXPECTED_SEED0[0]: 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="exact authorized job universe"):
        module._reservations(
            path, [*module.EXPECTED_SEED0, *module.EXPECTED_SEED12], 64.0
        )


def test_legacy_sufs_note_is_bound_to_exact_sha256(tmp_path: Path) -> None:
    module = _module()
    note = tmp_path / "legacy.md"
    note.write_text("synthetic legacy provenance\n", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(note.read_bytes()).hexdigest()
    assert module._verify_sha256(note, digest, "legacy note") == digest
    with pytest.raises(ValueError, match="differs"):
        module._verify_sha256(note, "0" * 64, "legacy note")


def test_seed12_commit_map_is_bound_to_locked_wave(tmp_path: Path) -> None:
    module = _module()
    commits = tmp_path / "commits.json"
    commits.write_text(json.dumps({method: "a" * 40 for method in module.NATIVE}))
    digest = module.sha256_file(commits)
    locked = {"expected_commits_sha256": digest}
    seed0 = {"expected_commits_sha256": digest}
    assert module._verify_commit_map_binding(seed0, locked, commits) == digest
    commits.write_text(json.dumps({method: "b" * 40 for method in module.NATIVE}))
    with pytest.raises(ValueError, match="commit map changed"):
        module._verify_commit_map_binding(seed0, locked, commits)


def test_delivery_guard_builder_preserves_frozen_scientific_job_identity(
    tmp_path: Path,
) -> None:
    module = _module()
    raw = {
        "name": "full.drunet.seed1",
        "argv": [
            "/frozen/venv/python",
            "/frozen/segbench/train_seg.py",
            "--config",
            "private_data/configs/generated/full/drunet.yaml",
            "--resume-checkpoint",
            "private_data/runs/source/best.pt",
            "--seed",
            "1",
        ],
        "env": {"FROZEN_MODEL_SNAPSHOT": "private_data/models/snapshot"},
    }
    before = json.loads(json.dumps(raw))
    guarded = module._job_with_guards(raw, runs_root=tmp_path / "runs", ram_gib=8.0)

    # A delivery/control-plane commit may append only the queue guards.  The
    # scientific trainer, config, checkpoint/resume input, seed, and frozen
    # environment remain byte-for-byte those emitted by the training pin.
    assert raw == before
    assert guarded["name"] == before["name"]
    assert guarded["argv"] == before["argv"]
    assert guarded["env"] == before["env"]
    assert set(guarded) == {
        "name",
        "argv",
        "env",
        "completion_artifact",
        "resources",
        "job_identity_sha256",
    }
    assert guarded["completion_artifact"] == str(
        tmp_path / "runs/full/drunet/seed1/run.json"
    )
    assert guarded["resources"] == {"host_ram_gib": 8.0}
    assert guarded["job_identity_sha256"] == canonical_sha256(
        {
            "name": before["name"],
            "argv": before["argv"],
            "env": before["env"],
            "completion_artifact": guarded["completion_artifact"],
            "resources": guarded["resources"],
        }
    )


def test_remaining_sweep_universe_is_exact_and_excludes_completed_sufs() -> None:
    path = ROOT / "scripts" / "prepare_remaining_native_sweep.py"
    spec = importlib.util.spec_from_file_location("remaining_sweep_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    assert len(value.EXPECTED_NATIVE) == 14
    assert len(value.EXPECTED_RAW) == 16
    assert all("sufs_ft" not in name for name in value.EXPECTED_NATIVE)
    assert value.EXPECTED_RAW[-2:] == (
        "sweep.sufs_ft.lr0p0003.seed0",
        "sweep.sufs_ft.lr0p001.seed0",
    )


def test_remaining_sweep_requires_complete_hash_bound_stage2_delivery(
    tmp_path: Path,
) -> None:
    path = ROOT / "scripts" / "prepare_remaining_native_sweep.py"
    spec = importlib.util.spec_from_file_location("remaining_sweep_delivery_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bundle = tmp_path / "delivery"
    (bundle / "results").mkdir(parents=True)
    (bundle / "qa/overlays").mkdir(parents=True)
    import hashlib
    def write(relative: str, value: bytes) -> str:
        target = bundle / relative
        target.write_bytes(value)
        return hashlib.sha256(value).hexdigest()
    summary = write("results/summary.csv", b"synthetic summary\n")
    markdown = write("results/tables.md", b"synthetic table\n")
    latex = write("results/tables.tex", b"synthetic latex\n")
    qa = [{"curve_sha256": write("qa/training_curves.png", b"curve")}]
    for index in range(1, 11):
        qa.append(
            {
                "ordinal": index,
                "sha256": write(f"qa/overlays/overlay_{index:02d}.png", f"overlay-{index}".encode()),
            }
        )
    delivery = bundle / "delivery.json"
    delivery.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stage2_unet_heldout_delivery",
                "gate_sha256": "a" * 64,
                "inference_card_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "provenance": {
                    "capture_timing": "process_start",
                    "git_commit": "d" * 40,
                    "git_dirty": False,
                },
                "results": {
                    "summary_sha256": summary,
                    "tables_md_sha256": markdown,
                    "tables_tex_sha256": latex,
                    "qa": qa,
                },
            }
        ),
        encoding="utf-8",
    )
    assert module._validate_stage2_delivery(delivery)["checkpoint_sha256"] == "c" * 64
    (bundle / "qa/overlays/overlay_10.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="overlay universe changed"):
        module._validate_stage2_delivery(delivery)
