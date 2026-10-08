"""Synthetic CPU-only tests for immutable launch-time provenance."""
from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from segbench.deep_utils import apply_queue_batch_contract, checkpoint_payload, run_provenance


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def test_provenance_snapshot_survives_later_head_and_worktree_changes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "user.name", "Synthetic Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "one")
    first_head = _git(repo, "rev-parse", "HEAD")

    captured = run_provenance(
        ["synthetic-command", "--seed", "0"],
        capture_timing="process_start",
        repository=repo,
    )

    tracked.write_text("two\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "two")
    (repo / "later-untracked.txt").write_text("dirty\n", encoding="utf-8")

    assert captured["git_commit"] == first_head
    assert captured["git_dirty"] is False
    assert captured["capture_timing"] == "process_start"
    assert captured["captured_at_utc"].endswith("+00:00")
    assert captured["command_argv"] == ["synthetic-command", "--seed", "0"]
    assert "repository" not in captured

    later = run_provenance(["later"], repository=repo)
    assert later["git_commit"] != captured["git_commit"]
    assert later["git_dirty"] is True
    assert captured["git_commit"] == first_head
    assert captured["git_dirty"] is False


def test_checkpoint_embeds_the_already_captured_snapshot() -> None:
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    captured = {
        "command_argv": ["train"],
        "capture_timing": "process_start",
        "captured_at_utc": "2026-01-01T00:00:00+00:00",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "versions": {},
    }
    config = {
        "schema_version": 1,
        "method": "fixture",
        "task": "pixel",
        "data": {"normalization": "train_only"},
        "model": {},
        "training": {
            "learning_rate": 3e-4,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "threshold_candidates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        },
        "runtime": {"memory_fraction": 0.3},
    }
    apply_queue_batch_contract(config, {})
    payload = checkpoint_payload(
        model=model,
        method="fixture",
        task="pixel",
        config=config,
        normalization={"source_split": "train", "mean": 0.5, "std": 0.2},
        epoch=0,
        selected_threshold=0.5,
        validation_dice=0.5,
        parameter_count={"total": 2, "trainable": 2},
        provenance=captured,
    )
    captured["git_commit"] = "b" * 40
    assert payload["provenance"]["git_commit"] == "a" * 40
    assert payload["provenance"]["capture_timing"] == "process_start"
