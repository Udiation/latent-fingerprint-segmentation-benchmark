from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *parts], cwd=ROOT, text=True, capture_output=True, check=True)


def test_public_rows_are_aggregate_and_pending() -> None:
    with (ROOT / "results" / "public_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    assert all(row["table"] == "A" for row in rows)
    assert all(row["status"] == "pending" for row in rows)
    forbidden = {"image_id", "subject_id", "image_path", "mask_path", "filename"}
    assert forbidden.isdisjoint(rows[0])


def test_site_build_and_number_match() -> None:
    run("site/build.py")
    result = run("scripts/check_site_numbers.py")
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["result_rows"] == 18


def test_release_scan_passes() -> None:
    run("site/build.py")
    result = run("scripts/scan_release.py")
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["errors"] == []


def test_release_history_scan_runs_from_linked_worktree(monkeypatch, tmp_path: Path) -> None:
    script = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[1:3] == ["rev-parse", "--is-inside-work-tree"]:
            return subprocess.CompletedProcess(command, 0, "true\n", "")
        if command[1:3] == ["log", "-p"]:
            return subprocess.CompletedProcess(command, 0, "clean history\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    errors: list[str] = []
    module.scan_history(errors)
    assert errors == []
    assert ["git", "log", "-p", "--all", "--no-ext-diff"] in calls
