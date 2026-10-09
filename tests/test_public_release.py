from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

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
    manifest = json.loads((ROOT / "site" / "build_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (ROOT / "examples" / "synthetic" / "synthetic_receipt.json").read_text(encoding="utf-8")
    )
    statuses = [row["status"] for row in receipt["synthetic_demo"]["methods"]]
    assert manifest["synthetic_examples"] == 3
    assert manifest["synthetic_complete_methods"] == statuses.count("complete")
    assert manifest["synthetic_unavailable_methods"] == statuses.count("unavailable")


def test_release_scan_passes() -> None:
    run("site/build.py")
    result = run("scripts/scan_release.py", "--allow-limited-synthetic")
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
        if command[1:3] == ["log", "--format="]:
            return subprocess.CompletedProcess(command, 0, "clean history\n", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    errors: list[str] = []
    module.scan_history(errors)
    assert errors == []
    assert ["git", "log", "--format=", "-p", "--all", "--no-ext-diff"] in calls


def test_release_scan_rejects_private_tokens_in_path_content_and_history(
    monkeypatch, tmp_path: Path
) -> None:
    script = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_policy_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    errors: list[str] = []
    module._scan_text("0000" + "2314_X.png", "path", errors)
    module._scan_text("1Abcdefghijklmno" + "pqrstuvWXYZ789", "content", errors)
    module._scan_text("s" + "sh operator@" + "private-node" + ".local", "content", errors)
    assert any("SD302-like image identifier" in error for error in errors)
    assert any("opaque cloud-drive-like identifier" in error for error in errors)
    assert any("local infrastructure hostname" in error for error in errors)

    errors = []
    module._scan_text(
        "0000" + "2314_X " + "1000" + "PPI LP" + "01",
        "docs/DATASETS.md",
        errors,
        allow_identifier_explanation=True,
    )
    assert errors == []


def test_release_scan_rejects_unapproved_binary_suffix(monkeypatch, tmp_path: Path) -> None:
    script = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_binary_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    (tmp_path / "payload.zip").write_bytes(b"PK\x03\x04")
    monkeypatch.setattr(module, "scan_history", lambda errors: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scan_release.py", "--out", str(tmp_path / "scan.json")],
    )
    with pytest.raises(SystemExit):
        module.main()
    report = json.loads((tmp_path / "scan.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert any("unapproved binary or file type" in error for error in report["errors"])
