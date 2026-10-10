from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *parts], cwd=ROOT, text=True, capture_output=True, check=True)


def test_public_rows_are_aggregate_with_explicit_completion_status() -> None:
    with (ROOT / "results" / "public_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table_a = [
        row
        for row in rows
        if row["table"] == "A" and row["view"] == "overall" and row["group"] == "overall"
    ]
    assert len(table_a) == 18
    assert len({row["method"] for row in table_a}) == 18
    assert {row["status"] for row in rows} <= {"complete", "pending", "n/a"}
    assert all(row["status"] != "pending" or row["seeds_completed"] != "" for row in rows)
    forbidden = {"image_id", "subject_id", "image_path", "mask_path", "filename"}
    assert forbidden.isdisjoint(rows[0])

    assert all(row["status"] == "complete" for row in table_a)
    assert next(row for row in table_a if row["method"] == "segformer_b2")[
        "seeds_completed"
    ] == "3"

    table_b1 = [
        row
        for row in rows
        if row["table"] == "B1" and row["view"] == "overall" and row["group"] == "overall"
    ]
    assert len(table_b1) == 20
    assert {row["method"] for row in table_b1 if row["status"] == "complete"} == {
        "gmfs_default", "gmfs_tuned", "sufs_zero", "sufs_ft", "fingernet",
        "svm", "rf", "welm", "unet", "unetpp", "drunet", "r_unet",
        "lenet", "alexnet", "cnn_ae", "segformer_b2", "all_background",
        "all_foreground", "hull_as_prediction",
    }
    unavailable = [row for row in table_b1 if row["status"] == "n/a"]
    assert [(row["method"], row["seeds_completed"]) for row in unavailable] == [
        ("ra_runet_aligned", "0")
    ]
    assert all(
        row[key] == ""
        for row in table_b1
        for key in ("accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr")
    )
    assert all(
        row["n_images"] == "475" and row["n_subjects"] == "40"
        for row in table_b1 if row["status"] == "complete"
    )


def test_site_build_and_number_match() -> None:
    run("site/build.py")
    result = run("scripts/check_site_numbers.py")
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    with (ROOT / "results" / "public_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        expected_rows = list(csv.DictReader(handle))
    with (ROOT / "results" / "sd302b_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        amendments = list(csv.DictReader(handle))
    keys = ("table", "view", "group", "method")
    merged = {tuple(row[key] for key in keys): row for row in expected_rows}
    merged.update({tuple(row[key] for key in keys): row for row in amendments})
    assert payload["result_rows"] == len(merged)
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    assert "v1.1 locked aggregate snapshot" in html
    assert "RA-RUnet" in html
    assert "training — pending" in (
        ROOT / "site" / "assets" / "app.js"
    ).read_text(encoding="utf-8")
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
    result = run(
        "scripts/scan_release.py", "--phase", "working", "--allow-limited-synthetic"
    )
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
        if command[1:3] == ["log", "--format=%B%x00"]:
            return subprocess.CompletedProcess(command, 0, "clean message\n", "")
        if command[1:3] == ["log", "--format="]:
            return subprocess.CompletedProcess(command, 0, "clean history\n", "")
        if command[1:3] == ["rev-list", "--objects"]:
            return subprocess.CompletedProcess(command, 0, "", "")
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

    encoded_errors: list[str] = []
    module._scan_text("A" * 100, "payload", encoded_errors)
    module._scan_text("data:image/png;base64," + "A" * 100, "payload", encoded_errors)
    module._scan_text(
        "data:" + "image/png;base64," + "AAAA \n" * 30,
        "whitespace-chunked payload",
        encoded_errors,
    )
    module._scan_text(
        "data:" + "image/svg+xml,%3Csvg%20viewBox%3D0%200%3E",
        "non-base64 image payload",
        encoded_errors,
    )
    assert any("long encoded token" in error for error in encoded_errors)
    assert any("embedded image payload" in error for error in encoded_errors)
    assert any(
        "whitespace-chunked payload" in error and "embedded image payload" in error
        for error in encoded_errors
    )
    assert any(
        "non-base64 image payload" in error and "embedded image payload" in error
        for error in encoded_errors
    )

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
    monkeypatch.setattr(module, "scan_history", lambda *args, **kwargs: None)
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


def test_release_scan_rejects_directory_symlink(monkeypatch, tmp_path: Path) -> None:
    script = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_directory_link_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked-directory").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(module, "scan_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scan_release.py", "--out", str(tmp_path / "scan.json")],
    )
    with pytest.raises(SystemExit):
        module.main()
    report = json.loads((tmp_path / "scan.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert "symlink: linked-directory" in report["errors"]


def test_final_scan_clean_tree_probe_distinguishes_committed_and_dirty(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_clean_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "safe.txt").write_text("synthetic\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "safe.txt"], check=True)
    subprocess.run(
        [
            "git", "-C", str(tmp_path), "-c", "user.name=Synthetic Test",
            "-c", "user.email=synthetic@example.invalid", "commit", "-qm", "synthetic fixture",
        ],
        check=True,
    )
    clean, head = module.git_tree_status(tmp_path)
    assert clean is True and len(head) == 40
    (tmp_path / "safe.txt").write_text("changed\n", encoding="utf-8")
    dirty, unchanged_head = module.git_tree_status(tmp_path)
    assert dirty is False and unchanged_head == head


def test_normal_final_scan_still_rejects_pending_figures(
    monkeypatch, tmp_path: Path
) -> None:
    """The partial-release exception must never weaken normal final."""
    script = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_final_mode_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "repo"
    (root / "figures").mkdir(parents=True)
    (root / "figures" / "figure_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "segbench.aggregate-figure-manifest.v1",
                "status": "pending",
                "source_receipt": None,
                "figures": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(root), "-c", "user.name=Synthetic Test",
            "-c", "user.email=synthetic@example.invalid", "commit", "-qm",
            "synthetic fixture",
        ],
        check=True,
    )
    module.ROOT = root
    monkeypatch.setattr(module, "scan_history", lambda *args, **kwargs: None)
    report = tmp_path / "normal-final.json"
    monkeypatch.setattr(
        sys, "argv", ["scan_release.py", "--phase", "final", "--out", str(report)]
    )
    with pytest.raises(SystemExit):
        module.main()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any(
        "final publication scan requires imported aggregate figures" in error
        for error in payload["errors"]
    )


def test_superseded_provisional_receipt_rejects_final_output(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_provisional_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    for name in (
        "public_results.csv", "statistics.csv", "table_status.csv",
        "provisional_release_receipt.json",
    ):
        (result_dir / name).write_bytes((ROOT / "results" / name).read_bytes())
    receipt_path = result_dir / "provisional_release_receipt.json"
    pinned = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    # v1.0's receipt remains available for audit, but it must fail closed when
    # paired with the replaced v1.1 aggregate files.
    with pytest.raises(ValueError, match="not receipt-bound"):
        module.verify_provisional_release(tmp_path, pinned)


def test_history_scan_rejects_private_commit_message_and_deleted_raster(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("release_scan_history_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    image = tmp_path / "deleted.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nprivate fixture")
    subprocess.run(["git", "-C", str(tmp_path), "add", "deleted.png"], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=Synthetic Test",
        "-c", "user.email=synthetic@example.invalid", "commit", "-qm",
        "temporary file from " + "/" + "home/private/source",
    ], check=True)
    image.unlink()
    subprocess.run(["git", "-C", str(tmp_path), "add", "-u"], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=Synthetic Test",
        "-c", "user.email=synthetic@example.invalid", "commit", "-qm", "remove fixture",
    ], check=True)
    errors: list[str] = []
    module.scan_history(errors, root=tmp_path, approved_binary_hashes=set())
    assert any("commit message" in error and "absolute private path" in error for error in errors)
    assert any("historical raster/PDF" in error for error in errors)
