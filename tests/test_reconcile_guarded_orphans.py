"""Synthetic tests for fail-closed orphan-worker reconciliation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pytest
import yaml

from segbench.artifact_identity import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts/reconcile_guarded_orphans.py"
    spec = importlib.util.spec_from_file_location("reconcile_orphans_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Dict[str, Any]]:
    completion = tmp_path / "run.json"
    argv = ["python", "train.py", "--resume", "auto"]
    job: Dict[str, Any] = {
        "name": "sweep.synthetic.seed0",
        "argv": argv,
        "env": {},
        "completion_artifact": str(completion),
        "resources": {"host_ram_gib": 8.0},
    }
    identity = canonical_sha256(
        {
            "name": job["name"],
            "argv": argv,
            "env": {},
            "completion_artifact": str(completion),
            "resources": job["resources"],
        }
    )
    job["job_identity_sha256"] = identity
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        yaml.safe_dump({"launch_authorized": True, "jobs": [job]}),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queue_sha256": sha256_file(queue),
                "jobs": {
                    job["name"]: {
                        "identity_sha256": identity,
                        "status": "running",
                        "attempts": 1,
                        "ordinary_retries": 0,
                        "oom_retries": 0,
                        "preemptions": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return queue, ledger, job


def test_capture_requires_exact_live_argv_and_running_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue, ledger, job = _artifacts(tmp_path)
    monkeypatch.setattr(
        module,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": list(job["argv"]),
            "argv_sha256": canonical_sha256(job["argv"]),
        },
    )
    output = tmp_path / "capture.json"
    captured = module.capture(queue, ledger, {job["name"]: 4567}, output)
    assert captured["workers"][job["name"]]["pid"] == 4567
    assert captured["queue_sha256"] == sha256_file(queue)
    assert captured["schema_version"] == 2
    assert captured["workers"][job["name"]]["completion_artifact_absent"] is True
    assert captured["queue_snapshot"] == yaml.safe_load(queue.read_text())
    assert captured["ledger_snapshot_at_capture"] == json.loads(ledger.read_text())

    output.unlink()
    monkeypatch.setattr(
        module,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": ["python", "wrong.py"],
            "argv_sha256": canonical_sha256(["python", "wrong.py"]),
        },
    )
    with pytest.raises(ValueError, match="argv differs"):
        module.capture(queue, ledger, {job["name"]: 4567}, output)


def test_reconcile_never_signals_and_atomically_completes_after_run_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue, ledger, job = _artifacts(tmp_path)
    monkeypatch.setattr(
        module,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": list(job["argv"]),
            "argv_sha256": canonical_sha256(job["argv"]),
        },
    )
    capture_path = tmp_path / "capture.json"
    module.capture(queue, ledger, {job["name"]: 4567}, capture_path)
    preflight = module.reconciliation_preflight(queue, ledger, capture_path)
    assert preflight["workers"] == {job["name"]: "running"}
    Path(job["completion_artifact"]).write_text('{"complete": true}\n', encoding="utf-8")
    monkeypatch.setattr(module, "process_identity", lambda pid: None)
    proof = module.reconcile(
        queue,
        ledger,
        capture_path,
        tmp_path / "proof.json",
        poll_seconds=1,
    )
    entry = json.loads(ledger.read_text())["jobs"][job["name"]]
    assert entry["status"] == "completed"
    assert entry["artifact_sha256"] == sha256_file(Path(job["completion_artifact"]))
    assert entry["orphan_reconciliation"]["pid"] == 4567
    assert proof["ledger_sha256_after"] == sha256_file(ledger)
    # Replay validates capture path/hash/kind/workers and the reconciled card.
    replay = module.reconcile(
        queue,
        ledger,
        capture_path,
        tmp_path / "proof.json",
        poll_seconds=1,
    )
    assert replay == proof


def test_reconcile_fails_if_orphan_exits_without_run_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue, ledger, job = _artifacts(tmp_path)
    monkeypatch.setattr(
        module,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": list(job["argv"]),
            "argv_sha256": canonical_sha256(job["argv"]),
        },
    )
    capture_path = tmp_path / "capture.json"
    module.capture(queue, ledger, {job["name"]: 4567}, capture_path)
    monkeypatch.setattr(module, "process_identity", lambda pid: None)
    with pytest.raises(FileNotFoundError, match="without declared completion"):
        module.reconcile(
            queue,
            ledger,
            capture_path,
            tmp_path / "proof.json",
            poll_seconds=1,
        )


def test_capture_rejects_preexisting_completion_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue, ledger, job = _artifacts(tmp_path)
    Path(job["completion_artifact"]).write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": list(job["argv"]),
            "argv_sha256": canonical_sha256(job["argv"]),
        },
    )
    with pytest.raises(FileExistsError, match="already existed"):
        module.capture(
            queue, ledger, {job["name"]: 4567}, tmp_path / "capture.json"
        )


def test_existing_proof_rejects_changed_capture_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue, ledger, job = _artifacts(tmp_path)
    monkeypatch.setattr(
        module,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": list(job["argv"]),
            "argv_sha256": canonical_sha256(job["argv"]),
        },
    )
    capture_path = tmp_path / "capture.json"
    module.capture(queue, ledger, {job["name"]: 4567}, capture_path)
    Path(job["completion_artifact"]).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "process_identity", lambda pid: None)
    proof_path = tmp_path / "proof.json"
    module.reconcile(queue, ledger, capture_path, proof_path, poll_seconds=1)
    capture = json.loads(capture_path.read_text())
    capture["workers"][job["name"]]["start_ticks"] = 124
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(ValueError):
        module.reconcile(queue, ledger, capture_path, proof_path, poll_seconds=1)


def test_process_identity_missing_proc_is_clean_none() -> None:
    module = _module()
    assert module.process_identity(999_999_999) is None


def test_preflight_rejects_mutated_captured_ledger_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue, ledger, job = _artifacts(tmp_path)
    monkeypatch.setattr(
        module,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": list(job["argv"]),
            "argv_sha256": canonical_sha256(job["argv"]),
        },
    )
    capture_path = tmp_path / "capture.json"
    module.capture(queue, ledger, {job["name"]: 4567}, capture_path)
    payload = json.loads(ledger.read_text())
    payload["jobs"][job["name"]]["attempts"] = 99
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="allowed reconciliation evolution"):
        module.reconciliation_preflight(queue, ledger, capture_path)


def test_preflight_rejects_mutated_noncaptured_ledger_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue, ledger, job = _artifacts(tmp_path)
    queue_payload = yaml.safe_load(queue.read_text())
    other_completion = tmp_path / "other-run.json"
    other = {
        "name": "sweep.synthetic.other.seed0",
        "argv": ["python", "train.py", "--other"],
        "env": {},
        "completion_artifact": str(other_completion),
        "resources": {"host_ram_gib": 8.0},
    }
    other["job_identity_sha256"] = canonical_sha256(
        {
            "name": other["name"],
            "argv": other["argv"],
            "env": other["env"],
            "completion_artifact": other["completion_artifact"],
            "resources": other["resources"],
        }
    )
    queue_payload["jobs"].append(other)
    queue.write_text(yaml.safe_dump(queue_payload), encoding="utf-8")
    ledger_payload = json.loads(ledger.read_text())
    ledger_payload["queue_sha256"] = sha256_file(queue)
    ledger_payload["jobs"][other["name"]] = {
        "identity_sha256": other["job_identity_sha256"],
        "status": "pending",
        "attempts": 0,
        "ordinary_retries": 0,
        "oom_retries": 0,
        "preemptions": 0,
    }
    ledger.write_text(json.dumps(ledger_payload), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": list(job["argv"]),
            "argv_sha256": canonical_sha256(job["argv"]),
        },
    )
    capture_path = tmp_path / "capture.json"
    module.capture(queue, ledger, {job["name"]: 4567}, capture_path)
    changed = json.loads(ledger.read_text())
    changed["jobs"][other["name"]]["attempts"] = 1
    ledger.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="noncaptured ledger entry changed"):
        module.reconciliation_preflight(queue, ledger, capture_path)


def test_preflight_rejects_truncated_captured_running_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue, ledger, job = _artifacts(tmp_path)
    queue_payload = yaml.safe_load(queue.read_text())
    other = {
        "name": "sweep.synthetic.second.seed0",
        "argv": ["python", "train.py", "--second"],
        "env": {},
        "completion_artifact": str(tmp_path / "second" / "run.json"),
        "resources": {"host_ram_gib": 8.0},
    }
    other["job_identity_sha256"] = canonical_sha256(
        {
            "name": other["name"],
            "argv": other["argv"],
            "env": other["env"],
            "completion_artifact": other["completion_artifact"],
            "resources": other["resources"],
        }
    )
    queue_payload["jobs"].append(other)
    queue.write_text(yaml.safe_dump(queue_payload), encoding="utf-8")
    ledger_payload = json.loads(ledger.read_text())
    ledger_payload["queue_sha256"] = sha256_file(queue)
    ledger_payload["jobs"][other["name"]] = {
        "identity_sha256": other["job_identity_sha256"],
        "status": "running",
        "attempts": 1,
        "ordinary_retries": 0,
        "oom_retries": 0,
        "preemptions": 0,
    }
    ledger.write_text(json.dumps(ledger_payload), encoding="utf-8")

    def identity(pid: int) -> Dict[str, Any]:
        argv = job["argv"] if pid == 4567 else other["argv"]
        return {
            "pid": pid,
            "start_ticks": pid + 100,
            "argv": list(argv),
            "argv_sha256": canonical_sha256(argv),
        }

    monkeypatch.setattr(module, "process_identity", identity)
    capture_path = tmp_path / "capture.json"
    module.capture(
        queue,
        ledger,
        {job["name"]: 4567, other["name"]: 4568},
        capture_path,
    )
    truncated = json.loads(capture_path.read_text())
    del truncated["workers"][other["name"]]
    capture_path.write_text(json.dumps(truncated), encoding="utf-8")
    with pytest.raises(ValueError, match="worker universe differs"):
        module.reconciliation_preflight(queue, ledger, capture_path)
