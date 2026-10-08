"""Tests for append-only queue recovery after an audited infrastructure failure."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from segbench.artifact_identity import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts" / "prepare_queue_recovery.py"
    spec = importlib.util.spec_from_file_location("queue_recovery_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _job(name: str, completion: Path) -> dict:
    item = {
        "name": name,
        "argv": ["python", "train_seg.py", "--resume", "auto"],
        "env": {},
        "completion_artifact": str(completion),
        "resources": {"host_ram_gib": 4.0, "cpu_workers": 8},
    }
    item["job_identity_sha256"] = canonical_sha256(
        {
            "name": name,
            "argv": item["argv"],
            "env": {},
            "completion_artifact": str(completion),
            "resources": item["resources"],
        }
    )
    return item


def test_recovery_is_append_only_and_resets_only_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    complete = tmp_path / "complete.json"
    complete.write_text("{}\n", encoding="utf-8")
    failed = tmp_path / "failed.json"
    queue = tmp_path / "source.yaml"
    queue.write_text(
        yaml.safe_dump(
                {
                    "kind": module.guarded.ADOPTED_QUEUE_KIND,
                    "launch_authorized": True,
                "phase": "synthetic",
                "jobs": [_job("done", complete), _job("retry", failed)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module.guarded,
        "validate_adopted_queue_runtime",
        lambda *args: {"kind": "synthetic-binding"},
    )
    jobs = module.guarded.load_guarded_jobs(queue)
    ledger = module.guarded.initial_state(queue, jobs)
    ledger["jobs"]["done"].update(
        status="completed", artifact_sha256=sha256_file(complete)
    )
    ledger["jobs"]["retry"].update(
        status="failed", attempts=2, ordinary_retries=1, exit_code=1
    )
    ledger_path = tmp_path / "source-ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    note = tmp_path / "incident.md"
    note.write_text("Profiler dependency was unavailable before epoch completion.\n")
    source_queue_bytes = queue.read_bytes()
    source_ledger_bytes = ledger_path.read_bytes()
    args = module.parse_args(
        [
            "--source-queue", str(queue),
            "--source-ledger", str(ledger_path),
            "--incident-note", str(note),
            "--out-queue", str(tmp_path / "recovery.yaml"),
            "--out-state", str(tmp_path / "recovery-state.json"),
            "--proof-out", str(tmp_path / "recovery-proof.json"),
        ]
    )
    proof = module.prepare(args)
    assert queue.read_bytes() == source_queue_bytes
    assert ledger_path.read_bytes() == source_ledger_bytes
    recovered = json.loads((tmp_path / "recovery-state.json").read_text())
    assert recovered["jobs"]["done"]["status"] == "completed"
    assert recovered["jobs"]["retry"]["status"] == "pending"
    assert recovered["jobs"]["retry"]["attempts"] == 0
    assert recovered["jobs"]["retry"]["superseded_source_attempts"] == 2
    assert proof["source_artifacts_modified"] is False


def test_recovery_refuses_live_source_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue = tmp_path / "source.yaml"
    completion = tmp_path / "run.json"
    queue.write_text(
        yaml.safe_dump(
                {
                    "kind": module.guarded.ADOPTED_QUEUE_KIND,
                    "launch_authorized": True,
                "jobs": [_job("active", completion)],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module.guarded, "validate_adopted_queue_runtime", lambda *args: {"ok": True}
    )
    jobs = module.guarded.load_guarded_jobs(queue)
    ledger = module.guarded.initial_state(queue, jobs)
    ledger["jobs"]["active"].update(
        status="running", worker={"pid": 123, "start_ticks": 1}
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    note = tmp_path / "incident.md"
    note.write_text("audited incident\n", encoding="utf-8")
    monkeypatch.setattr(module.guarded, "_worker_is_exactly_live", lambda worker: True)
    args = module.parse_args(
        [
            "--source-queue", str(queue), "--source-ledger", str(ledger_path),
            "--incident-note", str(note), "--out-queue", str(tmp_path / "new.yaml"),
            "--out-state", str(tmp_path / "new.json"),
            "--proof-out", str(tmp_path / "proof.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="still live"):
        module.prepare(args)


def _external_completion_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, object, Path, Path, Path, dict]:
    module = _module()
    completion = tmp_path / "external-run.json"
    completion.write_text('{"complete": true}\n', encoding="utf-8")
    retry = tmp_path / "retry-run.json"
    external_job = _job("external", completion)
    queue = tmp_path / "source.yaml"
    queue.write_text(
        yaml.safe_dump(
            {
                "kind": module.guarded.ADOPTED_QUEUE_KIND,
                "launch_authorized": True,
                "jobs": [external_job, _job("retry", retry)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module.guarded,
        "validate_adopted_queue_runtime",
        lambda *args: {"kind": "synthetic-binding"},
    )
    jobs = module.guarded.load_guarded_jobs(queue)
    ledger = module.guarded.initial_state(queue, jobs)
    ledger["jobs"]["retry"].update(status="failed", attempts=2, exit_code=1)
    ledger_path = tmp_path / "source-ledger.json"
    ledger_path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    note = tmp_path / "incident.md"
    note.write_text("audited infrastructure incident\n", encoding="utf-8")
    worker = module.guarded.ExternalWorker(
        name="external",
        pid=43210,
        start_ticks=99,
        argv_sha256=canonical_sha256(external_job["argv"]),
        gpu=7,
        completion_artifact=completion,
        preempt_path=tmp_path / "external.preempt",
        host_ram_gib=4.0,
        cpu_workers=8,
    )
    external_record = {
        "name": "external",
        "pid": worker.pid,
        "start_ticks": worker.start_ticks,
        "argv_sha256": worker.argv_sha256,
        "gpu": worker.gpu,
        "completion_artifact": str(completion),
        "preempt_path": str(worker.preempt_path),
        "host_ram_gib": worker.host_ram_gib,
        "cpu_workers": worker.cpu_workers,
        "source_job_identity_sha256": external_job["job_identity_sha256"],
    }
    plan = {
        "allowed_gpus": [worker.gpu],
        "max_jobs_per_gpu": 3,
        "max_project_ram_gib": 32.0,
        "memory_fraction": 0.3,
        "cpu_budget": {"max_cpu_cores": 16},
        "workers": [external_record],
        "precompleted_workers": [],
    }
    plan_path = tmp_path / "source-external-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        module.guarded,
        "load_external_workers",
        lambda *args, **kwargs: ([worker], plan),
    )
    args = module.parse_args(
        [
            "--source-queue", str(queue),
            "--source-ledger", str(ledger_path),
            "--incident-note", str(note),
            "--source-external-plan", str(plan_path),
            "--out-queue", str(tmp_path / "recovery.yaml"),
            "--out-state", str(tmp_path / "recovery-state.json"),
            "--out-external-plan", str(tmp_path / "rebound-plan.json"),
            "--proof-out", str(tmp_path / "recovery-proof.json"),
        ]
    )
    return module, args, queue, ledger_path, plan_path, plan


def test_recovery_preserves_exact_external_completion_without_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, args, queue, ledger_path, plan_path, _ = _external_completion_case(
        tmp_path, monkeypatch
    )
    source_bytes = {
        queue: queue.read_bytes(),
        ledger_path: ledger_path.read_bytes(),
        plan_path: plan_path.read_bytes(),
    }
    monkeypatch.setattr(module.guarded, "_external_is_exactly_live", lambda worker: False)
    monkeypatch.setattr(module.guarded, "_process_start_ticks", lambda pid: None)

    proof = module.prepare(args)

    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    state = json.loads(args.out_state.read_text(encoding="utf-8"))
    entry = state["jobs"]["external"]
    assert entry["status"] == "completed"
    assert entry["completed_externally_during_recovery"] is True
    assert entry["source_external_plan_sha256"] == sha256_file(plan_path)
    assert state["jobs"]["retry"]["status"] == "pending"
    rebound = json.loads(args.out_external_plan.read_text(encoding="utf-8"))
    assert rebound["workers"] == []
    assert proof["preserved_completed_jobs"] == ["external"]
    assert proof["reset_unfinished_jobs"] == ["retry"]


def test_recovery_refuses_external_artifact_while_worker_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, args, *_ = _external_completion_case(tmp_path, monkeypatch)
    monkeypatch.setattr(module.guarded, "_external_is_exactly_live", lambda worker: True)
    with pytest.raises(RuntimeError, match="published its artifact but is still live"):
        module.prepare(args)


def test_recovery_refuses_external_completion_with_changed_job_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, args, _, _, plan_path, plan = _external_completion_case(
        tmp_path, monkeypatch
    )
    plan["workers"][0]["source_job_identity_sha256"] = "0" * 64
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="external-plan job identity changed"):
        module.prepare(args)
