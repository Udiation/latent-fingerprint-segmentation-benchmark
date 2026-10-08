#!/usr/bin/env python3
"""Capture and reconcile guarded-queue workers orphaned by controller failure.

This recovery never starts, stops, or signals a worker.  ``capture`` must run
while every declared worker is alive; it binds PID, procfs start time and exact
argument vector to the frozen queue and ledger.  ``reconcile`` polls only those
identities.  After a worker exits, it requires the declared ``run.json`` and
atomically changes only that ledger entry from ``running`` to ``completed``.
The normal guarded queue can then resume the remaining pending jobs safely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import canonical_sha256, sha256_file


KIND = "guarded_queue_orphan_capture"
CAPTURE_SCHEMA_VERSION = 2


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _read_queue(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("launch_authorized") is not True:
        raise ValueError("queue is not a launch-authorized object")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("queue lacks jobs")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if canonical_sha256(_read_json(path, "existing immutable artifact")) != canonical_sha256(value):
            raise FileExistsError(f"refusing to replace changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_identity(pid: int) -> Optional[Dict[str, Any]]:
    """Return exact argv plus a PID-reuse-resistant procfs start time."""
    proc = Path("/proc") / str(pid)
    try:
        before = (
            proc.joinpath("stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .strip()
            .split()
        )
        start_ticks_before = int(before[19])
        raw = [part for part in proc.joinpath("cmdline").read_bytes().split(b"\0") if part]
        argv = [part.decode("utf-8", errors="surrogateescape") for part in raw]
        after = (
            proc.joinpath("stat")
            .read_text(encoding="utf-8")
            .rsplit(")", 1)[1]
            .strip()
            .split()
        )
        start_ticks_after = int(after[19])
    except (OSError, ProcessLookupError, ValueError, IndexError):
        return None
    if start_ticks_before != start_ticks_after:
        raise RuntimeError(f"PID {pid} changed identity during procfs observation")
    return {
        "pid": pid,
        "start_ticks": start_ticks_after,
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
    }


def _job_map(queue: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for job in queue["jobs"]:
        if not isinstance(job, dict) or not isinstance(job.get("name"), str):
            raise ValueError("queue contains an invalid job")
        name = str(job["name"])
        if name in result:
            raise ValueError(f"duplicate queue job {name}")
        result[name] = job
    return result


def _ledger(path: Path, queue_path: Path, jobs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    ledger = _read_json(path, "guarded queue ledger")
    if ledger.get("queue_sha256") != sha256_file(queue_path):
        raise ValueError("ledger is not bound to the queue")
    entries = ledger.get("jobs")
    if not isinstance(entries, dict) or set(entries) != set(jobs):
        raise ValueError("ledger universe differs from queue")
    for name, job in jobs.items():
        entry = entries[name]
        identity = canonical_sha256(
            {
                "name": job.get("name"),
                "argv": job.get("argv"),
                "env": job.get("env", {}),
                "completion_artifact": job.get("completion_artifact"),
                "resources": job.get("resources"),
            }
        )
        if job.get("job_identity_sha256") != identity or entry.get(
            "identity_sha256"
        ) != identity:
            raise ValueError(f"{name}: queue/ledger job identity mismatch")
    return ledger


def parse_job_pids(values: Sequence[str]) -> Dict[str, int]:
    output: Dict[str, int] = {}
    for value in values:
        name, separator, pid_text = value.rpartition("=")
        if not separator or not name:
            raise ValueError(f"invalid --job-pid {value!r}; expected NAME=PID")
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise ValueError(f"invalid PID in --job-pid {value!r}") from exc
        if pid <= 1 or name in output or pid in output.values():
            raise ValueError("job names and worker PIDs must be unique")
        output[name] = pid
    if not output:
        raise ValueError("at least one --job-pid is required")
    return output


def capture(
    queue_path: Path, ledger_path: Path, job_pids: Mapping[str, int], output: Path
) -> Dict[str, Any]:
    """Bind live orphan workers to exact frozen queue entries."""
    queue = _read_queue(queue_path)
    jobs = _job_map(queue)
    ledger = _ledger(ledger_path, queue_path, jobs)
    entries = ledger["jobs"]
    running = {name for name, entry in entries.items() if entry.get("status") == "running"}
    if set(job_pids) != running:
        raise ValueError(
            f"declared orphan universe {sorted(job_pids)} != running ledger universe {sorted(running)}"
        )
    workers = {}
    for name in sorted(job_pids):
        completion = Path(str(jobs[name].get("completion_artifact", "")))
        if completion.exists():
            raise FileExistsError(
                f"{name}: completion artifact already existed before orphan capture"
            )
        identity = process_identity(int(job_pids[name]))
        if identity is None:
            raise ProcessLookupError(f"{name}: PID {job_pids[name]} is not alive")
        expected_argv = jobs[name].get("argv")
        if identity["argv"] != expected_argv:
            raise ValueError(f"{name}: live worker argv differs from frozen queue")
        if completion.exists():
            raise FileExistsError(
                f"{name}: completion artifact appeared during orphan capture"
            )
        workers[name] = {
            "pid": identity["pid"],
            "start_ticks": identity["start_ticks"],
            "argv_sha256": identity["argv_sha256"],
            "completion_artifact": str(jobs[name]["completion_artifact"]),
            "completion_artifact_absent": True,
        }
    queue_text = queue_path.read_text(encoding="utf-8")
    ledger_text = ledger_path.read_text(encoding="utf-8")
    if yaml.safe_load(queue_text) != queue or json.loads(ledger_text) != ledger:
        raise RuntimeError("queue or ledger changed while orphan capture was assembled")
    queue_byte_sha = hashlib.sha256(queue_text.encode("utf-8")).hexdigest()
    ledger_byte_sha = hashlib.sha256(ledger_text.encode("utf-8")).hexdigest()
    if queue_byte_sha != sha256_file(queue_path) or ledger_byte_sha != sha256_file(
        ledger_path
    ):
        raise RuntimeError("queue or ledger bytes changed during orphan capture")
    payload = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "kind": KIND,
        "captured_at_utc": _utc(),
        "queue": str(queue_path),
        "queue_sha256": queue_byte_sha,
        "ledger": str(ledger_path),
        "ledger_sha256_at_capture": ledger_byte_sha,
        "queue_snapshot_sha256": canonical_sha256(queue),
        "ledger_snapshot_sha256_at_capture": canonical_sha256(ledger),
        "queue_snapshot_text": queue_text,
        "ledger_snapshot_text_at_capture": ledger_text,
        "queue_snapshot": queue,
        "ledger_snapshot_at_capture": ledger,
        "workers": workers,
    }
    _write_once(output, payload)
    return payload


def _captured_identity(record: Mapping[str, Any], argv: Sequence[str]) -> Dict[str, Any]:
    return {
        "pid": int(record["pid"]),
        "start_ticks": int(record["start_ticks"]),
        "argv": list(argv),
        "argv_sha256": str(record["argv_sha256"]),
    }


def _validate_capture(
    captured: Mapping[str, Any],
    capture_path: Path,
    queue_path: Path,
    ledger_path: Path,
    queue: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Validate the immutable v2 capture, including its exact snapshots."""
    if captured.get("schema_version") != CAPTURE_SCHEMA_VERSION or captured.get(
        "kind"
    ) != KIND:
        raise ValueError("orphan capture schema/kind changed")
    if captured.get("queue_sha256") != sha256_file(queue_path):
        raise ValueError("orphan capture is not bound to this queue")
    if Path(str(captured.get("queue", ""))).resolve() != queue_path.resolve() or Path(
        str(captured.get("ledger", ""))
    ).resolve() != ledger_path.resolve():
        raise ValueError("orphan capture queue/ledger paths changed")
    queue_snapshot = captured.get("queue_snapshot")
    ledger_snapshot = captured.get("ledger_snapshot_at_capture")
    if not isinstance(queue_snapshot, dict) or not isinstance(ledger_snapshot, dict):
        raise ValueError("orphan capture lacks exact queue/ledger snapshots")
    if canonical_sha256(queue_snapshot) != captured.get(
        "queue_snapshot_sha256"
    ) or canonical_sha256(ledger_snapshot) != captured.get(
        "ledger_snapshot_sha256_at_capture"
    ):
        raise ValueError("orphan capture snapshot digest changed")
    if queue_snapshot != queue:
        raise ValueError("captured queue snapshot differs from frozen queue")
    if ledger_snapshot.get("queue_sha256") != captured.get("queue_sha256"):
        raise ValueError("captured ledger snapshot is not queue-bound")
    snapshot_entries = ledger_snapshot.get("jobs")
    if not isinstance(snapshot_entries, dict) or set(snapshot_entries) != set(jobs):
        raise ValueError("captured ledger snapshot universe differs from queue")
    queue_text = captured.get("queue_snapshot_text")
    ledger_text = captured.get("ledger_snapshot_text_at_capture")
    if not isinstance(queue_text, str) or not isinstance(ledger_text, str):
        raise ValueError("orphan capture lacks byte-exact queue/ledger text")
    if hashlib.sha256(queue_text.encode("utf-8")).hexdigest() != captured.get(
        "queue_sha256"
    ) or hashlib.sha256(ledger_text.encode("utf-8")).hexdigest() != captured.get(
        "ledger_sha256_at_capture"
    ):
        raise ValueError("orphan capture byte snapshot digest changed")
    if yaml.safe_load(queue_text) != queue_snapshot or json.loads(
        ledger_text
    ) != ledger_snapshot:
        raise ValueError("orphan capture parsed and byte snapshots disagree")
    workers = captured.get("workers")
    if not isinstance(workers, dict) or not workers:
        raise ValueError("orphan capture lacks workers")
    captured_running = {
        name
        for name, entry in snapshot_entries.items()
        if isinstance(entry, dict) and entry.get("status") == "running"
    }
    if set(workers) != captured_running:
        raise ValueError("orphan capture worker universe differs from captured running jobs")
    for name, record in workers.items():
        if not isinstance(record, dict) or record.get(
            "completion_artifact_absent"
        ) is not True:
            raise ValueError(f"{name}: capture lacks stale-artifact exclusion proof")
        if snapshot_entries[name].get("status") != "running":
            raise ValueError(f"{name}: worker was not running in captured ledger")
    return workers


def _validate_ledger_evolution(
    captured: Mapping[str, Any],
    capture_path: Path,
    current: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Allow only this reconciler's exact captured running→completed edits."""
    snapshot = captured["ledger_snapshot_at_capture"]
    snapshot_entries = snapshot["jobs"]
    current_entries = current["jobs"]
    snapshot_top = {key: value for key, value in snapshot.items() if key != "jobs"}
    current_top = {key: value for key, value in current.items() if key != "jobs"}
    if current_top != snapshot_top:
        raise ValueError("current ledger top-level fields changed after orphan capture")
    workers = captured["workers"]
    capture_sha = sha256_file(capture_path)
    for name in jobs:
        before = snapshot_entries[name]
        observed = current_entries[name]
        if name not in workers:
            if observed != before:
                raise ValueError(f"{name}: noncaptured ledger entry changed after capture")
            continue
        if observed == before:
            if observed.get("status") != "running":
                raise ValueError(f"{name}: captured baseline entry was not running")
            continue
        if observed.get("status") != "completed":
            raise ValueError(f"{name}: ledger is not an allowed reconciliation evolution")
        record = workers[name]
        completion = Path(str(record["completion_artifact"]))
        if not completion.is_file():
            raise ValueError(f"{name}: completed ledger entry lacks its run card")
        artifact_sha = sha256_file(completion)
        expected_reconciliation = {
            "capture_sha256": capture_sha,
            "pid": int(record["pid"]),
            "start_ticks": int(record["start_ticks"]),
            "argv_sha256": str(record["argv_sha256"]),
            "artifact_sha256": artifact_sha,
        }
        expected = dict(before)
        expected.update(
            status="completed",
            artifact_sha256=artifact_sha,
            completed_at_utc=observed.get("completed_at_utc"),
            orphan_reconciliation=expected_reconciliation,
        )
        if (
            not isinstance(observed.get("completed_at_utc"), str)
            or not observed["completed_at_utc"]
            or observed != expected
        ):
            raise ValueError(f"{name}: ledger is not an allowed reconciliation evolution")


def reconciliation_preflight(
    queue_path: Path, ledger_path: Path, capture_path: Path
) -> Dict[str, Any]:
    """Validate an immutable capture against the current queue and ledger."""
    captured = _read_json(capture_path, "orphan capture")
    queue = _read_queue(queue_path)
    jobs = _job_map(queue)
    workers = _validate_capture(
        captured, capture_path, queue_path, ledger_path, queue, jobs
    )
    ledger = _ledger(ledger_path, queue_path, jobs)
    _validate_ledger_evolution(captured, capture_path, ledger, jobs)
    statuses: Dict[str, str] = {}
    for name in sorted(workers):
        status = str(ledger["jobs"][name].get("status", ""))
        if status not in {"running", "completed"}:
            raise ValueError(f"{name}: current ledger status {status!r} is not reconcilable")
        statuses[name] = status
    return {
        "schema_version": 1,
        "kind": "guarded_queue_orphan_reconciliation_preflight",
        "capture": str(capture_path),
        "capture_sha256": sha256_file(capture_path),
        "queue_sha256": sha256_file(queue_path),
        "ledger_sha256_current": sha256_file(ledger_path),
        "workers": statuses,
    }


def validate_reconciliation_proof(
    queue_path: Path,
    ledger_path: Path,
    capture_path: Path,
    proof_path: Path,
) -> Dict[str, Any]:
    """Replay an existing proof against the capture and current ledger.

    This is deliberately read-only.  It is shared with the unattended
    supervisor so a supervisor restart cannot accept a stale proof merely
    because the file exists.
    """
    reconciliation_preflight(queue_path, ledger_path, capture_path)
    captured = _read_json(capture_path, "orphan capture")
    queue = _read_queue(queue_path)
    jobs = _job_map(queue)
    workers = captured["workers"]
    proof = _read_json(proof_path, "existing reconciliation proof")
    if (
        proof.get("schema_version") != 1
        or proof.get("kind") != "guarded_queue_orphan_reconciliation"
        or proof.get("queue_sha256") != sha256_file(queue_path)
        or Path(str(proof.get("capture", ""))).resolve() != capture_path.resolve()
        or proof.get("capture_sha256") != sha256_file(capture_path)
        or Path(str(proof.get("ledger", ""))).resolve() != ledger_path.resolve()
        or proof.get("ledger_sha256_after") != sha256_file(ledger_path)
    ):
        raise ValueError("existing reconciliation proof no longer verifies")
    proof_workers = proof.get("workers")
    if not isinstance(proof_workers, dict) or set(proof_workers) != set(workers):
        raise ValueError("existing reconciliation proof worker universe changed")
    ledger = _ledger(ledger_path, queue_path, jobs)
    for name, record in workers.items():
        reconciliation = proof_workers[name]
        entry = ledger["jobs"][name]
        if (
            entry.get("status") != "completed"
            or not isinstance(reconciliation, dict)
            or reconciliation != entry.get("orphan_reconciliation")
            or entry.get("artifact_sha256") != reconciliation.get("artifact_sha256")
        ):
            raise ValueError(f"{name}: proof differs from reconciled ledger")
        if (
            reconciliation.get("capture_sha256") != sha256_file(capture_path)
            or reconciliation.get("pid") != int(record["pid"])
            or reconciliation.get("start_ticks") != int(record["start_ticks"])
            or reconciliation.get("argv_sha256") != record["argv_sha256"]
        ):
            raise ValueError(f"{name}: proof identity differs from capture")
        completion = Path(str(record["completion_artifact"]))
        if not completion.is_file() or reconciliation.get(
            "artifact_sha256"
        ) != sha256_file(completion):
            raise ValueError(f"{name}: proof completion artifact changed")
    return proof


def reconcile(
    queue_path: Path,
    ledger_path: Path,
    capture_path: Path,
    proof_out: Path,
    *,
    poll_seconds: float,
) -> Dict[str, Any]:
    """Wait for captured workers and atomically reconcile completed cards."""
    reconciliation_preflight(queue_path, ledger_path, capture_path)
    captured = _read_json(capture_path, "orphan capture")
    queue = _read_queue(queue_path)
    jobs = _job_map(queue)
    workers = captured["workers"]
    if proof_out.exists():
        return validate_reconciliation_proof(
            queue_path, ledger_path, capture_path, proof_out
        )

    reconciled: Dict[str, Any] = {}
    while len(reconciled) != len(workers):
        progressed = False
        ledger = _ledger(ledger_path, queue_path, jobs)
        _validate_ledger_evolution(captured, capture_path, ledger, jobs)
        for name in sorted(workers):
            if name in reconciled:
                continue
            record = workers[name]
            entry = ledger["jobs"][name]
            if str(record.get("completion_artifact", "")) != str(
                jobs[name].get("completion_artifact", "")
            ):
                raise ValueError(f"{name}: captured completion path differs from queue")
            if record.get("argv_sha256") != canonical_sha256(jobs[name]["argv"]):
                raise ValueError(f"{name}: captured argv digest differs from queue")
            expected = _captured_identity(record, jobs[name]["argv"])
            live = process_identity(int(record["pid"]))
            if live is not None:
                if live != expected:
                    raise ValueError(f"{name}: PID was reused or worker identity changed")
                continue
            completion = Path(str(record["completion_artifact"]))
            if completion.name != "run.json" or not completion.is_file():
                raise FileNotFoundError(
                    f"{name}: orphan exited without declared completion run.json"
                )
            artifact_sha = sha256_file(completion)
            reconciliation = {
                "capture_sha256": sha256_file(capture_path),
                "pid": int(record["pid"]),
                "start_ticks": int(record["start_ticks"]),
                "argv_sha256": str(record["argv_sha256"]),
                "artifact_sha256": artifact_sha,
            }
            if entry.get("status") == "completed":
                if entry.get("artifact_sha256") != artifact_sha or entry.get(
                    "orphan_reconciliation"
                ) != reconciliation:
                    raise ValueError(f"{name}: prior orphan reconciliation differs")
            elif entry.get("status") == "running":
                entry.update(
                    status="completed",
                    artifact_sha256=artifact_sha,
                    completed_at_utc=_utc(),
                    orphan_reconciliation=reconciliation,
                )
                _atomic_json(ledger_path, ledger)
            else:
                raise ValueError(f"{name}: ledger status changed to {entry.get('status')!r}")
            reconciled[name] = reconciliation
            progressed = True
        if len(reconciled) != len(workers) and not progressed:
            time.sleep(poll_seconds)
    proof = {
        "schema_version": 1,
        "kind": "guarded_queue_orphan_reconciliation",
        "queue_sha256": sha256_file(queue_path),
        "capture": str(capture_path),
        "capture_sha256": sha256_file(capture_path),
        "ledger": str(ledger_path),
        "ledger_sha256_after": sha256_file(ledger_path),
        "workers": reconciled,
        "completed_at_utc": _utc(),
    }
    _write_once(proof_out, proof)
    return proof


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--queue", required=True, type=Path)
    capture_parser.add_argument("--ledger", required=True, type=Path)
    capture_parser.add_argument("--job-pid", action="append", default=[])
    capture_parser.add_argument("--out", required=True, type=Path)
    reconcile_parser = sub.add_parser("reconcile")
    reconcile_parser.add_argument("--queue", required=True, type=Path)
    reconcile_parser.add_argument("--ledger", required=True, type=Path)
    reconcile_parser.add_argument("--capture", required=True, type=Path)
    reconcile_parser.add_argument("--proof-out", required=True, type=Path)
    reconcile_parser.add_argument("--poll-seconds", type=float, default=30.0)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--queue", required=True, type=Path)
    preflight_parser.add_argument("--ledger", required=True, type=Path)
    preflight_parser.add_argument("--capture", required=True, type=Path)
    args = parser.parse_args(argv)
    if getattr(args, "poll_seconds", 1.0) < 1.0:
        parser.error("--poll-seconds must be at least one second")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "capture":
        payload = capture(args.queue, args.ledger, parse_job_pids(args.job_pid), args.out)
    elif args.command == "reconcile":
        payload = reconcile(
            args.queue,
            args.ledger,
            args.capture,
            args.proof_out,
            poll_seconds=args.poll_seconds,
        )
    else:
        payload = reconciliation_preflight(args.queue, args.ledger, args.capture)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
