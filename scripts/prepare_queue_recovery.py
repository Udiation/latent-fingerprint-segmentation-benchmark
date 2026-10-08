#!/usr/bin/env python3
"""Create an append-only guarded-queue recovery after an audited incident.

The source queue and ledger are read-only.  Completed jobs retain their exact
artifact identities; non-completed jobs receive a fresh retry budget in a new
ledger whose queue binds the source artifacts and an operator incident note.
An existing external-worker adoption plan is rebound without changing worker
PIDs, argv, resources, or the scientific execution binding.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import canonical_sha256, sha256_file
import run_queue_guarded as guarded


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _read_yaml(path: Path, description: str) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a YAML object")
    return value


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    _write_once(path, json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _write_yaml_once(path: Path, value: Mapping[str, Any]) -> None:
    _write_once(path, yaml.safe_dump(dict(value), sort_keys=False))


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    source_queue = _read_yaml(args.source_queue, "source queue")
    source_ledger = _read_json(args.source_ledger, "source ledger")
    incident_note = args.incident_note.read_text(encoding="utf-8")
    if not incident_note.strip():
        raise ValueError("incident note is empty")
    if source_ledger.get("queue_sha256") != sha256_file(args.source_queue):
        raise ValueError("source ledger is not bound to the source queue")
    jobs = guarded.load_guarded_jobs(args.source_queue)
    entries = source_ledger.get("jobs")
    if not isinstance(entries, dict) or set(entries) != {job.name for job in jobs}:
        raise ValueError("source ledger job universe differs from the source queue")
    source_binding = guarded.validate_adopted_queue_runtime(
        args.source_queue, source_queue
    )
    if source_binding is None:
        raise ValueError("recovery requires an authenticated adopted training queue")

    external_workers: Dict[str, guarded.ExternalWorker] = {}
    external_worker_records: Dict[str, Dict[str, Any]] = {}
    external_precompleted: Dict[str, Dict[str, Any]] = {}
    source_plan: Optional[Dict[str, Any]] = None
    if args.source_external_plan is not None:
        source_plan = _read_json(args.source_external_plan, "source external plan")
        cpu_budget = source_plan.get("cpu_budget")
        if not isinstance(cpu_budget, dict):
            raise ValueError("source external plan lacks its CPU budget")
        workers, validated_plan = guarded.load_external_workers(
            args.source_external_plan,
            queue=args.source_queue,
            gpus=list(source_plan.get("allowed_gpus", [])),
            max_per_gpu=int(source_plan.get("max_jobs_per_gpu", 0)),
            max_project_ram_gib=float(source_plan.get("max_project_ram_gib", 0.0)),
            max_cpu_cores=int(cpu_budget.get("max_cpu_cores", 0)),
            memory_fraction=float(source_plan.get("memory_fraction", 0.0)),
        )
        source_plan = validated_plan
        external_workers = {worker.name: worker for worker in workers}
        external_worker_records = {
            str(item.get("name")): dict(item)
            for item in source_plan.get("workers", [])
            if isinstance(item, dict)
        }
        external_precompleted = {
            str(item.get("name")): dict(item)
            for item in source_plan.get("precompleted_workers", [])
            if isinstance(item, dict)
        }

    recovery_queue = copy.deepcopy(source_queue)
    recovery_queue["recovery"] = {
        "schema_version": 1,
        "kind": "append_only_guarded_queue_recovery",
        "source_queue": str(args.source_queue),
        "source_queue_sha256": sha256_file(args.source_queue),
        "source_ledger": str(args.source_ledger),
        "source_ledger_sha256": sha256_file(args.source_ledger),
        "incident_note": str(args.incident_note),
        "incident_note_sha256": sha256_file(args.incident_note),
        "source_ledger_unchanged": True,
    }
    _write_yaml_once(args.out_queue, recovery_queue)
    new_jobs = guarded.load_guarded_jobs(args.out_queue)
    jobs_by_name = {job.name: job for job in new_jobs}
    captured_external_names = set(external_worker_records) | set(external_precompleted)
    if not captured_external_names <= set(jobs_by_name):
        raise ValueError("source external plan contains a job outside the source queue")
    for name in captured_external_names:
        record = external_worker_records.get(name) or external_precompleted[name]
        job = jobs_by_name[name]
        if (
            record.get("source_job_identity_sha256") != job.identity_sha256
            or record.get("argv_sha256")
            != canonical_sha256(job.base_job.argv)
            or Path(str(record.get("completion_artifact", ""))).resolve()
            != job.completion_artifact.resolve()
        ):
            raise ValueError(f"{name}: external-plan job identity changed")
        worker = external_workers.get(name)
        if worker is not None and (
            worker.cpu_workers != job.cpu_workers
            or worker.host_ram_gib != job.host_ram_gib
        ):
            raise ValueError(f"{name}: external-plan resource identity changed")
    new_state = guarded.initial_state(args.out_queue, new_jobs)
    recovered = []
    preserved = []
    for job in new_jobs:
        source = entries[job.name]
        status = source.get("status")
        target = new_state["jobs"][job.name]
        if status == "completed":
            if (
                not job.completion_artifact.is_file()
                or sha256_file(job.completion_artifact)
                != source.get("artifact_sha256")
            ):
                raise ValueError(f"{job.name}: completed artifact changed")
            target.update(
                status="completed",
                artifact_sha256=str(source["artifact_sha256"]),
                completed_at_utc=str(source.get("completed_at_utc", "")),
                preserved_from_source_ledger=True,
            )
            preserved.append(job.name)
            continue
        if status == "running":
            worker = source.get("worker")
            if isinstance(worker, Mapping) and guarded._worker_is_exactly_live(worker):
                raise RuntimeError(
                    f"{job.name}: source-ledger worker is still live; reconcile first"
                )
        if job.completion_artifact.exists():
            worker = external_workers.get(job.name)
            precompleted = external_precompleted.get(job.name)
            if worker is None and precompleted is None:
                raise ValueError(
                    f"{job.name}: unauthenticated non-completed artifact exists"
                )
            if worker is not None:
                if guarded._external_is_exactly_live(worker):
                    raise RuntimeError(
                        f"{job.name}: external worker published its artifact but is still live"
                    )
                if guarded._process_start_ticks(worker.pid) is not None:
                    raise RuntimeError(f"{job.name}: external worker PID was reused")
                if (
                    worker.completion_artifact.resolve()
                    != job.completion_artifact.resolve()
                ):
                    raise ValueError(f"{job.name}: external completion path changed")
            else:
                assert precompleted is not None
                if (
                    Path(str(precompleted.get("completion_artifact", ""))).resolve()
                    != job.completion_artifact.resolve()
                    or precompleted.get("artifact_sha256")
                    != sha256_file(job.completion_artifact)
                ):
                    raise ValueError(f"{job.name}: precompleted external artifact changed")
            target.update(
                status="completed",
                artifact_sha256=sha256_file(job.completion_artifact),
                completed_externally_during_recovery=True,
                source_external_plan_sha256=sha256_file(args.source_external_plan),
            )
            preserved.append(job.name)
            continue
        if status not in {"pending", "running", "retrying", "preempted", "failed"}:
            raise ValueError(f"{job.name}: unsupported source status {status!r}")
        target["superseded_source_status"] = str(status)
        target["superseded_source_attempts"] = int(source.get("attempts", 0))
        target["recovery_authorization_sha256"] = sha256_file(args.incident_note)
        recovered.append(job.name)
    if not recovered:
        raise ValueError("source ledger has no unfinished jobs to recover")
    _write_json_once(args.out_state, new_state)

    external_plan_identity = None
    if args.source_external_plan is not None:
        assert source_plan is not None
        rebound = copy.deepcopy(source_plan)
        rebound["supplemental_queue"] = str(args.out_queue)
        rebound["supplemental_queue_sha256"] = sha256_file(args.out_queue)
        rebound["recovery_source_external_plan"] = {
            "path": str(args.source_external_plan),
            "sha256": sha256_file(args.source_external_plan),
        }
        rebound["precompleted_workers"] = [
            item
            for item in rebound.get("precompleted_workers", [])
            if item.get("name") not in set(preserved)
        ]
        rebound["workers"] = [
            item
            for item in rebound.get("workers", [])
            if item.get("name") not in set(preserved)
        ]
        _write_json_once(args.out_external_plan, rebound)
        external_plan_identity = {
            "path": str(args.out_external_plan),
            "sha256": sha256_file(args.out_external_plan),
        }
    elif args.out_external_plan is not None:
        raise ValueError("--out-external-plan requires --source-external-plan")

    proof = {
        "schema_version": 1,
        "kind": "append_only_guarded_queue_recovery_proof",
        "source_queue": {
            "path": str(args.source_queue),
            "sha256": sha256_file(args.source_queue),
        },
        "source_ledger": {
            "path": str(args.source_ledger),
            "sha256": sha256_file(args.source_ledger),
        },
        "incident_note": {
            "path": str(args.incident_note),
            "sha256": sha256_file(args.incident_note),
        },
        "recovery_queue": {
            "path": str(args.out_queue),
            "sha256": sha256_file(args.out_queue),
        },
        "recovery_state": {
            "path": str(args.out_state),
            "sha256": sha256_file(args.out_state),
        },
        "external_plan": external_plan_identity,
        "execution_binding_sha256": canonical_sha256(source_binding),
        "preserved_completed_jobs": preserved,
        "reset_unfinished_jobs": recovered,
        "source_artifacts_modified": False,
    }
    _write_json_once(args.proof_out, proof)
    return proof


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", required=True, type=Path)
    parser.add_argument("--source-ledger", required=True, type=Path)
    parser.add_argument("--incident-note", required=True, type=Path)
    parser.add_argument("--out-queue", required=True, type=Path)
    parser.add_argument("--out-state", required=True, type=Path)
    parser.add_argument("--proof-out", required=True, type=Path)
    parser.add_argument("--source-external-plan", type=Path)
    parser.add_argument("--out-external-plan", type=Path)
    args = parser.parse_args(argv)
    if (args.source_external_plan is None) != (args.out_external_plan is None):
        parser.error("external-plan inputs and outputs must be supplied together")
    source_paths = {args.source_queue.resolve(), args.source_ledger.resolve()}
    output_paths = {
        args.out_queue.resolve(), args.out_state.resolve(), args.proof_out.resolve()
    }
    if args.out_external_plan is not None:
        output_paths.add(args.out_external_plan.resolve())
    if source_paths & output_paths or len(output_paths) != (
        4 if args.out_external_plan is not None else 3
    ):
        parser.error("recovery outputs must be distinct from immutable source artifacts")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    proof = prepare(parse_args(argv))
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
