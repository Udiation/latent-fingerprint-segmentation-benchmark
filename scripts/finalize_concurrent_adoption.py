#!/usr/bin/env python3
"""Finalize the mixed legacy/cache Stage-3 concurrency queue.

The source queue is never rewritten. Captured U-Net++ jobs retain their exact
legacy argv/config; jobs that had never launched use the receipt-bound 500-ppi
cache, eight-worker loader and operator-authorized threshold amendment.  The
output remains bound to the mixed queue instead of falsely claiming that these
prospective amendments were byte-identical to the obsolete source queue.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import canonical_sha256, sha256_file
import reconcile_guarded_orphans as orphan
import run_queue_guarded as guarded


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _read_yaml(path: Path, description: str) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
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


def _queue_identity(item: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "name": item["name"],
            "argv": item["argv"],
            "env": item.get("env", {}),
            "completion_artifact": item["completion_artifact"],
            "resources": item["resources"],
        }
    )


def _jobs(queue: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = queue.get("jobs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("queue lacks jobs")
    result: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("queue contains invalid job")
        name = str(item["name"])
        if name in result or item.get("job_identity_sha256") != _queue_identity(item):
            raise ValueError(f"invalid or duplicate queue job: {name}")
        result[name] = item
    return result


def finalize(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    # This also proves the source ledger still has only the evolution permitted
    # by its original immutable orphan capture.
    orphan.reconciliation_preflight(args.source_queue, args.source_ledger, args.capture)
    source_queue = _read_yaml(args.source_queue, "source queue")
    source_ledger = _read_json(args.source_ledger, "source ledger")
    capture = _read_json(args.capture, "capture")
    supplemental_queue = _read_yaml(args.supplemental_queue, "supplemental queue")
    supplemental_ledger = _read_json(args.supplemental_ledger, "supplemental ledger")
    plan = _read_json(args.external_plan, "external-worker plan")
    worker_proof = _read_json(args.external_proof, "external-worker proof")

    identities = {
        "source_queue": sha256_file(args.source_queue),
        "source_ledger": sha256_file(args.source_ledger),
        "capture": sha256_file(args.capture),
        "supplemental_queue": sha256_file(args.supplemental_queue),
        "supplemental_ledger": sha256_file(args.supplemental_ledger),
        "external_plan": sha256_file(args.external_plan),
        "external_proof": sha256_file(args.external_proof),
    }
    if source_ledger.get("queue_sha256") != identities["source_queue"]:
        raise ValueError("source ledger is not source-queue-bound")
    if supplemental_ledger.get("queue_sha256") != identities["supplemental_queue"]:
        raise ValueError("supplemental ledger is not supplemental-queue-bound")
    expected_plan = {
        "kind": "guarded_queue_external_worker_plan",
        "supplemental_queue_sha256": identities["supplemental_queue"],
        "source_queue_sha256": identities["source_queue"],
        "source_ledger_sha256_at_adoption": identities["source_ledger"],
        "capture_sha256": identities["capture"],
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            raise ValueError(f"external plan binding changed: {key}")
    expected_proof = {
        "kind": "guarded_queue_external_worker_completion",
        "external_workers_plan_sha256": identities["external_plan"],
        "supplemental_queue_sha256": identities["supplemental_queue"],
        "supplemental_ledger_sha256": identities["supplemental_ledger"],
    }
    for key, expected in expected_proof.items():
        if worker_proof.get(key) != expected:
            raise ValueError(f"external completion proof binding changed: {key}")

    source_jobs = _jobs(source_queue)
    supplemental_jobs = _jobs(supplemental_queue)
    source_entries = source_ledger.get("jobs")
    supplemental_entries = supplemental_ledger.get("jobs")
    captured_workers = capture.get("workers")
    proof_workers = worker_proof.get("workers")
    if not all(
        isinstance(value, dict)
        for value in (source_entries, supplemental_entries, captured_workers, proof_workers)
    ):
        raise ValueError("ledger/capture/proof job maps are invalid")
    if set(source_entries) != set(source_jobs):
        raise ValueError("source ledger universe differs from source queue")
    if set(supplemental_entries) != set(supplemental_jobs):
        raise ValueError("supplemental ledger universe differs from its queue")
    if set(captured_workers) != set(proof_workers):
        raise ValueError("external completion proof universe differs from capture")
    if set(source_jobs) != set(supplemental_jobs):
        raise ValueError("supplemental universe does not reproduce source")

    execution_binding = guarded.validate_adopted_queue_runtime(
        args.supplemental_queue, supplemental_queue
    )
    if execution_binding is None or plan.get(
        "execution_binding_sha256"
    ) != canonical_sha256(execution_binding):
        raise ValueError("finalizer execution binding differs from adoption plan")
    legacy_names = set(execution_binding["legacy_external_job_names"])
    amended_names = set(execution_binding["cache_amended_job_names"])
    if legacy_names != set(captured_workers) or legacy_names | amended_names != set(
        source_jobs
    ):
        raise ValueError("legacy/amended job partition differs from the capture")

    merged_entries: Dict[str, Dict[str, Any]] = {}
    for name, source_job in source_jobs.items():
        source_entry = source_entries[name]
        if source_entry.get("identity_sha256") != source_job["job_identity_sha256"]:
            raise ValueError(f"{name}: source ledger identity changed")
        completion = Path(str(source_job["completion_artifact"]))
        if not completion.is_file():
            raise FileNotFoundError(f"{name}: completion artifact is absent")
        artifact_sha = sha256_file(completion)
        supplemental_job = supplemental_jobs[name]
        supplemental_entry = supplemental_entries[name]
        for key in ("name", "env", "completion_artifact"):
            if supplemental_job.get(key, {} if key == "env" else None) != source_job.get(
                key, {} if key == "env" else None
            ):
                raise ValueError(f"{name}: immutable job field changed: {key}")
        source_resources = dict(source_job.get("resources", {}))
        supplemental_resources = dict(supplemental_job.get("resources", {}))
        source_ram = source_resources.pop("host_ram_gib", None)
        supplemental_ram = supplemental_resources.pop("host_ram_gib", None)
        supplemental_cpu = supplemental_resources.pop("cpu_workers", None)
        if (
            source_resources != supplemental_resources
            or not isinstance(source_ram, (int, float))
            or not isinstance(supplemental_ram, (int, float))
            or type(supplemental_cpu) is not int
            or supplemental_job.get("source_job_identity_sha256")
            != source_job["job_identity_sha256"]
        ):
            raise ValueError(f"{name}: scheduler resource amendment is invalid")
        if name in legacy_names:
            if (
                supplemental_job.get("argv") != source_job.get("argv")
                or supplemental_job.get("adoption_role") != "legacy_captured_job"
            ):
                raise ValueError(f"{name}: captured legacy argv changed")
        else:
            source_argv = list(source_job["argv"])
            amended_argv = list(supplemental_job["argv"])
            expected = list(source_argv)
            expected[1] = amended_argv[1]
            source_config_index = expected.index("--config") + 1
            amended_config_index = amended_argv.index("--config") + 1
            expected[source_config_index] = amended_argv[amended_config_index]
            expected.extend(
                [
                    "--cache-dir",
                    str(execution_binding["cache_dir"]),
                    "--threshold-amendment",
                    str(execution_binding["threshold_amendment"]),
                ]
            )
            if (
                amended_argv != expected
                or supplemental_job.get("adoption_role")
                != "cache_amended_pending_job"
                or supplemental_cpu != 8
            ):
                raise ValueError(f"{name}: pending-job amendment exceeded policy")
        if supplemental_entry.get("identity_sha256") != supplemental_job[
            "job_identity_sha256"
        ] or supplemental_entry.get("status") != "completed":
            raise ValueError(f"{name}: supplemental job is not completed/authenticated")
        if supplemental_entry.get("artifact_sha256") != artifact_sha:
            raise ValueError(f"{name}: supplemental artifact digest changed")
        merged = dict(supplemental_entry)
        merged["identity_sha256"] = supplemental_job["job_identity_sha256"]
        merged["source_job_identity_sha256"] = source_job["job_identity_sha256"]
        merged["attempts"] = int(source_entry.get("attempts", 0)) + int(
            supplemental_entry.get("attempts", 0)
        )
        for counter in ("ordinary_retries", "oom_retries", "preemptions"):
            merged[counter] = int(source_entry.get(counter, 0)) + int(
                supplemental_entry.get(counter, 0)
            )
        adoption = {
            "role": supplemental_job["adoption_role"],
            "supplemental_queue_sha256": identities["supplemental_queue"],
            "supplemental_ledger_sha256": identities["supplemental_ledger"],
            "supplemental_job_identity_sha256": supplemental_job[
                "job_identity_sha256"
            ],
            "source_declared_host_ram_gib": float(source_ram),
            "runtime_reserved_host_ram_gib": float(supplemental_ram),
            "runtime_reserved_cpu_workers": supplemental_cpu,
        }
        if name in captured_workers:
            record = captured_workers[name]
            completed = proof_workers[name]
            identity_expected = {
                "pid": int(record["pid"]),
                "start_ticks": int(record["start_ticks"]),
                "argv_sha256": str(record["argv_sha256"]),
            }
            for key, value in identity_expected.items():
                if completed.get(key) != value:
                    raise ValueError(f"{name}: adopted-worker proof changed: {key}")
            adoption.update(
                role="captured_external_worker",
                capture_sha256=identities["capture"],
                external_plan_sha256=identities["external_plan"],
                external_proof_sha256=identities["external_proof"],
                external_outcome=completed.get("outcome"),
                gpu=completed.get("gpu"),
                **identity_expected,
            )
            if completed.get("outcome") == "completed_externally" and completed.get(
                "artifact_sha256"
            ) != artifact_sha:
                raise ValueError(f"{name}: external artifact digest changed")
            if completed.get("outcome") not in {
                "completed_externally",
                "completed_before_adoption",
                "resume_required",
            }:
                raise ValueError(f"{name}: invalid external-worker outcome")
        merged["concurrency_adoption"] = adoption
        merged.pop("worker", None)
        merged_entries[name] = merged

    runtime_policy = supplemental_queue.get("runtime_policy")
    if not isinstance(runtime_policy, dict):
        raise ValueError("supplemental queue lacks its runtime resource policy")
    aggregate_workers = runtime_policy.get(
        "aggregate_data_workers_at_full_concurrency"
    )
    non_loader_capacity = runtime_policy.get(
        "cpu_capacity_left_for_non_loader_work"
    )
    if type(aggregate_workers) is not int or type(non_loader_capacity) is not int:
        raise ValueError("supplemental runtime resource policy is invalid")

    merged_ledger = {
        "schema_version": 1,
        "queue_sha256": identities["supplemental_queue"],
        "jobs": merged_entries,
        "concurrency_adoption": {
            "schema_version": 1,
            "source_ledger_sha256": identities["source_ledger"],
            "supplemental_queue_sha256": identities["supplemental_queue"],
            "supplemental_ledger_sha256": identities["supplemental_ledger"],
            "external_plan_sha256": identities["external_plan"],
            "external_proof_sha256": identities["external_proof"],
            "execution_binding_sha256": canonical_sha256(execution_binding),
            "captured_jobs_argv_unchanged": True,
            "pending_jobs_cache_and_threshold_amended": True,
            "source_queue_was_not_rewritten": True,
            "new_jobs_use_verified_500ppi_cache": True,
            "new_jobs_use_eight_loader_workers": True,
            "preserved_sweep_metrics_selection_ineligible_until_expanded_grid_reselection": True,
            "full_concurrency_data_worker_reservation": aggregate_workers,
            "non_loader_cpu_capacity": non_loader_capacity,
        },
    }
    proof = {
        "schema_version": 1,
        "kind": "stage3_concurrency_adoption_finalization",
        "identities": identities,
        "source_job_count": len(source_jobs),
        "captured_worker_count": len(captured_workers),
        "supplemental_job_count": len(supplemental_jobs),
        "terminal_completed_count": sum(
            entry.get("status") == "completed" for entry in merged_entries.values()
        ),
        "merged_ledger": str(args.merged_ledger_out),
        "merged_ledger_sha256": canonical_sha256(merged_ledger),
        "ground_truth_opened": False,
        "result_metrics_opened": False,
        "downstream_stage3_delivery_contract": {
            "ordered_source_images_sha256": (
                "canonical_sha256(sorted([{image_id,source_sha256}, ...], "
                "key=image_id)) "
                "over all 2010 Table-A test sources"
            ),
            "threshold_amendment_sha256": execution_binding[
                "threshold_amendment_sha256"
            ],
            "probability_cards_required_for_validation_and_test": True,
            "per_epoch_training_profile_cards_required": True,
            "computed_after_native_training_and_inference": True,
            "threshold_policy_attestation": {
                "allowed_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "validation_only": True,
                "old_grid_rejected": True,
                "amended_records_cover_all_successful_native_seeds_except_stage2_unet_seed0": True,
                "stage2_unet_seed0_reuse_required_fields": [
                    "legacy_provenance_delivery_sha256",
                    "threshold_reselection_delivery_sha256",
                    "replacement_scientific_rows_sha256",
                    "reused_reselected_not_rescored",
                ],
                "amended_training_record_required_fields": [
                    "method",
                    "seed",
                    "selected_threshold",
                    "config",
                    "run_card",
                    "checkpoint",
                    "raw_validation_probability_card",
                    "selected_validation_probability_card",
                    "threshold_selector_receipt",
                    "threshold_bound_inference_card",
                    "test_probability_card",
                    "per_epoch_training_profile_card",
                ],
                "preserved_unetpp_sweep_selection_rule": {
                    "legacy_run_metric_eligible": False,
                    "required_frozen_checkpoint_validation_selector_receipts": 2,
                    "learning_rate_selected_from": "expanded_grid_validation_dice_only",
                    "full_seed0_may_launch_before_both_receipts": False,
                },
            },
        },
    }
    _write_once(args.merged_ledger_out, merged_ledger)
    proof["merged_ledger_file_sha256"] = sha256_file(args.merged_ledger_out)
    _write_once(args.proof_out, proof)
    return merged_ledger, proof


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", required=True, type=Path)
    parser.add_argument("--source-ledger", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--supplemental-queue", required=True, type=Path)
    parser.add_argument("--supplemental-ledger", required=True, type=Path)
    parser.add_argument("--external-plan", required=True, type=Path)
    parser.add_argument("--external-proof", required=True, type=Path)
    parser.add_argument("--merged-ledger-out", required=True, type=Path)
    parser.add_argument("--proof-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _, proof = finalize(args)
    print(json.dumps({"status": "finalized", "proof": proof}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
