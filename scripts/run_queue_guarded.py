#!/usr/bin/env python3
"""Resumable GPU queue with artifact ledger, one retry, and host-RAM guard."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json, canonical_sha256, sha256_file
from segbench.deep_utils import build_queue_batch_contract, config_digest
import run_queue as base

GIB = 1024**3
TIMING_ISOLATED_PHASES = {
    "stage3_native_test_inference",
    "stage3_drunet_mc_dropout_aux",
}
ADOPTED_QUEUE_KIND = "stage3_concurrency_supplemental_queue"
ADOPTED_BINDING_KIND = "stage3_cache_concurrency_execution_binding"
ADOPTED_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ADOPTED_DATALOADER = {
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
}
EXPECTED_ABSENT_QUALITY_MASKS = 66
OOM_BATCH_POLICY = {
    "schema_version": 1,
    "kind": "single_oom_half_batch_double_accumulation",
    "max_oom_retries": 1,
    "only_transition": "base_to_half_batch_double_accumulation",
    "effective_batch_preserved": True,
}


def _oom_transition_evidence() -> Dict[str, Any]:
    """Return immutable ledger evidence for the one authorized OOM retry."""
    retry = base.Job(name="batch-contract", argv=["true"], env={}, oom_retries=1)
    return {
        "from_oom_retry_count": 0,
        "to_oom_retry_count": 1,
        "batch_scale": 0.5,
        "gradient_accumulation_scale": 2.0,
        "effective_batch_preserved": True,
        "launcher_environment": base.queue_batch_environment(retry),
        "policy_sha256": canonical_sha256(OOM_BATCH_POLICY),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_artifact(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


@dataclass
class GuardedJob:
    base_job: base.Job
    completion_artifact: Path
    host_ram_gib: float
    cpu_workers: int
    identity_sha256: str
    ordinary_retries: int = 0

    @property
    def name(self) -> str:
        return self.base_job.name


@dataclass
class GuardedRunning:
    job: GuardedJob
    process: base.Running


@dataclass(frozen=True)
class ExternalWorker:
    """Exactly identified worker inherited from a failed prior controller."""

    name: str
    pid: int
    start_ticks: int
    argv_sha256: str
    gpu: int
    completion_artifact: Path
    preempt_path: Path
    host_ram_gib: float
    cpu_workers: int


@dataclass(frozen=True)
class CoexistingWorker:
    """Read-only worker owned by a separately authenticated live controller.

    Unlike ``ExternalWorker``, this controller never adopts, resumes, signals,
    or writes state for this process.  It is counted only for GPU slots,
    process ownership, RAM, and CPU reservations.
    """

    name: str
    pid: int
    start_ticks: int
    argv_sha256: str
    gpu: int
    host_ram_gib: float
    cpu_workers: int


def _declared_identity(item: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "name": item["name"],
            "argv": item["argv"],
            "env": item.get("env", {}),
            "completion_artifact": item["completion_artifact"],
            "resources": item["resources"],
        }
    )


def _single_option(argv: Sequence[str], option: str) -> str:
    if list(argv).count(option) != 1:
        raise ValueError(f"job argv must contain exactly one {option}")
    index = list(argv).index(option)
    if index + 1 >= len(argv):
        raise ValueError(f"job argv {option} lacks a value")
    return str(argv[index + 1])


def validate_adopted_queue_runtime(
    queue: Path, payload: Optional[Mapping[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Replay cache/config/code bindings before every amended-job launch."""
    if payload is None:
        loaded = yaml.safe_load(queue.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("queue must be an object")
        payload = loaded
    if payload.get("kind") != ADOPTED_QUEUE_KIND:
        if "execution_binding" in payload:
            raise ValueError("execution binding is valid only on the adopted queue")
        return None
    binding = payload.get("execution_binding")
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != 1
        or binding.get("kind") != ADOPTED_BINDING_KIND
    ):
        raise ValueError("adopted queue lacks its execution binding")
    def validate_code_identity(value: Any, description: str) -> Path:
        if not isinstance(value, dict):
            raise ValueError(f"adopted queue lacks its {description} code identity")
        root = Path(str(value.get("root", "")))
        if not root.is_absolute() or not root.is_dir():
            raise ValueError(f"adopted {description} code root is unavailable")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
        )
        if (
            commit != value.get("git_commit")
            or tree != value.get("git_tree")
            or status.strip()
        ):
            raise ValueError(f"adopted {description} code identity or cleanliness changed")
        file_hashes = value.get("files_sha256")
        if not isinstance(file_hashes, dict) or not file_hashes:
            raise ValueError(f"adopted {description} code file hashes are absent")
        for relative, expected in file_hashes.items():
            path = root / str(relative)
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"adopted {description} code file changed: {relative}")
        return root

    code = binding.get("code")
    code_root = validate_code_identity(code, "default")
    job_code = binding.get("job_code", {})
    if not isinstance(job_code, dict):
        raise ValueError("adopted per-job code identities are invalid")
    job_code_roots = {
        str(name): validate_code_identity(value, f"job {name}")
        for name, value in job_code.items()
    }

    cache_dir = Path(str(binding.get("cache_dir", "")))
    receipt = Path(str(binding.get("cache_receipt", "")))
    manifest = Path(str(binding.get("cache_manifest", "")))
    if (
        not cache_dir.is_absolute()
        or receipt.resolve() != (cache_dir / "receipt.json").resolve()
        or not receipt.is_file()
        or sha256_file(receipt) != binding.get("cache_receipt_sha256")
        or not manifest.is_absolute()
        or not manifest.is_file()
        or sha256_file(manifest) != binding.get("cache_manifest_sha256")
    ):
        raise ValueError("adopted cache receipt/manifest path or digest changed")
    with manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_by_id = {str(row.get("image_id", "")): row for row in manifest_rows}
    if (
        not manifest_by_id
        or "" in manifest_by_id
        or len(manifest_by_id) != len(manifest_rows)
    ):
        raise ValueError("adopted cache manifest ids are invalid or duplicated")
    cache_payload = json.loads(receipt.read_text(encoding="utf-8"))
    verification = cache_payload.get("verification", {})
    required_mask_columns = ["mask_path", "quality_mask_path"]
    if (
        cache_payload.get("schema_version") != 1
        or cache_payload.get("kind") != "uint8_500ppi_preprocessing_cache"
        or cache_payload.get("target_ppi") != 500.0
        or cache_payload.get("image_format") != "uint8_npy"
        or cache_payload.get("mask_format") != "uint8_0_255_npy"
        or cache_payload.get("debug_limited") is not False
        or cache_payload.get("manifest_sha256")
        != binding.get("cache_manifest_sha256")
        or sorted(cache_payload.get("mask_columns", []))
        != sorted(required_mask_columns)
        or not isinstance(verification, dict)
        or verification.get("requested_count") != 20
        or verification.get("verified_count") != 20
        or verification.get("array_identity") is not True
    ):
        raise ValueError("adopted cache receipt no longer passes its identity gate")
    records = cache_payload.get("records")
    if (
        not isinstance(records, list)
        or cache_payload.get("record_count") != len(records)
    ):
        raise ValueError("adopted cache receipt records are absent")
    by_id = {
        str(record.get("image_id")): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("image_id"), str)
    }
    if len(by_id) != len(records):
        raise ValueError("adopted cache receipt ids are invalid or duplicated")
    if set(by_id) != set(manifest_by_id):
        raise ValueError("adopted cache does not cover the complete manifest")
    for record in records:
        image_id = str(record["image_id"])
        shape = record.get("shape")
        if (
            Path(image_id).name != image_id
            or image_id in {".", ".."}
            or not _is_sha256(record.get("image_sha256"))
            or not _is_sha256(record.get("source_image_sha256"))
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise ValueError("adopted cache record image identity or shape is invalid")
        for key in ("mask_sha256", "source_mask_sha256"):
            hashes = record.get(key)
            if not isinstance(hashes, dict) or set(hashes) != set(required_mask_columns):
                raise ValueError("adopted cache record lacks ROI/quality mask keys")
            if any(value is not None and not _is_sha256(value) for value in hashes.values()):
                raise ValueError("adopted cache record contains an invalid mask digest")
        if any(
            (record["mask_sha256"][column] is None)
            != (record["source_mask_sha256"][column] is None)
            for column in required_mask_columns
        ):
            raise ValueError("adopted source/cache mask absence disagrees")
        row = manifest_by_id[image_id]
        for column in required_mask_columns:
            source_present = bool(str(row.get(column, "")).strip())
            if source_present != (record["source_mask_sha256"][column] is not None):
                raise ValueError("adopted manifest/cache mask presence disagrees")
        if not (cache_dir / "images" / f"{image_id}.npy").is_file():
            raise FileNotFoundError(f"adopted cached image is absent: {image_id}")
        for column in required_mask_columns:
            cached_mask = cache_dir / "masks" / column / f"{image_id}.npy"
            if (record["mask_sha256"][column] is not None) != cached_mask.is_file():
                raise ValueError("adopted cache receipt/artifact mask presence disagrees")
    if sum(
        record["mask_sha256"]["quality_mask_path"] is None
        for record in records
    ) != EXPECTED_ABSENT_QUALITY_MASKS:
        raise ValueError("adopted cache does not represent 66 absent quality masks")
    verified_ids = verification.get("image_ids")
    if not isinstance(verified_ids, list) or len(verified_ids) != 20:
        raise ValueError("adopted cache verification ids changed")
    for image_id in verified_ids:
        if Path(str(image_id)).name != str(image_id) or str(image_id) in {".", ".."}:
            raise ValueError("unsafe adopted cache verification id")
        record = by_id.get(str(image_id))
        if not isinstance(record, dict):
            raise ValueError("adopted cache verification record is absent")
        image = cache_dir / "images" / f"{image_id}.npy"
        if not image.is_file() or sha256_file(image) != record.get("image_sha256"):
            raise ValueError(f"verified cached image changed: {image_id}")
        row = manifest_by_id[str(image_id)]
        source_image = _manifest_artifact(str(row.get("image_path", "")), manifest)
        if (
            not source_image.is_file()
            or sha256_file(source_image) != record.get("source_image_sha256")
        ):
            raise ValueError(f"verified source image changed: {image_id}")
        for column in required_mask_columns:
            expected_mask = record["mask_sha256"][column]
            if expected_mask is None:
                continue
            mask = cache_dir / "masks" / column / f"{image_id}.npy"
            if not mask.is_file() or sha256_file(mask) != expected_mask:
                raise ValueError(f"verified cached mask changed: {image_id}/{column}")
            source_mask = _manifest_artifact(str(row.get(column, "")), manifest)
            if (
                not source_mask.is_file()
                or sha256_file(source_mask)
                != record["source_mask_sha256"][column]
            ):
                raise ValueError(f"verified source mask changed: {image_id}/{column}")
    amendment = Path(str(binding.get("threshold_amendment", "")))
    if not amendment.is_file() or sha256_file(amendment) != binding.get(
        "threshold_amendment_sha256"
    ):
        raise ValueError("threshold amendment digest changed")
    amendment_payload = json.loads(amendment.read_text(encoding="utf-8"))
    replacement = amendment_payload.get("replacement_policy", {})
    if (
        amendment_payload.get("kind") != "stage4_threshold_policy_amendment"
        or not isinstance(replacement, dict)
        or replacement.get("allowed_trainable_grid") != ADOPTED_THRESHOLDS
        or replacement.get("selection_data") != "validation_only"
    ):
        raise ValueError("threshold amendment policy changed")

    jobs = payload.get("jobs")
    runtime_policy = payload.get("runtime_policy")
    hashes = binding.get("generated_config_sha256")
    batch_hashes = binding.get("generated_batch_contract_sha256")
    legacy_hashes = binding.get("legacy_job_files_sha256")
    legacy = set(binding.get("legacy_external_job_names", []))
    amended = set(binding.get("cache_amended_job_names", []))
    if (
        not isinstance(jobs, list)
        or not isinstance(runtime_policy, dict)
        or not isinstance(hashes, dict)
        or not isinstance(batch_hashes, dict)
        or not isinstance(legacy_hashes, dict)
        or binding.get("dataloader") != ADOPTED_DATALOADER
        or binding.get("oom_batch_policy") != OOM_BATCH_POLICY
        or legacy & amended
        or legacy | amended != {str(item.get("name")) for item in jobs}
        or set(job_code_roots) - {str(item.get("name")) for item in jobs}
        or set(hashes) != amended
        or set(batch_hashes) != amended
        or set(legacy_hashes) != legacy
    ):
        raise ValueError("adopted job/config universe changed")
    for item in jobs:
        name = str(item["name"])
        argv = item.get("argv")
        if not isinstance(argv, list):
            raise ValueError(f"{name}: argv is invalid")
        role = item.get("adoption_role")
        resources = item.get("resources")
        if (
            not isinstance(resources, dict)
            or type(resources.get("cpu_workers")) is not int
            or resources.get("host_ram_gib")
            != runtime_policy.get("host_ram_gib_per_job")
        ):
            raise ValueError(f"{name}: CPU-worker reservation is absent")
        if name in legacy:
            config_path = Path(_single_option(argv, "--config"))
            legacy_files = legacy_hashes[name]
            entrypoint_path = Path(argv[1]) if len(argv) > 1 else Path()
            if (
                not isinstance(legacy_files, dict)
                or str(entrypoint_path) != legacy_files.get("entrypoint")
                or not entrypoint_path.is_file()
                or sha256_file(entrypoint_path)
                != legacy_files.get("entrypoint_sha256")
                or str(config_path) != legacy_files.get("config")
                or not config_path.is_file()
                or sha256_file(config_path) != legacy_files.get("config_sha256")
            ):
                raise ValueError(f"{name}: legacy entrypoint/config changed")
            legacy_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            legacy_training = (
                legacy_config.get("training", {})
                if isinstance(legacy_config, dict)
                else {}
            )
            if (
                role != "legacy_captured_job"
                or "--cache-dir" in argv
                or "--threshold-amendment" in argv
                or resources["cpu_workers"]
                != int(legacy_training.get("num_workers", -1))
            ):
                raise ValueError(f"{name}: legacy captured contract changed")
            continue
        if role != "cache_amended_pending_job":
            raise ValueError(f"{name}: amended role changed")
        if Path(_single_option(argv, "--cache-dir")).resolve() != cache_dir.resolve():
            raise ValueError(f"{name}: cache directory differs from the binding")
        if Path(_single_option(argv, "--threshold-amendment")).resolve() != amendment.resolve():
            raise ValueError(f"{name}: threshold amendment differs from the binding")
        entrypoint = Path(argv[1]).resolve() if len(argv) > 1 else Path()
        selected_code_root = job_code_roots.get(name, code_root)
        if entrypoint.parent != (selected_code_root / "segbench" / "scripts").resolve():
            raise ValueError(f"{name}: training entrypoint escaped the bound code tree")
        config_path = Path(_single_option(argv, "--config"))
        if not config_path.is_file() or sha256_file(config_path) != hashes[name]:
            raise ValueError(f"{name}: generated config digest changed")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        training = config.get("training", {}) if isinstance(config, dict) else {}
        data = config.get("data", {}) if isinstance(config, dict) else {}
        runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
        configured_fraction = (
            float(runtime.get("memory_fraction", math.nan))
            if isinstance(runtime, dict)
            else math.nan
        )
        expected_batch_contract = (
            build_queue_batch_contract(training) if isinstance(training, dict) else None
        )
        expected_batch_digest = (
            config_digest(expected_batch_contract)
            if isinstance(expected_batch_contract, dict)
            else None
        )
        if (
            not isinstance(training, dict)
            or any(
                training.get(key) != value
                for key, value in ADOPTED_DATALOADER.items()
            )
            or resources["cpu_workers"] != 8
            or training.get("threshold_candidates") != ADOPTED_THRESHOLDS
            or runtime.get("queue_batch_contract") != expected_batch_contract
            or runtime.get("queue_batch_contract_sha256") != expected_batch_digest
            or batch_hashes.get(name) != expected_batch_digest
            or not isinstance(data, dict)
            or float(data.get("target_ppi", math.nan)) != 500.0
            or not isinstance(runtime, dict)
            or not math.isfinite(configured_fraction)
            or not math.isclose(
                configured_fraction,
                float(runtime_policy.get("memory_fraction", math.nan)),
                abs_tol=1e-12,
            )
            or sha256_file(Path(_single_option(argv, "--manifest")))
            != binding.get("cache_manifest_sha256")
        ):
            raise ValueError(f"{name}: amended loader/scientific config changed")
    return dict(binding)


def load_guarded_jobs(path: Path) -> List[GuardedJob]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("launch_authorized") is not True:
        raise ValueError("queue is absent or has not passed its phase barrier")
    raw = payload.get("jobs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("authorized queue must contain non-empty jobs")
    validate_adopted_queue_runtime(path, payload)
    # Reuse every argv/name/env/resume validation from the original queue.
    base_jobs = base.load_jobs(path)
    guarded = []
    for item, base_job in zip(raw, base_jobs):
        completion = item.get("completion_artifact")
        resources = item.get("resources")
        if not isinstance(completion, str) or not completion:
            raise ValueError(f"{base_job.name}: completion_artifact is required")
        if not isinstance(resources, dict) or "host_ram_gib" not in resources:
            raise ValueError(f"{base_job.name}: declared host_ram_gib is required")
        ram = float(resources["host_ram_gib"])
        cpu_workers = resources.get("cpu_workers", 0)
        if not math.isfinite(ram) or ram <= 0.0:
            raise ValueError(f"{base_job.name}: invalid host RAM reservation")
        if type(cpu_workers) is not int or cpu_workers < 0:
            raise ValueError(f"{base_job.name}: invalid CPU-worker reservation")
        observed_identity = _declared_identity(item)
        if item.get("job_identity_sha256") != observed_identity:
            raise ValueError(f"{base_job.name}: job identity hash mismatch")
        guarded.append(
            GuardedJob(
                base_job,
                Path(completion),
                ram,
                cpu_workers,
                observed_identity,
            )
        )
    return guarded


def queue_launcher_attestation(
    queue: Path,
    *,
    gpus: Sequence[int],
    max_per_gpu: int,
    memory_fraction: float,
) -> Optional[Dict[str, Any]]:
    """Validate and attest inference-only timing-isolation requirements.

    Training queues predate this policy and intentionally keep their v1 ledger
    schema.  A Stage-3 held-out inference queue is never accepted without the
    registered single-job-per-GPU policy and an exact launcher attestation.
    """
    payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("queue must be a YAML object")
    phase = payload.get("phase")
    if phase not in TIMING_ISOLATED_PHASES:
        return None
    required = payload.get("required_max_jobs_per_gpu")
    required_args = payload.get("required_launcher_args")
    if type(required) is not int or required != 1:
        raise ValueError("inference queue must require exactly one job per GPU")
    if required_args != ["--max-per-gpu", "1"]:
        raise ValueError("inference queue launcher requirements changed")
    if max_per_gpu != required:
        raise ValueError("launcher --max-per-gpu differs from inference queue policy")
    registered_gpus = payload.get("required_gpu_indices")
    if (
        not isinstance(registered_gpus, list)
        or not registered_gpus
        or any(type(value) is not int or value < 0 for value in registered_gpus)
        or len(set(registered_gpus)) != len(registered_gpus)
    ):
        raise ValueError("timing-isolated queue lacks registered GPU indices")
    if list(gpus) != registered_gpus:
        raise ValueError("launcher GPUs differ from timing-isolated queue policy")
    if not abs(float(memory_fraction) - 0.3) <= 1e-12:
        raise ValueError("inference timing launcher must use memory fraction 0.3")
    return {
        "schema_version": 1,
        "queue_phase": phase,
        "queue_sha256": sha256_file(queue),
        "gpus": registered_gpus,
        "max_per_gpu": 1,
        "memory_fraction": 0.3,
        "required_launcher_args": ["--max-per-gpu", "1"],
    }


def initial_state(
    queue: Path,
    jobs: Sequence[GuardedJob],
    launcher_attestation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "schema_version": 2 if launcher_attestation is not None else 1,
        "queue_sha256": sha256_file(queue),
        "jobs": {
            job.name: {
                "identity_sha256": job.identity_sha256,
                "status": "pending",
                "attempts": 0,
                "ordinary_retries": 0,
                "oom_retries": 0,
                "preemptions": 0,
            }
            for job in jobs
        },
    }
    if launcher_attestation is not None:
        state["launcher_attestation"] = dict(launcher_attestation)
    return state


def load_state(
    path: Path,
    queue: Path,
    jobs: Sequence[GuardedJob],
    launcher_attestation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    expected = initial_state(queue, jobs, launcher_attestation)
    if not path.exists():
        atomic_json(path, expected)
        return expected
    state = json.loads(path.read_text(encoding="utf-8"))
    if launcher_attestation is not None:
        if state.get("schema_version") != 2 or state.get(
            "launcher_attestation"
        ) != dict(launcher_attestation):
            raise ValueError(
                "inference ledger lacks the exact launcher timing attestation"
            )
    elif state.get("schema_version") != 1 or "launcher_attestation" in state:
        raise ValueError("legacy training ledger schema changed")
    if state.get("queue_sha256") != expected["queue_sha256"]:
        raise ValueError("queue file changed after ledger creation")
    if set(state.get("jobs", {})) != set(expected["jobs"]):
        raise ValueError("ledger job universe differs from queue")
    for job in jobs:
        entry = state["jobs"][job.name]
        if entry.get("identity_sha256") != job.identity_sha256:
            raise ValueError(f"{job.name}: ledger identity differs from queue")
        status = entry.get("status")
        if status == "completed":
            if not job.completion_artifact.is_file():
                raise FileNotFoundError(f"{job.name}: completed artifact disappeared")
            if sha256_file(job.completion_artifact) != entry.get("artifact_sha256"):
                raise ValueError(f"{job.name}: completed artifact changed")
        elif status == "running":
            worker = entry.get("worker")
            if not isinstance(worker, Mapping):
                raise RuntimeError(
                    f"{job.name}: running ledger lacks worker identity; explicit "
                    "orphan reconciliation is required"
                )
            expected_launch_environment = base.queue_batch_environment(
                base.Job(
                    name=job.name,
                    argv=list(job.base_job.argv),
                    env=dict(job.base_job.env),
                    oom_retries=int(entry.get("oom_retries", 0)),
                )
            )
            if worker.get("queue_batch_environment") != expected_launch_environment:
                raise ValueError(
                    f"{job.name}: running worker batch environment differs from ledger"
                )
            if _worker_is_exactly_live(worker):
                raise RuntimeError(
                    f"{job.name}: exact worker remains live after controller exit; "
                    "explicit orphan reconciliation is required"
                )
            elif job.completion_artifact.is_file():
                # The exact worker vanished but durably published its declared
                # immutable completion artifact before the controller died.
                complete_job(state, job)
                entry["recovered_after_controller_exit"] = True
                entry.pop("worker", None)
            else:
                entry["status"] = "pending"
                entry.pop("worker", None)
        elif status in {"retrying", "preempted"}:
            entry["status"] = "pending"
            entry.pop("worker", None)
        elif status not in {"pending", "failed"}:
            raise ValueError(f"{job.name}: invalid ledger state {status!r}")
        counters = {
            key: entry.get(key, 0)
            for key in ("ordinary_retries", "oom_retries", "preemptions")
        }
        if any(type(value) is not int or value < 0 for value in counters.values()):
            raise ValueError(f"{job.name}: ledger retry counters are invalid")
        if counters["oom_retries"] not in {0, 1}:
            raise ValueError(f"{job.name}: ledger exceeds the single OOM retry")
        transition = entry.get("oom_batch_transition")
        if counters["oom_retries"] == 0 and transition is not None:
            raise ValueError(f"{job.name}: ledger has an unearned OOM transition")
        if counters["oom_retries"] == 1 and transition != _oom_transition_evidence():
            raise ValueError(f"{job.name}: ledger OOM transition evidence changed")
        job.ordinary_retries = int(entry.get("ordinary_retries", 0))
        job.base_job.oom_retries = int(entry.get("oom_retries", 0))
        job.base_job.preemptions = int(entry.get("preemptions", 0))
    atomic_json(path, state)
    return state


def complete_job(state: Dict[str, Any], job: GuardedJob) -> None:
    if not job.completion_artifact.is_file():
        raise FileNotFoundError(
            f"{job.name}: exit 0 without declared completion artifact"
        )
    entry = state["jobs"][job.name]
    entry.update(
        status="completed",
        artifact_sha256=sha256_file(job.completion_artifact),
        completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    entry.pop("worker", None)


def _process_start_ticks(pid: int) -> Optional[int]:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # comm is parenthesized and may contain spaces; fields following its
        # final ')' begin at field 3, and starttime is field 22.
        return int(text.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _process_argv_sha256(pid: int) -> Optional[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    argv = [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]
    return canonical_sha256(argv) if argv else None


def _worker_is_exactly_live(worker: Mapping[str, Any]) -> bool:
    pid = worker.get("pid")
    start_ticks = worker.get("start_ticks")
    argv_sha = worker.get("argv_sha256")
    if type(pid) is not int or type(start_ticks) is not int or not isinstance(argv_sha, str):
        return False
    return (
        _process_start_ticks(pid) == start_ticks
        and _process_argv_sha256(pid) == argv_sha
    )


def _external_is_exactly_live(worker: ExternalWorker) -> bool:
    return (
        _process_start_ticks(worker.pid) == worker.start_ticks
        and _process_argv_sha256(worker.pid) == worker.argv_sha256
    )


def _coexisting_is_exactly_live(worker: CoexistingWorker) -> bool:
    return (
        _process_start_ticks(worker.pid) == worker.start_ticks
        and _process_argv_sha256(worker.pid) == worker.argv_sha256
    )


def load_coexisting_controller_plan(
    path: Path,
    *,
    target_queue: Path,
    gpus: Sequence[int],
    max_per_gpu: int,
    max_project_ram_gib: float,
    max_cpu_cores: int,
    memory_fraction: float,
) -> Dict[str, Any]:
    """Authenticate an immutable, read-only controller-coexistence contract."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("coexisting-controller plan must be an object")
    expected = {
        "schema_version": 1,
        "kind": "guarded_queue_coexisting_controller_plan",
        "target_queue_sha256": sha256_file(target_queue),
        "allowed_gpus": list(gpus),
        "max_jobs_per_gpu": max_per_gpu,
        "max_project_ram_gib": max_project_ram_gib,
        "max_cpu_cores": max_cpu_cores,
        "memory_fraction": memory_fraction,
        "read_only_source": True,
        "may_signal_source": False,
        "may_write_source_ledger": False,
    }
    disagreements = {
        key: (value.get(key), required)
        for key, required in expected.items()
        if value.get(key) != required
    }
    if disagreements:
        raise ValueError(
            f"coexisting-controller plan binding changed: {disagreements}"
        )
    source_queue = Path(str(value.get("source_queue", "")))
    source_state = Path(str(value.get("source_state", "")))
    source_proof = Path(str(value.get("source_proof", "")))
    if (
        not source_queue.is_absolute()
        or not source_state.is_absolute()
        or not source_proof.is_absolute()
        or not source_queue.is_file()
        or not source_state.is_file()
        or not source_proof.is_file()
        or sha256_file(source_queue) != value.get("source_queue_sha256")
        or sha256_file(source_proof) != value.get("source_proof_sha256")
    ):
        raise ValueError("coexisting source queue/state/proof identity changed")
    # Replays every cache/config/code binding carried by the mixed recovery.
    if validate_adopted_queue_runtime(source_queue) is None:
        raise ValueError("coexisting source is not an authenticated adopted queue")
    source_jobs = load_guarded_jobs(source_queue)
    declared = value.get("source_jobs")
    observed = [
        {
            "name": job.name,
            "identity_sha256": job.identity_sha256,
            "completion_artifact": str(job.completion_artifact),
            "host_ram_gib": job.host_ram_gib,
            "cpu_workers": job.cpu_workers,
        }
        for job in source_jobs
    ]
    if declared != observed or not observed:
        raise ValueError("coexisting source job universe changed")
    controller = value.get("source_controller")
    if (
        not isinstance(controller, dict)
        or type(controller.get("pid")) is not int
        or type(controller.get("start_ticks")) is not int
        or not _is_sha256(controller.get("argv_sha256"))
    ):
        raise ValueError("coexisting source controller identity is invalid")
    return value


def observe_coexisting_workers(plan: Mapping[str, Any]) -> List[CoexistingWorker]:
    """Return one stable, exactly authenticated read-only source snapshot.

    The source ledger is read twice.  If it changes during inspection the
    operation retries; no launch decision is made from a torn snapshot.
    Once co-scheduling starts, the source may only move monotonically from
    ``running`` to ``completed``.  Retry/preemption/failure therefore fails
    closed instead of racing a second controller's relaunch.
    """
    source_queue = Path(str(plan["source_queue"]))
    source_state = Path(str(plan["source_state"]))
    if sha256_file(source_queue) != plan["source_queue_sha256"]:
        raise ValueError("coexisting source queue changed")
    if validate_adopted_queue_runtime(source_queue) is None:
        raise ValueError("coexisting source queue lost its execution binding")
    source_jobs = {job.name: job for job in load_guarded_jobs(source_queue)}
    declared_names = [str(item["name"]) for item in plan["source_jobs"]]
    if list(source_jobs) != declared_names:
        raise ValueError("coexisting source job order/universe changed")
    stable: Optional[bytes] = None
    payload: Optional[Dict[str, Any]] = None
    for _ in range(5):
        first = source_state.read_bytes()
        candidate = json.loads(first)
        second = source_state.read_bytes()
        if first == second and isinstance(candidate, dict):
            stable, payload = first, candidate
            break
        time.sleep(0.05)
    if stable is None or payload is None:
        raise RuntimeError("coexisting source ledger did not yield a stable snapshot")
    if payload.get("queue_sha256") != plan["source_queue_sha256"]:
        raise ValueError("coexisting source ledger is not bound to its queue")
    states = payload.get("jobs")
    if not isinstance(states, dict) or set(states) != set(source_jobs):
        raise ValueError("coexisting source ledger job universe changed")
    running: List[CoexistingWorker] = []
    for name, job in source_jobs.items():
        entry = states[name]
        if not isinstance(entry, dict) or entry.get("identity_sha256") != job.identity_sha256:
            raise ValueError(f"{name}: coexisting ledger identity changed")
        status = entry.get("status")
        if status == "completed":
            if (
                not job.completion_artifact.is_file()
                or entry.get("artifact_sha256") != sha256_file(job.completion_artifact)
            ):
                raise ValueError(f"{name}: coexisting completion artifact changed")
            continue
        if status != "running":
            raise RuntimeError(
                f"{name}: coexisting source left the running/completed contract: {status!r}"
            )
        worker = entry.get("worker")
        if not isinstance(worker, dict):
            raise ValueError(f"{name}: coexisting running identity is absent")
        observed = CoexistingWorker(
            name=name,
            pid=int(worker.get("pid", -1)),
            start_ticks=int(worker.get("start_ticks", -1)),
            argv_sha256=str(worker.get("argv_sha256", "")),
            gpu=int(worker.get("gpu", -1)),
            host_ram_gib=job.host_ram_gib,
            cpu_workers=job.cpu_workers,
        )
        if observed.gpu not in plan["allowed_gpus"]:
            raise ValueError(f"{name}: coexisting worker escaped allowed GPUs")
        if not _coexisting_is_exactly_live(observed):
            # Exit-before-ledger-update is safe only when the declared output
            # already exists; do not count a vanished PID as owned.
            if job.completion_artifact.is_file() and _process_start_ticks(observed.pid) is None:
                continue
            raise RuntimeError(f"{name}: coexisting worker PID identity changed")
        running.append(observed)
    controller = plan["source_controller"]
    if running and not (
        _process_start_ticks(int(controller["pid"])) == int(controller["start_ticks"])
        and _process_argv_sha256(int(controller["pid"])) == controller["argv_sha256"]
    ):
        raise RuntimeError("coexisting source controller identity changed")
    per_gpu = {
        gpu: sum(worker.gpu == gpu for worker in running)
        for gpu in plan["allowed_gpus"]
    }
    source_cap = int(plan.get("source_max_jobs_per_gpu", 0))
    if source_cap not in {1, 2, 3} or any(count > source_cap for count in per_gpu.values()):
        raise ValueError("coexisting source exceeds its registered GPU cap")
    return running


def load_external_workers(
    path: Path,
    *,
    queue: Path,
    gpus: Sequence[int],
    max_per_gpu: int,
    max_project_ram_gib: float,
    max_cpu_cores: int,
    memory_fraction: float,
) -> tuple[List[ExternalWorker], Dict[str, Any]]:
    """Load a write-once adoption plan and authenticate its live workers.

    The plan is produced only after the former controller and reconciler have
    exited.  It binds the supplemental queue, source capture, physical device
    assignment, cooperative preemption path, and conservative RAM override.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("external-worker plan must be an object")
    expected = {
        "schema_version": 1,
        "kind": "guarded_queue_external_worker_plan",
        "supplemental_queue_sha256": sha256_file(queue),
        "allowed_gpus": list(gpus),
        "max_jobs_per_gpu": max_per_gpu,
        "max_project_ram_gib": max_project_ram_gib,
        "memory_fraction": memory_fraction,
    }
    for key, observed in expected.items():
        if value.get(key) != observed:
            raise ValueError(f"external-worker plan binding changed: {key}")
    cpu_budget = value.get("cpu_budget")
    expected_aggregate_workers = (
        len(gpus) * max_per_gpu * ADOPTED_DATALOADER["num_workers"]
    )
    if (
        not isinstance(cpu_budget, dict)
        or cpu_budget.get("max_cpu_cores") != max_cpu_cores
        or cpu_budget.get("aggregate_data_workers") != expected_aggregate_workers
        or cpu_budget.get("headroom_for_controller_and_reporting")
        != max_cpu_cores - expected_aggregate_workers
    ):
        raise ValueError("external-worker CPU budget changed")
    queue_payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    if not isinstance(queue_payload, dict):
        raise ValueError("supplemental queue is invalid")
    binding = validate_adopted_queue_runtime(queue, queue_payload)
    if binding is None or value.get("execution_binding_sha256") != canonical_sha256(
        binding
    ):
        raise ValueError("external-worker plan does not bind the execution contract")
    runtime_policy = queue_payload.get("runtime_policy", {})
    if (
        value.get("reservation_override_gib")
        != runtime_policy.get("host_ram_gib_per_job")
        or value.get("generated_config_sha256")
        != binding.get("generated_config_sha256")
        or value.get("generated_batch_contract_sha256")
        != binding.get("generated_batch_contract_sha256")
        or value.get("oom_batch_policy") != OOM_BATCH_POLICY
        or binding.get("oom_batch_policy") != OOM_BATCH_POLICY
    ):
        raise ValueError("external-worker plan resource/batch binding changed")
    raw = value.get("workers")
    if not isinstance(raw, list):
        raise ValueError("external-worker plan workers must be a list")
    workers: List[ExternalWorker] = []
    names: set[str] = set()
    pids: set[int] = set()
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError("external-worker record is not an object")
        name = record.get("name")
        pid = record.get("pid")
        start_ticks = record.get("start_ticks")
        argv_sha = record.get("argv_sha256")
        gpu = record.get("gpu")
        completion = Path(str(record.get("completion_artifact", "")))
        preempt = Path(str(record.get("preempt_path", "")))
        reservation = record.get("host_ram_gib")
        cpu_workers = record.get("cpu_workers")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or type(pid) is not int
            or pid <= 1
            or pid in pids
            or type(start_ticks) is not int
            or start_ticks <= 0
            or not isinstance(argv_sha, str)
            or len(argv_sha) != 64
            or type(gpu) is not int
            or gpu not in gpus
            or not completion.is_absolute()
            or not preempt.is_absolute()
            or not isinstance(reservation, (int, float))
            or not math.isfinite(float(reservation))
            or float(reservation) <= 0.0
            or float(reservation) > max_project_ram_gib
            or type(cpu_workers) is not int
            or cpu_workers < 0
            or cpu_workers > max_cpu_cores
        ):
            raise ValueError(f"invalid external-worker record: {name!r}")
        worker = ExternalWorker(
            name=name,
            pid=pid,
            start_ticks=start_ticks,
            argv_sha256=argv_sha,
            gpu=gpu,
            completion_artifact=completion,
            preempt_path=preempt,
            host_ram_gib=float(reservation),
            cpu_workers=cpu_workers,
        )
        if not _external_is_exactly_live(worker) and not completion.is_file():
            current_start = _process_start_ticks(pid)
            if current_start is not None:
                raise RuntimeError(f"{name}: external worker PID identity changed")
            # A captured training worker may have cooperatively checkpointed
            # between plan creation and controller launch.  The same frozen
            # supplemental argv uses --resume auto, so main() safely relaunches
            # this exact job rather than treating it as a duplicate.
        workers.append(worker)
        names.add(name)
        pids.add(pid)
    per_gpu = {
        gpu: sum(worker.gpu == gpu for worker in workers) for gpu in gpus
    }
    if any(count > max_per_gpu for count in per_gpu.values()):
        raise ValueError("external workers already exceed per-GPU concurrency cap")
    if sum(worker.host_ram_gib for worker in workers) > max_project_ram_gib:
        raise ValueError("external-worker reservations exceed project RAM cap")
    if sum(worker.cpu_workers for worker in workers) > max_cpu_cores:
        raise ValueError("external-worker CPU reservations exceed project CPU cap")
    precompleted = value.get("precompleted_workers", [])
    if not isinstance(precompleted, list) or any(
        not isinstance(record, dict) for record in precompleted
    ):
        raise ValueError("external-worker plan precompleted records are invalid")
    precompleted_names = [record.get("name") for record in precompleted]
    if (
        any(not isinstance(name, str) or not name for name in precompleted_names)
        or len(precompleted_names) != len(set(precompleted_names))
        or set(precompleted_names) & names
        or not workers and not precompleted
    ):
        raise ValueError("external-worker plan captured universe is invalid")
    return workers, value


def worker_identity(process: object, argv: Sequence[str], gpu: int) -> Dict[str, Any]:
    pid = int(process.pid)
    start_ticks = _process_start_ticks(pid)
    observed_argv = _process_argv_sha256(pid)
    expected_argv = canonical_sha256(list(argv))
    if start_ticks is None or observed_argv != expected_argv:
        raise RuntimeError("launched worker identity could not be authenticated")
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "argv_sha256": observed_argv,
        "gpu": int(gpu),
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def authenticate_or_terminate(
    launched: base.Running, argv: Sequence[str], gpu: int
) -> Dict[str, Any]:
    """Authenticate a just-started child or synchronously stop the owned child."""
    try:
        return worker_identity(launched.process, argv, gpu)
    except BaseException:
        launched.process.terminate()
        launched.process.wait(timeout=30)
        launched.log_handle.close()
        launched.preempt_path.unlink(missing_ok=True)
        raise


def retry_action(code: int, job: GuardedJob, *, oom: bool) -> str:
    """Return one of complete/preempt/retry_oom/retry_failure/fail."""
    if job.base_job.oom_retries not in {0, 1}:
        raise ValueError("queue job exceeds the single authorized OOM retry")
    if code == 0:
        return "complete"
    if code == base.PREEMPT_EXIT:
        return "preempt"
    if oom and job.base_job.oom_retries == 0:
        return "retry_oom"
    if not oom and job.ordinary_retries == 0:
        return "retry_failure"
    return "fail"


def launch_candidate_transaction(
    *,
    candidate: GuardedJob,
    gpu: int,
    pending: List[GuardedJob],
    state: Dict[str, Any],
    state_path: Path,
    log_dir: Path,
    memory_fraction: float,
    stop_requested: Callable[[], bool],
    launcher: Callable[[base.Job, int, Path, float], base.Running] = base.launch,
) -> Optional[base.Running]:
    """Persist a reversible launch transition and never spawn after a stop check."""
    if stop_requested():
        return None
    pending.remove(candidate)
    entry = state["jobs"][candidate.name]
    previous_status = entry.get("status", "pending")
    previous_attempts = int(entry.get("attempts", 0))
    entry["status"] = "running"
    entry["attempts"] = previous_attempts + 1
    atomic_json(state_path, state)
    if stop_requested():
        entry["status"] = previous_status
        entry["attempts"] = previous_attempts
        pending.insert(0, candidate)
        atomic_json(state_path, state)
        return None
    try:
        launched = launcher(candidate.base_job, gpu, log_dir, memory_fraction)
    except BaseException:
        entry["status"] = previous_status
        entry["attempts"] = previous_attempts
        pending.insert(0, candidate)
        atomic_json(state_path, state)
        raise
    if stop_requested():
        launched.preempt_path.touch()
        launched.preempt_requested = True
    return launched


def _pid_pss_bytes(pid: int) -> int:
    rollup = Path(f"/proc/{pid}/smaps_rollup")
    try:
        for line in rollup.read_text(encoding="utf-8").splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    # A worker may exit between descendant enumeration and either procfs read.
    # Linux then raises any of several OSError subclasses (including
    # ProcessLookupError); disappearance is a zero-usage observation, not a
    # controller-fatal condition.
    except (OSError, ValueError, IndexError):
        pass
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def owned_ram_gib(running: Sequence[GuardedRunning]) -> float:
    pids = set()
    for item in running:
        pids.update(base.descendants(item.process.process.pid))
    return sum(_pid_pss_bytes(pid) for pid in pids) / GIB


def external_owned_ram_gib(
    workers: Sequence[ExternalWorker | CoexistingWorker],
) -> float:
    pids: set[int] = set()
    for worker in workers:
        pids.update(base.descendants(worker.pid))
    return sum(_pid_pss_bytes(pid) for pid in pids) / GIB


def owned_pids_including_external(
    running: Sequence[GuardedRunning],
    external: Sequence[ExternalWorker | CoexistingWorker],
    gpus: Sequence[int],
) -> Dict[int, set[int]]:
    """Return all locally launched and exactly adopted process trees by GPU."""
    owned = base.owned_pids_by_gpu([item.process for item in running], gpus)
    for worker in external:
        owned.setdefault(worker.gpu, set()).update(base.descendants(worker.pid))
    return owned


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / GIB
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def can_reserve(
    job: GuardedJob,
    running: Sequence[GuardedRunning],
    *,
    max_project_ram_gib: float,
    min_available_ram_gib: float,
    external_reserved_gib: float = 0.0,
    external_owned_gib: float = 0.0,
    max_cpu_cores: int = 0,
    external_cpu_workers: int = 0,
) -> bool:
    reserved = external_reserved_gib + sum(item.job.host_ram_gib for item in running)
    observed = external_owned_gib + owned_ram_gib(running)
    if reserved + job.host_ram_gib > max_project_ram_gib:
        return False
    if observed + job.host_ram_gib > max_project_ram_gib:
        return False
    if max_cpu_cores and (
        external_cpu_workers
        + sum(item.job.cpu_workers for item in running)
        + job.cpu_workers
        > max_cpu_cores
    ):
        return False
    return available_ram_gib() >= min_available_ram_gib + job.host_ram_gib


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--max-per-gpu", type=int, default=3)
    parser.add_argument("--min-free-mib", type=int, default=40 * 1024)
    parser.add_argument("--foreign-process-mib", type=int, default=2 * 1024)
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--min-disk-free-fraction", type=float, default=0.15)
    parser.add_argument("--max-project-ram-gib", type=float, required=True)
    parser.add_argument("--max-cpu-cores", type=int, default=0)
    parser.add_argument("--min-available-ram-gib", type=float, default=0.0)
    parser.add_argument("--external-workers-plan", type=Path)
    parser.add_argument("--external-workers-proof", type=Path)
    parser.add_argument("--coexisting-controller-plan", type=Path)
    args = parser.parse_args(argv)
    args.gpus = base.parse_gpu_list(args.gpus)
    if not 1 <= args.max_per_gpu <= 3:
        parser.error("--max-per-gpu must be in 1..3")
    if not 0.0 < args.memory_fraction <= 0.3:
        parser.error("--memory-fraction must be in (0,0.3]")
    if not math.isfinite(args.max_project_ram_gib) or args.max_project_ram_gib <= 0.0:
        parser.error("--max-project-ram-gib must be a positive runtime cap")
    if args.max_cpu_cores < 0:
        parser.error("--max-cpu-cores must be non-negative")
    if (args.external_workers_plan is None) != (args.external_workers_proof is None):
        parser.error(
            "--external-workers-plan and --external-workers-proof must be used together"
        )
    if args.coexisting_controller_plan is not None and args.external_workers_plan is not None:
        parser.error(
            "read-only controller coexistence cannot be combined with worker adoption"
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # This precedes queue loading and ledger creation: an unavailable profiling
    # dependency must never consume attempts or mutate immutable state.
    base.require_gpu_telemetry_dependency()
    jobs = load_guarded_jobs(args.jobs)
    queue_payload = yaml.safe_load(args.jobs.read_text(encoding="utf-8"))
    if isinstance(queue_payload, dict) and queue_payload.get("kind") == ADOPTED_QUEUE_KIND:
        policy = queue_payload.get("runtime_policy", {})
        expected_aggregate_workers = (
            len(args.gpus) * args.max_per_gpu * ADOPTED_DATALOADER["num_workers"]
        )
        if (
            not isinstance(policy, dict)
            or policy.get("allowed_gpus") != list(args.gpus)
            or policy.get("max_jobs_per_gpu") != args.max_per_gpu
            or not math.isclose(
                float(policy.get("memory_fraction", math.nan)),
                args.memory_fraction,
                abs_tol=1e-12,
            )
            or policy.get("max_project_ram_gib") != args.max_project_ram_gib
            or args.max_cpu_cores != policy.get("max_cpu_cores")
            or policy.get("host_ram_gib_per_job")
            != (jobs[0].host_ram_gib if jobs else None)
            or any(
                not math.isclose(
                    job.host_ram_gib,
                    float(policy.get("host_ram_gib_per_job", math.nan)),
                    abs_tol=1e-12,
                )
                for job in jobs
            )
            or not math.isclose(
                float(policy.get("aggregate_gpu_memory_fraction", math.nan)),
                args.max_per_gpu * args.memory_fraction,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(policy.get("min_disk_free_fraction", math.nan)),
                args.min_disk_free_fraction,
                abs_tol=1e-12,
            )
            or policy.get("max_data_workers_per_job")
            != ADOPTED_DATALOADER["num_workers"]
            or policy.get("aggregate_data_workers_at_full_concurrency")
            != expected_aggregate_workers
            or policy.get("cpu_capacity_left_for_non_loader_work")
            != args.max_cpu_cores - expected_aggregate_workers
        ):
            raise ValueError("launcher resources differ from adopted queue policy")
    if any(job.host_ram_gib > args.max_project_ram_gib for job in jobs):
        raise ValueError("a queue reservation exceeds --max-project-ram-gib")
    attestation = queue_launcher_attestation(
        args.jobs,
        gpus=args.gpus,
        max_per_gpu=args.max_per_gpu,
        memory_fraction=args.memory_fraction,
    )
    state = load_state(args.state, args.jobs, jobs, attestation)
    pending = [job for job in jobs if state["jobs"][job.name]["status"] != "completed"]
    if any(state["jobs"][job.name]["status"] == "failed" for job in pending):
        raise RuntimeError("ledger contains a terminal failed job; explicit audit required")
    log_dir = args.results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    base.disk_guard(args.results, args.min_disk_free_fraction)
    running: List[GuardedRunning] = []
    external: List[ExternalWorker] = []
    external_plan: Optional[Dict[str, Any]] = None
    external_completed: Dict[str, Dict[str, Any]] = {}
    coexisting_plan: Optional[Dict[str, Any]] = None
    if args.external_workers_plan is not None:
        external, external_plan = load_external_workers(
            args.external_workers_plan,
            queue=args.jobs,
            gpus=args.gpus,
            max_per_gpu=args.max_per_gpu,
            max_project_ram_gib=args.max_project_ram_gib,
            max_cpu_cores=args.max_cpu_cores,
            memory_fraction=args.memory_fraction,
        )
        by_name = {job.name: job for job in jobs}
        if any(
            worker.name not in by_name
            or worker.completion_artifact.resolve()
            != by_name[worker.name].completion_artifact.resolve()
            or not math.isclose(
                worker.host_ram_gib,
                by_name[worker.name].host_ram_gib,
                abs_tol=1e-12,
            )
            or worker.cpu_workers != by_name[worker.name].cpu_workers
            for worker in external
        ):
            raise ValueError(
                "external workers are not exact jobs in the supplemental queue"
            )
        for record in external_plan.get("precompleted_workers", []):
            name = str(record["name"])
            matching = next((job for job in pending if job.name == name), None)
            if matching is None:
                raise ValueError(f"{name}: precompleted job is not pending")
            artifact = matching.completion_artifact
            if (
                not artifact.is_file()
                or record.get("artifact_sha256") != sha256_file(artifact)
            ):
                raise ValueError(f"{name}: precompleted artifact changed")
            pending.remove(matching)
            complete_job(state, matching)
            adoption = {
                "outcome": "completed_before_adoption",
                "pid": int(record["pid"]),
                "start_ticks": int(record["start_ticks"]),
                "argv_sha256": str(record["argv_sha256"]),
                "gpu": None,
                "artifact_sha256": str(record["artifact_sha256"]),
            }
            state["jobs"][name]["external_adoption"] = adoption
            external_completed[name] = adoption
        if external_plan.get("precompleted_workers"):
            atomic_json(args.state, state)
    if args.coexisting_controller_plan is not None:
        coexisting_plan = load_coexisting_controller_plan(
            args.coexisting_controller_plan,
            target_queue=args.jobs,
            gpus=args.gpus,
            max_per_gpu=args.max_per_gpu,
            max_project_ram_gib=args.max_project_ram_gib,
            max_cpu_cores=args.max_cpu_cores,
            memory_fraction=args.memory_fraction,
        )
    failed: List[str] = []
    stop_requested = False

    def request_stop(signum, frame) -> None:  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        for item in running:
            item.process.preempt_path.touch()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while pending or running or external:
        coexisting = (
            observe_coexisting_workers(coexisting_plan)
            if coexisting_plan is not None
            else []
        )
        devices, processes = base.gpu_snapshot()
        remaining_external: List[ExternalWorker] = []
        for worker in external:
            prior = state["jobs"].get(worker.name, {}).get("external_adoption")
            if isinstance(prior, dict):
                identity = {
                    "gpu": worker.gpu,
                    "pid": worker.pid,
                    "start_ticks": worker.start_ticks,
                    "argv_sha256": worker.argv_sha256,
                }
                if any(prior.get(key) != value for key, value in identity.items()):
                    raise ValueError(
                        f"{worker.name}: persisted external adoption identity changed"
                    )
                outcome = prior.get("outcome")
                if outcome == "completed_externally":
                    if (
                        not worker.completion_artifact.is_file()
                        or prior.get("artifact_sha256")
                        != sha256_file(worker.completion_artifact)
                    ):
                        raise ValueError(
                            f"{worker.name}: persisted external artifact changed"
                        )
                elif outcome != "resume_required":
                    raise ValueError(
                        f"{worker.name}: persisted external outcome is invalid"
                    )
                external_completed[worker.name] = dict(prior)
                continue
            if _external_is_exactly_live(worker):
                remaining_external.append(worker)
                continue
            if _process_start_ticks(worker.pid) is not None:
                raise RuntimeError(
                    f"{worker.name}: adopted PID was reused or changed identity"
                )
            matching = next((job for job in pending if job.name == worker.name), None)
            record = {
                "gpu": worker.gpu,
                "pid": worker.pid,
                "start_ticks": worker.start_ticks,
                "argv_sha256": worker.argv_sha256,
            }
            if worker.completion_artifact.is_file():
                artifact_sha = sha256_file(worker.completion_artifact)
                if matching is not None:
                    pending.remove(matching)
                    complete_job(state, matching)
                    state["jobs"][worker.name]["external_adoption"] = {
                        **record,
                        "outcome": "completed_externally",
                        "artifact_sha256": artifact_sha,
                    }
                    atomic_json(args.state, state)
                elif not (
                    state["jobs"].get(worker.name, {}).get("status") == "completed"
                    and state["jobs"][worker.name].get("artifact_sha256")
                    == artifact_sha
                ):
                    raise RuntimeError(
                        f"{worker.name}: external completion has no matching ledger job"
                    )
                record.update(
                    outcome="completed_externally",
                    artifact_sha256=artifact_sha,
                )
            else:
                # The frozen argv uses --resume auto.  Once the exact captured
                # worker has vanished, retaining its queue job as pending is a
                # safe checkpoint resume rather than a duplicate launch.
                if matching is None:
                    raise RuntimeError(
                        f"{worker.name}: vanished external worker is not resumable"
                    )
                record["outcome"] = "resume_required"
                state["jobs"][worker.name]["external_adoption"] = dict(record)
                atomic_json(args.state, state)
            external_completed[worker.name] = record
        external = remaining_external

        occupancy = [*external, *coexisting]
        owned = owned_pids_including_external(running, occupancy, args.gpus)

        # The union of adopted and locally launched process trees is owned.
        # A foreign heavy process cooperatively preempts all project workers on
        # that physical GPU and prevents any additional launch.
        for gpu in args.gpus:
            if base.foreign_heavy_process(
                gpu, devices, processes, owned[gpu], args.foreign_process_mib
            ):
                for worker in external:
                    if worker.gpu == gpu:
                        worker.preempt_path.touch()
                for item in running:
                    if item.process.gpu == gpu and not item.process.preempt_requested:
                        item.process.preempt_path.touch()
                        item.process.preempt_requested = True
                        item.process.log_handle.write(
                            "queue: foreign GPU process; checkpoint requested\n"
                        )
        finished: List[GuardedRunning] = []
        for item in running:
            code = item.process.process.poll()
            if code is None:
                if (
                    external_owned_ram_gib(occupancy) + owned_ram_gib(running)
                    > args.max_project_ram_gib
                ):
                    item.process.preempt_path.touch()
                    item.process.preempt_requested = True
                    item.process.log_handle.write("queue: project RAM cap exceeded; checkpoint requested\n")
                elif base.foreign_heavy_process(
                    item.process.gpu, devices, processes, owned[item.process.gpu],
                    args.foreign_process_mib,
                ) and not item.process.preempt_requested:
                    item.process.preempt_path.touch()
                    item.process.preempt_requested = True
                    item.process.log_handle.write("queue: foreign GPU process; checkpoint requested\n")
                continue
            item.process.log_handle.write(f"queue: process exited with code {code}\n")
            item.process.log_handle.close()
            item.process.preempt_path.unlink(missing_ok=True)
            oom = base.tail_contains_oom(
                item.process.log_path, start_offset=item.process.log_start
            )
            action = retry_action(code, item.job, oom=oom)
            entry = state["jobs"][item.job.name]
            if action == "complete":
                complete_job(state, item.job)
            elif action == "preempt":
                item.job.base_job.preemptions += 1
                entry.update(status="preempted", preemptions=item.job.base_job.preemptions)
                entry.pop("worker", None)
                pending.insert(0, item.job)
            elif action == "retry_oom":
                item.job.base_job.oom_retries = 1
                entry.update(
                    status="retrying",
                    oom_retries=1,
                    oom_batch_transition=_oom_transition_evidence(),
                )
                entry.pop("worker", None)
                pending.insert(0, item.job)
            elif action == "retry_failure":
                item.job.ordinary_retries = 1
                entry.update(status="retrying", ordinary_retries=1)
                entry.pop("worker", None)
                pending.insert(0, item.job)
            else:
                entry.update(status="failed", exit_code=code)
                entry.pop("worker", None)
                failed.append(item.job.name)
            atomic_json(args.state, state)
            finished.append(item)
        running = [item for item in running if item not in finished]
        if stop_requested:
            for item in running:
                item.process.preempt_path.touch()
                item.process.preempt_requested = True
            if not running:
                break
            time.sleep(args.poll_seconds)
            continue
        for gpu in args.gpus:
            slots = args.max_per_gpu - sum(
                item.process.gpu == gpu for item in running
            ) - sum(worker.gpu == gpu for worker in occupancy)
            for _ in range(max(slots, 0)):
                if stop_requested:
                    break
                external_reserved = sum(
                    worker.host_ram_gib for worker in occupancy
                )
                external_owned = external_owned_ram_gib(occupancy)
                external_cpu = sum(worker.cpu_workers for worker in occupancy)
                candidate = next(
                    (
                        job for job in pending
                        if job.name not in {worker.name for worker in external}
                        if can_reserve(
                            job, running,
                            max_project_ram_gib=args.max_project_ram_gib,
                            min_available_ram_gib=args.min_available_ram_gib,
                            external_reserved_gib=external_reserved,
                            external_owned_gib=external_owned,
                            max_cpu_cores=args.max_cpu_cores,
                            external_cpu_workers=external_cpu,
                        )
                    ),
                    None,
                )
                if candidate is None:
                    break
                devices, processes = base.gpu_snapshot()
                owned = owned_pids_including_external(
                    running, occupancy, args.gpus
                )
                if gpu not in devices or devices[gpu][1] < args.min_free_mib:
                    break
                if base.foreign_heavy_process(
                    gpu, devices, processes, owned[gpu], args.foreign_process_mib
                ):
                    break
                if stop_requested:
                    break
                base.disk_guard(args.results, args.min_disk_free_fraction)
                # Re-authenticate clean code, generated configs, amendment and
                # cache receipt immediately before each launch.  A plan-time
                # hash alone would not detect later local mutation.
                validate_adopted_queue_runtime(args.jobs)
                if stop_requested:
                    break
                launched = launch_candidate_transaction(
                    candidate=candidate,
                    gpu=gpu,
                    pending=pending,
                    state=state,
                    state_path=args.state,
                    log_dir=log_dir,
                    memory_fraction=args.memory_fraction,
                    stop_requested=lambda: stop_requested,
                )
                if launched is None:
                    break
                entry = state["jobs"][candidate.name]
                worker_identity = authenticate_or_terminate(
                    launched, candidate.base_job.argv, gpu
                )
                worker_identity["queue_batch_environment"] = (
                    base.queue_batch_environment(candidate.base_job)
                )
                entry["worker"] = worker_identity
                atomic_json(args.state, state)
                running.append(GuardedRunning(candidate, launched))
                if stop_requested:
                    launched.preempt_path.touch()
                    launched.preempt_requested = True
                    break
        if pending or running or external:
            time.sleep(args.poll_seconds)
    if failed:
        print("failed jobs: " + ", ".join(failed), file=sys.stderr)
        return 1
    if args.external_workers_proof is not None and not stop_requested:
        assert args.external_workers_plan is not None and external_plan is not None
        proof = {
            "schema_version": 1,
            "kind": "guarded_queue_external_worker_completion",
            "external_workers_plan": str(args.external_workers_plan),
            "external_workers_plan_sha256": sha256_file(args.external_workers_plan),
            "supplemental_queue": str(args.jobs),
            "supplemental_queue_sha256": sha256_file(args.jobs),
            "supplemental_ledger": str(args.state),
            "supplemental_ledger_sha256": sha256_file(args.state),
            "workers": external_completed,
        }
        if args.external_workers_proof.exists():
            observed = json.loads(
                args.external_workers_proof.read_text(encoding="utf-8")
            )
            comparable = {
                key: value for key, value in observed.items()
                if key != "completed_at_utc"
            }
            if comparable != proof or not isinstance(
                observed.get("completed_at_utc"), str
            ):
                raise FileExistsError("external-worker proof already differs")
        else:
            proof["completed_at_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            atomic_json(args.external_workers_proof, proof)
    return base.PREEMPT_EXIT if stop_requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
