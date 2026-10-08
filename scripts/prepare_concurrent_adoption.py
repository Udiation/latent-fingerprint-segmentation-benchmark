#!/usr/bin/env python3
"""Freeze a fail-closed supplemental queue around authenticated orphan workers.

This control-plane tool never launches, signals, or writes to a live worker.  It
must be run only after the superseded queue controller and reconciler exit.  It
keeps their two captured workers in place, creates a disjoint queue for the
formerly pending jobs, and records enough identity and resource evidence for
``run_queue_guarded.py --external-workers-plan`` to account for both universes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import canonical_sha256, sha256_file
from segbench.deep_utils import build_queue_batch_contract, config_digest
import reconcile_guarded_orphans as orphan


GIB = 1024**3
CONTROLLER_NAMES = {
    "run_queue_guarded.py",
    "reconcile_guarded_orphans.py",
    "run_stage3_supervisor.py",
}
PROTOCOL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
PROTOCOL_DATALOADER = {
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
}
CACHE_KIND = "uint8_500ppi_preprocessing_cache"
THRESHOLD_AMENDMENT_KIND = "stage4_threshold_policy_amendment"
EXPECTED_ABSENT_QUALITY_MASKS = 66
EXPECTED_SOURCE_PHASE = "learning_rate_sweep_remaining_native"
OOM_BATCH_POLICY = {
    "schema_version": 1,
    "kind": "single_oom_half_batch_double_accumulation",
    "max_oom_retries": 1,
    "only_transition": "base_to_half_batch_double_accumulation",
    "effective_batch_preserved": True,
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


def _write_once_text(path: Path, text: str) -> None:
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


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_once_text(path, json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


def _proc_argv(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [
        part.decode("utf-8", errors="surrogateescape")
        for part in raw.split(b"\0")
        if part
    ]


def conflicting_controllers(
    queue: Path, ledger: Path, superseded_config: Optional[Path]
) -> Dict[int, list[str]]:
    """Find live control processes that could launch or mutate this queue."""
    needles = {str(queue), str(ledger)}
    if superseded_config is not None:
        needles.add(str(superseded_config))
    conflicts: Dict[int, list[str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = _proc_argv(int(entry.name))
        except OSError:
            continue
        if not argv or not any(Path(value).name in CONTROLLER_NAMES for value in argv):
            continue
        if needles.intersection(argv):
            conflicts[int(entry.name)] = argv
    return conflicts


def _environment(pid: int) -> Dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    values: Dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        values[key.decode(errors="surrogateescape")] = value.decode(
            errors="surrogateescape"
        )
    return values


def _status_kib(pid: int, key: str) -> int:
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    raise ValueError(f"PID {pid} status lacks {key}")


def _pss_kib(pid: int) -> int:
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1])
    raise ValueError(f"PID {pid} smaps_rollup lacks Pss")


def _job_identity(item: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "name": item["name"],
            "argv": item["argv"],
            "env": item.get("env", {}),
            "completion_artifact": item["completion_artifact"],
            "resources": item["resources"],
        }
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _option(argv: Sequence[str], name: str) -> str:
    if list(argv).count(name) != 1:
        raise ValueError(f"job argv must contain exactly one {name}")
    index = list(argv).index(name)
    if index + 1 >= len(argv):
        raise ValueError(f"job argv {name} lacks a value")
    return str(argv[index + 1])


def _replace_option(argv: Sequence[str], name: str, value: str) -> list[str]:
    output = list(argv)
    if output.count(name) != 1:
        raise ValueError(f"job argv must contain exactly one {name}")
    index = output.index(name)
    if index + 1 >= len(output):
        raise ValueError(f"job argv {name} lacks a value")
    output[index + 1] = value
    return output


def _git_identity(root: Path, expected_commit: str) -> Dict[str, Any]:
    """Authenticate one immutable clean worktree used by every new job."""
    root = root.resolve()
    if len(expected_commit) != 40 or any(ch not in "0123456789abcdef" for ch in expected_commit):
        raise ValueError("expected code commit must be a lowercase 40-hex SHA")
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if observed != expected_commit:
        raise ValueError("new-job code worktree is not at the expected commit")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    )
    if status.strip():
        raise ValueError("new-job code worktree is dirty")
    scripts = root / "segbench" / "scripts"
    library = root / "segbench" / "segbench"
    required = [
        scripts / "train_seg.py",
        scripts / "train_patch.py",
        scripts / "train_sufs.py",
        scripts / "infer_masks.py",
        scripts / "select_threshold_from_probabilities.py",
        scripts / "prepare_deep_jobs.py",
        library / "deep_utils.py",
        library / "preprocess_cache.py",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required cache-enabled code is absent: {path}")
        relative = path.relative_to(root)
        subprocess.check_call(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return {
        "root": str(root),
        "git_commit": observed,
        "git_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip(),
        "files_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in required
        },
    }


def _validate_cache_receipt(
    cache_dir: Path, manifest: Path
) -> tuple[Dict[str, Any], Path]:
    cache_dir = cache_dir.resolve()
    receipt_path = cache_dir / "receipt.json"
    receipt = _read_json(receipt_path, "500-ppi cache receipt")
    expected = {
        "schema_version": 1,
        "kind": CACHE_KIND,
        "manifest_sha256": sha256_file(manifest),
        "target_ppi": 500.0,
        "image_format": "uint8_npy",
        "mask_format": "uint8_0_255_npy",
        "debug_limited": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"cache receipt mismatch for {key}")
    required_mask_columns = ["mask_path", "quality_mask_path"]
    if sorted(receipt.get("mask_columns", [])) != sorted(required_mask_columns):
        raise ValueError("cache receipt must contain ROI and quality mask columns")
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != receipt.get("record_count"):
        raise ValueError("cache receipt record count is invalid")
    image_ids = [
        record.get("image_id") for record in records if isinstance(record, dict)
    ]
    if len(image_ids) != len(records) or len(set(image_ids)) != len(records):
        raise ValueError("cache receipt record ids are invalid or duplicated")
    with manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_ids = [str(row.get("image_id", "")) for row in manifest_rows]
    if (
        not manifest_ids
        or any(not value for value in manifest_ids)
        or len(set(manifest_ids)) != len(manifest_ids)
        or set(manifest_ids) != set(image_ids)
    ):
        raise ValueError("cache receipt does not cover the complete source manifest")
    manifest_by_id = {str(row["image_id"]): row for row in manifest_rows}
    for record in records:
        image_id = str(record["image_id"])
        if Path(image_id).name != image_id or image_id in {".", ".."}:
            raise ValueError("unsafe cache record id")
        shape = record.get("shape")
        if (
            not _is_sha256(record.get("image_sha256"))
            or not _is_sha256(record.get("source_image_sha256"))
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise ValueError("cache record image identity or shape is invalid")
        for key in ("mask_sha256", "source_mask_sha256"):
            hashes = record.get(key)
            if not isinstance(hashes, dict) or set(hashes) != set(required_mask_columns):
                raise ValueError("cache record does not explicitly cover both mask columns")
            if any(value is not None and not _is_sha256(value) for value in hashes.values()):
                raise ValueError("cache record contains an invalid mask digest")
        if any(
            (record["mask_sha256"][column] is None)
            != (record["source_mask_sha256"][column] is None)
            for column in required_mask_columns
        ):
            raise ValueError("source/cache mask absence disagrees")
        row = manifest_by_id[image_id]
        for column in required_mask_columns:
            source_present = bool(str(row.get(column, "")).strip())
            if source_present != (record["source_mask_sha256"][column] is not None):
                raise ValueError("manifest/cache mask presence disagrees")
        cached_image = cache_dir / "images" / f"{image_id}.npy"
        if not cached_image.is_file():
            raise FileNotFoundError(f"cached image is absent: {image_id}")
        for column in required_mask_columns:
            cached_mask = cache_dir / "masks" / column / f"{image_id}.npy"
            if (record["mask_sha256"][column] is not None) != cached_mask.is_file():
                raise ValueError("cache receipt/artifact mask presence disagrees")
    if sum(
        record["mask_sha256"]["quality_mask_path"] is None
        for record in records
    ) != EXPECTED_ABSENT_QUALITY_MASKS:
        raise ValueError("cache receipt does not represent exactly 66 absent quality masks")
    verification = receipt.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("requested_count") != 20
        or verification.get("verified_count") != 20
        or verification.get("array_identity") is not True
        or not isinstance(verification.get("image_ids"), list)
        or len(verification["image_ids"]) != 20
    ):
        raise ValueError("cache receipt lacks the required 20-image identity gate")
    by_id = {str(record["image_id"]): record for record in records}
    for image_id in verification["image_ids"]:
        if Path(str(image_id)).name != str(image_id) or str(image_id) in {".", ".."}:
            raise ValueError("unsafe cache verification id")
        record = by_id.get(str(image_id))
        if not isinstance(record, dict):
            raise ValueError("cache verification id is absent from receipt records")
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
        mask_hashes = record.get("mask_sha256")
        source_mask_hashes = record.get("source_mask_sha256")
        if (
            not isinstance(mask_hashes, dict)
            or set(mask_hashes) != set(required_mask_columns)
            or not isinstance(source_mask_hashes, dict)
            or set(source_mask_hashes) != set(required_mask_columns)
        ):
            raise ValueError("cache record does not explicitly cover both mask columns")
        for column in required_mask_columns:
            expected_mask = mask_hashes[column]
            expected_source = source_mask_hashes[column]
            if (expected_mask is None) != (expected_source is None):
                raise ValueError("source/cache mask absence disagrees")
            if expected_mask is None:
                continue
            mask = cache_dir / "masks" / column / f"{image_id}.npy"
            if not mask.is_file() or sha256_file(mask) != expected_mask:
                raise ValueError(f"verified cached mask changed: {image_id}/{column}")
            source_mask = _manifest_artifact(str(row.get(column, "")), manifest)
            if (
                not source_mask.is_file()
                or sha256_file(source_mask) != expected_source
            ):
                raise ValueError(f"verified source mask changed: {image_id}/{column}")
    return receipt, receipt_path


def _validate_threshold_amendment(path: Path) -> Dict[str, Any]:
    amendment = _read_json(path, "threshold policy amendment")
    replacement = amendment.get("replacement_policy")
    attestations = amendment.get("required_attestations")
    if (
        amendment.get("schema_version") != 1
        or amendment.get("kind") != THRESHOLD_AMENDMENT_KIND
        or not isinstance(replacement, dict)
        or replacement.get("allowed_trainable_grid") != PROTOCOL_THRESHOLDS
        or replacement.get("selection_data") != "validation_only"
        or not isinstance(attestations, dict)
        or attestations.get("validation_only") is not True
        or attestations.get("test_or_heldout_opened_for_selection") is not False
    ):
        raise ValueError("threshold policy amendment does not authorize the new grid")
    return amendment


def _write_generated_config(
    source: Path, destination: Path, *, memory_fraction: float
) -> tuple[Dict[str, Any], str]:
    """Write a config whose only policy changes are workers and threshold grid."""
    original = _read_yaml(source, f"source config {source.name}")
    amended = json.loads(json.dumps(original))
    training = amended.get("training")
    data = amended.get("data")
    runtime = amended.get("runtime")
    if not all(isinstance(value, dict) for value in (training, data, runtime)):
        raise ValueError(f"{source}: config sections are invalid")
    if float(data.get("target_ppi", math.nan)) != 500.0:
        raise ValueError(f"{source}: target ppi is not 500")
    configured_fraction = float(runtime.get("memory_fraction", math.nan))
    if not math.isclose(configured_fraction, memory_fraction, abs_tol=1e-12):
        raise ValueError(f"{source}: memory fraction differs from the runtime plan")
    for key, value in PROTOCOL_DATALOADER.items():
        training[key] = value
    training["threshold_candidates"] = list(PROTOCOL_THRESHOLDS)
    batch_contract = build_queue_batch_contract(training)
    runtime["queue_batch_contract"] = batch_contract
    runtime["queue_batch_contract_sha256"] = config_digest(batch_contract)
    text = yaml.safe_dump(amended, sort_keys=False)
    _write_once_text(destination, text)
    observed = _read_yaml(destination, f"generated config {destination.name}")
    if observed != amended:
        raise RuntimeError("generated config did not round-trip exactly")
    return amended, sha256_file(destination)


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    restricted_paths = (
        args.source_queue,
        args.source_ledger,
        args.capture,
        args.supplemental_queue,
        args.plan_out,
        args.code_root,
        args.cache_dir,
        args.threshold_amendment,
        args.generated_config_dir,
    )
    if args.superseded_supervisor_config is not None:
        restricted_paths += (args.superseded_supervisor_config,)
    if any(not _inside(path, args.resource_root) for path in restricted_paths):
        raise ValueError("adoption input/output escaped the restricted project root")
    orphan.reconciliation_preflight(args.source_queue, args.source_ledger, args.capture)
    conflicts = conflicting_controllers(
        args.source_queue, args.source_ledger, args.superseded_supervisor_config
    )
    if conflicts:
        raise RuntimeError(
            "superseded controller/reconciler is still live: "
            + ", ".join(str(pid) for pid in sorted(conflicts))
        )
    queue = _read_yaml(args.source_queue, "source queue")
    ledger = _read_json(args.source_ledger, "source ledger")
    capture = _read_json(args.capture, "orphan capture")
    jobs = queue.get("jobs")
    entries = ledger.get("jobs")
    workers = capture.get("workers")
    if not isinstance(jobs, list) or not isinstance(entries, dict) or not isinstance(
        workers, dict
    ):
        raise ValueError("source queue/ledger/capture structure changed")
    if queue.get("phase") != EXPECTED_SOURCE_PHASE:
        raise ValueError("source queue is not the preserved learning-rate sweep")
    if set(entries) != {str(job.get("name")) for job in jobs}:
        raise ValueError("source queue and ledger universes differ")
    running = {
        name for name, entry in entries.items() if entry.get("status") == "running"
    }
    pending = {
        name for name, entry in entries.items() if entry.get("status") == "pending"
    }
    completed = {
        name for name, entry in entries.items() if entry.get("status") == "completed"
    }
    if (
        set(workers) != running | completed
        or running | completed | pending != set(entries)
        or completed - set(workers)
    ):
        raise ValueError(
            "source ledger must contain only captured running/completed and pending jobs"
        )

    allowed = [int(value) for value in args.allowed_gpus.split(",")]
    if len(allowed) != len(set(allowed)) or not allowed or any(value < 0 for value in allowed):
        raise ValueError("allowed GPUs must be unique non-negative indices")
    if not 1 <= args.max_per_gpu <= 3:
        raise ValueError("max-per-gpu must be within 1..3")
    if (
        not math.isfinite(args.memory_fraction)
        or not 0.0 < args.memory_fraction <= 0.3
        or args.max_per_gpu * args.memory_fraction > 0.9 + 1e-12
    ):
        raise ValueError("aggregate per-GPU memory-fraction cap is invalid")
    if not math.isfinite(args.reservation_gib) or args.reservation_gib <= 0.0:
        raise ValueError("reservation must be positive and finite")
    if (
        not math.isfinite(args.max_project_ram_gib)
        or args.max_project_ram_gib <= 0.0
        or args.max_cpu_cores <= 0
    ):
        raise ValueError("project resource caps must be positive runtime values")
    slots = len(allowed) * args.max_per_gpu
    if slots * args.reservation_gib > args.max_project_ram_gib:
        raise ValueError("aggregate slot reservations exceed project RAM cap")
    if slots * args.max_data_workers_per_job > args.max_cpu_cores:
        raise ValueError("aggregate DataLoader worker budget exceeds CPU cap")
    if args.max_data_workers_per_job != PROTOCOL_DATALOADER["num_workers"]:
        raise ValueError("new jobs require exactly eight DataLoader workers")
    usage = shutil.disk_usage(args.resource_root)
    if usage.free / usage.total < args.min_disk_free_fraction:
        raise RuntimeError("disk-free floor is not met")
    source_worker_counts: Dict[str, int] = {}
    for job in jobs:
        argv = job.get("argv")
        if not isinstance(argv, list):
            raise ValueError(f"{job.get('name')}: argv is invalid")
        config_path = Path(_option(argv, "--config"))
        config = _read_yaml(config_path, f"{job.get('name')} config")
        training = config.get("training")
        runtime = config.get("runtime")
        configured_fraction = (
            float(runtime.get("memory_fraction", math.nan))
            if isinstance(runtime, dict)
            else math.nan
        )
        if (
            not isinstance(training, dict)
            or int(training.get("num_workers", -1)) > args.max_data_workers_per_job
            or int(training.get("num_workers", -1)) < 0
            or not isinstance(runtime, dict)
            or not math.isfinite(configured_fraction)
            or not 0.0 < configured_fraction <= args.memory_fraction + 1e-12
        ):
            raise ValueError(f"{job.get('name')}: config exceeds concurrency budget")
        source_worker_counts[str(job.get("name"))] = int(training["num_workers"])

    manifests = {Path(_option(job["argv"], "--manifest")).resolve() for job in jobs}
    if len(manifests) != 1:
        raise ValueError("source jobs do not share one manifest")
    manifest = next(iter(manifests))
    code_identity = _git_identity(args.code_root, args.expected_code_commit)
    cache_receipt, cache_receipt_path = _validate_cache_receipt(
        args.cache_dir, manifest
    )
    _validate_threshold_amendment(args.threshold_amendment)
    if sha256_file(args.threshold_amendment) != args.expected_threshold_amendment_sha256:
        raise ValueError("threshold amendment digest differs from the operator pin")

    worker_records = []
    precompleted_records = []
    for name in sorted(running):
        record = workers[name]
        identity = orphan.process_identity(int(record["pid"]))
        source_job = next(job for job in jobs if job["name"] == name)
        completion = Path(str(record["completion_artifact"]))
        if not completion.is_absolute() or not _inside(completion, args.resource_root):
            raise ValueError(f"{name}: completion artifact escaped resource root")
        if identity is None:
            if not completion.is_file():
                raise RuntimeError(f"{name}: captured worker vanished without completion")
            precompleted_records.append(
                {
                    "name": name,
                    "pid": int(record["pid"]),
                    "start_ticks": int(record["start_ticks"]),
                    "argv_sha256": str(record["argv_sha256"]),
                    "completion_artifact": str(completion),
                    "artifact_sha256": sha256_file(completion),
                    "source_job_identity_sha256": source_job[
                        "job_identity_sha256"
                    ],
                }
            )
            continue
        expected = {
            "pid": int(record["pid"]),
            "start_ticks": int(record["start_ticks"]),
            "argv": list(source_job["argv"]),
            "argv_sha256": str(record["argv_sha256"]),
        }
        if identity != expected:
            raise RuntimeError(f"{name}: captured worker is no longer exactly live")
        environment = _environment(identity["pid"])
        try:
            gpu = int(environment["CUDA_VISIBLE_DEVICES"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{name}: physical GPU assignment is unavailable") from exc
        preempt = Path(environment.get("SEG_PREEMPT_FILE", ""))
        inherited_fraction = float(
            environment.get("SEG_GPU_MEMORY_FRACTION", math.nan)
        )
        if (
            gpu not in allowed
            or not preempt.is_absolute()
            or not _inside(preempt, args.resource_root)
            or not math.isfinite(inherited_fraction)
            or not 0.0 < inherited_fraction <= args.memory_fraction + 1e-12
            or any(
                environment.get(name) != "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            )
        ):
            raise ValueError(f"{name}: worker GPU/preemption contract changed")
        if not completion.is_absolute() or not _inside(completion, args.resource_root):
            raise ValueError(f"{name}: completion artifact escaped resource root")
        if completion.exists():
            raise FileExistsError(f"{name}: live worker already has completion artifact")
        worker_records.append(
            {
                "name": name,
                "pid": identity["pid"],
                "start_ticks": identity["start_ticks"],
                "argv_sha256": identity["argv_sha256"],
                "gpu": gpu,
                "completion_artifact": str(completion),
                "preempt_path": str(preempt),
                "host_ram_gib": args.reservation_gib,
                "cpu_workers": source_worker_counts[name],
                "source_job_identity_sha256": source_job["job_identity_sha256"],
                "observed_pss_gib": _pss_kib(identity["pid"]) * 1024 / GIB,
                "observed_vm_hwm_gib": _status_kib(identity["pid"], "VmHWM")
                * 1024
                / GIB,
            }
        )
    for name in sorted(completed):
        record = workers[name]
        source_job = next(job for job in jobs if job["name"] == name)
        completion = Path(str(record["completion_artifact"]))
        entry = entries[name]
        if (
            not completion.is_file()
            or entry.get("artifact_sha256") != sha256_file(completion)
        ):
            raise ValueError(f"{name}: reconciled completion artifact changed")
        precompleted_records.append(
            {
                "name": name,
                "pid": int(record["pid"]),
                "start_ticks": int(record["start_ticks"]),
                "argv_sha256": str(record["argv_sha256"]),
                "completion_artifact": str(completion),
                "artifact_sha256": sha256_file(completion),
                "source_job_identity_sha256": source_job[
                    "job_identity_sha256"
                ],
            }
        )
    per_gpu = {
        gpu: sum(record["gpu"] == gpu for record in worker_records) for gpu in allowed
    }
    if any(value > args.max_per_gpu for value in per_gpu.values()):
        raise ValueError("captured workers already exceed per-GPU cap")

    supplemental_jobs = []
    generated_config_hashes: Dict[str, str] = {}
    generated_batch_contract_hashes: Dict[str, str] = {}
    legacy_file_hashes: Dict[str, Dict[str, str]] = {}
    captured_names = set(workers)
    for source_job in jobs:
        name = str(source_job["name"])
        item = dict(source_job)
        resources = dict(item.get("resources", {}))
        declared = float(resources.get("host_ram_gib", 0.0))
        if not math.isfinite(declared) or declared <= 0.0:
            raise ValueError(f"{name}: invalid source RAM reservation")
        resources["host_ram_gib"] = args.reservation_gib
        resources["cpu_workers"] = (
            PROTOCOL_DATALOADER["num_workers"]
            if name in pending
            else source_worker_counts[name]
        )
        item["resources"] = resources
        item["source_job_identity_sha256"] = item["job_identity_sha256"]
        if name in pending:
            source_argv = list(source_job["argv"])
            entrypoint_name = Path(source_argv[1]).name if len(source_argv) > 1 else ""
            if entrypoint_name not in {"train_seg.py", "train_patch.py"}:
                raise ValueError(f"{name}: unsupported training entrypoint")
            entrypoint = (
                Path(code_identity["root"])
                / "segbench"
                / "scripts"
                / entrypoint_name
            )
            source_config = Path(_option(source_argv, "--config"))
            generated_config = args.generated_config_dir / f"{name}.yaml"
            generated, generated_config_hashes[name] = _write_generated_config(
                source_config,
                generated_config,
                memory_fraction=args.memory_fraction,
            )
            generated_batch_contract_hashes[name] = str(
                generated["runtime"]["queue_batch_contract_sha256"]
            )
            amended_argv = list(source_argv)
            amended_argv[1] = str(entrypoint)
            amended_argv = _replace_option(
                amended_argv, "--config", str(generated_config)
            )
            if "--cache-dir" in amended_argv or "--threshold-amendment" in amended_argv:
                raise ValueError(f"{name}: source job unexpectedly used amended inputs")
            amended_argv.extend(
                [
                    "--cache-dir",
                    str(args.cache_dir.resolve()),
                    "--threshold-amendment",
                    str(args.threshold_amendment.resolve()),
                ]
            )
            item["argv"] = amended_argv
            item["adoption_role"] = "cache_amended_pending_job"
        elif name in captured_names:
            # The two already-running workers keep their byte-exact argv/config.
            # If one cooperatively preempts, this same entry resumes it unchanged.
            item["adoption_role"] = "legacy_captured_job"
            legacy_entrypoint = Path(item["argv"][1])
            legacy_config = Path(_option(item["argv"], "--config"))
            legacy_file_hashes[name] = {
                "entrypoint": str(legacy_entrypoint),
                "entrypoint_sha256": sha256_file(legacy_entrypoint),
                "config": str(legacy_config),
                "config_sha256": sha256_file(legacy_config),
            }
        else:
            raise ValueError(f"{name}: source status is not adoptable")
        item["job_identity_sha256"] = _job_identity(item)
        supplemental_jobs.append(item)
    if len(supplemental_jobs) != len(jobs):
        raise ValueError("supplemental queue does not exactly cover source jobs")
    supplemental = {
        "schema_version": 1,
        "kind": "stage3_concurrency_supplemental_queue",
        "phase": queue.get("phase"),
        "launch_authorized": True,
        "source_queue": {
            "path": str(args.source_queue),
            "sha256": sha256_file(args.source_queue),
        },
        "source_ledger_sha256_at_adoption": sha256_file(args.source_ledger),
        "source_capture": {
            "path": str(args.capture),
            "sha256": sha256_file(args.capture),
        },
        "execution_binding": {
            "schema_version": 1,
            "kind": "stage3_cache_concurrency_execution_binding",
            "code": code_identity,
            "cache_dir": str(args.cache_dir.resolve()),
            "cache_receipt": str(cache_receipt_path),
            "cache_receipt_sha256": sha256_file(cache_receipt_path),
            "cache_manifest_sha256": cache_receipt["manifest_sha256"],
            "cache_manifest": str(manifest),
            "threshold_amendment": str(args.threshold_amendment.resolve()),
            "threshold_amendment_sha256": sha256_file(args.threshold_amendment),
            "generated_config_sha256": generated_config_hashes,
            "generated_batch_contract_sha256": generated_batch_contract_hashes,
            "legacy_job_files_sha256": legacy_file_hashes,
            "legacy_external_job_names": sorted(captured_names),
            "cache_amended_job_names": sorted(pending),
            "dataloader": dict(PROTOCOL_DATALOADER),
            "oom_batch_policy": dict(OOM_BATCH_POLICY),
        },
        "runtime_policy": {
            "allowed_gpus": allowed,
            "max_jobs_per_gpu": args.max_per_gpu,
            "memory_fraction": args.memory_fraction,
            "aggregate_gpu_memory_fraction": args.max_per_gpu
            * args.memory_fraction,
            "max_project_ram_gib": args.max_project_ram_gib,
            "host_ram_gib_per_job": args.reservation_gib,
            "max_cpu_cores": args.max_cpu_cores,
            "max_data_workers_per_job": args.max_data_workers_per_job,
            "aggregate_data_workers_at_full_concurrency": slots
            * PROTOCOL_DATALOADER["num_workers"],
            "cpu_capacity_left_for_non_loader_work": args.max_cpu_cores
            - slots * PROTOCOL_DATALOADER["num_workers"],
            "min_disk_free_fraction": args.min_disk_free_fraction,
        },
        "jobs": supplemental_jobs,
    }
    queue_text = yaml.safe_dump(supplemental, sort_keys=False)
    _write_once_text(args.supplemental_queue, queue_text)
    plan = {
        "schema_version": 1,
        "kind": "guarded_queue_external_worker_plan",
        "supplemental_queue": str(args.supplemental_queue),
        "supplemental_queue_sha256": sha256_file(args.supplemental_queue),
        "source_queue": str(args.source_queue),
        "source_queue_sha256": sha256_file(args.source_queue),
        "source_ledger": str(args.source_ledger),
        "source_ledger_sha256_at_adoption": sha256_file(args.source_ledger),
        "capture": str(args.capture),
        "capture_sha256": sha256_file(args.capture),
        "execution_binding_sha256": canonical_sha256(
            supplemental["execution_binding"]
        ),
        "code_git_commit": code_identity["git_commit"],
        "cache_receipt_sha256": sha256_file(cache_receipt_path),
        "threshold_amendment_sha256": sha256_file(args.threshold_amendment),
        "generated_config_sha256": generated_config_hashes,
        "generated_batch_contract_sha256": generated_batch_contract_hashes,
        "oom_batch_policy": dict(OOM_BATCH_POLICY),
        "allowed_gpus": allowed,
        "max_jobs_per_gpu": args.max_per_gpu,
        "memory_fraction": args.memory_fraction,
        "max_project_ram_gib": args.max_project_ram_gib,
        "reservation_override_gib": args.reservation_gib,
        "reservation_evidence": {
            "kind": "operator_authorized_measured_override",
            "source_declared_gib_per_job": sorted(
                {float(job["resources"]["host_ram_gib"]) for job in jobs}
            ),
            "max_observed_external_pss_gib": max(
                (record["observed_pss_gib"] for record in worker_records),
                default=None,
            ),
            "max_observed_external_vm_hwm_gib": max(
                (record["observed_vm_hwm_gib"] for record in worker_records),
                default=None,
            ),
            "aggregate_slot_reservation_gib": slots * args.reservation_gib,
        },
        "cpu_budget": {
            "max_cpu_cores": args.max_cpu_cores,
            "max_data_workers_per_job": args.max_data_workers_per_job,
            "aggregate_data_workers": slots * args.max_data_workers_per_job,
            "initial_adopted_data_workers": sum(
                record["cpu_workers"] for record in worker_records
            ),
            "headroom_for_controller_and_reporting": args.max_cpu_cores
            - slots * args.max_data_workers_per_job,
        },
        "workers": worker_records,
        "precompleted_workers": precompleted_records,
        "source_statuses": {
            name: str(entries[name]["status"]) for name in sorted(entries)
        },
        "captured_source_names": sorted(workers),
        "pending_source_names": sorted(pending),
        "ground_truth_opened": False,
        "result_metrics_opened": False,
        "live_processes_modified": False,
    }
    _write_once_json(args.plan_out, plan)
    return plan


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", required=True, type=Path)
    parser.add_argument("--source-ledger", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--superseded-supervisor-config", type=Path)
    parser.add_argument("--supplemental-queue", required=True, type=Path)
    parser.add_argument("--plan-out", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--threshold-amendment", required=True, type=Path)
    parser.add_argument("--expected-threshold-amendment-sha256", required=True)
    parser.add_argument("--generated-config-dir", required=True, type=Path)
    parser.add_argument("--allowed-gpus", required=True)
    parser.add_argument("--max-per-gpu", required=True, type=int)
    parser.add_argument("--memory-fraction", default=0.3, type=float)
    parser.add_argument("--reservation-gib", required=True, type=float)
    parser.add_argument("--max-project-ram-gib", required=True, type=float)
    parser.add_argument("--max-cpu-cores", required=True, type=int)
    parser.add_argument("--max-data-workers-per-job", required=True, type=int)
    parser.add_argument("--resource-root", required=True, type=Path)
    parser.add_argument("--min-disk-free-fraction", default=0.15, type=float)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plan = prepare(args)
    print(json.dumps({"status": "prepared", "plan_sha256": canonical_sha256(plan)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
