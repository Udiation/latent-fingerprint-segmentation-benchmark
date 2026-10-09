#!/usr/bin/env python3
"""Wait for an authenticated validation handoff, then launch full seed-0 jobs.

``prepare`` is control-plane-only: it validates and pins the exact upstream
handoff plan and preparation receipt, writes an immutable controller plan, and
launches nothing. ``run`` may be started before the upstream handoff finishes.
It polls for the write-once completion receipt, authenticates the selected
full-seed-0 queue and its preparation proof, and launches that queue only via
``run_queue_guarded.py`` with the frozen v3 resource policy.

This controller never opens a manifest, split, image, mask, probability map,
checkpoint, or held-out result. It accepts training entry points only, and all
new outputs must live below ``full_v3_expanded_grid/<method>/seed0``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once, canonical_sha256, sha256_file
import autonomous_validation_handoff as upstream
import run_queue_guarded as guarded


SPEC_KIND = "autonomous_seed0_handoff_spec"
PLAN_KIND = "autonomous_seed0_handoff_plan"
AUTH_KIND = "selected_full_seed0_launch_authorization"
METHODS = {
    "unet", "unetpp", "drunet", "ra_runet", "lenet", "alexnet",
    "cnn_ae", "segformer_sub", "sufs_ft",
}
JOB_NAME = re.compile(r"^full\.([a-z0-9_]+)\.seed0$")
HEX40 = set("0123456789abcdef")
RUNTIME_KEYS = {
    "allowed_gpus",
    "max_jobs_per_gpu",
    "memory_fraction",
    "max_cpu_cores",
    "max_project_ram_gib",
    "min_disk_free_fraction",
    "min_free_mib",
    "foreign_process_mib",
    "min_available_ram_gib",
    "host_ram_gib_per_job",
    "cpu_workers_per_job",
    "reservation_evidence",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the metadata-only prepare or authenticated run interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--spec", required=True, type=Path)
    prepare_parser.add_argument("--code-root", required=True, type=Path)
    prepare_parser.add_argument("--out", required=True, type=Path)
    prepare_parser.add_argument("--proof-out", required=True, type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", required=True, type=Path)
    run_parser.add_argument("--plan-sha256", required=True)
    return parser.parse_args(argv)


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    """Read one control-plane JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _stable_json(path: Path, description: str) -> tuple[Dict[str, Any], str]:
    """Read a stable write-once JSON snapshot and return its byte digest."""
    for _ in range(10):
        first = path.read_bytes()
        value = json.loads(first)
        if first == path.read_bytes() and isinstance(value, dict):
            return value, sha256_file(path)
        time.sleep(0.05)
    raise RuntimeError(f"{description} did not yield a stable snapshot")


def _inside(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves inside ``root`` without requiring it."""
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _validated_runtime_policy(value: Any) -> Dict[str, Any]:
    """Validate a fully explicit private runtime policy, with no defaults."""
    if not isinstance(value, dict) or set(value) != RUNTIME_KEYS:
        raise ValueError("runtime policy must provide the exact required key set")
    gpus = value.get("allowed_gpus")
    integer_keys = {
        "max_jobs_per_gpu", "max_cpu_cores", "min_free_mib",
        "foreign_process_mib", "cpu_workers_per_job",
    }
    numeric_keys = {
        "memory_fraction", "max_project_ram_gib", "min_disk_free_fraction",
        "min_available_ram_gib", "host_ram_gib_per_job",
    }
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(type(item) is not int or item < 0 for item in gpus)
        or len(gpus) != len(set(gpus))
        or any(type(value.get(key)) is not int or value[key] <= 0 for key in integer_keys)
        or any(
            not isinstance(value.get(key), (int, float))
            or not math.isfinite(float(value[key]))
            or float(value[key]) <= 0.0
            for key in numeric_keys
        )
        or not 0.0 < float(value["memory_fraction"]) <= 1.0
        or not 0.0 < float(value["min_disk_free_fraction"]) < 1.0
    ):
        raise ValueError("runtime policy contains invalid resource values")
    slots = len(gpus) * int(value["max_jobs_per_gpu"])
    aggregate_ram = slots * float(value["host_ram_gib_per_job"])
    aggregate_workers = slots * int(value["cpu_workers_per_job"])
    aggregate_fraction = int(value["max_jobs_per_gpu"]) * float(
        value["memory_fraction"]
    )
    if aggregate_ram > float(value["max_project_ram_gib"]):
        raise ValueError("runtime reservations cannot fill every registered GPU slot")
    if aggregate_workers > int(value["max_cpu_cores"]):
        raise ValueError("DataLoader reservations cannot fill every registered GPU slot")
    if aggregate_fraction > 1.0:
        raise ValueError("per-GPU process-memory reservations exceed one device")
    evidence = value.get("reservation_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
        raise ValueError("runtime policy reservation evidence is absent")
    evidence_path = Path(str(evidence.get("path", "")))
    evidence_sha256 = str(evidence.get("sha256", ""))
    if (
        not evidence_path.is_absolute()
        or not evidence_path.is_file()
        or len(evidence_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in evidence_sha256
        )
        or sha256_file(evidence_path) != evidence_sha256
    ):
        raise ValueError("runtime policy reservation evidence identity changed")
    return {
        "allowed_gpus": list(gpus),
        "max_jobs_per_gpu": int(value["max_jobs_per_gpu"]),
        "memory_fraction": float(value["memory_fraction"]),
        "max_cpu_cores": int(value["max_cpu_cores"]),
        "max_project_ram_gib": float(value["max_project_ram_gib"]),
        "min_disk_free_fraction": float(value["min_disk_free_fraction"]),
        "min_free_mib": int(value["min_free_mib"]),
        "foreign_process_mib": int(value["foreign_process_mib"]),
        "min_available_ram_gib": float(value["min_available_ram_gib"]),
        "host_ram_gib_per_job": float(value["host_ram_gib_per_job"]),
        "cpu_workers_per_job": int(value["cpu_workers_per_job"]),
        "reservation_evidence": {
            "path": str(evidence_path),
            "sha256": evidence_sha256,
        },
    }


def _resource_reservation(policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the queue-visible reservation bound by the runtime policy."""
    return {
        "host_ram_gib_per_job": float(policy["host_ram_gib_per_job"]),
        "cpu_workers_per_job": int(policy["cpu_workers_per_job"]),
        "reservation_evidence": dict(policy["reservation_evidence"]),
    }


def _single_option(argv: Sequence[str], option: str) -> str:
    if list(argv).count(option) != 1:
        raise ValueError(f"training argv must contain exactly one {option}")
    index = list(argv).index(option)
    if index + 1 >= len(argv):
        raise ValueError(f"training argv {option} lacks a value")
    return str(argv[index + 1])


def _code_identity(root: Path) -> Dict[str, Any]:
    """Pin the controller and the guarded launcher to one clean Git tree."""
    root = root.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).strip()
    if len(commit) != 40 or any(character not in HEX40 for character in commit) or dirty:
        raise ValueError("seed-0 controller code must be a clean immutable commit")
    files: Dict[str, str] = {}
    for relative in (
        "scripts/autonomous_seed0_handoff.py",
        "scripts/run_queue_guarded.py",
        "scripts/run_queue.py",
        "segbench/artifact_identity.py",
    ):
        path = root / "segbench" / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = sha256_file(path)
    return {
        "root": str(root),
        "git_commit": commit,
        "git_tree": tree,
        "files_sha256": files,
    }


def _verify_code(value: Mapping[str, Any]) -> Path:
    root = Path(str(value.get("root", "")))
    if _code_identity(root) != value:
        raise ValueError("seed-0 controller code identity changed")
    return root


def _verify_upstream(
    plan: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Authenticate the immutable upstream plan, proof, and code identity."""
    source = plan.get("source_handoff")
    if not isinstance(source, dict):
        raise ValueError("source handoff binding is absent")
    plan_path = Path(str(source.get("plan", "")))
    proof_path = Path(str(source.get("preparation_proof", "")))
    if (
        not plan_path.is_absolute()
        or not proof_path.is_absolute()
        or not plan_path.is_file()
        or not proof_path.is_file()
        or sha256_file(plan_path) != source.get("plan_sha256")
        or sha256_file(proof_path) != source.get("preparation_proof_sha256")
    ):
        raise ValueError("source handoff plan/preparation identity changed")
    upstream_plan = _read_json(plan_path, "source handoff plan")
    if (
        upstream_plan.get("kind") != upstream.PLAN_KIND
        or upstream_plan.get("test_or_heldout_opened") is not False
        or upstream_plan.get("gpu_launched") is not False
    ):
        raise ValueError("source handoff plan contract changed")
    if upstream_plan.get("code") != source.get("code"):
        raise ValueError("source handoff code binding changed")
    upstream._verify_code(upstream_plan["code"])
    runtime_policy = _validated_runtime_policy(plan.get("runtime_policy"))
    upstream_resources = upstream._validated_selected_full_resources(
        upstream_plan.get("selected_full_resources")
    )
    if upstream_resources != _resource_reservation(runtime_policy):
        raise ValueError("source selected full-job resource binding changed")
    proof = _read_json(proof_path, "source handoff preparation proof")
    expected = {
        "kind": "autonomous_validation_handoff_preparation",
        "status": "READY_NOT_LAUNCHED",
        "plan": str(plan_path),
        "plan_sha256": source["plan_sha256"],
        "code_commit": upstream_plan["code"]["git_commit"],
        "source_controller_authenticated": True,
        "test_or_heldout_opened": False,
        "gpu_launched": False,
    }
    disagreements = {
        key: (proof.get(key), value)
        for key, value in expected.items()
        if proof.get(key) != value
    }
    if disagreements:
        raise ValueError(f"source handoff preparation proof changed: {disagreements}")
    expected_completion = Path(str(source.get("completion", "")))
    if expected_completion != Path(str(upstream_plan.get("handoff_root", ""))) / "completion.json":
        raise ValueError("source handoff completion path changed")
    return upstream_plan, proof


def prepare(
    spec_path: Path, code_root: Path, out: Path, proof_out: Path
) -> Dict[str, Any]:
    """Write an immutable, no-launch controller plan."""
    spec = _read_json(spec_path, "seed-0 handoff specification")
    if spec.get("schema_version") != 1 or spec.get("kind") != SPEC_KIND:
        raise ValueError("seed-0 handoff specification kind changed")
    source_plan_path = Path(str(spec.get("source_handoff_plan", "")))
    source_proof_path = Path(str(spec.get("source_handoff_preparation_proof", "")))
    if (
        not source_plan_path.is_absolute()
        or not source_proof_path.is_absolute()
        or not source_plan_path.is_file()
        or not source_proof_path.is_file()
        or sha256_file(source_plan_path) != spec.get("source_handoff_plan_sha256")
        or sha256_file(source_proof_path)
        != spec.get("source_handoff_preparation_proof_sha256")
    ):
        raise ValueError("specified source handoff identities differ")
    source_plan = _read_json(source_plan_path, "source handoff plan")
    restricted_root = Path(str(source_plan.get("restricted_root", "")))
    controller_root = Path(str(spec.get("controller_root", "")))
    if (
        not restricted_root.is_absolute()
        or not controller_root.is_absolute()
        or not _inside(controller_root, restricted_root)
    ):
        raise ValueError("controller root must be inside the restricted result root")
    source = {
        "plan": str(source_plan_path),
        "plan_sha256": sha256_file(source_plan_path),
        "preparation_proof": str(source_proof_path),
        "preparation_proof_sha256": sha256_file(source_proof_path),
        "completion": str(Path(str(source_plan.get("handoff_root", ""))) / "completion.json"),
        "code": source_plan.get("code"),
    }
    code = _code_identity(code_root)
    runtime_policy = _validated_runtime_policy(spec.get("runtime_policy"))
    upstream_resources = upstream._validated_selected_full_resources(
        source_plan.get("selected_full_resources")
    )
    if upstream_resources != _resource_reservation(runtime_policy):
        raise ValueError("seed-0 runtime policy differs from source selected resources")
    value = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "code": code,
        "specification": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "source_handoff": source,
        "restricted_root": str(restricted_root),
        "controller_root": str(controller_root),
        "state": str(controller_root / "selected_full_seed0_queue_state.json"),
        "log": str(controller_root / "selected_full_seed0_queue.log"),
        "launch_authorization": str(controller_root / "launch_authorization.json"),
        "completion": str(controller_root / "completion.json"),
        "source_wait_seconds": float(spec.get("source_wait_seconds", 172800.0)),
        "poll_seconds": float(spec.get("poll_seconds", 10.0)),
        "runtime_policy": runtime_policy,
        "test_or_heldout_opened": False,
        "gpu_launched": False,
    }
    if (
        not math.isfinite(value["source_wait_seconds"])
        or value["source_wait_seconds"] <= 0
        or not math.isfinite(value["poll_seconds"])
        or value["poll_seconds"] <= 0
    ):
        raise ValueError("polling intervals must be finite and positive")
    # Reuse the runtime verifier on the exact constructed binding before publish.
    _verify_upstream(value)
    atomic_json_once(out, value)
    proof = {
        "schema_version": 1,
        "kind": "autonomous_seed0_handoff_preparation",
        "status": "READY_NOT_LAUNCHED",
        "plan": str(out),
        "plan_sha256": sha256_file(out),
        "source_handoff_plan_sha256": source["plan_sha256"],
        "source_handoff_preparation_proof_sha256": source["preparation_proof_sha256"],
        "code_commit": code["git_commit"],
        "test_or_heldout_opened": False,
        "gpu_launched": False,
    }
    atomic_json_once(proof_out, proof)
    return proof


def _forbid_heldout_tokens(argv: Sequence[str]) -> None:
    for item in argv:
        lowered = str(item).lower()
        if (
            lowered in {"test", "heldout", "held-out"}
            or lowered.startswith("--test")
            or lowered.startswith("--heldout")
            or lowered.startswith("--held-out")
        ):
            raise ValueError("selected full training argv contains a held-out token")


def _validate_job(
    item: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    decision: Mapping[str, Any],
    runtime_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate one train/val-only, versioned full seed-0 job."""
    name = str(item.get("name", ""))
    match = JOB_NAME.fullmatch(name)
    if match is None or match.group(1) not in METHODS:
        raise ValueError(f"invalid selected full seed-0 job name: {name!r}")
    method = match.group(1)
    argv = item.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise ValueError(f"{name}: argv is invalid")
    _forbid_heldout_tokens(argv)
    if len(argv) < 2:
        raise ValueError(f"{name}: training entry point is absent")
    code_root = Path(str(source_plan["code"]["root"]))
    expected_entrypoint = {
        "sufs_ft": "train_sufs.py",
        "lenet": "train_patch.py",
        "alexnet": "train_patch.py",
        "cnn_ae": "train_patch.py",
    }.get(method, "train_seg.py")
    entrypoint = Path(argv[1])
    if entrypoint != code_root / "segbench" / "scripts" / expected_entrypoint:
        raise ValueError(f"{name}: training entry point escaped the pinned code tree")
    exact_options = {
        "--manifest": str(source_plan["manifest"]),
        "--splits": str(source_plan["splits"]),
        "--cache-dir": str(source_plan["cache_dir"]),
        "--threshold-amendment": str(source_plan["training_threshold_amendment"]),
        "--seed": "0",
        "--resume": "auto",
    }
    for option, expected in exact_options.items():
        if _single_option(argv, option) != expected:
            raise ValueError(f"{name}: {option} differs from the upstream binding")
    out = Path(_single_option(argv, "--out"))
    expected_out = (
        Path(str(source_plan["runs_root"]))
        / "full_v3_expanded_grid"
        / method
        / "seed0"
    )
    if out != expected_out or not _inside(out, Path(str(source_plan["runs_root"]))):
        raise ValueError(f"{name}: output is not the fresh versioned seed-0 path")
    completion = Path(str(item.get("completion_artifact", "")))
    if completion != out / "run.json":
        raise ValueError(f"{name}: completion artifact differs from its output")
    if decision.get("method") != method or decision.get("action") != "fresh_full_seed0":
        raise ValueError(f"{name}: preparation decision changed")
    if method == "sufs_ft":
        selection = Path(_single_option(argv, "--selection"))
        if (
            _single_option(argv, "--mode") != "full"
            or selection != Path(str(decision.get("dedicated_selection", "")))
            or not selection.is_file()
            or sha256_file(selection) != decision.get("dedicated_selection_sha256")
        ):
            raise ValueError("sufs_ft: dedicated validation selection changed")
    else:
        config = Path(_single_option(argv, "--config"))
        if (
            config != Path(str(decision.get("config", "")))
            or not config.is_file()
            or sha256_file(config) != decision.get("config_sha256")
        ):
            raise ValueError(f"{name}: generated full-training config changed")
    resources = item.get("resources")
    if (
        not isinstance(resources, dict)
        or float(resources.get("host_ram_gib", math.nan))
        != float(runtime_policy["host_ram_gib_per_job"])
        or resources.get("cpu_workers") != runtime_policy["cpu_workers_per_job"]
    ):
        raise ValueError(f"{name}: resource reservation changed")
    return {
        "name": name,
        "method": method,
        "identity_sha256": str(item.get("job_identity_sha256", "")),
        "output": str(out),
        "completion_artifact": str(completion),
    }


def _authenticate_selected(
    plan: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    completion: Mapping[str, Any],
    completion_sha256: str,
) -> Dict[str, Any]:
    """Authenticate the completion, selected queue, proof, configs, and jobs."""
    source = plan["source_handoff"]
    runtime_policy = _validated_runtime_policy(plan.get("runtime_policy"))
    expected_completion = {
        "kind": "autonomous_validation_handoff_completion",
        "status": "COMPLETE_NO_TEST_OPENED",
        "plan_sha256": source["plan_sha256"],
        "test_or_heldout_opened": False,
        "source_ledger_written": False,
        "source_signalled": False,
        "selected_full_seed0_launched": False,
    }
    disagreements = {
        key: (completion.get(key), value)
        for key, value in expected_completion.items()
        if completion.get(key) != value
    }
    if disagreements:
        raise ValueError(f"source handoff completion changed: {disagreements}")
    handoff_root = Path(str(source_plan["handoff_root"]))
    queue_path = Path(str(completion.get("selected_full_seed0_queue", "")))
    proof_path = handoff_root / "selected_full_seed0_preparation.json"
    if queue_path != handoff_root / "selected_full_seed0_queue.yaml":
        raise ValueError("selected full seed-0 queue path changed")
    if (
        not queue_path.is_file()
        or not proof_path.is_file()
        or sha256_file(queue_path) != completion.get("selected_full_seed0_queue_sha256")
    ):
        raise ValueError("selected full seed-0 queue/proof is absent or changed")
    proof, proof_sha256 = _stable_json(proof_path, "selected queue preparation proof")
    queue_sha256 = sha256_file(queue_path)
    queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    if not isinstance(queue, dict):
        raise ValueError("selected full seed-0 queue must be an object")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("selected full seed-0 job list is absent")
    expected_status = "READY_NOT_LAUNCHED" if jobs else "NO_NEW_JOBS"
    expected_proof = {
        "kind": "selected_full_seed0_queue_preparation",
        "status": expected_status,
        "test_or_heldout_opened": False,
        "queue": str(queue_path),
        "queue_sha256": queue_sha256,
        "jobs": [str(item.get("name", "")) for item in jobs],
        "resource_reservation": _resource_reservation(runtime_policy),
        "gpu_launched": False,
    }
    proof_disagreements = {
        key: (proof.get(key), value)
        for key, value in expected_proof.items()
        if proof.get(key) != value
    }
    if proof_disagreements:
        raise ValueError(f"selected queue preparation proof changed: {proof_disagreements}")
    if (
        queue.get("schema_version") != 1
        or queue.get("phase") != "selected_full_seed0_training"
        or queue.get("selection_split") != "val"
        or queue.get("test_inference_authorized") is not False
        or queue.get("launch_authorized") is not bool(jobs)
        or queue.get("resource_reservation")
        != expected_proof["resource_reservation"]
    ):
        raise ValueError("selected full seed-0 queue scientific barrier changed")
    summary_paths = [
        str(completion.get("initial_selection_summary", "")),
        str(completion.get("late_selection_summary", "")),
    ]
    declared_summaries = proof.get("selection_summaries")
    if not isinstance(declared_summaries, list) or [
        str(item.get("path", "")) for item in declared_summaries
    ] != summary_paths:
        raise ValueError("selected queue validation-summary universe changed")
    for record in declared_summaries:
        path = Path(str(record.get("path", "")))
        if (
            not path.is_absolute()
            or not _inside(path, handoff_root)
            or not path.is_file()
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError("selected queue validation-summary identity changed")
    decisions = proof.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("selected queue decisions are absent")
    by_method: Dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("selected queue decision is invalid")
        method = str(decision.get("method", ""))
        if method not in METHODS or method in by_method:
            raise ValueError("selected queue decision method changed or duplicated")
        if decision.get("action") not in {"fresh_full_seed0", "reuse_existing_full"}:
            raise ValueError("selected queue decision action changed")
        by_method[method] = decision
    queue_methods = []
    validated_jobs = []
    for item in jobs:
        if not isinstance(item, dict):
            raise ValueError("selected full seed-0 job is invalid")
        match = JOB_NAME.fullmatch(str(item.get("name", "")))
        method = match.group(1) if match else ""
        if method not in by_method:
            raise ValueError("selected full seed-0 job lacks its decision")
        validated_jobs.append(
            _validate_job(item, source_plan, by_method[method], runtime_policy)
        )
        queue_methods.append(method)
    if len(queue_methods) != len(set(queue_methods)):
        raise ValueError("selected full seed-0 method is duplicated")
    fresh = {
        method for method, decision in by_method.items()
        if decision.get("action") == "fresh_full_seed0"
    }
    if fresh != set(queue_methods):
        raise ValueError("selected queue jobs differ from fresh-training decisions")
    if jobs:
        loaded = guarded.load_guarded_jobs(queue_path)
        if [job.name for job in loaded] != [record["name"] for record in validated_jobs]:
            raise ValueError("guarded launcher parsed a different selected job universe")
    return {
        "schema_version": 1,
        "kind": AUTH_KIND,
        "status": "AUTHORIZED_NOT_LAUNCHED" if jobs else "NO_NEW_JOBS",
        "source_handoff_plan": source["plan"],
        "source_handoff_plan_sha256": source["plan_sha256"],
        "source_handoff_completion": source["completion"],
        "source_handoff_completion_sha256": completion_sha256,
        "selected_queue": str(queue_path),
        "selected_queue_sha256": queue_sha256,
        "selected_preparation_proof": str(proof_path),
        "selected_preparation_proof_sha256": proof_sha256,
        "controller_code": dict(plan["code"]),
        "source_code": dict(source_plan["code"]),
        "jobs": validated_jobs,
        "runtime_policy": runtime_policy,
        "test_or_heldout_opened": False,
        "gpu_launched": False,
    }


def _wait_for_completion(
    plan: Mapping[str, Any], sleep: Callable[[float], None] = time.sleep
) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    """Poll until the exact upstream completion appears or the deadline expires."""
    deadline = time.monotonic() + float(plan["source_wait_seconds"])
    completion_path = Path(str(plan["source_handoff"]["completion"]))
    while time.monotonic() < deadline:
        source_plan, _ = _verify_upstream(plan)
        if completion_path.is_file():
            completion, digest = _stable_json(completion_path, "source handoff completion")
            return source_plan, completion, digest
        sleep(float(plan["poll_seconds"]))
    raise TimeoutError("timed out waiting for the authenticated validation handoff")


def _guarded_argv(plan: Mapping[str, Any], authorization: Mapping[str, Any]) -> list[str]:
    """Return the one frozen guarded-launch command."""
    root = Path(str(plan["code"]["root"]))
    policy = plan["runtime_policy"]
    return [
        str(Path(sys.executable)),
        str(root / "segbench/scripts/run_queue_guarded.py"),
        "--jobs", str(authorization["selected_queue"]),
        "--results", str(plan["restricted_root"]),
        "--state", str(plan["state"]),
        "--gpus", ",".join(str(item) for item in policy["allowed_gpus"]),
        "--max-per-gpu", str(policy["max_jobs_per_gpu"]),
        "--min-free-mib", str(policy["min_free_mib"]),
        "--foreign-process-mib", str(policy["foreign_process_mib"]),
        "--memory-fraction", str(policy["memory_fraction"]),
        "--poll-seconds", str(plan["poll_seconds"]),
        "--min-disk-free-fraction", str(policy["min_disk_free_fraction"]),
        "--max-project-ram-gib", str(int(policy["max_project_ram_gib"])),
        "--max-cpu-cores", str(policy["max_cpu_cores"]),
        "--min-available-ram-gib", str(int(policy["min_available_ram_gib"])),
    ]


def _run_guarded(argv: Sequence[str], *, cwd: Path, log: Path) -> int:
    """Run the guarded queue synchronously; child jobs inherit nice >= 10."""
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("seed-0 handoff controller must run under nice -n 10")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("guarded-seed0 argv=" + json.dumps(list(argv)) + "\n")
        handle.flush()
        result = subprocess.run(
            list(argv), cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    return int(result.returncode)


def _verify_final_state(plan: Mapping[str, Any], authorization: Mapping[str, Any]) -> Dict[str, Any]:
    state_path = Path(str(plan["state"]))
    state, state_sha256 = _stable_json(state_path, "selected seed-0 queue ledger")
    if state.get("queue_sha256") != authorization["selected_queue_sha256"]:
        raise ValueError("selected seed-0 ledger/queue binding changed")
    records = state.get("jobs")
    expected = {record["name"]: record for record in authorization["jobs"]}
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError("selected seed-0 ledger job universe changed")
    completed = []
    for name, expected_job in expected.items():
        record = records[name]
        artifact = Path(expected_job["completion_artifact"])
        if (
            record.get("status") != "completed"
            or record.get("identity_sha256") != expected_job["identity_sha256"]
            or not artifact.is_file()
            or record.get("artifact_sha256") != sha256_file(artifact)
        ):
            raise ValueError(f"{name}: guarded completion identity changed")
        completed.append({
            "name": name,
            "artifact": str(artifact),
            "artifact_sha256": record["artifact_sha256"],
        })
    return {"path": str(state_path), "sha256": state_sha256, "jobs": completed}


def run(
    plan_path: Path,
    expected_sha256: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    launcher: Callable[..., int] = _run_guarded,
) -> Dict[str, Any]:
    """Wait, authenticate, and execute the selected full seed-0 queue."""
    if sha256_file(plan_path) != expected_sha256:
        raise ValueError("seed-0 handoff plan SHA-256 changed")
    plan = _read_json(plan_path, "seed-0 handoff plan")
    if (
        plan.get("kind") != PLAN_KIND
        or plan.get("test_or_heldout_opened") is not False
        or plan.get("gpu_launched") is not False
    ):
        raise ValueError("seed-0 handoff plan contract changed")
    if _validated_runtime_policy(plan.get("runtime_policy")) != plan["runtime_policy"]:
        raise ValueError("seed-0 handoff runtime policy changed")
    root = _verify_code(plan["code"])
    source_plan, source_completion, completion_sha256 = _wait_for_completion(
        plan, sleep=sleep
    )
    authorization = _authenticate_selected(
        plan, source_plan, source_completion, completion_sha256
    )
    authorization_path = Path(str(plan["launch_authorization"]))
    atomic_json_once(authorization_path, authorization)
    if authorization["status"] == "NO_NEW_JOBS":
        result = {
            "schema_version": 1,
            "kind": "autonomous_seed0_handoff_completion",
            "status": "COMPLETE_NO_NEW_JOBS",
            "plan_sha256": expected_sha256,
            "launch_authorization_sha256": sha256_file(authorization_path),
            "test_or_heldout_opened": False,
            "gpu_launched": False,
        }
        atomic_json_once(Path(str(plan["completion"])), result)
        return result
    state_path = Path(str(plan["state"]))
    if not state_path.exists():
        occupied = [
            record["output"] for record in authorization["jobs"]
            if Path(record["output"]).exists()
            and any(Path(record["output"]).iterdir())
        ]
        if occupied:
            raise FileExistsError(
                "unledgered versioned seed-0 output paths already contain data: "
                + ", ".join(occupied)
            )
    # Close the authorization-to-exec window as far as the filesystem permits.
    _verify_code(plan["code"])
    current_source, _ = _verify_upstream(plan)
    current_completion, current_completion_sha = _stable_json(
        Path(str(plan["source_handoff"]["completion"])),
        "source handoff completion",
    )
    current_authorization = _authenticate_selected(
        plan, current_source, current_completion, current_completion_sha
    )
    if canonical_sha256(current_authorization) != canonical_sha256(authorization):
        raise ValueError("selected full seed-0 authorization changed before launch")
    argv = _guarded_argv(plan, authorization)
    exit_code = launcher(argv, cwd=root, log=Path(str(plan["log"])))
    if exit_code != 0:
        raise RuntimeError(f"guarded selected seed-0 queue exited with {exit_code}")
    _verify_code(plan["code"])
    final_source, _ = _verify_upstream(plan)
    final_completion, final_completion_sha = _stable_json(
        Path(str(plan["source_handoff"]["completion"])),
        "source handoff completion",
    )
    final_authorization = _authenticate_selected(
        plan, final_source, final_completion, final_completion_sha
    )
    if canonical_sha256(final_authorization) != canonical_sha256(authorization):
        raise ValueError("selected full seed-0 authorization changed during execution")
    ledger = _verify_final_state(plan, authorization)
    result = {
        "schema_version": 1,
        "kind": "autonomous_seed0_handoff_completion",
        "status": "COMPLETE",
        "plan_sha256": expected_sha256,
        "launch_authorization": str(authorization_path),
        "launch_authorization_sha256": sha256_file(authorization_path),
        "selected_queue_sha256": authorization["selected_queue_sha256"],
        "selected_preparation_proof_sha256": authorization[
            "selected_preparation_proof_sha256"
        ],
        "guarded_ledger": ledger,
        "test_or_heldout_opened": False,
        "gpu_launched": True,
    }
    atomic_json_once(Path(str(plan["completion"])), result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.spec, args.code_root, args.out, args.proof_out)
    else:
        result = run(args.plan, args.plan_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
