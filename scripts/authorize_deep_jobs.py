#!/usr/bin/env python3
"""Create two immutable deep-training waves with a hard seed-0 barrier.

The first command emits only the seven not-yet-run native seed-0 jobs and a
non-launchable locked plan for sixteen seed-1/2 jobs.  The second command emits
the seed-1/2 launch queue only after all eight native seed-0 run cards pass the
same scientific/provenance checks.  SUFS fine-tuning is proved but excluded
because its three full runs are already complete.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import (
    atomic_json,
    canonical_sha256,
    require_clean_process_start,
    sha256_file,
)
import prepare_deep_jobs as planner


NATIVE = tuple(planner.METHOD_PRIORITY)
EXPECTED_SEED0 = tuple(f"full.{method}.seed0" for method in NATIVE if method != "unet")
EXPECTED_SEED12 = tuple(
    f"full.{method}.seed{seed}" for method in NATIVE for seed in (1, 2)
)


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON mapping")
    return value


def _read_yaml(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a YAML mapping")
    return value


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    planner.atomic_text(path, yaml.safe_dump(dict(value), sort_keys=False))


def _commit_map(path: Path) -> Dict[str, str]:
    value = _read_json(path, "expected commit map")
    if set(value) != set(NATIVE):
        raise ValueError("commit map must contain exactly all native methods")
    for method, commit in value.items():
        if not isinstance(commit, str) or len(commit) != 40 or any(
            ch not in "0123456789abcdef" for ch in commit
        ):
            raise ValueError(f"{method}: expected commit is not a 40-hex SHA")
    return {str(key): str(item) for key, item in value.items()}


def _reservations(
    path: Path, expected: Sequence[str], max_ram_gib: float
) -> Dict[str, float]:
    """Load declared, run-card-independent RAM reservations in GiB."""
    value = _read_json(path, "host-RAM reservation map")
    if set(value) != set(expected):
        raise ValueError("RAM map must cover the exact authorized job universe")
    result: Dict[str, float] = {}
    for name, amount in value.items():
        amount = float(amount)
        if not 0.0 < amount <= max_ram_gib:
            raise ValueError(f"{name}: RAM reservation exceeds the runtime cap")
        result[str(name)] = amount
    return result


def _job_with_guards(
    job: Mapping[str, Any], *, runs_root: Path, ram_gib: float
) -> Dict[str, Any]:
    # This script can be newer than the frozen scientific planner.  Its sole
    # authority is to add resource/completion guards; trainer argv (including
    # script, config, seed, output/resume or checkpoint arguments), name, and
    # environment must cross the pin boundary byte-for-byte unchanged.
    scientific_fields = {
        "name": copy.deepcopy(job.get("name")),
        "argv": copy.deepcopy(job.get("argv")),
        "env": copy.deepcopy(job.get("env", {})),
    }
    copied = copy.deepcopy(dict(job))
    match = planner.JOB_NAME_RE.fullmatch(str(copied.get("name")))
    if match is None:
        raise ValueError(f"invalid full job name: {copied.get('name')!r}")
    method_name, seed_text = match.groups()
    seed = int(seed_text)
    copied["completion_artifact"] = str(
        runs_root / "full" / method_name / f"seed{seed}" / "run.json"
    )
    copied["resources"] = {"host_ram_gib": ram_gib}
    observed_scientific_fields = {
        "name": copied.get("name"),
        "argv": copied.get("argv"),
        "env": copied.get("env", {}),
    }
    if observed_scientific_fields != scientific_fields:
        raise AssertionError("queue guard builder altered frozen scientific job fields")
    copied["job_identity_sha256"] = canonical_sha256(
        {
            "name": copied["name"],
            "argv": copied["argv"],
            "env": copied.get("env", {}),
            "completion_artifact": copied["completion_artifact"],
            "resources": copied["resources"],
        }
    )
    return copied


def _require_run_commit(run_path: Path, commit: str, description: str) -> None:
    run = _read_json(run_path, description)
    require_clean_process_start(run, commit, description=description)


def _verify_sha256(path: Path, expected: str, description: str) -> str:
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError(f"{description}: expected digest is not lowercase SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{description}: SHA-256 differs from registered digest")
    return observed


def _selection_provenance(
    global_selection: Mapping[str, Any], commits: Mapping[str, str], runs_root: Path
) -> None:
    methods = global_selection["methods"]
    for method in NATIVE:
        candidates = methods[method].get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise ValueError(f"{method}: selection must contain exactly two candidates")
        for candidate in candidates:
            _require_run_commit(
                Path(str(candidate["run_card"])),
                commits[method],
                f"{method} LR-selection candidate",
            )
def _validate_completed(
    *,
    global_selection: Mapping[str, Any],
    method: str,
    seed: int,
    runs_root: Path,
    commit: str | None,
    enforce_clean_process_start: bool = True,
    sufs_selection_sha256: str | None = None,
) -> Dict[str, Any]:
    method_selection = global_selection["methods"][method]
    if method == "sufs_ft":
        config_sha = str(method_selection["source_config_sha256"])
    elif method == "unet" and seed == 0:
        config_sha = str(method_selection["reused_seed0_config_sha256"])
    else:
        config_sha = str(method_selection["full_config_sha256"])
    proof = planner.validate_completed_full_run(
        runs_root / "full" / method / f"seed{seed}",
        method=method,
        seed=seed,
        manifest_sha256=str(global_selection["manifest_sha256"]),
        splits_sha256=str(global_selection["splits_sha256"]),
        source_config_sha256=config_sha,
        selected_learning_rate=float(method_selection["selected_learning_rate"]),
        selection_sha256=sufs_selection_sha256,
    )
    if enforce_clean_process_start:
        if commit is None:
            raise ValueError(f"full.{method}.seed{seed}: expected commit is required")
        _require_run_commit(Path(proof["run_card"]), commit, f"full.{method}.seed{seed}")
    return proof


def prepare_seed0(args: argparse.Namespace) -> None:
    commits = _commit_map(args.expected_commits)
    sufs_note_digest = _verify_sha256(
        args.sufs_provenance_note,
        args.sufs_provenance_note_sha256,
        "legacy SUFS provenance note",
    )
    global_selection, _, existing_sufs = planner.validate_staged_selections(
        args.global_selection, args.pilot_selection, args.sufs_selection
    )
    global_digest = sha256_file(args.global_selection)
    _selection_provenance(global_selection, commits, args.runs_root)
    _, pilot = planner.load_queue(
        args.pilot_seed0_jobs,
        description="pilot seed-0 queue",
        selection_sha256=global_digest,
        expected_names=["full.unet.seed0"],
    )
    _, remaining = planner.load_queue(
        args.remaining_seed0_jobs,
        description="remaining seed-0 queue",
        selection_sha256=global_digest,
        expected_names=[*EXPECTED_SEED0, "full.sufs_ft.seed0"],
    )
    _, seed12 = planner.load_queue(
        args.seed12_jobs,
        description="seed-1/2 queue",
        selection_sha256=global_digest,
        expected_names=[*EXPECTED_SEED12, "full.sufs_ft.seed1", "full.sufs_ft.seed2"],
    )
    for job in [*pilot, *remaining, *seed12]:
        planner.validate_queue_job(
            job, global_selection=global_selection, runs_root=args.runs_root
        )
    completed = {
        "full.unet.seed0": _validate_completed(
            global_selection=global_selection,
            method="unet",
            seed=0,
            runs_root=args.runs_root,
            commit=commits["unet"],
        )
    }
    sufs_digest = sha256_file(args.sufs_selection)
    for seed in planner.FULL_SEEDS:
        completed[f"full.sufs_ft.seed{seed}"] = _validate_completed(
            global_selection=global_selection,
            method="sufs_ft",
            seed=seed,
            runs_root=args.runs_root,
            commit=None,
            enforce_clean_process_start=False,
            sufs_selection_sha256=sufs_digest,
        )
    native0 = [job for job in remaining if job["name"] in EXPECTED_SEED0]
    native12 = [job for job in seed12 if job["name"] in EXPECTED_SEED12]
    if [job["name"] for job in native0] != list(EXPECTED_SEED0):
        raise ValueError("seed-0 wave differs from the seven-job registered universe")
    if [job["name"] for job in native12] != list(EXPECTED_SEED12):
        raise ValueError("seed-1/2 wave differs from the sixteen-job registered universe")
    if not 0.0 < float(args.max_project_ram_gib):
        raise ValueError("project RAM limit must be a positive runtime cap")
    reservations = _reservations(
        args.ram_reservations,
        [*EXPECTED_SEED0, *EXPECTED_SEED12],
        float(args.max_project_ram_gib),
    )
    guarded0 = [
        _job_with_guards(job, runs_root=args.runs_root, ram_gib=reservations[job["name"]])
        for job in native0
    ]
    guarded12 = [
        _job_with_guards(job, runs_root=args.runs_root, ram_gib=reservations[job["name"]])
        for job in native12
    ]
    common = {
        "schema_version": 2,
        "global_selection": str(args.global_selection),
        "global_selection_sha256": global_digest,
        "expected_commits": str(args.expected_commits),
        "expected_commits_sha256": sha256_file(args.expected_commits),
        "legacy_sufs_provenance_note": str(args.sufs_provenance_note),
        "legacy_sufs_provenance_note_sha256": sufs_note_digest,
        "host_ram_limit_gib": float(args.max_project_ram_gib),
    }
    seed0_payload = {
        **common,
        "phase": "full_seed0_native",
        "launch_authorized": True,
        "jobs": guarded0,
    }
    _write_yaml(args.seed0_out, seed0_payload)
    locked_payload = {
        **common,
        "phase": "full_seed12_native_locked",
        "launch_authorized": False,
        "barrier": "all eight native seed-0 runs validated",
        "seed0_queue_sha256": sha256_file(args.seed0_out),
        "jobs_locked": guarded12,
    }
    _write_yaml(args.seed12_plan_out, locked_payload)
    proof = {
        "schema_version": 1,
        "operation": "prepare_breadth_first_deep_waves",
        "global_selection_sha256": global_digest,
        "seed0_queue": {"path": str(args.seed0_out), "sha256": sha256_file(args.seed0_out)},
        "seed12_locked_plan": {
            "path": str(args.seed12_plan_out),
            "sha256": sha256_file(args.seed12_plan_out),
        },
        "completed_before_wave": completed,
        "seed0_jobs": list(EXPECTED_SEED0),
        "seed12_jobs_locked": list(EXPECTED_SEED12),
        "sufs_selection_sha256": sufs_digest,
        "sufs_selection_semantically_equal": bool(existing_sufs),
        "legacy_sufs_provenance_note_sha256": sufs_note_digest,
    }
    atomic_json(args.proof_out, proof)


def _verify_commit_map_binding(
    seed0: Mapping[str, Any], locked: Mapping[str, Any], path: Path
) -> str:
    digest = sha256_file(path)
    if (
        locked.get("expected_commits_sha256") != digest
        or seed0.get("expected_commits_sha256") != digest
    ):
        raise ValueError("expected-commit map changed after wave preparation")
    return digest


def authorize_seed12(args: argparse.Namespace) -> None:
    proof = _read_json(args.prepare_proof, "prepare proof")
    seed0 = _read_yaml(args.seed0_queue, "seed-0 queue")
    locked = _read_yaml(args.seed12_plan, "locked seed-1/2 plan")
    if sha256_file(args.seed0_queue) != proof["seed0_queue"]["sha256"]:
        raise ValueError("seed-0 queue hash differs from prepare proof")
    if sha256_file(args.seed12_plan) != proof["seed12_locked_plan"]["sha256"]:
        raise ValueError("locked seed-1/2 plan hash differs from prepare proof")
    if seed0.get("launch_authorized") is not True or locked.get("launch_authorized") is not False:
        raise ValueError("wave authorization states are invalid")
    if "jobs" in locked or [job["name"] for job in locked.get("jobs_locked", [])] != list(EXPECTED_SEED12):
        raise ValueError("locked plan is launchable or its job universe changed")
    _verify_commit_map_binding(seed0, locked, args.expected_commits)
    commits = _commit_map(args.expected_commits)
    global_selection = _read_json(args.global_selection, "global selection")
    if sha256_file(args.global_selection) != locked["global_selection_sha256"]:
        raise ValueError("global selection changed after seed-0 authorization")
    barrier = {}
    barrier["full.unet.seed0"] = _validate_completed(
        global_selection=global_selection,
        method="unet",
        seed=0,
        runs_root=args.runs_root,
        commit=commits["unet"],
    )
    for method in NATIVE:
        if method == "unet":
            continue
        barrier[f"full.{method}.seed0"] = _validate_completed(
            global_selection=global_selection,
            method=method,
            seed=0,
            runs_root=args.runs_root,
            commit=commits[method],
        )
    authorized = dict(locked)
    authorized.pop("jobs_locked")
    authorized["phase"] = "full_seed12_native_authorized"
    authorized["launch_authorized"] = True
    authorized["barrier_proof"] = barrier
    authorized["barrier_proof_sha256"] = canonical_sha256(barrier)
    authorized["jobs"] = locked["jobs_locked"]
    _write_yaml(args.seed12_out, authorized)
    atomic_json(
        args.proof_out,
        {
            "schema_version": 1,
            "operation": "authorize_seed12_after_seed0_barrier",
            "seed0_barrier": barrier,
            "seed0_queue_sha256": sha256_file(args.seed0_queue),
            "seed12_locked_plan_sha256": sha256_file(args.seed12_plan),
            "seed12_authorized_queue_sha256": sha256_file(args.seed12_out),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-seed0")
    for name in (
        "global-selection", "pilot-selection", "sufs-selection",
        "pilot-seed0-jobs", "remaining-seed0-jobs", "seed12-jobs",
        "expected-commits", "ram-reservations", "runs-root", "seed0-out",
        "sufs-provenance-note",
        "seed12-plan-out", "proof-out",
    ):
        prepare.add_argument(f"--{name}", required=True, type=Path)
    prepare.add_argument("--sufs-provenance-note-sha256", required=True)
    prepare.add_argument("--max-project-ram-gib", required=True, type=float)
    authorize = sub.add_parser("authorize-seed12")
    for name in (
        "global-selection", "expected-commits", "runs-root", "prepare-proof",
        "seed0-queue", "seed12-plan", "seed12-out", "proof-out",
    ):
        authorize.add_argument(f"--{name}", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare-seed0":
        prepare_seed0(arguments)
    else:
        authorize_seed12(arguments)
