#!/usr/bin/env python3
"""Prepare or run the authenticated sweep-to-validation handoff.

``prepare`` is results-blind and launches nothing. ``run`` watches one exact
mixed-recovery controller, co-schedules only validation probability replay in
the unused registered slots, finalizes validation metadata, adds the later
DRUnet/RUnet/AlexNet/SegFormer replay wave when its immutable run cards exist,
and prepares (but does not launch) selected full seed-0 training jobs.

No command in this controller accepts a test split.  Source ledgers are opened
read-only and are never edited or signalled.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once, canonical_sha256, sha256_file
import run_queue_guarded as guarded


PLAN_KIND = "autonomous_validation_handoff_plan"
HEX40 = set("0123456789abcdef")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--spec", required=True, type=Path)
    prepare.add_argument("--code-root", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--proof-out", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--plan-sha256", required=True)
    return parser.parse_args(argv)


def _read(path: Path, description: str) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _process_start_ticks(pid: int) -> Optional[int]:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _process_argv_sha256(pid: int) -> Optional[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    argv = [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]
    return canonical_sha256(argv) if argv else None


def _project_root(root: Path) -> Path:
    """Resolve either the standalone release layout or its source subdirectory."""
    root = root.resolve()
    candidates = (root, root / "segbench")
    matches = [
        candidate
        for candidate in candidates
        if (candidate / "scripts" / "autonomous_validation_handoff.py").is_file()
        and (candidate / "segbench" / "artifact_identity.py").is_file()
    ]
    if len(matches) != 1:
        raise ValueError("code root does not resolve to exactly one benchmark project")
    return matches[0]


def _code_identity(root: Path) -> Dict[str, Any]:
    root = root.resolve()
    project = _project_root(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True).strip()
    if len(commit) != 40 or any(c not in HEX40 for c in commit) or dirty:
        raise ValueError("handoff code must be a clean immutable commit")
    files = {}
    for relative in (
        "scripts/autonomous_validation_handoff.py",
        "scripts/run_queue_guarded.py",
        "scripts/prepare_validation_probability_replays.py",
        "scripts/run_validation_probability_replay.py",
        "scripts/finalize_validation_replay_selection.py",
        "scripts/prepare_selected_seed0_queue.py",
    ):
        path = project / relative
        files[relative] = sha256_file(path)
    return {
        "root": str(root),
        "project_root": str(project),
        "git_commit": commit,
        "git_tree": tree,
        "files_sha256": files,
    }


def _verify_code(value: Mapping[str, Any]) -> Path:
    root = Path(str(value.get("root", "")))
    observed = _code_identity(root)
    if observed != value:
        raise ValueError("handoff code identity changed")
    return Path(str(observed["project_root"]))


def _runtime_policy(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the environment-specific policy supplied by the operator."""
    value = plan.get("runtime_policy")
    if not isinstance(value, dict):
        raise ValueError("handoff runtime policy is absent")
    gpus = value.get("allowed_gpu_indices")
    checks = (
        isinstance(gpus, list) and bool(gpus)
        and all(type(item) is int and item >= 0 for item in gpus)
        and len(gpus) == len(set(gpus)),
        type(value.get("max_jobs_per_gpu")) is int and value["max_jobs_per_gpu"] > 0,
        type(value.get("min_free_mib")) is int and value["min_free_mib"] > 0,
        type(value.get("foreign_process_mib")) is int and value["foreign_process_mib"] >= 0,
        isinstance(value.get("memory_fraction"), (int, float))
        and 0 < float(value["memory_fraction"]) <= 1,
        isinstance(value.get("min_disk_free_fraction"), (int, float))
        and 0 < float(value["min_disk_free_fraction"]) < 1,
        isinstance(value.get("max_project_ram_gib"), (int, float))
        and float(value["max_project_ram_gib"]) > 0,
        type(value.get("max_cpu_cores")) is int and value["max_cpu_cores"] > 0,
        isinstance(value.get("min_available_ram_gib"), (int, float))
        and float(value["min_available_ram_gib"]) >= 0,
        isinstance(value.get("seed0_host_ram_gib"), (int, float))
        and float(value["seed0_host_ram_gib"]) > 0,
        type(value.get("seed0_cpu_workers")) is int
        and value["seed0_cpu_workers"] > 0,
    )
    if not all(checks):
        raise ValueError("handoff runtime policy is malformed")
    return dict(value)


def prepare(spec_path: Path, code_root: Path, out: Path, proof_out: Path) -> Dict[str, Any]:
    spec = _read(spec_path, "handoff specification")
    if spec.get("schema_version") != 1 or spec.get("kind") != "autonomous_validation_handoff_spec":
        raise ValueError("handoff specification kind changed")
    code = _code_identity(code_root)
    _runtime_policy(spec)
    transition = spec.get("transition_controller")
    if not isinstance(transition, dict):
        raise ValueError("transition controller identity is absent")
    pid = int(transition.get("pid", -1))
    allowed = transition.get("allowed_mixed_controller_argv")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("future mixed-controller argv bindings are absent")
    for item in allowed:
        if not isinstance(item, dict) or canonical_sha256(item.get("argv")) != item.get("argv_sha256"):
            raise ValueError("future mixed-controller argv digest changed")
    observed_start = _process_start_ticks(pid)
    observed_argv = _process_argv_sha256(pid)
    accepted_argv = {
        str(transition.get("current_argv_sha256")): "transition_watcher",
        **{
            str(item["argv_sha256"]): "mixed_recovery_controller"
            for item in allowed
        },
    }
    if (
        observed_start != transition.get("start_ticks")
        or observed_argv not in accepted_argv
    ):
        raise ValueError(
            "transition/mixed controller is not an exact live preregistered process"
        )
    initial_plan = Path(str(spec.get("initial_replay_plan", "")))
    initial_queue = Path(str(spec.get("initial_replay_queue", "")))
    for path in (initial_plan, initial_queue):
        if not path.is_absolute() or not path.is_file():
            raise ValueError("initial replay plan/queue is absent")
    replay = _read(initial_plan, "initial replay plan")
    queue = yaml.safe_load(initial_queue.read_text(encoding="utf-8"))
    if (
        replay.get("selection_split") != "val"
        or replay.get("test_or_heldout_opened") is not False
        or not isinstance(queue, dict)
        or queue.get("selection_split") != "val"
        or queue.get("test_inference_authorized") is not False
    ):
        raise ValueError("initial replay is not validation-only")
    plan = dict(spec)
    plan.update(
        kind=PLAN_KIND,
        code=code,
        specification={"path": str(spec_path), "sha256": sha256_file(spec_path)},
        initial_replay_plan_sha256=sha256_file(initial_plan),
        initial_replay_queue_sha256=sha256_file(initial_queue),
        controller_phase_at_preparation=accepted_argv[observed_argv],
        controller_argv_sha256_at_preparation=observed_argv,
        test_or_heldout_opened=False,
        gpu_launched=False,
    )
    atomic_json_once(out, plan)
    proof = {
        "schema_version": 1,
        "kind": "autonomous_validation_handoff_preparation",
        "status": "READY_NOT_LAUNCHED",
        "plan": str(out),
        "plan_sha256": sha256_file(out),
        "code_commit": code["git_commit"],
        "source_controller_authenticated": True,
        "source_controller_phase": accepted_argv[observed_argv],
        "source_controller_argv_sha256": observed_argv,
        "test_or_heldout_opened": False,
        "gpu_launched": False,
    }
    atomic_json_once(proof_out, proof)
    return proof


def _controller_identity(plan: Mapping[str, Any]) -> Dict[str, Any]:
    transition = plan["transition_controller"]
    pid = int(transition["pid"])
    if _process_start_ticks(pid) != int(transition["start_ticks"]):
        raise RuntimeError("transition/mixed controller PID start identity changed")
    observed = _process_argv_sha256(pid)
    matches = [
        item for item in transition["allowed_mixed_controller_argv"]
        if item["argv_sha256"] == observed
    ]
    if len(matches) != 1:
        raise RuntimeError("mixed controller argv is not the preregistered command")
    return {"pid": pid, "start_ticks": int(transition["start_ticks"]), "argv_sha256": observed, "argv": matches[0]["argv"]}


def _completed_source_controller_identity(
    plan: Mapping[str, Any], source_queue: Path,
) -> Dict[str, Any]:
    """Recover the preregistered controller argv after a clean queue exit.

    A finished controller is expected to disappear.  Its queue records the
    exact max-per-GPU value, which selects exactly one of the preregistered
    post-exec commands.  PID reuse or a still-live but changed process fails.
    """
    transition = plan["transition_controller"]
    pid = int(transition["pid"])
    observed_start = _process_start_ticks(pid)
    if observed_start is not None:
        # If it still exists, require the normal exact live authentication.
        return _controller_identity(plan)
    payload = yaml.safe_load(source_queue.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("completed source queue must be an object")
    cap = payload.get("runtime_policy", {}).get("max_jobs_per_gpu")
    if type(cap) is not int or cap <= 0:
        raise ValueError("completed source queue lacks its registered GPU cap")
    matches = []
    for item in transition["allowed_mixed_controller_argv"]:
        argv = list(item["argv"])
        if argv.count("--max-per-gpu") != 1:
            raise ValueError("preregistered mixed-controller argv has an invalid cap")
        index = argv.index("--max-per-gpu")
        if index + 1 < len(argv) and int(argv[index + 1]) == cap:
            matches.append(item)
    if len(matches) != 1:
        raise ValueError("completed source cap does not select exactly one controller argv")
    selected = matches[0]
    return {
        "pid": pid,
        "start_ticks": int(transition["start_ticks"]),
        "argv_sha256": selected["argv_sha256"],
        "argv": selected["argv"],
        "controller_exited_after_completion": True,
    }


def _stable_state(path: Path) -> Dict[str, Any]:
    for _ in range(10):
        first = path.read_bytes()
        value = json.loads(first)
        if first == path.read_bytes() and isinstance(value, dict):
            return value
        time.sleep(0.1)
    raise RuntimeError("source ledger did not yield a stable snapshot")


def _wait_for_source(plan: Mapping[str, Any]) -> tuple[Dict[str, Any], list[guarded.GuardedJob]]:
    queue_path = Path(plan["source_mixed_queue"])
    state_path = Path(plan["source_mixed_state"])
    proof_path = Path(plan["source_mixed_proof"])
    deadline = time.monotonic() + float(plan.get("source_wait_seconds", 86400))
    while time.monotonic() < deadline:
        if queue_path.is_file() and state_path.is_file() and proof_path.is_file():
            jobs = guarded.load_guarded_jobs(queue_path)
            state = _stable_state(state_path)
            if state.get("queue_sha256") != sha256_file(queue_path):
                raise ValueError("source mixed ledger/queue binding changed")
            statuses = {name: item.get("status") for name, item in state.get("jobs", {}).items()}
            if any(value == "failed" for value in statuses.values()):
                raise RuntimeError("source mixed recovery contains a failed job")
            if statuses and all(value in {"running", "completed"} for value in statuses.values()):
                if all(value == "completed" for value in statuses.values()):
                    # Authenticate every durable completion before accepting an
                    # expected controller exit. The read-only coexistence guard
                    # rechecks these identities on every scheduling cycle.
                    by_name = {job.name: job for job in jobs}
                    for name, entry in state["jobs"].items():
                        job = by_name[name]
                        if (
                            not job.completion_artifact.is_file()
                            or entry.get("artifact_sha256")
                            != sha256_file(job.completion_artifact)
                        ):
                            raise ValueError(f"{name}: completed source artifact changed")
                    controller = _completed_source_controller_identity(plan, queue_path)
                else:
                    controller = _controller_identity(plan)
                return controller, jobs
        time.sleep(float(plan.get("poll_seconds", 10)))
    raise TimeoutError("timed out waiting for a stable mixed recovery queue")


def _coexistence_plan(
    plan: Mapping[str, Any], controller: Mapping[str, Any], source_jobs: Sequence[guarded.GuardedJob],
    target_queue: Path, output: Path,
) -> Dict[str, Any]:
    source_queue = Path(plan["source_mixed_queue"])
    source_proof = Path(plan["source_mixed_proof"])
    argv = list(controller["argv"])
    max_per_gpu = int(argv[argv.index("--max-per-gpu") + 1])
    policy = _runtime_policy(plan)
    value = {
        "schema_version": 1,
        "kind": "guarded_queue_coexisting_controller_plan",
        "target_queue_sha256": sha256_file(target_queue),
        "allowed_gpus": list(policy["allowed_gpu_indices"]),
        "max_jobs_per_gpu": int(policy["max_jobs_per_gpu"]),
        "source_max_jobs_per_gpu": max_per_gpu,
        "max_project_ram_gib": float(policy["max_project_ram_gib"]),
        "max_cpu_cores": int(policy["max_cpu_cores"]),
        "memory_fraction": float(policy["memory_fraction"]),
        "read_only_source": True,
        "may_signal_source": False,
        "may_write_source_ledger": False,
        "source_queue": str(source_queue),
        "source_queue_sha256": sha256_file(source_queue),
        "source_state": str(plan["source_mixed_state"]),
        "source_proof": str(source_proof),
        "source_proof_sha256": sha256_file(source_proof),
        "source_controller": {key: controller[key] for key in ("pid", "start_ticks", "argv_sha256")},
        "source_jobs": [
            {
                "name": job.name,
                "identity_sha256": job.identity_sha256,
                "completion_artifact": str(job.completion_artifact),
                "host_ram_gib": job.host_ram_gib,
                "cpu_workers": job.cpu_workers,
            }
            for job in source_jobs
        ],
    }
    atomic_json_once(output, value)
    return value


def _run_child(argv: Sequence[str], *, cwd: Path, label: str, log: Path) -> None:
    forbidden = {"test", "heldout"}
    if any(str(item).lower() in forbidden for item in argv):
        raise ValueError(f"{label}: forbidden split token in child argv")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(label + " argv=" + json.dumps(list(argv)) + "\n")
        handle.flush()
        result = subprocess.run(list(argv), cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} exited with {result.returncode}")


def _queue_has_jobs(path: Path) -> bool:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(value, dict) and value.get("launch_authorized") is True and bool(value.get("jobs"))


def _guard_argv(plan: Mapping[str, Any], queue: Path, state: Path, coexist: Path) -> list[str]:
    root = Path(str(plan["code"]["project_root"]))
    policy = _runtime_policy(plan)
    gpu_csv = ",".join(str(item) for item in policy["allowed_gpu_indices"])
    return [
        str(plan["python"]), str(root / "scripts/run_queue_guarded.py"),
        "--jobs", str(queue), "--results", str(plan["restricted_root"]),
        "--state", str(state), "--gpus", gpu_csv,
        "--max-per-gpu", str(policy["max_jobs_per_gpu"]),
        "--min-free-mib", str(policy["min_free_mib"]),
        "--foreign-process-mib", str(policy["foreign_process_mib"]),
        "--memory-fraction", str(policy["memory_fraction"]),
        "--poll-seconds", str(plan.get("poll_seconds", 10)),
        "--min-disk-free-fraction", str(policy["min_disk_free_fraction"]),
        "--max-project-ram-gib", str(policy["max_project_ram_gib"]),
        "--max-cpu-cores", str(policy["max_cpu_cores"]),
        "--min-available-ram-gib", str(policy["min_available_ram_gib"]),
        "--coexisting-controller-plan", str(coexist),
    ]


def _late_inventory(plan: Mapping[str, Any], output: Path) -> None:
    initial = _read(Path(plan["initial_replay_plan"]), "initial replay plan")
    initial_candidates = initial.get("candidates")
    if not isinstance(initial_candidates, list) or not initial_candidates:
        raise ValueError("initial replay plan lacks common input bindings")
    common = initial_candidates[0].get("inputs")
    if not isinstance(common, dict):
        raise ValueError("initial replay candidate lacks input bindings")
    candidates = []
    for item in plan["late_candidates"]:
        for rate, slug in ((0.0003, "0p0003"), (0.001, "0p001")):
            run_dir = Path(str(item["run_root"])) / f"lr_{slug}" / "seed0"
            candidates.append({
                "name": f"{item['method']}.lr{slug}.seed0",
                "producer": "native", "method": item["method"], "role": "sweep",
                "selection_group": f"{item['method']}.seed0", "learning_rate": rate,
                "seed": 0, "config": str(item[f"config_{slug}"]),
                "checkpoint": str(run_dir / "best.pt"),
                "training_card": str(run_dir / "run.json"),
                "host_ram_gib": float(item.get("host_ram_gib", 24)),
                "cpu_workers": 0,
            })
    value = {
        "schema_version": 1,
        "kind": "dataset1_validation_probability_replay_candidates",
        "python": plan["python"],
        "manifest": common["manifest"]["path"],
        "splits": common["splits"]["path"],
        "cache_dir": common["cache"]["path"],
        "cache_compatibility_receipt": common["cache_compatibility"]["path"],
        "threshold_amendment": common["threshold_amendment"]["path"],
        "restricted_root": plan["restricted_root"],
        "output_root": str(Path(plan["handoff_root"]) / "late_replay_outputs"),
        "candidates": candidates,
    }
    if output.exists():
        observed = yaml.safe_load(output.read_text(encoding="utf-8"))
        if canonical_sha256(observed) != canonical_sha256(value):
            raise FileExistsError("late replay inventory changed")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def _wait_late_cards(plan: Mapping[str, Any]) -> None:
    deadline = time.monotonic() + float(plan.get("late_wait_seconds", 172800))
    paths = []
    for item in plan["late_candidates"]:
        for slug in ("0p0003", "0p001"):
            root = Path(item["run_root"]) / f"lr_{slug}" / "seed0"
            paths.extend([root / "best.pt", root / "run.json"])
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            return
        time.sleep(float(plan.get("poll_seconds", 10)))
    raise TimeoutError("late sweep run cards did not become complete")


def run(plan_path: Path, expected_sha: str) -> Dict[str, Any]:
    if sha256_file(plan_path) != expected_sha:
        raise ValueError("handoff plan SHA-256 changed")
    plan = _read(plan_path, "handoff plan")
    if plan.get("kind") != PLAN_KIND or plan.get("test_or_heldout_opened") is not False:
        raise ValueError("handoff plan contract changed")
    root = _verify_code(plan["code"])
    controller, source_jobs = _wait_for_source(plan)
    handoff = Path(plan["handoff_root"])
    log = handoff / "handoff.log"
    initial_queue = Path(plan["initial_replay_queue"])
    coexist = handoff / "initial_coexistence.json"
    _coexistence_plan(plan, controller, source_jobs, initial_queue, coexist)
    _run_child(
        _guard_argv(plan, initial_queue, handoff / "initial_queue_state.json", coexist),
        cwd=root, label="initial-validation-replay", log=log,
    )
    selections = handoff / "initial_selections"
    deferred = handoff / "initial_deferred_queue.yaml"
    finalizer = [
        str(plan["python"]), str(root / "scripts/finalize_validation_replay_selection.py"),
        "--plan", str(plan["initial_replay_plan"]), "--plan-sha256", str(plan["initial_replay_plan_sha256"]),
        "--out-dir", str(selections), "--eligible-queue-out", str(deferred),
    ]
    _run_child(finalizer, cwd=root, label="initial-validation-finalizer", log=log)
    if _queue_has_jobs(deferred):
        deferred_coexist = handoff / "deferred_coexistence.json"
        _coexistence_plan(plan, controller, source_jobs, deferred, deferred_coexist)
        _run_child(
            _guard_argv(plan, deferred, handoff / "deferred_queue_state.json", deferred_coexist),
            cwd=root, label="deferred-full-checkpoint-validation-replay", log=log,
        )

    _wait_late_cards(plan)
    late_inventory = handoff / "late_replay_candidates.yaml"
    late_plan = handoff / "late_replay_plan.json"
    late_queue = handoff / "late_replay_queue.yaml"
    late_proof = handoff / "late_replay_preparation.json"
    _late_inventory(plan, late_inventory)
    compiler = [
        str(plan["python"]), str(root / "scripts/prepare_validation_probability_replays.py"),
        "--candidates", str(late_inventory), "--code-root", str(root),
        "--out-plan", str(late_plan), "--out-queue", str(late_queue),
        "--proof-out", str(late_proof),
    ]
    _run_child(compiler, cwd=root, label="late-replay-compiler", log=log)
    late_coexist = handoff / "late_coexistence.json"
    _coexistence_plan(plan, controller, source_jobs, late_queue, late_coexist)
    _run_child(
        _guard_argv(plan, late_queue, handoff / "late_queue_state.json", late_coexist),
        cwd=root, label="late-validation-replay", log=log,
    )
    late_selections = handoff / "late_selections"
    _run_child([
        str(plan["python"]), str(root / "scripts/finalize_validation_replay_selection.py"),
        "--plan", str(late_plan), "--plan-sha256", sha256_file(late_plan),
        "--out-dir", str(late_selections),
    ], cwd=root, label="late-validation-finalizer", log=log)

    seed0_queue = handoff / "selected_full_seed0_queue.yaml"
    seed0_proof = handoff / "selected_full_seed0_preparation.json"
    builder = [
        str(plan["python"]), str(root / "scripts/prepare_selected_seed0_queue.py"),
        "--selection-summary", str(selections / "summary.json"),
        "--selection-summary", str(late_selections / "summary.json"),
        "--code-root", str(root), "--python", str(plan["python"]),
        "--manifest", str(plan["manifest"]), "--splits", str(plan["splits"]),
        "--cache-dir", str(plan["cache_dir"]),
        "--threshold-amendment", str(plan["training_threshold_amendment"]),
        "--runs-root", str(plan["runs_root"]),
        "--config-out", str(handoff / "selected_full_configs"),
        "--queue-out", str(seed0_queue), "--proof-out", str(seed0_proof),
        "--segformer-dir", str(plan["segformer_dir"]),
        "--job-host-ram-gib", str(_runtime_policy(plan)["seed0_host_ram_gib"]),
        "--job-cpu-workers", str(_runtime_policy(plan)["seed0_cpu_workers"]),
    ]
    _run_child(builder, cwd=root, label="selected-full-seed0-planner", log=log)
    result = {
        "schema_version": 1,
        "kind": "autonomous_validation_handoff_completion",
        "status": "COMPLETE_NO_TEST_OPENED",
        "plan_sha256": expected_sha,
        "test_or_heldout_opened": False,
        "source_ledger_written": False,
        "source_signalled": False,
        "initial_selection_summary": str(selections / "summary.json"),
        "late_selection_summary": str(late_selections / "summary.json"),
        "selected_full_seed0_queue": str(seed0_queue),
        "selected_full_seed0_queue_sha256": sha256_file(seed0_queue),
        "selected_full_seed0_launched": False,
    }
    atomic_json_once(handoff / "completion.json", result)
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
