#!/usr/bin/env python3
"""Resume-safe, fail-closed supervisor for the native Stage 3 pipeline.

The supervisor deliberately does not know how to score a mask and never opens
the manifest, a test image, a prediction, or ground truth.  It only advances
through already registered producers and verifies their control-plane
artifacts.  In particular, it first invokes the narrowly scoped orphan-worker
reconciler (which may update only attested ``running`` sweep ledger entries),
then resumes the unchanged frozen sweep queue.  It enforces the seed-0 barrier,
permits terminal seed-1/2 failures only as input to the separate >=2-seed
readiness gate, and does not invoke the one-time finalizer until inference has
a terminal, hash-bound ledger.

All commands are argument vectors (never shell strings).  Every producer is
executed from a clean, pinned Git worktree, under lowered CPU priority.  The
configuration, state, logs, queues, ledgers and outputs live in the restricted
results area; the code contains no machine-specific paths or device choices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import yaml

import _path  # noqa: F401
from reconcile_guarded_orphans import validate_reconciliation_proof
from segbench.artifact_identity import canonical_sha256, sha256_file


SCHEMA_VERSION = 1
KIND = "stage3_native_unattended_supervisor"
DYNAMIC_SHA256 = "@supervisor:producer-sha256"
STEP_ORDER = (
    "reconcile_sweep",
    "resume_sweep",
    "select_full",
    "prepare_seed0",
    "run_seed0",
    "authorize_seed12",
    "run_seed12",
    "prepare_inference",
    "run_inference",
    "run_auxiliary",
    "finalize",
)
TERMINAL = {"completed", "failed"}


class SupervisorError(RuntimeError):
    """A fail-closed supervisor validation or execution failure."""


class SupervisorInterrupted(RuntimeError):
    """A requested resumable stop, not a scientific gate failure."""


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _set_child_priority() -> None:
    """Set the child to the registered absolute niceness.

    The supervisor itself may already run under ``nice -n 10``.  Incremental
    ``os.nice(10)`` would then create a nice-19 queue controller whose
    unprivileged workers cannot restore nice 10.  An absolute assignment keeps
    controller and workers on the same registered priority.
    """
    os.setpriority(os.PRIO_PROCESS, 0, 10)


def _git_blob_sha1(path: Path) -> str:
    """Compute Git's blob identity for pinned small snapshot metadata."""
    digest = hashlib.sha1()  # noqa: S324 - required Git object identity
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _read_yaml(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a YAML object: {path}")
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


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _disjoint(left: Path, right: Path) -> bool:
    """Return whether neither path contains the other after resolution."""
    return not _inside(left, right) and not _inside(right, left)


def _option(argv: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"expected exactly one {name} value")
    return argv[positions[0] + 1]


def _hex(value: object, length: int, description: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{description} must be lowercase {length}-hex")
    return value


def _queue_job_identity(job: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "name": job.get("name"),
            "argv": job.get("argv"),
            "env": job.get("env", {}),
            "completion_artifact": job.get("completion_artifact"),
            "resources": job.get("resources"),
        }
    )


def _proc_start_ticks(proc: Path) -> int:
    suffix = proc.joinpath("stat").read_text(encoding="utf-8").rsplit(")", 1)[
        1
    ].strip().split()
    return int(suffix[19])


def _proc_argv(proc: Path) -> list[str]:
    values = proc.joinpath("cmdline").read_bytes().split(b"\0")
    return [
        value.decode("utf-8", errors="surrogateescape") for value in values if value
    ]


def _live_process_identity(pid: int) -> Optional[Dict[str, Any]]:
    """Return a PID-reuse-resistant process identity from procfs."""
    proc = Path("/proc") / str(pid)
    try:
        start_ticks_before = _proc_start_ticks(proc)
        command = _proc_argv(proc)
        start_ticks_after = _proc_start_ticks(proc)
    except (OSError, ProcessLookupError, ValueError, IndexError):
        return None
    if start_ticks_before != start_ticks_after:
        return None
    return {
        "pid": pid,
        "start_ticks": start_ticks_after,
        "argv_sha256": canonical_sha256(command),
    }


def inspect_ledger(
    queue_path: Path,
    ledger_path: Path,
    expected_names: Sequence[str],
    *,
    allow_terminal_failures: bool,
    require_timing_attestation: bool = False,
    allowed_gpus: Optional[Sequence[int]] = None,
    max_host_ram_gib: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate a guarded queue/ledger pair without changing either file."""
    queue = _read_yaml(queue_path, "guarded queue")
    if queue.get("launch_authorized") is not True:
        raise ValueError("guarded queue is not launch-authorized")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or [job.get("name") for job in jobs] != list(
        expected_names
    ):
        raise ValueError("guarded queue job universe/order changed")
    by_name: Dict[str, Mapping[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("guarded queue contains a non-object job")
        name = str(job.get("name", ""))
        if job.get("job_identity_sha256") != _queue_job_identity(job):
            raise ValueError(f"{name}: queue identity does not reproduce")
        completion = Path(str(job.get("completion_artifact", "")))
        resources = job.get("resources")
        if not completion.is_absolute():
            raise ValueError(f"{name}: completion artifact is not absolute")
        host_ram = float(resources.get("host_ram_gib", 0.0)) if isinstance(
            resources, dict
        ) else 0.0
        if not math.isfinite(host_ram) or host_ram <= 0.0 or (
            max_host_ram_gib is not None and host_ram > max_host_ram_gib
        ):
            raise ValueError(f"{name}: invalid host-RAM reservation")
        by_name[name] = job

    if require_timing_attestation:
        queue_phase = queue.get("phase")
        if not isinstance(queue_phase, str) or not queue_phase:
            raise ValueError("timing-isolated queue lacks its registered phase")
        if queue.get("required_max_jobs_per_gpu") != 1 or queue.get(
            "required_launcher_args"
        ) != ["--max-per-gpu", "1"]:
            raise ValueError("timing-isolated queue policy changed")
        if allowed_gpus is None:
            raise ValueError("timing-isolated ledger validation needs allowed_gpus")
        expected_attestation = {
            "schema_version": 1,
            "queue_phase": queue_phase,
            "queue_sha256": sha256_file(queue_path),
            "gpus": [int(value) for value in allowed_gpus],
            "max_per_gpu": 1,
            "memory_fraction": 0.3,
            "required_launcher_args": ["--max-per-gpu", "1"],
        }

    if not ledger_path.is_file():
        return {
            "terminal": False,
            "completed": [],
            "failed": [],
            "nonterminal": {name: "not_started" for name in expected_names},
        }
    ledger = _read_json(ledger_path, "guarded queue ledger")
    if ledger.get("queue_sha256") != sha256_file(queue_path):
        raise ValueError("guarded queue ledger is not bound to the queue")
    states = ledger.get("jobs")
    if not isinstance(states, dict) or set(states) != set(expected_names):
        raise ValueError("guarded queue ledger universe changed")

    if require_timing_attestation:
        if ledger.get("schema_version") != 2 or ledger.get(
            "launcher_attestation"
        ) != expected_attestation:
            raise ValueError("inference ledger lacks exact timing-isolation attestation")
    elif ledger.get("schema_version") != 1:
        raise ValueError("training ledger schema changed")

    completed, failed, nonterminal = [], [], {}
    for name in expected_names:
        state = states[name]
        if not isinstance(state, dict):
            raise ValueError(f"{name}: ledger entry is not an object")
        if state.get("identity_sha256") != by_name[name].get("job_identity_sha256"):
            raise ValueError(f"{name}: ledger identity differs from queue")
        status = state.get("status")
        if status == "completed":
            completion = Path(str(by_name[name]["completion_artifact"]))
            if not completion.is_file() or sha256_file(completion) != state.get(
                "artifact_sha256"
            ):
                raise ValueError(f"{name}: completed artifact disappeared or changed")
            completed.append(name)
        elif status == "failed":
            if int(state.get("attempts", 0)) < 2:
                raise ValueError(f"{name}: terminal failure lacks retry evidence")
            failed.append(name)
        elif status in {"pending", "running", "retrying", "preempted"}:
            nonterminal[name] = status
        else:
            raise ValueError(f"{name}: invalid ledger status {status!r}")
    terminal = not nonterminal
    if terminal and failed and not allow_terminal_failures:
        raise SupervisorError("terminal queue failures: " + ", ".join(failed))
    return {
        "terminal": terminal,
        "completed": completed,
        "failed": failed,
        "nonterminal": nonterminal,
        "queue_sha256": sha256_file(queue_path),
        "ledger_sha256": sha256_file(ledger_path),
    }


def _initial_state(config_sha256: str) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "config_sha256": config_sha256,
        "status": "running",
        "current_step": STEP_ORDER[0],
        "heartbeat_utc": _utc(),
        "steps": {
            name: {"status": "pending", "attempts": 0} for name in STEP_ORDER
        },
        "history": [],
    }


class Supervisor:
    """One deterministic, resumable Stage 3 state machine."""

    def __init__(
        self,
        config_path: Path,
        state_path: Path,
        log_path: Path,
        *,
        poll_seconds: float,
    ) -> None:
        self.config_path = config_path.resolve()
        self.state_path = state_path.resolve()
        self.log_path = log_path.resolve()
        self.poll_seconds = poll_seconds
        self.config = _read_yaml(self.config_path, "supervisor configuration")
        self._validate_config()
        config_sha = sha256_file(self.config_path)
        self.config_sha256 = config_sha
        if self.state_path.exists():
            self.state = _read_json(self.state_path, "supervisor state")
            if self.state.get("config_sha256") != config_sha:
                raise ValueError("supervisor configuration changed after state creation")
            if self.state.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("supervisor state schema changed")
        else:
            self.state = _initial_state(config_sha)
            self._save()
        self.stop_requested = False
        self.child: Optional[subprocess.Popen[Any]] = None

    @property
    def restricted_root(self) -> Path:
        return Path(str(self.config["restricted_root"])).resolve()

    def _event(self, event: str, **fields: Any) -> None:
        record = {"at_utc": _utc(), "event": event, **fields}
        self.state["history"].append(record)
        # Keep persistent state compact; the JSONL log has the full history.
        self.state["history"] = self.state["history"][-200:]
        _append_jsonl(self.log_path, record)
        self._save()

    def _save(self) -> None:
        self.state["heartbeat_utc"] = _utc()
        _atomic_json(self.state_path, self.state)

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.get("schema_version") != SCHEMA_VERSION or cfg.get("kind") != KIND:
            raise ValueError("unsupported supervisor configuration schema/kind")
        if cfg.get("launch_authorized") is not True:
            raise ValueError("supervisor configuration is not launch-authorized")
        root = Path(str(cfg.get("restricted_root", "")))
        if not root.is_absolute():
            raise ValueError("restricted_root must be absolute")
        for path in (self.config_path, self.state_path, self.log_path):
            if not _inside(path, root):
                raise ValueError(f"supervisor control artifact must stay under restricted_root: {path}")
        resources = cfg.get("resources")
        if not isinstance(resources, dict):
            raise ValueError("resources must be an object")
        allowed = resources.get("allowed_gpus")
        if (
            not isinstance(allowed, list)
            or not allowed
            or any(type(value) is not int or value < 0 for value in allowed)
            or len(allowed) != len(set(allowed))
        ):
            raise ValueError("allowed_gpus must be a non-empty unique list")
        max_jobs = resources.get("max_jobs_per_gpu")
        if type(max_jobs) is not int or not 1 <= max_jobs <= 3:
            raise ValueError("max_jobs_per_gpu must be within the registered cap")
        max_memory_fraction = float(resources.get("max_memory_fraction", 0.0))
        if not math.isfinite(max_memory_fraction) or not 0.0 < max_memory_fraction <= 0.3:
            raise ValueError("max_memory_fraction must be within the registered cap")
        max_cpu_cores = resources.get("max_cpu_cores")
        if type(max_cpu_cores) is not int or max_cpu_cores <= 0:
            raise ValueError("max_cpu_cores must be a positive runtime cap")
        max_ram_gib = float(resources.get("max_ram_gib", 0.0))
        if not math.isfinite(max_ram_gib) or max_ram_gib <= 0.0:
            raise ValueError("max_ram_gib must be a positive runtime cap")
        min_disk = float(resources.get("min_disk_free_fraction", 0.0))
        if not math.isfinite(min_disk) or not 0.15 <= min_disk < 1.0:
            raise ValueError("minimum disk-free fraction is below the registered floor")
        resource_root = Path(str(resources.get("resource_root", "")))
        if not resource_root.is_absolute() or not _inside(resource_root, root):
            raise ValueError("resource_root must be inside restricted_root")

        bindings = cfg.get("bindings")
        if not isinstance(bindings, dict) or not isinstance(
            bindings.get("manifest"), dict
        ):
            raise ValueError("bindings.manifest is required")
        manifest_binding = bindings["manifest"]
        manifest = Path(str(manifest_binding.get("path", "")))
        if (
            not manifest.is_absolute()
            or not _inside(manifest, root)
            or manifest.name != "latent_source.csv"
        ):
            raise ValueError("Stage 3 must use the bound latent_source.csv manifest")
        expected_manifest_sha = _hex(
            manifest_binding.get("sha256"), 64, "Stage 3 manifest digest"
        )
        if not manifest.is_file() or sha256_file(manifest) != expected_manifest_sha:
            raise ValueError("bound Stage 3 manifest bytes changed")
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if "source_sha256" not in header:
            raise ValueError("bound Stage 3 manifest lacks source_sha256")

        pins = cfg.get("code_pins")
        if not isinstance(pins, dict) or not pins:
            raise ValueError("code_pins must be a non-empty mapping")
        for name, pin in pins.items():
            if not isinstance(pin, dict):
                raise ValueError(f"code pin {name} is not an object")
            repo = Path(str(pin.get("repo_root", "")))
            if not repo.is_absolute():
                raise ValueError(f"code pin {name} repo_root must be absolute")
            _hex(pin.get("expected_commit"), 40, f"code pin {name}")
        roles = cfg.get("role_pins")
        if not isinstance(roles, dict) or any(
            roles.get(name) not in pins for name in ("training", "inference")
        ):
            raise ValueError("role_pins must bind training and inference code pins")

        sweep = cfg.get("sweep")
        if not isinstance(sweep, dict):
            raise ValueError("sweep must be an object")
        if len(sweep.get("expected_names", [])) != 14:
            raise ValueError("existing sweep must contain exactly 14 registered jobs")
        for key in ("queue", "ledger"):
            path = Path(str(sweep.get(key, "")))
            if not path.is_absolute() or not _inside(path, root):
                raise ValueError(f"sweep {key} must be inside restricted_root")
        _hex(sweep.get("queue_sha256"), 64, "frozen sweep queue digest")

        actions = cfg.get("actions")
        if not isinstance(actions, dict) or set(actions) != set(STEP_ORDER):
            raise ValueError("actions must contain exactly the registered Stage 3 steps")
        for step in STEP_ORDER:
            self._validate_action_definition(step, actions[step])
        delivery_pin = actions["prepare_inference"]["code_pin"]
        for step in (
            "reconcile_sweep",
            "resume_sweep",
            "prepare_seed0",
            "run_seed0",
            "authorize_seed12",
            "run_seed12",
            "run_inference",
            "run_auxiliary",
            "finalize",
        ):
            if actions[step]["code_pin"] != delivery_pin:
                raise ValueError(f"{step} must use the attested delivery runner pin")
        training_pin = roles["training"]
        if actions["select_full"]["code_pin"] != training_pin:
            raise ValueError("select_full must use the frozen training pin")

        # The delivery-pinned authorizer is control-plane-only.  It may add
        # queue guards, but it must consume the exact selection and raw job
        # queues emitted by the frozen training pin.  Requiring these
        # producer/consumer bindings prevents changing a trainer entrypoint,
        # generated config, resume/checkpoint argument, or job environment by
        # swapping an intermediate queue at the pin boundary.
        selection = actions["select_full"]
        selection_paths = {
            option: str(Path(_option(selection["argv"], option)))
            for option in (
                "--selection-out",
                "--pilot-seed0-jobs-out",
                "--remaining-seed0-jobs-out",
                "--seed12-jobs-out",
            )
        }
        sufs_selection_path = str(
            Path(selection_paths["--selection-out"]).with_name(
                "sufs_ft_selection.json"
            )
        )
        required_selection_outputs = {
            *selection_paths.values(),
            sufs_selection_path,
        }
        if not required_selection_outputs <= set(selection.get("outputs", [])):
            raise ValueError(
                "select_full must declare every selection and raw scientific queue output"
            )

        prepare = actions["prepare_seed0"]
        prepare_cross = {
            (item.get("option"), item.get("path"), item.get("producer_step"))
            for item in prepare.get("cross_step_inputs", [])
        }
        for option, path in (
            ("--global-selection", selection_paths["--selection-out"]),
            ("--pilot-seed0-jobs", selection_paths["--pilot-seed0-jobs-out"]),
            (
                "--remaining-seed0-jobs",
                selection_paths["--remaining-seed0-jobs-out"],
            ),
            ("--seed12-jobs", selection_paths["--seed12-jobs-out"]),
        ):
            if (option, path, "select_full") not in prepare_cross:
                raise ValueError(
                    f"prepare_seed0 must hash-bind frozen training output {option}"
                )

        # The generated SUFS selection remains an explicit select_full output
        # referenced by the global selection, but downstream authorization
        # must independently cross-check it against the earlier audited SUFS
        # artifact rather than consuming the generated copy as its authority.
        audited_sufs = cfg.get("audited_sufs_selection")
        if not isinstance(audited_sufs, dict):
            raise ValueError("audited_sufs_selection is required")
        audited_sufs_path = Path(str(audited_sufs.get("path", "")))
        audited_sufs_sha = _hex(
            audited_sufs.get("sha256"), 64, "audited SUFS selection digest"
        )
        if (
            not audited_sufs_path.is_absolute()
            or not _inside(audited_sufs_path, root)
            or audited_sufs_path.resolve() == Path(sufs_selection_path).resolve()
            or not audited_sufs_path.is_file()
            or sha256_file(audited_sufs_path) != audited_sufs_sha
        ):
            raise ValueError("audited SUFS selection path/digest is invalid")
        for consumer in ("prepare_seed0", "prepare_inference"):
            action = actions[consumer]
            if Path(_option(action["argv"], "--sufs-selection")).resolve() != (
                audited_sufs_path.resolve()
            ):
                raise ValueError(f"{consumer} must consume the audited SUFS selection")
            bindings = [
                item
                for item in action.get("bound_inputs", [])
                if item.get("option") == "--sufs-selection"
                and Path(str(item.get("path", ""))).resolve()
                == audited_sufs_path.resolve()
                and item.get("sha256") == audited_sufs_sha
            ]
            if len(bindings) != 1:
                raise ValueError(
                    f"{consumer} must hash-bind the audited SUFS selection"
                )

        authorize = actions["authorize_seed12"]
        authorize_cross = {
            (item.get("option"), item.get("path"), item.get("producer_step"))
            for item in authorize.get("cross_step_inputs", [])
        }
        for option, producer_option in (
            ("--global-selection", "--selection-out"),
            ("--prepare-proof", "--proof-out"),
            ("--seed0-queue", "--seed0-out"),
            ("--seed12-plan", "--seed12-plan-out"),
        ):
            producer = "select_full" if option == "--global-selection" else "prepare_seed0"
            source_action = actions[producer]
            path = str(Path(_option(source_action["argv"], producer_option)))
            if (option, path, producer) not in authorize_cross:
                raise ValueError(
                    f"authorize_seed12 must hash-bind barrier input {option}"
                )
        self._validate_selection_environment()
        reconciliation = actions["reconcile_sweep"]
        reconcile_argv = reconciliation["argv"]
        if Path(_option(reconcile_argv, "--queue")).resolve() != Path(
            str(sweep["queue"])
        ).resolve() or Path(_option(reconcile_argv, "--ledger")).resolve() != Path(
            str(sweep["ledger"])
        ).resolve():
            raise ValueError("reconcile_sweep must use the frozen sweep queue/ledger")
        capture_path = Path(_option(reconcile_argv, "--capture"))
        capture_bindings = [
            item
            for item in reconciliation.get("bound_inputs", [])
            if item.get("option") == "--capture"
            and Path(str(item.get("path", ""))).resolve() == capture_path.resolve()
        ]
        if len(capture_bindings) != 1:
            raise ValueError("reconcile_sweep must hash-bind its orphan capture")
        proof_path = Path(_option(reconcile_argv, "--proof-out"))
        if reconciliation.get("outputs") != [str(proof_path)]:
            raise ValueError("reconcile_sweep must declare only its proof output")
        for step in ("select_full", "prepare_inference", "finalize"):
            argv = actions[step]["argv"]
            if Path(_option(argv, "--manifest")).resolve() != manifest.resolve():
                raise ValueError(f"{step}: --manifest is not bound latent_source.csv")
        preparation = actions["prepare_inference"]
        prepared_paths = {
            str(Path(_option(preparation["argv"], option)))
            for option in (
                "--gate-out",
                "--plan-out",
                "--queue-out",
                "--prep-bundle-out",
                "--aux-plan-out",
                "--aux-queue-out",
            )
        }
        if not prepared_paths <= set(preparation.get("outputs", [])):
            raise ValueError(
                "prepare_inference must record deterministic and auxiliary planning outputs"
            )

        # Every planner data/output root is private, and the MC-dropout extra
        # has no shared directory with deterministic masks, cards, or work.
        root_options = (
            "--runs-root",
            "--masks-root",
            "--cards-root",
            "--work-root",
            "--stage2-masks",
            "--aux-masks-root",
            "--aux-variance-root",
            "--aux-work-root",
        )
        planner_roots: Dict[str, Path] = {}
        for option in root_options:
            path = Path(_option(preparation["argv"], option))
            if not path.is_absolute() or not _inside(path, root):
                raise ValueError(f"prepare_inference {option} must be inside restricted_root")
            planner_roots[option] = path
        for auxiliary in (
            "--aux-masks-root",
            "--aux-variance-root",
            "--aux-work-root",
        ):
            for deterministic in ("--masks-root", "--cards-root", "--work-root"):
                if not _disjoint(planner_roots[auxiliary], planner_roots[deterministic]):
                    raise ValueError(
                        f"{auxiliary} must be disjoint from deterministic {deterministic}"
                    )
        auxiliary_options = (
            "--aux-masks-root",
            "--aux-variance-root",
            "--aux-work-root",
        )
        for index, left in enumerate(auxiliary_options):
            for right in auxiliary_options[index + 1 :]:
                if not _disjoint(planner_roots[left], planner_roots[right]):
                    raise ValueError(f"{left} and {right} must be disjoint")
        if Path(_option(preparation["argv"], "--restricted-root")).resolve() != (
            root.resolve()
        ):
            raise ValueError("prepare_inference restricted root differs from configuration")
        configured_gpu_csv = ",".join(
            str(int(value)) for value in resources["allowed_gpus"]
        )
        if _option(preparation["argv"], "--allowed-gpus") != configured_gpu_csv:
            raise ValueError("prepare_inference GPU allow-list differs from configuration")
        if float(_option(preparation["argv"], "--max-project-ram-gib")) != float(
            resources["max_ram_gib"]
        ):
            raise ValueError("prepare_inference RAM cap differs from configuration")
        aux_ram = float(_option(preparation["argv"], "--aux-ram-gib"))
        if not 0.0 < aux_ram <= float(resources["max_ram_gib"]):
            raise ValueError("prepare_inference auxiliary RAM reservation exceeds cap")

        inference_queue = str(Path(_option(preparation["argv"], "--queue-out")))
        auxiliary_queue = str(Path(_option(preparation["argv"], "--aux-queue-out")))
        for consumer, queue_path in (
            ("run_inference", inference_queue),
            ("run_auxiliary", auxiliary_queue),
        ):
            action = actions[consumer]
            if str(Path(action["queue"])) != queue_path:
                raise ValueError(f"{consumer} must consume its exact prepared queue")
            cross = {
                (item.get("option"), item.get("path"), item.get("producer_step"))
                for item in action.get("cross_step_inputs", [])
            }
            if ("--jobs", queue_path, "prepare_inference") not in cross:
                raise ValueError(f"{consumer} must hash-bind its prepared queue")
        inference_action = actions["run_inference"]
        if (
            inference_action.get("expected_names_from_step")
            != "prepare_inference"
            or "expected_names" in inference_action
        ):
            raise ValueError(
                "run_inference names must derive from the frozen prepared queue"
            )
        aux_card = str(Path(_option(preparation["argv"], "--aux-card-out")))
        if aux_card not in actions["run_auxiliary"].get("outputs", []):
            raise ValueError("run_auxiliary must record its runtime completion card")

        finalizer = actions["finalize"]
        prep_path = str(Path(_option(preparation["argv"], "--prep-bundle-out")))
        cross = {
            (item.get("option"), item.get("path"), item.get("producer_step"))
            for item in finalizer.get("cross_step_inputs", [])
        }
        dynamic = {
            (item.get("option"), item.get("path"), item.get("producer_step"))
            for item in finalizer.get("dynamic_sha256_inputs", [])
        }
        if ("--prep-bundle", prep_path, "prepare_inference") not in cross or (
            "--expected-prep-bundle-sha256", prep_path, "prepare_inference"
        ) not in dynamic:
            raise ValueError("finalizer must bind the prepared bundle path and digest")
        aux_plan_path = str(Path(_option(preparation["argv"], "--aux-plan-out")))
        aux_queue_path = str(Path(_option(preparation["argv"], "--aux-queue-out")))
        aux_ledger_path = str(Path(actions["run_auxiliary"]["ledger"]))
        aux_card_path = str(Path(_option(preparation["argv"], "--aux-card-out")))
        for required in (
            ("--aux-plan", aux_plan_path, "prepare_inference"),
            ("--aux-queue", aux_queue_path, "prepare_inference"),
            ("--aux-ledger", aux_ledger_path, "run_auxiliary"),
            ("--aux-card", aux_card_path, "run_auxiliary"),
        ):
            if required not in cross:
                raise ValueError(f"finalizer lacks auxiliary binding {required[0]}")
        bundle_root = Path(_option(finalizer["argv"], "--bundle-out"))
        if not bundle_root.is_absolute() or not _inside(bundle_root, root):
            raise ValueError("finalizer --bundle-out must be inside restricted_root")
        expected_delivery = str(bundle_root / "delivery.json")
        if Path(_option(finalizer["argv"], "--declared-delivery")).resolve() != Path(
            expected_delivery
        ).resolve():
            raise ValueError("finalizer declared delivery differs from bundle-out")
        if Path(_option(finalizer["argv"], "--restricted-root")).resolve() != (
            root.resolve()
        ):
            raise ValueError("finalizer restricted root differs from configuration")
        for option in (
            "--gate",
            "--plan",
            "--prep-bundle",
            "--inference-queue",
            "--inference-ledger",
            "--aux-plan",
            "--aux-queue",
            "--aux-ledger",
            "--aux-card",
            "--manifest",
            "--splits",
        ):
            path = Path(_option(finalizer["argv"], option))
            if not path.is_absolute() or not _inside(path, root):
                raise ValueError(f"finalizer {option} must be inside restricted_root")
        if finalizer.get("outputs") != [expected_delivery]:
            raise ValueError("finalizer output must be exactly bundle-out/delivery.json")

    def _validate_selection_environment(self) -> None:
        """Bind selection-only environment inputs to frozen prior evidence."""
        cfg = self.config
        root = self.restricted_root
        actions = cfg["actions"]
        selection = actions["select_full"]
        environment = selection.get("env")
        required_environment = {
            "SEGBENCH_EXPECTED_COMMIT_MAP",
            "SEGBENCH_SEGFORMER_B2_DIR",
        }
        if not isinstance(environment, dict) or set(environment) != required_environment:
            raise ValueError(
                "select_full must declare only its two frozen selection environment inputs"
            )

        commit_map = Path(str(environment["SEGBENCH_EXPECTED_COMMIT_MAP"]))
        commit_bindings = [
            item
            for item in selection.get("bound_inputs", [])
            if item.get("env_key") == "SEGBENCH_EXPECTED_COMMIT_MAP"
            and Path(str(item.get("path", ""))).resolve() == commit_map.resolve()
        ]
        if len(commit_bindings) != 1:
            raise ValueError("select_full must hash-bind its expected-commit env input")
        commit_sha = str(commit_bindings[0].get("sha256", ""))
        if (
            not commit_map.is_absolute()
            or not _inside(commit_map, root)
            or not commit_map.is_file()
            or sha256_file(commit_map) != commit_sha
        ):
            raise ValueError("select_full expected-commit env input changed")
        for consumer in ("prepare_seed0", "authorize_seed12", "prepare_inference"):
            action = actions[consumer]
            if Path(_option(action["argv"], "--expected-commits")).resolve() != (
                commit_map.resolve()
            ):
                raise ValueError(f"{consumer} expected-commit map differs from selection")
            matching = [
                item
                for item in action.get("bound_inputs", [])
                if item.get("option") == "--expected-commits"
                and Path(str(item.get("path", ""))).resolve()
                == commit_map.resolve()
                and item.get("sha256") == commit_sha
            ]
            if len(matching) != 1:
                raise ValueError(f"{consumer} must hash-bind the expected-commit map")

        registration = cfg.get("segformer_snapshot")
        if not isinstance(registration, dict):
            raise ValueError("segformer_snapshot registration is required")
        snapshot = Path(str(registration.get("path", "")))
        revision = str(registration.get("revision", ""))
        if (
            not snapshot.is_absolute()
            or not _inside(snapshot, root)
            or not snapshot.is_dir()
            or not revision
            or snapshot.name != revision
            or Path(str(environment["SEGBENCH_SEGFORMER_B2_DIR"])).resolve()
            != snapshot.resolve()
        ):
            raise ValueError("SegFormer snapshot path/revision binding changed")

        files = registration.get("files")
        expected_files = {
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
        }
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ValueError("SegFormer snapshot file universe changed")
        for name in sorted(expected_files):
            record = files[name]
            if not isinstance(record, dict):
                raise ValueError(f"SegFormer file registration is invalid: {name}")
            path = Path(str(record.get("path", "")))
            expected_path = snapshot / name
            expected_sha = _hex(
                record.get("sha256"), 64, f"SegFormer {name} digest"
            )
            if (
                path.resolve() != expected_path.resolve()
                or not path.is_file()
                or not _inside(path, root)
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != expected_sha
            ):
                raise ValueError(f"SegFormer snapshot file identity changed: {name}")
            blob = record.get("git_blob_sha1")
            if blob is not None and (
                not isinstance(blob, str)
                or len(blob) != 40
                or _git_blob_sha1(path) != blob
            ):
                raise ValueError(f"SegFormer Git blob identity changed: {name}")

        base = registration.get("base_config")
        training_pin = cfg["code_pins"][cfg["role_pins"]["training"]]
        expected_base_path = (
            Path(str(training_pin["repo_root"]))
            / "segbench"
            / "configs"
            / "segformer_sub.yaml"
        )
        if not isinstance(base, dict):
            raise ValueError("SegFormer base-config registration is required")
        base_path = Path(str(base.get("path", "")))
        if (
            base_path.resolve() != expected_base_path.resolve()
            or not base_path.is_file()
            or sha256_file(base_path)
            != _hex(base.get("sha256"), 64, "SegFormer base-config digest")
        ):
            raise ValueError("SegFormer base-config identity changed")

        queue_path = Path(str(cfg["sweep"]["queue"]))
        if (
            not queue_path.is_file()
            or sha256_file(queue_path) != cfg["sweep"]["queue_sha256"]
        ):
            raise ValueError("frozen sweep queue changed before SegFormer validation")
        queue = _read_yaml(queue_path, "frozen native sweep queue")
        jobs = queue.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("frozen sweep queue lacks jobs")
        expected_names = (
            "sweep.segformer_sub.lr0p0003.seed0",
            "sweep.segformer_sub.lr0p001.seed0",
        )
        by_name = {
            str(job.get("name")): job
            for job in jobs
            if isinstance(job, dict) and job.get("name") in expected_names
        }
        if set(by_name) != set(expected_names):
            raise ValueError("frozen sweep queue lacks both SegFormer cells")
        registered_configs = registration.get("sweep_configs")
        if not isinstance(registered_configs, dict) or set(
            registered_configs
        ) != set(expected_names):
            raise ValueError("SegFormer sweep-config registration changed")

        config_paths = [base_path]
        for name in expected_names:
            job = by_name[name]
            if job.get("env") != {
                "SEGBENCH_SEGFORMER_B2_DIR": str(snapshot)
            }:
                raise ValueError(f"{name}: frozen snapshot environment changed")
            if job.get("job_identity_sha256") != _queue_job_identity(job):
                raise ValueError(f"{name}: frozen job identity changed")
            config_path = Path(_option(job["argv"], "--config"))
            record = registered_configs[name]
            if not isinstance(record, dict) or config_path.resolve() != Path(
                str(record.get("path", ""))
            ).resolve():
                raise ValueError(f"{name}: registered config path changed")
            expected_sha = _hex(
                record.get("sha256"), 64, f"{name} config digest"
            )
            if not config_path.is_file() or sha256_file(config_path) != expected_sha:
                raise ValueError(f"{name}: registered config bytes changed")
            config_paths.append(config_path)

        for config_path in config_paths:
            model = _read_yaml(config_path, "pinned SegFormer config").get("model")
            if not isinstance(model, dict) or {
                "pretrained": model.get("pretrained"),
                "pretrained_model": model.get("pretrained_model"),
                "pretrained_revision": model.get("pretrained_revision"),
                "pretrained_path_env": model.get("pretrained_path_env"),
                "local_files_only": model.get("local_files_only"),
            } != {
                "pretrained": True,
                "pretrained_model": "nvidia/mit-b2",
                "pretrained_revision": revision,
                "pretrained_path_env": "SEGBENCH_SEGFORMER_B2_DIR",
                "local_files_only": True,
            }:
                raise ValueError(f"SegFormer model pin changed: {config_path}")
            declared_files = model.get("pretrained_files")
            if not isinstance(declared_files, dict) or set(
                declared_files
            ) != expected_files:
                raise ValueError(f"SegFormer file declarations changed: {config_path}")
            for name, declared in declared_files.items():
                record = files[name]
                if not isinstance(declared, dict):
                    raise ValueError(f"SegFormer file declaration is invalid: {name}")
                if "sha256" in declared and (
                    declared.get("sha256") != record.get("sha256")
                    or declared.get("bytes") != record.get("bytes")
                ):
                    raise ValueError(f"SegFormer SHA-256 declaration changed: {name}")
                if "git_blob_sha1" in declared and declared.get(
                    "git_blob_sha1"
                ) != record.get("git_blob_sha1"):
                    raise ValueError(f"SegFormer Git blob declaration changed: {name}")

    def _validate_action_definition(self, step: str, action: object) -> None:
        if not isinstance(action, dict):
            raise ValueError(f"{step}: action must be an object")
        expected_type = (
            "guarded_queue"
            if step
            in {
                "resume_sweep",
                "run_seed0",
                "run_seed12",
                "run_inference",
                "run_auxiliary",
            }
            else "finalizer"
            if step == "finalize"
            else "command"
        )
        if action.get("type") != expected_type:
            raise ValueError(f"{step}: action type must be {expected_type}")
        pin_name = action.get("code_pin")
        if pin_name not in self.config["code_pins"]:
            raise ValueError(f"{step}: unknown code pin")
        argv = action.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(value, str) and value for value in argv
        ):
            raise ValueError(f"{step}: argv must be a non-empty string list")
        if any(any(token in value for token in (";", "&&", "||", "`", "$(")) for value in argv):
            raise ValueError(f"{step}: shell syntax is forbidden")
        environment = action.get("env", {})
        if not isinstance(environment, dict) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in environment.items()
        ):
            raise ValueError(f"{step}: env must be a string mapping")
        entrypoint = Path(str(action.get("entrypoint", "")))
        if entrypoint.is_absolute() or ".." in entrypoint.parts:
            raise ValueError(f"{step}: entrypoint must be relative to its pinned repo")
        pin_root = Path(str(self.config["code_pins"][pin_name]["repo_root"]))
        expected_entrypoint = (pin_root / entrypoint).resolve()
        if len(argv) < 2 or Path(argv[1]).resolve() != expected_entrypoint:
            raise ValueError(f"{step}: argv does not invoke the pinned entrypoint")
        child_log = Path(str(action.get("child_log", "")))
        if not child_log.is_absolute() or not _inside(child_log, self.restricted_root):
            raise ValueError(f"{step}: child_log must be inside restricted_root")
        outputs = action.get("outputs", [])
        if not isinstance(outputs, list) or not all(
            Path(str(path)).is_absolute() and _inside(Path(str(path)), self.restricted_root)
            for path in outputs
        ):
            raise ValueError(f"{step}: outputs must be absolute restricted paths")
        if len(outputs) != len(set(outputs)):
            raise ValueError(f"{step}: outputs must be unique")
        for binding in action.get("bound_inputs", []):
            if not isinstance(binding, dict):
                raise ValueError(f"{step}: bound input is not an object")
            path = Path(str(binding.get("path", "")))
            if not path.is_absolute() or not _inside(path, self.restricted_root):
                raise ValueError(f"{step}: bound input must be a restricted path")
            _hex(binding.get("sha256"), 64, f"{step} bound input digest")
            option = binding.get("option")
            env_key = binding.get("env_key")
            if (option is None) == (env_key is None):
                raise ValueError(
                    f"{step}: bound input needs exactly one option or env_key"
                )
            if option is not None:
                option = str(option)
                if not option.startswith("--") or Path(
                    _option(argv, option)
                ).resolve() != path.resolve():
                    raise ValueError(f"{step}: bound input option/path differs")
            else:
                env_key = str(env_key)
                environment = action.get("env")
                if (
                    not env_key
                    or not isinstance(environment, dict)
                    or environment.get(env_key) != str(path)
                ):
                    raise ValueError(f"{step}: bound input env_key/path differs")
        for binding in action.get("cross_step_inputs", []):
            if not isinstance(binding, dict):
                raise ValueError(f"{step}: cross-step input is not an object")
            producer = str(binding.get("producer_step", ""))
            if producer not in STEP_ORDER or STEP_ORDER.index(producer) >= STEP_ORDER.index(step):
                raise ValueError(f"{step}: invalid cross-step producer")
            path = Path(str(binding.get("path", "")))
            if str(path) not in self.config.get("actions", {}).get(producer, {}).get(
                "outputs", []
            ):
                raise ValueError(f"{step}: cross-step input is not a declared producer output")
            option = str(binding.get("option", ""))
            if not option.startswith("--") or Path(_option(argv, option)).resolve() != path.resolve():
                raise ValueError(f"{step}: cross-step input option/path differs")
        for binding in action.get("dynamic_sha256_inputs", []):
            if not isinstance(binding, dict):
                raise ValueError(f"{step}: dynamic SHA-256 input is not an object")
            producer = str(binding.get("producer_step", ""))
            path = Path(str(binding.get("path", "")))
            if producer not in STEP_ORDER or STEP_ORDER.index(producer) >= STEP_ORDER.index(step):
                raise ValueError(f"{step}: invalid dynamic SHA-256 producer")
            if str(path) not in self.config.get("actions", {}).get(producer, {}).get(
                "outputs", []
            ):
                raise ValueError(f"{step}: dynamic SHA-256 path is not a producer output")
            option = str(binding.get("option", ""))
            if not option.startswith("--") or _option(argv, option) != DYNAMIC_SHA256:
                raise ValueError(f"{step}: dynamic SHA-256 option lacks its sentinel")

        expected_script = {
            "reconcile_sweep": ("reconcile_guarded_orphans.py", "reconcile"),
            "resume_sweep": ("run_queue_guarded.py", None),
            "select_full": ("prepare_deep_jobs.py", "select-full"),
            "prepare_seed0": ("authorize_deep_jobs.py", "prepare-seed0"),
            "run_seed0": ("run_queue_guarded.py", None),
            "authorize_seed12": ("authorize_deep_jobs.py", "authorize-seed12"),
            "run_seed12": ("run_queue_guarded.py", None),
            "prepare_inference": ("prepare_stage3_inference.py", None),
            "run_inference": ("run_queue_guarded.py", None),
            "run_auxiliary": ("run_queue_guarded.py", None),
            "finalize": ("finalize_stage3_native.py", None),
        }[step]
        if expected_entrypoint.name != expected_script[0]:
            raise ValueError(f"{step}: wrong registered entrypoint")
        if expected_script[1] is not None and (
            len(argv) < 3 or argv[2] != expected_script[1]
        ):
            raise ValueError(f"{step}: wrong registered subcommand")

        if action["type"] == "guarded_queue":
            self._validate_runner_argv(step, action)
        elif step == "select_full":
            training_pin = str(self.config["role_pins"]["training"])
            expected_code = Path(
                str(self.config["code_pins"][training_pin]["repo_root"])
            ) / "segbench"
            if Path(_option(argv, "--code-root")).resolve() != expected_code.resolve():
                raise ValueError("select_full --code-root is not the training pin")
        elif step == "prepare_inference":
            delivery_pin = self.config["code_pins"][pin_name]
            if _option(argv, "--expected-planner-git-commit") != delivery_pin[
                "expected_commit"
            ]:
                raise ValueError("prepare_inference planner commit is not its code pin")
            inference_name = str(self.config["role_pins"]["inference"])
            inference_pin = self.config["code_pins"][inference_name]
            if _option(argv, "--inference-git-commit") != inference_pin[
                "expected_commit"
            ]:
                raise ValueError("prepare_inference inference commit is not pinned")
            expected_code = Path(str(inference_pin["repo_root"])) / "segbench"
            if Path(_option(argv, "--code-root")).resolve() != expected_code.resolve():
                raise ValueError("prepare_inference --code-root is not the inference pin")
            if _option(argv, "--finalizer-git-commit") != delivery_pin[
                "expected_commit"
            ]:
                raise ValueError("prepare_inference finalizer commit is not pinned")
            if float(_option(argv, "--max-project-ram-gib")) != float(
                self.config["resources"]["max_ram_gib"]
            ):
                raise ValueError("prepare_inference project RAM cap differs")
        elif step == "prepare_seed0":
            if float(_option(argv, "--max-project-ram-gib")) != float(
                self.config["resources"]["max_ram_gib"]
            ):
                raise ValueError("prepare_seed0 project RAM cap differs")
        elif step == "finalize":
            commit = self.config["code_pins"][pin_name]["expected_commit"]
            if _option(argv, "--expected-finalizer-git-commit") != commit:
                raise ValueError("finalizer commit is not its code pin")
            if _option(argv, "--expected-planner-git-commit") != commit:
                raise ValueError("finalizer planner commit is not its code pin")

    def _validate_runner_argv(self, step: str, action: Mapping[str, Any]) -> None:
        argv = action["argv"]
        resources = self.config["resources"]
        gpus = [int(value) for value in _option(argv, "--gpus").split(",")]
        if gpus != [int(value) for value in resources["allowed_gpus"]]:
            raise ValueError(f"{step}: GPU allow-list differs from configuration")
        max_per_gpu = int(_option(argv, "--max-per-gpu"))
        expected_max = 1 if step in {"run_inference", "run_auxiliary"} else int(
            resources["max_jobs_per_gpu"]
        )
        if max_per_gpu != expected_max:
            raise ValueError(f"{step}: max-per-gpu must be {expected_max}")
        requested_fraction = float(_option(argv, "--memory-fraction"))
        if (
            not math.isfinite(requested_fraction)
            or not 0.0 < requested_fraction <= float(resources["max_memory_fraction"])
        ):
            raise ValueError(f"{step}: memory fraction exceeds cap")
        if float(_option(argv, "--max-project-ram-gib")) != float(
            resources["max_ram_gib"]
        ):
            raise ValueError(f"{step}: project RAM cap differs")
        requested_disk_floor = float(_option(argv, "--min-disk-free-fraction"))
        if (
            not math.isfinite(requested_disk_floor)
            or not float(resources["min_disk_free_fraction"])
            <= requested_disk_floor
            < 1.0
        ):
            raise ValueError(f"{step}: disk floor is weaker than registered")
        queue = Path(_option(argv, "--jobs"))
        ledger = Path(_option(argv, "--state"))
        if queue.resolve() != Path(str(action.get("queue", ""))).resolve():
            raise ValueError(f"{step}: runner queue differs from action")
        if ledger.resolve() != Path(str(action.get("ledger", ""))).resolve():
            raise ValueError(f"{step}: runner ledger differs from action")
        if str(ledger) not in action.get("outputs", []):
            raise ValueError(f"{step}: runner ledger must be a recorded output")
        if step in {"run_inference", "run_auxiliary"} and max_per_gpu != 1:
            raise ValueError("held-out GPU evaluation must use one job per GPU")
        if step == "resume_sweep":
            sweep = self.config["sweep"]
            if queue.resolve() != Path(str(sweep["queue"])).resolve() or ledger.resolve() != Path(
                str(sweep["ledger"])
            ).resolve():
                raise ValueError("resume_sweep must use the frozen sweep queue/ledger")

    def _verify_code_pin(self, name: str) -> None:
        pin = self.config["code_pins"][name]
        root = Path(str(pin["repo_root"]))
        expected = str(pin["expected_commit"])
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
        ).strip()
        if observed != expected or dirty:
            raise SupervisorError(f"code pin {name} is not the clean expected worktree")

    def _disk_guard(self) -> None:
        root = Path(str(self.config["resources"]["resource_root"]))
        usage = shutil.disk_usage(root)
        free = usage.free / usage.total
        minimum = float(self.config["resources"]["min_disk_free_fraction"])
        if free < minimum:
            raise SupervisorError(
                f"disk guard: {free:.1%} free is below {minimum:.1%}"
            )

    def _verify_action_inputs(self, step: str, action: Mapping[str, Any]) -> None:
        """Rehash immutable inputs and outputs carried across stage barriers."""
        manifest_binding = self.config["bindings"]["manifest"]
        manifest = Path(str(manifest_binding["path"]))
        if not manifest.is_file() or sha256_file(manifest) != manifest_binding["sha256"]:
            raise SupervisorError("bound latent_source.csv changed")
        sweep_queue = Path(str(self.config["sweep"]["queue"]))
        if not sweep_queue.is_file() or sha256_file(sweep_queue) != self.config["sweep"][
            "queue_sha256"
        ]:
            raise SupervisorError("frozen sweep queue changed")
        for binding in action.get("bound_inputs", []):
            path = Path(str(binding["path"]))
            if not path.is_file() or sha256_file(path) != binding["sha256"]:
                raise SupervisorError(f"{step}: immutable bound input changed: {path}")
        for binding in action.get("cross_step_inputs", []):
            producer = str(binding["producer_step"])
            path = Path(str(binding["path"]))
            producer_evidence = self.state["steps"][producer].get("evidence", {})
            outputs = producer_evidence.get("outputs", {})
            expected = outputs.get(str(path)) if isinstance(outputs, dict) else None
            if expected is None:
                raise SupervisorError(
                    f"{step}: producer has no recorded digest for cross-step input {path}"
                )
            if not path.is_file() or sha256_file(path) != expected:
                raise SupervisorError(f"{step}: cross-step input changed: {path}")
        if step == "select_full":
            try:
                self._validate_selection_environment()
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise SupervisorError(
                    "select_full frozen environment changed before launch"
                ) from exc

    def _resolved_argv(self, step: str, action: Mapping[str, Any]) -> list[str]:
        """Resolve only state-bound SHA sentinels; never interpolate shell text."""
        argv = list(action["argv"])
        for binding in action.get("dynamic_sha256_inputs", []):
            producer = str(binding["producer_step"])
            path = Path(str(binding["path"]))
            expected = self.state["steps"][producer].get("evidence", {}).get(
                "outputs", {}
            ).get(str(path))
            if not isinstance(expected, str) or len(expected) != 64:
                raise SupervisorError(
                    f"{step}: producer digest unavailable for dynamic option"
                )
            option = str(binding["option"])
            index = argv.index(option) + 1
            if argv[index] != DYNAMIC_SHA256:
                raise SupervisorError(f"{step}: dynamic SHA sentinel changed")
            argv[index] = expected
        return argv

    def _mark_complete(self, step: str, evidence: Mapping[str, Any]) -> None:
        self.state["steps"][step].update(
            status="completed", completed_at_utc=_utc(), evidence=dict(evidence)
        )
        next_index = STEP_ORDER.index(step) + 1
        if next_index == len(STEP_ORDER):
            self.state["status"] = "completed"
            self.state["current_step"] = None
        else:
            self.state["current_step"] = STEP_ORDER[next_index]
        self._event("step_completed", step=step, evidence=dict(evidence))

    def _mark_failed(self, step: str, error: BaseException) -> None:
        message = f"{type(error).__name__}: {error}"
        self.state["status"] = "failed"
        self.state["current_step"] = step
        self.state["steps"][step].update(
            status="failed", failed_at_utc=_utc(), error=message
        )
        self._event("step_failed", step=step, error=message)

    def _verify_completed(self) -> None:
        for step in STEP_ORDER:
            record = self.state["steps"][step]
            if record.get("status") != "completed":
                continue
            evidence = record.get("evidence", {})
            output_hashes = evidence.get("outputs", {})
            if not isinstance(output_hashes, dict):
                raise SupervisorError(f"{step}: completed output evidence changed")
            for path_text, digest in output_hashes.items():
                path = Path(path_text)
                if not path.is_file() or sha256_file(path) != digest:
                    raise SupervisorError(f"{step}: completed output changed: {path}")
            if step == "run_auxiliary":
                current = self._auxiliary_evidence()
                if any(evidence.get(key) != value for key, value in current.items()):
                    raise SupervisorError(
                        "run_auxiliary: authenticated variance evidence changed"
                    )

    def _recover_running(self, step: str, action: Mapping[str, Any]) -> bool:
        """Wait for an orphaned child, then prove outputs or resume safely."""
        record: MutableMapping[str, Any] = self.state["steps"][step]
        if record.get("status") != "running":
            return False
        expected_process = record.get("process_identity")
        pid = record.get("pid")
        if isinstance(pid, int) and isinstance(expected_process, dict):
            while _live_process_identity(pid) == expected_process:
                if self.stop_requested:
                    raise SupervisorInterrupted("stop requested while adopting child")
                self._save()
                time.sleep(self.poll_seconds)
        record.pop("pid", None)
        record.pop("process_identity", None)
        # The former child is gone.  Accept it only when all declared evidence
        # validates.  Otherwise return the step to pending; guarded queues then
        # resume from their own immutable ledgers/checkpoints.
        try:
            if action["type"] == "guarded_queue":
                evidence = self._queue_evidence(step, action)
                evidence["outputs"] = self._output_evidence(action)
                if step == "run_auxiliary":
                    evidence.update(self._auxiliary_evidence())
            else:
                evidence = self._semantic_outputs(step, action)
        except (FileNotFoundError, SupervisorError, ValueError):
            record["status"] = "pending"
            self._event("orphan_not_complete", step=step)
            return False
        evidence["recovered_after_supervisor_restart"] = True
        self._mark_complete(step, evidence)
        return True

    def _output_evidence(self, action: Mapping[str, Any]) -> Dict[str, str]:
        result = {}
        for value in action.get("outputs", []):
            path = Path(str(value))
            if not path.is_file():
                raise SupervisorError(f"declared action output is missing: {path}")
            result[str(path)] = sha256_file(path)
        return result

    def _queue_evidence(self, step: str, action: Mapping[str, Any]) -> Dict[str, Any]:
        if step == "run_inference":
            expected = self.state["steps"]["prepare_inference"].get(
                "evidence", {}
            ).get("inference_expected_names")
        else:
            expected = action.get("expected_names")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{step}: expected_names is required")
        evidence = inspect_ledger(
            Path(str(action["queue"])),
            Path(str(action["ledger"])),
            [str(name) for name in expected],
            allow_terminal_failures=step == "run_seed12",
            require_timing_attestation=step in {"run_inference", "run_auxiliary"},
            allowed_gpus=self.config["resources"]["allowed_gpus"],
            max_host_ram_gib=float(self.config["resources"]["max_ram_gib"]),
        )
        if not evidence["terminal"]:
            raise SupervisorError(f"{step}: runner exited before its ledger was terminal")
        return evidence

    def _auxiliary_evidence(self) -> Dict[str, Any]:
        """Authenticate the GT-blind DRUnet variance set after its queue exits."""
        preparation = self.config["actions"]["prepare_inference"]
        plan_path = Path(_option(preparation["argv"], "--aux-plan-out"))
        card_path = Path(_option(preparation["argv"], "--aux-card-out"))
        plan = _read_json(plan_path, "DRUnet MC-dropout plan")
        card = _read_json(card_path, "DRUnet MC-dropout completion card")
        plan_claims = {
            "schema_version": 1,
            "kind": "stage3_drunet_mc_dropout_plan",
            "ground_truth_blind": True,
            "test_inputs_hashed": True,
            "test_images_decoded": False,
            "test_labels_opened": False,
            "timing_excluded": True,
            "mask_outputs_forbidden": True,
            "mc_dropout_passes": 20,
        }
        if any(plan.get(key) != value for key, value in plan_claims.items()):
            raise SupervisorError("DRUnet auxiliary plan claims changed after planning")
        image_ids = plan.get("test_image_ids")
        shapes = plan.get("expected_shapes")
        if (
            not isinstance(image_ids, list)
            or len(image_ids) != 20
            or not isinstance(shapes, dict)
            or set(shapes) != set(image_ids)
        ):
            raise SupervisorError("DRUnet auxiliary plan lacks its exact shape universe")
        run = plan.get("run")
        selected_seed = run.get("seed") if isinstance(run, dict) else None
        expected_card = {
            "schema_version": 1,
            "kind": "stage3_drunet_mc_dropout_completion_card",
            "method": "drunet",
            "seed": selected_seed,
            "plan_sha256": sha256_file(plan_path),
            "ground_truth_opened": False,
            "mask_outputs_written": False,
            "timing_excluded": True,
            "mc_dropout_passes": 20,
            "image_id_set_sha256": plan.get("image_id_set_sha256"),
            "ordered_source_images_sha256": plan.get(
                "ordered_source_images_sha256"
            ),
        }
        for key, value in expected_card.items():
            if card.get(key) != value:
                raise SupervisorError(f"DRUnet auxiliary card binding changed: {key}")
        provenance = card.get("provenance")
        inference_pin_name = str(self.config["role_pins"]["inference"])
        inference_commit = self.config["code_pins"][inference_pin_name][
            "expected_commit"
        ]
        if not isinstance(provenance, dict) or {
            "capture_timing": provenance.get("capture_timing"),
            "git_commit": provenance.get("git_commit"),
            "git_dirty": provenance.get("git_dirty"),
        } != {
            "capture_timing": "process_start",
            "git_commit": inference_commit,
            "git_dirty": False,
        }:
            raise SupervisorError("DRUnet auxiliary card lacks pinned start provenance")
        plan_identities = plan.get("identities")
        card_identities = card.get("identities")
        if not isinstance(plan_identities, dict) or not isinstance(card_identities, dict):
            raise SupervisorError("DRUnet auxiliary identity bindings are missing")
        for name in ("config", "checkpoint", "manifest", "splits"):
            if card_identities.get(name) != plan_identities.get(name):
                raise SupervisorError(f"DRUnet auxiliary {name} identity changed")

        variance_set = card.get("variance_set")
        records = variance_set.get("records") if isinstance(variance_set, dict) else None
        if (
            not isinstance(records, list)
            or variance_set.get("count") != 20
            or len(records) != 20
            or variance_set.get("sha256") != canonical_sha256(records)
            or [record.get("image_id") for record in records] != image_ids
        ):
            raise SupervisorError("DRUnet auxiliary variance-set identity changed")
        variance_root = Path(str(plan.get("variance_out", "")))
        if not variance_root.is_absolute() or not _inside(
            variance_root, self.restricted_root
        ):
            raise SupervisorError("DRUnet auxiliary variance root escaped restricted area")
        expected_paths: list[Path] = []
        for record in records:
            if not isinstance(record, dict):
                raise SupervisorError("DRUnet auxiliary variance record is not an object")
            filename = record.get("filename")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".npy")
            ):
                raise SupervisorError("DRUnet auxiliary variance filename is unsafe")
            path = variance_root / filename
            expected_paths.append(path)
            if (
                not path.is_file()
                or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != sha256_file(path)
            ):
                raise SupervisorError("DRUnet auxiliary variance file identity changed")
            expected_shape = shapes.get(str(record.get("image_id")))
            if (
                not isinstance(expected_shape, list)
                or len(expected_shape) != 2
                or any(type(value) is not int or value <= 0 for value in expected_shape)
                or record.get("shape") != expected_shape
                or record.get("dtype") != "float32"
            ):
                raise SupervisorError("DRUnet auxiliary variance shape contract changed")
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise SupervisorError(
                    f"DRUnet auxiliary variance array is unreadable: {path}"
                ) from exc
            if (
                list(array.shape) != expected_shape
                or array.dtype != np.dtype("float32")
                or not np.isfinite(array).all()
                or bool((array < 0).any())
            ):
                raise SupervisorError("DRUnet auxiliary variance array is invalid")
        observed_paths = sorted(
            path.resolve() for path in variance_root.rglob("*") if path.is_file()
        )
        if observed_paths != sorted(path.resolve() for path in expected_paths):
            raise SupervisorError("DRUnet auxiliary variance directory has extra files")
        reserved_masks = Path(str(plan.get("reserved_masks_root", "")))
        if not reserved_masks.is_absolute() or not _inside(
            reserved_masks, self.restricted_root
        ):
            raise SupervisorError("DRUnet auxiliary reserved mask root escaped")
        if reserved_masks.exists() and any(
            path.is_file() for path in reserved_masks.rglob("*")
        ):
            raise SupervisorError("DRUnet auxiliary job wrote forbidden mask files")
        return {
            "auxiliary_card_sha256": sha256_file(card_path),
            "auxiliary_plan_sha256": sha256_file(plan_path),
            "auxiliary_variance_set_sha256": variance_set["sha256"],
            "auxiliary_image_id_set_sha256": card["image_id_set_sha256"],
            "auxiliary_variance_count": 20,
            "auxiliary_timing_excluded": True,
        }

    def _semantic_outputs(self, step: str, action: Mapping[str, Any]) -> Dict[str, Any]:
        output_hashes = self._output_evidence(action)
        evidence: Dict[str, Any] = {"outputs": output_hashes}
        if step == "reconcile_sweep":
            queue = Path(_option(action["argv"], "--queue"))
            ledger = Path(_option(action["argv"], "--ledger"))
            capture = Path(_option(action["argv"], "--capture"))
            proof_path = Path(_option(action["argv"], "--proof-out"))
            proof = validate_reconciliation_proof(
                queue, ledger, capture, proof_path
            )
            evidence.update(
                reconciliation_proof_sha256=sha256_file(proof_path),
                reconciliation_capture_sha256=sha256_file(capture),
                reconciliation_ledger_sha256=sha256_file(ledger),
                reconciliation_workers=sorted(proof["workers"]),
            )
        elif step == "prepare_inference":
            gate = _read_json(Path(_option(action["argv"], "--gate-out")), "inference gate")
            plan_path = Path(_option(action["argv"], "--plan-out"))
            queue_path = Path(_option(action["argv"], "--queue-out"))
            plan = _read_json(plan_path, "inference plan")
            queue = _read_yaml(queue_path, "inference queue")
            if gate.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
                raise SupervisorError("Stage 3 >=2-seed readiness gate failed")
            prep_path = Path(_option(action["argv"], "--prep-bundle-out"))
            prep = _read_json(prep_path, "frozen inference preparation bundle")
            if gate.get("kind") != "stage3_native_training_readiness_gate":
                raise SupervisorError("inference gate kind changed")
            if plan.get("kind") != "stage3_native_inference_plan":
                raise SupervisorError("inference plan kind changed")
            if prep.get("kind") != "stage3_native_frozen_preparation_bundle":
                raise SupervisorError("inference preparation-bundle kind changed")
            runtime_policy = prep.get("runtime_policy")
            if not isinstance(runtime_policy, dict) or float(
                runtime_policy.get("max_project_ram_gib", 0.0)
            ) != float(self.config["resources"]["max_ram_gib"]):
                raise SupervisorError(
                    "inference preparation RAM cap differs from configuration"
                )
            if queue.get("phase") != "stage3_native_test_inference":
                raise SupervisorError("inference queue phase changed")
            for item, description in (
                (gate, "gate"),
                (plan, "plan"),
                (queue, "queue"),
                (prep, "preparation bundle"),
            ):
                if (
                    item.get("ground_truth_blind") is not True
                    or item.get("test_inputs_hashed") is not True
                    or item.get("test_images_decoded") is not False
                    or item.get("test_labels_opened") is not False
                ):
                    raise SupervisorError(f"inference {description} is not GT-blind")
            if queue.get("launch_authorized") is not True:
                raise SupervisorError("inference queue is not launch-authorized")
            if queue.get("required_max_jobs_per_gpu") != 1 or queue.get(
                "required_launcher_args"
            ) != ["--max-per-gpu", "1"]:
                raise SupervisorError("inference queue lacks timing-isolation policy")
            if queue.get("required_gpu_indices") != [
                int(value) for value in self.config["resources"]["allowed_gpus"]
            ]:
                raise SupervisorError("inference queue GPU policy differs from configuration")
            queue_jobs = queue.get("jobs")
            inference_names = [
                item.get("name") if isinstance(item, dict) else None
                for item in queue_jobs
            ] if isinstance(queue_jobs, list) else []
            if (
                not inference_names
                or any(not isinstance(name, str) or not name for name in inference_names)
                or len(inference_names) != len(set(inference_names))
            ):
                raise SupervisorError("inference queue job universe/order is invalid")
            for job in queue_jobs:
                if job.get("job_identity_sha256") != _queue_job_identity(job):
                    raise SupervisorError("inference queue job identity does not reproduce")
            gate_sha = sha256_file(Path(_option(action["argv"], "--gate-out")))
            plan_sha = sha256_file(plan_path)
            queue_sha = sha256_file(queue_path)
            if plan.get("gate_sha256") != gate_sha or queue.get("gate_sha256") != gate_sha:
                raise SupervisorError("inference artifacts do not bind the frozen gate")
            if queue.get("plan_sha256") != plan_sha:
                raise SupervisorError("inference queue does not bind the frozen plan")
            for name, path, digest in (
                ("gate", Path(_option(action["argv"], "--gate-out")), gate_sha),
                ("plan", plan_path, plan_sha),
                ("queue", queue_path, queue_sha),
            ):
                identity = prep.get(name)
                if not isinstance(identity, dict) or Path(
                    str(identity.get("path", ""))
                ).resolve() != path.resolve() or identity.get("sha256") != digest:
                    raise SupervisorError(f"preparation bundle does not bind {name}")

            aux_plan_path = Path(_option(action["argv"], "--aux-plan-out"))
            aux_queue_path = Path(_option(action["argv"], "--aux-queue-out"))
            aux_card_path = Path(_option(action["argv"], "--aux-card-out"))
            aux_plan = _read_json(aux_plan_path, "DRUnet MC-dropout plan")
            aux_queue = _read_yaml(aux_queue_path, "DRUnet MC-dropout queue")
            if aux_plan.get("kind") != "stage3_drunet_mc_dropout_plan":
                raise SupervisorError("DRUnet auxiliary plan kind changed")
            if aux_queue.get("phase") != "stage3_drunet_mc_dropout_aux":
                raise SupervisorError("DRUnet auxiliary queue phase changed")
            for item, description in (
                (aux_plan, "plan"),
                (aux_queue, "queue"),
            ):
                if (
                    item.get("ground_truth_blind") is not True
                    or item.get("test_inputs_hashed") is not True
                    or item.get("test_images_decoded") is not False
                    or item.get("test_labels_opened") is not False
                ):
                    raise SupervisorError(
                        f"DRUnet auxiliary {description} is not GT-blind at planning"
                    )
                if item.get("mc_dropout_passes") != 20:
                    raise SupervisorError(
                        f"DRUnet auxiliary {description} must register 20 passes"
                    )
                if item.get("timing_excluded") is not True:
                    raise SupervisorError(
                        f"DRUnet auxiliary {description} is not timing-excluded"
                    )
                if item.get("mask_outputs_forbidden") is not True:
                    raise SupervisorError(
                        f"DRUnet auxiliary {description} permits mask outputs"
                    )
            image_ids = aux_plan.get("test_image_ids")
            if (
                not isinstance(image_ids, list)
                or len(image_ids) != 20
                or not all(isinstance(value, str) and value for value in image_ids)
                or image_ids != sorted(image_ids)
                or len(set(image_ids)) != 20
            ):
                raise SupervisorError(
                    "DRUnet auxiliary plan must freeze 20 unique sorted test IDs"
                )
            if aux_plan.get("image_id_set_sha256") != canonical_sha256(image_ids):
                raise SupervisorError("DRUnet auxiliary image-ID digest does not reproduce")
            expected_shapes = aux_plan.get("expected_shapes")
            if (
                not isinstance(expected_shapes, dict)
                or set(expected_shapes) != set(image_ids)
                or any(
                    not isinstance(shape, list)
                    or len(shape) != 2
                    or any(type(value) is not int or value <= 0 for value in shape)
                    for shape in expected_shapes.values()
                )
            ):
                raise SupervisorError("DRUnet auxiliary expected-shape universe changed")
            _hex(
                aux_plan.get("ordered_source_images_sha256"),
                64,
                "DRUnet ordered source-image digest",
            )
            aux_jobs = aux_queue.get("jobs")
            if not isinstance(aux_jobs, list) or len(aux_jobs) != 1:
                raise SupervisorError("DRUnet auxiliary queue must contain exactly one job")
            if aux_queue.get("launch_authorized") is not True:
                raise SupervisorError("DRUnet auxiliary queue is not launch-authorized")
            if aux_queue.get("required_max_jobs_per_gpu") != 1 or aux_queue.get(
                "required_launcher_args"
            ) != ["--max-per-gpu", "1"]:
                raise SupervisorError("DRUnet auxiliary queue lacks isolation policy")
            if aux_queue.get("required_gpu_indices") != [
                int(value) for value in self.config["resources"]["allowed_gpus"]
            ]:
                raise SupervisorError(
                    "DRUnet auxiliary queue GPU policy differs from configuration"
                )
            aux_plan_sha = sha256_file(aux_plan_path)
            aux_queue_sha = sha256_file(aux_queue_path)
            if aux_queue.get("plan_sha256") != aux_plan_sha:
                raise SupervisorError("DRUnet auxiliary queue does not bind its plan")
            if Path(str(aux_jobs[0].get("completion_artifact", ""))).resolve() != (
                aux_card_path.resolve()
            ):
                raise SupervisorError("DRUnet auxiliary completion card path changed")
            expected_aux_names = self.config["actions"]["run_auxiliary"].get(
                "expected_names"
            )
            if [aux_jobs[0].get("name")] != expected_aux_names:
                raise SupervisorError("DRUnet auxiliary job identity/name changed")
            for name, path, digest in (
                ("aux_plan", aux_plan_path, aux_plan_sha),
                ("aux_queue", aux_queue_path, aux_queue_sha),
            ):
                identity = prep.get(name)
                if not isinstance(identity, dict) or Path(
                    str(identity.get("path", ""))
                ).resolve() != path.resolve() or identity.get("sha256") != digest:
                    raise SupervisorError(
                        f"preparation bundle does not bind {name}"
                    )
            evidence.update(
                gate_status=gate["status"],
                gate_sha256=gate_sha,
                plan_sha256=plan_sha,
                queue_sha256=queue_sha,
                prep_bundle_sha256=sha256_file(prep_path),
                auxiliary_plan_sha256=aux_plan_sha,
                auxiliary_queue_sha256=aux_queue_sha,
                auxiliary_image_id_set_sha256=aux_plan["image_id_set_sha256"],
                inference_expected_names=inference_names,
            )
        elif step == "finalize":
            delivery_path = Path(_option(action["argv"], "--bundle-out")) / "delivery.json"
            delivery = _read_json(delivery_path, "Stage 3 delivery")
            if delivery.get("kind") != "stage3_native_heldout_delivery":
                raise SupervisorError("finalizer did not publish the registered delivery")
            auxiliary = delivery.get("auxiliary")
            preparation = self.config["actions"]["prepare_inference"]
            auxiliary_action = self.config["actions"]["run_auxiliary"]
            auxiliary_state = self.state["steps"]["run_auxiliary"].get("evidence", {})
            expected_auxiliary = {
                "plan_sha256": sha256_file(
                    Path(_option(preparation["argv"], "--aux-plan-out"))
                ),
                "queue_sha256": sha256_file(Path(str(auxiliary_action["queue"]))),
                "ledger_sha256": sha256_file(Path(str(auxiliary_action["ledger"]))),
                "completion_card_sha256": auxiliary_state.get(
                    "auxiliary_card_sha256"
                ),
                "variance_set_sha256": auxiliary_state.get(
                    "auxiliary_variance_set_sha256"
                ),
                "image_id_set_sha256": auxiliary_state.get(
                    "auxiliary_image_id_set_sha256"
                ),
                "mc_dropout_passes": 20,
                "timing_excluded": True,
                "scoring_excluded": True,
            }
            if not isinstance(auxiliary, dict) or any(
                auxiliary.get(key) != value for key, value in expected_auxiliary.items()
            ):
                raise SupervisorError("final delivery auxiliary binding changed")
            evidence["delivery_sha256"] = sha256_file(delivery_path)
        return evidence

    def _launch(self, step: str, action: Mapping[str, Any]) -> int:
        self._verify_code_pin(str(action["code_pin"]))
        self._disk_guard()
        self._verify_action_inputs(step, action)
        step_record: MutableMapping[str, Any] = self.state["steps"][step]
        step_record["attempts"] = int(step_record.get("attempts", 0)) + 1
        step_record.update(status="running", started_at_utc=_utc())
        child_log = Path(str(action["child_log"]))
        child_log.parent.mkdir(parents=True, exist_ok=True)
        handle = child_log.open("a", encoding="utf-8", buffering=1)
        handle.write(f"\n=== supervisor launch {_utc()} step={step} ===\n")
        argv = self._resolved_argv(step, action)
        handle.write("argv=" + json.dumps(argv) + "\n")
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in action.get("env", {}).items()})
        env["MAX_CPU_CORES"] = str(self.config["resources"]["max_cpu_cores"])
        env["MAX_RAM_GB"] = str(self.config["resources"]["max_ram_gib"])

        self.child = subprocess.Popen(
            argv,
            cwd=Path(str(self.config["code_pins"][action["code_pin"]]["repo_root"])),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            preexec_fn=_set_child_priority,
        )
        step_record["pid"] = self.child.pid
        identity = _live_process_identity(self.child.pid)
        if identity is None:
            self.child.terminate()
            raise SupervisorError(f"{step}: could not attest launched process identity")
        step_record["process_identity"] = identity
        self._event("child_started", step=step, pid=self.child.pid)
        while self.child.poll() is None:
            self._save()
            if self.stop_requested:
                self.child.terminate()
            time.sleep(self.poll_seconds)
        code = int(self.child.returncode)
        handle.write(f"supervisor: child exited with code {code}\n")
        handle.close()
        step_record["exit_code"] = code
        step_record.pop("pid", None)
        step_record.pop("process_identity", None)
        self.child = None
        self._event("child_exited", step=step, exit_code=code)
        return code

    def advance_once(self) -> str:
        """Advance at most one state-machine step."""
        self._verify_completed()
        if self.state.get("status") == "completed":
            return "completed"
        if self.state.get("status") == "failed":
            raise SupervisorError("supervisor is in a failed state; audit is required")
        step = next(
            name
            for name in STEP_ORDER
            if self.state["steps"][name].get("status") != "completed"
        )
        self.state["current_step"] = step
        try:
            action = self.config["actions"][step]
            if self._recover_running(step, action):
                return "completed" if self.state["status"] == "completed" else "advanced"
            code = self._launch(step, action)
            if self.stop_requested:
                self.state["steps"][step]["status"] = "pending"
                self.state["status"] = "running"
                self._event("supervisor_interrupted", step=step, child_exit_code=code)
                raise SupervisorInterrupted("resumable stop requested")
            if action["type"] == "guarded_queue":
                evidence = self._queue_evidence(step, action)
                if code != 0 and not (step == "run_seed12" and evidence["failed"]):
                    raise SupervisorError(f"{step} exited with code {code}")
                evidence["outputs"] = self._output_evidence(action)
                if step == "run_auxiliary":
                    evidence.update(self._auxiliary_evidence())
            else:
                if code != 0:
                    raise SupervisorError(f"{step} exited with code {code}")
                evidence = self._semantic_outputs(step, action)
            self._mark_complete(step, evidence)
            return "completed" if self.state["status"] == "completed" else "advanced"
        except SupervisorInterrupted:
            raise
        except BaseException as exc:
            self._mark_failed(step, exc)
            raise

    def run(self, *, once: bool = False) -> str:
        """Run until delivery, failure, or one requested transition."""
        while True:
            if self.stop_requested:
                raise SupervisorInterrupted("resumable stop requested")
            result = self.advance_once()
            if result == "completed" or once:
                return result
            if result == "waiting":
                time.sleep(self.poll_seconds)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate config and clean code pins without launching any action",
    )
    args = parser.parse_args(argv)
    if args.poll_seconds < 1.0:
        parser.error("--poll-seconds must be at least one second")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    supervisor = Supervisor(
        args.config, args.state, args.log, poll_seconds=args.poll_seconds
    )
    if args.validate_only:
        for pin_name in supervisor.config["code_pins"]:
            supervisor._verify_code_pin(str(pin_name))
        print(
            json.dumps(
                {
                    "status": "validated",
                    "config": str(args.config),
                    "config_sha256": supervisor.config_sha256,
                    "state": str(args.state),
                },
                sort_keys=True,
            )
        )
        return 0

    def stop(signum: int, frame: object) -> None:  # noqa: ARG001
        supervisor.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        result = supervisor.run(once=args.once)
    except SupervisorInterrupted as exc:
        print(f"Stage 3 supervisor paused: {exc}", file=sys.stderr)
        return 75
    except BaseException as exc:
        print(f"Stage 3 supervisor stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result, "state": str(args.state)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
