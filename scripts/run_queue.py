#!/usr/bin/env python3
"""Resource-aware GPU job queue for reproducible segmentation experiments.

The queue never chooses devices implicitly. Pass the permitted physical GPU
indices with ``--gpus`` (or ``SEGBENCH_GPUS``). Jobs are JSON/YAML objects with
``name`` and an argument-vector ``argv``; shell command strings are rejected.
Training programs cooperate through ``SEG_PREEMPT_FILE`` and exit code 75.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import yaml


PREEMPT_EXIT = 75
OOM_PATTERN = re.compile(r"(?:CUDA\s+out\s+of\s+memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED)", re.I)
SAFE_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def require_gpu_telemetry_dependency() -> None:
    """Fail before ledger creation when epoch-level GPU telemetry cannot run."""
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception as exc:
        raise RuntimeError(
            "GPU training requires the nvidia-ml-py package and a working NVML "
            "runtime; install the pinned project requirements before launch"
        ) from exc


@dataclass
class Job:
    """A queue entry loaded from a declarative job file."""

    name: str
    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    oom_retries: int = 0
    preemptions: int = 0


@dataclass
class Running:
    """Runtime state for a launched job."""

    job: Job
    gpu: int
    process: subprocess.Popen
    log_handle: object
    log_path: Path
    preempt_path: Path
    log_start: int = 0
    preempt_requested: bool = False


def load_jobs(path: Path) -> List[Job]:
    """Load and validate a YAML or JSON list of jobs."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, dict):
        data = data.get("jobs")
    if not isinstance(data, list) or not data:
        raise ValueError("job file must contain a non-empty list (or {jobs: [...]})")
    jobs: List[Job] = []
    names: Set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"job {index} is not an object")
        name, argv = item.get("name"), item.get("argv")
        if (
            not isinstance(name, str)
            or not SAFE_JOB_NAME.fullmatch(name)
            or name in {".", ".."}
            or name in names
        ):
            raise ValueError(f"job {index} has an empty or duplicate name")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError(f"job {name!r} must provide a non-empty argv string list")
        training_entrypoints = {"train_seg.py", "train_patch.py", "train_sufs.py"}
        if any(Path(argument).name in training_entrypoints for argument in argv):
            if "--resume" not in argv:
                raise ValueError(
                    f"training job {name!r} must declare --resume (normally 'auto')"
                )
            resume_index = argv.index("--resume")
            if resume_index + 1 >= len(argv) or argv[resume_index + 1] == "none":
                raise ValueError(
                    f"training job {name!r} must use a resumable --resume value"
                )
        env = item.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, (str, int, float)) for k, v in env.items()):
            raise ValueError(f"job {name!r} env must be a scalar mapping")
        jobs.append(Job(name=name, argv=list(argv), env={k: str(v) for k, v in env.items()}))
        names.add(name)
    return jobs


def parse_gpu_list(value: str) -> List[int]:
    """Parse a comma-separated, duplicate-free physical GPU allow-list."""
    try:
        gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid --gpus value {value!r}") from exc
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain unique non-negative indices")
    return gpus


def _run_nvidia_smi(query: str) -> List[List[str]]:
    command = [
        "nvidia-smi",
        f"--query-{query.split(':', 1)[0]}={query.split(':', 1)[1]}",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    return [[part.strip() for part in line.split(",")] for line in output.splitlines() if line.strip()]


def gpu_snapshot() -> Tuple[Dict[int, Tuple[str, int]], Dict[str, List[Tuple[int, int]]]]:
    """Return {index: (uuid, free MiB)} and {uuid: [(pid, used MiB)]}."""
    devices: Dict[int, Tuple[str, int]] = {}
    for index, uuid, free in _run_nvidia_smi("gpu:index,uuid,memory.free"):
        devices[int(index)] = (uuid, int(float(free)))
    processes: Dict[str, List[Tuple[int, int]]] = {}
    try:
        rows = _run_nvidia_smi("compute-apps:gpu_uuid,pid,used_memory")
    except subprocess.CalledProcessError:
        rows = []
    for uuid, pid, used in rows:
        processes.setdefault(uuid, []).append((int(pid), int(float(used))))
    return devices, processes


def descendants(root_pid: int) -> Set[int]:
    """Find a process and all descendants from procfs without extra packages."""
    parents: Dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            parents[int(fields[0])] = int(fields[3])
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    owned = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in owned and pid not in owned:
                owned.add(pid)
                changed = True
    return owned


def owned_pids_by_gpu(running: Sequence[Running], gpus: Iterable[int]) -> Dict[int, Set[int]]:
    """Return the union of every queue-owned process tree on each GPU.

    Ownership must be computed across sibling jobs.  Otherwise, when multiple
    queue jobs share a GPU, each job's CUDA process is incorrectly classified
    as foreign by every other job and all of them preempt one another.
    """
    owned: Dict[int, Set[int]] = {gpu: set() for gpu in gpus}
    for item in running:
        owned.setdefault(item.gpu, set()).update(descendants(item.process.pid))
    return owned


def foreign_heavy_process(
    gpu: int,
    devices: Mapping[int, Tuple[str, int]],
    processes: Mapping[str, List[Tuple[int, int]]],
    owned_pids: Set[int],
    threshold_mib: int,
) -> bool:
    """Return true when an unowned process exceeds the declared memory limit."""
    if gpu not in devices:
        raise RuntimeError(f"GPU index {gpu} is not reported by nvidia-smi")
    uuid, _ = devices[gpu]
    return any(pid not in owned_pids and used > threshold_mib for pid, used in processes.get(uuid, []))


def disk_guard(path: Path, minimum_free_fraction: float) -> None:
    usage = shutil.disk_usage(path)
    free_fraction = usage.free / usage.total
    if free_fraction < minimum_free_fraction:
        raise RuntimeError(
            f"disk guard: {free_fraction:.1%} free is below {minimum_free_fraction:.1%}"
        )


def queue_batch_environment(job: Job) -> Dict[str, str]:
    """Return the only launcher batch states permitted by the queue protocol."""
    if job.oom_retries == 0:
        return {
            "SEG_OOM_RETRY_COUNT": "0",
            "SEG_BATCH_SCALE": "1.0",
            "SEG_GRAD_ACCUM_SCALE": "1.0",
        }
    if job.oom_retries == 1:
        return {
            "SEG_OOM_RETRY_COUNT": "1",
            "SEG_BATCH_SCALE": "0.5",
            "SEG_GRAD_ACCUM_SCALE": "2.0",
        }
    raise ValueError("queue permits at most one OOM batch transition")


def launch(job: Job, gpu: int, log_dir: Path, memory_fraction: float) -> Running:
    """Launch one argv-safe job with the physical device hidden behind local index 0."""
    log_path = log_dir / f"{job.name}.log"
    preempt_path = log_dir / f".{job.name}.preempt"
    preempt_path.unlink(missing_ok=True)
    log_start = log_path.stat().st_size if log_path.exists() else 0
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    handle.write(
        f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S%z')} gpu={gpu} "
        f"oom_retry={job.oom_retries} preemptions={job.preemptions} ===\n"
    )
    handle.write("argv=" + json.dumps(job.argv) + "\n")
    env: MutableMapping[str, str] = os.environ.copy()
    env.update(job.env)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "SEG_GPU_MEMORY_FRACTION": str(memory_fraction),
            "SEG_PREEMPT_FILE": str(preempt_path),
            **queue_batch_environment(job),
            # Keep library thread pools bounded. DataLoader workers are declared
            # separately in each scientific config and the queue may run six
            # processes at once.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OPENCV_FOR_THREADS_NUM": "1",
        }
    )

    def lower_priority() -> None:
        # Set an absolute value: the guarded controller itself may already run
        # under nice, and incremental os.nice(10) would compound to nice 19.
        os.setpriority(os.PRIO_PROCESS, 0, 10)

    process = subprocess.Popen(
        job.argv,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        preexec_fn=lower_priority,
    )
    return Running(job, gpu, process, handle, log_path, preempt_path, log_start=log_start)


def launch_next_unless_stopping(
    pending: List[Job],
    *,
    gpu: int,
    log_dir: Path,
    memory_fraction: float,
    stop_requested: Callable[[], bool],
    launcher: Callable[[Job, int, Path, float], Running] = launch,
) -> Optional[Running]:
    """Pop and launch atomically from the controller's stop-policy perspective."""
    if stop_requested() or not pending:
        return None
    job = pending.pop(0)
    if stop_requested():
        pending.insert(0, job)
        return None
    try:
        launched = launcher(job, gpu, log_dir, memory_fraction)
    except BaseException:
        pending.insert(0, job)
        raise
    if stop_requested():
        launched.preempt_path.touch()
        launched.preempt_requested = True
    return launched


def tail_contains_oom(
    path: Path, *, start_offset: int = 0, max_bytes: int = 1_000_000
) -> bool:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(start_offset, end - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return False
    return OOM_PATTERN.search(text) is not None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--gpus", default=os.environ.get("SEGBENCH_GPUS", ""))
    parser.add_argument("--max-per-gpu", type=int, default=3)
    parser.add_argument("--min-free-mib", type=int, default=40 * 1024)
    parser.add_argument("--foreign-process-mib", type=int, default=2 * 1024)
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--min-disk-free-fraction", type=float, default=0.15)
    args = parser.parse_args(argv)
    if not args.gpus:
        parser.error("--gpus or SEGBENCH_GPUS is required; devices are never inferred")
    args.gpus = parse_gpu_list(args.gpus)
    if args.max_per_gpu < 1 or args.max_per_gpu > 3:
        parser.error("--max-per-gpu must be in 1..3")
    if not 0 < args.memory_fraction <= 0.3:
        parser.error("--memory-fraction must be in (0, 0.3]")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")
    if args.min_free_mib < 1 or args.foreign_process_mib < 0:
        parser.error("GPU memory thresholds must be non-negative")
    if not 0 < args.min_disk_free_fraction < 1:
        parser.error("--min-disk-free-fraction must be in (0, 1)")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    require_gpu_telemetry_dependency()
    jobs = load_jobs(args.jobs)
    log_dir = args.results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    disk_guard(args.results, args.min_disk_free_fraction)
    pending = list(jobs)
    running: List[Running] = []
    failed: List[str] = []
    stop_requested = False

    def request_stop(signum, frame) -> None:  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True
        for item in running:
            item.preempt_path.touch()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while pending or running:
        devices, processes = gpu_snapshot()
        owned_by_gpu = owned_pids_by_gpu(running, args.gpus)
        finished: List[Running] = []
        for item in running:
            code = item.process.poll()
            if code is None:
                if foreign_heavy_process(
                    item.gpu,
                    devices,
                    processes,
                    owned_by_gpu[item.gpu],
                    args.foreign_process_mib,
                ) and not item.preempt_requested:
                    item.preempt_path.touch()
                    item.preempt_requested = True
                    item.log_handle.write("queue: foreign process detected; checkpoint requested\n")
                continue
            item.log_handle.write(f"queue: process exited with code {code}\n")
            item.log_handle.close()
            item.preempt_path.unlink(missing_ok=True)
            if code == 0:
                pass
            elif code == PREEMPT_EXIT:
                item.job.preemptions += 1
                pending.insert(0, item.job)
            elif item.job.oom_retries == 0 and tail_contains_oom(
                item.log_path, start_offset=item.log_start
            ):
                item.job.oom_retries = 1
                pending.insert(0, item.job)
            else:
                failed.append(item.job.name)
            finished.append(item)
        running = [item for item in running if item not in finished]

        if stop_requested:
            for item in running:
                item.preempt_path.touch()
                item.preempt_requested = True
            if not running:
                break
            time.sleep(args.poll_seconds)
            continue

        for gpu in args.gpus:
            if not pending:
                break
            slots = args.max_per_gpu - sum(item.gpu == gpu for item in running)
            if slots <= 0:
                continue
            for _ in range(slots):
                if stop_requested or not pending:
                    break
                # Re-query immediately before every launch.  A single snapshot
                # cannot safely authorize several sibling launches because GPU
                # memory or a foreign process may change after the first one.
                devices, processes = gpu_snapshot()
                owned_by_gpu = owned_pids_by_gpu(running, args.gpus)
                if gpu not in devices:
                    break
                _, free_mib = devices[gpu]
                if free_mib < args.min_free_mib:
                    break
                if foreign_heavy_process(
                    gpu,
                    devices,
                    processes,
                    owned_by_gpu[gpu],
                    args.foreign_process_mib,
                ):
                    break
                if stop_requested:
                    break
                disk_guard(args.results, args.min_disk_free_fraction)
                if stop_requested:
                    break
                launched = launch_next_unless_stopping(
                    pending,
                    gpu=gpu,
                    log_dir=log_dir,
                    memory_fraction=args.memory_fraction,
                    stop_requested=lambda: stop_requested,
                )
                if launched is None:
                    break
                running.append(launched)
                if stop_requested:
                    launched.preempt_path.touch()
                    launched.preempt_requested = True
                    break

        if pending or running:
            time.sleep(args.poll_seconds)

    if failed:
        print("failed jobs: " + ", ".join(failed), file=sys.stderr)
        return 1
    return PREEMPT_EXIT if stop_requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
