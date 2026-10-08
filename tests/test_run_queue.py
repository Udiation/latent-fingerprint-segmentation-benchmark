"""CPU-only safety tests for the shared-machine GPU queue."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_queue as queue  # noqa: E402


def test_job_names_cannot_escape_log_directory(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.yaml"
    jobs.write_text(
        "jobs:\n  - name: ../escape\n    argv: [python, train.py]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        queue.load_jobs(jobs)


def test_training_jobs_must_be_resumable(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.yaml"
    jobs.write_text(
        "jobs:\n"
        "  - name: train\n"
        "    argv: [python, scripts/train_seg.py, --config, config.yaml]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must declare --resume"):
        queue.load_jobs(jobs)

    jobs.write_text(
        "jobs:\n"
        "  - name: train-sufs\n"
        "    argv: [python, scripts/train_sufs.py, --config, config.yaml]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must declare --resume"):
        queue.load_jobs(jobs)


def test_sibling_jobs_are_owned_not_foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    first = queue.Running(
        queue.Job("first", ["python", "a.py"]),
        0,
        SimpleNamespace(pid=101),
        SimpleNamespace(),
        Path("first.log"),
        Path("first.preempt"),
    )
    second = queue.Running(
        queue.Job("second", ["python", "b.py"]),
        0,
        SimpleNamespace(pid=202),
        SimpleNamespace(),
        Path("second.log"),
        Path("second.preempt"),
    )
    process_trees = {101: {101, 111}, 202: {202, 222}}
    monkeypatch.setattr(queue, "descendants", lambda pid: process_trees[pid])
    owned = queue.owned_pids_by_gpu([first, second], [0, 1])
    devices = {0: ("gpu-zero", 100_000), 1: ("gpu-one", 100_000)}
    own_processes = {"gpu-zero": [(111, 10_000), (222, 10_000)]}
    assert not queue.foreign_heavy_process(
        0, devices, own_processes, owned[0], threshold_mib=2_048
    )
    with_foreign = {"gpu-zero": own_processes["gpu-zero"] + [(999, 2_049)]}
    assert queue.foreign_heavy_process(
        0, devices, with_foreign, owned[0], threshold_mib=2_048
    )


def test_queue_rejects_more_than_three_jobs_per_gpu(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        queue.parse_args(
            [
                "--jobs",
                str(tmp_path / "jobs.yaml"),
                "--results",
                str(tmp_path),
                "--gpus",
                "0,1",
                "--max-per-gpu",
                "4",
            ]
        )


def test_launch_sets_memory_retry_and_thread_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class DummyProcess:
        pid = 123

    def fake_popen(argv: list[str], **kwargs: object) -> DummyProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        return DummyProcess()

    monkeypatch.setattr(queue.subprocess, "Popen", fake_popen)
    priority_calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        queue.os,
        "setpriority",
        lambda which, who, priority: priority_calls.append((which, who, priority)),
    )
    job = queue.Job("retry", ["python", "train.py"], oom_retries=1)
    running = queue.launch(job, 0, tmp_path, 0.3)
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["SEG_GPU_MEMORY_FRACTION"] == "0.3"
    assert environment["SEG_BATCH_SCALE"] == "0.5"
    assert environment["SEG_GRAD_ACCUM_SCALE"] == "2.0"
    assert environment["SEG_OOM_RETRY_COUNT"] == "1"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        assert environment[key] == "1"
    preexec = captured["preexec_fn"]
    assert callable(preexec)
    preexec()
    assert priority_calls == [(queue.os.PRIO_PROCESS, 0, 10)]
    running.log_handle.close()


def test_queue_batch_environment_rejects_arbitrary_retry_count() -> None:
    with pytest.raises(ValueError, match="at most one"):
        queue.queue_batch_environment(
            queue.Job("invalid", ["python", "train.py"], oom_retries=2)
        )


def test_gpu_telemetry_preflight_fails_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(queue.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="nvidia-ml-py"):
        queue.require_gpu_telemetry_dependency()


def test_stop_between_pop_and_launch_never_spawns(tmp_path: Path) -> None:
    pending = [queue.Job("one", ["python", "train.py"])]
    checks = iter((False, True))
    launched = []

    result = queue.launch_next_unless_stopping(
        pending,
        gpu=0,
        log_dir=tmp_path,
        memory_fraction=0.3,
        stop_requested=lambda: next(checks),
        launcher=lambda *args: launched.append(args),
    )
    assert result is None
    assert not launched
    assert [job.name for job in pending] == ["one"]


def test_oom_detection_is_scoped_to_current_launch(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    old = "CUDA out of memory in an earlier launch\n"
    log.write_text(old + "ordinary failure in this launch\n", encoding="utf-8")
    assert queue.tail_contains_oom(log)
    assert not queue.tail_contains_oom(log, start_offset=len(old.encode("utf-8")))
