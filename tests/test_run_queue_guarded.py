"""CPU-only tests of the Stage-3 persistent queue guards."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts" / "run_queue_guarded.py"
    spec = importlib.util.spec_from_file_location("guarded_queue_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _queue(tmp_path: Path) -> tuple[Path, Path]:
    module = _module()
    completion = tmp_path / "run.json"
    raw = {
        "name": "full.unet.seed0",
        "argv": [
            "python", "train_seg.py", "--resume", "auto", "--out", str(tmp_path)
        ],
        "env": {},
        "completion_artifact": str(completion),
        "resources": {"host_ram_gib": 8.0},
    }
    raw["job_identity_sha256"] = module._declared_identity(raw)
    path = tmp_path / "queue.yaml"
    path.write_text(
        yaml.safe_dump({"launch_authorized": True, "jobs": [raw]}),
        encoding="utf-8",
    )
    return path, completion


def test_completed_ledger_skips_only_hash_identical_artifact(tmp_path: Path) -> None:
    module = _module()
    queue_path, completion = _queue(tmp_path)
    jobs = module.load_guarded_jobs(queue_path)
    state_path = tmp_path / "state.json"
    state = module.load_state(state_path, queue_path, jobs)
    completion.write_text("complete", encoding="utf-8")
    module.complete_job(state, jobs[0])
    module.atomic_json(state_path, state)
    resumed = module.load_state(state_path, queue_path, jobs)
    assert resumed["jobs"][jobs[0].name]["status"] == "completed"
    completion.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="completed artifact changed"):
        module.load_state(state_path, queue_path, jobs)


def test_queue_change_after_ledger_is_rejected(tmp_path: Path) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    jobs = module.load_guarded_jobs(queue_path)
    state_path = tmp_path / "state.json"
    module.load_state(state_path, queue_path, jobs)
    payload = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    payload["extra"] = True
    queue_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="queue file changed"):
        module.load_state(state_path, queue_path, jobs)


def test_one_ordinary_retry_and_one_oom_retry_are_selective(tmp_path: Path) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    job = module.load_guarded_jobs(queue_path)[0]
    assert module.retry_action(1, job, oom=False) == "retry_failure"
    job.ordinary_retries = 1
    assert module.retry_action(1, job, oom=False) == "fail"
    job.ordinary_retries = 0
    assert module.retry_action(76, job, oom=True) == "retry_oom"
    job.base_job.oom_retries = 1
    assert module.retry_action(76, job, oom=True) == "fail"


def test_stop_after_ledger_transition_rolls_back_without_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    job = module.load_guarded_jobs(queue_path)[0]
    pending = [job]
    state = module.initial_state(queue_path, [job])
    stopped = False
    writes = 0
    launched = []

    def write_state(path: Path, payload: dict) -> None:
        nonlocal stopped, writes
        writes += 1
        if writes == 1:
            stopped = True

    monkeypatch.setattr(module, "atomic_json", write_state)
    result = module.launch_candidate_transaction(
        candidate=job,
        gpu=0,
        pending=pending,
        state=state,
        state_path=tmp_path / "state.json",
        log_dir=tmp_path,
        memory_fraction=0.3,
        stop_requested=lambda: stopped,
        launcher=lambda *args: launched.append(args),
    )
    assert result is None
    assert not launched
    assert pending == [job]
    assert state["jobs"][job.name]["status"] == "pending"
    assert state["jobs"][job.name]["attempts"] == 0
    assert writes == 2


def test_ledger_rejects_unbound_or_repeated_oom_override(tmp_path: Path) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    jobs = module.load_guarded_jobs(queue_path)
    ledger = tmp_path / "ledger.json"
    state = module.initial_state(queue_path, jobs)
    entry = state["jobs"][jobs[0].name]
    entry["oom_retries"] = 1
    module.atomic_json(ledger, state)
    with pytest.raises(ValueError, match="transition evidence"):
        module.load_state(ledger, queue_path, jobs)

    entry["oom_batch_transition"] = {
        "from_oom_retry_count": 0,
        "to_oom_retry_count": 1,
        "batch_scale": 0.5,
        "gradient_accumulation_scale": 2.0,
        "effective_batch_preserved": True,
        "launcher_environment": {
            "SEG_OOM_RETRY_COUNT": "1",
            "SEG_BATCH_SCALE": "0.5",
            "SEG_GRAD_ACCUM_SCALE": "2.0",
        },
        "policy_sha256": module.canonical_sha256(module.OOM_BATCH_POLICY),
    }
    module.atomic_json(ledger, state)
    loaded = module.load_state(ledger, queue_path, jobs)
    assert loaded["jobs"][jobs[0].name]["oom_retries"] == 1

    loaded["jobs"][jobs[0].name]["oom_retries"] = 2
    module.atomic_json(ledger, loaded)
    with pytest.raises(ValueError, match="single OOM retry"):
        module.load_state(ledger, queue_path, jobs)


def test_reservation_guard_enforces_runtime_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    job = module.load_guarded_jobs(queue_path)[0]
    job.host_ram_gib = 12.0
    running = [
            module.GuardedRunning(
            module.GuardedJob(
                job.base_job,
                job.completion_artifact,
                50.0,
                0,
                job.identity_sha256,
            ),
            object(),
        )
    ]
    monkeypatch.setattr(module, "owned_ram_gib", lambda _: 1.0)
    monkeypatch.setattr(module, "available_ram_gib", lambda: 1000.0)
    assert not module.can_reserve(
        job, running, max_project_ram_gib=60.0, min_available_ram_gib=0.0
    )


def test_external_reservations_count_toward_runtime_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    job = module.load_guarded_jobs(queue_path)[0]
    job.host_ram_gib = 10.0
    monkeypatch.setattr(module, "owned_ram_gib", lambda _: 0.5)
    monkeypatch.setattr(module, "available_ram_gib", lambda: 1000.0)
    assert module.can_reserve(
        job,
        [],
        max_project_ram_gib=64.0,
        min_available_ram_gib=8.0,
        external_reserved_gib=50.0,
        external_owned_gib=3.0,
    )
    assert not module.can_reserve(
        job,
        [],
        max_project_ram_gib=64.0,
        min_available_ram_gib=8.0,
        external_reserved_gib=60.0,
        external_owned_gib=3.0,
    )


def test_cpu_worker_reservations_enforce_runtime_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    job = module.load_guarded_jobs(queue_path)[0]
    job.cpu_workers = 8
    monkeypatch.setattr(module, "owned_ram_gib", lambda _: 0.0)
    monkeypatch.setattr(module, "available_ram_gib", lambda: 1000.0)
    assert module.can_reserve(
        job,
        [],
        max_project_ram_gib=64.0,
        min_available_ram_gib=0.0,
        max_cpu_cores=40,
        external_cpu_workers=24,
    )
    assert not module.can_reserve(
        job,
        [],
        max_project_ram_gib=64.0,
        min_available_ram_gib=0.0,
        max_cpu_cores=40,
        external_cpu_workers=40,
    )


def test_adopted_worker_tree_is_owned_during_every_foreign_process_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    worker = module.ExternalWorker(
        name="captured",
        pid=4242,
        start_ticks=7,
        argv_sha256="a" * 64,
        gpu=0,
        completion_artifact=tmp_path / "run.json",
        preempt_path=tmp_path / "preempt",
        host_ram_gib=12.0,
        cpu_workers=4,
    )
    monkeypatch.setattr(module.base, "owned_pids_by_gpu", lambda running, gpus: {0: set(), 2: set()})
    monkeypatch.setattr(module.base, "descendants", lambda pid: {pid, pid + 1})
    owned = module.owned_pids_including_external([], [worker], [0, 2])
    assert owned == {0: {4242, 4243}, 2: set()}


def test_adopted_queue_revalidates_code_cache_and_config_before_launch(
    tmp_path: Path,
) -> None:
    module = _module()
    code = tmp_path / "code"
    required = [
        code / "segbench/scripts/train_seg.py",
        code / "segbench/scripts/train_patch.py",
        code / "segbench/segbench/deep_utils.py",
        code / "segbench/segbench/preprocess_cache.py",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.name}\n", encoding="utf-8")
    subprocess.check_call(["git", "init", "-q"], cwd=code)
    subprocess.check_call(["git", "config", "user.email", "test@example.invalid"], cwd=code)
    subprocess.check_call(["git", "config", "user.name", "Test"], cwd=code)
    subprocess.check_call(["git", "add", "."], cwd=code)
    subprocess.check_call(["git", "commit", "-qm", "fixture"], cwd=code)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=code, text=True).strip()

    manifest = tmp_path / "manifest.csv"
    source = tmp_path / "source"
    source.mkdir()
    cache = tmp_path / "cache"
    records = []
    manifest_lines = ["image_id,image_path,mask_path,quality_mask_path"]
    for index in range(86):
        image_id = f"i{index:02d}"
        source_image = source / f"{image_id}.image"
        source_roi = source / f"{image_id}.roi"
        source_quality = source / f"{image_id}.quality"
        source_image.write_bytes(b"source-image" + bytes([index]))
        source_roi.write_bytes(b"source-roi" + bytes([index]))
        if index < 20:
            source_quality.write_bytes(b"source-quality" + bytes([index]))
        manifest_lines.append(
            f"{image_id},{source_image},{source_roi},"
            + (str(source_quality) if index < 20 else "")
        )
        image = cache / "images" / f"{image_id}.npy"
        roi = cache / "masks/mask_path" / f"{image_id}.npy"
        quality = cache / "masks/quality_mask_path" / f"{image_id}.npy"
        artifacts = [(image, b"i"), (roi, b"r")]
        if index < 20:
            artifacts.append((quality, b"q"))
        for path, value in artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value + bytes([index]))
        records.append(
            {
                "image_id": image_id,
                "shape": [8, 9],
                "image_sha256": module.sha256_file(image),
                "source_image_sha256": module.sha256_file(source_image),
                "mask_sha256": {
                    "mask_path": module.sha256_file(roi),
                    "quality_mask_path": module.sha256_file(quality) if index < 20 else None,
                },
                "source_mask_sha256": {
                    "mask_path": module.sha256_file(source_roi),
                    "quality_mask_path": module.sha256_file(source_quality) if index < 20 else None,
                },
            }
        )
    manifest.write_text("\n".join(manifest_lines) + "\n")
    receipt = cache / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "uint8_500ppi_preprocessing_cache",
                "manifest_sha256": module.sha256_file(manifest),
                "target_ppi": 500.0,
                "mask_columns": ["mask_path", "quality_mask_path"],
                "image_format": "uint8_npy",
                "mask_format": "uint8_0_255_npy",
                "record_count": len(records),
                "debug_limited": False,
                "records": records,
                "verification": {
                    "requested_count": 20,
                    "verified_count": 20,
                    "array_identity": True,
                    "image_ids": [record["image_id"] for record in records[:20]],
                },
            }
        )
    )
    amendment = tmp_path / "amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "kind": "stage4_threshold_policy_amendment",
                "replacement_policy": {
                    "allowed_trainable_grid": module.ADOPTED_THRESHOLDS,
                    "selection_data": "validation_only",
                },
            }
        )
    )
    config = tmp_path / "config.yaml"
    training = {
        **module.ADOPTED_DATALOADER,
        "batch_size": 4,
        "gradient_accumulation": 1,
        "threshold_candidates": module.ADOPTED_THRESHOLDS,
    }
    batch_contract = module.build_queue_batch_contract(training)
    config.write_text(
        yaml.safe_dump(
            {
                "training": training,
                "data": {"target_ppi": 500},
                "runtime": {
                    "memory_fraction": 0.3,
                    "queue_batch_contract": batch_contract,
                    "queue_batch_contract_sha256": module.config_digest(
                        batch_contract
                    ),
                },
            }
        )
    )
    argv = [
        "python",
        str(required[0]),
        "--resume",
        "auto",
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--cache-dir",
        str(cache),
        "--threshold-amendment",
        str(amendment),
    ]
    item = {
        "name": "pending",
        "argv": argv,
        "env": {},
        "completion_artifact": str(tmp_path / "run.json"),
        "resources": {"host_ram_gib": 12.0, "cpu_workers": 8},
        "adoption_role": "cache_amended_pending_job",
    }
    item["job_identity_sha256"] = module._declared_identity(item)
    binding = {
        "schema_version": 1,
        "kind": "stage3_cache_concurrency_execution_binding",
        "code": {
            "root": str(code),
            "git_commit": commit,
            "git_tree": tree,
            "files_sha256": {
                str(path.relative_to(code)): module.sha256_file(path) for path in required
            },
        },
        "cache_dir": str(cache),
        "cache_receipt": str(receipt),
        "cache_receipt_sha256": module.sha256_file(receipt),
        "cache_manifest_sha256": module.sha256_file(manifest),
        "cache_manifest": str(manifest),
        "threshold_amendment": str(amendment),
        "threshold_amendment_sha256": module.sha256_file(amendment),
        "generated_config_sha256": {"pending": module.sha256_file(config)},
        "generated_batch_contract_sha256": {
            "pending": module.config_digest(batch_contract)
        },
        "legacy_job_files_sha256": {},
        "legacy_external_job_names": [],
        "cache_amended_job_names": ["pending"],
        "dataloader": module.ADOPTED_DATALOADER,
        "oom_batch_policy": module.OOM_BATCH_POLICY,
    }
    queue = tmp_path / "adopted.yaml"
    queue.write_text(
        yaml.safe_dump(
            {
                "kind": module.ADOPTED_QUEUE_KIND,
                "launch_authorized": True,
                "execution_binding": binding,
                "runtime_policy": {
                    "host_ram_gib_per_job": 12.0,
                    "memory_fraction": 0.3,
                },
                "jobs": [item],
            },
            sort_keys=False,
        )
    )
    assert module.validate_adopted_queue_runtime(queue) == binding
    config.write_text(config.read_text() + "changed: true\n")
    with pytest.raises(ValueError, match="config digest changed"):
        module.validate_adopted_queue_runtime(queue)


def test_external_worker_plan_is_queue_and_launcher_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue_path, completion = _queue(tmp_path)
    worker = {
        "name": "external.sweep",
        "pid": 4242,
        "start_ticks": 777,
        "argv_sha256": "a" * 64,
        "gpu": 0,
        "completion_artifact": str(completion),
        "preempt_path": str(tmp_path / "preempt"),
        "host_ram_gib": 12.0,
        "cpu_workers": 4,
    }
    binding = {
        "kind": "synthetic",
        "generated_config_sha256": {},
        "generated_batch_contract_sha256": {},
        "oom_batch_policy": module.OOM_BATCH_POLICY,
    }
    queue_payload = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    queue_payload["runtime_policy"] = {"host_ram_gib_per_job": 12.0}
    queue_path.write_text(yaml.safe_dump(queue_payload), encoding="utf-8")
    plan = {
        "schema_version": 1,
        "kind": "guarded_queue_external_worker_plan",
        "supplemental_queue_sha256": module.sha256_file(queue_path),
        "allowed_gpus": [0, 2],
        "max_jobs_per_gpu": 2,
        "max_project_ram_gib": 64.0,
        "memory_fraction": 0.3,
        "reservation_override_gib": 12.0,
        "generated_config_sha256": {},
        "generated_batch_contract_sha256": {},
        "oom_batch_policy": module.OOM_BATCH_POLICY,
        "execution_binding_sha256": module.canonical_sha256(binding),
        "cpu_budget": {
            "max_cpu_cores": 40,
            "aggregate_data_workers": 32,
            "headroom_for_controller_and_reporting": 8,
        },
        "workers": [worker],
    }
    path = tmp_path / "external.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(module, "_external_is_exactly_live", lambda _: True)
    monkeypatch.setattr(
        module, "validate_adopted_queue_runtime", lambda *args, **kwargs: binding
    )
    workers, loaded = module.load_external_workers(
        path,
        queue=queue_path,
        gpus=[0, 2],
        max_per_gpu=2,
        max_project_ram_gib=64.0,
        max_cpu_cores=40,
        memory_fraction=0.3,
    )
    assert loaded == plan
    assert [(item.name, item.gpu, item.host_ram_gib, item.cpu_workers) for item in workers] == [
        ("external.sweep", 0, 12.0, 4)
    ]
    with pytest.raises(ValueError, match="binding changed"):
        module.load_external_workers(
            path,
            queue=queue_path,
            gpus=[0, 2],
            max_per_gpu=1,
            max_project_ram_gib=64.0,
            max_cpu_cores=40,
            memory_fraction=0.3,
        )


def test_inference_queue_requires_and_records_exact_timing_attestation(
    tmp_path: Path,
) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    payload = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    payload.update(
        phase="stage3_native_test_inference",
        required_max_jobs_per_gpu=1,
        required_launcher_args=["--max-per-gpu", "1"],
        required_gpu_indices=[2, 5],
    )
    queue_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    jobs = module.load_guarded_jobs(queue_path)
    attestation = module.queue_launcher_attestation(
        queue_path, gpus=[2, 5], max_per_gpu=1, memory_fraction=0.3
    )
    assert attestation == {
        "schema_version": 1,
        "queue_phase": "stage3_native_test_inference",
        "queue_sha256": module.sha256_file(queue_path),
        "gpus": [2, 5],
        "max_per_gpu": 1,
        "memory_fraction": 0.3,
        "required_launcher_args": ["--max-per-gpu", "1"],
    }
    ledger = tmp_path / "inference-ledger.json"
    state = module.load_state(ledger, queue_path, jobs, attestation)
    assert state["schema_version"] == 2
    assert state["launcher_attestation"] == attestation

    legacy = tmp_path / "legacy-inference-ledger.json"
    module.atomic_json(legacy, module.initial_state(queue_path, jobs))
    with pytest.raises(ValueError, match="lacks the exact launcher"):
        module.load_state(legacy, queue_path, jobs, attestation)


@pytest.mark.parametrize(
    ("max_per_gpu", "memory_fraction", "message"),
    [
        (2, 0.3, "max-per-gpu differs"),
        (1, 0.2, "memory fraction 0.3"),
    ],
)
def test_inference_queue_rejects_launcher_policy_mismatch(
    tmp_path: Path,
    max_per_gpu: int,
    memory_fraction: float,
    message: str,
) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    payload = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    payload.update(
        phase="stage3_native_test_inference",
        required_max_jobs_per_gpu=1,
        required_launcher_args=["--max-per-gpu", "1"],
        required_gpu_indices=[2, 5],
    )
    queue_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        module.queue_launcher_attestation(
            queue_path,
            gpus=[2, 5],
            max_per_gpu=max_per_gpu,
            memory_fraction=memory_fraction,
        )


def test_procfs_process_disappearance_is_not_controller_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    original = Path.read_text

    def vanished(self: Path, *args, **kwargs):
        if str(self).startswith("/proc/424242/"):
            raise ProcessLookupError(3, "process vanished", str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", vanished)
    assert module._pid_pss_bytes(424242) == 0


def test_restart_refuses_to_duplicate_exact_live_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    jobs = module.load_guarded_jobs(queue_path)
    ledger = tmp_path / "ledger.json"
    state = module.initial_state(queue_path, jobs)
    state["jobs"][jobs[0].name].update(
        status="running",
        worker={
            "pid": 4321,
            "start_ticks": 9876,
            "argv_sha256": module.canonical_sha256(jobs[0].base_job.argv),
            "gpu": 1,
            "queue_batch_environment": {
                "SEG_OOM_RETRY_COUNT": "0",
                "SEG_BATCH_SCALE": "1.0",
                "SEG_GRAD_ACCUM_SCALE": "1.0",
            },
        },
    )
    module.atomic_json(ledger, state)
    monkeypatch.setattr(module, "_worker_is_exactly_live", lambda _: True)
    with pytest.raises(RuntimeError, match="orphan reconciliation"):
        module.load_state(ledger, queue_path, jobs)


def test_restart_refuses_running_crash_window_without_worker_identity(
    tmp_path: Path,
) -> None:
    module = _module()
    queue_path, _ = _queue(tmp_path)
    jobs = module.load_guarded_jobs(queue_path)
    ledger = tmp_path / "ledger.json"
    state = module.initial_state(queue_path, jobs)
    state["jobs"][jobs[0].name]["status"] = "running"
    module.atomic_json(ledger, state)
    with pytest.raises(RuntimeError, match="lacks worker identity"):
        module.load_state(ledger, queue_path, jobs)


def test_worker_identity_failure_terminates_owned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()

    class Process:
        pid = 999

        def __init__(self):
            self.terminated = False
            self.waited = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == 30
            self.waited = True

    class Log:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    process, log = Process(), Log()
    preempt = tmp_path / "preempt"
    preempt.touch()
    launched = SimpleNamespace(process=process, log_handle=log, preempt_path=preempt)
    monkeypatch.setattr(
        module,
        "worker_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("identity")),
    )
    with pytest.raises(RuntimeError, match="identity"):
        module.authenticate_or_terminate(launched, ["python", "job.py"], 7)
    assert process.terminated and process.waited and log.closed
    assert not preempt.exists()


def test_restart_recovers_vanished_worker_with_exact_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    queue_path, completion = _queue(tmp_path)
    jobs = module.load_guarded_jobs(queue_path)
    ledger = tmp_path / "ledger.json"
    state = module.initial_state(queue_path, jobs)
    state["jobs"][jobs[0].name].update(
        status="running",
        worker={
            "pid": 4321,
            "start_ticks": 9876,
            "argv_sha256": module.canonical_sha256(jobs[0].base_job.argv),
            "gpu": 1,
            "queue_batch_environment": {
                "SEG_OOM_RETRY_COUNT": "0",
                "SEG_BATCH_SCALE": "1.0",
                "SEG_GRAD_ACCUM_SCALE": "1.0",
            },
        },
    )
    module.atomic_json(ledger, state)
    completion.write_text("immutable completion", encoding="utf-8")
    monkeypatch.setattr(module, "_worker_is_exactly_live", lambda _: False)
    resumed = module.load_state(ledger, queue_path, jobs)
    entry = resumed["jobs"][jobs[0].name]
    assert entry["status"] == "completed"
    assert entry["artifact_sha256"] == module.sha256_file(completion)
    assert entry["recovered_after_controller_exit"] is True
    assert "worker" not in entry
