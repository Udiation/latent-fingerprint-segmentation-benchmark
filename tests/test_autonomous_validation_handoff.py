"""Synthetic, CPU-only tests for autonomous validation handoff controls."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from segbench.artifact_identity import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module(filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _file(path: Path, content: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _runtime_policy() -> dict:
    """Small synthetic resource policy; no deployment topology is encoded."""
    return {
        "allowed_gpu_indices": [0],
        "max_jobs_per_gpu": 1,
        "min_free_mib": 1024,
        "foreign_process_mib": 512,
        "memory_fraction": 0.25,
        "min_disk_free_fraction": 0.1,
        "max_project_ram_gib": 48.0,
        "max_cpu_cores": 4,
        "min_available_ram_gib": 2.0,
        "seed0_host_ram_gib": 12.0,
        "seed0_cpu_workers": 2,
    }


def test_coexisting_snapshot_authenticates_controller_and_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module("run_queue_guarded.py")
    target = _file(tmp_path / "target.yaml")
    source = _file(tmp_path / "source.yaml")
    proof = _file(tmp_path / "proof.json")
    completion = tmp_path / "run.json"
    job = module.GuardedJob(
        module.base.Job("sweep.alexnet.lr0p0003.seed0", ["python", "train.py"]),
        completion,
        16.0,
        8,
        "a" * 64,
    )
    state = {
        "queue_sha256": sha256_file(source),
        "jobs": {
            job.name: {
                "identity_sha256": job.identity_sha256,
                "status": "running",
                "worker": {
                    "pid": 101,
                    "start_ticks": 202,
                    "argv_sha256": "b" * 64,
                    "gpu": 0,
                },
            }
        },
    }
    state_path = _file(tmp_path / "state.json", json.dumps(state))
    plan = {
        "schema_version": 1,
        "kind": "guarded_queue_coexisting_controller_plan",
        "target_queue_sha256": sha256_file(target),
        "allowed_gpus": [0],
        "max_jobs_per_gpu": 1,
        "source_max_jobs_per_gpu": 2,
        "max_project_ram_gib": 48.0,
        "max_cpu_cores": 4,
        "memory_fraction": 0.25,
        "read_only_source": True,
        "may_signal_source": False,
        "may_write_source_ledger": False,
        "source_queue": str(source),
        "source_queue_sha256": sha256_file(source),
        "source_state": str(state_path),
        "source_proof": str(proof),
        "source_proof_sha256": sha256_file(proof),
        "source_controller": {
            "pid": 303,
            "start_ticks": 404,
            "argv_sha256": "c" * 64,
        },
        "source_jobs": [{
            "name": job.name,
            "identity_sha256": job.identity_sha256,
            "completion_artifact": str(completion),
            "host_ram_gib": 16.0,
            "cpu_workers": 8,
        }],
    }
    plan_path = _file(tmp_path / "coexist.json", json.dumps(plan))
    monkeypatch.setattr(module, "validate_adopted_queue_runtime", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(module, "load_guarded_jobs", lambda *_a, **_k: [job])
    monkeypatch.setattr(
        module,
        "_process_start_ticks",
        lambda pid: {101: 202, 303: 404}.get(pid),
    )
    monkeypatch.setattr(
        module,
        "_process_argv_sha256",
        lambda pid: {101: "b" * 64, 303: "c" * 64}.get(pid),
    )
    loaded = module.load_coexisting_controller_plan(
        plan_path,
        target_queue=target,
        gpus=[0],
        max_per_gpu=1,
        max_project_ram_gib=48.0,
        max_cpu_cores=4,
        memory_fraction=0.25,
    )
    workers = module.observe_coexisting_workers(loaded)
    assert [(worker.name, worker.gpu) for worker in workers] == [(job.name, 0)]
    assert loaded["may_signal_source"] is False
    monkeypatch.setattr(module, "_process_argv_sha256", lambda _pid: "d" * 64)
    with pytest.raises(RuntimeError, match="PID identity|controller identity"):
        module.observe_coexisting_workers(loaded)


def test_selected_seed0_builder_uses_validation_receipt_only(tmp_path: Path) -> None:
    module = _module("prepare_selected_seed0_queue.py")
    code = tmp_path / "code"
    base = yaml.safe_load((ROOT / "configs/unetpp.yaml").read_text(encoding="utf-8"))
    _file(code / "configs/unetpp.yaml", yaml.safe_dump(base))
    _file(code / "scripts/train_seg.py")
    python = _file(tmp_path / "python")
    manifest = _file(tmp_path / "manifest.csv", "image_id\nsynthetic\n")
    splits = _file(tmp_path / "splits.json", "{}")
    cache = tmp_path / "cache"
    _file(
        cache / "receipt.json",
        json.dumps({
            "manifest_sha256": sha256_file(manifest),
            "target_ppi": 500.0,
            "debug_limited": False,
        }),
    )
    amendment = _file(
        tmp_path / "amendment.json",
        json.dumps({
            "kind": "stage4_threshold_policy_amendment",
            "replacement_policy": {
                "allowed_trainable_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "selection_data": "validation_only",
            },
        }),
    )
    selection_path = tmp_path / "unetpp.seed0.json"
    selection = {
        "kind": "expanded_grid_learning_rate_selection",
        "selection_source": "validation_only",
        "test_or_heldout_opened": False,
        "method": "unetpp",
        "selected_learning_rate": 0.0003,
        "selected_threshold": 0.2,
    }
    _file(selection_path, json.dumps(selection))
    summary = _file(
        tmp_path / "summary.json",
        json.dumps({
            "kind": "validation_replay_selection_summary",
            "selection_source": "validation_only",
            "test_or_heldout_opened": False,
            "groups": [{
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
            }],
            "deferred_candidates": [],
        }),
    )
    args = SimpleNamespace(
        selection_summary=[summary], code_root=code, python=python,
        manifest=manifest, splits=splits, cache_dir=cache,
        threshold_amendment=amendment, runs_root=tmp_path / "runs",
        config_out=tmp_path / "configs", queue_out=tmp_path / "queue.yaml",
        proof_out=tmp_path / "proof.json", segformer_dir=None,
        job_host_ram_gib=12.0, job_cpu_workers=2,
    )
    result = module.prepare(args)
    assert result["status"] == "READY_NOT_LAUNCHED"
    assert result["test_or_heldout_opened"] is False
    queue = yaml.safe_load(args.queue_out.read_text(encoding="utf-8"))
    assert queue["test_inference_authorized"] is False
    assert [job["name"] for job in queue["jobs"]] == ["full.unetpp.seed0"]
    assert "test" not in queue["jobs"][0]["argv"]
    assert "--split" not in queue["jobs"][0]["argv"]
    generated = yaml.safe_load((args.config_out / "unetpp.yaml").read_text())
    assert generated["training"]["learning_rate"] == 0.0003
    assert generated["training"]["threshold_candidates"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert queue["jobs"][0]["resources"]["host_ram_gib"] == 12.0


def test_selected_seed0_reuse_requires_completed_final_validation_replay(
    tmp_path: Path,
) -> None:
    module = _module("prepare_selected_seed0_queue.py")
    selector = _file(tmp_path / "selector.json", "{}")
    completion_path = tmp_path / "completion.json"
    candidate = {
        "name": "unet.final.seed0",
        "method": "unet",
        "role": "final",
        "candidate_identity_sha256": "c" * 64,
        "outputs": {"completion": str(completion_path)},
    }
    plan_path = _file(
        tmp_path / "plan.json",
        json.dumps({
            "kind": "dataset1_validation_probability_replay_plan",
            "selection_split": "val",
            "test_or_heldout_opened": False,
            "candidates": [candidate],
        }),
    )
    completion = {
        "kind": "validation_probability_replay_completion",
        "status": "complete",
        "complete": True,
        "plan_sha256": sha256_file(plan_path),
        "candidate": candidate["name"],
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "method": "unet",
        "role": "final",
        "selection_split": "val",
        "test_or_heldout_opened": False,
        "selector": str(selector),
        "selector_sha256": sha256_file(selector),
        "selected_threshold": 0.2,
        "selected_validation_dice": 0.8,
    }
    _file(completion_path, json.dumps(completion))
    receipt = module._completed_deferred_replay(
        {"plan": str(plan_path), "plan_sha256": sha256_file(plan_path)},
        {
            "candidate": candidate["name"],
            "candidate_identity_sha256": candidate["candidate_identity_sha256"],
            "method": "unet",
            "eligible": True,
        },
    )
    assert receipt["selector_sha256"] == sha256_file(selector)
    assert receipt["selected_threshold"] == 0.2

    completion["test_or_heldout_opened"] = True
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="completion changed"):
        module._completed_deferred_replay(
            {"plan": str(plan_path), "plan_sha256": sha256_file(plan_path)},
            {
                "candidate": candidate["name"],
                "candidate_identity_sha256": candidate["candidate_identity_sha256"],
                "method": "unet",
                "eligible": True,
            },
        )


def test_selected_seed0_builder_emits_fresh_versioned_sufs_job(
    tmp_path: Path,
) -> None:
    module = _module("prepare_selected_seed0_queue.py")
    code = tmp_path / "code"
    config = yaml.safe_load((ROOT / "configs/sufs_ft.yaml").read_text(encoding="utf-8"))
    _file(code / "configs/sufs_ft.yaml", yaml.safe_dump(config, sort_keys=False))
    _file(code / "scripts/train_sufs.py")
    python = _file(tmp_path / "python")
    manifest = _file(tmp_path / "manifest.csv", "image_id\nsynthetic\n")
    splits = _file(tmp_path / "splits.json", "{}")
    cache = tmp_path / "cache"
    _file(cache / "receipt.json", json.dumps({
        "manifest_sha256": sha256_file(manifest),
        "target_ppi": 500.0,
        "debug_limited": False,
    }))
    amendment = _file(tmp_path / "amendment.json", json.dumps({
        "kind": "stage4_threshold_policy_amendment",
        "replacement_policy": {
            "allowed_trainable_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "selection_data": "validation_only",
        },
    }))
    selection_path = tmp_path / "sufs_ft.seed0.json"
    selection = {
        "kind": "expanded_grid_learning_rate_selection",
        "selection_source": "validation_only",
        "selection_metric": "macro_image_dice_after_per_candidate_threshold_selection",
        "test_or_heldout_opened": False,
        "method": "sufs_ft",
        "tie_break": "candidate_inventory_order",
        "selected_learning_rate": 0.001,
        "selected_threshold": 0.2,
        "selected_validation_dice": 0.81,
        "candidates": [
            {
                "learning_rate": 0.0003,
                "selected_validation_dice": 0.80,
                "selected_threshold": 0.3,
                "checkpoint_sha256": "a" * 64,
                "selector_sha256": "b" * 64,
                "completion_sha256": "c" * 64,
            },
            {
                "learning_rate": 0.001,
                "selected_validation_dice": 0.81,
                "selected_threshold": 0.2,
                "checkpoint_sha256": "d" * 64,
                "selector_sha256": "e" * 64,
                "completion_sha256": "f" * 64,
            },
        ],
    }
    _file(selection_path, json.dumps(selection))
    summary = _file(tmp_path / "summary.json", json.dumps({
        "kind": "validation_replay_selection_summary",
        "selection_source": "validation_only",
        "test_or_heldout_opened": False,
        "groups": [{"path": str(selection_path), "sha256": sha256_file(selection_path)}],
        "deferred_candidates": [{
            "candidate": "sufs_ft.final.seed0",
            "method": "sufs_ft",
            "eligible": False,
        }],
    }))
    args = SimpleNamespace(
        selection_summary=[summary], code_root=code, python=python,
        manifest=manifest, splits=splits, cache_dir=cache,
        threshold_amendment=amendment, runs_root=tmp_path / "runs",
        config_out=tmp_path / "configs", queue_out=tmp_path / "queue.yaml",
        proof_out=tmp_path / "proof.json", segformer_dir=None,
        job_host_ram_gib=12.0, job_cpu_workers=2,
    )
    result = module.prepare(args)
    assert result["jobs"] == ["full.sufs_ft.seed0"]
    queue = yaml.safe_load(args.queue_out.read_text(encoding="utf-8"))
    job = queue["jobs"][0]
    assert "full_v3_expanded_grid/sufs_ft/seed0" in job["completion_artifact"]
    assert "--mode" in job["argv"] and "full" in job["argv"]
    adapter = json.loads((args.config_out / "sufs_ft_selection.json").read_text())
    assert adapter["selected_learning_rate"] == 0.001
    assert adapter["test_or_heldout_opened"] is False
    assert adapter["source_config_sha256"] == sha256_file(code / "configs/sufs_ft.yaml")


def test_handoff_prepare_binds_current_and_future_controller_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module("autonomous_validation_handoff.py")
    initial_plan = _file(
        tmp_path / "initial.json",
        json.dumps({"selection_split": "val", "test_or_heldout_opened": False}),
    )
    initial_queue = _file(
        tmp_path / "initial.yaml",
        yaml.safe_dump({"selection_split": "val", "test_inference_authorized": False}),
    )
    future = ["python", "guard.py", "--max-per-gpu", "2"]
    spec = _file(
        tmp_path / "spec.yaml",
        yaml.safe_dump({
            "schema_version": 1,
            "kind": "autonomous_validation_handoff_spec",
            "transition_controller": {
                "pid": 10,
                "start_ticks": 20,
                "current_argv_sha256": "a" * 64,
                "allowed_mixed_controller_argv": [{
                    "argv": future,
                    "argv_sha256": canonical_sha256(future),
                }],
            },
            "initial_replay_plan": str(initial_plan),
            "initial_replay_queue": str(initial_queue),
            "runtime_policy": _runtime_policy(),
        }),
    )
    monkeypatch.setattr(module, "_process_start_ticks", lambda _pid: 20)
    monkeypatch.setattr(module, "_process_argv_sha256", lambda _pid: "a" * 64)
    monkeypatch.setattr(module, "_code_identity", lambda _root: {
        "root": str(tmp_path), "git_commit": "b" * 40,
        "git_tree": "c" * 40, "files_sha256": {"x": "d" * 64},
    })
    proof = module.prepare(spec, tmp_path, tmp_path / "plan.json", tmp_path / "proof.json")
    assert proof["status"] == "READY_NOT_LAUNCHED"
    assert proof["gpu_launched"] is False
    assert proof["test_or_heldout_opened"] is False
    assert proof["source_controller_phase"] == "transition_watcher"


def test_handoff_prepare_accepts_only_preregistered_post_exec_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module("autonomous_validation_handoff.py")
    initial_plan = _file(
        tmp_path / "initial.json",
        json.dumps({"selection_split": "val", "test_or_heldout_opened": False}),
    )
    initial_queue = _file(
        tmp_path / "initial.yaml",
        yaml.safe_dump({"selection_split": "val", "test_inference_authorized": False}),
    )
    future = ["python", "guard.py", "--max-per-gpu", "2"]
    future_sha = canonical_sha256(future)
    spec = _file(
        tmp_path / "spec.yaml",
        yaml.safe_dump({
            "schema_version": 1,
            "kind": "autonomous_validation_handoff_spec",
            "transition_controller": {
                "pid": 10,
                "start_ticks": 20,
                "current_argv_sha256": "a" * 64,
                "allowed_mixed_controller_argv": [{
                    "argv": future,
                    "argv_sha256": future_sha,
                }],
            },
            "initial_replay_plan": str(initial_plan),
            "initial_replay_queue": str(initial_queue),
            "runtime_policy": _runtime_policy(),
        }),
    )
    monkeypatch.setattr(module, "_process_start_ticks", lambda _pid: 20)
    monkeypatch.setattr(module, "_process_argv_sha256", lambda _pid: future_sha)
    monkeypatch.setattr(module, "_code_identity", lambda _root: {
        "root": str(tmp_path), "git_commit": "b" * 40,
        "git_tree": "c" * 40, "files_sha256": {"x": "d" * 64},
    })
    proof = module.prepare(
        spec, tmp_path, tmp_path / "plan.json", tmp_path / "proof.json",
    )
    assert proof["source_controller_phase"] == "mixed_recovery_controller"
    assert proof["source_controller_argv_sha256"] == future_sha

    monkeypatch.setattr(module, "_process_argv_sha256", lambda _pid: "e" * 64)
    with pytest.raises(ValueError, match="not an exact live preregistered process"):
        module.prepare(
            spec, tmp_path, tmp_path / "changed-plan.json",
            tmp_path / "changed-proof.json",
        )


def test_completed_source_accepts_expected_controller_exit_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module("autonomous_validation_handoff.py")
    argv_one = ["python", "guard.py", "--max-per-gpu", "1"]
    argv_two = ["python", "guard.py", "--max-per-gpu", "2"]
    plan = {
        "transition_controller": {
            "pid": 10,
            "start_ticks": 20,
            "allowed_mixed_controller_argv": [
                {"argv": argv_one, "argv_sha256": canonical_sha256(argv_one)},
                {"argv": argv_two, "argv_sha256": canonical_sha256(argv_two)},
            ],
        }
    }
    queue = _file(
        tmp_path / "source.yaml",
        yaml.safe_dump({"runtime_policy": {"max_jobs_per_gpu": 2}}),
    )
    monkeypatch.setattr(module, "_process_start_ticks", lambda _pid: None)
    identity = module._completed_source_controller_identity(plan, queue)
    assert identity["argv"] == argv_two
    assert identity["controller_exited_after_completion"] is True

    monkeypatch.setattr(module, "_process_start_ticks", lambda _pid: 999)
    monkeypatch.setattr(module, "_process_argv_sha256", lambda _pid: "f" * 64)
    with pytest.raises(RuntimeError, match="PID start identity changed"):
        module._completed_source_controller_identity(plan, queue)


def test_handoff_run_is_validation_only_and_stops_at_seed0_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module("autonomous_validation_handoff.py")
    code_root = tmp_path / "code"
    code_root.mkdir()
    initial_plan = _file(
        tmp_path / "initial_plan.json",
        json.dumps({
            "candidates": [{"inputs": {"manifest": {"path": "m"}}}],
        }),
    )
    initial_queue = _file(tmp_path / "initial_queue.yaml", "jobs: []\n")
    plan_path = tmp_path / "handoff_plan.json"
    handoff = tmp_path / "handoff"
    plan = {
        "kind": module.PLAN_KIND,
        "test_or_heldout_opened": False,
        "code": {"root": str(code_root)},
        "handoff_root": str(handoff),
        "initial_replay_plan": str(initial_plan),
        "initial_replay_plan_sha256": sha256_file(initial_plan),
        "initial_replay_queue": str(initial_queue),
        "python": "/synthetic/python",
        "restricted_root": str(tmp_path),
        "manifest": str(tmp_path / "manifest.csv"),
        "splits": str(tmp_path / "splits.json"),
        "cache_dir": str(tmp_path / "cache"),
        "training_threshold_amendment": str(tmp_path / "amendment.json"),
        "runs_root": str(tmp_path / "runs"),
        "segformer_dir": str(tmp_path / "segformer"),
        "runtime_policy": _runtime_policy(),
    }
    _file(plan_path, json.dumps(plan))
    monkeypatch.setattr(module, "_verify_code", lambda _value: code_root)
    monkeypatch.setattr(module, "_wait_for_source", lambda _plan: ({
        "pid": 1, "start_ticks": 2, "argv_sha256": "a" * 64,
        "argv": ["python", "guard", "--max-per-gpu", "1"],
    }, []))
    monkeypatch.setattr(
        module, "_coexistence_plan",
        lambda *_args, **_kwargs: {"read_only_source": True},
    )
    monkeypatch.setattr(module, "_wait_late_cards", lambda _plan: None)
    monkeypatch.setattr(module, "_late_inventory", lambda _plan, output: _file(output, "{}\n"))
    monkeypatch.setattr(module, "_guard_argv", lambda _p, q, _s, _c: [
        "/synthetic/python", "validation_guard", "--queue", str(q),
    ])
    labels: list[str] = []

    def fake_child(argv, *, cwd, label, log):  # noqa: ANN001
        assert cwd == code_root
        assert all(str(token).lower() not in {"test", "heldout"} for token in argv)
        labels.append(label)
        if label == "initial-validation-finalizer":
            out = Path(argv[argv.index("--out-dir") + 1])
            _file(out / "summary.json", json.dumps({"kind": "synthetic"}))
            queue = Path(argv[argv.index("--eligible-queue-out") + 1])
            _file(queue, yaml.safe_dump({"launch_authorized": False, "jobs": []}))
        elif label == "late-replay-compiler":
            _file(Path(argv[argv.index("--out-plan") + 1]), "{}")
            _file(Path(argv[argv.index("--out-queue") + 1]), "jobs: []\n")
            _file(Path(argv[argv.index("--proof-out") + 1]), "{}")
        elif label == "late-validation-finalizer":
            out = Path(argv[argv.index("--out-dir") + 1])
            _file(out / "summary.json", json.dumps({"kind": "synthetic"}))
        elif label == "selected-full-seed0-planner":
            queue = Path(argv[argv.index("--queue-out") + 1])
            _file(queue, yaml.safe_dump({"launch_authorized": True, "jobs": [{"name": "synthetic"}]}))
            _file(Path(argv[argv.index("--proof-out") + 1]), "{}")

    monkeypatch.setattr(module, "_run_child", fake_child)
    result = module.run(plan_path, sha256_file(plan_path))
    assert labels == [
        "initial-validation-replay",
        "initial-validation-finalizer",
        "late-replay-compiler",
        "late-validation-replay",
        "late-validation-finalizer",
        "selected-full-seed0-planner",
    ]
    assert result["status"] == "COMPLETE_NO_TEST_OPENED"
    assert result["selected_full_seed0_launched"] is False
    assert result["source_ledger_written"] is False
    assert result["source_signalled"] is False
