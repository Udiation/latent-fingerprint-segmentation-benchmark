"""Synthetic tests for the resumable, GT-blind Stage 3 supervisor."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from segbench.artifact_identity import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts" / "run_stage3_supervisor.py"
    spec = importlib.util.spec_from_file_location("stage3_supervisor_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _job(name: str, completion: Path) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "name": name,
        "argv": ["python", "synthetic.py", "--name", name],
        "env": {},
        "completion_artifact": str(completion),
        "resources": {"host_ram_gib": 8.0},
    }
    item["job_identity_sha256"] = canonical_sha256(
        {
            "name": item["name"],
            "argv": item["argv"],
            "env": item["env"],
            "completion_artifact": item["completion_artifact"],
            "resources": item["resources"],
        }
    )
    return item


def _queue_and_ledger(
    tmp_path: Path,
    names: list[str],
    statuses: list[str],
    *,
    inference: bool = False,
    allowed_gpus: list[int] | None = None,
    job_overrides: Dict[str, Dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    jobs = []
    entries = {}
    for index, (name, status) in enumerate(zip(names, statuses)):
        completion = tmp_path / f"complete-{index}.json"
        if status == "completed":
            completion.write_text(json.dumps({"name": name}), encoding="utf-8")
        item = _job(name, completion)
        if job_overrides and name in job_overrides:
            item.update(json.loads(json.dumps(job_overrides[name])))
            item["job_identity_sha256"] = canonical_sha256(
                {
                    "name": item["name"],
                    "argv": item["argv"],
                    "env": item.get("env", {}),
                    "completion_artifact": item["completion_artifact"],
                    "resources": item["resources"],
                }
            )
        jobs.append(item)
        entry: Dict[str, Any] = {
            "identity_sha256": item["job_identity_sha256"],
            "status": status,
            "attempts": 2 if status == "failed" else 1,
            "ordinary_retries": 1 if status == "failed" else 0,
            "oom_retries": 0,
            "preemptions": 0,
        }
        if status == "completed":
            entry["artifact_sha256"] = sha256_file(completion)
        entries[name] = entry
    queue_payload: Dict[str, Any] = {"launch_authorized": True, "jobs": jobs}
    if inference:
        if allowed_gpus is None:
            raise ValueError("timing-isolated test queue needs synthetic GPU IDs")
        queue_payload.update(
            phase="stage3_native_test_inference",
            required_max_jobs_per_gpu=1,
            required_launcher_args=["--max-per-gpu", "1"],
            required_gpu_indices=allowed_gpus,
        )
    queue = tmp_path / ("inference.yaml" if inference else "training.yaml")
    queue.write_text(yaml.safe_dump(queue_payload, sort_keys=False), encoding="utf-8")
    ledger_payload: Dict[str, Any] = {
        "schema_version": 2 if inference else 1,
        "queue_sha256": sha256_file(queue),
        "jobs": entries,
    }
    if inference:
        ledger_payload["launcher_attestation"] = {
            "schema_version": 1,
            "queue_phase": "stage3_native_test_inference",
            "queue_sha256": sha256_file(queue),
            "gpus": allowed_gpus,
            "max_per_gpu": 1,
            "memory_fraction": 0.3,
            "required_launcher_args": ["--max-per-gpu", "1"],
        }
    ledger = tmp_path / ("inference.json" if inference else "training.json")
    ledger.write_text(json.dumps(ledger_payload), encoding="utf-8")
    return queue, ledger


def _config(tmp_path: Path, module: Any) -> tuple[Path, Path, Path]:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    manifest = restricted / "latent_source.csv"
    manifest.write_text(
        "image_id,image_path,mask_path,subject_id,subset,ppi,source_sha256\n",
        encoding="utf-8",
    )
    splits = restricted / "splits.json"
    splits.write_text('{"seed": 42, "splits": {}}\n', encoding="utf-8")
    pin = tmp_path / "pinned"
    (pin / "segbench/scripts").mkdir(parents=True)
    python = "/synthetic/venv/python"
    # Deliberately non-physical synthetic IDs. Production IDs come only from
    # the restricted runtime configuration.
    allowed_gpus = [100 + offset for offset in range(2)]
    gpu_csv = ",".join(str(value) for value in allowed_gpus)
    max_cpu_cores = 8
    max_ram_gib = 32
    gate = restricted / "gate.json"
    plan = restricted / "plan.json"
    inference_queue = restricted / "inference.yaml"
    prep_bundle = restricted / "prep_bundle.json"
    aux_plan = restricted / "auxiliary" / "plan.json"
    aux_queue = restricted / "auxiliary" / "queue.yaml"
    aux_card = restricted / "auxiliary" / "card.json"
    global_selection = restricted / "selection" / "selection.json"
    generated_sufs_selection = restricted / "selection" / "sufs_ft_selection.json"
    audited_sufs_selection = restricted / "audited" / "sufs_ft_selection.json"
    pilot_seed0_raw = restricted / "jobs" / "pilot_seed0_raw.yaml"
    remaining_seed0_raw = restricted / "jobs" / "remaining_seed0_raw.yaml"
    seed12_raw = restricted / "jobs" / "seed12_raw.yaml"
    pilot_selection = restricted / "selection" / "pilot.json"
    expected_commits = restricted / "jobs" / "expected_commits.json"
    ram_reservations = restricted / "jobs" / "ram.json"
    sufs_note = restricted / "reports" / "sufs_provenance.md"
    seed0_authorized = restricted / "run_seed0.yaml"
    seed12_locked = restricted / "jobs" / "seed12_locked.yaml"
    prepare_proof = restricted / "jobs" / "prepare_proof.json"
    seed12_authorized = restricted / "run_seed12.yaml"
    seed12_authorization = restricted / "jobs" / "seed12_authorization.json"
    for path, contents in (
        (pilot_selection, "{}\n"),
        (expected_commits, "{}\n"),
        (ram_reservations, "{}\n"),
        (sufs_note, "synthetic provenance\n"),
        (audited_sufs_selection, '{"audited": true}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    snapshot = restricted / "models" / "segformer" / "snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    snapshot_files = {
        "config.json": b'{"model_type":"segformer"}\n',
        "preprocessor_config.json": b'{"size":512}\n',
        "pytorch_model.bin": b"synthetic-weights",
    }
    for name, contents in snapshot_files.items():
        (snapshot / name).write_bytes(contents)
    file_records = {
        name: {
            "path": str(snapshot / name),
            "sha256": sha256_file(snapshot / name),
            "bytes": (snapshot / name).stat().st_size,
            **(
                {"git_blob_sha1": module._git_blob_sha1(snapshot / name)}
                if name != "pytorch_model.bin"
                else {}
            ),
        }
        for name in snapshot_files
    }
    model_config = {
        "model": {
            "pretrained": True,
            "pretrained_model": "nvidia/mit-b2",
            "pretrained_revision": snapshot.name,
            "pretrained_path_env": "SEGBENCH_SEGFORMER_B2_DIR",
            "pretrained_files": {
                "config.json": {
                    "git_blob_sha1": file_records["config.json"]["git_blob_sha1"]
                },
                "preprocessor_config.json": {
                    "git_blob_sha1": file_records["preprocessor_config.json"][
                        "git_blob_sha1"
                    ]
                },
                "pytorch_model.bin": {
                    "sha256": file_records["pytorch_model.bin"]["sha256"],
                    "bytes": file_records["pytorch_model.bin"]["bytes"],
                },
            },
            "local_files_only": True,
        }
    }
    base_config = pin / "segbench" / "configs" / "segformer_sub.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    sweep_names = [
        "sweep.unetpp.lr0p0003.seed0",
        "sweep.unetpp.lr0p001.seed0",
        "sweep.drunet.lr0p0003.seed0",
        "sweep.drunet.lr0p001.seed0",
        "sweep.ra_runet.lr0p0003.seed0",
        "sweep.ra_runet.lr0p001.seed0",
        "sweep.lenet.lr0p0003.seed0",
        "sweep.lenet.lr0p001.seed0",
        "sweep.alexnet.lr0p0003.seed0",
        "sweep.alexnet.lr0p001.seed0",
        "sweep.cnn_ae.lr0p0003.seed0",
        "sweep.cnn_ae.lr0p001.seed0",
        "sweep.segformer_sub.lr0p0003.seed0",
        "sweep.segformer_sub.lr0p001.seed0",
    ]
    segformer_configs: Dict[str, Path] = {}
    segformer_overrides: Dict[str, Dict[str, Any]] = {}
    for name in sweep_names[-2:]:
        slug = "0p0003" if "0p0003" in name else "0p001"
        config_path = restricted / "configs" / f"segformer_sub_lr{slug}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
        segformer_configs[name] = config_path
        segformer_overrides[name] = {
            "argv": [
                python,
                str(pin / "segbench" / "scripts" / "train_seg.py"),
                "--config",
                str(config_path),
                "--seed",
                "0",
            ],
            "env": {"SEGBENCH_SEGFORMER_B2_DIR": str(snapshot)},
        }
    sweep_queue, sweep_ledger = _queue_and_ledger(
        restricted,
        sweep_names,
        ["completed"] * 14,
        job_overrides=segformer_overrides,
    )
    orphan_capture = restricted / "orphan_capture.json"
    orphan_capture.write_text('{"synthetic": true}\n', encoding="utf-8")
    actions: Dict[str, Any] = {}
    scripts = {
        "reconcile_sweep": ("reconcile_guarded_orphans.py", ["reconcile"]),
        "resume_sweep": ("run_queue_guarded.py", []),
        "select_full": ("prepare_deep_jobs.py", ["select-full"]),
        "prepare_seed0": ("authorize_deep_jobs.py", ["prepare-seed0"]),
        "run_seed0": ("run_queue_guarded.py", []),
        "authorize_seed12": ("authorize_deep_jobs.py", ["authorize-seed12"]),
        "run_seed12": ("run_queue_guarded.py", []),
        "prepare_inference": ("prepare_stage3_inference.py", []),
        "run_inference": ("run_queue_guarded.py", []),
        "run_auxiliary": ("run_queue_guarded.py", []),
        "finalize": ("finalize_stage3_native.py", []),
    }
    for step, (script, subcommand) in scripts.items():
        entrypoint = Path("segbench/scripts") / script
        argv = [python, str(pin / entrypoint), *subcommand]
        action: Dict[str, Any] = {
            "type": (
                "guarded_queue"
                if step.startswith("run_") or step == "resume_sweep"
                else "finalizer" if step == "finalize" else "command"
            ),
            "code_pin": (
                "delivery"
                if step
                in {
                    "reconcile_sweep",
                    "resume_sweep",
                    "prepare_seed0",
                    "run_seed0",
                    "authorize_seed12",
                    "run_seed12",
                    "prepare_inference",
                    "run_inference",
                    "run_auxiliary",
                    "finalize",
                }
                else "training"
            ),
            "entrypoint": str(entrypoint),
            "argv": argv,
            "child_log": str(restricted / "logs" / f"{step}.log"),
            "outputs": [],
        }
        if step == "reconcile_sweep":
            proof = restricted / "orphan_reconciliation.json"
            argv.extend(
                [
                    "--queue", str(sweep_queue),
                    "--ledger", str(sweep_ledger),
                    "--capture", str(orphan_capture),
                    "--proof-out", str(proof),
                ]
            )
            action["outputs"] = [str(proof)]
            action["bound_inputs"] = [
                {
                    "option": "--capture",
                    "path": str(orphan_capture),
                    "sha256": sha256_file(orphan_capture),
                }
            ]
        if step == "select_full":
            argv.extend(
                [
                    "--code-root", str(pin / "segbench"),
                    "--manifest", str(manifest),
                    "--splits", str(splits),
                    "--selection-out", str(global_selection),
                    "--pilot-seed0-jobs-out", str(pilot_seed0_raw),
                    "--remaining-seed0-jobs-out", str(remaining_seed0_raw),
                    "--seed12-jobs-out", str(seed12_raw),
                ]
            )
            action["outputs"] = [
                str(global_selection),
                str(generated_sufs_selection),
                str(pilot_seed0_raw),
                str(remaining_seed0_raw),
                str(seed12_raw),
            ]
            action["env"] = {
                "SEGBENCH_EXPECTED_COMMIT_MAP": str(expected_commits),
                "SEGBENCH_SEGFORMER_B2_DIR": str(snapshot),
            }
            action["bound_inputs"] = [
                {
                    "env_key": "SEGBENCH_EXPECTED_COMMIT_MAP",
                    "path": str(expected_commits),
                    "sha256": sha256_file(expected_commits),
                }
            ]
        if step == "prepare_seed0":
            argv.extend(
                [
                    "--global-selection", str(global_selection),
                    "--pilot-selection", str(pilot_selection),
                    "--sufs-selection", str(audited_sufs_selection),
                    "--pilot-seed0-jobs", str(pilot_seed0_raw),
                    "--remaining-seed0-jobs", str(remaining_seed0_raw),
                    "--seed12-jobs", str(seed12_raw),
                    "--expected-commits", str(expected_commits),
                    "--ram-reservations", str(ram_reservations),
                    "--runs-root", str(restricted / "runs" / "deep"),
                    "--seed0-out", str(seed0_authorized),
                    "--seed12-plan-out", str(seed12_locked),
                    "--proof-out", str(prepare_proof),
                    "--sufs-provenance-note", str(sufs_note),
                    "--sufs-provenance-note-sha256", sha256_file(sufs_note),
                    "--max-project-ram-gib", str(max_ram_gib),
                ]
            )
            action["cross_step_inputs"] = [
                {"option": option, "path": str(path), "producer_step": "select_full"}
                for option, path in (
                    ("--global-selection", global_selection),
                    ("--pilot-seed0-jobs", pilot_seed0_raw),
                    ("--remaining-seed0-jobs", remaining_seed0_raw),
                    ("--seed12-jobs", seed12_raw),
                )
            ]
            action["bound_inputs"] = [
                {
                    "option": "--sufs-selection",
                    "path": str(audited_sufs_selection),
                    "sha256": sha256_file(audited_sufs_selection),
                },
                {
                    "option": "--expected-commits",
                    "path": str(expected_commits),
                    "sha256": sha256_file(expected_commits),
                },
            ]
            action["outputs"] = [
                str(seed0_authorized),
                str(seed12_locked),
                str(prepare_proof),
            ]
        if step == "authorize_seed12":
            argv.extend(
                [
                    "--global-selection", str(global_selection),
                    "--expected-commits", str(expected_commits),
                    "--runs-root", str(restricted / "runs" / "deep"),
                    "--prepare-proof", str(prepare_proof),
                    "--seed0-queue", str(seed0_authorized),
                    "--seed12-plan", str(seed12_locked),
                    "--seed12-out", str(seed12_authorized),
                    "--proof-out", str(seed12_authorization),
                ]
            )
            action["cross_step_inputs"] = [
                {
                    "option": "--global-selection",
                    "path": str(global_selection),
                    "producer_step": "select_full",
                },
                *[
                    {
                        "option": option,
                        "path": str(path),
                        "producer_step": "prepare_seed0",
                    }
                    for option, path in (
                        ("--prepare-proof", prepare_proof),
                        ("--seed0-queue", seed0_authorized),
                        ("--seed12-plan", seed12_locked),
                    )
                ],
            ]
            action["bound_inputs"] = [
                {
                    "option": "--expected-commits",
                    "path": str(expected_commits),
                    "sha256": sha256_file(expected_commits),
                }
            ]
            action["outputs"] = [str(seed12_authorized), str(seed12_authorization)]
        if step == "prepare_inference":
            argv.extend(
                [
                    "--expected-planner-git-commit", "b" * 40,
                    "--inference-git-commit", "a" * 40,
                    "--finalizer-git-commit", "b" * 40,
                    "--code-root", str(pin / "segbench"),
                    "--manifest", str(manifest),
                    "--splits", str(splits),
                    "--sufs-selection", str(audited_sufs_selection),
                    "--expected-commits", str(expected_commits),
                    "--gate-out", str(gate),
                    "--plan-out", str(plan),
                    "--queue-out", str(inference_queue),
                    "--prep-bundle-out", str(prep_bundle),
                    "--runs-root", str(restricted / "runs" / "deep"),
                    "--masks-root", str(restricted / "masks" / "deterministic"),
                    "--cards-root", str(restricted / "cards" / "deterministic"),
                    "--work-root", str(restricted / "work" / "deterministic"),
                    "--stage2-masks", str(restricted / "masks" / "stage2"),
                    "--aux-plan-out", str(aux_plan),
                    "--aux-queue-out", str(aux_queue),
                    "--aux-card-out", str(aux_card),
                    "--aux-masks-root", str(restricted / "auxiliary" / "masks"),
                    "--aux-variance-root", str(restricted / "auxiliary" / "variance"),
                    "--aux-work-root", str(restricted / "auxiliary" / "work"),
                    "--aux-ram-gib", "8",
                    "--max-project-ram-gib", str(max_ram_gib),
                    "--restricted-root", str(restricted),
                    "--allowed-gpus", gpu_csv,
                ]
            )
            action["outputs"] = [
                str(gate), str(plan), str(inference_queue), str(prep_bundle),
                str(aux_plan), str(aux_queue),
            ]
            action["bound_inputs"] = [
                {
                    "option": "--sufs-selection",
                    "path": str(audited_sufs_selection),
                    "sha256": sha256_file(audited_sufs_selection),
                },
                {
                    "option": "--expected-commits",
                    "path": str(expected_commits),
                    "sha256": sha256_file(expected_commits),
                },
            ]
        if step == "finalize":
            delivery = restricted / "bundle" / "delivery.json"
            argv.extend(
                [
                    "--expected-finalizer-git-commit", "b" * 40,
                    "--expected-planner-git-commit", "b" * 40,
                    "--manifest", str(manifest),
                    "--splits", str(splits),
                    "--gate", str(gate),
                    "--plan", str(plan),
                    "--prep-bundle", str(prep_bundle),
                    "--expected-prep-bundle-sha256", module.DYNAMIC_SHA256,
                    "--inference-queue", str(inference_queue),
                    "--inference-ledger", str(restricted / "run_inference.json"),
                    "--aux-plan", str(aux_plan),
                    "--aux-queue", str(aux_queue),
                    "--aux-ledger", str(restricted / "run_auxiliary.json"),
                    "--aux-card", str(aux_card),
                    "--bundle-out", str(restricted / "bundle"),
                    "--restricted-root", str(restricted),
                    "--declared-delivery", str(delivery),
                ]
            )
            action["cross_step_inputs"] = [
                {
                    "option": "--gate",
                    "path": str(gate),
                    "producer_step": "prepare_inference",
                },
                {
                    "option": "--plan",
                    "path": str(plan),
                    "producer_step": "prepare_inference",
                },
                {
                    "option": "--prep-bundle",
                    "path": str(prep_bundle),
                    "producer_step": "prepare_inference",
                },
                {
                    "option": "--inference-queue",
                    "path": str(inference_queue),
                    "producer_step": "prepare_inference",
                },
                {
                    "option": "--inference-ledger",
                    "path": str(restricted / "run_inference.json"),
                    "producer_step": "run_inference",
                },
                {
                    "option": "--aux-plan",
                    "path": str(aux_plan),
                    "producer_step": "prepare_inference",
                },
                {
                    "option": "--aux-queue",
                    "path": str(aux_queue),
                    "producer_step": "prepare_inference",
                },
                {
                    "option": "--aux-ledger",
                    "path": str(restricted / "run_auxiliary.json"),
                    "producer_step": "run_auxiliary",
                },
                {
                    "option": "--aux-card",
                    "path": str(aux_card),
                    "producer_step": "run_auxiliary",
                },
            ]
            action["bound_inputs"] = [
                {
                    "option": "--splits",
                    "path": str(splits),
                    "sha256": sha256_file(splits),
                }
            ]
            action["dynamic_sha256_inputs"] = [
                {
                    "option": "--expected-prep-bundle-sha256",
                    "path": str(prep_bundle),
                    "producer_step": "prepare_inference",
                }
            ]
            action["outputs"] = [str(delivery)]
        if step.startswith("run_") or step == "resume_sweep":
            queue = (
                sweep_queue
                if step == "resume_sweep"
                else inference_queue
                if step == "run_inference"
                else aux_queue
                if step == "run_auxiliary"
                else restricted / f"{step}.yaml"
            )
            ledger = sweep_ledger if step == "resume_sweep" else restricted / f"{step}.json"
            max_per = 1 if step in {"run_inference", "run_auxiliary"} else 3
            argv.extend(
                [
                    "--jobs", str(queue),
                    "--state", str(ledger),
                    "--results", str(restricted / "runs" / step),
                    "--gpus", gpu_csv,
                    "--max-per-gpu", str(max_per),
                    "--memory-fraction", "0.3",
                    "--min-disk-free-fraction", "0.15",
                    "--max-project-ram-gib", str(max_ram_gib),
                ]
            )
            action.update(
                queue=str(queue),
                ledger=str(ledger),
            )
            if step == "run_inference":
                action["expected_names_from_step"] = "prepare_inference"
            else:
                action["expected_names"] = (
                    sweep_names
                    if step == "resume_sweep"
                    else ["stage3.drunet.mc_dropout_aux"]
                    if step == "run_auxiliary"
                    else [f"{step}.job"]
                )
            action["outputs"] = [str(ledger)] + (
                [str(aux_card)] if step == "run_auxiliary" else []
            )
            if step in {"run_inference", "run_auxiliary"}:
                action["cross_step_inputs"] = [
                    {
                        "option": "--jobs",
                        "path": str(queue),
                        "producer_step": "prepare_inference",
                    }
                ]
        actions[step] = action
    config = {
        "schema_version": 1,
        "kind": module.KIND,
        "launch_authorized": True,
        "restricted_root": str(restricted),
        "resources": {
            "allowed_gpus": allowed_gpus,
            "max_jobs_per_gpu": 3,
            "max_memory_fraction": 0.3,
            "max_cpu_cores": max_cpu_cores,
            "max_ram_gib": max_ram_gib,
            "min_disk_free_fraction": 0.15,
            "resource_root": str(restricted),
        },
        "bindings": {
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)}
        },
        "audited_sufs_selection": {
            "path": str(audited_sufs_selection),
            "sha256": sha256_file(audited_sufs_selection),
        },
        "segformer_snapshot": {
            "path": str(snapshot),
            "revision": snapshot.name,
            "files": file_records,
            "base_config": {
                "path": str(base_config),
                "sha256": sha256_file(base_config),
            },
            "sweep_configs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in segformer_configs.items()
            },
        },
        "role_pins": {"training": "training", "inference": "training"},
        "code_pins": {
            "training": {"repo_root": str(pin), "expected_commit": "a" * 40},
            "delivery": {"repo_root": str(pin), "expected_commit": "b" * 40},
        },
        "sweep": {
            "queue": str(sweep_queue),
            "ledger": str(sweep_ledger),
            "queue_sha256": sha256_file(sweep_queue),
            "expected_names": sweep_names,
        },
        "actions": actions,
    }
    config_path = restricted / "supervisor.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, restricted / "state.json", restricted / "supervisor.jsonl"


def test_read_only_sweep_barrier_and_exact_completion_hashes(tmp_path: Path) -> None:
    module = _module()
    names = ["one", "two"]
    queue, ledger = _queue_and_ledger(tmp_path, names, ["completed", "completed"])
    before = queue.read_bytes()
    evidence = module.inspect_ledger(
        queue, ledger, names, allow_terminal_failures=False
    )
    assert evidence["terminal"] is True
    assert evidence["completed"] == names
    assert queue.read_bytes() == before
    completion = Path(yaml.safe_load(queue.read_text())["jobs"][0]["completion_artifact"])
    completion.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="disappeared or changed"):
        module.inspect_ledger(queue, ledger, names, allow_terminal_failures=False)


def test_live_process_identity_rejects_procfs_pid_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    ticks = iter([100, 101])
    monkeypatch.setattr(module, "_proc_start_ticks", lambda proc: next(ticks))
    monkeypatch.setattr(module, "_proc_argv", lambda proc: ["python", "worker.py"])
    assert module._live_process_identity(12345) is None


def test_terminal_seed12_failure_requires_retry_but_can_reach_planner(
    tmp_path: Path,
) -> None:
    module = _module()
    queue, ledger = _queue_and_ledger(tmp_path, ["seed1", "seed2"], ["completed", "failed"])
    evidence = module.inspect_ledger(
        queue, ledger, ["seed1", "seed2"], allow_terminal_failures=True
    )
    assert evidence["terminal"] is True
    assert evidence["failed"] == ["seed2"]
    data = json.loads(ledger.read_text())
    data["jobs"]["seed2"]["attempts"] = 1
    ledger.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks retry evidence"):
        module.inspect_ledger(
            queue, ledger, ["seed1", "seed2"], allow_terminal_failures=True
        )


def test_inference_ledger_requires_single_job_timing_attestation(
    tmp_path: Path,
) -> None:
    module = _module()
    allowed_gpus = [100 + offset for offset in range(2)]
    queue, ledger = _queue_and_ledger(
        tmp_path,
        ["infer.one"],
        ["completed"],
        inference=True,
        allowed_gpus=allowed_gpus,
    )
    evidence = module.inspect_ledger(
        queue,
        ledger,
        ["infer.one"],
        allow_terminal_failures=False,
        require_timing_attestation=True,
        allowed_gpus=allowed_gpus,
    )
    assert evidence["terminal"] is True
    value = json.loads(ledger.read_text())
    value["launcher_attestation"]["max_per_gpu"] = 2
    ledger.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="timing-isolation"):
        module.inspect_ledger(
            queue,
            ledger,
            ["infer.one"],
            allow_terminal_failures=False,
            require_timing_attestation=True,
            allowed_gpus=allowed_gpus,
        )


def test_timing_policy_is_checked_before_a_ledger_exists(tmp_path: Path) -> None:
    module = _module()
    allowed_gpus = [100 + offset for offset in range(2)]
    queue, ledger = _queue_and_ledger(
        tmp_path,
        ["infer.one"],
        ["pending"],
        inference=True,
        allowed_gpus=allowed_gpus,
    )
    ledger.unlink()
    value = yaml.safe_load(queue.read_text(encoding="utf-8"))
    value["required_max_jobs_per_gpu"] = 2
    queue.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="timing-isolated queue policy changed"):
        module.inspect_ledger(
            queue,
            ledger,
            ["infer.one"],
            allow_terminal_failures=False,
            require_timing_attestation=True,
            allowed_gpus=allowed_gpus,
        )


def test_configuration_rejects_nonisolated_inference(tmp_path: Path) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text())
    argv = payload["actions"]["run_inference"]["argv"]
    argv[argv.index("--max-per-gpu") + 1] = "2"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="max-per-gpu must be 1"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_configuration_rejects_action_type_bypass(tmp_path: Path) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["actions"]["run_inference"]["type"] = "command"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="action type must be guarded_queue"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_configuration_rejects_static_inference_seed_universe(tmp_path: Path) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    action = payload["actions"]["run_inference"]
    action.pop("expected_names_from_step")
    action["expected_names"] = ["infer.drunet.seed0", "infer.drunet.seed1"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="derive from the frozen prepared queue"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


@pytest.mark.parametrize("step", ["prepare_seed0", "authorize_seed12"])
def test_control_plane_authorizers_require_delivery_pin(
    tmp_path: Path, step: str
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["actions"][step]["code_pin"] = "training"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="attested delivery runner pin"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_scientific_selector_requires_frozen_training_pin(tmp_path: Path) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["actions"]["select_full"]["code_pin"] = "delivery"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen training pin"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_delivery_authorizer_requires_exact_raw_training_queue_binding(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prepare = payload["actions"]["prepare_seed0"]
    prepare["cross_step_inputs"] = [
        item
        for item in prepare["cross_step_inputs"]
        if item["option"] != "--remaining-seed0-jobs"
    ]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen training output"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_seed12_barrier_requires_exact_locked_plan_binding(tmp_path: Path) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    authorize = payload["actions"]["authorize_seed12"]
    authorize["cross_step_inputs"] = [
        item
        for item in authorize["cross_step_inputs"]
        if item["option"] != "--seed12-plan"
    ]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="hash-bind barrier input"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_audited_sufs_is_static_and_distinct_from_generated_selection(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audited = payload["audited_sufs_selection"]
    # Derive the generated path from the registered selection output rather
    # than relying on the argv position in the actual assertion.
    selection_argv = payload["actions"]["select_full"]["argv"]
    selection_path = Path(
        selection_argv[selection_argv.index("--selection-out") + 1]
    )
    generated = str(selection_path.with_name("sufs_ft_selection.json"))
    assert audited["path"] != generated
    assert generated in payload["actions"]["select_full"]["outputs"]
    for consumer in ("prepare_seed0", "prepare_inference"):
        action = payload["actions"][consumer]
        index = action["argv"].index("--sufs-selection") + 1
        assert action["argv"][index] == audited["path"]
        assert not any(
            item.get("option") == "--sufs-selection"
            for item in action.get("cross_step_inputs", [])
        )
    module.Supervisor(config_path, state, log, poll_seconds=1)


def test_generated_sufs_cannot_replace_independently_audited_selection(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    select_argv = payload["actions"]["select_full"]["argv"]
    generated = str(
        Path(select_argv[select_argv.index("--selection-out") + 1]).with_name(
            "sufs_ft_selection.json"
        )
    )
    generated_path = Path(generated)
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text('{"generated": true}\n', encoding="utf-8")
    prepare = payload["actions"]["prepare_seed0"]
    prepare["argv"][prepare["argv"].index("--sufs-selection") + 1] = generated
    binding = next(
        item
        for item in prepare["bound_inputs"]
        if item.get("option") == "--sufs-selection"
    )
    binding.update(path=generated, sha256=sha256_file(generated_path))
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="consume the audited SUFS selection"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


@pytest.mark.parametrize(
    "key", ["SEGBENCH_EXPECTED_COMMIT_MAP", "SEGBENCH_SEGFORMER_B2_DIR"]
)
def test_selection_environment_is_mandatory(tmp_path: Path, key: str) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["actions"]["select_full"]["env"].pop(key)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="two frozen selection environment inputs|bound input env_key/path differs",
    ):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_expected_commit_env_is_hash_bound_and_rechecked_before_launch(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    supervisor = module.Supervisor(config_path, state, log, poll_seconds=1)
    action = supervisor.config["actions"]["select_full"]
    commit_map = Path(action["env"]["SEGBENCH_EXPECTED_COMMIT_MAP"])
    commit_map.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(module.SupervisorError, match="immutable bound input changed"):
        supervisor._verify_action_inputs("select_full", action)


def test_segformer_environment_must_agree_with_both_frozen_sweep_jobs(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    queue_path = Path(payload["sweep"]["queue"])
    queue = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    job = next(item for item in queue["jobs"] if "segformer_sub" in item["name"])
    job["env"]["SEGBENCH_SEGFORMER_B2_DIR"] = str(tmp_path / "wrong")
    job["job_identity_sha256"] = module._queue_job_identity(job)
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False), encoding="utf-8")
    payload["sweep"]["queue_sha256"] = sha256_file(queue_path)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen snapshot environment changed"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_segformer_snapshot_files_are_rechecked_before_selection_launch(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    supervisor = module.Supervisor(config_path, state, log, poll_seconds=1)
    action = supervisor.config["actions"]["select_full"]
    model = Path(
        supervisor.config["segformer_snapshot"]["files"]["pytorch_model.bin"][
            "path"
        ]
    )
    model.write_bytes(b"changed")
    with pytest.raises(
        module.SupervisorError, match="frozen environment changed before launch"
    ):
        supervisor._verify_action_inputs("select_full", action)


def test_absolute_child_priority_stays_ten_from_an_already_nice_parent() -> None:
    script = f"""
import importlib.util
import os
from pathlib import Path
path = Path({str(ROOT / 'scripts' / 'run_stage3_supervisor.py')!r})
spec = importlib.util.spec_from_file_location('priority_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
os.setpriority(os.PRIO_PROCESS, 0, 10)
assert os.getpriority(os.PRIO_PROCESS, 0) == 10
module._set_child_priority()
assert os.getpriority(os.PRIO_PROCESS, 0) == 10
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'scripts'}"},
    )


def test_validate_only_cli_prints_bound_config_digest_without_launching(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pin = Path(payload["code_pins"]["training"]["repo_root"])
    (pin / "tracked.txt").write_text("synthetic clean pin\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=pin, check=True)
    subprocess.run(["git", "add", "."], cwd=pin, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=segbench-test",
            "-c",
            "user.email=segbench-test@example.invalid",
            "commit",
            "-qm",
            "synthetic pin",
        ],
        cwd=pin,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=pin, text=True
    ).strip()
    for value in payload["code_pins"].values():
        value["expected_commit"] = commit
    prepare = payload["actions"]["prepare_inference"]["argv"]
    for option in (
        "--expected-planner-git-commit",
        "--inference-git-commit",
        "--finalizer-git-commit",
    ):
        prepare[prepare.index(option) + 1] = commit
    finalize = payload["actions"]["finalize"]["argv"]
    for option in (
        "--expected-finalizer-git-commit",
        "--expected-planner-git-commit",
    ):
        finalize[finalize.index(option) + 1] = commit
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_stage3_supervisor.py"),
            "--config",
            str(config_path),
            "--state",
            str(state),
            "--log",
            str(log),
            "--validate-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'scripts'}"},
    )
    report = json.loads(result.stdout)
    assert report == {
        "status": "validated",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "state": str(state),
    }
    assert state.is_file()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("max_jobs_per_gpu", 0, "max_jobs_per_gpu"),
        ("max_memory_fraction", 0.0, "max_memory_fraction"),
        ("max_memory_fraction", float("nan"), "max_memory_fraction"),
        ("allowed_gpus", [-1], "allowed_gpus"),
    ],
)
def test_configuration_rejects_invalid_runtime_resource_caps(
    tmp_path: Path, key: str, value: Any, message: str
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["resources"][key] = value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_configuration_rejects_auxiliary_output_overlap(tmp_path: Path) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text())
    argv = payload["actions"]["prepare_inference"]["argv"]
    deterministic = argv[argv.index("--masks-root") + 1]
    argv[argv.index("--aux-variance-root") + 1] = str(Path(deterministic) / "aux")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must be disjoint"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_configuration_requires_exact_declared_delivery(tmp_path: Path) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    payload = yaml.safe_load(config_path.read_text())
    payload["actions"]["finalize"]["outputs"] = [
        str(tmp_path / "restricted" / "bundle" / "wrong.json")
    ]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly bundle-out/delivery.json"):
        module.Supervisor(config_path, state, log, poll_seconds=1)


def test_prepare_semantics_accept_exact_real_planner_fields(
    tmp_path: Path,
) -> None:
    """Exercise the real semantic validator; do not hide it behind a mock."""
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    supervisor = module.Supervisor(config_path, state, log, poll_seconds=1)
    action = supervisor.config["actions"]["prepare_inference"]
    gate_path = Path(module._option(action["argv"], "--gate-out"))
    plan_path = Path(module._option(action["argv"], "--plan-out"))
    queue_path = Path(module._option(action["argv"], "--queue-out"))
    prep_path = Path(module._option(action["argv"], "--prep-bundle-out"))
    aux_plan_path = Path(module._option(action["argv"], "--aux-plan-out"))
    aux_queue_path = Path(module._option(action["argv"], "--aux-queue-out"))
    aux_plan_path.parent.mkdir(parents=True, exist_ok=True)
    aux_plan_path = Path(module._option(action["argv"], "--aux-plan-out"))
    aux_queue_path = Path(module._option(action["argv"], "--aux-queue-out"))
    aux_plan_path.parent.mkdir(parents=True, exist_ok=True)
    aux_card_path = Path(module._option(action["argv"], "--aux-card-out"))
    allowed_gpus = supervisor.config["resources"]["allowed_gpus"]
    claims = {
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
    }
    gate_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stage3_native_training_readiness_gate",
                "status": "PASS_WITH_WARNINGS",
                "methods": {
                    "drunet": {
                        "valid_seeds": [0, 2],
                        "failed_seeds": [1],
                        "missing_seeds": [],
                    }
                },
                **claims,
            }
        ),
        encoding="utf-8",
    )
    gate_sha = sha256_file(gate_path)
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stage3_native_inference_plan",
                "gate_sha256": gate_sha,
                **claims,
            }
        ),
        encoding="utf-8",
    )
    plan_sha = sha256_file(plan_path)
    inference_jobs = [
        _job(
            f"infer.drunet.seed{seed}",
            tmp_path
            / "restricted"
            / "cards"
            / "deterministic"
            / f"seed{seed}.json",
        )
        for seed in (0, 2)
    ]
    queue_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "phase": "stage3_native_test_inference",
                "launch_authorized": True,
                "gate_sha256": gate_sha,
                "plan_sha256": plan_sha,
                "required_max_jobs_per_gpu": 1,
                "required_launcher_args": ["--max-per-gpu", "1"],
                "required_gpu_indices": allowed_gpus,
                # Seed 1 is terminally failed and therefore intentionally
                # absent despite the gate passing with warnings.
                "jobs": inference_jobs,
                **claims,
            }
        ),
        encoding="utf-8",
    )
    aux_ids = [f"synthetic-test-{index:02d}" for index in range(20)]
    aux_plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stage3_drunet_mc_dropout_plan",
                "test_image_ids": aux_ids,
                "image_id_set_sha256": canonical_sha256(aux_ids),
                "expected_shapes": {image_id: [2, 3] for image_id in aux_ids},
                "ordered_source_images_sha256": canonical_sha256(
                    [{"image_id": value, "source_sha256": "a" * 64} for value in aux_ids]
                ),
                "mc_dropout_passes": 20,
                "timing_excluded": True,
                "mask_outputs_forbidden": True,
                **claims,
            }
        ),
        encoding="utf-8",
    )
    aux_plan_sha = sha256_file(aux_plan_path)
    aux_job = _job("stage3.drunet.mc_dropout_aux", aux_card_path)
    aux_queue_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "phase": "stage3_drunet_mc_dropout_aux",
                "launch_authorized": True,
                "plan_sha256": aux_plan_sha,
                "required_max_jobs_per_gpu": 1,
                "required_launcher_args": ["--max-per-gpu", "1"],
                "required_gpu_indices": allowed_gpus,
                "mc_dropout_passes": 20,
                "timing_excluded": True,
                "mask_outputs_forbidden": True,
                "jobs": [aux_job],
                **claims,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    prep_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stage3_native_frozen_preparation_bundle",
                "gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
                "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
                "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
                "aux_plan": {
                    "path": str(aux_plan_path),
                    "sha256": sha256_file(aux_plan_path),
                },
                "aux_queue": {
                    "path": str(aux_queue_path),
                    "sha256": sha256_file(aux_queue_path),
                },
                "runtime_policy": {
                    "max_project_ram_gib": supervisor.config["resources"][
                        "max_ram_gib"
                    ]
                },
                **claims,
            }
        ),
        encoding="utf-8",
    )
    evidence = supervisor._semantic_outputs("prepare_inference", action)
    assert evidence["gate_status"] == "PASS_WITH_WARNINGS"
    assert evidence["prep_bundle_sha256"] == sha256_file(prep_path)
    assert evidence["auxiliary_image_id_set_sha256"] == canonical_sha256(aux_ids)
    assert evidence["inference_expected_names"] == [
        "infer.drunet.seed0",
        "infer.drunet.seed2",
    ]
    assert "infer.drunet.seed1" not in evidence["inference_expected_names"]

    # The guarded-queue completion check consumes that frozen dynamic
    # universe, rather than a prospective all-seeds list from configuration.
    run_action = supervisor.config["actions"]["run_inference"]
    ledger_path = Path(run_action["ledger"])
    entries = {}
    for job in inference_jobs:
        card = Path(job["completion_artifact"])
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(json.dumps({"job": job["name"]}), encoding="utf-8")
        entries[job["name"]] = {
            "identity_sha256": job["job_identity_sha256"],
            "status": "completed",
            "attempts": 1,
            "ordinary_retries": 0,
            "oom_retries": 0,
            "preemptions": 0,
            "artifact_sha256": sha256_file(card),
        }
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "queue_sha256": sha256_file(queue_path),
                "launcher_attestation": {
                    "schema_version": 1,
                    "queue_phase": "stage3_native_test_inference",
                    "queue_sha256": sha256_file(queue_path),
                    "gpus": allowed_gpus,
                    "max_per_gpu": 1,
                    "memory_fraction": 0.3,
                    "required_launcher_args": ["--max-per-gpu", "1"],
                },
                "jobs": entries,
            }
        ),
        encoding="utf-8",
    )
    supervisor.state["steps"]["prepare_inference"]["evidence"] = evidence
    terminal = supervisor._queue_evidence("run_inference", run_action)
    assert terminal["terminal"] is True
    assert terminal["completed"] == evidence["inference_expected_names"]


def test_prepare_semantics_rejects_obsolete_blindness_field(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    supervisor = module.Supervisor(config_path, state, log, poll_seconds=1)
    action = supervisor.config["actions"]["prepare_inference"]
    gate_path = Path(module._option(action["argv"], "--gate-out"))
    plan_path = Path(module._option(action["argv"], "--plan-out"))
    queue_path = Path(module._option(action["argv"], "--queue-out"))
    prep_path = Path(module._option(action["argv"], "--prep-bundle-out"))
    aux_plan_path = Path(module._option(action["argv"], "--aux-plan-out"))
    aux_queue_path = Path(module._option(action["argv"], "--aux-queue-out"))
    aux_plan_path.parent.mkdir(parents=True, exist_ok=True)
    obsolete = {"ground_truth_blind": True, "test_data_opened": False}
    gate_path.write_text(
        json.dumps(
            {
                "kind": "stage3_native_training_readiness_gate",
                "status": "PASS",
                **obsolete,
            }
        ),
        encoding="utf-8",
    )
    plan_path.write_text(
        json.dumps({"kind": "stage3_native_inference_plan", **obsolete}),
        encoding="utf-8",
    )
    queue_path.write_text(
        yaml.safe_dump(
            {
                "phase": "stage3_native_test_inference",
                "launch_authorized": True,
                **obsolete,
            }
        ),
        encoding="utf-8",
    )
    prep_path.write_text(
        json.dumps(
            {
                "kind": "stage3_native_frozen_preparation_bundle",
                "runtime_policy": {
                    "max_project_ram_gib": supervisor.config["resources"][
                        "max_ram_gib"
                    ]
                },
                **obsolete,
            }
        ),
        encoding="utf-8",
    )
    aux_plan_path.write_text("{}\n", encoding="utf-8")
    aux_queue_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(module.SupervisorError, match="not GT-blind"):
        supervisor._semantic_outputs("prepare_inference", action)


def test_auxiliary_semantics_authenticate_exact_variance_universe(
    tmp_path: Path,
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    supervisor = module.Supervisor(config_path, state, log, poll_seconds=1)
    preparation = supervisor.config["actions"]["prepare_inference"]
    plan_path = Path(module._option(preparation["argv"], "--aux-plan-out"))
    card_path = Path(module._option(preparation["argv"], "--aux-card-out"))
    variance_root = Path(module._option(preparation["argv"], "--aux-variance-root"))
    reserved_masks = Path(module._option(preparation["argv"], "--aux-masks-root"))
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    variance_root.mkdir(parents=True, exist_ok=True)
    image_ids = [f"synthetic-test-{index:02d}" for index in range(20)]
    source_digest = canonical_sha256(
        [{"image_id": value, "source_sha256": "a" * 64} for value in image_ids]
    )
    identities = {
        name: {
            "path": str(tmp_path / "restricted" / f"{name}.synthetic"),
            "sha256": canonical_sha256(name),
        }
        for name in ("config", "checkpoint", "manifest", "splits")
    }
    plan = {
        "schema_version": 1,
        "kind": "stage3_drunet_mc_dropout_plan",
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "timing_excluded": True,
        "mask_outputs_forbidden": True,
        "mc_dropout_passes": 20,
        "test_image_ids": image_ids,
        "image_id_set_sha256": canonical_sha256(image_ids),
        "ordered_source_images_sha256": source_digest,
        "expected_shapes": {image_id: [2, 3] for image_id in image_ids},
        "run": {"method": "drunet", "seed": 0},
        "identities": identities,
        "variance_out": str(variance_root),
        "reserved_masks_root": str(reserved_masks),
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    records = []
    for index, image_id in enumerate(image_ids):
        path = variance_root / f"{image_id}_variance.npy"
        np.save(path, np.full((2, 3), index / 100.0, dtype=np.float32))
        records.append(
            {
                "image_id": image_id,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": [2, 3],
                "dtype": "float32",
            }
        )
    card = {
        "schema_version": 1,
        "kind": "stage3_drunet_mc_dropout_completion_card",
        "method": "drunet",
        "seed": 0,
        "plan_sha256": sha256_file(plan_path),
        "ground_truth_opened": False,
        "mask_outputs_written": False,
        "timing_excluded": True,
        "mc_dropout_passes": 20,
        "image_id_set_sha256": canonical_sha256(image_ids),
        "ordered_source_images_sha256": source_digest,
        "variance_set": {
            "count": 20,
            "records": records,
            "sha256": canonical_sha256(records),
        },
        "identities": identities,
        "provenance": {
            "capture_timing": "process_start",
            "git_commit": "a" * 40,
            "git_dirty": False,
        },
    }
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card), encoding="utf-8")
    evidence = supervisor._auxiliary_evidence()
    assert evidence["auxiliary_variance_count"] == 20
    assert evidence["auxiliary_variance_set_sha256"] == canonical_sha256(records)
    (variance_root / "unexpected.npy").write_bytes(b"stale")
    with pytest.raises(module.SupervisorError, match="extra files"):
        supervisor._auxiliary_evidence()


def test_state_machine_orders_barriers_and_allows_only_seed12_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config_path, state, log = _config(tmp_path, module)
    supervisor = module.Supervisor(config_path, state, log, poll_seconds=1)
    launched: list[str] = []

    def launch(step: str, action: Dict[str, Any]) -> int:  # noqa: ARG001
        launched.append(step)
        return 1 if step == "run_seed12" else 0

    monkeypatch.setattr(supervisor, "_launch", launch)
    monkeypatch.setattr(
        supervisor,
        "_queue_evidence",
        lambda step, action: {
            "terminal": True,
            "completed": ["ok"],
            "failed": ["failed-seed"] if step == "run_seed12" else [],
            "nonterminal": {},
        },
    )
    monkeypatch.setattr(supervisor, "_output_evidence", lambda action: {})
    monkeypatch.setattr(
        supervisor,
        "_semantic_outputs",
        lambda step, action: {"outputs": {}, "synthetic": step},
    )
    monkeypatch.setattr(
        supervisor,
        "_auxiliary_evidence",
        lambda: {"synthetic_auxiliary": True},
    )
    while supervisor.state["status"] != "completed":
        supervisor.advance_once()
    assert launched == list(module.STEP_ORDER)
    assert supervisor.state["steps"]["run_seed12"]["evidence"]["failed"] == [
        "failed-seed"
    ]
    assert module.STEP_ORDER.index("prepare_inference") > module.STEP_ORDER.index(
        "run_seed12"
    )
    assert module.STEP_ORDER.index("finalize") > module.STEP_ORDER.index(
        "run_inference"
    )
    assert module.STEP_ORDER.index("run_auxiliary") > module.STEP_ORDER.index(
        "run_inference"
    )
    assert module.STEP_ORDER.index("finalize") > module.STEP_ORDER.index(
        "run_auxiliary"
    )


def test_source_has_no_ground_truth_reader() -> None:
    source = (ROOT / "scripts/run_stage3_supervisor.py").read_text(encoding="utf-8")
    assert "read_mask(" not in source
    assert "cv2.imread" not in source
    assert "Image.open" not in source
