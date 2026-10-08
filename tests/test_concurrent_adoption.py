"""Synthetic checks for the Stage-3 concurrency adoption handoff."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from segbench.artifact_identity import canonical_sha256, sha256_file


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name + "_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity(item):
    return canonical_sha256(
        {
            "name": item["name"],
            "argv": item["argv"],
            "env": item.get("env", {}),
            "completion_artifact": item["completion_artifact"],
            "resources": item["resources"],
        }
    )


def _source(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,image_path,mask_path,subject_id,subset,ppi\n", encoding="utf-8")
    entrypoint = tmp_path / "train_seg.py"
    entrypoint.write_text("# synthetic legacy entrypoint\n", encoding="utf-8")
    jobs = []
    for name in ("captured", "pending"):
        config = tmp_path / f"{name}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "data": {"target_ppi": 500},
                    "training": {
                        "num_workers": 4,
                        "batch_size": 4,
                        "gradient_accumulation": 1,
                        "threshold_candidates": [0.3, 0.4, 0.5, 0.6, 0.7],
                    },
                    "runtime": {"memory_fraction": 0.3},
                }
            ),
            encoding="utf-8",
        )
        item = {
            "name": name,
            "argv": [
                "python",
                str(entrypoint),
                "--config",
                str(config),
                "--manifest",
                str(manifest),
                "--resume",
                "auto",
                "--name",
                name,
            ],
            "env": {},
            "completion_artifact": str(tmp_path / name / "run.json"),
            "resources": {"host_ram_gib": 40.0},
        }
        item["job_identity_sha256"] = _identity(item)
        jobs.append(item)
    queue = tmp_path / "source.yaml"
    queue.write_text(
        yaml.safe_dump(
            {
                "launch_authorized": True,
                "phase": "learning_rate_sweep_remaining_native",
                "jobs": jobs,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "source.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queue_sha256": sha256_file(queue),
                "jobs": {
                    "captured": {
                        "identity_sha256": jobs[0]["job_identity_sha256"],
                        "status": "running",
                        "attempts": 1,
                        "ordinary_retries": 0,
                        "oom_retries": 0,
                        "preemptions": 0,
                    },
                    "pending": {
                        "identity_sha256": jobs[1]["job_identity_sha256"],
                        "status": "pending",
                        "attempts": 0,
                        "ordinary_retries": 0,
                        "oom_retries": 0,
                        "preemptions": 0,
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "workers": {
                    "captured": {
                        "pid": 4242,
                        "start_ticks": 123,
                        "argv_sha256": canonical_sha256(jobs[0]["argv"]),
                        "completion_artifact": jobs[0]["completion_artifact"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return queue, ledger, capture, jobs


def _args(tmp_path: Path, queue: Path, ledger: Path, capture: Path):
    return argparse.Namespace(
        source_queue=queue,
        source_ledger=ledger,
        capture=capture,
        superseded_supervisor_config=tmp_path / "old-supervisor.yaml",
        supplemental_queue=tmp_path / "supplemental.yaml",
        plan_out=tmp_path / "external-plan.json",
        code_root=tmp_path / "code-root",
        expected_code_commit="a" * 40,
        cache_dir=tmp_path / "cache",
        threshold_amendment=tmp_path / "threshold-amendment.json",
        expected_threshold_amendment_sha256=(
            sha256_file(tmp_path / "threshold-amendment.json")
            if (tmp_path / "threshold-amendment.json").is_file()
            else "0" * 64
        ),
        generated_config_dir=tmp_path / "generated-configs",
        allowed_gpus="0,2",
        max_per_gpu=2,
        memory_fraction=0.3,
        reservation_gib=12.0,
        max_project_ram_gib=64.0,
        max_cpu_cores=40,
        max_data_workers_per_job=8,
        resource_root=tmp_path,
        min_disk_free_fraction=0.15,
    )


def _mock_amendment_inputs(
    module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    receipt_path = cache / "receipt.json"
    receipt = {
        "manifest_sha256": "m" * 64,
        "target_ppi": 500.0,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    amendment = tmp_path / "threshold-amendment.json"
    amendment.write_text(json.dumps({"kind": "synthetic"}), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_git_identity",
        lambda root, commit: {
            "root": str((tmp_path / "code-root").resolve()),
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "files_sha256": {"segbench/scripts/train_seg.py": "c" * 64},
        },
    )
    monkeypatch.setattr(
        module,
        "_validate_cache_receipt",
        lambda cache_dir, manifest: (receipt, receipt_path),
    )
    monkeypatch.setattr(module, "_validate_threshold_amendment", lambda path: {})


def test_cache_receipt_requires_production_20_image_identity_gate(tmp_path: Path) -> None:
    module = _load("prepare_concurrent_adoption")
    manifest = tmp_path / "manifest.csv"
    source = tmp_path / "source"
    source.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "images").mkdir()
    (cache / "masks" / "mask_path").mkdir(parents=True)
    (cache / "masks" / "quality_mask_path").mkdir(parents=True)
    records = []
    manifest_lines = ["image_id,image_path,mask_path,quality_mask_path"]
    for index in range(86):
        image_id = f"i{index:02d}"
        source_image = source / f"{image_id}.image"
        source_mask = source / f"{image_id}.roi"
        source_quality = source / f"{image_id}.quality"
        source_image.write_bytes(f"source-image-{index}".encode())
        source_mask.write_bytes(f"source-mask-{index}".encode())
        if index < 20:
            source_quality.write_bytes(f"source-quality-{index}".encode())
        manifest_lines.append(
            f"{image_id},{source_image},{source_mask},"
            + (str(source_quality) if index < 20 else "")
        )
        image = cache / "images" / f"{image_id}.npy"
        mask = cache / "masks" / "mask_path" / f"{image_id}.npy"
        quality = cache / "masks" / "quality_mask_path" / f"{image_id}.npy"
        image.write_bytes(f"image-{index}".encode())
        mask.write_bytes(f"mask-{index}".encode())
        if index < 20:
            quality.write_bytes(f"quality-{index}".encode())
        records.append(
            {
                "image_id": image_id,
                "shape": [8, 9],
                "image_sha256": sha256_file(image),
                "source_image_sha256": sha256_file(source_image),
                "mask_sha256": {
                    "mask_path": sha256_file(mask),
                    "quality_mask_path": sha256_file(quality) if index < 20 else None,
                },
                "source_mask_sha256": {
                    "mask_path": sha256_file(source_mask),
                    "quality_mask_path": sha256_file(source_quality) if index < 20 else None,
                },
            }
        )
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "kind": "uint8_500ppi_preprocessing_cache",
        "manifest_sha256": sha256_file(manifest),
        "target_ppi": 500.0,
        "mask_columns": ["mask_path", "quality_mask_path"],
        "image_format": "uint8_npy",
        "mask_format": "uint8_0_255_npy",
        "record_count": len(records),
        "records": records,
        "verification": {
            "requested_count": 20,
            "verified_count": 20,
            "array_identity": True,
            "image_ids": [record["image_id"] for record in records[:20]],
        },
        "debug_limited": False,
    }
    (cache / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    observed, path = module._validate_cache_receipt(cache, manifest)
    assert observed == receipt
    assert path == cache / "receipt.json"
    receipt["mask_columns"] = ["mask_path"]
    (cache / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="ROI and quality"):
        module._validate_cache_receipt(cache, manifest)
    receipt["mask_columns"] = ["mask_path", "quality_mask_path"]
    receipt["verification"]["verified_count"] = 19
    (cache / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="20-image identity"):
        module._validate_cache_receipt(cache, manifest)


def test_prepare_preserves_captured_and_amends_only_pending_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("prepare_concurrent_adoption")
    queue, ledger, capture, jobs = _source(tmp_path)
    monkeypatch.setattr(module.orphan, "reconciliation_preflight", lambda *args: {})
    monkeypatch.setattr(module, "conflicting_controllers", lambda *args: {})
    monkeypatch.setattr(
        module.orphan,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": jobs[0]["argv"],
            "argv_sha256": canonical_sha256(jobs[0]["argv"]),
        },
    )
    monkeypatch.setattr(
        module,
        "_environment",
        lambda pid: {
            "CUDA_VISIBLE_DEVICES": "0",
            "SEG_PREEMPT_FILE": str(tmp_path / "captured.preempt"),
            "SEG_GPU_MEMORY_FRACTION": "0.3",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    )
    monkeypatch.setattr(module, "_pss_kib", lambda pid: 2 * 1024**2)
    monkeypatch.setattr(module, "_status_kib", lambda pid, key: 4 * 1024**2)
    _mock_amendment_inputs(module, tmp_path, monkeypatch)
    args = _args(tmp_path, queue, ledger, capture)
    plan = module.prepare(args)
    supplement = yaml.safe_load(args.supplemental_queue.read_text())
    assert [item["name"] for item in supplement["jobs"]] == ["captured", "pending"]
    pending = supplement["jobs"][1]
    assert pending["argv"] != jobs[1]["argv"]
    assert pending["argv"][-4:] == [
        "--cache-dir",
        str(args.cache_dir.resolve()),
        "--threshold-amendment",
        str(args.threshold_amendment.resolve()),
    ]
    assert Path(pending["argv"][1]).name == "train_seg.py"
    assert pending["adoption_role"] == "cache_amended_pending_job"
    assert pending["env"] == jobs[1]["env"]
    assert pending["completion_artifact"] == jobs[1]["completion_artifact"]
    assert pending["resources"] == {"host_ram_gib": 12.0, "cpu_workers": 8}
    captured = supplement["jobs"][0]
    assert captured["argv"] == jobs[0]["argv"]
    assert captured["resources"] == {"host_ram_gib": 12.0, "cpu_workers": 4}
    assert pending["source_job_identity_sha256"] == jobs[1]["job_identity_sha256"]
    assert plan["workers"][0]["gpu"] == 0
    assert plan["reservation_evidence"]["aggregate_slot_reservation_gib"] == 48.0
    assert plan["cpu_budget"]["aggregate_data_workers"] == 32
    assert plan["cpu_budget"]["headroom_for_controller_and_reporting"] == 8
    assert supplement["execution_binding"]["dataloader"] == {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    generated = yaml.safe_load(Path(pending["argv"][pending["argv"].index("--config") + 1]).read_text())
    assert {
        key: generated["training"][key]
        for key in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor")
    } == supplement["execution_binding"]["dataloader"]
    assert generated["runtime"]["queue_batch_contract_sha256"] == supplement[
        "execution_binding"
    ]["generated_batch_contract_sha256"]["pending"]
    assert plan["ground_truth_opened"] is False
    assert plan["result_metrics_opened"] is False


def test_prepare_refuses_while_old_controller_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("prepare_concurrent_adoption")
    queue, ledger, capture, _ = _source(tmp_path)
    monkeypatch.setattr(module.orphan, "reconciliation_preflight", lambda *args: {})
    monkeypatch.setattr(module, "conflicting_controllers", lambda *args: {99: ["old"]})
    with pytest.raises(RuntimeError, match="still live"):
        module.prepare(_args(tmp_path, queue, ledger, capture))


def test_finalize_builds_mixed_queue_bound_terminal_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = _load("prepare_concurrent_adoption")
    finalize = _load("finalize_concurrent_adoption")
    queue, ledger, capture, jobs = _source(tmp_path)
    monkeypatch.setattr(prepare.orphan, "reconciliation_preflight", lambda *args: {})
    monkeypatch.setattr(prepare, "conflicting_controllers", lambda *args: {})
    monkeypatch.setattr(
        prepare.orphan,
        "process_identity",
        lambda pid: {
            "pid": pid,
            "start_ticks": 123,
            "argv": jobs[0]["argv"],
            "argv_sha256": canonical_sha256(jobs[0]["argv"]),
        },
    )
    monkeypatch.setattr(
        prepare,
        "_environment",
        lambda pid: {
            "CUDA_VISIBLE_DEVICES": "0",
            "SEG_PREEMPT_FILE": str(tmp_path / "captured.preempt"),
            "SEG_GPU_MEMORY_FRACTION": "0.3",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    )
    monkeypatch.setattr(prepare, "_pss_kib", lambda pid: 1024)
    monkeypatch.setattr(prepare, "_status_kib", lambda pid, key: 2048)
    _mock_amendment_inputs(prepare, tmp_path, monkeypatch)
    pargs = _args(tmp_path, queue, ledger, capture)
    prepare.prepare(pargs)

    for item in jobs:
        card = Path(item["completion_artifact"])
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(json.dumps({"name": item["name"]}), encoding="utf-8")
    supplemental_queue = yaml.safe_load(pargs.supplemental_queue.read_text())
    supplemental_jobs = {item["name"]: item for item in supplemental_queue["jobs"]}
    supplemental_job = supplemental_jobs["pending"]
    supplemental_ledger = tmp_path / "supplemental-ledger.json"
    supplemental_ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queue_sha256": sha256_file(pargs.supplemental_queue),
                "jobs": {
                    "captured": {
                        "identity_sha256": supplemental_jobs["captured"][
                            "job_identity_sha256"
                        ],
                        "status": "completed",
                        "attempts": 0,
                        "ordinary_retries": 0,
                        "oom_retries": 0,
                        "preemptions": 0,
                        "artifact_sha256": sha256_file(
                            Path(supplemental_jobs["captured"]["completion_artifact"])
                        ),
                    },
                    "pending": {
                        "identity_sha256": supplemental_job["job_identity_sha256"],
                        "status": "completed",
                        "attempts": 1,
                        "ordinary_retries": 0,
                        "oom_retries": 0,
                        "preemptions": 0,
                        "artifact_sha256": sha256_file(
                            Path(supplemental_job["completion_artifact"])
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    external_proof = tmp_path / "external-proof.json"
    external_proof.write_text(
        json.dumps(
            {
                "kind": "guarded_queue_external_worker_completion",
                "external_workers_plan_sha256": sha256_file(pargs.plan_out),
                "supplemental_queue_sha256": sha256_file(pargs.supplemental_queue),
                "supplemental_ledger_sha256": sha256_file(supplemental_ledger),
                "workers": {
                    "captured": {
                        "pid": 4242,
                        "start_ticks": 123,
                        "argv_sha256": canonical_sha256(jobs[0]["argv"]),
                        "artifact_sha256": sha256_file(
                            Path(jobs[0]["completion_artifact"])
                        ),
                        "gpu": 0,
                        "outcome": "completed_externally",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(finalize.orphan, "reconciliation_preflight", lambda *args: {})
    monkeypatch.setattr(
        finalize.guarded,
        "validate_adopted_queue_runtime",
        lambda *args, **kwargs: supplemental_queue["execution_binding"],
    )
    fargs = SimpleNamespace(
        source_queue=queue,
        source_ledger=ledger,
        capture=capture,
        supplemental_queue=pargs.supplemental_queue,
        supplemental_ledger=supplemental_ledger,
        external_plan=pargs.plan_out,
        external_proof=external_proof,
        merged_ledger_out=tmp_path / "merged.json",
        proof_out=tmp_path / "merge-proof.json",
    )
    merged, proof = finalize.finalize(fargs)
    assert merged["queue_sha256"] == sha256_file(pargs.supplemental_queue)
    assert {entry["status"] for entry in merged["jobs"].values()} == {"completed"}
    assert merged["jobs"]["pending"]["identity_sha256"] == supplemental_job[
        "job_identity_sha256"
    ]
    assert merged["jobs"]["pending"]["source_job_identity_sha256"] == jobs[1][
        "job_identity_sha256"
    ]
    assert proof["terminal_completed_count"] == 2
    assert proof["ground_truth_opened"] is False
