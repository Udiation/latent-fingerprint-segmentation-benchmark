"""Synthetic, ground-truth-free tests for the Stage 3 inference barrier."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd
import pytest
import yaml

from segbench.artifact_identity import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _module():
    path = ROOT / "scripts" / "prepare_stage3_inference.py"
    spec = importlib.util.spec_from_file_location("stage3_inference_planner_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _provenance(commit: str) -> Dict[str, Any]:
    return {
        "capture_timing": "process_start",
        "git_commit": commit,
        "git_dirty": False,
    }


def _full_run(
    module: Any, tmp_path: Path, *, method: str, seed: int, commit: str
) -> tuple[Dict[str, Any], Path]:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "full" / method / f"seed{seed}"
    run_dir.mkdir(parents=True)
    config = tmp_path / f"{method}.yaml"
    config.write_text("model:\n  method: synthetic\n", encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"best-checkpoint")
    (run_dir / "last.pt").write_bytes(b"last-checkpoint")
    manifest_sha = "1" * 64
    splits_sha = "2" * 64
    run = {
        "method": method,
        "seed": seed,
        "epochs_requested": module.planner.FULL_EPOCHS,
        "epochs_completed": 2,
        "manifest_sha256": manifest_sha,
        "split_sha256": splits_sha,
        "source_config_sha256": sha256_file(config),
        "learning_rate": 3e-4,
        "debug_limited": False,
        "checkpoint_sha256": {
            "best.pt": sha256_file(run_dir / "best.pt"),
            "last.pt": sha256_file(run_dir / "last.pt"),
        },
        "best_validation_dice": 0.7,
        "selected_validation_threshold": 0.5,
        "parameter_count": {"total": 1234},
        "provenance": _provenance(commit),
    }
    (run_dir / "run.json").write_text(
        json.dumps(run, sort_keys=True) + "\n", encoding="utf-8"
    )
    selection = {
        "manifest_sha256": manifest_sha,
        "splits_sha256": splits_sha,
        "methods": {
            method: {
                "full_config": str(config),
                "full_config_sha256": sha256_file(config),
                "selected_learning_rate": 3e-4,
            }
        },
    }
    return selection, runs_root


def _guarded_job(module: Any, name: str, completion: Path) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "name": name,
        "argv": ["python", "synthetic.py", "--run", name],
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


def test_manifest_metadata_count_never_opens_gt(tmp_path: Path) -> None:
    module = _module()
    heldout_image = tmp_path / "heldout.png"
    heldout_image.write_bytes(b"source-image-bytes")
    nonprint_image = tmp_path / "nonprint.png"
    nonprint_image.write_bytes(b"excluded-source-bytes")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_id": "heldout-1",
                "image_path": str(heldout_image),
                "source_sha256": sha256_file(heldout_image),
                "mask_path": "/must-not-open/roi.png",
                "quality_mask_path": "/must-not-open/quality.png",
                "subject_id": "S1",
                "subset": "finger_latent",
                "ppi": 1000,
                "examiner_value": "VALUE",
            },
            {
                "image_id": "nonprint-1",
                "image_path": str(nonprint_image),
                "source_sha256": sha256_file(nonprint_image),
                "mask_path": "/must-not-open/nonprint-roi.png",
                "subject_id": "S2",
                "subset": "finger_latent",
                "ppi": 1000,
                "examiner_value": "NONPRINT",
            },
        ]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"splits": {"S1": "test", "S2": "test"}}),
        encoding="utf-8",
    )
    universe = module._test_universe(manifest, splits, expected_count=1)
    assert universe["prediction_count"] == 1
    assert universe["image_id_set_sha256"] == canonical_sha256(["heldout-1"])
    assert universe["ordered_source_images_sha256"] == canonical_sha256(
        [
            {
                "image_id": "heldout-1",
                "source_sha256": sha256_file(heldout_image),
            }
        ]
    )
    assert universe["test_inputs_hashed"] is True
    assert universe["test_images_decoded"] is False
    assert universe["test_labels_opened"] is False
    heldout_image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source image bytes changed"):
        module._test_universe(manifest, splits, expected_count=1)


def test_manifest_without_source_hash_fails_before_gate(tmp_path: Path) -> None:
    module = _module()
    image = tmp_path / "heldout.png"
    image.write_bytes(b"source")
    manifest = tmp_path / "latent.csv"
    pd.DataFrame(
        [
            {
                "image_id": "heldout-1",
                "image_path": str(image),
                "mask_path": "/must-not-open/roi.png",
                "subject_id": "S1",
                "subset": "finger_latent",
                "ppi": 1000,
                "examiner_value": "VALUE",
            }
        ]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits": {"S1": "test"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="source identities"):
        module._test_universe(manifest, splits, expected_count=1)


def test_independent_lr_reconstruction_binds_ledger_card_config_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "NATIVE_METHODS", ("unetpp",))
    commit = "a" * 40
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic manifest\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text("{}\n", encoding="utf-8")
    generated = tmp_path / "generated"
    full_path = generated / "full" / "unetpp.yaml"
    full_path.parent.mkdir(parents=True)
    candidates = []
    jobs = {}
    scores = {0.0003: 0.7, 0.001: 0.7}  # tie must select the lower LR
    for rate in module.planner.LEARNING_RATES:
        slug = module.planner.lr_slug(rate)
        source = generated / "sweep" / f"unetpp_lr{slug}.yaml"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"training:\n  learning_rate: {rate}\n", encoding="utf-8")
        run_dir = tmp_path / "runs" / "sweep" / "unetpp" / f"lr_{slug}" / "seed0"
        run_dir.mkdir(parents=True)
        for name in ("best.pt", "last.pt"):
            (run_dir / name).write_bytes(f"{slug}-{name}".encode())
        run = {
            "method": "unetpp",
            "seed": 0,
            "epochs_requested": module.planner.SWEEP_EPOCHS,
            "epochs_completed": module.planner.SWEEP_EPOCHS,
            "manifest_sha256": sha256_file(manifest),
            "split_sha256": sha256_file(splits),
            "source_config_sha256": sha256_file(source),
            "learning_rate": rate,
            "debug_limited": False,
            "best_validation_dice": scores[rate],
            "best_epoch": 3,
            "selected_validation_threshold": 0.5,
            "config_sha256": "b" * 64,
            "checkpoint_sha256": {
                name: sha256_file(run_dir / name) for name in ("best.pt", "last.pt")
            },
            "provenance": _provenance(commit),
        }
        run_path = run_dir / "run.json"
        run_path.write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
        candidates.append(
            {
                "learning_rate": rate,
                "best_validation_dice": scores[rate],
                "best_epoch": 3,
                "selected_validation_threshold": 0.5,
                "config_sha256": "b" * 64,
                "best_checkpoint_sha256": run["checkpoint_sha256"]["best.pt"],
                "run_card": str(run_path),
            }
        )
        job_name = f"sweep.unetpp.lr{slug}.seed0"
        jobs[job_name] = {
            "name": job_name,
            "argv": [
                "python", "train.py", "--config", str(source), "--out",
                str(run_dir), "--seed", "0",
            ],
            "completion_artifact": str(run_path),
            "ledger_artifact_sha256": sha256_file(run_path),
        }
    full_path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "learning_rate": 0.0003,
                    "epochs": module.planner.FULL_EPOCHS,
                    "patience": module.planner.FULL_PATIENCE,
                },
                "experiment": {
                    "phase": "full_training",
                    "learning_rate_selected_on": "val_best_dice",
                    "selected_learning_rate": 0.0003,
                    "sweep_epochs": module.planner.SWEEP_EPOCHS,
                    "full_seeds": list(module.planner.FULL_SEEDS),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    selection = {
        "selection_split": "val",
        "selection_metric": "best_validation_dice",
        "manifest_sha256": sha256_file(manifest),
        "splits_sha256": sha256_file(splits),
        "methods": {
            "unetpp": {
                "candidates": candidates,
                "selected_learning_rate": 0.0003,
                "selected_validation_dice": 0.7,
                "full_config": str(full_path),
                "full_config_sha256": sha256_file(full_path),
            }
        },
    }
    monkeypatch.setattr(
        module.planner,
        "validate_staged_selections",
        lambda *args: (selection, {}, {}),
    )
    selected, evidence = module._validate_global_selection(
        global_selection_path=tmp_path / "selection.json",
        pilot_selection_path=tmp_path / "pilot.json",
        sufs_selection_path=tmp_path / "sufs.json",
        expected_commits={"unetpp": commit},
        manifest=manifest,
        splits=splits,
        native_sweep_jobs=jobs,
    )
    assert selected is selection
    assert evidence["unetpp"]["selected_learning_rate"] == 0.0003

    low_best = (
        tmp_path / "runs/sweep/unetpp/lr_0p0003/seed0/best.pt"
    )
    original = low_best.read_bytes()
    low_best.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint changed"):
        module._validate_global_selection(
            global_selection_path=tmp_path / "selection.json",
            pilot_selection_path=tmp_path / "pilot.json",
            sufs_selection_path=tmp_path / "sufs.json",
            expected_commits={"unetpp": commit},
            manifest=manifest,
            splits=splits,
            native_sweep_jobs=jobs,
        )
    low_best.write_bytes(original)
    run_path = Path(candidates[0]["run_card"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["best_epoch"] = 4
    run_path.write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from (ledger|run card)"):
        module._validate_global_selection(
            global_selection_path=tmp_path / "selection.json",
            pilot_selection_path=tmp_path / "pilot.json",
            sufs_selection_path=tmp_path / "sufs.json",
            expected_commits={"unetpp": commit},
            manifest=manifest,
            splits=splits,
            native_sweep_jobs=jobs,
        )


def test_wrong_per_seed_training_commit_is_rejected(tmp_path: Path) -> None:
    module = _module()
    expected_commit = "a" * 40
    selection, runs_root = _full_run(
        module, tmp_path, method="unet", seed=1, commit="b" * 40
    )
    with pytest.raises(ValueError, match="provenance"):
        module._validate_full_run(
            method="unet",
            seed=1,
            expected_commit=expected_commit,
            selection=selection,
            runs_root=runs_root,
        )


def test_checkpoint_or_run_card_tampering_is_rejected(tmp_path: Path) -> None:
    module = _module()
    commit = "a" * 40
    selection, runs_root = _full_run(
        module, tmp_path, method="unet", seed=1, commit=commit
    )
    record = module._validate_full_run(
        method="unet",
        seed=1,
        expected_commit=commit,
        selection=selection,
        runs_root=runs_root,
    )
    assert record["checkpoint_sha256"] == sha256_file(
        runs_root / "full/unet/seed1/best.pt"
    )
    (runs_root / "full/unet/seed1/best.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash"):
        module._validate_full_run(
            method="unet",
            seed=1,
            expected_commit=commit,
            selection=selection,
            runs_root=runs_root,
        )
    (runs_root / "full/unet/seed1/best.pt").write_bytes(b"best-checkpoint")
    Path(selection["methods"]["unet"]["full_config"]).write_text(
        "changed: true\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="selected config bytes changed"):
        module._validate_full_run(
            method="unet",
            seed=1,
            expected_commit=commit,
            selection=selection,
            runs_root=runs_root,
        )


def test_less_than_two_valid_seeds_is_explicitly_insufficient() -> None:
    module = _module()
    methods = {
        method: {"valid_seeds": [0, 1], "failed_seeds": [], "missing_seeds": [2]}
        for method in module.NATIVE_METHODS
    }
    methods["drunet"] = {
        "valid_seeds": [0],
        "failed_seeds": [1],
        "missing_seeds": [2],
    }
    assert module._insufficient_methods(methods) == {"drunet": [0]}


def test_stage2_unet_reuse_binds_delivery_and_exact_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    commit = "c" * 40
    checkpoint_sha = "d" * 64
    ids = ["heldout-1", "heldout-2"]
    masks = tmp_path / "stage2-masks"
    masks.mkdir()
    prediction_records = []
    for image_id in ids:
        path = masks / f"{image_id}.png"
        path.write_bytes(f"mask:{image_id}".encode())
        prediction_records.append(
            {
                "image_id": image_id,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    prediction_sha = canonical_sha256(prediction_records)
    card = tmp_path / "unet-seed0-inference.json"
    payload = {
        "kind": "complete_unbiased_inference_card",
        "method": "unet",
        "subset": "finger_latent",
        "split": "test",
        "debug_limited": False,
        "identities": {
            "checkpoint_sha256": checkpoint_sha,
            "prediction_count": len(ids),
            "prediction_set_sha256": prediction_sha,
        },
        "prediction_set": {
            "count": len(ids),
            "sha256": prediction_sha,
            "records": prediction_records,
        },
        "provenance": _provenance(commit),
    }
    card.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    delivery = tmp_path / "delivery.json"
    delivery.write_text("{}\n", encoding="utf-8")
    candidate = {
        "learning_rate": 0.0003,
        "best_validation_dice": 0.7,
        "selected_validation_threshold": 0.5,
        "run_card": str(tmp_path / "sweep/run.json"),
        "run_card_sha256": "1" * 64,
        "source_config": str(tmp_path / "sweep.yaml"),
        "source_config_sha256": "2" * 64,
        "checkpoint_sha256": {"best.pt": "3" * 64, "last.pt": "4" * 64},
    }
    pilot_selection = tmp_path / "pilot-selection.json"
    pilot_selection.write_text(
        json.dumps(
            {
                "selected_learning_rate": 0.0003,
                "selected_validation_dice": 0.7,
                "candidates": [candidate],
            }
        ),
        encoding="utf-8",
    )
    stage2_gate = tmp_path / "stage2-gate.json"
    stage2_gate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "evidence": {
                    "selected_learning_rate": 0.0003,
                    "selected_sweep_validation_dice": 0.7,
                    "selection_candidates": [
                        {
                            "learning_rate": candidate["learning_rate"],
                            "validation_dice": candidate["best_validation_dice"],
                            "selected_threshold": candidate[
                                "selected_validation_threshold"
                            ],
                            "run_card": candidate["run_card"],
                            "run_card_sha256": candidate["run_card_sha256"],
                            "source_config": candidate["source_config"],
                            "source_config_sha256": candidate[
                                "source_config_sha256"
                            ],
                            "checkpoint_sha256": candidate["checkpoint_sha256"],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_validate_stage2_delivery",
        lambda _: {
            "checkpoint_sha256": checkpoint_sha,
            "inference_card_sha256": sha256_file(card),
            "gate_sha256": sha256_file(stage2_gate),
        },
    )
    reused = module._validate_stage2_reuse(
        delivery_path=delivery,
        inference_card_path=card,
        unet_run={"checkpoint_sha256": checkpoint_sha},
        expected_inference_commit=commit,
        expected_prediction_count=len(ids),
        expected_image_id_set_sha256=canonical_sha256(sorted(ids)),
        masks_root=masks,
        stage2_gate_path=stage2_gate,
        pilot_selection_path=pilot_selection,
    )
    assert reused["method"] == "unet" and reused["seed"] == 0
    assert reused["delivery"] == {
        "path": str(delivery),
        "sha256": sha256_file(delivery),
    }
    first_mask = masks / f"{ids[0]}.png"
    original_mask = first_mask.read_bytes()
    first_mask.write_bytes(b"substituted-stage2-mask")
    with pytest.raises(ValueError, match="prediction bytes changed"):
        module._validate_stage2_reuse(
            delivery_path=delivery,
            inference_card_path=card,
            unet_run={"checkpoint_sha256": checkpoint_sha},
            expected_inference_commit=commit,
            expected_prediction_count=len(ids),
            expected_image_id_set_sha256=canonical_sha256(sorted(ids)),
            masks_root=masks,
            stage2_gate_path=stage2_gate,
            pilot_selection_path=pilot_selection,
        )
    first_mask.write_bytes(original_mask)
    payload["prediction_set"]["records"][0]["image_id"] = "other"
    card.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_validate_stage2_delivery",
        lambda _: {
            "checkpoint_sha256": checkpoint_sha,
            "inference_card_sha256": sha256_file(card),
            "gate_sha256": sha256_file(stage2_gate),
        },
    )
    with pytest.raises(ValueError, match="held-out universe"):
        module._validate_stage2_reuse(
            delivery_path=delivery,
            inference_card_path=card,
            unet_run={"checkpoint_sha256": checkpoint_sha},
            expected_inference_commit=commit,
            expected_prediction_count=len(ids),
            expected_image_id_set_sha256=canonical_sha256(sorted(ids)),
            masks_root=masks,
            stage2_gate_path=stage2_gate,
            pilot_selection_path=pilot_selection,
        )


def test_duplicate_or_tampered_training_jobs_fail_closed(tmp_path: Path) -> None:
    module = _module()
    name = "full.unet.seed1"
    first = _guarded_job(module, name, tmp_path / "first.json")
    second = _guarded_job(module, name, tmp_path / "second.json")
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        yaml.safe_dump({"launch_authorized": True, "jobs": [first, second]}),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"queue_sha256": sha256_file(queue), "jobs": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate training job"):
        module._training_wave(queue, ledger, [name, name], "synthetic")

    only = _guarded_job(module, name, tmp_path / "only.json")
    queue.write_text(
        yaml.safe_dump({"launch_authorized": True, "jobs": [only]}),
        encoding="utf-8",
    )
    ledger.write_text(
        json.dumps(
            {
                "queue_sha256": sha256_file(queue),
                "jobs": {
                    name: {
                        "identity_sha256": only["job_identity_sha256"],
                        "status": "completed",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    payload = yaml.safe_load(queue.read_text(encoding="utf-8"))
    payload["jobs"][0]["argv"].append("--tampered")
    queue.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        module._training_wave(queue, ledger, [name], "synthetic")


def test_terminal_failed_training_entry_proves_retry_and_preemption_semantics(
    tmp_path: Path,
) -> None:
    module = _module()
    name = "full.unet.seed1"
    job = _guarded_job(module, name, tmp_path / "run.json")
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        yaml.safe_dump({"launch_authorized": True, "jobs": [job]}),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"

    def write_entry(**updates: Any) -> None:
        entry = {
            "identity_sha256": job["job_identity_sha256"],
            "status": "failed",
            "attempts": 3,
            "ordinary_retries": 1,
            "oom_retries": 0,
            "preemptions": 1,
            "exit_code": 1,
        }
        entry.update(updates)
        ledger.write_text(
            json.dumps(
                {"queue_sha256": sha256_file(queue), "jobs": {name: entry}}
            ),
            encoding="utf-8",
        )

    write_entry()
    module._training_wave(queue, ledger, [name], "synthetic")
    write_entry(attempts=1, ordinary_retries=0, preemptions=0)
    with pytest.raises(ValueError, match="did not consume a retry"):
        module._training_wave(queue, ledger, [name], "synthetic")
    write_entry(attempts=4, ordinary_retries=2, preemptions=1)
    with pytest.raises(ValueError, match="exceeds registered retry limits"):
        module._training_wave(queue, ledger, [name], "synthetic")
    write_entry(attempts=2, ordinary_retries=1, preemptions=1)
    with pytest.raises(ValueError, match="attempts do not prove"):
        module._training_wave(queue, ledger, [name], "synthetic")
    write_entry(
        attempts=2,
        ordinary_retries=0,
        oom_retries=1,
        preemptions=0,
        exit_code=76,
    )
    module._training_wave(queue, ledger, [name], "synthetic")


def test_inference_job_and_immutable_outputs_are_deterministic(tmp_path: Path) -> None:
    module = _module()
    receipt = tmp_path / "selector.json"
    receipt.write_text("{}\n", encoding="utf-8")
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "receipt.json").write_text("{}\n", encoding="utf-8")
    record = {
        "status": "valid",
        "method": "unetpp",
        "seed": 2,
        "run_card": str(tmp_path / "run.json"),
        "run_card_sha256": "1" * 64,
        "config": str(tmp_path / "config.yaml"),
        "config_sha256": "2" * 64,
        "checkpoint": str(tmp_path / "best.pt"),
        "checkpoint_sha256": "3" * 64,
        "training_git_commit": "4" * 40,
        "threshold_receipt": str(receipt),
        "threshold_receipt_sha256": sha256_file(receipt),
    }
    kwargs = {
        "record": record,
        "python_executable": Path("/venv/bin/python"),
        "code_root": Path("/pinned/segbench"),
        "inference_commit": "5" * 40,
        "manifest": Path("private_data/manifest.csv"),
        "splits": Path("private_data/splits.json"),
        "masks_root": tmp_path / "masks",
        "cards_root": tmp_path / "cards",
        "work_root": tmp_path / "work",
        "cache_dir": cache,
        "threshold_amendment": amendment,
        "ram_gib": 16.0,
        "segformer_env": None,
    }
    first = module._inference_job(**kwargs)
    second = module._inference_job(**kwargs)
    assert first == second
    job, planned = first
    assert job["name"] == "infer.unetpp.seed2"
    assert planned["expected_training_git_commit"] == "4" * 40
    assert planned["card_out"] == job["completion_artifact"]
    for option, expected in (
        ("--threshold-receipt", receipt),
        ("--threshold-amendment", amendment),
        ("--cache-dir", cache),
    ):
        assert job["argv"][job["argv"].index(option) + 1] == str(expected)

    output = tmp_path / "immutable.json"
    module._write_json_once(output, {"b": 2, "a": 1})
    original = output.read_bytes()
    module._write_json_once(output, {"a": 1, "b": 2})
    assert output.read_bytes() == original
    with pytest.raises(FileExistsError, match="refusing to replace"):
        module._write_json_once(output, {"a": 9, "b": 2})


def test_full_run_commit_map_is_exact(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "commits.json"
    mapping = {
        f"{method}.seed{seed}": "6" * 40
        for method in module.NATIVE_METHODS
        for seed in module.SEEDS
    }
    mapping["unet.seed0"] = "7" * 40
    path.write_text(json.dumps(mapping), encoding="utf-8")
    assert module._full_run_commit_map(path)["unet.seed0"] == "7" * 40
    mapping.pop("drunet.seed2")
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 24"):
        module._full_run_commit_map(path)
