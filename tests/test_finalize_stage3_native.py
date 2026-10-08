"""Synthetic, CPU-only tests for the one-time Stage 3 scoring transaction."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))


def _module():
    path = SCRIPTS / "finalize_stage3_native.py"
    spec = importlib.util.spec_from_file_location("finalize_stage3_native_test", path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _score(
    manifest: Path,
    splits: Path,
    predictions: Path,
    out: Path,
    *,
    seed: int,
    run_id: str,
    table: str,
    gt: str,
) -> None:
    command = [
        sys.executable,
        str(SCRIPTS / "evaluate_masks.py"),
        "--manifest", str(manifest),
        "--splits", str(splits),
        "--subsets", "finger_latent",
        "--method", "unet",
        "--display-name", "U-Net",
        "--pred-template", str(predictions / "{image_id}.png"),
        "--pred-ppi", "500",
        "--target-ppi", "500",
        "--split", "test",
        "--gt-column", gt,
        "--exclude", "examiner_value=NONPRINT",
        "--group-by", "examiner_value",
        "--seed", str(seed),
        "--year", "2022",
        "--family", "Encoder-decoder",
        "--training-data", "SD302-Latent train split",
        "--comment", "LR and threshold selected on validation only",
        "--params-m", "1.0",
        "--device", "GPU",
        "--threshold", "0.5",
        "--run-id", run_id,
        "--table", table,
        "--ms-per-img", "1.0",
        "--out", str(out),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _build_case(tmp_path: Path, module) -> SimpleNamespace:
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    groups = ("VALUE", "LIMITED", "NOVALUE", "MIXED")
    split_map = {}
    for index, group in enumerate(groups):
        image_id, subject = f"img{index}", f"s{index}"
        split_map[subject] = "test"
        image = np.full((4, 4), 128, np.uint8)
        roi = np.zeros((4, 4), np.uint8)
        roi[index % 2 : index % 2 + 2, 1:3] = 255
        quality = np.rot90(roi).copy()
        image_path = data / f"{image_id}_image.png"
        roi_path = data / f"{image_id}_roi.png"
        quality_path = data / f"{image_id}_quality.png"
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(roi_path), roi)
        cv2.imwrite(str(quality_path), quality)
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(roi_path),
                "quality_mask_path": str(quality_path),
                "subject_id": subject,
                "subset": "finger_latent",
                "ppi": 500,
                "examiner_value": group,
                "source_sha256": module.sha256_file(image_path),
                "roi_union_sha256": module.sha256_file(roi_path),
                "quality_union_sha256": module.sha256_file(quality_path),
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    _json(splits, {"seed": 42, "splits": split_map})
    ordered = module._ordered_source_gt(manifest)
    ordered_sources = module._ordered_source_images(manifest, splits)
    audit = tmp_path / "manifest.audit.json"
    _json(audit, {"manifest_sha256": module.sha256_file(manifest), "ordered_source_gt_sha256": ordered})

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for row in rows:
        # The prediction need not be good, only deterministic and complete.
        source = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(predictions / f"{row['image_id']}.png"), source)
    prediction_set = module.prediction_set_identity(
        (row["image_id"], predictions / f"{row['image_id']}.png") for row in rows
    )

    training_commit, inference_commit, finalizer_commit, planner_commit = (
        "a" * 40,
        "b" * 40,
        "c" * 40,
        "d" * 40,
    )
    config = tmp_path / "unet.yaml"
    config.write_text("method: unet\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    run_card = tmp_path / "run.json"
    checkpoint_sha = module.sha256_file(checkpoint)
    _json(
        run_card,
        {
            "method": "unet",
            "seed": 1,
            "debug_limited": False,
            "manifest_sha256": module.sha256_file(manifest),
            "split_sha256": module.sha256_file(splits),
            "source_config_sha256": module.sha256_file(config),
            "checkpoint_sha256": {"best.pt": checkpoint_sha},
            "selected_validation_threshold": 0.5,
            "best_validation_dice": 0.75,
            "parameter_count": {"total": 1_000_000, "trainable": 1_000_000},
            "provenance": {
                "capture_timing": "process_start",
                "git_commit": training_commit,
                "git_dirty": False,
            },
        },
    )
    card = tmp_path / "inference.json"
    _json(
        card,
        {
            "schema_version": 1,
            "kind": "complete_unbiased_inference_card",
            "method": "unet",
            "task": "pixel",
            "subset": "finger_latent",
            "split": "test",
            "device": "CUDA",
            "debug_limited": False,
            "selected_validation_threshold": 0.5,
            "identities": {
                "source_config_sha256": module.sha256_file(config),
                "checkpoint_sha256": checkpoint_sha,
                "authorized_checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": module.sha256_file(manifest),
                "split_sha256": module.sha256_file(splits),
                "prediction_set_sha256": prediction_set["sha256"],
                "prediction_count": len(rows),
                "ordered_source_images_sha256": ordered_sources,
            },
            "timing": {
                "warmup_images": 10,
                "mean_ms_per_img": 1.0,
                "per_image": [
                    {"image_id": row["image_id"], "ms_per_img": 1.0} for row in rows
                ],
            },
            "prediction_set": prediction_set,
            "provenance": {
                "capture_timing": "process_start",
                "git_commit": inference_commit,
                "git_dirty": False,
            },
        },
    )

    # Build the already-delivered Stage 2 U-Net seed-0 evidence.
    stage2 = tmp_path / "stage2"
    stage2_results = stage2 / "results"
    stage2_results.mkdir(parents=True)
    _score(manifest, splits, predictions, stage2_results, seed=0, run_id=checkpoint_sha, table="A", gt="mask_path")
    _score(manifest, splits, predictions, stage2_results, seed=0, run_id=checkpoint_sha, table="A_prime", gt="quality_mask_path")
    stage2_card = stage2 / "inference_card.json"
    shutil_card = json.loads(card.read_text(encoding="utf-8"))
    _json(stage2_card, shutil_card)
    delivery = stage2 / "delivery.json"
    stage2_summary_sha = module.sha256_file(stage2_results / "summary.csv")
    scoring_evidence = {"summary_sha256": stage2_summary_sha}
    for table in ("A", "A_prime"):
        matches = list(stage2_results.glob(f"per_image_unet_finger_latent_{table}_seed0_identity-*.csv"))
        assert len(matches) == 1
        scoring_evidence[table] = {
            "per_image_file": matches[0].name,
            "per_image_sha256": module.sha256_file(matches[0]),
            "n_images": len(rows),
            "groups": {"overall": 4, "VALUE": 1, "LIMITED": 1, "NOVALUE": 1, "MIXED": 1},
        }
    _json(
        delivery,
        {
            "schema_version": 1,
            "kind": "stage2_unet_heldout_delivery",
            "checkpoint_sha256": checkpoint_sha,
            "inference_card_sha256": module.sha256_file(stage2_card),
            "results": {
                "summary_sha256": stage2_summary_sha,
                "scoring_evidence": scoring_evidence,
            },
        },
    )

    gate = tmp_path / "gate.json"
    _json(
        gate,
        {
            "schema_version": 1,
            "kind": "stage3_native_training_readiness_gate",
            "status": "PASS_WITH_WARNINGS",
            "ground_truth_blind": True,
            "test_inputs_hashed": True,
            "test_images_decoded": False,
            "test_labels_opened": False,
            "expected_prediction_count": len(rows),
            "identities": {
                "manifest_sha256": module.sha256_file(manifest),
                "splits_sha256": module.sha256_file(splits),
            },
            "methods": {
                "unet": {
                    "valid_seeds": [0, 1],
                    "failed_seeds": [],
                    "missing_seeds": [],
                    "seeds": {
                        "0": {"status": "valid"},
                        "1": {"status": "valid"},
                    },
                }
            },
            "test_universe": {
                "ordered_source_images_sha256": ordered_sources,
            },
        },
    )
    code = {
        "planner": SCRIPTS / "prepare_stage3_inference.py",
        "finalizer": SCRIPTS / "finalize_stage3_native.py",
        "evaluator": SCRIPTS / "evaluate_masks.py",
        "metrics": ROOT / "segbench" / "metrics.py",
        "io": ROOT / "segbench" / "io.py",
        "summary": ROOT / "segbench" / "summary.py",
        "table_builder": SCRIPTS / "make_tables.py",
    }
    inference_code = {
        "wrapper": SCRIPTS / "infer_masks_repro.py",
        "auxiliary_wrapper": SCRIPTS / "infer_drunet_mc_dropout.py",
        "runner": SCRIPTS / "infer_masks.py",
        "deep_utils": ROOT / "segbench" / "deep_utils.py",
        "deep_models": ROOT / "segbench" / "deep_models.py",
        "artifact_identity": ROOT / "segbench" / "artifact_identity.py",
    }
    plan = tmp_path / "plan.json"
    run = {
        "method": "unet",
        "seed": 1,
        "run_card": str(run_card),
        "run_card_sha256": module.sha256_file(run_card),
        "config": str(config),
        "config_sha256": module.sha256_file(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "training_git_commit": training_commit,
        "expected_training_git_commit": training_commit,
        "expected_inference_git_commit": inference_commit,
        "masks_out": str(predictions),
        "card_out": str(card),
        "work_dir": str(tmp_path / "work"),
        "prediction_set_sha256": prediction_set["sha256"],
        "best_validation_dice": 0.75,
        "selected_validation_threshold": 0.5,
        "parameter_count": 1_000_000,
    }
    job = {
        "name": "infer.unet.seed1",
        "argv": ["python", "infer_masks_repro.py"],
        "env": {},
        "completion_artifact": str(card),
        "resources": {"host_ram_gib": 1.0},
    }
    job["job_identity_sha256"] = module.canonical_sha256(
        {
            "name": job["name"],
            "argv": job["argv"],
            "env": job["env"],
            "completion_artifact": job["completion_artifact"],
            "resources": job["resources"],
        }
    )
    run["job_identity_sha256"] = job["job_identity_sha256"]
    reused = {
        "method": "unet",
        "seed": 0,
        "delivery": {"path": str(delivery), "sha256": module.sha256_file(delivery)},
        "inference_card": {"path": str(stage2_card), "sha256": module.sha256_file(stage2_card)},
        "checkpoint_sha256": checkpoint_sha,
        "prediction_set_sha256": prediction_set["sha256"],
        "prediction_count": len(rows),
        "masks_out": str(predictions),
        "expected_inference_git_commit": inference_commit,
    }
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    gate_payload.update(
        expected_inference_git_commit=inference_commit,
        scorer_code={
            name: {"path": str(path), "sha256": module.sha256_file(path)}
            for name, path in code.items()
        },
        inference_code={
            name: {"path": str(path), "sha256": module.sha256_file(path)}
            for name, path in inference_code.items()
        },
        reused=reused,
        planner_provenance={
            "capture_timing": "process_start",
            "git_commit": planner_commit,
            "git_dirty": False,
        },
    )
    gate_payload["methods"]["unet"]["seeds"]["1"] = {
        "status": "valid",
        **{
            key: run[key]
            for key in (
                "method", "seed", "run_card", "run_card_sha256", "config",
                "config_sha256", "checkpoint", "checkpoint_sha256",
                "training_git_commit", "best_validation_dice",
                "selected_validation_threshold", "parameter_count",
            )
        },
    }
    _json(gate, gate_payload)
    _json(
        plan,
        {
            "schema_version": 1,
            "kind": "stage3_native_inference_plan",
            "ground_truth_blind": True,
            "test_inputs_hashed": True,
            "test_images_decoded": False,
            "test_labels_opened": False,
            "gate": str(gate),
            "gate_sha256": module.sha256_file(gate),
            "expected_prediction_count": len(rows),
            "expected_inference_git_commit": inference_commit,
            "manifest_audit": {"path": str(audit), "sha256": module.sha256_file(audit)},
            "ordered_source_gt_sha256": ordered,
            "ordered_source_images_sha256": ordered_sources,
            "code": {
                name: {"path": str(path), "sha256": module.sha256_file(path)}
                for name, path in code.items()
            },
            "inference_code": {
                name: {"path": str(path), "sha256": module.sha256_file(path)}
                for name, path in inference_code.items()
            },
            "reused": reused,
            "runs": [run],
        },
    )
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "phase": "stage3_native_test_inference",
                "launch_authorized": True,
                "plan_sha256": module.sha256_file(plan),
                "required_max_jobs_per_gpu": 1,
                "required_launcher_args": ["--max-per-gpu", "1"],
                "required_gpu_indices": [2, 5],
                "jobs": [job],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    prep = tmp_path / "prep.json"
    _json(
        prep,
        {
            "schema_version": 1,
            "kind": "stage3_native_frozen_preparation_bundle",
            "gate": {"path": str(gate), "sha256": module.sha256_file(gate)},
            "plan": {"path": str(plan), "sha256": module.sha256_file(plan)},
            "queue": {"path": str(queue), "sha256": module.sha256_file(queue)},
            "planner": {
                "path": str(SCRIPTS / "prepare_stage3_inference.py"),
                "sha256": module.sha256_file(SCRIPTS / "prepare_stage3_inference.py"),
            },
            "expected_planner_git_commit": planner_commit,
            "expected_inference_git_commit": inference_commit,
            "expected_finalizer_git_commit": finalizer_commit,
            "inference_code": gate_payload["inference_code"],
            "scorer_code": gate_payload["scorer_code"],
            "manifest": {"path": str(manifest), "sha256": module.sha256_file(manifest)},
            "splits": {"path": str(splits), "sha256": module.sha256_file(splits)},
            "ground_truth_blind": True,
            "test_inputs_hashed": True,
            "test_images_decoded": False,
            "test_labels_opened": False,
            "runtime_policy": {
                "gpu_indices": [2, 5],
                "max_jobs_per_gpu": 1,
                "memory_fraction": 0.3,
                "max_project_ram_gib": 900.0,
            },
        },
    )
    ledger = tmp_path / "ledger.json"
    _json(
        ledger,
        {
            "schema_version": 2,
            "queue_sha256": module.sha256_file(queue),
            "launcher_attestation": {
                "schema_version": 1,
                "queue_phase": "stage3_native_test_inference",
                "queue_sha256": module.sha256_file(queue),
                "gpus": [2, 5],
                "max_per_gpu": 1,
                "memory_fraction": 0.3,
                "required_launcher_args": ["--max-per-gpu", "1"],
            },
            "jobs": {
                "infer.unet.seed1": {
                    "status": "completed",
                    "identity_sha256": job["job_identity_sha256"],
                    "artifact_sha256": module.sha256_file(card),
                }
            },
        },
    )
    args = SimpleNamespace(
        gate=gate,
        plan=plan,
        prep_bundle=prep,
        expected_prep_bundle_sha256=module.sha256_file(prep),
        inference_queue=queue,
        inference_ledger=ledger,
        manifest=manifest,
        splits=splits,
        bundle_out=tmp_path / "bundle",
        restricted_root=tmp_path,
        declared_delivery=tmp_path / "bundle" / "delivery.json",
        expected_finalizer_git_commit=finalizer_commit,
        expected_planner_git_commit=planner_commit,
    )
    provenance = {
        "capture_timing": "process_start",
        "git_commit": finalizer_commit,
        "git_dirty": False,
    }
    return SimpleNamespace(
        args=args,
        provenance=provenance,
        predictions=predictions,
        rows=rows,
    )


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _module()
    monkeypatch.setattr(module, "NATIVE_METHODS", ("unet",))
    monkeypatch.setattr(module, "REQUIRED_SEEDS", (0, 1))
    monkeypatch.setattr(
        module,
        "_validate_auxiliary",
        lambda *_args, **_kwargs: {
            "plan_sha256": "1" * 64,
            "queue_sha256": "2" * 64,
            "ledger_sha256": "3" * 64,
            "completion_card_sha256": "4" * 64,
            "variance_set_sha256": "5" * 64,
            "timing_excluded": True,
            "scoring_excluded": True,
        },
    )
    return module, _build_case(tmp_path, module)


def test_atomic_publish_reuses_unet0_and_second_call_never_rescores(case, monkeypatch) -> None:
    module, built = case
    calls = []
    original = module._invoke

    def guarded(command):
        assert not built.args.bundle_out.exists()
        calls.append(list(command))
        original(command)

    monkeypatch.setattr(module, "_invoke", guarded)
    delivery = module.run(built.args, built.provenance)
    assert len(calls) == 2
    assert all("--seed" in command and command[command.index("--seed") + 1] == "1" for command in calls)
    assert delivery["stage2_unet_seed0"]["reused_not_rescored"] is True
    assert built.args.bundle_out.is_dir()
    assert (built.args.bundle_out / "delivery.json").is_file()
    summary = pd.read_csv(built.args.bundle_out / "results" / "summary.csv")
    assert "MIXED" in set(summary.group_value.astype(str))
    assert set(pd.to_numeric(summary.seed)) == {0, 1}
    assert delivery["gate_evidence"]["methods"]["unet"]["n_successful"] == 2

    calls.clear()
    replay = module.run(built.args, built.provenance)
    assert calls == []
    assert replay["outputs"] == delivery["outputs"]


def test_prediction_tampering_fails_before_scoring_and_leaves_no_journal(case) -> None:
    module, built = case
    target = built.predictions / f"{built.rows[0]['image_id']}.png"
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="prediction changed"):
        module.run(built.args, built.provenance)
    assert not module._journal_path(built.args.bundle_out).exists()
    assert not built.args.bundle_out.exists()
    assert not module._lock_path(built.args.bundle_out).exists()


def test_recursive_extra_prediction_file_is_rejected(case) -> None:
    module, built = case
    nested = built.predictions / "nested"
    nested.mkdir()
    (nested / "extra.png").write_bytes(b"unregistered")
    with pytest.raises(ValueError, match="prediction directory universe changed"):
        module.run(built.args, built.provenance)
    assert not module._journal_path(built.args.bundle_out).exists()


@pytest.mark.parametrize("artifact", ("gate", "plan"))
def test_substituted_gate_or_plan_is_rejected_by_external_prep_binding(
    case, artifact: str
) -> None:
    module, built = case
    path = getattr(built.args, artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["substituted"] = True
    _json(path, payload)
    with pytest.raises(
        ValueError,
        match=(
            "preparation bundle binding changed|"
            "inference plan does not bind the training-readiness gate"
        ),
    ):
        module.run(built.args, built.provenance)
    assert not module._journal_path(built.args.bundle_out).exists()


def test_source_image_tampering_is_rejected_before_scoring(case) -> None:
    module, built = case
    Path(built.rows[0]["image_path"]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source image bytes changed"):
        module.run(built.args, built.provenance)
    assert not module._journal_path(built.args.bundle_out).exists()


def test_bundle_escape_is_rejected(case) -> None:
    module, built = case
    built.args.bundle_out = built.args.restricted_root.parent / "escaped-bundle"
    built.args.declared_delivery = built.args.bundle_out / "delivery.json"
    with pytest.raises(ValueError, match="escapes the restricted root"):
        module.run(built.args, built.provenance)


def test_replay_rejects_mutated_scientific_result_without_rescoring(
    case, monkeypatch
) -> None:
    module, built = case
    calls = []
    original = module._invoke

    def invoke(command):
        calls.append(list(command))
        return original(command)

    monkeypatch.setattr(module, "_invoke", invoke)
    module.run(built.args, built.provenance)
    assert len(calls) == 2
    per_image = next(
        (built.args.bundle_out / "results").glob(
            "per_image_unet_finger_latent_A_seed1_identity-*.csv"
        )
    )
    frame = pd.read_csv(per_image)
    frame.loc[0, "tp"] = int(frame.loc[0, "tp"]) + 1
    frame.to_csv(per_image, index=False)
    calls.clear()
    with pytest.raises(ValueError, match="output bytes changed"):
        module.run(built.args, built.provenance)
    assert calls == []


@pytest.mark.parametrize("artifact", ("ground_truth", "prediction"))
def test_persistent_midscore_input_mutation_aborts_before_delivery(
    case, monkeypatch, artifact: str
) -> None:
    module, built = case
    original = module._invoke
    mutated = False

    def invoke(command):
        nonlocal mutated
        if not mutated:
            target = (
                Path(built.rows[0]["mask_path"])
                if artifact == "ground_truth"
                else built.predictions / f"{built.rows[0]['image_id']}.png"
            )
            image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            assert image is not None
            assert cv2.imwrite(str(target), 255 - image)
            mutated = True
        return original(command)

    monkeypatch.setattr(module, "_invoke", invoke)
    with pytest.raises(ValueError, match="changed"):
        module.run(built.args, built.provenance)
    assert mutated
    assert not built.args.bundle_out.exists()
    assert module._journal_path(built.args.bundle_out).is_file()
    assert module._failure_path(built.args.bundle_out).is_file()
    journal = json.loads(
        module._journal_path(built.args.bundle_out).read_text(encoding="utf-8")
    )
    assert Path(journal["staging_directory"]).is_dir()


def test_postscore_failure_retains_staging_and_refuses_automatic_rescore(
    case, monkeypatch
) -> None:
    module, built = case
    calls = []
    original_invoke = module._invoke
    original_validate = module._validate_seed_results

    def invoke(command):
        calls.append(list(command))
        original_invoke(command)

    count = 0

    def fail_after_stage2(result_dir, evidence, universes, *, prediction_dir):
        nonlocal count
        count += 1
        if count == 1:  # Stage 2 reuse is validated before the journal.
            return original_validate(
                result_dir, evidence, universes, prediction_dir=prediction_dir
            )
        raise ValueError("synthetic post-score validation failure")

    monkeypatch.setattr(module, "_invoke", invoke)
    monkeypatch.setattr(module, "_validate_seed_results", fail_after_stage2)
    with pytest.raises(ValueError, match="synthetic post-score"):
        module.run(built.args, built.provenance)
    assert len(calls) == 2
    journal = module._journal_path(built.args.bundle_out)
    failure = module._failure_path(built.args.bundle_out)
    assert journal.is_file() and failure.is_file()
    failed = json.loads(failure.read_text(encoding="utf-8"))
    assert Path(failed["staging_directory"]).is_dir()
    assert module._lock_path(built.args.bundle_out).exists()

    calls.clear()
    with pytest.raises(RuntimeError, match="automatic rescoring is forbidden"):
        module.run(built.args, built.provenance)
    assert calls == []


def test_gate_evidence_reports_every_missing_and_failed_seed(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "NATIVE_METHODS", ("unet", "drunet"))
    gate = {
        "methods": {
            "unet": {"failed_seeds": [], "missing_seeds": [2]},
            "drunet": {"failed_seeds": [1], "missing_seeds": []},
        }
    }
    evidence = module._gate_evidence(
        [
            {"method": "unet", "seed": 0},
            {"method": "unet", "seed": 1},
            {"method": "drunet", "seed": 0},
            {"method": "drunet", "seed": 2},
        ],
        gate,
    )
    assert evidence["status"] == "PASS"
    assert evidence["methods"]["unet"]["missing_seeds"] == [2]
    assert evidence["methods"]["drunet"]["failed_seeds"] == [1]


def test_auxiliary_chain_replays_manifest_and_rejects_reserved_masks(
    tmp_path: Path,
) -> None:
    module = _module()
    data = tmp_path / "sources"
    data.mkdir()
    rows = []
    split_map = {}
    image_ids = [f"aux{index:02d}" for index in range(20)]
    for index, image_id in enumerate(image_ids):
        source = data / f"{image_id}.png"
        source.write_bytes(f"source-{index}".encode())
        subject = f"subject{index:02d}"
        split_map[subject] = "test"
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(source),
                "mask_path": "",
                "subject_id": subject,
                "subset": "finger_latent",
                "ppi": 500,
                "examiner_value": "VALUE",
                "source_sha256": module.sha256_file(source),
                "target_height": 2,
                "target_width": 3,
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    _json(splits, {"splits": split_map})
    config = tmp_path / "drunet.yaml"
    checkpoint = tmp_path / "best.pt"
    run_card = tmp_path / "run.json"
    config.write_text("method: drunet\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    run_card.write_text("{}\n", encoding="utf-8")
    training_commit, inference_commit = "a" * 40, "b" * 40
    seed_record = {
        "status": "valid",
        "method": "drunet",
        "seed": 0,
        "run_card": str(run_card),
        "run_card_sha256": module.sha256_file(run_card),
        "config": str(config),
        "config_sha256": module.sha256_file(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": module.sha256_file(checkpoint),
        "best_validation_dice": 0.75,
        "selected_validation_threshold": 0.5,
        "parameter_count": 123,
        "training_git_commit": training_commit,
    }
    gate_path = tmp_path / "gate.json"
    gate = {"methods": {"drunet": {"seeds": {"0": seed_record}}}}
    _json(gate_path, gate)
    wrapper = SCRIPTS / "infer_drunet_mc_dropout.py"
    deterministic_path = tmp_path / "deterministic-plan.json"
    deterministic = {
        "expected_inference_git_commit": inference_commit,
        "inference_code": {
            "auxiliary_wrapper": {
                "path": str(wrapper),
                "sha256": module.sha256_file(wrapper),
            }
        },
    }
    _json(deterministic_path, deterministic)
    variance = tmp_path / "variance"
    variance.mkdir()
    variance_records = []
    expected_shapes = {}
    for index, image_id in enumerate(image_ids):
        path = variance / f"{image_id}_variance.npy"
        array = np.full((2, 3), index, np.float32)
        np.save(path, array, allow_pickle=False)
        expected_shapes[image_id] = [2, 3]
        variance_records.append(
            {
                "image_id": image_id,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": module.sha256_file(path),
                "shape": [2, 3],
                "dtype": "float32",
            }
        )
    source_records = [
        {"image_id": row["image_id"], "source_sha256": row["source_sha256"]}
        for row in rows
    ]
    reserved_masks = tmp_path / "reserved-masks"
    auxiliary_work = tmp_path / "aux-work"
    card = tmp_path / "aux-card.json"
    run = {
        **seed_record,
        "expected_training_git_commit": training_commit,
        "expected_inference_git_commit": inference_commit,
    }
    aux_plan_path = tmp_path / "aux-plan.json"
    aux_plan = {
        "schema_version": 1,
        "kind": "stage3_drunet_mc_dropout_plan",
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "timing_excluded": True,
        "mask_outputs_forbidden": True,
        "mc_dropout_passes": 20,
        "selection_rule": "first_20_lexicographic_heldout_ids",
        "test_image_ids": image_ids,
        "image_id_set_sha256": module.canonical_sha256(image_ids),
        "ordered_source_images_sha256": module.canonical_sha256(source_records),
        "expected_shapes": expected_shapes,
        "gate": {"path": str(gate_path), "sha256": module.sha256_file(gate_path)},
        "deterministic_plan": {
            "path": str(deterministic_path),
            "sha256": module.sha256_file(deterministic_path),
        },
        "run": run,
        "identities": {
            name: {"path": str(path), "sha256": module.sha256_file(path)}
            for name, path in {
                "config": config,
                "checkpoint": checkpoint,
                "run_card": run_card,
                "manifest": manifest,
                "splits": splits,
                "wrapper": wrapper,
            }.items()
        },
        "reserved_masks_root": str(reserved_masks),
        "variance_out": str(variance),
        "work_dir": str(auxiliary_work),
        "card_out": str(card),
    }
    _json(aux_plan_path, aux_plan)
    _json(
        card,
        {
            "schema_version": 1,
            "kind": "stage3_drunet_mc_dropout_completion_card",
            "method": "drunet",
            "seed": 0,
            "plan_sha256": module.sha256_file(aux_plan_path),
            "ground_truth_opened": False,
            "mask_outputs_written": False,
            "timing_excluded": True,
            "mc_dropout_passes": 20,
            "image_id_set_sha256": module.canonical_sha256(image_ids),
            "ordered_source_images_sha256": module.canonical_sha256(source_records),
            "variance_set": {
                "count": 20,
                "records": variance_records,
                "sha256": module.canonical_sha256(variance_records),
            },
            "identities": {
                name: {"path": str(path), "sha256": module.sha256_file(path)}
                for name, path in {
                    "config": config,
                    "checkpoint": checkpoint,
                    "manifest": manifest,
                    "splits": splits,
                }.items()
            },
            "provenance": {
                "capture_timing": "process_start",
                "git_commit": inference_commit,
                "git_dirty": False,
            },
        },
    )
    job = {
        "name": "aux.drunet.seed0.mc_dropout20",
        "argv": [
            "python",
            str(wrapper),
            "--plan", str(aux_plan_path),
            "--expected-plan-sha256", module.sha256_file(aux_plan_path),
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--manifest", str(manifest),
            "--splits", str(splits),
            "--variance-out", str(variance),
            "--card-out", str(card),
            "--work-dir", str(auxiliary_work),
            "--expected-git-commit", inference_commit,
            "--expected-checkpoint-sha256", module.sha256_file(checkpoint),
            "--seed", "0",
            "--device", "cuda",
            "--memory-fraction", "0.3",
        ],
        "completion_artifact": str(card),
        "resources": {"host_ram_gib": 4.0},
    }
    job["job_identity_sha256"] = module.canonical_sha256(
        {
            "name": job["name"],
            "argv": job["argv"],
            "env": {},
            "completion_artifact": job["completion_artifact"],
            "resources": job["resources"],
        }
    )
    aux_queue_path = tmp_path / "aux-queue.yaml"
    aux_queue = {
        "schema_version": 2,
        "phase": "stage3_drunet_mc_dropout_aux",
        "launch_authorized": True,
        "timing_excluded": True,
        "mask_outputs_forbidden": True,
        "mc_dropout_passes": 20,
        "required_max_jobs_per_gpu": 1,
        "required_launcher_args": ["--max-per-gpu", "1"],
        "required_gpu_indices": [7],
        "plan_sha256": module.sha256_file(aux_plan_path),
        "jobs": [job],
    }
    aux_queue_path.write_text(yaml.safe_dump(aux_queue), encoding="utf-8")
    aux_ledger = tmp_path / "aux-ledger.json"
    _json(
        aux_ledger,
        {
            "schema_version": 2,
            "queue_sha256": module.sha256_file(aux_queue_path),
            "launcher_attestation": {
                "schema_version": 1,
                "queue_phase": "stage3_drunet_mc_dropout_aux",
                "queue_sha256": module.sha256_file(aux_queue_path),
                "gpus": [7],
                "max_per_gpu": 1,
                "memory_fraction": 0.3,
                "required_launcher_args": ["--max-per-gpu", "1"],
            },
            "jobs": {
                job["name"]: {
                    "status": "completed",
                    "identity_sha256": job["job_identity_sha256"],
                    "artifact_sha256": module.sha256_file(card),
                }
            },
        },
    )
    prep = tmp_path / "prep.json"
    _json(
        prep,
        {
            "aux_plan": {
                "path": str(aux_plan_path),
                "sha256": module.sha256_file(aux_plan_path),
            },
                "aux_queue": {
                    "path": str(aux_queue_path),
                    "sha256": module.sha256_file(aux_queue_path),
                },
                "runtime_policy": {"gpu_indices": [7]},
            },
    )
    args = SimpleNamespace(
        aux_plan=aux_plan_path,
        aux_queue=aux_queue_path,
        aux_ledger=aux_ledger,
        aux_card=card,
        restricted_root=tmp_path,
        prep_bundle=prep,
        gate=gate_path,
        plan=deterministic_path,
        manifest=manifest,
        splits=splits,
    )
    evidence = module._validate_auxiliary(args, gate, deterministic)
    assert evidence["variance_set_sha256"] == module.canonical_sha256(
        variance_records
    )
    assert evidence["reserved_masks_empty"] is True

    reserved_masks.mkdir()
    (reserved_masks / "forbidden.png").write_bytes(b"not allowed")
    with pytest.raises(ValueError, match="reserved mask root is not empty"):
        module._validate_auxiliary(args, gate, deterministic)
