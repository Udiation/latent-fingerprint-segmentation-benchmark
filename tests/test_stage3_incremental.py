"""Synthetic-only tests for incremental G3 scoring and immutable G4 import."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Dict, Optional

import pandas as pd
import pytest
import yaml

from segbench.artifact_identity import canonical_sha256, sha256_file
from segbench import stage3_incremental as inc


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _identity(path: Path) -> Dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _probability_card(path: Path, method: str, seed: int, split: str,
                      config_sha: str, checkpoint_sha: str) -> Path:
    return _json(path, {
        "schema_version": 1,
        "kind": inc.PROBABILITY_CARD_KIND,
        "status": "complete",
        "complete": True,
        "method": method,
        "task": "patch" if method in {"lenet", "alexnet", "cnn_ae"} else "pixel",
        "seed": seed,
        "split": split,
        "debug_limited": False,
        "source_config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
    })


def _seed0_case(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "restricted"
    root.mkdir()
    images = root / "images"
    images.mkdir()
    rows = []
    for index in range(2):
        image = images / f"synthetic-{index}.bin"
        image.write_bytes(f"image-{index}".encode())
        rows.append({
            "image_id": f"synthetic-{index}",
            "image_path": str(image),
            "source_sha256": sha256_file(image),
            "mask_path": "/must-not-open/roi.png",
            "quality_mask_path": "/must-not-open/quality.png",
            "subject_id": f"S{index}",
            "subset": "finger_latent",
            "examiner_value": "VALUE",
            "ppi": 1000,
        })
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    splits = _json(root / "splits.json", {"splits": {"S0": "test", "S1": "test"}})
    amendment = _json(root / "amendment.json", {
        "schema_version": 1,
        "kind": "stage4_threshold_policy_amendment",
        "replacement_policy": {
            "allowed_trainable_grid": list(inc.AMENDED_GRID),
            "selection_data": "validation_only",
        },
    })
    cache = _json(root / "cache" / "receipt.json", {
        "schema_version": 1,
        "kind": "uint8_500ppi_preprocessing_cache",
        "manifest_sha256": sha256_file(manifest),
        "target_ppi": 500.0,
        "mask_columns": ["mask_path", "quality_mask_path"],
        "debug_limited": False,
    })
    stage2 = _json(root / "stage2" / "delivery.json", {
        "schema_version": 1,
        "kind": "stage2_threshold_reselection_delivery",
    })
    runs = []
    for method in inc.SEED0_INFERENCE_METHODS:
        method_root = root / "runs" / method
        config = method_root / "config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(yaml.safe_dump({
            "schema_version": 1,
            "method": method,
            "task": "patch" if method in {"lenet", "alexnet", "cnn_ae"} else "pixel",
            "seed": 0,
        }), encoding="utf-8")
        checkpoint = method_root / "best.pt"
        checkpoint.write_bytes(f"checkpoint-{method}".encode())
        run = _json(method_root / "run.json", {
            "method": method,
            "seed": 0,
            "source_config_sha256": sha256_file(config),
            "checkpoint_sha256": {"best.pt": sha256_file(checkpoint)},
            "parameter_count": {"total": 1000},
        })
        raw_val = _probability_card(
            method_root / "raw_val.json", method, 0, "val",
            sha256_file(config), sha256_file(checkpoint),
        )
        selected_val = _probability_card(
            method_root / "selected_val.json", method, 0, "val",
            sha256_file(config), sha256_file(checkpoint),
        )
        selector = _json(method_root / "selector.json", {
            "schema_version": 1,
            "kind": inc.SELECTOR_KIND,
            "method": method,
            "task": "patch" if method in {"lenet", "alexnet", "cnn_ae"} else "pixel",
            "seed": 0,
            "selection_split": "val",
            "selection_source": "validation_only",
            "validation_only": True,
            "test_or_heldout_opened": False,
            "source_config_sha256": sha256_file(config),
            "checkpoint_sha256": sha256_file(checkpoint),
            "manifest_sha256": sha256_file(manifest),
            "split_sha256": sha256_file(splits),
            "amendment_sha256": sha256_file(amendment),
            "threshold_amendment_sha256": sha256_file(amendment),
            "raw_validation_probability_card_sha256": sha256_file(raw_val),
            "validation_probability_card_sha256": sha256_file(selected_val),
            "probabilities_are_pre_threshold": True,
            "derived_from_binary_mask": False,
            "debug_limited": False,
            "allowed_grid": list(inc.AMENDED_GRID),
            "selected_threshold": 0.3,
        })
        profile = _json(method_root / "profile.json", {"epochs": 3})
        runs.append({
            "method": method,
            "seed": 0,
            "config": _identity(config),
            "run_card": _identity(run),
            "checkpoint": _identity(checkpoint),
            "selector_receipt": _identity(selector),
            "raw_validation_probability_card": _identity(raw_val),
            "validation_probability_card": _identity(selected_val),
            "training_lineage": "amended_training",
            "per_epoch_training_profile_card": _identity(profile),
            "masks_out": str(root / "outputs" / method / "masks"),
            "probabilities_out": str(root / "outputs" / method / "probabilities"),
            "card_out": str(root / "outputs" / method / "inference.json"),
            "work_dir": str(root / "outputs" / method / "work"),
            "seed_delivery_out": str(root / "seed_deliveries" / f"{method}-0"),
        })
    spec = _json(root / "seed0_spec.json", {
        "schema_version": 1,
        "kind": inc.PLAN_SPEC_KIND,
        "manifest": _identity(manifest),
        "splits": _identity(splits),
        "threshold_amendment": _identity(amendment),
        "cache_receipt": _identity(cache),
        "stage2_unet_seed0_delivery": _identity(stage2),
        "expected_prediction_count": 2,
        "code_root": str(ROOT),
        "expected_inference_git_commit": "a" * 40,
        "expected_scoring_git_commit": "b" * 40,
        "python_executable": sys.executable,
        "allowed_gpu_indices": [0],
        "ram_gib": {method: 2.0 for method in inc.SEED0_INFERENCE_METHODS},
        "runs": runs,
    })
    return SimpleNamespace(
        root=root, spec=spec, manifest=manifest, splits=splits,
        amendment=amendment, cache=cache, runs=runs,
    )


def test_seed0_plan_is_exact_ground_truth_blind_and_not_a_g4_pass(tmp_path: Path) -> None:
    case = _seed0_case(tmp_path)
    gate_path, plan_path, queue_path = (
        case.root / "gate.json", case.root / "plan.json", case.root / "queue.yaml"
    )
    gate, plan, queue = inc.prepare_seed0(
        spec_path=case.spec,
        expected_spec_sha256=sha256_file(case.spec),
        restricted_root=case.root,
        gate_out=gate_path,
        plan_out=plan_path,
        queue_out=queue_path,
    )
    assert gate["status"] == "PASS"
    assert gate["satisfies_stage4_multi_seed_gate"] is False
    assert gate["stage4_minimum_remains"] == 2
    assert gate["test_universe"]["test_labels_opened"] is False
    assert {record["method"] for record in plan["runs"]} == set(inc.SEED0_INFERENCE_METHODS)
    assert len(queue["jobs"]) == 7
    assert queue["required_max_jobs_per_gpu"] == 1
    assert queue["required_launcher_args"] == ["--max-per-gpu", "1"]
    assert all("--split" in job["argv"] and "test" in job["argv"] for job in queue["jobs"])
    # The deliberately absent label paths prove the planner never opens GT.
    assert not Path("/must-not-open/roi.png").exists()


def test_seed0_plan_fails_when_a_native_method_is_missing(tmp_path: Path) -> None:
    case = _seed0_case(tmp_path)
    spec = json.loads(case.spec.read_text())
    spec["runs"].pop()
    _json(case.spec, spec)
    with pytest.raises(inc.IncrementalError, match="exactly the seven"):
        inc.prepare_seed0(
            spec_path=case.spec,
            expected_spec_sha256=sha256_file(case.spec),
            restricted_root=case.root,
            gate_out=case.root / "gate.json",
            plan_out=case.root / "plan.json",
            queue_out=case.root / "queue.yaml",
        )

def test_per_seed_finalizer_scores_once_then_only_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _seed0_case(tmp_path)
    gate_path, plan_path, queue_path = case.root / "gate.json", case.root / "plan.json", case.root / "queue.yaml"
    _, plan, _ = inc.prepare_seed0(
        spec_path=case.spec,
        expected_spec_sha256=sha256_file(case.spec),
        restricted_root=case.root,
        gate_out=gate_path,
        plan_out=plan_path,
        queue_out=queue_path,
    )
    record = plan["runs"][0]
    method = record["method"]
    outputs = record["outputs"]
    Path(outputs["work_dir"]).mkdir(parents=True)
    Path(outputs["masks_out"]).mkdir(parents=True)
    Path(outputs["probabilities_out"]).mkdir(parents=True)
    test_card = _json(Path(outputs["work_dir"]) / "full_timed_pass.json", {
        "schema_version": 1,
        "kind": inc.PROBABILITY_CARD_KIND,
        "status": "complete",
        "complete": True,
        "method": method,
        "task": record["task"],
        "seed": 0,
        "split": "test",
        "labels_opened": False,
        "threshold_selected_before_test": True,
        "selected_validation_threshold": record["selected_threshold"],
        "identities": {
            "source_config_sha256": record["artifacts"]["config"]["sha256"],
            "checkpoint_sha256": record["artifacts"]["checkpoint"]["sha256"],
            "selector_receipt_sha256": record["artifacts"]["selector_receipt"]["sha256"],
            "amendment_sha256": plan["data_inputs"]["threshold_amendment"]["sha256"],
        },
    })
    _json(Path(outputs["card_out"]), {
        "schema_version": 1,
        "kind": inc.UNBIASED_CARD_KIND,
        "method": method,
        "seed": 0,
        "split": "test",
        "debug_limited": False,
        "identities": {
            "full_timed_pass_card_sha256": sha256_file(test_card),
            "checkpoint_sha256": record["artifacts"]["checkpoint"]["sha256"],
        },
    })
    calls = []

    def fake_run(command, cwd, check):
        calls.append(list(command))
        result_dir = Path(command[command.index("--results-dir") + 1])
        receipt_path = Path(command[command.index("--receipt-out") + 1])
        rows = []
        invocations = []
        for table in ("A", "A_prime"):
            per = result_dir / f"per_image_{method}_finger_latent_{table}_seed0_identity-synthetic.csv"
            pd.DataFrame([{"image_id": "synthetic-0", "dice": 0.5}]).to_csv(per, index=False)
            rows.append({
                "method": method, "seed": 0, "table": table, "group_value": "overall",
                "n_missing": 0, "run_id": "run", "training_data": "SD302-Latent train split",
            })
            invocations.append({"table": table, "per_image": _identity(per)})
        summary = result_dir / "summary.csv"
        pd.DataFrame(rows).to_csv(summary, index=False)
        _json(receipt_path, {
            "schema_version": 1,
            "kind": inc.SCORING_RECEIPT_KIND,
            "method": method,
            "seed": 0,
            "selected_threshold": record["selected_threshold"],
            "prediction_count": 2,
            "invocations": invocations,
            "scientific_summary": _identity(summary),
        })
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(inc.subprocess, "run", fake_run)
    monkeypatch.setattr(inc, "_validate_code_root", lambda *_: None)
    bundle = Path(outputs["seed_delivery_out"])
    first = inc.finalize_seed(
        plan_path=plan_path,
        expected_plan_sha256=sha256_file(plan_path),
        restricted_root=case.root,
        method=method,
        seed=0,
        bundle_out=bundle,
        repo_root=ROOT.parent,
        expected_git_commit="b" * 40,
    )
    assert first["automatic_heldout_rescore_permitted"] is False
    assert len(calls) == 1
    assert calls[0].count("--prediction-dir") == 1
    assert calls[0].count("--probability-dir") == 1
    receipt = json.loads((bundle / "scoring_receipt.json").read_text())
    assert {row["table"] for row in receipt["invocations"]} == {"A", "A_prime"}
    second = inc.finalize_seed(
        plan_path=plan_path,
        expected_plan_sha256=sha256_file(plan_path),
        restricted_root=case.root,
        method=method,
        seed=0,
        bundle_out=bundle,
        repo_root=ROOT.parent,
        expected_git_commit="b" * 40,
    )
    assert second == first
    assert len(calls) == 1


def test_per_seed_finalizer_rejects_an_unplanned_bundle_path(tmp_path: Path) -> None:
    case = _seed0_case(tmp_path)
    plan_path = case.root / "plan.json"
    inc.prepare_seed0(
        spec_path=case.spec,
        expected_spec_sha256=sha256_file(case.spec),
        restricted_root=case.root,
        gate_out=case.root / "gate.json",
        plan_out=plan_path,
        queue_out=case.root / "queue.yaml",
    )
    method = json.loads(plan_path.read_text())["runs"][0]["method"]
    with pytest.raises(inc.IncrementalError, match="differs from the frozen plan"):
        inc.finalize_seed(
            plan_path=plan_path,
            expected_plan_sha256=sha256_file(plan_path),
            restricted_root=case.root,
            method=method,
            seed=0,
            bundle_out=case.root / "different-delivery-path",
            repo_root=ROOT.parent,
            expected_git_commit="b" * 40,
        )


def _summary(path: Path, method: str, seed: int) -> Path:
    rows = [{
        "method": method, "seed": seed, "table": table, "group_value": "overall",
        "n_missing": 0, "run_id": f"{method}-{seed}",
        "training_data": "synthetic train",
    } for table in ("A", "A_prime")]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _scored_bundle(
    root: Path,
    method: str,
    seed: int,
    digests: Dict[str, str],
    test_universe: Optional[Dict[str, Any]] = None,
) -> Path:
    bundle = root / "seed_deliveries" / f"{method}-{seed}"
    results = bundle / "results"
    summary = _summary(results / "summary.csv", method, seed)
    invocations = []
    for table in ("A", "A_prime"):
        per = results / f"per_image_{method}_{table}_seed{seed}.csv"
        pd.DataFrame([{"image_id": "synthetic", "dice": 0.5}]).to_csv(per, index=False)
        invocations.append({"table": table, "per_image": _identity(per)})
    receipt = _json(bundle / "scoring_receipt.json", {
        "schema_version": 1,
        "kind": inc.SCORING_RECEIPT_KIND,
        "method": method,
        "seed": seed,
        "invocations": invocations,
        "scientific_summary": _identity(summary),
    })
    plan = _json(root / "plans" / f"{method}-{seed}.json", {"method": method, "seed": seed})
    journal = _json(bundle.parent / f".{bundle.name}.score-once-journal.json", {
        "method": method, "seed": seed,
    })
    delivery = {
        "schema_version": 1,
        "kind": inc.SEED_DELIVERY_KIND,
        "method": method,
        "seed": seed,
        "automatic_heldout_rescore_permitted": False,
        "plan": _identity(plan),
        "training": dict(digests),
        "test_universe": test_universe or {"ordered_source_images_sha256": "f" * 64},
        "scoring_receipt_sha256": sha256_file(receipt),
        "journal_sha256": sha256_file(journal),
    }
    delivery["outputs"] = inc._directory_hashes(bundle, exclude=("delivery.json",))
    return _json(bundle / "delivery.json", delivery)


def test_additional_plan_binds_prior_seed_and_cannot_reopen_it(tmp_path: Path) -> None:
    case = _seed0_case(tmp_path)
    universe = inc._test_universe(case.manifest, case.splits)
    priors = []
    for index, prior_method in enumerate(inc.SEED0_INFERENCE_METHODS):
        prior_digests = {
            "config_sha256": hashlib.sha256(f"config-{index}".encode()).hexdigest(),
            "run_card_sha256": hashlib.sha256(f"run-{index}".encode()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(f"checkpoint-{index}".encode()).hexdigest(),
        }
        priors.append(_scored_bundle(
            case.root, prior_method, 0, prior_digests, test_universe=universe,
        ))
    method, seed = "unetpp", 1
    run_root = case.root / "additional" / method
    run_root.mkdir(parents=True)
    config = run_root / "config.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1, "method": method, "task": "pixel", "seed": seed,
    }), encoding="utf-8")
    checkpoint = run_root / "best.pt"
    checkpoint.write_bytes(b"additional checkpoint")
    run_card = _json(run_root / "run.json", {
        "method": method,
        "seed": seed,
        "source_config_sha256": sha256_file(config),
        "checkpoint_sha256": {"best.pt": sha256_file(checkpoint)},
        "parameter_count": {"total": 1000},
    })
    raw_val = _probability_card(
        run_root / "raw_val.json", method, seed, "val",
        sha256_file(config), sha256_file(checkpoint),
    )
    selected_val = _probability_card(
        run_root / "selected_val.json", method, seed, "val",
        sha256_file(config), sha256_file(checkpoint),
    )
    selector = _json(run_root / "selector.json", {
        "schema_version": 1,
        "kind": inc.SELECTOR_KIND,
        "method": method,
        "task": "pixel",
        "seed": seed,
        "selection_split": "val",
        "selection_source": "validation_only",
        "validation_only": True,
        "test_or_heldout_opened": False,
        "source_config_sha256": sha256_file(config),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(case.manifest),
        "split_sha256": sha256_file(case.splits),
        "amendment_sha256": sha256_file(case.amendment),
        "threshold_amendment_sha256": sha256_file(case.amendment),
        "raw_validation_probability_card_sha256": sha256_file(raw_val),
        "validation_probability_card_sha256": sha256_file(selected_val),
        "probabilities_are_pre_threshold": True,
        "derived_from_binary_mask": False,
        "debug_limited": False,
        "allowed_grid": list(inc.AMENDED_GRID),
        "selected_threshold": 0.4,
    })
    profile = _json(run_root / "profile.json", {"epochs": 3})
    record = {
        "method": method,
        "seed": seed,
        "config": _identity(config),
        "run_card": _identity(run_card),
        "checkpoint": _identity(checkpoint),
        "selector_receipt": _identity(selector),
        "raw_validation_probability_card": _identity(raw_val),
        "validation_probability_card": _identity(selected_val),
        "training_lineage": "amended_training",
        "per_epoch_training_profile_card": _identity(profile),
        "masks_out": str(case.root / "additional_outputs" / "masks"),
        "probabilities_out": str(case.root / "additional_outputs" / "probabilities"),
        "card_out": str(case.root / "additional_outputs" / "inference.json"),
        "work_dir": str(case.root / "additional_outputs" / "work"),
        "seed_delivery_out": str(case.root / "seed_deliveries" / f"{method}-{seed}"),
    }
    spec = _json(case.root / "additional_spec.json", {
        "schema_version": 1,
        "kind": inc.ADDITIONAL_PLAN_SPEC_KIND,
        "manifest": _identity(case.manifest),
        "splits": _identity(case.splits),
        "threshold_amendment": _identity(case.amendment),
        "cache_receipt": _identity(case.cache),
        "stage2_unet_seed0_delivery": _identity(case.root / "stage2" / "delivery.json"),
        "expected_prediction_count": 2,
        "code_root": str(ROOT),
        "expected_inference_git_commit": "a" * 40,
        "expected_scoring_git_commit": "b" * 40,
        "python_executable": sys.executable,
        "allowed_gpu_indices": [0],
        "prior_seed_deliveries": [_identity(path) for path in priors],
        "ram_gib": {"unetpp.seed1": 2.0},
        "runs": [record],
    })
    gate, plan, queue = inc.prepare_additional(
        spec_path=spec,
        expected_spec_sha256=sha256_file(spec),
        restricted_root=case.root,
        gate_out=case.root / "additional_gate.json",
        plan_out=case.root / "additional_plan.json",
        queue_out=case.root / "additional_queue.yaml",
    )
    assert gate["satisfies_stage4_multi_seed_gate"] is False
    assert {row["method"] for row in plan["prior_seed_deliveries"]} == set(inc.SEED0_INFERENCE_METHODS)
    assert all(row["seed"] == 0 for row in plan["prior_seed_deliveries"])
    assert [(row["method"], row["seed"]) for row in plan["runs"]] == [(method, seed)]
    assert queue["jobs"][0]["name"] == "infer.unetpp.seed1"


def _import_case(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "restricted"
    root.mkdir()
    manifest = root / "manifest.csv"
    manifest.write_text("synthetic\n", encoding="utf-8")
    splits = _json(root / "splits.json", {"splits": {}})
    methods = {}
    delivery_paths = []
    for method in inc.NATIVE_METHODS:
        valid = [0, 1]
        seeds = {}
        for seed in inc.REQUIRED_SEEDS:
            digests = {
                "config_sha256": hashlib.sha256(f"config-{method}-{seed}".encode()).hexdigest(),
                "run_card_sha256": hashlib.sha256(f"run-{method}-{seed}".encode()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(f"checkpoint-{method}-{seed}".encode()).hexdigest(),
            }
            seeds[str(seed)] = {"status": "valid" if seed in valid else "missing", **digests}
            if seed in valid and (method, seed) != ("unet", 0):
                delivery_paths.append(_scored_bundle(root, method, seed, digests))
        methods[method] = {
            "valid_seeds": valid,
            "failed_seeds": [],
            "missing_seeds": [2],
            "seeds": seeds,
        }
    gate = _json(root / "readiness.json", {
        "schema_version": 1,
        "kind": inc.READINESS_GATE_KIND,
        "status": "PASS_WITH_WARNINGS",
        "minimum_valid_seeds_per_method": 2,
        "test_universe": {"ordered_source_images_sha256": "f" * 64},
        "identities": {
            "manifest_sha256": sha256_file(manifest),
            "splits_sha256": sha256_file(splits),
        },
        "methods": methods,
    })
    stage2 = root / "stage2"
    stage2_results = stage2 / "results"
    stage2_summary = _summary(stage2_results / "summary.csv", "unet", 0)
    stage2_invocations = []
    for table in ("A", "A_prime"):
        per = stage2_results / f"per_image_unet_{table}_seed0.csv"
        pd.DataFrame([{"image_id": "synthetic", "dice": 0.5}]).to_csv(per, index=False)
        stage2_invocations.append({"table": table, "per_image": _identity(per)})
    stage2_receipt = _json(stage2 / "scoring_receipt.json", {
        "schema_version": 1,
        "kind": inc.SCORING_RECEIPT_KIND,
        "method": "unet",
        "seed": 0,
        "invocations": stage2_invocations,
        "scientific_summary": _identity(stage2_summary),
    })
    unet0_gate = methods["unet"]["seeds"]["0"]
    stage2_payload = {
        "schema_version": 1,
        "kind": "stage2_threshold_reselection_delivery",
        "outputs": {"results/summary.csv": sha256_file(stage2_summary)},
        "threshold_policy_attestation": {
            "records": [{
                "config": {"sha256": unet0_gate["config_sha256"]},
                "run_card": {"sha256": unet0_gate["run_card_sha256"]},
                "checkpoint": {"sha256": unet0_gate["checkpoint_sha256"]},
                "scoring_receipt": _identity(stage2_receipt),
            }],
        },
    }
    stage2_delivery = _json(stage2 / "delivery.json", stage2_payload)
    spec = _json(root / "import_spec.json", {
        "schema_version": 1,
        "kind": inc.IMPORT_SPEC_KIND,
        "manifest": _identity(manifest),
        "splits": _identity(splits),
        "readiness_gate": _identity(gate),
        "stage2_unet_seed0_delivery": _identity(stage2_delivery),
        "seed_deliveries": [_identity(path) for path in delivery_paths],
    })
    return SimpleNamespace(root=root, spec=spec, gate=gate, stage2=stage2_delivery)


def test_import_requires_g4_two_seed_gate_and_never_rescores(tmp_path: Path) -> None:
    case = _import_case(tmp_path)
    out = case.root / "combined"
    delivery = inc.import_deliveries(
        spec_path=case.spec,
        expected_spec_sha256=sha256_file(case.spec),
        restricted_root=case.root,
        bundle_out=out,
    )
    assert delivery["gate_evidence"]["status"] == "PASS"
    assert delivery["stage2_unet_seed0"]["reused_reselected_not_rescored"] is True
    assert all(row["imported_not_rescored"] for row in delivery["newly_scored_seeds"])
    assert len(delivery["newly_scored_seeds"]) == 15
    combined = pd.read_csv(out / "results" / "summary.csv")
    assert len(combined) == 32  # 16 successful method/seed runs x two tables
    before = inc._directory_hashes(out, exclude=("delivery.json",))
    assert inc.import_deliveries(
        spec_path=case.spec,
        expected_spec_sha256=sha256_file(case.spec),
        restricted_root=case.root,
        bundle_out=out,
    ) == delivery
    assert inc._directory_hashes(out, exclude=("delivery.json",)) == before


def test_import_rejects_one_seed_method(tmp_path: Path) -> None:
    case = _import_case(tmp_path)
    gate = json.loads(case.gate.read_text())
    gate["methods"]["drunet"]["valid_seeds"] = [0]
    gate["methods"]["drunet"]["missing_seeds"] = [1, 2]
    gate["methods"]["drunet"]["seeds"]["1"]["status"] = "missing"
    _json(case.gate, gate)
    spec = json.loads(case.spec.read_text())
    spec["readiness_gate"] = _identity(case.gate)
    _json(case.spec, spec)
    with pytest.raises(inc.IncrementalError, match="does not satisfy incremental G3/G4"):
        inc.import_deliveries(
            spec_path=case.spec,
            expected_spec_sha256=sha256_file(case.spec),
            restricted_root=case.root,
            bundle_out=case.root / "combined",
        )


def test_import_requires_seed0_even_when_two_later_seeds_are_valid(tmp_path: Path) -> None:
    case = _import_case(tmp_path)
    gate = json.loads(case.gate.read_text())
    gate["methods"]["drunet"]["valid_seeds"] = [1, 2]
    gate["methods"]["drunet"]["missing_seeds"] = [0]
    gate["methods"]["drunet"]["seeds"]["0"]["status"] = "missing"
    gate["methods"]["drunet"]["seeds"]["2"]["status"] = "valid"
    _json(case.gate, gate)
    spec = json.loads(case.spec.read_text())
    spec["readiness_gate"] = _identity(case.gate)
    _json(case.spec, spec)
    with pytest.raises(inc.IncrementalError, match="does not satisfy incremental G3/G4"):
        inc.import_deliveries(
            spec_path=case.spec,
            expected_spec_sha256=sha256_file(case.spec),
            restricted_root=case.root,
            bundle_out=case.root / "combined",
        )
