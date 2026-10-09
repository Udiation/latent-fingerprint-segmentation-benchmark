"""Synthetic coverage for the fail-closed SD302g cross-domain handoff."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_b1_crossdomain")
finalize = load("finalize_b1_crossdomain")
manifest_builder = load("prepare_sd302g_inference_manifest")
infer_masks = load("infer_masks")

from segbench import probability_artifacts  # noqa: E402
from segbench import b1_crossdomain  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_case(tmp_path: Path) -> tuple[Namespace, dict[str, Path]]:
    subjects = {"subject-a": "test", "subject-b": "test"}
    splits = tmp_path / "splits.json"
    write_json(splits, {"seed": 7, "splits": subjects})
    split_hash = prepare.sha256_file(splits)

    verified = tmp_path / "verified"
    rows = []
    for index, (image_id, subject) in enumerate((('example-a', 'subject-a'), ('example-b', 'subject-b'))):
        image = np.full((20, 24), 80 + index, dtype=np.uint8)
        hull = np.zeros_like(image)
        hull[4:17, 5:20] = 255
        image_path = verified / "images" / f"{image_id}.png"
        hull_path = verified / "hulls" / f"{image_id}.png"
        points_path = verified / "minutiae" / f"{image_id}.csv"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        hull_path.parent.mkdir(parents=True, exist_ok=True)
        points_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(hull_path), hull)
        write_csv(points_path, [{"x_px": 7, "y_px": 6}, {"x_px": 17, "y_px": 14}])
        rows.append(
            {
                "image_id": image_id,
                "image_path": f"images/{image_id}.png",
                "hull_path": f"hulls/{image_id}.png",
                "minutiae_path": f"minutiae/{image_id}.csv",
                "subject_id": subject,
                "split": "test",
                "group": "baseline" if index == 0 else "challengers",
                "device": f"device-{index}",
                "capture": "roll" if index == 0 else "slap",
                "width": 24,
                "height": 20,
                "ppi": 1000,
            }
        )
    verified_manifest = verified / "manifest.csv"
    write_csv(verified_manifest, rows)
    write_json(
        verified / "provenance.json",
        {
            "schema": "cureblock.sd302g.build-provenance.v1",
            "protocol": prepare.PROTOCOL,
            "manifest_sha256": prepare.sha256_file(verified_manifest),
            "split_sha256": split_hash,
            "verification_status": "SYNTHETIC_TEST_ONLY",
            "bootstrap": {
                "seed": 7,
                "replicates": 1000,
                "unit": "subject",
                "interval": "percentile_95",
                "paired": True,
            },
        },
    )
    inference_manifest = tmp_path / "private" / "b1_inference.csv"
    manifest_builder.build(verified_manifest, splits, inference_manifest, "finger_exemplar")
    inference_receipt = inference_manifest.with_suffix(".csv.receipt.json")

    dataset1_manifest = tmp_path / "dataset1.csv"
    dataset1_manifest.write_text("synthetic Dataset-1 identity\n", encoding="utf-8")
    config = tmp_path / "unet.yaml"
    config.write_text("method: unet\ntask: pixel\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    amendment = tmp_path / "amendment.json"
    write_json(amendment, {"synthetic": True})
    commit = "a" * 40
    run_card = tmp_path / "run.json"
    write_json(
        run_card,
        {
            "method": "unet",
            "task": "pixel",
            "seed": 0,
            "debug_limited": False,
            "source_config_sha256": prepare.sha256_file(config),
            "manifest_sha256": prepare.sha256_file(dataset1_manifest),
            "split_sha256": split_hash,
            "threshold_amendment_sha256": prepare.sha256_file(amendment),
            "checkpoint_sha256": {"best.pt": prepare.sha256_file(checkpoint)},
            "parameter_count": {"total": 1_234_567, "trainable": 1_234_567},
            "provenance": {
                "capture_timing": "process_start",
                "git_commit": commit,
                "git_dirty": False,
            },
        },
    )
    selector = tmp_path / "selector.json"
    scores = {f"{value:.8g}": 0.7 + index / 100 for index, value in enumerate((.1, .2, .3, .4, .5, .6, .7))}
    scores["0.5"] = 0.99
    write_json(
        selector,
        {
            "schema_version": 1,
            "kind": "expanded_validation_threshold_selector_receipt",
            "method": "unet",
            "task": "pixel",
            "seed": 0,
            "selection_split": "val",
            "checkpoint_sha256": prepare.sha256_file(checkpoint),
            "source_config_sha256": prepare.sha256_file(config),
            "manifest_sha256": prepare.sha256_file(dataset1_manifest),
            "split_sha256": split_hash,
            "threshold_amendment_sha256": prepare.sha256_file(amendment),
            "amendment_sha256": prepare.sha256_file(amendment),
            "selection_source": "validation_only",
            "validation_only": True,
            "test_or_heldout_opened": False,
            "probabilities_are_pre_threshold": True,
            "storage_dtype": "float16",
            "derived_from_binary_mask": False,
            "debug_limited": False,
            "allowed_grid": [.1, .2, .3, .4, .5, .6, .7],
            "threshold_candidates": [.1, .2, .3, .4, .5, .6, .7],
            "threshold_scores": scores,
            "selected_threshold": .5,
            "validation_probability_card_sha256": "b" * 64,
        },
    )
    gate = tmp_path / "reference_gate.json"
    report = tmp_path / "reference_report.json"
    write_json(gate, {"synthetic": True})
    write_json(report, {"synthetic": True})
    predictions = tmp_path / "outputs" / "masks"
    probabilities = tmp_path / "outputs" / "probabilities"
    timing_card = tmp_path / "outputs" / "timing.json"
    score_out = tmp_path / "outputs" / "scored"
    journal = tmp_path / "outputs" / ".scored.journal.json"
    authorization = tmp_path / "outputs" / "authorization.json"
    plan = tmp_path / "outputs" / "plan.json"
    args = Namespace(
        method_key="unet_seed0",
        display_name="U-Net (seed 0)",
        config=config,
        checkpoint=checkpoint,
        training_run_card=run_card,
        selector_receipt=selector,
        threshold_amendment=amendment,
        dataset1_manifest=dataset1_manifest,
        b1_verified_manifest=verified_manifest,
        b1_inference_manifest=inference_manifest,
        b1_inference_receipt=inference_receipt,
        reference_gate=gate,
        reference_report=report,
        splits=splits,
        inference_script=SCRIPTS / "infer_masks.py",
        finalizer_script=SCRIPTS / "finalize_b1_crossdomain.py",
        evaluator=SCRIPTS / "evaluate_hull.py",
        python=Path(sys.executable),
        expected_inference_commit=commit,
        device="cpu",
        memory_fraction=.3,
        warmup=0,
        predictions=predictions,
        probabilities=probabilities,
        inference_card=timing_card,
        score_out=score_out,
        score_journal=journal,
        authorization_out=authorization,
        plan_out=plan,
        synthetic_test_only=True,
    )
    paths = {
        "verified": verified,
        "verified_manifest": verified_manifest,
        "inference_manifest": inference_manifest,
        "selector": selector,
        "amendment": amendment,
        "config": config,
        "checkpoint": checkpoint,
        "splits": splits,
        "predictions": predictions,
        "probabilities": probabilities,
        "timing": timing_card,
        "score": score_out,
        "journal": journal,
        "authorization": authorization,
        "plan": plan,
    }
    return args, paths


def materialize_inference(paths: dict[str, Path], commit: str = "a" * 40) -> None:
    paths["predictions"].mkdir(parents=True)
    paths["probabilities"].mkdir(parents=True)
    _, inference_rows = finalize.read_csv(paths["inference_manifest"], "inference")
    test_rows = sorted(inference_rows, key=lambda row: row["image_id"])
    per_image = []
    probability_items = []
    for row in test_rows:
        image_id = row["image_id"]
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        assert image is not None
        probability = np.ones((image.shape[0] // 2, image.shape[1] // 2), dtype=np.float16)
        probability_path = paths["probabilities"] / f"{image_id}.npy"
        np.save(probability_path, probability, allow_pickle=False)
        mask_path = paths["predictions"] / f"{image_id}.png"
        assert cv2.imwrite(str(mask_path), np.full(probability.shape, 255, dtype=np.uint8))
        probability_items.append((image_id, probability_path))
        per_image.append(
            {
                "image_id": image_id,
                "ms_per_img": 2.0,
                "mask_sha256": finalize.sha256_file(mask_path),
                "probability_sha256": finalize.sha256_file(probability_path),
            }
        )
    source_hashes = {row["image_id"]: row["source_sha256"] for row in test_rows}
    probability_set = probability_artifacts.set_identity(
        probability_items, source_sha256=source_hashes
    )
    probability_set["complete"] = True
    authorization_hash = finalize.sha256_file(paths["authorization"])
    authorization = finalize.read_json(paths["authorization"], "authorization")
    model = authorization["model"]
    target = authorization["target_domain"]
    write_json(
        paths["timing"],
        {
            "schema_version": 1,
            "status": "complete",
            "complete": True,
            "kind": "complete_probability_map_card",
            "method": "unet",
            "task": "pixel",
            "seed": 0,
            "subset": "finger_exemplar",
            "split": "test",
            "selected_validation_threshold": .5,
            "threshold_receipt_sha256": model["selector_receipt_sha256"],
            "threshold_amendment_sha256": model["threshold_amendment_sha256"],
            "cross_domain_authorization_sha256": authorization_hash,
            "source_config_sha256": model["source_config_sha256"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "manifest_sha256": target["manifest_sha256"],
            "split_sha256": target["split_sha256"],
            "labels_opened": False,
            "probabilities_are_pre_threshold": True,
            "derived_from_binary_mask": False,
            "binary_mask_reinterpreted": False,
            "probability_maps_required": True,
            "debug_limited": False,
            "preempted": False,
            "target_ppi": 500.0,
            "mean_ms_per_img": 2.0,
            "device": "CPU",
            "per_image": per_image,
            "probability_set": probability_set,
            "mask_derivation_verification": {
                "complete": True,
                "count": 2,
                "all_masks_rederived_from_retained_probabilities": True,
            },
            "provenance": {
                "capture_timing": "process_start",
                "git_commit": commit,
                "git_dirty": False,
            },
        },
    )


def test_prepares_ground_truth_blind_cross_domain_plan(tmp_path: Path) -> None:
    args, paths = synthetic_case(tmp_path)
    plan = prepare.prepare(args)
    authorization = prepare.read_json(paths["authorization"], "authorization")
    assert plan["launch_performed"] is False
    assert authorization["constraints"]["sd302g_supervised_fitting"] is False
    assert authorization["target_domain"]["test_subjects"] == 2
    assert "--cross-domain-authorization" in plan["inference"]["argv"]
    assert all(row["mask_path"] == "" for row in prepare.read_csv(paths["inference_manifest"], "inference")[1])


def test_selector_accepts_source_manifest_only_with_bound_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, paths = synthetic_case(tmp_path)
    prepare.prepare(args)
    authorization = prepare.read_json(paths["authorization"], "authorization")
    monkeypatch.setattr(
        infer_masks.probability_artifacts,
        "validate_threshold_amendment",
        lambda _path: prepare.sha256_file(paths["amendment"]),
    )
    namespace = Namespace(
        threshold_amendment=paths["amendment"],
        threshold_receipt=paths["selector"],
        checkpoint=paths["checkpoint"],
        config=paths["config"],
        manifest=paths["inference_manifest"],
        splits=paths["splits"],
        split="test",
        limit=0,
    )
    threshold, _ = infer_masks._threshold_from_receipt(
        namespace, {"method": "unet", "task": "pixel", "seed": 0}, {}, authorization
    )
    assert threshold == .5
    with pytest.raises(ValueError, match="provenance mismatch"):
        infer_masks._threshold_from_receipt(
            namespace, {"method": "unet", "task": "pixel", "seed": 0}, {}, None
        )


def test_inference_authorization_fails_closed_if_target_manifest_changes(
    tmp_path: Path,
) -> None:
    args, paths = synthetic_case(tmp_path)
    prepare.prepare(args)
    b1_crossdomain.validate_for_inference(
        paths["authorization"],
        method="unet",
        task="pixel",
        seed=0,
        config_path=paths["config"],
        checkpoint_path=paths["checkpoint"],
        selector_path=paths["selector"],
        amendment_path=paths["amendment"],
        target_manifest_path=paths["inference_manifest"],
        split_path=paths["splits"],
        subset="finger_exemplar",
        split="test",
        limit=0,
    )
    paths["inference_manifest"].write_text(
        paths["inference_manifest"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target binding changed"):
        b1_crossdomain.validate_for_inference(
            paths["authorization"],
            method="unet",
            task="pixel",
            seed=0,
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            selector_path=paths["selector"],
            amendment_path=paths["amendment"],
            target_manifest_path=paths["inference_manifest"],
            split_path=paths["splits"],
            subset="finger_exemplar",
            split="test",
            limit=0,
        )


def test_exactly_once_scoring_retains_outputs_and_breakdowns(tmp_path: Path) -> None:
    args, paths = synthetic_case(tmp_path)
    prepare.prepare(args)
    materialize_inference(paths)
    receipt = finalize.run(Namespace(authorization=paths["authorization"]))
    assert receipt["no_supervised_sd302g_fitting"] is True
    assert receipt["evaluation"]["hull_reference_gate"]["passed"] is True
    assert set(receipt["evaluation"]["breakdowns"]) >= {"overall", "device", "capture"}
    assert paths["predictions"].is_dir() and paths["probabilities"].is_dir()
    assert paths["journal"].is_file() and paths["score"].is_dir()
    # An authenticated completed replay returns the receipt without invoking metrics again.
    again = finalize.run(Namespace(authorization=paths["authorization"]))
    assert again["score_once_journal_sha256"] == receipt["score_once_journal_sha256"]


def test_journal_blocks_retry_after_interrupted_scoring(tmp_path: Path) -> None:
    args, paths = synthetic_case(tmp_path)
    prepare.prepare(args)
    materialize_inference(paths)
    write_json(paths["journal"], {"kind": "interrupted"})
    with pytest.raises(finalize.FinalizationError, match="automatic retry is forbidden"):
        finalize.run(Namespace(authorization=paths["authorization"]))


def test_rejects_training_run_with_target_domain_fit(tmp_path: Path) -> None:
    args, _ = synthetic_case(tmp_path)
    run = prepare.read_json(args.training_run_card, "run")
    run = dict(run)
    run["target_manifest_sha256"] = "c" * 64
    write_json(args.training_run_card, run)
    with pytest.raises(prepare.PreparationError, match="supervised or aligned"):
        prepare.prepare(args)
