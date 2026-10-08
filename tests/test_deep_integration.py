"""Tiny CPU end-to-end smoke test for train -> infer -> shared evaluator."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from segbench.artifact_identity import sha256_file
from segbench.deep_models import build_model
from segbench.deep_utils import resume_config_digest


ROOT = Path(__file__).resolve().parents[1]


def _write_threshold_receipt(
    path: Path,
    config_path: Path,
    checkpoint_path: Path,
    manifest: Path,
    splits: Path,
) -> Path:
    """Write an internally consistent synthetic validation-selection receipt."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    candidates = sorted(float(value) for value in config["training"]["threshold_candidates"])
    selected = float(checkpoint["selected_threshold"])
    assert selected in candidates
    scores = {f"{value:.8g}": (1.0 if value == selected else 0.0) for value in candidates}
    payload = {
        "schema_version": 1,
        "kind": "expanded_validation_threshold_selector_receipt",
        "method": config["method"],
        "task": config["task"],
        "selection_split": "val",
        "selection_source": "validation_only",
        "validation_only": True,
        "test_or_heldout_opened": False,
        "probabilities_are_pre_threshold": True,
        "storage_dtype": "float16",
        "derived_from_binary_mask": False,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest),
        "split_sha256": sha256_file(splits),
        "threshold_amendment_sha256": None,
        "amendment_sha256": None,
        "seed": int(checkpoint["config"]["seed"]),
        "validation_probability_card_sha256": "d" * 64,
        "threshold_candidates": candidates,
        "allowed_grid": candidates,
        "threshold_scores": scores,
        "selected_threshold": selected,
        "debug_limited": False,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_example(root: Path, subject: str, split: str, offset: int) -> dict[str, object]:
    yy, xx = np.mgrid[:32, :32]
    mask = yy < 16 + offset
    image = np.where(mask, 40 + offset, 210 - offset).astype(np.uint8)
    image_path = root / f"{subject}_image.png"
    mask_path = root / f"{subject}_mask.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    return {
        "image_id": f"syn_{subject}",
        "image_path": str(image_path),
        "source_sha256": sha256_file(image_path),
        "mask_path": str(mask_path),
        "subject_id": subject,
        "subset": "finger_latent",
        "ppi": 500,
        "split": split,
    }


def test_tiny_cpu_pixel_workflow(tmp_path: Path) -> None:
    rows = [
        _write_example(tmp_path, "train_a", "train", 0),
        _write_example(tmp_path, "train_b", "train", 1),
        _write_example(tmp_path, "val_a", "val", 0),
        _write_example(tmp_path, "test_a", "test", 1),
    ]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{key: value for key, value in row.items() if key != "split"} for row in rows]).to_csv(
        manifest, index=False
    )
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"seed": 0, "splits": {row["subject_id"]: row["split"] for row in rows}}),
        encoding="utf-8",
    )
    config = yaml.safe_load((ROOT / "configs" / "unet.yaml").read_text(encoding="utf-8"))
    config["data"]["crop_size"] = 32
    config["model"]["base_channels"] = 2
    config["training"].update(
        epochs=1,
        patience=1,
        batch_size=2,
        num_workers=8,
    )
    config["inference"].update(tile_size=0, tile_overlap=0)
    config["runtime"].update(device="cpu", memory_fraction=1.0)
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "run"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["SEG_BATCH_SCALE"] = "0.5"
    environment["SEG_GRAD_ACCUM_SCALE"] = "2.0"
    environment["SEG_OOM_RETRY_COUNT"] = "1"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_seg.py"),
            "--allow-unamended-debug",
            "--limit-train", "2", "--limit-val", "1",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--out",
            str(run_dir),
            "--seed",
            "1",
        ],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert {path.name for path in run_dir.glob("*.pt")} == {"best.pt", "last.pt"}
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["training"]["batch_size"] == 1
    assert checkpoint["config"]["training"]["gradient_accumulation"] == 2
    assert checkpoint["config"]["seed"] == 1
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    profile_card = json.loads(
        (run_dir / "per_epoch_profile.json").read_text(encoding="utf-8")
    )
    assert profile_card["kind"] == "per_epoch_training_profile_card"
    assert profile_card["run_card_sha256"] == sha256_file(run_dir / "run.json")
    assert profile_card["epochs_recorded"] == 1
    assert profile_card["complete"] is True
    assert run_record["last_epoch"] == 0
    assert run_record["epochs_completed"] == 1
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert np.isfinite(history[0]["validation_loss"])
    assert run_record["validation_loss_first"] == history[0]["validation_loss"]
    assert run_record["validation_loss_last"] == history[0]["validation_loss"]
    assert run_record["validation_loss_min"] == history[0]["validation_loss"]
    assert run_record["source_config_sha256"]
    assert set(run_record["checkpoint_sha256"]) == {"best.pt", "last.pt"}
    assert all(
        len(value) == 64 for value in run_record["checkpoint_sha256"].values()
    )
    masks = tmp_path / "predictions"
    timing = tmp_path / "timing.json"
    threshold_receipt = _write_threshold_receipt(
        tmp_path / "threshold.json",
        config_path,
        run_dir / "best.pt",
        manifest,
        splits,
    )
    inference_command = [
            sys.executable,
            str(ROOT / "scripts" / "infer_masks.py"),
            "--allow-unamended-debug",
            "--limit", "1",
            "--config",
            str(config_path),
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--subset",
            "finger_latent",
            "--split",
            "test",
            "--out",
            str(masks),
            "--timing-out",
            str(timing),
            "--threshold-receipt",
            str(threshold_receipt),
            "--warmup",
            "0",
        ]
    subprocess.run(
        inference_command,
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    prediction = cv2.imread(str(masks / "syn_test_a.png"), cv2.IMREAD_UNCHANGED)
    assert prediction is not None and prediction.shape == (32, 32)
    assert set(np.unique(prediction).tolist()).issubset({0, 255})
    probability = np.load(tmp_path / "predictions_probabilities" / "syn_test_a.npy")
    assert probability.dtype == np.float16 and probability.shape == (32, 32)
    timing_card = json.loads(timing.read_text())
    assert timing_card["probability_set"]["complete"] is True
    assert timing_card["probabilities_are_pre_threshold"] is True
    assert timing_card["derived_from_binary_mask"] is False
    first_timing_records = timing_card["per_image"]
    immutable_card_sha = sha256_file(timing)
    immutable_mask_sha = sha256_file(masks / "syn_test_a.png")
    immutable_probability_sha = sha256_file(
        tmp_path / "predictions_probabilities" / "syn_test_a.npy"
    )
    subprocess.run(
        [*inference_command, "--skip-valid"],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    resumed_timing = json.loads(timing.read_text(encoding="utf-8"))
    assert sha256_file(timing) == immutable_card_sha
    assert sha256_file(masks / "syn_test_a.png") == immutable_mask_sha
    assert sha256_file(
        tmp_path / "predictions_probabilities" / "syn_test_a.npy"
    ) == immutable_probability_sha
    assert resumed_timing["per_image"] == first_timing_records
    assert resumed_timing["timing_resume_policy"] == "authenticated_per_image_records"
    refused = subprocess.run(
        inference_command,
        check=False,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert sha256_file(timing) == immutable_card_sha
    assert sha256_file(masks / "syn_test_a.png") == immutable_mask_sha
    assert sha256_file(
        tmp_path / "predictions_probabilities" / "syn_test_a.npy"
    ) == immutable_probability_sha
    result_dir = tmp_path / "scores"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_masks.py"),
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--subsets",
            "finger_latent",
            "--method",
            "unet_smoke",
            "--display-name",
            "U-Net synthetic smoke",
            "--pred-template",
            str(masks / "{image_id}.png"),
            "--pred-ppi",
            "500",
            "--out",
            str(result_dir),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    summary = pd.read_csv(result_dir / "summary.csv")
    assert len(summary) == 1 and int(summary.iloc[0]["n_images"]) == 1

    preempt_file = tmp_path / "preempt.request"
    preempt_file.touch()
    preempt_environment = environment.copy()
    preempt_environment["SEG_PREEMPT_FILE"] = str(preempt_file)
    preempt_run = tmp_path / "preempt_run"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_seg.py"),
            "--allow-unamended-debug",
            "--limit-train", "2", "--limit-val", "1",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--out",
            str(preempt_run),
        ],
        check=False,
        cwd=ROOT,
        env=preempt_environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 75
    assert (preempt_run / "last.pt").is_file()

    patch_config = yaml.safe_load(
        (ROOT / "configs" / "lenet.yaml").read_text(encoding="utf-8")
    )
    patch_config["data"]["blocks_per_class_per_image"] = 2
    patch_config["training"].update(
        epochs=1,
        patience=1,
        batch_size=4,
        num_workers=8,
    )
    patch_config["inference"].update(patch_batch_size=4, min_component_blocks=1)
    patch_config["runtime"].update(device="cpu", memory_fraction=1.0)
    patch_config_path = tmp_path / "tiny_patch.yaml"
    patch_config_path.write_text(
        yaml.safe_dump(patch_config, sort_keys=False), encoding="utf-8"
    )
    patch_run = tmp_path / "patch_run"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_patch.py"),
            "--allow-unamended-debug",
            "--limit-train", "2", "--limit-val", "1",
            "--config",
            str(patch_config_path),
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--out",
            str(patch_run),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert {path.name for path in patch_run.glob("*.pt")} == {"best.pt", "last.pt"}
    patch_profile = json.loads(
        (patch_run / "per_epoch_profile.json").read_text(encoding="utf-8")
    )
    assert patch_profile["kind"] == "per_epoch_training_profile_card"
    assert patch_profile["run_card_sha256"] == sha256_file(patch_run / "run.json")
    patch_masks = tmp_path / "patch_predictions"
    patch_threshold_receipt = _write_threshold_receipt(
        tmp_path / "patch_threshold.json",
        patch_config_path,
        patch_run / "best.pt",
        manifest,
        splits,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "infer_masks.py"),
            "--allow-unamended-debug",
            "--limit", "1",
            "--config",
            str(patch_config_path),
            "--checkpoint",
            str(patch_run / "best.pt"),
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--subset",
            "finger_latent",
            "--split",
            "test",
            "--out",
            str(patch_masks),
            "--timing-out",
            str(tmp_path / "patch_timing.json"),
            "--threshold-receipt",
            str(patch_threshold_receipt),
            "--warmup",
            "0",
        ],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    patch_prediction = cv2.imread(
        str(patch_masks / "syn_test_a.png"), cv2.IMREAD_UNCHANGED
    )
    assert patch_prediction is not None and patch_prediction.shape == (32, 32)
    assert set(np.unique(patch_prediction).tolist()).issubset({0, 255})
    patch_probability = np.load(
        tmp_path / "patch_predictions_probabilities" / "syn_test_a.npy"
    )
    assert patch_probability.dtype == np.float16 and patch_probability.ndim == 2
    patch_card = json.loads((tmp_path / "patch_timing.json").read_text())
    assert patch_card["probability_set"]["complete"] is True

    autoencoder_config = yaml.safe_load(
        (ROOT / "configs" / "cnn_ae.yaml").read_text(encoding="utf-8")
    )
    autoencoder_config["data"]["blocks_per_class_per_image"] = 2
    autoencoder_config["model"]["latent_channels"] = 8
    autoencoder_config["training"].update(
        epochs=1,
        patience=1,
        batch_size=4,
        num_workers=8,
        autoencoder_pretrain_epochs=1,
    )
    autoencoder_config["inference"].update(
        patch_batch_size=4, min_component_blocks=1
    )
    autoencoder_config["runtime"].update(device="cpu", memory_fraction=1.0)
    autoencoder_config_path = tmp_path / "tiny_autoencoder.yaml"
    autoencoder_config_path.write_text(
        yaml.safe_dump(autoencoder_config, sort_keys=False), encoding="utf-8"
    )
    autoencoder_run = tmp_path / "autoencoder_run"
    autoencoder_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_patch.py"),
        "--allow-unamended-debug",
        "--limit-train", "2", "--limit-val", "1",
        "--config",
        str(autoencoder_config_path),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--out",
        str(autoencoder_run),
    ]
    autoencoder_preempt = tmp_path / "autoencoder.preempt"
    autoencoder_preempt.touch()
    autoencoder_environment = environment.copy()
    autoencoder_environment["SEG_PREEMPT_FILE"] = str(autoencoder_preempt)
    interrupted_autoencoder = subprocess.run(
        autoencoder_command,
        check=False,
        cwd=ROOT,
        env=autoencoder_environment,
        capture_output=True,
        text=True,
    )
    assert interrupted_autoencoder.returncode == 75
    assert (autoencoder_run / "pretrain_last.pt").is_file()
    autoencoder_preempt.unlink()
    subprocess.run(
        autoencoder_command + ["--resume", "auto"],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    autoencoder_record = json.loads(
        (autoencoder_run / "run.json").read_text(encoding="utf-8")
    )
    assert len(autoencoder_record["autoencoder_pretrain_loss"]) == 1
    assert {path.name for path in autoencoder_run.glob("*.pt")} == {
        "best.pt",
        "last.pt",
    }


def test_drunet_mc_dropout_variance_extra_is_separate_from_table_mask(
    tmp_path: Path,
) -> None:
    rows = [
        _write_example(tmp_path, "train_a", "train", 0),
        _write_example(tmp_path, "test_a", "test", 1),
    ]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "split"} for row in rows]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {"seed": 0, "splits": {row["subject_id"]: row["split"] for row in rows}}
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load(
        (ROOT / "configs" / "drunet.yaml").read_text(encoding="utf-8")
    )
    config["model"]["base_channels"] = 2
    config["inference"].update(tile_size=0, tile_overlap=0)
    config["runtime"].update(device="cpu", memory_fraction=1.0)
    config_path = tmp_path / "tiny_drunet.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    torch.manual_seed(17)
    model = build_model("drunet", config["model"])
    checkpoint = {
        "config": config,
        "resume_config_sha256": resume_config_digest(config),
        "method": "drunet",
        "task": "pixel",
        "normalization": {"mean": 0.5, "std": 0.25, "source_split": "train"},
        "selected_threshold": 0.5,
        "model_state": model.state_dict(),
    }
    checkpoint_path = tmp_path / "drunet.pt"
    torch.save(checkpoint, checkpoint_path)
    threshold_receipt = _write_threshold_receipt(
        tmp_path / "drunet_threshold.json",
        config_path,
        checkpoint_path,
        manifest,
        splits,
    )

    masks = tmp_path / "masks"
    variance_dir = tmp_path / "private_variance"
    timing = tmp_path / "timing.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable,
        str(ROOT / "scripts" / "infer_masks.py"),
        "--allow-unamended-debug",
        "--limit", "1",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--subset",
        "finger_latent",
        "--split",
        "test",
        "--out",
        str(masks),
        "--timing-out",
        str(timing),
        "--threshold-receipt",
        str(threshold_receipt),
        "--warmup",
        "0",
        "--mc-dropout-samples",
        "6",
        "--mc-variance-out",
        str(variance_dir),
    ]
    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    table_mask = cv2.imread(
        str(masks / "syn_test_a.png"), cv2.IMREAD_UNCHANGED
    )
    assert table_mask is not None
    assert set(np.unique(table_mask).tolist()).issubset({0, 255})
    variance = np.load(
        variance_dir / "syn_test_a_variance.npy", allow_pickle=False
    )
    assert variance.shape == table_mask.shape
    assert variance.dtype == np.float32
    assert np.isfinite(variance).all() and (variance >= 0.0).all()
    assert float(variance.max()) > 0.0
    report = json.loads(timing.read_text(encoding="utf-8"))
    assert report["mc_dropout"] == {
        "enabled": True,
        "samples": 6,
        "requested_images": 1,
        "written": 1,
        "skipped_valid": 0,
        "output_format": "float32_npy",
        "deterministic_table_masks": True,
        "excluded_from_ms_per_img": True,
        "per_image_stable_seed": True,
    }
    variance_path = variance_dir / "syn_test_a_variance.npy"
    variance_path.unlink()
    subprocess.run(
        command + ["--skip-valid"],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    resumed_variance = np.load(variance_path, allow_pickle=False)
    np.testing.assert_array_equal(resumed_variance, variance)
    resumed_report = json.loads(timing.read_text(encoding="utf-8"))
    assert resumed_report == report
    assert resumed_report["mc_dropout"]["written"] == 1


def test_epoch_boundary_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    rows = [
        _write_example(tmp_path, "train_a", "train", 0),
        _write_example(tmp_path, "train_b", "train", 1),
        _write_example(tmp_path, "val_a", "val", 0),
    ]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "split"} for row in rows]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {"seed": 0, "splits": {row["subject_id"]: row["split"] for row in rows}}
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load((ROOT / "configs" / "unet.yaml").read_text(encoding="utf-8"))
    config["data"]["crop_size"] = 32
    config["model"]["base_channels"] = 2
    config["training"].update(epochs=2, patience=2, batch_size=2, num_workers=8)
    config["inference"].update(tile_size=0, tile_overlap=0)
    config["runtime"].update(device="cpu", memory_fraction=1.0)
    config_path = tmp_path / "resume.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""

    command_prefix = [
        sys.executable,
        str(ROOT / "scripts" / "train_seg.py"),
        "--allow-unamended-debug",
        "--limit-train", "2", "--limit-val", "1",
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
    ]
    uninterrupted = tmp_path / "uninterrupted"
    subprocess.run(
        command_prefix + ["--out", str(uninterrupted)],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    resumed = tmp_path / "resumed"
    preempt = tmp_path / "resume.preempt"
    preempt.touch()
    interrupted_environment = environment.copy()
    interrupted_environment["SEG_PREEMPT_FILE"] = str(preempt)
    first = subprocess.run(
        command_prefix + ["--out", str(resumed)],
        check=False,
        cwd=ROOT,
        env=interrupted_environment,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 75
    preempt.unlink()
    subprocess.run(
        command_prefix + ["--out", str(resumed), "--resume", "auto"],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    direct_checkpoint = torch.load(
        uninterrupted / "last.pt", map_location="cpu", weights_only=False
    )
    resumed_checkpoint = torch.load(
        resumed / "last.pt", map_location="cpu", weights_only=False
    )
    assert direct_checkpoint["epoch"] == resumed_checkpoint["epoch"] == 1
    for name, direct_tensor in direct_checkpoint["model_state"].items():
        torch.testing.assert_close(
            direct_tensor,
            resumed_checkpoint["model_state"][name],
            rtol=0,
            atol=0,
        )
    direct_history = json.loads((uninterrupted / "history.json").read_text())
    resumed_history = json.loads((resumed / "history.json").read_text())
    def without_operational_profiles(history: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {key: value for key, value in entry.items() if key != "epoch_profile"}
            for entry in history
        ]

    assert without_operational_profiles(direct_history) == without_operational_profiles(
        resumed_history
    )
    for history in (direct_history, resumed_history):
        for entry in history:
            profile = entry["epoch_profile"]
            assert profile["data_wait_seconds"] >= 0
            assert profile["compute_seconds"] >= 0
