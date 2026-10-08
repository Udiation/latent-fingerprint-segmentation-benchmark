"""Synthetic-only tests for neural data and inference utilities."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import yaml

import segbench.deep_utils as deep_utils
from segbench.deep_utils import (
    PatchDataset,
    PixelCropDataset,
    UnlabelledPatchDataset,
    apply_queue_batch_contract,
    assert_compatible_subject_splits,
    binary_dice,
    blocks_to_postprocessed_mask,
    checkpoint_payload,
    compute_train_normalization,
    examiner_exclusions_for_split,
    full_image_validation_loss,
    load_config,
    predict_pixel_probability,
    read_normalized_image,
    resume_config_digest,
    select_validation_threshold,
    validate_resume_batch_transition,
)


ROOT = Path(__file__).resolve().parents[1]


AUGMENTATION = {
    "horizontal_flip_probability": 0.5,
    "max_rotation_degrees": 15.0,
    "scale_min": 0.9,
    "scale_max": 1.1,
    "brightness": 0.2,
    "contrast": 0.2,
    "gaussian_noise_std": 0.02,
}


def _write_pair(root: Path, name: str, image: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    image_path = root / f"{name}_image.png"
    mask_path = root / f"{name}_mask.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    return {
        "image_id": name,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "subject_id": name,
        "subset": "finger_latent",
        "ppi": 500,
    }


def test_normalization_reads_training_rows_only(tmp_path: Path) -> None:
    train = np.tile(np.array([0, 128], dtype=np.uint8), (16, 8))
    val = np.full((16, 16), 255, dtype=np.uint8)
    mask = np.ones((16, 16), dtype=bool)
    train_row = _write_pair(tmp_path, "train", train, mask)
    _write_pair(tmp_path, "val", val, mask)
    stats = compute_train_normalization(pd.DataFrame([train_row]), 500)
    assert stats["source_split"] == "train"
    assert stats["mean"] == pytest.approx(float(train.mean() / 255.0))
    assert stats["mean"] < 0.6


def test_invalid_ppi_fails_instead_of_guessing(tmp_path: Path) -> None:
    row = _write_pair(
        tmp_path,
        "bad",
        np.arange(64, dtype=np.uint8).reshape(8, 8),
        np.ones((8, 8), dtype=bool),
    )
    row["ppi"] = 0
    with pytest.raises(ValueError, match="invalid ppi"):
        read_normalized_image(row, 500)


def test_image_and_mask_use_separate_ppi_and_canonical_shape(tmp_path: Path) -> None:
    image = np.arange(30 * 40, dtype=np.uint8).reshape(30, 40)
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:16, 6:24] = True
    row = _write_pair(tmp_path, "grids", image, mask)
    row.update(
        ppi=1200,
        mask_ppi=1000,
        target_ppi=500,
        target_width=15,
        target_height=10,
    )
    normalized, resized_mask = deep_utils.read_pair(row, 500, "mask_path")
    assert normalized.shape == (10, 15)
    assert resized_mask.shape == (10, 15)
    expected_mask = cv2.resize(
        mask.astype(np.uint8), (15, 10), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    np.testing.assert_array_equal(resized_mask, expected_mask)


def test_nonprint_exclusion_is_test_only() -> None:
    data_config = {"exclude_examiner_values_from_test": ["NONPRINT"]}
    assert examiner_exclusions_for_split(data_config, "train") == ()
    assert examiner_exclusions_for_split(data_config, "val") == ()
    assert examiner_exclusions_for_split(data_config, "test") == ("NONPRINT",)


def test_config_rejects_off_protocol_learning_rate_and_threshold_grid(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "unet.yaml").read_text(encoding="utf-8")
    )
    path = tmp_path / "config.yaml"
    config["training"]["learning_rate"] = 1e-4
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered sweep"):
        load_config(path)

    config["training"]["learning_rate"] = 3e-4
    config["training"]["threshold_candidates"] = [0.4, 0.5, 0.6]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered validation grid"):
        load_config(path)


def _batch_config(batch_size: int = 4, accumulation: int = 2) -> dict[str, object]:
    return {
        "training": {
            "batch_size": batch_size,
            "gradient_accumulation": accumulation,
        },
        "runtime": {},
    }


def test_queue_batch_contract_preserves_effective_batch_and_allows_one_retry() -> None:
    base = _batch_config()
    base_active = apply_queue_batch_contract(base, {})
    assert base_active["batch_size"] == 4
    assert base_active["gradient_accumulation"] == 2
    assert base_active["effective_batch_size"] == 8

    checkpoint = {
        "queue_batch_contract_sha256": base_active["contract_sha256"],
        "queue_batch_active_state": dict(base_active),
    }
    retry = _batch_config()
    retry_active = apply_queue_batch_contract(
        retry,
        {
            "SEG_OOM_RETRY_COUNT": "1",
            "SEG_BATCH_SCALE": "0.5",
            "SEG_GRAD_ACCUM_SCALE": "2.0",
        },
    )
    assert retry_active["batch_size"] == 2
    assert retry_active["gradient_accumulation"] == 4
    assert retry_active["effective_batch_size"] == 8
    validate_resume_batch_transition(checkpoint, retry)


@pytest.mark.parametrize(
    "environment",
    [
        {"SEG_OOM_RETRY_COUNT": "0", "SEG_BATCH_SCALE": "0.5", "SEG_GRAD_ACCUM_SCALE": "2.0"},
        {"SEG_OOM_RETRY_COUNT": "1", "SEG_BATCH_SCALE": "0.25", "SEG_GRAD_ACCUM_SCALE": "4.0"},
        {"SEG_OOM_RETRY_COUNT": "2", "SEG_BATCH_SCALE": "0.5", "SEG_GRAD_ACCUM_SCALE": "2.0"},
    ],
)
def test_queue_batch_contract_rejects_arbitrary_scale_or_retry_count(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="exactly base or the single authorized"):
        apply_queue_batch_contract(_batch_config(), environment)


def test_queue_batch_contract_rejects_reverse_transition_and_tamper() -> None:
    retry = _batch_config()
    retry_active = apply_queue_batch_contract(
        retry,
        {
            "SEG_OOM_RETRY_COUNT": "1",
            "SEG_BATCH_SCALE": "0.5",
            "SEG_GRAD_ACCUM_SCALE": "2.0",
        },
    )
    checkpoint = {
        "queue_batch_contract_sha256": retry_active["contract_sha256"],
        "queue_batch_active_state": dict(retry_active),
    }
    base = _batch_config()
    apply_queue_batch_contract(base, {})
    with pytest.raises(ValueError, match="reverse or unauthorized"):
        validate_resume_batch_transition(checkpoint, base)

    checkpoint["queue_batch_active_state"]["batch_size"] = 3
    with pytest.raises(ValueError, match="differs from its contract"):
        validate_resume_batch_transition(checkpoint, retry)


def test_queue_batch_contract_requires_even_base_batch() -> None:
    with pytest.raises(ValueError, match="must be even"):
        apply_queue_batch_contract(_batch_config(batch_size=3), {})


def test_domain_alignment_rejects_subject_split_disagreement(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    target = tmp_path / "target.json"
    primary.write_text(
        json.dumps({"splits": {"S1": "train", "S2": "test"}}), encoding="utf-8"
    )
    target.write_text(
        json.dumps({"splits": {"S1": "test", "S3": "train"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="split mismatch"):
        assert_compatible_subject_splits(primary, target)


def test_domain_alignment_rejects_subjects_outside_authoritative_split(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.json"
    target = tmp_path / "target.json"
    primary.write_text(
        json.dumps({"splits": {"S1": "train", "S2": "test"}}), encoding="utf-8"
    )
    target.write_text(
        json.dumps({"splits": {"S1": "train", "S3": "train"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="absent from the authoritative"):
        assert_compatible_subject_splits(primary, target)


def test_pixel_crop_is_reproducible_by_seed_epoch_index(tmp_path: Path) -> None:
    image = np.arange(80 * 80, dtype=np.uint8).reshape(80, 80)
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:60, 15:55] = True
    row = _write_pair(tmp_path, "sample", image, mask)
    stats = compute_train_normalization(pd.DataFrame([row]), 500)
    dataset = PixelCropDataset(
        pd.DataFrame([row]),
        target_ppi=500,
        mask_column="mask_path",
        crop_size=64,
        normalization=stats,
        augmentation=AUGMENTATION,
        seed=7,
    )
    dataset.set_epoch(3)
    first = dataset[0]
    second = dataset[0]
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


def test_patch_index_is_exactly_balanced_and_deterministic(tmp_path: Path) -> None:
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32] = True
    row = _write_pair(tmp_path, "blocks", image, mask)
    stats = compute_train_normalization(pd.DataFrame([row]), 500)
    arguments = dict(
        rows=pd.DataFrame([row]),
        target_ppi=500,
        mask_column="mask_path",
        patch_size=32,
        block_size=16,
        blocks_per_class_per_image=8,
        normalization=stats,
        augmentation={**AUGMENTATION, "max_rotation_degrees": 0.0, "scale_min": 1.0, "scale_max": 1.0},
        seed=11,
    )
    first = PatchDataset(**arguments)
    second = PatchDataset(**arguments)
    labels = [sample[2] for sample in first.samples]
    assert labels.count(True) == labels.count(False)
    assert first.samples == second.samples
    patch, target = first[0]
    assert patch.shape == (1, 32, 32)
    assert target.item() in {0.0, 1.0}


def test_patch_label_is_recomputed_after_geometric_augmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32] = True
    row = _write_pair(tmp_path, "relabel", image, mask)
    stats = compute_train_normalization(pd.DataFrame([row]), 500)
    dataset = PatchDataset(
        pd.DataFrame([row]),
        target_ppi=500,
        mask_column="mask_path",
        patch_size=32,
        block_size=16,
        blocks_per_class_per_image=8,
        normalization=stats,
        augmentation=AUGMENTATION,
        seed=3,
    )
    negative_index = next(
        index for index, sample in enumerate(dataset.samples) if sample[2] is False
    )

    def force_foreground(
        patch: np.ndarray,
        patch_mask: np.ndarray,
        augmentation: object,
        rng: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        del augmentation, rng
        return patch, np.ones_like(patch_mask, dtype=bool)

    monkeypatch.setattr(deep_utils, "augment_pair", force_foreground)
    _, target = dataset[negative_index]
    assert target.item() == 1.0


def test_autoencoder_pretraining_dataset_never_reads_masks(tmp_path: Path) -> None:
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    row = _write_pair(tmp_path, "unlabelled", image, np.zeros((64, 64), dtype=bool))
    stats = compute_train_normalization(pd.DataFrame([row]), 500)
    row["mask_path"] = str(tmp_path / "must_not_be_opened.png")
    dataset = UnlabelledPatchDataset(
        pd.DataFrame([row]),
        target_ppi=500,
        patch_size=64,
        block_size=16,
        patches_per_image=4,
        normalization=stats,
        augmentation=AUGMENTATION,
        seed=4,
    )
    assert len(dataset) == 4
    assert dataset[0].shape == (1, 64, 64)


@pytest.mark.parametrize("patch_size", [32, 64])
def test_patch_paths_cover_partial_final_blocks(
    tmp_path: Path, patch_size: int
) -> None:
    """Labelled, unlabelled and inference paths handle ceil-grid boundaries."""
    height, width = 37, 53
    image = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    mask = np.zeros((height, width), dtype=bool)
    mask[:24, :32] = True
    row = _write_pair(tmp_path, f"partial_{patch_size}", image, mask)
    rows = pd.DataFrame([row])
    stats = compute_train_normalization(rows, 500)
    identity_augmentation = {
        **AUGMENTATION,
        "horizontal_flip_probability": 0.0,
        "max_rotation_degrees": 0.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "gaussian_noise_std": 0.0,
    }
    grid_h, grid_w = deep_utils.block_features.grid_shape(height, width)
    final_index = grid_h * grid_w - 1

    labelled = PatchDataset(
        rows,
        target_ppi=500,
        mask_column="mask_path",
        patch_size=patch_size,
        block_size=16,
        blocks_per_class_per_image=2,
        normalization=stats,
        augmentation=identity_augmentation,
        seed=8,
    )
    labelled.samples = [(0, final_index, False)]
    patch, target = labelled[0]
    assert patch.shape == (1, patch_size, patch_size)
    expected = deep_utils.block_features.block_labels(mask, grid_h, grid_w).ravel()[
        final_index
    ]
    assert bool(target.item()) is bool(expected)

    unlabelled = UnlabelledPatchDataset(
        rows,
        target_ppi=500,
        patch_size=patch_size,
        block_size=16,
        patches_per_image=1,
        normalization=stats,
        augmentation=identity_augmentation,
        seed=9,
    )
    unlabelled.samples = [(0, final_index)]
    assert unlabelled[0].shape == (1, patch_size, patch_size)

    patches, extracted_h, extracted_w = deep_utils._extract_all_patches(
        image.astype(np.float32), patch_size, 16
    )
    assert (extracted_h, extracted_w) == (grid_h, grid_w)
    assert patches.shape == (grid_h * grid_w, patch_size, patch_size)


def test_validation_threshold_selection_has_deterministic_tie_rule() -> None:
    probability = np.full((4, 4), 0.5, dtype=np.float32)
    target = np.ones((4, 4), dtype=bool)
    threshold, score, scores = select_validation_threshold(
        [probability], [target], [0.3, 0.5, 0.7]
    )
    assert threshold == 0.5
    assert score == 1.0
    assert set(scores) == {"0.3", "0.5", "0.7"}


def test_full_image_validation_loss_matches_configured_training_objective() -> None:
    logits = torch.tensor(
        [
            [[[2.0, -1.0], [0.5, -3.0]]],
            [[[-0.5, 1.5], [3.0, -2.0]]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    probabilities = torch.sigmoid(logits).numpy()[:, 0]
    targets = target.numpy()[:, 0].astype(bool)
    observed = full_image_validation_loss(
        list(probabilities),
        list(targets),
        bce_weight=1.0,
        dice_weight=1.0,
    )
    expected = deep_utils.segmentation_loss(
        logits,
        target,
        bce_weight=1.0,
        dice_weight=1.0,
    )
    assert observed == pytest.approx(float(expected), rel=1e-6, abs=1e-6)


def test_full_image_validation_loss_rejects_non_finite_probability() -> None:
    with pytest.raises(FloatingPointError, match="non-finite probability"):
        full_image_validation_loss(
            [np.array([[np.nan]], dtype=np.float32)],
            [np.array([[True]])],
            bce_weight=1.0,
            dice_weight=1.0,
        )


def test_full_and_tiled_pointwise_inference_match() -> None:
    model = nn.Conv2d(1, 1, kernel_size=1)
    with torch.no_grad():
        model.weight.fill_(0.75)
        model.bias.fill_(-0.1)
    image = np.random.default_rng(5).normal(size=(70, 90)).astype(np.float32)
    full = predict_pixel_probability(
        model, image, torch.device("cpu"), pad_multiple=32, tile_size=0, tile_overlap=0
    )
    tiled = predict_pixel_probability(
        model, image, torch.device("cpu"), pad_multiple=32, tile_size=32, tile_overlap=8
    )
    np.testing.assert_allclose(full, tiled, rtol=1e-6, atol=1e-6)
    assert full.shape == image.shape


def test_mc_dropout_is_stochastic_while_batchnorm_stays_frozen() -> None:
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=1, bias=False),
        nn.BatchNorm2d(4),
        nn.Dropout2d(0.5),
        nn.Conv2d(4, 1, kernel_size=1, bias=False),
    )
    image = np.ones((32, 32), dtype=np.float32)
    deterministic_a = predict_pixel_probability(
        model,
        image,
        torch.device("cpu"),
        pad_multiple=32,
        tile_size=0,
        tile_overlap=0,
    )
    deterministic_b = predict_pixel_probability(
        model,
        image,
        torch.device("cpu"),
        pad_multiple=32,
        tile_size=0,
        tile_overlap=0,
    )
    np.testing.assert_array_equal(deterministic_a, deterministic_b)

    torch.manual_seed(11)
    stochastic = [
        predict_pixel_probability(
            model,
            image,
            torch.device("cpu"),
            pad_multiple=32,
            tile_size=0,
            tile_overlap=0,
            enable_dropout=True,
        )
        for _ in range(6)
    ]
    assert any(not np.array_equal(stochastic[0], item) for item in stochastic[1:])
    assert model[1].training is False
    assert model[2].training is True


def test_resume_identity_allows_only_queue_mechanics() -> None:
    base = {
        "method": "unet",
        "task": "pixel",
        "seed": 0,
        "data": {"normalization": "train_only"},
        "model": {"base_channels": 4},
        "augmentation": {"brightness": 0.2},
        "training": {
            "learning_rate": 0.001,
            "batch_size": 4,
            "gradient_accumulation": 1,
            "num_workers": 2,
        },
        "runtime": {"device": "auto", "memory_fraction": 0.3},
    }
    retry = json.loads(json.dumps(base))
    retry["training"].update(batch_size=2, gradient_accumulation=2, num_workers=1)
    retry["runtime"].update(device="cuda", memory_fraction=0.2, queue_overrides={})
    assert resume_config_digest(base) == resume_config_digest(retry)
    changed = json.loads(json.dumps(retry))
    changed["training"]["learning_rate"] = 0.01
    assert resume_config_digest(base) != resume_config_digest(changed)


def test_memory_fraction_resolves_indexless_cuda_to_current_logical_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, int]] = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: calls.append((fraction, device)),
    )

    deep_utils.configure_memory_fraction(torch.device("cuda"), 0.3)

    assert calls == [(0.3, 2)]


def test_memory_fraction_preserves_explicit_cuda_index_and_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, int]] = []
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("explicit CUDA index must not query current_device"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: calls.append((fraction, device)),
    )

    deep_utils.configure_memory_fraction(torch.device("cuda:1"), 0.2)
    assert calls == [(0.2, 1)]

    with pytest.raises(ValueError, match="per-process ceiling"):
        deep_utils.configure_memory_fraction(torch.device("cuda"), 0.31)
