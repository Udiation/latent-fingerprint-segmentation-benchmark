"""CPU-only protocol and integration tests for fine-tuned SUFS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import keras
import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import infer_sufs  # noqa: E402
import select_sufs_lr  # noqa: E402
import train_sufs  # noqa: E402
from segbench.deep_utils import load_config  # noqa: E402
from segbench.sufs_finetune import (  # noqa: E402
    PROTOCOL_LEARNING_RATES,
    build_released_sufs,
    predict_sufs_probability,
    protocol_config_digest,
    released_weights_path,
    sha256_file,
    state_dict_cpu,
    sufs_probability_loss,
)
from segbench.pyfing_compat import TorchSufs  # noqa: E402
from pyfing.segmentation import Sufs  # noqa: E402


CONFIG = ROOT / "configs" / "sufs_ft.yaml"


def test_config_pins_released_weights_and_protocol() -> None:
    """The committed config must identify the exact installed SUFS artifact."""
    config = load_config(CONFIG)
    weights = released_weights_path(config)
    assert sha256_file(weights) == config["pretrained"]["weights_sha256"]
    assert config["training"]["sweep_epochs"] == 30
    assert config["runtime"]["memory_fraction"] <= 0.3
    assert config["data"]["normalization"] == "train_only"


def test_zero_shot_compatibility_wrapper_is_bit_exact_to_released_sufs() -> None:
    """The Torch conversion fix must not alter the separate zero-shot row."""
    torch.set_num_threads(1)
    image = np.random.default_rng(19).integers(0, 256, (35, 47), dtype=np.uint8)
    keras.backend.clear_session()
    released = Sufs()
    with torch.inference_mode():
        expected = released.run(image)
    keras.backend.clear_session()
    compatible = TorchSufs()
    np.testing.assert_array_equal(compatible.run(image), expected)


def test_probability_loss_uses_sigmoid_values_not_logits() -> None:
    """BCE+Dice is computed directly on SUFS's final sigmoid probabilities."""
    probability = torch.tensor(
        [[[[0.8], [0.2]], [[0.6], [0.1]]]], requires_grad=True
    )
    target = torch.tensor([[[[1.0], [0.0]], [[1.0], [0.0]]]])
    observed = sufs_probability_loss(
        probability, target, bce_weight=1.0, dice_weight=1.0
    )
    bce = torch.nn.functional.binary_cross_entropy(probability, target)
    intersection = (probability * target).sum()
    expected = bce + 1.0 - (2.0 * intersection + 1e-6) / (
        probability.sum() + target.sum() + 1e-6
    )
    torch.testing.assert_close(observed, expected)
    observed.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_released_model_supports_training_and_restores_full_image_shape() -> None:
    """One synthetic CPU update exercises Keras-Torch autograd and padding."""
    torch.set_num_threads(1)
    config = load_config(CONFIG)
    segmenter, model, provenance = build_released_sufs(config, torch.device("cpu"))
    assert provenance["parameter_count"] == 7_083_793
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    image = torch.zeros((1, 64, 64, 1), dtype=torch.float32)
    target = torch.zeros_like(image)
    target[:, 16:48, 16:48] = 1.0
    before = next(model.parameters()).detach().clone()
    probability = model(image, training=True)
    loss = sufs_probability_loss(
        probability, target, bce_weight=1.0, dice_weight=1.0
    )
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()).detach())
    trained_state = state_dict_cpu(model)
    restored_segmenter, restored, _ = build_released_sufs(config, torch.device("cpu"))
    restored.load_state_dict(trained_state, strict=True)
    for key, value in state_dict_cpu(restored).items():
        torch.testing.assert_close(value, trained_state[key])
    result = predict_sufs_probability(
        restored_segmenter,
        np.zeros((37, 53), dtype=np.float32),
        torch.device("cpu"),
    )
    assert result.shape == (37, 53)
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0 and float(result.max()) <= 1.0


def test_sweep_mode_enforces_equal_budget_and_seed_zero(tmp_path: Path) -> None:
    args = argparse.Namespace(
        config=CONFIG,
        manifest=tmp_path / "manifest.csv",
        splits=tmp_path / "splits.json",
        out=tmp_path / "out",
        mode="sweep",
        learning_rate=3e-4,
        selection=None,
        seed=0,
        device="cpu",
        memory_fraction=0.3,
        resume="none",
        limit_train=0,
        limit_val=0,
    )
    config, selection = train_sufs._resolved_config(args)
    assert selection is None
    assert config["training"]["epochs"] == 30
    assert config["training"]["learning_rate"] == 3e-4
    args.seed = 1
    with pytest.raises(ValueError, match="seed 0"):
        train_sufs._resolved_config(args)


def test_protocol_digest_ignores_only_sweep_and_queue_mechanics() -> None:
    config = load_config(CONFIG)
    changed = json.loads(json.dumps(config))
    changed["seed"] = 2
    changed["training"]["learning_rate"] = 3e-4
    changed["training"]["epochs"] = 30
    changed["training"]["batch_size"] = 2
    changed["training"]["gradient_accumulation"] = 2
    changed["training"]["num_workers"] = 1
    changed["runtime"]["queue_overrides"] = {"batch_scale": 0.5}
    assert protocol_config_digest(changed) == protocol_config_digest(config)
    changed["training"]["weight_decay"] = 0.0
    assert protocol_config_digest(changed) != protocol_config_digest(config)


def _sweep_card(tmp_path: Path, learning_rate: float, score: float) -> Path:
    path = tmp_path / f"run-{learning_rate}.json"
    payload = {
        "method": "sufs_ft",
        "mode": "sweep",
        "seed": 0,
        "epochs_requested": 30,
        "epochs_completed": 30,
        "debug_limited": False,
        "learning_rate": learning_rate,
        "best_validation_dice": score,
        "best_epoch": 12,
        "protocol_config_sha256": "a" * 64,
        "source_config_sha256": "e" * 64,
        "manifest_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "released": {"weights_sha256": "d" * 64},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_lr_selection_requires_both_rates_and_uses_prospective_tie_rule(
    tmp_path: Path,
) -> None:
    low = _sweep_card(tmp_path, PROTOCOL_LEARNING_RATES[0], 0.8)
    high = _sweep_card(tmp_path, PROTOCOL_LEARNING_RATES[1], 0.8)
    selected = select_sufs_lr.select([high, low])
    assert selected["selection_split"] == "val"
    assert selected["selected_learning_rate"] == PROTOCOL_LEARNING_RATES[0]
    duplicate = _sweep_card(tmp_path, PROTOCOL_LEARNING_RATES[0], 0.9)
    with pytest.raises(ValueError, match="both preregistered"):
        select_sufs_lr.select([low, duplicate])


def test_inference_never_reads_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic mask generation receives images only and writes 0/255 PNGs."""
    config = load_config(CONFIG)
    checkpoint_config = json.loads(json.dumps(config))
    checkpoint_config["seed"] = 0
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("fixture-manifest", encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text("fixture-split", encoding="utf-8")
    image = np.linspace(0.0, 1.0, 35 * 41, dtype=np.float32).reshape(35, 41)
    source_image = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(source_image), np.rint(image * 255).astype(np.uint8))
    rows = pd.DataFrame(
        [
            {
                "image_id": "synthetic-one",
                    "image_path": str(source_image),
                    "source_sha256": sha256_file(source_image),
                "mask_path": "must-not-be-opened.png",
            }
        ]
    )

    class DummyModel(torch.nn.Module):
        def load_state_dict(self, state: object, strict: bool = True) -> object:
            assert state == {}
            assert strict
            return SimpleNamespace()

        def eval(self) -> "DummyModel":
            return self

    monkeypatch.setattr(infer_sufs.torch, "load", lambda *_args, **_kwargs: {
        "method": "sufs_ft",
        "task": "pixel",
        "mode": "full",
        "config": checkpoint_config,
        "released": {"weights_sha256": "e" * 64},
        "model_state": {},
        "normalization": {"mean": 0.5, "std": 0.25, "source_split": "train"},
        "selected_threshold": 0.5,
        "best_epoch": 7,
        "validation_dice": 0.75,
    })
    monkeypatch.setattr(
        infer_sufs,
        "build_released_sufs",
        lambda *_args, **_kwargs: (
            SimpleNamespace(model=DummyModel()),
            DummyModel(),
            {"weights_sha256": "e" * 64, "parameter_count": 1},
        ),
    )
    monkeypatch.setattr(infer_sufs, "load_split_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(infer_sufs, "read_normalized_image", lambda *_args, **_kwargs: image)
    monkeypatch.setattr(
        infer_sufs,
        "predict_sufs_probability",
        lambda *_args, **_kwargs: np.where(image > 0.5, 0.9, 0.1).astype(np.float32),
    )
    monkeypatch.setattr(infer_sufs, "run_provenance", lambda *_args: {"fixture": True})
    threshold_receipt = tmp_path / "threshold.json"
    candidates = sorted(float(value) for value in config["training"]["threshold_candidates"])
    threshold_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "expanded_validation_threshold_selector_receipt",
                "method": "sufs_ft",
                "task": "pixel",
                "selection_split": "val",
                "selection_source": "validation_only",
                "validation_only": True,
                "test_or_heldout_opened": False,
                "probabilities_are_pre_threshold": True,
                "storage_dtype": "float16",
                "derived_from_binary_mask": False,
                "checkpoint_sha256": sha256_file(checkpoint),
                "source_config_sha256": sha256_file(CONFIG),
                "manifest_sha256": sha256_file(manifest),
                "split_sha256": sha256_file(splits),
                "threshold_amendment_sha256": None,
                "amendment_sha256": None,
                "seed": 0,
                "validation_probability_card_sha256": "d" * 64,
                "threshold_candidates": candidates,
                "allowed_grid": candidates,
                "threshold_scores": {
                    f"{value:.8g}": (1.0 if value == 0.5 else 0.0)
                    for value in candidates
                },
                "selected_threshold": 0.5,
                "debug_limited": False,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=CONFIG,
        checkpoint=checkpoint,
        manifest=manifest,
        splits=splits,
        subset="finger_latent",
        split="test",
        out=tmp_path / "masks",
        run_card=tmp_path / "inference.json",
        threshold_receipt=threshold_receipt,
        device="cpu",
        memory_fraction=0.3,
        warmup=0,
        limit=1,
        skip_valid=False,
        allow_sweep_checkpoint=False,
        allow_unamended_debug=True,
    )
    record = infer_sufs.infer(args)
    prediction = cv2.imread(
        str(args.out / "synthetic-one.png"), cv2.IMREAD_GRAYSCALE
    )
    assert set(np.unique(prediction).tolist()) == {0, 255}
    assert record["prediction_count"] == 1
    assert record["identity"]["threshold"] == 0.5
    assert protocol_config_digest(checkpoint_config) == protocol_config_digest(config)
