"""CPU-only architecture and loss tests for neural benchmark methods."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from segbench.deep_models import (
    RARUNet,
    RecurrentUNet,
    build_model,
    gradient_reverse,
    parameter_counts,
    verify_pinned_pretrained_snapshot,
)
from segbench.deep_utils import segmentation_loss


ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = (
    "unet",
    "unetpp",
    "drunet",
    "ra_runet",
    "ra_runet_aligned",
    "lenet",
    "alexnet",
    "cnn_ae",
    "segformer_sub",
)


def test_configs_share_fairness_controls() -> None:
    configs = [
        yaml.safe_load((ROOT / "configs" / f"{key}.yaml").read_text(encoding="utf-8"))
        for key in CONFIG_KEYS
    ]
    assert {config["training"]["learning_rate"] for config in configs} == {0.001}
    assert {
        tuple(config["training"]["threshold_candidates"]) for config in configs
    } == {(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)}
    assert {config["data"]["normalization"] for config in configs} == {"train_only"}
    assert {config["training"]["patience"] for config in configs} == {15}
    assert {config["runtime"]["memory_fraction"] for config in configs} == {0.3}


@pytest.mark.parametrize("method", ["unet", "drunet", "ra_runet"])
def test_pixel_models_preserve_shape(method: str) -> None:
    config = {
        "in_channels": 1,
        "base_channels": 4,
        "levels": 4,
        "recurrent_steps": 2,
        "dropout": 0.2 if method == "drunet" else 0.0,
    }
    model = build_model(method, config)
    output = model(torch.randn(2, 1, 64, 96))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 1, 64, 96)
    counts = parameter_counts(model)
    assert counts["total"] == counts["trainable"] > 0


def test_unetpp_deep_supervision_and_final_inference_head() -> None:
    model = build_model(
        "unetpp",
        {"in_channels": 1, "base_channels": 4, "levels": 4, "deep_supervision": True},
    )
    image = torch.randn(2, 1, 64, 96)
    model.train()
    outputs = model(image)
    assert isinstance(outputs, list) and len(outputs) == 4
    assert all(output.shape == image.shape for output in outputs)
    target = torch.rand_like(image)
    combined = segmentation_loss(outputs, target, bce_weight=1.0, dice_weight=1.0)
    explicit = torch.stack(
        [segmentation_loss(output, target, bce_weight=1.0, dice_weight=1.0) for output in outputs]
    ).mean()
    torch.testing.assert_close(combined, explicit)
    model.eval()
    assert model(image).shape == image.shape


def test_drunet_eval_is_deterministic_with_dropout_disabled_by_eval() -> None:
    model = build_model(
        "drunet",
        {
            "in_channels": 1,
            "base_channels": 4,
            "levels": 4,
            "recurrent_steps": 2,
            "dropout": 0.2,
        },
    )
    image = torch.randn(1, 1, 64, 64)
    model.eval()
    with torch.no_grad():
        first = model(image)
        second = model(image)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_gradient_reversal_changes_only_gradient_sign_and_scale() -> None:
    value = torch.tensor([2.0], requires_grad=True)
    output = gradient_reverse(value, 0.25)
    torch.testing.assert_close(output, value)
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([-0.25]))


def test_ra_runet_omits_domain_head_when_alignment_is_disabled() -> None:
    backbone = build_model(
        "ra_runet",
        {
            "in_channels": 1,
            "base_channels": 4,
            "levels": 4,
            "recurrent_steps": 2,
            "dropout": 0.0,
            "alignment": {"enabled": False},
        },
    )
    assert isinstance(backbone, RecurrentUNet)
    image = torch.randn(2, 1, 64, 64)
    assert backbone(image).shape == image.shape

    model = build_model(
        "ra_runet",
        {
            "in_channels": 1,
            "base_channels": 4,
            "levels": 4,
            "recurrent_steps": 2,
            "dropout": 0.0,
            "alignment": {"enabled": True},
        },
    )
    assert isinstance(model, RARUNet)
    segmentation = model(image)
    assert segmentation.shape == image.shape
    segmentation, domain = model(image, return_domain=True, gradient_reversal_weight=0.1)
    assert segmentation.shape == image.shape
    assert domain.shape == (2,)


@pytest.mark.parametrize(
    ("method", "shape"),
    [("lenet", (2, 1, 32, 32)), ("alexnet", (2, 1, 64, 64)), ("cnn_ae", (2, 1, 64, 64))],
)
def test_patch_models_emit_one_logit_per_patch(method: str, shape: tuple[int, ...]) -> None:
    config = {"in_channels": 1, "dropout": 0.2, "latent_channels": 16}
    logits = build_model(method, config)(torch.randn(*shape))
    assert logits.shape == (2,)


def test_segmentation_loss_is_finite_for_empty_and_full_targets() -> None:
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    for target in (torch.zeros_like(logits), torch.ones_like(logits)):
        loss = segmentation_loss(logits, target, bce_weight=1.0, dice_weight=1.0)
        assert torch.isfinite(loss)


def _git_blob(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()  # nosec


def test_pinned_snapshot_is_content_verified(tmp_path: Path) -> None:
    config = b"{}\n"
    weights = b"synthetic weights"
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "pytorch_model.bin").write_bytes(weights)
    declared = {
        "config.json": {"git_blob_sha1": _git_blob(config)},
        "pytorch_model.bin": {
            "sha256": hashlib.sha256(weights).hexdigest(),
            "bytes": len(weights),
        },
    }
    observed = verify_pinned_pretrained_snapshot(tmp_path, declared)
    assert observed["pytorch_model.bin"]["bytes"] == len(weights)
    (tmp_path / "pytorch_model.bin").write_bytes(weights + b"tamper")
    with pytest.raises(ValueError, match="byte-size mismatch|SHA-256 mismatch"):
        verify_pinned_pretrained_snapshot(tmp_path, declared)


def test_segformer_rejects_mutable_or_network_initialization() -> None:
    with pytest.raises(ValueError, match="local-only"):
        build_model(
            "segformer_sub",
            {"in_channels": 1, "pretrained": True, "local_files_only": False},
        )
