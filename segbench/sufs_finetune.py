"""Faithful PyFing SUFS fine-tuning helpers for the Keras Torch backend.

The released SUFS network ends in a sigmoid.  This module therefore computes
binary cross entropy and soft Dice directly on probabilities (never as logits),
while retaining the model architecture, padding convention and released
weights shipped by PyFing 0.6.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

if os.environ.get("KERAS_BACKEND", "torch").lower() != "torch":
    raise RuntimeError("SUFS fine-tuning requires KERAS_BACKEND=torch")
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import keras
import numpy as np
import torch
from torch import Tensor, nn

from pyfing.segmentation import Sufs, SufsParameters

from segbench.pyfing_compat import TorchSufs


PROTOCOL_LEARNING_RATES = (3e-4, 1e-3)
EXPECTED_PACKAGE = "pyfing"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def released_weights_path(config: Mapping[str, Any]) -> Path:
    """Locate and validate the exact SUFS weights shipped by installed PyFing."""
    pretrained = config.get("pretrained")
    if not isinstance(pretrained, Mapping):
        raise ValueError("SUFS config is missing pretrained provenance")
    package = str(pretrained.get("package", ""))
    if package != EXPECTED_PACKAGE:
        raise ValueError(f"SUFS must use the released {EXPECTED_PACKAGE} package")
    observed_version = importlib.metadata.version(package)
    if observed_version != str(pretrained.get("version", "")):
        raise ValueError(
            f"installed {package} version {observed_version} differs from the "
            f"configured version {pretrained.get('version')!r}"
        )
    package_root = Path(inspect.getfile(Sufs)).resolve().parent
    weights = package_root / "models" / str(pretrained.get("weights_file", ""))
    if not weights.is_file():
        raise FileNotFoundError(f"released SUFS weights are missing: {weights}")
    observed_digest = sha256_file(weights)
    expected_digest = str(pretrained.get("weights_sha256", ""))
    if observed_digest != expected_digest:
        raise ValueError(
            "released SUFS weights fail the configured SHA-256 check: "
            f"observed {observed_digest}, expected {expected_digest}"
        )
    return weights


def validate_model_config(config: Mapping[str, Any]) -> Dict[str, int]:
    """Validate the frozen SUFS input geometry and return normalized values."""
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("SUFS config is missing model settings")
    if model_config.get("implementation") != "pyfing_sufs_0.6":
        raise ValueError("unknown SUFS implementation")
    values = {
        key: int(model_config[key])
        for key in (
            "dnn_input_dpi",
            "dnn_input_size_multiple",
            "image_dpi",
            "border",
        )
    }
    if values != {
        "dnn_input_dpi": 500,
        "dnn_input_size_multiple": 64,
        "image_dpi": 500,
        "border": 33,
    }:
        raise ValueError("SUFS input geometry must match the released PyFing defaults")
    if float(config["data"]["target_ppi"]) != 500.0:
        raise ValueError("SUFS benchmark inputs must be on the 500-PPI grid")
    if int(config["inference"].get("batch_size", 0)) != 1:
        raise ValueError("SUFS timing protocol requires inference batch size 1")
    return values


def build_released_sufs(
    config: Mapping[str, Any], device: torch.device
) -> Tuple[TorchSufs, nn.Module, Dict[str, Any]]:
    """Build the released SUFS model and move it to the selected Torch device."""
    weights = released_weights_path(config)
    geometry = validate_model_config(config)
    # Keras layer names are global counters and are embedded in the Torch state
    # dict.  Resetting the session makes independently built checkpoints stable.
    keras.backend.clear_session()
    parameters = SufsParameters(**geometry, threshold=0.5)
    segmenter = TorchSufs(parameters=parameters, model_weights=str(weights))
    model = segmenter.model
    if not isinstance(model, nn.Module):
        raise TypeError("Keras Torch SUFS did not expose a torch.nn.Module")
    model.to(device)
    provenance = {
        "package": EXPECTED_PACKAGE,
        "version": importlib.metadata.version(EXPECTED_PACKAGE),
        "weights_file": weights.name,
        "weights_sha256": sha256_file(weights),
        "keras_version": keras.__version__,
        "keras_backend": keras.backend.backend(),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    if provenance["keras_backend"] != "torch":
        raise RuntimeError("SUFS model was not built with the Keras Torch backend")
    return segmenter, model, provenance


def validate_shared_training_protocol(config: Mapping[str, Any]) -> None:
    """Fail if SUFS is given an easier or different pixel-model protocol."""
    data = config["data"]
    training = config["training"]
    augmentation = config["augmentation"]
    expected_augmentation = {
        "horizontal_flip_probability": 0.5,
        "max_rotation_degrees": 15.0,
        "scale_min": 0.9,
        "scale_max": 1.1,
        "brightness": 0.2,
        "contrast": 0.2,
        "gaussian_noise_std": 0.02,
    }
    observed_augmentation = {
        key: float(augmentation.get(key, float("nan")))
        for key in expected_augmentation
    }
    if observed_augmentation != expected_augmentation:
        raise ValueError("SUFS augmentations differ from the shared pixel-model protocol")
    expected_training = {
        "weight_decay": 1e-4,
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "min_learning_rate": 0.0,
    }
    for key, expected in expected_training.items():
        if not math.isclose(
            float(training.get(key, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"SUFS {key} differs from the shared protocol")
    if (
        int(data.get("crop_size", 0)) != 512
        or int(training.get("epochs", 0)) != 100
        or int(training.get("sweep_epochs", 0)) != 30
        or int(training.get("patience", 0)) != 15
        or training.get("scheduler") != "cosine"
    ):
        raise ValueError("SUFS crop, epoch, patience or scheduler settings drifted")
    if float(config["runtime"].get("memory_fraction", 1.0)) > 0.3:
        raise ValueError("SUFS CUDA memory fraction exceeds the project ceiling")


def sufs_probability_loss(
    probability: Tensor,
    target: Tensor,
    *,
    bce_weight: float,
    dice_weight: float,
    epsilon: float = 1e-6,
) -> Tensor:
    """Return BCE plus soft Dice for a sigmoid probability output."""
    if probability.shape != target.shape:
        raise ValueError(
            f"SUFS probability shape {tuple(probability.shape)} differs from "
            f"target {tuple(target.shape)}"
        )
    if probability.ndim < 2:
        raise ValueError("SUFS probability tensor must include batch and pixel dimensions")
    if not math.isfinite(bce_weight) or not math.isfinite(dice_weight):
        raise ValueError("loss weights must be finite")
    if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
        raise ValueError("loss weights must be non-negative with a positive sum")
    clipped = probability.clamp(min=epsilon, max=1.0 - epsilon)
    bce = torch.nn.functional.binary_cross_entropy(clipped, target)
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice = (
        1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)
    ).mean()
    return float(bce_weight) * bce + float(dice_weight) * dice


def predict_sufs_probability(
    segmenter: TorchSufs,
    normalized_image: np.ndarray,
    device: torch.device,
    *,
    autocast_bf16: bool = False,
) -> np.ndarray:
    """Run full-image SUFS inference with the released padding convention."""
    if normalized_image.ndim != 2:
        raise ValueError("SUFS expects one two-dimensional grayscale image")
    if not np.isfinite(normalized_image).all():
        raise ValueError("SUFS input contains NaN or Inf")
    original_h, original_w = normalized_image.shape
    size_info = segmenter._compute_size(original_w, original_h)
    adjusted = segmenter._adjust_input(normalized_image, *size_info)
    tensor = torch.from_numpy(
        np.ascontiguousarray(adjusted[None, ..., None], dtype=np.float32)
    ).to(device)
    segmenter.model.eval()
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_bf16,
        ):
            output = segmenter.model(tensor, training=False)
    if not isinstance(output, Tensor):
        raise TypeError("Keras Torch SUFS returned a non-Tensor output")
    probability = output[0, ..., 0].detach().float().cpu().numpy().astype(np.float32)
    width, height, left, top, right, bottom = size_info
    del right, bottom
    probability = probability[top : top + height, left : left + width]
    if probability.shape != (original_h, original_w):
        probability = cv2.resize(
            probability,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )
    if probability.shape != normalized_image.shape:
        raise RuntimeError("SUFS failed to restore the original image shape")
    return np.clip(probability, 0.0, 1.0)


def protocol_config_digest(config: Mapping[str, Any]) -> str:
    """Hash the shared SUFS protocol while ignoring sweep-varying fields."""
    payload = json.loads(json.dumps(config))
    payload.pop("seed", None)
    training = payload.get("training", {})
    for key in (
        "learning_rate",
        "epochs",
        "batch_size",
        "gradient_accumulation",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
    ):
        training.pop(key, None)
    runtime = payload.get("runtime", {})
    for key in (
        "device",
        "memory_fraction",
        "queue_overrides",
        "preprocess_cache_receipt_sha256",
        "threshold_amendment_sha256",
    ):
        runtime.pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_learning_rate(value: float) -> float:
    """Require one of the two pre-registered neural sweep rates."""
    value = float(value)
    for candidate in PROTOCOL_LEARNING_RATES:
        if math.isclose(value, candidate, rel_tol=0.0, abs_tol=1e-12):
            return candidate
    raise ValueError(
        f"SUFS learning rate must be one of {PROTOCOL_LEARNING_RATES}, got {value}"
    )


def state_dict_cpu(model: nn.Module) -> Dict[str, Tensor]:
    """Copy a Keras Torch model state to portable CPU tensors."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def prediction_set_digest(paths: Sequence[Path]) -> str:
    """Hash names and contents of a complete prediction set."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
