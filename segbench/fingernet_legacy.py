"""Faithful inference-only compatibility path for released FingerNet weights.

The authors released a Keras 2.0.2/TensorFlow HDF5 checkpoint.  This module
reconstructs only the dependency path from the input to the released
``seg_out`` tensor and loads those tensors directly by their legacy layer
names.  It deliberately preserves the release's normalization, 8-pixel grid,
0.5 rounding threshold, and 5x5 morphological opening.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("KERAS_BACKEND", "torch")
if os.environ["KERAS_BACKEND"].lower() != "torch":
    raise RuntimeError("FingerNet compatibility inference requires KERAS_BACKEND=torch")

import cv2
import h5py
import keras
import numpy as np
import torch
from keras import layers, ops


RELEASE_WEIGHTS_SHA256 = (
    "1c2c9ec33472d59e31bdb480bdee57aa9e109dc5b91bf01c0466154ee45c2ccd"
)
RELEASE_COMMIT = "aee18f3a77e2f25e7e0c072e3beb89a2878f3290"
RELEASE_KERAS_VERSION = "2.0.2"
RELEASE_BACKEND = "tensorflow"
FULL_RELEASE_STORED_SCALAR_COUNT = 4_718_014
FULL_RELEASE_PARAMETER_COUNT = 4_709_694
SEGMENTATION_GRAPH_STORED_SCALAR_COUNT = 3_615_171
SEGMENTATION_GRAPH_PARAMETER_COUNT = 3_610_563

BACKBONE_BLOCKS: Sequence[Tuple[str, int]] = (
    ("1_1", 64),
    ("1_2", 64),
    ("2_1", 128),
    ("2_2", 128),
    ("3_1", 256),
    ("3_2", 256),
    ("3_3", 256),
)
SEGMENTATION_WEIGHT_LAYERS: Sequence[str] = (
    "conv1_1", "bn-1_1", "prelu-1_1",
    "conv1_2", "bn-1_2", "prelu-1_2",
    "conv2_1", "bn-2_1", "prelu-2_1",
    "conv2_2", "bn-2_2", "prelu-2_2",
    "conv3_1", "bn-3_1", "prelu-3_1",
    "conv3_2", "bn-3_2", "prelu-3_2",
    "conv3_3", "bn-3_3", "prelu-3_3",
    "conv4_1", "bn-4_1", "prelu-4_1",
    "atrousconv4_2", "bn-4_2", "prelu-4_2",
    "atrousconv4_3", "bn-4_3", "prelu-4_3",
    "convseg_1_1", "bn-seg_1_1", "prelu-seg_1_1", "seg_1_2",
    "convseg_2_1", "bn-seg_2_1", "prelu-seg_2_1", "seg_2_2",
    "convseg_3_1", "bn-seg_3_1", "prelu-seg_3_1", "seg_3_2",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: Any) -> str:
    """Decode one legacy HDF5 attribute value."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _datasets(group: h5py.Group) -> Iterable[h5py.Dataset]:
    """Yield every dataset below an HDF5 group."""
    for value in group.values():
        if isinstance(value, h5py.Dataset):
            yield value
        elif isinstance(value, h5py.Group):
            yield from _datasets(value)


def audit_release_weights(path: Path) -> Dict[str, Any]:
    """Fail closed unless ``path`` is the authors' complete released checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != RELEASE_WEIGHTS_SHA256:
        raise ValueError(
            "FingerNet weights SHA-256 mismatch: the benchmark accepts only the "
            "authors' released checkpoint"
        )
    with h5py.File(path, "r") as handle:
        keras_version = _decode(handle.attrs.get("keras_version", ""))
        backend = _decode(handle.attrs.get("backend", ""))
        layer_names = [_decode(value) for value in handle.attrs.get("layer_names", [])]
        missing = sorted(set(SEGMENTATION_WEIGHT_LAYERS) - set(layer_names))
        if missing:
            raise ValueError(f"released checkpoint is missing segmentation layers: {missing}")
        stored_scalar_count = 0
        parameter_count = 0
        nonfinite: List[str] = []
        for dataset in _datasets(handle):
            scalar_count = int(np.prod(dataset.shape, dtype=np.int64))
            stored_scalar_count += scalar_count
            if not dataset.name.endswith(("/moving_mean:0", "/moving_variance:0")):
                parameter_count += scalar_count
            if not np.isfinite(dataset[()]).all():
                nonfinite.append(dataset.name)
        if stored_scalar_count != FULL_RELEASE_STORED_SCALAR_COUNT:
            raise ValueError(
                f"released checkpoint has {stored_scalar_count} scalars, expected "
                f"{FULL_RELEASE_STORED_SCALAR_COUNT}"
            )
        if parameter_count != FULL_RELEASE_PARAMETER_COUNT:
            raise ValueError(
                f"released checkpoint has {parameter_count} learned parameters, "
                f"expected {FULL_RELEASE_PARAMETER_COUNT}"
            )
        if keras_version != RELEASE_KERAS_VERSION or backend != RELEASE_BACKEND:
            raise ValueError(
                "released checkpoint metadata disagrees with Keras 2.0.2/TensorFlow"
            )
        if nonfinite:
            raise ValueError(f"released checkpoint has non-finite tensors: {nonfinite}")
    return {
        "sha256": digest,
        "keras_version": keras_version,
        "backend": backend,
        "layer_count": len(layer_names),
        "stored_scalar_count": stored_scalar_count,
        "parameter_count": parameter_count,
    }


class ReleaseImageNormalization(layers.Layer):
    """Exact population-variance normalization from the released graph."""

    def call(self, inputs: Any) -> Any:
        mean = ops.mean(inputs, axis=(1, 2, 3), keepdims=True)
        variance = ops.var(inputs, axis=(1, 2, 3), keepdims=True)
        magnitude = ops.sqrt(ops.square(inputs - mean) / variance)
        return ops.where(inputs > mean, magnitude, -magnitude)


def _conv_bn_prelu(
    bottom: Any,
    filters: int,
    kernel: int,
    release_name: str,
    dilation: int = 1,
) -> Any:
    """Rebuild the release's Conv2D -> BatchNorm -> PReLU block."""
    prefix = "conv" if dilation == 1 else "atrousconv"
    top = layers.Conv2D(
        filters,
        kernel,
        padding="same",
        dilation_rate=dilation,
        name=prefix + release_name,
    )(bottom)
    top = layers.BatchNormalization(name="bn-" + release_name)(top)
    return layers.PReLU(shared_axes=(1, 2), name="prelu-" + release_name)(top)


def build_segmentation_model() -> keras.Model:
    """Build the exact released input-to-``seg_out`` dependency graph."""
    image = keras.Input(shape=(None, None, 1), name="input_2")
    features = ReleaseImageNormalization(name="img_norm")(image)
    for index, (name, filters) in enumerate(BACKBONE_BLOCKS):
        features = _conv_bn_prelu(features, filters, 3, name)
        if index in (1, 3, 6):
            features = layers.MaxPooling2D(pool_size=2, strides=2)(features)

    logits = []
    for branch, dilation in enumerate((1, 4, 8), start=1):
        scale = _conv_bn_prelu(features, 256, 3, f"4_{branch}", dilation=dilation)
        scale = _conv_bn_prelu(scale, 128, 1, f"seg_{branch}_1")
        logits.append(
            layers.Conv2D(1, 1, padding="same", name=f"seg_{branch}_2")(scale)
        )
    fused = layers.Add(name="seg_fusion_sum")(logits)
    output = layers.Activation("sigmoid", name="seg_out")(fused)
    model = keras.Model(image, output, name="fingernet_release_segmentation")
    if model.count_params() != SEGMENTATION_GRAPH_STORED_SCALAR_COUNT:
        raise AssertionError(
            f"segmentation graph has {model.count_params()} parameters, expected "
            f"{SEGMENTATION_GRAPH_STORED_SCALAR_COUNT} stored scalars"
        )
    trainable_count = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
    if trainable_count != SEGMENTATION_GRAPH_PARAMETER_COUNT:
        raise AssertionError(
            f"segmentation graph has {trainable_count} learned parameters, expected "
            f"{SEGMENTATION_GRAPH_PARAMETER_COUNT}"
        )
    return model


def load_release_segmentation_weights(model: keras.Model, path: Path) -> List[str]:
    """Load and shape-check every released tensor used by ``model``."""
    audit_release_weights(path)
    loaded: List[str] = []
    with h5py.File(path, "r") as handle:
        for layer in model.layers:
            if not layer.weights:
                continue
            if layer.name not in SEGMENTATION_WEIGHT_LAYERS:
                raise ValueError(f"unexpected weighted compatibility layer {layer.name!r}")
            group = handle[layer.name]
            names = [_decode(value) for value in group.attrs["weight_names"]]
            values = [np.asarray(group[name]) for name in names]
            expected = [tuple(variable.shape) for variable in layer.weights]
            actual = [tuple(value.shape) for value in values]
            if actual != expected:
                raise ValueError(
                    f"legacy tensor shapes for {layer.name} are {actual}, expected {expected}"
                )
            layer.set_weights(values)
            loaded.extend(names)
    if set(layer.name for layer in model.layers if layer.weights) != set(
        SEGMENTATION_WEIGHT_LAYERS
    ):
        raise ValueError("compatibility graph did not consume every declared segmentation layer")
    return loaded


def release_postprocess(coarse_probability: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Apply the release's rounding, coarse 5x5 opening, and nearest upsampling."""
    probability = np.asarray(coarse_probability)
    if probability.ndim != 2:
        raise ValueError(f"expected 2-D coarse probability map, got {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError("FingerNet produced a non-finite segmentation probability")
    coarse = np.round(probability).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, kernel)
    upsampled = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1).astype(bool)
    expected_crop = (output_shape[0] // 8 * 8, output_shape[1] // 8 * 8)
    if upsampled.shape != expected_crop:
        raise ValueError(
            f"coarse output expands to {upsampled.shape}, expected cropped grid {expected_crop}"
        )
    # The release floors each input dimension to a multiple of eight.  Preserve
    # that field of view on a full-size benchmark canvas without rescaling it.
    output = np.zeros(output_shape, dtype=bool)
    output[: expected_crop[0], : expected_crop[1]] = upsampled
    return output


class LegacyFingerNetSegmenter:
    """Inference-only adapter around the released segmentation graph."""

    def __init__(self, weights_path: Path, device: str = "cpu") -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.release_audit = audit_release_weights(weights_path)
        self.model = build_segmentation_model()
        self.loaded_weight_names = load_release_segmentation_weights(
            self.model, weights_path
        )
        self.model.to(torch.device(device))
        self.model.eval()
        self.device = torch.device(device)

    def _coarse_probability(self, image_u8: np.ndarray) -> np.ndarray:
        """Run the released graph and return its unthresholded coarse output."""
        image = np.asarray(image_u8)
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("FingerNet expects one 2-D uint8 grayscale image")
        crop_h, crop_w = image.shape[0] // 8 * 8, image.shape[1] // 8 * 8
        if crop_h == 0 or crop_w == 0:
            raise ValueError("FingerNet input dimensions must each be at least eight pixels")
        cropped = image[:crop_h, :crop_w].astype(np.float32) / 255.0
        if float(cropped.var()) == 0.0:
            raise ValueError("released FingerNet normalization is undefined for a constant image")
        tensor = torch.from_numpy(cropped[None, ..., None]).to(self.device)
        with torch.inference_mode():
            probability = self.model(tensor, training=False)
        coarse = probability.detach().float().cpu().numpy()[0, ..., 0]
        if coarse.ndim != 2 or not np.isfinite(coarse).all():
            raise RuntimeError("FingerNet produced an invalid probability map")
        return np.clip(coarse, 0.0, 1.0)

    def probability(self, image_u8: np.ndarray) -> np.ndarray:
        """Return raw segmentation probabilities expanded to the input grid."""
        image = np.asarray(image_u8)
        coarse = self._coarse_probability(image)
        expanded = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)
        output = np.zeros(image.shape, dtype=np.float32)
        output[: expanded.shape[0], : expanded.shape[1]] = expanded
        return output

    def run_with_probability(
        self, image_u8: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the official mask and raw expanded probability in one pass."""
        image = np.asarray(image_u8)
        coarse = self._coarse_probability(image)
        expanded = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)
        probability = np.zeros(image.shape, dtype=np.float32)
        probability[: expanded.shape[0], : expanded.shape[1]] = expanded
        return release_postprocess(coarse, image.shape), probability

    @staticmethod
    def mask_from_probability(probability: np.ndarray) -> np.ndarray:
        """Reconstruct the released mask exactly from an expanded raw map."""
        value = np.asarray(probability)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("FingerNet probability must be a finite 2-D array")
        crop_h, crop_w = value.shape[0] // 8 * 8, value.shape[1] // 8 * 8
        coarse = value[:crop_h:8, :crop_w:8]
        return release_postprocess(coarse, value.shape)

    def run(self, image_u8: np.ndarray) -> np.ndarray:
        """Return the official binary segmentation output on the input grid."""
        image = np.asarray(image_u8)
        return release_postprocess(self._coarse_probability(image), image.shape)

    @property
    def device_label(self) -> str:
        """Return the table's non-identifying device category."""
        return "GPU" if self.device.type == "cuda" else "CPU"
