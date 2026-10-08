"""Narrow PyFing compatibility support for the Keras Torch backend.

PyFing 0.6 calls ``Tensor.numpy()`` directly in ``Sufs.run``. Torch tensors
require an explicit detach (and CUDA tensors require a host transfer). This
wrapper preserves PyFing's size handling, weights, threshold, and output logic;
only the tensor-to-NumPy conversion changes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import cv2
from pyfing.segmentation import Sufs


def host_numpy(value: Any) -> np.ndarray:
    """Convert a Keras/Torch prediction tensor to a host NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


class TorchSufs(Sufs):
    """SUFS with backend-safe prediction conversion and unchanged algorithm."""

    def _raw_probability(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("SUFS expects one two-dimensional uint8 image")
        original_h, original_w = image.shape
        size_info = self._compute_size(original_w, original_h)
        adjusted = self._adjust_input(image, *size_info)
        # ``training=False`` controls BatchNorm but does not disable PyTorch's
        # autograd tape.  Inference mode avoids retaining a full activation
        # graph while leaving the released model and threshold untouched.
        with torch.inference_mode():
            predictions = host_numpy(
                self.model(adjusted[np.newaxis, ..., np.newaxis], training=False)
            )
        probability = predictions[0].squeeze().astype(np.float32)
        if probability.ndim != 2 or not np.isfinite(probability).all():
            raise RuntimeError("SUFS produced an invalid raw probability map")
        return probability, size_info

    def probability(self, image: np.ndarray) -> np.ndarray:
        """Return the released sigmoid output on the original image grid."""
        original_h, original_w = image.shape
        probability, size_info = self._raw_probability(image)
        w, h, border_left, border_top, _border_right, _border_bottom = size_info
        probability = probability[
            border_top : border_top + h,
            border_left : border_left + w,
        ]
        if w != original_w or h != original_h:
            probability = cv2.resize(
                probability,
                (original_w, original_h),
                # Released SUFS thresholds before _adjust_output, whose resize
                # is nearest-neighbour.  Sampling sigmoid values with the same
                # geometry makes threshold(stored probability) exactly recover
                # the released mask; linear mixing would silently differ.
                interpolation=cv2.INTER_NEAREST,
            )
        if probability.shape != image.shape or not np.isfinite(probability).all():
            raise RuntimeError("SUFS produced an invalid probability map")
        return np.clip(probability, 0.0, 1.0)

    def run(self, image: np.ndarray, intermediate_results: Any = None) -> np.ndarray:
        """Segment one uint8 image using the released SUFS implementation."""
        if intermediate_results is not None:
            raise NotImplementedError("intermediate SUFS activations are not exposed")
        original_h, original_w = image.shape
        probability, size_info = self._raw_probability(image)
        mask = np.where(
            probability < self.parameters.threshold, 0, 255
        ).astype(np.uint8)
        return self._adjust_output(
            mask, *size_info, original_w, original_h
        )
