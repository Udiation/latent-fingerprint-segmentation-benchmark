from __future__ import annotations

import numpy as np

from segbench.metrics import image_metrics


def test_identity_mask_is_perfect() -> None:
    ground_truth = np.zeros((48, 64), dtype=bool)
    ground_truth[9:39, 12:52] = True
    result = image_metrics(ground_truth.copy(), ground_truth)
    assert result["dice"] == 1.0
    assert result["jaccard"] == 1.0
    assert result["mdr"] == 0.0
    assert result["fdr"] == 0.0


def test_all_background_misses_foreground() -> None:
    ground_truth = np.zeros((32, 32), dtype=bool)
    ground_truth[8:24, 8:24] = True
    result = image_metrics(np.zeros_like(ground_truth), ground_truth)
    assert result["dice"] == 0.0
    assert result["mdr"] == 1.0
    assert result["fdr"] == 0.0

