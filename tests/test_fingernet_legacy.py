import importlib.util
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

os.environ.setdefault("KERAS_BACKEND", "torch")

from segbench.fingernet_legacy import (  # noqa: E402
    LegacyFingerNetSegmenter,
    RELEASE_COMMIT,
    RELEASE_WEIGHTS_SHA256,
    release_postprocess,
)


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts" / "run_fingernet.py"
SPEC = importlib.util.spec_from_file_location("run_fingernet", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_release_postprocess_preserves_release_crop_without_rescaling():
    coarse = np.ones((2, 3), dtype=np.float32)
    output = release_postprocess(coarse, (19, 27))
    assert output.shape == (19, 27)
    assert output[:16, :24].all()
    assert not output[16:, :].any()
    assert not output[:, 24:].any()

    coarse = np.ones((6, 6), dtype=np.float32)
    output = release_postprocess(coarse, (51, 53))
    assert output[:48, :48].all()
    assert not output[48:, :].any()
    assert not output[:, 48:].any()


def test_release_postprocess_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        release_postprocess(np.asarray([[np.nan]], dtype=np.float32), (8, 8))


def test_stored_float16_probability_reconstructs_release_postprocess():
    coarse = np.linspace(0.1, 0.9, 30, dtype=np.float32).reshape(5, 6)
    expanded = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)
    probability = np.zeros((43, 51), dtype=np.float32)
    probability[:40, :48] = expanded.astype(np.float16).astype(np.float32)
    expected = release_postprocess(coarse.astype(np.float16), probability.shape)
    observed = LegacyFingerNetSegmenter.mask_from_probability(probability)
    assert np.array_equal(observed, expected)


def test_frozen_config_matches_release(tmp_path):
    config = {
        "method": "fingernet",
        "display_name": "FingerNet (Tang et al. 2017)",
        "target_ppi": 500.0,
        "released_weights": {"sha256": RELEASE_WEIGHTS_SHA256},
        "source": {"commit": RELEASE_COMMIT},
        "inference": {
            "batch_size": 1,
            "coarse_stride": 8,
            "threshold": 0.5,
            "dimension_policy": "floor_to_stride_then_background_fill",
            "postprocess": {"operation": "opening", "kernel": [5, 5]},
        },
    }
    path = tmp_path / "fingernet.yaml"
    path.write_text(yaml.safe_dump(config))
    assert RUNNER.validate_config(path)["target_ppi"] == 500.0
    config["inference"]["threshold"] = 0.4
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="deviates"):
        RUNNER.validate_config(path)


def test_load_rows_does_not_read_or_require_ground_truth(tmp_path):
    image = np.full((24, 24), 127, np.uint8)
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), image)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_id": "kept",
                "image_path": str(image_path),
                "mask_path": "/does/not/exist.png",
                "subject_id": "s1",
                "subset": "finger_latent",
                "ppi": 500,
                "examiner_value": "VALUE",
            },
            {
                "image_id": "excluded_nonprint",
                "image_path": str(image_path),
                "mask_path": "/also/missing.png",
                "subject_id": "s2",
                "subset": "finger_latent",
                "ppi": 500,
                "examiner_value": "NONPRINT",
            },
        ]
    ).to_csv(manifest, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits": {"s1": "test", "s2": "test"}}))
    rows = RUNNER.load_rows(manifest, splits, "finger_latent", "test", 0)
    assert rows.image_id.tolist() == ["kept"]


@pytest.mark.skipif(
    not os.environ.get("FINGERNET_RELEASE_WEIGHTS"),
    reason="set FINGERNET_RELEASE_WEIGHTS for release-checkpoint integration",
)
def test_release_checkpoint_audit_opt_in():
    from segbench.fingernet_legacy import audit_release_weights

    audit = audit_release_weights(Path(os.environ["FINGERNET_RELEASE_WEIGHTS"]))
    assert audit["sha256"] == RELEASE_WEIGHTS_SHA256
    assert audit["stored_scalar_count"] == 4_718_014
    assert audit["parameter_count"] == 4_709_694
