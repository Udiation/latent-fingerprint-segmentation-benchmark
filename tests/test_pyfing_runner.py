"""CPU-only integrity tests for PyFing tuning and mask generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pyfing as runner  # noqa: E402


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a synthetic PNG fixture and require success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def write_manifest_and_splits(
    root: Path, rows: list[dict[str, object]], assignments: dict[str, str]
) -> tuple[Path, Path]:
    """Write minimal synthetic protocol inputs."""
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    splits = root / "splits.json"
    splits.write_text(
        json.dumps({"seed": 42, "splits": assignments}), encoding="utf-8"
    )
    return manifest, splits


def generation_args(
    root: Path, manifest: Path, splits: Path, **overrides: object
) -> argparse.Namespace:
    """Return arguments for one synthetic GMFS generation run."""
    values: dict[str, object] = {
        "method": "gmfs",
        "display_name": "GMFS (default)",
        "manifest": manifest,
        "splits": splits,
        "subset": "finger_latent",
        "split": "test",
        "target_ppi": 500.0,
        "parameters_json": None,
        "output_root": root / "predictions",
        "run_card": root / "run.json",
        "warmup": 0,
        "progress_every": 1,
        "memory_fraction": 0.3,
        "limit": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_preregistered_grid_is_default_plus_sixteen_unique_corners() -> None:
    """The committed grid gives GMFS exactly its declared validation budget."""
    config = yaml.safe_load((ROOT / "configs" / "gmfs.yaml").read_text())
    candidates = runner.gmfs_candidates(config)
    assert len(candidates) == 17
    assert runner.gmfs_default() in candidates
    assert len({json.dumps(item, sort_keys=True) for item in candidates}) == 17


def test_gmfs_default_matches_installed_pyfing() -> None:
    """The explicit default must not drift from the pinned implementation."""
    pytest.importorskip("pyfing")
    from pyfing.segmentation import GmfsParameters

    released = GmfsParameters()
    observed = {key: getattr(released, key) for key in runner.GMFS_PARAMETER_KEYS}
    assert observed == runner.gmfs_default()


@pytest.mark.parametrize(
    "field,value",
    [("sigma", float("nan")), ("closing_count", 1.5), ("image_dpi", 500.5)],
)
def test_gmfs_parameter_validation_fails_closed(field: str, value: float) -> None:
    parameters = runner.gmfs_default()
    parameters[field] = value
    with pytest.raises(ValueError):
        runner.validate_gmfs_parameters(parameters)


def test_load_rows_uses_requested_split_and_nonprint_is_test_only(tmp_path: Path) -> None:
    rows = []
    assignments = {"train-subject": "train", "val-subject": "val", "test-subject": "test"}
    for subject, split in assignments.items():
        for examiner in ("VALUE", "NONPRINT"):
            rows.append(
                {
                    "image_id": f"{split}-{examiner}",
                    "image_path": "unused.png",
                    "mask_path": "unused-mask.png",
                    "subject_id": subject,
                    "subset": "finger_latent",
                    "ppi": 500,
                    "examiner_value": examiner,
                }
            )
    manifest, splits = write_manifest_and_splits(tmp_path, rows, assignments)
    val = runner.load_rows(manifest, splits, "finger_latent", "val", "mask_path", 0)
    test = runner.load_rows(manifest, splits, "finger_latent", "test", None, 0)
    assert set(val.image_id) == {"val-VALUE", "val-NONPRINT"}
    assert list(test.image_id) == ["test-VALUE"]


def test_image_and_gt_use_distinct_ppi_on_one_audited_grid(tmp_path: Path) -> None:
    image = np.arange(120 * 240, dtype=np.uint8).reshape(120, 240)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:80, 40:160] = 255
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    write_png(image_path, image)
    write_png(mask_path, mask)
    row = {
        "image_id": "mixed-grid",
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "ppi": 1200,
        "mask_ppi": 1000,
        "target_ppi": 500,
        "target_height": 50,
        "target_width": 100,
    }
    resized_image = runner.read_image_500(row, 500)
    resized_mask = runner.read_gt_500(row, "mask_path", 500, resized_image.shape)
    assert resized_image.shape == (50, 100)
    assert resized_mask.shape == (50, 100)


def test_tuning_reads_validation_rows_only_and_records_audit_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "image_id": split,
            "image_path": f"never-read-{split}.png",
            "mask_path": f"never-read-{split}-mask.png",
            "subject_id": split,
            "subset": "finger_latent",
            "ppi": 500,
        }
        for split in ("train", "val", "test")
    ]
    manifest, splits = write_manifest_and_splits(
        tmp_path, rows, {split: split for split in ("train", "val", "test")}
    )
    seen: list[str] = []

    def fake_score(
        row: dict[str, object], candidates: list[dict[str, object]], *_: object
    ) -> tuple[str, list[float]]:
        seen.append(str(row["image_id"]))
        return str(row["image_id"]), [0.5] * len(candidates)

    monkeypatch.setattr(runner, "score_gmfs_row", fake_score)
    output = tmp_path / "gmfs.json"
    args = argparse.Namespace(
        manifest=manifest,
        splits=splits,
        subset="finger_latent",
        mask_column="mask_path",
        config=ROOT / "configs" / "gmfs.yaml",
        out_json=output,
        n_jobs=1,
        limit=0,
    )
    runner.tune_gmfs(args)
    payload = json.loads(output.read_text())
    assert seen == ["val"]
    assert payload["selection_split"] == "val"
    assert payload["selection_metric"] == "macro_dice"
    assert payload["mask_column"] == "mask_path"
    assert payload["subset"] == "finger_latent"
    assert output.with_suffix(".per_image.csv").is_file()


def tuned_payload(manifest: Path, splits: Path) -> dict[str, object]:
    """Build a minimal internally consistent validation artifact."""
    selected = runner.gmfs_default()
    return {
        "schema_version": 1,
        "method": "gmfs_tuned",
        "selection_split": "val",
        "selection_metric": "macro_dice",
        "subset": "finger_latent",
        "mask_column": "mask_path",
        "target_ppi": 500.0,
        "n_images": 1,
        "candidate_count": 1,
        "selected_index": 0,
        "selected": selected,
        "selected_val_dice": 0.75,
        "candidates": [{"index": 0, "parameters": selected, "val_dice": 0.75}],
        "manifest_sha256": runner.sha256_file(manifest),
        "splits_sha256": runner.sha256_file(splits),
        "config_sha256": "a" * 64,
        "implementation": runner.implementation_provenance("gmfs", None),
    }


def test_tuned_parameters_are_bound_to_validation_artifact_and_data(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    splits = tmp_path / "splits.json"
    manifest.write_text("manifest-v1", encoding="utf-8")
    splits.write_text("splits-v1", encoding="utf-8")
    artifact = tmp_path / "selected.json"
    artifact.write_text(json.dumps(tuned_payload(manifest, splits)), encoding="utf-8")
    parameters, provenance = runner.load_tuned_parameters(
        artifact, manifest, splits, "finger_latent"
    )
    assert parameters == runner.gmfs_default()
    assert provenance["selection_split"] == "val"

    payload = json.loads(artifact.read_text())
    payload["selection_split"] = "test"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        runner.load_tuned_parameters(artifact, manifest, splits, "finger_latent")


def test_generation_never_reads_gt_and_resume_hash_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
    image_path = tmp_path / "image.png"
    write_png(image_path, image)
    rows = [
        {
            "image_id": "synthetic-sample",
            "image_path": str(image_path),
            "mask_path": str(tmp_path / "must-not-be-read.png"),
            "subject_id": "subject",
            "subset": "finger_latent",
            "ppi": 500,
        }
    ]
    manifest, splits = write_manifest_and_splits(
        tmp_path, rows, {"subject": "test"}
    )
    calls = {"count": 0}

    class DummySegmenter:
        def run(self, source: np.ndarray) -> np.ndarray:
            calls["count"] += 1
            result = np.zeros_like(source, dtype=np.uint8)
            result[4:-4, 4:-4] = 255
            return result

    monkeypatch.setattr(runner, "build_segmenter", lambda *_: (DummySegmenter(), 0))
    monkeypatch.setattr(
        runner, "implementation_provenance", lambda *_: {"implementation": "fixture"}
    )
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "fixture")
    monkeypatch.setattr(
        runner, "read_mask", lambda *_: pytest.fail("generation read ground truth")
    )
    args = generation_args(tmp_path, manifest, splits)

    runner.generate(args)
    target = args.output_root / "synthetic-sample.png"
    assert calls["count"] == 1
    assert set(np.unique(cv2.imread(str(target), cv2.IMREAD_GRAYSCALE))) == {0, 255}

    # A published complete card is immutable and prevents output mutation.
    target_sha = runner.sha256_file(target)
    card_sha = runner.sha256_file(args.run_card)
    with pytest.raises(ValueError, match="immutable complete"):
        runner.generate(args)
    assert calls["count"] == 1
    assert runner.sha256_file(target) == target_sha
    assert runner.sha256_file(args.run_card) == card_sha

    # Even after external tampering, rerun refuses before model or output writes.
    write_png(target, np.zeros_like(image))
    tampered_sha = runner.sha256_file(target)
    with pytest.raises(ValueError, match="immutable complete"):
        runner.generate(args)
    assert calls["count"] == 1
    assert runner.sha256_file(target) == tampered_sha
    run_card = json.loads(args.run_card.read_text())
    assert run_card["device"] == "CPU"
    assert run_card["prediction_count"] == 1
    assert len(run_card["prediction_set_sha256"]) == 64


def test_torch_sufs_conversion_preserves_shape_polarity_and_disables_autograd() -> None:
    pytest.importorskip("pyfing")
    from segbench.pyfing_compat import TorchSufs, host_numpy

    tensor = torch.tensor([1.0], requires_grad=True)
    np.testing.assert_array_equal(host_numpy(tensor), np.array([1.0], dtype=np.float32))

    observed: dict[str, object] = {}

    class FakeModel:
        def __call__(self, value: np.ndarray, training: bool) -> torch.Tensor:
            observed["shape"] = value.shape
            observed["training"] = training
            observed["grad_enabled"] = torch.is_grad_enabled()
            return torch.full(value.shape, 0.75)

    segmenter = TorchSufs.__new__(TorchSufs)
    segmenter.parameters = SimpleNamespace(
        dnn_input_dpi=500,
        image_dpi=500,
        dnn_input_size_multiple=1,
        border=0,
        threshold=0.5,
    )
    segmenter.model = FakeModel()
    image = np.zeros((7, 9), dtype=np.uint8)
    result = segmenter.run(image)
    assert observed == {
        "shape": (1, 7, 9, 1),
        "training": False,
        "grad_enabled": False,
    }
    assert result.shape == image.shape
    assert np.all(result == 255)


def test_torch_sufs_stored_probability_matches_threshold_before_nearest_resize() -> None:
    pytest.importorskip("pyfing")
    from segbench.pyfing_compat import TorchSufs
    from segbench.probability_artifacts import storage_roundtrip

    class FakeModel:
        def __call__(self, value: np.ndarray, training: bool) -> torch.Tensor:
            output = np.full(value.shape, 0.49, dtype=np.float32)
            output[:, :, output.shape[2] // 2 :, :] = 0.51
            return torch.from_numpy(output)

    segmenter = TorchSufs.__new__(TorchSufs)
    segmenter.parameters = SimpleNamespace(
        dnn_input_dpi=500,
        image_dpi=1000,
        dnn_input_size_multiple=1,
        border=0,
        threshold=0.5,
    )
    segmenter.model = FakeModel()
    image = np.zeros((11, 13), dtype=np.uint8)
    stored = storage_roundtrip(segmenter.probability(image))
    expected = segmenter.run(image) > 0
    assert np.array_equal(stored >= 0.5, expected)


def test_torch_sufs_matches_released_model_computation() -> None:
    """The compatibility wrapper changes conversion only, not SUFS predictions."""
    pytest.importorskip("pyfing")
    from segbench.pyfing_compat import TorchSufs, host_numpy

    torch.set_num_threads(1)
    segmenter = TorchSufs()
    image = np.random.default_rng(4).integers(0, 256, (32, 40), dtype=np.uint8)
    size_info = segmenter._compute_size(image.shape[1], image.shape[0])
    adjusted = segmenter._adjust_input(image, *size_info)
    with torch.inference_mode():
        # PyFing's released ``run`` supplies this implicit channel shape. Keras
        # normalizes it to the same four-dimensional output.
        released = host_numpy(segmenter.model(adjusted[np.newaxis, ...], training=False))
    expected = np.where(
        released[0].squeeze() < segmenter.parameters.threshold, 0, 255
    ).astype(np.uint8)
    expected = segmenter._adjust_output(
        expected, *size_info, image.shape[1], image.shape[0]
    )
    np.testing.assert_array_equal(segmenter.run(image), expected)
