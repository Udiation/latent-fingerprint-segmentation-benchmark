"""Synthetic-only tests for the receipt-authenticated private demo backend."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_demo_receipt import build as build_receipt  # noqa: E402
from demo.app import render_compare, render_explore, render_hull, render_upload  # noqa: E402
from demo.restricted_backend import (  # noqa: E402
    RECEIPT_ENV,
    RECEIPT_KIND,
    RECEIPT_SHA_ENV,
    ROOT_ENV,
    ArtifactStore,
    ReceiptError,
    RestrictedDemoBackend,
    create_backend,
)
from segbench.metrics import image_metrics  # noqa: E402
from segbench.deep_models import build_model  # noqa: E402


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _png(path: Path, value: np.ndarray) -> None:
    assert cv2.imwrite(str(path), value)


def _save_probability(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, value.astype(np.float16), allow_pickle=False)


def _hull_metrics(prediction: np.ndarray, hull: np.ndarray, points: np.ndarray) -> dict[str, float]:
    area = int(prediction.sum())
    return {
        "mc": float(prediction[points[:, 1], points[:, 0]].mean()),
        "hr": float(np.count_nonzero(prediction & hull) / hull.sum()),
        "par": float(prediction.sum() / hull.sum()),
        "ohf": float(np.count_nonzero(prediction & ~hull) / area) if area else 0.0,
    }


def _fixture(root: Path) -> tuple[Path, str]:
    root.mkdir()
    image = np.full((32, 40), 165, dtype=np.uint8)
    image[5:27, 7:33] = 90
    ground_truth = np.zeros_like(image, dtype=bool)
    ground_truth[6:26, 8:32] = True
    hull_image = np.full((36, 44), 135, dtype=np.uint8)
    hull_image[7:29, 9:35] = 70
    hull = np.zeros_like(hull_image, dtype=bool)
    hull[9:27, 12:32] = True
    points = np.asarray([[14, 11], [29, 12], [30, 24], [15, 25]], dtype=np.int64)

    probability_a = np.where(ground_truth, 0.9, 0.1).astype(np.float32)
    mask_b = np.zeros_like(ground_truth)
    mask_b[8:28, 10:34] = True
    probability_b = np.where(mask_b, 0.8, 0.2).astype(np.float32)
    b1_probability_a = np.where(hull, 0.9, 0.1).astype(np.float32)
    b1_probability_b = np.full_like(hull_image, 0.2, dtype=np.float32)

    _png(root / "test.png", image)
    _png(root / "gt.png", ground_truth.astype(np.uint8) * 255)
    _png(root / "b1.png", hull_image)
    _png(root / "hull.png", hull.astype(np.uint8) * 255)
    _save_probability(root / "test_a.npy", probability_a)
    _save_probability(root / "test_b.npy", probability_b)
    _save_probability(root / "test_unet.npy", probability_a)
    _save_probability(root / "b1_a.npy", b1_probability_a)
    _save_probability(root / "b1_b.npy", b1_probability_b)
    _save_probability(root / "b1_unet.npy", b1_probability_a)
    _write_csv(
        root / "points.csv",
        ["x_px", "y_px"],
        [{"x_px": int(x), "y_px": int(y)} for x, y in points],
    )

    model = DummyClassifier(strategy="constant", constant=True)
    model.fit(np.zeros((2, 8), dtype=np.float32), np.asarray([False, True]))
    joblib.dump(model, root / "classical.joblib")
    deep_config = {
        "schema_version": 1,
        "method": "unet",
        "task": "pixel",
        "seed": 0,
        "data": {"target_ppi": 500, "normalization": "train_only"},
        "model": {"in_channels": 1, "base_channels": 1, "levels": 4},
        "training": {"learning_rate": 0.001, "threshold_candidates": [0.5]},
        "runtime": {"memory_fraction": 0.1, "acceleration": {
            "precision": "fp32", "channels_last": False,
        }},
        "inference": {"pad_multiple": 16, "tile_size": 64, "tile_overlap": 8},
    }
    deep_model = build_model("unet", deep_config["model"])
    (root / "unet.yaml").write_text(
        yaml.safe_dump(deep_config, sort_keys=True), encoding="utf-8"
    )
    torch.save(
        {
            "method": "unet",
            "task": "pixel",
            "config": deep_config,
            "normalization": {"mean": 0.5, "std": 0.25},
            "model_state": deep_model.state_dict(),
        },
        root / "unet.pt",
    )

    _write_csv(
        root / "test_manifest.csv",
        [
            "example_key", "label", "image_id", "examiner_value", "ppi",
            "image_artifact", "ground_truth_artifact", "split",
        ],
        [{
            "example_key": "test-one", "label": "Synthetic test one",
            "image_id": "synthetic-test-one", "examiner_value": "VALUE", "ppi": 500,
            "image_artifact": "test_image", "ground_truth_artifact": "test_gt",
            "split": "test",
        }],
    )
    _write_csv(
        root / "b1_manifest.csv",
        [
            "example_key", "label", "image_id", "group", "device", "capture", "ppi",
            "image_artifact", "hull_artifact", "minutiae_artifact", "split",
        ],
        [{
            "example_key": "b1-one", "label": "Synthetic exemplar one",
            "image_id": "synthetic-b1-one", "group": "synthetic", "device": "SYN",
            "capture": "roll", "ppi": 500, "image_artifact": "b1_image",
            "hull_artifact": "b1_hull", "minutiae_artifact": "b1_points", "split": "test",
        }],
    )
    predictions = []
    for dataset, example, prefix in (
        ("test", "test-one", "test"), ("b1", "b1-one", "b1")
    ):
        for method in ("method_a", "method_b", "unet"):
            predictions.append({
                "example_key": example,
                "method": method,
                "artifact": (
                    f"{prefix}_{method[-1]}_probability"
                    if method != "unet"
                    else f"{prefix}_unet_probability"
                ),
                "artifact_kind": "probability_npy",
                "representation": "pixel",
                "block_size": 16,
                "min_component_blocks": 4,
            })
        _write_csv(
            root / f"{dataset}_predictions.csv",
            [
                "example_key", "method", "artifact", "artifact_kind", "representation",
                "block_size", "min_component_blocks",
            ],
            predictions[-3:],
        )

    test_results = []
    for method, mask in (
        ("method_a", ground_truth), ("method_b", mask_b), ("unet", ground_truth)
    ):
        metrics = image_metrics(mask, ground_truth)
        test_results.append({
            "example_key": "test-one", "method": method,
            "dice": metrics["dice"], "mdr": metrics["mdr"], "fdr": metrics["fdr"],
            "ms_per_img": 2.5 if method == "method_a" else 3.5,
        })
    _write_csv(
        root / "test_results.csv",
        ["example_key", "method", "dice", "mdr", "fdr", "ms_per_img"],
        test_results,
    )
    b1_results = []
    for method, mask in (
        ("method_a", hull), ("method_b", np.zeros_like(hull)), ("unet", hull)
    ):
        metrics = _hull_metrics(mask, hull, points)
        b1_results.append({
            "example_key": "b1-one", "method": method, **metrics,
            "ms_per_img": 2.7 if method == "method_a" else 3.7,
        })
    _write_csv(
        root / "b1_results.csv",
        ["example_key", "method", "mc", "hr", "par", "ohf", "ms_per_img"],
        b1_results,
    )

    kinds = {
        "test_image": ("test.png", "image"),
        "test_gt": ("gt.png", "binary_mask"),
        "b1_image": ("b1.png", "image"),
        "b1_hull": ("hull.png", "binary_mask"),
        "b1_points": ("points.csv", "csv"),
        "test_manifest": ("test_manifest.csv", "csv"),
        "b1_manifest": ("b1_manifest.csv", "csv"),
        "test_results": ("test_results.csv", "csv"),
        "b1_results": ("b1_results.csv", "csv"),
        "test_predictions": ("test_predictions.csv", "csv"),
        "b1_predictions": ("b1_predictions.csv", "csv"),
        "test_a_probability": ("test_a.npy", "probability_npy"),
        "test_b_probability": ("test_b.npy", "probability_npy"),
        "test_unet_probability": ("test_unet.npy", "probability_npy"),
        "b1_a_probability": ("b1_a.npy", "probability_npy"),
        "b1_b_probability": ("b1_b.npy", "probability_npy"),
        "b1_unet_probability": ("b1_unet.npy", "probability_npy"),
        "classical_model": ("classical.joblib", "joblib_model"),
        "unet_config": ("unet.yaml", "yaml"),
        "unet_checkpoint": ("unet.pt", "torch_checkpoint"),
    }
    spec = {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "target_ppi": 500,
        "artifacts": {
            key: {"path": path, "kind": kind} for key, (path, kind) in kinds.items()
        },
        "methods": [
            {
                "key": "method_a", "display_name": "Method A", "reported_threshold": 0.5,
                "runner": {
                    "type": "classical_joblib", "model_artifact": "classical_model",
                    "target_ppi": 500, "min_component_blocks": 4,
                },
            },
            {
                "key": "method_b", "display_name": "Method B", "reported_threshold": 0.5,
                "runner": {"type": "precomputed_only"},
            },
            {
                "key": "unet", "display_name": "Tiny U-Net", "reported_threshold": 0.5,
                "runner": {
                    "type": "deep_checkpoint", "config_artifact": "unet_config",
                    "model_artifact": "unet_checkpoint",
                },
            },
        ],
        "datasets": {
            "test": {
                "manifest": "test_manifest", "results": "test_results",
                "predictions": "test_predictions",
            },
            "b1": {
                "manifest": "b1_manifest", "results": "b1_results",
                "predictions": "b1_predictions",
            },
        },
    }
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    receipt = root / "receipt.json"
    evidence = build_receipt(root, spec_path, receipt)
    return receipt, str(evidence["sha256"])


def _backend(tmp_path: Path) -> tuple[RestrictedDemoBackend, Path, str]:
    root = tmp_path / "private_fixture"
    receipt, digest = _fixture(root)
    store = ArtifactStore(root, receipt, digest)
    return RestrictedDemoBackend(store), root, digest


def test_receipt_backend_serves_test_and_b1_panels(tmp_path: Path) -> None:
    backend, _, _ = _backend(tmp_path)
    assert list(backend.method_names()) == ["Method A", "Method B", "Tiny U-Net"]
    assert list(backend.upload_method_names()) == ["Method A", "Tiny U-Net"]
    assert backend.supports_threshold_exploration("Method A") is True
    source = backend.load_example("test-one")
    assert source.image.shape == source.ground_truth.shape == (32, 40)
    prediction = backend.predict("Method A", source.image, source.ppi)
    assert np.array_equal(prediction.mask, source.ground_truth)
    hull = backend.load_hull_example("b1-one")
    assert hull.image.shape == hull.hull.shape == (36, 44)
    assert hull.minutiae_xy.shape == (4, 2)
    html = render_hull(backend, "b1-one", "Method A")
    assert "100.00%" in html and "1.0000" in html


def test_comparison_is_derived_and_cross_checked_against_results(tmp_path: Path) -> None:
    backend, root, digest = _backend(tmp_path)
    rows = backend.comparison_rows("Method A", "Method B")
    assert len(rows) == 1 and rows[0].dice_a == pytest.approx(1.0)
    html = render_compare(backend, "test-one", "Method A", "Method B")
    assert "Synthetic test one" in html
    # A changed result file is rejected by artifact authentication before a
    # misleading comparison can be shown.
    with (root / "test_results.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ReceiptError, match="size/type|SHA-256"):
        RestrictedDemoBackend(ArtifactStore(root, root / "receipt.json", digest))


def test_factory_requires_independent_receipt_digest(tmp_path: Path) -> None:
    _, root, digest = _backend(tmp_path)
    environment = {
        ROOT_ENV: str(root),
        RECEIPT_ENV: str(root / "receipt.json"),
        RECEIPT_SHA_ENV: digest,
    }
    backend = create_backend(environment)
    assert list(backend.method_names()) == ["Method A", "Method B", "Tiny U-Net"]
    with pytest.raises(ReceiptError, match="receipt SHA-256"):
        create_backend({**environment, RECEIPT_SHA_ENV: "0" * 64})


def test_validly_receipted_but_inconsistent_metric_is_rejected(tmp_path: Path) -> None:
    _, root, _ = _backend(tmp_path)
    with (root / "test_results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["dice"] = "0.25"
    _write_csv(
        root / "test_results.csv",
        ["example_key", "method", "dice", "mdr", "fdr", "ms_per_img"],
        rows,
    )
    second = root / "inconsistent_receipt.json"
    evidence = build_receipt(root, root / "spec.json", second)
    backend = RestrictedDemoBackend(ArtifactStore(root, second, str(evidence["sha256"])))
    with pytest.raises(ReceiptError, match="derived dice"):
        backend.comparison_rows("Method A", "Method B")


def test_artifact_mutation_and_symlink_fail_closed(tmp_path: Path) -> None:
    _, root, digest = _backend(tmp_path)
    receipt = root / "receipt.json"
    receipt_link = root / "receipt_link.json"
    receipt_link.symlink_to(receipt)
    with pytest.raises(ReceiptError, match="symlink"):
        ArtifactStore(root, receipt_link, digest)
    store = ArtifactStore(root, receipt, digest)
    (root / "test_a.npy").write_bytes(b"changed")
    with pytest.raises(ReceiptError, match="size/type|SHA-256"):
        store.bytes("test_a_probability", {"probability_npy"})

    other = tmp_path / "other.npy"
    other.write_bytes(b"untrusted")
    (root / "test_a.npy").unlink()
    (root / "test_a.npy").symlink_to(other)
    with pytest.raises(ReceiptError, match="symlink"):
        store.bytes("test_a_probability", {"probability_npy"})


def test_upload_inference_uses_authenticated_model_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, _ = _backend(tmp_path)
    image = np.full((20, 22), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    import base64

    data_url = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.chdir(scratch)
    before = list(scratch.iterdir())
    html = render_upload(backend, data_url, 500, ["Method A"])
    assert "processed in memory" in html
    assert list(scratch.iterdir()) == before
    with pytest.raises(Exception, match="Select at least one available method|frozen panels only"):
        render_upload(backend, data_url, 500, ["Method B"])


def test_probability_exploration_and_binary_result_separation(tmp_path: Path) -> None:
    backend, _, _ = _backend(tmp_path)
    source = backend.load_example("test-one")
    reported = backend.predict("Method A", source.image, source.ppi)
    explored = backend.predict("Method A", source.image, source.ppi, threshold=0.95)
    assert reported.mask.any() and not explored.mask.any()
    html = render_explore(backend, "test-one", ["Method A"], 0.95)
    assert "Exploration only" in html
    assert "1.0000" in html  # reported-threshold Dice, not the exploratory mask


def test_deep_checkpoint_upload_is_weights_only_and_in_memory(tmp_path: Path) -> None:
    backend, _, _ = _backend(tmp_path)
    image = np.full((21, 25), 128, dtype=np.uint8)
    prediction = backend.predict("Tiny U-Net", image, 500)
    assert prediction.mask.shape == image.shape
    assert prediction.mask.dtype == np.bool_
    assert prediction.milliseconds >= 0.0


def test_receipt_builder_refuses_escape_and_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("x\n1\n", encoding="utf-8")
    spec = {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "target_ppi": 500,
        "artifacts": {"bad": {"path": "../outside.csv", "kind": "csv"}},
        "methods": [],
        "datasets": {},
    }
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        build_receipt(root, spec_path, root / "receipt.json")


def test_receipt_contains_no_absolute_paths(tmp_path: Path) -> None:
    _, root, _ = _backend(tmp_path)
    payload = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    for artifact in payload["artifacts"].values():
        assert not Path(artifact["path"]).is_absolute()
        assert ".." not in Path(artifact["path"]).parts
