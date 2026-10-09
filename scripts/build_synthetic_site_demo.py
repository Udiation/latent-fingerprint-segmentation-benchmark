#!/usr/bin/env python3
"""Build receipt-bound, synthetic-only method outputs for the public website.

This is deliberately a demonstration builder, not a benchmark runner.  It
uses a tiny deterministic CPU training budget, train-only synthetic subjects,
one validation subject for threshold selection, and never emits scores.  The
architectures and inference functions are the same implementations used by the
benchmark.  Methods whose required released weights are not distributed in
the public repository are recorded as unavailable rather than imitated.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import itertools
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

# This process is CPU-only even on a workstation that happens to expose CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import pandas as pd
import yaml

import _path  # noqa: F401
from segbench import features as block_features
from segbench.io import read_gray, read_mask, to_target_ppi
from segbench.metrics import image_metrics
from segbench.models import build as build_classical
from segbench.postprocess import postprocess_blocks
from segbench.synthetic_demo import (
    EXTERNAL_UNAVAILABLE,
    METHOD_KEYS,
    canonical_sha256,
    method_record_fingerprint,
    prediction_set_sha256,
    sha256_file,
    validate_synthetic_demo_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC = ROOT / "examples" / "synthetic"
LABELS = {
    "gmfs_default": "GMFS (default)",
    "gmfs_tuned": "GMFS (synthetic-val tuned)",
    "sufs_zero": "SUFS (released weights)",
    "sufs_ft": "SUFS (synthetic micro-fine-tune)",
    "svm": "SVM (Nyström RBF)",
    "rf": "Random forest",
    "welm": "Weighted ELM",
    "lenet": "LeNet",
    "alexnet": "AlexNet",
    "cnn_ae": "CNN + stacked autoencoder",
    "unet": "U-Net",
    "unetpp": "UNet++",
    "drunet": "DRUnet",
    "runet_no_alignment": "RUnet (no alignment)",
    "segformer_sub": "SegFormer-B2 (substitute†)",
    "fingernet": "FingerNet (released weights)",
}
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _source_hashes(*relative: str) -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in relative}


def _dataset_receipt(root: Path) -> dict[str, Any]:
    path = root / "synthetic_receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "segbench_synthetic_dataset":
        raise ValueError("input is not the generated synthetic dataset")
    generator = ROOT / "scripts" / "make_synthetic_demo.py"
    if payload.get("generator_sha256") != sha256_file(generator):
        raise ValueError("synthetic dataset generator identity mismatch")
    base_assets = []
    for raw in payload.get("assets", []):
        if raw.get("role") == "method_prediction":
            continue
        relative = Path(str(raw.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".png":
            raise ValueError("synthetic dataset receipt has an unsafe asset path")
        path_on_disk = root / relative
        if not path_on_disk.is_file() or sha256_file(path_on_disk) != raw.get("sha256"):
            raise ValueError(f"synthetic dataset asset identity mismatch: {relative}")
        base_assets.append(dict(raw))
    payload["assets"] = base_assets
    payload.pop("synthetic_demo", None)
    return payload


def _resolve_manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    candidate = root / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"synthetic manifest asset is absent: {Path(value).name}")


def _load_data(root: Path) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    manifest = pd.read_csv(root / "manifest.csv")
    required = {"image_id", "image_path", "mask_path", "subject_id", "subset", "ppi"}
    if not required.issubset(manifest.columns):
        raise ValueError("synthetic manifest lacks required columns")
    latent = manifest[manifest["subset"] == "finger_latent"].copy()
    latent = latent.sort_values("image_id", kind="stable").reset_index(drop=True)
    if latent["subject_id"].nunique() < 4 or len(latent) < 4:
        raise ValueError("synthetic method demo requires at least four latent subjects")
    loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in latent.to_dict("records"):
        source = read_gray(_resolve_manifest_path(root, str(row["image_path"])))
        target = read_mask(_resolve_manifest_path(root, str(row["mask_path"])))
        image_500 = to_target_ppi(source, float(row["ppi"]), 500.0)
        mask_500 = to_target_ppi(target, float(row["ppi"]), 500.0, is_mask=True)
        if image_500.shape != mask_500.shape:
            raise ValueError(f"synthetic image/mask mismatch: {row['image_id']}")
        loaded[str(row["image_id"])] = (source, image_500, mask_500)
    return latent, loaded


def _split_subjects(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    subjects = sorted(str(value) for value in frame["subject_id"].unique())
    train = subjects[:-2]
    validation = [subjects[-2]]
    held_out = [subjects[-1]]
    if not train or set(train) & set(validation + held_out):
        raise ValueError("invalid synthetic subject split")
    return train, validation, held_out


def _rows_for(frame: pd.DataFrame, subjects: Sequence[str]) -> list[dict[str, Any]]:
    return frame[frame["subject_id"].astype(str).isin(subjects)].to_dict("records")


def _normalization(rows: Sequence[Mapping[str, Any]], data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    pixels = np.concatenate(
        [data[str(row["image_id"])][1].astype(np.float32).ravel() / 255.0 for row in rows]
    )
    mean = float(pixels.mean())
    std = float(pixels.std())
    if not np.isfinite(std) or std < 1e-6:
        raise ValueError("synthetic train-only normalization is degenerate")
    return mean, std


def _threshold(
    probabilities: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for value in THRESHOLDS:
        scores[f"{value:.1f}"] = float(
            np.mean(
                [
                    image_metrics(probability >= value, target)["dice"]
                    for probability, target in zip(probabilities, targets)
                ]
            )
        )
    chosen = max(THRESHOLDS, key=lambda value: (scores[f"{value:.1f}"], -abs(value - 0.5), -value))
    return float(chosen), scores


def _block_training_arrays(
    rows: Sequence[Mapping[str, Any]], data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for row in rows:
        image = data[str(row["image_id"])][1]
        target = data[str(row["image_id"])][2]
        values = block_features.extract_block_features(image)
        height, width, count = values.shape
        features.append(values.reshape(-1, count))
        labels.append(block_features.block_labels(target, height, width).reshape(-1))
    return np.concatenate(features), np.concatenate(labels)


def _classical_methods(
    methods: set[str],
    train_rows: Sequence[Mapping[str, Any]],
    display_rows: Sequence[Mapping[str, Any]],
    data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    selected = methods & {"svm", "rf", "welm"}
    if not selected:
        return {}
    train_x, train_y = _block_training_arrays(train_rows, data)
    params = {
        "svm": {"gamma": 0.2, "C": 1.0, "n_components": min(128, len(train_x))},
        "rf": {"n_estimators": 32, "max_depth": 20, "min_samples_leaf": 1},
        "welm": {"n_hidden": 64, "C": 100.0},
    }
    outputs: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for method in METHOD_KEYS:
        if method not in selected:
            continue
        model = build_classical(method, params[method], seed=7, n_jobs=1).fit(train_x, train_y)
        predictions = {}
        for row in display_rows:
            image = data[str(row["image_id"])][1]
            values = block_features.extract_block_features(image)
            height, width, count = values.shape
            blocks = np.asarray(model.predict(values.reshape(-1, count))).reshape(height, width)
            blocks = postprocess_blocks(blocks, min_component_blocks=4)
            predictions[str(row["image_id"])] = block_features.blocks_to_mask(blocks, image.shape)
        outputs[method] = (
            predictions,
            {
                "implementation_sha256": _source_hashes(
                    "segbench/features.py", "segbench/models.py", "segbench/postprocess.py"
                ),
                "synthetic_demo_parameters": params[method],
                "train_blocks": int(len(train_x)),
                "seed": 7,
            },
        )
    return outputs


def _gmfs_methods(
    methods: set[str],
    validation_rows: Sequence[Mapping[str, Any]],
    display_rows: Sequence[Mapping[str, Any]],
    data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    selected = methods & {"gmfs_default", "gmfs_tuned"}
    if not selected:
        return {}
    from pyfing.segmentation import Gmfs, GmfsParameters

    config = yaml.safe_load((ROOT / "configs" / "gmfs.yaml").read_text(encoding="utf-8"))
    default = dict(config["default"])
    default["image_dpi"] = int(config["target_ppi"])
    candidates = [default]
    grid = config["tuning"]["candidates"]
    for sigma, threshold, closing, opening in itertools.product(
        grid["sigma"], grid["threshold"], grid["closing_count"], grid["opening_count"]
    ):
        candidates.append(
            {
                "sigma": float(sigma),
                "percentile": int(default["percentile"]),
                "threshold": float(threshold),
                "closing_count": int(closing),
                "opening_count": int(opening),
                "image_dpi": 500,
            }
        )
    unique = {json.dumps(value, sort_keys=True): value for value in candidates}
    candidates = [unique[key] for key in sorted(unique)]
    chosen = default
    if "gmfs_tuned" in selected:
        scored = []
        for parameters in candidates:
            segmenter = Gmfs(GmfsParameters(**parameters))
            dice = []
            for row in validation_rows:
                image = data[str(row["image_id"])][1]
                target = data[str(row["image_id"])][2]
                dice.append(float(image_metrics(segmenter.run(image) > 0, target)["dice"]))
            scored.append((float(np.mean(dice)), json.dumps(parameters, sort_keys=True), parameters))
        chosen = max(scored, key=lambda item: (item[0], item[1]))[2]
    outputs = {}
    for method, parameters in (("gmfs_default", default), ("gmfs_tuned", chosen)):
        if method not in selected:
            continue
        segmenter = Gmfs(GmfsParameters(**parameters))
        predictions = {
            str(row["image_id"]): segmenter.run(data[str(row["image_id"])][1]) > 0
            for row in display_rows
        }
        outputs[method] = (
            predictions,
            {
                "implementation_sha256": _source_hashes("scripts/run_pyfing.py"),
                "package": "pyfing",
                "package_version": _version("pyfing"),
                "synthetic_demo_parameters": parameters,
                "validation_candidates": len(candidates) if method == "gmfs_tuned" else 1,
            },
        )
    return outputs


def _sufs_zero(
    validation_rows: Sequence[Mapping[str, Any]],
    display_rows: Sequence[Mapping[str, Any]],
    data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from segbench.pyfing_compat import TorchSufs
    from pyfing.segmentation import Sufs

    segmenter = TorchSufs()
    val_probabilities = [segmenter.probability(data[str(row["image_id"])][1]) for row in validation_rows]
    threshold, scores = _threshold(
        val_probabilities, [data[str(row["image_id"])][2] for row in validation_rows]
    )
    predictions = {
        str(row["image_id"]): segmenter.probability(data[str(row["image_id"])][1]) >= threshold
        for row in display_rows
    }
    return predictions, {
        "implementation_sha256": _source_hashes(
            "segbench/pyfing_compat.py", "scripts/run_pyfing.py"
        ),
        "package": "pyfing",
        "package_version": _version("pyfing"),
        "keras_version": _version("keras"),
        "released_weights_sha256": sha256_file(
            Path(inspect.getfile(Sufs)).resolve().parent / "models" / "SUFS.weights.h5"
        ),
        "selected_threshold": threshold,
        "validation_threshold_scores": scores,
    }


def _torch_runtime(threads: int, seed: int = 7):
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch


def _pixel_methods(
    methods: set[str],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    display_rows: Sequence[Mapping[str, Any]],
    data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    steps: int,
    threads: int,
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    selected = methods & {"unet", "unetpp", "drunet", "runet_no_alignment"}
    if not selected:
        return {}
    torch = _torch_runtime(threads)
    from segbench.deep_models import build_model, parameter_counts
    from segbench.deep_utils import predict_pixel_probability, segmentation_loss

    mean, std = _normalization(train_rows, data)
    train_images = []
    train_targets = []
    for row in train_rows:
        image = data[str(row["image_id"])][1].astype(np.float32) / 255.0
        target = data[str(row["image_id"])][2].astype(np.float32)
        train_images.append(cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA))
        train_targets.append(cv2.resize(target, (64, 64), interpolation=cv2.INTER_NEAREST))
    batch = torch.from_numpy(((np.stack(train_images) - mean) / std)[:, None].astype(np.float32))
    target_batch = torch.from_numpy(np.stack(train_targets)[:, None].astype(np.float32))
    config_name = {
        "runet_no_alignment": "ra_runet",
        "unet": "unet",
        "unetpp": "unetpp",
        "drunet": "drunet",
    }
    outputs = {}
    for method in METHOD_KEYS:
        if method not in selected:
            continue
        torch.manual_seed(7)
        source_method = "ra_runet" if method == "runet_no_alignment" else method
        config_path = ROOT / "configs" / f"{config_name[method]}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = build_model(source_method, config["model"]).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for _ in range(steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(
                model(batch), target_batch, bce_weight=1.0, dice_weight=1.0
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_probabilities = []
        val_targets = []
        for row in validation_rows:
            image = data[str(row["image_id"])][1].astype(np.float32) / 255.0
            val_probabilities.append(
                predict_pixel_probability(
                    model,
                    (image - mean) / std,
                    torch.device("cpu"),
                    pad_multiple=32,
                    tile_size=0,
                    tile_overlap=0,
                )
            )
            val_targets.append(data[str(row["image_id"])][2])
        threshold, scores = _threshold(val_probabilities, val_targets)
        predictions = {}
        for row in display_rows:
            image = data[str(row["image_id"])][1].astype(np.float32) / 255.0
            probability = predict_pixel_probability(
                model,
                (image - mean) / std,
                torch.device("cpu"),
                pad_multiple=32,
                tile_size=0,
                tile_overlap=0,
            )
            predictions[str(row["image_id"])] = probability >= threshold
        outputs[method] = (
            predictions,
            {
                "implementation_sha256": _source_hashes(
                    "segbench/deep_models.py", "segbench/deep_utils.py", config_path.relative_to(ROOT).as_posix()
                ),
                "torch_version": _version("torch"),
                "seed": 7,
                "synthetic_micro_training_steps": steps,
                "synthetic_micro_crop": [64, 64],
                "loss": "BCE+Dice",
                "train_only_normalization": {"mean": mean, "std": std},
                "selected_threshold": threshold,
                "validation_threshold_scores": scores,
                "parameter_count": parameter_counts(model)["total"],
                "final_training_loss": losses[-1],
            },
        )
    return outputs


def _balanced_patch_batch(
    rows: Sequence[Mapping[str, Any]],
    data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    patch_size: int,
    mean: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray]:
    from segbench.deep_utils import _extract_all_patches

    all_patches = []
    all_targets = []
    rng = np.random.default_rng(7)
    for row in rows:
        image = data[str(row["image_id"])][1].astype(np.float32) / 255.0
        target = data[str(row["image_id"])][2]
        patches, height, width = _extract_all_patches(image, patch_size, 16)
        labels = block_features.block_labels(target, height, width).reshape(-1)
        chosen = []
        for value in (False, True):
            candidates = np.flatnonzero(labels == value)
            if len(candidates):
                chosen.extend(rng.choice(candidates, size=min(16, len(candidates)), replace=False))
        all_patches.append((patches[np.asarray(chosen)] - mean) / std)
        all_targets.append(labels[np.asarray(chosen)].astype(np.float32))
    return np.concatenate(all_patches), np.concatenate(all_targets)


def _patch_methods(
    methods: set[str],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    display_rows: Sequence[Mapping[str, Any]],
    data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    steps: int,
    threads: int,
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    selected = methods & {"lenet", "alexnet", "cnn_ae"}
    if not selected:
        return {}
    torch = _torch_runtime(threads)
    from segbench.deep_models import ConvAutoencoder, build_model, parameter_counts
    from segbench.deep_utils import blocks_to_postprocessed_mask, predict_patch_probability

    mean, std = _normalization(train_rows, data)
    patch_sizes = {"lenet": 32, "alexnet": 64, "cnn_ae": 64}
    outputs = {}
    for method in METHOD_KEYS:
        if method not in selected:
            continue
        torch.manual_seed(7)
        config_path = ROOT / "configs" / f"{method}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        patches, labels = _balanced_patch_batch(
            train_rows, data, patch_size=patch_sizes[method], mean=mean, std=std
        )
        batch = torch.from_numpy(patches[:, None].astype(np.float32))
        targets = torch.from_numpy(labels.astype(np.float32))
        model = build_model(method, config["model"]).cpu()
        pretrain_losses: list[float] = []
        if method == "cnn_ae":
            autoencoder = ConvAutoencoder(
                in_channels=1, latent_channels=int(config["model"]["latent_channels"])
            ).cpu()
            pretrain_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(steps):
                pretrain_optimizer.zero_grad(set_to_none=True)
                reconstruction = autoencoder(batch)
                pretrain_loss = torch.nn.functional.mse_loss(reconstruction, batch)
                pretrain_loss.backward()
                pretrain_optimizer.step()
                pretrain_losses.append(float(pretrain_loss.detach()))
            model.encoder.load_state_dict(autoencoder.encoder.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for _ in range(steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        def probability(row: Mapping[str, Any]) -> np.ndarray:
            image = data[str(row["image_id"])][1].astype(np.float32) / 255.0
            return predict_patch_probability(
                model,
                (image - mean) / std,
                torch.device("cpu"),
                patch_size=patch_sizes[method],
                block_size=16,
                batch_size=128,
            )

        val_probabilities = [probability(row) for row in validation_rows]
        val_targets = [data[str(row["image_id"])][2] for row in validation_rows]
        scores: dict[str, float] = {}
        for threshold in THRESHOLDS:
            values = []
            for row, block_probability, target in zip(
                validation_rows, val_probabilities, val_targets
            ):
                prediction = blocks_to_postprocessed_mask(
                    block_probability,
                    target.shape,
                    threshold=threshold,
                    min_component_blocks=4,
                )
                values.append(float(image_metrics(prediction, target)["dice"]))
            scores[f"{threshold:.1f}"] = float(np.mean(values))
        threshold = max(
            THRESHOLDS,
            key=lambda value: (scores[f"{value:.1f}"], -abs(value - 0.5), -value),
        )
        predictions = {}
        for row in display_rows:
            target_shape = data[str(row["image_id"])][2].shape
            predictions[str(row["image_id"])] = blocks_to_postprocessed_mask(
                probability(row),
                target_shape,
                threshold=float(threshold),
                min_component_blocks=4,
            )
        outputs[method] = (
            predictions,
            {
                "implementation_sha256": _source_hashes(
                    "segbench/deep_models.py", "segbench/deep_utils.py", config_path.relative_to(ROOT).as_posix()
                ),
                "torch_version": _version("torch"),
                "seed": 7,
                "synthetic_micro_training_steps": steps,
                "synthetic_autoencoder_pretrain_steps": steps if method == "cnn_ae" else 0,
                "patch_size": patch_sizes[method],
                "block_size": 16,
                "train_only_normalization": {"mean": mean, "std": std},
                "selected_threshold": float(threshold),
                "validation_threshold_scores": scores,
                "parameter_count": parameter_counts(model)["total"],
                "final_training_loss": losses[-1],
                "final_pretraining_loss": pretrain_losses[-1] if pretrain_losses else None,
            },
        )
    return outputs


def _sufs_finetuned(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    display_rows: Sequence[Mapping[str, Any]],
    data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    steps: int,
    threads: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    torch = _torch_runtime(threads)
    from segbench.sufs_finetune import (
        build_released_sufs,
        predict_sufs_probability,
        sufs_probability_loss,
    )

    config_path = ROOT / "configs" / "sufs_ft.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    segmenter, model, released = build_released_sufs(config, torch.device("cpu"))
    mean, std = _normalization(train_rows, data)
    images = []
    targets = []
    for row in train_rows:
        image = data[str(row["image_id"])][1].astype(np.float32) / 255.0
        target = data[str(row["image_id"])][2].astype(np.float32)
        images.append(cv2.resize((image - mean) / std, (128, 128), interpolation=cv2.INTER_AREA))
        targets.append(cv2.resize(target, (128, 128), interpolation=cv2.INTER_NEAREST))
    batch = torch.from_numpy(np.stack(images)[..., None].astype(np.float32))
    target_batch = torch.from_numpy(np.stack(targets)[..., None].astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        probability = model(batch, training=True)
        loss = sufs_probability_loss(
            probability, target_batch, bce_weight=1.0, dice_weight=1.0
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    def predict(row: Mapping[str, Any]) -> np.ndarray:
        image = data[str(row["image_id"])][1].astype(np.float32) / 255.0
        return predict_sufs_probability(
            segmenter, (image - mean) / std, torch.device("cpu")
        )

    val_probabilities = [predict(row) for row in validation_rows]
    threshold, scores = _threshold(
        val_probabilities, [data[str(row["image_id"])][2] for row in validation_rows]
    )
    predictions = {
        str(row["image_id"]): predict(row) >= threshold for row in display_rows
    }
    return predictions, {
        "implementation_sha256": _source_hashes(
            "segbench/pyfing_compat.py",
            "segbench/sufs_finetune.py",
            "scripts/train_sufs.py",
            "configs/sufs_ft.yaml",
        ),
        "torch_version": _version("torch"),
        "keras_version": _version("keras"),
        "package": released["package"],
        "package_version": released["version"],
        "released_weights_sha256": released["weights_sha256"],
        "seed": 7,
        "synthetic_micro_training_steps": steps,
        "synthetic_micro_crop": [128, 128],
        "train_only_normalization": {"mean": mean, "std": std},
        "selected_threshold": threshold,
        "validation_threshold_scores": scores,
        "parameter_count": released["parameter_count"],
        "final_training_loss": losses[-1],
    }


def _atomic_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial.png")
    if not cv2.imwrite(str(temporary), np.where(mask, 255, 0).astype(np.uint8)):
        raise OSError(f"could not write {path.name}")
    os.replace(temporary, path)


def _record(
    method: str,
    status: str,
    *,
    reason: str = "",
    execution: Mapping[str, Any] | None = None,
    prediction_set: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "method": method,
        "label": LABELS[method],
        "status": status,
        "reason": reason,
        "prediction_set_sha256": prediction_set,
        "execution": dict(execution) if execution is not None else None,
    }
    record["fingerprint_sha256"] = method_record_fingerprint(record)
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="all, or one or more stable method keys",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.steps < 1 or args.threads < 1:
        raise ValueError("steps and threads must be positive")
    requested = set(METHOD_KEYS if args.methods == ["all"] else args.methods)
    unknown = requested - set(METHOD_KEYS)
    if unknown:
        raise ValueError(f"unknown method keys: {sorted(unknown)}")
    root = args.synthetic_root.resolve()
    receipt = _dataset_receipt(root)
    frame, data = _load_data(root)
    train_subjects, validation_subjects, held_out_subjects = _split_subjects(frame)
    train_rows = _rows_for(frame, train_subjects)
    validation_rows = _rows_for(frame, validation_subjects)
    display_rows = frame.head(3).to_dict("records")
    display_stems = [str(row["image_id"]) for row in display_rows]

    prediction_root = root / "predictions"
    if prediction_root.exists():
        shutil.rmtree(prediction_root)
    completed: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    completed.update(_classical_methods(requested, train_rows, display_rows, data))

    pyfing_error: str | None = None
    if requested & {"gmfs_default", "gmfs_tuned", "sufs_zero", "sufs_ft"}:
        try:
            completed.update(
                _gmfs_methods(requested, validation_rows, display_rows, data)
            )
            if "sufs_zero" in requested:
                completed["sufs_zero"] = _sufs_zero(
                    validation_rows, display_rows, data
                )
            if "sufs_ft" in requested:
                completed["sufs_ft"] = _sufs_finetuned(
                    train_rows,
                    validation_rows,
                    display_rows,
                    data,
                    steps=args.steps,
                    threads=args.threads,
                )
        except (ImportError, ModuleNotFoundError, FileNotFoundError, RuntimeError) as exc:
            pyfing_error = type(exc).__name__
            for method in {"gmfs_default", "gmfs_tuned", "sufs_zero", "sufs_ft"}:
                completed.pop(method, None)

    completed.update(
        _pixel_methods(
            requested,
            train_rows,
            validation_rows,
            display_rows,
            data,
            steps=args.steps,
            threads=args.threads,
        )
    )
    completed.update(
        _patch_methods(
            requested,
            train_rows,
            validation_rows,
            display_rows,
            data,
            steps=args.steps,
            threads=args.threads,
        )
    )

    prediction_assets: list[dict[str, Any]] = []
    method_records: list[dict[str, Any]] = []
    for method in METHOD_KEYS:
        if method in completed:
            predictions, execution = completed[method]
            assets = []
            for row in display_rows:
                source_stem = str(row["image_id"])
                prediction = np.asarray(predictions[source_stem]).astype(bool)
                source_shape = data[source_stem][0].shape
                rendered = cv2.resize(
                    prediction.astype(np.uint8),
                    (source_shape[1], source_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                relative = Path("predictions") / method / f"{source_stem}.png"
                path = root / relative
                _atomic_png(path, rendered)
                asset = {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "mime_type": "image/png",
                    "role": "method_prediction",
                    "method": method,
                    "label": LABELS[method],
                    "source_stem": source_stem,
                    "inference_ppi": 500,
                    "display_nearest_resized_to_source_grid": True,
                }
                assets.append(asset)
                prediction_assets.append(asset)
            method_records.append(
                _record(
                    method,
                    "complete",
                    execution=execution,
                    prediction_set=prediction_set_sha256(assets),
                )
            )
        elif method in EXTERNAL_UNAVAILABLE:
            method_records.append(
                _record(method, "unavailable", reason=EXTERNAL_UNAVAILABLE[method])
            )
        elif method in requested and method.startswith(("gmfs", "sufs")) and pyfing_error:
            method_records.append(
                _record(
                    method,
                    "unavailable",
                    reason=(
                        "The optional PyFing runtime or its released weights were "
                        f"unavailable in this artifact-build environment ({pyfing_error})."
                    ),
                )
            )
        elif method in requested:
            raise RuntimeError(f"requested synthetic method did not complete: {method}")
        else:
            method_records.append(
                _record(
                    method,
                    "not_built",
                    reason="This method was not requested by the current artifact-build command.",
                )
            )

    receipt["assets"].extend(prediction_assets)
    demo = {
        "schema_version": 1,
        "kind": "segbench_synthetic_method_demo",
        "benchmark_evidence": False,
        "device": "CPU",
        "seed": 7,
        "synthetic_micro_training_steps": args.steps,
        "threshold_candidates": list(THRESHOLDS),
        "manifest_sha256": sha256_file(root / "manifest.csv"),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "held_out_subjects": held_out_subjects,
        "display_source_stems": display_stems,
        "methods": method_records,
        "method_record_set_sha256": canonical_sha256(method_records),
        "notice": (
            "Synthetic-only micro-runs demonstrate code paths and are not benchmark "
            "measurements, model comparisons, or evidence about biometric performance."
        ),
    }
    receipt["synthetic_demo"] = demo
    target = root / "synthetic_receipt.json"
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    validate_synthetic_demo_receipt(root, receipt, verify_files=True, verify_builder=True)
    print(
        json.dumps(
            {
                "status": "PASS",
                "synthetic_only": True,
                "benchmark_evidence": False,
                "complete": sum(row["status"] == "complete" for row in method_records),
                "unavailable": sum(row["status"] == "unavailable" for row in method_records),
                "not_built": sum(row["status"] == "not_built" for row in method_records),
                "predictions": len(prediction_assets),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
