#!/usr/bin/env python3
"""Generate 500-ppi binary masks from a frozen neural checkpoint.

This script never computes benchmark metrics.  It writes masks only; final
scores must be produced by ``evaluate_masks.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once
from segbench.deep_models import build_model
from segbench.deep_utils import (
    PROTOCOL_THRESHOLDS,
    atomic_json,
    blocks_to_postprocessed_mask,
    configure_memory_fraction,
    examiner_exclusions_for_split,
    load_config,
    load_split_rows,
    predict_patch_probability,
    predict_pixel_probability,
    read_normalized_image,
    resume_config_digest,
    resolve_device,
    run_provenance,
    seed_everything,
    verify_source_image,
    file_digest,
)
from segbench import probability_artifacts
from segbench import preprocess_cache


SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PREEMPT_EXIT = 75


def preempt_requested() -> bool:
    """Honor the shared queue's cooperative stop request between images."""
    value = os.environ.get("SEG_PREEMPT_FILE")
    return bool(value and Path(value).exists())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--probability-out",
        type=Path,
        help="float16 .npy probability maps (defaults beside --out)",
    )
    parser.add_argument(
        "--threshold-receipt",
        type=Path,
        help="immutable validation-only threshold receipt",
    )
    parser.add_argument(
        "--threshold-amendment",
        type=Path,
        help="exact operator-authorized threshold-policy amendment",
    )
    parser.add_argument("--allow-unamended-debug", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--timing-out", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument(
        "--seed",
        type=int,
        help="expected training seed; defaults to the seed recorded in the checkpoint",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-valid", action="store_true")
    parser.add_argument(
        "--mc-dropout-samples",
        type=int,
        default=0,
        help=(
            "DRUnet only: stochastic forward passes used for the private "
            "MC-dropout variance-map extra (0 disables it)"
        ),
    )
    parser.add_argument(
        "--mc-variance-out",
        type=Path,
        help="private output directory for float32 .npy variance maps",
    )
    parser.add_argument(
        "--mc-limit",
        type=int,
        default=20,
        help="number of ordered test images receiving variance maps (protocol default: 20)",
    )
    return parser.parse_args(argv)


def _safe_output(out: Path, image_id: str) -> Path:
    if not SAFE_IMAGE_ID.fullmatch(image_id) or image_id in {".", ".."}:
        raise ValueError(f"unsafe image_id for output path: {image_id!r}")
    destination = out / f"{image_id}.png"
    if destination.parent.resolve() != out.resolve():
        raise ValueError(f"image_id escapes output directory: {image_id!r}")
    return destination


def _safe_variance_output(out: Path, image_id: str) -> Path:
    """Return a non-escaping path for one private MC-dropout artifact."""
    if not SAFE_IMAGE_ID.fullmatch(image_id) or image_id in {".", ".."}:
        raise ValueError(f"unsafe image_id for output path: {image_id!r}")
    destination = out / f"{image_id}_variance.npy"
    if destination.parent.resolve() != out.resolve():
        raise ValueError(f"image_id escapes output directory: {image_id!r}")
    return destination


def _atomic_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".png", mask.astype(np.uint8))
    if not encoded_ok:
        raise RuntimeError(f"failed to encode prediction: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".png")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    """Atomically write one NumPy array so resumable runs never see partial maps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _valid_existing(path: Path, shape: Tuple[int, int]) -> bool:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2 or image.shape != shape or image.dtype != np.uint8:
        return False
    return set(np.unique(image).tolist()).issubset({0, 255})


def _valid_existing_variance(path: Path, shape: Tuple[int, int]) -> bool:
    """Validate a resumable float32 variance map without trusting pickle data."""
    try:
        variance = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return False
    return (
        variance.shape == shape
        and variance.dtype == np.float32
        and bool(np.isfinite(variance).all())
        and bool((variance >= 0.0).all())
    )


def _predict(
    model: torch.nn.Module,
    image: np.ndarray,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    normalized = (image - float(normalization["mean"])) / float(normalization["std"])
    acceleration = config.get("runtime", {}).get(
        "acceleration", {"precision": "fp32", "channels_last": False}
    )
    if config["task"] == "pixel":
        inference = config["inference"]
        probability = predict_pixel_probability(
            model,
            normalized,
            device,
            pad_multiple=int(inference["pad_multiple"]),
            tile_size=int(inference["tile_size"]),
            tile_overlap=int(inference["tile_overlap"]),
            autocast_bf16=acceleration.get("precision") == "bf16",
            channels_last=bool(acceleration.get("channels_last", False)),
        )
        probability = probability_artifacts.storage_roundtrip(probability)
        return probability, _mask_from_probability(
            probability, image.shape, config, threshold
        )
    if config["task"] == "patch":
        data_config = config["data"]
        inference = config["inference"]
        block_probability = predict_patch_probability(
            model,
            normalized,
            device,
            patch_size=int(data_config["patch_size"]),
            block_size=int(data_config["block_size"]),
            batch_size=int(inference["patch_batch_size"]),
            autocast_bf16=acceleration.get("precision") == "bf16",
            channels_last=bool(acceleration.get("channels_last", False)),
        )
        block_probability = probability_artifacts.storage_roundtrip(block_probability)
        prediction = _mask_from_probability(
            block_probability, image.shape, config, threshold
        )
        return block_probability, prediction
    raise ValueError(f"unknown checkpoint task: {config['task']!r}")


def _mask_from_probability(
    probability: np.ndarray,
    image_shape: Tuple[int, int],
    config: Mapping[str, Any],
    threshold: float,
) -> np.ndarray:
    """Apply the frozen threshold/postprocess path to retained probability bytes."""
    if config["task"] == "pixel":
        if probability.shape != image_shape:
            raise ValueError("pixel probability shape differs from output image")
        return probability >= threshold
    if config["task"] == "patch":
        return blocks_to_postprocessed_mask(
            probability,
            image_shape,
            threshold=threshold,
            min_component_blocks=int(config["inference"]["min_component_blocks"]),
        )
    raise ValueError(f"unknown checkpoint task: {config['task']!r}")


def _verify_retained_probability_masks(
    records: Sequence[Mapping[str, Any]],
    *,
    mask_dir: Path,
    probability_dir: Path,
    config: Mapping[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """Byte-check every mask against a fresh derivation from its stored map."""
    evidence = []
    for row in records:
        image_id = str(row["image_id"])
        mask_path = _safe_output(mask_dir, image_id)
        saved = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if saved is None or saved.ndim != 2:
            raise RuntimeError(f"{image_id}: final binary mask is unreadable")
        probability_path = probability_artifacts.safe_path(
            probability_dir, image_id
        )
        probability = probability_artifacts.read(
            probability_path,
            _probability_shape(config["task"], saved.shape, config),
        )
        derived = _mask_from_probability(
            probability, saved.shape, config, threshold
        ).astype(np.uint8) * 255
        if not np.array_equal(derived, saved):
            raise RuntimeError(
                f"{image_id}: retained probability does not reproduce final mask"
            )
        evidence.append(
            {
                "image_id": image_id,
                "mask_sha256": file_digest(mask_path),
                "probability_sha256": file_digest(probability_path),
            }
        )
    canonical = json.dumps(
        evidence, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "complete": len(evidence) == len(records),
        "count": len(evidence),
        "records_sha256": hashlib.sha256(canonical).hexdigest(),
        "all_masks_rederived_from_retained_probabilities": True,
    }


def _probability_shape(
    task: str, image_shape: Tuple[int, int], config: Mapping[str, Any]
) -> Tuple[int, int]:
    if task == "pixel":
        return image_shape
    if task == "patch":
        from segbench import features as block_features

        return block_features.grid_shape(*image_shape)
    raise ValueError(f"unknown checkpoint task: {task!r}")


def _threshold_from_receipt(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> Tuple[float, Optional[str]]:
    amendment_path = getattr(args, "threshold_amendment", None)
    amendment_sha256 = (
        probability_artifacts.validate_threshold_amendment(amendment_path)
        if amendment_path is not None
        else None
    )
    candidates = {
        float(item)
        for item in (
            PROTOCOL_THRESHOLDS
            if amendment_sha256 is not None
            else config["training"]["threshold_candidates"]
        )
    }
    if args.threshold_receipt is None:
        if args.split == "test":
            raise ValueError(
                "test inference requires an immutable validation threshold receipt"
            )
        threshold = float(checkpoint["selected_threshold"])
        if threshold not in candidates:
            raise ValueError(
                "checkpoint threshold was not selected from the declared validation grid"
            )
        return threshold, None
    receipt = json.loads(args.threshold_receipt.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "kind": "expanded_validation_threshold_selector_receipt",
        "method": config["method"],
        "task": config["task"],
        "selection_split": "val",
        "checkpoint_sha256": file_digest(args.checkpoint),
        "source_config_sha256": file_digest(args.config),
        "manifest_sha256": file_digest(args.manifest),
        "split_sha256": file_digest(args.splits),
        "threshold_amendment_sha256": amendment_sha256,
        "amendment_sha256": amendment_sha256,
        "seed": int(config["seed"]),
        "selection_source": "validation_only",
        "validation_only": True,
        "test_or_heldout_opened": False,
        "probabilities_are_pre_threshold": True,
        "storage_dtype": probability_artifacts.STORAGE_DTYPE,
        "derived_from_binary_mask": False,
    }
    disagreements = {
        key: (receipt.get(key), value)
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if disagreements:
        raise ValueError(f"threshold receipt provenance mismatch: {disagreements}")
    validation_card_sha256 = receipt.get("validation_probability_card_sha256")
    if not isinstance(validation_card_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", validation_card_sha256
    ):
        raise ValueError("threshold receipt lacks a validation probability-card identity")
    if [float(value) for value in receipt.get("threshold_candidates", [])] != sorted(
        candidates
    ):
        raise ValueError("threshold receipt candidate grid differs from config")
    if [float(value) for value in receipt.get("allowed_grid", [])] != sorted(candidates):
        raise ValueError("threshold receipt allowed grid differs from config")
    scores = receipt.get("threshold_scores")
    expected_score_keys = {f"{value:.8g}" for value in candidates}
    if not isinstance(scores, Mapping) or set(scores) != expected_score_keys:
        raise ValueError("threshold receipt lacks the complete validation score grid")
    numeric_scores = {float(key): float(value) for key, value in scores.items()}
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in numeric_scores.values()):
        raise ValueError("threshold receipt contains an invalid validation score")
    threshold = float(receipt["selected_threshold"])
    if threshold not in candidates:
        raise ValueError("threshold receipt selected a value outside the grid")
    expected_threshold, _ = max(
        numeric_scores.items(),
        key=lambda item: (item[1], -abs(item[0] - 0.5), -item[0]),
    )
    if threshold != expected_threshold:
        raise ValueError("threshold receipt selection disagrees with its score grid")
    if args.split == "test" and not args.limit and receipt.get("debug_limited") is not False:
        raise ValueError("full test inference requires a full-validation threshold receipt")
    return threshold, file_digest(args.threshold_receipt)


def _mc_dropout_variance(
    model: torch.nn.Module,
    image: np.ndarray,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    device: torch.device,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Estimate predictive variance while keeping normalisation layers frozen."""
    if samples < 2:
        raise ValueError("MC-dropout variance requires at least two samples")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    normalized = (image - float(normalization["mean"])) / float(normalization["std"])
    inference = config["inference"]
    predictions = [
        predict_pixel_probability(
            model,
            normalized,
            device,
            pad_multiple=int(inference["pad_multiple"]),
            tile_size=int(inference["tile_size"]),
            tile_overlap=int(inference["tile_overlap"]),
            enable_dropout=True,
        )
        for _ in range(samples)
    ]
    variance = np.var(np.stack(predictions, axis=0), axis=0, ddof=0)
    return variance.astype(np.float32, copy=False)


def infer(
    args: argparse.Namespace,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one checkpoint over a declared manifest split."""
    start_provenance = dict(
        provenance
        if provenance is not None
        else run_provenance(sys.argv)
    )
    threshold_amendment = getattr(args, "threshold_amendment", None)
    threshold_amendment_sha256 = (
        probability_artifacts.validate_threshold_amendment(threshold_amendment)
        if threshold_amendment is not None
        else None
    )
    config = load_config(
        args.config,
        allow_amendment_legacy_thresholds=threshold_amendment_sha256 is not None,
    )
    debug_device = getattr(args, "device", None) or str(
        config["runtime"].get("device", "auto")
    )
    if threshold_amendment_sha256 is None and not (
        getattr(args, "allow_unamended_debug", False) and debug_device == "cpu"
        and int(args.limit) > 0
    ):
        raise ValueError("production inference requires the exact threshold amendment")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint lacks its resolved training configuration")
    checkpoint_seed = int(checkpoint_config.get("seed", config.get("seed", 0)))
    config = dict(config)
    config["seed"] = int(args.seed if args.seed is not None else checkpoint_seed)
    config["runtime"] = dict(config["runtime"])
    if "training_data_identity" in checkpoint_config:
        config["training_data_identity"] = checkpoint_config["training_data_identity"]
    checkpoint_runtime = checkpoint_config.get("runtime", {})
    for key in (
        "acceleration",
        "queue_batch_contract",
        "queue_batch_contract_sha256",
    ):
        if key in checkpoint_runtime:
            config["runtime"][key] = checkpoint_runtime[key]
    if checkpoint.get("resume_config_sha256") != resume_config_digest(config):
        raise ValueError("checkpoint and supplied scientific configuration differ")
    if checkpoint.get("method") != config["method"] or checkpoint.get("task") != config["task"]:
        raise ValueError("checkpoint model identity differs from supplied configuration")
    if args.mc_dropout_samples < 0:
        raise ValueError("mc-dropout-samples must be non-negative")
    mc_enabled = args.mc_dropout_samples > 0
    if mc_enabled:
        if config["method"] != "drunet" or config["task"] != "pixel":
            raise ValueError("MC-dropout variance export is available only for DRUnet")
        if args.mc_dropout_samples < 2:
            raise ValueError("MC-dropout variance requires at least two samples")
        if args.mc_variance_out is None:
            raise ValueError("--mc-variance-out is required with --mc-dropout-samples")
        if args.mc_limit <= 0:
            raise ValueError("mc-limit must be positive")
    elif args.mc_variance_out is not None:
        raise ValueError("--mc-variance-out requires --mc-dropout-samples")
    config["runtime"] = dict(config["runtime"])
    if args.device is not None:
        config["runtime"]["device"] = args.device
    memory_override = (
        args.memory_fraction
        if args.memory_fraction is not None
        else os.environ.get("SEG_GPU_MEMORY_FRACTION")
    )
    if memory_override is not None:
        config["runtime"]["memory_fraction"] = float(memory_override)
    seed_everything(
        int(config.get("seed", 0)), bool(config["runtime"].get("deterministic", True))
    )
    device = resolve_device(str(config["runtime"].get("device", "auto")))
    configure_memory_fraction(device, float(config["runtime"]["memory_fraction"]))
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("source_split") != "train":
        raise ValueError("checkpoint lacks training-only normalization metadata")
    threshold, threshold_receipt_sha256 = _threshold_from_receipt(
        args, config, checkpoint
    )
    model = build_model(str(config["method"]), config["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    acceleration = config.get("runtime", {}).get(
        "acceleration", {"precision": "fp32", "channels_last": False}
    )
    if acceleration.get("channels_last"):
        model.to(memory_format=torch.channels_last)
    data_config = config["data"]
    rows = load_split_rows(
        args.manifest,
        args.splits,
        subset=args.subset,
        split=args.split,
        mask_column=str(data_config["mask_column"]),
        require_mask=False,
        exclude_examiner_values=examiner_exclusions_for_split(data_config, args.split),
        limit=args.limit,
    )
    records = rows.to_dict("records")
    cache_receipt_sha256 = None
    if args.cache_dir is not None:
        preprocess_cache.load_receipt(
            args.cache_dir,
            manifest_sha256=file_digest(args.manifest),
            target_ppi=float(data_config["target_ppi"]),
            mask_columns=[str(data_config["mask_column"])],
            required_ids=[str(row["image_id"]) for row in records],
            verify_artifact_hashes=True,
        )
        cache_receipt_sha256 = file_digest(args.cache_dir / "receipt.json")
    # Cached arrays accelerate training/validation and non-timed exports.  Held-out
    # paper timing always retains decode + resample semantics from the frozen protocol.
    inference_cache_dir = args.cache_dir if args.split != "test" else None
    if args.probability_out is None:
        args.probability_out = args.out.parent / f"{args.out.name}_probabilities"
    args.probability_out.mkdir(parents=True, exist_ok=True)
    # Authenticate every frozen source exactly once before warm-up or timing.
    # Hashing is provenance validation, not model preprocessing, and therefore
    # must never inflate the reported per-image runtime.
    for row in sorted(records, key=lambda item: str(item["image_id"])):
        verify_source_image(row)
    source_hashes = probability_artifacts.source_hash_map(records)
    run_identity = {
        "schema_version": 1,
        "method": config["method"],
        "task": config["task"],
        "seed": int(config["seed"]),
        "subset": args.subset,
        "split": args.split,
        "limit": int(args.limit),
        "selected_threshold": threshold,
        "threshold_receipt_sha256": threshold_receipt_sha256,
        "selector_receipt_sha256": threshold_receipt_sha256,
        "threshold_amendment_sha256": threshold_amendment_sha256,
        "source_config_sha256": file_digest(args.config),
        "checkpoint_sha256": file_digest(args.checkpoint),
        "manifest_sha256": file_digest(args.manifest),
        "split_sha256": file_digest(args.splits),
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
        "mc_dropout_samples": int(args.mc_dropout_samples),
        "mc_limit": int(args.mc_limit),
    }
    progress_path = args.timing_out.with_name(f"{args.timing_out.stem}.progress.json")
    if args.timing_out.is_file() and not args.skip_valid:
        raise ValueError(
            "immutable complete inference card exists; use --skip-valid to authenticate it"
        )
    prior_timing_records: Dict[str, Dict[str, Any]] = {}
    previous_complete: Optional[Dict[str, Any]] = None
    resume_path = (
        args.timing_out
        if args.timing_out.is_file()
        else progress_path
    )
    if args.skip_valid and resume_path.is_file():
        try:
            previous_timing = json.loads(resume_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing timing card is invalid") from exc
        if previous_timing.get("run_identity") != run_identity:
            raise ValueError("existing timing card belongs to different inference inputs")
        prior_timing_records = probability_artifacts.authenticated_resume_records(
            previous_timing.get("per_image", []),
            allowed_ids=[str(row["image_id"]) for row in records],
            timing_key="ms_per_img",
        )
        if previous_timing.get("complete") is True:
            if set(prior_timing_records) != {
                str(row["image_id"]) for row in records
            }:
                raise ValueError("complete timing card has an incomplete image universe")
            previous_complete = dict(previous_timing)
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    for warmup_index in range(args.warmup):
        row = records[warmup_index % len(records)]
        image = read_normalized_image(
            row, float(data_config["target_ppi"]), cache_dir=inference_cache_dir
        )
        _predict(model, image, config, normalization, threshold, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timing_records: Dict[str, Dict[str, Any]] = {}
    errors = []
    skipped = 0
    written = 0
    mc_written = 0
    mc_skipped = 0
    preempted = False
    args.out.mkdir(parents=True, exist_ok=True)
    if mc_enabled:
        assert args.mc_variance_out is not None
        args.mc_variance_out.mkdir(parents=True, exist_ok=True)
    for record_index, row in enumerate(records):
        if preempt_requested():
            preempted = True
            break
        image_id = str(row["image_id"])
        try:
            path = _safe_output(args.out, image_id)
            probability_path = probability_artifacts.safe_path(
                args.probability_out, image_id
            )
            mc_selected = mc_enabled and record_index < args.mc_limit
            variance_path = (
                _safe_variance_output(args.mc_variance_out, image_id)
                if mc_selected and args.mc_variance_out is not None
                else None
            )
            if args.skip_valid and (
                path.exists()
                or probability_path.exists()
                or (variance_path and variance_path.exists())
            ):
                image = read_normalized_image(
                    row,
                    float(data_config["target_ppi"]),
                    cache_dir=inference_cache_dir,
                )
            else:
                image = None
            mask_is_valid = bool(
                image is not None and path.exists() and _valid_existing(path, image.shape)
            )
            probability_is_valid = False
            if image is not None and probability_path.exists():
                try:
                    probability_artifacts.read(
                        probability_path,
                        _probability_shape(config["task"], image.shape, config),
                    )
                    probability_is_valid = True
                except ValueError:
                    probability_is_valid = False
            prior_timing = prior_timing_records.get(image_id, {})
            authenticated_prior = bool(
                isinstance(prior_timing.get("ms_per_img"), (float, int))
                and np.isfinite(float(prior_timing["ms_per_img"]))
                and prior_timing.get("mask_sha256") == (
                    file_digest(path) if mask_is_valid else None
                )
                and prior_timing.get("probability_sha256") == (
                    file_digest(probability_path) if probability_is_valid else None
                )
            )
            variance_is_valid = bool(
                image is not None
                and variance_path is not None
                and variance_path.exists()
                and _valid_existing_variance(variance_path, image.shape)
            )
            if (
                mask_is_valid
                and probability_is_valid
                and authenticated_prior
                and (not mc_selected or variance_is_valid)
            ):
                skipped += 1
                timing_records[image_id] = prior_timing
                if mc_selected:
                    mc_skipped += 1
                continue
            if not (mask_is_valid and probability_is_valid and authenticated_prior):
                if previous_complete is not None:
                    raise ValueError(
                        f"{image_id}: immutable complete inference artifact changed"
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                # Always read inside the measured region, including when an
                # invalid resumable artifact was inspected above.  The timing
                # protocol includes image I/O and resolution preprocessing.
                image = read_normalized_image(
                    row,
                    float(data_config["target_ppi"]),
                    cache_dir=inference_cache_dir,
                )
                probability, prediction = _predict(
                    model, image, config, normalization, threshold, device
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if prediction.shape != image.shape:
                    raise RuntimeError(
                        f"prediction shape {prediction.shape} != expected {image.shape}"
                    )
                probability_artifacts.atomic_write(probability_path, probability)
                _atomic_png(path, prediction.astype(np.uint8) * 255)
                timing_records[image_id] = {
                    "image_id": image_id,
                    "ms_per_img": elapsed_ms,
                    "mask_sha256": file_digest(path),
                    "probability_sha256": file_digest(probability_path),
                }
                written += 1
            else:
                skipped += 1
                timing_records[image_id] = prior_timing
            if mc_selected:
                assert variance_path is not None
                if variance_is_valid:
                    mc_skipped += 1
                else:
                    if image is None:
                        image = read_normalized_image(
                            row,
                            float(data_config["target_ppi"]),
                            cache_dir=inference_cache_dir,
                        )
                    variance = _mc_dropout_variance(
                        model,
                        image,
                        config,
                        normalization,
                        device,
                        args.mc_dropout_samples,
                        (
                            int(config["seed"])
                            + int.from_bytes(
                                hashlib.sha256(image_id.encode("utf-8")).digest()[:8],
                                "big",
                            )
                        )
                        % (2**63 - 1),
                    )
                    if variance.shape != image.shape or not np.isfinite(variance).all():
                        raise RuntimeError("invalid MC-dropout variance map")
                    _atomic_npy(variance_path, variance)
                    mc_written += 1
        except Exception as exc:
            errors.append(
                {"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"}
            )
        if (record_index + 1) % 50 == 0 or errors:
            atomic_json(
                progress_path,
                {
                    "status": "running",
                    "run_identity": run_identity,
                    "per_image": [
                        timing_records[key] for key in sorted(timing_records)
                    ],
                    "errors": errors,
                },
            )
    # Re-authenticate after every decode but outside the measured interval.
    # Any source mutation during inference invalidates the run without charging
    # provenance hashing to the reported model latency.
    for row in sorted(records, key=lambda item: str(item["image_id"])):
        verify_source_image(row)
    probability_items = [
        (
            str(row["image_id"]),
            probability_artifacts.safe_path(args.probability_out, str(row["image_id"])),
        )
        for row in records
        if probability_artifacts.safe_path(
            args.probability_out, str(row["image_id"])
        ).is_file()
    ]
    probability_set = probability_artifacts.set_identity(
        probability_items, source_sha256=source_hashes
    )
    probability_set["complete"] = len(probability_items) == len(records)
    if not preempted and not errors and not probability_set["complete"]:
        raise RuntimeError("successful inference lacks a complete probability set")
    if not preempted and not errors and len(timing_records) != len(records):
        raise RuntimeError("successful inference lacks complete per-image timing records")
    mask_derivation_verification = None
    if not preempted and not errors:
        mask_derivation_verification = _verify_retained_probability_masks(
            records,
            mask_dir=args.out,
            probability_dir=args.probability_out,
            config=config,
            threshold=threshold,
        )
    report = {
        "schema_version": 1,
        "status": "preempted" if preempted else ("failed" if errors else "complete"),
        "complete": not preempted and not errors and probability_set["complete"],
        "run_identity": run_identity,
        "method": config["method"],
        "task": config["task"],
        "seed": int(config["seed"]),
        "subset": args.subset,
        "split": args.split,
        "kind": "complete_probability_map_card",
        "selected_validation_threshold": threshold,
        "threshold_receipt_sha256": threshold_receipt_sha256,
        "threshold_amendment_sha256": threshold_amendment_sha256,
        "threshold_candidates": sorted(
            float(value)
            for value in (
                PROTOCOL_THRESHOLDS
                if threshold_amendment_sha256 is not None
                else config["training"]["threshold_candidates"]
            )
        ),
        "probabilities_are_pre_threshold": True,
        "storage_format": probability_artifacts.FORMAT,
        "storage_dtype": probability_artifacts.STORAGE_DTYPE,
        "derived_from_binary_mask": False,
        "binary_mask_reinterpreted": False,
        "labels_opened": False,
        "selection_source": "unselected_raw",
        "threshold_selected_before_test": (
            args.split != "test" or threshold_receipt_sha256 is not None
        ),
        "target_ppi": float(data_config["target_ppi"]),
        "device": device.type.upper(),
        "warmup_images": args.warmup,
        "written": written,
        "skipped_valid": skipped,
        "errors": errors,
        "mean_ms_per_img": float(
            np.mean([item["ms_per_img"] for item in timing_records.values()])
        )
        if timing_records
        else float("nan"),
        "per_image": [timing_records[key] for key in sorted(timing_records)],
        "timing_resume_policy": "authenticated_per_image_records",
        "probability_set": probability_set,
        "mask_derivation_verification": mask_derivation_verification,
        "probability_maps_required": True,
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
        "cache_used_for_image_decode": inference_cache_dir is not None,
        "paper_timing_preprocess": (
            "on_the_fly_decode_and_resample" if args.split == "test" else "not_paper_timing"
        ),
        "source_config_sha256": file_digest(args.config),
        "checkpoint_sha256": file_digest(args.checkpoint),
        "manifest_sha256": file_digest(args.manifest),
        "split_sha256": file_digest(args.splits),
        "ordered_source_images_sha256": probability_set.get(
            "ordered_source_images_sha256"
        ),
        "identities": {
            "source_config_sha256": file_digest(args.config),
            "checkpoint_sha256": file_digest(args.checkpoint),
            "manifest_sha256": file_digest(args.manifest),
            "split_sha256": file_digest(args.splits),
            "selector_receipt_sha256": threshold_receipt_sha256,
            "amendment_sha256": threshold_amendment_sha256,
        },
        "debug_limited": bool(args.limit or args.allow_unamended_debug),
        "preempted": preempted,
        "mc_dropout": {
            "enabled": mc_enabled,
            "samples": int(args.mc_dropout_samples),
            "requested_images": int(min(args.mc_limit, len(records))) if mc_enabled else 0,
            "written": mc_written,
            "skipped_valid": mc_skipped,
            "output_format": "float32_npy" if mc_enabled else None,
            "deterministic_table_masks": True,
            "excluded_from_ms_per_img": True,
            "per_image_stable_seed": True,
        },
        "provenance": start_provenance,
    }
    if preempted or errors:
        atomic_json(progress_path, report)
    elif previous_complete is not None:
        previous_set = previous_complete.get("probability_set", {})
        if previous_set.get("sha256") != probability_set.get("sha256"):
            raise ValueError("immutable complete probability card artifact set changed")
        report = previous_complete
    else:
        atomic_json_once(args.timing_out, report)
    if preempted:
        raise SystemExit(PREEMPT_EXIT)
    if errors:
        raise RuntimeError(
            f"inference failed for {len(errors)} item(s); see {args.timing_out}"
        )
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    print(infer(parse_args(argv), provenance))


if __name__ == "__main__":
    main()
