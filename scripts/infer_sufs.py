#!/usr/bin/env python3
"""Generate 500-PPI masks from a frozen fine-tuned SUFS checkpoint.

The script never opens a ground-truth mask or computes a benchmark metric.
Final scores are produced only by ``evaluate_masks.py``.
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

if os.environ.get("KERAS_BACKEND", "torch").lower() != "torch":
    raise RuntimeError("infer_sufs.py requires KERAS_BACKEND=torch")
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import torch

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once
from segbench.deep_utils import (
    PROTOCOL_THRESHOLDS,
    atomic_json,
    configure_memory_fraction,
    examiner_exclusions_for_split,
    file_digest,
    load_config,
    load_split_rows,
    read_normalized_image,
    run_provenance,
    seed_everything,
)
from segbench import probability_artifacts
from segbench import preprocess_cache
from segbench.sufs_finetune import (
    build_released_sufs,
    predict_sufs_probability,
    prediction_set_digest,
    protocol_config_digest,
    sha256_file,
)


SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the mask-only inference interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--probability-out", type=Path)
    parser.add_argument("--threshold-receipt", type=Path)
    parser.add_argument("--threshold-amendment", type=Path)
    parser.add_argument("--allow-unamended-debug", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--run-card", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-valid", action="store_true")
    parser.add_argument(
        "--allow-sweep-checkpoint",
        action="store_true",
        help="debug only; final table masks must come from selected full runs",
    )
    return parser.parse_args(argv)


def _safe_output(root: Path, image_id: str) -> Path:
    """Return a non-escaping PNG path for one manifest image id."""
    if not SAFE_IMAGE_ID.fullmatch(image_id) or image_id in {".", ".."}:
        raise ValueError(f"unsafe image_id for output path: {image_id!r}")
    output = root / f"{image_id}.png"
    if output.parent.resolve() != root.resolve():
        raise ValueError(f"image_id escapes output directory: {image_id!r}")
    return output


def _atomic_png(path: Path, mask: np.ndarray) -> None:
    """Atomically save one two-level uint8 prediction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8))
    if not ok:
        raise RuntimeError(f"could not encode {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".png")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _valid_prediction(
    path: Path,
    shape: Tuple[int, int],
    expected_sha256: Optional[str],
) -> bool:
    """Require correct shape, binary pixels and (when known) exact contents."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        image is None
        or image.ndim != 2
        or image.dtype != np.uint8
        or image.shape != shape
        or not set(np.unique(image).tolist()).issubset({0, 255})
    ):
        return False
    return expected_sha256 is None or sha256_file(path) == expected_sha256


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["runtime"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("runtime device must be auto, cpu or cuda")
    return torch.device(requested)


def infer(
    args: argparse.Namespace,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run frozen SUFS inference without reading any target labels."""
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
    base_config = load_config(
        args.config,
        allow_amendment_legacy_thresholds=threshold_amendment_sha256 is not None,
    )
    debug_device = getattr(args, "device", None) or str(
        base_config["runtime"].get("device", "auto")
    )
    if threshold_amendment_sha256 is None and not (
        getattr(args, "allow_unamended_debug", False) and debug_device == "cpu"
        and int(args.limit) > 0
    ):
        raise ValueError("production SUFS inference requires the exact threshold amendment")
    if base_config["method"] != "sufs_ft" or base_config["task"] != "pixel":
        raise ValueError("infer_sufs.py accepts only the sufs_ft pixel config")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("SUFS checkpoint lacks its resolved configuration")
    if checkpoint.get("method") != "sufs_ft" or checkpoint.get("task") != "pixel":
        raise ValueError("checkpoint is not a fine-tuned SUFS pixel model")
    if checkpoint.get("mode") != "full" and not args.allow_sweep_checkpoint:
        raise ValueError("final masks require a validation-selected full SUFS run")
    if protocol_config_digest(checkpoint_config) != protocol_config_digest(base_config):
        raise ValueError("checkpoint and supplied SUFS protocol configurations differ")
    config = json.loads(json.dumps(checkpoint_config))
    if args.device is not None:
        config["runtime"]["device"] = args.device
    memory_override = (
        args.memory_fraction
        if args.memory_fraction is not None
        else os.environ.get("SEG_GPU_MEMORY_FRACTION")
    )
    if memory_override is not None:
        config["runtime"]["memory_fraction"] = float(memory_override)
    seed_everything(int(config["seed"]), bool(config["runtime"].get("deterministic", True)))
    device = _device(config)
    configure_memory_fraction(device, float(config["runtime"]["memory_fraction"]))
    segmenter, model, released = build_released_sufs(config, device)
    if checkpoint.get("released", {}).get("weights_sha256") != released["weights_sha256"]:
        raise ValueError("checkpoint was initialized from different released SUFS weights")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("source_split") != "train":
        raise ValueError("checkpoint lacks training-only normalization")
    threshold = float(checkpoint["selected_threshold"])
    historical_thresholds = {
        float(item) for item in checkpoint_config["training"]["threshold_candidates"]
    }
    if threshold not in historical_thresholds:
        raise ValueError("checkpoint threshold was not selected from the validation grid")
    declared_thresholds = (
        set(map(float, PROTOCOL_THRESHOLDS))
        if threshold_amendment_sha256 is not None
        else historical_thresholds
    )
    threshold_receipt = getattr(args, "threshold_receipt", None)
    cache_dir = getattr(args, "cache_dir", None)
    probability_out = getattr(args, "probability_out", None)
    threshold_receipt_sha256 = None
    if args.split == "test" and threshold_receipt is None:
        raise ValueError(
            "test inference requires an immutable validation threshold receipt"
        )
    if threshold_receipt is not None:
        receipt = json.loads(threshold_receipt.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "kind": "expanded_validation_threshold_selector_receipt",
            "method": "sufs_ft",
            "task": "pixel",
            "selection_split": "val",
            "checkpoint_sha256": file_digest(args.checkpoint),
            "source_config_sha256": file_digest(args.config),
            "manifest_sha256": file_digest(args.manifest),
            "split_sha256": file_digest(args.splits),
            "threshold_amendment_sha256": threshold_amendment_sha256,
            "amendment_sha256": threshold_amendment_sha256,
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
            raise ValueError(
                "threshold receipt lacks a validation probability-card identity"
            )
        if [float(value) for value in receipt.get("threshold_candidates", [])] != sorted(
            declared_thresholds
        ):
            raise ValueError("threshold receipt grid differs from SUFS config")
        if [float(value) for value in receipt.get("allowed_grid", [])] != sorted(
            declared_thresholds
        ):
            raise ValueError("threshold receipt allowed grid differs from SUFS config")
        scores = receipt.get("threshold_scores")
        expected_score_keys = {f"{value:.8g}" for value in declared_thresholds}
        if not isinstance(scores, Mapping) or set(scores) != expected_score_keys:
            raise ValueError("threshold receipt lacks the complete validation score grid")
        numeric_scores = {float(key): float(value) for key, value in scores.items()}
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in numeric_scores.values()
        ):
            raise ValueError("threshold receipt contains an invalid validation score")
        threshold = float(receipt["selected_threshold"])
        if threshold not in declared_thresholds:
            raise ValueError("threshold receipt chose a value outside the grid")
        expected_threshold, _ = max(
            numeric_scores.items(),
            key=lambda item: (item[1], -abs(item[0] - 0.5), -item[0]),
        )
        if threshold != expected_threshold:
            raise ValueError("threshold receipt selection disagrees with its score grid")
        if args.split == "test" and not args.limit and receipt.get("debug_limited") is not False:
            raise ValueError("full test inference requires a full-validation threshold receipt")
        threshold_receipt_sha256 = file_digest(threshold_receipt)
    data = config["data"]
    rows = load_split_rows(
        args.manifest,
        args.splits,
        subset=args.subset,
        split=args.split,
        mask_column=str(data["mask_column"]),
        require_mask=False,
        exclude_examiner_values=examiner_exclusions_for_split(data, args.split),
        limit=args.limit,
    )
    records = rows.to_dict("records")
    source_hashes = probability_artifacts.source_hash_map(records)
    cache_receipt_sha256 = None
    if cache_dir is not None:
        preprocess_cache.load_receipt(
            cache_dir,
            manifest_sha256=file_digest(args.manifest),
            target_ppi=float(data["target_ppi"]),
            mask_columns=[str(data["mask_column"])],
            required_ids=[str(row["image_id"]) for row in records],
            verify_artifact_hashes=True,
        )
        cache_receipt_sha256 = file_digest(cache_dir / "receipt.json")
    if probability_out is None:
        probability_out = args.out.parent / f"{args.out.name}_probabilities"
    probability_out.mkdir(parents=True, exist_ok=True)
    inference_cache_dir = cache_dir if args.split != "test" else None
    acceleration = checkpoint_config.get("runtime", {}).get(
        "acceleration", {"precision": "fp32", "channels_last": False}
    )
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    for index in range(args.warmup):
        row = records[index % len(records)]
        image = read_normalized_image(
            row, float(data["target_ppi"]), cache_dir=inference_cache_dir
        )
        normalized = (image - float(normalization["mean"])) / float(normalization["std"])
        predict_sufs_probability(
            segmenter,
            normalized,
            device,
            autocast_bf16=acceleration.get("precision") == "bf16",
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    identity = {
        "schema_version": 1,
        "method": "sufs_ft",
        "checkpoint_sha256": file_digest(args.checkpoint),
        "manifest_sha256": file_digest(args.manifest),
        "split_sha256": file_digest(args.splits),
        "subset": args.subset,
        "split": args.split,
        "target_ppi": float(data["target_ppi"]),
        "threshold": threshold,
        "threshold_receipt_sha256": threshold_receipt_sha256,
        "selector_receipt_sha256": threshold_receipt_sha256,
        "threshold_amendment_sha256": threshold_amendment_sha256,
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
    }
    progress_path = args.run_card.with_name(f"{args.run_card.stem}.progress.json")
    if args.run_card.is_file() and not args.skip_valid:
        raise ValueError(
            "immutable complete SUFS inference card exists; use --skip-valid to authenticate it"
        )
    previous: Dict[str, Any] = {}
    previous_complete: Optional[Dict[str, Any]] = None
    resume_path = args.run_card if args.run_card.is_file() else progress_path
    if resume_path.is_file():
        candidate = json.loads(resume_path.read_text(encoding="utf-8"))
        if candidate.get("identity") == identity:
            previous = candidate
            if candidate.get("complete") is True:
                previous_complete = dict(candidate)
        elif args.skip_valid:
            raise ValueError("existing SUFS run card belongs to different inputs")
    prior_records = probability_artifacts.authenticated_resume_records(
        previous.get("predictions", []),
        allowed_ids=[str(row["image_id"]) for row in records],
        timing_key="ms_per_img",
    )
    if previous_complete is not None and set(prior_records) != {
        str(row["image_id"]) for row in records
    }:
        raise ValueError("complete SUFS card has an incomplete image universe")
    predictions: Dict[str, Dict[str, Any]] = {}
    errors = []
    args.out.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(records):
        image_id = str(row["image_id"])
        try:
            path = _safe_output(args.out, image_id)
            probability_path = probability_artifacts.safe_path(
                probability_out, image_id
            )
            prior = prior_records.get(image_id, {})
            image = (
                read_normalized_image(
                    row, float(data["target_ppi"]), cache_dir=inference_cache_dir
                )
                if args.skip_valid and (path.is_file() or probability_path.is_file())
                else None
            )
            probability_valid = False
            if image is not None and probability_path.is_file():
                try:
                    probability_artifacts.read(probability_path, image.shape)
                    probability_valid = True
                except ValueError:
                    probability_valid = False
            valid = bool(
                args.skip_valid
                and image is not None
                and path.is_file()
                and _valid_prediction(path, image.shape, prior.get("sha256"))
                and probability_valid
                and sha256_file(probability_path) == prior.get("probability_sha256")
                and isinstance(prior.get("ms_per_img"), (float, int))
            )
            if valid:
                predictions[image_id] = prior
                continue
            if previous_complete is not None:
                raise ValueError(
                    f"{image_id}: immutable complete SUFS inference artifact changed"
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            # I/O, target-grid resampling, train-only normalization, inference,
            # cropping and thresholding are all inside the timing region.
            image = read_normalized_image(
                row, float(data["target_ppi"]), cache_dir=inference_cache_dir
            )
            normalized = (image - float(normalization["mean"])) / float(normalization["std"])
            probability = predict_sufs_probability(
                segmenter,
                normalized,
                device,
                autocast_bf16=acceleration.get("precision") == "bf16",
            )
            probability = probability_artifacts.storage_roundtrip(probability)
            prediction = probability >= threshold
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            probability_artifacts.atomic_write(probability_path, probability)
            _atomic_png(path, prediction.astype(np.uint8) * 255)
            predictions[image_id] = {
                "image_id": image_id,
                "ms_per_img": elapsed_ms,
                "sha256": sha256_file(path),
                "probability_sha256": sha256_file(probability_path),
            }
        except Exception as exc:  # keep later images resumable, then fail loudly
            errors.append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
        partial = {
            "status": "running",
            "identity": identity,
            "predictions": [predictions[key] for key in sorted(predictions)],
            "errors": errors,
        }
        if (index + 1) % 10 == 0 or errors:
            atomic_json(progress_path, partial)
    paths = [_safe_output(args.out, str(row["image_id"])) for row in records]
    probability_paths = [
        probability_artifacts.safe_path(probability_out, str(row["image_id"]))
        for row in records
    ]
    if (
        errors
        or len(predictions) != len(records)
        or not all(path.is_file() for path in paths)
        or not all(path.is_file() for path in probability_paths)
    ):
        failed = {
            "status": "failed",
            "identity": identity,
            "predictions": [predictions[key] for key in sorted(predictions)],
            "errors": errors,
        }
        atomic_json(progress_path, failed)
        raise RuntimeError(f"SUFS inference failed for {len(errors)} images")
    timings = [float(predictions[key]["ms_per_img"]) for key in sorted(predictions)]
    probability_set = probability_artifacts.set_identity(
        [
            (str(row["image_id"]), path)
            for row, path in zip(records, probability_paths)
        ],
        source_sha256=source_hashes,
    )
    probability_set["complete"] = len(probability_paths) == len(records)
    derivation_records = []
    for row, mask_path, probability_path in zip(
        records, paths, probability_paths
    ):
        image_id = str(row["image_id"])
        saved = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if saved is None or saved.ndim != 2:
            raise RuntimeError(f"{image_id}: final SUFS mask is unreadable")
        stored = probability_artifacts.read(probability_path, saved.shape)
        derived = (stored >= threshold).astype(np.uint8) * 255
        if not np.array_equal(derived, saved):
            raise RuntimeError(
                f"{image_id}: retained SUFS probability does not reproduce final mask"
            )
        derivation_records.append(
            {
                "image_id": image_id,
                "mask_sha256": sha256_file(mask_path),
                "probability_sha256": sha256_file(probability_path),
            }
        )
    derivation_digest = hashlib.sha256(
        json.dumps(
            derivation_records, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    record = {
        "schema_version": 1,
        "status": "complete",
        "complete": True,
        "kind": "complete_probability_map_card",
        "method": "sufs_ft",
        "task": "pixel",
        "seed": int(config["seed"]),
        "subset": args.subset,
        "split": args.split,
        "identity": identity,
        "display_name": config["display_name"],
        "framework": "pyfing-keras-torch",
        "released": released,
        "selected_validation_threshold": threshold,
        "threshold_candidates": sorted(declared_thresholds),
        "threshold_amendment_sha256": threshold_amendment_sha256,
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
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_validation_dice": float(checkpoint["validation_dice"]),
        "device": device.type.upper(),
        "parameter_count": released["parameter_count"],
        "prediction_count": len(predictions),
        "ms_per_img": float(np.mean(timings)),
        "prediction_set_sha256": prediction_set_digest(paths),
        "probability_set": probability_set,
        "mask_derivation_verification": {
            "complete": len(derivation_records) == len(records),
            "count": len(derivation_records),
            "records_sha256": derivation_digest,
            "all_masks_rederived_from_retained_probabilities": True,
        },
        "threshold_receipt_sha256": threshold_receipt_sha256,
        "source_config_sha256": file_digest(args.config),
        "checkpoint_sha256": file_digest(args.checkpoint),
        "manifest_sha256": file_digest(args.manifest),
        "split_sha256": file_digest(args.splits),
        "ordered_source_images_sha256": probability_set.get(
            "ordered_source_images_sha256"
        ),
        "cache_used_for_image_decode": inference_cache_dir is not None,
        "paper_timing_preprocess": (
            "on_the_fly_decode_and_resample" if args.split == "test" else "not_paper_timing"
        ),
        "identities": {
            "source_config_sha256": file_digest(args.config),
            "checkpoint_sha256": file_digest(args.checkpoint),
            "manifest_sha256": file_digest(args.manifest),
            "split_sha256": file_digest(args.splits),
            "selector_receipt_sha256": threshold_receipt_sha256,
            "amendment_sha256": threshold_amendment_sha256,
        },
        "predictions": [predictions[key] for key in sorted(predictions)],
        "errors": [],
        "debug_limited": bool(
            args.limit or args.allow_sweep_checkpoint or args.allow_unamended_debug
        ),
        "provenance": start_provenance,
    }
    if previous_complete is not None:
        if previous_complete.get("probability_set", {}).get("sha256") != (
            probability_set.get("sha256")
        ):
            raise ValueError("immutable complete SUFS probability set changed")
        return previous_complete
    atomic_json_once(args.run_card, record)
    return record


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point with queue-compatible OOM status."""
    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    try:
        print(infer(parse_args(argv), provenance), flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"CUDA OOM: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(76) from exc


if __name__ == "__main__":
    main()
