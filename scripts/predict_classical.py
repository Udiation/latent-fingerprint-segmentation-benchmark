#!/usr/bin/env python3
"""Generate classical-model masks without reading held-out ground truth.

Hyperparameter selection and fitting are performed by the development-only
``run_classical.py`` command. This command loads one frozen fitted model, runs batch-one
inference on a declared split, and writes 500-PPI binary masks.  Final metrics
are produced only by ``evaluate_masks.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import _path  # noqa: F401
from segbench import features as F
from segbench.deep_utils import run_provenance
from segbench.io import (
    attach_split,
    load_manifest,
    manifest_target_shape,
    read_gray,
    to_target_ppi,
)
from segbench.models import fitted_scalar_count, set_single_thread
from segbench.postprocess import postprocess_blocks


def ensure_nice(target: int = 10) -> int:
    """Lower this CPU timing job to at least the project scheduling priority."""
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < target:
        os.nice(target - current)
    return os.getpriority(os.PRIO_PROCESS, 0)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through an atomic sibling replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary 0/255 PNG atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.stem + ".partial.png")
    if not cv2.imwrite(str(temporary), np.where(mask, 255, 0).astype(np.uint8)):
        raise OSError(f"failed to write prediction {temporary}")
    os.replace(temporary, path)


def load_rows(
    manifest: Path,
    splits_path: Path,
    subset: str,
    split: str,
    limit: int,
) -> Any:
    """Load only image metadata for a deterministic partition."""
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    rows = attach_split(load_manifest(manifest), payload["splits"])
    rows = rows[(rows["subset"] == subset) & (rows["split"] == split)].copy()
    if split == "test" and "examiner_value" in rows.columns:
        rows = rows[rows["examiner_value"].astype(str).str.upper() != "NONPRINT"]
    rows = rows.sort_values("image_id", kind="stable").reset_index(drop=True)
    if limit:
        rows = rows.head(limit).copy()
    if rows.empty:
        raise ValueError(f"no rows for subset={subset!r}, split={split!r}")
    # Predictions need image metadata only. Clearing annotation-bearing columns
    # prevents future code changes from accidentally opening held-out labels.
    for column in rows.columns:
        if "mask" in column.lower():
            rows[column] = ""
    return rows


def extract_image_features(
    row: Mapping[str, Any], target_ppi: float
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """Decode/resample an image and extract its shared block features."""
    started = time.perf_counter()
    image = to_target_ppi(read_gray(row["image_path"]), float(row["ppi"]), target_ppi)
    declared = manifest_target_shape(row, target_ppi)
    if declared is not None and image.shape != declared:
        raise ValueError(
            f"{row['image_id']}: decoded target shape {image.shape} != declared {declared}"
        )
    features = F.extract_block_features(image)
    return features, image.shape, (time.perf_counter() - started) * 1000.0


def infer_one(
    model: Any,
    row: Mapping[str, Any],
    target_ppi: float,
    min_component_blocks: int,
) -> Tuple[np.ndarray, float]:
    """Run complete single-thread preprocessing, prediction, and cleanup."""
    features, shape, preprocessing_ms = extract_image_features(row, target_ppi)
    hb, wb, n_features = features.shape
    started = time.perf_counter()
    block_mask = np.asarray(
        model.predict(features.reshape(-1, n_features))
    ).astype(bool).reshape(hb, wb)
    block_mask = postprocess_blocks(block_mask, min_component_blocks)
    prediction = F.blocks_to_mask(block_mask, shape)
    inference_ms = (time.perf_counter() - started) * 1000.0
    return prediction, preprocessing_ms + inference_ms


def valid_existing(path: Path, shape: Tuple[int, int]) -> bool:
    """Return true only for a complete binary PNG on the declared grid."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (
        image is not None
        and image.shape == shape
        and set(np.unique(image).tolist()).issubset({0, 255})
    )


def expected_target_shape(row: Mapping[str, Any], target_ppi: float) -> Tuple[int, int]:
    """Use the audited grid when present, otherwise derive a legacy smoke-test grid."""
    declared = manifest_target_shape(row, target_ppi)
    if declared is not None:
        return declared
    return to_target_ppi(
        read_gray(row["image_path"]), float(row["ppi"]), target_ppi
    ).shape


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--training-card",
        required=True,
        type=Path,
        help="audit card written beside the frozen model by run_classical.py",
    )
    parser.add_argument("--method", choices=("svm", "rf", "welm"), required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--subset", default="finger_latent")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--target-ppi", type=float, default=500.0)
    parser.add_argument("--min-component-blocks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-card", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    args = parse_args(argv)
    nice_level = ensure_nice()
    if args.target_ppi <= 0 or args.min_component_blocks < 1:
        raise ValueError("target PPI and minimum component size must be positive")
    cv2.setNumThreads(1)
    rows = load_rows(args.manifest, args.splits, args.subset, args.split, args.limit)
    model = set_single_thread(joblib.load(args.model))
    training_card = json.loads(args.training_card.read_text(encoding="utf-8"))
    model_sha256 = sha256_file(args.model)
    if training_card.get("method") != args.method:
        raise ValueError("training card method disagrees with --method")
    if training_card.get("seed") != args.seed:
        raise ValueError("training card seed disagrees with --seed")
    if training_card.get("model_sha256") != model_sha256:
        raise ValueError("frozen model hash disagrees with its training card")
    if args.split == "test" and training_card.get("debug_limited"):
        raise ValueError("a debug-limited classical model cannot score the test split")
    scalar_count = fitted_scalar_count(model)
    if int(training_card.get("fitted_scalar_count", -1)) != scalar_count:
        raise ValueError("fitted scalar count disagrees with the training card")
    identity = {
        "schema_version": 1,
        "method": args.method,
        "seed": args.seed,
        "model_sha256": model_sha256,
        "training_card_sha256": sha256_file(args.training_card),
        "manifest_sha256": sha256_file(args.manifest),
        "splits_sha256": sha256_file(args.splits),
        "subset": args.subset,
        "split": args.split,
        "target_ppi": args.target_ppi,
        "min_component_blocks": args.min_component_blocks,
        "limit": args.limit,
    }
    checkpoint_path = args.run_card.with_suffix(".checkpoint.json")
    completed: Dict[str, float] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("prediction checkpoint identity disagrees with this run")
        completed = {
            str(key): float(value) for key, value in checkpoint["completed_ms"].items()
        }

    records = rows.to_dict("records")
    with threadpool_limits(limits=1):
        for row in records[: min(args.warmup, len(records))]:
            prediction, _ = infer_one(
                model, row, args.target_ppi, args.min_component_blocks
            )
            if prediction.shape != expected_target_shape(row, args.target_ppi):
                raise ValueError(f"{row['image_id']}: warm-up output shape mismatch")

        started = time.perf_counter()
        for index, row in enumerate(records, start=1):
            image_id = str(row["image_id"])
            target = args.output_root / f"{image_id}.png"
            declared = expected_target_shape(row, args.target_ppi)
            if image_id in completed and valid_existing(target, declared):
                continue
            prediction, elapsed_ms = infer_one(
                model, row, args.target_ppi, args.min_component_blocks
            )
            if prediction.shape != declared:
                raise ValueError(
                    f"{image_id}: prediction shape {prediction.shape} != {declared}"
                )
            atomic_mask(target, prediction)
            completed[image_id] = elapsed_ms
            if index % args.progress_every == 0 or index == len(records):
                atomic_json(
                    checkpoint_path,
                    {"identity": identity, "completed_ms": completed},
                )
                print(
                    f"{args.method}: {index}/{len(records)} complete={len(completed)}",
                    flush=True,
                )

    if len(completed) != len(records):
        raise RuntimeError(f"completed {len(completed)} masks for {len(records)} rows")
    timings = np.asarray([completed[str(row["image_id"])] for row in records], dtype=float)
    payload = {
        **identity,
        "display_name": args.display_name,
        "prediction_count": len(records),
        "missing_count": 0,
        "warmup_images": min(args.warmup, len(records)),
        "ms_per_img": float(timings.mean()),
        "ms_per_img_std": float(timings.std(ddof=0)),
        "device": "CPU",
        "nice_level": nice_level,
        "parameter_semantics": training_card["parameter_semantics"],
        "fitted_scalar_count": scalar_count,
        "params_m": scalar_count / 1_000_000.0,
        "selected_parameters": training_card["selected_parameters"],
        "selected_validation_balanced_accuracy": training_card[
            "selected_validation_balanced_accuracy"
        ],
        "serialized_model_bytes": args.model.stat().st_size,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "command": list(sys.argv),
        "provenance": provenance,
    }
    atomic_json(args.run_card, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
