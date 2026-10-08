#!/usr/bin/env python3
"""Select and fit classical block segmenters using train/validation data only.

This program deliberately has no test-inference or scoring path. It chooses a
configuration by balanced block accuracy on the validation partition, refits
that configuration on training blocks, and writes a frozen model plus an audit
card. ``predict_classical.py`` generates held-out masks without opening ground
truth, and only ``evaluate_masks.py`` may populate final metric tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import sklearn
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

import _path  # noqa: F401
from segbench import features as F
from segbench.io import (
    attach_split,
    load_manifest,
    manifest_target_shape,
    mask_ppi_for,
    match_shape,
    read_gray,
    read_mask,
    to_target_ppi,
)
from segbench.models import DISPLAY, PARAM_GRIDS, build, fitted_scalar_count


METHODS = ("svm", "rf", "welm")


def ensure_nice(target: int = 10) -> int:
    """Lower this CPU job to at least the project scheduling priority."""
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
    """Write JSON through a sibling replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_joblib(path: Path, payload: Any) -> None:
    """Write one compressed joblib artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, path)


def source_identity() -> Dict[str, str]:
    """Hash code that defines classical data preparation and fitting."""
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "segbench" / "features.py",
        root / "segbench" / "io.py",
        root / "segbench" / "models.py",
    ]
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def git_identity() -> Dict[str, Any]:
    """Record source revision without modifying the repository."""
    root = Path(__file__).resolve().parents[1]

    def run(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *arguments],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": status not in {"", "unavailable"},
    }


def featurize(
    row: Mapping[str, Any], target_ppi: float, mask_column: str
) -> Dict[str, Any]:
    """Read one development image and its declared ground truth."""
    cv2.setNumThreads(1)
    image = to_target_ppi(read_gray(row["image_path"]), row["ppi"], target_ppi)
    declared = manifest_target_shape(row, target_ppi)
    if declared is not None and image.shape != declared:
        raise ValueError(
            f"{row['image_id']}: image target shape {image.shape} != declared {declared}"
        )
    ground_truth = to_target_ppi(
        read_mask(row[mask_column]),
        mask_ppi_for(row, mask_column),
        target_ppi,
        is_mask=True,
    )
    ground_truth = match_shape(ground_truth, image.shape)
    return {
        "image_id": str(row["image_id"]),
        "features": F.extract_block_features(image),
        "ground_truth": ground_truth,
    }


def sample_blocks(
    item: Mapping[str, Any], per_class: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Take a deterministic balanced sample from one image's block grid."""
    features = item["features"]
    hb, wb, n_features = features.shape
    labels = F.block_labels(item["ground_truth"], hb, wb).ravel()
    values = features.reshape(-1, n_features)
    selected = []
    for label in (True, False):
        candidates = np.flatnonzero(labels == label)
        if len(candidates):
            selected.append(
                rng.choice(
                    candidates, size=min(per_class, len(candidates)), replace=False
                )
            )
    if not selected:
        return np.empty((0, n_features), np.float32), np.empty(0, bool)
    indices = np.concatenate(selected)
    return values[indices], labels[indices]


def build_block_set(
    rows: Any,
    *,
    target_ppi: float,
    mask_column: str,
    blocks_per_class: int,
    cap: int,
    seed: int,
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create one reproducible train or validation block matrix."""
    records = rows.to_dict("records")
    if not records:
        raise ValueError("cannot build a block set from zero images")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        items = Parallel(n_jobs=n_jobs)(
            delayed(featurize)(record, target_ppi, mask_column)
            for record in records
        )
    rng = np.random.default_rng(seed)
    samples = [sample_blocks(item, blocks_per_class, rng) for item in items]
    nonempty = [(values, labels) for values, labels in samples if len(values)]
    if not nonempty:
        raise ValueError("no labelled blocks were sampled")
    values = np.concatenate([item[0] for item in nonempty])
    labels = np.concatenate([item[1] for item in nonempty])
    if len(values) > cap:
        keep = rng.choice(len(values), size=cap, replace=False)
        values, labels = values[keep], labels[keep]
    if np.unique(labels).size != 2:
        raise ValueError("block set must contain both foreground and background")
    return values, labels


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute equally weighted foreground/background block accuracy."""
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    if labels.shape != predictions.shape or np.unique(labels).size != 2:
        raise ValueError("balanced accuracy requires aligned two-class arrays")
    true_positive_rate = float(predictions[labels].mean())
    true_negative_rate = float((~predictions[~labels]).mean())
    return (true_positive_rate + true_negative_rate) / 2.0


def tune_and_fit(
    method: str,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    val_values: np.ndarray,
    val_labels: np.ndarray,
    *,
    tune_blocks: int,
    seed: int,
    n_jobs: int,
) -> Tuple[Any, Dict[str, Any], float, list[Dict[str, Any]]]:
    """Select on validation only, then fit the selected model on train only."""
    tuning_rng = np.random.default_rng(seed + 2)
    tune_count = min(tune_blocks, len(train_values))
    train_indices = tuning_rng.choice(
        len(train_values), size=tune_count, replace=False
    )
    scores: list[Dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for candidate_index, parameters in enumerate(PARAM_GRIDS[method]):
            candidate = build(method, parameters, seed, n_jobs).fit(
                train_values[train_indices], train_labels[train_indices]
            )
            score = balanced_accuracy(
                val_labels,
                np.asarray(candidate.predict(val_values), dtype=bool),
            )
            scores.append(
                {
                    "candidate_index": candidate_index,
                    "parameters": parameters,
                    "validation_balanced_accuracy": score,
                }
            )
            print(
                f"[{method}] candidate={candidate_index} {parameters} "
                f"val_balanced_accuracy={score:.6f}",
                flush=True,
            )
        # Candidate order is the predeclared deterministic tie-break.
        selected = max(
            scores,
            key=lambda item: (
                float(item["validation_balanced_accuracy"]),
                -int(item["candidate_index"]),
            ),
        )
        parameters = dict(selected["parameters"])
        model = build(method, parameters, seed, n_jobs).fit(
            train_values, train_labels
        )
    return (
        model,
        parameters,
        float(selected["validation_balanced_accuracy"]),
        scores,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the train/validation-only CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--train-subset", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-ppi", type=float, default=500.0)
    parser.add_argument("--mask-column", default="mask_path")
    parser.add_argument("--blocks-per-class", type=int, default=64)
    parser.add_argument("--max-train-blocks", type=int, default=300_000)
    parser.add_argument("--max-val-blocks", type=int, default=100_000)
    parser.add_argument("--tune-blocks", type=int, default=60_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    # Kept only to turn old invocations into an explicit, safe failure.
    parser.add_argument("--eval-subsets", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--save-masks", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.eval_subsets or args.save_masks:
        parser.error(
            "run_classical.py never opens test data; use predict_classical.py "
            "then evaluate_masks.py"
        )
    for name in (
        "blocks_per_class",
        "max_train_blocks",
        "max_val_blocks",
        "tune_blocks",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.target_ppi <= 0 or args.n_jobs < 1 or args.n_jobs > 64:
        parser.error("target PPI must be positive and n-jobs must be in 1..64")
    if args.limit < 0 or args.limit_train < 0 or args.limit_val < 0:
        parser.error("debug limits cannot be negative")
    if args.limit and (args.limit_train or args.limit_val):
        parser.error("use --limit or the split-specific limits, not both")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Build/resume development features, tune, and freeze each estimator."""
    args = parse_args(argv)
    nice_level = ensure_nice()
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "models").mkdir(parents=True, exist_ok=True)
    (args.out / "cards").mkdir(parents=True, exist_ok=True)

    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    frame = attach_split(load_manifest(args.manifest), split_payload["splits"])
    if args.mask_column not in frame.columns:
        raise ValueError(f"manifest has no mask column {args.mask_column!r}")
    frame[args.mask_column] = frame[args.mask_column].fillna("").astype(str)
    frame = frame[
        (frame["subset"] == args.train_subset)
        & frame["split"].isin(["train", "val"])
        & (frame[args.mask_column].str.len() > 0)
    ].copy()

    def partition(name: str) -> Any:
        rows = frame[frame["split"] == name].sort_values(
            "image_id", kind="stable"
        )
        limit = args.limit or getattr(args, f"limit_{name}")
        return rows.head(limit).copy() if limit else rows.copy()

    train_rows, val_rows = partition("train"), partition("val")
    if train_rows.empty or val_rows.empty:
        raise ValueError(
            f"development split is empty: train={len(train_rows)} val={len(val_rows)}"
        )

    code_hashes = source_identity()
    identity = {
        "schema_version": 2,
        "manifest_sha256": sha256_file(args.manifest),
        "splits_sha256": sha256_file(args.splits),
        "train_subset": args.train_subset,
        "mask_column": args.mask_column,
        "target_ppi": args.target_ppi,
        "blocks_per_class": args.blocks_per_class,
        "max_train_blocks": args.max_train_blocks,
        "max_val_blocks": args.max_val_blocks,
        "limit_train": int(args.limit or args.limit_train),
        "limit_val": int(args.limit or args.limit_val),
        "seed": args.seed,
        "code_sha256": code_hashes,
    }
    cache_path = args.feature_cache or (
        args.out / "cache" / f"blocks_{args.train_subset}_{args.mask_column}.joblib"
    )
    cache: Optional[Mapping[str, Any]] = None
    if args.resume and cache_path.exists():
        candidate = joblib.load(cache_path)
        if candidate.get("identity") == identity:
            cache = candidate
            print(f"resuming verified feature cache {cache_path}", flush=True)
    if cache is None:
        train_values, train_labels = build_block_set(
            train_rows,
            target_ppi=args.target_ppi,
            mask_column=args.mask_column,
            blocks_per_class=args.blocks_per_class,
            cap=args.max_train_blocks,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        val_values, val_labels = build_block_set(
            val_rows,
            target_ppi=args.target_ppi,
            mask_column=args.mask_column,
            blocks_per_class=args.blocks_per_class,
            cap=args.max_val_blocks,
            seed=args.seed + 1,
            n_jobs=args.n_jobs,
        )
        cache = {
            "identity": identity,
            "train_values": train_values,
            "train_labels": train_labels,
            "val_values": val_values,
            "val_labels": val_labels,
        }
        atomic_joblib(cache_path, cache)
    train_values = np.asarray(cache["train_values"])
    train_labels = np.asarray(cache["train_labels"], dtype=bool)
    val_values = np.asarray(cache["val_values"])
    val_labels = np.asarray(cache["val_labels"], dtype=bool)
    print(
        f"development only: train_images={len(train_rows)} val_images={len(val_rows)} "
        f"train_blocks={len(train_values)} val_blocks={len(val_values)}",
        flush=True,
    )

    aggregate: Dict[str, Any] = {"identity": identity, "methods": {}}
    for method in args.methods:
        model_path = args.out / "models" / f"{method}_{args.train_subset}.joblib"
        card_path = args.out / "cards" / f"{method}_{args.train_subset}.json"
        method_identity = {
            **identity,
            "method": method,
            "parameter_grid": PARAM_GRIDS[method],
            "tune_blocks": args.tune_blocks,
            "selection_metric": "validation_balanced_block_accuracy",
            "selection_tie_break": "predeclared_candidate_order",
        }
        if args.resume and model_path.exists() and card_path.exists():
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if (
                card.get("identity") == method_identity
                and card.get("model_sha256") == sha256_file(model_path)
            ):
                print(f"[{method}] resuming verified frozen model", flush=True)
                aggregate["methods"][method] = card
                continue

        started = time.monotonic()
        model, selected, validation_score, candidate_scores = tune_and_fit(
            method,
            train_values,
            train_labels,
            val_values,
            val_labels,
            tune_blocks=args.tune_blocks,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        scalar_count = fitted_scalar_count(model)
        atomic_joblib(model_path, model)
        card = {
            "identity": method_identity,
            "method": method,
            "display_name": DISPLAY[method],
            "implementation_status": "reimplemented",
            "device": "CPU",
            "nice_level": nice_level,
            "seed": args.seed,
            "train_images": len(train_rows),
            "validation_images": len(val_rows),
            "train_blocks": len(train_values),
            "validation_blocks": len(val_values),
            "selected_parameters": selected,
            "selected_validation_balanced_accuracy": validation_score,
            "candidate_scores": candidate_scores,
            "parameter_semantics": (
                "numeric scalars in fitted inference representation; not neural "
                "trainable-parameter equivalents"
            ),
            "fitted_scalar_count": scalar_count,
            "params_m": scalar_count / 1_000_000.0,
            "model_sha256": sha256_file(model_path),
            "model_bytes": model_path.stat().st_size,
            "elapsed_seconds": time.monotonic() - started,
            "debug_limited": bool(
                args.limit or args.limit_train or args.limit_val
            ),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "git": git_identity(),
            "command": list(sys.argv),
        }
        atomic_json(card_path, card)
        aggregate["methods"][method] = card
        print(
            f"[{method}] frozen val_balanced_accuracy={validation_score:.6f} "
            f"fitted_scalars={scalar_count}",
            flush=True,
        )

    atomic_json(args.out / f"chosen_params_{args.train_subset}.json", aggregate)
    print("classical development complete; no test rows were opened", flush=True)


if __name__ == "__main__":
    main()
