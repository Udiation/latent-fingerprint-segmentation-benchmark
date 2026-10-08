#!/usr/bin/env python3
"""Tune or run PyFing GMFS/SUFS under the shared benchmark protocol.

``tune-gmfs`` reads validation ground truth only and freezes the selected
parameter tuple. ``generate`` never reads target masks; it writes binary
500-PPI predictions for later scoring exclusively through ``evaluate_masks``.
Long generation is resumable through an identity-checked restricted checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

if os.environ.get("KERAS_BACKEND", "torch").lower() != "torch":
    raise RuntimeError("run_pyfing requires KERAS_BACKEND=torch")
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import joblib
import numpy as np
import pandas as pd
import yaml

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once
from segbench.deep_utils import run_provenance
from segbench import probability_artifacts, preprocess_cache
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
from segbench.metrics import image_metrics


CHECKPOINT_SCHEMA_VERSION = 2
GMFS_PARAMETER_KEYS = {
    "sigma",
    "percentile",
    "threshold",
    "closing_count",
    "opening_count",
    "image_dpi",
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through an atomic sibling rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_mask(path: Path, mask: np.ndarray) -> None:
    """Write a binary mask atomically as 0/255 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.stem + ".partial.png")
    binary = np.where(mask, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(temporary), binary):
        raise OSError(f"failed to write {temporary}")
    os.replace(temporary, path)


def atomic_dataframe_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV through an atomic sibling rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_rows(
    manifest_path: Path,
    split_path: Path,
    subset: str,
    split: str,
    mask_column: str | None,
    limit: int,
) -> pd.DataFrame:
    """Load a deterministic split, excluding NONPRINT only from test."""
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    frame = attach_split(load_manifest(manifest_path), split_payload["splits"])
    frame = frame[(frame["subset"] == subset) & (frame["split"] == split)].copy()
    if split == "test" and "examiner_value" in frame.columns:
        frame = frame[frame["examiner_value"].astype(str).str.upper() != "NONPRINT"]
    if mask_column:
        if mask_column not in frame.columns:
            raise ValueError(f"manifest has no mask column {mask_column!r}")
        frame[mask_column] = frame[mask_column].fillna("").astype(str)
        frame = frame[frame[mask_column].str.len() > 0]
    frame = frame.sort_values("image_id", kind="stable").reset_index(drop=True)
    if limit:
        frame = frame.head(limit).copy()
    if frame.empty:
        raise ValueError(f"no rows for subset={subset!r}, split={split!r}")
    return frame


def read_image_500(
    row: Mapping[str, Any],
    target_ppi: float,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Decode and area-resample one grayscale source image."""
    if cache_dir is not None:
        image = np.rint(
            preprocess_cache.read_image(cache_dir, str(row["image_id"])) * 255.0
        ).astype(np.uint8)
    else:
        image = to_target_ppi(
            read_gray(row["image_path"]), float(row["ppi"]), target_ppi
        )
    declared = manifest_target_shape(row, target_ppi)
    if declared is not None and image.shape != declared:
        raise ValueError(
            f"{row['image_id']}: decoded target shape {image.shape} != declared {declared}"
        )
    return image


def read_gt_500(
    row: Mapping[str, Any], mask_column: str, target_ppi: float, shape: Tuple[int, int]
) -> np.ndarray:
    """Decode a GT mask using its own PPI and align rounding to the image grid."""
    mask = to_target_ppi(
        read_mask(row[mask_column]),
        mask_ppi_for(row, mask_column),
        target_ppi,
        is_mask=True,
    )
    if max(abs(mask.shape[0] - shape[0]), abs(mask.shape[1] - shape[1])) > 1:
        raise ValueError(
            f"{row['image_id']}: resampled GT shape {mask.shape} is incompatible "
            f"with image grid {shape}"
        )
    return match_shape(mask, shape)


def gmfs_default() -> Dict[str, float | int]:
    """Return the released PyFing 0.6 defaults, fully explicit."""
    return {
        "sigma": 13 / 3,
        "percentile": 95,
        "threshold": 0.2,
        "closing_count": 6,
        "opening_count": 12,
        "image_dpi": 500,
    }


def gmfs_candidates(config: Mapping[str, Any]) -> List[Dict[str, float | int]]:
    """Construct the preregistered default-plus-corners validation grid."""
    if float(config.get("target_ppi", 0)) != 500.0:
        raise ValueError("the benchmark GMFS grid must run at 500 PPI")
    tuning = config.get("tuning", {})
    if tuning.get("metric") != "macro_dice" or tuning.get("split") != "val":
        raise ValueError("GMFS selection must be preregistered as validation macro Dice")
    declared_default = dict(config["default"])
    declared_default["image_dpi"] = int(config["target_ppi"])
    if set(declared_default) != GMFS_PARAMETER_KEYS:
        raise ValueError("GMFS default has missing or unexpected parameter keys")
    if declared_default != gmfs_default():
        raise ValueError("configured GMFS default disagrees with the released PyFing default")
    grid = tuning["candidates"]
    if grid.get("policy") != "default_plus_symmetric_corners":
        raise ValueError("unsupported GMFS candidate policy")
    candidates = [declared_default]
    for sigma, threshold, closing, opening in itertools.product(
        grid["sigma"], grid["threshold"], grid["closing_count"], grid["opening_count"]
    ):
        candidates.append(
            {
                "sigma": float(sigma),
                "percentile": int(declared_default["percentile"]),
                "threshold": float(threshold),
                "closing_count": int(closing),
                "opening_count": int(opening),
                "image_dpi": int(config["target_ppi"]),
            }
        )
    unique = {json.dumps(item, sort_keys=True): item for item in candidates}
    return [unique[key] for key in sorted(unique)]


def validate_gmfs_parameters(parameters: Mapping[str, Any]) -> Dict[str, float | int]:
    """Validate and normalize one explicit GMFS parameter mapping."""
    if set(parameters) != GMFS_PARAMETER_KEYS:
        raise ValueError("GMFS parameters have missing or unexpected keys")

    def exact_int(name: str) -> int:
        numeric = float(parameters[name])
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"GMFS {name} must be a finite integer")
        return int(numeric)

    normalized: Dict[str, float | int] = {
        "sigma": float(parameters["sigma"]),
        "percentile": exact_int("percentile"),
        "threshold": float(parameters["threshold"]),
        "closing_count": exact_int("closing_count"),
        "opening_count": exact_int("opening_count"),
        "image_dpi": exact_int("image_dpi"),
    }
    if not np.isfinite(normalized["sigma"]) or normalized["sigma"] <= 0:
        raise ValueError("GMFS sigma must be positive")
    if not 0 <= normalized["percentile"] <= 100:
        raise ValueError("GMFS percentile must be in [0, 100]")
    if not np.isfinite(normalized["threshold"]) or normalized["threshold"] < 0:
        raise ValueError("GMFS threshold must be non-negative")
    if normalized["closing_count"] < 0 or normalized["opening_count"] < 0:
        raise ValueError("GMFS morphology iteration counts must be non-negative")
    if normalized["image_dpi"] != 500:
        raise ValueError("benchmark GMFS parameters must declare image_dpi=500")
    return normalized


def build_gmfs(parameters: Mapping[str, Any]):
    """Instantiate GMFS lazily so SUFS-only workers do not import Keras early."""
    from pyfing.segmentation import Gmfs, GmfsParameters

    return Gmfs(GmfsParameters(**validate_gmfs_parameters(parameters)))


def load_tuned_parameters(
    path: Path, manifest_path: Path, split_path: Path, subset: str
) -> Tuple[Dict[str, float | int], Dict[str, Any]]:
    """Load a validation-tuned artifact and bind it to its exact data protocol."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "method": "gmfs_tuned",
        "selection_split": "val",
        "selection_metric": "macro_dice",
        "subset": subset,
        "target_ppi": 500.0,
        "manifest_sha256": sha256_file(manifest_path),
        "splits_sha256": sha256_file(split_path),
        "implementation": implementation_provenance("gmfs", None),
    }
    disagreements = {
        key: (payload.get(key), expected)
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if disagreements:
        raise ValueError(f"GMFS tuning artifact provenance mismatch: {disagreements}")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("GMFS tuning artifact has no selected parameter mapping")
    parameters = validate_gmfs_parameters(selected)
    selected_index = payload.get("selected_index")
    candidates = payload.get("candidates")
    config_digest = payload.get("config_sha256")
    if (
        not isinstance(selected_index, int)
        or not isinstance(candidates, list)
        or payload.get("candidate_count") != len(candidates)
        or not isinstance(payload.get("n_images"), int)
        or payload["n_images"] < 1
        or not payload.get("mask_column")
        or not isinstance(config_digest, str)
        or len(config_digest) != 64
        or selected_index < 0
        or selected_index >= len(candidates)
    ):
        raise ValueError("GMFS tuning artifact has inconsistent selection evidence")
    normalized_candidates: List[Tuple[Dict[str, float | int], float]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or candidate.get("index") != index:
            raise ValueError("GMFS tuning artifact has invalid candidate indices")
        candidate_parameters = candidate.get("parameters")
        score = candidate.get("val_dice")
        if not isinstance(candidate_parameters, Mapping):
            raise ValueError("GMFS tuning artifact has invalid candidate parameters")
        normalized_candidate = validate_gmfs_parameters(candidate_parameters)
        if not isinstance(score, (int, float)) or not np.isfinite(float(score)):
            raise ValueError("GMFS tuning artifact has an invalid validation score")
        normalized_candidates.append((normalized_candidate, float(score)))
    expected_index = sorted(
        range(len(normalized_candidates)),
        key=lambda index: (
            -normalized_candidates[index][1],
            float(normalized_candidates[index][0]["sigma"]),
            float(normalized_candidates[index][0]["threshold"]),
            int(normalized_candidates[index][0]["closing_count"]),
            int(normalized_candidates[index][0]["opening_count"]),
        ),
    )[0]
    if (
        selected_index != expected_index
        or normalized_candidates[selected_index][0] != parameters
        or normalized_candidates[selected_index][1]
        != float(payload.get("selected_val_dice", float("nan")))
    ):
        raise ValueError("GMFS tuning artifact selection does not follow the tie rule")
    provenance = {
        "kind": "validation_artifact",
        "sha256": sha256_file(path),
        "selection_split": payload["selection_split"],
        "selection_metric": payload["selection_metric"],
        "selection_mask_column": payload.get("mask_column", ""),
        "selected_val_dice": payload.get("selected_val_dice"),
        "config_sha256": config_digest,
    }
    return parameters, provenance


def score_gmfs_row(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    target_ppi: float,
    mask_column: str,
) -> Tuple[str, List[float]]:
    """Score every frozen GMFS candidate on one validation image."""
    cv2.setNumThreads(1)
    image = read_image_500(row, target_ppi)
    gt = read_gt_500(row, mask_column, target_ppi, image.shape)
    scores: List[float] = []
    for parameters in candidates:
        prediction = build_gmfs(parameters).run(image) > 0
        prediction = match_shape(prediction, image.shape)
        scores.append(float(image_metrics(prediction, gt)["dice"]))
    return str(row["image_id"]), scores


def tune_gmfs(args: argparse.Namespace) -> None:
    """Run the preregistered GMFS grid on validation only."""
    provenance = getattr(args, "_start_provenance", None) or run_provenance(sys.argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target_ppi = float(config["target_ppi"])
    candidates = gmfs_candidates(config)
    rows = load_rows(
        args.manifest, args.splits, args.subset, "val", args.mask_column, args.limit
    )
    started = time.perf_counter()
    results = joblib.Parallel(n_jobs=args.n_jobs, prefer="processes", batch_size=1)(
        joblib.delayed(score_gmfs_row)(row, candidates, target_ppi, args.mask_column)
        for row in rows.to_dict("records")
    )
    image_ids = [item[0] for item in results]
    matrix = np.asarray([item[1] for item in results], dtype=np.float64)
    means = matrix.mean(axis=0)

    parameter_tuples = [
        (
            float(item["sigma"]),
            float(item["threshold"]),
            int(item["closing_count"]),
            int(item["opening_count"]),
        )
        for item in candidates
    ]
    selected_index = sorted(
        range(len(candidates)), key=lambda index: (-means[index], parameter_tuples[index])
    )[0]
    payload = {
        "schema_version": 1,
        "method": "gmfs_tuned",
        "selection_split": "val",
        "selection_metric": "macro_dice",
        "subset": args.subset,
        "mask_column": args.mask_column,
        "target_ppi": target_ppi,
        "n_images": len(rows),
        "candidate_count": len(candidates),
        "selected_index": selected_index,
        "selected": candidates[selected_index],
        "selected_val_dice": float(means[selected_index]),
        "candidates": [
            {"index": index, "parameters": item, "val_dice": float(means[index])}
            for index, item in enumerate(candidates)
        ],
        "manifest_sha256": sha256_file(args.manifest),
        "splits_sha256": sha256_file(args.splits),
        "config_sha256": sha256_file(args.config),
        "implementation": implementation_provenance("gmfs", None),
        "wall_seconds": time.perf_counter() - started,
        "command": sys.argv,
        "provenance": provenance,
    }
    per_image = pd.DataFrame(
        matrix, columns=[f"candidate_{index:02d}" for index in range(len(candidates))]
    )
    per_image.insert(0, "image_id", image_ids)
    atomic_dataframe_csv(args.out_json.with_suffix(".per_image.csv"), per_image)
    # Publish the selection artifact last, after all of its audit evidence exists.
    atomic_json(args.out_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def configure_accelerator(memory_fraction: float) -> str:
    """Apply the per-process fraction before SUFS model allocation."""
    import torch

    if not 0.0 < memory_fraction <= 0.3:
        raise ValueError("memory fraction must be in (0, 0.3]")
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch allows this setting only before inter-op work starts. The
            # queue also enforces single-thread library environment variables.
            pass
        return "CPU"
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    return "GPU"


def build_segmenter(method: str, parameters: Mapping[str, Any] | None):
    """Build one frozen PyFing segmenter and return it with parameter count."""
    if method == "gmfs":
        return build_gmfs(parameters or gmfs_default()), 0
    if method != "sufs":
        raise ValueError(f"unknown method {method!r}")
    from segbench.pyfing_compat import TorchSufs

    segmenter = TorchSufs()
    return segmenter, int(segmenter.model.count_params())


def implementation_provenance(method: str, segmenter: Any) -> Dict[str, Any]:
    """Fingerprint executable PyFing code and, for SUFS, released weights."""
    from segbench import io as io_module
    from segbench import metrics as metrics_module
    from pyfing import segmentation

    payload: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "pyfing_version": importlib.metadata.version("pyfing"),
        "numpy_version": importlib.metadata.version("numpy"),
        "opencv_version": importlib.metadata.version("opencv-python-headless"),
        "runner_sha256": sha256_file(Path(__file__)),
        "io_sha256": sha256_file(Path(inspect.getfile(io_module))),
        "metrics_sha256": sha256_file(Path(inspect.getfile(metrics_module))),
        "pyfing_segmentation_sha256": sha256_file(Path(inspect.getfile(segmentation))),
    }
    if method == "sufs":
        weights = Path(inspect.getfile(segmentation)).parent / "models" / "SUFS.weights.h5"
        if not weights.is_file():
            raise FileNotFoundError("released SUFS weights are missing")
        payload.update(
            {
                "keras_version": importlib.metadata.version("keras"),
                "torch_version": importlib.metadata.version("torch"),
                "sufs_weights_sha256": sha256_file(weights),
                "compatibility_sha256": sha256_file(
                    Path(inspect.getfile(type(segmenter)))
                ),
            }
        )
    return payload


def segmenter_device(segmenter: Any, anticipated: str) -> str:
    """Record the device that actually owns SUFS model variables."""
    model = getattr(segmenter, "model", None)
    variables = list(getattr(model, "variables", [])) if model is not None else []
    devices = {
        str(getattr(getattr(variable, "value", variable), "device", "")).lower()
        for variable in variables
    }
    devices.discard("")
    if any("cuda" in device for device in devices):
        return "GPU"
    if devices:
        return "CPU"
    return anticipated


def validate_existing_mask(
    path: Path, shape: Tuple[int, int], expected_sha256: str
) -> bool:
    """Return true only for a complete binary PNG of the expected shape."""
    existing = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (
        existing is not None
        and existing.shape == shape
        and set(np.unique(existing).tolist()).issubset({0, 255})
        and sha256_file(path) == expected_sha256
    )


def generate(args: argparse.Namespace) -> None:
    if args.run_card.is_file():
        raise ValueError(
            "immutable complete PyFing run card already exists; choose a new output"
        )
    """Generate resumable masks without reading any ground-truth pixels."""
    provenance = getattr(args, "_start_provenance", None) or run_provenance(sys.argv)
    memory_fraction = float(os.environ.get("SEG_GPU_MEMORY_FRACTION", args.memory_fraction))
    anticipated_device = (
        configure_accelerator(memory_fraction) if args.method == "sufs" else "CPU"
    )
    cv2.setNumThreads(1)
    parameters: Mapping[str, Any] | None = None
    parameters_source: Mapping[str, Any] = {"kind": "released_default"}
    if args.method == "gmfs":
        if args.parameters_json:
            parameters, parameters_source = load_tuned_parameters(
                args.parameters_json, args.manifest, args.splits, args.subset
            )
        else:
            parameters = gmfs_default()
    elif args.parameters_json:
        raise ValueError("--parameters-json is valid only for GMFS")
    segmenter, parameter_count = build_segmenter(args.method, parameters)
    device = segmenter_device(segmenter, anticipated_device)
    implementation = implementation_provenance(args.method, segmenter)
    rows = load_rows(args.manifest, args.splits, args.subset, args.split, None, args.limit)
    cache_dir = getattr(args, "cache_dir", None)
    probability_root = getattr(args, "probability_root", None)
    cache_receipt_sha256 = None
    if cache_dir is not None:
        receipt = preprocess_cache.load_receipt(
            cache_dir,
            manifest_sha256=sha256_file(args.manifest),
            target_ppi=float(args.target_ppi),
            mask_columns=[],
            required_ids=[str(value) for value in rows["image_id"]],
            verify_artifact_hashes=True,
        )
        del receipt
        cache_receipt_sha256 = sha256_file(cache_dir / "receipt.json")
    inference_cache_dir = cache_dir if args.split != "test" else None
    source_hashes = probability_artifacts.source_hash_map(rows.to_dict("records"))
    if args.method == "sufs":
        if probability_root is None:
            probability_root = (
                args.output_root.parent / f"{args.output_root.name}_probabilities"
            )
        probability_root.mkdir(parents=True, exist_ok=True)

    identity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": args.method,
        "parameters": parameters or "released_sufs_weights",
        "parameters_source": parameters_source,
        "implementation": implementation,
        "device": device,
        "manifest_sha256": sha256_file(args.manifest),
        "splits_sha256": sha256_file(args.splits),
        # Singular spelling is the probability-card contract; keep the legacy
        # plural key as well because it is part of the resumable-run identity.
        "split_sha256": sha256_file(args.splits),
        "subset": args.subset,
        "split": args.split,
        "target_ppi": args.target_ppi,
        "limit": args.limit,
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
    }
    checkpoint_path = args.run_card.with_suffix(".checkpoint.json")
    completed: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("generation checkpoint identity disagrees with this run")
        for key, value in checkpoint.get("completed", {}).items():
            if not isinstance(value, Mapping):
                raise ValueError("generation checkpoint has an invalid completion record")
            completed[str(key)] = {
                "ms": float(value["ms"]),
                "sha256": str(value["sha256"]),
                "probability_sha256": value.get("probability_sha256"),
            }
        completed = probability_artifacts.authenticated_resume_records(
            [
                {"image_id": image_id, **record}
                for image_id, record in completed.items()
            ],
            allowed_ids=[str(value) for value in rows["image_id"]],
            timing_key="ms",
        )

    warmup_rows = rows.head(min(args.warmup, len(rows))).to_dict("records")
    for row in warmup_rows:
        image = read_image_500(row, args.target_ppi, inference_cache_dir)
        if args.method == "sufs":
            probability = probability_artifacts.storage_roundtrip(
                segmenter.probability(image)
            )
            prediction = probability >= float(segmenter.parameters.threshold)
            released = segmenter.run(image) > 0
            if not np.array_equal(prediction, released):
                raise RuntimeError(
                    "SUFS probability threshold does not reproduce released output"
                )
        else:
            prediction = segmenter.run(image) > 0
        if prediction.shape != image.shape:
            raise ValueError(f"{row['image_id']}: warm-up output shape mismatch")

    started = time.perf_counter()
    errors: List[Dict[str, str]] = []
    retained_preexisting_masks = 0
    for index, row in enumerate(rows.to_dict("records"), start=1):
        image_id = str(row["image_id"])
        target = args.output_root / f"{image_id}.png"
        probability_path = (
            probability_artifacts.safe_path(probability_root, image_id)
            if args.method == "sufs" and probability_root is not None
            else None
        )
        declared_shape = manifest_target_shape(row, args.target_ppi)
        if image_id in completed:
            expected_shape = declared_shape
            if expected_shape is None:
                expected_shape = read_image_500(
                    row, args.target_ppi, inference_cache_dir
                ).shape
            if validate_existing_mask(
                target, expected_shape, completed[image_id]["sha256"]
            ) and (
                probability_path is None
                or (
                    probability_path.is_file()
                    and sha256_file(probability_path)
                    == completed[image_id].get("probability_sha256")
                )
            ):
                continue
            completed.pop(image_id)
        try:
            before = time.perf_counter()
            image = read_image_500(row, args.target_ppi, inference_cache_dir)
            probability = None
            if args.method == "sufs":
                probability = probability_artifacts.storage_roundtrip(
                    segmenter.probability(image)
                )
                prediction = probability >= float(segmenter.parameters.threshold)
                if args.split == "val" and index <= 50:
                    released = segmenter.run(image) > 0
                    if not np.array_equal(prediction, released):
                        raise RuntimeError(
                            "SUFS raw probability does not reproduce released val mask"
                        )
            else:
                prediction = segmenter.run(image) > 0
            prediction = match_shape(prediction, image.shape)
            elapsed_ms = (time.perf_counter() - before) * 1000.0
            if target.is_file():
                frozen = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
                if (
                    frozen is None
                    or frozen.shape != prediction.shape
                    or not np.array_equal(frozen > 0, prediction)
                ):
                    raise RuntimeError(
                        f"{image_id}: pre-existing frozen prediction differs"
                    )
                retained_preexisting_masks += 1
            else:
                atomic_mask(target, prediction)
            if probability_path is not None and probability is not None:
                probability_artifacts.atomic_write(probability_path, probability)
            completed[image_id] = {
                "ms": elapsed_ms,
                "sha256": sha256_file(target),
                "probability_sha256": (
                    sha256_file(probability_path)
                    if probability_path is not None
                    else None
                ),
            }
        except Exception as exc:
            errors.append({"image_id": image_id, "error": repr(exc)})
        if index % args.progress_every == 0 or index == len(rows):
            atomic_json(
                checkpoint_path,
                {"identity": identity, "completed": completed, "errors": errors},
            )
            print(
                f"{args.method}: {index}/{len(rows)} complete={len(completed)} "
                f"errors={len(errors)}",
                flush=True,
            )
    if errors:
        raise RuntimeError(f"{len(errors)} prediction errors; inspect {checkpoint_path}")
    if len(completed) != len(rows):
        raise RuntimeError(f"completed {len(completed)} masks for {len(rows)} rows")

    # This is deliberately a post-pass, rather than only an in-loop assertion:
    # a resumed run may skip every already-complete record.  Re-reading the
    # retained float16 artifact proves that the bytes we will deliver (not the
    # pre-storage float32 tensor) reproduce both the released implementation and
    # the saved binary mask on the deterministic validation audit set.
    validation_equivalence_images = 0
    complete_probability_equivalence_images = 0
    probability_equivalence_records: List[Dict[str, str]] = []
    if args.method == "sufs":
        for audit_index, row in enumerate(rows.to_dict("records"), start=1):
            image_id = str(row["image_id"])
            image = read_image_500(row, args.target_ppi, inference_cache_dir)
            stored_path = probability_artifacts.safe_path(probability_root, image_id)
            stored = probability_artifacts.read(
                stored_path,
                image.shape,
            )
            reconstructed = stored >= float(segmenter.parameters.threshold)
            released = segmenter.run(image) > 0
            saved = cv2.imread(
                str(args.output_root / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE
            )
            if saved is None or not np.array_equal(reconstructed, saved > 0):
                raise RuntimeError(
                    "stored SUFS probability does not reproduce its saved val mask"
                )
            if not np.array_equal(reconstructed, released):
                raise RuntimeError(
                    "stored SUFS probability does not reproduce released output"
                )
            complete_probability_equivalence_images += 1
            if args.split == "val" and audit_index <= 50:
                validation_equivalence_images += 1
            probability_equivalence_records.append(
                {
                    "image_id": image_id,
                    "mask_sha256": sha256_file(
                        args.output_root / f"{image_id}.png"
                    ),
                    "probability_sha256": sha256_file(stored_path),
                }
            )
    equivalence_digest = (
        hashlib.sha256(
            json.dumps(
                probability_equivalence_records,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if probability_equivalence_records
        else None
    )

    timings = np.asarray(
        [completed[str(row.image_id)]["ms"] for row in rows.itertuples()], dtype=float
    )
    mask_set_digest = hashlib.sha256()
    for image_id in sorted(completed):
        mask_set_digest.update(
            f"{image_id}:{completed[image_id]['sha256']}\n".encode("utf-8")
        )
    versions = {
        "python": sys.version.split()[0],
        "pyfing": importlib.metadata.version("pyfing"),
        "keras": importlib.metadata.version("keras"),
        "numpy": importlib.metadata.version("numpy"),
        "opencv-python-headless": importlib.metadata.version(
            "opencv-python-headless"
        ),
    }
    if args.method == "sufs":
        versions["torch"] = importlib.metadata.version("torch")
    probability_set = None
    if args.method == "sufs" and probability_root is not None:
        probability_set = probability_artifacts.set_identity(
            [
                (
                    str(row.image_id),
                    probability_artifacts.safe_path(
                        probability_root, str(row.image_id)
                    ),
                )
                for row in rows.itertuples()
            ],
            source_sha256=source_hashes,
        )
        probability_set["complete"] = len(probability_set["records"]) == len(rows)
    payload = {
        **identity,
        "schema_version": 1 if args.method == "sufs" else identity["schema_version"],
        "display_name": args.display_name,
        "prediction_count": len(completed),
        "missing_count": 0,
        "warmup_images": len(warmup_rows),
        "ms_per_img": float(timings.mean()),
        "ms_per_img_std": float(timings.std(ddof=0)),
        "parameter_count": parameter_count,
        "params_m": parameter_count / 1_000_000,
        "device": device,
        "prediction_set_sha256": mask_set_digest.hexdigest(),
        "preexisting_frozen_masks_retained": retained_preexisting_masks,
        "probability_set": probability_set,
        "kind": (
            "complete_probability_map_card" if args.method == "sufs" else None
        ),
        "complete": True if args.method == "sufs" else None,
        "seed": None if args.method == "sufs" else None,
        "task": "pixel" if args.method == "sufs" else None,
        "probabilities_are_pre_threshold": args.method == "sufs",
        "storage_format": (
            probability_artifacts.FORMAT if args.method == "sufs" else None
        ),
        "storage_dtype": (
            probability_artifacts.STORAGE_DTYPE if args.method == "sufs" else None
        ),
        "derived_from_binary_mask": False if args.method == "sufs" else None,
        "binary_mask_reinterpreted": False if args.method == "sufs" else None,
        "labels_opened": False,
        "threshold_source": "released_fixed" if args.method == "sufs" else None,
        "released_threshold_retuned": False if args.method == "sufs" else None,
        "selected_threshold": (
            float(segmenter.parameters.threshold) if args.method == "sufs" else None
        ),
        "selection_source": (
            "not_used_released_fixed" if args.method == "sufs" else None
        ),
        "threshold_fixed_before_test": args.method == "sufs",
        "validation_equivalence_images": (
            validation_equivalence_images
        ),
        "validation_equivalence_byte_identical": (
            validation_equivalence_images == min(50, len(rows))
            if args.method == "sufs" and args.split == "val"
            else None
        ),
        "complete_probability_equivalence_images": (
            complete_probability_equivalence_images
            if args.method == "sufs"
            else None
        ),
        "complete_probability_equivalence_byte_identical": (
            complete_probability_equivalence_images == len(rows)
            if args.method == "sufs"
            else None
        ),
        "mask_derivation_verification": (
            {
                "complete": complete_probability_equivalence_images == len(rows),
                "count": complete_probability_equivalence_images,
                "records_sha256": equivalence_digest,
                "all_masks_rederived_from_retained_probabilities": True,
                "all_masks_match_released_runner": True,
            }
            if args.method == "sufs"
            else None
        ),
        "cache_used_for_image_decode": inference_cache_dir is not None,
        "paper_timing_preprocess": (
            "on_the_fly_decode_and_resample" if args.split == "test" else "not_paper_timing"
        ),
        "ordered_source_images_sha256": (
            probability_set.get("ordered_source_images_sha256")
            if probability_set is not None
            else None
        ),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "command": sys.argv,
        "versions": versions,
        "provenance": provenance,
    }
    atomic_json_once(args.run_card, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune = subparsers.add_parser("tune-gmfs")
    tune.add_argument("--manifest", required=True, type=Path)
    tune.add_argument("--splits", required=True, type=Path)
    tune.add_argument("--subset", default="finger_latent")
    tune.add_argument("--mask-column", default="mask_path")
    tune.add_argument("--config", required=True, type=Path)
    tune.add_argument("--out-json", required=True, type=Path)
    tune.add_argument("--n-jobs", type=int, default=1)
    tune.add_argument("--limit", type=int, default=0)

    run = subparsers.add_parser("generate")
    run.add_argument("--method", choices=("gmfs", "sufs"), required=True)
    run.add_argument("--display-name", required=True)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--splits", required=True, type=Path)
    run.add_argument("--subset", default="finger_latent")
    run.add_argument("--split", choices=("train", "val", "test"), default="test")
    run.add_argument("--target-ppi", type=float, default=500.0)
    run.add_argument("--parameters-json", type=Path)
    run.add_argument("--output-root", required=True, type=Path)
    run.add_argument("--probability-root", type=Path)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--run-card", required=True, type=Path)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--progress-every", type=int, default=50)
    run.add_argument("--memory-fraction", type=float, default=0.3)
    run.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    provenance = run_provenance(sys.argv, capture_timing="process_start")
    args = parse_args()
    args._start_provenance = provenance
    if args.command == "tune-gmfs":
        if args.n_jobs < 1 or args.n_jobs > 64:
            raise ValueError("--n-jobs must be in 1..64")
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        tune_gmfs(args)
    else:
        if args.target_ppi != 500.0:
            raise ValueError("benchmark predictions must be generated at 500 PPI")
        if args.limit < 0 or args.warmup < 0 or args.progress_every < 1:
            raise ValueError(
                "--limit/--warmup must be non-negative and --progress-every positive"
            )
        generate(args)


if __name__ == "__main__":
    main()
