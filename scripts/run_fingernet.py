#!/usr/bin/env python3
"""Generate official FingerNet segmentation masks without reading ground truth.

This is an inference-only zero-shot workflow.  Selection and scoring remain in
the common benchmark tools; this command only writes binary 500-PPI PNG masks
and a provenance-rich restricted run card.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import torch
import yaml

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once
from segbench.deep_utils import run_provenance
from segbench import probability_artifacts, preprocess_cache
from segbench.fingernet_legacy import (
    FULL_RELEASE_PARAMETER_COUNT,
    RELEASE_COMMIT,
    RELEASE_WEIGHTS_SHA256,
    SEGMENTATION_GRAPH_PARAMETER_COUNT,
    LegacyFingerNetSegmenter,
    audit_release_weights,
    sha256_file,
)
from segbench.io import attach_split, load_manifest, manifest_target_shape, read_gray, to_target_ppi


CHECKPOINT_SCHEMA_VERSION = 1
SOURCE_FILE_HASHES = {
    "README.md": "bc104234719afabd115c56ebe905393edb64833447f63fcad41e207e92f1ec44",
    "src/train_test_deploy.py": "ca43702763c3edf1ec66379a3af404f14c33785f0bf5b8ebc6a4232ae29b2003",
    "src/utils.py": "b338956d8f2afba66341c29b62a35fd7e405a10f1c7ba8216d646f58336cb51e",
}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through an atomic sibling rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_mask(path: Path, mask: np.ndarray) -> None:
    """Write one binary mask through an atomic sibling rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.stem + ".partial.png")
    encoded = np.where(mask, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(temporary), encoded):
        raise OSError(f"failed to write {temporary}")
    os.replace(temporary, path)


def validate_config(path: Path) -> Dict[str, Any]:
    """Load the frozen zero-shot configuration and reject silent deviations."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "target_ppi": 500.0,
        "weights_sha256": RELEASE_WEIGHTS_SHA256,
        "source_commit": RELEASE_COMMIT,
        "batch_size": 1,
        "coarse_stride": 8,
        "threshold": 0.5,
        "dimension_policy": "floor_to_stride_then_background_fill",
        "operation": "opening",
        "kernel": [5, 5],
    }
    actual = {
        "target_ppi": float(config["target_ppi"]),
        "weights_sha256": config["released_weights"]["sha256"],
        "source_commit": config["source"]["commit"],
        "batch_size": int(config["inference"]["batch_size"]),
        "coarse_stride": int(config["inference"]["coarse_stride"]),
        "threshold": float(config["inference"]["threshold"]),
        "dimension_policy": config["inference"]["dimension_policy"],
        "operation": config["inference"]["postprocess"]["operation"],
        "kernel": list(config["inference"]["postprocess"]["kernel"]),
    }
    if actual != expected:
        raise ValueError(f"FingerNet config deviates from the released workflow: {actual}")
    return config


def audit_source_repository(path: Path) -> Dict[str, Any]:
    """Verify the pinned archival checkout and the original source file bytes."""
    if not (path / ".git").is_dir():
        raise FileNotFoundError(f"not a git checkout: {path}")
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != RELEASE_COMMIT:
        raise ValueError(f"FingerNet source is at {head}, expected {RELEASE_COMMIT}")
    hashes = {}
    for relative, expected in SOURCE_FILE_HASHES.items():
        digest = sha256_file(path / relative)
        if digest != expected:
            raise ValueError(f"archived FingerNet source hash mismatch: {relative}")
        hashes[relative] = digest
    return {
        "paper_repository": "https://github.com/felixTY/FingerNet",
        "paper_repository_status_at_audit": "not_found",
        "archival_repository": "https://github.com/592692070/FingerNet",
        "commit": head,
        "source_sha256": hashes,
        "license_evidence": (
            "README declares MIT; the preserved checkout contains no LICENSE file"
        ),
    }


def load_rows(
    manifest_path: Path,
    split_path: Path,
    subset: str,
    split: str,
    limit: int,
):
    """Load deterministic inference rows without consulting any GT column."""
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    frame = attach_split(load_manifest(manifest_path), payload["splits"])
    frame = frame[(frame["subset"] == subset) & (frame["split"] == split)].copy()
    if split == "test" and "examiner_value" in frame.columns:
        frame = frame[frame["examiner_value"].astype(str).str.upper() != "NONPRINT"]
    frame = frame.sort_values("image_id", kind="stable").reset_index(drop=True)
    if limit:
        frame = frame.head(limit).copy()
    if frame.empty:
        raise ValueError(f"no rows for subset={subset!r}, split={split!r}")
    return frame


def read_image_500(
    row: Mapping[str, Any], target_ppi: float, cache_dir: Path | None = None
) -> np.ndarray:
    """Decode and resolution-normalize one image before timed inference."""
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


def configure_device(requested: str, memory_fraction: float) -> str:
    """Configure CPU or one queue-restricted logical CUDA device."""
    if not 0.0 < memory_fraction <= 0.3:
        raise ValueError("CUDA memory fraction must be in (0, 0.3]")
    if requested == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        return "cpu"
    if requested != "cuda":
        raise ValueError("--device must be cpu or cuda")
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if len(visible) != 1:
        raise RuntimeError("CUDA inference requires exactly one queue-assigned visible device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but is unavailable")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
    return "cuda"


def validate_existing_mask(path: Path, shape: Tuple[int, int], digest: str) -> bool:
    """Accept a resumed output only when shape, encoding, and digest match."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (
        mask is not None
        and mask.shape == shape
        and set(np.unique(mask).tolist()).issubset({0, 255})
        and sha256_file(path) == digest
    )


def generate(args: argparse.Namespace) -> None:
    if args.run_card.is_file():
        raise ValueError(
            "immutable complete FingerNet run card already exists; choose a new output"
        )
    """Generate masks and a restricted run card, without reading target masks."""
    provenance = getattr(args, "_start_provenance", None) or run_provenance(sys.argv)
    config = validate_config(args.config)
    source = audit_source_repository(args.source_repo)
    release = audit_release_weights(args.weights)
    memory_fraction = float(
        os.environ.get(
            "SEG_GPU_MEMORY_FRACTION",
            config["inference"]["cuda_memory_fraction"],
        )
    )
    device = configure_device(args.device, memory_fraction)
    cv2.setNumThreads(1)
    segmenter = LegacyFingerNetSegmenter(args.weights, device=device)
    rows = load_rows(args.manifest, args.splits, args.subset, args.split, args.limit)
    cache_dir = getattr(args, "cache_dir", None)
    probability_root_arg = getattr(args, "probability_root", None)
    cache_receipt_sha256 = None
    if cache_dir is not None:
        preprocess_cache.load_receipt(
            cache_dir,
            manifest_sha256=sha256_file(args.manifest),
            target_ppi=float(config["target_ppi"]),
            mask_columns=[],
            required_ids=[str(value) for value in rows["image_id"]],
            verify_artifact_hashes=True,
        )
        cache_receipt_sha256 = sha256_file(cache_dir / "receipt.json")
    inference_cache_dir = cache_dir if args.split != "test" else None
    source_hashes = probability_artifacts.source_hash_map(rows.to_dict("records"))
    probability_root = probability_root_arg or (
        args.output_root.parent / f"{args.output_root.name}_probabilities"
    )
    probability_root.mkdir(parents=True, exist_ok=True)

    implementation = {
        "runner_sha256": sha256_file(Path(__file__)),
        "compatibility_sha256": sha256_file(Path(inspect.getfile(LegacyFingerNetSegmenter))),
        "config_sha256": sha256_file(args.config),
        "source": source,
        "release_weights": release,
    }
    identity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": "fingernet",
        "implementation": implementation,
        "manifest_sha256": sha256_file(args.manifest),
        "splits_sha256": sha256_file(args.splits),
        # Probability-card validators use the singular field; retain the
        # original plural field for compatibility with run checkpoints.
        "split_sha256": sha256_file(args.splits),
        "subset": args.subset,
        "split": args.split,
        "target_ppi": float(config["target_ppi"]),
        "limit": args.limit,
        "device": segmenter.device_label,
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
    }
    checkpoint_path = args.run_card.with_suffix(".checkpoint.json")
    completed: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("FingerNet checkpoint identity disagrees with this run")
        completed = {
            str(key): {
                "ms": float(value["ms"]),
                "sha256": str(value["sha256"]),
                "probability_sha256": value.get("probability_sha256"),
            }
            for key, value in checkpoint.get("completed", {}).items()
        }
        completed = probability_artifacts.authenticated_resume_records(
            [
                {"image_id": image_id, **record}
                for image_id, record in completed.items()
            ],
            allowed_ids=[str(value) for value in rows["image_id"]],
            timing_key="ms",
        )

    warmup = int(config["inference"]["warmup_images"])
    for row in rows.head(min(warmup, len(rows))).to_dict("records"):
        prediction, probability = segmenter.run_with_probability(
            read_image_500(row, float(config["target_ppi"]), inference_cache_dir)
        )
        probability = probability_artifacts.storage_roundtrip(probability)
        if prediction.ndim != 2:
            raise ValueError("FingerNet warm-up produced a non-2-D mask")
        if probability.shape != prediction.shape:
            raise ValueError("FingerNet warm-up probability shape mismatch")
        if not np.array_equal(
            prediction, segmenter.mask_from_probability(probability)
        ):
            raise RuntimeError(
                "float16 FingerNet probability cannot reproduce the released mask"
            )

    errors = []
    started = time.perf_counter()
    retained_preexisting_masks = 0
    for index, row in enumerate(rows.to_dict("records"), start=1):
        image_id = str(row["image_id"])
        target = args.output_root / f"{image_id}.png"
        probability_path = probability_artifacts.safe_path(
            probability_root, image_id
        )
        declared = manifest_target_shape(row, float(config["target_ppi"]))
        if image_id in completed:
            shape = declared or read_image_500(
                row, float(config["target_ppi"]), inference_cache_dir
            ).shape
            probability_valid = False
            if probability_path.is_file():
                try:
                    probability_artifacts.read(probability_path, shape)
                    probability_valid = (
                        sha256_file(probability_path)
                        == completed[image_id].get("probability_sha256")
                    )
                except ValueError:
                    probability_valid = False
            if validate_existing_mask(
                target, shape, completed[image_id]["sha256"]
            ) and probability_valid:
                continue
            completed.pop(image_id)
        try:
            before = time.perf_counter()
            image = read_image_500(
                row, float(config["target_ppi"]), inference_cache_dir
            )
            released_prediction, probability = segmenter.run_with_probability(image)
            probability = probability_artifacts.storage_roundtrip(probability)
            prediction = segmenter.mask_from_probability(probability)
            if not np.array_equal(prediction, released_prediction):
                raise RuntimeError(
                    "float16 FingerNet probability cannot reproduce released output"
                )
            elapsed_ms = (time.perf_counter() - before) * 1000.0
            if args.split == "val" and index <= 50:
                official = segmenter.run(image)
                if not np.array_equal(prediction, official):
                    raise RuntimeError(
                        "FingerNet raw probability does not reproduce released val mask"
                    )
            if target.is_file():
                frozen = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
                if (
                    frozen is None
                    or frozen.shape != prediction.shape
                    or not np.array_equal(frozen > 0, prediction)
                ):
                    raise RuntimeError(
                        f"{image_id}: pre-existing frozen FingerNet prediction differs"
                    )
                retained_preexisting_masks += 1
            else:
                atomic_mask(target, prediction)
            probability_artifacts.atomic_write(probability_path, probability)
            completed[image_id] = {
                "ms": elapsed_ms,
                "sha256": sha256_file(target),
                "probability_sha256": sha256_file(probability_path),
            }
        except Exception as exc:
            errors.append({"image_id": image_id, "error": repr(exc)})
        if index % args.progress_every == 0 or index == len(rows):
            atomic_json(
                checkpoint_path,
                {"identity": identity, "completed": completed, "errors": errors},
            )
            print(
                f"fingernet: {index}/{len(rows)} complete={len(completed)} errors={len(errors)}",
                flush=True,
            )
    if errors:
        raise RuntimeError(f"{len(errors)} FingerNet errors; inspect {checkpoint_path}")
    if len(completed) != len(rows):
        raise RuntimeError(f"completed {len(completed)} masks for {len(rows)} rows")

    # Re-run the released implementation for every retained artifact, including
    # after checkpoint resume.  This proves both the float16-map derivation and
    # released-runner equivalence over the complete delivered universe.
    validation_equivalence_images = 0
    complete_probability_equivalence_images = 0
    probability_equivalence_records: List[Dict[str, str]] = []
    for audit_index, row in enumerate(rows.to_dict("records"), start=1):
        image_id = str(row["image_id"])
        image = read_image_500(
            row, float(config["target_ppi"]), inference_cache_dir
        )
        stored_path = probability_artifacts.safe_path(probability_root, image_id)
        stored = probability_artifacts.read(
            stored_path,
            image.shape,
        )
        reconstructed = segmenter.mask_from_probability(stored)
        released = segmenter.run(image)
        saved = cv2.imread(
            str(args.output_root / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE
        )
        if saved is None or not np.array_equal(reconstructed, saved > 0):
            raise RuntimeError(
                "stored FingerNet probability does not reproduce its saved mask"
            )
        if not np.array_equal(reconstructed, released):
            raise RuntimeError(
                "stored FingerNet probability does not reproduce released output"
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
    equivalence_digest = hashlib.sha256(
        json.dumps(
            probability_equivalence_records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    timings = np.asarray(
        [completed[str(row.image_id)]["ms"] for row in rows.itertuples()], dtype=float
    )
    prediction_digest = hashlib.sha256()
    for image_id in sorted(completed):
        prediction_digest.update(
            f"{image_id}:{completed[image_id]['sha256']}\n".encode("utf-8")
        )
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
        "schema_version": 1,
        "display_name": config["display_name"],
        "prediction_count": len(completed),
        "missing_count": 0,
        "warmup_images": min(warmup, len(rows)),
        "batch_size": 1,
        "ms_per_img": float(timings.mean()),
        "ms_per_img_std": float(timings.std(ddof=0)),
        "parameter_count": FULL_RELEASE_PARAMETER_COUNT,
        "params_m": FULL_RELEASE_PARAMETER_COUNT / 1_000_000,
        "executed_segmentation_graph_parameter_count": SEGMENTATION_GRAPH_PARAMETER_COUNT,
        "device": segmenter.device_label,
        "prediction_set_sha256": prediction_digest.hexdigest(),
        "preexisting_frozen_masks_retained": retained_preexisting_masks,
        "probability_set": probability_set,
        "kind": "complete_probability_map_card",
        "complete": True,
        "seed": None,
        "task": "pixel",
        "probabilities_are_pre_threshold": True,
        "storage_format": probability_artifacts.FORMAT,
        "storage_dtype": probability_artifacts.STORAGE_DTYPE,
        "derived_from_binary_mask": False,
        "binary_mask_reinterpreted": False,
        "labels_opened": False,
        "threshold_source": "released_fixed",
        "released_threshold_retuned": False,
        "selected_threshold": float(config["inference"]["threshold"]),
        "selection_source": "not_used_released_fixed",
        "threshold_fixed_before_test": True,
        "validation_equivalence_images": (
            validation_equivalence_images
        ),
        "validation_equivalence_byte_identical": (
            validation_equivalence_images == min(50, len(rows))
            if args.split == "val"
            else None
        ),
        "complete_probability_equivalence_images": (
            complete_probability_equivalence_images
        ),
        "complete_probability_equivalence_byte_identical": (
            complete_probability_equivalence_images == len(rows)
        ),
        "mask_derivation_verification": {
            "complete": complete_probability_equivalence_images == len(rows),
            "count": complete_probability_equivalence_images,
            "records_sha256": equivalence_digest,
            "all_masks_rederived_from_retained_probabilities": True,
            "all_masks_match_released_runner": True,
        },
        "cache_used_for_image_decode": inference_cache_dir is not None,
        "paper_timing_preprocess": (
            "on_the_fly_decode_and_resample" if args.split == "test" else "not_paper_timing"
        ),
        "ordered_source_images_sha256": probability_set.get(
            "ordered_source_images_sha256"
        ),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "command": sys.argv,
        "training_data_disclosure": (
            "Authors report CISL24218, a private mixture of rolled and latent prints; "
            "the released README does not identify SD302 as training data"
        ),
        "versions": {
            "python": sys.version.split()[0],
            "keras": importlib.metadata.version("keras"),
            "torch": importlib.metadata.version("torch"),
            "numpy": importlib.metadata.version("numpy"),
            "opencv-python-headless": importlib.metadata.version("opencv-python-headless"),
            "h5py": importlib.metadata.version("h5py"),
        },
        "provenance": provenance,
    }
    atomic_json_once(args.run_card, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    """Parse the zero-shot generation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--subset", default="finger_latent")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--probability-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--run-card", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.limit < 0 or args.progress_every < 1:
        parser.error("--limit must be non-negative and --progress-every positive")
    return args


if __name__ == "__main__":
    start_provenance = run_provenance(sys.argv, capture_timing="process_start")
    parsed_args = parse_args()
    parsed_args._start_provenance = start_provenance
    generate(parsed_args)
