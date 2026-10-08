#!/usr/bin/env python3
"""Produce a frozen DRUnet MC-dropout uncertainty set, never table masks.

This qualitative auxiliary job is deliberately separate from deterministic
benchmark inference and timing.  It consumes a GT-blind plan containing an
exact 20-image universe and writes only float32 NumPy variance maps plus one
immutable completion card.  Ground-truth paths are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

import _path  # noqa: F401
from segbench.artifact_identity import (
    atomic_json,
    atomic_json_once,
    canonical_sha256,
    require_clean_process_start,
    sha256_file,
)
from segbench.deep_models import build_model
from segbench.deep_utils import (
    configure_memory_fraction,
    examiner_exclusions_for_split,
    load_config,
    load_split_rows,
    read_normalized_image,
    resume_config_digest,
    resolve_device,
    run_provenance,
    seed_everything,
    verify_source_image,
)
import infer_masks


MC_DROPOUT_PASSES = 20
MC_DROPOUT_IMAGE_COUNT = 20


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _require_sha(value: str, description: str, length: int = 64) -> str:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{description} is not lowercase {length}-hex")
    return value


def _variance_set_identity(paths: Mapping[str, Path]) -> Dict[str, Any]:
    records = []
    for image_id in sorted(paths):
        path = paths[image_id]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        records.append(
            {
                "image_id": image_id,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    return {
        "count": len(records),
        "records": records,
        "sha256": canonical_sha256(records),
    }


def _exact_variance_files(root: Path, image_ids: Sequence[str]) -> Dict[str, Path]:
    expected = {
        image_id: infer_masks._safe_variance_output(root, image_id)
        for image_id in image_ids
    }
    observed = sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []
    if observed != sorted(expected.values()):
        raise ValueError("auxiliary variance directory has missing or extra files")
    return expected


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--variance-out", required=True, type=Path)
    parser.add_argument("--card-out", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    return parser.parse_args(argv)


def run(args: argparse.Namespace, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    require_clean_process_start(
        {"provenance": provenance},
        args.expected_git_commit,
        description="DRUnet MC-dropout auxiliary inference",
    )
    expected_plan = _require_sha(args.expected_plan_sha256, "expected plan digest")
    if sha256_file(args.plan) != expected_plan:
        raise ValueError("DRUnet auxiliary plan differs from registered bytes")
    plan = _read_json(args.plan, "DRUnet auxiliary plan")
    claims = {
        "schema_version": 1,
        "kind": "stage3_drunet_mc_dropout_plan",
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "timing_excluded": True,
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "mask_outputs_forbidden": True,
    }
    if any(plan.get(key) != value for key, value in claims.items()):
        raise ValueError("DRUnet auxiliary plan claims changed")
    image_ids = plan.get("test_image_ids")
    if (
        not isinstance(image_ids, list)
        or image_ids != sorted(image_ids)
        or len(image_ids) != MC_DROPOUT_IMAGE_COUNT
        or len(set(image_ids)) != MC_DROPOUT_IMAGE_COUNT
        or any(not isinstance(value, str) or not value for value in image_ids)
        or plan.get("image_id_set_sha256") != canonical_sha256(image_ids)
    ):
        raise ValueError("DRUnet auxiliary image universe changed")
    expected_shapes = plan.get("expected_shapes")
    if (
        not isinstance(expected_shapes, dict)
        or set(expected_shapes) != set(image_ids)
        or any(
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
            for shape in expected_shapes.values()
        )
    ):
        raise ValueError("DRUnet auxiliary expected shapes changed")
    run = plan.get("run")
    if not isinstance(run, dict) or run.get("method") != "drunet" or int(run.get("seed", -1)) != args.seed:
        raise ValueError("DRUnet auxiliary run identity changed")
    identity_paths = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "splits": args.splits,
    }
    for name, path in identity_paths.items():
        expected = plan.get("identities", {}).get(name)
        if not isinstance(expected, dict) or Path(str(expected.get("path", ""))).resolve() != path.resolve():
            raise ValueError(f"DRUnet auxiliary {name} path changed")
        if expected.get("sha256") != sha256_file(path):
            raise ValueError(f"DRUnet auxiliary {name} bytes changed")
    checkpoint_sha = _require_sha(
        args.expected_checkpoint_sha256, "expected checkpoint digest"
    )
    if checkpoint_sha != run.get("checkpoint_sha256") or sha256_file(args.checkpoint) != checkpoint_sha:
        raise ValueError("DRUnet auxiliary checkpoint changed")
    if str(args.variance_out.resolve()) != str(Path(str(plan.get("variance_out", ""))).resolve()):
        raise ValueError("DRUnet auxiliary variance output changed")
    if str(args.card_out.resolve()) != str(Path(str(plan.get("card_out", ""))).resolve()):
        raise ValueError("DRUnet auxiliary card output changed")
    if str(args.work_dir.resolve()) != str(Path(str(plan.get("work_dir", ""))).resolve()):
        raise ValueError("DRUnet auxiliary work directory changed")
    if run.get("expected_inference_git_commit") != args.expected_git_commit:
        raise ValueError("DRUnet auxiliary inference commit changed")
    if args.card_out.exists():
        existing = _read_json(args.card_out, "existing DRUnet auxiliary card")
        exact = _exact_variance_files(args.variance_out, image_ids)
        require_clean_process_start(
            existing,
            args.expected_git_commit,
            description="existing DRUnet MC-dropout completion card",
        )
        if (
            existing.get("schema_version") != 1
            or existing.get("kind")
            != "stage3_drunet_mc_dropout_completion_card"
            or existing.get("method") != "drunet"
            or int(existing.get("seed", -1)) != args.seed
            or existing.get("plan_sha256") != expected_plan
            or existing.get("ground_truth_opened") is not False
            or existing.get("mask_outputs_written") is not False
            or existing.get("timing_excluded") is not True
            or existing.get("mc_dropout_passes") != MC_DROPOUT_PASSES
            or existing.get("image_id_set_sha256") != canonical_sha256(image_ids)
        ):
            raise FileExistsError("existing DRUnet auxiliary card identity changed")
        if existing.get("variance_set") != _variance_set_identity(exact):
            raise ValueError("existing DRUnet auxiliary variance set changed")
        for name, path in identity_paths.items():
            if existing.get("identities", {}).get(name) != {
                "path": str(path),
                "sha256": sha256_file(path),
            }:
                raise ValueError(f"existing DRUnet auxiliary {name} identity changed")
        return existing

    config = load_config(args.config)
    if config.get("method") != "drunet" or config.get("task") != "pixel":
        raise ValueError("auxiliary inference requires the native DRUnet pixel model")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("checkpoint lacks its resolved training configuration")
    resolved = dict(config)
    resolved["seed"] = args.seed
    resolved["runtime"] = dict(resolved["runtime"])
    if "training_data_identity" in checkpoint_config:
        resolved["training_data_identity"] = checkpoint_config["training_data_identity"]
    checkpoint_runtime = checkpoint_config.get("runtime", {})
    for key in (
        "acceleration",
        "queue_batch_contract",
        "queue_batch_contract_sha256",
    ):
        if key in checkpoint_runtime:
            resolved["runtime"][key] = checkpoint_runtime[key]
    if checkpoint.get("resume_config_sha256") != resume_config_digest(resolved):
        raise ValueError("checkpoint and supplied auxiliary configuration differ")
    if checkpoint.get("method") != "drunet" or checkpoint.get("task") != "pixel":
        raise ValueError("checkpoint is not a DRUnet pixel model")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("source_split") != "train":
        raise ValueError("checkpoint lacks train-only normalization metadata")

    data = resolved["data"]
    rows = load_split_rows(
        args.manifest,
        args.splits,
        subset="finger_latent",
        split="test",
        mask_column=str(data["mask_column"]),
        require_mask=False,
        exclude_examiner_values=examiner_exclusions_for_split(data, "test"),
    )
    by_id = {str(row["image_id"]): row for row in rows.to_dict("records")}
    if set(image_ids) - set(by_id):
        raise ValueError("DRUnet auxiliary plan contains unknown held-out IDs")
    selected = [by_id[image_id] for image_id in image_ids]
    source_records = []
    for row in selected:
        source_sha = verify_source_image(row)
        source_records.append(
            {"image_id": str(row["image_id"]), "source_sha256": source_sha}
        )
    if canonical_sha256(source_records) != plan.get("ordered_source_images_sha256"):
        raise ValueError("DRUnet auxiliary source-image bytes changed")

    resolved["runtime"] = dict(resolved["runtime"])
    resolved["runtime"]["device"] = args.device
    resolved["runtime"]["memory_fraction"] = args.memory_fraction
    seed_everything(args.seed, bool(resolved["runtime"].get("deterministic", True)))
    device = resolve_device(str(resolved["runtime"]["device"]))
    configure_memory_fraction(device, float(resolved["runtime"]["memory_fraction"]))
    model = build_model("drunet", resolved["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.variance_out.mkdir(parents=True, exist_ok=True)
    progress_path = args.work_dir / "variance_progress.json"
    progress: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "stage3_drunet_mc_dropout_progress",
        "plan_sha256": expected_plan,
        "checkpoint_sha256": checkpoint_sha,
        "ordered_source_images_sha256": canonical_sha256(source_records),
        "records": {},
    }
    if progress_path.exists():
        observed_progress = _read_json(
            progress_path, "DRUnet MC-dropout progress journal"
        )
        expected_header = {
            key: progress[key]
            for key in (
                "schema_version", "kind", "plan_sha256", "checkpoint_sha256",
                "ordered_source_images_sha256",
            )
        }
        if any(observed_progress.get(key) != value for key, value in expected_header.items()):
            raise ValueError("DRUnet auxiliary progress belongs to another run")
        observed_records = observed_progress.get("records")
        if not isinstance(observed_records, dict) or set(observed_records) - set(image_ids):
            raise ValueError("DRUnet auxiliary progress universe changed")
        progress = observed_progress
    for row in selected:
        if infer_masks.preempt_requested():
            raise SystemExit(infer_masks.PREEMPT_EXIT)
        image_id = str(row["image_id"])
        image = read_normalized_image(row, float(data["target_ppi"]))
        if list(image.shape) != expected_shapes[image_id]:
            raise ValueError(f"DRUnet auxiliary target shape changed: {image_id}")
        output = infer_masks._safe_variance_output(args.variance_out, image_id)
        progress_record = progress["records"].get(image_id)
        if isinstance(progress_record, dict) and (
            progress_record.get("sha256") == (sha256_file(output) if output.is_file() else None)
            and progress_record.get("shape") == list(image.shape)
            and progress_record.get("dtype") == "float32"
            and infer_masks._valid_existing_variance(output, image.shape)
        ):
            continue
        # A shape-valid file without an authenticated progress entry is stale
        # by definition and is deterministically recomputed, never adopted.
        stable_seed = (
            args.seed
            + int.from_bytes(
                hashlib.sha256(image_id.encode("utf-8")).digest()[:8],
                "big",
            )
        ) % (2**63 - 1)
        variance = infer_masks._mc_dropout_variance(
            model,
            image,
            resolved,
            normalization,
            device,
            MC_DROPOUT_PASSES,
            stable_seed,
        )
        if variance.shape != image.shape or variance.dtype != np.float32 or not np.isfinite(variance).all():
            raise RuntimeError(f"invalid DRUnet variance map: {image_id}")
        infer_masks._atomic_npy(output, variance)
        progress["records"][image_id] = {
            "sha256": sha256_file(output),
            "shape": list(variance.shape),
            "dtype": str(variance.dtype),
        }
        atomic_json(progress_path, progress)

    exact = _exact_variance_files(args.variance_out, image_ids)
    for image_id, path in exact.items():
        image = read_normalized_image(by_id[image_id], float(data["target_ppi"]))
        if list(image.shape) != expected_shapes[image_id]:
            raise ValueError(f"DRUnet auxiliary final shape changed: {image_id}")
        if not infer_masks._valid_existing_variance(path, image.shape):
            raise ValueError(f"invalid final DRUnet variance map: {image_id}")
    post_source_records = []
    for row in selected:
        post_source_records.append(
            {
                "image_id": str(row["image_id"]),
                "source_sha256": verify_source_image(row),
            }
        )
    if post_source_records != source_records:
        raise ValueError("DRUnet auxiliary source images changed during inference")
    variance_set = _variance_set_identity(exact)
    card = {
        "schema_version": 1,
        "kind": "stage3_drunet_mc_dropout_completion_card",
        "method": "drunet",
        "seed": args.seed,
        "plan_sha256": expected_plan,
        "ground_truth_opened": False,
        "mask_outputs_written": False,
        "timing_excluded": True,
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "image_id_set_sha256": canonical_sha256(image_ids),
        "ordered_source_images_sha256": canonical_sha256(source_records),
        "variance_set": variance_set,
        "identities": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in identity_paths.items()
        },
        "provenance": dict(provenance),
    }
    atomic_json_once(args.card_out, card)
    return card


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    print(json.dumps(run(args, provenance), sort_keys=True))


if __name__ == "__main__":
    main()
