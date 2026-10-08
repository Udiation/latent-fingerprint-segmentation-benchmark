#!/usr/bin/env python3
"""Resumable mask generation with one unbiased, immutable inference card.

A completion pass repairs/skips existing masks but is never used for timing.
It is followed by an all-image timed pass (with the registered warm-up) that
must write every prediction.  Thus a resumed run cannot report timings only
for the unfinished tail.  The final card is published only after the full
prediction set and every input/checkpoint hash have been verified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import torch

import _path  # noqa: F401
from segbench.artifact_identity import (
    atomic_json_once,
    canonical_sha256,
    prediction_set_identity,
    require_clean_process_start,
    sha256_file,
)
from segbench.deep_utils import (
    examiner_exclusions_for_split,
    load_config,
    load_split_rows,
    read_normalized_image,
    run_provenance,
    verify_source_image,
)
from segbench import probability_artifacts
import infer_masks


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
    parser.add_argument("--threshold-amendment", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--card-out", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def _base_command(args: argparse.Namespace, timing_out: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("infer_masks.py")),
        "--config", str(args.config),
        "--checkpoint", str(args.checkpoint),
        "--manifest", str(args.manifest),
        "--splits", str(args.splits),
        "--subset", args.subset,
        "--split", args.split,
        "--out", str(args.out),
        "--probability-out", str(args.probability_out),
        "--timing-out", str(timing_out),
        "--device", args.device,
        "--memory-fraction", str(args.memory_fraction),
        "--warmup", str(args.warmup),
    ]
    if args.threshold_receipt is not None:
        command.extend(("--threshold-receipt", str(args.threshold_receipt)))
    command.extend(("--threshold-amendment", str(args.threshold_amendment)))
    if args.cache_dir is not None:
        command.extend(("--cache-dir", str(args.cache_dir)))
    if args.seed is not None:
        command.extend(("--seed", str(args.seed)))
    if args.limit:
        command.extend(("--limit", str(args.limit)))
    return command


def _run(command: Sequence[str]) -> None:
    completed = subprocess.run(list(command), check=False)
    if completed.returncode == 75:
        raise SystemExit(75)
    completed.check_returncode()


def _read_card(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"inference card is not an object: {path}")
    return value


def require_expected_checkpoint(path: Path, expected: str) -> str:
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("expected checkpoint digest is not lowercase SHA-256")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError("checkpoint SHA-256 differs from authorized identity")
    return observed


def validate_full_timing(report: Mapping[str, Any], expected_ids: Sequence[str]) -> None:
    """Reject partial/resume-tail timing so ms/img remains comparable."""
    if report.get("errors"):
        raise ValueError("timed inference contains errors")
    if int(report.get("skipped_valid", -1)) != 0:
        raise ValueError("timed inference skipped existing predictions")
    if int(report.get("written", -1)) != len(expected_ids):
        raise ValueError("timed inference did not write the full prediction set")
    rows = report.get("per_image")
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError("timed inference lacks one timing per expected image")
    observed = [row.get("image_id") for row in rows]
    if observed != list(expected_ids):
        raise ValueError("timing order/universe differs from the frozen split")
def _rows(args: argparse.Namespace, config: Mapping[str, Any]):
    data = config["data"]
    return load_split_rows(
        args.manifest,
        args.splits,
        subset=args.subset,
        split=args.split,
        mask_column=str(data["mask_column"]),
        require_mask=False,
        exclude_examiner_values=examiner_exclusions_for_split(data, args.split),
        limit=args.limit,
    )


def _valid_snapshot(
    args: argparse.Namespace,
    rows,
    target_ppi: float,
    config: Mapping[str, Any],
) -> Dict[str, Dict[str, str]]:
    """Hash only already-valid masks; deterministic rerun must preserve them."""
    result: Dict[str, Dict[str, str]] = {}
    for row in rows.to_dict("records"):
        image_id = str(row["image_id"])
        path = infer_masks._safe_output(args.out, image_id)
        if not path.exists():
            continue
        image = read_normalized_image(row, target_ppi, cache_dir=args.cache_dir)
        if infer_masks._valid_existing(path, image.shape):
            probability_path = probability_artifacts.safe_path(
                args.probability_out, image_id
            )
            try:
                probability_artifacts.read(
                    probability_path,
                    infer_masks._probability_shape(
                        str(config["task"]), image.shape, config
                    ),
                )
            except (FileNotFoundError, ValueError):
                continue
            result[image_id] = {
                "mask": sha256_file(path),
                "probability": sha256_file(probability_path),
            }
    return result


def run(args: argparse.Namespace, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    require_clean_process_start(
        {"provenance": provenance},
        args.expected_git_commit,
        description="inference wrapper",
    )
    for path in (args.config, args.checkpoint, args.manifest, args.splits):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.threshold_receipt is None:
        raise ValueError("production inference wrapper requires a threshold receipt")
    for path in (args.threshold_receipt, args.threshold_amendment):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.cache_dir is None or not (args.cache_dir / "receipt.json").is_file():
        raise FileNotFoundError("production inference wrapper requires a verified cache")
    checkpoint_digest = require_expected_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256
    )
    config = load_config(args.config)
    if args.probability_out is None:
        args.probability_out = args.out.parent / f"{args.out.name}_probabilities"
    rows = _rows(args, config)
    # Hash every source before either child inference pass.  Source hashing is
    # deliberately outside the measured loop; a second pass below closes the
    # time-of-check/time-of-use window before the immutable card is published.
    for row in rows.to_dict("records"):
        verify_source_image(row)
    source_records = [
        {
            "image_id": str(row["image_id"]),
            "source_sha256": str(row["source_sha256"]),
        }
        for row in rows.sort_values("image_id").to_dict("records")
    ]
    ordered_source_images_sha256 = canonical_sha256(source_records)
    expected_ids = [str(value) for value in rows["image_id"].tolist()]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("frozen inference set is empty or has duplicate ids")
    target_ppi = float(config["data"]["target_ppi"])
    args.work_dir.mkdir(parents=True, exist_ok=True)
    completion_card = args.work_dir / "completion_pass.json"
    timed_card = args.work_dir / "full_timed_pass.json"
    preexisting = _valid_snapshot(args, rows, target_ppi, config)

    # Phase 1 is resumable completion only; its timing is deliberately ignored.
    completion_command = _base_command(args, completion_card) + ["--skip-valid"]
    _run(completion_command)
    completion = _read_card(completion_card)
    require_clean_process_start(
        completion, args.expected_git_commit, description="completion inference child"
    )
    if completion.get("errors"):
        raise ValueError("completion pass contains inference errors")

    # Phase 2 always recomputes every image after a fresh warm-up.
    timed_command = _base_command(args, timed_card)
    _run(timed_command)
    timed = _read_card(timed_card)
    require_clean_process_start(
        timed, args.expected_git_commit, description="timed inference child"
    )
    validate_full_timing(timed, expected_ids)

    for row in rows.to_dict("records"):
        verify_source_image(row)

    items = []
    for row in rows.to_dict("records"):
        image_id = str(row["image_id"])
        path = infer_masks._safe_output(args.out, image_id)
        image = read_normalized_image(row, target_ppi, cache_dir=args.cache_dir)
        if not infer_masks._valid_existing(path, image.shape):
            raise ValueError(f"invalid final prediction: {image_id}")
        items.append((image_id, path))
    probability_items = []
    for row in rows.to_dict("records"):
        image_id = str(row["image_id"])
        image = read_normalized_image(row, target_ppi, cache_dir=args.cache_dir)
        path = probability_artifacts.safe_path(args.probability_out, image_id)
        probability_artifacts.read(
            path,
            infer_masks._probability_shape(str(config["task"]), image.shape, config),
        )
        probability_items.append((image_id, path))
    # The shape-validation pass above performs the final source decodes.  Bind
    # those exact bytes too, after the decode and before publishing the card.
    for row in rows.to_dict("records"):
        verify_source_image(row)
    prediction_set = prediction_set_identity(items)
    probability_set = probability_artifacts.set_identity(probability_items)
    final_hashes = {record["image_id"]: record["sha256"] for record in prediction_set["records"]}
    changed = {
        image_id: {
            "mask_before": digests["mask"],
            "mask_after": final_hashes[image_id],
            "probability_before": digests["probability"],
            "probability_after": next(
                item["sha256"]
                for item in probability_set["records"]
                if item["image_id"] == image_id
            ),
        }
        for image_id, digests in preexisting.items()
        if final_hashes.get(image_id) != digests["mask"]
        or next(
            item["sha256"]
            for item in probability_set["records"]
            if item["image_id"] == image_id
        )
        != digests["probability"]
    }
    if changed:
        raise ValueError(
            "deterministic full pass changed valid resumed masks: "
            + ", ".join(sorted(changed)[:10])
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    final = {
        "schema_version": 1,
        "kind": "complete_unbiased_inference_card",
        "method": timed["method"],
        "task": timed["task"],
        "subset": args.subset,
        "split": args.split,
        "device": timed["device"],
        "debug_limited": bool(args.limit),
        "selected_validation_threshold": timed["selected_validation_threshold"],
        "identities": {
            "source_config_sha256": sha256_file(args.config),
            "checkpoint_sha256": checkpoint_digest,
            "authorized_checkpoint_sha256": args.expected_checkpoint_sha256,
            "checkpoint_resume_config_sha256": checkpoint.get("resume_config_sha256"),
            "manifest_sha256": sha256_file(args.manifest),
            "split_sha256": sha256_file(args.splits),
            "completion_pass_card_sha256": sha256_file(completion_card),
            "full_timed_pass_card_sha256": sha256_file(timed_card),
            "prediction_set_sha256": prediction_set["sha256"],
            "prediction_count": prediction_set["count"],
            "probability_set_sha256": probability_set["sha256"],
            "probability_count": probability_set["count"],
            "ordered_source_images_sha256": ordered_source_images_sha256,
            "threshold_selector_receipt_sha256": sha256_file(
                args.threshold_receipt
            ),
            "threshold_amendment_sha256": sha256_file(
                args.threshold_amendment
            ),
            "preprocess_cache_receipt_sha256": sha256_file(
                args.cache_dir / "receipt.json"
            ),
        },
        "timing": {
            "protocol": "fresh warmup then every frozen image recomputed once",
            "warmup_images": timed["warmup_images"],
            "mean_ms_per_img": timed["mean_ms_per_img"],
            "per_image": timed["per_image"],
            "completion_pass_timing_excluded": True,
            "includes_preprocessing_and_postprocessing": True,
            "batch_size": 1,
        },
        "prediction_set": prediction_set,
        "probability_set": probability_set,
        "resume_evidence": {
            "valid_masks_before_completion": len(preexisting),
            "byte_changed_after_full_deterministic_pass": 0,
            "probability_byte_changed_after_full_deterministic_pass": 0,
        },
        "provenance": dict(provenance),
    }
    atomic_json_once(args.card_out, final)
    return final


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    print(json.dumps(run(args, provenance), sort_keys=True))


if __name__ == "__main__":
    main()
