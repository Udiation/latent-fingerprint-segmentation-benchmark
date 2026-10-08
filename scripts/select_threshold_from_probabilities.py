#!/usr/bin/env python3
"""Freeze a validation-only threshold selection from saved probability maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once, sha256_file
from segbench import probability_artifacts, preprocess_cache
from segbench.deep_utils import (
    PROTOCOL_THRESHOLDS,
    binary_dice,
    blocks_to_postprocessed_mask,
    load_config,
    load_split_rows,
    read_pair,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--probability-dir", required=True, type=Path)
    parser.add_argument("--probability-card", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--selected-probability-card-out", type=Path)
    parser.add_argument("--threshold-amendment", type=Path)
    parser.add_argument("--allow-unamended-debug", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def _probability_set(card: Mapping[str, Any]) -> Mapping[str, Any]:
    value = card.get("probability_set")
    if not isinstance(value, Mapping):
        raise ValueError("probability card lacks a probability_set")
    if value.get("format") != probability_artifacts.FORMAT:
        raise ValueError("probability-card format is not float16_npy")
    if value.get("storage_dtype") != probability_artifacts.STORAGE_DTYPE:
        raise ValueError("probability-card storage dtype is not float16")
    if value.get("complete") is not True:
        raise ValueError("probability-card set is not marked complete")
    return value


def select(args: argparse.Namespace) -> Dict[str, Any]:
    amendment_path = getattr(args, "threshold_amendment", None)
    amendment_sha256 = None
    if amendment_path is not None:
        amendment_sha256 = probability_artifacts.validate_threshold_amendment(
            amendment_path
        )
    elif not (getattr(args, "allow_unamended_debug", False) and args.limit > 0):
        raise ValueError(
            "production threshold selection requires the exact threshold amendment"
        )
    config = load_config(
        args.config,
        allow_amendment_legacy_thresholds=amendment_sha256 is not None,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != config["method"] or checkpoint.get("task") != config["task"]:
        raise ValueError("checkpoint and configuration model identities differ")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping) or "seed" not in checkpoint_config:
        raise ValueError("checkpoint lacks its exact training seed")
    checkpoint_seed = int(checkpoint_config["seed"])
    card = json.loads(args.probability_card.read_text(encoding="utf-8"))
    expected_card_invariants = {
        "status": "complete",
        "complete": True,
        "kind": "complete_probability_map_card",
        "probabilities_are_pre_threshold": True,
        "storage_format": probability_artifacts.FORMAT,
        "storage_dtype": probability_artifacts.STORAGE_DTYPE,
        "derived_from_binary_mask": False,
        "binary_mask_reinterpreted": False,
        "labels_opened": False,
        "selection_source": "unselected_raw",
    }
    invariant_disagreements = {
        key: (card.get(key), value)
        for key, value in expected_card_invariants.items()
        if card.get(key) != value
    }
    if invariant_disagreements:
        raise ValueError(
            f"probability card fails threshold-selection invariants: {invariant_disagreements}"
        )
    if card.get("split") != "val":
        raise ValueError("threshold selection accepts validation probabilities only")
    if card.get("method") != config["method"] or card.get("task") != config["task"]:
        raise ValueError("probability card method/task differs from configuration")
    if card.get("seed") != checkpoint_seed:
        raise ValueError("probability card seed differs from checkpoint")
    if amendment_path is not None:
        if card.get("threshold_amendment_sha256") != amendment_sha256:
            raise ValueError("probability card is not bound to the supplied amendment")
    identities = card.get("identities")
    expected_identities = {
        "source_config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest_sha256": sha256_file(args.manifest),
        "split_sha256": sha256_file(args.splits),
    }
    if not isinstance(identities, Mapping) or any(
        identities.get(key) != value for key, value in expected_identities.items()
    ):
        raise ValueError("probability card provenance differs from selection inputs")
    data = config["data"]
    rows = load_split_rows(
        args.manifest,
        args.splits,
        subset=args.subset,
        split="val",
        mask_column=str(data["mask_column"]),
        require_mask=True,
        limit=args.limit,
    )
    cache_receipt_sha256 = None
    if args.cache_dir is not None:
        preprocess_cache.load_receipt(
            args.cache_dir,
            manifest_sha256=sha256_file(args.manifest),
            target_ppi=float(data["target_ppi"]),
            mask_columns=[str(data["mask_column"])],
            required_ids=[str(value) for value in rows["image_id"]],
            verify_artifact_hashes=True,
        )
        cache_receipt_sha256 = sha256_file(args.cache_dir / "receipt.json")
    if card.get("preprocess_cache_receipt_sha256") != cache_receipt_sha256:
        raise ValueError("probability card cache identity differs from selection inputs")
    probability_set = _probability_set(card)
    records = probability_set.get("records")
    if not isinstance(records, list):
        raise ValueError("probability card lacks artifact records")
    by_id = {
        str(item.get("image_id")): item
        for item in records
        if isinstance(item, Mapping) and isinstance(item.get("image_id"), str)
    }
    expected_ids = [str(value) for value in rows["image_id"].tolist()]
    if set(by_id) != set(expected_ids) or len(by_id) != len(records):
        raise ValueError("validation probability universe differs from frozen val rows")
    if probability_set.get("count") != len(expected_ids):
        raise ValueError("validation probability count differs from frozen val rows")
    canonical_records = [
        {
            "image_id": str(item["image_id"]),
            "sha256": item.get("sha256"),
            "bytes": item.get("bytes"),
        }
        for item in sorted(records, key=lambda value: str(value["image_id"]))
    ]
    if probability_set.get("sha256") != probability_artifacts.canonical_sha256(
        canonical_records
    ):
        raise ValueError("probability-card set identity is invalid")
    source_hashes = probability_artifacts.source_hash_map(rows.to_dict("records"))
    entries: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows.to_dict("records"):
        image_id = str(row["image_id"])
        path = probability_artifacts.safe_path(args.probability_dir, image_id)
        record = by_id[image_id]
        if (
            path.name != record.get("filename")
            or str(path.resolve()) != record.get("path")
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
            or record.get("source_sha256") != source_hashes[image_id]
        ):
            raise ValueError(f"validation probability bytes changed: {image_id}")
        _, target = read_pair(
            row,
            float(data["target_ppi"]),
            str(data["mask_column"]),
            cache_dir=args.cache_dir,
        )
        probability = probability_artifacts.read(path)
        if config["task"] == "pixel" and probability.shape != target.shape:
            raise ValueError(f"pixel probability shape mismatch: {image_id}")
        entries.append((probability, target))
    candidates = sorted(
        float(value)
        for value in (
            PROTOCOL_THRESHOLDS
            if amendment_sha256 is not None
            else config["training"]["threshold_candidates"]
        )
    )
    scores: Dict[str, float] = {}
    for threshold in candidates:
        dice = []
        for probability, target in entries:
            if config["task"] == "pixel":
                prediction = probability >= threshold
            else:
                prediction = blocks_to_postprocessed_mask(
                    probability,
                    target.shape,
                    threshold=threshold,
                    min_component_blocks=int(
                        config["inference"]["min_component_blocks"]
                    ),
                )
            dice.append(binary_dice(prediction, target))
        scores[f"{threshold:.8g}"] = float(np.mean(dice))
    selected, selected_score = max(
        ((float(key), score) for key, score in scores.items()),
        key=lambda item: (item[1], -abs(item[0] - 0.5), -item[0]),
    )
    selected_card = dict(card)
    selected_card.update(
        {
            "schema_version": 1,
            "kind": "complete_probability_map_card",
            "status": "complete",
            "complete": True,
            "selection_source": "validation_only",
            "threshold_scores": scores,
            "selected_threshold": selected,
            "selected_validation_threshold": selected,
            "threshold_candidates": candidates,
            "threshold_amendment_sha256": amendment_sha256,
            "probabilities_are_pre_threshold": True,
            "storage_format": probability_artifacts.FORMAT,
            "storage_dtype": probability_artifacts.STORAGE_DTYPE,
            "derived_from_binary_mask": False,
            "binary_mask_reinterpreted": False,
            "labels_opened": True,
        }
    )
    selected_card_out = getattr(args, "selected_probability_card_out", None)
    if selected_card_out is None:
        selected_card_out = args.out.with_name(
            f"{args.out.stem}.probability_card.json"
        )
    atomic_json_once(selected_card_out, selected_card)
    receipt: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "expanded_validation_threshold_selector_receipt",
        "method": config["method"],
        "task": config["task"],
        "seed": checkpoint_seed,
        "selection_split": "val",
        "selection_metric": "macro_image_dice",
        "selection_source": "validation_only",
        "validation_only": True,
        "test_or_heldout_opened": False,
        "tie_break": "closest_to_0.5_then_lower",
        "allowed_grid": candidates,
        "threshold_candidates": candidates,
        "threshold_scores": scores,
        "selected_threshold": selected,
        "selected_validation_dice": selected_score,
        "validation_images": len(entries),
        "probability_set_sha256": probability_set.get("sha256"),
        "raw_validation_probability_card_sha256": sha256_file(args.probability_card),
        "validation_probability_card_sha256": sha256_file(selected_card_out),
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
        "checkpoint_sha256": expected_identities["checkpoint_sha256"],
        "source_config_sha256": expected_identities["source_config_sha256"],
        "manifest_sha256": expected_identities["manifest_sha256"],
        "split_sha256": expected_identities["split_sha256"],
        "probability_format": probability_artifacts.FORMAT,
        "threshold_amendment_sha256": amendment_sha256,
        "amendment_sha256": amendment_sha256,
        "probabilities_are_pre_threshold": True,
        "storage_dtype": probability_artifacts.STORAGE_DTYPE,
        "derived_from_binary_mask": False,
        "debug_limited": bool(args.limit),
    }
    atomic_json_once(args.out, receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> None:
    print(json.dumps(select(parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
