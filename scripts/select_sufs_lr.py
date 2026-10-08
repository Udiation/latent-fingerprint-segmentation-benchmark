#!/usr/bin/env python3
"""Freeze the SUFS learning rate from both seed-0 validation sweep runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import _path  # noqa: F401
from segbench.deep_utils import atomic_json, file_digest
from segbench.sufs_finetune import PROTOCOL_LEARNING_RATES, validate_learning_rate


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse two completed sweep run cards and an output artifact path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=2, required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def select(run_paths: Sequence[Path]) -> Dict[str, Any]:
    """Validate equal-budget evidence and choose highest validation Dice."""
    if len(run_paths) != 2:
        raise ValueError("exactly two SUFS learning-rate run cards are required")
    candidates = []
    shared_keys = (
        "protocol_config_sha256",
        "source_config_sha256",
        "manifest_sha256",
        "split_sha256",
    )
    shared: Dict[str, Any] = {}
    released_digest = None
    for path in run_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("method") != "sufs_ft" or payload.get("mode") != "sweep":
            raise ValueError(f"{path} is not a SUFS sweep run card")
        if payload.get("seed") != 0:
            raise ValueError(f"{path} did not use the preregistered sweep seed 0")
        if payload.get("epochs_requested") != 30 or payload.get("epochs_completed") != 30:
            raise ValueError(f"{path} did not complete the equal 30-epoch budget")
        if payload.get("debug_limited"):
            raise ValueError(f"{path} is a debug-limited run and cannot select a rate")
        learning_rate = validate_learning_rate(payload.get("learning_rate"))
        score = float(payload.get("best_validation_dice"))
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{path} has an invalid validation Dice")
        for key in shared_keys:
            if key not in shared:
                shared[key] = payload.get(key)
            elif payload.get(key) != shared[key]:
                raise ValueError(f"SUFS sweep runs disagree on {key}")
        observed_released = payload.get("released", {}).get("weights_sha256")
        if not isinstance(observed_released, str) or len(observed_released) != 64:
            raise ValueError(f"{path} lacks released-weight provenance")
        if released_digest is None:
            released_digest = observed_released
        elif observed_released != released_digest:
            raise ValueError("SUFS sweep runs used different released weights")
        candidates.append(
            {
                "learning_rate": learning_rate,
                "best_validation_dice": score,
                "best_epoch": int(payload["best_epoch"]),
                "run_card_sha256": file_digest(path),
            }
        )
    if {item["learning_rate"] for item in candidates} != set(PROTOCOL_LEARNING_RATES):
        raise ValueError("SUFS sweep does not cover both preregistered rates")
    candidates.sort(key=lambda item: item["learning_rate"])
    # Lower learning rate is the prospective exact-tie rule.
    selected = max(
        candidates,
        key=lambda item: (item["best_validation_dice"], -item["learning_rate"]),
    )
    return {
        "schema_version": 1,
        "method": "sufs_ft",
        "selection_split": "val",
        "selection_metric": "best_macro_dice",
        **shared,
        "released_weights_sha256": released_digest,
        "tie_rule": "lower_learning_rate",
        "candidates": candidates,
        "selected_learning_rate": selected["learning_rate"],
        "selected_validation_dice": selected["best_validation_dice"],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Write an immutable validation-only selection artifact."""
    args = parse_args(argv)
    result = select(args.runs)
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
