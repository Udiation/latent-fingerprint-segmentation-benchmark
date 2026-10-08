#!/usr/bin/env python3
"""Accept bf16+channels-last only from a paired two-epoch val-Dice check."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once, sha256_file


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-run-card", required=True, type=Path)
    parser.add_argument("--bf16-run-card", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run card: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"run card is not a mapping: {path}")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    fp32 = _load(args.fp32_run_card)
    bf16 = _load(args.bf16_run_card)
    identity_keys = (
        "method",
        "task",
        "seed",
        "source_config_sha256",
        "manifest_sha256",
        "split_sha256",
    )
    disagreements = {
        key: (fp32.get(key), bf16.get(key))
        for key in identity_keys
        if fp32.get(key) != bf16.get(key)
    }
    if disagreements:
        raise ValueError(f"acceleration comparison identities differ: {disagreements}")
    for name, card, precision, channels_last in (
        ("fp32", fp32, "fp32", False),
        ("bf16", bf16, "bf16", True),
    ):
        if int(card.get("epochs_completed", -1)) != 2:
            raise ValueError(f"{name} run must complete exactly two epochs")
        acceleration = card.get("acceleration")
        expected = {
            "precision": precision,
            "channels_last": channels_last,
            "evaluation_only": True,
            "production_eligible": False,
        }
        if not isinstance(acceleration, Mapping) or any(
            acceleration.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"{name} card is not the declared evaluation mode")
    fp32_dice = float(fp32.get("best_validation_dice", float("nan")))
    bf16_dice = float(bf16.get("best_validation_dice", float("nan")))
    delta = abs(fp32_dice - bf16_dice)
    if not all(math.isfinite(value) for value in (fp32_dice, bf16_dice, delta)):
        raise ValueError("acceleration comparison has non-finite validation Dice")
    if not delta < 0.005:
        raise ValueError(
            f"bf16+channels-last rejected: absolute validation-Dice delta {delta:.8f}"
        )
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "kind": "bf16_channels_last_two_epoch_acceptance",
        "accepted": True,
        "epochs": 2,
        "method": fp32["method"],
        "task": fp32["task"],
        "seed": int(fp32["seed"]),
        "source_config_sha256": fp32["source_config_sha256"],
        "manifest_sha256": fp32["manifest_sha256"],
        "split_sha256": fp32["split_sha256"],
        "fp32_validation_dice": fp32_dice,
        "bf16_validation_dice": bf16_dice,
        "absolute_validation_dice_delta": delta,
        "acceptance_threshold_exclusive": 0.005,
        "fp32_run_card": {
            "path": str(args.fp32_run_card.resolve()),
            "sha256": sha256_file(args.fp32_run_card),
        },
        "bf16_run_card": {
            "path": str(args.bf16_run_card.resolve()),
            "sha256": sha256_file(args.bf16_run_card),
        },
    }
    atomic_json_once(args.out, receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> None:
    print(json.dumps(build(parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
