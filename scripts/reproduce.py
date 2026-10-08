#!/usr/bin/env python3
"""Fail-closed dispatcher for paper table reproduction entry points."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def parse_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing local config: {path}")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"invalid local config line: {line}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", choices=["A", "A_prime", "B1", "D"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    values = parse_env(args.config)
    required = {"A": ["MANIFEST", "SPLITS", "PREDICTIONS", "OUTPUT"], "A_prime": ["MANIFEST", "SPLITS", "PREDICTIONS", "OUTPUT"], "B1": ["HULL_MANIFEST", "SPLITS", "PREDICTIONS", "OUTPUT"], "D": ["MANIFEST", "SPLITS", "PREDICTIONS", "OUTPUT"]}[args.table]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise SystemExit("local config lacks: " + ", ".join(missing))
    script = "scripts/evaluate_hull.py" if args.table == "B1" else "scripts/evaluate_masks.py"
    command = ["python", script, "--help"]
    print(f"Table {args.table}: validated local inputs; evaluator entry point is {script}")
    if args.dry_run:
        print("DRY RUN:", " ".join(command))
    else:
        subprocess.run(command, check=True)
        print("The release skeleton validates dispatch only; use docs/PROTOCOL.md to bind the frozen model registry before production scoring.")


if __name__ == "__main__":
    main()

