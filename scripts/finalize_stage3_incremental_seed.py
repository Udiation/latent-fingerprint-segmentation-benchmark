#!/usr/bin/env python3
"""Publish one planned Stage-3 seed score exactly once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import _path  # noqa: F401
from segbench.stage3_incremental import finalize_seed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--restricted-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--bundle-out", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    delivery = finalize_seed(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        restricted_root=args.restricted_root,
        method=args.method,
        seed=args.seed,
        bundle_out=args.bundle_out,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
    )
    print(json.dumps(delivery, sort_keys=True))


if __name__ == "__main__":
    main()
