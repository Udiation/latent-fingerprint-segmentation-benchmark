#!/usr/bin/env python3
"""Freeze the incremental seed-0 held-out inference gate, plan, and queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import _path  # noqa: F401
from segbench.stage3_incremental import prepare_seed0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--restricted-root", required=True, type=Path)
    parser.add_argument("--gate-out", required=True, type=Path)
    parser.add_argument("--plan-out", required=True, type=Path)
    parser.add_argument("--queue-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    gate, plan, queue = prepare_seed0(
        spec_path=args.spec,
        expected_spec_sha256=args.expected_spec_sha256,
        restricted_root=args.restricted_root,
        gate_out=args.gate_out,
        plan_out=args.plan_out,
        queue_out=args.queue_out,
    )
    print(json.dumps({
        "gate_status": gate["status"],
        "planned_runs": len(plan["runs"]),
        "queued_jobs": len(queue["jobs"]),
        "satisfies_stage4_multi_seed_gate": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
