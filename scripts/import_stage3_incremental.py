#!/usr/bin/env python3
"""Import immutable seed deliveries into the multi-seed Stage-3 bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import _path  # noqa: F401
from segbench.stage3_incremental import import_deliveries


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--expected-spec-sha256", required=True)
    parser.add_argument("--restricted-root", required=True, type=Path)
    parser.add_argument("--bundle-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    delivery = import_deliveries(
        spec_path=args.spec,
        expected_spec_sha256=args.expected_spec_sha256,
        restricted_root=args.restricted_root,
        bundle_out=args.bundle_out,
    )
    print(json.dumps({
        "kind": delivery["kind"],
        "seed_delivery_count": delivery["incremental_import"]["seed_delivery_count"],
        "gate_status": delivery["gate_evidence"]["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
