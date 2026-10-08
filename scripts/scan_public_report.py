#!/usr/bin/env python3
"""Independently scan a built public report and its synthetic asset receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401
from report.privacy import scan_public_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--synthetic-receipt", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument(
        "--generator", type=Path,
        default=Path(__file__).resolve().parent / "make_synthetic_demo.py",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = scan_public_html(
        args.index,
        synthetic_receipt=args.synthetic_receipt,
        synthetic_root=args.synthetic_root,
        generator_path=args.generator,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(f"refusing to replace scan report: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
