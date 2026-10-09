#!/usr/bin/env python3
"""Prepare a GT-blind inference manifest from a private SD302g hull manifest.

The hull builder deliberately uses ``hull_path`` instead of ``mask_path`` so
the minutiae hull cannot be mistaken for complete pixel ground truth.  Generic
inference runners nevertheless require the common manifest schema.  This
adapter preserves that separation: it writes an empty ``mask_path`` column,
resolves only the source images, and verifies the already-frozen subject split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping


REQUIRED = {
    "image_id",
    "image_path",
    "subject_id",
    "split",
    "ppi",
    "group",
    "device",
    "capture",
}
OUTPUT_COLUMNS = (
    "image_id",
    "image_path",
    "mask_path",
    "subject_id",
    "subset",
    "ppi",
    "group",
    "device",
    "capture",
    "source_sha256",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Return a stable digest for a JSON-compatible value."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def outside_repo(path: Path) -> None:
    """Reject private output paths inside the source repository."""
    repo = Path(__file__).resolve().parents[2]
    try:
        path.resolve().relative_to(repo)
    except ValueError:
        return
    raise ValueError("private SD302g manifests must be written outside the repository")


def load_split(path: Path) -> Mapping[str, str]:
    """Load and validate the frozen subject-to-split mapping."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("splits") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("split file lacks a non-empty splits mapping")
    allowed = {"train", "val", "test"}
    normalized = {str(key): str(value) for key, value in mapping.items()}
    if set(normalized.values()) - allowed:
        raise ValueError("split file contains an unsupported partition")
    return normalized


def build(source: Path, splits: Path, out: Path, subset: str) -> Dict[str, object]:
    """Create the GT-blind manifest and return its aggregate receipt."""
    outside_repo(out)
    mapping = load_split(splits)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"source manifest is missing columns: {sorted(missing)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("source manifest is empty")

    root = source.parent.resolve()
    seen = set()
    rows: List[Dict[str, str]] = []
    split_counts = {name: 0 for name in ("train", "val", "test")}
    for row in source_rows:
        image_id = str(row["image_id"])
        subject = str(row["subject_id"])
        if image_id in seen:
            raise ValueError("source manifest contains duplicate image_id values")
        seen.add(image_id)
        expected_split = mapping.get(subject)
        if expected_split is None or expected_split != row["split"]:
            raise ValueError("source manifest disagrees with the frozen subject split")
        relative = Path(str(row["image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source image path must be relative to the source manifest")
        image = (root / relative).resolve()
        try:
            image.relative_to(root)
        except ValueError as exc:
            raise ValueError("source image path escapes the artifact root") from exc
        if not image.is_file():
            raise FileNotFoundError(f"missing source image for manifest row {len(rows) + 1}")
        ppi = float(row["ppi"])
        if ppi <= 0:
            raise ValueError("ppi must be positive")
        source_sha256 = sha256_file(image)
        split_counts[expected_split] += 1
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image),
                "mask_path": "",
                "subject_id": subject,
                "subset": subset,
                "ppi": f"{ppi:g}",
                "group": str(row["group"]),
                "device": str(row["device"]),
                "capture": str(row["capture"]),
                "source_sha256": source_sha256,
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out)
    receipt: Dict[str, object] = {
        "schema_version": 1,
        "kind": "sd302g_gt_blind_inference_manifest",
        "source_manifest_sha256": sha256_file(source),
        "split_sha256": sha256_file(splits),
        "output_sha256": sha256_file(out),
        "rows": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "split_counts": split_counts,
        "subset": subset,
        "mask_paths_empty": all(not row["mask_path"] for row in rows),
        "ordered_source_images_sha256": canonical_sha256(
            [
                {
                    "image_id": row["image_id"],
                    "source_sha256": row["source_sha256"],
                }
                for row in sorted(rows, key=lambda item: item["image_id"])
            ]
        ),
    }
    receipt_path = out.with_suffix(out.suffix + ".receipt.json")
    receipt_temp = receipt_path.with_name(receipt_path.name + ".partial")
    receipt_temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(receipt_temp, receipt_path)
    return receipt


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subset", default="finger_exemplar")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.splits, args.out, args.subset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
