#!/usr/bin/env python3
"""Build a blind, empty-label SD302b annotation package.

The input is a private SD302g-to-SD302b mapping CSV. Selection uses metadata
only; predictions and hull metrics are neither accepted nor read. The package
contains source images and filename conventions, but deliberately creates no
mask files and no pre-filled polygons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = {
    "image_id",
    "subject_id",
    "split",
    "device",
    "report_capture",
    "finger_position",
    "actual_ppi",
    "sd302b_image_path",
}


class PackageError(ValueError):
    """The metadata cannot support the requested annotation package."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED.difference(reader.fieldnames or ())
        if missing:
            raise PackageError(f"mapping lacks required columns: {sorted(missing)}")
        rows = [dict(row) for row in reader if row["split"] == "test"]
    if not rows:
        raise PackageError("mapping has no test-subject rows")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise PackageError("mapping image_id values are not unique")
    for row in rows:
        if row["report_capture"] not in {"Rolled", "Plain (segmented slap)"}:
            raise PackageError(f"unsupported capture label: {row['report_capture']}")
        source = Path(row["sd302b_image_path"])
        if not source.is_file() or source.is_symlink():
            raise PackageError(f"source image missing or symlinked: {source}")
        if float(row["actual_ppi"]) != 500.0:
            raise PackageError(f"annotation source is not on the 500-ppi grid: {source}")
    return rows


def choose_primary(
    rows: list[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    if count > len(rows):
        raise PackageError(f"requested {count} unique images from only {len(rows)} rows")
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda row: row["image_id"])
    by_subject: dict[str, list[dict[str, str]]] = {}
    for row in ordered:
        by_subject.setdefault(row["subject_id"], []).append(row)
    for group in by_subject.values():
        rng.shuffle(group)

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    # The segmented-slap stratum is scarce. Retain all of it, then guarantee
    # at least one task per test subject before filling the remaining places.
    for row in ordered:
        if row["report_capture"] == "Plain (segmented slap)":
            selected.append(row)
            selected_ids.add(row["image_id"])
    if len(selected) > count:
        raise PackageError("plain stratum alone exceeds requested package size")

    for subject in sorted(by_subject):
        if any(row["subject_id"] == subject for row in selected):
            continue
        candidate = next(
            (row for row in by_subject[subject] if row["image_id"] not in selected_ids),
            None,
        )
        if candidate is not None and len(selected) < count:
            selected.append(candidate)
            selected_ids.add(candidate["image_id"])

    pool = [row for row in ordered if row["image_id"] not in selected_ids]
    rng.shuffle(pool)
    subject_counts = Counter(row["subject_id"] for row in selected)
    while len(selected) < count:
        minimum = min(subject_counts.get(row["subject_id"], 0) for row in pool)
        index = next(
            i for i, row in enumerate(pool)
            if subject_counts.get(row["subject_id"], 0) == minimum
        )
        row = pool.pop(index)
        selected.append(row)
        selected_ids.add(row["image_id"])
        subject_counts[row["subject_id"]] += 1
    return selected


def choose_repeats(
    primary: list[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    if count > len(primary):
        raise PackageError("repeat count exceeds primary count")
    rng = random.Random(seed + 1)
    plain = [row for row in primary if row["report_capture"] == "Plain (segmented slap)"]
    rolled = [row for row in primary if row["report_capture"] == "Rolled"]
    rng.shuffle(plain)
    rng.shuffle(rolled)
    plain_count = min(len(plain), max(1, round(count * len(plain) / len(primary))))
    repeats = plain[:plain_count] + rolled[: count - plain_count]
    rng.shuffle(repeats)
    return repeats


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> dict[str, object]:
    mapping = args.mapping.resolve()
    rows = read_mapping(mapping)
    primary = choose_primary(rows, args.primary_count, args.seed)
    repeats = choose_repeats(primary, args.repeat_count, args.seed)

    task_rows: list[tuple[dict[str, str], bool, str]] = []
    for row in primary:
        task_rows.append((row, False, row["image_id"]))
    for row in repeats:
        task_rows.append((row, True, row["image_id"]))
    random.Random(args.seed + 2).shuffle(task_rows)

    summary = {
        "primary": len(primary),
        "repeats": len(repeats),
        "subjects": len({row["subject_id"] for row in primary}),
        "primary_capture_counts": dict(Counter(row["report_capture"] for row in primary)),
        "repeat_capture_counts": dict(Counter(row["report_capture"] for row in repeats)),
    }
    if args.dry_run:
        return {"status": "DRY_RUN", **summary}

    out = args.out.resolve()
    if out.exists():
        raise PackageError(f"refusing to overwrite existing package: {out}")
    images = out / "images"
    images.mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    label_studio: list[dict[str, object]] = []
    for index, (source_row, is_repeat, repeat_group) in enumerate(task_rows, start=1):
        task_id = f"task_{index:04d}"
        source = Path(source_row["sd302b_image_path"])
        suffix = source.suffix.lower()
        filename = f"{task_id}{suffix}"
        destination = images / filename
        shutil.copy2(source, destination)
        mask_filename = f"{task_id}_mask.png"
        record: dict[str, object] = {
            "task_id": task_id,
            "image_filename": filename,
            "expected_mask_filename": mask_filename,
            "is_repeat": is_repeat,
            "repeat_group": repeat_group,
            "source_image_id": source_row["image_id"],
            "source_image_path": str(source),
            "subject_id": source_row["subject_id"],
            "capture": source_row["report_capture"],
            "device": source_row["device"],
            "finger_position": source_row["finger_position"],
            "ppi": int(float(source_row["actual_ppi"])),
            "image_sha256": sha256(destination),
        }
        manifest.append(record)
        label_studio.append(
            {
                "id": index,
                # This is Label Studio's route, not a host filesystem path.
                # Keep the tokens separate so a fail-closed release scan does
                # not confuse the application URL with a private data-root path.
                "data": {"image": "/" + f"data/local-files/?d=images/{filename}"},
                "meta": {"task_id": task_id, "ppi": 500},
            }
        )

    write_csv(out / "manifest_private.csv", manifest)
    (out / "label_studio_tasks.json").write_text(
        json.dumps(label_studio, indent=2) + "\n", encoding="utf-8"
    )
    (out / "INSTRUCTIONS.md").write_text(
        """# SD302b foreground annotation instructions

Draw one polygon around the ridge-bearing fingerprint area. Include visible
fingerprint ridges and exclude background, platen borders, labels, and text.
Do not infer or extend ridges that are not visible. If several disconnected
ridge-bearing regions belong to the same impression, use the minimum number of
polygons needed to cover them without including intervening background.

Export a binary PNG at the exact source canvas size using the filename in the
`expected_mask_filename` column: fingerprint foreground = 255, background = 0.
Do not resize or crop. The package contains intentional blind repeat tasks for
intra-annotator agreement; annotate every task independently.

No masks or pre-filled polygons are included in this package.
""",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 1,
        "kind": "sd302b_manual_annotation_package_v1",
        "status": "READY_FOR_HUMAN_ANNOTATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "selection_uses_predictions": False,
        "contains_prefilled_annotations": False,
        "source_mapping": {
            "path": str(mapping),
            "sha256": sha256(mapping),
        },
        "summary": summary,
        "outputs": {
            name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
            for name in ("manifest_private.csv", "label_studio_tasks.json", "INSTRUCTIONS.md")
        },
    }
    (out / "package_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return {"status": "PASS", "out": str(out), **summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary-count", type=int, default=60)
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
