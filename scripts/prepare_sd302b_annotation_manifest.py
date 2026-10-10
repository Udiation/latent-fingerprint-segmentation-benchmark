#!/usr/bin/env python3
"""Select a deterministic SD302b-only 60+20 manual-annotation manifest.

The command authenticates the private exact-mapping receipt and writes only a
CSV/receipt.  It does not copy images, create masks, or reinterpret any method
output as ground truth.  The scarce segmented-slap test stratum is fully
represented and repeated; rolled devices are deliberately stratified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


PRIMARY_QUOTAS = {
    ("slap", "R"): 6,
    ("slap", "S"): 3,
    ("roll", "U"): 10,
    ("roll", "V"): 41,
}
REPEAT_QUOTAS = {
    ("slap", "R"): 6,
    ("slap", "S"): 3,
    ("roll", "U"): 4,
    ("roll", "V"): 7,
}
COLUMNS = (
    "task_id",
    "role",
    "repeat_of_task_id",
    "repeat_group",
    "image_id",
    "subject_id",
    "device",
    "source_capture",
    "report_capture",
    "source_image_path",
    "width",
    "height",
    "ppi",
    "selection_rank_sha256",
)


class SelectionError(ValueError):
    """The mapping or requested annotation sample is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outside_repository(path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        path.resolve().relative_to(repository)
    except ValueError:
        return
    raise SelectionError(f"annotation manifest must remain outside repository: {path}")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"cannot read mapping receipt: {path}") from exc
    if not isinstance(value, dict):
        raise SelectionError("mapping receipt is not an object")
    return value


def _load_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise SelectionError(f"cannot read mapped test CSV: {path}") from exc
    return rows


def _rank(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def _diverse_select(rows: Sequence[Mapping[str, str]], quota: int, salt: str) -> list[dict[str, str]]:
    if quota > len(rows):
        raise SelectionError(f"quota {quota} exceeds stratum size {len(rows)}")
    by_subject: Dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(dict(row))
    for subject_rows in by_subject.values():
        subject_rows.sort(key=lambda row: _rank(salt, row["image_id"]))
    subjects = sorted(by_subject, key=lambda subject: _rank(f"{salt}-subject", subject))
    selected: list[dict[str, str]] = []
    round_index = 0
    while len(selected) < quota:
        progress = False
        for subject in subjects:
            candidates = by_subject[subject]
            if round_index < len(candidates):
                selected.append(candidates[round_index])
                progress = True
                if len(selected) == quota:
                    break
        if not progress:
            raise SelectionError("subject-balanced selection exhausted unexpectedly")
        round_index += 1
    return selected


def build(
    mapping_receipt: Path,
    output_csv: Path,
    *,
    primary_quotas: Mapping[tuple[str, str], int] = PRIMARY_QUOTAS,
    repeat_quotas: Mapping[tuple[str, str], int] = REPEAT_QUOTAS,
    expected_test_records: int = 439,
) -> Mapping[str, Any]:
    mapping_receipt = mapping_receipt.resolve()
    output_csv = output_csv.resolve()
    outside_repository(output_csv)
    receipt_path = output_csv.with_suffix(".receipt.json")
    if output_csv.exists() or receipt_path.exists():
        raise SelectionError("refusing to overwrite annotation selection outputs")
    receipt = _load_json(mapping_receipt)
    if (
        receipt.get("kind") != "sd302b_sd302g_exact_mapping_receipt_v1"
        or receipt.get("status") != "REFERENCE_VERIFIED_MAPPING"
    ):
        raise SelectionError("mapping receipt is not reference verified")
    source = mapping_receipt.parent / "mapping_test_private.csv"
    identity = receipt.get("outputs", {}).get("mapping_test_private.csv", {})
    if (
        not source.is_file()
        or source.is_symlink()
        or sha256_file(source) != identity.get("sha256")
    ):
        raise SelectionError("mapped test CSV differs from its receipt")
    rows = _load_rows(source)
    if len(rows) != expected_test_records:
        raise SelectionError("mapped test universe changed")
    if any(
        row["split"] != "test"
        or row["group"] != "baseline"
        or row["canonical_pixels_equal"].lower() != "true"
        for row in rows
    ):
        raise SelectionError("annotation candidates are not exact mapped test records")

    strata: Dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[(row["source_capture"], row["device"])].append(row)
    if not set(primary_quotas).issubset(strata):
        raise SelectionError("one or more annotation strata are absent")
    primary: list[dict[str, str]] = []
    selected_by_stratum: Dict[tuple[str, str], list[dict[str, str]]] = {}
    for key, quota in primary_quotas.items():
        chosen = _diverse_select(strata[key], quota, f"sd302b-mask-primary-v1-{key}")
        selected_by_stratum[key] = chosen
        primary.extend(chosen)
    if len({row["image_id"] for row in primary}) != sum(primary_quotas.values()):
        raise SelectionError("primary annotation identities are not unique")

    primary.sort(
        key=lambda row: (
            row["report_capture"], row["device"],
            _rank("sd302b-mask-primary-order-v1", row["image_id"]),
        )
    )
    tasks: list[Dict[str, object]] = []
    task_by_image: Dict[str, str] = {}
    rank_by_image: Dict[str, str] = {}
    for index, row in enumerate(primary, start=1):
        task_id = f"SD302B-MASK-{index:03d}"
        rank = _rank("sd302b-mask-primary-v1", row["image_id"])
        task_by_image[row["image_id"]] = task_id
        rank_by_image[row["image_id"]] = rank
        tasks.append({
            "task_id": task_id, "role": "primary", "repeat_of_task_id": "",
            "repeat_group": task_id, "image_id": row["image_id"],
            "subject_id": row["subject_id"], "device": row["device"],
            "source_capture": row["source_capture"],
            "report_capture": row["report_capture"],
            "source_image_path": row["sd302b_image_path"],
            "width": row["sd302b_width"], "height": row["sd302b_height"],
            "ppi": row["actual_ppi"], "selection_rank_sha256": rank,
        })
    repeats: list[dict[str, str]] = []
    for key, quota in repeat_quotas.items():
        if key not in selected_by_stratum:
            raise SelectionError(f"repeat stratum lacks primary records: {key}")
        repeats.extend(
            _diverse_select(
                selected_by_stratum[key], quota, f"sd302b-mask-repeat-v1-{key}"
            )
        )
    repeats.sort(key=lambda row: _rank("sd302b-mask-repeat-order-v1", row["image_id"]))
    for offset, row in enumerate(repeats, start=len(primary) + 1):
        primary_task = task_by_image[row["image_id"]]
        tasks.append({
            "task_id": f"SD302B-MASK-{offset:03d}", "role": "repeat",
            "repeat_of_task_id": primary_task, "repeat_group": primary_task,
            "image_id": row["image_id"], "subject_id": row["subject_id"],
            "device": row["device"], "source_capture": row["source_capture"],
            "report_capture": row["report_capture"],
            "source_image_path": row["sd302b_image_path"],
            "width": row["sd302b_width"], "height": row["sd302b_height"],
            "ppi": row["actual_ppi"],
            "selection_rank_sha256": rank_by_image[row["image_id"]],
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(tasks)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_csv)
    finally:
        temporary.unlink(missing_ok=True)
    output_receipt: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "sd302b_manual_annotation_selection_receipt_v1",
        "status": "PACKAGE_MAPPING_READY_IMAGES_NOT_COPIED",
        "no_images_copied": True,
        "no_masks_present": True,
        "selection_signal": False,
        "input": {
            "mapping_receipt_path": str(mapping_receipt),
            "mapping_receipt_sha256": sha256_file(mapping_receipt),
            "mapping_test_sha256": sha256_file(source),
        },
        "selection": {
            "primary_images": len(primary),
            "repeat_tasks": len(repeats),
            "total_tasks": len(tasks),
            "primary_quotas": {f"{key[0]}:{key[1]}": value for key, value in primary_quotas.items()},
            "repeat_quotas": {f"{key[0]}:{key[1]}": value for key, value in repeat_quotas.items()},
            "primary_capture_counts": dict(sorted(Counter(row["report_capture"] for row in primary).items())),
            "repeat_capture_counts": dict(sorted(Counter(row["report_capture"] for row in repeats).items())),
            "subject_balancing": "deterministic round-robin by subject within each stratum",
            "rank_rule": "SHA256 over fixed salt and private image/subject identity",
        },
        "output": {
            "path": str(output_csv),
            "sha256": sha256_file(output_csv),
            "bytes": output_csv.stat().st_size,
            "rows": len(tasks),
        },
        "caveat": (
            "Only nine mapped segmented-slap test images exist; all nine are primary "
            "items and repeats. 'Plain' means segmented slap, not a separately "
            "documented plain-impression capture."
        ),
    }
    receipt_tmp = receipt_path.with_name(f".{receipt_path.name}.tmp-{os.getpid()}")
    try:
        with receipt_tmp.open("w", encoding="utf-8") as handle:
            json.dump(output_receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(receipt_tmp, receipt_path)
    finally:
        receipt_tmp.unlink(missing_ok=True)
    return output_receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = build(args.mapping_receipt, args.out)
    except (SelectionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
