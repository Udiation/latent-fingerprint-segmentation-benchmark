#!/usr/bin/env python3
"""Build an auditable uint8 NumPy cache of the shared 500-ppi preprocessing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import cv2
from joblib import Parallel, delayed, parallel_config

import _path  # noqa: F401
from segbench.deep_utils import file_digest, read_normalized_image, read_pair
from segbench.io import load_manifest
from segbench.preprocess_cache import (
    CACHE_SCHEMA_VERSION,
    atomic_json,
    atomic_npy,
    deterministic_sample_ids,
    image_path,
    implementation_sha256,
    mask_path,
    read_image,
    read_pair as read_cached_pair,
    sha256_file,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--mask-columns",
        nargs="+",
        default=["mask_path", "quality_mask_path"],
    )
    parser.add_argument("--target-ppi", type=float, default=500.0)
    parser.add_argument("--verify-count", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def _as_u8(image: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    return scaled.astype(np.uint8)


def _ordered_ids_sha256(image_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for image_id in image_ids:
        digest.update(str(image_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _implementation_sha256() -> dict[str, str]:
    """Bind cache bytes to every module that defines preprocessing semantics."""
    return implementation_sha256()


def _prior_records(
    args: argparse.Namespace,
    columns: list[str],
    expected_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    receipt_path = args.out / "receipt.json"
    state_path = args.out / "build_state.json"
    source_path = receipt_path if receipt_path.is_file() else state_path
    if not source_path.is_file():
        return {}
    try:
        receipt = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("existing cache state is invalid; use a new cache directory") from exc
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "uint8_500ppi_preprocessing_cache",
        "manifest_sha256": file_digest(args.manifest),
        "target_ppi": float(args.target_ppi),
        "mask_columns": columns,
        "record_count": len(expected_ids),
        "ordered_image_ids_sha256": _ordered_ids_sha256(expected_ids),
        "debug_limited": bool(args.limit),
        "implementation_sha256": _implementation_sha256(),
    }
    disagreements = {
        key: (receipt.get(key), value)
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if disagreements:
        raise ValueError(f"existing cache receipt belongs to other inputs: {disagreements}")
    records = receipt.get("records") if source_path == receipt_path else None
    if source_path == state_path:
        records = []
        record_root = args.out / ".records"
        for record_path in sorted(record_root.glob("*.json")) if record_root.exists() else []:
            try:
                records.append(json.loads(record_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid cache journal record: {record_path.name}") from exc
    if not isinstance(records, list):
        raise ValueError("existing cache receipt lacks records")
    result = {
        str(record["image_id"]): dict(record)
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("image_id"), str)
    }
    expected_id_set = set(map(str, expected_ids))
    if len(result) != len(records):
        raise ValueError("existing cache state contains duplicate/invalid journal ids")
    if source_path == receipt_path and set(result) != expected_id_set:
        raise ValueError("existing cache receipt has a different image-id universe")
    if source_path == state_path and not set(result).issubset(expected_id_set):
        raise ValueError("cache journal contains ids outside the active build universe")
    return result


def _remove_stale_partials(root: Path) -> None:
    """Repair interrupted atomic writes without touching canonical artifacts."""
    if not root.exists():
        return
    for path in sorted(root.rglob(".partial-*")):
        if path.is_file():
            path.unlink()


def _journal_path(root: Path, image_id: str) -> Path:
    """Return a safe per-record journal path for interrupted initial builds."""
    safe_stem = image_path(root, image_id).stem
    return root / ".records" / f"{safe_stem}.json"


def _publish_record(root: Path, record: dict[str, object]) -> dict[str, object]:
    atomic_json(_journal_path(root, str(record["image_id"])), record)
    return record


def _build_record(
    row: Mapping[str, Any],
    *,
    out: Path,
    columns: Sequence[str],
    target_ppi: float,
    prior: Mapping[str, Any] | None,
) -> dict[str, object]:
    cv2.setNumThreads(1)
    image_id = str(row["image_id"])
    source_image_sha256 = file_digest(Path(str(row["image_path"])))
    pinned = row.get("source_sha256")
    if isinstance(pinned, str) and pinned and pinned != source_image_sha256:
        raise ValueError(f"{image_id}: manifest source_sha256 mismatch")
    source_mask_hashes: dict[str, Optional[str]] = {}
    for column in columns:
        value = row.get(column)
        if value is None or str(value).strip() == "" or str(value).lower() == "nan":
            source_mask_hashes[column] = None
        else:
            source_mask_hashes[column] = file_digest(Path(str(value)))
    destination = image_path(out, image_id)
    if (
        prior is not None
        and prior.get("source_image_sha256") == source_image_sha256
        and prior.get("source_mask_sha256") == source_mask_hashes
        and destination.is_file()
        and sha256_file(destination) == prior.get("image_sha256")
    ):
        mask_hashes = prior.get("mask_sha256", {})
        reusable = isinstance(mask_hashes, Mapping)
        for column in columns:
            expected = mask_hashes.get(column) if reusable else None
            if source_mask_hashes[column] is None:
                reusable = reusable and expected is None
            else:
                path = mask_path(out, column, image_id)
                reusable = reusable and path.is_file() and sha256_file(path) == expected
        if reusable:
            cached = np.load(destination, allow_pickle=False)
            return _publish_record(out, {
                "image_id": image_id,
                "shape": list(cached.shape),
                "source_image_sha256": source_image_sha256,
                "image_sha256": str(prior["image_sha256"]),
                "source_mask_sha256": source_mask_hashes,
                "mask_sha256": dict(mask_hashes),
            })
    image = read_normalized_image(row, target_ppi)
    image_u8 = _as_u8(image)
    atomic_npy(destination, image_u8)
    mask_hashes: dict[str, Optional[str]] = {}
    for column in columns:
        if source_mask_hashes[column] is None:
            mask_hashes[column] = None
            continue
        _, mask = read_pair(row, target_ppi, column)
        mask_destination = mask_path(out, column, image_id)
        atomic_npy(mask_destination, mask.astype(np.uint8) * 255)
        mask_hashes[column] = sha256_file(mask_destination)
    return _publish_record(out, {
        "image_id": image_id,
        "shape": list(image_u8.shape),
        "source_image_sha256": source_image_sha256,
        "image_sha256": sha256_file(destination),
        "source_mask_sha256": source_mask_hashes,
        "mask_sha256": mask_hashes,
    })


def build(args: argparse.Namespace) -> dict[str, object]:
    if float(args.target_ppi) != 500.0:
        raise ValueError("the benchmark preprocessing cache must be exactly 500 ppi")
    if args.verify_count != 20 and not args.limit:
        raise ValueError("production cache verification is frozen at 20 images")
    if not 1 <= int(args.n_jobs) <= 32:
        raise ValueError("--n-jobs must be in 1..32")
    frame = load_manifest(args.manifest).sort_values("image_id", kind="stable")
    if args.limit:
        frame = frame.head(args.limit).copy()
    columns = sorted(set(map(str, args.mask_columns)))
    if not columns:
        raise ValueError("at least one mask column is required")
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"manifest lacks mask columns: {missing_columns}")
    _remove_stale_partials(args.out)
    row_records = frame.to_dict("records")
    expected_ids = [str(row["image_id"]) for row in row_records]
    prior = _prior_records(args, columns, expected_ids)
    atomic_json(
        args.out / "build_state.json",
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": "uint8_500ppi_preprocessing_cache",
            "manifest_sha256": file_digest(args.manifest),
            "target_ppi": float(args.target_ppi),
            "mask_columns": columns,
            "record_count": len(expected_ids),
            "ordered_image_ids_sha256": _ordered_ids_sha256(expected_ids),
            "debug_limited": bool(args.limit),
            "implementation_sha256": _implementation_sha256(),
        },
    )
    with parallel_config(backend="loky", inner_max_num_threads=1):
        records = Parallel(n_jobs=int(args.n_jobs))(
            delayed(_build_record)(
                row,
                out=args.out,
                columns=columns,
                target_ppi=float(args.target_ppi),
                prior=prior.get(str(row["image_id"])),
            )
            for row in row_records
        )
    by_id = {str(row["image_id"]): row for row in frame.to_dict("records")}
    verified_ids = deterministic_sample_ids(list(by_id), args.verify_count)
    for image_id in verified_ids:
        row = by_id[image_id]
        on_the_fly = read_normalized_image(row, args.target_ppi)
        cached = read_image(args.out, image_id)
        if not np.array_equal(_as_u8(on_the_fly), _as_u8(cached)):
            raise RuntimeError(f"cached image differs from on-the-fly preprocessing: {image_id}")
        for column in columns:
            value = row.get(column)
            if value is None or str(value).strip() == "" or str(value).lower() == "nan":
                continue
            fly_image, fly_mask = read_pair(row, args.target_ppi, column)
            cache_image, cache_mask = read_cached_pair(args.out, image_id, column)
            if not np.array_equal(_as_u8(fly_image), _as_u8(cache_image)):
                raise RuntimeError(f"cache image mismatch during pair audit: {image_id}")
            if not np.array_equal(fly_mask, cache_mask):
                raise RuntimeError(f"cache mask mismatch: {image_id}/{column}")
    receipt: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "uint8_500ppi_preprocessing_cache",
        "manifest_sha256": file_digest(args.manifest),
        "target_ppi": float(args.target_ppi),
        "mask_columns": columns,
        "image_format": "uint8_npy",
        "mask_format": "uint8_0_255_npy",
        "record_count": len(records),
        "ordered_image_ids_sha256": _ordered_ids_sha256(expected_ids),
        "records": records,
        "verification": {
            "algorithm": "sha256(image_id), lexical tie-break",
            "requested_count": int(args.verify_count),
            "verified_count": len(verified_ids),
            "image_ids": verified_ids,
            "array_identity": True,
        },
        "debug_limited": bool(args.limit),
        "implementation_sha256": _implementation_sha256(),
    }
    atomic_json(args.out / "receipt.json", receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> None:
    receipt = build(parse_args(argv))
    print(
        json.dumps(
            {
                "record_count": receipt["record_count"],
                "verified_count": receipt["verification"]["verified_count"],
                "debug_limited": receipt["debug_limited"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
