"""Auditable 500-ppi uint8 preprocessing-cache helpers.

The cache is an operational acceleration only.  Its immutable receipt binds the
source manifest and every cached array; scientific configuration and split
selection remain outside this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


CACHE_SCHEMA_VERSION = 1
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> Dict[str, str]:
    """Return current identities of every file defining cached array semantics."""
    package = Path(__file__).resolve().parent
    project = package.parent
    paths = {
        "builder": project / "scripts" / "cache_preprocessed.py",
        "deep_utils": package / "deep_utils.py",
        "io": package / "io.py",
        "preprocess_cache": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"cache implementation files are missing: {missing}")
    return {key: sha256_file(path) for key, path in paths.items()}


def _safe_component(value: str, *, column: bool = False) -> str:
    expression = SAFE_COLUMN if column else SAFE_NAME
    if not expression.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe cache component: {value!r}")
    return value


def image_path(root: Path | str, image_id: str) -> Path:
    """Return the non-escaping cache path for one resampled image."""
    base = Path(root)
    return base / "images" / f"{_safe_component(str(image_id))}.npy"


def mask_path(root: Path | str, mask_column: str, image_id: str) -> Path:
    """Return the non-escaping cache path for one resampled mask."""
    base = Path(root)
    return (
        base
        / "masks"
        / _safe_component(str(mask_column), column=True)
        / f"{_safe_component(str(image_id))}.npy"
    )


def atomic_npy(path: Path, array: np.ndarray) -> None:
    """Write an array atomically without pickle support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".partial-", suffix=".npy"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic JSON document atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".partial-", suffix=".json"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_uint8(path: Path, shape: Tuple[int, int] | None = None) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid cached array: {path.name}") from exc
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(f"cached array must be two-dimensional uint8: {path.name}")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"cached array shape mismatch: {path.name}")
    return array


def read_image(root: Path | str, image_id: str) -> np.ndarray:
    """Read one cached image as float32 in [0, 1]."""
    return _load_uint8(image_path(root, image_id)).astype(np.float32) / 255.0


def read_pair(
    root: Path | str, image_id: str, mask_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Read one cached image and its aligned boolean mask."""
    image_u8 = _load_uint8(image_path(root, image_id))
    mask_u8 = _load_uint8(mask_path(root, mask_column, image_id), image_u8.shape)
    if not set(np.unique(mask_u8).tolist()).issubset({0, 255}):
        raise ValueError("cached mask is not binary 0/255")
    return image_u8.astype(np.float32) / 255.0, mask_u8 > 0


def deterministic_sample_ids(image_ids: Sequence[str], count: int) -> list[str]:
    """Choose a stable content-independent audit sample."""
    if count < 0:
        raise ValueError("verification count must be non-negative")
    unique = sorted({str(value) for value in image_ids})
    ordered = sorted(
        unique,
        key=lambda value: (hashlib.sha256(value.encode("utf-8")).digest(), value),
    )
    return ordered[: min(count, len(ordered))]


def load_receipt(
    root: Path | str,
    *,
    manifest_sha256: str,
    target_ppi: float,
    mask_columns: Sequence[str],
    required_ids: Sequence[str] = (),
    verify_artifact_hashes: bool = False,
) -> Dict[str, Any]:
    """Validate and return a cache receipt before cached arrays are consumed."""
    root = Path(root)
    receipt_path = root / "receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("preprocessing cache lacks a valid receipt.json") from exc
    expected_columns = sorted(str(value) for value in mask_columns)
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "uint8_500ppi_preprocessing_cache",
        "manifest_sha256": manifest_sha256,
        "target_ppi": float(target_ppi),
        "image_format": "uint8_npy",
        "mask_format": "uint8_0_255_npy",
        "implementation_sha256": implementation_sha256(),
    }
    disagreements = {
        key: (receipt.get(key), value)
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if disagreements:
        raise ValueError(f"preprocessing-cache receipt mismatch: {disagreements}")
    receipt_columns = receipt.get("mask_columns")
    if not isinstance(receipt_columns, list) or not set(expected_columns).issubset(
        set(map(str, receipt_columns))
    ):
        raise ValueError("preprocessing cache lacks a required mask column")
    records = receipt.get("records")
    if not isinstance(records, list):
        raise ValueError("preprocessing-cache receipt has no records")
    by_id = {
        str(record.get("image_id")): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("image_id"), str)
    }
    if len(by_id) != len(records):
        raise ValueError("preprocessing-cache receipt contains duplicate/invalid ids")
    missing = sorted(set(map(str, required_ids)) - set(by_id))
    if missing:
        raise ValueError(f"preprocessing cache is missing required ids, e.g. {missing[:10]}")
    if verify_artifact_hashes:
        for image_id in sorted(set(map(str, required_ids)) or set(by_id)):
            record = by_id[image_id]
            image = image_path(root, image_id)
            if sha256_file(image) != record.get("image_sha256"):
                raise ValueError(f"cached image bytes changed: {image_id}")
            mask_hashes = record.get("mask_sha256", {})
            for column in expected_columns:
                expected_hash = mask_hashes.get(column)
                if expected_hash is None:
                    continue
                if sha256_file(mask_path(root, column, image_id)) != expected_hash:
                    raise ValueError(f"cached mask bytes changed: {image_id}/{column}")
    return receipt
