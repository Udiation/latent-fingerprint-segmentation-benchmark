"""Loss-preserving probability artifacts for validation-only threshold selection."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FORMAT = "float16_npy"
STORAGE_DTYPE = "float16"
THRESHOLD_AMENDMENT_SHA256 = (
    "bc03e127606fae78b1fb727b74852c2ec92b945a082ca890549d106cb932780a"
)


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON for independently reproducible set identities."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_threshold_amendment(path: Path | str) -> str:
    """Require the exact operator-authorized threshold-policy amendment."""
    path = Path(path)
    digest = sha256_file(path)
    if digest != THRESHOLD_AMENDMENT_SHA256:
        raise ValueError(
            "threshold amendment bytes differ from the frozen operator authorization"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("threshold amendment is not valid JSON") from exc
    if payload.get("kind") != "stage4_threshold_policy_amendment":
        raise ValueError("unexpected threshold amendment kind")
    grid = payload.get("replacement_policy", {}).get("allowed_trainable_grid")
    if grid != [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        raise ValueError("threshold amendment does not authorize the frozen grid")
    return digest


def safe_path(root: Path | str, image_id: str) -> Path:
    """Return a probability path that cannot escape ``root``."""
    image_id = str(image_id)
    if not SAFE_IMAGE_ID.fullmatch(image_id) or image_id in {".", ".."}:
        raise ValueError(f"unsafe probability image_id: {image_id!r}")
    base = Path(root)
    destination = base / f"{image_id}.npy"
    if destination.parent.resolve() != base.resolve():
        raise ValueError("probability path escapes its output directory")
    return destination


def atomic_write(path: Path, probability: np.ndarray) -> None:
    """Atomically save a finite [0,1] probability map as float16."""
    value = np.asarray(probability)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("probability map must be finite and two-dimensional")
    if bool((value < 0.0).any()) or bool((value > 1.0).any()):
        raise ValueError("probability map values must lie in [0,1]")
    encoded = value.astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, encoded, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def storage_roundtrip(probability: np.ndarray) -> np.ndarray:
    """Return exactly the numeric values retained by the float16 artifact."""
    value = np.asarray(probability)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("probability map must be finite and two-dimensional")
    if bool((value < 0.0).any()) or bool((value > 1.0).any()):
        raise ValueError("probability map values must lie in [0,1]")
    return value.astype(np.float16).astype(np.float32)


def read(path: Path | str, expected_shape: Tuple[int, int] | None = None) -> np.ndarray:
    """Load and validate one float16 probability map."""
    try:
        value = np.load(Path(path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid probability map: {Path(path).name}") from exc
    if value.ndim != 2 or value.dtype != np.float16:
        raise ValueError("probability map must be two-dimensional float16")
    if expected_shape is not None and tuple(value.shape) != tuple(expected_shape):
        raise ValueError("probability map has the wrong shape")
    if not np.isfinite(value).all() or bool((value < 0).any()) or bool((value > 1).any()):
        raise ValueError("probability map contains an invalid value")
    return value.astype(np.float32)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash_map(rows: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Bind image ids to source bytes, verifying a pinned digest when present."""
    result: Dict[str, str] = {}
    for row in rows:
        image_id = str(row["image_id"])
        if image_id in result:
            raise ValueError(f"duplicate source image id: {image_id}")
        observed = sha256_file(Path(str(row["image_path"])))
        pinned = row.get("source_sha256")
        if isinstance(pinned, str) and pinned and pinned != observed:
            raise ValueError(f"source image bytes changed: {image_id}")
        result[image_id] = observed
    return result


def authenticated_resume_records(
    records: Iterable[Mapping[str, Any]],
    *,
    allowed_ids: Sequence[str],
    timing_key: str,
) -> Dict[str, Dict[str, Any]]:
    """Validate resumable per-image records without collapsing bad identities."""
    allowed = set(map(str, allowed_ids))
    result: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("image_id"), str
        ):
            raise ValueError("resumable inference record has no valid image_id")
        image_id = str(record["image_id"])
        if image_id in result:
            raise ValueError(f"duplicate resumable inference id: {image_id}")
        if image_id not in allowed:
            raise ValueError(f"unknown resumable inference id: {image_id}")
        timing = record.get(timing_key)
        if (
            not isinstance(timing, (float, int))
            or isinstance(timing, bool)
            or not np.isfinite(float(timing))
            or float(timing) < 0.0
        ):
            raise ValueError(f"invalid resumable timing for {image_id}")
        result[image_id] = dict(record)
    return result


def set_identity(
    items: Sequence[tuple[str, Path]],
    source_sha256: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Bind the complete probability universe, filenames and bytes."""
    ordered = sorted((str(image_id), Path(path)) for image_id, path in items)
    if len({image_id for image_id, _ in ordered}) != len(ordered):
        raise ValueError("duplicate image ids in probability set")
    records = []
    aggregate = hashlib.sha256()
    for image_id, path in ordered:
        digest = sha256_file(path)
        probability = read(path)
        byte_count = path.stat().st_size
        record = {
            "image_id": image_id,
            "filename": path.name,
            "path": str(path.resolve()),
            "sha256": digest,
            "bytes": byte_count,
            "shape": list(probability.shape),
            "dtype": STORAGE_DTYPE,
        }
        if source_sha256 is not None:
            source_digest = source_sha256.get(image_id)
            if not isinstance(source_digest, str) or len(source_digest) != 64:
                raise ValueError(f"missing source SHA-256 for {image_id}")
            record["source_sha256"] = source_digest
        records.append(record)
        aggregate.update(image_id.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(path.name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
    canonical_records = [
        {
            "image_id": record["image_id"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        for record in records
    ]
    identity = {
        "format": FORMAT,
        "storage_dtype": STORAGE_DTYPE,
        "count": len(records),
        "sha256": canonical_sha256(canonical_records),
        "legacy_sha256": aggregate.hexdigest(),
        "records": records,
    }
    if source_sha256 is not None:
        identity["ordered_source_images_sha256"] = canonical_sha256(
            [
                {
                    "image_id": record["image_id"],
                    "source_sha256": record["source_sha256"],
                }
                for record in records
            ]
        )
    return identity
