"""Small, dependency-free helpers for immutable artifact identities."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


HEX40 = set("0123456789abcdef")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path | str) -> str:
    value = Path(path).read_bytes()
    digest = hashlib.sha1()  # nosec B324: Git object identity, not security
    digest.update(f"blob {len(value)}\0".encode("ascii"))
    digest.update(value)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def atomic_json_once(path: Path, value: Any) -> None:
    """Publish an immutable final card, accepting only an identical replay."""
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if canonical_sha256(observed) != canonical_sha256(value):
            raise FileExistsError(f"refusing to replace non-identical artifact: {path}")
        return
    atomic_json(path, value)


def require_clean_process_start(
    run: Mapping[str, Any], expected_commit: str, *, description: str
) -> None:
    if len(expected_commit) != 40 or any(ch not in HEX40 for ch in expected_commit):
        raise ValueError(f"{description}: expected commit is not an immutable SHA")
    provenance = run.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{description}: missing provenance")
    expected = {
        "capture_timing": "process_start",
        "git_commit": expected_commit,
        "git_dirty": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"{description}: provenance {key}={provenance.get(key)!r}; "
                f"expected {value!r}"
            )


def prediction_set_identity(items: Iterable[tuple[str, Path]]) -> Mapping[str, Any]:
    records = []
    for image_id, path in sorted(items, key=lambda pair: pair[0]):
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {"image_id": image_id, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    ids = [record["image_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction set contains duplicate image ids")
    return {
        "count": len(records),
        "sha256": canonical_sha256(records),
        "records": records,
    }
