#!/usr/bin/env python3
"""Build an immutable restricted-demo receipt from a path-only JSON spec.

The spec and every artifact must live below ``--root``.  The output contains
only relative paths plus byte counts and SHA-256 identities.  It belongs in
restricted storage and must never be committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import _path  # noqa: F401
from demo.restricted_backend import RECEIPT_KIND


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, path: Path, label: str) -> Path:
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside --root") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside --root") from exc
    return resolved


def build(root_value: Path, spec_value: Path, out_value: Path) -> Mapping[str, Any]:
    if root_value.is_symlink() or not root_value.is_dir():
        raise ValueError("--root must be a regular directory")
    root = root_value.resolve(strict=True)
    spec_path = _inside(root, spec_value, "--spec")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("--spec is not valid UTF-8 JSON") from exc
    if not isinstance(spec, dict):
        raise ValueError("spec root must be an object")
    if spec.get("schema_version") != 1 or spec.get("kind") != RECEIPT_KIND:
        raise ValueError("spec has the wrong restricted-demo schema")
    raw_artifacts = spec.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise ValueError("spec artifacts must be a non-empty object")
    artifacts: dict[str, dict[str, Any]] = {}
    for key, value in raw_artifacts.items():
        if not isinstance(value, dict) or set(value) != {"path", "kind"}:
            raise ValueError(
                f"artifact {key!r} must contain exactly path and kind before sealing"
            )
        relative = Path(str(value["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError(f"artifact {key!r} has an unsafe relative path")
        path = _inside(root, root / relative, f"artifact {key!r}")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"artifact {key!r} is not a regular file")
        artifacts[str(key)] = {
            "path": relative.as_posix(),
            "kind": value["kind"],
            "bytes": int(info.st_size),
            "sha256": _sha256(path),
        }
    receipt = dict(spec)
    receipt["artifacts"] = artifacts
    out_parent = out_value.parent.resolve(strict=True)
    try:
        out_parent.relative_to(root)
    except ValueError as exc:
        raise ValueError("--out must be inside --root") from exc
    if out_value.exists() or out_value.is_symlink():
        raise FileExistsError(f"refusing to replace receipt: {out_value}")
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = out_value.with_name(out_value.name + ".partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out_value)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "receipt": out_value.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "artifact_count": len(artifacts),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.root, args.spec, args.out), sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
