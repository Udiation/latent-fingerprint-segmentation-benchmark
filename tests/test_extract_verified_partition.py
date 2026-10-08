"""Tests for the resumable, checksummed ZIP partition extractor."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_verified_partition.py"
SPEC = importlib.util.spec_from_file_location("extract_verified_partition", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(data: bytes) -> str:
    """Return a test payload digest."""
    return hashlib.sha256(data).hexdigest()


def make_archive(path: Path) -> None:
    """Create a tiny partition with a matching release-style checksum CSV."""
    payloads = {"s1/a.png": b"alpha", "s2/b.png": b"beta"}
    rows = ["filename,sha256"] + [
        f"{name},{digest(data)}" for name, data in payloads.items()
    ]
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in payloads.items():
            bundle.writestr("release/images/" + name, data)
        bundle.writestr("release/checksums.csv", "\n".join(rows) + "\n")


def test_extract_is_verified_and_resumable(tmp_path: Path) -> None:
    """A second run verifies and skips both already-extracted files."""
    archive, out = tmp_path / "release.zip", tmp_path / "out"
    make_archive(archive)
    first = MODULE.extract(
        archive, out, "release/images", "release/checksums.csv", 2, 0.01
    )
    second = MODULE.extract(
        archive, out, "release/images", "release/checksums.csv", 2, 0.01
    )
    assert first == {"files": 2, "extracted": 2, "skipped": 0}
    assert second == {"files": 2, "extracted": 0, "skipped": 2}
    assert (out / "s1" / "a.png").read_bytes() == b"alpha"
    assert not list(out.rglob("*.partial"))


def test_checksum_manifest_rejects_parent_traversal() -> None:
    """Checksum paths cannot escape the selected output root."""
    data = f"filename,sha256\n../bad,{digest(b'x')}\n".encode()
    with pytest.raises(ValueError, match="unsafe checksum path"):
        MODULE.parse_checksums(data)


def test_partition_requires_exact_member_set(tmp_path: Path) -> None:
    """Unchecked extra members under the selected prefix fail closed."""
    archive, out = tmp_path / "release.zip", tmp_path / "out"
    make_archive(archive)
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("release/images/unlisted.png", b"unlisted")
    with pytest.raises(ValueError, match="archive/checksum member mismatch"):
        MODULE.extract(
            archive, out, "release/images", "release/checksums.csv", 2, 0.01
        )
