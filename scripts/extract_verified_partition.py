#!/usr/bin/env python3
"""Safely extract and checksum one declared partition of a ZIP archive.

The checksum CSV must contain ``filename,sha256`` and its filenames are
interpreted relative to ``--member-prefix``. Existing valid files are skipped,
which makes interrupted extraction resumable. Files are streamed to a sibling
``.partial`` path and atomically renamed only after their checksum matches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Mapping


CHUNK_BYTES = 8 * 1024 * 1024


def parse_checksums(data: bytes) -> Dict[str, str]:
    """Parse a two-column release checksum CSV.

    Args:
        data: Raw CSV bytes.

    Returns:
        A relative POSIX path to lowercase SHA-256 mapping.

    Raises:
        ValueError: If the schema, a path, or a digest is invalid.
    """
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if rows.fieldnames != ["filename", "sha256"]:
        raise ValueError(f"unexpected checksum columns: {rows.fieldnames}")
    checksums: Dict[str, str] = {}
    for number, row in enumerate(rows, start=2):
        relative = PurePosixPath(row["filename"])
        digest = row["sha256"].strip().lower()
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe checksum path on row {number}: {relative}")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"invalid SHA-256 on row {number}")
        key = relative.as_posix()
        if key in checksums:
            raise ValueError(f"duplicate checksum path: {key}")
        checksums[key] = digest
    if not checksums:
        raise ValueError("checksum CSV contains no file rows")
    return checksums


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_members(prefix: str, checksums: Mapping[str, str]) -> Dict[str, str]:
    """Return archive-member to expected-digest mappings."""
    normalized = PurePosixPath(prefix).as_posix().rstrip("/") + "/"
    return {normalized + relative: digest for relative, digest in checksums.items()}


def validate_member_set(
    archive_names: Iterable[str], members: Mapping[str, str], prefix: str
) -> None:
    """Require exact agreement between the partition and checksum manifest."""
    normalized = PurePosixPath(prefix).as_posix().rstrip("/") + "/"
    actual = {name for name in archive_names if name.startswith(normalized) and not name.endswith("/")}
    expected = set(members)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        unexpected = sorted(actual - expected)[:10]
        raise ValueError(
            "archive/checksum member mismatch: "
            f"expected={len(expected)} actual={len(actual)} "
            f"missing={missing} unexpected={unexpected}"
        )


def check_disk(out: Path, required_bytes: int, minimum_free_fraction: float) -> None:
    """Fail before extraction if the requested write would breach the disk floor."""
    probe = out
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    floor = int(usage.total * minimum_free_fraction)
    if usage.free - required_bytes < floor:
        raise RuntimeError(
            f"disk guard: write needs {required_bytes} bytes but would leave "
            f"less than {minimum_free_fraction:.1%} free"
        )


def extract(
    archive: Path,
    out: Path,
    member_prefix: str,
    checksum_member: str,
    expected_count: int,
    minimum_free_fraction: float,
) -> Dict[str, int]:
    """Extract a checksummed partition, resuming verified existing files."""
    with zipfile.ZipFile(archive) as bundle:
        checksums = parse_checksums(bundle.read(checksum_member))
        if len(checksums) != expected_count:
            raise ValueError(
                f"checksum row count {len(checksums)} != expected {expected_count}"
            )
        members = required_members(member_prefix, checksums)
        validate_member_set((item.filename for item in bundle.infolist()), members, member_prefix)
        info_by_name = {item.filename: item for item in bundle.infolist()}

        required_bytes = sum(info_by_name[name].file_size for name in members)
        check_disk(out, required_bytes, minimum_free_fraction)
        out.mkdir(parents=True, exist_ok=True)

        extracted = skipped = 0
        for index, (member, expected_digest) in enumerate(members.items(), start=1):
            relative = PurePosixPath(member).relative_to(PurePosixPath(member_prefix))
            destination = out.joinpath(*relative.parts)
            if destination.is_file() and sha256_file(destination) == expected_digest:
                skipped += 1
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            partial.unlink(missing_ok=True)
            digest = hashlib.sha256()
            try:
                with bundle.open(member) as source, partial.open("wb") as target:
                    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                        digest.update(chunk)
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                if digest.hexdigest() != expected_digest:
                    raise ValueError(f"checksum mismatch after extraction: {member}")
                os.replace(partial, destination)
            finally:
                partial.unlink(missing_ok=True)
            extracted += 1
            if index % 250 == 0 or index == len(members):
                print(
                    f"verified {index:,}/{len(members):,}; "
                    f"extracted={extracted:,} skipped={skipped:,}",
                    flush=True,
                )
    return {"files": len(members), "extracted": extracted, "skipped": skipped}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--member-prefix", required=True)
    parser.add_argument("--checksum-member", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--minimum-free-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if not 0.0 < args.minimum_free_fraction < 1.0:
        raise ValueError("--minimum-free-fraction must be between 0 and 1")
    started = time.monotonic()
    result = extract(
        args.archive.resolve(),
        args.out.resolve(),
        args.member_prefix,
        args.checksum_member,
        args.expected_count,
        args.minimum_free_fraction,
    )
    print(f"complete: {result} wall_seconds={time.monotonic() - started:.1f}")


if __name__ == "__main__":
    main()
