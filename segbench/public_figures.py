"""Receipt-bound contract for aggregate-only public benchmark figures."""
from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Mapping


MANIFEST_KIND = "segbench.aggregate-figure-manifest.v1"
SOURCE_KIND = "stage5_aggregate_package_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FIGURE_KINDS = {
    "dice_boxplot": "dice_distribution",
    "dice_vs_ms": "efficiency_pareto",
    "dice_by_examiner_value": "examiner_value",
    "hr_by_device": "exemplar_device",
}
MIME_BY_SUFFIX = {".png": "image/png", ".pdf": "application/pdf"}


class FigureManifestError(ValueError):
    """A public figure or its provenance manifest is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pending_manifest() -> dict[str, Any]:
    """Return the only valid empty manifest used before final publication."""
    return {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "status": "pending",
        "source_receipt": None,
        "figures": [],
    }


def _safe_path(value: object) -> Path:
    if not isinstance(value, str):
        raise FigureManifestError("figure path is not a string")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "figures"
        or path.suffix.lower() not in MIME_BY_SUFFIX
    ):
        raise FigureManifestError(f"unsafe aggregate figure path: {value!r}")
    if path.stem not in FIGURE_KINDS:
        raise FigureManifestError(f"unapproved aggregate figure name: {value!r}")
    return path


def _check_magic(path: Path, mime: str) -> None:
    prefix = path.read_bytes()[:8]
    if mime == "image/png" and prefix != b"\x89PNG\r\n\x1a\n":
        raise FigureManifestError(f"{path.name} is not a PNG")
    if mime == "application/pdf" and not prefix.startswith(b"%PDF-"):
        raise FigureManifestError(f"{path.name} is not a PDF")


def build_manifest(
    root: Path,
    figure_paths: list[Path],
    *,
    source_receipt_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic manifest for already-copied aggregate figures."""
    if HEX64.fullmatch(source_receipt_sha256) is None:
        raise FigureManifestError("source receipt SHA-256 is invalid")
    rows = []
    seen: set[str] = set()
    for raw in sorted(figure_paths, key=lambda item: item.as_posix()):
        try:
            relative = raw.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise FigureManifestError("aggregate figure escapes repository root") from exc
        relative = _safe_path(relative.as_posix())
        key = relative.as_posix()
        if key in seen:
            raise FigureManifestError(f"duplicate aggregate figure: {key}")
        seen.add(key)
        absolute = root / relative
        if not absolute.is_file() or absolute.is_symlink():
            raise FigureManifestError(f"aggregate figure is absent or a symlink: {key}")
        mime = MIME_BY_SUFFIX[relative.suffix.lower()]
        _check_magic(absolute, mime)
        rows.append({
            "path": key,
            "sha256": sha256_file(absolute),
            "mime_type": mime,
            "kind": FIGURE_KINDS[relative.stem],
            "source_receipt_sha256": source_receipt_sha256,
        })
    return {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "status": "complete",
        "source_receipt": {
            "kind": SOURCE_KIND,
            "sha256": source_receipt_sha256,
        },
        "figures": rows,
    }


def validate_figure_manifest(
    root: Path,
    manifest_path: Path | None = None,
    *,
    expected_source_receipt_sha256: str | None = None,
) -> dict[str, Mapping[str, str]]:
    """Validate the manifest and return approved repository-relative figures."""
    path = manifest_path or root / "figures" / "figure_manifest.json"
    if not path.is_file() or path.is_symlink():
        raise FigureManifestError("aggregate figure manifest is absent or a symlink")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FigureManifestError("aggregate figure manifest is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "kind", "status", "source_receipt", "figures"
    }:
        raise FigureManifestError("aggregate figure manifest schema changed")
    if payload.get("schema_version") != 1 or payload.get("kind") != MANIFEST_KIND:
        raise FigureManifestError("aggregate figure manifest kind changed")
    rows = payload.get("figures")
    if not isinstance(rows, list):
        raise FigureManifestError("aggregate figure list is malformed")
    if payload.get("status") == "pending":
        if rows or payload.get("source_receipt") is not None:
            raise FigureManifestError("pending figure manifest contains published artifacts")
        if expected_source_receipt_sha256 is not None:
            raise FigureManifestError("expected source receipt has no published figures")
        return {}
    if payload.get("status") != "complete" or not rows:
        raise FigureManifestError("complete figure manifest has no figures")
    source = payload.get("source_receipt")
    if (
        not isinstance(source, dict)
        or set(source) != {"kind", "sha256"}
        or source.get("kind") != SOURCE_KIND
        or HEX64.fullmatch(str(source.get("sha256", ""))) is None
    ):
        raise FigureManifestError("figure source receipt is invalid")
    source_digest = str(source["sha256"])
    if expected_source_receipt_sha256 is not None and source_digest != expected_source_receipt_sha256:
        raise FigureManifestError("figure source receipt differs from the independently pinned receipt")
    approved: dict[str, Mapping[str, str]] = {}
    for index, item in enumerate(rows):
        if not isinstance(item, dict) or set(item) != {
            "path", "sha256", "mime_type", "kind", "source_receipt_sha256"
        }:
            raise FigureManifestError(f"figure row {index} schema changed")
        relative = _safe_path(item["path"])
        key = relative.as_posix()
        if key in approved:
            raise FigureManifestError(f"duplicate aggregate figure: {key}")
        expected_mime = MIME_BY_SUFFIX[relative.suffix.lower()]
        guessed, _ = mimetypes.guess_type(relative.name)
        if item.get("mime_type") != expected_mime or guessed != expected_mime:
            raise FigureManifestError(f"figure MIME differs from its suffix: {key}")
        if item.get("kind") != FIGURE_KINDS[relative.stem]:
            raise FigureManifestError(f"figure kind differs from its declared plot: {key}")
        if item.get("source_receipt_sha256") != source_digest:
            raise FigureManifestError(f"figure source receipt changed: {key}")
        digest = str(item.get("sha256", ""))
        if HEX64.fullmatch(digest) is None:
            raise FigureManifestError(f"figure SHA-256 is invalid: {key}")
        absolute = root / relative
        if not absolute.is_file() or absolute.is_symlink():
            raise FigureManifestError(f"manifest figure is absent or a symlink: {key}")
        if sha256_file(absolute) != digest:
            raise FigureManifestError(f"manifest figure bytes changed: {key}")
        _check_magic(absolute, expected_mime)
        approved[key] = {str(k): str(v) for k, v in item.items()}
    return approved
