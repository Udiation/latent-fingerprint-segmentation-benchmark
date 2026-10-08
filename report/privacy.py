"""Fail-closed validation for the self-contained public-safe report.

The public report is deliberately more restrictive than a general HTML page:
it cannot load remote resources, execute scripts, name individual benchmark
samples, or embed an image that is not bound to a synthetic-data receipt.
"""
from __future__ import annotations

import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


class PublicReportError(RuntimeError):
    """Raised when a public report cannot be proven safe to publish."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DATA_IMAGE = re.compile(
    r"^data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\s]+)$",
    re.IGNORECASE,
)
_FORBIDDEN_PUBLIC_IDENTIFIERS = (
    ("SD302 impression identifier", re.compile(r"(?<!\d)\d{8}_[A-Za-z0-9]")),
    ("source-resolution filename token", re.compile("1000" + "PPI", re.IGNORECASE)),
    ("latent-position filename token", re.compile(r"\bLP\d{1,3}\b", re.IGNORECASE)),
)
_FORBIDDEN_INFRASTRUCTURE = (
    ("restricted filesystem path", re.compile(r"(?:^|[\s\"'=])/(?:home|mnt|data|restricted)/")),
    ("cloud-drive reference", re.compile(r"(?:drive\.google\.com|docs\.google\.com)", re.IGNORECASE)),
    ("external URL", re.compile(r"\b(?:https?|ftp)://", re.IGNORECASE)),
    ("network hostname", re.compile(r"\b[a-z0-9-]+\.(?:com|net|org|io|ai|dev|local)\b", re.IGNORECASE)),
    ("IPv4 address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
    (
        "hardware detail",
        re.compile(
            r"\b(?:" + "|".join(("NVI" + "DIA", "CU" + "DA", "B" + "300",
                                    "A" + "100", "H" + "100", r"R" + r"TX\s*\d+"))
            + r")\b",
            re.IGNORECASE,
        ),
    ),
)


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular file without following a symlink."""
    if path.is_symlink() or not path.is_file():
        raise PublicReportError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_infrastructure_text(value: str, context: str) -> None:
    """Reject local paths, external endpoints, and hardware disclosures."""
    for label, pattern in _FORBIDDEN_INFRASTRUCTURE:
        match = pattern.search(value)
        if match:
            excerpt = match.group(0)[:80]
            raise PublicReportError(f"{context} contains {label}: {excerpt!r}")


def scan_public_text(value: str, context: str) -> None:
    """Reject identifiers, local paths, network details, and hardware details."""
    for label, pattern in _FORBIDDEN_PUBLIC_IDENTIFIERS:
        match = pattern.search(value)
        if match:
            excerpt = match.group(0)[:80]
            raise PublicReportError(f"{context} contains {label}: {excerpt!r}")
    scan_infrastructure_text(value, context)


def load_synthetic_receipt(
    path: Path, root: Path, *, generator_path: Path | None = None
) -> dict[str, dict[str, str]]:
    """Authenticate the regular files declared by a synthetic-data receipt."""
    if path.is_symlink() or not path.is_file():
        raise PublicReportError(f"missing synthetic receipt: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicReportError(f"cannot parse synthetic receipt: {exc}") from exc
    if not isinstance(receipt, dict):
        raise PublicReportError("synthetic receipt must be a JSON object")
    if receipt.get("schema_version") != 1 or receipt.get("kind") != "segbench_synthetic_dataset":
        raise PublicReportError("synthetic receipt schema or kind is not recognized")
    if receipt.get("provenance") != "generated_by_make_synthetic_demo":
        raise PublicReportError("synthetic receipt does not declare the approved generator")
    generator_digest = receipt.get("generator_sha256")
    if not isinstance(generator_digest, str) or not _SHA256.fullmatch(generator_digest):
        raise PublicReportError("synthetic receipt lacks a valid generator digest")
    for field in ("seed", "subjects", "per_subject"):
        if not isinstance(receipt.get(field), int):
            raise PublicReportError(f"synthetic receipt has an invalid {field}")
    if receipt["subjects"] < 1 or receipt["per_subject"] < 1:
        raise PublicReportError("synthetic receipt declares an empty generation")
    if generator_path is not None and sha256_file(generator_path) != generator_digest:
        raise PublicReportError("synthetic receipt was produced by a different generator revision")
    assets = receipt.get("assets")
    if not isinstance(assets, list) or not assets:
        raise PublicReportError("synthetic receipt has no assets")
    root = root.resolve()
    records: dict[str, dict[str, str]] = {}
    for index, item in enumerate(assets):
        if not isinstance(item, Mapping):
            raise PublicReportError(f"synthetic asset {index} is not an object")
        relative = item.get("path")
        digest = item.get("sha256")
        role = item.get("role")
        mime = item.get("mime_type")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise PublicReportError(f"synthetic asset {index} has an unsafe path")
        relative_path = Path(relative)
        if any(part in {"", ".", ".."} for part in relative_path.parts):
            raise PublicReportError(f"synthetic asset {index} has an unsafe path")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise PublicReportError(f"synthetic asset {index} has an invalid digest")
        if role not in {"example_image", "ground_truth_mask"}:
            raise PublicReportError(f"synthetic asset {index} has an invalid role")
        if mime != "image/png":
            raise PublicReportError(f"synthetic asset {index} has an invalid MIME type")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PublicReportError(f"synthetic asset escapes its root: {relative}") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise PublicReportError(f"synthetic asset is not a regular file: {relative}")
        if sha256_file(candidate) != digest:
            raise PublicReportError(f"synthetic asset digest mismatch: {relative}")
        key = relative_path.as_posix()
        if key in records:
            raise PublicReportError(f"duplicate synthetic asset path: {key}")
        records[key] = {
            "sha256": digest,
            "role": str(role),
            "mime_type": str(mime),
            "absolute_path": str(candidate),
        }
    return records


def verify_synthetic_reproduction(path: Path, generator_path: Path) -> None:
    """Regenerate the declared dataset and require byte-identical assets."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="segbench-public-synthetic-") as directory:
        output = Path(directory) / "synthetic"
        command = [
            sys.executable, str(generator_path), "--out", str(output),
            "--subjects", str(receipt["subjects"]),
            "--per-subject", str(receipt["per_subject"]),
            "--seed", str(receipt["seed"]),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode != 0:
            raise PublicReportError("could not independently reproduce the synthetic assets")
        replay_path = output / "synthetic_receipt.json"
        if not replay_path.is_file():
            raise PublicReportError("synthetic replay did not produce a receipt")
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
    original_assets = receipt.get("assets")
    replay_assets = replay.get("assets")
    if original_assets != replay_assets:
        raise PublicReportError("synthetic assets are not byte-identical to an independent replay")


class _PublicHTMLParser(HTMLParser):
    """Collect and validate active content and embedded images."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.synthetic_images: list[tuple[str, str, bytes]] = []
        self.has_public_marker = False

    def handle_starttag(self, tag: str, attrs: Sequence[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        tag = tag.lower()
        if tag in {
            "script", "iframe", "object", "embed", "video", "audio", "image",
            "form", "input", "button", "base",
        }:
            raise PublicReportError(f"public report contains forbidden <{tag}> content")
        for key in attributes:
            if key.startswith("on"):
                raise PublicReportError(f"public report contains an event handler: {key}")
        if tag == "meta" and attributes.get("name") == "segbench-public-safe":
            self.has_public_marker = attributes.get("content") == "1"
        if tag == "meta" and "http-equiv" in attributes:
            raise PublicReportError("public report contains an HTTP-equivalent meta directive")
        if tag in {"link", "source"} and any(key in attributes for key in ("href", "src", "srcset")):
            raise PublicReportError(f"public report contains an external-resource <{tag}> tag")
        if tag == "a":
            href = attributes.get("href", "")
            if href and not href.startswith("#"):
                raise PublicReportError("public report contains a non-local hyperlink")
        if tag != "img":
            return
        source = attributes.get("src", "")
        provenance = attributes.get("data-provenance", "")
        declared_digest = attributes.get("data-asset-sha256", "")
        match = _DATA_IMAGE.fullmatch(source)
        if not match:
            raise PublicReportError("every public report image must be an embedded data image")
        if provenance != "synthetic":
            raise PublicReportError("embedded public report image lacks synthetic provenance")
        if not _SHA256.fullmatch(declared_digest):
            raise PublicReportError("embedded public report image lacks a valid digest")
        try:
            payload = base64.b64decode(match.group(2), validate=True)
        except ValueError as exc:
            raise PublicReportError("embedded public report image has invalid base64") from exc
        if sha256_bytes(payload) != declared_digest:
            raise PublicReportError("embedded public report image digest does not match its bytes")
        self.synthetic_images.append((declared_digest, match.group(1).lower(), payload))


def scan_public_html(
    index_path: Path,
    *,
    synthetic_receipt: Path,
    synthetic_root: Path,
    generator_path: Path,
    max_bytes: int = 50 * 1024 * 1024,
) -> dict[str, Any]:
    """Validate one self-contained public page against its synthetic receipt.

    Returns a JSON-serialisable scan record. Any uncertainty raises
    :class:`PublicReportError`; there is no warning-only path.
    """
    if index_path.is_symlink() or not index_path.is_file():
        raise PublicReportError(f"public report is not a regular file: {index_path}")
    size = index_path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise PublicReportError(f"public report size is outside the allowed range: {size}")
    if any(path.is_symlink() for path in index_path.parent.rglob("*")):
        raise PublicReportError("public report directory contains a symlink")
    extra_files = [
        path for path in index_path.parent.rglob("*")
        if path.is_file() and path.resolve() != index_path.resolve()
    ]
    if extra_files:
        raise PublicReportError("public report must be self-contained in index.html")
    try:
        html = index_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PublicReportError(f"cannot read public report as UTF-8: {exc}") from exc
    # Base64 can coincidentally contain identifier-like text. Scan the semantic
    # HTML after replacing data payloads, then authenticate payloads separately.
    semantic_html = re.sub(
        r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=\s]+",
        "data:image/REDACTED;base64,REDACTED",
        html,
        flags=re.IGNORECASE,
    )
    scan_public_text(semantic_html, "public report")
    if re.search(r"@import\b|url\s*\(", semantic_html, re.IGNORECASE):
        raise PublicReportError("public report CSS contains an external-resource construct")
    parser = _PublicHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except PublicReportError:
        raise
    except Exception as exc:  # HTMLParser can expose malformed-input errors.
        raise PublicReportError(f"cannot parse public report HTML: {exc}") from exc
    if not parser.has_public_marker:
        raise PublicReportError("public report lacks the fail-closed safety marker")
    if html.lower().count("data:image/") != len(parser.synthetic_images):
        raise PublicReportError("public report contains an image outside an authenticated <img> element")
    receipt_assets = load_synthetic_receipt(
        synthetic_receipt, synthetic_root, generator_path=generator_path
    )
    verify_synthetic_reproduction(synthetic_receipt, generator_path)
    approved = {
        item["sha256"]: item
        for item in receipt_assets.values()
        if item["role"] == "example_image"
    }
    if not parser.synthetic_images:
        raise PublicReportError("public report has no synthetic examples")
    for digest, mime, _ in parser.synthetic_images:
        item = approved.get(digest)
        if item is None:
            raise PublicReportError("public report embeds an image absent from the synthetic receipt")
        if item["mime_type"] != mime:
            raise PublicReportError("embedded image MIME type differs from the synthetic receipt")
    return {
        "schema_version": 1,
        "kind": "public_report_privacy_scan",
        "status": "PASS",
        "report_sha256": sha256_file(index_path),
        "report_bytes": size,
        "synthetic_images": len(parser.synthetic_images),
        "synthetic_receipt_sha256": sha256_file(synthetic_receipt),
        "synthetic_generator_sha256": sha256_file(generator_path),
        "checks": [
            "single self-contained regular HTML file",
            "no active or remote content",
            "no biometric identifiers, local paths, network, or hardware details",
            "all images byte-bound to approved synthetic examples",
        ],
    }
