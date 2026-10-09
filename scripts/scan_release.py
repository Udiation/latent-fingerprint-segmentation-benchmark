#!/usr/bin/env python3
"""Fail-closed privacy and hygiene scan for the public release tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segbench.synthetic_demo import validate_synthetic_demo_receipt

MAX_BYTES = 10 * 1024 * 1024
BANNED_ARTIFACT_SUFFIXES = {
    ".bin", ".ckpt", ".h5", ".joblib", ".npy", ".npz", ".onnx", ".pt",
    ".pth", ".safetensors",
}
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".toml",
    ".cff", ".html", ".css", ".js", ".svg", ".sh", ".ipynb",
}
ALLOWED_NON_TEXT_NAMES = {"LICENSE", ".gitignore"}
_OPAQUE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{25,80}(?![A-Za-z0-9_-])"
)


def patterns() -> list[tuple[str, re.Pattern[str]]]:
    return [
        ("absolute private path", re.compile("/" + "(?:home|mnt|data|restricted)/")),
        ("cloud-drive URL", re.compile("(?:drive\\." + "google\\.com|docs\\." + "google\\.com)", re.I)),
        ("IPv4 address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
        ("SD302-like image identifier", re.compile(r"(?<![A-Za-z0-9])\d{8}_[A-Za-z0-9]")),
        ("source-resolution filename token", re.compile("1000" + "PPI", re.I)),
        ("latent-position filename token", re.compile(r"\bLP\d{2}\b", re.I)),
        (
            "local infrastructure hostname",
            re.compile(
                r"(?:\b[a-z0-9-]+\.local\b|"
                r"\b(?:ssh|scp)\s+(?:[^\s@]+@)?[A-Za-z0-9._-]+)",
                re.I,
            ),
        ),
    ]


def opaque_identifier(value: str) -> str | None:
    """Return a cloud-drive-like token without embedding any private denylist.

    Cryptographic digests are uniformly hexadecimal, while repository symbols
    tend to contain several word separators.  A bounded mixed-case token with
    a digit and no more than two separators is therefore treated as an opaque
    external identifier and rejected.  This covers raw cloud file IDs without
    confusing SHA-1/SHA-256 values or descriptive Python identifiers for one.
    """
    for token in _OPAQUE_TOKEN.findall(value):
        if (
            any(character.islower() for character in token)
            and any(character.isupper() for character in token)
            and any(character.isdigit() for character in token)
            and sum(character in "_-" for character in token) <= 2
        ):
            return token
    return None


def files() -> Iterable[Path]:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if ".git" in path.parts or any(
            part in {".smoke", ".venv", ".pytest_cache", "__pycache__"}
            for part in relative.parts
        ) or not path.is_file():
            continue
        yield path


def approved_synthetic_hashes() -> set[str]:
    receipt = ROOT / "examples" / "synthetic" / "synthetic_receipt.json"
    if not receipt.is_file():
        return set()
    data = json.loads(receipt.read_text(encoding="utf-8"))
    approved_roles = {"example_image", "ground_truth_mask", "method_prediction"}
    return {
        str(item.get("sha256"))
        for item in data.get("assets", [])
        if item.get("role") in approved_roles
    }


def _scan_text(
    value: str,
    context: str,
    errors: list[str],
    *,
    allow_identifier_explanation: bool = False,
) -> None:
    """Scan semantic text or a repository path with one shared policy."""
    for label, pattern in patterns():
        if allow_identifier_explanation and label in {
            "SD302-like image identifier",
            "source-resolution filename token",
            "latent-position filename token",
        }:
            continue
        if pattern.search(value):
            errors.append(f"{context}: contains {label}")
    token = opaque_identifier(value)
    if token is not None:
        errors.append(f"{context}: contains opaque cloud-drive-like identifier")


def scan_history(errors: list[str]) -> None:
    """Scan every committed patch, excluding commit metadata such as author mail.

    The current-tree scan alone is insufficient: a secret or biometric ID that
    was deleted in a later commit remains public after a push.  We track each
    diff's path so the narrowly scoped naming explanation remains the sole
    exception for SD302-style example tokens.
    """
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if probe.returncode or probe.stdout.strip() != "true":
        return
    result = subprocess.run(
        ["git", "log", "--format=", "-p", "--all", "--no-ext-diff"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        errors.append("could not inspect git history")
        return
    current_path = ""
    for line_number, line in enumerate(result.stdout.splitlines(), start=1):
        if line.startswith("diff --git a/"):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            current_path = match.group(2) if match else ""
            _scan_text(
                current_path,
                f"git history path at patch line {line_number}",
                errors,
                allow_identifier_explanation=current_path == "docs/DATASETS.md",
            )
            continue
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        _scan_text(
            line[1:],
            f"git history {current_path or 'unknown path'} at patch line {line_number}",
            errors,
            allow_identifier_explanation=current_path == "docs/DATASETS.md",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "release_scan.json")
    parser.add_argument(
        "--allow-limited-synthetic",
        action="store_true",
        help="CI-only: accept a receipt-valid subset build with at least one completed method",
    )
    args = parser.parse_args()
    errors: list[str] = []
    approved_hashes = approved_synthetic_hashes()
    synthetic_root = ROOT / "examples" / "synthetic"
    synthetic_receipt = synthetic_root / "synthetic_receipt.json"
    try:
        payload = json.loads(synthetic_receipt.read_text(encoding="utf-8"))
        index = validate_synthetic_demo_receipt(synthetic_root, payload)
        statuses = [row["status"] for row in index["methods"].values()]
        expected_complete = 14
        if statuses.count("unavailable") != 2:
            errors.append("synthetic demo does not declare both external-weight methods unavailable")
        if args.allow_limited_synthetic:
            if statuses.count("complete") < 1:
                errors.append("limited synthetic demo executes no method")
        elif statuses.count("complete") != expected_complete:
            errors.append(
                f"synthetic demo has {statuses.count('complete')} complete methods; "
                f"release requires {expected_complete}"
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"synthetic method-demo receipt failed validation: {type(exc).__name__}")
    checked = 0
    for path in files():
        checked += 1
        rel = path.relative_to(ROOT)
        _scan_text(
            rel.as_posix(),
            f"repository path {rel}",
            errors,
            allow_identifier_explanation=rel == Path("docs/DATASETS.md"),
        )
        if path.is_symlink():
            errors.append(f"symlink: {rel}")
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            errors.append(f"file over 10 MB: {rel}")
        if path.suffix.lower() in BANNED_ARTIFACT_SUFFIXES:
            errors.append(f"model/data artifact extension: {rel}")
        if path.suffix.lower() in RASTER_SUFFIXES:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if rel.parts[:2] == ("examples", "synthetic") or rel.parts[:3] == ("site", "assets", "synthetic"):
                if digest not in approved_hashes:
                    errors.append(f"raster lacks synthetic receipt provenance: {rel}")
            elif rel.parts[:1] == ("figures",):
                errors.append(f"plot raster requires a generated figure manifest: {rel}")
            else:
                errors.append(f"raster outside approved synthetic/figure directories: {rel}")
        if (
            path.suffix.lower() not in TEXT_SUFFIXES | RASTER_SUFFIXES
            and path.name not in ALLOWED_NON_TEXT_NAMES
        ):
            errors.append(f"unapproved binary or file type: {rel}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in ALLOWED_NON_TEXT_NAMES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF8 text artifact: {rel}")
            continue
        _scan_text(
            text,
            str(rel),
            errors,
            allow_identifier_explanation=path == ROOT / "docs" / "DATASETS.md",
        )
    scan_history(errors)
    report = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "files_checked": checked,
        "approved_synthetic_hashes": len(approved_hashes),
        "checks": [
            "no biometric filename patterns outside the naming explanation",
            "no private absolute paths, cloud-drive references, or IP addresses",
            "no local infrastructure hostnames or opaque cloud-drive-like identifiers",
            "no trained-weight or data-array artifacts",
            "all raster assets are receipt-bound synthetic images",
            "synthetic method outputs pass provenance validation and release completeness",
            "no unapproved binary file types",
            "no file exceeds 10 MB",
            "git patch history has no private paths, identifiers, infrastructure, or cloud-drive tokens",
        ],
        "errors": errors,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
