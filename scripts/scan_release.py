#!/usr/bin/env python3
"""Fail-closed privacy and hygiene scan for the public release tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
BANNED_ARTIFACT_SUFFIXES = {".pt", ".pth", ".joblib", ".npy", ".h5"}
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".toml",
    ".cff", ".html", ".css", ".js", ".svg", ".sh", ".ipynb",
}


def patterns() -> list[tuple[str, re.Pattern[str]]]:
    return [
        ("absolute private path", re.compile("/" + "(?:home|mnt|data|restricted)/")),
        ("cloud-drive URL", re.compile("(?:drive\\." + "google\\.com|docs\\." + "google\\.com)", re.I)),
        ("IPv4 address", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
        ("SD302-like image identifier", re.compile(r"(?<![A-Za-z0-9])\d{8}_[A-Za-z0-9]")),
        ("source-resolution filename token", re.compile("1000" + "PPI", re.I)),
        ("latent-position filename token", re.compile(r"\bLP\d{2}\b", re.I)),
    ]


def files() -> Iterable[Path]:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if ".git" in path.parts or any(part in {".smoke", ".venv", ".pytest_cache", "__pycache__", "private_data", "local"} for part in relative.parts) or not path.is_file():
            continue
        yield path


def allowed_identifier_doc(path: Path, label: str) -> bool:
    return path == ROOT / "docs" / "DATASETS.md" and label in {
        "SD302-like image identifier", "source-resolution filename token", "latent-position filename token"
    }


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


def scan_history(errors: list[str]) -> None:
    if not (ROOT / ".git").is_dir():
        return
    result = subprocess.run(["git", "log", "-p", "--all", "--no-ext-diff"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        errors.append("could not inspect git history")
        return
    history = result.stdout
    history_patterns = patterns()[:3]
    for label, pattern in history_patterns:
        match = pattern.search(history)
        if match:
            errors.append(f"git history contains {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "release_scan.json")
    args = parser.parse_args()
    errors: list[str] = []
    approved_hashes = approved_synthetic_hashes()
    checked = 0
    for path in files():
        checked += 1
        rel = path.relative_to(ROOT)
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
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF8 text artifact: {rel}")
            continue
        for label, pattern in patterns():
            if allowed_identifier_doc(path, label):
                continue
            if pattern.search(text):
                errors.append(f"{rel}: contains {label}")
    scan_history(errors)
    report = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "files_checked": checked,
        "approved_synthetic_hashes": len(approved_hashes),
        "checks": [
            "no biometric filename patterns outside the naming explanation",
            "no private absolute paths, cloud-drive references, or IP addresses",
            "no trained-weight or data-array artifacts",
            "all raster assets are receipt-bound synthetic images",
            "no file exceeds 10 MB",
            "git patch history has no private paths, cloud-drive URLs, or IP addresses",
        ],
        "errors": errors,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
