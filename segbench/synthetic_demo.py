"""Validation helpers for the public, synthetic-only method demonstration.

The public site must never infer that an image is synthetic merely from its
filename.  Instead, every source, target and prediction is content-addressed
by ``synthetic_receipt.json``.  This module validates the stronger method-run
receipt written by :mod:`scripts.build_synthetic_site_demo` without importing
PyTorch or any optional model runtime, so the site and privacy scanner can use
it in a core-only environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


METHOD_KEYS = (
    "gmfs_default",
    "gmfs_tuned",
    "sufs_zero",
    "sufs_ft",
    "svm",
    "rf",
    "welm",
    "lenet",
    "alexnet",
    "cnn_ae",
    "unet",
    "unetpp",
    "drunet",
    "runet_no_alignment",
    "segformer_sub",
    "fingernet",
)
METHOD_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
METHOD_STATUSES = {"complete", "unavailable", "not_built"}
EXTERNAL_UNAVAILABLE = {
    "segformer_sub": (
        "The public repository does not ship the pinned ImageNet SegFormer "
        "snapshot required by the benchmark implementation."
    ),
    "fingernet": (
        "The public repository does not ship the official FingerNet weights "
        "or its archival legacy runtime."
    ),
}


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    """Hash a JSON-compatible value with a stable canonical encoding."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prediction_set_sha256(items: list[Mapping[str, Any]]) -> str:
    """Hash the declared method/source/path/content identity of predictions."""
    rows = [
        {
            "method": str(item["method"]),
            "source_stem": str(item["source_stem"]),
            "path": str(item["path"]),
            "sha256": str(item["sha256"]),
        }
        for item in sorted(items, key=lambda row: str(row["source_stem"]))
    ]
    return canonical_sha256(rows)


def method_record_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the fingerprint of a method record excluding its fingerprint."""
    payload = {key: value for key, value in record.items() if key != "fingerprint_sha256"}
    return canonical_sha256(payload)


def _safe_relative_png(value: Any) -> Path:
    path = Path(str(value))
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix.lower() != ".png"
        or not path.parts
    ):
        raise ValueError(f"unsafe synthetic PNG path: {value!r}")
    return path


def validate_synthetic_demo_receipt(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    verify_files: bool = True,
    verify_builder: bool = True,
    builder_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a synthetic method receipt and return its normalized index.

    Args:
        root: Directory containing ``synthetic_receipt.json`` and its assets.
        receipt: Parsed receipt mapping.
        verify_files: Re-hash every declared file when true.
        verify_builder: Require the recorded builder bytes to match the current
            public builder script. Tests may disable only this check when
            validating a copied fixture outside the repository.
        builder_path: Explicit staged-builder path. Site builds in an isolated
            workspace pass this to avoid relying on Python's module cache.

    Returns:
        A mapping containing method records and prediction assets by key.

    Raises:
        ValueError: If any identity, completeness or availability rule fails.
    """
    demo = receipt.get("synthetic_demo")
    if not isinstance(demo, Mapping) or demo.get("schema_version") != 1:
        raise ValueError("synthetic receipt lacks a schema-v1 method demo")
    if demo.get("kind") != "segbench_synthetic_method_demo":
        raise ValueError("unexpected synthetic method-demo kind")
    if demo.get("device") != "CPU" or demo.get("benchmark_evidence") is not False:
        raise ValueError("synthetic demo must declare CPU and benchmark_evidence=false")
    manifest = root / "manifest.csv"
    if not manifest.is_file() or demo.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("synthetic demo manifest identity mismatch")
    if verify_builder:
        builder = builder_path or (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_synthetic_site_demo.py"
        )
        if not builder.is_file() or demo.get("builder_sha256") != sha256_file(builder):
            raise ValueError("synthetic demo builder identity mismatch")

    display_stems = demo.get("display_source_stems")
    train_subjects = demo.get("train_subjects")
    validation_subjects = demo.get("validation_subjects")
    if (
        not isinstance(display_stems, list)
        or len(display_stems) != 3
        or len(set(map(str, display_stems))) != 3
    ):
        raise ValueError("synthetic demo must bind exactly three unique display sources")
    if not isinstance(train_subjects, list) or not train_subjects:
        raise ValueError("synthetic demo lacks train-subject provenance")
    if not isinstance(validation_subjects, list) or not validation_subjects:
        raise ValueError("synthetic demo lacks validation-subject provenance")
    if set(map(str, train_subjects)) & set(map(str, validation_subjects)):
        raise ValueError("synthetic train and validation subjects overlap")

    assets = receipt.get("assets")
    if not isinstance(assets, list):
        raise ValueError("synthetic receipt assets must be a list")
    prediction_assets: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHOD_KEYS
    }
    seen_paths: set[str] = set()
    for raw in assets:
        if not isinstance(raw, Mapping):
            raise ValueError("synthetic receipt contains a malformed asset")
        path = _safe_relative_png(raw.get("path"))
        relative = path.as_posix()
        if relative in seen_paths:
            raise ValueError(f"duplicate synthetic asset path: {relative}")
        seen_paths.add(relative)
        if raw.get("mime_type") != "image/png":
            raise ValueError(f"synthetic asset has unexpected MIME type: {relative}")
        digest = str(raw.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"synthetic asset lacks a SHA-256 digest: {relative}")
        if verify_files:
            file_path = root / path
            if not file_path.is_file() or sha256_file(file_path) != digest:
                raise ValueError(f"synthetic asset bytes differ from receipt: {relative}")
        if raw.get("role") != "method_prediction":
            continue
        method = str(raw.get("method", ""))
        source_stem = str(raw.get("source_stem", ""))
        if method not in prediction_assets or not METHOD_KEY_PATTERN.fullmatch(method):
            raise ValueError(f"unknown synthetic method key: {method!r}")
        if source_stem not in set(map(str, display_stems)):
            raise ValueError(f"prediction references an unapproved source: {source_stem!r}")
        prediction_assets[method].append(dict(raw))

    raw_records = demo.get("methods")
    if not isinstance(raw_records, list) or len(raw_records) != len(METHOD_KEYS):
        raise ValueError("synthetic demo must contain exactly one record per method row")
    records: dict[str, dict[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("synthetic demo contains a malformed method record")
        record = dict(raw)
        method = str(record.get("method", ""))
        if method not in METHOD_KEYS or method in records:
            raise ValueError(f"duplicate or unknown synthetic method record: {method!r}")
        status = str(record.get("status", ""))
        if status not in METHOD_STATUSES:
            raise ValueError(f"invalid synthetic method status for {method}: {status!r}")
        if not str(record.get("label", "")).strip():
            raise ValueError(f"synthetic method {method} lacks a display label")
        if record.get("fingerprint_sha256") != method_record_fingerprint(record):
            raise ValueError(f"synthetic method record fingerprint mismatch: {method}")
        predictions = prediction_assets[method]
        if status == "complete":
            if sorted(str(item["source_stem"]) for item in predictions) != sorted(
                map(str, display_stems)
            ):
                raise ValueError(f"complete method {method} lacks all three predictions")
            if record.get("prediction_set_sha256") != prediction_set_sha256(predictions):
                raise ValueError(f"prediction-set identity mismatch: {method}")
            if not isinstance(record.get("execution"), Mapping):
                raise ValueError(f"complete method {method} lacks execution provenance")
            if str(record.get("reason", "")):
                raise ValueError(f"complete method {method} must not carry a failure reason")
        else:
            if predictions:
                raise ValueError(f"non-complete method {method} has fabricated predictions")
            if not str(record.get("reason", "")).strip():
                raise ValueError(f"non-complete method {method} lacks an explicit reason")
            if record.get("prediction_set_sha256") not in {None, ""}:
                raise ValueError(f"non-complete method {method} declares a prediction set")
        records[method] = record
    if tuple(records) != METHOD_KEYS:
        raise ValueError("synthetic method records are not in the frozen public order")

    record_set = [records[key] for key in METHOD_KEYS]
    if demo.get("method_record_set_sha256") != canonical_sha256(record_set):
        raise ValueError("synthetic method-record set identity mismatch")
    return {
        "display_source_stems": list(map(str, display_stems)),
        "methods": records,
        "predictions": prediction_assets,
    }
