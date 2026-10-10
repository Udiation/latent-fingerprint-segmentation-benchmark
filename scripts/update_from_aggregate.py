#!/usr/bin/env python3
"""Import one privacy-screened Stage-4 aggregate bundle into the public tree.

The importer accepts exactly three aggregate CSVs plus a hash manifest. It
never opens per-image results, images, masks, probability maps, checkpoints,
or weights. All inputs are validated and the documentation/site are rebuilt in
a temporary workspace before any tracked output is replaced.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from types import ModuleType
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_KIND = "stage4_public_aggregate_bundle"
REPORT_KIND = "public_aggregate_update_report"
EXPECTED_FILES = ("public_results.csv", "statistics.csv", "table_status.csv")
MANIFEST_NAME = "aggregate_manifest.json"
MAX_INPUT_BYTES = 5 * 1024 * 1024
LOWER_HEX = set("0123456789abcdef")
README_RESULTS_START = "<!-- BEGIN GENERATED HEADLINE RESULTS -->"
README_RESULTS_END = "<!-- END GENERATED HEADLINE RESULTS -->"
README_B1_START = "<!-- BEGIN GENERATED B1 RESULTS -->"
README_B1_END = "<!-- END GENERATED B1 RESULTS -->"

RESULT_COLUMNS = (
    "table",
    "view",
    "group",
    "year",
    "method",
    "approach",
    "family",
    "training_data",
    "comment",
    "status",
    "seeds_completed",
    "n_images",
    "n_subjects",
    "accuracy",
    "accuracy_seed_std",
    "error_rate",
    "error_rate_seed_std",
    "dice",
    "dice_seed_std",
    "jaccard",
    "jaccard_seed_std",
    "miou",
    "miou_seed_std",
    "mdr",
    "mdr_seed_std",
    "fdr",
    "fdr_seed_std",
    "mc",
    "mc_seed_std",
    "hr",
    "hr_seed_std",
    "par",
    "par_seed_std",
    "ohf",
    "ohf_seed_std",
    "params_m",
    "params_m_seed_std",
    "ms_per_img",
    "ms_per_img_seed_std",
    "device",
    "ci_lower",
    "ci_upper",
)
STATISTIC_COLUMNS = (
    "table",
    "metric",
    "comparison",
    "estimate",
    "ci_lower",
    "ci_upper",
    "p_holm",
    "significant",
    "status",
)
TABLE_COLUMNS = ("table", "title", "status", "note")
ALLOWED_STATUSES = {"pending", "complete", "n/a"}
HULL_TABLES = {"B1", "D"}
UNIT_INTERVAL_FIELDS = {
    "accuracy",
    "error_rate",
    "dice",
    "jaccard",
    "miou",
    "mdr",
    "fdr",
    "mc",
    "hr",
    "ohf",
    "ci_lower",
    "ci_upper",
    "accuracy_seed_std",
    "error_rate_seed_std",
    "dice_seed_std",
    "jaccard_seed_std",
    "miou_seed_std",
    "mdr_seed_std",
    "fdr_seed_std",
    "mc_seed_std",
    "hr_seed_std",
    "ohf_seed_std",
}
FORBIDDEN_FIELD_NAMES = {
    "image_id",
    "subject_id",
    "sample_id",
    "record_id",
    "image_path",
    "mask_path",
    "filename",
    "file_path",
    "checkpoint_path",
    "weight_path",
}


class AggregateImportError(ValueError):
    """The candidate aggregate bundle is unsafe, malformed, or inconsistent."""


@dataclass(frozen=True)
class AggregateBundle:
    """Fully validated aggregate bytes and parsed rows."""

    manifest_sha256: str
    file_sha256: Mapping[str, str]
    raw_files: Mapping[str, bytes]
    results: tuple[Mapping[str, str], ...]
    statistics: tuple[Mapping[str, str], ...]
    tables: tuple[Mapping[str, str], ...]


class _PayloadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.capture = False
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if (
            tag == "script"
            and values.get("id") == "public-data"
            and values.get("type") == "application/json"
        ):
            self.capture = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.capture:
            self.capture = False

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.parts.append(data)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in LOWER_HEX for character in value)
    )


def _read_regular(path: Path, description: str) -> bytes:
    """Read a bounded regular file without following a final symlink."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise AggregateImportError("symlink-safe reads are unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AggregateImportError(f"cannot open {description} as a regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AggregateImportError(f"{description} is not a regular file")
        if metadata.st_size > MAX_INPUT_BYTES:
            raise AggregateImportError(f"{description} exceeds the aggregate size limit")
        chunks: list[bytes] = []
        remaining = MAX_INPUT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_INPUT_BYTES:
            raise AggregateImportError(f"{description} exceeds the aggregate size limit")
        observed = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise AggregateImportError(f"{description} changed while being read")
        return data
    finally:
        os.close(descriptor)


def _forbidden_text_reason(value: str) -> str | None:
    """Return why an aggregate cell is unsafe, or ``None`` when it is safe."""
    if re.fullmatch(
        r"\s*selection_group:\s*(?:mixed|native)\.[a-z0-9_]+\.seed\d+\s*",
        value,
        re.IGNORECASE,
    ):
        # Validation-selection group syntax, not a DNS/infrastructure host.
        return None
    folded = value.casefold()
    forbidden_names = "|".join(sorted(re.escape(item) for item in FORBIDDEN_FIELD_NAMES))
    checks = (
        (
            "per-image identifier field",
            re.compile(r"(?<![a-z0-9])(?:" + forbidden_names + r")(?![a-z0-9])", re.I),
        ),
        (
            "biometric-style sample identifier",
            re.compile(r"(?<![A-Za-z0-9])\d{8}_[A-Za-z0-9]"),
        ),
        (
            "source-resolution filename token",
            re.compile("1000" + "PPI", re.I),
        ),
        ("latent-position filename token", re.compile(r"\bLP\d{2}\b", re.I)),
        (
            "absolute private path",
            re.compile(
                r"(?<![:A-Za-z0-9])/"
                r"(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+",
                re.I,
            ),
        ),
        ("Windows absolute path", re.compile(r"\b[A-Za-z]:[\\/]")),
        (
            "IPv4 address",
            re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
        ),
        ("IPv6 address", re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}:){2,}")),
        (
            "weight or data-array path",
            re.compile(r"(?:^|[\\/\s])[^\s]+\.(?:pt|pth|h5|joblib|npy)(?:$|\s)", re.I),
        ),
        (
            "host name",
            re.compile(
                r"(?:\b(?:localhost|host(?:name)?|server)\s*[:=]|"
                r"\b(?:node|compute|server|host)[-_]?[A-Za-z0-9]+\b|"
                r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}\b)",
                re.I,
            ),
        ),
        (
            "cloud-drive reference",
            re.compile(
                r"(?:drive\." + r"google\.com|docs\." + r"google\.com|uc\?id=)",
                re.I,
            ),
        ),
        ("embedded image payload", re.compile(r"(?:data:image/|base64,)", re.I)),
    )
    for label, pattern in checks:
        if pattern.search(value):
            return label
    for token in re.findall(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{25,}(?![A-Za-z0-9_-])", value):
        if any(c.islower() for c in token) and any(c.isupper() for c in token) and any(
            c.isdigit() for c in token
        ):
            return "opaque cloud-drive-like identifier"
    if "\x00" in value:
        return "NUL byte"
    if len(value) > 1000:
        return "oversized aggregate cell"
    if "\r" in value or "\n" in value:
        return "multiline aggregate cell"
    if value[:1] in {"=", "+", "@"}:
        return "spreadsheet formula cell"
    if "data:image" in folded:
        return "embedded image payload"
    return None


def _scan_cell(value: str, location: str) -> None:
    reason = _forbidden_text_reason(value)
    if reason:
        raise AggregateImportError(f"{location} contains {reason}")


def _decode_utf8(data: bytes, description: str) -> str:
    if data.startswith((b"\x89PNG", b"\xff\xd8\xff", b"RIFF", b"PK\x03\x04")):
        raise AggregateImportError(f"{description} contains binary/image/archive bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AggregateImportError(f"{description} is not UTF-8 text") from exc
    if text.startswith("\ufeff"):
        raise AggregateImportError(f"{description} has an unexpected byte-order mark")
    return text


def _read_csv(
    data: bytes, name: str, expected_columns: Sequence[str]
) -> list[dict[str, str]]:
    text = _decode_utf8(data, name)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != tuple(expected_columns):
        raise AggregateImportError(
            f"{name} columns changed; expected {list(expected_columns)}"
        )
    rows: list[dict[str, str]] = []
    for number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise AggregateImportError(f"{name}:{number} has a malformed field count")
        normalized = {key: str(value) for key, value in row.items()}
        for key, value in normalized.items():
            if value != value.strip():
                raise AggregateImportError(f"{name}:{number}:{key} has edge whitespace")
            _scan_cell(value, f"{name}:{number}:{key}")
        rows.append(normalized)
    if not rows:
        raise AggregateImportError(f"{name} contains no aggregate rows")
    return rows


def _number(
    value: str,
    location: str,
    *,
    minimum: float,
    maximum: float | None = None,
    allow_blank: bool = True,
) -> float | None:
    if value == "" and allow_blank:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise AggregateImportError(f"{location} is not numeric") from exc
    if not math.isfinite(number) or number < minimum or (
        maximum is not None and number > maximum
    ):
        raise AggregateImportError(f"{location} is outside its allowed range")
    return number


def _validate_results(rows: Sequence[Mapping[str, str]]) -> None:
    identities: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(rows, start=2):
        prefix = f"public_results.csv:{index}"
        for key in (
            "table",
            "view",
            "group",
            "method",
            "approach",
            "family",
            "training_data",
            "comment",
            "status",
        ):
            if not row[key]:
                raise AggregateImportError(f"{prefix}:{key} is empty")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", row["table"]):
            raise AggregateImportError(f"{prefix}:table is not a public table key")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", row["method"]):
            raise AggregateImportError(f"{prefix}:method is not a stable method key")
        identity = (row["table"], row["view"], row["group"], row["method"])
        if identity in identities:
            raise AggregateImportError(f"{prefix} duplicates table/view/group/method {identity}")
        identities.add(identity)
        if row["status"] not in ALLOWED_STATUSES:
            raise AggregateImportError(f"{prefix}:status is not recognised")
        if row["year"]:
            if not row["year"].isdigit() or not 1900 <= int(row["year"]) <= 2100:
                raise AggregateImportError(f"{prefix}:year is invalid")
        if not row["seeds_completed"].isdigit() or not 0 <= int(row["seeds_completed"]) <= 100:
            raise AggregateImportError(f"{prefix}:seeds_completed is invalid")
        for field in UNIT_INTERVAL_FIELDS:
            _number(row[field], f"{prefix}:{field}", minimum=0.0, maximum=1.0)
        for field in (
            "par", "par_seed_std", "params_m", "params_m_seed_std",
            "ms_per_img", "ms_per_img_seed_std",
        ):
            _number(row[field], f"{prefix}:{field}", minimum=0.0)
        for field in ("n_images", "n_subjects"):
            value = row[field]
            if value and (not value.isdigit() or int(value) <= 0):
                raise AggregateImportError(f"{prefix}:{field} must be a positive integer")
        if row["device"] not in {"", "CPU", "GPU"}:
            raise AggregateImportError(f"{prefix}:device must be blank, CPU, or GPU")
        bounds = (row["ci_lower"], row["ci_upper"])
        if (bounds[0] == "") != (bounds[1] == ""):
            raise AggregateImportError(f"{prefix} has only one confidence bound")
        if bounds[0] and float(bounds[0]) > float(bounds[1]):
            raise AggregateImportError(f"{prefix} confidence bounds are reversed")
        if row["status"] == "complete":
            required = (
                ("mc", "hr", "par", "ohf")
                if row["table"] in HULL_TABLES
                else ("dice", "jaccard", "mdr", "fdr", "accuracy")
            )
            if any(row[field] == "" for field in required):
                raise AggregateImportError(f"{prefix} complete row lacks a metric")


def _validate_tables(rows: Sequence[Mapping[str, str]]) -> set[str]:
    observed: set[str] = set()
    for index, row in enumerate(rows, start=2):
        prefix = f"table_status.csv:{index}"
        if not row["table"] or not row["title"] or not row["note"]:
            raise AggregateImportError(f"{prefix} has an empty required field")
        if row["table"] in observed:
            raise AggregateImportError(f"{prefix} duplicates a table key")
        observed.add(row["table"])
        if row["status"] not in ALLOWED_STATUSES:
            raise AggregateImportError(f"{prefix}:status is not recognised")
    return observed


def _validate_statistics(rows: Sequence[Mapping[str, str]]) -> None:
    identities: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows, start=2):
        prefix = f"statistics.csv:{index}"
        if not row["table"] or not row["metric"] or not row["comparison"]:
            raise AggregateImportError(f"{prefix} has an empty identity field")
        identity = (row["table"], row["metric"], row["comparison"])
        if identity in identities:
            raise AggregateImportError(f"{prefix} duplicates a statistical comparison")
        identities.add(identity)
        if row["status"] not in ALLOWED_STATUSES:
            raise AggregateImportError(f"{prefix}:status is not recognised")
        _number(row["estimate"], f"{prefix}:estimate", minimum=-1.0, maximum=1.0)
        _number(row["ci_lower"], f"{prefix}:ci_lower", minimum=-1.0, maximum=1.0)
        _number(row["ci_upper"], f"{prefix}:ci_upper", minimum=-1.0, maximum=1.0)
        _number(row["p_holm"], f"{prefix}:p_holm", minimum=0.0, maximum=1.0)
        if row["significant"] not in {"", "true", "false"}:
            raise AggregateImportError(f"{prefix}:significant must be true, false, or blank")
        lower, upper = row["ci_lower"], row["ci_upper"]
        if (lower == "") != (upper == ""):
            raise AggregateImportError(f"{prefix} has only one confidence bound")
        if lower and float(lower) > float(upper):
            raise AggregateImportError(f"{prefix} confidence bounds are reversed")


def validate_bundle(bundle_dir: Path) -> AggregateBundle:
    """Authenticate the exact bundle inventory, hashes, schemas, and content."""
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise AggregateImportError("aggregate bundle must be a non-symlink directory")
    expected_inventory = {MANIFEST_NAME, *EXPECTED_FILES}
    observed_inventory = {entry.name for entry in bundle_dir.iterdir()}
    if observed_inventory != expected_inventory:
        unexpected = sorted(observed_inventory - expected_inventory)
        missing = sorted(expected_inventory - observed_inventory)
        raise AggregateImportError(
            f"aggregate bundle inventory changed; missing={missing}, unexpected={unexpected}"
        )
    manifest_bytes = _read_regular(bundle_dir / MANIFEST_NAME, "aggregate manifest")
    manifest_text = _decode_utf8(manifest_bytes, "aggregate manifest")
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise AggregateImportError("aggregate manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "kind", "files"}:
        raise AggregateImportError("aggregate manifest fields changed")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != BUNDLE_KIND:
        raise AggregateImportError("aggregate manifest schema or kind changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_FILES):
        raise AggregateImportError("aggregate manifest file universe changed")
    if any(not _valid_sha256(value) for value in files.values()):
        raise AggregateImportError("aggregate manifest contains an invalid SHA-256")

    raw: dict[str, bytes] = {}
    for name in EXPECTED_FILES:
        data = _read_regular(bundle_dir / name, name)
        if _sha256(data) != files[name]:
            raise AggregateImportError(f"{name} differs from the aggregate manifest")
        raw[name] = data
    results = _read_csv(raw["public_results.csv"], "public_results.csv", RESULT_COLUMNS)
    statistics = _read_csv(raw["statistics.csv"], "statistics.csv", STATISTIC_COLUMNS)
    tables = _read_csv(raw["table_status.csv"], "table_status.csv", TABLE_COLUMNS)
    _validate_results(results)
    _validate_statistics(statistics)
    table_keys = _validate_tables(tables)
    referenced = {row["table"] for row in results} | {row["table"] for row in statistics}
    if not referenced.issubset(table_keys):
        raise AggregateImportError(
            "aggregate rows reference undeclared tables: " + ", ".join(sorted(referenced - table_keys))
        )
    return AggregateBundle(
        manifest_sha256=_sha256(manifest_bytes),
        file_sha256={name: str(files[name]) for name in EXPECTED_FILES},
        raw_files=raw,
        results=tuple(results),
        statistics=tuple(statistics),
        tables=tuple(tables),
    )


def _markdown(value: str) -> str:
    return value.replace("|", "\\|")


def _percent(value: str) -> str:
    return "pending" if value == "" else f"{float(value) * 100:.2f}%"


def _with_std(value: str, std: str, *, percent: bool = False) -> str:
    """Render a mean and its across-seed standard deviation when present."""
    if value == "":
        return "pending"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    rendered = f"{float(value) * scale:.2f}{suffix}"
    if std != "":
        rendered += f" ± {float(std) * scale:.2f}{suffix}"
    return rendered


def _display_status(row: Mapping[str, str]) -> str:
    """Render the one explicit training-pending state without changing source data."""
    if row["status"] == "pending" and "training — pending" in row["comment"]:
        return "training — pending"
    return row["status"]


def render_results_markdown(bundle: AggregateBundle) -> str:
    """Render the public results page from the same rows consumed by the site."""
    lines = [
        "# Results",
        "",
        "<!-- Generated by scripts/update_from_aggregate.py; do not edit by hand. -->",
        "",
        "All values on this page come from the hash-bound public aggregate bundle.",
        "No per-image records, biometric media, trained weights, or sample identifiers are imported.",
        "",
    ]
    for table in bundle.tables:
        display_table = table["table"].replace("_prime", "′").replace("_value", "-value")
        lines.extend(
            [
                f"## Table {_markdown(display_table)}: {_markdown(table['title'])}",
                "",
                f"Status: **{_markdown(table['status'])}** — {_markdown(table['note'])}",
                "",
            ]
        )
        rows = [row for row in bundle.results if row["table"] == table["table"]]
        if not rows:
            lines.extend(["No aggregate rows are published for this table yet.", ""])
            continue
        if table["table"] in HULL_TABLES:
            lines.extend(
                [
                    "| Approach | Group | Status | Seeds | MC | HR | PAR | OHF | "
                    "Params (M) | ms/img | Device |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in rows:
                values = (
                    _markdown(row["approach"]),
                    _markdown(row["group"]),
                    _display_status(row),
                    row["seeds_completed"],
                    _with_std(row["mc"], row["mc_seed_std"], percent=True),
                    _with_std(row["hr"], row["hr_seed_std"], percent=True),
                    _with_std(row["par"], row["par_seed_std"]),
                    _with_std(row["ohf"], row["ohf_seed_std"], percent=True),
                    _with_std(row["params_m"], row["params_m_seed_std"]),
                    _with_std(row["ms_per_img"], row["ms_per_img_seed_std"]),
                    row["device"] or "—",
                )
                lines.append("| " + " | ".join(values) + " |")
        else:
            lines.extend(
                [
                    "| Approach | Group | Family | Status | Seeds | Dice | Jaccard | "
                    "MDR | FDR | Params (M) | ms/img | Device |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in rows:
                values = (
                    _markdown(row["approach"]),
                    _markdown(row["group"]),
                    _markdown(row["family"]),
                    _display_status(row),
                    row["seeds_completed"],
                    _with_std(row["dice"], row["dice_seed_std"], percent=True),
                    _with_std(row["jaccard"], row["jaccard_seed_std"], percent=True),
                    _with_std(row["mdr"], row["mdr_seed_std"], percent=True),
                    _with_std(row["fdr"], row["fdr_seed_std"], percent=True),
                    _with_std(row["params_m"], row["params_m_seed_std"]),
                    _with_std(row["ms_per_img"], row["ms_per_img_seed_std"]),
                    row["device"] or "—",
                )
                lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(["## Aggregate statistical comparisons", ""])
    lines.extend(
        [
            "| Table | Metric | Comparison | Estimate | 95% CI | Holm p | Significant | Status |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in bundle.statistics:
        interval = (
            "pending"
            if not row["ci_lower"]
            else f"[{row['ci_lower']}, {row['ci_upper']}]"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(row["table"]),
                    _markdown(row["metric"]),
                    _markdown(row["comparison"]),
                    row["estimate"] or "pending",
                    interval,
                    row["p_holm"] or "pending",
                    row["significant"] or "pending",
                    row["status"],
                )
            )
            + " |"
        )
    completed = sum(row["status"] == "complete" for row in bundle.results)
    pending = sum(row["status"] == "pending" for row in bundle.results)
    factual = (
        f"This v1.0 provisional snapshot contains {completed} completed aggregate rows "
        f"and {pending} pending rows. Completed values are published exactly as supplied "
        "by the authenticated bundle; missing values are not inferred. The v1.1 refresh "
        "will use the next authenticated aggregate bundle."
        if pending
        else f"All {completed} aggregate rows in this snapshot are complete."
    )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "The public figure set is regenerated only from the same aggregate rows. "
            "Plots whose required aggregates are incomplete remain **pending**.",
            "No qualitative fingerprint imagery is published.",
            "",
            "## Factual reading",
            "",
            factual,
            "",
            "",
            "Source digests:",
            "",
            *[
                f"- `{name}`: `{bundle.file_sha256[name]}`"
                for name in EXPECTED_FILES
            ],
            "",
        ]
    )
    return "\n".join(lines)


def render_readme_headline(rows: Sequence[Mapping[str, str]]) -> str:
    """Render the README headline table from aggregate Table A rows."""
    table_rows = [
        row
        for row in rows
        if row["table"] == "A" and row["view"] == "overall" and row["group"] == "overall"
    ]
    lines = [
        README_RESULTS_START,
        "| Family | Approach | Status | Dice | Jaccard | MDR | FDR | Seeds | Params (M) | ms/img (device) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        runtime = _with_std(row["ms_per_img"], row["ms_per_img_seed_std"])
        if row["device"]:
            runtime += f" ({row['device']})"
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(row["family"]),
                    _markdown(row["approach"]),
                    _display_status(row),
                    _with_std(row["dice"], row["dice_seed_std"], percent=True),
                    _with_std(row["jaccard"], row["jaccard_seed_std"], percent=True),
                    _with_std(row["mdr"], row["mdr_seed_std"], percent=True),
                    _with_std(row["fdr"], row["fdr_seed_std"], percent=True),
                    row["seeds_completed"],
                    _with_std(row["params_m"], row["params_m_seed_std"]),
                    runtime,
                )
            )
            + " |"
        )
    lines.append(README_RESULTS_END)
    return "\n".join(lines)


def render_readme_b1(rows: Sequence[Mapping[str, str]]) -> str:
    """Render the provisional Table B1 rows without treating its hull as full GT."""
    table_rows = [
        row
        for row in rows
        if row["table"] == "B1" and row["view"] == "overall" and row["group"] == "overall"
    ]
    lines = [
        README_B1_START,
        "| Approach | Status | MC | HR | PAR | OHF | Seeds | Params (M) | ms/img (device) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        runtime = _with_std(row["ms_per_img"], row["ms_per_img_seed_std"])
        if row["device"]:
            runtime += f" ({row['device']})"
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(row["approach"]),
                    _display_status(row),
                    _with_std(row["mc"], row["mc_seed_std"], percent=True),
                    _with_std(row["hr"], row["hr_seed_std"], percent=True),
                    _with_std(row["par"], row["par_seed_std"]),
                    _with_std(row["ohf"], row["ohf_seed_std"], percent=True),
                    row["seeds_completed"],
                    _with_std(row["params_m"], row["params_m_seed_std"]),
                    runtime,
                )
            )
            + " |"
        )
    lines.append(README_B1_END)
    return "\n".join(lines)


def render_readme(readme: str, rows: Sequence[Mapping[str, str]]) -> str:
    """Replace only the marker-bound README table, refusing ambiguous input."""
    if readme.count(README_RESULTS_START) != 1 or readme.count(README_RESULTS_END) != 1:
        raise AggregateImportError("README generated-result markers are missing or duplicated")
    start = readme.index(README_RESULTS_START)
    end = readme.index(README_RESULTS_END, start) + len(README_RESULTS_END)
    rendered = readme[:start] + render_readme_headline(rows) + readme[end:]
    if rendered.count(README_B1_START) != 1 or rendered.count(README_B1_END) != 1:
        raise AggregateImportError("README generated-B1 markers are missing or duplicated")
    b1_start = rendered.index(README_B1_START)
    b1_end = rendered.index(README_B1_END, b1_start) + len(README_B1_END)
    return rendered[:b1_start] + render_readme_b1(rows) + rendered[b1_end:]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AggregateImportError("cannot load the public site builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_site_matches(site_dir: Path, bundle: AggregateBundle) -> None:
    parser = _PayloadParser()
    parser.feed((site_dir / "index.html").read_text(encoding="utf-8"))
    if not parser.parts:
        raise AggregateImportError("rebuilt site has no aggregate JSON payload")
    payload = json.loads("".join(parser.parts))
    expected = {
        "results": list(bundle.results),
        "tables": list(bundle.tables),
        "statistics": list(bundle.statistics),
    }
    for key, rows in expected.items():
        if payload.get(key) != rows:
            raise AggregateImportError(f"rebuilt site differs from {key} aggregate rows")
    manifest = json.loads((site_dir / "build_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_sha256") != dict(bundle.file_sha256):
        raise AggregateImportError("rebuilt site source hashes differ from the aggregate bundle")


def _stage_outputs(bundle: AggregateBundle, repo_root: Path) -> dict[str, bytes]:
    """Build every aggregate-dependent output in an isolated workspace."""
    required = (
        repo_root / "README.md",
        repo_root / "site" / "build.py",
        repo_root / "site" / "assets",
        repo_root / "examples" / "synthetic",
    )
    if not all(path.exists() for path in required):
        raise AggregateImportError("repository lacks the public site or synthetic fixtures")
    with tempfile.TemporaryDirectory(prefix="segbench-aggregate-update-") as temp:
        workspace = Path(temp)
        results_dir = workspace / "results"
        results_dir.mkdir()
        for name, data in bundle.raw_files.items():
            (results_dir / name).write_bytes(data)
        shutil.copytree(repo_root / "site", workspace / "site")
        shutil.copytree(repo_root / "examples" / "synthetic", workspace / "examples" / "synthetic")
        (workspace / "segbench").mkdir()
        shutil.copy2(ROOT / "segbench" / "__init__.py", workspace / "segbench" / "__init__.py")
        shutil.copy2(
            ROOT / "segbench" / "synthetic_demo.py",
            workspace / "segbench" / "synthetic_demo.py",
        )
        (workspace / "scripts").mkdir()
        shutil.copy2(
            ROOT / "scripts" / "build_synthetic_site_demo.py",
            workspace / "scripts" / "build_synthetic_site_demo.py",
        )
        docs_dir = workspace / "docs"
        docs_dir.mkdir()
        results_markdown = render_results_markdown(bundle).encode("utf-8")
        (docs_dir / "RESULTS.md").write_bytes(results_markdown)
        readme = render_readme(
            (repo_root / "README.md").read_text(encoding="utf-8"), bundle.results
        ).encode("utf-8")

        builder = _load_module("segbench_staged_site_builder", workspace / "site" / "build.py")
        with contextlib.redirect_stdout(io.StringIO()):
            builder.main()
        _assert_site_matches(workspace / "site", bundle)
        outputs = {
            **{f"results/{name}": data for name, data in bundle.raw_files.items()},
            "docs/RESULTS.md": results_markdown,
            "README.md": readme,
            "site/index.html": (workspace / "site" / "index.html").read_bytes(),
            "site/build_manifest.json": (workspace / "site" / "build_manifest.json").read_bytes(),
            "site/assets/results/public_results.csv": (
                workspace / "site" / "assets" / "results" / "public_results.csv"
            ).read_bytes(),
        }
    return outputs


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or (path.exists() and path.is_symlink()):
        raise AggregateImportError(f"refusing symlink output: {path.name}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _apply_outputs(outputs: Mapping[str, bytes], repo_root: Path) -> None:
    allowed = {
        *(f"results/{name}" for name in EXPECTED_FILES),
        "docs/RESULTS.md",
        "README.md",
        "site/index.html",
        "site/build_manifest.json",
        "site/assets/results/public_results.csv",
    }
    if set(outputs) != allowed:
        raise AggregateImportError("internal public-output universe changed")
    previous: dict[str, bytes | None] = {}
    try:
        for relative, data in outputs.items():
            target = repo_root / relative
            if repo_root.is_symlink() or target.is_symlink():
                raise AggregateImportError(f"refusing symlink output: {relative}")
            current = repo_root
            for part in Path(relative).parts[:-1]:
                current = current / part
                if current.exists() and current.is_symlink():
                    raise AggregateImportError(
                        f"refusing output beneath a symlink: {relative}"
                    )
            previous[relative] = target.read_bytes() if target.is_file() else None
            _atomic_write(target, data)
    except Exception:
        for relative, data in previous.items():
            target = repo_root / relative
            if data is None:
                if target.exists() and target.is_file() and not target.is_symlink():
                    target.unlink()
            else:
                _atomic_write(target, data)
        raise


def run_update(
    bundle_dir: Path, repo_root: Path = ROOT, *, dry_run: bool
) -> dict[str, object]:
    """Validate, rebuild, number-check, and optionally install one bundle."""
    bundle = validate_bundle(bundle_dir)
    outputs = _stage_outputs(bundle, repo_root)
    if not dry_run:
        _apply_outputs(outputs, repo_root)
    return {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "PASS",
        "mode": "dry-run" if dry_run else "apply",
        "mutations_applied": not dry_run,
        "importer_sha256": _sha256(Path(__file__).read_bytes()),
        "bundle": {
            "schema_version": 1,
            "kind": BUNDLE_KIND,
            "manifest_sha256": bundle.manifest_sha256,
            "files": dict(bundle.file_sha256),
        },
        "aggregate_rows": {
            "results": len(bundle.results),
            "statistics": len(bundle.statistics),
            "tables": len(bundle.tables),
        },
        "output_sha256": {name: _sha256(data) for name, data in sorted(outputs.items())},
        "checks": [
            "exact manifest-bound CSV inventory",
            "aggregate-only schemas with no per-image columns",
            "no biometric media, trained artifacts, private paths, hosts, IPs, or cloud-drive strings",
            "README marker table rendered from aggregate rows",
            "documentation rendered from aggregate rows",
            "site payload and source hashes match aggregate rows byte-for-byte",
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_update(args.bundle, dry_run=args.dry_run)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(args.report, encoded)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
