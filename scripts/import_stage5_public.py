#!/usr/bin/env python3
"""Convert an authenticated canonical Stage-5 public package into this release.

The independently supplied receipt digest is the trust root.  The converter
verifies the complete ``public/`` inventory against that receipt, validates the
canonical aggregate schemas, deterministically derives the compact website
CSVs, stages every generated output, and only then performs a rollback-safe
apply.  It never reads private images, masks, per-image rows, or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segbench.public_figures import (  # noqa: E402
    FIGURE_KINDS,
    MIME_BY_SUFFIX,
    SOURCE_KIND,
    build_manifest,
)


RECEIPT_KIND = SOURCE_KIND
IMPORT_KIND = "stage5_public_release_import_receipt_v1"
REPORT_KIND = "stage5_public_release_import_report_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_TEXT_BYTES = 10 * 1024 * 1024
CORE_FILES = {
    "tables.csv", "statistics.csv", "statistics.json", "team_crosscheck.json",
    "cross_domain_deltas.csv", "table_metadata.json", "dataset_summary.json",
    "reproducibility.json", "release_universe.json",
}
CANONICAL_TABLE_COLUMNS = (
    "table", "view", "group_by", "group_value", "method", "display_name",
    "year", "family", "training_data", "comment", "status", "reason",
    "n_seeds", "n_images", "n_subjects", "n_missing", "params_m",
    "params_m_seed_std", "ms_per_img", "ms_per_img_seed_std", "device",
    "subset", "split", "ground_truth", "seed_ids", "accuracy_mean",
    "accuracy_seed_std", "error_rate_mean", "error_rate_seed_std", "dice_mean",
    "dice_seed_std", "jaccard_mean", "jaccard_seed_std", "miou_mean",
    "miou_seed_std", "mdr_mean", "mdr_seed_std", "fdr_mean", "fdr_seed_std",
    "mc_mean", "mc_seed_std", "hr_mean", "hr_seed_std", "par_mean",
    "par_seed_std", "ohf_mean", "ohf_seed_std", "accuracy_ci_low",
    "accuracy_ci_high", "error_rate_ci_low", "error_rate_ci_high", "dice_ci_low",
    "dice_ci_high", "jaccard_ci_low", "jaccard_ci_high", "miou_ci_low",
    "miou_ci_high", "mdr_ci_low", "mdr_ci_high", "fdr_ci_low", "fdr_ci_high",
    "mc_ci_low", "mc_ci_high", "hr_ci_low", "hr_ci_high", "par_ci_low",
    "par_ci_high", "ohf_ci_low", "ohf_ci_high",
)
CANONICAL_STAT_COLUMNS = (
    "table", "metric", "comparison", "reference_method", "method", "estimate",
    "ci_lower", "ci_upper", "p_value", "p_holm", "significant", "n_images",
    "n_subjects", "status", "reason", "receipt_status",
)
TABLE_ORDER = ("A", "A_value", "A_prime", "B1", "D")
TABLE_STEMS = {
    "A": "table_a", "A_value": "table_a_value", "A_prime": "table_a_prime",
    "B1": "table_b1", "D": "table_d",
}
EXPECTED_VIEWS = {
    "A": {"overall"},
    "A_value": {"examiner_value"},
    "A_prime": {"overall"},
    "B1": {"overall", "group", "device", "capture"},
    "D": {"latent_to_exemplar"},
}
DATASET1_METHODS = frozenset({
    "gmfs_default", "gmfs_tuned", "sufs", "sufs_ft", "fingernet", "svm",
    "rf", "welm", "unet", "unetpp", "drunet", "ra_runet", "lenet",
    "alexnet", "cnn_ae", "segformer_sub", "all_background", "all_foreground",
})
EXAMINER_GROUPS = frozenset({"VALUE", "LIMITED", "NOVALUE"})
HULL_METHODS = DATASET1_METHODS | {"ra_runet_aligned", "hull_as_prediction"}
CROSS_DOMAIN_METHODS = frozenset({
    "gmfs_tuned", "sufs_ft", "svm", "rf", "welm", "unet", "unetpp",
    "drunet", "ra_runet", "lenet", "alexnet", "cnn_ae", "segformer_sub",
})
METHOD_MAP = {"sufs": "sufs_zero", "ra_runet": "r_unet", "segformer_sub": "segformer_b2"}
CONFIG_METHOD_MAP = {
    "ra_runet": "ra_runet", "ra_runet_aligned": "ra_runet",
    "segformer_sub": "segformer_sub", "sufs_zero": "sufs"
}
CONFIG_CANONICAL_METHOD = {
    "sufs_zero": "sufs", "ra_runet_aligned": "ra_runet_aligned",
}
THRESHOLD_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
UNSEEDED_THRESHOLD_METHODS = frozenset({"sufs", "fingernet"})
EXPECTED_CONFIG_NAMES = frozenset({
    "alexnet.yaml", "cnn_ae.yaml", "drunet.yaml", "fingernet.yaml",
    "gmfs.yaml", "lenet.yaml", "ra_runet.yaml", "ra_runet_aligned.yaml",
    "rf.yaml", "segformer_sub.yaml", "sufs_ft.yaml", "sufs_zero.yaml",
    "svm.yaml", "unet.yaml", "unetpp.yaml", "welm.yaml",
})
IMPORT_RECEIPT_FIELDS = {
    "schema_version", "kind", "status", "source_receipt",
    "source_public_inventory_sha256", "converter_sha256",
    "converter_dependencies", "converter_dependencies_sha256",
    "contains_per_image_rows", "contains_biometric_media", "release_metadata",
    "outputs", "outputs_sha256",
}


class Stage5ImportError(ValueError):
    """The canonical package or proposed public mutation is invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def _converter_dependencies(root: Path = ROOT) -> dict[str, str]:
    fixed = {
        "README.md",
        "scripts/import_stage5_public.py", "scripts/update_from_aggregate.py",
        "scripts/build_synthetic_site_demo.py", "segbench/public_figures.py",
        "segbench/synthetic_demo.py", "site/build.py",
    }
    paths = {root / relative for relative in fixed}
    for base in (root / "examples/synthetic", root / "site/assets"):
        for path in base.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and not path.is_symlink() and relative != "site/assets/results/public_results.csv":
                paths.add(path)
    missing = [path for path in paths if not path.is_file() or path.is_symlink()]
    if missing:
        raise Stage5ImportError("converter dependency inventory is incomplete")
    dependencies: dict[str, str] = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if relative == "README.md":
            text = _decode_text(data, "README publication template")
            start, end = UPDATE.README_RESULTS_START, UPDATE.README_RESULTS_END
            if text.count(start) != 1 or text.count(end) != 1:
                raise Stage5ImportError("README publication template markers changed")
            left = text.index(start) + len(start)
            right = text.index(end, left)
            data = (text[:left] + "\n<generated-results/>\n" + text[right:]).encode()
        dependencies[relative] = _sha256(data)
    return dependencies


def _load_update_module() -> ModuleType:
    path = ROOT / "scripts" / "update_from_aggregate.py"
    spec = importlib.util.spec_from_file_location("segbench_public_aggregate_update", path)
    if spec is None or spec.loader is None:
        raise Stage5ImportError("cannot load aggregate release builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


UPDATE = _load_update_module()


def _regular_bytes(path: Path, label: str, *, limit: int = MAX_TEXT_BYTES) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        raise Stage5ImportError("symlink-safe reads are unavailable")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise Stage5ImportError(f"cannot open {label} as a regular file") from exc
    try:
        meta = os.fstat(descriptor)
        if not stat.S_ISREG(meta.st_mode) or meta.st_size > limit:
            raise Stage5ImportError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > limit:
            raise Stage5ImportError(f"{label} exceeds the size limit")
        observed = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(observed.st_mode) or (observed.st_dev, observed.st_ino) != (meta.st_dev, meta.st_ino):
            raise Stage5ImportError(f"{label} changed while being read")
        return data
    finally:
        os.close(descriptor)


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise Stage5ImportError(f"unsafe receipt path: {value!r}")
    return path


def _decode_text(data: bytes, label: str) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Stage5ImportError(f"{label} is not UTF-8") from exc
    if text.startswith("\ufeff"):
        raise Stage5ImportError(f"{label} has a byte-order mark")
    return text


def _scan_text(text: str, label: str) -> None:
    for line_number, line in enumerate(text.splitlines(), 1):
        reason = UPDATE._forbidden_text_reason(line)
        if reason:
            raise Stage5ImportError(f"{label}:{line_number} contains {reason}")


def _scan_document(text: str, label: str) -> None:
    """Scan a whole aggregate document without treating a row as one cell."""
    for line_number, line in enumerate(text.splitlines(), 1):
        reason = UPDATE._forbidden_text_reason(line)
        if reason in {"oversized aggregate cell", "multiline aggregate cell"}:
            reason = None
        if reason:
            raise Stage5ImportError(f"{label}:{line_number} contains {reason}")
        if re.search(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{81,}(?![A-Za-z0-9+/=_-])", line):
            raise Stage5ImportError(f"{label}:{line_number} contains a long encoded token")


def _has_opaque_external_identifier(value: str) -> bool:
    for token in re.findall(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{25,80}(?![A-Za-z0-9_-])", value):
        if (
            any(character.islower() for character in token)
            and any(character.isupper() for character in token)
            and any(character.isdigit() for character in token)
            and sum(character in "_-" for character in token) <= 2
        ):
            return True
    return False


def _csv_rows(data: bytes, name: str, columns: Sequence[str]) -> list[dict[str, str]]:
    text = _decode_text(data, name)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != tuple(columns):
        raise Stage5ImportError(f"{name} canonical columns changed")
    rows: list[dict[str, str]] = []
    for line, raw in enumerate(reader, 2):
        if None in raw or any(value is None for value in raw.values()):
            raise Stage5ImportError(f"{name}:{line} has a malformed field count")
        row = {key: str(value) for key, value in raw.items()}
        for key, value in row.items():
            if value != value.strip():
                raise Stage5ImportError(f"{name}:{line}:{key} has edge whitespace")
            _scan_text(value, f"{name}:{line}:{key}")
        rows.append(row)
    if not rows:
        raise Stage5ImportError(f"{name} is empty")
    return rows


def _number(value: str, label: str, minimum: float, maximum: float | None = None) -> float | None:
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise Stage5ImportError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise Stage5ImportError(f"{label} is outside its allowed range")
    return result


def _validate_canonical_tables(rows: list[dict[str, str]]) -> None:
    identities: set[tuple[str, str, str, str]] = set()
    for line, row in enumerate(rows, 2):
        prefix = f"tables.csv:{line}"
        if row["table"] not in TABLE_ORDER or row["status"] not in {"ok", "unavailable"}:
            raise Stage5ImportError(f"{prefix} has an unsupported table/status")
        if row["view"] not in EXPECTED_VIEWS[row["table"]]:
            raise Stage5ImportError(f"{prefix} has an unsupported table/view pairing")
        if not row["view"] or not row["group_value"] or not row["method"] or not row["display_name"]:
            raise Stage5ImportError(f"{prefix} has an empty identity")
        identity = (row["table"], row["view"], row["group_value"], row["method"])
        if identity in identities:
            raise Stage5ImportError(f"{prefix} duplicates {identity}")
        identities.add(identity)
        for field in ("n_seeds", "n_images", "n_subjects", "n_missing"):
            if row[field] and (not row[field].isdigit() or int(row[field]) < 0):
                raise Stage5ImportError(f"{prefix}:{field} is invalid")
        seed_ids = [item.strip() for item in row["seed_ids"].split(",") if item.strip()]
        if (
            not row["n_seeds"]
            or int(row["n_seeds"]) != len(seed_ids)
            or len(seed_ids) != len(set(seed_ids))
        ):
            raise Stage5ImportError(f"{prefix}:seed_ids differs from n_seeds")
        if row["n_images"] and row["n_missing"] and int(row["n_missing"]) > int(row["n_images"]):
            raise Stage5ImportError(f"{prefix}:n_missing exceeds n_images")
        for metric in ("accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr", "mc", "hr", "ohf"):
            for suffix in ("mean", "seed_std", "ci_low", "ci_high"):
                _number(row[f"{metric}_{suffix}"], f"{prefix}:{metric}_{suffix}", 0.0, 1.0)
            low, high, mean = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"], row[f"{metric}_mean"]
            if (low == "") != (high == ""):
                raise Stage5ImportError(f"{prefix}:{metric} has only one confidence bound")
            if low and float(low) > float(high):
                raise Stage5ImportError(f"{prefix}:{metric} confidence interval is inconsistent")
        for suffix in ("mean", "seed_std", "ci_low", "ci_high"):
            _number(row[f"par_{suffix}"], f"{prefix}:par_{suffix}", 0.0)
        par_low, par_high, par_mean = row["par_ci_low"], row["par_ci_high"], row["par_mean"]
        if (par_low == "") != (par_high == ""):
            raise Stage5ImportError(f"{prefix}:par has only one confidence bound")
        if par_low and float(par_low) > float(par_high):
            raise Stage5ImportError(f"{prefix}:par confidence interval is inconsistent")
        for field in ("params_m", "params_m_seed_std", "ms_per_img", "ms_per_img_seed_std"):
            _number(row[field], f"{prefix}:{field}", 0.0)
        if row["device"] not in {"", "CPU", "GPU"}:
            raise Stage5ImportError(f"{prefix}:device is invalid")
        metric_family = (
            ("mc", "hr", "par", "ohf") if row["table"] in {"B1", "D"}
            else ("accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr")
        )
        if row["status"] == "ok" and any(row[f"{metric}_mean"] == "" for metric in metric_family):
            raise Stage5ImportError(f"{prefix} complete row lacks a metric")
        if row["status"] == "unavailable" and not row["reason"]:
            raise Stage5ImportError(f"{prefix} unavailable row lacks a reason")
        if row["status"] == "unavailable" and any(
            row[f"{metric}_{suffix}"]
            for metric in metric_family
            for suffix in ("mean", "seed_std", "ci_low", "ci_high")
        ):
            raise Stage5ImportError(f"{prefix} unavailable row contains a metric")
    by_table = {
        table: {row["view"] for row in rows if row["table"] == table}
        for table in TABLE_ORDER
    }
    if set(by_table) != set(TABLE_ORDER) or any(
        by_table[table] != EXPECTED_VIEWS[table] for table in TABLE_ORDER
    ):
        raise Stage5ImportError("canonical Stage-5 table/view universe is incomplete")

    def require_method_universe(
        table: str, view: str, group: str, expected: frozenset[str],
    ) -> None:
        observed = {
            row["method"] for row in rows
            if row["table"] == table and row["view"] == view
            and row["group_value"] == group
        }
        if observed != expected:
            raise Stage5ImportError(
                f"canonical method universe is incomplete for {table}/{view}/{group}; "
                f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
            )

    require_method_universe("A", "overall", "overall", DATASET1_METHODS)
    require_method_universe("A_prime", "overall", "overall", DATASET1_METHODS)
    observed_examiner_groups = {
        row["group_value"] for row in rows
        if row["table"] == "A_value" and row["view"] == "examiner_value"
    }
    if observed_examiner_groups != EXAMINER_GROUPS:
        raise Stage5ImportError(
            "canonical examiner-value group universe is incomplete; "
            f"missing={sorted(EXAMINER_GROUPS - observed_examiner_groups)}, "
            f"unexpected={sorted(observed_examiner_groups - EXAMINER_GROUPS)}"
        )
    for group in sorted(EXAMINER_GROUPS):
        require_method_universe("A_value", "examiner_value", group, DATASET1_METHODS)
    for view in sorted(EXPECTED_VIEWS["B1"]):
        groups = {
            row["group_value"] for row in rows
            if row["table"] == "B1" and row["view"] == view
        }
        if not groups:
            raise Stage5ImportError(f"canonical B1/{view} group universe is empty")
        for group in sorted(groups):
            require_method_universe("B1", view, group, HULL_METHODS)
    require_method_universe(
        "D", "latent_to_exemplar", "overall", CROSS_DOMAIN_METHODS,
    )


def _validate_release_universe(data: bytes, rows: list[dict[str, str]]) -> None:
    """Require a receipt-bound declaration equal to the canonical row projection."""
    try:
        payload = json.loads(_decode_text(data, "release_universe.json"))
    except json.JSONDecodeError as exc:
        raise Stage5ImportError("release universe is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "kind", "rows"}
        or payload.get("schema_version") != 1
        or payload.get("kind") != "stage5_public_release_universe_v1"
        or not isinstance(payload.get("rows"), list)
    ):
        raise Stage5ImportError("release universe schema changed")
    declared: dict[tuple[str, str, str], frozenset[str]] = {}
    declared_order: list[tuple[str, str, str]] = []
    for index, raw in enumerate(payload["rows"]):
        if not isinstance(raw, dict) or set(raw) != {
            "table", "view", "group_value", "methods"
        }:
            raise Stage5ImportError(f"release universe row {index} schema changed")
        key = (raw["table"], raw["view"], raw["group_value"])
        methods = raw["methods"]
        if (
            not all(isinstance(value, str) and value for value in key)
            or not isinstance(methods, list)
            or not methods
            or any(not isinstance(method, str) or not method for method in methods)
            or methods != sorted(set(methods))
            or key in declared
        ):
            raise Stage5ImportError(f"release universe row {index} is not canonical")
        declared[key] = frozenset(methods)
        declared_order.append(key)
    if declared_order != sorted(declared_order):
        raise Stage5ImportError("release universe rows are not canonically sorted")
    observed: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        key = (row["table"], row["view"], row["group_value"])
        observed.setdefault(key, set()).add(row["method"])
    normalized = {key: frozenset(methods) for key, methods in observed.items()}
    if declared != normalized:
        raise Stage5ImportError("receipt-bound release universe differs from tables.csv")


def _normalize_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return format(value, ".15g")
    text = str(value)
    if text.casefold() in {"true", "false"}:
        return text.casefold()
    try:
        number = float(text)
    except ValueError:
        return text
    return format(number, ".15g") if math.isfinite(number) else ""


def _validate_statistics_json(csv_rows: list[dict[str, str]], data: bytes) -> None:
    try:
        payload = json.loads(_decode_text(data, "statistics.json"))
    except json.JSONDecodeError as exc:
        raise Stage5ImportError("statistics.json is invalid") from exc
    if not isinstance(payload, list) or len(payload) != len(csv_rows):
        raise Stage5ImportError("statistics JSON/CSV row counts differ")
    for index, (csv_row, json_row) in enumerate(zip(csv_rows, payload)):
        if not isinstance(json_row, dict) or set(json_row) != set(CANONICAL_STAT_COLUMNS):
            raise Stage5ImportError(f"statistics.json row {index} schema changed")
        for column in CANONICAL_STAT_COLUMNS:
            if _normalize_scalar(csv_row[column]) != _normalize_scalar(json_row[column]):
                raise Stage5ImportError(f"statistics JSON/CSV differ at row {index}:{column}")


def _validate_statistics(rows: list[dict[str, str]]) -> None:
    allowed = {
        "ok", "unavailable", "UNAVAILABLE", "N/A", "complete", "COMPLETE",
        "descriptive_only",
    }
    identities: set[tuple[str, str, str, str]] = set()
    for line, row in enumerate(rows, 2):
        prefix = f"statistics.csv:{line}"
        if row["table"] not in TABLE_ORDER or row["status"] not in allowed:
            raise Stage5ImportError(f"{prefix} has an unsupported table/status")
        if not row["metric"] or not row["comparison"]:
            raise Stage5ImportError(f"{prefix} has an empty identity")
        identity = (row["table"], row["metric"], row["comparison"], row["method"])
        if identity in identities:
            raise Stage5ImportError(f"{prefix} duplicates a statistical row")
        identities.add(identity)
        for field in ("estimate", "ci_lower", "ci_upper"):
            _number(row[field], f"{prefix}:{field}", -1.0, 1.0)
        for field in ("p_value", "p_holm"):
            _number(row[field], f"{prefix}:{field}", 0.0, 1.0)
        for field in ("n_images", "n_subjects"):
            if row[field] and (not row[field].isdigit() or int(row[field]) < 0):
                raise Stage5ImportError(f"{prefix}:{field} is invalid")
        if _normalize_scalar(row["significant"]) not in {"", "true", "false"}:
            raise Stage5ImportError(f"{prefix}:significant is invalid")
        low, high, estimate = row["ci_lower"], row["ci_upper"], row["estimate"]
        if (low == "") != (high == ""):
            raise Stage5ImportError(f"{prefix} has only one confidence bound")
        if low and float(low) > float(high):
            raise Stage5ImportError(f"{prefix} confidence interval is inconsistent")
        unavailable = row["status"] in {"unavailable", "UNAVAILABLE", "N/A"}
        if unavailable and not row["reason"]:
            raise Stage5ImportError(f"{prefix} unavailable row lacks a reason")
        if unavailable and any(row[field] for field in ("estimate", "ci_lower", "ci_upper", "p_value", "p_holm")):
            raise Stage5ImportError(f"{prefix} unavailable row contains statistics")


def _reconcile_unavailable_point_estimates(
    rows: list[dict[str, str]], tables: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Normalize one authenticated producer edge without inventing evidence.

    The canonical producer can serialize an unavailable point-estimate row with
    a blank status plus placeholder [0, 1] bounds.  A matching unavailable
    table row is authoritative.  Only that exact, evidence-free case is
    normalized; every other blank status remains a hard schema error.
    """
    output: list[dict[str, str]] = []
    reconciled = 0
    for row in rows:
        item = dict(row)
        if not item["status"]:
            matches = [
                candidate for candidate in tables
                if candidate["table"] == item["table"]
                and candidate["method"] == item["method"]
                and candidate["status"] == "unavailable"
                and candidate["group_value"] == "overall"
            ]
            if (
                item["comparison"] != "point estimate"
                or item["estimate"]
                or len(matches) != 1
                or not matches[0]["reason"]
            ):
                raise Stage5ImportError("blank canonical statistic status is not reconcilable")
            item["status"] = "UNAVAILABLE"
            item["reason"] = matches[0]["reason"]
            for field in ("ci_lower", "ci_upper", "p_value", "p_holm", "significant"):
                item[field] = ""
            reconciled += 1
        output.append(item)
    return output, reconciled


def _validate_metadata(data: bytes, tables: list[dict[str, str]]) -> list[dict[str, str]]:
    try:
        payload = json.loads(_decode_text(data, "table_metadata.json"))
    except json.JSONDecodeError as exc:
        raise Stage5ImportError("table metadata is invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"tables"} or not isinstance(payload["tables"], list):
        raise Stage5ImportError("table metadata schema changed")
    records: list[dict[str, str]] = []
    keys: set[tuple[str, str]] = set()
    for index, raw in enumerate(payload["tables"]):
        if not isinstance(raw, dict) or set(raw) != {"table", "view", "title", "caption"}:
            raise Stage5ImportError(f"table metadata row {index} schema changed")
        row = {key: str(value) for key, value in raw.items()}
        if row["table"] not in TABLE_ORDER or not row["view"] or not row["title"] or not row["caption"]:
            raise Stage5ImportError(f"table metadata row {index} is incomplete")
        _scan_text(json.dumps(row, sort_keys=True), f"table metadata row {index}")
        key = (row["table"], row["view"])
        if key in keys:
            raise Stage5ImportError(f"table metadata row {index} is duplicated")
        keys.add(key)
        records.append(row)
    expected = {(row["table"], row["view"]) for row in tables}
    if keys != expected:
        raise Stage5ImportError("table metadata views differ from tables.csv")
    return records


def _load_receipt(package: Path, expected_digest: str) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    if HEX64.fullmatch(expected_digest) is None:
        raise Stage5ImportError("expected receipt SHA-256 is invalid")
    if package.is_symlink() or not package.is_dir():
        raise Stage5ImportError("Stage-5 package is not a regular directory")
    receipt_bytes = _regular_bytes(package / "package_receipt.json", "Stage-5 package receipt")
    if _sha256(receipt_bytes) != expected_digest:
        raise Stage5ImportError("Stage-5 receipt differs from the independently pinned SHA-256")
    try:
        receipt = json.loads(_decode_text(receipt_bytes, "Stage-5 package receipt"))
    except json.JSONDecodeError as exc:
        raise Stage5ImportError("Stage-5 receipt is invalid JSON") from exc
    if not isinstance(receipt, dict):
        raise Stage5ImportError("Stage-5 receipt root is not an object")
    outputs = receipt.get("outputs")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != RECEIPT_KIND
        or receipt.get("status") != "PASS"
        or receipt.get("contains_per_image_rows") is not False
        or receipt.get("contains_subject_or_image_ids") is not False
        or receipt.get("public_scan", {}).get("status") != "PASS"
        or not isinstance(outputs, dict)
        or receipt.get("outputs_sha256") != _canonical_sha256(outputs)
    ):
        raise Stage5ImportError("Stage-5 receipt did not pass the public aggregate contract")
    public_expected: dict[str, str] = {}
    for raw, digest in outputs.items():
        relative = _safe_relative(str(raw))
        if HEX64.fullmatch(str(digest)) is None:
            raise Stage5ImportError("Stage-5 receipt contains an invalid artifact digest")
        if relative.parts[:1] == ("public",):
            public_expected[Path(*relative.parts[1:]).as_posix()] = str(digest)
    public = package / "public"
    if public.is_symlink() or not public.is_dir():
        raise Stage5ImportError("Stage-5 public directory is absent or a symlink")
    observed_paths: list[Path] = []
    for entry in sorted(public.rglob("*")):
        if entry.is_symlink():
            raise Stage5ImportError("Stage-5 public tree contains a symlink")
        if entry.is_file():
            observed_paths.append(entry)
    observed = {path.relative_to(public).as_posix() for path in observed_paths}
    if observed != set(public_expected):
        raise Stage5ImportError("Stage-5 public file universe differs from its receipt")
    files: dict[str, bytes] = {}
    for path in observed_paths:
        relative = path.relative_to(public).as_posix()
        data = _regular_bytes(path, f"Stage-5 public artifact {relative}")
        if _sha256(data) != public_expected[relative]:
            raise Stage5ImportError(f"Stage-5 public artifact bytes changed: {relative}")
        files[relative] = data
    return receipt, receipt_bytes, files


def _validate_inventory(files: Mapping[str, bytes], table_keys: set[str]) -> dict[str, list[str]]:
    paths = set(files)
    if not CORE_FILES.issubset(paths):
        raise Stage5ImportError(f"canonical public package lacks {sorted(CORE_FILES - paths)}")
    expected_tables = {
        f"tables/{TABLE_STEMS[table]}.{suffix}"
        for table in table_keys for suffix in ("csv", "md", "tex")
    }
    expected_figures = {
        f"figures/{stem}{suffix}"
        for stem in FIGURE_KINDS for suffix in MIME_BY_SUFFIX
    }
    config_paths = sorted(path for path in paths if path.startswith("configs/"))
    split_paths = sorted(path for path in paths if path.startswith("splits/"))
    if split_paths not in ([], ["splits/sd302_subject_split.json"]):
        raise Stage5ImportError("canonical split artifact universe changed")
    allowed = CORE_FILES | expected_tables | expected_figures | set(config_paths) | set(split_paths)
    if paths != allowed:
        raise Stage5ImportError(
            f"canonical public inventory changed; missing={sorted(allowed - paths)}, "
            f"unexpected={sorted(paths - allowed)}"
        )
    if any(Path(path).parent != Path("configs") or Path(path).suffix != ".yaml" for path in config_paths):
        raise Stage5ImportError("canonical config path is unsafe")
    return {"configs": config_paths, "splits": split_paths, "tables": sorted(expected_tables), "figures": sorted(expected_figures)}


def _format_year(value: str) -> str:
    if value == "":
        return ""
    number = float(value)
    if not number.is_integer() or not 1900 <= number <= 2100:
        raise Stage5ImportError(f"invalid canonical method year: {value}")
    return str(int(number))


def _release_result_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        method = METHOD_MAP.get(row["method"], row["method"])
        status = "complete" if row["status"] == "ok" else "n/a"
        primary = "hr" if row["table"] in {"B1", "D"} else "dice"
        comment = row["comment"]
        if status == "n/a" and row["reason"] and row["reason"] not in comment:
            comment = f"{comment}; N/A: {row['reason']}" if comment else f"N/A: {row['reason']}"
        converted = {column: "" for column in UPDATE.RESULT_COLUMNS}
        converted.update({
            "table": row["table"], "view": row["view"], "group": row["group_value"],
            "year": _format_year(row["year"]), "method": method,
            "approach": row["display_name"], "family": row["family"],
            "training_data": row["training_data"], "comment": comment,
            "status": status, "seeds_completed": str(int(row["n_seeds"] or "0")),
            "n_images": "" if not row["n_images"] or int(row["n_images"]) == 0 else str(int(row["n_images"])),
            "n_subjects": "" if not row["n_subjects"] or int(row["n_subjects"]) == 0 else str(int(row["n_subjects"])),
            "accuracy": row["accuracy_mean"],
            "accuracy_seed_std": row["accuracy_seed_std"],
            "error_rate": row["error_rate_mean"],
            "error_rate_seed_std": row["error_rate_seed_std"],
            "dice": row["dice_mean"], "dice_seed_std": row["dice_seed_std"],
            "jaccard": row["jaccard_mean"],
            "jaccard_seed_std": row["jaccard_seed_std"],
            "miou": row["miou_mean"], "miou_seed_std": row["miou_seed_std"],
            "mdr": row["mdr_mean"], "mdr_seed_std": row["mdr_seed_std"],
            "fdr": row["fdr_mean"], "fdr_seed_std": row["fdr_seed_std"],
            "mc": row["mc_mean"], "mc_seed_std": row["mc_seed_std"],
            "hr": row["hr_mean"], "hr_seed_std": row["hr_seed_std"],
            "par": row["par_mean"], "par_seed_std": row["par_seed_std"],
            "ohf": row["ohf_mean"], "ohf_seed_std": row["ohf_seed_std"],
            "params_m": row["params_m"],
            "params_m_seed_std": row["params_m_seed_std"],
            "ms_per_img": row["ms_per_img"],
            "ms_per_img_seed_std": row["ms_per_img_seed_std"],
            "device": row["device"],
            "ci_lower": row[f"{primary}_ci_low"], "ci_upper": row[f"{primary}_ci_high"],
        })
        if status == "n/a":
            for field in (
                "accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr",
                "mc", "hr", "par", "ohf", "ci_lower", "ci_upper",
            ):
                converted[field] = ""
            for field in (
                "accuracy_seed_std", "error_rate_seed_std", "dice_seed_std",
                "jaccard_seed_std", "miou_seed_std", "mdr_seed_std",
                "fdr_seed_std", "mc_seed_std", "hr_seed_std", "par_seed_std",
                "ohf_seed_std",
            ):
                converted[field] = ""
        output.append(converted)
    return output


def _release_stat_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        comparison = row["comparison"]
        if row["method"] and (comparison in {"point estimate", "unavailable"} or (row["table"], row["metric"], comparison) in identities):
            comparison = f"{comparison}: {METHOD_MAP.get(row['method'], row['method'])}"
        if row["reason"] and str(row["status"]).upper() in {"N/A", "UNAVAILABLE"}:
            comparison = f"{comparison} — {row['reason']}"
        identity = (row["table"], row["metric"], comparison)
        if identity in identities:
            raise Stage5ImportError(f"canonical statistics collapse to duplicate {identity}")
        identities.add(identity)
        unavailable = str(row["status"]).upper() in {"N/A", "UNAVAILABLE"}
        output.append({
            "table": row["table"], "metric": row["metric"], "comparison": comparison,
            "estimate": row["estimate"], "ci_lower": row["ci_lower"], "ci_upper": row["ci_upper"],
            "p_holm": row["p_holm"],
            "significant": _normalize_scalar(row["significant"]) if row["significant"] else "",
            "status": "n/a" if unavailable else "complete",
        })
    return output


def _release_table_rows(metadata: list[dict[str, str]], result_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for table in TABLE_ORDER:
        rows = [row for row in result_rows if row["table"] == table]
        if not rows:
            continue
        candidates = [row for row in metadata if row["table"] == table]
        preferred = next((row for row in candidates if row["view"] in {"overall", "latent_to_exemplar", "examiner_value"}), candidates[0])
        output.append({"table": table, "title": preferred["title"], "status": "complete", "note": preferred["caption"]})
    return output


def _csv_bytes(columns: Sequence[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _validate_configs(
    files: Mapping[str, bytes], paths: list[str], table_rows: list[dict[str, str]],
) -> dict[str, bytes]:
    output: dict[str, bytes] = {}
    allowed_names = EXPECTED_CONFIG_NAMES

    def contains_pending(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().casefold() == "pending"
        if isinstance(value, dict):
            return any(contains_pending(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_pending(item) for item in value)
        return False

    for source in paths:
        name = Path(source).name
        if name not in allowed_names:
            raise Stage5ImportError(f"canonical package contains an unknown config: {name}")
        text = _decode_text(files[source], source)
        # A configuration may name schema fields such as ``mask_path``; that
        # is not a per-image record. Retain every other public-text guard.
        for line_number, line in enumerate(text.splitlines(), 1):
            reason = UPDATE._forbidden_text_reason(line)
            if reason == "absolute private path" and re.fullmatch(
                r"\s*(?:paper_repository|archival_repository):\s*"
                r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?\s*",
                line,
            ):
                reason = None
            if reason == "opaque cloud-drive-like identifier" and not _has_opaque_external_identifier(line):
                reason = None
            if reason == "per-image identifier field" and re.fullmatch(
                r"\s*(?:image|mask|subject)_column:\s*(?:image_path|mask_path|subject_id)\s*", line
            ):
                reason = None
            if reason == "weight or data-array path" and re.fullmatch(
                r"\s*weights_file:\s*[A-Za-z0-9._-]+\.(?:h5|pt|pth|joblib|npy)\s*", line,
                re.IGNORECASE,
            ):
                reason = None
            if reason == "host name" and re.fullmatch(
                r"\s*[A-Za-z0-9_-]+\.(?:json|bin|txt|yaml|yml):\s*", line,
                re.IGNORECASE,
            ):
                reason = None
            if reason:
                raise Stage5ImportError(f"{source}:{line_number} contains {reason}")
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise Stage5ImportError(f"canonical config is invalid YAML: {name}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("method"), str):
            raise Stage5ImportError(f"canonical config schema is invalid: {name}")
        method = str(payload["method"])
        expected_name = name.removesuffix(".yaml")
        if method != CONFIG_METHOD_MAP.get(expected_name, expected_name):
            raise Stage5ImportError(f"canonical config method/name mismatch: {name}")
        if contains_pending(payload):
            raise Stage5ImportError(f"canonical config still contains a pending choice: {name}")
        task = str(payload.get("task", ""))
        canonical_method = CONFIG_CANONICAL_METHOD.get(expected_name, expected_name)
        if task in {"pixel", "patch"} or method == "fingernet":
            selection = payload.get("selection")
            if not isinstance(selection, dict) or selection.get("status") not in {
                "selected", "unavailable"
            }:
                raise Stage5ImportError(f"canonical selector status is invalid: {name}")
            all_applicable = [
                row for row in table_rows
                if row["method"] == canonical_method
                and (
                    (canonical_method == "ra_runet_aligned" and row["table"] == "B1")
                    or (canonical_method != "ra_runet_aligned" and row["table"] == "A")
                )
            ]
            applicable = [
                row for row in all_applicable
                if row["view"] == "overall" and row["group_value"] == "overall"
            ]
            if len(applicable) != 1:
                raise Stage5ImportError(f"canonical selector has no unique result row: {name}")
            result = applicable[0]
            if selection["status"] == "unavailable":
                if (
                    result["status"] != "unavailable"
                    or not result["reason"]
                    or selection.get("reason") != result["reason"]
                    or any(
                        row["status"] != "unavailable"
                        or row["reason"] != result["reason"]
                        for row in all_applicable
                    )
                    or "chosen_threshold" in selection
                    or "threshold_by_seed" in selection
                ):
                    raise Stage5ImportError(
                        f"unavailable selector is not bound to its table reason: {name}"
                    )
            else:
                if (
                    result["status"] != "ok"
                    or any(row["status"] != "ok" for row in all_applicable)
                    or selection.get("allowed_grid") != list(THRESHOLD_GRID)
                ):
                    raise Stage5ImportError(f"selected threshold is not validation-grid bound: {name}")
                if canonical_method in UNSEEDED_THRESHOLD_METHODS:
                    chosen = selection.get("chosen_threshold")
                    if (
                        not isinstance(chosen, (int, float)) or isinstance(chosen, bool)
                        or float(chosen) not in THRESHOLD_GRID
                        or "threshold_by_seed" in selection
                    ):
                        raise Stage5ImportError(f"canonical unseeded threshold is invalid: {name}")
                else:
                    seed_ids = [item.strip() for item in result["seed_ids"].split(",") if item.strip()]
                    mapping = selection.get("threshold_by_seed")
                    if (
                        not result["n_seeds"].isdigit()
                        or int(result["n_seeds"]) != len(seed_ids)
                        or len(seed_ids) != len(set(seed_ids))
                        or not isinstance(mapping, dict)
                        or set(mapping) != set(seed_ids)
                        or "chosen_threshold" in selection
                        or any(
                            not re.fullmatch(r"\d+", str(seed))
                            or not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or float(value) not in THRESHOLD_GRID
                            for seed, value in mapping.items()
                        )
                    ):
                        raise Stage5ImportError(
                            f"canonical per-seed thresholds are incomplete or invalid: {name}"
                        )
        if task == "block":
            selection = payload.get("selection")
            applicable = [
                row for row in table_rows
                if row["method"] == canonical_method and row["table"] == "A"
                and row["view"] == "overall" and row["group_value"] == "overall"
            ]
            if (
                len(applicable) != 1
                or not isinstance(selection, dict)
                or selection.get("status") not in {"selected", "unavailable"}
            ):
                raise Stage5ImportError(f"canonical block selector is invalid: {name}")
            result = applicable[0]
            if selection["status"] == "unavailable":
                if (
                    result["status"] != "unavailable"
                    or not result["reason"]
                    or selection.get("reason") != result["reason"]
                    or "chosen" in selection
                    or "chosen_by_seed" in selection
                ):
                    raise Stage5ImportError(
                        f"unavailable block selector is not bound to its table reason: {name}"
                    )
            else:
                seed_ids = [item.strip() for item in result["seed_ids"].split(",") if item.strip()]
                mapping = selection.get("chosen_by_seed")
                chosen = selection.get("chosen")
                valid_mapping = (
                    isinstance(mapping, dict)
                    and set(mapping) == set(seed_ids)
                    and all(
                        re.fullmatch(r"\d+", str(seed)) is not None
                        and isinstance(parameters, dict) and bool(parameters)
                        for seed, parameters in mapping.items()
                    )
                )
                values = list(mapping.values()) if isinstance(mapping, dict) else []
                if (
                    result["status"] != "ok"
                    or not result["n_seeds"].isdigit()
                    or int(result["n_seeds"]) != len(seed_ids)
                    or len(seed_ids) != len(set(seed_ids))
                    or not valid_mapping
                    or (
                        chosen is not None
                        and (
                            not isinstance(chosen, dict)
                            or not values
                            or any(parameters != chosen for parameters in values)
                        )
                    )
                ):
                    raise Stage5ImportError(
                        f"canonical per-seed block parameters are incomplete or invalid: {name}"
                    )
        if method == "gmfs":
            tuning = payload.get("tuning")
            if not isinstance(tuning, dict) or not isinstance(tuning.get("chosen"), dict):
                raise Stage5ImportError("canonical GMFS config lacks final validation parameters")
        output[f"configs/{name}"] = files[source]
    return output


def _validate_split(data: bytes) -> bytes:
    try:
        payload = json.loads(_decode_text(data, "public subject split"))
    except json.JSONDecodeError as exc:
        raise Stage5ImportError("public subject split is invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "status", "seed", "expected_subject_counts", "splits"}:
        raise Stage5ImportError("public subject split schema changed")
    if payload["schema_version"] != 1 or payload["status"] != "complete" or not isinstance(payload["seed"], int):
        raise Stage5ImportError("public subject split is not complete")
    expected = {"train": 136, "val": 24, "test": 40}
    if payload["expected_subject_counts"] != expected or not isinstance(payload["splits"], dict):
        raise Stage5ImportError("public subject split counts changed")
    counts = {key: 0 for key in expected}
    for subject, split in payload["splits"].items():
        if not isinstance(subject, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,32}", subject) or split not in counts:
            raise Stage5ImportError("public subject split contains an unsafe subject assignment")
        counts[split] += 1
    if counts != expected or len(payload["splits"]) != 200:
        raise Stage5ImportError("public subject split does not contain 136/24/40 unique subjects")
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _git_state(root: Path) -> tuple[bool | None, str]:
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, capture_output=True, text=True)
    if probe.returncode:
        return None, ""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True, check=True)
    if head.returncode:
        return False, ""
    return not status.stdout.strip(), head.stdout.strip()


def _build_outputs(
    package: Path,
    expected_receipt_sha256: str,
    repo_root: Path,
    *,
    require_release_metadata: bool,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    receipt, _, files = _load_receipt(package, expected_receipt_sha256)
    table_rows = _csv_rows(files["tables.csv"], "tables.csv", CANONICAL_TABLE_COLUMNS)
    _validate_canonical_tables(table_rows)
    _validate_release_universe(files["release_universe.json"], table_rows)
    stat_rows = _csv_rows(files["statistics.csv"], "statistics.csv", CANONICAL_STAT_COLUMNS)
    _validate_statistics_json(stat_rows, files["statistics.json"])
    stat_rows, reconciled_statistics = _reconcile_unavailable_point_estimates(
        stat_rows, table_rows
    )
    _validate_statistics(stat_rows)
    metadata = _validate_metadata(files["table_metadata.json"], table_rows)
    table_keys = {row["table"] for row in table_rows}
    if table_keys != set(TABLE_ORDER):
        raise Stage5ImportError("canonical Stage-5 table universe is incomplete")
    inventory = _validate_inventory(files, table_keys)
    for path, data in files.items():
        if path.startswith(("figures/", "configs/", "splits/")):
            continue
        _scan_document(_decode_text(data, path), path)

    public_results = _release_result_rows(table_rows)
    public_statistics = _release_stat_rows(stat_rows)
    public_tables = _release_table_rows(metadata, public_results)
    with tempfile.TemporaryDirectory(prefix="segbench-stage5-convert-") as temporary:
        temp = Path(temporary)
        bundle = temp / "bundle"
        bundle.mkdir()
        generated = {
            "public_results.csv": _csv_bytes(UPDATE.RESULT_COLUMNS, public_results),
            "statistics.csv": _csv_bytes(UPDATE.STATISTIC_COLUMNS, public_statistics),
            "table_status.csv": _csv_bytes(UPDATE.TABLE_COLUMNS, public_tables),
        }
        for name, data in generated.items():
            (bundle / name).write_bytes(data)
        manifest = {
            "schema_version": 1, "kind": UPDATE.BUNDLE_KIND,
            "files": {name: _sha256(data) for name, data in generated.items()},
        }
        (bundle / UPDATE.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        compact = UPDATE.validate_bundle(bundle)
        outputs = UPDATE._stage_outputs(compact, repo_root)

        for source in inventory["tables"]:
            outputs[f"results/tables/{Path(source).name}"] = files[source]
        for source in sorted(CORE_FILES):
            outputs[f"results/canonical/{source}"] = files[source]

        figure_paths: list[Path] = []
        for source in inventory["figures"]:
            destination = temp / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(files[source])
            figure_paths.append(destination)
            outputs[source] = files[source]
        figure_manifest = build_manifest(temp, figure_paths, source_receipt_sha256=expected_receipt_sha256)
        outputs["figures/figure_manifest.json"] = (json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n").encode()

    config_outputs = _validate_configs(files, inventory["configs"], table_rows)
    outputs.update(config_outputs)
    split_imported = bool(inventory["splits"])
    if split_imported:
        outputs["splits/sd302_subject_split.json"] = _validate_split(files[inventory["splits"][0]])
    expected_configs = {f"configs/{name}" for name in EXPECTED_CONFIG_NAMES}
    if require_release_metadata and (set(config_outputs) != expected_configs or not split_imported):
        raise Stage5ImportError(
            "final release metadata is absent: canonical package must receipt-bind the complete "
            "public/configs/*.yaml universe with final validation choices and "
            "public/splits/sd302_subject_split.json"
        )
    source_inventory = {
        key: receipt["outputs"][f"public/{key}"] for key in sorted(files)
    }
    details = {
        "source_receipt_sha256": expected_receipt_sha256,
        "source_public_inventory_sha256": _canonical_sha256(source_inventory),
        "source_public_files": len(source_inventory),
        "aggregate_rows": {"results": len(public_results), "statistics": len(public_statistics), "tables": len(public_tables)},
        "canonical_reconciliations": {
            "unavailable_point_estimates": reconciled_statistics,
        },
        "release_metadata": {
            "configs": "imported" if config_outputs else "absent_from_canonical_package",
            "config_count": len(config_outputs),
            "subject_split": "imported" if split_imported else "absent_from_canonical_package",
        },
    }
    return outputs, details


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or (path.exists() and path.is_symlink()):
        raise Stage5ImportError(f"refusing symlink output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _allowed_import_output(relative: str) -> bool:
    fixed = {
        "README.md", "docs/RESULTS.md", "site/index.html", "site/build_manifest.json",
        "site/assets/results/public_results.csv", "results/public_results.csv",
        "results/statistics.csv", "results/table_status.csv",
        "splits/sd302_subject_split.json", "figures/figure_manifest.json",
    }
    if relative in fixed:
        return True
    path = Path(relative)
    if path.parent == Path("results/canonical"):
        return path.name in CORE_FILES
    if path.parent == Path("results/tables"):
        return path.name in {
            f"{stem}.{suffix}" for stem in TABLE_STEMS.values()
            for suffix in ("csv", "md", "tex")
        }
    if path.parent == Path("figures"):
        return path.stem in FIGURE_KINDS and path.suffix in MIME_BY_SUFFIX
    if path.parent == Path("configs"):
        return path.name in EXPECTED_CONFIG_NAMES
    return False


def _safe_output_path(root: Path, relative: str) -> Path:
    rel = _safe_relative(relative)
    if not _allowed_import_output(rel.as_posix()) and rel.as_posix() != "results/stage5_import_receipt.json":
        raise Stage5ImportError(f"import output is outside the publication allowlist: {relative}")
    root_resolved = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise Stage5ImportError("release root is absent or a symlink")
    target = root / rel
    current = target.parent
    while current != root:
        if current.exists() and current.is_symlink():
            raise Stage5ImportError(f"refusing symlink ancestor for output: {relative}")
        current = current.parent
    try:
        target.resolve(strict=False).relative_to(root_resolved)
    except ValueError as exc:
        raise Stage5ImportError(f"import output escapes release root: {relative}") from exc
    return target


def _validate_import_receipt(root: Path, data: bytes) -> dict[str, Any]:
    try:
        receipt = json.loads(_decode_text(data, "previous import receipt"))
    except json.JSONDecodeError as exc:
        raise Stage5ImportError("previous import receipt is invalid") from exc
    if not isinstance(receipt, dict) or set(receipt) != IMPORT_RECEIPT_FIELDS:
        raise Stage5ImportError("previous import receipt schema changed")
    source = receipt.get("source_receipt")
    dependencies = receipt.get("converter_dependencies")
    outputs = receipt.get("outputs")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != IMPORT_KIND
        or receipt.get("status") != "PASS"
        or receipt.get("contains_per_image_rows") is not False
        or receipt.get("contains_biometric_media") is not False
        or not isinstance(source, dict)
        or set(source) != {"kind", "sha256"}
        or source.get("kind") != RECEIPT_KIND
        or HEX64.fullmatch(str(source.get("sha256", ""))) is None
        or HEX64.fullmatch(str(receipt.get("source_public_inventory_sha256", ""))) is None
        or HEX64.fullmatch(str(receipt.get("converter_sha256", ""))) is None
        or not isinstance(dependencies, dict)
        or not isinstance(outputs, dict)
        or receipt.get("converter_dependencies_sha256") != _canonical_sha256(dependencies)
        or receipt.get("outputs_sha256") != _canonical_sha256(outputs)
    ):
        raise Stage5ImportError("previous import receipt failed provenance validation")
    for relative, digest in dependencies.items():
        _safe_relative(str(relative))
        if HEX64.fullmatch(str(digest)) is None:
            raise Stage5ImportError("previous import receipt contains an invalid dependency digest")
    for relative, digest in outputs.items():
        if not isinstance(relative, str) or not _allowed_import_output(relative):
            raise Stage5ImportError("previous import receipt contains an unsafe output path")
        if HEX64.fullmatch(str(digest)) is None:
            raise Stage5ImportError("previous import receipt contains an invalid output digest")
        target = _safe_output_path(root, relative)
        if not target.is_file() or target.is_symlink() or _sha256(target.read_bytes()) != digest:
            raise Stage5ImportError(f"previous imported output changed outside the bridge: {relative}")
    return receipt


def _previous_outputs(root: Path) -> set[str]:
    path = root / "results" / "stage5_import_receipt.json"
    if not path.exists():
        return set()
    if path.is_symlink() or not path.is_file():
        raise Stage5ImportError("previous import receipt is not a regular file")
    receipt = _validate_import_receipt(root, _regular_bytes(path, "previous import receipt"))
    return set(receipt["outputs"])


def _apply(outputs: dict[str, bytes], repo_root: Path, import_receipt: bytes) -> None:
    old = _previous_outputs(repo_root)
    next_paths = set(outputs)
    deletions = old - next_paths
    touched = sorted(next_paths | deletions | {"results/stage5_import_receipt.json"})
    previous: dict[str, bytes | None] = {}
    try:
        for relative in touched:
            path = _safe_output_path(repo_root, relative)
            if path.is_symlink():
                raise Stage5ImportError(f"refusing symlink output: {relative}")
            previous[relative] = path.read_bytes() if path.is_file() else None
        for relative in sorted(deletions):
            _safe_output_path(repo_root, relative).unlink()
        for relative, data in sorted(outputs.items()):
            _atomic_write(_safe_output_path(repo_root, relative), data)
        _atomic_write(_safe_output_path(repo_root, "results/stage5_import_receipt.json"), import_receipt)
    except Exception:
        for relative, data in reversed(list(previous.items())):
            path = _safe_output_path(repo_root, relative)
            if data is None:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            else:
                _atomic_write(path, data)
        raise


def run_import(
    package: Path,
    expected_receipt_sha256: str,
    *,
    repo_root: Path = ROOT,
    dry_run: bool,
    require_release_metadata: bool = False,
) -> dict[str, Any]:
    """Validate, convert, stage, and optionally apply one canonical package."""
    clean, initial_head = _git_state(repo_root)
    if not dry_run and not require_release_metadata:
        raise Stage5ImportError("controlled apply requires final release metadata")
    if not dry_run and clean is not True:
        raise Stage5ImportError("controlled apply requires a clean Git worktree at process start")
    outputs, details = _build_outputs(
        package, expected_receipt_sha256, repo_root,
        require_release_metadata=require_release_metadata,
    )
    output_hashes = {path: _sha256(data) for path, data in sorted(outputs.items())}
    dependencies = _converter_dependencies()
    import_receipt = {
        "schema_version": 1, "kind": IMPORT_KIND, "status": "PASS",
        "source_receipt": {"kind": RECEIPT_KIND, "sha256": expected_receipt_sha256},
        "source_public_inventory_sha256": details["source_public_inventory_sha256"],
        "converter_sha256": _sha256(Path(__file__).read_bytes()),
        "converter_dependencies": dependencies,
        "converter_dependencies_sha256": _canonical_sha256(dependencies),
        "contains_per_image_rows": False, "contains_biometric_media": False,
        "release_metadata": details["release_metadata"], "outputs": output_hashes,
        "outputs_sha256": _canonical_sha256(output_hashes),
    }
    encoded_receipt = (json.dumps(import_receipt, indent=2, sort_keys=True) + "\n").encode()
    if not dry_run:
        clean_before_apply, head_before_apply = _git_state(repo_root)
        if clean_before_apply is not True or head_before_apply != initial_head:
            raise Stage5ImportError("Git worktree or HEAD changed before controlled apply")
        _apply(outputs, repo_root, encoded_receipt)
    return {
        "schema_version": 1, "kind": REPORT_KIND, "status": "PASS",
        "mode": "dry-run" if dry_run else "controlled-apply",
        "mutations_applied": not dry_run,
        "git_tree_clean_at_start": clean,
        **details,
        "output_sha256": output_hashes,
        "import_receipt_sha256": _sha256(encoded_receipt),
        "next_phase": (
            "review dry-run, then controlled apply"
            if dry_run else
            "run working scan, commit the exact outputs, then run a final clean-tree scan"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--require-release-metadata", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report_resolved = args.report.resolve()
    for forbidden_root, label in ((ROOT.resolve(), "release repository"), (args.package.resolve(), "Stage-5 package")):
        try:
            report_resolved.relative_to(forbidden_root)
        except ValueError:
            continue
        raise Stage5ImportError(f"import report must be outside the {label}")
    report = run_import(
        args.package, args.expected_receipt_sha256, dry_run=args.dry_run,
        require_release_metadata=args.require_release_metadata,
    )
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(args.report, encoded)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
