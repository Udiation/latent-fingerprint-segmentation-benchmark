#!/usr/bin/env python3
"""Aggregate SD302b live-scan tables from authenticated, locked B1 evidence.

The command never runs a model and never treats a minutiae hull as a complete
foreground mask.  It authenticates the frozen SD302g B1 adoption receipts and
their per-image evaluator CSVs, restricts them to the independently verified
SD302b baseline mapping, and reports MC, HR, PAR, OHF, and predicted foreground
area fraction.  ``Plain`` is always printed as ``Plain (segmented slap)``.

Agreement is a separate stability analysis, not accuracy.  It is computed only
when every retained binary mask set can be reauthenticated against its adoption
receipt.  Its seed-consensus policy must be written and independently hash
pinned before the aggregation command is allowed to read metric outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

# Keep direct ``python scripts/...`` execution consistent with the documented
# runbooks without requiring an ambient PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segbench.artifact_identity import canonical_sha256, sha256_file
from segbench.b1_statistics import (
    B1StatisticsError,
    validate_per_image,
    validate_universe,
)


BOOTSTRAP_SEED = 20260908
BOOTSTRAP_REPLICATES = 1_000
EXPECTED_MAPPING_KIND = "sd302b_sd302g_exact_mapping_receipt_v1"
EXPECTED_B1_REPORT_KIND = "sd302g_b1_reference_verified_report"
REFERENCE_METHODS = (
    "all_background",
    "all_foreground",
    "hull_as_prediction",
    "hull_dilated_10px_500ppi",
)
RA_RUNET_ALIGNED_PENDING_REASON = (
    "Pending: no authenticated aligned RA-RUnet checkpoint was completed before the "
    "benchmark freeze; never substitute the scored no-alignment RUnet."
)
METRICS = ("mc", "hr", "par", "ohf", "predicted_area_fraction")
TABLE_COLUMNS = (
    "method",
    "display_name",
    "family",
    "training_data",
    "status",
    "reason",
    "capture",
    "device",
    "n_images",
    "n_subjects",
    "n_seeds",
    "seed_ids",
    "ms_per_img_mean",
    "ms_per_img_seed_std",
    "params_m_mean",
    "params_m_seed_std",
    "mc_mean",
    "mc_seed_std",
    "mc_ci_low",
    "mc_ci_high",
    "hr_mean",
    "hr_seed_std",
    "hr_ci_low",
    "hr_ci_high",
    "par_mean",
    "par_seed_std",
    "par_ci_low",
    "par_ci_high",
    "ohf_mean",
    "ohf_seed_std",
    "ohf_ci_low",
    "ohf_ci_high",
    "predicted_area_fraction_mean",
    "predicted_area_fraction_seed_std",
    "predicted_area_fraction_ci_low",
    "predicted_area_fraction_ci_high",
    "bootstrap_seed",
    "bootstrap_replicates",
    "bootstrap_unit",
    "bootstrap_status",
    "interpretation",
)


class AggregateError(ValueError):
    """Locked evidence cannot support the requested Table-b claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def outside_repository(path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        path.resolve().relative_to(repository)
    except ValueError:
        return
    raise AggregateError(f"restricted output must be outside repository: {path}")


def load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregateError(f"{label} is not readable JSON: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_csv(path: Path, label: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise AggregateError(f"{label} is not readable: {path}") from exc
    require(bool(columns), f"{label} has no header")
    return columns, rows


def regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    require(resolved.is_file() and not resolved.is_symlink(), f"{label} missing/symlink: {path}")
    return resolved


def _identity(path: Path) -> Dict[str, object]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def consensus_policy() -> Dict[str, Any]:
    """Return the immutable policy that must exist before metrics are opened."""
    return {
        "schema_version": 1,
        "kind": "sd302b_inter_method_agreement_policy_v1",
        "status": "PREREGISTERED",
        "selection_signal": False,
        "scope": (
            "all status=score table methods in the authenticated production B1 "
            "method universe; deterministic references and N/A methods excluded"
        ),
        "retained_mask_gate": {
            "required": True,
            "full_b1_test_records_per_adoption": 475,
            "authentication": (
                "both legacy prediction_set_sha256 and canonical "
                "prediction_records_sha256 must match the adoption receipt"
            ),
            "on_missing_complete_set": "do not compute agreement; report pending",
            "on_identity_mismatch": "fail closed",
        },
        "seed_consensus": {
            "seeded_methods": {
                "required_seed_ids": [0, 1, 2],
                "rule": "strict pixelwise majority",
                "foreground_votes_required": 2,
            },
            "unseeded_methods": "the single authenticated binary mask",
            "binary_values": [0, 255],
            "resizing": "forbidden; masks must equal the canonical SD302b canvas",
        },
        "agreement": {
            "metric": "Dice",
            "empty_empty_value": 1.0,
            "per_image_statistic": "mean over all unordered method pairs",
            "summary_statistic": "image-macro mean",
            "interpretation": "agreement, not accuracy",
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "unit": "subject",
            "interval": "percentile_95",
        },
    }


def write_policy(path: Path) -> Mapping[str, Any]:
    path = path.resolve()
    outside_repository(path)
    if path.exists():
        raise AggregateError(f"refusing to overwrite preregistered policy: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = consensus_policy()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        _write_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**payload, "sha256": sha256_file(path), "path": str(path)}


def authenticate_policy(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    path = regular_file(path, "consensus policy")
    require(len(expected_sha256) == 64, "consensus policy requires an independently pinned SHA-256")
    observed = sha256_file(path)
    require(observed == expected_sha256, "consensus policy SHA-256 differs from the independent pin")
    payload = load_json(path, "consensus policy")
    require(
        canonical_sha256(payload) == canonical_sha256(consensus_policy()),
        "consensus policy content differs from the preregistered implementation",
    )
    return payload


def authenticate_mapping(
    receipt_path: Path,
    *,
    expected_test_records: int,
    expected_full_records: int,
) -> tuple[Mapping[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    receipt_path = regular_file(receipt_path, "mapping receipt")
    receipt = load_json(receipt_path, "mapping receipt")
    require(
        receipt.get("kind") == EXPECTED_MAPPING_KIND
        and receipt.get("status") == "REFERENCE_VERIFIED_MAPPING",
        "mapping receipt is not reference verified",
    )
    outputs = receipt.get("outputs")
    require(isinstance(outputs, dict), "mapping receipt lacks output identities")
    loaded: Dict[str, list[dict[str, str]]] = {}
    headers: Dict[str, tuple[str, ...]] = {}
    for name in (
        "mapping_full_private.csv",
        "mapping_test_private.csv",
        "inference_manifest.csv",
    ):
        record = outputs.get(name)
        require(isinstance(record, dict), f"mapping receipt omits {name}")
        path = regular_file(receipt_path.parent / name, name)
        require(
            sha256_file(path) == record.get("sha256")
            and path.stat().st_size == int(record.get("bytes", -1)),
            f"{name} bytes differ from the mapping receipt",
        )
        headers[name], loaded[name] = load_csv(path, name)
        require(len(loaded[name]) == int(record.get("rows", -1)), f"{name} row count changed")
    full = loaded["mapping_full_private.csv"]
    test = loaded["mapping_test_private.csv"]
    inference = loaded["inference_manifest.csv"]
    require(len(full) == expected_full_records, "full SD302b mapping universe changed")
    require(len(test) == expected_test_records, "test SD302b mapping universe changed")
    require(len(inference) == expected_full_records, "GT-blind inference universe changed")
    test_ids = [row["image_id"] for row in test]
    require(len(test_ids) == len(set(test_ids)), "test mapping contains duplicate identities")
    require(
        all(
            row["split"] == "test"
            and row["group"] == "baseline"
            and row["canonical_pixels_equal"].lower() == "true"
            and row["actual_ppi"] == "500"
            for row in test
        ),
        "test mapping is not an exact canonical baseline/test/500-ppi subset",
    )
    require(
        Counter(row["source_capture"] for row in test) == Counter({"roll": 430, "slap": 9})
        if expected_test_records == 439
        else True,
        "mapped test capture counts changed",
    )
    inference_header = set(headers["inference_manifest.csv"])
    require(
        not {"hull_path", "minutiae_path"} & inference_header
        and all(row.get("mask_path", "") == "" for row in inference),
        "inference manifest exposes ground-truth paths",
    )
    require(
        set(test_ids) == {row["image_id"] for row in inference if row["split"] == "test"},
        "inference and private test mapping identities differ",
    )
    return receipt, full, test


def _verify_report_outputs(receipt_path: Path, receipt: Mapping[str, Any]) -> None:
    outputs = receipt.get("outputs")
    require(isinstance(outputs, dict), "B1 report receipt lacks output hashes")
    names = {
        "report_sha256": "REPORT.md",
        "statistics_receipt_sha256": "statistics_receipt.json",
        "summary_b1_sha256": "summary_b1.csv",
        "table_b1_sha256": "table_b1.csv",
        "wilcoxon_holm_b1_sha256": "wilcoxon_holm_b1.csv",
    }
    for key, name in names.items():
        path = regular_file(receipt_path.parent / name, f"B1 report output {name}")
        require(sha256_file(path) == outputs.get(key), f"B1 report output changed: {name}")


def _verified_manifest_rows(path: Path) -> tuple[Mapping[str, dict[str, str]], set[str]]:
    columns, rows = load_csv(path, "verified SD302g manifest")
    required = {
        "image_id", "image_path", "hull_path", "minutiae_path", "subject_id",
        "split", "group", "device", "capture", "width", "height", "ppi",
    }
    require(required.issubset(columns), "verified SD302g manifest lacks required columns")
    by_id = {row["image_id"]: row for row in rows}
    require(len(by_id) == len(rows), "verified SD302g identities are not unique")
    test_ids = {row["image_id"] for row in rows if row["split"] == "test"}
    return by_id, test_ids


def _adoption_to_table(report: Mapping[str, Any]) -> Dict[str, str]:
    table_methods = report.get("table_methods")
    require(isinstance(table_methods, dict), "B1 report lacks table-method metadata")
    result: Dict[str, str] = {}
    for table_method, record in table_methods.items():
        require(isinstance(record, dict), f"malformed B1 table-method record: {table_method}")
        for adoption in record.get("adoption_methods", []):
            key = str(adoption)
            require(key not in result, f"adoption method appears twice: {key}")
            result[key] = str(table_method)
    return result


def authenticate_b1(
    report_path: Path,
    verified_manifest: Path,
    mapped: Sequence[Mapping[str, str]],
    *,
    require_production: bool,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Mapping[str, Any]],
    Dict[str, list[tuple[int | None, pd.DataFrame]]],
    Dict[str, Mapping[str, Any]],
    Dict[str, Path],
]:
    report_path = regular_file(report_path, "B1 report receipt")
    report = load_json(report_path, "B1 report receipt")
    require(
        report.get("kind") == EXPECTED_B1_REPORT_KIND
        and report.get("status") == "REFERENCE_VERIFIED"
        and report.get("schema_version") == 2
        and report.get("synthetic_test_only") is (not require_production),
        "B1 report is not the expected authenticated report type",
    )
    statistics = report.get("statistics")
    require(
        isinstance(statistics, dict)
        and statistics.get("status") == "COMPLETE"
        and statistics.get("per_image_evidence_authenticated") is True,
        "B1 report does not authenticate complete per-image evidence",
    )
    _verify_report_outputs(report_path, report)
    manifest_hash = sha256_file(verified_manifest)
    require(
        report.get("identities", {}).get("verified_manifest_sha256") == manifest_hash,
        "B1 report and verified SD302g manifest identities differ",
    )
    method_universe_record = report.get("method_universe")
    require(isinstance(method_universe_record, dict), "B1 report lacks its method universe")
    universe_path = regular_file(
        Path(str(method_universe_record.get("path", ""))), "B1 method universe"
    )
    require(
        sha256_file(universe_path) == method_universe_record.get("sha256")
        and universe_path.stat().st_size == int(method_universe_record.get("bytes", -1)),
        "B1 method universe differs from the report receipt",
    )
    universe_payload = load_json(universe_path, "B1 method universe")
    try:
        universe = validate_universe(universe_payload, require_production=require_production)
    except B1StatisticsError as exc:
        raise AggregateError(str(exc)) from exc

    verified_by_id, verified_test_ids = _verified_manifest_rows(verified_manifest)
    expected_total = 475 if require_production else int(report.get("n_images", 0))
    require(len(verified_test_ids) == expected_total, "verified B1 test universe changed")
    mapped_by_id = {row["image_id"]: row for row in mapped}
    require(set(mapped_by_id).issubset(verified_test_ids), "mapped IDs are outside verified B1 test")
    adoption_table = _adoption_to_table(report)
    adoption_records = report.get("adoption_receipts")
    require(isinstance(adoption_records, list) and adoption_records, "B1 report lacks adoptions")
    require(
        {str(item.get("method")) for item in adoption_records if isinstance(item, dict)}
        == set(report.get("methods", [])),
        "B1 report adoption and method sets differ",
    )
    bound_hashes = statistics.get("adoption_receipt_sha256")
    require(isinstance(bound_hashes, dict), "B1 statistics omit adoption hashes")

    frames: Dict[str, list[tuple[int | None, pd.DataFrame]]] = defaultdict(list)
    adoption_receipts: Dict[str, Mapping[str, Any]] = {}
    adoption_paths: Dict[str, Path] = {}
    common_identity: Mapping[str, Any] | None = None
    for item in adoption_records:
        require(isinstance(item, dict), "malformed B1 adoption index row")
        method = str(item.get("method", ""))
        receipt_path = regular_file(Path(str(item.get("path", ""))), f"{method} adoption receipt")
        receipt_hash = sha256_file(receipt_path)
        require(
            receipt_hash == item.get("sha256") == bound_hashes.get(method),
            f"{method}: adoption receipt identity differs from B1 report",
        )
        receipt = load_json(receipt_path, f"{method} adoption receipt")
        require(
            receipt.get("kind") == "sd302g_b1_reference_verified_adoption"
            and receipt.get("status") == "REFERENCE_VERIFIED"
            and receipt.get("method") == method
            and receipt.get("no_prediction_regeneration") is True
            and receipt.get("test_labels_opened_only_by_evaluate_hull") is True,
            f"{method}: adoption is not reference verified",
        )
        identities = receipt.get("identities")
        require(isinstance(identities, dict), f"{method}: adoption lacks identities")
        if common_identity is None:
            common_identity = identities
        for key in (
            "reference_gate_sha256", "reference_report_sha256",
            "verified_manifest_sha256", "split_sha256", "evaluate_hull_sha256",
        ):
            require(
                identities.get(key) == report.get("identities", {}).get(key)
                and identities.get(key) == common_identity.get(key),
                f"{method}: B1 protocol identity differs for {key}",
            )
        per_path = regular_file(
            receipt_path.parent / f"per_image_{method}_test.csv",
            f"{method} per-image evidence",
        )
        require(
            sha256_file(per_path) == receipt.get("evaluation", {}).get("per_image_sha256"),
            f"{method}: per-image evidence differs from adoption receipt",
        )
        try:
            raw = pd.read_csv(per_path, dtype={"image_id": str, "subject_id": str})
            validated = validate_per_image(raw, method)
        except (OSError, UnicodeError, pd.errors.ParserError, B1StatisticsError) as exc:
            raise AggregateError(f"{method}: invalid authenticated per-image evidence: {exc}") from exc
        require(
            len(validated) == expected_total
            and set(validated.image_id.astype(str)) == verified_test_ids,
            f"{method}: per-image B1 universe differs from verified test",
        )
        subset = validated[validated.image_id.astype(str).isin(mapped_by_id)].copy()
        require(
            len(subset) == len(mapped_by_id)
            and set(subset.image_id.astype(str)) == set(mapped_by_id),
            f"{method}: mapped baseline test subset is incomplete",
        )
        subset = subset.sort_values("image_id", kind="stable").reset_index(drop=True)
        areas: list[int] = []
        labels: list[str] = []
        for record in subset.to_dict("records"):
            image_id = str(record["image_id"])
            map_row = mapped_by_id[image_id]
            gt_row = verified_by_id[image_id]
            require(
                str(record["subject_id"]) == map_row["subject_id"] == gt_row["subject_id"]
                and str(record["device"]) == map_row["device"] == gt_row["device"]
                and str(record["capture"]) == map_row["source_capture"] == gt_row["capture"]
                and str(record["group"]) == "baseline",
                f"{method}: mapped metadata differs for one test record",
            )
            area = int(gt_row["width"]) * int(gt_row["height"])
            require(int(record["predicted_pixels"]) <= area, f"{method}: predicted area exceeds canvas")
            areas.append(area)
            labels.append(map_row["report_capture"])
        subset["image_pixels"] = areas
        subset["predicted_area_fraction"] = (
            pd.to_numeric(subset.predicted_pixels).to_numpy(float) / np.asarray(areas, float)
        )
        subset["report_capture"] = labels
        table_method = adoption_table.get(method)
        require(table_method in universe, f"{method}: adoption lacks a table-method binding")
        seed_raw = receipt.get("seed")
        seed = seed_raw if type(seed_raw) is int else None
        frames[str(table_method)].append((seed, subset))
        adoption_receipts[method] = receipt
        adoption_paths[method] = receipt_path

    expected_adoptions = set(adoption_table)
    require(set(adoption_receipts) == expected_adoptions, "authenticated adoption set is incomplete")
    for method, meta in universe.items():
        if meta["status"] != "score":
            require(not frames.get(method), f"{method}: N/A row unexpectedly has evidence")
            continue
        runs = frames.get(method, [])
        seeds = sorted(seed for seed, _ in runs if seed is not None)
        if meta["seed_regime"] == "seeded":
            require(seeds == [0, 1, 2] and len(runs) == 3, f"{method}: all seeds 0,1,2 required")
        else:
            require(len(runs) == 1 and runs[0][0] is None, f"{method}: one unseeded run required")
    return report, universe, dict(frames), adoption_receipts, adoption_paths


def _derived_seed(aggregate: str, value: str) -> int:
    digest = hashlib.sha256(
        f"{BOOTSTRAP_SEED}\0{aggregate}\0{value}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _cluster_intervals(
    frame: pd.DataFrame, metrics: Sequence[str], aggregate: str, value: str
) -> tuple[Dict[str, tuple[float, float]], str, int]:
    subjects = sorted(set(frame.subject_id.astype(str)))
    require(bool(subjects), "cannot bootstrap an empty stratum")
    seed = _derived_seed(aggregate, value)
    if len(subjects) == 1:
        draws = np.zeros((BOOTSTRAP_REPLICATES, 1), dtype=np.int64)
        status = "degenerate_one_subject"
    else:
        draws = np.random.Generator(np.random.PCG64(seed)).integers(
            0, len(subjects), size=(BOOTSTRAP_REPLICATES, len(subjects)), dtype=np.int64
        )
        status = "ok"
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    counts = np.zeros(len(subjects), dtype=np.int64)
    for subject in frame.subject_id.astype(str):
        counts[subject_index[subject]] += 1
    denominators = counts[draws].sum(axis=1)
    intervals: Dict[str, tuple[float, float]] = {}
    for metric in metrics:
        sums = np.zeros(len(subjects), dtype=np.float64)
        for subject, metric_value in zip(
            frame.subject_id.astype(str), frame[metric].astype(float)
        ):
            sums[subject_index[subject]] += metric_value
        values = sums[draws].sum(axis=1) / denominators
        low, high = np.quantile(values, [0.025, 0.975], method="linear")
        intervals[metric] = (float(low), float(high))
    return intervals, status, seed


def _canonical_method_frame(
    method: str,
    runs: Sequence[tuple[int | None, pd.DataFrame]],
) -> pd.DataFrame:
    require(bool(runs), f"{method}: no evidence frames")
    base = runs[0][1][
        [
            "image_id", "subject_id", "group", "device", "capture", "report_capture",
            "n_minutiae", "hull_pixels", "image_pixels",
        ]
    ].copy()
    identity = tuple(base.image_id.astype(str))
    for _, frame in runs[1:]:
        require(tuple(frame.image_id.astype(str)) == identity, f"{method}: seed pairing changed")
        for column in (
            "subject_id", "group", "device", "capture", "report_capture",
            "n_minutiae", "hull_pixels", "image_pixels",
        ):
            require(
                tuple(frame[column].astype(str)) == tuple(base[column].astype(str)),
                f"{method}: seed metadata changed: {column}",
            )
    for metric in METRICS:
        base[metric] = np.stack([frame[metric].to_numpy(float) for _, frame in runs]).mean(axis=0)
    return base


def _summary_row(
    method: str,
    meta: Mapping[str, Any],
    runs: Sequence[tuple[int | None, pd.DataFrame]],
    canonical: pd.DataFrame,
    part: pd.DataFrame,
    *,
    capture: str,
    device: str,
    aggregate: str,
    value: str,
) -> Dict[str, object]:
    intervals, bootstrap_status, seed = _cluster_intervals(part, METRICS, aggregate, value)
    ids = set(part.image_id.astype(str))
    row: Dict[str, object] = {
        "method": method,
        "display_name": meta["display_name"],
        "family": meta["family"],
        "training_data": meta["training_data"],
        "status": "ok",
        "reason": "",
        "capture": capture,
        "device": device,
        "n_images": len(part),
        "n_subjects": int(part.subject_id.nunique()),
        "n_seeds": len(runs),
        "seed_ids": ",".join("unseeded" if seed_id is None else str(seed_id) for seed_id, _ in runs),
        "ms_per_img_mean": meta.get("ms_per_img_mean", float("nan")),
        "ms_per_img_seed_std": meta.get("ms_per_img_seed_std", float("nan")),
        "params_m_mean": meta.get("params_m_mean", float("nan")),
        "params_m_seed_std": meta.get("params_m_seed_std", float("nan")),
        "bootstrap_seed": seed,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "subject",
        "bootstrap_status": bootstrap_status,
        "interpretation": "Hull GT is a lower bound of true foreground; pixel Dice not reported.",
    }
    del canonical  # The explicit argument documents that ``part`` is seed-canonical.
    for metric in METRICS:
        values = part[metric].to_numpy(float)
        seed_means = [
            float(frame[frame.image_id.astype(str).isin(ids)][metric].mean())
            for _, frame in runs
        ]
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_seed_std"] = (
            float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else float("nan")
        )
        row[f"{metric}_ci_low"] = intervals[metric][0]
        row[f"{metric}_ci_high"] = intervals[metric][1]
    return row


def _na_row(
    method: str,
    meta: Mapping[str, Any],
    *,
    capture: str,
    device: str,
    n_images: int,
    n_subjects: int,
) -> Dict[str, object]:
    reason = (
        RA_RUNET_ALIGNED_PENDING_REASON
        if method == "ra_runet_aligned"
        else meta["reason"]
    )
    row: Dict[str, object] = {
        "method": method,
        "display_name": meta["display_name"],
        "family": meta["family"],
        "training_data": meta["training_data"],
        "status": "N/A",
        "reason": reason,
        "capture": capture,
        "device": device,
        "n_images": n_images,
        "n_subjects": n_subjects,
        "n_seeds": 0,
        "seed_ids": "",
        "ms_per_img_mean": meta.get("ms_per_img_mean", float("nan")),
        "ms_per_img_seed_std": meta.get("ms_per_img_seed_std", float("nan")),
        "params_m_mean": meta.get("params_m_mean", float("nan")),
        "params_m_seed_std": meta.get("params_m_seed_std", float("nan")),
        "bootstrap_seed": "",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "subject",
        "bootstrap_status": "unavailable",
        "interpretation": "Hull GT is a lower bound of true foreground; pixel Dice not reported.",
    }
    for metric in METRICS:
        for suffix in ("mean", "seed_std", "ci_low", "ci_high"):
            row[f"{metric}_{suffix}"] = float("nan")
    return row


def _confined_artifact(manifest: Path, value: str, label: str) -> Path:
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} path escapes")
    root = manifest.parent.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AggregateError(f"{label} path escapes verified build") from exc
    return regular_file(path, label)


def _read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    require(value is not None, f"could not decode binary mask: {path}")
    assert value is not None
    require(set(int(item) for item in np.unique(value)).issubset({0, 255}), f"non-binary mask: {path}")
    if shape is not None:
        require(value.shape == shape, f"mask shape {value.shape} != canonical canvas {shape}: {path}")
    return value > 0


def _read_points(path: Path, width: int, height: int) -> list[tuple[int, int]]:
    columns, rows = load_csv(path, "verified minutiae")
    require({"x_px", "y_px"}.issubset(columns), "verified minutiae lack x_px/y_px")
    points: list[tuple[int, int]] = []
    for row in rows:
        x, y = int(row["x_px"]), int(row["y_px"])
        require(0 <= x < width and 0 <= y < height, "verified minutia is outside canvas")
        points.append((x, y))
    require(bool(points), "verified minutiae set is empty")
    return points


def build_reference_frames(
    verified_manifest: Path,
    mapped: Sequence[Mapping[str, str]],
) -> tuple[Dict[str, pd.DataFrame], str]:
    verified_by_id, _ = _verified_manifest_rows(verified_manifest)
    rows: Dict[str, list[dict[str, object]]] = {method: [] for method in REFERENCE_METHODS}
    hull_records: list[dict[str, object]] = []
    for map_row in sorted(mapped, key=lambda item: item["image_id"]):
        source = verified_by_id[map_row["image_id"]]
        width, height = int(source["width"]), int(source["height"])
        hull_path = _confined_artifact(verified_manifest, source["hull_path"], "verified hull")
        points_path = _confined_artifact(
            verified_manifest, source["minutiae_path"], "verified minutiae"
        )
        hull = _read_mask(hull_path, (height, width))
        points = _read_points(points_path, width, height)
        require(all(hull[y, x] for x, y in points), "verified hull omits an expert minutia")
        hull_pixels = int(hull.sum())
        image_pixels = int(hull.size)
        source_ppi = float(source["ppi"])
        radius = int(round(10.0 * source_ppi / 500.0))
        require(radius >= 1, "invalid physical dilation radius")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        dilated = cv2.dilate(hull.astype(np.uint8), kernel, iterations=1).astype(bool)
        dilated_pixels = int(dilated.sum())
        common = {
            "image_id": map_row["image_id"],
            "subject_id": map_row["subject_id"],
            "group": "baseline",
            "device": map_row["device"],
            "capture": map_row["source_capture"],
            "report_capture": map_row["report_capture"],
            "n_minutiae": len(points),
            "hull_pixels": hull_pixels,
            "image_pixels": image_pixels,
        }
        specifications = {
            "all_background": (0.0, 0.0, 0.0, 0.0, 0.0),
            "all_foreground": (
                1.0, 1.0, image_pixels / hull_pixels,
                (image_pixels - hull_pixels) / image_pixels, 1.0,
            ),
            "hull_as_prediction": (1.0, 1.0, 1.0, 0.0, hull_pixels / image_pixels),
            "hull_dilated_10px_500ppi": (
                1.0, 1.0, dilated_pixels / hull_pixels,
                (dilated_pixels - hull_pixels) / dilated_pixels,
                dilated_pixels / image_pixels,
            ),
        }
        for method, values in specifications.items():
            rows[method].append({**common, **dict(zip(METRICS, values))})
        hull_records.append(
            {
                "image_id": map_row["image_id"],
                "hull_sha256": sha256_file(hull_path),
                "minutiae_sha256": sha256_file(points_path),
                "native_dilation_radius_px": radius,
            }
        )
    return {method: pd.DataFrame(records) for method, records in rows.items()}, canonical_sha256(hull_records)


def _reference_meta(method: str) -> Dict[str, object]:
    names = {
        "all_background": "All-background",
        "all_foreground": "All-foreground",
        "hull_as_prediction": "Hull-as-prediction",
        "hull_dilated_10px_500ppi": "Hull + 10 px dilation (500 ppi)",
    }
    return {
        "display_name": names[method],
        "family": "Reference",
        "training_data": "deterministic reference",
        "ms_per_img_mean": float("nan"),
        "ms_per_img_seed_std": float("nan"),
        "params_m_mean": 0.0,
        "params_m_seed_std": float("nan"),
    }


def authenticate_efficiency_table(
    path: Path,
    expected_sha256: str,
    universe: Mapping[str, Mapping[str, Any]],
) -> tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    """Load only method-level efficiency metadata from a hash-pinned public table."""
    path = regular_file(path, "efficiency metadata table")
    require(
        len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256),
        "efficiency table SHA-256 must be a lower-case hexadecimal digest",
    )
    require(sha256_file(path) == expected_sha256, "efficiency metadata table hash differs")
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise AggregateError(f"invalid efficiency metadata table: {exc}") from exc
    required = {
        "method_key", "ms_per_img_mean", "ms_per_img_seed_std",
        "params_m_mean", "params_m_seed_std",
    }
    require(required.issubset(frame.columns), "efficiency metadata table lacks required columns")
    require(frame.method_key.notna().all(), "efficiency metadata table has an empty method key")
    keys = frame.method_key.astype(str)
    require(not keys.duplicated().any(), "efficiency metadata method keys are not unique")
    expected = {key for key, meta in universe.items() if meta["status"] == "score"}
    require(expected.issubset(set(keys)), "efficiency metadata omits a scored Table-b method")
    allowed = set(universe) | {"all_background", "all_foreground"}
    require(set(keys).issubset(allowed), "efficiency metadata contains an unknown method")
    result: Dict[str, Dict[str, float]] = {}
    for record in frame.to_dict("records"):
        method = str(record["method_key"])
        values: Dict[str, float] = {}
        for column in (
            "ms_per_img_mean", "ms_per_img_seed_std", "params_m_mean", "params_m_seed_std"
        ):
            value = float(record[column]) if pd.notna(record[column]) else float("nan")
            require(not math.isfinite(value) or value >= 0.0, f"negative efficiency value: {method}")
            values[column] = value
        if method in expected:
            require(
                math.isfinite(values["ms_per_img_mean"])
                and math.isfinite(values["params_m_mean"]),
                f"efficiency metadata is incomplete for {method}",
            )
        result[method] = values
    return result, _identity(path)


def build_tables(
    universe: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, Sequence[tuple[int | None, pd.DataFrame]]],
    references: Mapping[str, pd.DataFrame],
) -> tuple[list[Dict[str, object]], list[Dict[str, object]], list[Dict[str, object]], Dict[str, pd.DataFrame]]:
    canonical = {
        method: _canonical_method_frame(method, runs) for method, runs in frames.items()
    }
    base = next(iter(canonical.values()))
    capture_specs = (("Plain (segmented slap)", "slap"), ("Rolled", "roll"))
    capture_tables: Dict[str, list[Dict[str, object]]] = {label: [] for label, _ in capture_specs}
    for label, source_capture in capture_specs:
        selector = base.capture.astype(str) == source_capture
        n_images = int(selector.sum())
        n_subjects = int(base.loc[selector, "subject_id"].nunique())
        for method, meta in universe.items():
            if meta["status"] == "N/A":
                capture_tables[label].append(
                    _na_row(
                        method, meta, capture=label, device="all",
                        n_images=n_images, n_subjects=n_subjects,
                    )
                )
                continue
            part = canonical[method][canonical[method].capture.astype(str) == source_capture]
            capture_tables[label].append(
                _summary_row(
                    method, meta, frames[method], canonical[method], part,
                    capture=label, device="all", aggregate="capture", value=source_capture,
                )
            )
        for method in REFERENCE_METHODS:
            part = references[method][references[method].capture.astype(str) == source_capture]
            capture_tables[label].append(
                _summary_row(
                    method, _reference_meta(method), [(None, references[method])],
                    references[method], part, capture=label, device="all",
                    aggregate="capture", value=source_capture,
                )
            )

    device_rows: list[Dict[str, object]] = []
    for device in sorted(set(base.device.astype(str))):
        device_selector = base.device.astype(str) == device
        capture_values = set(base.loc[device_selector, "report_capture"].astype(str))
        require(len(capture_values) == 1, f"device {device} mixes capture semantics")
        capture_label = capture_values.pop()
        n_images = int(device_selector.sum())
        n_subjects = int(base.loc[device_selector, "subject_id"].nunique())
        for method, meta in universe.items():
            if meta["status"] == "N/A":
                device_rows.append(
                    _na_row(
                        method, meta, capture=capture_label, device=device,
                        n_images=n_images, n_subjects=n_subjects,
                    )
                )
                continue
            part = canonical[method][canonical[method].device.astype(str) == device]
            device_rows.append(
                _summary_row(
                    method, meta, frames[method], canonical[method], part,
                    capture=capture_label, device=device, aggregate="device", value=device,
                )
            )
        for method in REFERENCE_METHODS:
            part = references[method][references[method].device.astype(str) == device]
            device_rows.append(
                _summary_row(
                    method, _reference_meta(method), [(None, references[method])],
                    references[method], part, capture=capture_label, device=device,
                    aggregate="device", value=device,
                )
            )
    return (
        capture_tables["Plain (segmented slap)"],
        capture_tables["Rolled"],
        device_rows,
        canonical,
    )


def _legacy_prediction_digest(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["image_id"])):
        digest.update(f"{record['image_id']}:{record['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _mask_candidates(receipt_path: Path, method: str, fallback: Path | None) -> list[Path]:
    candidates = [receipt_path.parent / "masks"]
    if receipt_path.parent.name == "score":
        candidates.append(receipt_path.parent.parent / "masks")
    if fallback is not None:
        candidates.append(fallback / method)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def authenticate_mask_sets(
    adoption_receipts: Mapping[str, Mapping[str, Any]],
    adoption_paths: Mapping[str, Path],
    all_test_ids: set[str],
    fallback_root: Path | None,
) -> tuple[Dict[str, Dict[str, Path]] | None, Mapping[str, Any]]:
    result: Dict[str, Dict[str, Path]] = {}
    identities: Dict[str, Any] = {}
    missing: list[str] = []
    for method, receipt in adoption_receipts.items():
        existing = [
            path for path in _mask_candidates(adoption_paths[method], method, fallback_root)
            if path.is_dir() and not path.is_symlink()
        ]
        if not existing:
            missing.append(method)
            continue
        authenticated: list[tuple[Path, list[dict[str, object]], Dict[str, Path]]] = []
        for root in existing:
            files = sorted(root.glob("*.png"))
            by_id = {path.stem: path for path in files}
            if len(by_id) != len(files) or set(by_id) != all_test_ids:
                continue
            records = [
                {
                    "image_id": image_id,
                    "sha256": sha256_file(by_id[image_id]),
                    "bytes": by_id[image_id].stat().st_size,
                }
                for image_id in sorted(all_test_ids)
            ]
            frozen = receipt.get("frozen_prediction", {})
            canonical_digest = canonical_sha256(records)
            legacy_digest = _legacy_prediction_digest(records)
            # Two authenticated producers exist in the locked B1 chain.  The
            # older adopter stores the line-oriented digest as
            # ``prediction_set_sha256`` and the canonical record digest as
            # ``prediction_records_sha256``.  The newer cross-domain finalizer
            # stores the canonical record digest in both fields.  Accept only
            # those two explicit, self-identifying contracts.
            expected_set = frozen.get("prediction_set_sha256")
            expected_records = frozen.get("prediction_records_sha256")
            set_digest_matches = (
                canonical_digest == expected_set
                if expected_set == expected_records
                else legacy_digest == expected_set
            )
            if (
                set_digest_matches and canonical_digest == expected_records
            ):
                authenticated.append((root, records, by_id))
        require(
            len(authenticated) <= 1,
            f"{method}: multiple distinct retained-mask roots authenticate; specify a narrower root",
        )
        if not authenticated:
            raise AggregateError(
                f"{method}: retained masks exist but none authenticate against the adoption receipt"
            )
        root, records, by_id = authenticated[0]
        result[method] = by_id
        identities[method] = {
            "root": str(root),
            "count": len(records),
            "prediction_set_sha256": receipt["frozen_prediction"]["prediction_set_sha256"],
            "prediction_records_sha256": canonical_sha256(records),
            "prediction_set_identity_scheme": (
                "canonical_records"
                if receipt["frozen_prediction"]["prediction_set_sha256"]
                == receipt["frozen_prediction"]["prediction_records_sha256"]
                else "legacy_line_digest_plus_canonical_records"
            ),
        }
    if missing:
        return None, {
            "status": "PENDING_RETAINED_MASKS_UNAVAILABLE",
            "reason": f"complete authenticated retained masks unavailable for: {sorted(missing)}",
            "interpretation": "agreement, not accuracy",
        }
    return result, {
        "status": "AUTHENTICATED",
        "methods": identities,
        "interpretation": "agreement, not accuracy",
    }


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(left.sum()) + int(right.sum())
    return 1.0 if denominator == 0 else 2.0 * int(np.count_nonzero(left & right)) / denominator


def compute_agreement(
    universe: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, Sequence[tuple[int | None, pd.DataFrame]]],
    adoption_receipts: Mapping[str, Mapping[str, Any]],
    mask_sets: Mapping[str, Mapping[str, Path]],
    mapped: Sequence[Mapping[str, str]],
) -> tuple[list[Dict[str, object]], list[Dict[str, object]], list[Dict[str, object]]]:
    adoption_by_table: Dict[str, list[str]] = defaultdict(list)
    for adoption, receipt in adoption_receipts.items():
        table_method = str(receipt.get("table_method", ""))
        if not table_method:
            matches = [
                method for method, runs in frames.items()
                if any(str(frame.method.iloc[0]) == adoption for _, frame in runs)
            ]
            require(len(matches) == 1, f"{adoption}: cannot resolve table method for agreement")
            table_method = matches[0]
        adoption_by_table[table_method].append(adoption)
    methods = [method for method, meta in universe.items() if meta["status"] == "score"]
    require(set(adoption_by_table) == set(methods), "agreement method universe differs")

    per_image: list[Dict[str, object]] = []
    pair_records: list[Dict[str, object]] = []
    for map_row in sorted(mapped, key=lambda item: item["image_id"]):
        image_id = map_row["image_id"]
        shape = (int(map_row["sd302b_height"]), int(map_row["sd302b_width"]))
        consensus: Dict[str, np.ndarray] = {}
        for method in methods:
            adoptions = adoption_by_table[method]
            masks = [_read_mask(mask_sets[adoption][image_id], shape) for adoption in adoptions]
            meta = universe[method]
            if meta["seed_regime"] == "seeded":
                require(len(masks) == 3, f"{method}: consensus requires three seed masks")
                consensus[method] = np.stack(masks, axis=0).sum(axis=0) >= 2
            else:
                require(len(masks) == 1, f"{method}: unseeded consensus requires one mask")
                consensus[method] = masks[0]
        values: list[float] = []
        for left, right in itertools.combinations(methods, 2):
            value = _dice(consensus[left], consensus[right])
            values.append(value)
            pair_records.append(
                {
                    "image_id": image_id,
                    "subject_id": map_row["subject_id"],
                    "capture": map_row["report_capture"],
                    "device": map_row["device"],
                    "method_a": left,
                    "method_b": right,
                    "dice": value,
                }
            )
        per_image.append(
            {
                "image_id": image_id,
                "subject_id": map_row["subject_id"],
                "capture": map_row["report_capture"],
                "device": map_row["device"],
                "n_methods": len(methods),
                "n_pairs": len(values),
                "mean_pairwise_dice": float(np.mean(values)),
                "interpretation": "agreement, not accuracy",
            }
        )
    per_frame = pd.DataFrame(per_image)
    summary: list[Dict[str, object]] = []
    selectors: list[tuple[str, str, pd.DataFrame]] = []
    for capture in ("Plain (segmented slap)", "Rolled"):
        selectors.append(("capture", capture, per_frame[per_frame.capture == capture]))
    for device in sorted(set(per_frame.device.astype(str))):
        selectors.append(("device", device, per_frame[per_frame.device == device]))
    for aggregate, value, part in selectors:
        interval, status, seed = _cluster_intervals(
            part.rename(columns={"mean_pairwise_dice": "agreement"}),
            ("agreement",), aggregate, value,
        )
        summary.append(
            {
                "aggregate": aggregate,
                "value": value,
                "n_images": len(part),
                "n_subjects": int(part.subject_id.nunique()),
                "n_methods": int(part.n_methods.iloc[0]),
                "n_pairs": int(part.n_pairs.iloc[0]),
                "mean_pairwise_dice": float(part.mean_pairwise_dice.mean()),
                "ci_low": interval["agreement"][0],
                "ci_high": interval["agreement"][1],
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_unit": "subject",
                "bootstrap_status": status,
                "interpretation": "agreement, not accuracy",
            }
        )
    pair_frame = pd.DataFrame(pair_records)
    pair_summary = (
        pair_frame.groupby(["capture", "method_a", "method_b"], sort=True, as_index=False)
        .agg(n_images=("dice", "size"), mean_dice=("dice", "mean"))
        .to_dict("records")
    )
    for row in pair_summary:
        row["interpretation"] = "agreement, not accuracy"
    return per_image, summary, pair_summary


def _private_per_image_rows(
    frames: Mapping[str, Sequence[tuple[int | None, pd.DataFrame]]]
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for table_method, runs in frames.items():
        for seed, frame in runs:
            adoption_method = str(frame.method.iloc[0])
            for record in frame.to_dict("records"):
                rows.append(
                    {
                        "table_method": table_method,
                        "adoption_method": adoption_method,
                        "seed": "unseeded" if seed is None else seed,
                        "image_id": record["image_id"],
                        "subject_id": record["subject_id"],
                        "capture": record["report_capture"],
                        "device": record["device"],
                        "mc": record["mc"],
                        "hr": record["hr"],
                        "par": record["par"],
                        "ohf": record["ohf"],
                        "predicted_area_fraction": record["predicted_area_fraction"],
                    }
                )
    return rows


def _fmt(value: object, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{100 * number:.2f}" if percent else f"{number:.3f}"


def _markdown_table(rows: Sequence[Mapping[str, object]], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Hull GT is a lower bound of the true foreground; pixel Dice is not reported. "
        "All non-reference models were applied cross-domain; latent-trained rows therefore "
        "measure latent-to-live-scan generalisation. Params and latency are hash-pinned "
        "method metadata copied from the locked latent public aggregate; latency was not "
        "remeasured on these live-scan images.",
        "",
        "| Method | Family | Training | Params (M) | ms/img | Status | Seeds | N | MC % (95% CI) | HR % (95% CI) | PAR | OHF % (95% CI) | Pred. area % (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(
                f"| {row['display_name']} | {row['family']} | {row['training_data']} | "
                f"{_fmt(row['params_m_mean'])} | {_fmt(row['ms_per_img_mean'])} | "
                f"N/A: {row['reason']} | 0 | {row['n_images']} | — | — | — | — | — |"
            )
            continue
        lines.append(
            "| {display_name} | {family} | {training_data} | {params} | {latency} | "
            "ok | {n_seeds} | {n_images} | {mc} ({mc_l}, {mc_h}) | "
            "{hr} ({hr_l}, {hr_h}) | {par} | {ohf} ({ohf_l}, {ohf_h}) | "
            "{paf} ({paf_l}, {paf_h}) |".format(
                display_name=row["display_name"], family=row["family"],
                training_data=row["training_data"], params=_fmt(row["params_m_mean"]),
                latency=_fmt(row["ms_per_img_mean"]), n_seeds=row["n_seeds"],
                n_images=row["n_images"], mc=_fmt(row["mc_mean"], percent=True),
                mc_l=_fmt(row["mc_ci_low"], percent=True),
                mc_h=_fmt(row["mc_ci_high"], percent=True),
                hr=_fmt(row["hr_mean"], percent=True),
                hr_l=_fmt(row["hr_ci_low"], percent=True),
                hr_h=_fmt(row["hr_ci_high"], percent=True),
                par=_fmt(row["par_mean"]), ohf=_fmt(row["ohf_mean"], percent=True),
                ohf_l=_fmt(row["ohf_ci_low"], percent=True),
                ohf_h=_fmt(row["ohf_ci_high"], percent=True),
                paf=_fmt(row["predicted_area_fraction_mean"], percent=True),
                paf_l=_fmt(row["predicted_area_fraction_ci_low"], percent=True),
                paf_h=_fmt(row["predicted_area_fraction_ci_high"], percent=True),
            )
        )
    lines.extend([
        "",
        "MC: minutiae coverage; HR: hull recall; PAR: predicted area divided by hull area; "
        "OHF: fraction of predicted foreground outside the hull. CIs are 1,000-resample "
        "subject-cluster percentile intervals.",
        "",
    ])
    return "\n".join(lines)


def _caveats() -> str:
    return """# SD302b locked-evidence caveats

- No human pixel masks were used. The expert-minutiae convex hull is a lower bound of true fingerprint foreground; pixel Dice is not reported.
- SD302b documents operator-assisted rolls and 4-4-2 slaps. “Plain (segmented slap)” denotes individual `slap-segmented` images and is not represented as a separately documented plain-impression acquisition.
- The mapped endpoint contains 439 baseline test images: 430 rolls and 9 segmented slaps. The 36 SD302g challenger test records belong to SD302a and are excluded.
- No inference was rerun. The archived SD302b 500-ppi inputs were exhaustively pixel-equivalent to the corresponding embedded SD302g inputs at the canonical 500-ppi model canvas. Existing locked B1 outputs were therefore subset and aggregated as behavior-equivalent evidence.
- For native 1000-ppi SD302g records, archive equivalence was observed after Pillow LANCZOS resizing. This is exhaustive empirical equivalence, not an authenticated archive-creation recipe.
- Existing B1 counts were produced against the verified native-resolution hull after nearest-neighbour restoration of the 500-ppi prediction. Predicted-area fractions use the matching native canvas and are scale-equivalent to the retained 500-ppi masks.
- “Hull + 10 px dilation (500 ppi)” is a deterministic reference, not ground truth. It uses an elliptical radius of 10 pixels at 500 ppi, scaled to the verified native hull resolution.
- RA-RUnet with live-scan alignment has no authenticated aligned checkpoint and remains pending. The scored RUnet row is explicitly the no-alignment model.
- Inter-method Dice is labelled “agreement, not accuracy.” It is emitted only after every retained full-test mask set reauthenticates against its adoption receipt and uses the separately preregistered seed-consensus policy.
- Every scored trainable model was trained on latent images. These tables measure latent-to-live-scan generalisation; zero-shot rows retain their externally pretrained/default status.
"""


def aggregate(
    mapping_receipt_path: Path,
    b1_report_path: Path,
    verified_manifest: Path,
    policy_path: Path,
    policy_sha256: str,
    efficiency_table_path: Path,
    efficiency_table_sha256: str,
    output: Path,
    *,
    fallback_mask_root: Path | None = None,
    require_production: bool = True,
    expected_test_records: int = 439,
    expected_full_records: int = 2_194,
) -> Mapping[str, Any]:
    output = output.resolve()
    outside_repository(output)
    require(output.is_dir() and not output.is_symlink(), "mapping output directory is missing/symlink")
    # The policy is authenticated before any mapping or B1 metric output is opened.
    policy = authenticate_policy(policy_path, policy_sha256)
    mapping_receipt, _, mapped = authenticate_mapping(
        mapping_receipt_path,
        expected_test_records=expected_test_records,
        expected_full_records=expected_full_records,
    )
    verified_manifest = regular_file(verified_manifest, "verified SD302g manifest")
    require(
        sha256_file(verified_manifest)
        == mapping_receipt.get("inputs", {}).get("verified_manifest", {}).get("sha256"),
        "mapping and supplied verified manifest identities differ",
    )
    report, universe, frames, adoption_receipts, adoption_paths = authenticate_b1(
        b1_report_path, verified_manifest, mapped, require_production=require_production
    )
    efficiency, efficiency_identity = authenticate_efficiency_table(
        efficiency_table_path, efficiency_table_sha256, universe
    )
    universe = {
        method: {**meta, **efficiency.get(method, {})}
        for method, meta in universe.items()
    }
    require(
        report.get("identities", {}).get("split_sha256")
        == mapping_receipt.get("inputs", {}).get("frozen_split", {}).get("sha256"),
        "mapping and B1 report frozen-split identities differ",
    )

    references, hull_evidence_sha256 = build_reference_frames(verified_manifest, mapped)
    plain, rolled, devices, _ = build_tables(universe, frames, references)
    private_per_image = _private_per_image_rows(frames)
    _, all_test_ids = _verified_manifest_rows(verified_manifest)
    fallback = fallback_mask_root.resolve() if fallback_mask_root is not None else None
    mask_sets, mask_status = authenticate_mask_sets(
        adoption_receipts, adoption_paths, all_test_ids, fallback
    )
    agreement_private: list[Dict[str, object]] = []
    agreement_summary: list[Dict[str, object]] = []
    agreement_pairwise: list[Dict[str, object]] = []
    if mask_sets is not None:
        agreement_private, agreement_summary, agreement_pairwise = compute_agreement(
            universe, frames, adoption_receipts, mask_sets, mapped
        )
        mask_status = {**mask_status, "status": "COMPLETE", "policy_sha256": policy_sha256}

    output_names = (
        "table_b_plain_segmented_slap.csv",
        "table_b_rolled.csv",
        "table_b_by_device.csv",
        "table_b_plain_segmented_slap.md",
        "table_b_rolled.md",
        "per_image_locked_private.csv",
        "agreement_per_image_private.csv",
        "agreement_summary.csv",
        "agreement_pairwise.csv",
        "CAVEATS.md",
        "aggregation_receipt.json",
    )
    for name in output_names:
        require(not (output / name).exists(), f"refusing to overwrite aggregate output: {name}")
    staging = Path(tempfile.mkdtemp(prefix=".sd302b-aggregate-", dir=output.parent))
    try:
        _write_csv(staging / output_names[0], plain, TABLE_COLUMNS)
        _write_csv(staging / output_names[1], rolled, TABLE_COLUMNS)
        _write_csv(staging / output_names[2], devices, TABLE_COLUMNS)
        (staging / output_names[3]).write_text(
            _markdown_table(plain, "Table b — Plain (segmented slap)"), encoding="utf-8"
        )
        (staging / output_names[4]).write_text(
            _markdown_table(rolled, "Table b — Rolled"), encoding="utf-8"
        )
        private_columns = (
            "table_method", "adoption_method", "seed", "image_id", "subject_id",
            "capture", "device", "mc", "hr", "par", "ohf", "predicted_area_fraction",
        )
        _write_csv(staging / output_names[5], private_per_image, private_columns)
        agreement_private_columns = (
            "image_id", "subject_id", "capture", "device", "n_methods", "n_pairs",
            "mean_pairwise_dice", "interpretation",
        )
        _write_csv(staging / output_names[6], agreement_private, agreement_private_columns)
        agreement_summary_columns = (
            "aggregate", "value", "n_images", "n_subjects", "n_methods", "n_pairs",
            "mean_pairwise_dice", "ci_low", "ci_high", "bootstrap_seed",
            "bootstrap_replicates", "bootstrap_unit", "bootstrap_status", "interpretation",
        )
        _write_csv(staging / output_names[7], agreement_summary, agreement_summary_columns)
        agreement_pairwise_columns = (
            "capture", "method_a", "method_b", "n_images", "mean_dice", "interpretation",
        )
        _write_csv(staging / output_names[8], agreement_pairwise, agreement_pairwise_columns)
        (staging / output_names[9]).write_text(_caveats(), encoding="utf-8")

        adoption_evidence = {
            method: {
                "receipt": _identity(adoption_paths[method]),
                "per_image": _identity(
                    adoption_paths[method].parent / f"per_image_{method}_test.csv"
                ),
            }
            for method in sorted(adoption_paths)
        }
        produced = {}
        for name in output_names[:-1]:
            path = staging / name
            produced[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        receipt: Dict[str, Any] = {
            "schema_version": 1,
            "kind": "sd302b_locked_b1_aggregation_receipt_v1",
            "status": "LOCKED_EVIDENCE_AGGREGATED",
            "private_biometric_artifact": True,
            "no_model_inference_rerun": True,
            "no_human_ground_truth_used": True,
            "behavior_equivalence": {
                "claim": (
                    "locked B1 predictions were reused because every mapped SD302b archive "
                    "image was exactly equal at the canonical 500-ppi model input"
                ),
                "mapping_receipt_sha256": sha256_file(mapping_receipt_path),
                "transform_provenance": "observed_exhaustive_not_authenticated_archive_recipe",
            },
            "inputs": {
                "mapping_receipt": _identity(mapping_receipt_path.resolve()),
                "b1_report_receipt": _identity(b1_report_path.resolve()),
                "verified_manifest": _identity(verified_manifest),
                "consensus_policy": _identity(policy_path.resolve()),
                "consensus_policy_content_sha256": canonical_sha256(policy),
                "efficiency_metadata_table": efficiency_identity,
                "adoption_evidence": adoption_evidence,
                "verified_hull_minutiae_records_sha256": hull_evidence_sha256,
            },
            "subset": {
                "records": len(mapped),
                "subjects": len({row["subject_id"] for row in mapped}),
                "capture_counts": dict(sorted(Counter(row["report_capture"] for row in mapped).items())),
                "device_counts": dict(sorted(Counter(row["device"] for row in mapped).items())),
                "challenger_records_excluded": 36 if require_production else None,
            },
            "methods": {
                "scored": [key for key, meta in universe.items() if meta["status"] == "score"],
                "pending": {
                    key: (
                        RA_RUNET_ALIGNED_PENDING_REASON
                        if key == "ra_runet_aligned"
                        else meta["reason"]
                    )
                    for key, meta in universe.items()
                    if meta["status"] == "N/A"
                },
            },
            "statistics": {
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_unit": "subject",
                "interval": "percentile_95",
                "seed_aggregation": "per-image arithmetic mean before subject bootstrap",
            },
            "references": {
                "rows": list(REFERENCE_METHODS),
                "dilation": (
                    "elliptical radius 10 px at 500 ppi, scaled to verified native resolution"
                ),
                "interpretation": "deterministic context rows; not complete foreground ground truth",
            },
            "agreement": mask_status,
            "outputs": produced,
            "caveats_file": "CAVEATS.md",
        }
        _write_json(staging / output_names[-1], receipt)
        for name in output_names:
            os.replace(staging / name, output / name)
        return receipt
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    policy_parser = subparsers.add_parser(
        "write-policy", help="write the immutable agreement policy before metric access"
    )
    policy_parser.add_argument("--out", required=True, type=Path)
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="authenticate and aggregate the locked mapped evidence"
    )
    aggregate_parser.add_argument("--mapping-receipt", required=True, type=Path)
    aggregate_parser.add_argument("--b1-report-receipt", required=True, type=Path)
    aggregate_parser.add_argument("--verified-manifest", required=True, type=Path)
    aggregate_parser.add_argument("--consensus-policy", required=True, type=Path)
    aggregate_parser.add_argument("--consensus-policy-sha256", required=True)
    aggregate_parser.add_argument("--efficiency-table", required=True, type=Path)
    aggregate_parser.add_argument("--efficiency-table-sha256", required=True)
    aggregate_parser.add_argument("--fallback-mask-root", type=Path)
    aggregate_parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.command == "write-policy":
            result = write_policy(args.out)
        else:
            result = aggregate(
                args.mapping_receipt,
                args.b1_report_receipt,
                args.verified_manifest,
                args.consensus_policy,
                args.consensus_policy_sha256,
                args.efficiency_table,
                args.efficiency_table_sha256,
                args.out,
                fallback_mask_root=args.fallback_mask_root,
            )
    except (AggregateError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
