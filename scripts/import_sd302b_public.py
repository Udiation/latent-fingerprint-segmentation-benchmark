#!/usr/bin/env python3
"""Publish receipt-bound aggregate SD302b hull tables without private rows.

The source tables contain one aggregate row per method and capture type.  This
bridge verifies their frozen aggregation receipt, converts them to the public
result schema, and writes a second receipt that explicitly excludes per-image
identifiers and biometric media.  It never reads an image, mask, or checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "results" / "public_results.csv"

YEARS = {
    "gmfs_default": "2023", "gmfs_tuned": "2023", "sufs_zero": "2023",
    "sufs_ft": "2023", "fingernet": "2017", "svm": "1995", "rf": "2017",
    "welm": "2016", "unet": "2022", "unetpp": "2020", "drunet": "",
    "r_unet": "2021", "ra_runet_aligned": "2021", "lenet": "2019",
    "alexnet": "2019", "cnn_ae": "2023", "segformer_b2": "2021",
}
CPU_METHODS = {
    "gmfs_default", "gmfs_tuned", "svm", "rf", "welm",
    "all_background", "all_foreground", "hull_as_prediction",
    "hull_dilated_10px_500ppi",
}


class ImportError(ValueError):
    """The aggregate-only input cannot support a public release."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _identities(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, dict):
        if "sha256" in value and "bytes" in value:
            yield value
        for child in value.values():
            yield from _identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from _identities(child)


def _verify_source(path: Path, receipt: Mapping[str, Any]) -> None:
    digest = sha256_file(path)
    size = path.stat().st_size
    matches = [
        row for row in _identities(receipt)
        if row.get("sha256") == digest and int(row.get("bytes", -1)) == size
    ]
    if len(matches) != 1:
        raise ImportError(f"source table is not uniquely bound by aggregation receipt: {path.name}")


def _clean(value: str) -> str:
    if value == "" or value.lower() == "nan":
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return repr(number)


def _convert(
    rows: list[dict[str, str]], columns: list[str], *, table: str
) -> list[dict[str, str]]:
    required = {
        "method", "display_name", "family", "training_data", "status", "reason",
        "n_images", "n_subjects", "n_seeds", "mc_mean", "mc_seed_std",
        "hr_mean", "hr_seed_std", "hr_ci_low", "hr_ci_high", "par_mean",
        "par_seed_std", "ohf_mean", "ohf_seed_std", "params_m_mean",
        "params_m_seed_std", "ms_per_img_mean", "ms_per_img_seed_std",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else ()))
        raise ImportError(f"aggregate source schema is incomplete: {missing}")
    forbidden = {"image_id", "subject_id", "image_path", "mask_path", "filename"}
    if forbidden.intersection(rows[0]):
        raise ImportError("aggregate source unexpectedly contains private per-image fields")
    output: list[dict[str, str]] = []
    for source in rows:
        method = source["method"]
        status = "complete" if source["status"].lower() == "ok" else "n/a"
        row = {column: "" for column in columns}
        row.update({
            "table": table,
            "view": "overall",
            "group": "overall",
            "year": YEARS.get(method, ""),
            "method": method,
            "approach": source["display_name"],
            "family": source["family"],
            "training_data": source["training_data"],
            "comment": (
                source["reason"] if status == "n/a" else
                "latent-to-live-scan generalisation; expert-minutiae hull is a lower bound; pixel Dice not reported"
            ),
            "status": status,
            "seeds_completed": source["n_seeds"] if status == "complete" else "0",
            "n_images": source["n_images"],
            "n_subjects": source["n_subjects"],
            "mc": _clean(source["mc_mean"]),
            "mc_seed_std": _clean(source["mc_seed_std"]),
            "hr": _clean(source["hr_mean"]),
            "hr_seed_std": _clean(source["hr_seed_std"]),
            "par": _clean(source["par_mean"]),
            "par_seed_std": _clean(source["par_seed_std"]),
            "ohf": _clean(source["ohf_mean"]),
            "ohf_seed_std": _clean(source["ohf_seed_std"]),
            "params_m": _clean(source["params_m_mean"]),
            "params_m_seed_std": _clean(source["params_m_seed_std"]),
            "ms_per_img": _clean(source["ms_per_img_mean"]),
            "ms_per_img_seed_std": _clean(source["ms_per_img_seed_std"]),
            "device": "CPU" if method in CPU_METHODS else "GPU",
            "ci_lower": _clean(source["hr_ci_low"]),
            "ci_upper": _clean(source["hr_ci_high"]),
        })
        output.append(row)
    return output


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _identity(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolled", type=Path, required=True)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--aggregation-receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--schema-template", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--out-results", type=Path, required=True)
    parser.add_argument("--out-status", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    args = parser.parse_args()

    receipt_bytes = args.aggregation_receipt.read_bytes()
    observed_receipt = hashlib.sha256(receipt_bytes).hexdigest()
    if observed_receipt != args.expected_receipt_sha256:
        raise ImportError("aggregation receipt differs from independently supplied digest")
    receipt = json.loads(receipt_bytes)
    if (
        receipt.get("kind") != "sd302b_locked_b1_aggregation_receipt_v1"
        or receipt.get("status") != "LOCKED_EVIDENCE_AGGREGATED"
    ):
        raise ImportError("aggregation receipt is not complete")
    _verify_source(args.rolled, receipt)
    _verify_source(args.plain, receipt)

    columns, schema_rows = _read_csv(args.schema_template)
    if not columns or len(columns) != len(set(columns)):
        raise ImportError("public schema template is invalid")
    _, rolled = _read_csv(args.rolled)
    _, plain = _read_csv(args.plain)
    public = _convert(rolled, columns, table="B_rolled")
    public.extend(_convert(plain, columns, table="B_plain"))
    if any(row["status"] == "complete" and row["method"] == "ra_runet_aligned" for row in public):
        raise ImportError("aligned RA-RUnet cannot be published without an authenticated checkpoint")
    # The canonical Stage-5 bundle froze an obsolete data-transfer explanation
    # for the unavailable aligned row.  The later, receipt-bound SD302b
    # aggregation proves that the target data existed and that only an
    # authenticated aligned checkpoint was absent.  Publish one explicit row
    # amendment while retaining the canonical CSV unchanged for provenance.
    source_ra = [row for row in rolled if row["method"] == "ra_runet_aligned"]
    base_ra = [
        row for row in schema_rows
        if row.get("table") == "B1" and row.get("method") == "ra_runet_aligned"
    ]
    if source_ra:
        if len(source_ra) != 1 or len(base_ra) != 1:
            raise ImportError("RA-RUnet amendment inputs are ambiguous")
        amended = dict(base_ra[0])
        amended["comment"] = source_ra[0]["reason"]
        amended["status"] = "n/a"
        amended["seeds_completed"] = "0"
        public.append(amended)

    with tempfile.TemporaryDirectory(prefix="sd302b-public-") as temporary:
        temp = Path(temporary)
        results = temp / args.out_results.name
        statuses = temp / args.out_status.name
        _write_csv(results, columns, public)
        status_columns = ["table", "title", "status", "note"]
        status_rows = [
            {
                "table": "B1",
                "title": "Table B1 — SD302-Exemplar lower-bound evaluation",
                "status": "complete",
                "note": "Expert-minutiae hull lower-bound endpoint. RA-RUnet alignment is pending because no authenticated aligned checkpoint completed before the benchmark freeze; the no-alignment RUnet is not substituted.",
            },
            {
                "table": "B_rolled",
                "title": "Table b — SD302b rolled baseline captures",
                "status": "complete",
                "note": "430 exactly mapped test images from 40 subjects; expert-minutiae hull lower bound; latent-trained models are applied cross-domain; pixel Dice is not reported.",
            },
            {
                "table": "B_plain",
                "title": "Table b — SD302b plain (segmented-slap) baseline captures",
                "status": "complete",
                "note": "9 exactly mapped segmented-slap test images; expert-minutiae hull lower bound; labelled plain with this capture-semantic qualification; pixel Dice is not reported.",
            },
        ]
        _write_csv(statuses, status_columns, status_rows)
        args.out_results.parent.mkdir(parents=True, exist_ok=True)
        args.out_status.parent.mkdir(parents=True, exist_ok=True)
        os.replace(results, args.out_results)
        os.replace(statuses, args.out_status)

    public_receipt = {
        "schema_version": 1,
        "kind": "sd302b.aggregate-public-release.v1",
        "status": "PASS",
        "source_aggregation_receipt_sha256": observed_receipt,
        "contains_per_image_rows": False,
        "contains_biometric_media": False,
        "tables": {
            "B_rolled": len(rolled),
            "B_plain": len(plain),
            "B1_amendments": len([row for row in public if row["table"] == "B1"]),
        },
        "outputs": {
            args.out_results.name: _identity(args.out_results),
            args.out_status.name: _identity(args.out_status),
        },
    }
    args.out_receipt.write_text(
        json.dumps(public_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(public_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
