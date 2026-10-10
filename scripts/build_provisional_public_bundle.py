#!/usr/bin/env python3
"""Build a public-only provisional bundle from two authenticated aggregates.

This bridge is intentionally narrow. It combines the authenticated Dataset-1
aggregate snapshot with the separately authenticated, reference-verified B1
summary. It never reads per-image rows, biometric media, predictions, or model
weights. Missing B1 method results remain explicit pending rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_from_aggregate as update  # noqa: E402


SNAPSHOT_KIND = "aggregate_partial_publication_snapshot_receipt_v1"
B1_RECEIPT_KIND = "sd302g_b1_reference_verified_report"
B1_COLUMNS = (
    "method", "approach", "mc_percent", "mc_ci_low_percent",
    "mc_ci_high_percent", "hr_percent", "hr_ci_low_percent",
    "hr_ci_high_percent", "par", "ohf_percent", "params_m", "ms_per_img",
    "device", "n_images", "n_subjects", "verification_status",
)
B1_VERIFIED_METHODS = frozenset(
    {"gmfs_default", "sufs_zero", "all_background", "all_foreground", "hull_as_prediction"}
)
B1_ALL_METHODS = (
    "gmfs_default", "gmfs_tuned", "sufs_zero", "sufs_ft", "fingernet",
    "svm", "rf", "welm", "unet", "unetpp", "drunet", "r_unet", "lenet",
    "alexnet", "cnn_ae", "segformer_b2", "ra_runet_aligned",
    "all_background", "all_foreground", "hull_as_prediction",
)


class ProvisionalBundleError(ValueError):
    """Authenticated provisional inputs do not meet the frozen contract."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def regular_bytes(path: Path, label: str, *, limit: int = 5 * 1024 * 1024) -> bytes:
    data = update._read_regular(path, label)
    if len(data) > limit:
        raise ProvisionalBundleError(f"{label} exceeds size limit")
    return data


def receipt(path: Path, expected_sha256: str, label: str) -> Mapping[str, object]:
    data = regular_bytes(path, label)
    if sha256(data) != expected_sha256:
        raise ProvisionalBundleError(f"{label} digest mismatch")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvisionalBundleError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProvisionalBundleError(f"{label} must be an object")
    return value


def validate_snapshot_receipt(
    value: Mapping[str, object], bundle: update.AggregateBundle
) -> None:
    if (
        value.get("schema_version") != 1
        or value.get("kind") != SNAPSHOT_KIND
        or value.get("status") != "PASS"
        or value.get("snapshot_status") != "partial"
        or value.get("pending_methods") != ["segformer_sub"]
        or value.get("per_image_metrics_opened") is not False
        or value.get("scoring_invoked") is not False
    ):
        raise ProvisionalBundleError("Dataset-1 snapshot receipt contract mismatch")
    outputs = value.get("outputs")
    if not isinstance(outputs, dict) or canonical_sha256(outputs) != value.get("outputs_sha256"):
        raise ProvisionalBundleError("Dataset-1 snapshot output inventory mismatch")
    for name, digest in bundle.file_sha256.items():
        if outputs.get(f"aggregate_bundle/{name}") != digest:
            raise ProvisionalBundleError(f"Dataset-1 snapshot does not bind {name}")
    table_a = [
        row for row in bundle.results
        if row["table"] == "A" and row["view"] == "overall" and row["group"] == "overall"
    ]
    pending = {row["method"] for row in table_a if row["status"] == "pending"}
    if len(table_a) != 18 or pending != {"segformer_b2"}:
        raise ProvisionalBundleError("Dataset-1 snapshot has an unexpected pending universe")
    segformer = next(row for row in table_a if row["method"] == "segformer_b2")
    if segformer["seeds_completed"] != "2":
        raise ProvisionalBundleError("SegFormer provisional row must contain exactly two seeds")


def read_b1(path: Path, receipt_value: Mapping[str, object]) -> list[dict[str, str]]:
    data = regular_bytes(path, "B1 aggregate table")
    outputs = receipt_value.get("outputs")
    if (
        receipt_value.get("schema_version") != 1
        or receipt_value.get("kind") != B1_RECEIPT_KIND
        or receipt_value.get("status") != "REFERENCE_VERIFIED"
        or receipt_value.get("n_images") != 475
        or receipt_value.get("n_subjects") != 40
        or receipt_value.get("methods") != ["gmfs_default", "sufs_zero"]
        or not isinstance(outputs, dict)
        or outputs.get("table_b1_sha256") != sha256(data)
    ):
        raise ProvisionalBundleError("B1 reference receipt contract mismatch")
    text = update._decode_utf8(data, "B1 aggregate table")
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != B1_COLUMNS:
        raise ProvisionalBundleError("B1 aggregate table schema mismatch")
    rows = list(reader)
    if {row["method"] for row in rows} != B1_VERIFIED_METHODS or len(rows) != 5:
        raise ProvisionalBundleError("B1 aggregate table method universe mismatch")
    for row in rows:
        if (
            row["verification_status"] != "REFERENCE_VERIFIED"
            or row["n_images"] != "475"
            or row["n_subjects"] != "40"
        ):
            raise ProvisionalBundleError("B1 aggregate row is not reference-verified")
        for key in (
            "mc_percent", "mc_ci_low_percent", "mc_ci_high_percent",
            "hr_percent", "hr_ci_low_percent", "hr_ci_high_percent", "ohf_percent",
        ):
            number = float(row[key])
            if not 0.0 <= number <= 100.0:
                raise ProvisionalBundleError(f"B1 {key} is outside [0,100]")
        if float(row["par"]) < 0:
            raise ProvisionalBundleError("B1 PAR is negative")
    return rows


def blank_public_row() -> dict[str, str]:
    return {column: "" for column in update.RESULT_COLUMNS}


def reference_metadata(method: str) -> dict[str, str]:
    if method == "hull_as_prediction":
        return {
            "year": "", "approach": "Hull as prediction", "family": "Reference",
            "training_data": "None", "comment": "Lower-bound reference; not a method",
        }
    if method == "ra_runet_aligned":
        return {
            "year": "2021", "approach": "RA-RUnet (aligned)",
            "family": "Encoder–decoder",
            "training_data": "Labelled latent train subjects plus unlabelled live-scan train subjects",
            "comment": "Adversarial alignment evaluation pending",
        }
    raise ProvisionalBundleError(f"no standalone metadata for {method}")


def to_fraction(value: str) -> str:
    return str(float(value) / 100.0)


def build_b1_rows(
    base: update.AggregateBundle, verified_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    table_a = {
        row["method"]: row for row in base.results
        if row["table"] == "A" and row["view"] == "overall" and row["group"] == "overall"
    }
    verified = {row["method"]: row for row in verified_rows}
    output: list[dict[str, str]] = []
    for method in B1_ALL_METHODS:
        metadata = (
            {key: table_a[method][key] for key in (
                "year", "approach", "family", "training_data", "comment",
            )}
            if method in table_a
            else reference_metadata(method)
        )
        row = blank_public_row()
        row.update(
            {
                "table": "B1", "view": "overall", "group": "overall",
                "method": method, **metadata,
            }
        )
        source = verified.get(method)
        if source is None:
            row.update(
                {
                    "status": "pending", "seeds_completed": "0",
                    "comment": metadata["comment"] + "; cross-domain scoring pending",
                }
            )
        else:
            row.update(
                {
                    "status": "complete", "seeds_completed": "1",
                    "n_images": source["n_images"], "n_subjects": source["n_subjects"],
                    "mc": to_fraction(source["mc_percent"]),
                    "hr": to_fraction(source["hr_percent"]),
                    "par": source["par"], "ohf": to_fraction(source["ohf_percent"]),
                    "params_m": "" if source["params_m"] == "-" else source["params_m"],
                    "ms_per_img": "" if source["ms_per_img"] == "-" else source["ms_per_img"],
                    "device": "" if source["device"] == "-" else source["device"],
                    "ci_lower": to_fraction(source["hr_ci_low_percent"]),
                    "ci_upper": to_fraction(source["hr_ci_high_percent"]),
                    "comment": metadata["comment"] + "; reference-verified hull aggregate",
                }
            )
        output.append(row)
    return output


def write_csv(path: Path, columns: tuple[str, ...], rows: list[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--snapshot-receipt", type=Path, required=True)
    parser.add_argument("--expected-snapshot-receipt-sha256", required=True)
    parser.add_argument("--b1-table", type=Path, required=True)
    parser.add_argument("--b1-receipt", type=Path, required=True)
    parser.add_argument("--expected-b1-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ProvisionalBundleError("output bundle already exists")

    base = update.validate_bundle(args.base_bundle)
    snapshot = receipt(
        args.snapshot_receipt, args.expected_snapshot_receipt_sha256,
        "Dataset-1 snapshot receipt",
    )
    validate_snapshot_receipt(snapshot, base)
    b1_receipt = receipt(args.b1_receipt, args.expected_b1_receipt_sha256, "B1 receipt")
    b1_rows = read_b1(args.b1_table, b1_receipt)

    results = [dict(row) for row in base.results if row["table"] != "B1"]
    for row in results:
        if row["method"] == "segformer_b2" and row["status"] == "pending":
            row["comment"] = row["comment"] + "; training — pending"
    results.extend(build_b1_rows(base, b1_rows))
    tables = [dict(row) for row in base.tables]
    for row in tables:
        if row["table"] == "B1":
            row["status"] = "pending"
            row["note"] = (
                "Two methods and three references are reference-verified; "
                "all other B1 rows remain pending."
            )

    args.out.mkdir(parents=True)
    write_csv(args.out / "public_results.csv", update.RESULT_COLUMNS, results)
    write_csv(args.out / "statistics.csv", update.STATISTIC_COLUMNS, list(base.statistics))
    write_csv(args.out / "table_status.csv", update.TABLE_COLUMNS, tables)
    files = {
        name: sha256((args.out / name).read_bytes()) for name in update.EXPECTED_FILES
    }
    manifest = {"schema_version": 1, "kind": update.BUNDLE_KIND, "files": files}
    (args.out / update.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validated = update.validate_bundle(args.out)
    pending_a = sorted(
        row["method"] for row in validated.results
        if row["table"] == "A" and row["status"] == "pending"
    )
    complete_b1 = sorted(
        row["method"] for row in validated.results
        if row["table"] == "B1" and row["status"] == "complete"
    )
    pending_b1 = sorted(
        row["method"] for row in validated.results
        if row["table"] == "B1" and row["status"] == "pending"
    )
    if pending_a != ["segformer_b2"] or complete_b1 != sorted(B1_VERIFIED_METHODS) or len(pending_b1) != 15:
        raise ProvisionalBundleError("derived provisional status universe mismatch")
    public_receipt = {
        "schema_version": 1,
        "kind": "provisional_public_bundle_receipt_v1",
        "status": "PASS",
        "source_snapshot_receipt_sha256": args.expected_snapshot_receipt_sha256,
        "source_snapshot_outputs_sha256": snapshot["outputs_sha256"],
        "source_b1_receipt_sha256": args.expected_b1_receipt_sha256,
        "source_b1_table_sha256": b1_receipt["outputs"]["table_b1_sha256"],
        "output_manifest_sha256": sha256((args.out / update.MANIFEST_NAME).read_bytes()),
        "output_files": files,
        "pending_table_a_methods": pending_a,
        "complete_b1_rows": complete_b1,
        "pending_b1_rows": pending_b1,
        "contains_per_image_rows": False,
        "contains_biometric_media": False,
        "transformations": [
            "preserved authenticated Dataset-1 aggregate values",
            "labelled pending SegFormer rows as training pending",
            "converted reference-verified B1 percentages to fractions",
            "added explicit pending rows for unscored B1 methods",
        ],
    }
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(
        json.dumps(public_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(public_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
