from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _identity(path: Path) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def test_aggregate_bridge_emits_no_private_fields(tmp_path: Path) -> None:
    schema = ROOT / "results" / "public_results.csv"
    with schema.open(newline="", encoding="utf-8") as handle:
        columns = list(csv.DictReader(handle).fieldnames or ())
    source_columns = [
        "method", "display_name", "family", "training_data", "status", "reason",
        "capture", "device", "n_images", "n_subjects", "n_seeds", "seed_ids",
        "ms_per_img_mean", "ms_per_img_seed_std", "params_m_mean", "params_m_seed_std",
        "mc_mean", "mc_seed_std", "mc_ci_low", "mc_ci_high", "hr_mean",
        "hr_seed_std", "hr_ci_low", "hr_ci_high", "par_mean", "par_seed_std",
        "par_ci_low", "par_ci_high", "ohf_mean", "ohf_seed_std", "ohf_ci_low",
        "ohf_ci_high", "predicted_area_fraction_mean",
        "predicted_area_fraction_seed_std", "predicted_area_fraction_ci_low",
        "predicted_area_fraction_ci_high", "bootstrap_seed", "bootstrap_replicates",
        "bootstrap_unit", "bootstrap_status", "interpretation",
    ]
    base = {column: "" for column in source_columns}
    base.update({
        "method": "unet", "display_name": "U-Net", "family": "Encoder-decoder",
        "training_data": "latent train subjects", "status": "ok", "capture": "Rolled",
        "device": "all", "n_images": "4", "n_subjects": "2", "n_seeds": "3",
        "seed_ids": "0,1,2", "ms_per_img_mean": "12.5", "params_m_mean": "7.2",
        "mc_mean": "0.9", "mc_seed_std": "0.01", "hr_mean": "0.8",
        "hr_seed_std": "0.02", "hr_ci_low": "0.7", "hr_ci_high": "0.9",
        "par_mean": "1.4", "par_seed_std": "0.1", "ohf_mean": "0.3",
        "ohf_seed_std": "0.02",
    })
    rolled = tmp_path / "rolled.csv"
    plain = tmp_path / "plain.csv"
    _write_csv(rolled, source_columns, [base])
    plain_row = dict(base, capture="Plain (segmented slap)")
    _write_csv(plain, source_columns, [plain_row])
    receipt = {
        "kind": "sd302b_locked_b1_aggregation_receipt_v1",
        "status": "LOCKED_EVIDENCE_AGGREGATED",
        "outputs": {"rolled": _identity(rolled), "plain": _identity(plain)},
    }
    receipt_path = tmp_path / "aggregation.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    out_results = tmp_path / "sd302b_results.csv"
    out_status = tmp_path / "sd302b_table_status.csv"
    out_receipt = tmp_path / "sd302b_release_receipt.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "import_sd302b_public.py"),
        "--rolled", str(rolled), "--plain", str(plain),
        "--aggregation-receipt", str(receipt_path),
        "--expected-receipt-sha256", receipt_hash,
        "--schema-template", str(schema), "--out-results", str(out_results),
        "--out-status", str(out_status), "--out-receipt", str(out_receipt),
    ], check=True, capture_output=True, text=True)
    with out_results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and list(rows[0]) == columns
    assert {row["table"] for row in rows} == {"B_rolled", "B_plain"}
    assert not {"image_id", "subject_id", "image_path", "mask_path"}.intersection(rows[0])
    assert rows[0]["dice"] == "" and rows[0]["hr"] == "0.8"
    public_receipt = json.loads(out_receipt.read_text(encoding="utf-8"))
    assert public_receipt["contains_per_image_rows"] is False
    assert public_receipt["contains_biometric_media"] is False
