from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from report.build_report import ReportBuildError, build_html, main, publish_once  # noqa: E402
from report.privacy import PublicReportError, scan_public_html  # noqa: E402


def _write_inputs(tmp_path: Path) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tables = pd.DataFrame([
        {
            "table": "A", "view": "overall", "group_by": "", "group_value": "overall",
            "method": "unet", "display_name": "U-Net", "year": 2022,
            "family": "Encoder-decoder", "training_data": "SD302-Latent train split",
            "comment": "threshold selected on validation", "n_seeds": 3,
            "n_images": 120, "n_missing": 0, "dice_mean": 0.81,
            "dice_seed_std": 0.01, "mdr_mean": 0.14, "mdr_seed_std": 0.01,
            "fdr_mean": 0.12, "fdr_seed_std": 0.02, "params_m": 7.2,
            "params_m_seed_std": 0.0, "ms_per_img": 12.0, "ms_per_img_seed_std": 0.4,
            "device": "GPU", "subset": "finger_latent", "split": "test",
            "ground_truth": "examiner ROI", "seed_ids": "0,1,2",
        },
        {
            "table": "A", "view": "overall", "group_by": "", "group_value": "overall",
            "method": "all_background", "display_name": "All background", "year": "",
            "family": "Reference", "training_data": "none", "comment": "reference",
            "n_seeds": 1, "n_images": 120, "n_missing": 0, "dice_mean": 0.0,
            "dice_seed_std": np.nan, "mdr_mean": 1.0, "mdr_seed_std": np.nan,
            "fdr_mean": 0.0, "fdr_seed_std": np.nan, "params_m": 0.0,
            "params_m_seed_std": np.nan, "ms_per_img": 0.0, "ms_per_img_seed_std": np.nan,
            "device": "CPU", "subset": "finger_latent", "split": "test",
            "ground_truth": "examiner ROI", "seed_ids": "unseeded",
        },
        {
            "table": "A", "view": "examiner_value", "group_by": "examiner_value",
            "group_value": "VALUE", "method": "unet", "display_name": "U-Net", "year": 2022,
            "family": "Encoder-decoder", "training_data": "SD302-Latent train split",
            "comment": "threshold selected on validation", "n_seeds": 3,
            "n_images": 50, "n_missing": 0, "dice_mean": 0.86,
            "dice_seed_std": 0.01, "mdr_mean": 0.1, "mdr_seed_std": 0.01,
            "fdr_mean": 0.08, "fdr_seed_std": 0.01, "params_m": 7.2,
            "params_m_seed_std": 0.0, "ms_per_img": 12.0, "ms_per_img_seed_std": 0.4,
            "device": "GPU", "subset": "finger_latent", "split": "test",
            "ground_truth": "examiner ROI", "seed_ids": "0,1,2",
        },
    ])
    prime = tables[tables.table == "A"].copy()
    prime["table"] = "A_prime"
    prime["ground_truth"] = "usable-ridge area"
    tables = pd.concat([tables, prime], ignore_index=True)
    tables_path = tmp_path / "tables.csv"
    tables.to_csv(tables_path, index=False)
    metadata_path = tmp_path / "table_metadata.json"
    metadata_path.write_text(json.dumps({"tables": [
        {
            "table": str(table), "view": str(view),
            "title": f"Table {table} — {view}",
            "caption": "Aggregate test results; thresholds were selected on validation data.",
        }
        for table, view in tables[["table", "view"]].drop_duplicates().itertuples(index=False)
    ]}), encoding="utf-8")
    dataset_path = tmp_path / "datasets.json"
    dataset_path.write_text(json.dumps({"datasets": [{
        "dataset": "SD302-Latent", "status": "complete", "ground_truth": "examiner ROI",
        "train_images": 600, "val_images": 100, "test_images": 120,
        "train_subjects": 12, "val_subjects": 3, "test_subjects": 4,
        "notes": "Aggregate counts only",
    }]}), encoding="utf-8")
    stats_path = tmp_path / "statistics.csv"
    pd.DataFrame([{
        "table": "A", "metric": "Dice", "comparison": "U-Net vs baseline",
        "reference_method": "U-Net", "method": "Baseline", "estimate": 0.04,
        "ci_lower": 0.02, "ci_upper": 0.06, "p_value": 0.001,
        "p_holm": 0.004, "significant": True, "n_images": 120,
        "n_subjects": 4, "status": "complete",
    }]).to_csv(stats_path, index=False)
    method_path = tmp_path / "methodology.md"
    method_path.write_text("# Methods\n\nAll selection used validation data only.\n", encoding="utf-8")
    repro_path = tmp_path / "repro.json"
    repro_path.write_text(json.dumps({
        "git_commit": "a" * 40, "config_hashes": ["b" * 64], "seeds": [0, 1, 2],
        "dates": ["2026-10-08"], "data_manifest_sha256": "c" * 64,
    }), encoding="utf-8")
    synthetic = tmp_path / "synthetic"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_demo.py"),
         "--out", str(synthetic), "--subjects", "1", "--per-subject", "1"],
        check=True, capture_output=True, text=True,
    )
    receipt = synthetic / "synthetic_receipt.json"
    return argparse.Namespace(
        tables_csv=tables_path, table_metadata_json=metadata_path,
        dataset_summary_json=dataset_path,
        statistics_csv=stats_path, methodology=method_path,
        reproducibility_json=repro_path, out=tmp_path / "report", public=True,
        synthetic_root=synthetic, synthetic_receipt=receipt, synthetic_limit=8,
        private_gallery=None, private_audit=None, scan_report=None,
    )


def test_public_report_is_self_contained_and_passes_independent_scan(tmp_path: Path) -> None:
    args = _write_inputs(tmp_path)
    html = build_html(args)
    index, action = publish_once(args.out, html)
    assert action == "created"
    result = scan_public_html(
        index, synthetic_receipt=args.synthetic_receipt, synthetic_root=args.synthetic_root,
        generator_path=ROOT / "scripts" / "make_synthetic_demo.py",
    )
    assert result["status"] == "PASS"
    assert result["synthetic_images"] == 3
    assert list(args.out.iterdir()) == [index]
    assert "SD302-Latent" in html
    assert "syn_finger" not in html
    assert ("/" + "home/") not in html
    assert "<script" not in html.lower()


def test_public_cli_writes_scan_outside_report_directory(tmp_path: Path, capsys) -> None:
    args = _write_inputs(tmp_path)
    scan = tmp_path / "audit" / "scan.json"
    main([
        "--public", "--tables-csv", str(args.tables_csv),
        "--table-metadata-json", str(args.table_metadata_json),
        "--dataset-summary-json", str(args.dataset_summary_json),
        "--statistics-csv", str(args.statistics_csv),
        "--methodology", str(args.methodology),
        "--reproducibility-json", str(args.reproducibility_json),
        "--synthetic-root", str(args.synthetic_root),
        "--synthetic-receipt", str(args.synthetic_receipt),
        "--scan-report", str(scan), "--out", str(args.out),
    ])
    result = json.loads(capsys.readouterr().out)
    assert result["privacy_scan"]["status"] == "PASS"
    assert json.loads(scan.read_text(encoding="utf-8"))["status"] == "PASS"
    assert list(args.out.iterdir()) == [args.out / "index.html"]


def test_public_report_rejects_identifier_and_private_inputs(tmp_path: Path) -> None:
    args = _write_inputs(tmp_path)
    unsafe_identifier = "0000" + "2314_private"
    args.methodology.write_text(f"# Methods\n\nCase {unsafe_identifier}\n", encoding="utf-8")
    with pytest.raises(PublicReportError, match="identifier"):
        build_html(args)
    args = _write_inputs(tmp_path / "second")
    args.private_gallery = tmp_path / "gallery.json"
    with pytest.raises(ReportBuildError, match="rejects every private"):
        build_html(args)


def test_scan_rejects_tampered_or_unreceipted_image(tmp_path: Path) -> None:
    args = _write_inputs(tmp_path)
    html = build_html(args)
    index, _ = publish_once(args.out, html)
    text = index.read_text(encoding="utf-8")
    text = text.replace('data-provenance="synthetic"', 'data-provenance="real"', 1)
    index.write_text(text, encoding="utf-8")
    with pytest.raises(PublicReportError, match="synthetic provenance"):
        scan_public_html(
            index, synthetic_receipt=args.synthetic_receipt, synthetic_root=args.synthetic_root,
            generator_path=ROOT / "scripts" / "make_synthetic_demo.py",
        )


def test_scan_replays_generator_instead_of_trusting_edited_receipt(tmp_path: Path) -> None:
    args = _write_inputs(tmp_path)
    receipt = json.loads(args.synthetic_receipt.read_text(encoding="utf-8"))
    item = next(asset for asset in receipt["assets"] if asset["role"] == "example_image")
    image_path = args.synthetic_root / item["path"]
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image[0, 0] = (int(image[0, 0]) + 1) % 256
    assert cv2.imwrite(str(image_path), image)
    item["sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    args.synthetic_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    html = build_html(args)
    index, _ = publish_once(args.out, html)
    with pytest.raises(PublicReportError, match="independent replay"):
        scan_public_html(
            index, synthetic_receipt=args.synthetic_receipt, synthetic_root=args.synthetic_root,
            generator_path=ROOT / "scripts" / "make_synthetic_demo.py",
        )


def test_public_report_publication_is_write_once(tmp_path: Path) -> None:
    args = _write_inputs(tmp_path)
    html = build_html(args)
    _, first = publish_once(args.out, html)
    _, second = publish_once(args.out, html)
    assert (first, second) == ("created", "verified_identical")
    with pytest.raises(ReportBuildError, match="refusing to replace"):
        publish_once(args.out, html + "changed")


def test_private_report_requires_and_embeds_both_evidence_sections(tmp_path: Path) -> None:
    args = _write_inputs(tmp_path)
    args.public = False
    args.synthetic_root = None
    args.synthetic_receipt = None
    image = tmp_path / "private_evidence.png"
    assert cv2.imwrite(str(image), np.full((10, 12), 120, dtype=np.uint8))
    gallery = tmp_path / "galleries.json"
    gallery.write_text(json.dumps({
        "schema_version": 1, "kind": "private_report_galleries", "sections": [
            {"kind": "qualitative_grid", "title": "Qualitative grid",
             "source_examples": 8, "method_count": 2, "items": [
                {"path": str(image), "mime_type": "image/png", "caption": "Example", "alt": "Overlay"}
            ]},
            {"kind": "failure_gallery", "title": "Failure gallery",
             "failures_per_method": 3, "method_count": 2, "items": [
                {"path": str(image), "mime_type": "image/png", "caption": "Failure", "alt": "Overlay"}
            ]},
        ],
    }), encoding="utf-8")
    low = tmp_path / "low.csv"
    pd.DataFrame([{"image_id": f"private-{index}", "iou": 0.1 + index / 1000}
                  for index in range(20)]).to_csv(low, index=False)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "schema_version": 1, "kind": "private_low_roi_quality_table",
        "title": "Low ROI-quality agreement", "csv": str(low),
        "columns": ["image_id", "iou"],
    }), encoding="utf-8")
    args.private_gallery = gallery
    args.private_audit = audit
    html = build_html(args)
    assert "Qualitative grid" in html and "Failure gallery" in html
    assert "private-0" in html
    assert html.count("data:image/png;base64,") == 2


def test_synthetic_generator_emits_byte_bound_receipt(tmp_path: Path) -> None:
    out = tmp_path / "generated"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_demo.py"),
         "--out", str(out), "--subjects", "1", "--per-subject", "1"],
        check=True, capture_output=True, text=True,
    )
    receipt = json.loads((out / "synthetic_receipt.json").read_text(encoding="utf-8"))
    assert receipt["kind"] == "segbench_synthetic_dataset"
    assert receipt["provenance"] == "generated_by_make_synthetic_demo"
    assert receipt["generator_sha256"] == hashlib.sha256(
        (ROOT / "scripts" / "make_synthetic_demo.py").read_bytes()
    ).hexdigest()
    assert len(receipt["assets"]) == 6
    for item in receipt["assets"]:
        assert hashlib.sha256((out / item["path"]).read_bytes()).hexdigest() == item["sha256"]
