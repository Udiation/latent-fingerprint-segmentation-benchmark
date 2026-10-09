"""Synthetic-only tests for the fail-closed public aggregate importer."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "update_from_aggregate.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("public_aggregate_import_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _result_row(**overrides: str) -> dict[str, str]:
    row = {column: "" for column in MODULE.RESULT_COLUMNS}
    row.update(
        {
            "table": "A",
            "view": "overall",
            "group": "overall",
            "year": "2024",
            "method": "synthetic_method",
            "approach": "Synthetic method",
            "family": "Synthetic",
            "training_data": "Generated inputs",
            "comment": "Synthetic aggregate fixture",
            "status": "complete",
            "seeds_completed": "1",
            "dice": "0.75",
            "dice_seed_std": "0.01",
            "jaccard": "0.60",
            "jaccard_seed_std": "0.02",
            "mdr": "0.20",
            "mdr_seed_std": "0.03",
            "fdr": "0.10",
            "fdr_seed_std": "0.01",
            "accuracy": "0.85",
            "params_m": "1.25",
            "params_m_seed_std": "0.00",
            "ms_per_img": "4.5",
            "ms_per_img_seed_std": "0.2",
            "device": "CPU",
            "ci_lower": "0.70",
            "ci_upper": "0.80",
        }
    )
    row.update(overrides)
    return row


def _statistics_row(**overrides: str) -> dict[str, str]:
    row = {column: "" for column in MODULE.STATISTIC_COLUMNS}
    row.update(
        {
            "table": "A",
            "metric": "dice",
            "comparison": "Synthetic A versus Synthetic B",
            "estimate": "0.10",
            "ci_lower": "0.02",
            "ci_upper": "0.18",
            "p_holm": "0.04",
            "significant": "true",
            "status": "complete",
        }
    )
    row.update(overrides)
    return row


def _rehash(bundle: Path) -> None:
    files = {
        name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        for name in MODULE.EXPECTED_FILES
    }
    manifest = {
        "schema_version": 1,
        "kind": MODULE.BUNDLE_KIND,
        "files": files,
    }
    (bundle / MODULE.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _bundle(tmp_path: Path, *, result: dict[str, str] | None = None) -> Path:
    bundle = tmp_path / "aggregate"
    bundle.mkdir(parents=True)
    result = result or _result_row()
    table = result["table"]
    _write_csv(
        bundle / "public_results.csv",
        MODULE.RESULT_COLUMNS,
        [result],
    )
    _write_csv(
        bundle / "statistics.csv",
        MODULE.STATISTIC_COLUMNS,
        [
            _statistics_row(
                table=table,
                metric="hr" if table == "B1" else "dice",
            )
        ],
    )
    _write_csv(
        bundle / "table_status.csv",
        MODULE.TABLE_COLUMNS,
        [
            {
                "table": table,
                "title": "Synthetic aggregate table",
                "status": "complete",
                "note": "Generated inputs only",
            }
        ],
    )
    _rehash(bundle)
    return bundle


def _public_tree(tmp_path: Path) -> Path:
    root = tmp_path / "public-repo"
    root.mkdir()
    shutil.copy2(ROOT / "README.md", root / "README.md")
    shutil.copytree(ROOT / "site", root / "site")
    shutil.copytree(ROOT / "examples" / "synthetic", root / "examples" / "synthetic")
    (root / "results").mkdir()
    (root / "docs").mkdir()
    (root / "docs" / "RESULTS.md").write_text("old documentation\n", encoding="utf-8")
    return root


def test_dry_run_rebuilds_in_isolation_without_mutating_targets(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    root = _public_tree(tmp_path)
    old_docs = (root / "docs" / "RESULTS.md").read_bytes()
    old_readme = (root / "README.md").read_bytes()
    old_site = (root / "site" / "index.html").read_bytes()
    report = MODULE.run_update(bundle, repo_root=root, dry_run=True)
    assert report["status"] == "PASS"
    assert report["mode"] == "dry-run"
    assert report["mutations_applied"] is False
    assert report["aggregate_rows"] == {"results": 1, "statistics": 1, "tables": 1}
    assert (root / "docs" / "RESULTS.md").read_bytes() == old_docs
    assert (root / "README.md").read_bytes() == old_readme
    assert (root / "site" / "index.html").read_bytes() == old_site
    assert not (root / "results" / "public_results.csv").exists()


def test_apply_installs_only_validated_rows_and_rebuilt_outputs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    root = _public_tree(tmp_path)
    report = MODULE.run_update(bundle, repo_root=root, dry_run=False)
    assert report["status"] == "PASS"
    assert report["mutations_applied"] is True
    assert (root / "results" / "public_results.csv").read_bytes() == (
        bundle / "public_results.csv"
    ).read_bytes()
    documentation = (root / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert "75.00% ± 1.00%" in documentation
    assert "Synthetic aggregate table" in documentation
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "| Synthetic | Synthetic method | 75.00% ± 1.00% | 60.00% ± 2.00% |" in readme
    site = (root / "site" / "index.html").read_text(encoding="utf-8")
    assert '"dice":"0.75"' in site
    assert '"dice_seed_std":"0.01"' in site
    assert (root / "site" / "assets" / "results" / "public_results.csv").read_bytes() == (
        bundle / "public_results.csv"
    ).read_bytes()


def test_hull_aggregate_uses_lower_bound_metric_schema(tmp_path: Path) -> None:
    result = _result_row(
        table="B1",
        group="synthetic_device",
        accuracy="",
        dice="",
        jaccard="",
        mdr="",
        fdr="",
        mc="0.90",
        hr="0.80",
        par="1.25",
        ohf="0.20",
    )
    bundle = _bundle(tmp_path, result=result)
    validated = MODULE.validate_bundle(bundle)
    rendered = MODULE.render_results_markdown(validated)
    assert "| MC | HR | PAR | OHF |" in rendered
    assert "synthetic_device" in rendered
    assert "1.25" in rendered


def test_cli_writes_path_free_dry_run_receipt(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    report_path = tmp_path / "dry-run.json"
    before = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in (
            "README.md",
            "docs/RESULTS.md",
            "results/public_results.csv",
            "site/index.html",
        )
    }
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--bundle",
            str(bundle),
            "--dry-run",
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["status"] == "PASS"
    assert str(tmp_path) not in report_text
    assert report["bundle"]["kind"] == MODULE.BUNDLE_KIND
    after = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in before
    }
    assert after == before


def test_rejects_per_image_column_even_when_manifest_matches(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    columns = (*MODULE.RESULT_COLUMNS, "image_id")
    row = _result_row()
    row["image_id"] = "synthetic-record"
    _write_csv(bundle / "public_results.csv", columns, [row])
    _rehash(bundle)
    with pytest.raises(MODULE.AggregateImportError, match="columns changed"):
        MODULE.validate_bundle(bundle)


@pytest.mark.parametrize(
    ("unsafe", "reason"),
    [
        ("/" + "home/private/result.csv", "absolute private path"),
        ("203" + ".0.113.9", "IPv4 address"),
        ("node" + ".internal", "host name"),
        ("Ab9_" * 8, "opaque cloud-drive-like identifier"),
        ("0000" + "1234_X", "biometric-style sample identifier"),
        ("data:" + "image/png;base64,AAAA", "embedded image payload"),
        ("artifact" + ".pth", "weight or data-array path"),
    ],
)
def test_rejects_sensitive_strings_in_aggregate_cells(
    tmp_path: Path, unsafe: str, reason: str
) -> None:
    bundle = _bundle(tmp_path, result=_result_row(comment=unsafe))
    with pytest.raises(MODULE.AggregateImportError, match=reason):
        MODULE.validate_bundle(bundle)


@pytest.mark.parametrize("name", ["sample.png", "model.pth", "per_image.csv"])
def test_rejects_every_unlisted_bundle_file(tmp_path: Path, name: str) -> None:
    bundle = _bundle(tmp_path)
    (bundle / name).write_bytes(b"synthetic")
    with pytest.raises(MODULE.AggregateImportError, match="inventory changed"):
        MODULE.validate_bundle(bundle)


def test_rejects_hash_drift_and_symlink_inputs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with (bundle / "statistics.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(MODULE.AggregateImportError, match="differs from"):
        MODULE.validate_bundle(bundle)

    bundle = _bundle(tmp_path / "second")
    source = bundle / "public_results.csv"
    outside = tmp_path / "outside.csv"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(MODULE.AggregateImportError, match="regular file"):
        MODULE.validate_bundle(bundle)


def test_apply_refuses_symlink_output_without_touching_target(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    root = _public_tree(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("do not replace\n", encoding="utf-8")
    result_doc = root / "docs" / "RESULTS.md"
    result_doc.unlink()
    result_doc.symlink_to(outside)
    with pytest.raises(MODULE.AggregateImportError, match="symlink output"):
        MODULE.run_update(bundle, repo_root=root, dry_run=False)
    assert outside.read_text(encoding="utf-8") == "do not replace\n"
