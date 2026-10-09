"""Synthetic-only tests for the canonical Stage-5 publication bridge."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "import_stage5_public.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage5_public_import_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _load_scanner() -> ModuleType:
    path = ROOT / "scripts" / "scan_release.py"
    spec = importlib.util.spec_from_file_location("stage5_release_scan_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCANNER = _load_scanner()
SEEDED_METHODS = {
    "sufs_ft", "svm", "rf", "welm", "unet", "unetpp", "drunet",
    "ra_runet", "ra_runet_aligned", "lenet", "alexnet", "cnn_ae",
    "segformer_sub",
}


def _csv_bytes(columns: tuple[str, ...], rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _canonical_table(
    table: str = "A", *, view_override: str | None = None,
    group_override: str | None = None, method: str = "unet",
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in MODULE.CANONICAL_TABLE_COLUMNS}
    hull = table in {"B1", "D"}
    view = {
        "A": "overall", "A_value": "examiner_value", "A_prime": "overall",
        "B1": "overall", "D": "latent_to_exemplar",
    }[table]
    view = view_override or view
    group = group_override or ("VALUE" if table == "A_value" else "overall")
    seed_ids = "0,1,2" if method in SEEDED_METHODS else "unseeded"
    n_seeds = 3 if method in SEEDED_METHODS else 1
    row.update({
        "table": table, "view": view,
        "group_by": (
            "examiner_value" if table == "A_value"
            else (view if view in {"group", "device", "capture"} else "")
        ),
        "group_value": group,
        "method": method, "display_name": method.replace("_", " ").title(), "year": 2022,
        "family": "Encoder-decoder", "training_data": "Synthetic train subjects",
        "comment": "Synthetic aggregate fixture", "status": "ok", "reason": "",
        "n_seeds": n_seeds, "n_images": 8, "n_subjects": 4, "n_missing": 0,
        "params_m": 7.2, "params_m_seed_std": 0.0,
        "ms_per_img": 5.5, "ms_per_img_seed_std": 0.2, "device": "GPU",
        "subset": "sd302_exemplar" if hull else "finger_latent", "split": "test",
        "ground_truth": "minutiae hull" if hull else "examiner ROI",
        "seed_ids": seed_ids,
    })
    if hull:
        row.update({
            "mc_mean": 0.95, "hr_mean": 0.85, "hr_ci_low": 0.8,
            "hr_ci_high": 0.9, "hr_seed_std": 0.01,
            "par_mean": 1.2, "par_seed_std": 0.02,
            "ohf_mean": 0.2, "ohf_seed_std": 0.01,
        })
    else:
        row.update({
            "accuracy_mean": 0.8, "error_rate_mean": 0.2,
            "dice_mean": 0.75, "dice_seed_std": 0.01,
            "dice_ci_low": 0.7, "dice_ci_high": 0.8,
            "jaccard_mean": 0.6, "miou_mean": 0.7, "mdr_mean": 0.2,
            "fdr_mean": 0.1,
        })
    return row


def _canonical_stat() -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in MODULE.CANONICAL_STAT_COLUMNS}
    row.update({
        "table": "A", "metric": "Dice", "comparison": "point estimate",
        "reference_method": "", "method": "unet", "estimate": 0.75,
        "ci_lower": 0.7, "ci_upper": 0.8, "p_value": "", "p_holm": "",
        "significant": False, "n_images": 8, "n_subjects": 4,
        "status": "ok", "reason": "", "receipt_status": "",
    })
    return row


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _package(tmp_path: Path, *, metadata: bool = False) -> tuple[Path, str]:
    package = tmp_path / "stage5"
    public = package / "public"
    (public / "tables").mkdir(parents=True)
    (public / "figures").mkdir()
    tables = [
        *[
            _canonical_table(table, method=method)
            for table in ("A", "A_prime")
            for method in sorted(MODULE.DATASET1_METHODS)
        ],
        *[
            _canonical_table("A_value", group_override=group, method=method)
            for group in sorted(MODULE.EXAMINER_GROUPS)
            for method in sorted(MODULE.DATASET1_METHODS)
        ],
        *[
            _canonical_table("B1", view_override=view, group_override=group, method=method)
            for view, group in (
                ("overall", "overall"), ("group", "baseline"),
                ("device", "synthetic-device"), ("capture", "roll"),
            )
            for method in sorted(MODULE.HULL_METHODS)
        ],
        *[
            _canonical_table("D", method=method)
            for method in sorted(MODULE.CROSS_DOMAIN_METHODS)
        ],
    ]
    statistic = _canonical_stat()
    universe: dict[tuple[str, str, str], set[str]] = {}
    for row in tables:
        key = (str(row["table"]), str(row["view"]), str(row["group_value"]))
        universe.setdefault(key, set()).add(str(row["method"]))
    universe_payload = {
        "schema_version": 1,
        "kind": "stage5_public_release_universe_v1",
        "rows": [
            {
                "table": key[0], "view": key[1], "group_value": key[2],
                "methods": sorted(universe[key]),
            }
            for key in sorted(universe)
        ],
    }
    files: dict[str, bytes] = {
        "tables.csv": _csv_bytes(MODULE.CANONICAL_TABLE_COLUMNS, tables),
        "release_universe.json": (
            json.dumps(universe_payload, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "statistics.csv": _csv_bytes(MODULE.CANONICAL_STAT_COLUMNS, [statistic]),
        "statistics.json": (json.dumps([statistic], indent=2, sort_keys=True) + "\n").encode(),
        "team_crosscheck.json": b'{"selection_signal":false,"status":"UNAVAILABLE"}\n',
        "cross_domain_deltas.csv": b"table,metric,status\n",
        "table_metadata.json": (json.dumps({"tables": [{
            "table": row["table"], "view": row["view"],
            "title": f"Synthetic Table {row['table']}",
            "caption": "Synthetic aggregate counts; validation-only threshold selection",
        } for row in {
            (item["table"], item["view"]): item for item in tables
        }.values()]}, indent=2, sort_keys=True) + "\n").encode(),
        "dataset_summary.json": b'{"datasets":[{"dataset":"Synthetic","status":"complete"}]}\n',
        "reproducibility.json": (
            '{"config_hashes":["' + "b" * 64 + '"],"data_manifest_sha256":"'
            + "c" * 64 + '","dates":["2026-10-09"],"git_commit":"'
            + "a" * 40 + '","seeds":[0]}\n'
        ).encode(),
    }
    for table, stem in MODULE.TABLE_STEMS.items():
        metric, value = ("hr_mean", "0.85") if table in {"B1", "D"} else ("dice_mean", "0.75")
        files[f"tables/{stem}.csv"] = f"method,{metric}\nunet,{value}\n".encode()
        files[f"tables/{stem}.md"] = (
            f"### Synthetic Table {table}\n\n| Method | Metric |\n|---|---:|\n| U-Net | {value} |\n"
        ).encode()
        files[f"tables/{stem}.tex"] = (
            f"\\begin{{table}}\\caption{{Synthetic Table {table}}}\\end{{table}}\n"
        ).encode()
    for stem in MODULE.FIGURE_KINDS:
        files[f"figures/{stem}.png"] = b"\x89PNG\r\n\x1a\nsynthetic-aggregate-plot"
        files[f"figures/{stem}.pdf"] = b"%PDF-1.4\n% synthetic aggregate plot\n"
    if metadata:
        for config_path in sorted((ROOT / "configs").glob("*.yaml")):
            config = MODULE.yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config.setdefault("schema_version", 1)
            task = config.get("task")
            method = config.get("method")
            if task in {"pixel", "patch"} or method == "fingernet":
                if config_path.stem in {"sufs_zero", "fingernet"}:
                    config["selection"] = {
                        "status": "selected", "allowed_grid": list(MODULE.THRESHOLD_GRID),
                        "chosen_threshold": 0.5,
                    }
                else:
                    config["selection"] = {
                        "status": "selected", "allowed_grid": list(MODULE.THRESHOLD_GRID),
                        "threshold_by_seed": {"0": 0.5, "1": 0.5, "2": 0.5},
                    }
            if task == "block":
                config["selection"] = {
                    "status": "selected",
                    "chosen_by_seed": {
                        "0": {"synthetic": 1}, "1": {"synthetic": 1},
                        "2": {"synthetic": 1},
                    },
                    "chosen": {"synthetic": 1},
                }
            if method == "gmfs":
                config.setdefault("tuning", {})["chosen"] = {"synthetic": 1}
            files[f"configs/{config_path.name}"] = MODULE.yaml.safe_dump(
                config, sort_keys=False
            ).encode()
        assignments = {
            f"S{index:03d}": split
            for split, count, offset in (("train", 136, 0), ("val", 24, 136), ("test", 40, 160))
            for index in range(offset, offset + count)
        }
        split = {
            "schema_version": 1, "status": "complete", "seed": 7,
            "expected_subject_counts": {"train": 136, "val": 24, "test": 40},
            "splits": assignments,
        }
        files["splits/sd302_subject_split.json"] = (
            json.dumps(split, indent=2, sort_keys=True) + "\n"
        ).encode()
    for relative, data in files.items():
        path = public / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    outputs = {
        f"public/{relative}": hashlib.sha256(data).hexdigest()
        for relative, data in sorted(files.items())
    }
    receipt = {
        "schema_version": 1, "kind": MODULE.RECEIPT_KIND, "status": "PASS",
        "public_scan": {"status": "PASS"}, "contains_per_image_rows": False,
        "contains_subject_or_image_ids": False, "outputs": outputs,
        "outputs_sha256": _canonical_digest(outputs),
    }
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    (package / "package_receipt.json").write_bytes(receipt_bytes)
    return package, hashlib.sha256(receipt_bytes).hexdigest()


def _public_tree(tmp_path: Path, *, git: bool = False) -> Path:
    root = tmp_path / "release"
    root.mkdir(parents=True)
    shutil.copy2(ROOT / "README.md", root / "README.md")
    shutil.copytree(ROOT / "site", root / "site")
    shutil.copytree(ROOT / "examples" / "synthetic", root / "examples" / "synthetic")
    (root / "results").mkdir()
    (root / "docs").mkdir()
    (root / "docs" / "RESULTS.md").write_text("pending\n", encoding="utf-8")
    if git:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run([
            "git", "-C", str(root), "-c", "user.name=Synthetic Test",
            "-c", "user.email=synthetic@example.invalid", "commit", "-qm", "fixture",
        ], check=True)
    return root


def _refresh_receipt(package: Path, *relatives: str) -> str:
    receipt_path = package / "package_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for relative in relatives:
        receipt["outputs"][f"public/{relative}"] = hashlib.sha256(
            (package / "public" / relative).read_bytes()
        ).hexdigest()
    receipt["outputs_sha256"] = _canonical_digest(receipt["outputs"])
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    receipt_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_dry_run_converts_canonical_schema_without_hand_typed_metrics(tmp_path: Path) -> None:
    package, digest = _package(tmp_path)
    root = _public_tree(tmp_path)
    report = MODULE.run_import(package, digest, repo_root=root, dry_run=True)
    assert report["status"] == "PASS"
    assert report["mutations_applied"] is False
    assert report["aggregate_rows"] == {"results": 183, "statistics": 1, "tables": 5}
    assert report["release_metadata"]["configs"] == "absent_from_canonical_package"
    assert not (root / "results" / "public_results.csv").exists()


def test_controlled_apply_writes_compact_results_tables_figures_and_receipt(tmp_path: Path) -> None:
    package, digest = _package(tmp_path, metadata=True)
    root = _public_tree(tmp_path, git=True)
    (root / "configs").mkdir()
    (root / "splits").mkdir()
    report = MODULE.run_import(
        package, digest, repo_root=root, dry_run=False, require_release_metadata=True
    )
    assert report["status"] == "PASS"
    public = list(csv.DictReader((root / "results/public_results.csv").open(encoding="utf-8")))
    assert public[0]["dice"] == "0.75"
    assert public[0]["dice_seed_std"] == "0.01"
    assert public[0]["ms_per_img_seed_std"] == "0.2"
    assert public[0]["ci_lower"] == "0.7"
    table_d = next(row for row in public if row["table"] == "D")
    assert table_d["hr"] == "0.85"
    assert table_d["hr_seed_std"] == "0.01"
    assert table_d["ci_lower"] == "0.8"
    assert (root / "results/tables/table_a.tex").is_file()
    assert (root / "results/canonical/statistics.json").is_file()
    assert (root / "configs/unet.yaml").is_file()
    assert len(list((root / "configs").glob("*.yaml"))) == len(
        list((ROOT / "configs").glob("*.yaml"))
    )
    assert len(json.loads((root / "splits/sd302_subject_split.json").read_text())["splits"]) == 200
    manifest = json.loads((root / "figures/figure_manifest.json").read_text())
    assert manifest["source_receipt"]["sha256"] == digest
    assert len(manifest["figures"]) == 8
    receipt = json.loads((root / "results/stage5_import_receipt.json").read_text())
    assert receipt["source_receipt"]["sha256"] == digest
    approved = SCANNER.verify_imported_release(root, package, digest)
    assert len(approved) == 8

    # Even a coordinated edit of bytes, manifest, and local import receipt is
    # rejected because verification re-derives outputs from the source package.
    figure = root / "figures/dice_boxplot.png"
    figure.write_bytes(figure.read_bytes() + b"tampered")
    manifest_path = root / "figures/figure_manifest.json"
    figure_manifest = json.loads(manifest_path.read_text())
    row = next(item for item in figure_manifest["figures"] if item["path"] == "figures/dice_boxplot.png")
    row["sha256"] = hashlib.sha256(figure.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n")
    receipt_path = root / "results/stage5_import_receipt.json"
    changed_receipt = json.loads(receipt_path.read_text())
    changed_receipt["outputs"]["figures/dice_boxplot.png"] = row["sha256"]
    changed_receipt["outputs"]["figures/figure_manifest.json"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    changed_receipt["outputs_sha256"] = _canonical_digest(changed_receipt["outputs"])
    receipt_path.write_text(json.dumps(changed_receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="differs from source-derived"):
        SCANNER.verify_imported_release(root, package, digest)


def test_final_metadata_gate_reports_canonical_schema_gap(tmp_path: Path) -> None:
    package, digest = _package(tmp_path)
    with pytest.raises(MODULE.Stage5ImportError, match="final release metadata is absent"):
        MODULE.run_import(
            package, digest, repo_root=_public_tree(tmp_path), dry_run=True,
            require_release_metadata=True,
        )


def test_rejects_incomplete_method_or_examiner_group_universe(tmp_path: Path) -> None:
    package, _ = _package(tmp_path)
    public = package / "public"
    rows = list(csv.DictReader((public / "tables.csv").open(encoding="utf-8")))
    rows = [
        row for row in rows
        if not (row["table"] == "A" and row["method"] == "alexnet")
    ]
    (public / "tables.csv").write_bytes(_csv_bytes(MODULE.CANONICAL_TABLE_COLUMNS, rows))
    digest = _refresh_receipt(package, "tables.csv")
    with pytest.raises(MODULE.Stage5ImportError, match="method universe is incomplete"):
        MODULE.run_import(package, digest, repo_root=_public_tree(tmp_path), dry_run=True)

    second, _ = _package(tmp_path / "examiner")
    second_public = second / "public"
    rows = list(csv.DictReader((second_public / "tables.csv").open(encoding="utf-8")))
    rows = [row for row in rows if not (
        row["table"] == "A_value" and row["group_value"] == "LIMITED"
    )]
    (second_public / "tables.csv").write_bytes(
        _csv_bytes(MODULE.CANONICAL_TABLE_COLUMNS, rows)
    )
    second_digest = _refresh_receipt(second, "tables.csv")
    with pytest.raises(MODULE.Stage5ImportError, match="examiner-value group universe"):
        MODULE.run_import(
            second, second_digest, repo_root=_public_tree(tmp_path / "examiner"), dry_run=True
        )


def test_rejects_receipt_bound_universe_that_disagrees_with_tables(tmp_path: Path) -> None:
    package, _ = _package(tmp_path)
    path = package / "public/release_universe.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["methods"].remove(payload["rows"][0]["methods"][0])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _refresh_receipt(package, "release_universe.json")
    with pytest.raises(MODULE.Stage5ImportError, match="differs from tables.csv"):
        MODULE.run_import(package, digest, repo_root=_public_tree(tmp_path), dry_run=True)


def test_rejects_receipt_tampering_and_public_artifact_drift(tmp_path: Path) -> None:
    package, digest = _package(tmp_path)
    with pytest.raises(MODULE.Stage5ImportError, match="independently pinned"):
        MODULE.run_import(package, "0" * 64, repo_root=_public_tree(tmp_path), dry_run=True)
    (package / "public/tables.csv").write_bytes(b"changed")
    with pytest.raises(MODULE.Stage5ImportError, match="artifact bytes changed"):
        MODULE.run_import(package, digest, repo_root=_public_tree(tmp_path / "second"), dry_run=True)


def test_rejects_statistics_json_csv_disagreement(tmp_path: Path) -> None:
    package, _ = _package(tmp_path)
    payload = json.loads((package / "public/statistics.json").read_text())
    payload[0]["estimate"] = 0.5
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    (package / "public/statistics.json").write_bytes(data)
    receipt_path = package / "package_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"]["public/statistics.json"] = hashlib.sha256(data).hexdigest()
    receipt["outputs_sha256"] = _canonical_digest(receipt["outputs"])
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    with pytest.raises(MODULE.Stage5ImportError, match="JSON/CSV differ"):
        MODULE.run_import(package, digest, repo_root=_public_tree(tmp_path), dry_run=True)


def test_reconciles_only_authenticated_unavailable_point_estimate_edge(tmp_path: Path) -> None:
    package, _ = _package(tmp_path)
    public = package / "public"
    table_rows = list(csv.DictReader((public / "tables.csv").open(encoding="utf-8")))
    target = next(
        row for row in table_rows if row["table"] == "A" and row["method"] == "unet"
    )
    target["status"], target["reason"] = "unavailable", "synthetic runtime unavailable"
    for metric in ("accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr"):
        for suffix in ("mean", "seed_std", "ci_low", "ci_high"):
            target[f"{metric}_{suffix}"] = ""
    (public / "tables.csv").write_bytes(_csv_bytes(MODULE.CANONICAL_TABLE_COLUMNS, table_rows))
    stat_rows = list(csv.DictReader((public / "statistics.csv").open(encoding="utf-8")))
    stat_rows[0].update({
        "status": "", "reason": "", "estimate": "", "ci_lower": "0",
        "ci_upper": "1", "significant": "False",
    })
    (public / "statistics.csv").write_bytes(_csv_bytes(MODULE.CANONICAL_STAT_COLUMNS, stat_rows))
    (public / "statistics.json").write_text(
        json.dumps(stat_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _refresh_receipt(package, "tables.csv", "statistics.csv", "statistics.json")
    report = MODULE.run_import(package, digest, repo_root=_public_tree(tmp_path), dry_run=True)
    assert report["canonical_reconciliations"]["unavailable_point_estimates"] == 1


def test_rejects_unknown_statistic_status_and_raw_encoded_payload(tmp_path: Path) -> None:
    package, _ = _package(tmp_path)
    public = package / "public"
    stat_rows = list(csv.DictReader((public / "statistics.csv").open(encoding="utf-8")))
    stat_rows[0]["status"] = "FAIL"
    (public / "statistics.csv").write_bytes(_csv_bytes(MODULE.CANONICAL_STAT_COLUMNS, stat_rows))
    (public / "statistics.json").write_text(
        json.dumps(stat_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _refresh_receipt(package, "statistics.csv", "statistics.json")
    with pytest.raises(MODULE.Stage5ImportError, match="unsupported table/status"):
        MODULE.run_import(package, digest, repo_root=_public_tree(tmp_path), dry_run=True)

    package2, _ = _package(tmp_path / "encoded")
    path = package2 / "public/tables/table_a.md"
    path.write_text(path.read_text() + "\n" + "A" * 100 + "\n", encoding="utf-8")
    digest2 = _refresh_receipt(package2, "tables/table_a.md")
    with pytest.raises(MODULE.Stage5ImportError, match="long encoded token"):
        MODULE.run_import(package2, digest2, repo_root=_public_tree(tmp_path / "encoded"), dry_run=True)


def test_rejects_private_host_or_checkpoint_in_receipt_bound_config(tmp_path: Path) -> None:
    for index, injected in enumerate(("server: internal.example.invalid\n", "checkpoint: model/private.pth\n")):
        case = tmp_path / str(index)
        package, _ = _package(case, metadata=True)
        path = package / "public/configs/unet.yaml"
        path.write_text(path.read_text() + injected, encoding="utf-8")
        digest = _refresh_receipt(package, "configs/unet.yaml")
        with pytest.raises(MODULE.Stage5ImportError):
            MODULE.run_import(
                package, digest, repo_root=_public_tree(case), dry_run=True,
                require_release_metadata=True,
            )


def test_requires_complete_per_seed_thresholds_and_binds_unavailable_selector(
    tmp_path: Path,
) -> None:
    incomplete, _ = _package(tmp_path / "incomplete", metadata=True)
    config_path = incomplete / "public/configs/unet.yaml"
    config = MODULE.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["selection"]["threshold_by_seed"].pop("2")
    config_path.write_text(MODULE.yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    digest = _refresh_receipt(incomplete, "configs/unet.yaml")
    with pytest.raises(MODULE.Stage5ImportError, match="per-seed thresholds"):
        MODULE.run_import(
            incomplete, digest, repo_root=_public_tree(tmp_path / "incomplete"),
            dry_run=True, require_release_metadata=True,
        )

    block_package, _ = _package(tmp_path / "block", metadata=True)
    block_path = block_package / "public/configs/svm.yaml"
    block = MODULE.yaml.safe_load(block_path.read_text(encoding="utf-8"))
    block["selection"]["chosen_by_seed"].pop("1")
    block_path.write_text(MODULE.yaml.safe_dump(block, sort_keys=False), encoding="utf-8")
    block_digest = _refresh_receipt(block_package, "configs/svm.yaml")
    with pytest.raises(MODULE.Stage5ImportError, match="per-seed block parameters"):
        MODULE.run_import(
            block_package, block_digest, repo_root=_public_tree(tmp_path / "block"),
            dry_run=True, require_release_metadata=True,
        )

    unavailable, _ = _package(tmp_path / "unavailable", metadata=True)
    public = unavailable / "public"
    rows = list(csv.DictReader((public / "tables.csv").open(encoding="utf-8")))
    reason = "synthetic unlabeled target is unavailable"
    for row in rows:
        if row["table"] == "B1" and row["method"] == "ra_runet_aligned":
            row["status"], row["reason"] = "unavailable", reason
            for metric in ("mc", "hr", "par", "ohf"):
                for suffix in ("mean", "seed_std", "ci_low", "ci_high"):
                    row[f"{metric}_{suffix}"] = ""
    (public / "tables.csv").write_bytes(_csv_bytes(MODULE.CANONICAL_TABLE_COLUMNS, rows))
    aligned_path = public / "configs/ra_runet_aligned.yaml"
    aligned = MODULE.yaml.safe_load(aligned_path.read_text(encoding="utf-8"))
    aligned["selection"] = {"status": "unavailable", "reason": reason}
    aligned_path.write_text(MODULE.yaml.safe_dump(aligned, sort_keys=False), encoding="utf-8")
    digest = _refresh_receipt(
        unavailable, "tables.csv", "configs/ra_runet_aligned.yaml"
    )
    report = MODULE.run_import(
        unavailable, digest, repo_root=_public_tree(tmp_path / "unavailable"),
        dry_run=True, require_release_metadata=True,
    )
    assert report["status"] == "PASS"

    aligned["selection"]["reason"] = "different reason"
    aligned_path.write_text(MODULE.yaml.safe_dump(aligned, sort_keys=False), encoding="utf-8")
    digest = _refresh_receipt(unavailable, "configs/ra_runet_aligned.yaml")
    with pytest.raises(MODULE.Stage5ImportError, match="table reason"):
        MODULE.run_import(
            unavailable, digest, repo_root=_public_tree(tmp_path / "mismatch"),
            dry_run=True, require_release_metadata=True,
        )


def test_controlled_apply_requires_git_and_final_metadata(tmp_path: Path) -> None:
    package, digest = _package(tmp_path, metadata=True)
    with pytest.raises(MODULE.Stage5ImportError, match="clean Git worktree"):
        MODULE.run_import(
            package, digest, repo_root=_public_tree(tmp_path), dry_run=False,
            require_release_metadata=True,
        )
    root = _public_tree(tmp_path / "git", git=True)
    with pytest.raises(MODULE.Stage5ImportError, match="requires final release metadata"):
        MODULE.run_import(package, digest, repo_root=root, dry_run=False)
