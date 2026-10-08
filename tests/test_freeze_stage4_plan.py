"""Metadata-only tests for the frozen Stage 4 preregistration artifact."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_stage4_plan.py"
SPEC = importlib.util.spec_from_file_location("freeze_stage4_plan_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _case(tmp_path: Path) -> dict[str, object]:
    subjects = [f"T{index:03d}" for index in range(40)]
    splits = tmp_path / "splits.json"
    _write_json(
        splits,
        {"seed": MODULE.EXPECTED_SPLIT_SEED, "splits": {subject: "test" for subject in subjects}},
    )

    manifest = tmp_path / "manifest.csv"
    groups = [
        *("VALUE" for _ in range(743)),
        *("LIMITED" for _ in range(336)),
        *("NOVALUE" for _ in range(922)),
        *("MIXED" for _ in range(9)),
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image_id",
                "subject_id",
                "subset",
                "mask_path",
                "quality_mask_path",
                "examiner_value",
            ),
        )
        writer.writeheader()
        removed_quality = False
        for index, group in enumerate(groups):
            quality = f"/must/not/be/opened/quality_{index}.png"
            if group == "VALUE" and not removed_quality:
                quality = ""
                removed_quality = True
            writer.writerow(
                {
                    "image_id": f"synthetic_{index:05d}",
                    "subject_id": subjects[index % len(subjects)],
                    "subset": "finger_latent",
                    "mask_path": f"/must/not/be/opened/roi_{index}.png",
                    "quality_mask_path": quality,
                    "examiner_value": group,
                }
            )

    manifest_sha = MODULE.sha256_file(manifest)
    splits_sha = MODULE.sha256_file(splits)
    return {
        "subjects": subjects,
        "manifest": manifest,
        "manifest_sha": manifest_sha,
        "splits": splits,
        "splits_sha": splits_sha,
    }


def _plan(case: dict[str, object]) -> dict[str, object]:
    return MODULE.build_plan(
        manifest=case["manifest"],
        splits=case["splits"],
        expected_manifest_sha256=case["manifest_sha"],
        expected_splits_sha256=case["splits_sha"],
        expected_test_subject_sha256=MODULE.canonical_subject_digest(case["subjects"]),
        stage1_scoring_audit_sha256="1" * 64,
        stage1_finalizer_audit_sha256="2" * 64,
        stage1_journal_sha256="3" * 64,
        stage1_summary_sha256="4" * 64,
        stage1_summary_rows=180,
        stage1_table_metadata_sha256="5" * 64,
        generator={
            "git_commit": "a" * 40,
            "git_dirty": False,
            "script": "segbench/scripts/freeze_stage4_plan.py",
            "script_sha256": "b" * 64,
            "numpy_version": "synthetic-test",
        },
    )


def test_plan_freezes_exact_universe_draws_and_analysis_policy(tmp_path: Path) -> None:
    case = _case(tmp_path)
    plan = _plan(case)

    assert plan["kind"] == MODULE.KIND
    assert plan["results_opened_during_creation"] is False
    assert plan["heldout_payloads_opened_during_creation"] is False
    assert plan["test_universe"]["test_subject_count"] == 40
    assert plan["test_universe"]["test_subject_ids_embedded"] is False
    assert plan["test_universe"]["test_subject_sha256"] == MODULE.canonical_subject_digest(case["subjects"])
    assert plan["test_universe"]["tables"] == MODULE.EXPECTED_GROUP_COUNTS
    assert plan["bootstrap"]["seed"] == 20260908
    assert plan["bootstrap"]["replicates"] == 1000
    assert plan["bootstrap"]["draw_shape"] == [1000, 40]
    assert plan["bootstrap"]["subject_index_mapping"] == (
        "index i maps to the ith subject_id in ascending canonical sorted order"
    )
    assert plan["bootstrap"]["draw_sha256"] == (
        "c9beff3d21504c02c1b6f178d2392f696786a8976b7e93a413dc07a9b9727bca"
    )
    assert plan["method_universe"]["presentation_rows"] == 18
    assert plan["method_universe"]["real_method_rows"] == 16
    assert plan["method_universe"]["reference_rows"] == 2
    assert [row["key"] for row in plan["method_universe"]["methods"]] == [
        *MODULE.REAL_METHODS,
        *MODULE.REFERENCE_METHODS,
    ]
    assert plan["holm"]["fixed_comparisons"] == 15
    assert plan["wilcoxon"] == {
        "table": "A",
        "metric": "method-level per-image Dice",
        "pairing": "exact 2010-source-image universe",
        "alternative": "two-sided",
        "zero_method": "wilcox",
        "continuity_correction": False,
        "method": "asymptotic",
        "all_zero_rule": {"statistic": 0.0, "raw_p": 1.0, "status": "all_zero"},
        "effects": ["mean_paired_difference", "median_paired_difference", "rank_biserial"],
    }
    assert plan["qualitative_grid"]["quotas"] == {"VALUE": 3, "LIMITED": 3, "NOVALUE": 2}
    assert plan["failure_gallery"]["images_per_real_method"] == 3
    assert plan["external_tables"]["B1"]["status"] == "UNVERIFIED_COORDINATES"


def test_manifest_metadata_is_validated_without_opening_referenced_payloads(tmp_path: Path) -> None:
    case = _case(tmp_path)
    plan = _plan(case)
    assert plan["test_universe"]["referenced_payloads_opened"] is False
    assert not Path("/must/not/be/opened/roi_0.png").exists()


def test_wrong_group_count_fails_closed(tmp_path: Path) -> None:
    case = _case(tmp_path)
    manifest = case["manifest"]
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace(",VALUE\n", ",LIMITED\n", 1), encoding="utf-8")
    case["manifest_sha"] = MODULE.sha256_file(manifest)
    with pytest.raises(MODULE.PlanError, match="group counts differ"):
        _plan(case)


def test_changed_manifest_identity_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    manifest = case["manifest"]
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(MODULE.PlanError, match="manifest digest differs"):
        _plan(case)


def test_changed_test_subject_digest_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["subjects"] = [*case["subjects"][:-1], "DIFFERENT"]
    with pytest.raises(MODULE.PlanError, match="test-subject digest differs"):
        _plan(case)


def test_publish_is_write_once_idempotent_and_refuses_changed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    first = MODULE.canonical_json_bytes({"kind": "synthetic", "value": 1})
    second = MODULE.canonical_json_bytes({"kind": "synthetic", "value": 2})
    assert MODULE.publish_once(path, first) == "created"
    assert MODULE.publish_once(path, first) == "verified_identical"
    assert path.read_bytes() == first
    with pytest.raises(FileExistsError, match="refusing to replace"):
        MODULE.publish_once(path, second)
    assert path.read_bytes() == first


def test_subject_digest_preserves_leading_zeroes_and_terminal_lf() -> None:
    expected = MODULE.hashlib.sha256(b"0001\n0010\n0100\n").hexdigest()
    assert MODULE.canonical_subject_digest(["0100", "0001", "0010"]) == expected


def test_exact_production_subject_and_draw_digests_are_preregistered() -> None:
    assert MODULE.PRESCRIBED_TEST_SUBJECT_SHA256 == (
        "4cf381878beb5b4d8dfc12492a1803d498f067927c3f73f0bf0440be825dc684"
    )
    assert MODULE.EXPECTED_BOOTSTRAP_DRAW_SHA256 == (
        "c9beff3d21504c02c1b6f178d2392f696786a8976b7e93a413dc07a9b9727bca"
    )
    assert MODULE.bootstrap_draw_digest(20260908, 1000, 40) == (
        MODULE.EXPECTED_BOOTSTRAP_DRAW_SHA256
    )
