"""Synthetic-only tests for immutable Stage 4 delivery production."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FINALIZER = _load("stage4_finalizer_delivery_test", SCRIPTS / "finalize_stage4.py")
BUILDER = _load("stage4_delivery_builder_test", SCRIPTS / "build_stage4_deliveries.py")
FIXTURES = _load("stage4_fixture_helpers", Path(__file__).with_name("test_finalize_stage4.py"))


CODE = {
    "git_commit": "c" * 40,
    "git_dirty": False,
    "script_sha256": BUILDER.sha256_file(SCRIPTS / "build_stage4_deliveries.py"),
}


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _spec(root: Path, name: str, kind: str, fields: Mapping[str, Any]) -> Path:
    return _json(
        root / "builder_specs" / f"{name}.json",
        {"schema_version": 1, "kind": kind, **dict(fields)},
    )


def _record_spec(record: Mapping[str, Any], inference_path: Path) -> dict[str, Any]:
    result = {
        "method": record["method"],
        "seed": record["seed"],
        "training_lineage": record["training_lineage"],
        "inference_card_path": str(inference_path),
    }
    for name in (
        "config",
        "run_card",
        "checkpoint",
        "selector_receipt",
        "raw_validation_probability_card",
        "validation_probability_card",
        "test_probability_card",
    ):
        result[f"{name}_path"] = record[name]["path"]
    if record["training_lineage"] == "legacy_completed_before_amendment":
        result["legacy_completion_evidence_path"] = record[
            "legacy_completion_evidence"
        ]["path"]
    else:
        result["per_epoch_training_profile_card_path"] = record[
            "per_epoch_training_profile_card"
        ]["path"]
        result["preprocess_cache_receipt_path"] = record[
            "preprocess_cache_receipt"
        ]["path"]
    old_scoring = Path(record["scoring_receipt"]["path"])
    scoring = json.loads(old_scoring.read_text(encoding="utf-8"))
    scoring["identities"]["inference_card_sha256"] = FINALIZER.sha256_file(
        inference_path
    )
    new_scoring = old_scoring.with_name("scoring_for_builder.json")
    _json(new_scoring, scoring)
    result["scoring_receipt_path"] = str(new_scoring)
    return result


def _trainable_wrapper(
    case: SimpleNamespace,
    record: Mapping[str, Any],
    name: str,
) -> tuple[Path, dict[str, Any]]:
    fields = {
        "method": record["method"],
        "seed": record["seed"],
        "amendment_path": str(case.amendment),
        "expected_amendment_sha256": case.amendment_sha,
    }
    for key in (
        "config",
        "checkpoint",
        "selector_receipt",
        "raw_validation_probability_card",
        "validation_probability_card",
        "test_probability_card",
    ):
        fields[f"{key}_path"] = record[key]["path"]
    spec = _spec(case.restricted, name, BUILDER.TRAINABLE_SPEC_KIND, fields)
    out = case.restricted / "produced_wrappers" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = BUILDER.build_trainable_inference_card(
        json.loads(spec.read_text()),
        spec_path=spec,
        out=out,
        restricted_root=case.restricted,
        code_identity=CODE,
    )
    return out, payload


def _prepared_case(tmp_path: Path) -> SimpleNamespace:
    case = FIXTURES._case(tmp_path)
    stage1_old = json.loads(case.stage1_reselection_delivery.read_text())
    stage2_old = json.loads(case.stage2_reselection_delivery.read_text())
    stage3_old = json.loads(case.stage3_delivery.read_text())

    stage1_specs = []
    for record in stage1_old["threshold_policy_attestation"]["records"]:
        wrapper, _ = _trainable_wrapper(
            case, record, f"stage1-{record['method']}-seed{record['seed']}"
        )
        stage1_specs.append(_record_spec(record, wrapper))

    stage2_specs = []
    for record in stage2_old["threshold_policy_attestation"]["records"]:
        wrapper, _ = _trainable_wrapper(
            case, record, f"stage2-{record['method']}-seed{record['seed']}"
        )
        stage2_specs.append(_record_spec(record, wrapper))

    stage3_specs = []
    for record in stage3_old["threshold_policy_attestation"]["records"]:
        wrapper, _ = _trainable_wrapper(
            case, record, f"stage3-{record['method']}-seed{record['seed']}"
        )
        stage3_specs.append(_record_spec(record, wrapper))

    released_by_method = {
        record["method"]: record
        for record in stage1_old["probability_retention_attestation"]["records"]
    }
    released_specs = []
    for method in FINALIZER.RELEASED_NEURAL_METHODS:
        released = released_by_method[method]
        released_spec = _spec(
            case.restricted,
            f"released-{method}",
            BUILDER.RELEASED_SPEC_KIND,
            {
                "method": method,
                "validation_probability_card_path": released[
                    "validation_probability_card"
                ]["path"],
                "test_probability_card_path": released["test_probability_card"]["path"],
                "frozen_stage1_run_card_path": str(
                    case.restricted
                    / "released_probability_evidence"
                    / method
                    / "frozen_stage1_run.json"
                ),
            },
        )
        released_out = (
            case.restricted / "produced_wrappers" / f"released-{method}.json"
        )
        BUILDER.build_released_inference_card(
            json.loads(released_spec.read_text()),
            spec_path=released_spec,
            out=released_out,
            restricted_root=case.restricted,
            code_identity=CODE,
        )
        released_specs.append(
            {
                "method": method,
                "availability": "available",
                "frozen_stage1_reference_artifact_name": "scoring_audit",
                "inference_card_path": str(released_out),
                "validation_probability_card_path": released[
                    "validation_probability_card"
                ]["path"],
                "test_probability_card_path": released["test_probability_card"]["path"],
            }
        )

    stage1_spec = _spec(
        case.restricted,
        "stage1-delivery",
        BUILDER.STAGE1_SPEC_KIND,
        {
            "amendment_path": str(case.amendment),
            "expected_amendment_sha256": case.amendment_sha,
            "manifest_path": str(case.manifest),
            "splits_path": str(case.splits),
            "frozen_stage1_summary_path": str(case.stage1_summary),
            "frozen_stage1_artifact_paths": {
                name: str(path) for name, path in case.stage1_paths.items()
            },
            "replacement_summary_path": str(
                case.stage1_reselection_delivery.parent / "results" / "summary.csv"
            ),
            "trainable_records": stage1_specs,
            "released_records": released_specs,
        },
    )
    produced_stage1 = case.restricted / "produced_stage1"
    BUILDER.build_stage1_delivery(
        json.loads(stage1_spec.read_text()),
        spec_path=stage1_spec,
        out_dir=produced_stage1,
        restricted_root=case.restricted,
        code_identity=CODE,
    )

    stage2_spec = _spec(
        case.restricted,
        "stage2-delivery",
        BUILDER.STAGE2_SPEC_KIND,
        {
            "amendment_path": str(case.amendment),
            "expected_amendment_sha256": case.amendment_sha,
            "manifest_path": str(case.manifest),
            "splits_path": str(case.splits),
            "legacy_stage2_delivery_path": str(case.stage2_delivery),
            "replacement_summary_path": str(
                case.stage2_reselection_delivery.parent / "results" / "summary.csv"
            ),
            "trainable_records": stage2_specs,
        },
    )
    produced_stage2 = case.restricted / "produced_stage2"
    BUILDER.build_stage2_delivery(
        json.loads(stage2_spec.read_text()),
        spec_path=stage2_spec,
        out_dir=produced_stage2,
        restricted_root=case.restricted,
        code_identity=CODE,
    )

    base_stage3 = case.restricted / "base_stage3"
    (base_stage3 / "results").mkdir(parents=True)
    shutil.copyfile(case.stage3_summary, base_stage3 / "results" / "summary.csv")
    base_payload = dict(stage3_old)
    base_payload.pop("threshold_policy_attestation")
    base_payload["outputs"] = {
        "results/summary.csv": FINALIZER.sha256_file(
            base_stage3 / "results" / "summary.csv"
        )
    }
    _json(base_stage3 / "delivery.json", base_payload)

    stage3_spec = _spec(
        case.restricted,
        "stage3-delivery",
        BUILDER.STAGE3_SPEC_KIND,
        {
            "amendment_path": str(case.amendment),
            "expected_amendment_sha256": case.amendment_sha,
            "manifest_path": str(case.manifest),
            "splits_path": str(case.splits),
            "plan_path": str(case.plan),
            "base_stage3_delivery_path": str(base_stage3 / "delivery.json"),
            "legacy_stage2_delivery_path": str(case.stage2_delivery),
            "stage2_reselection_delivery_path": str(
                produced_stage2 / "delivery.json"
            ),
            "trainable_records": stage3_specs,
        },
    )
    produced_stage3 = case.restricted / "produced_stage3"
    BUILDER.build_stage3_delivery(
        json.loads(stage3_spec.read_text()),
        spec_path=stage3_spec,
        out_dir=produced_stage3,
        restricted_root=case.restricted,
        code_identity=CODE,
    )
    case.produced_stage1 = produced_stage1 / "delivery.json"
    case.produced_stage2 = produced_stage2 / "delivery.json"
    case.produced_stage3 = produced_stage3 / "delivery.json"
    case.stage1_builder_spec = stage1_spec
    return case


def test_produced_deliveries_form_a_valid_results_blind_input_lock(
    tmp_path: Path,
) -> None:
    case = _prepared_case(tmp_path)
    lock = FINALIZER.build_input_lock(
        plan_path=case.plan,
        expected_plan_sha256=case.plan_sha,
        amendment_path=case.amendment,
        expected_amendment_sha256=case.amendment_sha,
        stage1_paths=case.stage1_paths,
        stage1_reselection_delivery_path=case.produced_stage1,
        stage2_delivery_path=case.stage2_delivery,
        stage2_reselection_delivery_path=case.produced_stage2,
        stage3_delivery_path=case.produced_stage3,
        restricted_root=case.restricted,
        code_identity=case.code,
    )
    assert lock["creation_scope"]["scientific_csv_content_parsed"] is True
    assert lock["creation_scope"]["metric_values_inspected"] is True
    assert (
        lock["creation_scope"]["metric_use"]
        == "integrity_replay_only_no_selection"
    )
    assert lock["creation_scope"]["model_selection_performed"] is False
    assert lock["stage1_reselection"]["threshold_policy_evidence"]["records"]
    assert lock["stage3"]["threshold_policy_evidence"]["records"]


def test_trainable_wrapper_rejects_raw_validation_label_access(tmp_path: Path) -> None:
    case = FIXTURES._case(tmp_path)
    delivery = json.loads(case.stage2_reselection_delivery.read_text())
    record = delivery["threshold_policy_attestation"]["records"][0]
    raw_path = Path(record["raw_validation_probability_card"]["path"])
    raw = json.loads(raw_path.read_text())
    raw["labels_opened"] = True
    _json(raw_path, raw)
    fields = {
        "method": record["method"],
        "seed": record["seed"],
        "amendment_path": str(case.amendment),
        "expected_amendment_sha256": case.amendment_sha,
    }
    for key in (
        "config",
        "checkpoint",
        "selector_receipt",
        "raw_validation_probability_card",
        "validation_probability_card",
        "test_probability_card",
    ):
        fields[f"{key}_path"] = record[key]["path"]
    spec = _spec(case.restricted, "bad-raw", BUILDER.TRAINABLE_SPEC_KIND, fields)
    out = case.restricted / "bad-raw-wrapper.json"
    with pytest.raises(BUILDER.stage4.Stage4Error, match="selector receipt|raw validation"):
        BUILDER.build_trainable_inference_card(
            json.loads(spec.read_text()),
            spec_path=spec,
            out=out,
            restricted_root=case.restricted,
            code_identity=CODE,
        )
    assert not out.exists()


def test_trainable_wrapper_dry_run_validates_without_writing(tmp_path: Path) -> None:
    case = FIXTURES._case(tmp_path)
    delivery = json.loads(case.stage2_reselection_delivery.read_text())
    record = delivery["threshold_policy_attestation"]["records"][0]
    fields = {
        "method": record["method"],
        "seed": record["seed"],
        "amendment_path": str(case.amendment),
        "expected_amendment_sha256": case.amendment_sha,
    }
    for key in (
        "config",
        "checkpoint",
        "selector_receipt",
        "raw_validation_probability_card",
        "validation_probability_card",
        "test_probability_card",
    ):
        fields[f"{key}_path"] = record[key]["path"]
    spec = _spec(case.restricted, "dry-run", BUILDER.TRAINABLE_SPEC_KIND, fields)
    out = case.restricted / "dry-run" / "inference.json"
    out.parent.mkdir(parents=True)
    payload = BUILDER.build_trainable_inference_card(
        json.loads(spec.read_text()),
        spec_path=spec,
        out=out,
        restricted_root=case.restricted,
        code_identity=CODE,
        publish=False,
    )
    assert payload["kind"] == BUILDER.stage4.TRAINABLE_INFERENCE_CARD_KIND
    assert not out.exists()


def test_released_wrapper_rejects_frozen_prediction_mismatch(tmp_path: Path) -> None:
    case = FIXTURES._case(tmp_path)
    delivery = json.loads(case.stage1_reselection_delivery.read_text())
    record = next(
        item
        for item in delivery["probability_retention_attestation"]["records"]
        if item["method"] == "sufs"
    )
    frozen = case.restricted / "released_probability_evidence" / "sufs" / "frozen_stage1_run.json"
    payload = json.loads(frozen.read_text())
    payload["prediction_set_sha256"] = "0" * 64
    _json(frozen, payload)
    spec = _spec(
        case.restricted,
        "bad-released",
        BUILDER.RELEASED_SPEC_KIND,
        {
            "method": "sufs",
            "validation_probability_card_path": record[
                "validation_probability_card"
            ]["path"],
            "test_probability_card_path": record["test_probability_card"]["path"],
            "frozen_stage1_run_card_path": str(frozen),
        },
    )
    with pytest.raises(BUILDER.stage4.Stage4Error, match="differ.*frozen"):
        BUILDER.build_released_inference_card(
            json.loads(spec.read_text()),
            spec_path=spec,
            out=case.restricted / "bad-released-wrapper.json",
            restricted_root=case.restricted,
            code_identity=CODE,
        )


def test_bundle_is_write_once_and_rejects_changed_spec_identity(tmp_path: Path) -> None:
    case = _prepared_case(tmp_path)
    original = case.produced_stage1.read_bytes()
    spec = json.loads(case.stage1_builder_spec.read_text())
    spec["released_records"].reverse()
    _json(case.stage1_builder_spec, spec)
    with pytest.raises(BUILDER.stage4.Stage4Error, match="refusing to replace"):
        BUILDER.build_stage1_delivery(
            spec,
            spec_path=case.stage1_builder_spec,
            out_dir=case.produced_stage1.parent,
            restricted_root=case.restricted,
            code_identity=CODE,
        )
    assert case.produced_stage1.read_bytes() == original


def test_bundle_publication_does_not_replace_symlink_created_during_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restricted = tmp_path / "restricted"
    source = restricted / "inputs" / "synthetic.txt"
    source.parent.mkdir(parents=True)
    source.write_text("synthetic input\n", encoding="utf-8")
    victim = restricted / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel.txt"
    sentinel.write_text("must survive\n", encoding="utf-8")
    out = restricted / "delivery-race"
    original_rename = BUILDER.stage4._rename_directory_noreplace
    injected = False

    def racing_rename(stage: Path, destination: Path) -> None:
        nonlocal injected
        assert destination == out.resolve()
        destination.symlink_to(victim, target_is_directory=True)
        injected = True
        original_rename(stage, destination)

    monkeypatch.setattr(
        BUILDER.stage4, "_rename_directory_noreplace", racing_rename
    )
    with pytest.raises(BUILDER.stage4.Stage4Error, match="not a directory"):
        BUILDER._publish_bundle(
            out,
            {"results/synthetic.txt": source},
            {"schema_version": 1, "kind": "synthetic_test_delivery"},
            restricted,
            publish=True,
            validate=lambda _path, _payload: None,
        )
    assert injected
    assert out.is_symlink()
    assert out.resolve() == victim.resolve()
    assert sentinel.read_text(encoding="utf-8") == "must survive\n"


def test_preflight_rejects_unexpected_spec_fields(tmp_path: Path) -> None:
    case = FIXTURES._case(tmp_path)
    delivery = json.loads(case.stage2_reselection_delivery.read_text())
    record = delivery["threshold_policy_attestation"]["records"][0]
    fields = {
        "method": record["method"],
        "seed": record["seed"],
        "amendment_path": str(case.amendment),
        "expected_amendment_sha256": case.amendment_sha,
        "unexpected_input": "must-not-be-ignored",
    }
    for key in (
        "config",
        "checkpoint",
        "selector_receipt",
        "raw_validation_probability_card",
        "validation_probability_card",
        "test_probability_card",
    ):
        fields[f"{key}_path"] = record[key]["path"]
    spec = _spec(case.restricted, "unexpected", BUILDER.TRAINABLE_SPEC_KIND, fields)
    out = case.restricted / "unexpected-wrapper.json"
    with pytest.raises(BUILDER.stage4.Stage4Error, match="unexpected=.*unexpected_input"):
        BUILDER.build_trainable_inference_card(
            json.loads(spec.read_text()),
            spec_path=spec,
            out=out,
            restricted_root=case.restricted,
            code_identity=CODE,
        )
    assert not out.exists()

    fields.pop("unexpected_input")
    fields.pop("config_path")
    missing_spec = _spec(
        case.restricted, "missing", BUILDER.TRAINABLE_SPEC_KIND, fields
    )
    missing_out = case.restricted / "missing-wrapper.json"
    with pytest.raises(BUILDER.stage4.Stage4Error, match="missing=.*config_path"):
        BUILDER.build_trainable_inference_card(
            json.loads(missing_spec.read_text()),
            spec_path=missing_spec,
            out=missing_out,
            restricted_root=case.restricted,
            code_identity=CODE,
        )
    assert not missing_out.exists()


def test_stage3_rejects_missing_amended_run(tmp_path: Path) -> None:
    case = _prepared_case(tmp_path)
    delivery = json.loads(case.produced_stage3.read_text())
    delivery["threshold_policy_attestation"]["records"].pop()
    # Direct finalizer validation proves a producer cannot omit one successful run.
    policies = FINALIZER._validate_plan(json.loads(case.plan.read_text()))
    successful = FINALIZER._validate_stage3_seed_evidence(delivery, policies)
    required = {
        (method, seed)
        for method, seeds in successful.items()
        for seed in seeds
        if not (method == "unet" and seed == 0)
    }
    with pytest.raises(FINALIZER.Stage4Error, match="run universe changed"):
        FINALIZER._validate_threshold_attestation(
            delivery,
            required_records=required,
                legacy_records=set(),
                amendment_sha256=case.amendment_sha,
                manifest_path=case.manifest,
                splits_path=case.splits,
                manifest_sha256=FINALIZER.sha256_file(case.manifest),
            splits_sha256=FINALIZER.sha256_file(case.splits),
            restricted_root=case.restricted,
            expected_universes=FINALIZER._manifest_split_probability_universes(
                case.manifest, case.splits
            ),
            scientific_summary_sha256=delivery["outputs"]["results/summary.csv"],
            scientific_summary_path=case.produced_stage3.parent
            / "results"
            / "summary.csv",
            expected_producer_commit=CODE["git_commit"],
            description="Stage 3 delivery",
        )
