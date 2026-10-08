#!/usr/bin/env python3
"""Build immutable, fail-closed inputs for the Stage 4 finalizer.

The producer has five transactions:

``trainable-inference``
    Wrap a validation selector and the selected validation/test probability
    cards in a non-self-referential inference identity.

``released-inference``
    Bind zero-shot validation/test probability cards to the already-frozen
    Stage 1 prediction-set identity.  Released thresholds are never tuned.

``stage1`` and ``stage2``
    Publish the superseding threshold-reselection bundles.  Scientific CSV
    bytes are copied and hashed, but metric columns are never parsed.

``stage3``
    Copy an immutable native Stage 3 bundle and add the complete amended-run
    threshold attestation plus the separately reselected U-Net seed-0 binding.

Every final artifact is write-once.  Inputs and output locations must remain
under the caller-supplied restricted root.  Standalone identity wrappers do not
interpret metric values.  Before a delivery is published, the shared finalizer
validation replays probability/mask derivations and evaluator outputs from the
bound artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import _path  # noqa: F401
import finalize_stage4 as stage4
from segbench.artifact_identity import canonical_sha256, sha256_file


TRAINABLE_SPEC_KIND = "stage4_trainable_inference_spec"
RELEASED_SPEC_KIND = "stage4_released_inference_spec"
STAGE1_SPEC_KIND = "stage4_stage1_reselection_delivery_spec"
STAGE2_SPEC_KIND = "stage4_stage2_reselection_delivery_spec"
STAGE3_SPEC_KIND = "stage4_stage3_delivery_spec"
BUILDER_KINDS = {
    "trainable-inference": TRAINABLE_SPEC_KIND,
    "released-inference": RELEASED_SPEC_KIND,
    "stage1": STAGE1_SPEC_KIND,
    "stage2": STAGE2_SPEC_KIND,
    "stage3": STAGE3_SPEC_KIND,
}

COMMON_SPEC_KEYS = {"schema_version", "kind"}
SPEC_KEYS = {
    "trainable-inference": COMMON_SPEC_KEYS
    | {
        "method",
        "seed",
        "amendment_path",
        "expected_amendment_sha256",
        "config_path",
        "checkpoint_path",
        "selector_receipt_path",
        "raw_validation_probability_card_path",
        "validation_probability_card_path",
        "test_probability_card_path",
    },
    "released-inference": COMMON_SPEC_KEYS
    | {
        "method",
        "validation_probability_card_path",
        "test_probability_card_path",
        "frozen_stage1_run_card_path",
    },
    "stage1": COMMON_SPEC_KEYS
    | {
        "amendment_path",
        "expected_amendment_sha256",
        "manifest_path",
        "splits_path",
        "frozen_stage1_summary_path",
        "frozen_stage1_artifact_paths",
        "replacement_summary_path",
        "trainable_records",
        "released_records",
    },
    "stage2": COMMON_SPEC_KEYS
    | {
        "amendment_path",
        "expected_amendment_sha256",
        "manifest_path",
        "splits_path",
        "legacy_stage2_delivery_path",
        "replacement_summary_path",
        "trainable_records",
    },
    "stage3": COMMON_SPEC_KEYS
    | {
        "amendment_path",
        "expected_amendment_sha256",
        "manifest_path",
        "splits_path",
        "plan_path",
        "base_stage3_delivery_path",
        "legacy_stage2_delivery_path",
        "stage2_reselection_delivery_path",
        "trainable_records",
    },
}


def _validate_spec_shape(spec: Mapping[str, Any], command: str) -> None:
    expected = SPEC_KEYS[command]
    observed = set(spec)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    stage4.require(
        not missing and not unexpected,
        f"{command} spec fields changed; missing={missing}, unexpected={unexpected}",
    )


def _read_spec(path: Path, restricted_root: Path, expected_kind: str) -> Dict[str, Any]:
    resolved = stage4._require_regular(path, restricted_root, "delivery-builder spec")
    payload = stage4._read_json(resolved, "delivery-builder spec")
    stage4.require(
        payload.get("schema_version") == 1 and payload.get("kind") == expected_kind,
        f"unexpected delivery-builder spec kind; expected {expected_kind}",
    )
    command = next(name for name, kind in BUILDER_KINDS.items() if kind == expected_kind)
    _validate_spec_shape(payload, command)
    return payload


def _input_path(
    value: object,
    restricted_root: Path,
    description: str,
) -> Path:
    return stage4._require_regular(Path(str(value or "")), restricted_root, description)


def _identity_for(path: Path, restricted_root: Path, description: str) -> Dict[str, str]:
    resolved = _input_path(path, restricted_root, description)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _probability_universe(card: Mapping[str, Any], description: str) -> tuple[int, str]:
    probability_set = card.get("probability_set")
    stage4.require(isinstance(probability_set, Mapping), f"{description} lacks probability set")
    count = probability_set.get("count")
    digest = card.get("ordered_source_images_sha256")
    stage4.require(type(count) is int and count > 0, f"{description} has invalid probability count")
    return int(count), stage4._sha(digest, f"{description} source-universe digest")


def _derivation_summary(card: Mapping[str, Any], description: str) -> Dict[str, Any]:
    probability_set = card.get("probability_set")
    derivation = card.get("mask_derivation_verification")
    stage4.require(
        isinstance(probability_set, Mapping)
        and isinstance(derivation, Mapping)
        and derivation.get("complete") is True
        and int(derivation.get("count", -1)) == int(probability_set.get("count", -2))
        and derivation.get("all_masks_rederived_from_retained_probabilities") is True,
        f"{description} lacks complete probability-to-mask derivation evidence",
    )
    return {
        "count": int(derivation["count"]),
        "records_sha256": stage4._sha(
            derivation.get("records_sha256"), f"{description} derivation digest"
        ),
        "all_masks_rederived_from_retained_probabilities": True,
    }


def _amendment_identity(spec: Mapping[str, Any], restricted_root: Path) -> Dict[str, str]:
    identity = _identity_for(
        Path(str(spec.get("amendment_path", ""))),
        restricted_root,
        "threshold-policy amendment",
    )
    expected = spec.get("expected_amendment_sha256", stage4.REGISTERED_AMENDMENT_SHA256)
    stage4.require(
        identity["sha256"] == stage4._sha(expected, "expected threshold amendment digest"),
        "threshold amendment differs from the builder spec",
    )
    amendment = stage4._read_json(Path(identity["path"]), "threshold-policy amendment")
    stage4.require(
        amendment.get("schema_version") == 1
        and amendment.get("kind") == stage4.AMENDMENT_KIND
        and amendment.get("supersedes_exactly") == ["threshold_policy"]
        and tuple(
            amendment.get("replacement_policy", {}).get(
                "allowed_trainable_grid", ()
            )
        )
        == stage4.REPLACEMENT_THRESHOLD_GRID,
        "threshold amendment schema or replacement grid changed",
    )
    return identity


def _publish_json_candidate(
    out: Path,
    payload: Mapping[str, Any],
    restricted_root: Path,
    *,
    publish: bool,
) -> Dict[str, Any]:
    resolved_parent = stage4._require_under(out.parent, restricted_root, "artifact parent")
    stage4.require(resolved_parent.is_dir(), f"artifact parent does not exist: {resolved_parent}")
    stage4._require_under(out, restricted_root, "artifact output")
    if publish:
        stage4._publish_json_once(out, payload)
    return dict(payload)


def _producer_metadata(
    code_identity: Mapping[str, Any],
    spec_path: Path,
    spec: Mapping[str, Any],
    *,
    scientific_artifacts_validated: bool = False,
) -> Dict[str, Any]:
    """Bind the exact spec bytes that were parsed for this transaction."""
    spec_bytes = stage4._read_regular_bytes_nofollow(
        spec_path, "delivery-builder spec"
    )
    try:
        bound_spec = json.loads(spec_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise stage4.Stage4Error("delivery-builder spec changed after parsing") from exc
    stage4.require(
        bound_spec == dict(spec),
        "delivery-builder spec changed after it was parsed",
    )
    return {
        "git_commit": stage4._commit(
            code_identity.get("git_commit"), "delivery producer commit"
        ),
        "git_dirty": False,
        "script_sha256": stage4._sha(
            code_identity.get("script_sha256"), "delivery producer script digest"
        ),
        "spec": {
            "path": str(spec_path.resolve()),
            "sha256": hashlib.sha256(spec_bytes).hexdigest(),
        },
        # Inference-card transactions only bind identities.  Stage-delivery
        # transactions invoke the shared finalizer validation, which parses and
        # independently replays evaluator outputs before publication.
        "results_or_metric_values_parsed": scientific_artifacts_validated,
        "payload_bytes_hashed": True,
    }


def build_trainable_inference_card(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    out: Path,
    restricted_root: Path,
    code_identity: Mapping[str, Any],
    publish: bool = True,
) -> Dict[str, Any]:
    """Build a wrapper that removes the test-card self-hash cycle."""
    _validate_spec_shape(spec, "trainable-inference")
    amendment = _amendment_identity(spec, restricted_root)
    method, seed = str(spec.get("method", "")), spec.get("seed")
    stage4.require(
        method in stage4.TRAINABLE_NEURAL_METHODS and type(seed) is int,
        "trainable inference spec has invalid method/seed",
    )
    identities = {
        name: _identity_for(
            Path(str(spec.get(f"{name}_path", ""))),
            restricted_root,
            f"{method}/seed{seed} {name}",
        )
        for name in (
            "config",
            "checkpoint",
            "selector_receipt",
            "raw_validation_probability_card",
            "validation_probability_card",
            "test_probability_card",
        )
    }
    selector = stage4._read_json(
        Path(identities["selector_receipt"]["path"]), "threshold selector receipt"
    )
    validation = stage4._read_json(
        Path(identities["validation_probability_card"]["path"]),
        "selected validation probability card",
    )
    raw_validation = stage4._read_json(
        Path(identities["raw_validation_probability_card"]["path"]),
        "raw validation probability card",
    )
    test = stage4._read_json(
        Path(identities["test_probability_card"]["path"]),
        "test probability card",
    )
    selected = stage4._selected_threshold(
        selector.get("selected_threshold"), "selector threshold"
    )
    selector_scores, winning_threshold, winning_score = (
        stage4._validated_threshold_scores(
            selector.get("threshold_scores"), "selector threshold scores"
        )
    )
    validation_scores, validation_winner, validation_score = (
        stage4._validated_threshold_scores(
            validation.get("threshold_scores"),
            "selected validation-card threshold scores",
        )
    )
    stage4.require(
        selector.get("selection_split") == "val"
        and selector.get("selection_metric") == "macro_image_dice"
        and selector.get("selection_source") == "validation_only"
        and selector.get("tie_break") == "closest_to_0.5_then_lower"
        and selector_scores == validation_scores
        and winning_threshold == validation_winner == selected
        and isinstance(selector.get("selected_validation_dice"), (int, float))
        and not isinstance(selector.get("selected_validation_dice"), bool)
        and math.isclose(
            float(selector.get("selected_validation_dice")),
            winning_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(winning_score, validation_score, rel_tol=0.0, abs_tol=1e-12),
        "selector does not reproduce the frozen finite argmax/tie-break",
    )
    expected_selector = {
        "schema_version": 1,
        "kind": stage4.SELECTOR_RECEIPT_KIND,
        "method": method,
        "seed": seed,
        "source_config_sha256": identities["config"]["sha256"],
        "checkpoint_sha256": identities["checkpoint"]["sha256"],
        "amendment_sha256": amendment["sha256"],
        "raw_validation_probability_card_sha256": identities[
            "raw_validation_probability_card"
        ]["sha256"],
        "validation_probability_card_sha256": identities[
            "validation_probability_card"
        ]["sha256"],
        "validation_only": True,
        "test_or_heldout_opened": False,
    }
    disagreements = {
        key: (selector.get(key), expected)
        for key, expected in expected_selector.items()
        if selector.get(key) != expected
    }
    stage4.require(not disagreements, f"selector receipt identity mismatch: {disagreements}")
    stage4.require(
        tuple(selector.get("allowed_grid", ())) == stage4.REPLACEMENT_THRESHOLD_GRID,
        "selector receipt uses the wrong threshold grid",
    )
    for split, card in (("val", raw_validation), ("val", validation), ("test", test)):
        stage4.require(
            card.get("schema_version") == 1
            and card.get("kind") == stage4.PROBABILITY_CARD_KIND
            and card.get("method") == method
            and card.get("seed") == seed
            and card.get("split") == split
            and card.get("source_config_sha256") == identities["config"]["sha256"]
            and card.get("checkpoint_sha256") == identities["checkpoint"]["sha256"]
            and card.get("threshold_amendment_sha256") == amendment["sha256"],
            f"{method}/seed{seed}: {split} probability-card lineage changed",
        )
        _derivation_summary(card, f"{method}/seed{seed} {split} probability card")
    stage4.require(
        raw_validation.get("labels_opened") is False
        and raw_validation.get("selection_source") == "unselected_raw"
        and "threshold_scores" not in raw_validation,
        "raw validation probability card contains selection evidence",
    )
    stage4.require(
        validation.get("selection_source") == "validation_only"
        and validation.get("labels_opened") is True
        and stage4._selected_threshold(
            validation.get("selected_validation_threshold", validation.get("selected_threshold")),
            "selected validation-card threshold",
        )
        == selected,
        "selected validation probability card disagrees with selector",
    )
    stage4.require(
        test.get("labels_opened") is False
        and test.get("selection_source") == "unselected_raw"
        and test.get("threshold_selected_before_test") is True
        and test.get("threshold_receipt_sha256")
        == identities["selector_receipt"]["sha256"]
        and stage4._selected_threshold(
            test.get("selected_validation_threshold"),
            f"{method}/seed{seed} test-card threshold",
        )
        == selected,
        "test probability card does not bind the validation-only selector",
    )
    stage4.require(
        validation.get("manifest_sha256") == test.get("manifest_sha256")
        and validation.get("split_sha256") == test.get("split_sha256"),
        "validation/test probability cards use different data identities",
    )
    payload = {
        "schema_version": 1,
        "kind": stage4.TRAINABLE_INFERENCE_CARD_KIND,
        "method": method,
        "seed": seed,
        "selected_validation_threshold": selected,
        "selection_frozen_before_test_inference": True,
        "test_labels_opened": False,
        "identities": {
            "source_config_sha256": identities["config"]["sha256"],
            "checkpoint_sha256": identities["checkpoint"]["sha256"],
            "selector_receipt_sha256": identities["selector_receipt"]["sha256"],
            "raw_validation_probability_card_sha256": identities[
                "raw_validation_probability_card"
            ]["sha256"],
            "validation_probability_card_sha256": identities[
                "validation_probability_card"
            ]["sha256"],
            "test_probability_card_sha256": identities["test_probability_card"][
                "sha256"
            ],
            "amendment_sha256": amendment["sha256"],
            "manifest_sha256": test["manifest_sha256"],
            "split_sha256": test["split_sha256"],
        },
        "producer": _producer_metadata(code_identity, spec_path, spec),
    }
    return _publish_json_candidate(
        out, payload, restricted_root, publish=publish
    )


def build_released_inference_card(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    out: Path,
    restricted_root: Path,
    code_identity: Mapping[str, Any],
    publish: bool = True,
) -> Dict[str, Any]:
    """Bind raw released probabilities to the exact frozen Stage 1 mask set."""
    _validate_spec_shape(spec, "released-inference")
    method = str(spec.get("method", ""))
    stage4.require(method in stage4.RELEASED_NEURAL_METHODS, "invalid released method")
    validation_identity = _identity_for(
        Path(str(spec.get("validation_probability_card_path", ""))),
        restricted_root,
        f"{method} validation probability card",
    )
    test_identity = _identity_for(
        Path(str(spec.get("test_probability_card_path", ""))),
        restricted_root,
        f"{method} test probability card",
    )
    frozen_identity = _identity_for(
        Path(str(spec.get("frozen_stage1_run_card_path", ""))),
        restricted_root,
        f"{method} frozen Stage 1 run card",
    )
    validation = stage4._read_json(
        Path(validation_identity["path"]), f"{method} validation probability card"
    )
    test = stage4._read_json(Path(test_identity["path"]), f"{method} test probability card")
    frozen = stage4._read_json(
        Path(frozen_identity["path"]), f"{method} frozen Stage 1 run card"
    )
    for split, card in (("val", validation), ("test", test)):
        stage4.require(
            card.get("schema_version") == 1
            and card.get("kind") == stage4.PROBABILITY_CARD_KIND
            and card.get("method") == method
            and card.get("seed") is None
            and card.get("split") == split
            and card.get("threshold_source") == "released_fixed"
            and card.get("released_threshold_retuned") is False,
            f"{method}: invalid {split} released probability card",
        )
        derivation = _derivation_summary(card, f"{method} {split} probability card")
        stage4.require(
            card.get("mask_derivation_verification", {}).get(
                "all_masks_match_released_runner"
            )
            is True,
            f"{method}: {split} masks do not match the released runner",
        )
        del derivation
    selected = stage4._selected_threshold(
        test.get("selected_threshold"), f"{method} released threshold"
    )
    stage4.require(
        stage4._selected_threshold(
            validation.get("selected_threshold"), f"{method} validation threshold"
        )
        == selected,
        f"{method}: validation/test released thresholds differ",
    )
    stage4.require(
        validation.get("manifest_sha256") == test.get("manifest_sha256")
        and validation.get("split_sha256") == test.get("split_sha256"),
        f"{method}: validation/test data identities differ",
    )
    expected_count, _ = _probability_universe(test, f"{method} test card")
    current_prediction_sha = stage4._sha(
        test.get("prediction_set_sha256"),
        f"{method} probability-derived prediction-set digest",
    )
    frozen_prediction_sha = stage4._sha(
        frozen.get("prediction_set_sha256"), f"{method} frozen prediction-set digest"
    )
    stage4.require(
        frozen.get("method") == method
        and int(frozen.get("prediction_count", -1)) == expected_count
        and int(test.get("prediction_count", -1)) == expected_count
        and frozen_prediction_sha == current_prediction_sha,
        f"{method}: probability-derived test masks differ from frozen Stage 1",
    )
    payload = {
        "schema_version": 1,
        "kind": stage4.RELEASED_INFERENCE_CARD_KIND,
        "method": method,
        "seed": None,
        "raw_probability_exposed": True,
        "binary_accepted_as_probability": False,
        "released_threshold_retuned": False,
        "threshold_source": "released_fixed",
        "selected_threshold": selected,
        "prediction_count": expected_count,
        "frozen_prediction_set_sha256": frozen_prediction_sha,
        "probability_derived_prediction_set_sha256": current_prediction_sha,
        "full_test_prediction_set_byte_identical": True,
        "identities": {
            "validation_probability_card_sha256": validation_identity["sha256"],
            "test_probability_card_sha256": test_identity["sha256"],
            "frozen_stage1_run_card": frozen_identity,
            "frozen_stage1_run_card_sha256": frozen_identity["sha256"],
        },
        "producer": _producer_metadata(code_identity, spec_path, spec),
    }
    return _publish_json_candidate(out, payload, restricted_root, publish=publish)


def _normalize_threshold_record(
    raw: Mapping[str, Any],
    restricted_root: Path,
) -> Dict[str, Any]:
    method, seed = str(raw.get("method", "")), raw.get("seed")
    lineage = raw.get("training_lineage")
    stage4.require(
        method in stage4.TRAINABLE_NEURAL_METHODS
        and type(seed) is int
        and lineage in {"legacy_completed_before_amendment", "amended_training"},
        "invalid trainable threshold-record identity",
    )
    common_keys = {
        "method",
        "seed",
        "training_lineage",
        "config_path",
        "run_card_path",
        "checkpoint_path",
        "selector_receipt_path",
        "inference_card_path",
        "raw_validation_probability_card_path",
        "validation_probability_card_path",
        "test_probability_card_path",
        "scoring_receipt_path",
    }
    lineage_keys = (
        {"legacy_completion_evidence_path"}
        if lineage == "legacy_completed_before_amendment"
        else {
            "per_epoch_training_profile_card_path",
            "preprocess_cache_receipt_path",
        }
    )
    expected_keys = common_keys | lineage_keys
    if lineage == "amended_training" and "acceleration_receipt_path" in raw:
        expected_keys = expected_keys | {"acceleration_receipt_path"}
    stage4.require(
        set(raw) == expected_keys,
        f"{method}/seed{seed}: threshold-record fields changed; "
        f"missing={sorted(expected_keys - set(raw))}, "
        f"unexpected={sorted(set(raw) - expected_keys)}",
    )
    result: Dict[str, Any] = {
        "method": method,
        "seed": seed,
        "training_lineage": lineage,
    }
    for name in (
        "config",
        "run_card",
        "checkpoint",
        "selector_receipt",
        "inference_card",
        "raw_validation_probability_card",
        "validation_probability_card",
        "test_probability_card",
        "scoring_receipt",
    ):
        result[name] = _identity_for(
            Path(str(raw.get(f"{name}_path", ""))),
            restricted_root,
            f"{method}/seed{seed} {name}",
        )
    selector = stage4._read_json(
        Path(result["selector_receipt"]["path"]), f"{method}/seed{seed} selector"
    )
    result["selected_threshold"] = stage4._selected_threshold(
        selector.get("selected_threshold"), f"{method}/seed{seed} selected threshold"
    )
    if lineage == "legacy_completed_before_amendment":
        for name in ("legacy_completion_evidence",):
            result[name] = _identity_for(
                Path(str(raw.get(f"{name}_path", ""))),
                restricted_root,
                f"{method}/seed{seed} {name}",
            )
    else:
        result["per_epoch_training_profile_card"] = _identity_for(
            Path(str(raw.get("per_epoch_training_profile_card_path", ""))),
            restricted_root,
            f"{method}/seed{seed} per-epoch profile",
        )
        result["preprocess_cache_receipt"] = _identity_for(
            Path(str(raw.get("preprocess_cache_receipt_path", ""))),
            restricted_root,
            f"{method}/seed{seed} preprocessing-cache receipt",
        )
        if "acceleration_receipt_path" in raw:
            result["acceleration_receipt"] = _identity_for(
                Path(str(raw.get("acceleration_receipt_path", ""))),
                restricted_root,
                f"{method}/seed{seed} acceleration receipt",
            )
    return result


def _threshold_attestation(
    amendment_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "amendment_sha256": amendment_sha256,
        "allowed_grid": list(stage4.REPLACEMENT_THRESHOLD_GRID),
        "validation_only": True,
        "old_grid_rejected": True,
        "records": sorted(
            [dict(record) for record in records],
            key=lambda item: (str(item["method"]), int(item["seed"])),
        ),
    }


def _normalize_released_record(
    raw: Mapping[str, Any],
    restricted_root: Path,
    frozen_stage1_artifacts: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    method = str(raw.get("method", ""))
    stage4.require(method in stage4.RELEASED_NEURAL_METHODS, "invalid released method")
    base: Dict[str, Any] = {
        "method": method,
        "seed": None,
        "released_threshold_retuned": False,
        "binary_accepted_as_probability": False,
    }
    availability = raw.get("availability")
    stage4.require(
        availability == "available",
        f"{method}: raw probability support exists; unavailable is not accepted",
    )
    expected_keys = {
        "method",
        "availability",
        "inference_card_path",
        "validation_probability_card_path",
        "test_probability_card_path",
        "frozen_stage1_reference_artifact_name",
    }
    stage4.require(
        set(raw) == expected_keys,
        f"{method}: available released-record fields changed; "
        f"missing={sorted(expected_keys - set(raw))}, "
        f"unexpected={sorted(set(raw) - expected_keys)}",
    )
    reference_name = str(raw.get("frozen_stage1_reference_artifact_name", ""))
    stage4.require(
        reference_name in {
            "scoring_audit",
            "finalizer_audit",
            "journal",
            "table_metadata",
        }
        and reference_name in frozen_stage1_artifacts,
        f"{method}: invalid frozen Stage 1 reference artifact",
    )
    inference = _identity_for(
        Path(str(raw.get("inference_card_path", ""))),
        restricted_root,
        f"{method} released inference card",
    )
    inference_payload = stage4._read_json(
        Path(inference["path"]), f"{method} released inference card"
    )
    base.update(
        {
            "availability": "available",
            "threshold_source": "released_fixed",
            "selected_threshold": stage4._selected_threshold(
                inference_payload.get("selected_threshold"),
                f"{method} released threshold",
            ),
            "inference_card": inference,
            "frozen_stage1_reference_artifact": {
                "name": reference_name,
                "sha256": frozen_stage1_artifacts[reference_name]["sha256"],
            },
            "validation_probability_card": _identity_for(
                Path(str(raw.get("validation_probability_card_path", ""))),
                restricted_root,
                f"{method} validation probability card",
            ),
            "test_probability_card": _identity_for(
                Path(str(raw.get("test_probability_card_path", ""))),
                restricted_root,
                f"{method} test probability card",
            ),
        }
    )
    return base


def _test_universe_from_threshold_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, str]:
    stage4.require(bool(records), "threshold-record list is empty")
    expected: Optional[tuple[int, str]] = None
    for record in records:
        card = stage4._read_json(
            Path(str(record["test_probability_card"]["path"])),
            "test probability card",
        )
        observed = _probability_universe(card, "test probability card")
        if expected is None:
            expected = observed
        else:
            stage4.require(
                observed == expected,
                "trainable probability cards use different test universes",
            )
    assert expected is not None
    return expected


def _bundle_inventory(
    sources: Mapping[str, Path], restricted_root: Path
) -> Dict[str, Path]:
    normalized: Dict[str, Path] = {}
    for relative, source in sources.items():
        name = stage4._safe_relative(relative, "delivery output path")
        stage4.require(name not in normalized, f"duplicate delivery output: {name}")
        normalized[name] = _input_path(source, restricted_root, f"delivery output {name}")
    stage4.require(bool(normalized), "delivery output set is empty")
    return normalized


def _verify_existing_bundle(
    out_dir: Path,
    expected_delivery: Mapping[str, Any],
    restricted_root: Path,
) -> None:
    delivery_path = out_dir / "delivery.json"
    observed = stage4._read_json(
        stage4._require_regular(delivery_path, restricted_root, "existing delivery"),
        "existing delivery",
    )
    stage4.require(
        stage4._canonical_json_bytes(observed)
        == stage4._canonical_json_bytes(expected_delivery),
        f"refusing to replace non-identical delivery: {delivery_path}",
    )
    stage4._validate_delivery_outputs(
        delivery_path, observed, restricted_root, "existing delivery"
    )


def _publish_bundle(
    out_dir: Path,
    sources: Mapping[str, Path],
    payload_without_outputs: Mapping[str, Any],
    restricted_root: Path,
    *,
    publish: bool,
    validate: Callable[[Path, Mapping[str, Any]], None],
) -> Dict[str, Any]:
    """Copy exact bytes and atomically publish a complete immutable bundle."""
    out_dir = Path(os.path.abspath(os.fspath(out_dir)))
    stage4._require_under(out_dir, restricted_root, "delivery bundle output")
    stage4.require(out_dir != restricted_root.resolve(), "delivery output cannot be root")
    source_inventory = _bundle_inventory(sources, restricted_root)
    output_hashes = {
        relative: sha256_file(source) for relative, source in source_inventory.items()
    }
    delivery = {**dict(payload_without_outputs), "outputs": output_hashes}
    if out_dir.exists():
        stage4.require(out_dir.is_dir() and not out_dir.is_symlink(), "existing delivery is not a directory")
        _verify_existing_bundle(out_dir, delivery, restricted_root)
        return delivery
    if not publish:
        return delivery
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent)
    )
    try:
        assert stage is not None
        for relative, source in source_inventory.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            stage4.require(
                sha256_file(destination) == output_hashes[relative],
                f"copied output changed: {relative}",
            )
        delivery_path = stage / "delivery.json"
        stage4._publish_json_once(delivery_path, delivery)
        validate(delivery_path, delivery)
        try:
            stage4._rename_directory_noreplace(stage, out_dir)
        except FileExistsError:
            stage4.require(
                out_dir.is_dir() and not out_dir.is_symlink(),
                "concurrent delivery destination is not a directory",
            )
            _verify_existing_bundle(out_dir, delivery, restricted_root)
        else:
            stage = None
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
    return delivery


def build_stage1_delivery(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    out_dir: Path,
    restricted_root: Path,
    code_identity: Mapping[str, Any],
    publish: bool = True,
) -> Dict[str, Any]:
    _validate_spec_shape(spec, "stage1")
    amendment = _amendment_identity(spec, restricted_root)
    manifest = _identity_for(Path(str(spec.get("manifest_path", ""))), restricted_root, "manifest")
    splits = _identity_for(Path(str(spec.get("splits_path", ""))), restricted_root, "splits")
    frozen_summary = _identity_for(
        Path(str(spec.get("frozen_stage1_summary_path", ""))),
        restricted_root,
        "frozen Stage 1 summary",
    )
    frozen_paths = spec.get("frozen_stage1_artifact_paths")
    expected_frozen_names = {
        "scoring_audit",
        "finalizer_audit",
        "journal",
        "summary",
        "table_metadata",
    }
    stage4.require(
        isinstance(frozen_paths, Mapping)
        and set(frozen_paths) == expected_frozen_names,
        "Stage 1 spec lacks the exact frozen artifact-path set",
    )
    frozen_artifacts = {
        name: _identity_for(
            Path(str(path)), restricted_root, f"frozen Stage 1 {name}"
        )
        for name, path in frozen_paths.items()
    }
    stage4.require(
        frozen_artifacts["summary"]["sha256"] == frozen_summary["sha256"],
        "frozen Stage 1 summary identities disagree",
    )
    replacement_summary = _input_path(
        spec.get("replacement_summary_path"), restricted_root, "Stage 1 replacement summary"
    )
    raw_records = spec.get("trainable_records")
    raw_released = spec.get("released_records")
    stage4.require(isinstance(raw_records, list) and isinstance(raw_released, list), "Stage 1 spec lacks record lists")
    records = [_normalize_threshold_record(row, restricted_root) for row in raw_records]
    released = [
        _normalize_released_record(row, restricted_root, frozen_artifacts)
        for row in raw_released
    ]
    expected_universes = stage4._manifest_split_probability_universes(
        Path(manifest["path"]), Path(splits["path"])
    )
    expected_count = int(expected_universes["test"]["count"])
    expected_universe = str(expected_universes["test"]["ordered_source_images_sha256"])
    summary_sha = sha256_file(replacement_summary)
    attestation = _threshold_attestation(amendment["sha256"], records)
    retention = {
        "schema_version": 1,
        "kind": stage4.PROBABILITY_RETENTION_KIND,
        "binary_masks_accepted_as_probability_maps": False,
        "records": sorted(released, key=lambda item: str(item["method"])),
    }
    payload = {
        "schema_version": 1,
        "kind": stage4.STAGE1_RESELECTION_KIND,
        "supersedes": {
            "stage1_summary_sha256": frozen_summary["sha256"],
            "methods": ["sufs_ft"],
        },
        "frozen_stage1_artifacts_sha256": canonical_sha256(
            {
                name: identity["sha256"]
                for name, identity in sorted(frozen_artifacts.items())
            }
        ),
        "data_inputs": {"manifest": manifest, "splits": splits},
        "ordered_source_images_sha256": expected_universe,
        "threshold_policy_attestation": attestation,
        "probability_retention_attestation": retention,
        "producer": _producer_metadata(
            code_identity,
            spec_path,
            spec,
            scientific_artifacts_validated=True,
        ),
    }
    required = {
        ("sufs_ft", seed) for seed in (0, 1, 2)
    }
    stage4._validate_threshold_attestation(
        payload,
        required_records=required,
        legacy_records=required,
        amendment_sha256=amendment["sha256"],
        manifest_path=Path(manifest["path"]),
        splits_path=Path(splits["path"]),
        manifest_sha256=manifest["sha256"],
        splits_sha256=splits["sha256"],
        restricted_root=restricted_root,
        expected_universes=expected_universes,
        scientific_summary_sha256=summary_sha,
        scientific_summary_path=replacement_summary,
        expected_producer_commit=str(code_identity["git_commit"]),
        description="Stage 1 threshold-reselection delivery",
    )
    stage4._validate_released_probability_retention(
        payload,
        frozen_stage1_artifacts=frozen_artifacts,
        manifest_path=Path(manifest["path"]),
        splits_path=Path(splits["path"]),
        manifest_sha256=manifest["sha256"],
        splits_sha256=splits["sha256"],
        restricted_root=restricted_root,
        expected_universes=expected_universes,
        expected_producer_commit=str(code_identity["git_commit"]),
        description="Stage 1 threshold-reselection delivery",
    )
    return _publish_bundle(
        out_dir,
        {"results/summary.csv": replacement_summary},
        payload,
        restricted_root,
        publish=publish,
        validate=lambda path, value: stage4._validate_delivery_outputs(
            path, value, restricted_root, "Stage 1 threshold-reselection delivery"
        ),
    )


def _scientific_row_identity(path: Path) -> str:
    """Hash only structural row keys; metric cells are deliberately ignored."""
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "seed", "table", "group_value"}
        stage4.require(
            reader.fieldnames is not None and required.issubset(reader.fieldnames),
            "replacement summary lacks structural row columns",
        )
        for row in reader:
            records.append({key: row[key] for key in sorted(required)})
    stage4.require(bool(records), "replacement summary has no scientific rows")
    records.sort(key=lambda item: tuple(item[key] for key in sorted(item)))
    return canonical_sha256(records)


def build_stage2_delivery(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    out_dir: Path,
    restricted_root: Path,
    code_identity: Mapping[str, Any],
    publish: bool = True,
) -> Dict[str, Any]:
    _validate_spec_shape(spec, "stage2")
    amendment = _amendment_identity(spec, restricted_root)
    manifest = _identity_for(Path(str(spec.get("manifest_path", ""))), restricted_root, "manifest")
    splits = _identity_for(Path(str(spec.get("splits_path", ""))), restricted_root, "splits")
    stage2 = _identity_for(
        Path(str(spec.get("legacy_stage2_delivery_path", ""))),
        restricted_root,
        "legacy Stage 2 delivery",
    )
    summary = _input_path(
        spec.get("replacement_summary_path"), restricted_root, "Stage 2 replacement summary"
    )
    raw_records = spec.get("trainable_records")
    stage4.require(isinstance(raw_records, list), "Stage 2 spec lacks trainable records")
    records = [_normalize_threshold_record(row, restricted_root) for row in raw_records]
    expected_universes = stage4._manifest_split_probability_universes(
        Path(manifest["path"]), Path(splits["path"])
    )
    expected_universe = str(expected_universes["test"]["ordered_source_images_sha256"])
    payload = {
        "schema_version": 1,
        "kind": stage4.STAGE2_RESELECTION_KIND,
        "supersedes": {
            "stage2_delivery_sha256": stage2["sha256"],
            "methods": ["unet"],
            "seeds": [0],
        },
        "ordered_source_images_sha256": expected_universe,
        "data_inputs": {"manifest": manifest, "splits": splits},
        "scientific_rows_sha256": _scientific_row_identity(summary),
        "threshold_policy_attestation": _threshold_attestation(
            amendment["sha256"], records
        ),
        "producer": _producer_metadata(
            code_identity,
            spec_path,
            spec,
            scientific_artifacts_validated=True,
        ),
    }
    stage4._validate_threshold_attestation(
        payload,
        required_records={("unet", 0)},
        legacy_records={("unet", 0)},
        amendment_sha256=amendment["sha256"],
        manifest_path=Path(manifest["path"]),
        splits_path=Path(splits["path"]),
        manifest_sha256=manifest["sha256"],
        splits_sha256=splits["sha256"],
        restricted_root=restricted_root,
        expected_universes=expected_universes,
        scientific_summary_sha256=sha256_file(summary),
        scientific_summary_path=summary,
        expected_producer_commit=str(code_identity["git_commit"]),
        description="Stage 2 threshold-reselection delivery",
    )
    return _publish_bundle(
        out_dir,
        {"results/summary.csv": summary},
        payload,
        restricted_root,
        publish=publish,
        validate=lambda path, value: stage4._validate_delivery_outputs(
            path, value, restricted_root, "Stage 2 threshold-reselection delivery"
        ),
    )


def build_stage3_delivery(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    out_dir: Path,
    restricted_root: Path,
    code_identity: Mapping[str, Any],
    publish: bool = True,
) -> Dict[str, Any]:
    _validate_spec_shape(spec, "stage3")
    amendment = _amendment_identity(spec, restricted_root)
    manifest = _identity_for(Path(str(spec.get("manifest_path", ""))), restricted_root, "manifest")
    splits = _identity_for(Path(str(spec.get("splits_path", ""))), restricted_root, "splits")
    plan_path = _input_path(spec.get("plan_path"), restricted_root, "Stage 4 plan")
    plan = stage4._read_json(plan_path, "Stage 4 plan")
    policies = stage4._validate_plan(plan)
    base_path = _input_path(
        spec.get("base_stage3_delivery_path"), restricted_root, "base Stage 3 delivery"
    )
    base = stage4._read_json(base_path, "base Stage 3 delivery")
    stage4.require(
        base.get("schema_version") == 1
        and base.get("kind") == stage4.STAGE3_KIND
        and base.get("manifest_sha256") == manifest["sha256"]
        and base.get("splits_sha256") == splits["sha256"],
        "base Stage 3 delivery identity changed",
    )
    stage4.require(
        "threshold_policy_attestation" not in base,
        "base Stage 3 delivery already contains a threshold attestation",
    )
    base_root, base_outputs = stage4._validate_delivery_outputs(
        base_path, base, restricted_root, "base Stage 3 delivery"
    )
    sources = {relative: base_root / relative for relative in base_outputs}
    stage2 = _identity_for(
        Path(str(spec.get("legacy_stage2_delivery_path", ""))),
        restricted_root,
        "legacy Stage 2 delivery",
    )
    stage2_reselection = _identity_for(
        Path(str(spec.get("stage2_reselection_delivery_path", ""))),
        restricted_root,
        "Stage 2 reselection delivery",
    )
    stage2_reselection_payload = stage4._read_json(
        Path(stage2_reselection["path"]), "Stage 2 reselection delivery"
    )
    replacement_rows_sha = stage4._sha(
        stage2_reselection_payload.get("scientific_rows_sha256"),
        "Stage 2 replacement scientific-row digest",
    )
    raw_records = spec.get("trainable_records")
    stage4.require(isinstance(raw_records, list), "Stage 3 spec lacks trainable records")
    records = [_normalize_threshold_record(row, restricted_root) for row in raw_records]
    base_scored = base.get("newly_scored_seeds")
    stage4.require(
        isinstance(base_scored, list),
        "base Stage 3 delivery lacks per-seed scoring lineage",
    )
    base_by_key = {
        (str(row.get("method", "")), row.get("seed")): row
        for row in base_scored
        if isinstance(row, Mapping)
    }
    stage4.require(
        len(base_by_key) == len(base_scored),
        "base Stage 3 delivery duplicates per-seed lineage",
    )
    for record in records:
        key = (str(record["method"]), record["seed"])
        prior = base_by_key.get(key)
        stage4.require(
            isinstance(prior, Mapping)
            and record["config"]["sha256"] == prior.get("config_sha256")
            and record["run_card"]["sha256"] == prior.get("run_card_sha256")
            and record["checkpoint"]["sha256"] == prior.get("checkpoint_sha256"),
            f"{key[0]}/seed{key[1]}: amended evidence changed the frozen "
            "Stage 3 training lineage",
        )
    expected_universes = stage4._manifest_split_probability_universes(
        Path(manifest["path"]), Path(splits["path"])
    )
    expected_count = int(expected_universes["test"]["count"])
    expected_universe = str(expected_universes["test"]["ordered_source_images_sha256"])
    stage4.require(
        expected_count == int(plan["test_universe"]["tables"]["A"]["overall"]),
        "amended Stage 3 probability-card count differs from the frozen plan",
    )
    stage4.require(
        base.get("ordered_source_images_sha256") == expected_universe,
        "base Stage 3 source universe differs from amended probability cards",
    )
    successful = stage4._validate_stage3_seed_evidence(base, policies)
    # These are final-table full runs only.  LR candidates have no test output;
    # the legacy-grid U-Net++ sweep replay remains an upstream prerequisite.
    expected_records = {
        (method, seed)
        for method, seeds in successful.items()
        if method in stage4.TRAINABLE_NEURAL_METHODS
        for seed in seeds
        if not (method == "unet" and seed == 0)
    }
    payload = dict(base)
    payload.pop("outputs", None)
    payload.update(
        {
            "stage2_unet_seed0": {
                "legacy_provenance_delivery_sha256": stage2["sha256"],
                "threshold_reselection_delivery_sha256": stage2_reselection[
                    "sha256"
                ],
                "replacement_scientific_rows_sha256": replacement_rows_sha,
                "reused_reselected_not_rescored": True,
            },
            "threshold_policy_attestation": _threshold_attestation(
                amendment["sha256"], records
            ),
            "data_inputs": {"manifest": manifest, "splits": splits},
            "producer": _producer_metadata(
                code_identity,
                spec_path,
                spec,
                scientific_artifacts_validated=True,
            ),
        }
    )
    stage4._validate_threshold_attestation(
        payload,
        required_records=expected_records,
        legacy_records=set(),
        amendment_sha256=amendment["sha256"],
        manifest_path=Path(manifest["path"]),
        splits_path=Path(splits["path"]),
        manifest_sha256=manifest["sha256"],
        splits_sha256=splits["sha256"],
        restricted_root=restricted_root,
        expected_universes=expected_universes,
        scientific_summary_sha256=base_outputs["results/summary.csv"],
        scientific_summary_path=base_root / "results" / "summary.csv",
        expected_producer_commit=str(code_identity["git_commit"]),
        description="Stage 3 delivery",
        expected_training_identities=base_by_key,
    )
    return _publish_bundle(
        out_dir,
        sources,
        payload,
        restricted_root,
        publish=publish,
        validate=lambda path, value: stage4._validate_delivery_outputs(
            path, value, restricted_root, "Stage 3 delivery"
        ),
    )


def git_code_identity(repo_root: Path, expected_commit: str) -> Dict[str, Any]:
    """Bind the builder to a clean, committed script at an exact Git SHA."""
    expected_commit = stage4._commit(expected_commit, "delivery producer commit")
    repo_root = repo_root.resolve()

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments], cwd=repo_root, check=False, capture_output=True
        )
        stage4.require(
            completed.returncode == 0,
            completed.stderr.decode("utf-8", errors="replace").strip(),
        )
        return completed.stdout

    stage4.require(
        Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        == repo_root,
        "--repo-root is not the Git top level",
    )
    stage4.require(
        git("rev-parse", "HEAD").decode("ascii").strip() == expected_commit,
        "repository HEAD differs from expected producer commit",
    )
    stage4.require(not git("status", "--porcelain"), "repository is dirty")
    script = Path(__file__).resolve()
    relative = script.relative_to(repo_root).as_posix()
    git("ls-files", "--error-unmatch", relative)
    committed = git("show", f"{expected_commit}:{relative}")
    stage4.require(committed == script.read_bytes(), "producer script differs from Git")
    return {
        "git_commit": expected_commit,
        "git_dirty": False,
        "script_sha256": sha256_file(script),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in BUILDER_KINDS:
        child = commands.add_parser(command)
        child.add_argument("--repo-root", required=True, type=Path)
        child.add_argument("--expected-git-commit", required=True)
        child.add_argument("--restricted-root", required=True, type=Path)
        child.add_argument("--spec", required=True, type=Path)
        if command.endswith("inference"):
            child.add_argument("--out", required=True, type=Path)
        else:
            child.add_argument("--out-dir", required=True, type=Path)
        child.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    restricted_root = args.restricted_root.resolve()
    stage4.require(restricted_root.is_dir(), "restricted root does not exist")
    spec = _read_spec(args.spec, restricted_root, BUILDER_KINDS[args.command])
    if args.command != "released-inference":
        stage4.require(
            spec.get("expected_amendment_sha256")
            == stage4.REGISTERED_AMENDMENT_SHA256,
            "production CLI requires the externally registered threshold amendment",
        )
    code = git_code_identity(args.repo_root, args.expected_git_commit)
    common = {
        "spec": spec,
        "spec_path": args.spec,
        "restricted_root": restricted_root,
        "code_identity": code,
        "publish": not args.dry_run,
    }
    if args.command == "trainable-inference":
        result = build_trainable_inference_card(out=args.out, **common)
    elif args.command == "released-inference":
        result = build_released_inference_card(out=args.out, **common)
    elif args.command == "stage1":
        result = build_stage1_delivery(out_dir=args.out_dir, **common)
    elif args.command == "stage2":
        result = build_stage2_delivery(out_dir=args.out_dir, **common)
    else:
        result = build_stage3_delivery(out_dir=args.out_dir, **common)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
