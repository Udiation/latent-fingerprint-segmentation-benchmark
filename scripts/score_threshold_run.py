#!/usr/bin/env python3
"""Score one frozen threshold run and publish immutable scoring provenance.

This wrapper is intentionally narrower than ``evaluate_masks.py``.  It
authenticates the frozen selector, inference card, test probability card, and
the exact retained probability/mask pair set before invoking the common
evaluator exactly once for Table A and once for Table A-prime.  Ground-truth
bytes and metric values are opened only by the evaluator; this process treats
the generated summary rows as opaque CSV strings when hashing them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

import _path  # noqa: F401
from segbench import features as block_features
from segbench.postprocess import postprocess_blocks


SCHEMA_VERSION = 1
RECEIPT_KIND = "superseding_threshold_scoring_receipt"
PROBABILITY_CARD_KIND = "complete_probability_map_card"
SELECTOR_RECEIPT_KIND = "expanded_validation_threshold_selector_receipt"
INFERENCE_CARD_KIND = "threshold_bound_inference_card"
AMENDMENT_KIND = "stage4_threshold_policy_amendment"
TABLES = (("A", "mask_path"), ("A_prime", "quality_mask_path"))
SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LOWER_HEX = set("0123456789abcdef")
EVALUATOR_BOOTSTRAP = (
    "import os,runpy,sys;"
    "p=sys.argv.pop(1);"
    "sys.path.insert(0,os.path.dirname(p));"
    "runpy.run_path(p,run_name='__main__')"
)


class ScoringError(RuntimeError):
    """A fail-closed scoring or provenance error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoringError(message)


def _sha(value: object, description: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in LOWER_HEX for character in value),
        f"{description} is not a lowercase SHA-256",
    )
    return str(value)


def _commit(value: object, description: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in LOWER_HEX for character in value),
        f"{description} is not an immutable lowercase Git SHA",
    )
    return str(value)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoringError(f"cannot read {description}: {path}: {exc}") from exc
    require(isinstance(value, dict), f"{description} is not a JSON object")
    return value


def _under(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ScoringError(f"{description} escapes the restricted root: {path}") from exc
    return resolved


def _regular(path: Path, root: Path, description: str) -> Path:
    resolved = _under(path, root, description)
    require(path.exists(), f"missing {description}: {path}")
    require(not path.is_symlink(), f"{description} must not be a symlink: {path}")
    require(path.is_file(), f"{description} is not a regular file: {path}")
    return resolved


def _directory(path: Path, root: Path, description: str) -> Path:
    resolved = _under(path, root, description)
    require(path.exists(), f"missing {description}: {path}")
    require(not path.is_symlink(), f"{description} must not be a symlink: {path}")
    require(path.is_dir(), f"{description} is not a directory: {path}")
    return resolved


def _prepare_directory(path: Path, root: Path, description: str) -> Path:
    resolved = _under(path, root, description)
    require(not path.is_symlink(), f"{description} must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    require(path.is_dir() and not path.is_symlink(), f"invalid {description}: {path}")
    return resolved


def _output_path(path: Path, root: Path, description: str) -> Path:
    resolved = _under(path, root, description)
    _prepare_directory(path.parent, root, f"{description} parent")
    require(not path.is_symlink(), f"{description} must not be a symlink: {path}")
    require(not path.exists() or path.is_file(), f"{description} is not a file: {path}")
    return resolved


def _test_target_shapes(
    manifest: Path, splits: Path
) -> Dict[str, tuple[int, int]]:
    """Derive the exact held-out image universe and canonical 500-ppi shapes."""
    split_payload = _read_json(splits, "subject split")
    assignments = split_payload.get("splits")
    require(isinstance(assignments, Mapping), "subject split lacks assignments")
    try:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "image_id",
                "subject_id",
                "subset",
                "examiner_value",
                "mask_path",
                "target_height",
                "target_width",
            }
            require(
                reader.fieldnames is not None and required.issubset(reader.fieldnames),
                "manifest lacks scoring-universe fields",
            )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ScoringError(f"cannot read scoring manifest: {exc}") from exc
    result: Dict[str, tuple[int, int]] = {}
    for row in rows:
        subject = str(row.get("subject_id", ""))
        if (
            str(row.get("subset", "")) != "finger_latent"
            or str(assignments.get(subject, "")) != "test"
            or str(row.get("examiner_value", "")).upper() == "NONPRINT"
            or not str(row.get("mask_path", "")).strip()
        ):
            continue
        image_id = str(row.get("image_id", ""))
        try:
            height = int(row.get("target_height", ""))
            width = int(row.get("target_width", ""))
        except (TypeError, ValueError) as exc:
            raise ScoringError(f"invalid target shape for {image_id}") from exc
        require(
            SAFE_IMAGE_ID.fullmatch(image_id) is not None
            and image_id not in result
            and height > 0
            and width > 0,
            f"invalid/duplicate manifest test record: {image_id!r}",
        )
        result[image_id] = (height, width)
    require(result, "manifest/split test universe is empty")
    return result


def _git(
    repo_root: Path, *arguments: str, description: str = "Git command"
) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    require(
        completed.returncode == 0,
        f"{description} failed: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )
    return completed.stdout


def committed_code_identity(
    repo_root: Path,
    expected_commit: str,
    expected_wrapper_sha256: str,
    expected_evaluator_sha256: str,
) -> tuple[Dict[str, Any], Path]:
    """Authenticate clean wrapper/evaluator bytes at one caller-pinned commit."""
    expected_commit = _commit(expected_commit, "expected scoring commit")
    expected_wrapper_sha256 = _sha(
        expected_wrapper_sha256, "expected scoring-wrapper digest"
    )
    expected_evaluator_sha256 = _sha(
        expected_evaluator_sha256, "expected common-evaluator digest"
    )
    repo_root = repo_root.resolve()
    require(repo_root.is_dir(), f"repository root is absent: {repo_root}")
    observed_root = Path(
        _git(repo_root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    require(observed_root == repo_root, "--repo-root is not the Git top level")
    observed_commit = _git(repo_root, "rev-parse", "HEAD").decode("ascii").strip()
    require(
        observed_commit == expected_commit,
        "repository HEAD differs from the expected scoring commit",
    )
    require(
        not _git(repo_root, "status", "--porcelain"),
        "repository is dirty; refusing to claim committed scoring code",
    )

    wrapper = Path(__file__).resolve()
    evaluator = wrapper.with_name("evaluate_masks.py")
    require(evaluator.is_file(), f"common evaluator is absent: {evaluator}")
    for path, expected, description in (
        (wrapper, expected_wrapper_sha256, "scoring wrapper"),
        (evaluator, expected_evaluator_sha256, "common evaluator"),
    ):
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ScoringError(f"{description} is outside the repository") from exc
        _git(
            repo_root,
            "ls-files",
            "--error-unmatch",
            relative,
            description=f"locating committed {description}",
        )
        committed = _git(
            repo_root,
            "show",
            f"{expected_commit}:{relative}",
            description=f"reading committed {description}",
        )
        current = path.read_bytes()
        require(committed == current, f"{description} bytes differ from Git")
        require(
            hashlib.sha256(current).hexdigest() == expected,
            f"{description} differs from its caller-pinned SHA-256",
        )
    return (
        {
            "git_commit": expected_commit,
            "git_dirty": False,
            "script_sha256": expected_wrapper_sha256,
            "evaluator_sha256": expected_evaluator_sha256,
        },
        evaluator,
    )


def _selected_threshold(value: object, description: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{description} is not numeric",
    )
    threshold = float(value)
    require(
        math.isfinite(threshold) and 0.0 <= threshold <= 1.0,
        f"{description} is non-finite or outside [0,1]",
    )
    return threshold


def _validate_control_cards(
    *,
    method: str,
    seed: int,
    manifest_sha256: str,
    splits_sha256: str,
    config_path: Path,
    probability_card_path: Path,
    selector_path: Path,
    amendment_path: Path,
    inference_path: Path,
) -> tuple[float, Dict[str, str], Dict[str, Any], str, Dict[str, Any]]:
    config_sha256 = sha256_file(config_path)
    selector_sha256 = sha256_file(selector_path)
    amendment_sha256 = sha256_file(amendment_path)
    probability_card_sha256 = sha256_file(probability_card_path)
    inference_sha256 = sha256_file(inference_path)
    selector = _read_json(selector_path, "threshold selector receipt")
    amendment = _read_json(amendment_path, "threshold-policy amendment")
    card = _read_json(probability_card_path, "test probability card")
    inference = _read_json(inference_path, "threshold-bound inference card")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ScoringError(f"cannot read source configuration: {config_path}: {exc}") from exc
    require(isinstance(config, Mapping), "source configuration is not a mapping")
    task = str(config.get("task", ""))
    require(
        config.get("schema_version") == 1
        and config.get("method") == method
        and config.get("seed") == seed
        and task in {"pixel", "patch"},
        "source configuration method/task/seed identity changed",
    )
    data_config = config.get("data")
    inference_config = config.get("inference")
    require(
        isinstance(data_config, Mapping)
        and float(data_config.get("target_ppi", 0.0)) == 500.0
        and data_config.get("subset") == "finger_latent"
        and isinstance(inference_config, Mapping),
        "source configuration scoring geometry changed",
    )
    min_component_blocks: Optional[int] = None
    if task == "patch":
        block_size = data_config.get(
            "block_size", inference_config.get("block_size")
        )
        raw_minimum = inference_config.get("min_component_blocks")
        require(
            type(block_size) is int
            and block_size == block_features.BLOCK == 16
            and type(raw_minimum) is int
            and raw_minimum >= 1,
            "patch configuration block postprocessing changed",
        )
        min_component_blocks = int(raw_minimum)

    selected = _selected_threshold(
        selector.get("selected_threshold"), "selector threshold"
    )
    require(
        selector.get("schema_version") == 1
        and selector.get("kind") == SELECTOR_RECEIPT_KIND
        and selector.get("method") == method
        and selector.get("task") == task
        and selector.get("seed") == seed
        and selector.get("selection_split") == "val"
        and selector.get("selection_source") == "validation_only"
        and selector.get("validation_only") is True
        and selector.get("test_or_heldout_opened") is False
        and selector.get("source_config_sha256") == config_sha256
        and selector.get("manifest_sha256") == manifest_sha256
        and selector.get("split_sha256") == splits_sha256
        and selector.get("amendment_sha256") == amendment_sha256,
        "selector receipt does not prove frozen validation-only selection",
    )
    require(
        amendment.get("schema_version") == 1
        and amendment.get("kind") == AMENDMENT_KIND,
        "threshold amendment schema or kind changed",
    )
    replacement = amendment.get("replacement_policy")
    require(isinstance(replacement, Mapping), "threshold amendment lacks replacement policy")
    allowed_grid = replacement.get("allowed_trainable_grid")
    require(
        isinstance(allowed_grid, list)
        and selected
        in [_selected_threshold(value, "amended threshold") for value in allowed_grid],
        "selected threshold is outside the amended grid",
    )

    require(
        card.get("schema_version") == 1
        and card.get("kind") == PROBABILITY_CARD_KIND
        and card.get("status") == "complete"
        and card.get("complete") is True
        and card.get("method") == method
        and card.get("seed") == seed
        and card.get("task") == task
        and card.get("split") == "test"
        and card.get("subset") == "finger_latent"
        and card.get("manifest_sha256") == manifest_sha256
        and card.get("split_sha256") == splits_sha256
        and card.get("threshold_receipt_sha256") == selector_sha256
        and card.get("threshold_amendment_sha256") == amendment_sha256
        and card.get("threshold_selected_before_test") is True
        and card.get("labels_opened") is False
        and card.get("selection_source") == "unselected_raw"
        and card.get("probabilities_are_pre_threshold") is True
        and card.get("storage_format") == "float16_npy"
        and card.get("storage_dtype") == "float16"
        and card.get("derived_from_binary_mask") is False
        and card.get("binary_mask_reinterpreted") is False
        and card.get("source_config_sha256") == config_sha256
        and _selected_threshold(
            card.get("selected_validation_threshold"), "test-card threshold"
        )
        == selected,
        "test probability card does not bind the frozen threshold/data lineage",
    )
    card_identities = card.get("identities")
    require(
        isinstance(card_identities, Mapping)
        and card_identities.get("source_config_sha256") == config_sha256
        and card_identities.get("manifest_sha256") == manifest_sha256
        and card_identities.get("split_sha256") == splits_sha256
        and card_identities.get("selector_receipt_sha256") == selector_sha256
        and card_identities.get("amendment_sha256") == amendment_sha256,
        "test probability-card identities changed",
    )

    identities = inference.get("identities")
    require(
        inference.get("schema_version") == 1
        and inference.get("kind") == INFERENCE_CARD_KIND
        and inference.get("method") == method
        and inference.get("seed") == seed
        and _selected_threshold(
            inference.get("selected_validation_threshold"),
            "inference-card threshold",
        )
        == selected
        and inference.get("selection_frozen_before_test_inference") is True
        and inference.get("test_labels_opened") is False
        and isinstance(identities, Mapping)
        and identities.get("source_config_sha256") == config_sha256
        and identities.get("manifest_sha256") == manifest_sha256
        and identities.get("split_sha256") == splits_sha256
        and identities.get("selector_receipt_sha256") == selector_sha256
        and identities.get("test_probability_card_sha256")
        == probability_card_sha256
        and identities.get("amendment_sha256") == amendment_sha256,
        "threshold-bound inference card does not bind the scoring inputs",
    )
    checkpoint_sha256 = _sha(
        identities.get("checkpoint_sha256"), "inference checkpoint digest"
    )
    require(
        _sha(card.get("checkpoint_sha256"), "test-card checkpoint digest")
        == checkpoint_sha256,
        "test probability card and inference card use different checkpoints",
    )
    return (
        selected,
        {
            "source_config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "splits_sha256": splits_sha256,
            "test_probability_card_sha256": probability_card_sha256,
            "selector_receipt_sha256": selector_sha256,
            "amendment_sha256": amendment_sha256,
            "inference_card_sha256": inference_sha256,
        },
        card,
        checkpoint_sha256,
        {
            "task": task,
            "block_size": block_features.BLOCK if task == "patch" else None,
            "min_component_blocks": min_component_blocks,
        },
    )


def _card_per_image_records(card: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    raw = card.get("per_image")
    if not isinstance(raw, list):
        raw = card.get("predictions")
    require(isinstance(raw, list), "test probability card lacks per-image mask records")
    result: Dict[str, Mapping[str, Any]] = {}
    for item in raw:
        require(isinstance(item, Mapping), "malformed per-image probability-card record")
        image_id = str(item.get("image_id", ""))
        require(image_id and image_id not in result, "duplicate per-image card identity")
        result[image_id] = item
    return result


def _exact_directory_files(root: Path, expected: set[str], description: str) -> None:
    observed: set[str] = set()
    for path in root.iterdir():
        require(not path.is_symlink(), f"symlink in {description}: {path.name}")
        require(path.is_file(), f"non-file in {description}: {path.name}")
        observed.add(path.name)
    require(
        observed == expected,
        f"{description} universe changed; missing={sorted(expected - observed)}, "
        f"unexpected={sorted(observed - expected)}",
    )


def verify_prediction_probability_derivation(
    *,
    card: Mapping[str, Any],
    prediction_dir: Path,
    probability_dir: Path,
    threshold: float,
    protocol: Mapping[str, Any],
    expected_shapes: Mapping[str, tuple[int, int]],
) -> tuple[int, str]:
    """Re-derive every binary mask and reproduce the card's record digest."""
    probability_set = card.get("probability_set")
    require(isinstance(probability_set, Mapping), "test card lacks a probability set")
    records = probability_set.get("records")
    require(
        probability_set.get("format") == "float16_npy"
        and probability_set.get("storage_dtype") == "float16"
        and probability_set.get("complete") is True
        and isinstance(records, list)
        and bool(records),
        "test probability set is incomplete or has the wrong storage format",
    )
    count = probability_set.get("count")
    require(type(count) is int and count == len(records), "probability-set count changed")
    image_ids: list[str] = []
    for record in records:
        require(isinstance(record, Mapping), "malformed probability-set record")
        image_id = str(record.get("image_id", ""))
        require(
            SAFE_IMAGE_ID.fullmatch(image_id) is not None and image_id not in {".", ".."},
            f"unsafe probability image id: {image_id!r}",
        )
        image_ids.append(image_id)
    require(
        image_ids == sorted(image_ids) and len(set(image_ids)) == len(image_ids),
        "probability records are not uniquely sorted by image id",
    )
    require(
        set(image_ids) == set(expected_shapes),
        "probability universe differs from the manifest/split test universe",
    )
    _exact_directory_files(
        prediction_dir,
        {f"{image_id}.png" for image_id in image_ids},
        "prediction-mask directory",
    )
    _exact_directory_files(
        probability_dir,
        {f"{image_id}.npy" for image_id in image_ids},
        "probability-map directory",
    )
    per_image = _card_per_image_records(card)
    require(set(per_image) == set(image_ids), "test-card per-image universe changed")

    canonical_probability_records = []
    derivation_records = []
    for record, image_id in zip(records, image_ids):
        probability_path = probability_dir / f"{image_id}.npy"
        mask_path = prediction_dir / f"{image_id}.png"
        require(
            record.get("filename") == probability_path.name
            and Path(str(record.get("path", ""))).resolve() == probability_path.resolve(),
            f"probability-card path changed for {image_id}",
        )
        probability_sha256 = sha256_file(probability_path)
        mask_sha256 = sha256_file(mask_path)
        probability_bytes = probability_path.stat().st_size
        require(
            record.get("sha256") == probability_sha256
            and record.get("bytes") == probability_bytes,
            f"retained probability bytes changed for {image_id}",
        )
        evidence = per_image[image_id]
        expected_mask_sha = evidence.get("mask_sha256", evidence.get("sha256"))
        require(
            expected_mask_sha == mask_sha256
            and evidence.get("probability_sha256") == probability_sha256,
            f"retained mask/probability identity changed for {image_id}",
        )
        try:
            stored = np.load(probability_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ScoringError(f"invalid retained probability map: {image_id}") from exc
        require(
            stored.ndim == 2
            and stored.dtype == np.float16
            and np.isfinite(stored).all()
            and not bool((stored < 0).any())
            and not bool((stored > 1).any()),
            f"invalid retained probability values for {image_id}",
        )
        require(
            record.get("shape") == list(stored.shape)
            and record.get("dtype") == "float16",
            f"probability-card array metadata changed for {image_id}",
        )
        saved = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        require(
            saved is not None
            and saved.ndim == 2
            and tuple(saved.shape) == tuple(expected_shapes[image_id])
            and set(np.unique(saved).tolist()).issubset({0, 255}),
            f"invalid retained binary mask for {image_id}",
        )
        task = str(protocol.get("task", ""))
        if task == "pixel":
            require(
                tuple(stored.shape) == tuple(saved.shape),
                f"pixel probability shape differs from mask for {image_id}",
            )
            derived = stored.astype(np.float32) >= threshold
        elif task == "patch":
            expected_grid = block_features.grid_shape(*saved.shape)
            require(
                tuple(stored.shape) == tuple(expected_grid),
                f"patch probability grid differs from mask shape for {image_id}",
            )
            minimum = protocol.get("min_component_blocks")
            require(
                type(minimum) is int and minimum >= 1,
                "patch postprocessing protocol is invalid",
            )
            block_mask = postprocess_blocks(
                stored.astype(np.float32) >= threshold,
                min_component_blocks=minimum,
            )
            derived = block_features.blocks_to_mask(block_mask, saved.shape)
        else:
            raise ScoringError(f"unsupported probability task: {task!r}")
        derived = derived.astype(np.uint8) * 255
        require(
            np.array_equal(derived, saved),
            f"retained probability does not reproduce mask for {image_id}",
        )
        canonical_probability_records.append(
            {
                "image_id": image_id,
                "sha256": probability_sha256,
                "bytes": probability_bytes,
            }
        )
        derivation_records.append(
            {
                "image_id": image_id,
                "mask_sha256": mask_sha256,
                "probability_sha256": probability_sha256,
            }
        )
    require(
        probability_set.get("sha256")
        == canonical_sha256(canonical_probability_records),
        "probability-set canonical digest changed",
    )
    derivation_sha256 = canonical_sha256(derivation_records)
    derivation = card.get("mask_derivation_verification")
    require(
        isinstance(derivation, Mapping)
        and derivation.get("complete") is True
        and derivation.get("count") == count
        and derivation.get("all_masks_rederived_from_retained_probabilities") is True
        and derivation.get("records_sha256") == derivation_sha256,
        "probability-to-mask derivation differs from the test card",
    )
    return count, derivation_sha256


def _safe(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "none"


def _evaluation_identity_suffix(run_id: str, training_data: str) -> str:
    encoded = json.dumps(
        {"run_id": run_id, "training_data": training_data},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "_identity-" + hashlib.sha256(encoded).hexdigest()


def _per_image_path(
    results_dir: Path,
    method: str,
    seed: int,
    table: str,
    run_id: str,
    training_data: str,
) -> Path:
    identity = _evaluation_identity_suffix(run_id, training_data)
    return results_dir / (
        f"per_image_{_safe(method)}_finger_latent_{_safe(table)}_seed{seed}"
        f"{identity}.csv"
    )


def _summary_rows(
    summary_path: Path,
    *,
    method: str,
    seed: int,
    table: str,
    gt_column: str,
    run_id: str,
    training_data: str,
) -> tuple[int, str]:
    """Hash exact row strings without converting or trusting metric fields."""
    require(
        summary_path.is_file() and not summary_path.is_symlink(),
        f"evaluator summary is absent: {summary_path}",
    )
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, "evaluator summary has no header")
            structural = {
                "method",
                "seed",
                "table",
                "subset",
                "split",
                "ground_truth",
                "group_by",
                "group_value",
                "run_id",
                "training_data",
            }
            require(
                structural.issubset(reader.fieldnames),
                "evaluator summary lacks structural columns",
            )
            rows = [
                dict(row)
                for row in reader
                if str(row.get("method", "")) == method
                and str(row.get("seed", "")) == str(seed)
                and str(row.get("table", "")) == table
            ]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ScoringError(f"cannot read evaluator summary: {exc}") from exc
    require(rows, f"evaluator summary lacks generated {table} rows")
    expected = {
        "method": method,
        "seed": str(seed),
        "table": table,
        "subset": "finger_latent",
        "split": "test",
        "ground_truth": gt_column,
        "group_by": "examiner_value",
        "run_id": run_id,
        "training_data": training_data,
    }
    for row in rows:
        require(
            all(str(row.get(key, "")) == value for key, value in expected.items()),
            f"{table} summary contains a row not generated by this invocation",
        )
    groups = [str(row.get("group_value", "")) for row in rows]
    require(
        "overall" in groups and len(groups) == len(set(groups)),
        f"{table} summary has an invalid aggregate-row universe",
    )
    rows.sort(key=lambda row: tuple(str(row.get(key, "")) for key in sorted(row)))
    return len(rows), canonical_sha256(rows)


def _evaluator_argv(
    *,
    evaluator: Path,
    manifest: Path,
    splits: Path,
    prediction_dir: Path,
    results_dir: Path,
    method: str,
    seed: int,
    display_name: str,
    year: str,
    family: str,
    training_data: str,
    comment: str,
    params_m: float,
    device: str,
    threshold: float,
    run_id: str,
    ms_per_img: float,
    table: str,
    gt_column: str,
) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-B",
        "-c",
        EVALUATOR_BOOTSTRAP,
        str(evaluator),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--subsets",
        "finger_latent",
        "--method",
        method,
        "--display-name",
        display_name,
        "--pred-template",
        str(prediction_dir / "{image_id}.png"),
        "--pred-ppi",
        "500",
        "--target-ppi",
        "500",
        "--split",
        "test",
        "--gt-column",
        gt_column,
        "--exclude",
        "examiner_value=NONPRINT",
        "--group-by",
        "examiner_value",
        "--seed",
        str(seed),
        "--year",
        year,
        "--family",
        family,
        "--training-data",
        training_data,
        "--comment",
        comment,
        "--params-m",
        str(params_m),
        "--device",
        device,
        "--threshold",
        str(threshold),
        "--run-id",
        run_id,
        "--table",
        table,
        "--ms-per-img",
        str(ms_per_img),
        "--out",
        str(results_dir),
    ]


def _publish_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Durably create one canonical JSON receipt without replacing any inode."""
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    require(hasattr(os, "O_NOFOLLOW"), "symlink-safe receipt publication is unavailable")
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        observed = _read_regular_bytes_nofollow(path, "existing scoring receipt")
        require(observed == encoded, f"refusing to replace receipt: {path}")
        return
    created_metadata = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        published_metadata = os.stat(path, follow_symlinks=False)
        require(
            stat.S_ISREG(published_metadata.st_mode)
            and (published_metadata.st_dev, published_metadata.st_ino)
            == (created_metadata.st_dev, created_metadata.st_ino),
            f"scoring receipt path changed during publication: {path}",
        )
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | os.O_NOFOLLOW,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        _unlink_owned_regular(path, created_metadata)
        raise


def _read_regular_bytes_nofollow(path: Path, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ScoringError(f"cannot open {description}: {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), f"{description} is not regular: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        current = os.stat(path, follow_symlinks=False)
        require(
            stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == (metadata.st_dev, metadata.st_ino),
            f"{description} path changed while it was read: {path}",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _unlink_owned_regular(path: Path, owned: os.stat_result) -> None:
    """Clean up only the exact regular inode created by this process."""
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    require(
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino),
        f"refusing to remove a replaced scoring receipt: {path}",
    )
    path.unlink()


def score(args: argparse.Namespace) -> Dict[str, Any]:
    restricted_root = args.restricted_root.resolve()
    require(
        args.restricted_root.is_dir() and not args.restricted_root.is_symlink(),
        f"restricted root is not a regular directory: {args.restricted_root}",
    )
    producer, evaluator = committed_code_identity(
        args.repo_root,
        args.expected_git_commit,
        args.expected_wrapper_sha256,
        args.expected_evaluator_sha256,
    )
    manifest = _regular(args.manifest, restricted_root, "manifest")
    splits = _regular(args.splits, restricted_root, "subject split")
    config_path = _regular(args.config, restricted_root, "source configuration")
    probability_card_path = _regular(
        args.test_probability_card, restricted_root, "test probability card"
    )
    selector_path = _regular(
        args.selector_receipt, restricted_root, "threshold selector receipt"
    )
    amendment_path = _regular(
        args.threshold_amendment, restricted_root, "threshold-policy amendment"
    )
    inference_path = _regular(
        args.inference_card, restricted_root, "threshold-bound inference card"
    )
    prediction_dir = _directory(
        args.prediction_dir, restricted_root, "prediction-mask directory"
    )
    probability_dir = _directory(
        args.probability_dir, restricted_root, "probability-map directory"
    )
    results_dir = _prepare_directory(args.results_dir, restricted_root, "results directory")
    receipt_out = _output_path(args.receipt_out, restricted_root, "scoring receipt")
    require(
        not receipt_out.exists(),
        "scoring receipt already exists; refusing to execute the test evaluator twice",
    )
    require(
        not any(results_dir.iterdir()),
        "results directory is not empty; refusing an unverifiable prior score",
    )

    manifest_sha256 = sha256_file(manifest)
    splits_sha256 = sha256_file(splits)
    expected_shapes = _test_target_shapes(manifest, splits)
    selected, identities, probability_card, run_id, derivation_protocol = (
        _validate_control_cards(
            method=args.method,
            seed=args.seed,
            manifest_sha256=manifest_sha256,
            splits_sha256=splits_sha256,
            config_path=config_path,
            probability_card_path=probability_card_path,
            selector_path=selector_path,
            amendment_path=amendment_path,
            inference_path=inference_path,
        )
    )
    prediction_count, derivation_sha256 = verify_prediction_probability_derivation(
        card=probability_card,
        prediction_dir=prediction_dir,
        probability_dir=probability_dir,
        threshold=selected,
        protocol=derivation_protocol,
        expected_shapes=expected_shapes,
    )
    timing_value = probability_card.get(
        "mean_ms_per_img", probability_card.get("ms_per_img")
    )
    require(
        isinstance(timing_value, (int, float))
        and not isinstance(timing_value, bool)
        and math.isfinite(float(timing_value))
        and float(timing_value) >= 0.0,
        "test probability card lacks a finite nonnegative inference time",
    )
    device_value = str(probability_card.get("device", "")).upper()
    device = "GPU" if device_value in {"CUDA", "GPU"} else device_value
    require(device in {"CPU", "GPU"}, "test probability card has an invalid device")
    require(
        math.isfinite(args.params_m) and args.params_m >= 0.0,
        "--params-m must be finite and nonnegative",
    )

    completed_invocations: list[
        tuple[str, str, list[str], subprocess.CompletedProcess[bytes]]
    ] = []
    child_environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in ("LANG", "LC_ALL", "LD_LIBRARY_PATH"):
        if os.environ.get(name):
            child_environment[name] = os.environ[name]
    for table, gt_column in TABLES:
        argv = _evaluator_argv(
            evaluator=evaluator,
            manifest=manifest,
            splits=splits,
            prediction_dir=prediction_dir,
            results_dir=results_dir,
            method=args.method,
            seed=args.seed,
            display_name=args.display_name,
            year=args.year,
            family=args.family,
            training_data=args.training_data,
            comment=args.comment,
            params_m=args.params_m,
            device=device,
            threshold=selected,
            run_id=run_id,
            ms_per_img=float(timing_value),
            table=table,
            gt_column=gt_column,
        )
        completed = subprocess.run(
            argv,
            cwd=args.repo_root.resolve(),
            env=child_environment,
            check=False,
            capture_output=True,
        )
        require(
            completed.returncode == 0,
            f"{table} evaluator exited {completed.returncode}: "
            + completed.stderr.decode("utf-8", errors="replace").strip(),
        )
        completed_invocations.append((table, gt_column, argv, completed))

    # Re-authenticate every non-label input after both evaluator processes.
    require(
        sha256_file(manifest) == manifest_sha256
        and sha256_file(splits) == splits_sha256
        and sha256_file(probability_card_path)
        == identities["test_probability_card_sha256"]
        and sha256_file(selector_path) == identities["selector_receipt_sha256"]
        and sha256_file(amendment_path) == identities["amendment_sha256"]
        and sha256_file(inference_path) == identities["inference_card_sha256"]
        and sha256_file(config_path) == identities["source_config_sha256"],
        "a scoring input changed during evaluator execution",
    )
    final_count, final_derivation_sha256 = verify_prediction_probability_derivation(
        card=probability_card,
        prediction_dir=prediction_dir,
        probability_dir=probability_dir,
        threshold=selected,
        protocol=derivation_protocol,
        expected_shapes=expected_shapes,
    )
    require(
        final_count == prediction_count
        and final_derivation_sha256 == derivation_sha256,
        "prediction/probability evidence changed during scoring",
    )

    invocations = []
    summary_path = results_dir / "summary.csv"
    for table, gt_column, argv, completed in completed_invocations:
        per_image_path = _per_image_path(
            results_dir,
            args.method,
            args.seed,
            table,
            run_id,
            args.training_data,
        )
        require(
            per_image_path.is_file() and not per_image_path.is_symlink(),
            f"{table} evaluator did not create its exact per-image CSV",
        )
        scientific_row_count, scientific_rows_sha256 = _summary_rows(
            summary_path,
            method=args.method,
            seed=args.seed,
            table=table,
            gt_column=gt_column,
            run_id=run_id,
            training_data=args.training_data,
        )
        invocations.append(
            {
                "table": table,
                "argv": argv,
                "exit_code": completed.returncode,
                "per_image": {
                    "path": str(per_image_path.resolve()),
                    "sha256": sha256_file(per_image_path),
                },
                "scientific_rows_sha256": scientific_rows_sha256,
                "scientific_row_count": scientific_row_count,
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            }
        )

    receipt: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "method": args.method,
        "seed": args.seed,
        "selected_threshold": selected,
        "selection_frozen_before_test_scoring": True,
        "test_labels_opened_only_by_evaluator": True,
        "manifest_path": str(manifest),
        "splits_path": str(splits),
        "identities": identities,
        "producer": producer,
        "invocations": invocations,
        "scientific_summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
        },
        "prediction_probability_derivation_records_sha256": derivation_sha256,
        "prediction_count": prediction_count,
    }
    _publish_json_once(receipt_out, receipt)
    return receipt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--restricted-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-probability-card", required=True, type=Path)
    parser.add_argument("--selector-receipt", required=True, type=Path)
    parser.add_argument("--threshold-amendment", required=True, type=Path)
    parser.add_argument("--inference-card", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--probability-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--receipt-out", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--year", default="")
    parser.add_argument("--family", default="")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--comment", default="")
    parser.add_argument("--params-m", required=True, type=float)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    receipt = score(parse_args(argv))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
