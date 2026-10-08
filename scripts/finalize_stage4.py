#!/usr/bin/env python3
"""Fail-closed Stage 4 input locking, preflight, and output publication.

The Stage 4 analysis is deliberately split into three transactions:

``lock``
    Authenticate the preregistered analysis plan and immutable Stage 1/2/3
    deliveries, independently replay the bound probability-to-mask and metric
    calculations as an integrity check, then freeze their identities.  The
    replay is not used for model selection.  Stage 2 is provenance-only; its
    summary is never added to the Stage 4 scientific inputs.

``preflight``
    Re-authenticate the input lock and validate the row *universe* of the
    Stage 1 unaffected, Stage 1 replacement, and Stage 3 summaries:
    method, seed, table, group, image count, and missing-prediction count.
    Metric values were already integrity-replayed while authenticating the
    input lock; neither pass performs model selection.

``publish``
    Authenticate an explicit output contract and a successful preflight
    receipt, copy exactly the contracted files into a temporary directory,
    and atomically publish a write-once delivery bundle.

This module never performs inference, publishes new held-out scores, performs
model selection, or carries out statistical analysis.  It only verifies the
already-frozen scientific outputs by independently replaying their arithmetic;
new scoring remains the responsibility of the committed evaluator wrapper,
and statistics remain in the separately reviewed Stage 4 statistics module.
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import _path  # noqa: F401
import yaml
from segbench import features as block_features
from segbench.artifact_identity import canonical_sha256, sha256_file
from segbench.io import read_gray, read_mask, to_target_ppi
from segbench.metrics import image_metrics, summarise
from segbench.postprocess import postprocess_blocks
from segbench.preprocess_cache import (
    implementation_sha256 as cache_implementation_sha256,
    image_path as cache_image_path,
    mask_path as cache_mask_path,
)


PLAN_KIND = "stage4_preregistered_analysis_plan"
LOCK_KIND = "stage4_frozen_input_lock"
PREFLIGHT_KIND = "stage4_summary_universe_preflight"
CONTRACT_KIND = "stage4_output_contract"
DELIVERY_KIND = "stage4_analysis_delivery"
STAGE2_KIND = "stage2_unet_heldout_delivery"
STAGE2_RESELECTION_KIND = "stage2_threshold_reselection_delivery"
STAGE3_KIND = "stage3_native_heldout_delivery"
AMENDMENT_KIND = "stage4_threshold_policy_amendment"
STAGE1_RESELECTION_KIND = "stage1_threshold_reselection_delivery"
PROBABILITY_CARD_KIND = "complete_probability_map_card"
PROBABILITY_RETENTION_KIND = "released_probability_retention_attestation"
RELEASED_INFERENCE_CARD_KIND = "released_probability_inference_card"
TRAINABLE_INFERENCE_CARD_KIND = "threshold_bound_inference_card"
SELECTOR_RECEIPT_KIND = "expanded_validation_threshold_selector_receipt"
SCORING_RECEIPT_KIND = "superseding_threshold_scoring_receipt"
SCORING_EVALUATOR_BOOTSTRAP = (
    "import os,runpy,sys;"
    "p=sys.argv.pop(1);"
    "sys.path.insert(0,os.path.dirname(p));"
    "runpy.run_path(p,run_name='__main__')"
)
PER_EPOCH_PROFILE_KIND = "per_epoch_training_profile_card"
LEGACY_COMPLETION_EVIDENCE_KIND = (
    "legacy_completion_before_threshold_amendment_evidence"
)

REGISTERED_PLAN_SHA256 = (
    "770e79852fcb4e6a78599aaa1f595fda4c27d0436ac84651a731c7661d02d22d"
)
REGISTERED_AMENDMENT_SHA256 = (
    "bc03e127606fae78b1fb727b74852c2ec92b945a082ca890549d106cb932780a"
)

PRIOR_THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
REPLACEMENT_THRESHOLD_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

TABLES = ("A", "A_prime")
NATIVE_METHODS = (
    "unet",
    "unetpp",
    "drunet",
    "ra_runet",
    "lenet",
    "alexnet",
    "cnn_ae",
    "segformer_sub",
)
TRAINABLE_NEURAL_METHODS = (
    "sufs_ft",
    *NATIVE_METHODS,
)
RELEASED_NEURAL_METHODS = ("sufs", "fingernet")
PIXEL_NEURAL_METHODS = (
    "sufs_ft",
    "unet",
    "unetpp",
    "drunet",
    "ra_runet",
    "segformer_sub",
    *RELEASED_NEURAL_METHODS,
)
PATCH_NEURAL_METHODS = ("lenet", "alexnet", "cnn_ae")
BLOCK_SIZE = 16
PROTOCOL_DATALOADER = {
    "num_workers": 8,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
}
METRICS = ("accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr")
LOWER_HEX = set("0123456789abcdef")


class Stage4Error(RuntimeError):
    """A fail-closed Stage 4 orchestration error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage4Error(message)


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


def _read_json(
    path: Path, description: str, *, expected_sha256: Optional[str] = None
) -> Dict[str, Any]:
    try:
        encoded = _read_regular_bytes_nofollow(path, description)
        if expected_sha256 is not None:
            require(
                hashlib.sha256(encoded).hexdigest() == expected_sha256,
                f"{description} bytes changed",
            )
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage4Error(f"cannot read {description}: {path}: {exc}") from exc
    require(isinstance(value, dict), f"{description} is not a JSON object")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _read_regular_bytes_nofollow(path: Path, description: str) -> bytes:
    """Read one immutable file without ever following a final-component symlink."""
    require(hasattr(os, "O_NOFOLLOW"), "symlink-safe immutable reads are unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage4Error(f"cannot open {description} as a regular file: {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), f"{description} is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise Stage4Error(
                f"{description} path changed while it was read: {path}: {exc}"
            ) from exc
        require(
            stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == (metadata.st_dev, metadata.st_ino),
            f"{description} path changed while it was read: {path}",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _publish_json_once(path: Path, value: Mapping[str, Any]) -> str:
    """Create a deterministic JSON file exactly once, allowing exact replay."""
    encoded = _canonical_json_bytes(value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Stage4Error(f"cannot create artifact parent directory: {path.parent}: {exc}") from exc
    require(
        path.parent.is_dir() and not path.parent.is_symlink(),
        f"artifact parent is absent or a symlink: {path.parent}",
    )
    require(hasattr(os, "O_NOFOLLOW"), "symlink-safe immutable publication is unavailable")
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
        observed = _read_regular_bytes_nofollow(path, "existing immutable artifact")
        require(observed == encoded, f"refusing to replace non-identical artifact: {path}")
        return "verified_identical"
    except OSError as exc:
        raise Stage4Error(f"cannot create immutable artifact: {path}: {exc}") from exc
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
            f"immutable artifact path changed during publication: {path}",
        )
    except BaseException:
        try:
            current = os.stat(path, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (
                created_metadata.st_dev,
                created_metadata.st_ino,
            ):
                path.unlink()
        except FileNotFoundError:
            pass
        raise
    return "created"


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any destination type."""
    require(source.is_dir() and not source.is_symlink(), f"publication stage is not a directory: {source}")
    require(
        destination.parent.is_dir() and not destination.parent.is_symlink(),
        f"publication parent is absent or a symlink: {destination.parent}",
    )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), str(destination))
        raise Stage4Error(
            f"atomic no-replace publication failed for {destination}: {os.strerror(error)}"
        )
    require(
        destination.is_dir() and not destination.is_symlink(),
        f"published destination is not a regular directory: {destination}",
    )


def _unlink_owned_regular(path: Path, owned: os.stat_result) -> None:
    """Remove only the exact regular file created by this process."""
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    require(
        stat.S_ISREG(current.st_mode)
        and (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino),
        f"refusing to remove a replaced publication lock: {path}",
    )
    path.unlink()


def _require_under(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise Stage4Error(f"{description} escapes the restricted root: {path}") from exc
    return resolved


def _require_regular(path: Path, root: Path, description: str) -> Path:
    resolved = _require_under(path, root, description)
    require(path.exists(), f"missing {description}: {path}")
    require(not path.is_symlink(), f"{description} must not be a symlink: {path}")
    require(path.is_file(), f"{description} is not a regular file: {path}")
    return resolved


def _identity(path: Path, root: Path, description: str) -> Dict[str, Any]:
    resolved = _require_regular(path, root, description)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _safe_relative(value: object, description: str) -> str:
    require(isinstance(value, str) and bool(value), f"{description} is empty")
    candidate = Path(str(value))
    require(not candidate.is_absolute(), f"{description} must be relative")
    require(".." not in candidate.parts and "." not in candidate.parts, f"unsafe {description}: {value}")
    normalized = candidate.as_posix()
    require(normalized not in {"", "delivery.json"}, f"reserved {description}: {value}")
    return normalized


def _directory_inventory(root: Path, *, exclude: Iterable[str] = ()) -> Dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), f"artifact root is not a regular directory: {root}")
    excluded = set(exclude)
    records: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise Stage4Error(f"symlink in immutable artifact tree: {relative}")
        if path.is_dir():
            continue
        require(path.is_file(), f"non-regular object in immutable artifact tree: {relative}")
        if relative not in excluded:
            records[relative] = sha256_file(path)
    return records


def _validate_amendment(
    path: Path,
    expected_sha256: str,
    plan: Mapping[str, Any],
    plan_sha256: str,
    restricted_root: Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Authenticate the sole prospective exception to the frozen plan."""
    identity = _identity(path, restricted_root, "threshold-policy amendment")
    require(
        identity["sha256"] == _sha(expected_sha256, "registered threshold amendment digest"),
        "threshold-policy amendment differs from external registration",
    )
    amendment = _read_json(
        Path(identity["path"]),
        "threshold-policy amendment",
        expected_sha256=identity["sha256"],
    )
    require(
        amendment.get("schema_version") == 1
        and amendment.get("kind") == AMENDMENT_KIND,
        "threshold-policy amendment schema or kind changed",
    )
    require(
        amendment.get("base_plan_sha256") == plan_sha256,
        "threshold-policy amendment does not bind the frozen analysis plan",
    )
    require(
        amendment.get("supersedes_exactly") == ["threshold_policy"],
        "threshold-policy amendment attempts to supersede another plan field",
    )
    prior = amendment.get("prior_policy")
    replacement = amendment.get("replacement_policy")
    require(
        isinstance(prior, Mapping) and prior == plan.get("threshold_policy"),
        "threshold-policy amendment prior policy differs from frozen plan",
    )
    require(
        tuple(prior.get("allowed_trainable_grid", ())) == PRIOR_THRESHOLD_GRID,
        "threshold-policy amendment prior grid changed",
    )
    require(isinstance(replacement, Mapping), "threshold amendment lacks replacement policy")
    require(
        tuple(replacement.get("allowed_trainable_grid", ()))
        == REPLACEMENT_THRESHOLD_GRID,
        "threshold-policy amendment replacement grid changed",
    )
    required_replacement = {
        "A_and_A_prime_reuse_identical_predictions": True,
        "binary_masks_accepted_as_probability_maps": False,
        "completed_trainable_pixel_models_reselected": True,
        "released_zero_shot_thresholds_retuned": False,
        "selection_data": "validation_only",
        "source": "frozen validation-selected run cards",
        "stage4_inference_or_rescoring_permitted": False,
        "stage4_reselection_permitted": False,
        "test_masks_regenerated_from_saved_probability_maps": True,
    }
    for key, expected in required_replacement.items():
        require(replacement.get(key) == expected, f"threshold amendment replacement field changed: {key}")
    require(
        amendment.get("required_attestations")
        == {
            "legacy_binary_only_outputs_must_be_superseded": True,
            "probability_maps_required_for_val_and_test": True,
            "test_or_heldout_opened_for_selection": False,
            "threshold_reported_per_trainable_pixel_run": True,
            "validation_only": True,
        },
        "threshold amendment attestation requirements changed",
    )
    creation_scope = amendment.get("creation_scope")
    require(
        creation_scope
        == {
            "ground_truth_opened": False,
            "images_opened": False,
            "model_or_checkpoint_opened": False,
            "prediction_or_probability_maps_opened": False,
            "results_opened": False,
        },
        "threshold amendment was not created results-blind",
    )
    generator = amendment.get("generator")
    require(isinstance(generator, Mapping), "threshold amendment lacks generator identity")
    _commit(generator.get("git_commit"), "threshold amendment generator commit")
    _sha(generator.get("script_sha256"), "threshold amendment generator digest")
    authorization = amendment.get("authorization")
    require(
        isinstance(authorization, Mapping)
        and authorization.get("source") == "operator_instruction"
        and bool(str(authorization.get("reason", "")).strip())
        and bool(str(authorization.get("recorded_at_utc", "")).strip()),
        "threshold amendment lacks operator authorization",
    )
    return amendment, identity


def _identity_from_record(
    record: object,
    restricted_root: Path,
    description: str,
) -> tuple[Path, str]:
    require(isinstance(record, Mapping), f"{description} identity is not an object")
    path = Path(str(record.get("path", "")))
    resolved = _require_regular(path, restricted_root, description)
    expected = _sha(record.get("sha256"), f"{description} digest")
    require(sha256_file(resolved) == expected, f"{description} bytes changed")
    return resolved, expected


def _positive_manifest_dimension(value: object, description: str) -> int:
    """Return one manifest-pinned integral output dimension."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage4Error(f"{description} is not numeric") from exc
    require(
        math.isfinite(numeric) and numeric > 0.0 and numeric.is_integer(),
        f"{description} is not a positive integer",
    )
    return int(numeric)


def _manifest_split_probability_universes(
    manifest_path: Path,
    splits_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Derive val/test probability universes from frozen manifest and split bytes.

    Probability cards are evidence *about* this universe; they are never allowed
    to define it.  The source digest and canonical output shape are therefore
    read from the manifest selected through the subject-disjoint split mapping.
    """
    split_payload = _read_json(splits_path, "subject split")
    subject_splits = split_payload.get("splits")
    require(isinstance(subject_splits, Mapping), "subject split lacks its splits mapping")
    normalized_splits: Dict[str, str] = {}
    for subject, split in subject_splits.items():
        subject_id = str(subject)
        split_name = str(split)
        require(
            subject_id and split_name in {"train", "val", "test"},
            "subject split contains an invalid subject or partition",
        )
        normalized_splits[subject_id] = split_name

    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "image_id",
                "image_path",
                "subject_id",
                "subset",
                "ppi",
                "mask_path",
                "quality_mask_path",
                "source_sha256",
                "target_height",
                "target_width",
                "examiner_value",
            }
            require(
                reader.fieldnames is not None and required.issubset(reader.fieldnames),
                "manifest lacks fields needed to derive probability universes",
            )
            manifest_rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise Stage4Error(f"cannot read probability-universe manifest: {exc}") from exc

    result: Dict[str, Dict[str, Any]] = {}
    global_ids: set[str] = set()
    all_records: Dict[str, Dict[str, Any]] = {}
    manifest_path_root = manifest_path.parent
    for row in manifest_rows:
        image_id = str(row.get("image_id", ""))
        require(image_id and image_id not in global_ids, "manifest has duplicate/empty image_id")
        global_ids.add(image_id)
        source_sha = _sha(row.get("source_sha256"), f"{image_id} source digest")
        raw_ppi = row.get("ppi")
        try:
            ppi = float(raw_ppi)
        except (TypeError, ValueError) as exc:
            raise Stage4Error(f"{image_id} ppi is not numeric") from exc
        require(math.isfinite(ppi) and ppi > 0.0, f"{image_id} ppi is invalid")

        def normalized_manifest_path(column: str) -> str:
            raw = str(row.get(column, "")).strip()
            if not raw:
                return ""
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = manifest_path_root / candidate
            return str(candidate.resolve())

        all_records[image_id] = {
            "image_id": image_id,
            "source_sha256": source_sha,
            "subject_id": str(row.get("subject_id", "")),
            "examiner_value": str(row.get("examiner_value", "")),
            "ppi": ppi,
            "image_path": normalized_manifest_path("image_path"),
            "mask_path": normalized_manifest_path("mask_path"),
            "quality_mask_path": normalized_manifest_path("quality_mask_path"),
            "target_shape": [
                _positive_manifest_dimension(
                    row.get("target_height"), f"{image_id} target height"
                ),
                _positive_manifest_dimension(
                    row.get("target_width"), f"{image_id} target width"
                ),
            ],
            "mask_presence": {
                column: bool(str(row.get(column, "")).strip())
                for column in ("mask_path", "quality_mask_path")
            },
        }
    result["all"] = {
        "count": len(all_records),
        "records": all_records,
        "image_id_set_sha256": canonical_sha256(sorted(all_records)),
    }
    for split in ("train", "val", "test"):
        records: Dict[str, Dict[str, Any]] = {}
        for row in manifest_rows:
            image_id = str(row.get("image_id", ""))
            if str(row.get("subset", "")) != "finger_latent":
                continue
            subject_id = str(row.get("subject_id", ""))
            require(subject_id in normalized_splits, f"manifest subject is absent from split: {subject_id}")
            if normalized_splits[subject_id] != split:
                continue
            if split == "test" and str(row.get("examiner_value", "")).upper() == "NONPRINT":
                continue
            # The shared selector and evaluator require ROI ground truth.  An
            # empty ROI row cannot legitimately enter either probability set.
            require(
                bool(str(row.get("mask_path", "")).strip()),
                f"{image_id}: {split} row lacks ROI ground truth",
            )
            records[image_id] = dict(all_records[image_id])
        require(records, f"manifest/split-derived {split} probability universe is empty")
        ordered_sources = [
            {
                "image_id": record["image_id"],
                "source_sha256": record["source_sha256"],
            }
            for record in sorted(records.values(), key=lambda item: item["image_id"])
        ]
        result[split] = {
            "count": len(records),
            "records": records,
            "image_id_set_sha256": canonical_sha256(sorted(records)),
            "ordered_source_images_sha256": canonical_sha256(ordered_sources),
        }
    return result


def _command_options(argv: object, description: str) -> tuple[str, Dict[str, str]]:
    """Parse a closed one-value CLI option set from process-start provenance."""
    require(isinstance(argv, list) and argv, f"{description} command argv is absent")
    values = [str(value) for value in argv]
    script_index = 0
    if Path(values[0]).name.startswith("python"):
        require(len(values) >= 2, f"{description} command lacks a script")
        script_index = 1
    script = Path(values[script_index]).name
    options: Dict[str, str] = {}
    index = script_index + 1
    if index < len(values) and not values[index].startswith("--"):
        options["__subcommand__"] = values[index]
        index += 1
    while index < len(values):
        flag = values[index]
        require(flag.startswith("--"), f"{description} contains a positional argument")
        require(flag not in options, f"{description} duplicates {flag}")
        if index + 1 >= len(values) or values[index + 1].startswith("--"):
            options[flag] = "true"
            index += 1
        else:
            options[flag] = values[index + 1]
            index += 2
    return script, options


def _validate_process_provenance(
    provenance: object,
    *,
    allowed_scripts: set[str],
    expected_paths: Mapping[str, Path],
    expected_values: Mapping[str, str],
    description: str,
) -> Dict[str, Any]:
    """Authenticate clean process-start provenance and its scientific argv."""
    require(isinstance(provenance, Mapping), f"{description} provenance is absent")
    commit = _commit(provenance.get("git_commit"), f"{description} Git commit")
    script, options = _command_options(provenance.get("command_argv"), description)
    require(
        provenance.get("capture_timing") == "process_start"
        and provenance.get("git_dirty") is False
        and script in allowed_scripts,
        f"{description} was not captured from a clean committed runner",
    )
    for flag, path in expected_paths.items():
        require(flag in options, f"{description} lacks {flag}")
        require(
            Path(options[flag]).resolve() == path.resolve(),
            f"{description} {flag} identity changed",
        )
    for flag, value in expected_values.items():
        require(options.get(flag) == value, f"{description} {flag} changed")
    for forbidden in (
        "--allow-unamended-debug",
        "--allow-sweep-checkpoint",
    ):
        require(forbidden not in options, f"{description} used {forbidden}")
    if "--limit" in options:
        require(options["--limit"] in {"0", "0.0"}, f"{description} is debug-limited")
    for flag in ("--limit-train", "--limit-val"):
        if flag in options:
            require(options[flag] in {"0", "0.0"}, f"{description} is debug-limited")
    return {"git_commit": commit, "script": script, "options": options}


def _validate_data_inputs(
    delivery: Mapping[str, Any],
    restricted_root: Path,
    description: str,
) -> tuple[Path, Path, str, str]:
    """Authenticate the manifest/split identities carried by a delivery."""
    inputs = delivery.get("data_inputs")
    require(isinstance(inputs, Mapping), f"{description} lacks data-input identities")
    manifest_path, manifest_sha = _identity_from_record(
        inputs.get("manifest"), restricted_root, f"{description} manifest"
    )
    splits_path, splits_sha = _identity_from_record(
        inputs.get("splits"), restricted_root, f"{description} subject split"
    )
    return manifest_path, splits_path, manifest_sha, splits_sha


def _validate_delivery_producer(
    delivery: Mapping[str, Any],
    *,
    expected_commit: str,
    restricted_root: Path,
    description: str,
    scientific_artifacts_validated: bool = False,
) -> Dict[str, Any]:
    """Bind a delivery to the committed builder and the immutable input spec."""
    producer = delivery.get("producer")
    require(isinstance(producer, Mapping), f"{description} lacks producer metadata")
    builder = Path(__file__).with_name("build_stage4_deliveries.py")
    require(builder.is_file(), "Stage 4 delivery builder source is absent")
    spec_path, spec_sha = _identity_from_record(
        producer.get("spec"), restricted_root, f"{description} producer spec"
    )
    del spec_path
    require(
        producer.get("git_commit") == expected_commit
        and producer.get("git_dirty") is False
        and producer.get("script_sha256") == sha256_file(builder)
        and producer.get("results_or_metric_values_parsed")
        is scientific_artifacts_validated
        and producer.get("payload_bytes_hashed") is True,
        f"{description} was not produced by the expected committed builder",
    )
    return {
        "git_commit": expected_commit,
        "script_sha256": sha256_file(builder),
        "spec_sha256": spec_sha,
    }


def _validate_probability_card(
    identity: object,
    *,
    split: str,
    method: str,
    seed: int,
    config_sha256: str,
    checkpoint_sha256: str,
    amendment_sha256: str,
    manifest_path: Path,
    splits_path: Path,
    manifest_sha256: str,
    splits_sha256: str,
    restricted_root: Path,
    expected_universes: Mapping[str, Mapping[str, Any]],
    validation_state: str = "selected",
) -> Dict[str, Any]:
    path, digest = _identity_from_record(
        identity, restricted_root, f"{method}/seed{seed} {split} probability card"
    )
    card = _read_json(
        path,
        f"{method}/seed{seed} {split} probability card",
        expected_sha256=digest,
    )
    require(
        card.get("schema_version") == 1
        and card.get("kind") == PROBABILITY_CARD_KIND,
        f"{method}/seed{seed}: invalid {split} probability-card schema",
    )
    require(card.get("method") == method and card.get("seed") == seed, f"{method}/seed{seed}: {split} probability-card identity changed")
    require(card.get("split") == split, f"{method}/seed{seed}: probability card has wrong split")
    require(card.get("source_config_sha256") == config_sha256, f"{method}/seed{seed}: {split} probability card config changed")
    require(
        card.get("checkpoint_sha256") == checkpoint_sha256,
        f"{method}/seed{seed}: {split} probability card checkpoint changed",
    )
    require(card.get("threshold_amendment_sha256") == amendment_sha256, f"{method}/seed{seed}: {split} probability card lacks amendment binding")
    require(
        card.get("manifest_sha256") == manifest_sha256
        and card.get("split_sha256") == splits_sha256,
        f"{method}/seed{seed}: {split} probability card data identity changed",
    )
    require(tuple(card.get("threshold_candidates", ())) == REPLACEMENT_THRESHOLD_GRID, f"{method}/seed{seed}: {split} probability card uses legacy grid")
    require(card.get("probabilities_are_pre_threshold") is True, f"{method}/seed{seed}: {split} maps are not pre-threshold probabilities")
    storage_dtype = card.get("storage_dtype")
    require(
        card.get("complete") is True
        and storage_dtype == "float16"
        and card.get("storage_format") == f"{storage_dtype}_npy"
        and card.get("derived_from_binary_mask") is False,
        f"{method}/seed{seed}: {split} probability storage violates the protocol",
    )
    if split == "val":
        require(
            validation_state in {"raw", "selected"},
            f"{method}/seed{seed}: invalid validation-card state",
        )
        if validation_state == "raw":
            require(
                card.get("labels_opened") is False
                and card.get("selection_source") == "unselected_raw"
                and "threshold_scores" not in card,
                f"{method}/seed{seed}: raw validation card contains selection evidence",
            )
        else:
            require(
                card.get("labels_opened") is True
                and card.get("selection_source") == "validation_only",
                f"{method}/seed{seed}: selected validation card lacks label-access evidence",
            )
            _, winning_threshold, winning_score = _validated_threshold_scores(
                card.get("threshold_scores"),
                f"{method}/seed{seed} validation threshold scores",
            )
            require(
                _selected_threshold(
                    card.get(
                        "selected_validation_threshold",
                        card.get("selected_threshold"),
                    ),
                    f"{method}/seed{seed} selected validation threshold",
                )
                == winning_threshold,
                f"{method}/seed{seed}: selected validation threshold is not the frozen argmax",
            )
            declared_score = card.get("selected_validation_dice", winning_score)
            require(
                isinstance(declared_score, (int, float))
                and not isinstance(declared_score, bool)
                and math.isfinite(float(declared_score))
                and math.isclose(float(declared_score), winning_score, rel_tol=0.0, abs_tol=1e-12),
                f"{method}/seed{seed}: selected validation Dice differs from threshold scores",
            )
    else:
        require(card.get("labels_opened") is False, f"{method}/seed{seed}: test probability generation opened labels")
        require(card.get("threshold_selected_before_test") is True, f"{method}/seed{seed}: test inference predates threshold selection")

    probability_summary = _validate_probability_set(
        card,
        split=split,
        method=method,
        seed=seed,
        restricted_root=restricted_root,
        expected_universes=expected_universes,
    )
    derivation = card.get("mask_derivation_verification")
    require(
        isinstance(derivation, Mapping)
        and derivation.get("complete") is True
        and int(derivation.get("count", -1)) == probability_summary["count"]
        and derivation.get("all_masks_rederived_from_retained_probabilities") is True,
        f"{method}/seed{seed}: {split} masks are not proven to derive from retained probabilities",
    )
    _sha(
        derivation.get("records_sha256"),
        f"{method}/seed{seed} {split} mask-derivation digest",
    )
    return {"path": str(path), "sha256": digest, **probability_summary}


def _validate_probability_set(
    card: Mapping[str, Any],
    *,
    split: str,
    method: str,
    seed: Optional[int],
    restricted_root: Path,
    expected_universes: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Authenticate probability bytes, semantics, source identity, and universe."""
    seed_label = "unseeded" if seed is None else f"seed{seed}"
    require(split in expected_universes, f"unknown expected probability split: {split}")
    expected = expected_universes[split]
    expected_records = expected.get("records")
    require(isinstance(expected_records, Mapping), f"invalid expected {split} universe")
    probability_set = card.get("probability_set")
    require(isinstance(probability_set, Mapping), f"{method}/{seed_label}: {split} probability set absent")
    records = probability_set.get("records")
    require(isinstance(records, list) and records, f"{method}/{seed_label}: empty {split} probability set")
    normalized = []
    source_records = []
    ids = set()
    for record in records:
        require(isinstance(record, Mapping), f"{method}/{seed_label}: malformed probability record")
        image_id = str(record.get("image_id", ""))
        require(image_id and image_id not in ids, f"{method}/{seed_label}: duplicate/empty {split} probability id")
        ids.add(image_id)
        probability_path = Path(str(record.get("path", "")))
        resolved = _require_regular(
            probability_path,
            restricted_root,
            f"{method}/{seed_label} {split} probability map {image_id}",
        )
        require(
            resolved.suffix.lower() == ".npy",
            f"{method}/{seed_label}: {split} probability map is not an NPY artifact",
        )
        probability_sha = _sha(record.get("sha256"), f"{method}/{seed_label} probability digest")
        require(sha256_file(resolved) == probability_sha, f"{method}/{seed_label}: {split} probability map changed: {image_id}")
        size = resolved.stat().st_size
        require(int(record.get("bytes", -1)) == size, f"{method}/{seed_label}: {split} probability-map size changed")
        require(image_id in expected_records, f"{method}/{seed_label}: unknown {split} probability id: {image_id}")
        expected_record = expected_records[image_id]
        require(isinstance(expected_record, Mapping), f"invalid expected identity for {image_id}")
        source_sha = _sha(
            record.get("source_sha256"),
            f"{method}/{seed_label} {split} source digest",
        )
        require(
            source_sha == expected_record.get("source_sha256"),
            f"{method}/{seed_label}: {split} source identity changed: {image_id}",
        )
        try:
            probability = np.load(resolved, allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise Stage4Error(
                f"{method}/{seed_label}: invalid {split} NPY probability map: {image_id}"
            ) from exc
        require(
            isinstance(probability, np.ndarray)
            and probability.ndim == 2
            and probability.dtype == np.float16,
            f"{method}/{seed_label}: {split} probability map must be 2-D float16: {image_id}",
        )
        require(
            np.isfinite(probability).all()
            and not bool((probability < 0.0).any())
            and not bool((probability > 1.0).any()),
            f"{method}/{seed_label}: {split} probability values are outside [0,1]: {image_id}",
        )
        declared_shape = record.get("shape")
        require(
            isinstance(declared_shape, list)
            and len(declared_shape) == 2
            and all(type(value) is int and value > 0 for value in declared_shape)
            and declared_shape == list(probability.shape)
            and record.get("dtype") == "float16",
            f"{method}/{seed_label}: {split} probability metadata differs from NPY: {image_id}",
        )
        target_shape = expected_record.get("target_shape")
        if method in PATCH_NEURAL_METHODS:
            target_shape = [
                (int(target_shape[0]) + BLOCK_SIZE - 1) // BLOCK_SIZE,
                (int(target_shape[1]) + BLOCK_SIZE - 1) // BLOCK_SIZE,
            ]
        require(
            list(probability.shape) == list(target_shape),
            f"{method}/{seed_label}: {split} probability shape differs from manifest: {image_id}",
        )
        normalized.append({"image_id": image_id, "sha256": probability_sha, "bytes": size})
        source_records.append({"image_id": image_id, "source_sha256": source_sha})
    normalized.sort(key=lambda item: item["image_id"])
    require(int(probability_set.get("count", -1)) == len(normalized), f"{method}/{seed_label}: {split} probability count changed")
    require(probability_set.get("sha256") == canonical_sha256(normalized), f"{method}/{seed_label}: {split} probability-set digest changed")
    require(
        ids == set(expected_records)
        and len(normalized) == int(expected.get("count", -1)),
        f"{method}/{seed_label}: {split} probability universe changed",
    )
    source_records.sort(key=lambda item: item["image_id"])
    source_digest = canonical_sha256(source_records)
    require(
        source_digest == expected.get("ordered_source_images_sha256")
        and card.get("ordered_source_images_sha256") == source_digest,
        f"{method}/{seed_label}: {split} probability image universe changed",
    )
    return {
        "count": len(normalized),
        "probability_set_sha256": probability_set["sha256"],
        "ordered_source_images_sha256": source_digest,
    }


def _validate_preprocess_cache_receipt(
    identity: object,
    *,
    manifest_sha256: str,
    expected_universes: Mapping[str, Mapping[str, Any]],
    restricted_root: Path,
    description: str,
) -> tuple[str, Dict[str, Any]]:
    """Validate the immutable 500-ppi cache and its 20-image identity gate."""
    path, digest = _identity_from_record(identity, restricted_root, description)
    receipt = _read_json(path, description, expected_sha256=digest)
    require(
        receipt.get("schema_version") == 1
        and receipt.get("kind") == "uint8_500ppi_preprocessing_cache"
        and receipt.get("manifest_sha256") == manifest_sha256
        and receipt.get("target_ppi") == 500.0
        and receipt.get("image_format") == "uint8_npy"
        and receipt.get("mask_format") == "uint8_0_255_npy"
        and receipt.get("debug_limited") is False,
        f"{description} schema/data identity changed",
    )
    mask_columns = receipt.get("mask_columns")
    require(
        isinstance(mask_columns, list)
        and set(mask_columns) == {"mask_path", "quality_mask_path"},
        f"{description} does not cover both ground-truth mask columns",
    )
    implementation = receipt.get("implementation_sha256")
    expected_implementation = cache_implementation_sha256()
    require(
        isinstance(implementation, Mapping)
        and set(implementation) == {"builder", "deep_utils", "io", "preprocess_cache"}
        and all(
            _sha(value, f"{description} {name} implementation digest")
            == expected_implementation[name]
            for name, value in implementation.items()
        ),
        f"{description} implementation identity is incomplete",
    )
    records = receipt.get("records")
    all_expected = expected_universes.get("all", {}).get("records")
    require(
        isinstance(records, list)
        and isinstance(all_expected, Mapping)
        and type(receipt.get("record_count")) is int
        and receipt.get("record_count") == len(records) == len(all_expected),
        f"{description} does not cover the complete manifest",
    )
    by_id: Dict[str, Mapping[str, Any]] = {}
    cache_root = path.parent
    for record in records:
        require(isinstance(record, Mapping), f"{description} contains a malformed record")
        image_id = str(record.get("image_id", ""))
        require(
            image_id in all_expected and image_id not in by_id,
            f"{description} contains an unknown/duplicate image id",
        )
        expected = all_expected[image_id]
        require(
            record.get("shape") == expected.get("target_shape")
            and record.get("source_image_sha256") == expected.get("source_sha256")
            and isinstance(record.get("mask_sha256"), Mapping)
            and set(record["mask_sha256"]) == set(mask_columns)
            and isinstance(record.get("source_mask_sha256"), Mapping)
            and set(record["source_mask_sha256"]) == set(mask_columns),
            f"{description} record identity changed: {image_id}",
        )
        cached_image_sha = _sha(
            record.get("image_sha256"), f"{description} cached image {image_id}"
        )
        cached_image_path = _require_regular(
            cache_image_path(cache_root, image_id),
            restricted_root,
            f"{description} cached image {image_id}",
        )
        require(
            sha256_file(cached_image_path) == cached_image_sha,
            f"{description} cached image bytes changed: {image_id}",
        )
        try:
            cached_image = np.load(cached_image_path, allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise Stage4Error(
                f"{description} cached image is not a valid NPY: {image_id}"
            ) from exc
        require(
            isinstance(cached_image, np.ndarray)
            and cached_image.ndim == 2
            and cached_image.dtype == np.uint8
            and list(cached_image.shape) == expected.get("target_shape"),
            f"{description} cached image shape/dtype changed: {image_id}",
        )
        for hash_group in (record["mask_sha256"], record["source_mask_sha256"]):
            for column, value in hash_group.items():
                if value is not None:
                    _sha(value, f"{description} {image_id}/{column} digest")
        require(
            all(
                (record["mask_sha256"][column] is None)
                == (record["source_mask_sha256"][column] is None)
                for column in mask_columns
            ),
            f"{description} cache/source mask presence differs: {image_id}",
        )
        require(
            all(
                (record["mask_sha256"][column] is not None)
                == bool(expected["mask_presence"][column])
                for column in mask_columns
            ),
            f"{description} manifest/cache mask presence differs: {image_id}",
        )
        for column in mask_columns:
            cached_mask_sha = record["mask_sha256"][column]
            if cached_mask_sha is None:
                continue
            cached_mask_path = _require_regular(
                cache_mask_path(cache_root, column, image_id),
                restricted_root,
                f"{description} cached mask {image_id}/{column}",
            )
            require(
                sha256_file(cached_mask_path) == cached_mask_sha,
                f"{description} cached mask bytes changed: {image_id}/{column}",
            )
            try:
                cached_mask = np.load(cached_mask_path, allow_pickle=False)
            except (OSError, ValueError, TypeError) as exc:
                raise Stage4Error(
                    f"{description} cached mask is not a valid NPY: "
                    f"{image_id}/{column}"
                ) from exc
            require(
                isinstance(cached_mask, np.ndarray)
                and cached_mask.ndim == 2
                and cached_mask.dtype == np.uint8
                and list(cached_mask.shape) == expected.get("target_shape")
                and set(np.unique(cached_mask).tolist()).issubset({0, 255}),
                f"{description} cached mask values/shape changed: "
                f"{image_id}/{column}",
            )
        by_id[image_id] = record
    require(set(by_id) == set(all_expected), f"{description} manifest universe changed")
    ordered_ids = [str(record["image_id"]) for record in records]
    require(
        ordered_ids == sorted(all_expected),
        f"{description} record order differs from the canonical manifest order",
    )
    ordered_digest = hashlib.sha256(
        "".join(f"{image_id}\n" for image_id in ordered_ids).encode("utf-8")
    ).hexdigest()
    require(
        receipt.get("ordered_image_ids_sha256") == ordered_digest,
        f"{description} ordered image universe changed",
    )
    verification = receipt.get("verification")
    required_verification_count = min(20, len(records))
    expected_verified_ids = sorted(
        ordered_ids,
        key=lambda value: (hashlib.sha256(value.encode("utf-8")).digest(), value),
    )[:required_verification_count]
    require(
        isinstance(verification, Mapping)
        and verification.get("algorithm") == "sha256(image_id), lexical tie-break"
        and verification.get("requested_count") == 20
        and verification.get("verified_count") == required_verification_count
        and verification.get("array_identity") is True
        and isinstance(verification.get("image_ids"), list)
        and list(map(str, verification["image_ids"])) == expected_verified_ids,
        f"{description} lacks the required 20-image byte-identity gate",
    )
    # Reproduce the advertised gate from the immutable source files.  A boolean
    # in the receipt is not evidence that cached preprocessing is equivalent.
    for image_id in expected_verified_ids:
        record = by_id[image_id]
        expected = all_expected[image_id]
        source_image_path = Path(str(expected.get("image_path", "")))
        require(
            source_image_path.is_file() and not source_image_path.is_symlink(),
            f"{description} source image is absent/non-regular: {image_id}",
        )
        require(
            sha256_file(source_image_path) == expected.get("source_sha256"),
            f"{description} source image bytes changed: {image_id}",
        )
        on_the_fly_image = to_target_ppi(
            read_gray(source_image_path), float(expected["ppi"]), 500.0
        ).astype(np.uint8, copy=False)
        cached_image = np.load(
            cache_image_path(cache_root, image_id), allow_pickle=False
        )
        require(
            np.array_equal(cached_image, on_the_fly_image),
            f"{description} cached image differs from on-the-fly preprocessing: "
            f"{image_id}",
        )
        for column in mask_columns:
            source_mask_value = str(expected.get(column, "")).strip()
            if not source_mask_value:
                continue
            source_mask_path = Path(source_mask_value)
            require(
                source_mask_path.is_file() and not source_mask_path.is_symlink(),
                f"{description} source mask is absent/non-regular: "
                f"{image_id}/{column}",
            )
            require(
                sha256_file(source_mask_path)
                == record["source_mask_sha256"][column],
                f"{description} source mask bytes changed: {image_id}/{column}",
            )
            on_the_fly_mask = (
                to_target_ppi(
                    read_mask(source_mask_path),
                    float(expected["ppi"]),
                    500.0,
                    is_mask=True,
                ).astype(np.uint8)
                * 255
            )
            cached_mask = np.load(
                cache_mask_path(cache_root, column, image_id), allow_pickle=False
            )
            require(
                np.array_equal(cached_mask, on_the_fly_mask),
                f"{description} cached mask differs from on-the-fly preprocessing: "
                f"{image_id}/{column}",
            )
    return digest, {
        "path": str(path),
        "sha256": digest,
        "record_count": len(records),
        "verified_count": required_verification_count,
    }


def _probability_to_mask(
    probability: np.ndarray,
    *,
    target_shape: tuple[int, int],
    threshold: float,
    method: str,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Apply the frozen inference geometry to one retained probability map."""
    if method in PIXEL_NEURAL_METHODS:
        require(
            tuple(probability.shape) == target_shape,
            f"{method}: pixel probability/target shape mismatch",
        )
        return probability.astype(np.float32) >= threshold
    require(method in PATCH_NEURAL_METHODS, f"unknown probability model: {method}")
    inference = config.get("inference")
    data = config.get("data")
    require(
        isinstance(inference, Mapping) and isinstance(data, Mapping),
        f"{method}: patch inference geometry is absent",
    )
    block_size = data.get("block_size", inference.get("block_size"))
    min_component_blocks = inference.get("min_component_blocks")
    require(
        type(block_size) is int
        and block_size == BLOCK_SIZE == block_features.BLOCK
        and type(min_component_blocks) is int
        and min_component_blocks >= 1
        and tuple(probability.shape) == block_features.grid_shape(*target_shape),
        f"{method}: patch inference geometry changed",
    )
    blocks = postprocess_blocks(
        probability.astype(np.float32) >= threshold,
        min_component_blocks=min_component_blocks,
    )
    return block_features.blocks_to_mask(blocks, target_shape)


def _recompute_validation_threshold_scores(
    card: Mapping[str, Any],
    *,
    method: str,
    config: Mapping[str, Any],
    expected_universes: Mapping[str, Mapping[str, Any]],
    restricted_root: Path,
) -> Dict[float, float]:
    """Recompute macro-image validation Dice from retained NPYs and ROI GT."""
    expected = expected_universes.get("val", {}).get("records")
    probability_set = card.get("probability_set")
    records = probability_set.get("records") if isinstance(probability_set, Mapping) else None
    require(
        isinstance(expected, Mapping) and isinstance(records, list),
        f"{method}: validation score evidence is incomplete",
    )
    by_id = {
        str(record.get("image_id", "")): record
        for record in records
        if isinstance(record, Mapping)
    }
    require(
        len(by_id) == len(records) and set(by_id) == set(expected),
        f"{method}: validation score universe changed",
    )
    entries: list[tuple[np.ndarray, np.ndarray]] = []
    for image_id in sorted(by_id):
        expected_record = expected[image_id]
        probability_path = _require_regular(
            Path(str(by_id[image_id].get("path", ""))),
            restricted_root,
            f"{method} validation probability {image_id}",
        )
        probability = np.load(probability_path, allow_pickle=False)
        mask_path = Path(str(expected_record.get("mask_path", "")))
        require(
            mask_path.is_file() and not mask_path.is_symlink(),
            f"{method}: validation GT is absent/non-regular: {image_id}",
        )
        target = to_target_ppi(
            read_mask(mask_path),
            float(expected_record["ppi"]),
            500.0,
            is_mask=True,
        )
        require(
            tuple(target.shape) == tuple(expected_record["target_shape"]),
            f"{method}: validation GT shape differs from manifest: {image_id}",
        )
        entries.append((probability, target))
    scores: Dict[float, float] = {}
    for threshold in REPLACEMENT_THRESHOLD_GRID:
        dice = []
        for probability, target in entries:
            prediction = _probability_to_mask(
                probability,
                target_shape=tuple(target.shape),
                threshold=threshold,
                method=method,
                config=config,
            )
            dice.append(float(image_metrics(prediction, target)["dice"]))
        scores[threshold] = float(np.mean(dice))
    return scores


def _validate_trainable_inference_provenance(
    card: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    split: str,
    expected_commit: str,
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    splits_path: Path,
    expected_count: int,
) -> Dict[str, Any]:
    """Bind retained NPYs to a complete clean committed inference command."""
    probability_set = card.get("probability_set")
    records = probability_set.get("records") if isinstance(probability_set, Mapping) else None
    require(isinstance(records, list) and records, f"{method}/seed{seed}: empty inference set")
    probability_parents = {
        Path(str(record.get("path", ""))).resolve().parent
        for record in records
        if isinstance(record, Mapping)
    }
    require(
        len(probability_parents) == 1,
        f"{method}/seed{seed}: inference probabilities span directories",
    )
    provenance = _validate_process_provenance(
        card.get("provenance"),
        allowed_scripts={"infer_sufs.py"} if method == "sufs_ft" else {"infer_masks.py"},
        expected_paths={
            "--config": config_path,
            "--checkpoint": checkpoint_path,
            "--manifest": manifest_path,
            "--splits": splits_path,
            "--probability-out": next(iter(probability_parents)),
        },
        expected_values={"--subset": "finger_latent", "--split": split},
        description=f"{method}/seed{seed} {split} inference",
    )
    require(
        provenance["git_commit"] == expected_commit,
        f"{method}/seed{seed}: {split} inference commit differs from frozen lineage",
    )
    require(
        card.get("status") == "complete"
        and card.get("complete") is True
        and card.get("debug_limited") is False
        and card.get("preempted") in {None, False}
        and (card.get("errors") is None or card.get("errors") == [])
        and int(probability_set.get("count", -1)) == expected_count,
        f"{method}/seed{seed}: {split} inference is partial/debug/failed",
    )
    options = provenance["options"]
    require(
        options.get("--seed", str(seed)) == str(seed),
        f"{method}/seed{seed}: {split} inference seed changed",
    )
    if split == "test" and provenance["script"] == "infer_masks.py":
        require(
            int(card.get("skipped_valid", -1)) == 0
            and int(card.get("written", -1)) == expected_count,
            f"{method}/seed{seed}: test timing was not a complete fresh pass",
        )
    timing_evidence: Optional[Dict[str, Any]] = None
    if split == "test":
        probability_ids = {
            str(record.get("image_id", ""))
            for record in records
            if isinstance(record, Mapping)
        }
        timing_records = (
            card.get("predictions") if method == "sufs_ft" else card.get("per_image")
        )
        require(
            isinstance(timing_records, list)
            and len(timing_records) == len(probability_ids),
            f"{method}/seed{seed}: test timing record count changed",
        )
        by_id: Dict[str, float] = {}
        for timing_record in timing_records:
            require(
                isinstance(timing_record, Mapping),
                f"{method}/seed{seed}: malformed test timing record",
            )
            image_id = str(timing_record.get("image_id", ""))
            value = timing_record.get("ms_per_img")
            require(
                image_id in probability_ids
                and image_id not in by_id
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0,
                f"{method}/seed{seed}: invalid/duplicate test timing record",
            )
            by_id[image_id] = float(value)
        require(
            set(by_id) == probability_ids,
            f"{method}/seed{seed}: test timing universe differs from probabilities",
        )
        declared_mean = card.get(
            "ms_per_img" if method == "sufs_ft" else "mean_ms_per_img"
        )
        recomputed_mean = float(np.mean([by_id[key] for key in sorted(by_id)]))
        require(
            isinstance(declared_mean, (int, float))
            and not isinstance(declared_mean, bool)
            and math.isfinite(float(declared_mean))
            and float(declared_mean) >= 0.0
            and math.isclose(
                float(declared_mean), recomputed_mean, rel_tol=0.0, abs_tol=1e-12
            ),
            f"{method}/seed{seed}: test timing mean differs from per-image records",
        )
        declared_warmup = (
            options.get("--warmup", "10")
            if method == "sufs_ft"
            else card.get("warmup_images")
        )
        try:
            warmup_images = int(declared_warmup)
        except (TypeError, ValueError) as exc:
            raise Stage4Error(
                f"{method}/seed{seed}: invalid test timing warm-up"
            ) from exc
        require(
            warmup_images == 10 and options.get("--warmup", "10") == "10",
            f"{method}/seed{seed}: test timing did not use the registered 10-image warm-up",
        )
        timing_evidence = {
            "count": len(by_id),
            "mean_ms_per_img": recomputed_mean,
            "warmup_images": warmup_images,
            "image_id_set_sha256": canonical_sha256(sorted(by_id)),
        }
    return {**provenance, "timing_evidence": timing_evidence}


def _validate_threshold_attestation(
    delivery: Mapping[str, Any],
    *,
    required_records: set[tuple[str, int]],
    legacy_records: set[tuple[str, int]],
    amendment_sha256: str,
    manifest_path: Path,
    splits_path: Path,
    manifest_sha256: str,
    splits_sha256: str,
    restricted_root: Path,
    expected_universes: Mapping[str, Mapping[str, Any]],
    scientific_summary_sha256: Optional[str],
    scientific_summary_path: Optional[Path],
    expected_producer_commit: str,
    description: str,
    expected_training_identities: Optional[
        Mapping[tuple[str, int], Mapping[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    """Bind amended runs and narrowly allow pre-amendment completed checkpoints."""
    require(legacy_records.issubset(required_records), f"{description} has an invalid legacy allowlist")
    attestation = delivery.get("threshold_policy_attestation")
    require(isinstance(attestation, Mapping), f"{description} lacks threshold-policy attestation")
    require(attestation.get("amendment_sha256") == amendment_sha256, f"{description} does not bind threshold amendment")
    require(tuple(attestation.get("allowed_grid", ())) == REPLACEMENT_THRESHOLD_GRID, f"{description} attests a legacy threshold grid")
    require(attestation.get("validation_only") is True, f"{description} threshold selection is not validation-only")
    require(attestation.get("old_grid_rejected") is True, f"{description} does not reject old-grid cards")
    records = attestation.get("records")
    require(isinstance(records, list), f"{description} threshold attestation lacks records")
    observed: set[tuple[str, int]] = set()
    checked = []
    for record in records:
        require(isinstance(record, Mapping), f"{description} has malformed threshold record")
        method, seed = str(record.get("method", "")), record.get("seed")
        require(method in TRAINABLE_NEURAL_METHODS and type(seed) is int, f"{description} has unexpected threshold record")
        key = (method, int(seed))
        require(key not in observed, f"{description} duplicates threshold record {key}")
        observed.add(key)
        lineage = record.get("training_lineage")
        expected_lineage = (
            "legacy_completed_before_amendment"
            if key in legacy_records
            else "amended_training"
        )
        require(lineage == expected_lineage, f"{method}/seed{seed}: invalid training-lineage exception")
        config_path, config_sha = _identity_from_record(
            record.get("config"), restricted_root, f"{method}/seed{seed} training config"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require(isinstance(config, Mapping), f"{method}/seed{seed}: config is not a mapping")
        training = config.get("training")
        expected_config_grid = (
            PRIOR_THRESHOLD_GRID if key in legacy_records else REPLACEMENT_THRESHOLD_GRID
        )
        require(
            isinstance(training, Mapping)
            and tuple(training.get("threshold_candidates", ())) == expected_config_grid,
            f"{method}/seed{seed}: training config grid differs from its declared lineage",
        )
        if key not in legacy_records:
            observed_loader = {
                name: training.get(name) for name in PROTOCOL_DATALOADER
            }
            require(
                observed_loader == PROTOCOL_DATALOADER,
                f"{method}/seed{seed}: source config loader settings differ from protocol",
            )
        run_path, run_sha = _identity_from_record(
            record.get("run_card"), restricted_root, f"{method}/seed{seed} training run card"
        )
        run = _read_json(
            run_path,
            f"{method}/seed{seed} training run card",
            expected_sha256=run_sha,
        )
        require(run.get("method") == method and run.get("seed") == seed, f"{method}/seed{seed}: run-card identity changed")
        require(run.get("source_config_sha256") == config_sha, f"{method}/seed{seed}: run card does not bind training config")
        checkpoint_path, checkpoint_sha = _identity_from_record(
            record.get("checkpoint"), restricted_root, f"{method}/seed{seed} checkpoint"
        )
        frozen_training: Optional[Mapping[str, Any]] = None
        expected_inference_commit = expected_producer_commit
        if expected_training_identities is not None:
            frozen_training = expected_training_identities.get(key)
            require(
                isinstance(frozen_training, Mapping)
                and frozen_training.get("config_sha256") == config_sha
                and frozen_training.get("run_card_sha256") == run_sha
                and frozen_training.get("checkpoint_sha256") == checkpoint_sha,
                f"{method}/seed{seed}: training lineage differs from frozen Stage 3",
            )
            expected_inference_commit = _commit(
                frozen_training.get("inference_git_commit"),
                f"{method}/seed{seed} frozen inference commit",
            )
            _sha(
                frozen_training.get("inference_card_sha256"),
                f"{method}/seed{seed} frozen inference-card digest",
            )
        if method == "sufs_ft":
            released = run.get("released")
            parameter_count = (
                released.get("parameter_count")
                if isinstance(released, Mapping)
                else None
            )
        else:
            counts = run.get("parameter_count")
            parameter_count = counts.get("total") if isinstance(counts, Mapping) else None
        require(
            type(parameter_count) is int and parameter_count > 0,
            f"{method}/seed{seed}: run card lacks a positive parameter count",
        )
        expected_params_m = int(parameter_count) / 1_000_000.0
        if frozen_training is not None:
            frozen_params = frozen_training.get("params_m")
            require(
                isinstance(frozen_params, (int, float))
                and not isinstance(frozen_params, bool)
                and math.isfinite(float(frozen_params))
                and math.isclose(
                    float(frozen_params),
                    expected_params_m,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{method}/seed{seed}: parameter count differs from frozen Stage 3",
            )
        if key not in legacy_records:
            run_checkpoints = run.get("checkpoint_sha256")
            require(
                run.get("manifest_sha256") == manifest_sha256
                and run.get("split_sha256") == splits_sha256
                and run.get("threshold_amendment_sha256") == amendment_sha256
                and run.get("task")
                == ("patch" if method in PATCH_NEURAL_METHODS else "pixel")
                and run.get("debug_limited") is False
                and run.get("train_images")
                == int(expected_universes["train"]["count"])
                and run.get("validation_images")
                == int(expected_universes["val"]["count"])
                and type(run.get("epochs_completed")) is int
                and int(run["epochs_completed"]) > 0
                and type(run.get("epochs_requested")) is int
                and int(run["epochs_requested"]) >= int(run["epochs_completed"])
                and isinstance(run_checkpoints, Mapping)
                and run_checkpoints.get("best.pt") == checkpoint_sha,
                f"{method}/seed{seed}: amended run-card scientific lineage changed",
            )
            train_provenance = _validate_process_provenance(
                run.get("provenance"),
                allowed_scripts={
                    "train_sufs.py"
                    if method == "sufs_ft"
                    else (
                        "train_patch.py"
                        if method in PATCH_NEURAL_METHODS
                        else "train_seg.py"
                    )
                },
                expected_paths={
                    "--config": config_path,
                    "--manifest": manifest_path,
                    "--splits": splits_path,
                    "--out": run_path.parent,
                },
                expected_values=(
                    {"--mode": "full"} if method == "sufs_ft" else {}
                ),
                description=f"{method}/seed{seed} training",
            )
            require(
                train_provenance["options"].get("--seed", str(seed)) == str(seed),
                f"{method}/seed{seed}: training command seed changed",
            )
        selected = _selected_threshold(
            record.get("selected_threshold"), f"{method}/seed{seed} selected threshold"
        )
        require(selected in REPLACEMENT_THRESHOLD_GRID, f"{method}/seed{seed}: selected threshold is outside amended grid")
        raw_validation = _validate_probability_card(
            record.get("raw_validation_probability_card"),
            split="val",
            method=method,
            seed=seed,
            config_sha256=config_sha,
            checkpoint_sha256=checkpoint_sha,
            amendment_sha256=amendment_sha256,
            manifest_sha256=manifest_sha256,
            splits_sha256=splits_sha256,
            manifest_path=manifest_path,
            splits_path=splits_path,
            restricted_root=restricted_root,
            expected_universes=expected_universes,
            validation_state="raw",
        )
        validation = _validate_probability_card(
            record.get("validation_probability_card"),
            split="val",
            method=method,
            seed=seed,
            config_sha256=config_sha,
            checkpoint_sha256=checkpoint_sha,
            amendment_sha256=amendment_sha256,
            manifest_sha256=manifest_sha256,
            splits_sha256=splits_sha256,
            manifest_path=manifest_path,
            splits_path=splits_path,
            restricted_root=restricted_root,
            expected_universes=expected_universes,
            validation_state="selected",
        )
        require(
            raw_validation["count"] == validation["count"]
            and raw_validation["probability_set_sha256"]
            == validation["probability_set_sha256"],
            f"{method}/seed{seed}: selector changed the validation probability set",
        )
        validation_payload = _read_json(
            Path(validation["path"]),
            f"{method}/seed{seed} selected validation probability card",
            expected_sha256=validation["sha256"],
        )
        _validate_trainable_inference_provenance(
            validation_payload,
            method=method,
            seed=seed,
            split="val",
            expected_commit=expected_inference_commit,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            splits_path=splits_path,
            expected_count=int(expected_universes["val"]["count"]),
        )
        require(
            _selected_threshold(
                validation_payload.get(
                    "selected_validation_threshold",
                    validation_payload.get("selected_threshold"),
                ),
                f"{method}/seed{seed} selected validation-card threshold",
            )
            == selected,
            f"{method}/seed{seed}: selected validation card differs from selector",
        )
        test = _validate_probability_card(
            record.get("test_probability_card"),
            split="test",
            method=method,
            seed=seed,
            config_sha256=config_sha,
            checkpoint_sha256=checkpoint_sha,
            amendment_sha256=amendment_sha256,
            manifest_sha256=manifest_sha256,
            splits_sha256=splits_sha256,
            manifest_path=manifest_path,
            splits_path=splits_path,
            restricted_root=restricted_root,
            expected_universes=expected_universes,
        )
        test_payload = _read_json(
            Path(test["path"]),
            f"{method}/seed{seed} test probability card",
            expected_sha256=test["sha256"],
        )
        test_provenance = _validate_trainable_inference_provenance(
            test_payload,
            method=method,
            seed=seed,
            split="test",
            expected_commit=expected_inference_commit,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            splits_path=splits_path,
            expected_count=int(expected_universes["test"]["count"]),
        )
        if method == "sufs_ft":
            require(
                test_payload.get("parameter_count") == parameter_count,
                f"{method}/seed{seed}: inference parameter count differs from run card",
            )
        require(
            test_payload.get("selection_source") == "unselected_raw"
            and _selected_threshold(
                test_payload.get("selected_validation_threshold"),
                f"{method}/seed{seed} test-card threshold",
            )
            == selected,
            f"{method}/seed{seed}: test card does not use the selected validation threshold",
        )
        inference_path, inference_sha = _identity_from_record(
            record.get("inference_card"), restricted_root, f"{method}/seed{seed} inference card"
        )
        inference = _read_json(
            inference_path,
            f"{method}/seed{seed} inference card",
            expected_sha256=inference_sha,
        )
        require(
            inference.get("schema_version") == 1
            and inference.get("kind") == TRAINABLE_INFERENCE_CARD_KIND
            and inference.get("method") == method
            and inference.get("seed") == seed
            and inference.get("selection_frozen_before_test_inference") is True
            and inference.get("test_labels_opened") is False,
            f"{method}/seed{seed}: inference-card identity changed",
        )
        _validate_delivery_producer(
            inference,
            expected_commit=expected_producer_commit,
            restricted_root=restricted_root,
            description=f"{method}/seed{seed} inference card",
        )
        require(
            _selected_threshold(
                inference.get("selected_validation_threshold"),
                f"{method}/seed{seed} inference threshold",
            )
            == selected,
            f"{method}/seed{seed}: inference-card threshold differs",
        )
        identities = inference.get("identities")
        selector_path, selector_sha = _identity_from_record(
            record.get("selector_receipt"),
            restricted_root,
            f"{method}/seed{seed} expanded-grid selector receipt",
        )
        selector = _read_json(
            selector_path,
            f"{method}/seed{seed} selector receipt",
            expected_sha256=selector_sha,
        )
        selector_scores, selector_winner, selector_winning_score = (
            _validated_threshold_scores(
                selector.get("threshold_scores"),
                f"{method}/seed{seed} selector threshold scores",
            )
        )
        validation_scores, validation_winner, validation_winning_score = (
            _validated_threshold_scores(
                validation_payload.get("threshold_scores"),
                f"{method}/seed{seed} selected-card threshold scores",
            )
        )
        recomputed_validation_scores = _recompute_validation_threshold_scores(
            validation_payload,
            method=method,
            config=config,
            expected_universes=expected_universes,
            restricted_root=restricted_root,
        )
        require(
            selector.get("schema_version") == 1
            and selector.get("kind") == SELECTOR_RECEIPT_KIND
            and selector.get("method") == method
            and selector.get("seed") == seed,
            f"{method}/seed{seed}: invalid selector receipt",
        )
        require(
            selector.get("source_config_sha256") == config_sha
            and selector.get("checkpoint_sha256") == checkpoint_sha
            and selector.get("amendment_sha256") == amendment_sha256
            and selector.get("raw_validation_probability_card_sha256")
            == raw_validation["sha256"]
            and selector.get("validation_probability_card_sha256")
            == validation["sha256"]
            and tuple(selector.get("allowed_grid", ()))
            == REPLACEMENT_THRESHOLD_GRID
            and selector.get("validation_only") is True
            and selector.get("test_or_heldout_opened") is False
            and selector.get("selection_split") == "val"
            and selector.get("selection_metric") == "macro_image_dice"
            and selector.get("selection_source") == "validation_only"
            and selector.get("tie_break") == "closest_to_0.5_then_lower"
            and selector_scores == validation_scores
            and all(
                math.isclose(
                    selector_scores[threshold],
                    recomputed_validation_scores[threshold],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for threshold in REPLACEMENT_THRESHOLD_GRID
            )
            and selector_winner == validation_winner == selected
            and math.isclose(
                selector_winning_score,
                validation_winning_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and isinstance(selector.get("selected_validation_dice"), (int, float))
            and not isinstance(selector.get("selected_validation_dice"), bool)
            and math.isfinite(float(selector.get("selected_validation_dice")))
            and math.isclose(
                float(selector.get("selected_validation_dice")),
                selector_winning_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and _selected_threshold(
                selector.get("selected_threshold"),
                f"{method}/seed{seed} selector threshold",
            )
            == selected,
            f"{method}/seed{seed}: selector receipt does not prove validation-only selection",
        )
        require(
            isinstance(identities, Mapping)
            and identities.get("source_config_sha256") == config_sha
            and identities.get("checkpoint_sha256") == checkpoint_sha
            and identities.get("selector_receipt_sha256") == selector_sha
            and identities.get("raw_validation_probability_card_sha256")
            == raw_validation["sha256"]
            and identities.get("validation_probability_card_sha256")
            == validation["sha256"]
            and identities.get("test_probability_card_sha256") == test["sha256"]
            and identities.get("amendment_sha256") == amendment_sha256,
            f"{method}/seed{seed}: inference card does not bind training lineage",
        )
        require(
            scientific_summary_sha256 is not None
            and scientific_summary_path is not None
            and sha256_file(scientific_summary_path) == scientific_summary_sha256,
            f"{description} lacks a superseding summary identity",
        )
        scoring_sha, scoring_evidence = _validate_scoring_receipt(
            record.get("scoring_receipt"),
            method=method,
            seed=seed,
            selected_threshold=selected,
            amendment_sha256=amendment_sha256,
            selector_sha256=selector_sha,
            inference_sha256=inference_sha,
            test_probability_card_sha256=test["sha256"],
            test_probability_payload=test_payload,
            source_config_sha256=config_sha,
            source_config=config,
            expected_params_m=expected_params_m,
            expected_ms_per_img=float(
                test_provenance["timing_evidence"]["mean_ms_per_img"]
            ),
            manifest_sha256=manifest_sha256,
            splits_sha256=splits_sha256,
            scientific_summary_path=scientific_summary_path,
            expected_universes=expected_universes,
            restricted_root=restricted_root,
            expected_commit=expected_producer_commit,
        )
        lineage_evidence: Dict[str, Any]
        if key in legacy_records:
            completion_path, completion_sha = _identity_from_record(
                record.get("legacy_completion_evidence"),
                restricted_root,
                f"{method}/seed{seed} legacy completion evidence",
            )
            completion = _read_json(
                completion_path,
                f"{method}/seed{seed} legacy completion evidence",
                expected_sha256=completion_sha,
            )
            require(
                completion.get("schema_version") == 1
                and completion.get("kind") == LEGACY_COMPLETION_EVIDENCE_KIND
                and completion.get("method") == method
                and completion.get("seed") == seed,
                f"{method}/seed{seed}: invalid external legacy-completion evidence",
            )
            require(
                completion.get("original_run_card_sha256") == run_sha
                and completion.get("original_config_sha256") == config_sha
                and completion.get("original_checkpoint_sha256") == checkpoint_sha
                and completion.get("threshold_amendment_sha256")
                == amendment_sha256
                and completion.get("completed_before_threshold_amendment") is True
                and completion.get("historical_run_card_modified") is False
                and bool(str(completion.get("evidence_basis", "")).strip()),
                f"{method}/seed{seed}: external legacy-completion evidence is not bound",
            )
            lineage_evidence = {
                "legacy_completion_evidence_sha256": completion_sha,
            }
        else:
            require(
                _selected_threshold(
                    run.get("selected_validation_threshold"),
                    f"{method}/seed{seed} run-card threshold",
                )
                == selected,
                f"{method}/seed{seed}: run-card threshold differs",
            )
            cache_sha, cache_evidence = _validate_preprocess_cache_receipt(
                record.get("preprocess_cache_receipt"),
                manifest_sha256=manifest_sha256,
                expected_universes=expected_universes,
                restricted_root=restricted_root,
                description=f"{method}/seed{seed} preprocessing-cache receipt",
            )
            training_options = train_provenance["options"]
            require(
                "--cache-dir" in training_options
                and Path(training_options["--cache-dir"]).resolve()
                == Path(cache_evidence["path"]).parent.resolve()
                and "--threshold-amendment" in training_options
                and sha256_file(Path(training_options["--threshold-amendment"]))
                == amendment_sha256,
                f"{method}/seed{seed}: training command does not bind cache/amendment",
            )
            require(
                run.get("dataloader") == PROTOCOL_DATALOADER
                and run.get("preprocess_cache_receipt_sha256") == cache_sha,
                f"{method}/seed{seed}: run card does not bind loader/cache protocol",
            )
            raw_validation_payload = _read_json(
                Path(raw_validation["path"]),
                f"{method}/seed{seed} raw validation probability card",
                expected_sha256=raw_validation["sha256"],
            )
            require(
                raw_validation_payload.get("preprocess_cache_receipt_sha256")
                == cache_sha
                and raw_validation_payload.get("cache_used_for_image_decode") is True
                and validation_payload.get("preprocess_cache_receipt_sha256")
                == cache_sha
                and validation_payload.get("cache_used_for_image_decode") is True,
                f"{method}/seed{seed}: validation probability cards do not bind the verified cache",
            )
            require(
                test_payload.get("cache_used_for_image_decode") is False
                and test_payload.get("paper_timing_preprocess")
                == "on_the_fly_decode_and_resample",
                f"{method}/seed{seed}: test timing does not include preprocessing",
            )
            acceleration = run.get("acceleration")
            require(
                isinstance(acceleration, Mapping)
                and acceleration.get("evaluation_only") is False
                and acceleration.get("production_eligible") is True,
                f"{method}/seed{seed}: non-production acceleration policy used",
            )
            precision = acceleration.get("precision")
            if precision == "fp32":
                require(
                    acceleration.get("channels_last") is False
                    and acceleration.get("acceptance_receipt_sha256") is None
                    and "acceleration_receipt" not in record,
                    f"{method}/seed{seed}: invalid fp32 acceleration policy",
                )
                acceleration_receipt_sha: Optional[str] = None
                require(
                    "--acceleration-receipt" not in training_options,
                    f"{method}/seed{seed}: fp32 training unexpectedly used an acceleration receipt",
                )
            else:
                require(
                    precision == "bf16" and acceleration.get("channels_last") is True,
                    f"{method}/seed{seed}: unsupported acceleration policy",
                )
                acceleration_path, acceleration_receipt_sha = _identity_from_record(
                    record.get("acceleration_receipt"),
                    restricted_root,
                    f"{method}/seed{seed} acceleration receipt",
                )
                acceleration_receipt = _read_json(
                    acceleration_path,
                    f"{method}/seed{seed} acceleration receipt",
                    expected_sha256=acceleration_receipt_sha,
                )
                require(
                    "--acceleration-receipt" in training_options
                    and Path(training_options["--acceleration-receipt"]).resolve()
                    == acceleration_path.resolve(),
                    f"{method}/seed{seed}: training command acceleration receipt changed",
                )
                delta = acceleration_receipt.get("absolute_validation_dice_delta")
                require(
                    acceleration.get("acceptance_receipt_sha256")
                    == acceleration_receipt_sha
                    and acceleration_receipt.get("schema_version") == 2
                    and acceleration_receipt.get("kind")
                    == "bf16_channels_last_two_epoch_acceptance"
                    and acceleration_receipt.get("accepted") is True
                    and acceleration_receipt.get("epochs") == 2
                    and acceleration_receipt.get("method") == method
                    and acceleration_receipt.get("seed") == seed
                    and acceleration_receipt.get("source_config_sha256")
                    == config_sha
                    and acceleration_receipt.get("manifest_sha256")
                    == manifest_sha256
                    and acceleration_receipt.get("split_sha256") == splits_sha256
                    and acceleration_receipt.get("task")
                    == ("patch" if method in PATCH_NEURAL_METHODS else "pixel")
                    and acceleration_receipt.get("acceptance_threshold_exclusive")
                    == 0.005
                    and isinstance(
                        acceleration_receipt.get("fp32_validation_dice"),
                        (int, float),
                    )
                    and isinstance(
                        acceleration_receipt.get("bf16_validation_dice"),
                        (int, float),
                    )
                    and not isinstance(
                        acceleration_receipt.get("fp32_validation_dice"), bool
                    )
                    and not isinstance(
                        acceleration_receipt.get("bf16_validation_dice"), bool
                    )
                    and isinstance(delta, (int, float))
                    and not isinstance(delta, bool)
                    and math.isfinite(
                        float(acceleration_receipt["fp32_validation_dice"])
                    )
                    and math.isfinite(
                        float(acceleration_receipt["bf16_validation_dice"])
                    )
                    and math.isfinite(float(delta))
                    and math.isclose(
                        abs(
                            float(acceleration_receipt["fp32_validation_dice"])
                            - float(acceleration_receipt["bf16_validation_dice"])
                        ),
                        float(delta),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and float(delta) < 0.005,
                    f"{method}/seed{seed}: bf16/channels-last equivalence is not proven",
                )
                for mode, expected_precision, expected_channels_last in (
                    ("fp32", "fp32", False),
                    ("bf16", "bf16", True),
                ):
                    paired_path, paired_sha = _identity_from_record(
                        acceleration_receipt.get(f"{mode}_run_card"),
                        restricted_root,
                        f"{method}/seed{seed} {mode} acceleration run card",
                    )
                    paired = _read_json(
                        paired_path,
                        f"{method}/seed{seed} {mode} acceleration run card",
                        expected_sha256=paired_sha,
                    )
                    paired_acceleration = paired.get("acceleration")
                    require(
                        paired.get("method") == method
                        and paired.get("seed") == seed
                        and paired.get("source_config_sha256") == config_sha
                        and paired.get("manifest_sha256") == manifest_sha256
                        and paired.get("split_sha256") == splits_sha256
                        and paired.get("epochs_completed") == 2
                        and paired.get("debug_limited") is False
                        and paired.get("train_images")
                        == int(expected_universes["train"]["count"])
                        and paired.get("validation_images")
                        == int(expected_universes["val"]["count"])
                        and isinstance(paired_acceleration, Mapping)
                        and paired_acceleration.get("precision")
                        == expected_precision
                        and paired_acceleration.get("channels_last")
                        is expected_channels_last
                        and paired_acceleration.get("evaluation_only") is True
                        and paired_acceleration.get("production_eligible") is False
                        and isinstance(
                            paired.get("best_validation_dice"), (int, float)
                        )
                        and not isinstance(paired.get("best_validation_dice"), bool)
                        and math.isfinite(float(paired["best_validation_dice"]))
                        and math.isclose(
                            float(paired["best_validation_dice"]),
                            float(acceleration_receipt[f"{mode}_validation_dice"]),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        ),
                        f"{method}/seed{seed}: {mode} acceleration run is not bound",
                    )
                    paired_provenance = _validate_process_provenance(
                        paired.get("provenance"),
                        allowed_scripts={
                            "train_sufs.py"
                            if method == "sufs_ft"
                            else (
                                "train_patch.py"
                                if method in PATCH_NEURAL_METHODS
                                else "train_seg.py"
                            )
                        },
                        expected_paths={
                            "--config": config_path,
                            "--manifest": manifest_path,
                            "--splits": splits_path,
                            "--out": paired_path.parent,
                        },
                        expected_values={
                            "--acceleration-evaluation": (
                                "fp32"
                                if mode == "fp32"
                                else "bf16_channels_last"
                            ),
                            **({"--mode": "full"} if method == "sufs_ft" else {}),
                        },
                        description=f"{method}/seed{seed} {mode} acceleration check",
                    )
                    require(
                        paired_provenance["options"].get("--seed", str(seed))
                        == str(seed),
                        f"{method}/seed{seed}: {mode} acceleration seed changed",
                    )
            profile_path, profile_sha = _identity_from_record(
                record.get("per_epoch_training_profile_card"),
                restricted_root,
                f"{method}/seed{seed} per-epoch profile",
            )
            profile = _read_json(
                profile_path,
                f"{method}/seed{seed} per-epoch profile",
                expected_sha256=profile_sha,
            )
            epoch_profiles = profile.get("epoch_profiles")
            epochs_recorded = profile.get("epochs_recorded")
            require(
                profile.get("schema_version") == 1
                and profile.get("kind") == PER_EPOCH_PROFILE_KIND
                and profile.get("method") == method
                and profile.get("seed") == seed
                and profile.get("source_config_sha256") == config_sha
                and profile.get("run_card_sha256") == run_sha
                and profile.get("complete") is True
                and profile.get("data_wait_seconds_recorded_per_epoch") is True
                and profile.get("model_compute_seconds_recorded_per_epoch") is True
                and profile.get("generic_device_utilisation_recorded") is True
                and type(epochs_recorded) is int
                and epochs_recorded > 0
                and epochs_recorded == run.get("epochs_completed")
                and isinstance(epoch_profiles, list)
                and len(epoch_profiles) == epochs_recorded,
                f"{method}/seed{seed}: amended training lacks complete per-epoch profiling",
            )
            for expected_epoch, epoch_profile in enumerate(epoch_profiles):
                require(
                    isinstance(epoch_profile, Mapping)
                    and epoch_profile.get("epoch") == expected_epoch,
                    f"{method}/seed{seed}: per-epoch profile order changed",
                )
                for field in ("data_wait_seconds", "compute_seconds"):
                    value = epoch_profile.get(field)
                    require(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) >= 0.0,
                        f"{method}/seed{seed}: invalid {field} in epoch profile",
                    )
                sample_count = epoch_profile.get("gpu_utilization_sample_count")
                require(
                    type(sample_count) is int and sample_count >= 0,
                    f"{method}/seed{seed}: invalid device-utilisation sample count",
                )
            require(
                profile.get("generic_device_utilisation_unavailable_reason") is None
                and all(
                    int(item["gpu_utilization_sample_count"]) > 0
                    and isinstance(
                        item.get("gpu_utilization_percent_mean"), (int, float)
                    )
                    and not isinstance(
                        item.get("gpu_utilization_percent_mean"), bool
                    )
                    and math.isfinite(float(item["gpu_utilization_percent_mean"]))
                    and 0.0
                    <= float(item["gpu_utilization_percent_mean"])
                    <= 100.0
                    for item in epoch_profiles
                ),
                f"{method}/seed{seed}: incomplete device-utilisation profile",
            )
            lineage_evidence = {
                "per_epoch_training_profile_card_sha256": profile_sha,
                "preprocess_cache_receipt_sha256": cache_sha,
                "preprocess_cache_evidence": cache_evidence,
                "acceleration_receipt_sha256": acceleration_receipt_sha,
            }
        lineage_evidence.update(
            {
                "selector_receipt_sha256": selector_sha,
                "scoring_receipt_sha256": scoring_sha,
                "scoring_evidence": scoring_evidence,
            }
        )
        checked.append(
            {
                "method": method,
                "seed": seed,
                "training_lineage": expected_lineage,
                "config_sha256": config_sha,
                "run_card_sha256": run_sha,
                "checkpoint_sha256": checkpoint_sha,
                "inference_card_sha256": inference_sha,
                "selected_threshold": selected,
                "raw_validation_probability_card": raw_validation,
                "validation_probability_card": validation,
                "test_probability_card": test,
                "selector_receipt_sha256": selector_sha,
                **lineage_evidence,
            }
        )
    require(observed == required_records, f"{description} threshold-attestation run universe changed")
    return {
        "amendment_sha256": amendment_sha256,
        "allowed_grid": list(REPLACEMENT_THRESHOLD_GRID),
        "records": sorted(checked, key=lambda item: (item["method"], item["seed"])),
    }


def _selected_threshold(value: object, description: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage4Error(f"{description} is not numeric") from exc
    require(math.isfinite(threshold) and 0.0 < threshold < 1.0, f"{description} is outside (0, 1)")
    return threshold


def _validated_threshold_scores(
    value: object, description: str
) -> tuple[Dict[float, float], float, float]:
    """Validate the frozen grid and reproduce its deterministic argmax."""
    require(isinstance(value, Mapping), f"{description} is not a mapping")
    scores: Dict[float, float] = {}
    for raw_threshold, raw_score in value.items():
        threshold = _selected_threshold(raw_threshold, f"{description} threshold")
        require(threshold not in scores, f"{description} duplicates a threshold")
        require(
            isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool),
            f"{description} contains a nonnumeric score",
        )
        score = float(raw_score)
        require(
            math.isfinite(score) and 0.0 <= score <= 1.0,
            f"{description} contains a non-finite/out-of-range score",
        )
        scores[threshold] = score
    require(
        set(scores) == set(REPLACEMENT_THRESHOLD_GRID),
        f"{description} does not contain the exact amended grid",
    )
    selected, selected_score = max(
        scores.items(), key=lambda item: (item[1], -abs(item[0] - 0.5), -item[0])
    )
    return scores, selected, selected_score


def _scientific_rows_for_run(
    summary_path: Path, method: str, seed: int, table: str
) -> tuple[int, str]:
    """Hash exact evaluator row strings without interpreting metric values."""
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, "scientific summary has no header")
            required = {"method", "seed", "table", "group_value"}
            require(required.issubset(reader.fieldnames), "scientific summary schema changed")
            rows = [
                dict(row)
                for row in reader
                if str(row.get("method", "")) == method
                and str(row.get("seed", "")) == str(seed)
                and str(row.get("table", "")) == table
            ]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise Stage4Error(f"cannot hash scientific summary rows: {exc}") from exc
    require(rows, f"{method}/seed{seed}: scientific summary lacks {table} rows")
    rows.sort(key=lambda row: tuple(str(row.get(key, "")) for key in sorted(row)))
    return len(rows), canonical_sha256(rows)


def _csv_metric_float(value: object, description: str) -> float:
    """Parse one evaluator metric while preserving its documented NaN cases.

    ``metrics.py`` returns NaN for MDR or FDR when the corresponding ground-
    truth class is absent.  Pandas serialises those values as ``nan``.  Infinity
    is never a legitimate metric, but rejecting NaN here would make a genuine
    evaluator output unverifiable for an otherwise valid all-one/all-zero GT.
    """
    # pandas writes NaN as an empty CSV field by default.  It may also read or
    # re-emit it as the literal ``nan``; both encodings are equivalent only
    # when the independently recomputed metric is itself NaN.
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage4Error(f"{description} is not numeric") from exc
    require(not math.isinf(result), f"{description} is infinite")
    return result


def _metric_equal(observed: object, expected: object, description: str) -> bool:
    observed_float = _csv_metric_float(observed, description)
    expected_float = float(expected)
    if math.isnan(expected_float):
        return math.isnan(observed_float)
    return math.isfinite(observed_float) and math.isclose(
        observed_float,
        expected_float,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _validate_scientific_metrics_replay(
    *,
    method: str,
    seed: int,
    table: str,
    prediction_dir: Path,
    per_image_path: Path,
    receipt_summary_path: Path,
    merged_summary_path: Path,
    expected_universes: Mapping[str, Mapping[str, Any]],
    expected_params_m: float,
    expected_ms_per_img: float,
    expected_threshold: float,
    expected_device: str,
) -> Dict[str, Any]:
    """Recompute exact confusion/metrics from frozen GT and scored PNGs."""
    expected_records = expected_universes.get("test", {}).get("records")
    require(isinstance(expected_records, Mapping), "test metric universe is absent")
    gt_column = "mask_path" if table == "A" else "quality_mask_path"
    scoring_records = {
        image_id: record
        for image_id, record in expected_records.items()
        if str(record.get(gt_column, "")).strip()
    }
    require(scoring_records, f"{method}/seed{seed}/{table}: scoring universe is empty")
    replay_rows = []
    for image_id in sorted(scoring_records):
        record = scoring_records[image_id]
        gt_path = Path(str(record.get(gt_column, "")))
        require(
            gt_path.is_file() and not gt_path.is_symlink(),
            f"{method}/seed{seed}/{table}: GT absent/non-regular: {image_id}",
        )
        gt = to_target_ppi(
            read_mask(gt_path),
            float(record["ppi"]),
            500.0,
            is_mask=True,
        )
        prediction_path = prediction_dir / f"{image_id}.png"
        require(
            prediction_path.is_file() and not prediction_path.is_symlink(),
            f"{method}/seed{seed}/{table}: prediction absent: {image_id}",
        )
        prediction = read_mask(prediction_path)
        require(
            tuple(gt.shape) == tuple(record["target_shape"])
            and prediction.shape == gt.shape,
            f"{method}/seed{seed}/{table}: scored shape mismatch: {image_id}",
        )
        metrics = image_metrics(prediction, gt)
        metrics.update(
            {
                "image_id": image_id,
                "subject_id": record["subject_id"],
                "examiner_value": record["examiner_value"],
                "missing": False,
            }
        )
        replay_rows.append(metrics)
    replay = pd.DataFrame(replay_rows)
    try:
        with per_image_path.open("r", encoding="utf-8", newline="") as handle:
            observed_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise Stage4Error(f"cannot parse evaluator per-image evidence: {exc}") from exc
    observed = {str(row.get("image_id", "")): row for row in observed_rows}
    require(
        len(observed) == len(observed_rows)
        and set(observed) == set(scoring_records),
        f"{method}/seed{seed}/{table}: evaluator per-image universe changed",
    )
    for row in replay_rows:
        image_id = row["image_id"]
        candidate = observed[image_id]
        require(
            str(candidate.get("subject_id", "")) == str(row["subject_id"])
            and str(candidate.get("examiner_value", ""))
            == str(row["examiner_value"])
            and str(candidate.get("missing", "")).lower() in {"false", "0"},
            f"{method}/seed{seed}/{table}: per-image identity changed: {image_id}",
        )
        for field in ("tp", "fp", "fn", "tn"):
            require(
                int(candidate.get(field, -1)) == int(row[field]),
                f"{method}/seed{seed}/{table}: {field} differs: {image_id}",
            )
        for field in METRICS:
            require(
                _metric_equal(
                    candidate.get(field),
                    row[field],
                    f"{method}/seed{seed}/{table}/{image_id}/{field}",
                ),
                f"{method}/seed{seed}/{table}: {field} differs: {image_id}",
            )

    expected_summaries: Dict[str, Dict[str, Any]] = {
        "overall": summarise(replay)
    }
    for group_value, group in replay.groupby("examiner_value", dropna=False):
        expected_summaries[str(group_value)] = summarise(group)

    def validate_summary(path: Path, description: str) -> None:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = [
                    dict(row)
                    for row in csv.DictReader(handle)
                    if str(row.get("method", "")) == method
                    and str(row.get("seed", "")) == str(seed)
                    and str(row.get("table", "")) == table
                ]
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise Stage4Error(f"cannot parse {description}: {exc}") from exc
        by_group = {str(row.get("group_value", "")): row for row in rows}
        require(
            len(by_group) == len(rows) and set(by_group) == set(expected_summaries),
            f"{method}/seed{seed}/{table}: {description} group universe changed",
        )
        for group_value, expected in expected_summaries.items():
            row = by_group[group_value]
            require(
                int(row.get("n_images", -1)) == int(expected["n_images"])
                and int(float(row.get("n_missing", -1))) == 0,
                f"{method}/seed{seed}/{table}: {description} count differs: "
                f"{group_value}",
            )
            for metric in METRICS:
                for suffix in ("mean", "std", "pooled"):
                    field = f"{metric}_{suffix}"
                    require(
                        field in row
                        and _metric_equal(
                            row[field],
                            expected[field],
                            f"{method}/seed{seed}/{table}/{group_value}/{field}",
                        ),
                        f"{method}/seed{seed}/{table}: {description} {field} "
                        f"differs: {group_value}",
                    )
            require(
                _metric_equal(
                    row.get("params_m"),
                    expected_params_m,
                    f"{method}/seed{seed}/{table}/{group_value}/params_m",
                )
                and _metric_equal(
                    row.get("ms_per_img"),
                    expected_ms_per_img,
                    f"{method}/seed{seed}/{table}/{group_value}/ms_per_img",
                )
                and _metric_equal(
                    row.get("threshold"),
                    expected_threshold,
                    f"{method}/seed{seed}/{table}/{group_value}/threshold",
                )
                and str(row.get("device", "")) == expected_device,
                f"{method}/seed{seed}/{table}: {description} runtime/model metadata "
                f"differs: {group_value}",
            )

    validate_summary(receipt_summary_path, "receipt evaluator summary")
    validate_summary(merged_summary_path, "merged scientific summary")
    return {
        "per_image_count": len(replay_rows),
        "groups": sorted(expected_summaries),
    }


def _argv_value(argv: Sequence[object], flag: str, description: str) -> str:
    """Return the unique value following one exact evaluator flag."""
    strings = [str(item) for item in argv]
    require(strings.count(flag) == 1, f"{description} must contain {flag} exactly once")
    index = strings.index(flag)
    require(index + 1 < len(strings), f"{description} lacks a value for {flag}")
    return strings[index + 1]


def _safe_evaluator_component(value: object) -> str:
    """Reproduce the common evaluator's filename-safe component mapping."""
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "none"


def _validate_scoring_argv(
    argv: object,
    *,
    evaluator: Path,
    receipt_manifest: Path,
    receipt_splits: Path,
    method: str,
    seed: int,
    table: str,
    selected_threshold: float,
    expected_params_m: float,
    expected_ms_per_img: float,
    test_probability_payload: Mapping[str, Any],
    restricted_root: Path,
) -> Dict[str, Any]:
    """Validate the complete, ordered common-evaluator command line.

    Merely finding a handful of expected flags is insufficient: a forged
    receipt could otherwise append ``--limit``, change the ground truth, or
    point the evaluator at an unrelated prediction set.  The wrapper emits one
    closed argv schema, so the finalizer reproduces that schema exactly.
    """
    require(isinstance(argv, list), f"{method}/seed{seed}: evaluator argv absent")
    values = [str(value) for value in argv]
    flags = [
        "--manifest",
        "--splits",
        "--subsets",
        "--method",
        "--display-name",
        "--pred-template",
        "--pred-ppi",
        "--target-ppi",
        "--split",
        "--gt-column",
        "--exclude",
        "--group-by",
        "--seed",
        "--year",
        "--family",
        "--training-data",
        "--comment",
        "--params-m",
        "--device",
        "--threshold",
        "--run-id",
        "--table",
        "--ms-per-img",
        "--out",
    ]
    require(
        len(values) == 6 + 2 * len(flags)
        and values[6::2] == flags
        and Path(values[0]).resolve() == Path(sys.executable).resolve()
        and values[1] == "-I"
        and values[2] == "-B"
        and values[3] == "-c"
        and values[4] == SCORING_EVALUATOR_BOOTSTRAP
        and Path(values[5]).resolve() == evaluator.resolve(),
        f"{method}/seed{seed}: evaluator argv schema changed",
    )
    arguments = dict(zip(flags, values[7::2]))
    expected_gt = "mask_path" if table == "A" else "quality_mask_path"
    run_id = _sha(
        test_probability_payload.get("checkpoint_sha256"),
        f"{method}/seed{seed} scoring run id",
    )
    timing = test_probability_payload.get(
        "mean_ms_per_img", test_probability_payload.get("ms_per_img")
    )
    require(
        isinstance(timing, (int, float))
        and not isinstance(timing, bool)
        and math.isfinite(float(timing))
        and float(timing) >= 0.0,
        f"{method}/seed{seed}: test inference time is invalid",
    )
    card_device = str(test_probability_payload.get("device", "")).upper()
    if card_device == "CUDA":
        card_device = "GPU"
    fixed = {
        "--manifest": str(receipt_manifest),
        "--splits": str(receipt_splits),
        "--subsets": "finger_latent",
        "--method": method,
        "--pred-ppi": "500",
        "--target-ppi": "500",
        "--split": "test",
        "--gt-column": expected_gt,
        "--exclude": "examiner_value=NONPRINT",
        "--group-by": "examiner_value",
        "--seed": str(seed),
        "--device": card_device,
        "--run-id": run_id,
        "--table": table,
    }
    require(
        all(arguments.get(flag) == expected for flag, expected in fixed.items())
        and bool(arguments["--display-name"].strip())
        and bool(arguments["--training-data"].strip())
        and card_device in {"CPU", "GPU"}
        and math.isclose(
            float(arguments["--threshold"]),
            selected_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(arguments["--ms-per-img"]),
            expected_ms_per_img,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(arguments["--params-m"]),
            expected_params_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{method}/seed{seed}: evaluator argv values changed",
    )
    require(
        math.isclose(float(timing), expected_ms_per_img, rel_tol=0.0, abs_tol=1e-12),
        f"{method}/seed{seed}: evaluator timing differs from verified per-image records",
    )
    prediction_template = arguments["--pred-template"]
    suffix = "{image_id}.png"
    require(
        prediction_template.endswith(suffix),
        f"{method}/seed{seed}: evaluator prediction template changed",
    )
    prediction_dir = _require_under(
        Path(prediction_template[: -len(suffix)]),
        restricted_root,
        f"{method}/seed{seed} scoring prediction directory",
    )
    require(
        prediction_dir.is_dir()
        and not prediction_dir.is_symlink()
        and prediction_template == str(prediction_dir / suffix),
        f"{method}/seed{seed}: evaluator prediction directory is invalid",
    )
    results_dir = _require_under(
        Path(arguments["--out"]),
        restricted_root,
        f"{method}/seed{seed} evaluator output directory",
    )
    require(
        results_dir.is_dir() and not results_dir.is_symlink(),
        f"{method}/seed{seed}: evaluator output directory is invalid",
    )
    return {
        "arguments": arguments,
        "prediction_dir": prediction_dir,
        "results_dir": results_dir,
        "run_id": run_id,
    }


def _recompute_scored_mask_derivation(
    *,
    card: Mapping[str, Any],
    prediction_dir: Path,
    selected_threshold: float,
    config: Mapping[str, Any],
    method: str,
    seed: int,
    restricted_root: Path,
) -> tuple[int, str]:
    """Independently prove that the evaluator saw thresholded NPY artifacts."""
    task = str(config.get("task", ""))
    require(
        (task == "pixel" and method in PIXEL_NEURAL_METHODS)
        or (task == "patch" and method in PATCH_NEURAL_METHODS),
        f"{method}/seed{seed}: source config task differs from method family",
    )
    inference_config = config.get("inference")
    data_config = config.get("data")
    require(
        isinstance(inference_config, Mapping) and isinstance(data_config, Mapping),
        f"{method}/seed{seed}: source config scoring geometry is absent",
    )
    min_component_blocks: Optional[int] = None
    if task == "patch":
        block_size = data_config.get(
            "block_size", inference_config.get("block_size")
        )
        min_component_blocks = inference_config.get("min_component_blocks")
        require(
            type(block_size) is int
            and block_size == BLOCK_SIZE == block_features.BLOCK
            and type(min_component_blocks) is int
            and min_component_blocks >= 1,
            f"{method}/seed{seed}: patch postprocessing config changed",
        )
    probability_set = card.get("probability_set")
    require(isinstance(probability_set, Mapping), f"{method}/seed{seed}: probability set absent")
    records = probability_set.get("records")
    require(isinstance(records, list) and records, f"{method}/seed{seed}: probability records absent")
    expected_prediction_names: set[str] = set()
    derivation_records = []
    for record in records:
        require(isinstance(record, Mapping), f"{method}/seed{seed}: malformed probability record")
        image_id = str(record.get("image_id", ""))
        require(image_id, f"{method}/seed{seed}: empty probability image id")
        probability_path = _require_regular(
            Path(str(record.get("path", ""))),
            restricted_root,
            f"{method}/seed{seed} retained probability {image_id}",
        )
        prediction_path = _require_regular(
            prediction_dir / f"{image_id}.png",
            restricted_root,
            f"{method}/seed{seed} scored prediction {image_id}",
        )
        expected_prediction_names.add(prediction_path.name)
        require(
            sha256_file(probability_path) == record.get("sha256"),
            f"{method}/seed{seed}: retained probability bytes changed: {image_id}",
        )
        try:
            probability = np.load(probability_path, allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise Stage4Error(
                f"{method}/seed{seed}: invalid retained probability NPY: {image_id}"
            ) from exc
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
        require(
            isinstance(probability, np.ndarray)
            and probability.ndim == 2
            and probability.dtype == np.float16
            and np.isfinite(probability).all()
            and not bool((probability < 0.0).any())
            and not bool((probability > 1.0).any())
            and prediction is not None
            and prediction.ndim == 2
            and set(np.unique(prediction).tolist()).issubset({0, 255}),
            f"{method}/seed{seed}: invalid scored probability/mask pair: {image_id}",
        )
        if task == "pixel":
            require(
                tuple(prediction.shape) == tuple(probability.shape),
                f"{method}/seed{seed}: pixel probability shape differs from mask: "
                f"{image_id}",
            )
            derived_bool = probability.astype(np.float32) >= selected_threshold
        else:
            require(
                tuple(probability.shape)
                == tuple(block_features.grid_shape(*prediction.shape)),
                f"{method}/seed{seed}: patch probability grid differs from mask: "
                f"{image_id}",
            )
            block_mask = postprocess_blocks(
                probability.astype(np.float32) >= selected_threshold,
                min_component_blocks=int(min_component_blocks),
            )
            derived_bool = block_features.blocks_to_mask(
                block_mask, prediction.shape
            )
        derived = derived_bool.astype(np.uint8) * 255
        require(
            np.array_equal(derived, prediction),
            f"{method}/seed{seed}: scored mask is not derived from probability: {image_id}",
        )
        derivation_records.append(
            {
                "image_id": image_id,
                "mask_sha256": sha256_file(prediction_path),
                "probability_sha256": sha256_file(probability_path),
            }
        )
    observed_names = {
        path.name
        for path in prediction_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    require(
        observed_names == expected_prediction_names
        and all(not path.is_symlink() for path in prediction_dir.iterdir()),
        f"{method}/seed{seed}: scored prediction directory universe changed",
    )
    derivation_records.sort(key=lambda item: item["image_id"])
    return len(derivation_records), canonical_sha256(derivation_records)


def _validate_scoring_receipt(
    identity: object,
    *,
    method: str,
    seed: int,
    selected_threshold: float,
    amendment_sha256: str,
    selector_sha256: str,
    inference_sha256: str,
    test_probability_card_sha256: str,
    test_probability_payload: Mapping[str, Any],
    source_config_sha256: str,
    source_config: Mapping[str, Any],
    expected_params_m: float,
    expected_ms_per_img: float,
    manifest_sha256: str,
    splits_sha256: str,
    scientific_summary_path: Path,
    expected_universes: Mapping[str, Mapping[str, Any]],
    restricted_root: Path,
    expected_commit: str,
) -> tuple[str, Dict[str, Any]]:
    """Verify scoring came from the committed common-evaluator wrapper."""
    path, digest = _identity_from_record(
        identity, restricted_root, f"{method}/seed{seed} scoring receipt"
    )
    receipt = _read_json(
        path,
        f"{method}/seed{seed} scoring receipt",
        expected_sha256=digest,
    )
    identities = receipt.get("identities")
    producer = receipt.get("producer")
    wrapper = Path(__file__).with_name("score_threshold_run.py")
    evaluator = Path(__file__).with_name("evaluate_masks.py")
    require(wrapper.is_file() and evaluator.is_file(), "committed scoring code is absent")
    require(
        receipt.get("schema_version") == 1
        and receipt.get("kind") == SCORING_RECEIPT_KIND
        and receipt.get("method") == method
        and receipt.get("seed") == seed
        and _selected_threshold(
            receipt.get("selected_threshold"),
            f"{method}/seed{seed} scored threshold",
        )
        == selected_threshold
        and receipt.get("selection_frozen_before_test_scoring") is True
        and receipt.get("test_labels_opened_only_by_evaluator") is True,
        f"{method}/seed{seed}: invalid scoring receipt identity",
    )
    require(
        isinstance(identities, Mapping)
        and identities.get("source_config_sha256") == source_config_sha256
        and identities.get("manifest_sha256") == manifest_sha256
        and identities.get("splits_sha256") == splits_sha256
        and identities.get("test_probability_card_sha256")
        == test_probability_card_sha256
        and identities.get("selector_receipt_sha256") == selector_sha256
        and identities.get("amendment_sha256") == amendment_sha256
        and identities.get("inference_card_sha256") == inference_sha256,
        f"{method}/seed{seed}: scoring receipt input identities changed",
    )
    receipt_manifest = _require_regular(
        Path(str(receipt.get("manifest_path", ""))),
        restricted_root,
        f"{method}/seed{seed} scoring manifest",
    )
    receipt_splits = _require_regular(
        Path(str(receipt.get("splits_path", ""))),
        restricted_root,
        f"{method}/seed{seed} scoring split",
    )
    require(
        sha256_file(receipt_manifest) == manifest_sha256
        and sha256_file(receipt_splits) == splits_sha256,
        f"{method}/seed{seed}: scoring manifest/split bytes changed",
    )
    require(
        isinstance(producer, Mapping)
        and producer.get("git_commit") == expected_commit
        and producer.get("git_dirty") is False
        and producer.get("script_sha256") == sha256_file(wrapper)
        and producer.get("evaluator_sha256") == sha256_file(evaluator),
        f"{method}/seed{seed}: scoring was not run by the committed wrapper/evaluator",
    )
    derivation = test_probability_payload.get("mask_derivation_verification")
    require(
        isinstance(derivation, Mapping)
        and receipt.get("prediction_probability_derivation_records_sha256")
        == derivation.get("records_sha256")
        and receipt.get("prediction_count") == derivation.get("count"),
        f"{method}/seed{seed}: scored masks differ from probability derivation",
    )
    invocations = receipt.get("invocations")
    require(isinstance(invocations, list) and len(invocations) == 2, f"{method}/seed{seed}: scoring invocations changed")
    receipt_summary_path, receipt_summary_sha = _identity_from_record(
        receipt.get("scientific_summary"),
        restricted_root,
        f"{method}/seed{seed} evaluator summary output",
    )
    observed_tables: set[str] = set()
    checked_invocations = []
    shared_prediction_dir: Optional[Path] = None
    shared_results_dir: Optional[Path] = None
    for invocation in invocations:
        require(isinstance(invocation, Mapping), f"{method}/seed{seed}: malformed scoring invocation")
        table = str(invocation.get("table", ""))
        require(table in TABLES and table not in observed_tables, f"{method}/seed{seed}: scoring table universe changed")
        observed_tables.add(table)
        command = _validate_scoring_argv(
            invocation.get("argv"),
            evaluator=evaluator,
            receipt_manifest=receipt_manifest,
            receipt_splits=receipt_splits,
            method=method,
            seed=seed,
            table=table,
            selected_threshold=selected_threshold,
            expected_params_m=expected_params_m,
            expected_ms_per_img=expected_ms_per_img,
            test_probability_payload=test_probability_payload,
            restricted_root=restricted_root,
        )
        require(
            invocation.get("exit_code") == 0,
            f"{method}/seed{seed}: evaluator exit binding changed",
        )
        prediction_dir = command["prediction_dir"]
        results_dir = command["results_dir"]
        if shared_prediction_dir is None:
            shared_prediction_dir = prediction_dir
            shared_results_dir = results_dir
        else:
            require(
                prediction_dir == shared_prediction_dir
                and results_dir == shared_results_dir,
                f"{method}/seed{seed}: evaluator invocations used different inputs/outputs",
            )
        require(
            receipt_summary_path == results_dir / "summary.csv",
            f"{method}/seed{seed}: evaluator summary path differs from argv",
        )
        per_image_path, per_image_sha = _identity_from_record(
            invocation.get("per_image"),
            restricted_root,
            f"{method}/seed{seed} {table} evaluator per-image output",
        )
        identity_payload = json.dumps(
            {
                "run_id": command["arguments"]["--run-id"],
                "training_data": command["arguments"]["--training-data"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        identity_suffix = "_identity-" + hashlib.sha256(identity_payload).hexdigest()
        expected_per_image = results_dir / (
            f"per_image_{_safe_evaluator_component(method)}_finger_latent_"
            f"{_safe_evaluator_component(table)}_seed{seed}{identity_suffix}.csv"
        )
        require(
            per_image_path == expected_per_image,
            f"{method}/seed{seed}: {table} per-image output path differs from argv",
        )
        replay_evidence = _validate_scientific_metrics_replay(
            method=method,
            seed=seed,
            table=table,
            prediction_dir=prediction_dir,
            per_image_path=per_image_path,
            receipt_summary_path=receipt_summary_path,
            merged_summary_path=scientific_summary_path,
            expected_universes=expected_universes,
            expected_params_m=expected_params_m,
            expected_ms_per_img=expected_ms_per_img,
            expected_threshold=selected_threshold,
            expected_device=command["arguments"]["--device"],
        )
        receipt_row_count, receipt_row_digest = _scientific_rows_for_run(
            receipt_summary_path, method, seed, table
        )
        merged_row_count, merged_row_digest = _scientific_rows_for_run(
            scientific_summary_path, method, seed, table
        )
        require(
            invocation.get("scientific_row_count")
            == receipt_row_count
            == merged_row_count
            and invocation.get("scientific_rows_sha256")
            == receipt_row_digest
            == merged_row_digest,
            f"{method}/seed{seed}: {table} scientific rows differ from evaluator receipt",
        )
        _sha(invocation.get("stdout_sha256"), f"{method}/seed{seed} evaluator stdout")
        _sha(invocation.get("stderr_sha256"), f"{method}/seed{seed} evaluator stderr")
        checked_invocations.append(
            {
                "table": table,
                "per_image_sha256": per_image_sha,
                "scientific_rows_sha256": receipt_row_digest,
                "scientific_row_count": receipt_row_count,
                "metric_replay": replay_evidence,
            }
        )
    require(observed_tables == set(TABLES), f"{method}/seed{seed}: incomplete evaluator table set")
    assert shared_prediction_dir is not None
    recomputed_count, recomputed_derivation = _recompute_scored_mask_derivation(
        card=test_probability_payload,
        prediction_dir=shared_prediction_dir,
        selected_threshold=selected_threshold,
        config=source_config,
        method=method,
        seed=seed,
        restricted_root=restricted_root,
    )
    require(
        recomputed_count == receipt.get("prediction_count")
        and recomputed_derivation
        == receipt.get("prediction_probability_derivation_records_sha256")
        == derivation.get("records_sha256"),
        f"{method}/seed{seed}: scored prediction derivation cannot be reproduced",
    )
    return digest, {
        "path": str(path),
        "sha256": digest,
        "scientific_summary_sha256": receipt_summary_sha,
        "prediction_probability_derivation_records_sha256": recomputed_derivation,
        "invocations": checked_invocations,
    }


def _validate_released_probability_card(
    identity: object,
    *,
    split: str,
    method: str,
    selected_threshold: float,
    manifest_path: Path,
    splits_path: Path,
    manifest_sha256: str,
    splits_sha256: str,
    restricted_root: Path,
    expected_universes: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    path, digest = _identity_from_record(
        identity,
        restricted_root,
        f"{method}/unseeded {split} released probability card",
    )
    card = _read_json(
        path,
        f"{method}/unseeded {split} released probability card",
        expected_sha256=digest,
    )
    require(
        card.get("schema_version") == 1
        and card.get("kind") == PROBABILITY_CARD_KIND,
        f"{method}: invalid {split} released probability-card schema",
    )
    require(
        card.get("method") == method and card.get("seed") is None,
        f"{method}: {split} released probability-card identity changed",
    )
    require(card.get("split") == split, f"{method}: released probability card has wrong split")
    require(
        card.get("manifest_sha256") == manifest_sha256
        and card.get("split_sha256") == splits_sha256,
        f"{method}: {split} released probability card data identity changed",
    )
    require(
        card.get("threshold_source") == "released_fixed"
        and card.get("released_threshold_retuned") is False,
        f"{method}: released threshold was retuned",
    )
    require(
        _selected_threshold(card.get("selected_threshold"), f"{method} released threshold")
        == selected_threshold,
        f"{method}: released probability-card threshold differs",
    )
    require(
        card.get("probabilities_are_pre_threshold") is True
        and card.get("binary_mask_reinterpreted") is False,
        f"{method}: binary output was accepted as a probability map",
    )
    storage_dtype = card.get("storage_dtype")
    require(
        card.get("complete") is True
        and storage_dtype == "float16"
        and card.get("storage_format") == f"{storage_dtype}_npy"
        and card.get("derived_from_binary_mask") is False,
        f"{method}: released probability storage violates the protocol",
    )
    require(
        "threshold_candidates" not in card and "threshold_scores" not in card,
        f"{method}: released zero-shot probability card contains retuning evidence",
    )
    if split == "val":
        require(
            card.get("selection_source") == "not_used_released_fixed",
            f"{method}: validation probabilities were used to retune a released threshold",
        )
    else:
        require(card.get("labels_opened") is False, f"{method}: test probability generation opened labels")
        require(
            card.get("threshold_fixed_before_test") is True,
            f"{method}: released threshold was not fixed before test inference",
        )
    probability_summary = _validate_probability_set(
        card,
        split=split,
        method=method,
        seed=None,
        restricted_root=restricted_root,
        expected_universes=expected_universes,
    )
    derivation = card.get("mask_derivation_verification")
    require(
        isinstance(derivation, Mapping)
        and derivation.get("complete") is True
        and int(derivation.get("count", -1)) == probability_summary["count"]
        and derivation.get("all_masks_rederived_from_retained_probabilities") is True
        and derivation.get("all_masks_match_released_runner") is True,
        f"{method}: {split} masks are not proven equivalent to retained probabilities and the released runner",
    )
    _sha(
        derivation.get("records_sha256"),
        f"{method} {split} released mask-derivation digest",
    )
    probability_records = card["probability_set"]["records"]
    probability_parents = {
        Path(str(record.get("path", ""))).resolve().parent
        for record in probability_records
        if isinstance(record, Mapping)
    }
    require(
        len(probability_parents) == 1,
        f"{method}: {split} released probabilities span directories",
    )
    provenance = _validate_process_provenance(
        card.get("provenance"),
        allowed_scripts={"run_pyfing.py"} if method == "sufs" else {"run_fingernet.py"},
        expected_paths={
            "--manifest": manifest_path,
            "--splits": splits_path,
            "--probability-root": next(iter(probability_parents)),
        },
        expected_values={
            "--subset": "finger_latent",
            "--split": split,
        },
        description=f"{method} {split} released inference",
    )
    output_root_value = provenance["options"].get("--output-root")
    require(output_root_value is not None, f"{method}: {split} inference lacks --output-root")
    output_root = _require_under(
        Path(output_root_value),
        restricted_root,
        f"{method} {split} released mask directory",
    )
    require(
        output_root.is_dir() and not output_root.is_symlink(),
        f"{method}: {split} released mask directory is invalid",
    )
    expected_records = expected_universes[split]["records"]
    expected_files = {f"{image_id}.png" for image_id in expected_records}
    observed_files = set()
    for child in output_root.iterdir():
        require(
            child.is_file() and not child.is_symlink(),
            f"{method}: invalid entry in {split} released mask directory",
        )
        observed_files.add(child.name)
    require(
        observed_files == expected_files,
        f"{method}: {split} released mask universe changed",
    )
    by_id = {str(item["image_id"]): item for item in probability_records}
    digest_builder = hashlib.sha256()
    derivation_records = []
    for image_id in sorted(expected_records):
        probability_path = Path(str(by_id[image_id]["path"]))
        probability = np.load(probability_path, allow_pickle=False).astype(np.float32)
        target_shape = tuple(expected_records[image_id]["target_shape"])
        if method == "sufs":
            derived = probability >= selected_threshold
        else:
            crop_h = probability.shape[0] // 8 * 8
            crop_w = probability.shape[1] // 8 * 8
            derived = np.zeros(target_shape, dtype=bool)
            if crop_h and crop_w:
                coarse = np.round(probability[:crop_h:8, :crop_w:8]).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, kernel)
                expanded = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1).astype(bool)
            else:
                expanded = np.zeros((crop_h, crop_w), dtype=bool)
            require(
                expanded.shape == (crop_h, crop_w)
                and crop_h <= target_shape[0]
                and crop_w <= target_shape[1],
                f"{method}: invalid expanded released probability shape for {image_id}",
            )
            derived[:crop_h, :crop_w] = expanded
        mask_path = output_root / f"{image_id}.png"
        saved = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        require(
            saved is not None
            and saved.shape == target_shape
            and set(np.unique(saved).tolist()).issubset({0, 255})
            and np.array_equal(saved > 0, derived),
            f"{method}: {split} mask is not derived from retained probability: {image_id}",
        )
        mask_sha = sha256_file(mask_path)
        probability_sha = sha256_file(probability_path)
        digest_builder.update(f"{image_id}:{mask_sha}\n".encode("utf-8"))
        derivation_records.append(
            {
                "image_id": image_id,
                "mask_sha256": mask_sha,
                "probability_sha256": probability_sha,
            }
        )
    prediction_digest = digest_builder.hexdigest()
    require(
        card.get("prediction_count") == len(expected_records)
        and card.get("prediction_set_sha256") == prediction_digest
        and derivation.get("records_sha256") == canonical_sha256(derivation_records),
        f"{method}: {split} released mask-set evidence changed",
    )
    return {
        "path": str(path),
        "sha256": digest,
        "prediction_directory": str(output_root),
        "prediction_set_sha256": prediction_digest,
        **probability_summary,
    }


def _validate_frozen_released_prediction_set(
    *,
    method: str,
    frozen: Mapping[str, Any],
    manifest_path: Path,
    splits_path: Path,
    restricted_root: Path,
    expected_universe: Mapping[str, Any],
) -> Dict[str, Any]:
    """Rehash the exact Stage 1 PNG universe from its recorded launch command."""
    provenance = frozen.get("provenance")
    command = (
        provenance.get("command_argv")
        if isinstance(provenance, Mapping)
        else frozen.get("command")
    )
    script, options = _command_options(command, f"{method} frozen Stage 1 run")
    require(
        script == ("run_pyfing.py" if method == "sufs" else "run_fingernet.py")
        and options.get("--subset") == "finger_latent"
        and options.get("--split") == "test"
        and "--manifest" in options
        and Path(options["--manifest"]).resolve() == manifest_path.resolve()
        and "--splits" in options
        and Path(options["--splits"]).resolve() == splits_path.resolve()
        and "--output-root" in options,
        f"{method}: frozen Stage 1 command does not bind its prediction set",
    )
    if method == "sufs":
        require(
            options.get("--method") == "sufs"
            and options.get("__subcommand__") in {None, "run"},
            "SUFS frozen Stage 1 command changed",
        )
    output_root = _require_under(
        Path(options["--output-root"]),
        restricted_root,
        f"{method} frozen Stage 1 prediction directory",
    )
    require(
        output_root.is_dir() and not output_root.is_symlink(),
        f"{method}: frozen Stage 1 prediction directory is invalid",
    )
    records = expected_universe.get("records")
    require(isinstance(records, Mapping) and records, "test universe is absent")
    expected_files = {f"{image_id}.png" for image_id in records}
    observed_files = set()
    for child in output_root.iterdir():
        require(
            child.is_file() and not child.is_symlink(),
            f"{method}: invalid frozen Stage 1 prediction entry",
        )
        observed_files.add(child.name)
    require(
        observed_files == expected_files,
        f"{method}: frozen Stage 1 prediction universe changed",
    )
    digest_builder = hashlib.sha256()
    for image_id in sorted(records):
        mask_path = output_root / f"{image_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        require(
            mask is not None
            and mask.shape == tuple(records[image_id]["target_shape"])
            and set(np.unique(mask).tolist()).issubset({0, 255}),
            f"{method}: invalid frozen Stage 1 mask: {image_id}",
        )
        digest_builder.update(
            f"{image_id}:{sha256_file(mask_path)}\n".encode("utf-8")
        )
    observed_digest = digest_builder.hexdigest()
    require(
        frozen.get("method") == method
        and int(frozen.get("prediction_count", -1)) == len(records)
        and frozen.get("prediction_set_sha256") == observed_digest,
        f"{method}: frozen Stage 1 prediction-set declaration differs from bytes",
    )
    return {
        "prediction_directory": str(output_root),
        "prediction_count": len(records),
        "prediction_set_sha256": observed_digest,
    }


def _validate_released_probability_retention(
    delivery: Mapping[str, Any],
    *,
    frozen_stage1_artifacts: Mapping[str, Mapping[str, Any]],
    manifest_path: Path,
    splits_path: Path,
    manifest_sha256: str,
    splits_sha256: str,
    restricted_root: Path,
    expected_universes: Mapping[str, Mapping[str, Any]],
    expected_producer_commit: str,
    description: str,
) -> Dict[str, Any]:
    """Require complete raw probabilities for every supported released runner."""
    attestation = delivery.get("probability_retention_attestation")
    require(isinstance(attestation, Mapping), f"{description} lacks released probability retention attestation")
    require(
        attestation.get("schema_version") == 1
        and attestation.get("kind") == PROBABILITY_RETENTION_KIND,
        f"{description} released probability retention schema changed",
    )
    require(
        attestation.get("binary_masks_accepted_as_probability_maps") is False,
        f"{description} permits binary masks as probabilities",
    )
    records = attestation.get("records")
    require(isinstance(records, list), f"{description} released probability retention records absent")
    observed: set[str] = set()
    checked = []
    for record in records:
        require(isinstance(record, Mapping), f"{description} has malformed released probability record")
        method = str(record.get("method", ""))
        require(
            method in RELEASED_NEURAL_METHODS
            and record.get("seed") is None
            and method not in observed,
            f"{description} has unexpected or duplicate released probability record",
        )
        observed.add(method)
        availability = record.get("availability")
        require(
            record.get("released_threshold_retuned") is False
            and record.get("binary_accepted_as_probability") is False,
            f"{method}: released threshold/probability policy changed",
        )
        require(
            availability == "available",
            f"{method}: raw probability support exists; unavailable is not a valid release state",
        )
        require(
            record.get("threshold_source") == "released_fixed",
            f"{method}: zero-shot threshold source changed",
        )
        selected = _selected_threshold(
            record.get("selected_threshold"), f"{method} released selected threshold"
        )
        inference_path, inference_sha = _identity_from_record(
            record.get("inference_card"),
            restricted_root,
            f"{method} released inference card",
        )
        inference = _read_json(
            inference_path,
            f"{method} released inference card",
            expected_sha256=inference_sha,
        )
        require(
            inference.get("schema_version") == 1
            and inference.get("kind") == RELEASED_INFERENCE_CARD_KIND
            and inference.get("method") == method
            and inference.get("seed") is None,
            f"{method}: released inference-card identity changed",
        )
        _validate_delivery_producer(
            inference,
            expected_commit=expected_producer_commit,
            restricted_root=restricted_root,
            description=f"{method} released inference card",
        )
        require(
            inference.get("raw_probability_exposed") is True
            and inference.get("binary_accepted_as_probability") is False
            and inference.get("released_threshold_retuned") is False
            and inference.get("threshold_source") == "released_fixed"
            and _selected_threshold(
                inference.get("selected_threshold"),
                f"{method} released inference threshold",
            )
            == selected,
            f"{method}: released inference card violates probability retention",
        )
        validation = _validate_released_probability_card(
            record.get("validation_probability_card"),
            split="val",
            method=method,
            selected_threshold=selected,
            manifest_path=manifest_path,
            splits_path=splits_path,
            manifest_sha256=manifest_sha256,
            splits_sha256=splits_sha256,
            restricted_root=restricted_root,
            expected_universes=expected_universes,
        )
        test = _validate_released_probability_card(
            record.get("test_probability_card"),
            split="test",
            method=method,
            selected_threshold=selected,
            manifest_path=manifest_path,
            splits_path=splits_path,
            manifest_sha256=manifest_sha256,
            splits_sha256=splits_sha256,
            restricted_root=restricted_root,
            expected_universes=expected_universes,
        )
        identities = inference.get("identities")
        require(
            isinstance(identities, Mapping)
            and identities.get("validation_probability_card_sha256")
            == validation["sha256"]
            and identities.get("test_probability_card_sha256") == test["sha256"],
            f"{method}: released inference card does not bind probability cards",
        )
        frozen_run_path, frozen_run_sha = _identity_from_record(
            identities.get("frozen_stage1_run_card"),
            restricted_root,
            f"{method} frozen Stage 1 run card",
        )
        require(
            identities.get("frozen_stage1_run_card_sha256") == frozen_run_sha,
            f"{method}: frozen Stage 1 run-card identities disagree",
        )
        frozen_run = _read_json(
            frozen_run_path,
            f"{method} frozen Stage 1 run card",
            expected_sha256=frozen_run_sha,
        )
        reference = record.get("frozen_stage1_reference_artifact")
        require(
            isinstance(reference, Mapping),
            f"{method}: frozen Stage 1 run card is not anchored to registered provenance",
        )
        reference_name = str(reference.get("name", ""))
        require(
            reference_name in {
                "scoring_audit",
                "finalizer_audit",
                "journal",
                "table_metadata",
            }
            and reference_name in frozen_stage1_artifacts,
            f"{method}: invalid frozen Stage 1 reference artifact",
        )
        frozen_reference = frozen_stage1_artifacts[reference_name]
        reference_sha = _sha(
            frozen_reference.get("sha256"),
            f"frozen Stage 1 {reference_name} digest",
        )
        require(
            reference.get("sha256") == reference_sha,
            f"{method}: frozen Stage 1 reference digest changed",
        )
        reference_path = _require_regular(
            Path(str(frozen_reference.get("path", ""))),
            restricted_root,
            f"frozen Stage 1 {reference_name}",
        )
        require(
            frozen_run_sha.encode("ascii") in reference_path.read_bytes(),
            f"{method}: frozen Stage 1 provenance does not mention the run card",
        )
        frozen_prediction_sha = _sha(
            inference.get("frozen_prediction_set_sha256"),
            f"{method} frozen Stage 1 prediction-set digest",
        )
        current_prediction_sha = _sha(
            inference.get("probability_derived_prediction_set_sha256"),
            f"{method} probability-derived prediction-set digest",
        )
        frozen_bytes = _validate_frozen_released_prediction_set(
            method=method,
            frozen=frozen_run,
            manifest_path=manifest_path,
            splits_path=splits_path,
            restricted_root=restricted_root,
            expected_universe=expected_universes["test"],
        )
        require(
            inference.get("full_test_prediction_set_byte_identical") is True
            and frozen_prediction_sha == current_prediction_sha
            and frozen_prediction_sha == frozen_bytes["prediction_set_sha256"]
            and current_prediction_sha == test["prediction_set_sha256"]
            and int(inference.get("prediction_count", -1))
            == int(expected_universes["test"]["count"]),
            f"{method}: probability-derived predictions are not bound byte-for-byte to frozen Stage 1",
        )
        checked.append(
            {
                "method": method,
                "seed": None,
                "availability": "available",
                "selected_threshold": selected,
                "inference_card": {"path": str(inference_path), "sha256": inference_sha},
                "frozen_stage1_run_card_sha256": frozen_run_sha,
                "frozen_stage1_run_card": {
                    "path": str(frozen_run_path),
                    "sha256": frozen_run_sha,
                },
                "frozen_stage1_reference_artifact": {
                    "name": reference_name,
                    "sha256": reference_sha,
                },
                "prediction_set_sha256": frozen_prediction_sha,
                "validation_probability_card": validation,
                "test_probability_card": test,
            }
        )
    require(observed == set(RELEASED_NEURAL_METHODS), f"{description} released neural method universe changed")
    return {
        "binary_masks_accepted_as_probability_maps": False,
        "records": sorted(checked, key=lambda item: item["method"]),
    }


def _validate_delivery_outputs(
    delivery_path: Path,
    delivery: Mapping[str, Any],
    restricted_root: Path,
    description: str,
) -> tuple[Path, Dict[str, str]]:
    outputs = delivery.get("outputs")
    require(isinstance(outputs, Mapping) and outputs, f"{description} lacks output identities")
    normalized = {
        _safe_relative(relative, f"{description} output path"): _sha(
            digest, f"{description} output digest"
        )
        for relative, digest in outputs.items()
    }
    root = delivery_path.parent
    _require_under(root, restricted_root, f"{description} bundle")
    observed = _directory_inventory(root, exclude={delivery_path.name})
    require(observed == normalized, f"{description} bundle file universe or bytes differ from delivery")
    return root, normalized


def _validate_plan(plan: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    require(plan.get("schema_version") == 1, "Stage 4 plan schema changed")
    require(plan.get("kind") == PLAN_KIND, "Stage 4 plan kind changed")
    require(plan.get("results_opened_during_creation") is False, "Stage 4 plan was not result-blind")
    require(
        plan.get("heldout_payloads_opened_during_creation") is False,
        "Stage 4 plan opened held-out payloads",
    )
    universe = plan.get("method_universe")
    require(isinstance(universe, Mapping), "Stage 4 plan lacks method universe")
    methods = universe.get("methods")
    require(isinstance(methods, list) and methods, "Stage 4 plan has no methods")
    records: Dict[str, Mapping[str, Any]] = {}
    for record in methods:
        require(isinstance(record, Mapping), "method policy is not an object")
        key = str(record.get("key", ""))
        require(key and key not in records, f"empty or duplicate method key: {key!r}")
        role = record.get("role")
        require(role in {"method", "reference"}, f"{key}: invalid role")
        policy = record.get("seed_policy")
        require(isinstance(policy, Mapping), f"{key}: missing seed policy")
        kind = policy.get("kind")
        require(kind in {"seeded", "unseeded"}, f"{key}: invalid seed policy kind")
        if kind == "seeded":
            seeds = policy.get("seeds")
            require(
                isinstance(seeds, list)
                and seeds
                and all(type(seed) is int for seed in seeds)
                and len(seeds) == len(set(seeds)),
                f"{key}: invalid seed universe",
            )
            minimum = policy.get("minimum_successful")
            target = policy.get("target_successful")
            require(
                type(minimum) is int
                and type(target) is int
                and 1 <= minimum <= target == len(seeds),
                f"{key}: invalid seed success policy",
            )
        else:
            require(
                policy.get("minimum_successful") == 1
                and policy.get("target_successful") == 1
                and "seeds" not in policy,
                f"{key}: invalid unseeded policy",
            )
        records[key] = record
    require(
        int(universe.get("presentation_rows", -1)) == len(records),
        "method presentation-row count changed",
    )
    require(set(NATIVE_METHODS).issubset(records), "Stage 4 plan lacks a native method")
    test_universe = plan.get("test_universe")
    require(isinstance(test_universe, Mapping), "Stage 4 plan lacks test universe")
    table_counts = test_universe.get("tables")
    require(isinstance(table_counts, Mapping) and set(table_counts) == set(TABLES), "table universe changed")
    expected_groups: Optional[set[str]] = None
    for table in TABLES:
        groups = table_counts.get(table)
        require(isinstance(groups, Mapping) and groups, f"{table}: missing group universe")
        keys = {str(key) for key in groups}
        require("overall" in keys, f"{table}: overall group missing")
        require(all(type(value) is int and value > 0 for value in groups.values()), f"{table}: invalid counts")
        non_overall = keys - {"overall"}
        if expected_groups is None:
            expected_groups = non_overall
        else:
            require(non_overall == expected_groups, "A and A_prime group labels differ")
    return records


def _policy_seeds(policy: Mapping[str, Any]) -> list[Optional[int]]:
    return [int(seed) for seed in policy["seeds"]] if policy.get("kind") == "seeded" else [None]


def _validate_stage3_seed_evidence(
    delivery: Mapping[str, Any], policies: Mapping[str, Mapping[str, Any]]
) -> Dict[str, list[int]]:
    gate = delivery.get("gate_evidence")
    require(isinstance(gate, Mapping) and gate.get("status") == "PASS", "Stage 3 seed gate is not PASS")
    methods = gate.get("methods")
    require(isinstance(methods, Mapping), "Stage 3 delivery lacks method gate evidence")
    require(set(methods) == set(NATIVE_METHODS), "Stage 3 method gate universe changed")
    successful: Dict[str, list[int]] = {}
    for method in NATIVE_METHODS:
        evidence = methods[method]
        require(isinstance(evidence, Mapping), f"{method}: invalid gate evidence")
        policy = policies[method]["seed_policy"]
        allowed = {int(seed) for seed in policy["seeds"]}
        seeds = evidence.get("successful_seeds")
        failed = evidence.get("failed_seeds")
        missing = evidence.get("missing_seeds")
        require(
            isinstance(seeds, list)
            and isinstance(failed, list)
            and isinstance(missing, list)
            and all(type(seed) is int for seed in [*seeds, *failed, *missing]),
            f"{method}: invalid successful/failed/missing seeds",
        )
        success_set, failed_set, missing_set = set(seeds), set(failed), set(missing)
        require(len(success_set) == len(seeds), f"{method}: duplicate successful seed")
        require(not (success_set & failed_set or success_set & missing_set or failed_set & missing_set), f"{method}: overlapping seed states")
        require(success_set | failed_set | missing_set == allowed, f"{method}: seed universe differs")
        minimum = int(policy["minimum_successful"])
        require(len(success_set) >= minimum, f"{method}: only {len(success_set)} successful seeds; need {minimum}")
        require(evidence.get("pass") is True, f"{method}: delivery does not mark seed gate PASS")
        successful[method] = sorted(success_set)

    newly_scored = delivery.get("newly_scored_seeds")
    require(isinstance(newly_scored, list), "Stage 3 delivery lacks newly-scored seed evidence")
    observed = []
    for row in newly_scored:
        require(isinstance(row, Mapping), "Stage 3 newly-scored seed is not an object")
        method, seed = str(row.get("method", "")), row.get("seed")
        require(method in NATIVE_METHODS and type(seed) is int, "unexpected Stage 3 scored seed")
        observed.append((method, int(seed)))
    require(len(observed) == len(set(observed)), "duplicate Stage 3 newly-scored seed")
    expected = {
        (method, seed)
        for method, seeds in successful.items()
        for seed in seeds
        if not (method == "unet" and seed == 0)
    }
    require(set(observed) == expected, "Stage 3 newly-scored seed universe differs from its gate")
    require(
        0 in successful["unet"]
        and isinstance(delivery.get("stage2_unet_seed0"), Mapping)
        and delivery["stage2_unet_seed0"].get("reused_reselected_not_rescored")
        is True,
        "Stage 3 does not prove reuse of the separately reselected Stage 2 "
        "U-Net seed 0",
    )
    return successful


def _expected_runs(
    policies: Mapping[str, Mapping[str, Any]], stage3_success: Mapping[str, Sequence[int]]
) -> Dict[str, list[Optional[int]]]:
    runs: Dict[str, list[Optional[int]]] = {}
    for method, record in policies.items():
        if method in NATIVE_METHODS:
            runs[method] = [int(seed) for seed in stage3_success[method]]
        else:
            policy = record["seed_policy"]
            seeds = _policy_seeds(policy)
            require(
                int(policy["minimum_successful"]) == int(policy["target_successful"]),
                f"{method}: a non-native partial seed row is not supported by the frozen Stage 1 identity",
            )
            runs[method] = seeds
    return runs


def _summary_row_count(runs: Mapping[str, Sequence[Optional[int]]], groups: Mapping[str, Mapping[str, int]]) -> int:
    views = sum(len(table_groups) for table_groups in groups.values())
    return sum(len(seeds) for seeds in runs.values()) * views


def build_input_lock(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    amendment_path: Path,
    expected_amendment_sha256: str,
    stage1_paths: Mapping[str, Path],
    stage1_reselection_delivery_path: Path,
    stage2_delivery_path: Path,
    stage2_reselection_delivery_path: Path,
    stage3_delivery_path: Path,
    restricted_root: Path,
    code_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Authenticate upstream artifacts and replay their frozen scientific bytes.

    The analysis plan and threshold choices already exist before this step.
    Metric parsing here is integrity validation only; it performs no model,
    threshold, checkpoint, seed, or row selection.
    """
    expected_plan_sha256 = _sha(expected_plan_sha256, "registered Stage 4 plan digest")
    plan_identity = _identity(plan_path, restricted_root, "Stage 4 analysis plan")
    require(plan_identity["sha256"] == expected_plan_sha256, "Stage 4 plan differs from registration")
    plan = _read_json(Path(plan_identity["path"]), "Stage 4 analysis plan")
    policies = _validate_plan(plan)
    expected_builder_commit = _commit(
        code_identity.get("git_commit"), "Stage 4 finalizer code commit"
    )
    amendment, amendment_identity = _validate_amendment(
        amendment_path,
        expected_amendment_sha256,
        plan,
        expected_plan_sha256,
        restricted_root,
    )

    declared_stage1 = plan.get("inputs", {}).get("stage1")
    require(isinstance(declared_stage1, Mapping), "Stage 4 plan lacks Stage 1 identities")
    expected_stage1 = {
        "scoring_audit": declared_stage1.get("scoring_audit_sha256"),
        "finalizer_audit": declared_stage1.get("finalizer_audit_sha256"),
        "journal": declared_stage1.get("journal_sha256"),
        "summary": declared_stage1.get("summary", {}).get("sha256"),
        "table_metadata": declared_stage1.get("table_metadata_sha256"),
    }
    require(set(stage1_paths) == set(expected_stage1), "Stage 1 artifact argument set changed")
    stage1: Dict[str, Any] = {}
    for name, path in stage1_paths.items():
        identity = _identity(path, restricted_root, f"Stage 1 {name}")
        require(identity["sha256"] == _sha(expected_stage1[name], f"planned Stage 1 {name} digest"), f"Stage 1 {name} differs from preregistration")
        stage1[name] = identity
    expected_stage1_rows = declared_stage1.get("summary", {}).get("rows")
    require(type(expected_stage1_rows) is int and expected_stage1_rows > 0, "Stage 1 row count is not frozen")
    stage1["summary"]["declared_rows"] = int(expected_stage1_rows)
    stage1_artifact_set_sha256 = canonical_sha256(
        {name: identity["sha256"] for name, identity in sorted(stage1.items())}
    )

    stage1_reselection_identity = _identity(
        stage1_reselection_delivery_path,
        restricted_root,
        "Stage 1 threshold-reselection delivery",
    )
    stage1_reselection = _read_json(
        Path(stage1_reselection_identity["path"]),
        "Stage 1 threshold-reselection delivery",
    )
    require(
        stage1_reselection.get("schema_version") == 1
        and stage1_reselection.get("kind") == STAGE1_RESELECTION_KIND,
        "Stage 1 threshold-reselection delivery schema or kind changed",
    )
    _validate_delivery_producer(
        stage1_reselection,
        expected_commit=expected_builder_commit,
        restricted_root=restricted_root,
        description="Stage 1 threshold-reselection delivery",
        scientific_artifacts_validated=True,
    )
    require(
        stage1_reselection.get("supersedes")
        == {
            "stage1_summary_sha256": stage1["summary"]["sha256"],
            "methods": ["sufs_ft"],
        },
        "Stage 1 reselection does not supersede exactly the frozen SUFS-ft rows",
    )
    require(
        stage1_reselection.get("frozen_stage1_artifacts_sha256")
        == stage1_artifact_set_sha256,
        "Stage 1 reselection does not bind the exact frozen Stage 1 artifact set",
    )
    stage1_reselection_root, stage1_reselection_outputs = _validate_delivery_outputs(
        Path(stage1_reselection_identity["path"]),
        stage1_reselection,
        restricted_root,
        "Stage 1 threshold-reselection delivery",
    )
    require(
        "results/summary.csv" in stage1_reselection_outputs,
        "Stage 1 reselection lacks replacement summary",
    )

    stage2_identity = _identity(stage2_delivery_path, restricted_root, "Stage 2 delivery")
    stage2 = _read_json(Path(stage2_identity["path"]), "Stage 2 delivery")
    required_stage2_kind = plan.get("future_input_binding", {}).get("stage2", {}).get("required_kind")
    require(stage2.get("schema_version") == 1 and stage2.get("kind") == required_stage2_kind == STAGE2_KIND, "Stage 2 delivery schema or kind changed")

    stage2_reselection_identity = _identity(
        stage2_reselection_delivery_path,
        restricted_root,
        "Stage 2 threshold-reselection delivery",
    )
    stage2_reselection = _read_json(
        Path(stage2_reselection_identity["path"]),
        "Stage 2 threshold-reselection delivery",
    )
    require(
        stage2_reselection.get("schema_version") == 1
        and stage2_reselection.get("kind") == STAGE2_RESELECTION_KIND,
        "Stage 2 threshold-reselection delivery schema or kind changed",
    )
    _validate_delivery_producer(
        stage2_reselection,
        expected_commit=expected_builder_commit,
        restricted_root=restricted_root,
        description="Stage 2 threshold-reselection delivery",
        scientific_artifacts_validated=True,
    )
    require(
        stage2_reselection.get("supersedes")
        == {
            "stage2_delivery_sha256": stage2_identity["sha256"],
            "methods": ["unet"],
            "seeds": [0],
        },
        "Stage 2 reselection does not supersede exactly legacy U-Net seed 0",
    )
    stage2_reselection_root, stage2_reselection_outputs = _validate_delivery_outputs(
        Path(stage2_reselection_identity["path"]),
        stage2_reselection,
        restricted_root,
        "Stage 2 threshold-reselection delivery",
    )
    require(
        "results/summary.csv" in stage2_reselection_outputs,
        "Stage 2 reselection lacks replacement summary",
    )
    stage2_reselection_rows_sha256 = _sha(
        stage2_reselection.get("scientific_rows_sha256"),
        "Stage 2 replacement scientific-row digest",
    )

    stage3_identity = _identity(stage3_delivery_path, restricted_root, "Stage 3 delivery")
    stage3 = _read_json(Path(stage3_identity["path"]), "Stage 3 delivery")
    required_stage3_kind = plan.get("future_input_binding", {}).get("stage3", {}).get("required_kind")
    require(stage3.get("schema_version") == 1 and stage3.get("kind") == required_stage3_kind == STAGE3_KIND, "Stage 3 delivery schema or kind changed")
    _validate_delivery_producer(
        stage3,
        expected_commit=expected_builder_commit,
        restricted_root=restricted_root,
        description="Stage 3 delivery",
        scientific_artifacts_validated=True,
    )
    manifest_sha256 = _sha(
        plan.get("inputs", {}).get("manifest", {}).get("sha256"),
        "planned manifest digest",
    )
    splits_sha256 = _sha(
        plan.get("inputs", {}).get("splits", {}).get("sha256"),
        "planned split digest",
    )
    require(stage3.get("manifest_sha256") == manifest_sha256, "Stage 3 manifest identity differs from plan")
    require(stage3.get("splits_sha256") == splits_sha256, "Stage 3 split identity differs from plan")
    reuse = stage3.get("stage2_unet_seed0")
    require(
        isinstance(reuse, Mapping)
        and reuse.get("legacy_provenance_delivery_sha256") == stage2_identity["sha256"]
        and reuse.get("threshold_reselection_delivery_sha256")
        == stage2_reselection_identity["sha256"]
        and reuse.get("replacement_scientific_rows_sha256")
        == stage2_reselection_rows_sha256
        and reuse.get("reused_reselected_not_rescored") is True,
        "Stage 3 does not bind both legacy and reselected Stage 2 U-Net seed 0",
    )

    stage3_root, normalized_outputs = _validate_delivery_outputs(
        Path(stage3_identity["path"]),
        stage3,
        restricted_root,
        "Stage 3 delivery",
    )
    require("results/summary.csv" in normalized_outputs, "Stage 3 delivery lacks results/summary.csv")

    input_identities = [
        _validate_data_inputs(delivery, restricted_root, description)
        for delivery, description in (
            (stage1_reselection, "Stage 1 threshold-reselection delivery"),
            (stage2_reselection, "Stage 2 threshold-reselection delivery"),
            (stage3, "Stage 3 delivery"),
        )
    ]
    manifest_path, splits_path, carried_manifest_sha, carried_splits_sha = input_identities[0]
    require(
        all(
            (item[2], item[3]) == (carried_manifest_sha, carried_splits_sha)
            for item in input_identities
        )
        and carried_manifest_sha == manifest_sha256
        and carried_splits_sha == splits_sha256,
        "Stage 1/2/3 data-input identities differ from the frozen plan",
    )
    expected_universes = _manifest_split_probability_universes(
        manifest_path, splits_path
    )
    expected_test_count = int(expected_universes["test"]["count"])
    require(
        expected_test_count == int(plan["test_universe"]["tables"]["A"]["overall"]),
        "manifest/split-derived test count differs from the frozen plan",
    )
    expected_test_universe_sha256 = _sha(
        expected_universes["test"]["ordered_source_images_sha256"],
        "manifest-derived ordered test source-image digest",
    )
    require(
        stage3.get("ordered_source_images_sha256") == expected_test_universe_sha256,
        "Stage 3 ordered test source-image digest differs from manifest/split",
    )
    for delivery, description in (
        (stage1_reselection, "Stage 1 threshold-reselection delivery"),
        (stage2_reselection, "Stage 2 threshold-reselection delivery"),
    ):
        require(
            delivery.get("ordered_source_images_sha256")
            == expected_test_universe_sha256,
            f"{description} test source-image universe differs from Stage 3",
        )
    amendment_sha256 = amendment_identity["sha256"]
    stage1_threshold_evidence = _validate_threshold_attestation(
        stage1_reselection,
        required_records={
            ("sufs_ft", int(seed))
            for seed in policies["sufs_ft"]["seed_policy"]["seeds"]
        },
        legacy_records={
            ("sufs_ft", int(seed))
            for seed in policies["sufs_ft"]["seed_policy"]["seeds"]
        },
        amendment_sha256=amendment_sha256,
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_path,
        splits_path=splits_path,
        splits_sha256=splits_sha256,
        restricted_root=restricted_root,
        expected_universes=expected_universes,
        scientific_summary_sha256=stage1_reselection_outputs["results/summary.csv"],
        scientific_summary_path=stage1_reselection_root / "results" / "summary.csv",
        expected_producer_commit=expected_builder_commit,
        description="Stage 1 threshold-reselection delivery",
    )
    stage2_threshold_evidence = _validate_threshold_attestation(
        stage2_reselection,
        required_records={("unet", 0)},
        legacy_records={("unet", 0)},
        amendment_sha256=amendment_sha256,
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_path,
        splits_path=splits_path,
        splits_sha256=splits_sha256,
        restricted_root=restricted_root,
        expected_universes=expected_universes,
        scientific_summary_sha256=stage2_reselection_outputs["results/summary.csv"],
        scientific_summary_path=stage2_reselection_root / "results" / "summary.csv",
        expected_producer_commit=expected_builder_commit,
        description="Stage 2 threshold-reselection delivery",
    )
    released_probability_evidence = _validate_released_probability_retention(
        stage1_reselection,
        frozen_stage1_artifacts=stage1,
        manifest_path=manifest_path,
        splits_path=splits_path,
        manifest_sha256=manifest_sha256,
        splits_sha256=splits_sha256,
        restricted_root=restricted_root,
        expected_universes=expected_universes,
        expected_producer_commit=expected_builder_commit,
        description="Stage 1 threshold-reselection delivery",
    )

    stage3_success = _validate_stage3_seed_evidence(stage3, policies)
    # Probability retention is scoped to final-table full runs: discarded LR
    # candidates never produced held-out outputs.  The two pre-amendment
    # U-Net++ sweep candidates are separately required upstream to replay their
    # expanded-grid LR selection from retained raw validation NPYs.  Nothing in
    # this final-run attestation weakens that legacy-sweep exception.
    stage3_amended_records = {
        (method, seed)
        for method, seeds in stage3_success.items()
        if method in TRAINABLE_NEURAL_METHODS
        for seed in seeds
        if not (method == "unet" and seed == 0)
    }
    stage3_scored_rows = stage3.get("newly_scored_seeds")
    require(
        isinstance(stage3_scored_rows, list),
        "Stage 3 delivery lacks frozen per-seed scoring lineage",
    )
    stage3_training_identities = {
        (str(row.get("method", "")), int(row.get("seed", -1))): row
        for row in stage3_scored_rows
        if isinstance(row, Mapping) and type(row.get("seed")) is int
    }
    require(
        len(stage3_training_identities) == len(stage3_scored_rows),
        "Stage 3 delivery duplicates/malforms frozen per-seed scoring lineage",
    )
    stage3_threshold_evidence = _validate_threshold_attestation(
        stage3,
        required_records=stage3_amended_records,
        legacy_records=set(),
        amendment_sha256=amendment_sha256,
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_path,
        splits_path=splits_path,
        splits_sha256=splits_sha256,
        restricted_root=restricted_root,
        expected_universes=expected_universes,
        scientific_summary_sha256=normalized_outputs["results/summary.csv"],
        scientific_summary_path=stage3_root / "results" / "summary.csv",
        expected_producer_commit=expected_builder_commit,
        description="Stage 3 delivery",
        expected_training_identities=stage3_training_identities,
    )
    runs = _expected_runs(policies, stage3_success)
    stage1_runs = {
        method: seeds
        for method, seeds in runs.items()
        if method not in NATIVE_METHODS and method != "sufs_ft"
    }
    stage1_reselection_runs = {"sufs_ft": runs["sufs_ft"]}
    stage3_runs = {method: seeds for method, seeds in runs.items() if method in NATIVE_METHODS}
    table_counts = plan["test_universe"]["tables"]
    stage1_scientific_rows = _summary_row_count(stage1_runs, table_counts)
    stage1_reselection_rows = _summary_row_count(stage1_reselection_runs, table_counts)
    require(
        stage1_scientific_rows + stage1_reselection_rows == expected_stage1_rows,
        "Stage 1 row count does not match the preregistered method/seed/group universe",
    )
    stage3_rows = _summary_row_count(stage3_runs, table_counts)

    code_commit = expected_builder_commit
    code_sha = _sha(code_identity.get("script_sha256"), "Stage 4 finalizer code digest")
    require(code_identity.get("git_dirty") is False, "Stage 4 finalizer code is dirty")
    return {
        "schema_version": 1,
        "kind": LOCK_KIND,
        "automatic_heldout_rescore_permitted": False,
        "stage4_reselection_permitted": False,
        "creation_scope": {
            "scientific_csv_content_parsed": True,
            "metric_values_inspected": True,
            "metric_use": "integrity_replay_only_no_selection",
            "model_selection_performed": False,
            "artifact_bytes_hashed": True,
            "delivery_metadata_validated": True,
        },
        "plan": plan_identity,
        "threshold_policy_amendment": {
            "artifact": amendment_identity,
            "kind": AMENDMENT_KIND,
            "supersedes_exactly": ["threshold_policy"],
            "prior_grid": list(PRIOR_THRESHOLD_GRID),
            "replacement_grid": list(REPLACEMENT_THRESHOLD_GRID),
            "validation_only": True,
            "affected_methods": list(TRAINABLE_NEURAL_METHODS),
        },
        "code": {
            "git_commit": code_commit,
            "git_dirty": False,
            "script": "segbench/scripts/finalize_stage4.py",
            "script_sha256": code_sha,
        },
        "stage1": {
            "role": "immutable provenance plus unaffected scientific rows",
            "artifacts": stage1,
            "excluded_superseded_methods": ["sufs_ft"],
        },
        "stage1_reselection": {
            "role": "superseding scientific rows for SUFS-ft",
            "delivery": stage1_reselection_identity,
            "kind": STAGE1_RESELECTION_KIND,
            "bundle_root": str(stage1_reselection_root.resolve()),
            "bundle_output_count": len(stage1_reselection_outputs),
            "bundle_outputs_sha256": canonical_sha256(stage1_reselection_outputs),
            "summary": {
                "path": str((stage1_reselection_root / "results" / "summary.csv").resolve()),
                "sha256": stage1_reselection_outputs["results/summary.csv"],
                "declared_rows": stage1_reselection_rows,
            },
            "threshold_policy_evidence": stage1_threshold_evidence,
            "released_probability_retention_evidence": released_probability_evidence,
        },
        "stage2": {
            "role": "immutable pre-amendment provenance only",
            "delivery": stage2_identity,
            "kind": STAGE2_KIND,
        },
        "stage2_reselection": {
            "role": "superseding U-Net seed 0 provenance; summary owned by Stage 3",
            "delivery": stage2_reselection_identity,
            "kind": STAGE2_RESELECTION_KIND,
            "bundle_root": str(stage2_reselection_root.resolve()),
            "bundle_output_count": len(stage2_reselection_outputs),
            "bundle_outputs_sha256": canonical_sha256(stage2_reselection_outputs),
            "summary": {
                "path": str(
                    (stage2_reselection_root / "results" / "summary.csv").resolve()
                ),
                "sha256": stage2_reselection_outputs["results/summary.csv"],
            },
            "scientific_rows_sha256": stage2_reselection_rows_sha256,
            "threshold_policy_evidence": stage2_threshold_evidence,
        },
        "stage3": {
            "role": "scientific input",
            "delivery": stage3_identity,
            "kind": STAGE3_KIND,
            "bundle_root": str(stage3_root.resolve()),
            "bundle_output_count": len(normalized_outputs),
            "bundle_outputs_sha256": canonical_sha256(normalized_outputs),
            "ordered_source_images_sha256": expected_test_universe_sha256,
            "summary": {
                "path": str((stage3_root / "results" / "summary.csv").resolve()),
                "sha256": normalized_outputs["results/summary.csv"],
                "declared_rows": stage3_rows,
            },
            "successful_seeds": stage3_success,
            "threshold_policy_evidence": stage3_threshold_evidence,
        },
        "scientific_inputs": {
            "combination": (
                "unaffected rows from frozen Stage 1 summary plus superseding "
                "SUFS-ft rows plus Stage 3 delivery summary"
            ),
            "stage2_summary_included": False,
            "summaries": [
                {
                    "owner": "stage1",
                    **stage1["summary"],
                    "excluded_methods": ["sufs_ft"],
                    "included_rows": stage1_scientific_rows,
                },
                {
                    "owner": "stage1_reselection",
                    "path": str((stage1_reselection_root / "results" / "summary.csv").resolve()),
                    "sha256": stage1_reselection_outputs["results/summary.csv"],
                    "declared_rows": stage1_reselection_rows,
                    "excluded_methods": [],
                    "included_rows": stage1_reselection_rows,
                },
                {
                    "owner": "stage3",
                    **{
                        "path": str((stage3_root / "results" / "summary.csv").resolve()),
                        "sha256": normalized_outputs["results/summary.csv"],
                        "declared_rows": stage3_rows,
                    },
                },
            ],
        },
        "expected_universe": {
            "tables": table_counts,
            "method_roles": {method: str(record["role"]) for method, record in policies.items()},
            "method_owner": {
                method: (
                    "stage3"
                    if method in NATIVE_METHODS
                    else "stage1_reselection"
                    if method == "sufs_ft"
                    else "stage1"
                )
                for method in policies
            },
            "runs": {
                method: ["unseeded" if seed is None else int(seed) for seed in seeds]
                for method, seeds in runs.items()
            },
            "stage1_physical_summary_rows": expected_stage1_rows,
            "stage1_scientific_rows": stage1_scientific_rows,
            "stage1_reselection_summary_rows": stage1_reselection_rows,
            "stage3_summary_rows": stage3_rows,
            "combined_summary_rows": stage1_scientific_rows + stage1_reselection_rows + stage3_rows,
        },
    }


def verify_input_lock(lock_path: Path, restricted_root: Path, code_identity: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-derive the complete input lock and require byte-semantic identity."""
    lock_resolved = _require_regular(lock_path, restricted_root, "Stage 4 input lock")
    lock = _read_json(lock_resolved, "Stage 4 input lock")
    require(lock.get("schema_version") == 1 and lock.get("kind") == LOCK_KIND, "Stage 4 input lock schema or kind changed")
    require(lock.get("automatic_heldout_rescore_permitted") is False, "input lock permits held-out rescoring")
    code = lock.get("code")
    require(isinstance(code, Mapping), "input lock lacks code identity")
    require(code.get("git_commit") == code_identity.get("git_commit"), "input lock was produced by a different commit")
    require(code.get("git_dirty") is False and code_identity.get("git_dirty") is False, "input lock or current code is dirty")
    require(code.get("script_sha256") == code_identity.get("script_sha256") == sha256_file(Path(__file__)), "Stage 4 finalizer bytes differ from input lock")
    plan_record = lock.get("plan")
    amendment_record = lock.get("threshold_policy_amendment", {}).get("artifact")
    artifacts = lock.get("stage1", {}).get("artifacts")
    require(isinstance(plan_record, Mapping), "input lock lacks Stage 4 plan identity")
    require(isinstance(amendment_record, Mapping), "input lock lacks threshold amendment identity")
    require(isinstance(artifacts, Mapping), "input lock lacks Stage 1 artifacts")
    required_stage1 = {
        "scoring_audit",
        "finalizer_audit",
        "journal",
        "summary",
        "table_metadata",
    }
    require(set(artifacts) == required_stage1, "input lock Stage 1 artifact universe changed")
    candidate = build_input_lock(
        plan_path=Path(str(plan_record.get("path", ""))),
        expected_plan_sha256=str(plan_record.get("sha256", "")),
        amendment_path=Path(str(amendment_record.get("path", ""))),
        expected_amendment_sha256=str(amendment_record.get("sha256", "")),
        stage1_paths={
            name: Path(str(artifacts[name].get("path", ""))) for name in required_stage1
        },
        stage1_reselection_delivery_path=Path(
            str(lock.get("stage1_reselection", {}).get("delivery", {}).get("path", ""))
        ),
        stage2_delivery_path=Path(
            str(lock.get("stage2", {}).get("delivery", {}).get("path", ""))
        ),
        stage2_reselection_delivery_path=Path(
            str(
                lock.get("stage2_reselection", {})
                .get("delivery", {})
                .get("path", "")
            )
        ),
        stage3_delivery_path=Path(
            str(lock.get("stage3", {}).get("delivery", {}).get("path", ""))
        ),
        restricted_root=restricted_root,
        code_identity=code_identity,
    )
    require(lock == candidate, "Stage 4 input lock differs from its deterministic derivation")
    return lock


def _canonical_seed(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return "unseeded"
    try:
        numeric = float(text)
    except ValueError as exc:
        raise Stage4Error(f"invalid seed value {text!r}") from exc
    require(math.isfinite(numeric) and numeric.is_integer(), f"invalid seed value {text!r}")
    return str(int(numeric))


def _integer_cell(value: object, description: str) -> int:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise Stage4Error(f"{description} is not numeric") from exc
    require(math.isfinite(numeric) and numeric.is_integer(), f"{description} is not an integer")
    return int(numeric)


def _read_summary_universe(
    record: Mapping[str, Any],
    owner: str,
    expected: Mapping[str, Any],
) -> Dict[str, Any]:
    path = Path(str(record["path"]))
    require(sha256_file(path) == record["sha256"], f"{owner} summary changed before preflight")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"method", "seed", "table", "group_value", "n_images", "n_missing", *(f"{metric}_mean" for metric in METRICS)}
        columns = set(reader.fieldnames or ())
        require(required.issubset(columns), f"{owner} summary lacks columns: {sorted(required - columns)}")
        rows = list(reader)
    require(len(rows) == int(record["declared_rows"]), f"{owner} summary row count changed")

    method_owner = expected["method_owner"]
    runs = expected["runs"]
    tables = expected["tables"]
    excluded_methods = set(record.get("excluded_methods", []))
    require(
        all(method in expected["method_owner"] for method in excluded_methods),
        f"{owner} summary declares an unknown excluded method",
    )
    identities = set()
    excluded_identities = set()
    observed_runs: Dict[str, set[str]] = {}
    for index, row in enumerate(rows, 2):
        method = str(row.get("method", "")).strip()
        require(method in method_owner, f"{owner} row {index}: unexpected method {method!r}")
        seed = _canonical_seed(row.get("seed"))
        allowed_seeds = {str(value) for value in runs[method]}
        require(seed in allowed_seeds, f"{owner} row {index}: unexpected seed {seed} for {method}")
        table = str(row.get("table", "")).strip()
        group = str(row.get("group_value", "")).strip()
        require(table in tables, f"{owner} row {index}: unexpected table {table!r}")
        require(group in tables[table], f"{owner} row {index}: unexpected group {group!r} for {table}")
        identity = (method, seed, table, group)
        if method in excluded_methods:
            require(
                method_owner[method] != owner,
                f"{owner}: cannot exclude an owned scientific method {method}",
            )
            require(identity not in excluded_identities, f"{owner}: duplicate excluded identity {identity}")
            excluded_identities.add(identity)
            continue
        require(method_owner[method] == owner, f"{owner} row {index}: method {method!r} belongs to {method_owner[method]}")
        require(identity not in identities, f"{owner}: duplicate scientific identity {identity}")
        identities.add(identity)
        observed_runs.setdefault(method, set()).add(seed)
        require(_integer_cell(row.get("n_images"), f"{owner} row {index} n_images") == int(tables[table][group]), f"{owner} row {index}: image/group universe changed")
        require(_integer_cell(row.get("n_missing"), f"{owner} row {index} n_missing") == 0, f"{owner} row {index}: nonzero missing prediction count")

    expected_methods = {method for method, candidate_owner in method_owner.items() if candidate_owner == owner}
    require(set(observed_runs) == expected_methods, f"{owner} method universe changed")
    for method in expected_methods:
        require(observed_runs[method] == {str(value) for value in runs[method]}, f"{owner}: seed universe changed for {method}")
    expected_identities = {
        (method, str(seed), table, group)
        for method in expected_methods
        for seed in runs[method]
        for table, group_counts in tables.items()
        for group in group_counts
    }
    require(identities == expected_identities, f"{owner} summary has missing or unexpected method/seed/table/group rows")
    expected_excluded = {
        (method, str(seed), table, group)
        for method in excluded_methods
        for seed in runs[method]
        for table, group_counts in tables.items()
        for group in group_counts
    }
    require(
        excluded_identities == expected_excluded,
        f"{owner} summary superseded-row universe changed",
    )
    require(
        len(identities) == int(record.get("included_rows", len(identities))),
        f"{owner} included scientific-row count changed",
    )
    return {
        "owner": owner,
        "path": str(path.resolve()),
        "sha256": record["sha256"],
        "physical_rows": len(rows),
        "rows": len(identities),
        "excluded_superseded_rows": len(excluded_identities),
        "scientific_identity_sha256": canonical_sha256(sorted(identities)),
        "methods": {method: sorted(seeds) for method, seeds in sorted(observed_runs.items())},
        "metric_columns_present": [f"{metric}_mean" for metric in METRICS],
        "metric_values_parsed_by_structural_pass": False,
    }


def preflight_summaries(lock_path: Path, restricted_root: Path, code_identity: Mapping[str, Any]) -> Dict[str, Any]:
    lock = verify_input_lock(lock_path, restricted_root, code_identity)
    summaries = lock.get("scientific_inputs", {}).get("summaries")
    require(isinstance(summaries, list) and len(summaries) == 3, "input lock must contain original Stage 1, superseding SUFS-ft, and Stage 3 summaries")
    by_owner = {str(record.get("owner")): record for record in summaries if isinstance(record, Mapping)}
    require(
        set(by_owner) == {"stage1", "stage1_reselection", "stage3"},
        "scientific summary ownership changed",
    )
    expected = lock.get("expected_universe")
    require(isinstance(expected, Mapping), "input lock lacks expected universe")
    records = [
        _read_summary_universe(by_owner[owner], owner, expected)
        for owner in ("stage1", "stage1_reselection", "stage3")
    ]
    require(sum(record["rows"] for record in records) == int(expected["combined_summary_rows"]), "combined summary row count changed")
    return {
        "schema_version": 1,
        "kind": PREFLIGHT_KIND,
        "status": "PASS",
        "input_lock_sha256": sha256_file(lock_path),
        "automatic_heldout_rescore_permitted": False,
        "scientific_csv_scope": {
            "row_identity_parsed": True,
            "image_and_group_counts_parsed": True,
            "missing_prediction_counts_parsed": True,
            # verify_input_lock independently replays the evaluator metrics
            # before this structural summary pass.
            "metric_values_parsed": True,
            "metric_use": "integrity_replay_only_no_selection",
            "model_selection_performed": False,
        },
        "summaries": records,
        "combined_rows": sum(record["rows"] for record in records),
        "combined_scientific_identity_sha256": canonical_sha256(
            [record["scientific_identity_sha256"] for record in records]
        ),
    }


def _validate_output_contract(
    contract_path: Path,
    receipt_path: Path,
    input_lock_path: Path,
    staging_dir: Path,
    restricted_root: Path,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    contract_resolved = _require_regular(contract_path, restricted_root, "Stage 4 output contract")
    receipt_resolved = _require_regular(receipt_path, restricted_root, "Stage 4 preflight receipt")
    staging_resolved = _require_under(staging_dir, restricted_root, "Stage 4 output staging directory")
    require(staging_dir.is_dir() and not staging_dir.is_symlink(), "Stage 4 output staging directory is absent or a symlink")
    contract = _read_json(contract_resolved, "Stage 4 output contract")
    receipt = _read_json(receipt_resolved, "Stage 4 preflight receipt")
    require(contract.get("schema_version") == 1 and contract.get("kind") == CONTRACT_KIND, "Stage 4 output contract schema or kind changed")
    require(receipt.get("schema_version") == 1 and receipt.get("kind") == PREFLIGHT_KIND and receipt.get("status") == "PASS", "Stage 4 preflight receipt is not PASS")
    input_lock_sha = sha256_file(input_lock_path)
    require(contract.get("input_lock_sha256") == receipt.get("input_lock_sha256") == input_lock_sha, "output contract/preflight does not bind this input lock")
    require(contract.get("preflight_receipt_sha256") == sha256_file(receipt_resolved), "output contract does not bind this preflight receipt")
    files = contract.get("files")
    require(isinstance(files, Mapping) and files, "output contract has no files")
    expected = {_safe_relative(path, "Stage 4 output path"): _sha(digest, "Stage 4 output digest") for path, digest in files.items()}
    observed = _directory_inventory(staging_resolved)
    require(observed == expected, "Stage 4 staging file universe or bytes differ from output contract")
    return contract, expected


def _verify_published_delivery(
    bundle_out: Path,
    input_lock_path: Path,
    receipt_path: Path,
    contract_path: Path,
    restricted_root: Path,
) -> Dict[str, Any]:
    bundle_resolved = _require_under(bundle_out, restricted_root, "published Stage 4 bundle")
    require(
        bundle_out.is_dir() and not bundle_out.is_symlink(),
        "published Stage 4 bundle is absent or a symlink",
    )
    delivery_path = bundle_resolved / "delivery.json"
    delivery = _read_json(delivery_path, "Stage 4 delivery")
    require(delivery.get("schema_version") == 1 and delivery.get("kind") == DELIVERY_KIND, "Stage 4 delivery schema or kind changed")
    require(delivery.get("input_lock_sha256") == sha256_file(input_lock_path), "published delivery input lock changed")
    require(delivery.get("preflight_receipt_sha256") == sha256_file(receipt_path), "published delivery preflight changed")
    require(delivery.get("output_contract_sha256") == sha256_file(contract_path), "published delivery contract changed")
    receipt = _read_json(
        _require_regular(receipt_path, restricted_root, "Stage 4 preflight receipt"),
        "Stage 4 preflight receipt",
    )
    contract = _read_json(
        _require_regular(contract_path, restricted_root, "Stage 4 output contract"),
        "Stage 4 output contract",
    )
    require(
        receipt.get("schema_version") == 1
        and receipt.get("kind") == PREFLIGHT_KIND
        and receipt.get("status") == "PASS"
        and receipt.get("input_lock_sha256") == sha256_file(input_lock_path),
        "published Stage 4 preflight receipt is not a bound PASS",
    )
    require(
        contract.get("schema_version") == 1
        and contract.get("kind") == CONTRACT_KIND
        and contract.get("input_lock_sha256") == sha256_file(input_lock_path)
        and contract.get("preflight_receipt_sha256") == sha256_file(receipt_path),
        "published Stage 4 output contract is not bound",
    )
    outputs = delivery.get("outputs")
    require(isinstance(outputs, Mapping), "published Stage 4 delivery lacks outputs")
    expected = {_safe_relative(path, "published Stage 4 output path"): _sha(digest, "published output digest") for path, digest in outputs.items()}
    contract_files = contract.get("files")
    require(isinstance(contract_files, Mapping), "published Stage 4 output contract lacks files")
    normalized_contract = {
        _safe_relative(path, "contracted Stage 4 output path"): _sha(
            digest, "contracted Stage 4 output digest"
        )
        for path, digest in contract_files.items()
    }
    require(expected == normalized_contract, "published Stage 4 delivery differs from output contract")
    require(_directory_inventory(bundle_resolved, exclude={"delivery.json"}) == expected, "published Stage 4 output bytes changed")
    return delivery


def publish_outputs(
    *,
    input_lock_path: Path,
    receipt_path: Path,
    contract_path: Path,
    staging_dir: Path,
    bundle_out: Path,
    restricted_root: Path,
    code_identity: Mapping[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    verify_input_lock(input_lock_path, restricted_root, code_identity)
    receipt = _read_json(receipt_path, "Stage 4 preflight receipt")
    require(receipt.get("input_lock_sha256") == sha256_file(input_lock_path), "preflight receipt does not bind input lock")
    _, outputs = _validate_output_contract(
        contract_path, receipt_path, input_lock_path, staging_dir, restricted_root
    )
    _require_under(bundle_out, restricted_root, "Stage 4 publication bundle")
    require(bundle_out.resolve() != staging_dir.resolve(), "Stage 4 staging and publication directories must differ")
    delivery = {
        "schema_version": 1,
        "kind": DELIVERY_KIND,
        "input_lock_sha256": sha256_file(input_lock_path),
        "preflight_receipt_sha256": sha256_file(receipt_path),
        "output_contract_sha256": sha256_file(contract_path),
        "outputs": outputs,
        "automatic_heldout_rescore_permitted": False,
        "provenance": {
            "git_commit": code_identity["git_commit"],
            "git_dirty": False,
            "script_sha256": code_identity["script_sha256"],
        },
    }
    if dry_run:
        require(not bundle_out.exists(), "dry-run refuses an already-published bundle; use verify-delivery")
        return delivery
    if bundle_out.exists():
        require(bundle_out.is_dir() and not bundle_out.is_symlink(), "published Stage 4 bundle is not a directory")
        return _verify_published_delivery(
            bundle_out, input_lock_path, receipt_path, contract_path, restricted_root
        )

    bundle_out.parent.mkdir(parents=True, exist_ok=True)
    require(
        bundle_out.parent.is_dir() and not bundle_out.parent.is_symlink(),
        f"Stage 4 publication parent is absent or a symlink: {bundle_out.parent}",
    )
    lock_path = bundle_out.parent / f".{bundle_out.name}.stage4-publish.lock"
    lock_flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        lock_fd = os.open(lock_path, lock_flags, 0o600)
    except FileExistsError as exc:
        raise Stage4Error(f"Stage 4 publication lock already exists: {lock_path}") from exc
    except OSError as exc:
        raise Stage4Error(f"cannot create Stage 4 publication lock: {lock_path}: {exc}") from exc
    lock_metadata = os.fstat(lock_fd)
    stage: Optional[Path] = None
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        stage = Path(tempfile.mkdtemp(prefix=f".{bundle_out.name}-stage4-", dir=bundle_out.parent))
        for relative, digest in outputs.items():
            source = staging_dir / relative
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            require(sha256_file(target) == digest, f"Stage 4 output changed while copying: {relative}")
        require(_directory_inventory(stage) == outputs, "copied Stage 4 output inventory changed")
        _publish_json_once(stage / "delivery.json", delivery)
        try:
            _rename_directory_noreplace(stage, bundle_out)
        except FileExistsError:
            # A concurrent publisher (or an injected destination) won the
            # name. Never replace it: only an exact, complete delivery is an
            # acceptable replay.
            pass
        else:
            stage = None
        if stage is not None:
            shutil.rmtree(stage)
            stage = None
        result = _verify_published_delivery(
            bundle_out, input_lock_path, receipt_path, contract_path, restricted_root
        )
        _unlink_owned_regular(lock_path, lock_metadata)
        return result
    except BaseException:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        _unlink_owned_regular(lock_path, lock_metadata)
        raise
    finally:
        os.close(lock_fd)


def git_code_identity(repo_root: Path, expected_commit: str) -> Dict[str, Any]:
    expected_commit = _commit(expected_commit, "Stage 4 finalizer commit")
    repo_root = repo_root.resolve()

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments], cwd=repo_root, check=False, capture_output=True
        )
        require(completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace").strip())
        return completed.stdout

    require(Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() == repo_root, "--repo-root is not the Git top level")
    require(git("rev-parse", "HEAD").decode("ascii").strip() == expected_commit, "repository HEAD differs from expected finalizer commit")
    require(not git("status", "--porcelain"), "repository is dirty; refuse Stage 4 orchestration")
    script = Path(__file__).resolve()
    try:
        relative = script.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise Stage4Error("Stage 4 finalizer script is outside the repository") from exc
    git("ls-files", "--error-unmatch", relative)
    committed = git("show", f"{expected_commit}:{relative}")
    require(committed == script.read_bytes(), "Stage 4 finalizer bytes differ from expected commit")
    return {
        "git_commit": expected_commit,
        "git_dirty": False,
        "script_sha256": hashlib.sha256(committed).hexdigest(),
    }


def _add_code_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--restricted-root", required=True, type=Path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    lock = commands.add_parser("lock", help="freeze Stage 4 upstream identities")
    _add_code_args(lock)
    lock.add_argument("--plan", required=True, type=Path)
    lock.add_argument("--expected-plan-sha256", required=True)
    lock.add_argument("--threshold-amendment", required=True, type=Path)
    lock.add_argument("--expected-threshold-amendment-sha256", required=True)
    lock.add_argument("--stage1-scoring-audit", required=True, type=Path)
    lock.add_argument("--stage1-finalizer-audit", required=True, type=Path)
    lock.add_argument("--stage1-journal", required=True, type=Path)
    lock.add_argument("--stage1-summary", required=True, type=Path)
    lock.add_argument("--stage1-table-metadata", required=True, type=Path)
    lock.add_argument("--stage1-reselection-delivery", required=True, type=Path)
    lock.add_argument("--stage2-delivery", required=True, type=Path)
    lock.add_argument("--stage2-reselection-delivery", required=True, type=Path)
    lock.add_argument("--stage3-delivery", required=True, type=Path)
    lock.add_argument("--out", required=True, type=Path)
    lock.add_argument("--dry-run", action="store_true")

    preflight = commands.add_parser("preflight", help="validate the frozen summary universe")
    _add_code_args(preflight)
    preflight.add_argument("--input-lock", required=True, type=Path)
    preflight.add_argument("--receipt-out", type=Path)

    publish = commands.add_parser("publish", help="atomically publish contracted Stage 4 outputs")
    _add_code_args(publish)
    publish.add_argument("--input-lock", required=True, type=Path)
    publish.add_argument("--preflight-receipt", required=True, type=Path)
    publish.add_argument("--output-contract", required=True, type=Path)
    publish.add_argument("--staging-dir", required=True, type=Path)
    publish.add_argument("--bundle-out", required=True, type=Path)
    publish.add_argument("--dry-run", action="store_true")

    verify = commands.add_parser("verify-delivery", help="verify an existing Stage 4 delivery")
    _add_code_args(verify)
    verify.add_argument("--input-lock", required=True, type=Path)
    verify.add_argument("--preflight-receipt", required=True, type=Path)
    verify.add_argument("--output-contract", required=True, type=Path)
    verify.add_argument("--bundle-out", required=True, type=Path)
    return parser.parse_args(argv)


def _require_registered_production_lock(lock_path: Path, restricted_root: Path) -> None:
    """Reject locks not rooted in the two externally frozen production artifacts."""
    lock = _read_json(
        _require_regular(lock_path, restricted_root, "registered Stage 4 input lock"),
        "registered Stage 4 input lock",
    )
    require(
        lock.get("plan", {}).get("sha256") == REGISTERED_PLAN_SHA256,
        "input lock does not bind the registered Stage 4 analysis plan",
    )
    require(
        lock.get("threshold_policy_amendment", {})
        .get("artifact", {})
        .get("sha256")
        == REGISTERED_AMENDMENT_SHA256,
        "input lock does not bind the registered threshold amendment",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    code_identity = git_code_identity(args.repo_root, args.expected_git_commit)
    if args.command == "lock":
        require(
            args.expected_plan_sha256 == REGISTERED_PLAN_SHA256,
            "CLI plan digest differs from the externally registered production plan",
        )
        require(
            args.expected_threshold_amendment_sha256
            == REGISTERED_AMENDMENT_SHA256,
            "CLI amendment digest differs from the externally registered production amendment",
        )
        candidate = build_input_lock(
            plan_path=args.plan,
            expected_plan_sha256=args.expected_plan_sha256,
            amendment_path=args.threshold_amendment,
            expected_amendment_sha256=args.expected_threshold_amendment_sha256,
            stage1_paths={
                "scoring_audit": args.stage1_scoring_audit,
                "finalizer_audit": args.stage1_finalizer_audit,
                "journal": args.stage1_journal,
                "summary": args.stage1_summary,
                "table_metadata": args.stage1_table_metadata,
            },
            stage1_reselection_delivery_path=args.stage1_reselection_delivery,
            stage2_delivery_path=args.stage2_delivery,
            stage2_reselection_delivery_path=args.stage2_reselection_delivery,
            stage3_delivery_path=args.stage3_delivery,
            restricted_root=args.restricted_root,
            code_identity=code_identity,
        )
        if args.dry_run:
            print(json.dumps({"status": "PASS", "dry_run": True, "candidate_sha256": canonical_sha256(candidate)}, sort_keys=True))
        else:
            _require_under(args.out, args.restricted_root, "Stage 4 input lock output")
            status = _publish_json_once(args.out, candidate)
            print(json.dumps({"status": status, "path": str(args.out), "sha256": sha256_file(args.out)}, sort_keys=True))
        return
    _require_registered_production_lock(args.input_lock, args.restricted_root)
    if args.command == "preflight":
        receipt = preflight_summaries(args.input_lock, args.restricted_root, code_identity)
        if args.receipt_out:
            _require_under(args.receipt_out, args.restricted_root, "Stage 4 preflight receipt")
            status = _publish_json_once(args.receipt_out, receipt)
            print(json.dumps({"status": status, "path": str(args.receipt_out), "sha256": sha256_file(args.receipt_out)}, sort_keys=True))
        else:
            print(json.dumps(receipt, sort_keys=True))
        return
    if args.command == "publish":
        delivery = publish_outputs(
            input_lock_path=args.input_lock,
            receipt_path=args.preflight_receipt,
            contract_path=args.output_contract,
            staging_dir=args.staging_dir,
            bundle_out=args.bundle_out,
            restricted_root=args.restricted_root,
            code_identity=code_identity,
            dry_run=args.dry_run,
        )
        print(json.dumps(delivery, sort_keys=True))
        return
    verify_input_lock(args.input_lock, args.restricted_root, code_identity)
    delivery = _verify_published_delivery(
        args.bundle_out,
        args.input_lock,
        args.preflight_receipt,
        args.output_contract,
        args.restricted_root,
    )
    print(json.dumps(delivery, sort_keys=True))


if __name__ == "__main__":
    main()
