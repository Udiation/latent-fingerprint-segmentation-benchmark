#!/usr/bin/env python3
"""Freeze Stage 3 native runs and prepare ground-truth-blind inference jobs.

This planner is the barrier between model training and held-out inference.  It
does not open a ground-truth mask, prediction mask, or test image.  It verifies
the validation-only selection, every available full-run card and checkpoint,
the two breadth-first training ledgers/proofs, and the already delivered U-Net
seed-0 inference.  A method needs at least two valid full seeds before any of
its models become eligible for the immutable inference plan.

Missing and terminally failed seeds remain explicit in the gate.  Scientific
identity failures (changed configuration, checkpoint, Git provenance, queue,
or proof) fail closed rather than being relabelled as ordinary run failures.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import (
    canonical_sha256,
    require_clean_process_start,
    sha256_file,
)
from segbench.deep_utils import run_provenance
from segbench.io import attach_split, load_manifest
import authorize_deep_jobs as authorization
import prepare_deep_jobs as planner
from prepare_remaining_native_sweep import _validate_stage2_delivery


SCHEMA_VERSION = 1
NATIVE_METHODS = tuple(planner.METHOD_PRIORITY)
SEEDS = tuple(planner.FULL_SEEDS)
EXPECTED_PREDICTION_COUNT = 2010
MC_DROPOUT_IMAGE_COUNT = 20
MC_DROPOUT_PASSES = 20
STAGE2_REUSED_RUN = ("unet", 0)
SEED0_NAMES = tuple(authorization.EXPECTED_SEED0)
SEED12_NAMES = tuple(authorization.EXPECTED_SEED12)
ALL_INFERENCE_NAMES = tuple(
    f"infer.{method}.seed{seed}"
    for method in NATIVE_METHODS
    for seed in SEEDS
    if (method, seed) != STAGE2_REUSED_RUN
)
HEX = set("0123456789abcdef")


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in HEX for character in value)
    )


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _read_yaml(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a YAML object: {path}")
    return value


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable JSON artifact, accepting an identical replay."""
    if path.exists():
        observed = _read_json(path, "existing immutable JSON artifact")
        if canonical_sha256(observed) != canonical_sha256(value):
            raise FileExistsError(f"refusing to replace changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_yaml_once(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable YAML artifact, accepting an identical replay."""
    if path.exists():
        observed = _read_yaml(path, "existing immutable YAML artifact")
        if canonical_sha256(observed) != canonical_sha256(value):
            raise FileExistsError(f"refusing to replace changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())


def _identity(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _stable_provenance(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove wall-clock-only fields so identical evidence is reproducible."""
    return {
        key: copy.deepcopy(provenance[key])
        for key in (
            "command_argv",
            "capture_timing",
            "git_commit",
            "git_dirty",
            "versions",
        )
        if key in provenance
    }


def _validate_commit(value: str, description: str) -> str:
    if not _is_hex(value, 40):
        raise ValueError(f"{description} must be an immutable 40-hex commit")
    return value


def _gpu_indices(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("allowed GPU indices must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result) or len(result) != len(set(result)):
        raise ValueError("allowed GPU indices must be non-negative and unique")
    return result


def _require_under(path: Path, root: Path, description: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{description} escapes the restricted root") from exc


def _require_disjoint(left: Path, right: Path, description: str) -> None:
    left_resolved, right_resolved = left.resolve(), right.resolve()
    try:
        left_resolved.relative_to(right_resolved)
    except ValueError:
        pass
    else:
        raise ValueError(f"{description} roots overlap")
    try:
        right_resolved.relative_to(left_resolved)
    except ValueError:
        return
    raise ValueError(f"{description} roots overlap")


def _validate_pinned_code_root(code_root: Path, expected_commit: str) -> None:
    """Require the inference implementation to be a clean pinned worktree."""
    entrypoint = code_root / "scripts" / "infer_masks_repro.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(entrypoint)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=code_root,
        text=True,
    ).strip()
    if commit != expected_commit or dirty:
        raise ValueError(
            "pinned inference code root does not match its clean expected commit"
        )


def _commit_map(path: Path) -> Dict[str, str]:
    value = _read_json(path, "historical expected-commit map")
    if set(value) != set(NATIVE_METHODS):
        raise ValueError("expected-commit map must contain exactly native methods")
    return {
        method: _validate_commit(str(value[method]), f"{method} historical commit")
        for method in NATIVE_METHODS
    }


def _full_run_commit_map(path: Path) -> Dict[str, str]:
    """Load the exact training commit expected for each method/seed run."""
    value = _read_json(path, "full-run expected-commit map")
    expected = {
        f"{method}.seed{seed}" for method in NATIVE_METHODS for seed in SEEDS
    }
    if set(value) != expected:
        raise ValueError("full-run commit map must contain exactly 24 method/seed keys")
    return {
        key: _validate_commit(str(value[key]), f"{key} training commit")
        for key in sorted(expected)
    }


def _load_reservations(path: Path, max_ram_gib: float) -> Dict[str, float]:
    """Require prospective RAM declarations for all possible new inferences."""
    value = _read_json(path, "inference RAM reservations")
    if set(value) != set(ALL_INFERENCE_NAMES):
        raise ValueError(
            "inference RAM map must cover exactly all 23 non-reused native runs"
        )
    result: Dict[str, float] = {}
    for name in ALL_INFERENCE_NAMES:
        amount = float(value[name])
        if not math.isfinite(amount) or not 0.0 < amount <= max_ram_gib:
            raise ValueError(f"{name}: RAM reservation exceeds the runtime cap")
        result[name] = amount
    return result


def _test_universe(
    manifest: Path, splits: Path, expected_count: int
) -> Dict[str, Any]:
    """Count test inputs from metadata only; never dereference a GT path."""
    split_payload = _read_json(splits, "subject split")
    mapping = split_payload.get("splits")
    if not isinstance(mapping, dict):
        raise ValueError("subject split lacks its splits mapping")
    frame = attach_split(load_manifest(manifest), mapping)
    if "examiner_value" not in frame.columns:
        raise ValueError("manifest lacks examiner_value needed for NONPRINT policy")
    rows = frame[
        (frame["subset"].astype(str) == "finger_latent")
        & (frame["split"].astype(str) == "test")
        & (frame["examiner_value"].astype(str).str.upper() != "NONPRINT")
    ]
    identifiers = [str(value) for value in rows["image_id"].tolist()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("held-out inference universe contains duplicate image ids")
    if len(identifiers) != expected_count:
        raise ValueError(
            f"held-out inference count {len(identifiers)} != expected {expected_count}"
        )
    required = {"image_path", "source_sha256"}
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"held-out manifest lacks source identities: {missing}")
    source_records = []
    for row in rows.sort_values("image_id").to_dict("records"):
        expected = row.get("source_sha256")
        if not _is_hex(expected, 64):
            raise ValueError(f"{row['image_id']}: invalid source_sha256")
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file() or sha256_file(image_path) != expected:
            raise ValueError(f"{row['image_id']}: source image bytes changed")
        source_records.append(
            {"image_id": str(row["image_id"]), "source_sha256": str(expected)}
        )
    return {
        "subset": "finger_latent",
        "split": "test",
        "excluded_examiner_values": ["NONPRINT"],
        "prediction_count": len(identifiers),
        "image_id_set_sha256": canonical_sha256(sorted(identifiers)),
        "ordered_source_images_sha256": canonical_sha256(source_records),
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
    }


def _auxiliary_sample(manifest: Path, splits: Path) -> Dict[str, Any]:
    """Freeze a deterministic 20-image qualitative sample without opening GT."""
    split_payload = _read_json(splits, "subject split")
    mapping = split_payload.get("splits")
    if not isinstance(mapping, dict):
        raise ValueError("subject split lacks its splits mapping")
    frame = attach_split(load_manifest(manifest), mapping)
    rows = frame[
        (frame["subset"].astype(str) == "finger_latent")
        & (frame["split"].astype(str) == "test")
        & (frame["examiner_value"].astype(str).str.upper() != "NONPRINT")
    ].sort_values("image_id").head(MC_DROPOUT_IMAGE_COUNT)
    required = {"source_sha256", "target_height", "target_width"}
    if len(rows) != MC_DROPOUT_IMAGE_COUNT or not required <= set(rows.columns):
        raise ValueError("cannot freeze the 20-image DRUnet auxiliary sample")
    image_ids = [str(value) for value in rows["image_id"].tolist()]
    source_records = []
    expected_shapes: Dict[str, list[int]] = {}
    for row in rows.to_dict("records"):
        expected = row.get("source_sha256")
        if not _is_hex(expected, 64):
            raise ValueError(f"{row['image_id']}: invalid source_sha256")
        source = Path(str(row["image_path"]))
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"{row['image_id']}: source image bytes changed")
        source_records.append(
            {"image_id": str(row["image_id"]), "source_sha256": str(expected)}
        )
        height, width = row.get("target_height"), row.get("target_width")
        if type(height) not in (int, float) or type(width) not in (int, float):
            raise ValueError(f"{row['image_id']}: invalid target shape")
        if not float(height).is_integer() or not float(width).is_integer():
            raise ValueError(f"{row['image_id']}: non-integral target shape")
        shape = [int(height), int(width)]
        if any(value <= 0 for value in shape):
            raise ValueError(f"{row['image_id']}: invalid target shape")
        expected_shapes[str(row["image_id"])] = shape
    return {
        "selection_rule": "first_20_lexicographic_heldout_ids",
        "test_image_ids": image_ids,
        "image_id_set_sha256": canonical_sha256(image_ids),
        "ordered_source_images_sha256": canonical_sha256(source_records),
        "expected_shapes": expected_shapes,
    }


def _manifest_audit_identity(
    audit_path: Path, manifest: Path
) -> tuple[Dict[str, Any], str]:
    """Bind a precomputed GT audit without opening any GT file."""
    audit = _read_json(audit_path, "latent manifest audit")
    if audit.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("manifest audit is not bound to the selected manifest")
    ordered_gt = audit.get("ordered_source_gt_sha256")
    if not _is_hex(ordered_gt, 64):
        raise ValueError("manifest audit lacks an ordered source-GT digest")
    return _identity(audit_path), str(ordered_gt)


def _scorer_code_identities(code_root: Path) -> Dict[str, Dict[str, Any]]:
    """Freeze every implementation later allowed to open held-out GT."""
    paths = {
        "planner": code_root / "scripts" / "prepare_stage3_inference.py",
        "finalizer": code_root / "scripts" / "finalize_stage3_native.py",
        "evaluator": code_root / "scripts" / "evaluate_masks.py",
        "metrics": code_root / "segbench" / "metrics.py",
        "io": code_root / "segbench" / "io.py",
        "summary": code_root / "segbench" / "summary.py",
        "table_builder": code_root / "scripts" / "make_tables.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing pinned scorer code: " + ", ".join(missing))
    return {name: _identity(path) for name, path in paths.items()}


def _inference_code_identities(code_root: Path) -> Dict[str, Dict[str, Any]]:
    """Freeze the complete project-owned deterministic inference surface."""
    paths = {
        "wrapper": code_root / "scripts" / "infer_masks_repro.py",
        "auxiliary_wrapper": code_root / "scripts" / "infer_drunet_mc_dropout.py",
        "runner": code_root / "scripts" / "infer_masks.py",
        "deep_utils": code_root / "segbench" / "deep_utils.py",
        "deep_models": code_root / "segbench" / "deep_models.py",
        "artifact_identity": code_root / "segbench" / "artifact_identity.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing pinned inference code: " + ", ".join(missing))
    return {name: _identity(path) for name, path in paths.items()}


def _validate_queue_identity(item: Mapping[str, Any]) -> str:
    expected = canonical_sha256(
        {
            "name": item.get("name"),
            "argv": item.get("argv"),
            "env": item.get("env", {}),
            "completion_artifact": item.get("completion_artifact"),
            "resources": item.get("resources"),
        }
    )
    if item.get("job_identity_sha256") != expected:
        raise ValueError(f"{item.get('name')}: training queue identity changed")
    return expected


def _training_wave(
    queue_path: Path,
    ledger_path: Path,
    expected_names: Sequence[str],
    description: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Mapping[str, Any]]]:
    queue = _read_yaml(queue_path, f"{description} queue")
    if queue.get("launch_authorized") is not True:
        raise ValueError(f"{description} queue was not authorized")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or [item.get("name") for item in jobs] != list(
        expected_names
    ):
        raise ValueError(f"{description} queue universe/order changed")
    by_name: Dict[str, Mapping[str, Any]] = {}
    for item in jobs:
        if not isinstance(item, dict):
            raise ValueError(f"{description} queue contains a non-mapping job")
        name = str(item.get("name"))
        if name in by_name:
            raise ValueError(f"duplicate training job: {name}")
        _validate_queue_identity(item)
        by_name[name] = item
    ledger = _read_json(ledger_path, f"{description} ledger")
    if ledger.get("queue_sha256") != sha256_file(queue_path):
        raise ValueError(f"{description} ledger is not bound to its queue")
    entries = ledger.get("jobs")
    if not isinstance(entries, dict) or set(entries) != set(expected_names):
        raise ValueError(f"{description} ledger universe changed")
    for name in expected_names:
        entry = entries[name]
        if not isinstance(entry, dict):
            raise ValueError(f"{name}: ledger entry is not a mapping")
        if entry.get("identity_sha256") != by_name[name]["job_identity_sha256"]:
            raise ValueError(f"{name}: ledger identity differs from queue")
        if entry.get("status") not in {
            "pending", "running", "retrying", "preempted", "completed", "failed"
        }:
            raise ValueError(f"{name}: invalid ledger status")
        if entry.get("status") == "failed":
            counters: Dict[str, int] = {}
            for key in (
                "attempts",
                "ordinary_retries",
                "oom_retries",
                "preemptions",
            ):
                value = entry.get(key)
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name}: failed ledger has invalid {key}")
                counters[key] = value
            if counters["ordinary_retries"] > 1 or counters["oom_retries"] > 1:
                raise ValueError(f"{name}: failed ledger exceeds registered retry limits")
            if counters["ordinary_retries"] + counters["oom_retries"] < 1:
                raise ValueError(f"{name}: terminal failure did not consume a retry")
            expected_attempts = (
                1
                + counters["ordinary_retries"]
                + counters["oom_retries"]
                + counters["preemptions"]
            )
            if counters["attempts"] < 2 or counters["attempts"] != expected_attempts:
                raise ValueError(
                    f"{name}: failed ledger attempts do not prove runner retry semantics"
                )
            exit_code = entry.get("exit_code")
            if type(exit_code) is not int or exit_code in {0, 75}:
                raise ValueError(f"{name}: terminal failure has invalid exit code")
    return queue, ledger, by_name


def _completed_native_sweep(
    queue_path: Path, ledger_path: Path
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Mapping[str, Any]]]:
    """Authenticate all 14 new native LR candidates through their queue ledger."""
    expected_names = tuple(
        f"sweep.{method}.lr{planner.lr_slug(rate)}.seed0"
        for method in NATIVE_METHODS
        if method != "unet"
        for rate in planner.LEARNING_RATES
    )
    queue, ledger, jobs = _training_wave(
        queue_path, ledger_path, expected_names, "native LR sweep"
    )
    for name in expected_names:
        entry = ledger["jobs"][name]
        if entry.get("status") != "completed":
            raise ValueError(f"{name}: LR candidate is not terminal-completed")
        card = Path(str(jobs[name].get("completion_artifact", "")))
        if not card.is_file() or sha256_file(card) != entry.get("artifact_sha256"):
            raise ValueError(f"{name}: LR candidate card differs from queue ledger")
        jobs[name] = {
            **dict(jobs[name]),
            "ledger_artifact_sha256": str(entry["artifact_sha256"]),
        }
    return queue, ledger, jobs


def _argv_value(job: Mapping[str, Any], option: str) -> str:
    argv = job.get("argv")
    if not isinstance(argv, list) or argv.count(option) != 1:
        raise ValueError(f"{job.get('name')}: expected exactly one {option}")
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise ValueError(f"{job.get('name')}: {option} has no value")
    return str(argv[index + 1])


def _validate_training_proofs(
    *,
    global_selection: Path,
    seed0_queue: Path,
    seed12_plan: Path,
    seed12_queue: Path,
    prepare_proof: Path,
    seed12_proof: Path,
) -> Dict[str, Any]:
    preparation = _read_json(prepare_proof, "full-wave preparation proof")
    authorization_proof = _read_json(seed12_proof, "seed-1/2 authorization proof")
    locked = _read_yaml(seed12_plan, "locked seed-1/2 plan")
    if locked.get("launch_authorized") is not False or "jobs" in locked:
        raise ValueError("seed-1/2 locked plan became launchable")
    if [item.get("name") for item in locked.get("jobs_locked", [])] != list(
        SEED12_NAMES
    ):
        raise ValueError("seed-1/2 locked universe changed")
    expected = {
        "global_selection_sha256": sha256_file(global_selection),
        "seed0_queue_sha256": sha256_file(seed0_queue),
        "seed12_plan_sha256": sha256_file(seed12_plan),
        "seed12_queue_sha256": sha256_file(seed12_queue),
    }
    if preparation.get("global_selection_sha256") != expected[
        "global_selection_sha256"
    ]:
        raise ValueError("preparation proof lost the global selection binding")
    if preparation.get("seed0_queue", {}).get("sha256") != expected[
        "seed0_queue_sha256"
    ]:
        raise ValueError("preparation proof lost the seed-0 queue binding")
    if preparation.get("seed12_locked_plan", {}).get("sha256") != expected[
        "seed12_plan_sha256"
    ]:
        raise ValueError("preparation proof lost the locked seed-1/2 binding")
    proof_expected = {
        "seed0_queue_sha256": expected["seed0_queue_sha256"],
        "seed12_locked_plan_sha256": expected["seed12_plan_sha256"],
        "seed12_authorized_queue_sha256": expected["seed12_queue_sha256"],
    }
    for key, value in proof_expected.items():
        if authorization_proof.get(key) != value:
            raise ValueError(f"seed-1/2 authorization proof changed: {key}")
    return {
        "prepare_proof": _identity(prepare_proof),
        "seed12_authorization_proof": _identity(seed12_proof),
        "seed12_locked_plan": _identity(seed12_plan),
    }


def _validate_global_selection(
    *,
    global_selection_path: Path,
    pilot_selection_path: Path,
    sufs_selection_path: Path,
    expected_commits: Mapping[str, str],
    manifest: Path,
    splits: Path,
    native_sweep_jobs: Mapping[str, Mapping[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    selection, _, _ = planner.validate_staged_selections(
        global_selection_path, pilot_selection_path, sufs_selection_path
    )
    if selection.get("selection_split") != "val" or selection.get(
        "selection_metric"
    ) != "best_validation_dice":
        raise ValueError("global selection is not validation-only")
    if selection.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("global selection manifest identity changed")
    if selection.get("splits_sha256") != sha256_file(splits):
        raise ValueError("global selection split identity changed")
    methods = selection.get("methods")
    if not isinstance(methods, dict) or not set(NATIVE_METHODS).issubset(methods):
        raise ValueError("global selection lacks native methods")
    independently_reconstructed: Dict[str, Any] = {}
    for method in NATIVE_METHODS:
        candidates = methods[method].get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise ValueError(f"{method}: selection candidate universe changed")
        rates = [float(candidate.get("learning_rate", math.nan)) for candidate in candidates]
        if len(set(rates)) != 2 or set(rates) != set(planner.LEARNING_RATES):
            raise ValueError(f"{method}: selection does not contain the exact LR grid")
        reconstructed = []
        for candidate in candidates:
            rate = float(candidate["learning_rate"])
            run_path = Path(str(candidate.get("run_card")))
            run = _read_json(run_path, f"{method} selection run card")
            require_clean_process_start(
                run,
                expected_commits[method],
                description=f"{method} validation-selection candidate",
            )
            expected_suffix = (
                "sweep", method, f"lr_{planner.lr_slug(rate)}", "seed0", "run.json"
            )
            if tuple(run_path.parts[-5:]) != expected_suffix:
                raise ValueError(f"{method}: sweep run-card path is not preregistered")
            expected_run = {
                "method": method,
                "seed": 0,
                "epochs_requested": planner.SWEEP_EPOCHS,
                "epochs_completed": planner.SWEEP_EPOCHS,
                "manifest_sha256": sha256_file(manifest),
                "split_sha256": sha256_file(splits),
                "debug_limited": False,
            }
            for key, expected in expected_run.items():
                if run.get(key) != expected:
                    raise ValueError(f"{method}: sweep run {key} changed")
            if not math.isclose(
                float(run.get("learning_rate", math.nan)), rate, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"{method}: sweep learning rate changed")
            run_score = float(run.get("best_validation_dice", math.nan))
            if not math.isfinite(run_score) or not 0.0 <= run_score <= 1.0:
                raise ValueError(f"{method}: sweep validation Dice is invalid")
            run_threshold = float(run.get("selected_validation_threshold", math.nan))
            if run_threshold not in planner.THRESHOLDS:
                raise ValueError(f"{method}: sweep threshold is not preregistered")
            score = run_score
            threshold = run_threshold
            selector_identity = None
            legacy_selector_required = False
            if method == "unetpp":
                sweep_job = native_sweep_jobs.get(
                    f"sweep.{method}.lr{planner.lr_slug(rate)}.seed0"
                )
                if isinstance(sweep_job, Mapping):
                    sweep_config_path = Path(_argv_value(sweep_job, "--config"))
                    sweep_config_payload = _read_yaml(
                        sweep_config_path, "unetpp sweep config"
                    )
                    sweep_training = sweep_config_payload.get("training", {})
                    legacy_selector_required = tuple(
                        float(value)
                        for value in sweep_training.get("threshold_candidates", ())
                    ) == tuple(planner.LEGACY_THRESHOLDS)
            if legacy_selector_required or candidate.get(
                "expanded_grid_selector_receipt"
            ) is not None:
                if candidate.get("legacy_run_metric_selection_ineligible") is not True:
                    raise ValueError("unetpp: legacy sweep metric was not made ineligible")
                selector_identity = candidate.get("expanded_grid_selector_receipt")
                if not isinstance(selector_identity, Mapping):
                    raise ValueError("unetpp: expanded-grid selector identity is absent")
                selector_path = Path(str(selector_identity.get("path", "")))
                if (
                    not selector_path.is_file()
                    or sha256_file(selector_path) != selector_identity.get("sha256")
                ):
                    raise ValueError("unetpp: expanded-grid selector receipt changed")
                selector = _read_json(selector_path, "unetpp expanded-grid selector")
                if (
                    selector.get("kind")
                    != "expanded_validation_threshold_selector_receipt"
                    or selector.get("selection_source") != "validation_only"
                    or selector.get("validation_only") is not True
                    or selector.get("test_or_heldout_opened") is not False
                    or tuple(float(value) for value in selector.get("allowed_grid", ()))
                    != tuple(planner.THRESHOLDS)
                    or selector.get("checkpoint_sha256")
                    != run["checkpoint_sha256"]["best.pt"]
                    or selector.get("source_config_sha256")
                    != run["source_config_sha256"]
                    or selector.get("manifest_sha256") != sha256_file(manifest)
                    or selector.get("split_sha256") != sha256_file(splits)
                    or selector.get("amendment_sha256")
                    != selector_identity.get("amendment_sha256")
                    or selector.get("threshold_amendment_sha256")
                    != selector_identity.get("amendment_sha256")
                    or selector.get("raw_validation_probability_card_sha256")
                    != selector_identity.get(
                        "raw_validation_probability_card_sha256"
                    )
                    or selector.get("validation_probability_card_sha256")
                    != selector_identity.get(
                        "selected_validation_probability_card_sha256"
                    )
                ):
                    raise ValueError("unetpp: expanded-grid selector provenance differs")
                score = float(selector.get("selected_validation_dice", math.nan))
                threshold = float(selector.get("selected_threshold", math.nan))
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("unetpp: expanded-grid validation Dice is invalid")
                if threshold not in planner.THRESHOLDS:
                    raise ValueError("unetpp: expanded-grid threshold is invalid")
            candidate_expected = {
                "best_validation_dice": score,
                "best_epoch": int(run["best_epoch"]),
                "selected_validation_threshold": threshold,
                "config_sha256": str(run["config_sha256"]),
                "best_checkpoint_sha256": str(run["checkpoint_sha256"]["best.pt"]),
            }
            for key, expected in candidate_expected.items():
                observed = candidate.get(key)
                if key in {"best_validation_dice", "selected_validation_threshold"}:
                    if not math.isclose(float(observed), float(expected), rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f"{method}: candidate {key} differs from run card")
                elif observed != expected:
                    raise ValueError(f"{method}: candidate {key} differs from run card")
            hashes = run.get("checkpoint_sha256")
            if not isinstance(hashes, Mapping) or set(hashes) != {"best.pt", "last.pt"}:
                raise ValueError(f"{method}: sweep checkpoint identities are incomplete")
            for name, digest in hashes.items():
                checkpoint = run_path.parent / name
                if not checkpoint.is_file() or sha256_file(checkpoint) != digest:
                    raise ValueError(f"{method}: sweep checkpoint changed: {name}")
            full_config_path = Path(str(methods[method].get("full_config", "")))
            source_config = (
                full_config_path.parent.parent
                / "sweep"
                / f"{method}_lr{planner.lr_slug(rate)}.yaml"
            )
            if (
                not source_config.is_file()
                or sha256_file(source_config) != run.get("source_config_sha256")
            ):
                raise ValueError(f"{method}: sweep source config changed")
            if method != "unet":
                job_name = f"sweep.{method}.lr{planner.lr_slug(rate)}.seed0"
                job = native_sweep_jobs.get(job_name)
                if not isinstance(job, Mapping):
                    raise ValueError(f"{method}: guarded sweep job is absent")
                if Path(str(job.get("completion_artifact", ""))).resolve() != run_path.resolve():
                    raise ValueError(f"{method}: sweep completion path differs from queue")
                if job.get("ledger_artifact_sha256") != sha256_file(run_path):
                    raise ValueError(f"{method}: sweep run card differs from ledger")
                if Path(_argv_value(job, "--config")).resolve() != source_config.resolve():
                    raise ValueError(f"{method}: sweep config differs from queue")
                if Path(_argv_value(job, "--out")).resolve() != run_path.parent.resolve():
                    raise ValueError(f"{method}: sweep output differs from queue")
                if _argv_value(job, "--seed") != "0":
                    raise ValueError(f"{method}: sweep seed differs from queue")
            reconstructed.append(
                {
                    "learning_rate": rate,
                    "best_validation_dice": score,
                    "selected_validation_threshold": threshold,
                    "run_card": _identity(run_path),
                    "source_config": _identity(source_config),
                    "checkpoints": {
                        name: {
                            "path": str(run_path.parent / name),
                            "sha256": str(hashes[name]),
                        }
                        for name in ("best.pt", "last.pt")
                    },
                    "training_git_commit": expected_commits[method],
                    **(
                        {
                            "legacy_run_metric_selection_ineligible": True,
                            "legacy_run_validation_dice": run_score,
                            "expanded_grid_selector_receipt": dict(
                                selector_identity
                            ),
                        }
                        if selector_identity is not None
                        else {}
                    ),
                }
            )
        chosen = max(
            reconstructed,
            key=lambda item: (
                item["best_validation_dice"],
                -planner.LEARNING_RATES.index(item["learning_rate"]),
            ),
        )
        record = methods[method]
        if not math.isclose(
            float(record.get("selected_learning_rate", math.nan)),
            float(chosen["learning_rate"]),
            rel_tol=0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(record.get("selected_validation_dice", math.nan)),
            float(chosen["best_validation_dice"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{method}: selected LR/metric is not deterministic argmax")
        full_config_path = Path(str(record.get("full_config", "")))
        if (
            not full_config_path.is_file()
            or sha256_file(full_config_path) != record.get("full_config_sha256")
        ):
            raise ValueError(f"{method}: selected full config bytes changed")
        full_config = _read_yaml(full_config_path, f"{method} selected full config")
        training = full_config.get("training")
        experiment = full_config.get("experiment")
        if not isinstance(training, Mapping) or not isinstance(experiment, Mapping):
            raise ValueError(f"{method}: selected full config structure changed")
        full_expected = {
            "learning_rate": chosen["learning_rate"],
            "epochs": planner.FULL_EPOCHS,
            "patience": planner.FULL_PATIENCE,
        }
        for key, expected in full_expected.items():
            observed = training.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"{method}: full config {key} changed")
            elif observed != expected:
                raise ValueError(f"{method}: full config {key} changed")
        if experiment != {
            "phase": "full_training",
            "learning_rate_selected_on": "val_best_dice",
            "selected_learning_rate": chosen["learning_rate"],
            "sweep_epochs": planner.SWEEP_EPOCHS,
            "full_seeds": list(planner.FULL_SEEDS),
        }:
            raise ValueError(f"{method}: full config experiment contract changed")
        independently_reconstructed[method] = {
            "learning_rate_candidates": sorted(rates),
            "tie_break": "lower_predeclared_learning_rate",
            "candidates": sorted(reconstructed, key=lambda item: item["learning_rate"]),
            "selected_learning_rate": chosen["learning_rate"],
            "selected_validation_dice": chosen["best_validation_dice"],
            "full_config": _identity(full_config_path),
        }
    return selection, independently_reconstructed


def _validate_full_run(
    *,
    method: str,
    seed: int,
    expected_commit: str,
    selection: Mapping[str, Any],
    runs_root: Path,
) -> Dict[str, Any]:
    method_selection = selection["methods"][method]
    reused_unet = method == "unet" and seed == 0
    config_key = "reused_seed0_config" if reused_unet else "full_config"
    config_sha_key = (
        "reused_seed0_config_sha256" if reused_unet else "full_config_sha256"
    )
    run_dir = runs_root / "full" / method / f"seed{seed}"
    proof = planner.validate_completed_full_run(
        run_dir,
        method=method,
        seed=seed,
        manifest_sha256=str(selection["manifest_sha256"]),
        splits_sha256=str(selection["splits_sha256"]),
        source_config_sha256=str(method_selection[config_sha_key]),
        selected_learning_rate=float(method_selection["selected_learning_rate"]),
    )
    run_path = run_dir / "run.json"
    run = _read_json(run_path, f"full.{method}.seed{seed} run card")
    require_clean_process_start(
        run, expected_commit, description=f"full.{method}.seed{seed}"
    )
    score = float(run.get("best_validation_dice", math.nan))
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"full.{method}.seed{seed}: invalid validation Dice")
    threshold = float(run.get("selected_validation_threshold", math.nan))
    if threshold not in planner.THRESHOLDS:
        raise ValueError(f"full.{method}.seed{seed}: invalid validation threshold")
    parameters = run.get("parameter_count")
    if not isinstance(parameters, dict) or int(parameters.get("total", 0)) <= 0:
        raise ValueError(f"full.{method}.seed{seed}: invalid parameter count")
    config_path = Path(str(method_selection[config_key]))
    if (
        not config_path.is_file()
        or sha256_file(config_path) != str(method_selection[config_sha_key])
    ):
        raise ValueError(f"full.{method}.seed{seed}: selected config bytes changed")
    checkpoint = run_dir / "best.pt"
    return {
        "status": "valid",
        "method": method,
        "seed": seed,
        "run_card": str(run_path),
        "run_card_sha256": proof["run_card_sha256"],
        "config": str(config_path),
        "config_sha256": str(method_selection[config_sha_key]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": proof["checkpoint_sha256"]["best.pt"],
        "last_checkpoint_sha256": proof["checkpoint_sha256"]["last.pt"],
        "training_git_commit": expected_commit,
        "epochs_completed": int(proof["epochs_completed"]),
        "best_validation_dice": score,
        "selected_validation_threshold": threshold,
        "parameter_count": int(parameters["total"]),
    }


def _production_inference_inputs(
    selection: Mapping[str, Any], cache_dir: Path, amendment: Path
) -> Dict[str, str]:
    """Re-authenticate the cache/amendment frozen by full-run selection."""
    production = selection.get("production_inputs")
    if not isinstance(production, Mapping):
        raise ValueError("global selection lacks production input identities")
    receipt = cache_dir.resolve() / "receipt.json"
    expected = {
        "cache_dir": str(cache_dir.resolve()),
        "cache_receipt": str(receipt),
        "cache_receipt_sha256": sha256_file(receipt),
        "threshold_amendment": str(amendment.resolve()),
        "threshold_amendment_sha256": sha256_file(amendment),
    }
    for key, value in expected.items():
        if production.get(key) != value:
            raise ValueError(f"production inference input changed: {key}")
    return expected


def _threshold_evidence_index(
    path: Path,
    *,
    selection: Mapping[str, Any],
    cache_receipt_sha256: str,
    amendment_sha256: str,
) -> Dict[tuple[str, int], Dict[str, Any]]:
    """Load the immutable validation-probability/selector handoff manifest."""
    payload = _read_json(path, "Stage 3 threshold-selector manifest")
    expected = {
        "schema_version": 1,
        "kind": "stage3_native_threshold_selector_manifest",
        "manifest_sha256": selection["manifest_sha256"],
        "splits_sha256": selection["splits_sha256"],
        "cache_receipt_sha256": cache_receipt_sha256,
        "threshold_amendment_sha256": amendment_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("threshold-selector manifest identity changed")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("threshold-selector manifest lacks records")
    result: Dict[tuple[str, int], Dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("threshold-selector manifest has a non-object record")
        key = (str(item.get("method")), int(item.get("seed", -1)))
        if key in result or key == STAGE2_REUSED_RUN:
            raise ValueError("threshold-selector manifest has a duplicate/reused record")
        result[key] = item
    return result


def _attach_threshold_evidence(
    record: Dict[str, Any],
    item: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    amendment_sha256: str,
) -> None:
    """Bind one full run to its raw validation probabilities and selector."""
    method, seed = str(record["method"]), int(record["seed"])
    identities: Dict[str, Dict[str, str]] = {}
    for key in (
        "threshold_receipt",
        "raw_validation_probability_card",
        "validation_probability_card",
    ):
        artifact = Path(str(item.get(key, "")))
        digest = str(item.get(f"{key}_sha256", ""))
        if not artifact.is_file() or not _is_hex(digest, 64) or sha256_file(artifact) != digest:
            raise ValueError(f"{method}/seed{seed}: {key} identity changed")
        identities[key] = {"path": str(artifact), "sha256": digest}
    selector = _read_json(
        Path(identities["threshold_receipt"]["path"]),
        f"{method}/seed{seed} threshold selector",
    )
    config = _read_yaml(Path(str(record["config"])), f"{method}/seed{seed} config")
    expected = {
        "schema_version": 1,
        "kind": "expanded_validation_threshold_selector_receipt",
        "method": method,
        "task": config.get("task"),
        "seed": seed,
        "selection_split": "val",
        "selection_source": "validation_only",
        "validation_only": True,
        "test_or_heldout_opened": False,
        "source_config_sha256": record["config_sha256"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "manifest_sha256": selection["manifest_sha256"],
        "split_sha256": selection["splits_sha256"],
        "threshold_amendment_sha256": amendment_sha256,
        "amendment_sha256": amendment_sha256,
        "raw_validation_probability_card_sha256": identities[
            "raw_validation_probability_card"
        ]["sha256"],
        "validation_probability_card_sha256": identities[
            "validation_probability_card"
        ]["sha256"],
        "probabilities_are_pre_threshold": True,
        "derived_from_binary_mask": False,
        "debug_limited": False,
    }
    disagreements = {
        key: (selector.get(key), value)
        for key, value in expected.items()
        if selector.get(key) != value
    }
    if disagreements:
        raise ValueError(
            f"{method}/seed{seed}: threshold selector provenance mismatch: "
            f"{disagreements}"
        )
    if tuple(float(value) for value in selector.get("allowed_grid", ())) != planner.THRESHOLDS:
        raise ValueError(f"{method}/seed{seed}: selector does not use expanded grid")
    if float(selector.get("selected_threshold", math.nan)) != float(
        record["selected_validation_threshold"]
    ):
        raise ValueError(f"{method}/seed{seed}: selector/run threshold changed")
    record.update(
        {
            "threshold_receipt": identities["threshold_receipt"]["path"],
            "threshold_receipt_sha256": identities["threshold_receipt"]["sha256"],
            "raw_validation_probability_card": identities[
                "raw_validation_probability_card"
            ],
            "validation_probability_card": identities[
                "validation_probability_card"
            ],
        }
    )


def _status_record(
    *,
    name: str,
    ledger_entry: Mapping[str, Any],
    queue_job: Mapping[str, Any],
    method: str,
    seed: int,
    selection: Mapping[str, Any],
    runs_root: Path,
    expected_commit: str,
) -> Dict[str, Any]:
    status = str(ledger_entry["status"])
    run_path = runs_root / "full" / method / f"seed{seed}" / "run.json"
    if status == "completed":
        if not run_path.is_file():
            raise FileNotFoundError(f"{name}: completed run card disappeared")
        if str(queue_job.get("completion_artifact")) != str(run_path):
            raise ValueError(f"{name}: completion path changed")
        if ledger_entry.get("artifact_sha256") != sha256_file(run_path):
            raise ValueError(f"{name}: completed run card differs from ledger")
        return _validate_full_run(
            method=method,
            seed=seed,
            expected_commit=expected_commit,
            selection=selection,
            runs_root=runs_root,
        )
    if run_path.exists():
        raise ValueError(f"{name}: non-completed ledger has a final run card")
    record: Dict[str, Any] = {
        "status": "failed" if status == "failed" else "missing",
        "ledger_status": status,
        "method": method,
        "seed": seed,
        "attempts": int(ledger_entry.get("attempts", 0)),
        "ordinary_retries": int(ledger_entry.get("ordinary_retries", 0)),
        "oom_retries": int(ledger_entry.get("oom_retries", 0)),
        "preemptions": int(ledger_entry.get("preemptions", 0)),
    }
    if status == "failed":
        record["exit_code"] = ledger_entry.get("exit_code")
    return record


def _insufficient_methods(methods: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Return every method that lacks the pre-registered two-seed minimum."""
    return {
        method: list(methods[method].get("valid_seeds", []))
        for method in NATIVE_METHODS
        if len(methods[method].get("valid_seeds", [])) < 2
    }


def _validate_stage2_reuse(
    *,
    delivery_path: Path,
    inference_card_path: Path,
    unet_run: Mapping[str, Any],
    expected_inference_commit: str,
    expected_prediction_count: int,
    expected_image_id_set_sha256: str,
    masks_root: Path,
    stage2_gate_path: Path,
    pilot_selection_path: Path,
) -> Dict[str, Any]:
    delivery = _validate_stage2_delivery(delivery_path)
    if delivery.get("gate_sha256") != sha256_file(stage2_gate_path):
        raise ValueError("Stage 2 delivery does not bind the supplied pilot gate")
    stage2_gate = _read_json(stage2_gate_path, "Stage 2 U-Net pilot gate")
    if stage2_gate.get("status") != "PASS":
        raise ValueError("Stage 2 U-Net pilot gate is not passing")
    pilot = _read_json(pilot_selection_path, "U-Net pilot selection")
    gate_evidence = stage2_gate.get("evidence")
    if not isinstance(gate_evidence, Mapping):
        raise ValueError("Stage 2 U-Net gate lacks candidate evidence")
    gate_candidates = gate_evidence.get("selection_candidates")
    pilot_candidates = pilot.get("candidates")
    if not isinstance(gate_candidates, list) or not isinstance(pilot_candidates, list):
        raise ValueError("Stage 2 U-Net selection candidates are absent")
    canonical_gate = []
    canonical_pilot = []
    for candidate in gate_candidates:
        canonical_gate.append(
            {
                "learning_rate": float(candidate["learning_rate"]),
                "best_validation_dice": float(candidate["validation_dice"]),
                "selected_validation_threshold": float(candidate["selected_threshold"]),
                "run_card": str(candidate["run_card"]),
                "run_card_sha256": str(candidate["run_card_sha256"]),
                "source_config": str(candidate["source_config"]),
                "source_config_sha256": str(candidate["source_config_sha256"]),
                "checkpoint_sha256": dict(candidate["checkpoint_sha256"]),
            }
        )
    for candidate in pilot_candidates:
        canonical_pilot.append(
            {
                "learning_rate": float(candidate["learning_rate"]),
                "best_validation_dice": float(candidate["best_validation_dice"]),
                "selected_validation_threshold": float(
                    candidate["selected_validation_threshold"]
                ),
                "run_card": str(candidate["run_card"]),
                "run_card_sha256": str(candidate["run_card_sha256"]),
                "source_config": str(candidate["source_config"]),
                "source_config_sha256": str(candidate["source_config_sha256"]),
                "checkpoint_sha256": dict(candidate["checkpoint_sha256"]),
            }
        )
    canonical_gate.sort(key=lambda item: item["learning_rate"])
    canonical_pilot.sort(key=lambda item: item["learning_rate"])
    if canonical_gate != canonical_pilot:
        raise ValueError("Stage 2 gate and U-Net pilot candidate identities differ")
    if (
        float(gate_evidence.get("selected_learning_rate", math.nan))
        != float(pilot.get("selected_learning_rate", math.nan))
        or float(gate_evidence.get("selected_sweep_validation_dice", math.nan))
        != float(pilot.get("selected_validation_dice", math.nan))
    ):
        raise ValueError("Stage 2 gate and U-Net pilot choice differ")
    if delivery.get("checkpoint_sha256") != unet_run["checkpoint_sha256"]:
        raise ValueError("Stage 2 delivery and U-Net seed-0 checkpoint differ")
    if delivery.get("inference_card_sha256") != sha256_file(inference_card_path):
        raise ValueError("Stage 2 delivery and immutable inference card differ")
    card = _read_json(inference_card_path, "Stage 2 U-Net inference card")
    if (
        card.get("kind") != "complete_unbiased_inference_card"
        or card.get("method") != "unet"
        or card.get("subset") != "finger_latent"
        or card.get("split") != "test"
        or card.get("debug_limited") is not False
    ):
        raise ValueError("Stage 2 inference card has the wrong scientific identity")
    identities = card.get("identities")
    if not isinstance(identities, dict):
        raise ValueError("Stage 2 inference card lacks identities")
    if identities.get("checkpoint_sha256") != unet_run["checkpoint_sha256"]:
        raise ValueError("Stage 2 inference used a different U-Net checkpoint")
    if identities.get("prediction_count") != expected_prediction_count:
        raise ValueError("Stage 2 inference prediction count changed")
    prediction_set = card.get("prediction_set")
    if not isinstance(prediction_set, dict) or prediction_set.get(
        "count"
    ) != expected_prediction_count:
        raise ValueError("Stage 2 inference prediction set is incomplete")
    records = prediction_set.get("records")
    if not isinstance(records, list) or len(records) != expected_prediction_count:
        raise ValueError("Stage 2 inference prediction records are incomplete")
    identifiers = [record.get("image_id") for record in records if isinstance(record, dict)]
    if len(identifiers) != expected_prediction_count or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Stage 2 inference prediction IDs are incomplete or duplicated")
    if canonical_sha256(sorted(map(str, identifiers))) != expected_image_id_set_sha256:
        raise ValueError("Stage 2 inference prediction IDs differ from the held-out universe")
    prediction_digest = prediction_set.get("sha256")
    if not _is_hex(prediction_digest, 64):
        raise ValueError("Stage 2 inference prediction set lacks an immutable digest")
    if (
        identities.get("prediction_set_sha256") != prediction_digest
        or identities.get("prediction_count") != expected_prediction_count
    ):
        raise ValueError("Stage 2 inference prediction identity is inconsistent")
    require_clean_process_start(
        card,
        expected_inference_commit,
        description="reused Stage 2 U-Net seed-0 inference",
    )
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Stage 2 prediction directory is absent: {masks_root}")
    observed_records = []
    expected_files = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Stage 2 prediction record is not an object")
        image_id = str(record.get("image_id", ""))
        if not image_id or Path(image_id).name != image_id:
            raise ValueError("Stage 2 prediction record has an unsafe image id")
        path = masks_root / f"{image_id}.png"
        if path.parent.resolve() != masks_root.resolve():
            raise ValueError("Stage 2 prediction path escapes its directory")
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Stage 2 prediction bytes changed: {image_id}")
        observed_records.append(
            {
                "image_id": image_id,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        expected_files.add(f"{image_id}.png")
    actual_files = {
        path.relative_to(masks_root).as_posix()
        for path in masks_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("Stage 2 prediction directory contains unregistered files")
    observed_records.sort(key=lambda item: item["image_id"])
    if canonical_sha256(observed_records) != prediction_digest:
        raise ValueError("Stage 2 prediction-set digest does not reproduce")
    return {
        "method": "unet",
        "seed": 0,
        "source": "stage2_delivery",
        "delivery": _identity(delivery_path),
        "stage2_gate": _identity(stage2_gate_path),
        "pilot_selection": _identity(pilot_selection_path),
        "inference_card": _identity(inference_card_path),
        "checkpoint_sha256": unet_run["checkpoint_sha256"],
        "prediction_set_sha256": prediction_digest,
        "prediction_count": expected_prediction_count,
        "masks_out": str(masks_root),
        "expected_inference_git_commit": expected_inference_commit,
    }


def _inference_job(
    *,
    record: Mapping[str, Any],
    python_executable: Path,
    code_root: Path,
    inference_commit: str,
    manifest: Path,
    splits: Path,
    masks_root: Path,
    cards_root: Path,
    work_root: Path,
    cache_dir: Path,
    threshold_amendment: Path,
    ram_gib: float,
    segformer_env: Optional[Mapping[str, str]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    method = str(record["method"])
    seed = int(record["seed"])
    name = f"infer.{method}.seed{seed}"
    masks = masks_root / method / f"seed{seed}"
    card = cards_root / f"{method}_seed{seed}.json"
    work = work_root / method / f"seed{seed}"
    argv = [
        str(python_executable),
        str(code_root / "scripts" / "infer_masks_repro.py"),
        "--config", str(record["config"]),
        "--checkpoint", str(record["checkpoint"]),
        "--manifest", str(manifest),
        "--splits", str(splits),
        "--subset", "finger_latent",
        "--split", "test",
        "--out", str(masks),
        "--card-out", str(card),
        "--work-dir", str(work),
        "--expected-git-commit", inference_commit,
        "--expected-checkpoint-sha256", str(record["checkpoint_sha256"]),
        "--device", "cuda",
        "--memory-fraction", "0.3",
        "--warmup", "10",
        "--seed", str(seed),
        "--threshold-receipt", str(record["threshold_receipt"]),
        "--threshold-amendment", str(threshold_amendment),
        "--cache-dir", str(cache_dir),
    ]
    env = dict(segformer_env or {}) if method == "segformer_sub" else {}
    resources = {"host_ram_gib": ram_gib}
    completion = str(card)
    job = {
        "name": name,
        "argv": argv,
        "completion_artifact": completion,
        "resources": resources,
    }
    if env:
        job["env"] = env
    job["job_identity_sha256"] = canonical_sha256(
        {
            "name": name,
            "argv": argv,
            "env": env,
            "completion_artifact": completion,
            "resources": resources,
        }
    )
    plan_record = {
        **copy.deepcopy(dict(record)),
        "expected_training_git_commit": str(record["training_git_commit"]),
        "expected_inference_git_commit": inference_commit,
        "masks_out": str(masks),
        "card_out": str(card),
        "inference_work_dir": str(work),
        "threshold_receipt_sha256": str(record["threshold_receipt_sha256"]),
        "threshold_amendment_sha256": sha256_file(threshold_amendment),
        "preprocess_cache_receipt_sha256": sha256_file(cache_dir / "receipt.json"),
        "job_name": name,
        "job_identity_sha256": job["job_identity_sha256"],
    }
    return job, plan_record


def build(args: argparse.Namespace, provenance: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Validate training evidence and return gate, plan, and guarded queue."""
    require_clean_process_start(
        {"provenance": provenance},
        args.expected_planner_git_commit,
        description="Stage 3 inference planner",
    )
    for path, description in (
        (args.runs_root, "training runs root"),
        (args.masks_root, "deterministic masks root"),
        (args.cards_root, "inference cards root"),
        (args.work_root, "inference work root"),
        (args.gate_out, "readiness gate output"),
        (args.plan_out, "inference plan output"),
        (args.queue_out, "inference queue output"),
        (args.prep_bundle_out, "preparation bundle output"),
        (args.aux_plan_out, "DRUnet auxiliary plan output"),
        (args.aux_queue_out, "DRUnet auxiliary queue output"),
        (args.aux_card_out, "DRUnet auxiliary completion card"),
        (args.aux_masks_root, "reserved DRUnet auxiliary masks root"),
        (args.aux_variance_root, "DRUnet auxiliary variance root"),
        (args.aux_work_root, "DRUnet auxiliary work root"),
        (args.threshold_selector_manifest, "threshold-selector manifest"),
        (args.cache_dir, "preprocessing cache"),
        (args.threshold_amendment, "threshold amendment"),
    ):
        _require_under(path, args.restricted_root, description)
    auxiliary_roots = (
        args.aux_masks_root,
        args.aux_variance_root,
        args.aux_work_root,
    )
    deterministic_roots = (args.masks_root, args.cards_root, args.work_root)
    for auxiliary in auxiliary_roots:
        for deterministic in deterministic_roots:
            _require_disjoint(
                auxiliary, deterministic, "auxiliary/deterministic output"
            )
    for deterministic in deterministic_roots:
        _require_disjoint(
            args.aux_card_out, deterministic, "auxiliary card/deterministic output"
        )
    for index, left in enumerate(auxiliary_roots):
        for right in auxiliary_roots[index + 1 :]:
            _require_disjoint(left, right, "DRUnet auxiliary output")
    max_project_ram_gib = float(args.max_project_ram_gib)
    if not math.isfinite(max_project_ram_gib) or max_project_ram_gib <= 0.0:
        raise ValueError("project RAM limit must be a positive runtime cap")
    aux_ram_gib = float(args.aux_ram_gib)
    if not math.isfinite(aux_ram_gib) or not 0.0 < aux_ram_gib <= max_project_ram_gib:
        raise ValueError("DRUnet auxiliary RAM reservation exceeds the runtime cap")
    inference_commit = _validate_commit(
        args.inference_git_commit, "inference Git commit"
    )
    allowed_gpus = _gpu_indices(args.allowed_gpus)
    stage2_inference_commit = _validate_commit(
        args.stage2_inference_git_commit, "Stage 2 inference Git commit"
    )
    _validate_pinned_code_root(args.code_root, inference_commit)
    expected_commits = _commit_map(args.expected_commits)
    full_run_commits = _full_run_commit_map(args.full_run_expected_commits)
    reservations = _load_reservations(
        args.ram_reservations, max_project_ram_gib
    )
    universe = _test_universe(
        args.manifest, args.splits, args.expected_prediction_count
    )
    auxiliary_sample = _auxiliary_sample(args.manifest, args.splits)
    manifest_audit, ordered_source_gt = _manifest_audit_identity(
        args.manifest_audit, args.manifest
    )
    scorer_code = _scorer_code_identities(args.code_root)
    inference_code = _inference_code_identities(args.code_root)
    native_sweep_queue, native_sweep_ledger, native_sweep_jobs = (
        _completed_native_sweep(args.native_sweep_queue, args.native_sweep_ledger)
    )
    selection, reconstructed_selection = _validate_global_selection(
        global_selection_path=args.global_selection,
        pilot_selection_path=args.pilot_selection,
        sufs_selection_path=args.sufs_selection,
        expected_commits=expected_commits,
        manifest=args.manifest,
        splits=args.splits,
        native_sweep_jobs=native_sweep_jobs,
    )
    production_inputs = _production_inference_inputs(
        selection, args.cache_dir, args.threshold_amendment
    )
    threshold_evidence = _threshold_evidence_index(
        args.threshold_selector_manifest,
        selection=selection,
        cache_receipt_sha256=production_inputs["cache_receipt_sha256"],
        amendment_sha256=production_inputs["threshold_amendment_sha256"],
    )
    seed0_queue, seed0_ledger, seed0_jobs = _training_wave(
        args.seed0_queue, args.seed0_ledger, SEED0_NAMES, "native seed-0"
    )
    seed12_queue, seed12_ledger, seed12_jobs = _training_wave(
        args.seed12_queue, args.seed12_ledger, SEED12_NAMES, "native seed-1/2"
    )
    nonterminal = {
        name: entry["status"]
        for ledger in (seed0_ledger, seed12_ledger)
        for name, entry in ledger["jobs"].items()
        if entry["status"] not in {"completed", "failed"}
    }
    if nonterminal:
        raise RuntimeError(
            "training ledgers are not terminal; readiness cannot be frozen: "
            + json.dumps(nonterminal, sort_keys=True)
        )
    if seed0_queue.get("global_selection_sha256") != sha256_file(
        args.global_selection
    ) or seed12_queue.get("global_selection_sha256") != sha256_file(
        args.global_selection
    ):
        raise ValueError("training queue global-selection binding changed")
    proof_identities = _validate_training_proofs(
        global_selection=args.global_selection,
        seed0_queue=args.seed0_queue,
        seed12_plan=args.seed12_plan,
        seed12_queue=args.seed12_queue,
        prepare_proof=args.prepare_proof,
        seed12_proof=args.seed12_proof,
    )

    methods: Dict[str, Any] = {}
    valid_records: list[Dict[str, Any]] = []
    unet0 = _validate_full_run(
        method="unet",
        seed=0,
        expected_commit=full_run_commits["unet.seed0"],
        selection=selection,
        runs_root=args.runs_root,
    )
    for method in NATIVE_METHODS:
        records: Dict[str, Dict[str, Any]] = {}
        for seed in SEEDS:
            if (method, seed) == STAGE2_REUSED_RUN:
                record = unet0
            else:
                name = f"full.{method}.seed{seed}"
                if seed == 0:
                    ledger_entry = seed0_ledger["jobs"][name]
                    queue_job = seed0_jobs[name]
                else:
                    ledger_entry = seed12_ledger["jobs"][name]
                    queue_job = seed12_jobs[name]
                record = _status_record(
                    name=name,
                    ledger_entry=ledger_entry,
                    queue_job=queue_job,
                    method=method,
                    seed=seed,
                    selection=selection,
                    runs_root=args.runs_root,
                    expected_commit=full_run_commits[f"{method}.seed{seed}"],
                )
            records[str(seed)] = record
            if record["status"] == "valid":
                valid_records.append(record)
        valid = [seed for seed in SEEDS if records[str(seed)]["status"] == "valid"]
        failed = [seed for seed in SEEDS if records[str(seed)]["status"] == "failed"]
        missing = [seed for seed in SEEDS if records[str(seed)]["status"] == "missing"]
        methods[method] = {
            "valid_seeds": valid,
            "failed_seeds": failed,
            "missing_seeds": missing,
            "seeds": records,
        }
    insufficient = _insufficient_methods(methods)
    expected_threshold_keys = {
        (str(record["method"]), int(record["seed"]))
        for record in valid_records
        if (record["method"], record["seed"]) != STAGE2_REUSED_RUN
    }
    if set(threshold_evidence) != expected_threshold_keys:
        raise ValueError(
            "threshold-selector manifest does not exactly cover valid amended runs"
        )
    for record in valid_records:
        key = (str(record["method"]), int(record["seed"]))
        if key != STAGE2_REUSED_RUN:
            _attach_threshold_evidence(
                record,
                threshold_evidence[key],
                selection=selection,
                amendment_sha256=production_inputs[
                    "threshold_amendment_sha256"
                ],
            )
    reused = _validate_stage2_reuse(
        delivery_path=args.stage2_delivery,
        inference_card_path=args.stage2_inference_card,
        unet_run=unet0,
        expected_inference_commit=stage2_inference_commit,
        expected_prediction_count=args.expected_prediction_count,
        expected_image_id_set_sha256=universe["image_id_set_sha256"],
        masks_root=args.stage2_masks,
        stage2_gate_path=args.stage2_gate,
        pilot_selection_path=args.pilot_selection,
    )
    has_warnings = any(
        methods[method]["failed_seeds"] or methods[method]["missing_seeds"]
        for method in NATIVE_METHODS
    )
    inputs = {
        "manifest": _identity(args.manifest),
        "splits": _identity(args.splits),
        "global_selection": _identity(args.global_selection),
        "pilot_selection": _identity(args.pilot_selection),
        "sufs_selection": _identity(args.sufs_selection),
        "expected_commits": _identity(args.expected_commits),
        "full_run_expected_commits": _identity(args.full_run_expected_commits),
        "manifest_audit": manifest_audit,
        "ram_reservations": _identity(args.ram_reservations),
        "native_sweep_queue": _identity(args.native_sweep_queue),
        "native_sweep_ledger": _identity(args.native_sweep_ledger),
        "seed0_queue": _identity(args.seed0_queue),
        "seed0_ledger": _identity(args.seed0_ledger),
        "seed12_queue": _identity(args.seed12_queue),
        "seed12_ledger": _identity(args.seed12_ledger),
        "stage2_delivery": _identity(args.stage2_delivery),
        "stage2_gate": _identity(args.stage2_gate),
        "stage2_inference_card": _identity(args.stage2_inference_card),
        "stage2_masks_prediction_set_sha256": reused["prediction_set_sha256"],
        "threshold_selector_manifest": _identity(args.threshold_selector_manifest),
        "cache_receipt": _identity(args.cache_dir / "receipt.json"),
        "threshold_amendment": _identity(args.threshold_amendment),
        **proof_identities,
    }
    gate = {
        "schema_version": SCHEMA_VERSION,
        "kind": "stage3_native_training_readiness_gate",
        "status": (
            "FAIL"
            if insufficient
            else "PASS_WITH_WARNINGS" if has_warnings else "PASS"
        ),
        "validation_only_selection": True,
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "test_metadata_opened": True,
        "minimum_valid_seeds_per_method": 2,
        "expected_prediction_count": args.expected_prediction_count,
        "test_universe": universe,
        "training_commit_policy": full_run_commits,
        "independently_reconstructed_lr_selection": reconstructed_selection,
        "stage2_inference_git_commit": stage2_inference_commit,
        "expected_inference_git_commit": inference_commit,
        "inference_code": inference_code,
        "scorer_code": scorer_code,
        "reused": reused,
        "methods": methods,
        "identities": {
            "manifest_sha256": sha256_file(args.manifest),
            "splits_sha256": sha256_file(args.splits),
        },
        "inputs": inputs,
        "planner_provenance": _stable_provenance(provenance),
    }
    _write_json_once(args.gate_out, gate)
    if insufficient:
        raise RuntimeError(
            "Stage 3 inference gate requires at least two valid seeds per method: "
            + json.dumps(insufficient, sort_keys=True)
        )

    segformer_env: Optional[Mapping[str, str]] = None
    for job in (*seed0_jobs.values(), *seed12_jobs.values()):
        if str(job.get("name", "")).startswith("full.segformer_sub."):
            observed = job.get("env")
            if not isinstance(observed, dict) or set(observed) != {
                "SEGBENCH_SEGFORMER_B2_DIR"
            }:
                raise ValueError("SegFormer training queue lacks its pinned snapshot")
            if segformer_env is None:
                segformer_env = {str(key): str(value) for key, value in observed.items()}
            elif dict(observed) != dict(segformer_env):
                raise ValueError("SegFormer snapshot differs across training jobs")
    if segformer_env is None:
        raise ValueError("SegFormer pinned environment is absent")

    jobs: list[Dict[str, Any]] = []
    planned_runs: list[Dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_cards: set[str] = set()
    for record in valid_records:
        if (record["method"], record["seed"]) == STAGE2_REUSED_RUN:
            continue
        name = f"infer.{record['method']}.seed{record['seed']}"
        job, plan_record = _inference_job(
            record=record,
            python_executable=args.python_executable,
            code_root=args.code_root,
            inference_commit=inference_commit,
            manifest=args.manifest,
            splits=args.splits,
            masks_root=args.masks_root,
            cards_root=args.cards_root,
            work_root=args.work_root,
            cache_dir=args.cache_dir.resolve(),
            threshold_amendment=args.threshold_amendment.resolve(),
            ram_gib=reservations[name],
            segformer_env=segformer_env,
        )
        if job["name"] in seen_names or job["completion_artifact"] in seen_cards:
            raise ValueError("duplicate Stage 3 inference job or completion artifact")
        seen_names.add(job["name"])
        seen_cards.add(job["completion_artifact"])
        jobs.append(job)
        planned_runs.append(plan_record)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "stage3_native_inference_plan",
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "ordered_source_images_sha256": universe[
            "ordered_source_images_sha256"
        ],
        "gate": str(args.gate_out),
        "gate_sha256": sha256_file(args.gate_out),
        "manifest_audit": manifest_audit,
        "ordered_source_gt_sha256": ordered_source_gt,
        "code": scorer_code,
        "inference_code": inference_code,
        "expected_prediction_count": args.expected_prediction_count,
        "expected_inference_git_commit": inference_commit,
        "reused": reused,
        "runs": planned_runs,
    }
    _write_json_once(args.plan_out, plan)
    queue = {
        "schema_version": 2,
        "phase": "stage3_native_test_inference",
        "launch_authorized": True,
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "required_max_jobs_per_gpu": 1,
        "required_launcher_args": ["--max-per-gpu", "1"],
        "required_gpu_indices": allowed_gpus,
        "timing_policy": "one fresh timed inference per GPU with no sibling timing job",
        "gate_sha256": sha256_file(args.gate_out),
        "plan_sha256": sha256_file(args.plan_out),
        "jobs": jobs,
    }
    _write_yaml_once(args.queue_out, queue)

    drunet_seed = min(methods["drunet"]["valid_seeds"])
    drunet_record = copy.deepcopy(methods["drunet"]["seeds"][str(drunet_seed)])
    if drunet_record.get("status") != "valid":
        raise ValueError("selected DRUnet auxiliary seed is not valid")
    auxiliary_plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": "stage3_drunet_mc_dropout_plan",
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "timing_excluded": True,
        "mask_outputs_forbidden": True,
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "selection_rule": auxiliary_sample["selection_rule"],
        "test_image_ids": auxiliary_sample["test_image_ids"],
        "image_id_set_sha256": auxiliary_sample["image_id_set_sha256"],
        "ordered_source_images_sha256": auxiliary_sample[
            "ordered_source_images_sha256"
        ],
        "expected_shapes": auxiliary_sample["expected_shapes"],
        "gate": _identity(args.gate_out),
        "deterministic_plan": _identity(args.plan_out),
        "run": {
            **drunet_record,
            "expected_training_git_commit": drunet_record[
                "training_git_commit"
            ],
            "expected_inference_git_commit": inference_commit,
        },
        "identities": {
            "config": _identity(Path(str(drunet_record["config"]))),
            "checkpoint": _identity(Path(str(drunet_record["checkpoint"]))),
            "run_card": _identity(Path(str(drunet_record["run_card"]))),
            "manifest": _identity(args.manifest),
            "splits": _identity(args.splits),
            "wrapper": inference_code["auxiliary_wrapper"],
        },
        "reserved_masks_root": str(args.aux_masks_root),
        "variance_out": str(args.aux_variance_root),
        "work_dir": str(args.aux_work_root),
        "card_out": str(args.aux_card_out),
    }
    _write_json_once(args.aux_plan_out, auxiliary_plan)
    aux_job_name = f"aux.drunet.seed{drunet_seed}.mc_dropout20"
    aux_argv = [
        str(args.python_executable),
        str(args.code_root / "scripts" / "infer_drunet_mc_dropout.py"),
        "--plan", str(args.aux_plan_out),
        "--expected-plan-sha256", sha256_file(args.aux_plan_out),
        "--config", str(drunet_record["config"]),
        "--checkpoint", str(drunet_record["checkpoint"]),
        "--manifest", str(args.manifest),
        "--splits", str(args.splits),
        "--variance-out", str(args.aux_variance_root),
        "--card-out", str(args.aux_card_out),
        "--work-dir", str(args.aux_work_root),
        "--expected-git-commit", inference_commit,
        "--expected-checkpoint-sha256", str(drunet_record["checkpoint_sha256"]),
        "--seed", str(drunet_seed),
        "--device", "cuda",
        "--memory-fraction", "0.3",
    ]
    aux_resources = {"host_ram_gib": aux_ram_gib}
    aux_job = {
        "name": aux_job_name,
        "argv": aux_argv,
        "completion_artifact": str(args.aux_card_out),
        "resources": aux_resources,
    }
    aux_job["job_identity_sha256"] = canonical_sha256(
        {
            "name": aux_job_name,
            "argv": aux_argv,
            "env": {},
            "completion_artifact": str(args.aux_card_out),
            "resources": aux_resources,
        }
    )
    auxiliary_queue = {
        "schema_version": 2,
        "phase": "stage3_drunet_mc_dropout_aux",
        "launch_authorized": True,
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "timing_excluded": True,
        "mask_outputs_forbidden": True,
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "required_max_jobs_per_gpu": 1,
        "required_launcher_args": ["--max-per-gpu", "1"],
        "required_gpu_indices": allowed_gpus,
        "plan_sha256": sha256_file(args.aux_plan_out),
        "jobs": [aux_job],
    }
    _write_yaml_once(args.aux_queue_out, auxiliary_queue)
    prep_bundle = {
        "schema_version": SCHEMA_VERSION,
        "kind": "stage3_native_frozen_preparation_bundle",
        "gate": _identity(args.gate_out),
        "plan": _identity(args.plan_out),
        "queue": _identity(args.queue_out),
        "aux_plan": _identity(args.aux_plan_out),
        "aux_queue": _identity(args.aux_queue_out),
        "planner": _identity(Path(__file__)),
        "expected_planner_git_commit": args.expected_planner_git_commit,
        "expected_inference_git_commit": inference_commit,
        "expected_finalizer_git_commit": _validate_commit(
            args.finalizer_git_commit, "finalizer Git commit"
        ),
        "inference_code": inference_code,
        "scorer_code": scorer_code,
        "manifest": _identity(args.manifest),
        "splits": _identity(args.splits),
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "runtime_policy": {
            "gpu_indices": allowed_gpus,
            "max_jobs_per_gpu": 1,
            "memory_fraction": 0.3,
            "max_project_ram_gib": max_project_ram_gib,
        },
    }
    _write_json_once(args.prep_bundle_out, prep_bundle)
    return gate, plan, queue


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "global-selection", "pilot-selection", "sufs-selection", "expected-commits",
        "full-run-expected-commits", "manifest", "manifest-audit", "splits",
        "runs-root", "native-sweep-queue", "native-sweep-ledger", "seed0-queue", "seed0-ledger",
        "seed12-plan", "seed12-queue", "seed12-ledger", "prepare-proof",
        "seed12-proof", "stage2-delivery", "stage2-gate", "stage2-inference-card", "stage2-masks", "code-root",
        "ram-reservations", "masks-root", "cards-root", "work-root", "gate-out",
        "plan-out", "queue-out", "prep-bundle-out", "restricted-root",
        "aux-plan-out", "aux-queue-out", "aux-card-out", "aux-masks-root",
        "aux-variance-root", "aux-work-root", "threshold-selector-manifest",
        "cache-dir", "threshold-amendment",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--expected-planner-git-commit", required=True)
    parser.add_argument("--inference-git-commit", required=True)
    parser.add_argument("--stage2-inference-git-commit", required=True)
    parser.add_argument("--finalizer-git-commit", required=True)
    parser.add_argument("--allowed-gpus", required=True)
    parser.add_argument("--max-project-ram-gib", required=True, type=float)
    parser.add_argument("--aux-ram-gib", required=True, type=float)
    parser.add_argument(
        "--expected-prediction-count", type=int, default=EXPECTED_PREDICTION_COUNT
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    gate, plan, queue = build(args, provenance)
    print(
        json.dumps(
            {
                "gate_status": gate["status"],
                "planned_inference_jobs": len(queue["jobs"]),
                "reused_inference_runs": 1 if plan.get("reused") else 0,
                "gate_sha256": sha256_file(args.gate_out),
                "plan_sha256": sha256_file(args.plan_out),
                "queue_sha256": sha256_file(args.queue_out),
                "prep_bundle_sha256": sha256_file(args.prep_bundle_out),
                "auxiliary_plan_sha256": sha256_file(args.aux_plan_out),
                "auxiliary_queue_sha256": sha256_file(args.aux_queue_out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
