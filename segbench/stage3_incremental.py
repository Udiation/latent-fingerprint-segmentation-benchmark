"""Incremental, one-time Stage-3 held-out delivery primitives.

The v3 protocol requires every completed seed-0 model to receive its single
held-out score before the multi-seed Stage-4 gate can be satisfied.  This
module keeps those concerns separate:

* :func:`prepare_seed0` freezes a ground-truth-blind inference plan for exactly
  the seven native seed-0 runs not already delivered by Stage 2;
* :func:`finalize_seed` invokes the shared threshold scorer at most once and
  publishes one immutable per-seed delivery; and
* :func:`import_deliveries` combines immutable seed deliveries only after the
  original two-successful-seeds-per-method readiness gate passes.

The importer never invokes inference or the evaluator.  Seed-0 scientific
files are copied byte-for-byte into the final Stage-3 bundle, making later
aggregation an import rather than a rescore.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import yaml

from segbench.artifact_identity import canonical_sha256, sha256_file


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
SEED0_INFERENCE_METHODS = tuple(method for method in NATIVE_METHODS if method != "unet")
REQUIRED_SEEDS = (0, 1, 2)
PLAN_SPEC_KIND = "stage3_incremental_seed0_plan_spec_v1"
ADDITIONAL_PLAN_SPEC_KIND = "stage3_incremental_additional_plan_spec_v1"
GATE_KIND = "stage3_incremental_seed0_readiness_gate_v1"
PLAN_KIND = "stage3_incremental_seed0_inference_plan_v1"
ADDITIONAL_GATE_KIND = "stage3_incremental_additional_readiness_gate_v1"
ADDITIONAL_PLAN_KIND = "stage3_incremental_additional_inference_plan_v1"
SEED_DELIVERY_KIND = "stage3_incremental_seed_delivery_v1"
IMPORT_SPEC_KIND = "stage3_incremental_import_spec_v1"
FINAL_DELIVERY_KIND = "stage3_native_heldout_delivery"
SELECTOR_KIND = "expanded_validation_threshold_selector_receipt"
UNBIASED_CARD_KIND = "complete_unbiased_inference_card"
PROBABILITY_CARD_KIND = "complete_probability_map_card"
BOUND_CARD_KIND = "threshold_bound_inference_card"
SCORING_RECEIPT_KIND = "superseding_threshold_scoring_receipt"
READINESS_GATE_KIND = "stage3_native_training_readiness_gate"
AMENDED_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
HEX = set("0123456789abcdef")

METHOD_METADATA: Mapping[str, Mapping[str, str]] = {
    "unet": {"display_name": "U-Net", "year": "2022", "family": "Encoder-decoder"},
    "unetpp": {"display_name": "UNet++", "year": "2020", "family": "Encoder-decoder"},
    "drunet": {"display_name": "DRUnet", "year": "", "family": "Encoder-decoder"},
    "ra_runet": {"display_name": "RUnet (no alignment)", "year": "2021", "family": "Encoder-decoder"},
    "lenet": {"display_name": "LeNet", "year": "2019", "family": "Patch CNN"},
    "alexnet": {"display_name": "AlexNet", "year": "2019", "family": "Patch CNN"},
    "cnn_ae": {"display_name": "CNN + stacked AE", "year": "2023", "family": "Patch CNN"},
    "segformer_sub": {"display_name": "SegFormer-B2 (substitute†)", "year": "2021", "family": "Transformer"},
}


class IncrementalError(RuntimeError):
    """An incremental Stage-3 artifact violates the frozen protocol."""


def require(condition: bool, message: str) -> None:
    """Fail closed when ``condition`` is false."""
    if not condition:
        raise IncrementalError(message)


def _sha(value: object, description: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and set(value) <= HEX,
            f"{description} is not a lowercase SHA-256")
    return str(value)


def _commit(value: object, description: str) -> str:
    require(isinstance(value, str) and len(value) == 40 and set(value) <= HEX,
            f"{description} is not an immutable Git SHA")
    return str(value)


def _under(path: Path, root: Path, description: str, *, file: bool = True) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise IncrementalError(f"{description} escapes the restricted root") from exc
    require(not path.is_symlink(), f"{description} must not be a symlink")
    require(path.is_file() if file else path.is_dir(), f"{description} is absent")
    return resolved


def _identity(record: object, root: Path, description: str) -> Dict[str, Any]:
    require(isinstance(record, Mapping), f"{description} identity is malformed")
    path = _under(Path(str(record.get("path", ""))), root, description)
    digest = _sha(record.get("sha256"), f"{description} digest")
    require(sha256_file(path) == digest, f"{description} bytes changed")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def identity(path: Path) -> Dict[str, Any]:
    """Return a stable regular-file identity."""
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_shared_inputs(
    manifest: Mapping[str, Any],
    amendment: Mapping[str, Any],
    cache: Mapping[str, Any],
) -> None:
    """Authenticate the uniform cache and expanded validation-only grid."""
    amendment_payload = _json(Path(str(amendment["path"])), "threshold amendment")
    replacement = amendment_payload.get("replacement_policy")
    require(
        amendment_payload.get("kind") == "stage4_threshold_policy_amendment"
        and isinstance(replacement, Mapping)
        and tuple(float(value) for value in replacement.get("allowed_trainable_grid", ()))
        == AMENDED_GRID
        and replacement.get("selection_data") == "validation_only",
        "threshold amendment no longer defines the expanded validation-only grid",
    )
    cache_payload = _json(Path(str(cache["path"])), "preprocessing cache receipt")
    require(
        cache_payload.get("schema_version") == 1
        and cache_payload.get("kind") == "uint8_500ppi_preprocessing_cache"
        and cache_payload.get("manifest_sha256") == manifest["sha256"]
        and float(cache_payload.get("target_ppi", math.nan)) == 500.0
        and cache_payload.get("mask_columns") == ["mask_path", "quality_mask_path"]
        and cache_payload.get("debug_limited") is False,
        "preprocessing cache receipt is not the complete 500-ppi corpus",
    )


def _json(path: Path, description: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IncrementalError(f"{description} is invalid JSON") from exc
    require(isinstance(value, dict), f"{description} is not an object")
    return value


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(_json(path, "existing immutable JSON") == dict(payload),
                f"refusing to replace changed artifact: {path}")
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_yaml_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        require(existing == dict(payload), f"refusing to replace changed artifact: {path}")
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())


def _output_path(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise IncrementalError(f"{description} escapes the restricted root") from exc
    require(not path.is_symlink(), f"{description} must not be a symlink")
    return resolved


def _require_disjoint(paths: Mapping[str, str], description: str) -> None:
    resolved = {name: Path(value).resolve() for name, value in paths.items()}
    for left_name, left in resolved.items():
        for right_name, right in resolved.items():
            if left_name >= right_name:
                continue
            try:
                left.relative_to(right)
                overlap = True
            except ValueError:
                try:
                    right.relative_to(left)
                    overlap = True
                except ValueError:
                    overlap = False
            require(not overlap, f"{description} overlap: {left_name}/{right_name}")


def _validate_code_root(repo_root: Path, code_root: Path, expected_commit: str) -> None:
    """Preflight committed scorer bytes before publishing the score journal."""
    expected = _commit(expected_commit, "incremental finalizer commit")
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=repo_root,
        check=False, capture_output=True, text=True,
    )
    require(completed.returncode == 0
            and Path(completed.stdout.strip()).resolve() == repo_root.resolve(),
            "--repo-root is not the Git top level")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        check=False, capture_output=True, text=True,
    )
    require(completed.returncode == 0 and completed.stdout.strip() == expected,
            "repository HEAD differs from expected finalizer commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root, check=False, capture_output=True, text=True,
    )
    require(dirty.returncode == 0 and not dirty.stdout.strip(),
            "repository is dirty before one-time scoring")
    for name in ("score_threshold_run.py", "evaluate_masks.py"):
        path = code_root / "scripts" / name
        require(path.is_file(), f"committed scoring code is absent: {name}")
        try:
            relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError as exc:
            raise IncrementalError("scoring code root is outside the repository") from exc
        located = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=repo_root,
            check=False, capture_output=True,
        )
        require(located.returncode == 0, f"scoring code is not committed: {relative}")
        committed = subprocess.run(
            ["git", "show", f"{expected}:{relative}"], cwd=repo_root,
            check=False, capture_output=True,
        )
        require(committed.returncode == 0 and committed.stdout == path.read_bytes(),
                f"scoring code differs from committed bytes: {relative}")


def _test_universe(manifest: Path, splits: Path) -> Dict[str, Any]:
    """Hash test source bytes without decoding images or opening labels."""
    split_payload = _json(splits, "subject split")
    assignments = split_payload.get("splits")
    require(isinstance(assignments, Mapping), "subject split lacks assignments")
    frame = pd.read_csv(manifest, dtype={"image_id": str, "subject_id": str})
    required = {"image_id", "image_path", "source_sha256", "subject_id", "subset", "examiner_value"}
    require(required.issubset(frame.columns), "manifest lacks source-universe fields")
    frame = frame[
        (frame.subset.astype(str) == "finger_latent")
        & (frame.subject_id.map(assignments).astype(str) == "test")
        & (frame.examiner_value.astype(str).str.upper() != "NONPRINT")
    ].sort_values("image_id", kind="stable")
    require(len(frame) > 0 and not frame.image_id.astype(str).duplicated().any(),
            "test source universe is empty or duplicated")
    records = []
    for row in frame.to_dict("records"):
        image_id = str(row["image_id"])
        expected = _sha(str(row["source_sha256"]), f"{image_id} source digest")
        source = Path(str(row["image_path"]))
        require(source.is_file() and not source.is_symlink(), f"{image_id}: source image absent")
        require(sha256_file(source) == expected, f"{image_id}: source image bytes changed")
        records.append({"image_id": image_id, "source_sha256": expected})
    return {
        "subset": "finger_latent",
        "split": "test",
        "excluded_examiner_values": ["NONPRINT"],
        "prediction_count": len(records),
        "image_id_set_sha256": canonical_sha256(sorted(row["image_id"] for row in records)),
        "ordered_source_images_sha256": canonical_sha256(records),
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
    }


def _validate_run_record(
    raw: object,
    root: Path,
    shared: Mapping[str, Any],
    *,
    allowed_methods: Sequence[str],
    allowed_seeds: Sequence[int],
) -> Dict[str, Any]:
    require(isinstance(raw, Mapping), "incremental run record is malformed")
    method, seed = str(raw.get("method", "")), raw.get("seed")
    require(method in set(allowed_methods) and type(seed) is int and seed in set(allowed_seeds),
            "incremental plan contains an unexpected method/seed")
    label = f"{method}/seed{seed}"
    artifact_names = (
        "config", "run_card", "checkpoint", "selector_receipt",
        "raw_validation_probability_card", "validation_probability_card",
    )
    artifacts = {
        name: _identity(raw.get(name), root, f"{label} {name}")
        for name in artifact_names
    }
    config = yaml.safe_load(Path(artifacts["config"]["path"]).read_text(encoding="utf-8"))
    run_card = _json(Path(artifacts["run_card"]["path"]), f"{label} run card")
    selector = _json(Path(artifacts["selector_receipt"]["path"]), f"{label} selector")
    require(
        isinstance(config, Mapping)
        and config.get("schema_version") == 1
        and config.get("method") == method
        and config.get("seed") == seed
        and config.get("task") in {"pixel", "patch"},
        f"{label} configuration identity changed",
    )
    require(
        run_card.get("method") == method
        and run_card.get("seed") == seed
        and run_card.get("source_config_sha256") == artifacts["config"]["sha256"]
        and run_card.get("checkpoint_sha256", {}).get("best.pt")
        == artifacts["checkpoint"]["sha256"],
        f"{label} training lineage changed",
    )
    threshold = float(selector.get("selected_threshold", math.nan))
    require(
        selector.get("schema_version") == 1
        and selector.get("kind") == SELECTOR_KIND
        and selector.get("method") == method
        and selector.get("task") == config.get("task")
        and selector.get("seed") == seed
        and selector.get("selection_split") == "val"
        and selector.get("selection_source") == "validation_only"
        and selector.get("validation_only") is True
        and selector.get("test_or_heldout_opened") is False
        and selector.get("source_config_sha256") == artifacts["config"]["sha256"]
        and selector.get("checkpoint_sha256") == artifacts["checkpoint"]["sha256"]
        and selector.get("manifest_sha256") == shared["manifest"]["sha256"]
        and selector.get("split_sha256") == shared["splits"]["sha256"]
        and selector.get("amendment_sha256") == shared["threshold_amendment"]["sha256"]
        and selector.get("threshold_amendment_sha256")
        == shared["threshold_amendment"]["sha256"]
        and selector.get("raw_validation_probability_card_sha256")
        == artifacts["raw_validation_probability_card"]["sha256"]
        and selector.get("validation_probability_card_sha256")
        == artifacts["validation_probability_card"]["sha256"]
        and selector.get("probabilities_are_pre_threshold") is True
        and selector.get("derived_from_binary_mask") is False
        and selector.get("debug_limited") is False
        and tuple(float(value) for value in selector.get("allowed_grid", ())) == AMENDED_GRID
        and threshold in AMENDED_GRID,
        f"{label} selector is not frozen validation-only evidence",
    )
    for name in ("raw_validation_probability_card", "validation_probability_card"):
        card = _json(Path(artifacts[name]["path"]), f"{label} {name}")
        require(
            card.get("schema_version") == 1
            and card.get("kind") == PROBABILITY_CARD_KIND
            and card.get("status") == "complete"
            and card.get("complete") is True
            and card.get("method") == method
            and card.get("seed") == seed
            and card.get("split") == "val"
            and card.get("debug_limited") is False
            and card.get("source_config_sha256") == artifacts["config"]["sha256"]
            and card.get("checkpoint_sha256") == artifacts["checkpoint"]["sha256"],
            f"{label} {name} lineage changed",
        )
    output_keys = (
        "masks_out", "probabilities_out", "card_out", "work_dir",
        "seed_delivery_out",
    )
    outputs = {key: str(_output_path(Path(str(raw.get(key, ""))), root, f"{method} {key}"))
               for key in output_keys}
    _require_disjoint(outputs, f"{label} output roots")
    training_lineage = str(raw.get("training_lineage", ""))
    require(training_lineage in {"legacy_completed_before_amendment", "amended_training"},
            f"{label} training-lineage label changed")
    lineage: Dict[str, Any] = {"training_lineage": training_lineage}
    if training_lineage == "legacy_completed_before_amendment":
        lineage["legacy_completion_evidence"] = _identity(
            raw.get("legacy_completion_evidence"), root,
            f"{label} legacy completion evidence",
        )
    else:
        lineage["per_epoch_training_profile_card"] = _identity(
            raw.get("per_epoch_training_profile_card"), root,
            f"{label} per-epoch training profile",
        )
        lineage["preprocess_cache_receipt"] = shared["cache_receipt"]
        if raw.get("acceleration_receipt") is not None:
            lineage["acceleration_receipt"] = _identity(
                raw.get("acceleration_receipt"), root,
                f"{label} acceleration receipt",
            )
    return {
        "method": method,
        "seed": seed,
        "task": str(config["task"]),
        "selected_threshold": threshold,
        "parameter_count": int(run_card.get("parameter_count", {}).get("total", 0)),
        "artifacts": artifacts,
        "outputs": outputs,
        **lineage,
    }


def _validate_run_output_universe(records: Sequence[Mapping[str, Any]]) -> None:
    """Require every planned output tree to be unique and non-overlapping."""
    outputs: Dict[str, str] = {}
    for record in records:
        label = f"{record['method']}.seed{record['seed']}"
        for name, value in record["outputs"].items():
            outputs[f"{label}.{name}"] = str(value)
    _require_disjoint(outputs, "incremental run output roots")


def prepare_seed0(
    *, spec_path: Path, expected_spec_sha256: str, restricted_root: Path,
    gate_out: Path, plan_out: Path, queue_out: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Freeze the G3 seed-0 inference wave without weakening the G4 gate."""
    root = restricted_root.resolve()
    for path, description in (
        (gate_out, "seed-0 gate output"),
        (plan_out, "seed-0 plan output"),
        (queue_out, "seed-0 queue output"),
    ):
        _output_path(path, root, description)
    require(root.is_dir() and not restricted_root.is_symlink(), "restricted root is invalid")
    spec_file = _under(spec_path, root, "seed-0 planning specification")
    require(sha256_file(spec_file) == _sha(expected_spec_sha256, "seed-0 spec digest"),
            "seed-0 planning specification bytes changed")
    spec = _json(spec_file, "seed-0 planning specification")
    require(spec.get("schema_version") == 1 and spec.get("kind") == PLAN_SPEC_KIND,
            "seed-0 planning specification schema/kind changed")
    manifest = _identity(spec.get("manifest"), root, "manifest")
    splits = _identity(spec.get("splits"), root, "subject split")
    amendment = _identity(spec.get("threshold_amendment"), root, "threshold amendment")
    cache = _identity(spec.get("cache_receipt"), root, "preprocessing cache receipt")
    _validate_shared_inputs(manifest, amendment, cache)
    stage2 = _identity(spec.get("stage2_unet_seed0_delivery"), root,
                       "Stage 2 U-Net seed-0 reselection delivery")
    stage2_payload = _json(Path(stage2["path"]), "Stage 2 U-Net seed-0 reselection delivery")
    require(
        stage2_payload.get("schema_version") == 1
        and stage2_payload.get("kind") == "stage2_threshold_reselection_delivery",
        "seed-0 plan does not bind the amended Stage 2 U-Net delivery",
    )
    shared = {
        "manifest": manifest,
        "splits": splits,
        "threshold_amendment": amendment,
        "cache_receipt": cache,
    }
    universe = _test_universe(Path(manifest["path"]), Path(splits["path"]))
    require(universe["prediction_count"] == int(spec.get("expected_prediction_count", -1)),
            "seed-0 expected prediction count changed")
    raw_records = spec.get("runs")
    require(isinstance(raw_records, list), "seed-0 specification lacks runs")
    records = [
        _validate_run_record(
            raw, root, shared,
            allowed_methods=SEED0_INFERENCE_METHODS,
            allowed_seeds=(0,),
        )
        for raw in raw_records
    ]
    by_method = {record["method"]: record for record in records}
    require(len(by_method) == len(records) and set(by_method) == set(SEED0_INFERENCE_METHODS),
            "seed-0 plan must cover exactly the seven non-U-Net native methods")
    _validate_run_output_universe(records)
    code_root = Path(str(spec.get("code_root", ""))).resolve()
    entrypoint = code_root / "scripts" / "infer_masks_repro.py"
    require(entrypoint.is_file() and not entrypoint.is_symlink(), "inference entrypoint is absent")
    inference_commit = _commit(spec.get("expected_inference_git_commit"), "inference commit")
    scoring_commit = _commit(spec.get("expected_scoring_git_commit"), "scoring commit")
    python = Path(str(spec.get("python_executable", ""))).resolve()
    require(python.is_file(), "Python executable is absent")
    allowed_gpus = spec.get("allowed_gpu_indices")
    require(
        isinstance(allowed_gpus, list)
        and allowed_gpus
        and all(type(value) is int and value >= 0 for value in allowed_gpus)
        and len(set(allowed_gpus)) == len(allowed_gpus),
        "allowed GPU indices are missing or malformed",
    )
    ram = spec.get("ram_gib")
    require(isinstance(ram, Mapping) and set(ram) == set(SEED0_INFERENCE_METHODS),
            "RAM reservation map must cover exactly the seven seed-0 jobs")
    jobs = []
    planned = []
    for method in SEED0_INFERENCE_METHODS:
        record = by_method[method]
        outputs = record["outputs"]
        argv = [
            str(python), str(entrypoint),
            "--config", record["artifacts"]["config"]["path"],
            "--checkpoint", record["artifacts"]["checkpoint"]["path"],
            "--manifest", manifest["path"], "--splits", splits["path"],
            "--subset", "finger_latent", "--split", "test",
            "--out", outputs["masks_out"],
            "--probability-out", outputs["probabilities_out"],
            "--threshold-receipt", record["artifacts"]["selector_receipt"]["path"],
            "--threshold-amendment", amendment["path"],
            "--cache-dir", str(Path(cache["path"]).parent),
            "--card-out", outputs["card_out"], "--work-dir", outputs["work_dir"],
            "--expected-git-commit", inference_commit,
            "--expected-checkpoint-sha256", record["artifacts"]["checkpoint"]["sha256"],
            "--device", "cuda", "--memory-fraction", "0.3", "--seed", "0",
        ]
        name = f"infer.{method}.seed0"
        job = {
            "name": name,
            "argv": argv,
            "completion_artifact": outputs["card_out"],
            "resources": {"host_ram_gib": float(ram[method])},
        }
        require(math.isfinite(job["resources"]["host_ram_gib"]) and job["resources"]["host_ram_gib"] > 0,
                f"{method}: invalid RAM reservation")
        job["job_identity_sha256"] = canonical_sha256(
            {"name": name, "argv": argv, "env": {},
             "completion_artifact": outputs["card_out"], "resources": job["resources"]}
        )
        jobs.append(job)
        planned.append(record)
    gate = {
        "schema_version": 1,
        "kind": GATE_KIND,
        "status": "PASS",
        "scope": "incremental_seed0_delivery_only",
        "minimum_valid_seeds_per_method": 1,
        "satisfies_stage4_multi_seed_gate": False,
        "stage4_minimum_remains": 2,
        "ground_truth_blind": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "test_universe": universe,
        "data_inputs": {**shared, "stage2_unet_seed0_delivery": stage2},
        "methods": {"unet": {"seed0": "stage2_reselection_reuse"},
                    **{method: {"seed0": "ready"} for method in SEED0_INFERENCE_METHODS}},
    }
    _write_json_once(gate_out, gate)
    plan = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "gate": identity(gate_out),
        "ground_truth_blind": True,
        "automatic_heldout_rescore_permitted": False,
        "expected_inference_git_commit": inference_commit,
        "expected_scoring_git_commit": scoring_commit,
        "code_root": str(code_root),
        "python_executable": str(python),
        "data_inputs": shared,
        "test_universe": universe,
        "runs": planned,
    }
    _write_json_once(plan_out, plan)
    queue = {
        "schema_version": 2,
        "phase": "stage3_native_test_inference",
        "incremental_wave": "seed0",
        "launch_authorized": True,
        "ground_truth_blind": True,
        "required_max_jobs_per_gpu": 1,
        "required_launcher_args": ["--max-per-gpu", "1"],
        "required_gpu_indices": list(allowed_gpus),
        "gate_sha256": sha256_file(gate_out),
        "plan_sha256": sha256_file(plan_out),
        "jobs": jobs,
    }
    _write_yaml_once(queue_out, queue)
    return gate, plan, queue


def prepare_additional(
    *, spec_path: Path, expected_spec_sha256: str, restricted_root: Path,
    gate_out: Path, plan_out: Path, queue_out: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Freeze later seed-1/2 inference without reopening delivered seed 0.

    The caller may include any non-empty subset of finished seed-1/2 runs.  A
    prior seed-delivery list is authenticated and its method/seed keys are
    forbidden from the new plan, so a completed seed can never be queued twice.
    This readiness artifact is operational only; it cannot satisfy G4.
    """
    root = restricted_root.resolve()
    require(root.is_dir() and not restricted_root.is_symlink(), "restricted root is invalid")
    for path, description in (
        (gate_out, "additional-seed gate output"),
        (plan_out, "additional-seed plan output"),
        (queue_out, "additional-seed queue output"),
    ):
        _output_path(path, root, description)
    spec_file = _under(spec_path, root, "additional-seed planning specification")
    require(sha256_file(spec_file) == _sha(expected_spec_sha256, "additional-seed spec digest"),
            "additional-seed planning specification bytes changed")
    spec = _json(spec_file, "additional-seed planning specification")
    require(spec.get("schema_version") == 1 and spec.get("kind") == ADDITIONAL_PLAN_SPEC_KIND,
            "additional-seed planning specification schema/kind changed")
    manifest = _identity(spec.get("manifest"), root, "manifest")
    splits = _identity(spec.get("splits"), root, "subject split")
    amendment = _identity(spec.get("threshold_amendment"), root, "threshold amendment")
    cache = _identity(spec.get("cache_receipt"), root, "preprocessing cache receipt")
    _validate_shared_inputs(manifest, amendment, cache)
    stage2 = _identity(spec.get("stage2_unet_seed0_delivery"), root,
                       "Stage 2 U-Net seed-0 reselection delivery")
    stage2_payload = _json(Path(stage2["path"]),
                           "Stage 2 U-Net seed-0 reselection delivery")
    require(
        stage2_payload.get("schema_version") == 1
        and stage2_payload.get("kind") == "stage2_threshold_reselection_delivery",
        "additional-seed plan does not bind the amended Stage 2 U-Net delivery",
    )
    shared = {
        "manifest": manifest,
        "splits": splits,
        "threshold_amendment": amendment,
        "cache_receipt": cache,
    }
    universe = _test_universe(Path(manifest["path"]), Path(splits["path"]))
    require(universe["prediction_count"] == int(spec.get("expected_prediction_count", -1)),
            "additional-seed expected prediction count changed")
    prior_raw = spec.get("prior_seed_deliveries")
    require(isinstance(prior_raw, list) and prior_raw,
            "additional-seed plan must bind prior seed deliveries")
    prior: list[Dict[str, Any]] = []
    prior_keys: set[tuple[str, int]] = set()
    for raw in prior_raw:
        item = _identity(raw, root, "prior incremental seed delivery")
        payload = _verify_seed_delivery(Path(item["path"]), root)
        require(payload.get("test_universe") == universe,
                "prior incremental seed delivery test universe changed")
        key = (str(payload.get("method", "")), int(payload.get("seed", -1)))
        require(key not in prior_keys, "duplicate prior incremental seed delivery")
        prior_keys.add(key)
        prior.append({"method": key[0], "seed": key[1], **item})
    required_seed0 = {(method, 0) for method in SEED0_INFERENCE_METHODS}
    require(required_seed0.issubset(prior_keys),
            "additional-seed plan requires all seven non-U-Net seed-0 deliveries")
    raw_records = spec.get("runs")
    require(isinstance(raw_records, list) and raw_records,
            "additional-seed specification lacks runs")
    records = [
        _validate_run_record(
            raw, root, shared,
            allowed_methods=NATIVE_METHODS,
            allowed_seeds=(1, 2),
        )
        for raw in raw_records
    ]
    keys = [(str(record["method"]), int(record["seed"])) for record in records]
    require(len(keys) == len(set(keys)) and not (set(keys) & prior_keys),
            "additional-seed plan duplicates a planned or delivered seed")
    _validate_run_output_universe(records)
    code_root = Path(str(spec.get("code_root", ""))).resolve()
    entrypoint = code_root / "scripts" / "infer_masks_repro.py"
    require(entrypoint.is_file() and not entrypoint.is_symlink(), "inference entrypoint is absent")
    inference_commit = _commit(spec.get("expected_inference_git_commit"), "inference commit")
    scoring_commit = _commit(spec.get("expected_scoring_git_commit"), "scoring commit")
    python = Path(str(spec.get("python_executable", ""))).resolve()
    require(python.is_file(), "Python executable is absent")
    allowed_gpus = spec.get("allowed_gpu_indices")
    require(
        isinstance(allowed_gpus, list)
        and allowed_gpus
        and all(type(value) is int and value >= 0 for value in allowed_gpus)
        and len(set(allowed_gpus)) == len(allowed_gpus),
        "allowed GPU indices are missing or malformed",
    )
    ram = spec.get("ram_gib")
    expected_ram = {f"{method}.seed{seed}" for method, seed in keys}
    require(isinstance(ram, Mapping) and set(ram) == expected_ram,
            "additional-seed RAM reservations differ from the run universe")
    jobs = []
    for record in sorted(records, key=lambda item: (str(item["method"]), int(item["seed"]))):
        method, seed = str(record["method"]), int(record["seed"])
        outputs = record["outputs"]
        argv = [
            str(python), str(entrypoint),
            "--config", record["artifacts"]["config"]["path"],
            "--checkpoint", record["artifacts"]["checkpoint"]["path"],
            "--manifest", manifest["path"], "--splits", splits["path"],
            "--subset", "finger_latent", "--split", "test",
            "--out", outputs["masks_out"],
            "--probability-out", outputs["probabilities_out"],
            "--threshold-receipt", record["artifacts"]["selector_receipt"]["path"],
            "--threshold-amendment", amendment["path"],
            "--cache-dir", str(Path(cache["path"]).parent),
            "--card-out", outputs["card_out"], "--work-dir", outputs["work_dir"],
            "--expected-git-commit", inference_commit,
            "--expected-checkpoint-sha256", record["artifacts"]["checkpoint"]["sha256"],
            "--device", "cuda", "--memory-fraction", "0.3", "--seed", str(seed),
        ]
        name = f"infer.{method}.seed{seed}"
        reservation = float(ram[f"{method}.seed{seed}"])
        require(math.isfinite(reservation) and reservation > 0,
                f"{method}/seed{seed}: invalid RAM reservation")
        resources = {"host_ram_gib": reservation}
        job = {
            "name": name,
            "argv": argv,
            "completion_artifact": outputs["card_out"],
            "resources": resources,
        }
        job["job_identity_sha256"] = canonical_sha256(
            {"name": name, "argv": argv, "env": {},
             "completion_artifact": outputs["card_out"], "resources": resources}
        )
        jobs.append(job)
    gate = {
        "schema_version": 1,
        "kind": ADDITIONAL_GATE_KIND,
        "status": "PASS",
        "scope": "incremental_additional_seed_delivery_only",
        "satisfies_stage4_multi_seed_gate": False,
        "stage4_minimum_remains": 2,
        "ground_truth_blind": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "test_universe": universe,
        "data_inputs": shared,
        "stage2_unet_seed0_delivery": stage2,
        "prior_seed_deliveries": prior,
        "planned_keys": [{"method": method, "seed": seed} for method, seed in sorted(keys)],
    }
    _write_json_once(gate_out, gate)
    plan = {
        "schema_version": 1,
        "kind": ADDITIONAL_PLAN_KIND,
        "gate": identity(gate_out),
        "ground_truth_blind": True,
        "automatic_heldout_rescore_permitted": False,
        "expected_inference_git_commit": inference_commit,
        "expected_scoring_git_commit": scoring_commit,
        "code_root": str(code_root),
        "python_executable": str(python),
        "data_inputs": shared,
        "stage2_unet_seed0_delivery": stage2,
        "test_universe": universe,
        "prior_seed_deliveries": prior,
        "runs": records,
    }
    _write_json_once(plan_out, plan)
    queue = {
        "schema_version": 2,
        "phase": "stage3_native_test_inference",
        "incremental_wave": "additional",
        "launch_authorized": True,
        "ground_truth_blind": True,
        "required_max_jobs_per_gpu": 1,
        "required_launcher_args": ["--max-per-gpu", "1"],
        "required_gpu_indices": list(allowed_gpus),
        "gate_sha256": sha256_file(gate_out),
        "plan_sha256": sha256_file(plan_out),
        "jobs": jobs,
    }
    _write_yaml_once(queue_out, queue)
    return gate, plan, queue


def _bound_inference_card(
    record: Mapping[str, Any], plan: Mapping[str, Any], test_card: Mapping[str, Any],
    test_card_path: Path,
) -> Dict[str, Any]:
    artifacts = record["artifacts"]
    selected = float(record["selected_threshold"])
    identities = test_card.get("identities")
    require(
        test_card.get("schema_version") == 1
        and test_card.get("kind") == PROBABILITY_CARD_KIND
        and test_card.get("status") == "complete"
        and test_card.get("complete") is True
        and test_card.get("method") == record["method"]
        and test_card.get("seed") == record["seed"]
        and test_card.get("split") == "test"
        and test_card.get("labels_opened") is False
        and test_card.get("threshold_selected_before_test") is True
        and float(test_card.get("selected_validation_threshold", math.nan)) == selected
        and isinstance(identities, Mapping)
        and identities.get("source_config_sha256") == artifacts["config"]["sha256"]
        and identities.get("checkpoint_sha256") == artifacts["checkpoint"]["sha256"]
        and identities.get("selector_receipt_sha256") == artifacts["selector_receipt"]["sha256"]
        and identities.get("amendment_sha256") == plan["data_inputs"]["threshold_amendment"]["sha256"],
        f"{record['method']}/seed{record['seed']}: test probability card lineage changed",
    )
    return {
        "schema_version": 1,
        "kind": BOUND_CARD_KIND,
        "method": record["method"],
        "seed": record["seed"],
        "selected_validation_threshold": selected,
        "selection_frozen_before_test_inference": True,
        "test_labels_opened": False,
        "identities": {
            "source_config_sha256": artifacts["config"]["sha256"],
            "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
            "selector_receipt_sha256": artifacts["selector_receipt"]["sha256"],
            "raw_validation_probability_card_sha256": artifacts["raw_validation_probability_card"]["sha256"],
            "validation_probability_card_sha256": artifacts["validation_probability_card"]["sha256"],
            "test_probability_card_sha256": sha256_file(test_card_path),
            "amendment_sha256": plan["data_inputs"]["threshold_amendment"]["sha256"],
            "manifest_sha256": plan["data_inputs"]["manifest"]["sha256"],
            "split_sha256": plan["data_inputs"]["splits"]["sha256"],
        },
    }


def _directory_hashes(root: Path, *, exclude: Sequence[str] = ()) -> Dict[str, str]:
    ignored = set(exclude)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in ignored
    }


def _verify_seed_delivery(path: Path, root: Path) -> Dict[str, Any]:
    delivery_path = _under(path, root, "incremental seed delivery")
    payload = _json(delivery_path, "incremental seed delivery")
    require(payload.get("schema_version") == 1 and payload.get("kind") == SEED_DELIVERY_KIND,
            "incremental seed delivery schema/kind changed")
    bundle = delivery_path.parent
    method, seed = str(payload.get("method", "")), payload.get("seed")
    require(
        method in NATIVE_METHODS
        and type(seed) is int
        and seed in REQUIRED_SEEDS
        and payload.get("automatic_heldout_rescore_permitted") is False,
        "incremental seed delivery identity/rescore policy changed",
    )
    outputs = payload.get("outputs")
    require(isinstance(outputs, Mapping), "incremental seed delivery lacks outputs")
    observed = _directory_hashes(bundle, exclude=("delivery.json",))
    require(dict(outputs) == observed, "incremental seed delivery output bytes changed")
    receipt = bundle / "scoring_receipt.json"
    require(
        receipt.is_file()
        and sha256_file(receipt) == payload.get("scoring_receipt_sha256"),
        "incremental seed scoring receipt bytes changed",
    )
    journal = bundle.parent / f".{bundle.name}.score-once-journal.json"
    require(
        journal.is_file() and sha256_file(journal) == payload.get("journal_sha256"),
        "incremental seed score-once journal bytes changed",
    )
    plan = payload.get("plan")
    require(isinstance(plan, Mapping), "incremental seed delivery lacks its frozen plan")
    _identity(plan, root, "incremental seed frozen plan")
    return payload

def finalize_seed(
    *, plan_path: Path, expected_plan_sha256: str, restricted_root: Path,
    method: str, seed: int, bundle_out: Path, repo_root: Path,
    expected_git_commit: str,
) -> Dict[str, Any]:
    """Score one planned seed exactly once and publish its immutable delivery."""
    root = restricted_root.resolve()
    plan_file = _under(plan_path, root, "incremental inference plan")
    require(sha256_file(plan_file) == _sha(expected_plan_sha256, "incremental plan digest"),
            "incremental inference plan bytes changed")
    plan = _json(plan_file, "incremental inference plan")
    require(plan.get("schema_version") == 1
            and plan.get("kind") in {PLAN_KIND, ADDITIONAL_PLAN_KIND},
            "incremental inference plan schema/kind changed")
    matches = [row for row in plan.get("runs", [])
               if isinstance(row, Mapping) and row.get("method") == method and row.get("seed") == seed]
    require(len(matches) == 1, "method/seed is not uniquely planned")
    record = matches[0]
    planned_bundle = Path(str(record["outputs"].get("seed_delivery_out", ""))).resolve()
    bundle = _output_path(bundle_out, root, "incremental seed bundle")
    require(bundle == planned_bundle, "incremental seed bundle differs from the frozen plan")
    commit = _commit(expected_git_commit, "incremental finalizer commit")
    require(
        commit == plan.get("expected_scoring_git_commit"),
        "scoring commit differs from the frozen incremental plan",
    )
    delivery_path = bundle / "delivery.json"
    if delivery_path.exists():
        return _verify_seed_delivery(delivery_path, root)
    require(not bundle.exists(), "partial incremental seed bundle exists; automatic rescore forbidden")
    journal = bundle.parent / f".{bundle.name}.score-once-journal.json"
    require(not journal.exists(), "incremental seed scoring already started; automatic rescore forbidden")
    code_root = Path(str(plan.get("code_root", ""))).resolve()
    _validate_code_root(repo_root, code_root, expected_git_commit)
    outputs = record["outputs"]
    unbiased_path = _under(Path(str(outputs["card_out"])), root, "unbiased inference card")
    unbiased = _json(unbiased_path, "unbiased inference card")
    require(
        unbiased.get("schema_version") == 1
        and unbiased.get("kind") == UNBIASED_CARD_KIND
        and unbiased.get("method") == method
        and unbiased.get("seed") == seed
        and unbiased.get("split") == "test"
        and unbiased.get("debug_limited") is False,
        "unbiased inference card identity changed",
    )
    full_timed_path = _under(Path(str(outputs["work_dir"])) / "full_timed_pass.json",
                             root, "full timed test probability card")
    test_card = _json(full_timed_path, "full timed test probability card")
    ids = unbiased.get("identities")
    require(isinstance(ids, Mapping)
            and ids.get("full_timed_pass_card_sha256") == sha256_file(full_timed_path)
            and ids.get("checkpoint_sha256") == record["artifacts"]["checkpoint"]["sha256"],
            "unbiased inference card does not bind the timed probability card")
    bound = _bound_inference_card(record, plan, test_card, full_timed_path)
    bundle.mkdir(parents=True)
    bound_path = bundle / "threshold_bound_inference_card.json"
    _write_json_once(bound_path, bound)
    results = bundle / "results"
    results.mkdir()
    receipt = bundle / "scoring_receipt.json"
    scorer = code_root / "scripts" / "score_threshold_run.py"
    evaluator = code_root / "scripts" / "evaluate_masks.py"
    require(scorer.is_file() and evaluator.is_file(), "shared scorer/evaluator is absent")
    run_card = _json(Path(record["artifacts"]["run_card"]["path"]), "training run card")
    params = int(run_card.get("parameter_count", {}).get("total", 0))
    require(params > 0, "training run card lacks parameter count")
    metadata = METHOD_METADATA[method]
    command = [
        str(Path(str(plan["python_executable"])).resolve()), str(scorer),
        "--repo-root", str(repo_root), "--expected-git-commit", commit,
        "--expected-wrapper-sha256", sha256_file(scorer),
        "--expected-evaluator-sha256", sha256_file(evaluator),
        "--restricted-root", str(root),
        "--manifest", plan["data_inputs"]["manifest"]["path"],
        "--splits", plan["data_inputs"]["splits"]["path"],
        "--config", record["artifacts"]["config"]["path"],
        "--test-probability-card", str(full_timed_path),
        "--selector-receipt", record["artifacts"]["selector_receipt"]["path"],
        "--threshold-amendment", plan["data_inputs"]["threshold_amendment"]["path"],
        "--inference-card", str(bound_path),
        "--prediction-dir", outputs["masks_out"],
        "--probability-dir", outputs["probabilities_out"],
        "--results-dir", str(results), "--receipt-out", str(receipt),
        "--method", method, "--seed", str(seed),
        "--display-name", metadata["display_name"], "--year", metadata["year"],
        "--family", metadata["family"], "--training-data", "SD302-Latent train split",
        "--comment", "LR and threshold selected on validation only",
        "--params-m", str(params / 1_000_000.0),
    ]
    journal_payload = {
        "schema_version": 1,
        "kind": "stage3_incremental_seed_scoring_journal_v1",
        "automatic_rescore_permitted": False,
        "plan": identity(plan_file),
        "method": method,
        "seed": seed,
        "inference_card": identity(unbiased_path),
        "test_probability_card": identity(full_timed_path),
        "threshold_bound_inference_card": identity(bound_path),
        "command_sha256": canonical_sha256(command),
    }
    _write_json_once(journal, journal_payload)
    completed = subprocess.run(command, cwd=repo_root, check=False)
    require(completed.returncode == 0, f"shared scorer exited {completed.returncode}; automatic retry forbidden")
    scoring = _json(receipt, "incremental scoring receipt")
    require(
        scoring.get("schema_version") == 1
        and scoring.get("kind") == SCORING_RECEIPT_KIND
        and scoring.get("method") == method
        and scoring.get("seed") == seed
        and scoring.get("prediction_count") == plan["test_universe"]["prediction_count"]
        and float(scoring.get("selected_threshold", math.nan))
        == float(record["selected_threshold"])
        and len(scoring.get("invocations", [])) == 2,
        "incremental scoring receipt is incomplete",
    )
    _receipt_files(scoring, root)
    scored_summary = _identity(
        scoring.get("scientific_summary"), root,
        f"{method}/seed{seed} scientific summary",
    )
    _summary_rows(Path(scored_summary["path"]), method, seed)
    threshold_record: Dict[str, Any] = {
        "method": method,
        "seed": seed,
        "training_lineage": record["training_lineage"],
        "config_path": record["artifacts"]["config"]["path"],
        "run_card_path": record["artifacts"]["run_card"]["path"],
        "checkpoint_path": record["artifacts"]["checkpoint"]["path"],
        "selector_receipt_path": record["artifacts"]["selector_receipt"]["path"],
        "inference_card_path": str(bound_path),
        "raw_validation_probability_card_path": record["artifacts"]["raw_validation_probability_card"]["path"],
        "validation_probability_card_path": record["artifacts"]["validation_probability_card"]["path"],
        "test_probability_card_path": str(full_timed_path),
        "scoring_receipt_path": str(receipt),
    }
    if record["training_lineage"] == "legacy_completed_before_amendment":
        threshold_record["legacy_completion_evidence_path"] = record["legacy_completion_evidence"]["path"]
    else:
        threshold_record["per_epoch_training_profile_card_path"] = record["per_epoch_training_profile_card"]["path"]
        threshold_record["preprocess_cache_receipt_path"] = record["preprocess_cache_receipt"]["path"]
        if "acceleration_receipt" in record:
            threshold_record["acceleration_receipt_path"] = record["acceleration_receipt"]["path"]
    payload = {
        "schema_version": 1,
        "kind": SEED_DELIVERY_KIND,
        "method": method,
        "seed": seed,
        "automatic_heldout_rescore_permitted": False,
        "plan": identity(plan_file),
        "training": {
            "config_sha256": record["artifacts"]["config"]["sha256"],
            "run_card_sha256": record["artifacts"]["run_card"]["sha256"],
            "checkpoint_sha256": record["artifacts"]["checkpoint"]["sha256"],
        },
        "test_universe": plan["test_universe"],
        "threshold_record": threshold_record,
        "scoring_receipt_sha256": sha256_file(receipt),
        "journal_sha256": sha256_file(journal),
    }
    payload["outputs"] = _directory_hashes(bundle, exclude=("delivery.json",))
    _write_json_once(delivery_path, payload)
    return _verify_seed_delivery(delivery_path, root)


def _receipt_files(receipt: Mapping[str, Any], root: Path) -> list[Path]:
    invocations = receipt.get("invocations")
    require(isinstance(invocations, list) and len(invocations) == 2,
            "scoring receipt does not contain exactly A and A-prime")
    tables = {str(item.get("table", "")) for item in invocations if isinstance(item, Mapping)}
    require(tables == {"A", "A_prime"}, "scoring receipt table universe changed")
    files = []
    for item in invocations:
        per = _identity(item.get("per_image"), root, "per-image evaluator CSV")
        files.append(Path(per["path"]))
    return files


def _summary_rows(path: Path, method: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str, "subject_id": str})
    required = {"method", "seed", "table", "group_value", "n_missing"}
    require(required.issubset(frame.columns) and len(frame) > 0,
            f"{method}/seed{seed}: summary schema is incomplete")
    require(set(frame.method.astype(str)) == {method}
            and set(pd.to_numeric(frame.seed, errors="raise").astype(int)) == {seed}
            and set(frame.table.astype(str)) == {"A", "A_prime"}
            and (pd.to_numeric(frame.n_missing, errors="raise") == 0).all(),
            f"{method}/seed{seed}: summary scientific identity changed")
    return frame


def import_deliveries(
    *, spec_path: Path, expected_spec_sha256: str, restricted_root: Path,
    bundle_out: Path,
) -> Dict[str, Any]:
    """Import one-time seed deliveries into the two-seed-gated Stage-3 bundle."""
    root = restricted_root.resolve()
    spec_file = _under(spec_path, root, "incremental import specification")
    require(sha256_file(spec_file) == _sha(expected_spec_sha256, "import spec digest"),
            "incremental import specification bytes changed")
    spec = _json(spec_file, "incremental import specification")
    require(spec.get("schema_version") == 1 and spec.get("kind") == IMPORT_SPEC_KIND,
            "incremental import specification schema/kind changed")
    manifest = _identity(spec.get("manifest"), root, "manifest")
    splits = _identity(spec.get("splits"), root, "subject split")
    gate_id = _identity(spec.get("readiness_gate"), root, "multi-seed readiness gate")
    gate = _json(Path(gate_id["path"]), "multi-seed readiness gate")
    require(
        gate.get("schema_version") == 1
        and gate.get("kind") == READINESS_GATE_KIND
        and gate.get("status") in {"PASS", "PASS_WITH_WARNINGS"}
        and gate.get("minimum_valid_seeds_per_method") == 2,
        "G4 readiness gate is not a passing two-seed gate",
    )
    identities = gate.get("identities")
    require(
        isinstance(identities, Mapping)
        and identities.get("manifest_sha256") == manifest["sha256"]
        and identities.get("splits_sha256") == splits["sha256"],
        "G4 readiness gate data identities differ from the import specification",
    )
    gate_methods = gate.get("methods")
    require(isinstance(gate_methods, Mapping) and set(gate_methods) == set(NATIVE_METHODS),
            "G4 readiness method universe changed")
    successful: Dict[str, list[int]] = {}
    for method in NATIVE_METHODS:
        record = gate_methods[method]
        valid = sorted(int(seed) for seed in record.get("valid_seeds", []))
        failed = sorted(int(seed) for seed in record.get("failed_seeds", []))
        missing = sorted(int(seed) for seed in record.get("missing_seeds", []))
        require(len(valid) >= 2 and 0 in valid
                and set(valid) | set(failed) | set(missing) == set(REQUIRED_SEEDS)
                and not (set(valid) & set(failed) or set(valid) & set(missing) or set(failed) & set(missing)),
                f"{method}: readiness seed partition does not satisfy incremental G3/G4")
        successful[method] = valid
    stage2_id = _identity(spec.get("stage2_unet_seed0_delivery"), root,
                          "Stage 2 U-Net seed-0 reselection delivery")
    stage2 = _json(Path(stage2_id["path"]), "Stage 2 U-Net seed-0 reselection delivery")
    require(stage2.get("schema_version") == 1
            and stage2.get("kind") == "stage2_threshold_reselection_delivery",
            "Stage 2 U-Net reselection delivery kind changed")
    stage2_outputs = stage2.get("outputs")
    require(isinstance(stage2_outputs, Mapping) and "results/summary.csv" in stage2_outputs,
            "Stage 2 U-Net delivery lacks replacement summary")
    stage2_summary = Path(stage2_id["path"]).parent / "results" / "summary.csv"
    require(stage2_summary.is_file() and sha256_file(stage2_summary) == stage2_outputs["results/summary.csv"],
            "Stage 2 U-Net summary bytes changed")
    summaries = [_summary_rows(stage2_summary, "unet", 0)]
    stage2_records = stage2.get("threshold_policy_attestation", {}).get("records")
    require(isinstance(stage2_records, list) and len(stage2_records) == 1,
            "Stage 2 U-Net delivery lacks one threshold record")
    stage2_receipt_identity = _identity(
        stage2_records[0].get("scoring_receipt"), root,
        "Stage 2 U-Net scoring receipt",
    )
    stage2_receipt_path = Path(stage2_receipt_identity["path"])
    stage2_receipt = _json(stage2_receipt_path,
                           "Stage 2 U-Net scoring receipt")
    require(
        stage2_receipt.get("kind") == SCORING_RECEIPT_KIND
        and stage2_receipt.get("method") == "unet"
        and stage2_receipt.get("seed") == 0,
        "Stage 2 U-Net scoring receipt identity changed",
    )
    stage2_gate_seed = gate_methods["unet"].get("seeds", {}).get("0")
    require(isinstance(stage2_gate_seed, Mapping) and stage2_gate_seed.get("status") == "valid",
            "G4 readiness gate does not include U-Net seed 0")
    stage2_threshold_record = stage2_records[0]
    for name, key in (
        ("config", "config_sha256"),
        ("run_card", "run_card_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    ):
        artifact = stage2_threshold_record.get(name)
        require(isinstance(artifact, Mapping)
                and artifact.get("sha256") == stage2_gate_seed.get(key),
                f"Stage 2 U-Net {name} differs from the G4 gate")
    source_per_image = _receipt_files(stage2_receipt, root)

    raw_deliveries = spec.get("seed_deliveries")
    require(isinstance(raw_deliveries, list), "incremental import lacks seed deliveries")
    deliveries: Dict[tuple[str, int], tuple[Dict[str, Any], Path]] = {}
    for raw in raw_deliveries:
        item = _identity(raw, root, "incremental seed delivery")
        payload = _verify_seed_delivery(Path(item["path"]), root)
        key = (str(payload.get("method", "")), int(payload.get("seed", -1)))
        require(key not in deliveries and key[0] in NATIVE_METHODS and key != ("unet", 0),
                "duplicate or unexpected incremental seed delivery")
        deliveries[key] = (payload, Path(item["path"]))
    expected = {(method, seed) for method, seeds in successful.items() for seed in seeds
                if (method, seed) != ("unet", 0)}
    require(set(deliveries) == expected,
            "incremental delivery universe does not exactly match the passing G4 gate")
    scored_records = []
    for key in sorted(expected):
        payload, delivery_path = deliveries[key]
        method, seed = key
        gate_seed = gate_methods[method].get("seeds", {}).get(str(seed))
        require(isinstance(gate_seed, Mapping) and gate_seed.get("status") == "valid",
                f"{method}/seed{seed}: gate seed evidence absent")
        training = payload.get("training")
        require(isinstance(training, Mapping)
                and training.get("config_sha256") == gate_seed.get("config_sha256")
                and training.get("run_card_sha256") == gate_seed.get("run_card_sha256")
                and training.get("checkpoint_sha256") == gate_seed.get("checkpoint_sha256"),
                f"{method}/seed{seed}: incremental delivery changed frozen training lineage")
        require(
            payload.get("test_universe") == gate.get("test_universe"),
            f"{method}/seed{seed}: incremental delivery test universe differs from the G4 gate",
        )
        receipt_path = delivery_path.parent / "scoring_receipt.json"
        require(receipt_path.is_file() and sha256_file(receipt_path) == payload.get("scoring_receipt_sha256"),
                f"{method}/seed{seed}: scoring receipt bytes changed")
        receipt = _json(receipt_path, f"{method}/seed{seed} scoring receipt")
        require(
            receipt.get("schema_version") == 1
            and receipt.get("kind") == SCORING_RECEIPT_KIND
            and receipt.get("method") == method
            and receipt.get("seed") == seed,
            f"{method}/seed{seed}: scoring receipt identity changed",
        )
        summary_identity = _identity(
            receipt.get("scientific_summary"), root,
            f"{method}/seed{seed} scientific summary",
        )
        summary_path = Path(summary_identity["path"])
        require(summary_path.is_file()
                and sha256_file(summary_path) == summary_identity["sha256"],
                f"{method}/seed{seed}: scientific summary bytes changed")
        summaries.append(_summary_rows(summary_path, method, seed))
        source_per_image.extend(_receipt_files(receipt, root))
        scored_records.append({
            "method": method,
            "seed": seed,
            "config_sha256": training["config_sha256"],
            "run_card_sha256": training["run_card_sha256"],
            "checkpoint_sha256": training["checkpoint_sha256"],
            "incremental_delivery_sha256": sha256_file(delivery_path),
            "scoring_receipt_sha256": sha256_file(receipt_path),
            "imported_not_rescored": True,
        })
    out = _output_path(bundle_out, root, "combined Stage 3 delivery")
    if (out / "delivery.json").exists():
        payload = _json(out / "delivery.json", "combined Stage 3 delivery")
        require(payload.get("kind") == FINAL_DELIVERY_KIND
                and payload.get("outputs") == _directory_hashes(out, exclude=("delivery.json",)),
                "combined Stage 3 delivery bytes changed")
        return payload
    require(not out.exists(), "partial combined Stage 3 delivery exists")
    staging = out.parent / f".{out.name}.incremental-import-{os.getpid()}"
    require(not staging.exists(), "incremental import staging directory already exists")
    results = staging / "results"
    results.mkdir(parents=True)
    combined = pd.concat(summaries, ignore_index=True)
    keys = ["method", "seed", "table", "group_value", "run_id", "training_data"]
    require(not combined.duplicated(keys).any(), "combined Stage 3 summary duplicates scientific rows")
    combined = combined.sort_values(keys, kind="stable")
    combined.to_csv(results / "summary.csv", index=False)
    observed_names: set[str] = set()
    for source in source_per_image:
        require(source.name not in observed_names, "combined Stage 3 per-image filename collision")
        observed_names.add(source.name)
        shutil.copyfile(source, results / source.name)
        require(sha256_file(source) == sha256_file(results / source.name),
                "per-image import changed bytes")
    gate_evidence = {
        "status": "PASS",
        "minimum_successful_seeds_per_method": 2,
        "methods": {
            method: {
                "successful_seeds": successful[method],
                "failed_seeds": sorted(int(seed) for seed in gate_methods[method].get("failed_seeds", [])),
                "missing_seeds": sorted(int(seed) for seed in gate_methods[method].get("missing_seeds", [])),
                "pass": len(successful[method]) >= 2,
            }
            for method in NATIVE_METHODS
        },
    }
    payload = {
        "schema_version": 1,
        "kind": FINAL_DELIVERY_KIND,
        "manifest_sha256": manifest["sha256"],
        "splits_sha256": splits["sha256"],
        "ordered_source_images_sha256": gate.get("test_universe", {}).get("ordered_source_images_sha256"),
        "automatic_heldout_rescore_permitted": False,
        "readiness_gate_sha256": gate_id["sha256"],
        "stage2_unet_seed0": {
            "reused_reselected_not_rescored": True,
            "threshold_reselection_delivery_sha256": stage2_id["sha256"],
        },
        "newly_scored_seeds": scored_records,
        "evaluator_invocations": 2 * len(scored_records),
        "gate_evidence": gate_evidence,
        "incremental_import": {
            "seed_delivery_count": len(scored_records),
            "all_imported_not_rescored": True,
            "seed_delivery_set_sha256": canonical_sha256(
                sorted(record["incremental_delivery_sha256"] for record in scored_records)
            ),
            "seed_deliveries": [
                {
                    "method": method,
                    "seed": seed,
                    "sha256": sha256_file(deliveries[(method, seed)][1]),
                }
                for method, seed in sorted(expected)
            ],
        },
    }
    payload["outputs"] = _directory_hashes(staging, exclude=("delivery.json",))
    _write_json_once(staging / "delivery.json", payload)
    os.replace(staging, out)
    return payload
