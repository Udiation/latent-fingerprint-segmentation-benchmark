#!/usr/bin/env python3
"""Fail-closed, one-time held-out scoring for the Stage 3 native models.

The validation-only gate and inference plan are frozen before inference.  This
program verifies those artifacts, the completed guarded inference queue, every
run/checkpoint/config/card/prediction, and the source-GT bytes before it creates
the write-once scoring journal.  Once that journal exists, an unpublished
failure can never be scored automatically again: the staging directory is kept
for audit and a new invocation refuses to run.

Metrics are produced exclusively by two invocations of ``evaluate_masks.py``
per newly scored seed (ROI Table A and quality-map Table A-prime).  U-Net seed
0 is copied from, and cryptographically verified against, the completed Stage
2 delivery; it is never scored again here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

import _path  # noqa: F401
from segbench.artifact_identity import (
    canonical_sha256,
    prediction_set_identity,
    require_clean_process_start,
    sha256_file,
)
from segbench.io import (
    attach_split,
    load_manifest,
    match_shape,
    read_mask,
    to_target_ppi,
)
from segbench.metrics import METRIC_KEYS, confusion, metrics_from_confusion, summarise
import infer_masks
import make_tables


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
REQUIRED_SEEDS = (0, 1, 2)
TABLES = {"A": "mask_path", "A_prime": "quality_mask_path"}
METHOD_METADATA: Mapping[str, Mapping[str, str]] = {
    "unet": {
        "display_name": "U-Net",
        "year": "2022",
        "family": "Encoder-decoder",
        "comment": "LR and threshold selected on validation only",
    },
    "unetpp": {
        "display_name": "UNet++",
        "year": "2020",
        "family": "Encoder-decoder",
        "comment": "Deep supervision; LR and threshold selected on validation only",
    },
    "drunet": {
        "display_name": "DRUnet",
        "year": "",
        "family": "Encoder-decoder",
        "comment": "Recurrent blocks with dropout; deterministic inference",
    },
    "ra_runet": {
        "display_name": "RUnet (no alignment)",
        "year": "2021",
        "family": "Encoder-decoder",
        "comment": "Dataset 1 has no target-domain alignment",
    },
    "lenet": {
        "display_name": "LeNet",
        "year": "2019",
        "family": "Patch CNN",
        "comment": "Block-based, 16x16 output blocks",
    },
    "alexnet": {
        "display_name": "AlexNet",
        "year": "2019",
        "family": "Patch CNN",
        "comment": "Block-based, 16x16 output blocks",
    },
    "cnn_ae": {
        "display_name": "CNN + stacked AE",
        "year": "2023",
        "family": "Patch CNN",
        "comment": "Train-split-only autoencoder pretraining",
    },
    "segformer_sub": {
        "display_name": "SegFormer-B2 (substitute†)",
        "year": "2021",
        "family": "Transformer",
        "comment": "Clearly labelled 2-D substitute for the OCT method",
    },
}


class PostScoreFailure(RuntimeError):
    """A failure after the held-out scoring journal was published."""


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} is not a JSON object")
    return value


def _sha(value: object, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{description} is not a lowercase SHA-256")
    return value


def _commit(value: object, description: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{description} is not an immutable Git SHA")
    return value


def _identity(record: object, description: str) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{description} identity is not an object")
    path = Path(str(record.get("path", "")))
    expected = _sha(record.get("sha256"), f"{description} digest")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{description} changed: {path}")
    return path, expected


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a durable JSON artifact exactly once (no identical overwrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _lock_path(bundle: Path) -> Path:
    return bundle.parent / f".{bundle.name}.stage3-native.lock"


def _journal_path(bundle: Path) -> Path:
    return bundle.parent / f".{bundle.name}.stage3-native.journal.json"


def _failure_path(bundle: Path) -> Path:
    return bundle.parent / f".{bundle.name}.stage3-native.failure.json"


def _require_under(path: Path, root: Path, description: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{description} escapes the restricted root") from exc


def _require_disjoint(left: Path, right: Path, description: str) -> None:
    left_resolved, right_resolved = left.resolve(), right.resolve()
    for candidate, parent in (
        (left_resolved, right_resolved),
        (right_resolved, left_resolved),
    ):
        try:
            candidate.relative_to(parent)
        except ValueError:
            continue
        raise ValueError(f"{description} roots overlap")


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)


def _ordered_source_gt(manifest_path: Path) -> str:
    """Rehash every ROI/quality source mask and reproduce audit digest v1."""
    frame = load_manifest(manifest_path)
    required = {
        "image_id",
        "mask_path",
        "quality_mask_path",
        "roi_union_sha256",
        "quality_union_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest lacks GT identity columns: {missing}")
    digest = hashlib.sha256(b"segbench-source-union-gt-v1\n")
    for row in frame.sort_values("image_id").to_dict("records"):
        image_id = str(row["image_id"])
        roi_expected = _sha(row["roi_union_sha256"], f"{image_id} ROI digest")
        if sha256_file(Path(str(row["mask_path"]))) != roi_expected:
            raise ValueError(f"{image_id}: ROI GT bytes changed")
        quality_path = "" if pd.isna(row["quality_mask_path"]) else str(row["quality_mask_path"])
        quality_expected = (
            "" if pd.isna(row["quality_union_sha256"]) else str(row["quality_union_sha256"])
        )
        if quality_path:
            quality_expected = _sha(quality_expected, f"{image_id} quality digest")
            if sha256_file(Path(quality_path)) != quality_expected:
                raise ValueError(f"{image_id}: quality GT bytes changed")
        elif quality_expected:
            raise ValueError(f"{image_id}: quality digest exists without a path")
        item = {
            "image_id": image_id,
            "quality_union_sha256": quality_expected,
            "roi_union_sha256": roi_expected,
        }
        digest.update(
            (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return digest.hexdigest()


def _test_universes(manifest: Path, splits: Path) -> Dict[str, pd.DataFrame]:
    split = _read_json(splits, "subject split").get("splits")
    if not isinstance(split, dict):
        raise ValueError("subject split lacks splits mapping")
    frame = attach_split(load_manifest(manifest), split)
    frame = frame[
        (frame.subset.astype(str) == "finger_latent")
        & (frame.split.astype(str) == "test")
        & (frame.examiner_value.astype(str) != "NONPRINT")
    ].copy()
    if frame.empty or frame.image_id.astype(str).duplicated().any():
        raise ValueError("held-out manifest universe is empty or duplicated")
    universes: Dict[str, pd.DataFrame] = {}
    for table, column in TABLES.items():
        paths = frame[column].fillna("").astype(str)
        part = frame[paths.str.len() > 0].copy()
        if part.empty or (part.examiner_value.fillna("").astype(str) == "").any():
            raise ValueError(f"{table}: empty universe or examiner group")
        universes[table] = part
    return universes


def _ordered_source_images(manifest: Path, splits: Path) -> str:
    """Hash every held-out source byte stream without decoding an image."""
    split = _read_json(splits, "subject split").get("splits")
    if not isinstance(split, dict):
        raise ValueError("subject split lacks splits mapping")
    frame = attach_split(load_manifest(manifest), split)
    frame = frame[
        (frame.subset.astype(str) == "finger_latent")
        & (frame.split.astype(str) == "test")
        & (frame.examiner_value.astype(str) != "NONPRINT")
    ].copy()
    if "source_sha256" not in frame:
        raise ValueError("manifest lacks held-out source_sha256 identities")
    records = []
    for row in frame.sort_values("image_id").to_dict("records"):
        expected = _sha(row.get("source_sha256"), f"{row['image_id']} source digest")
        path = Path(str(row["image_path"]))
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{row['image_id']}: held-out source image bytes changed")
        records.append(
            {"image_id": str(row["image_id"]), "source_sha256": expected}
        )
    return canonical_sha256(records)


def _validate_code_identities(plan: Mapping[str, Any]) -> Dict[str, str]:
    code = plan.get("code")
    if not isinstance(code, Mapping):
        raise ValueError("plan lacks scorer code identities")
    expected = {
        "planner": Path(__file__).with_name("prepare_stage3_inference.py"),
        "finalizer": Path(__file__),
        "evaluator": Path(__file__).with_name("evaluate_masks.py"),
        "metrics": Path(__file__).parents[1] / "segbench" / "metrics.py",
        "io": Path(__file__).parents[1] / "segbench" / "io.py",
        "summary": Path(__file__).parents[1] / "segbench" / "summary.py",
        "table_builder": Path(__file__).with_name("make_tables.py"),
    }
    if set(code) != set(expected):
        raise ValueError("plan code identity set changed")
    output = {}
    for name, actual_path in expected.items():
        path, digest = _identity(code[name], f"{name} code")
        # The inference planner runs from the pinned clean worktree, while this
        # finalizer is necessarily added by a later commit.  Permit the pinned
        # path only when its bytes are identical to the implementation invoked
        # below; path identity alone is neither necessary nor sufficient.
        if sha256_file(actual_path) != digest:
            raise ValueError(
                f"planned {name} bytes differ from the scoring implementation"
            )
        output[name] = digest
    return output


def _validate_inference_code_identities(plan: Mapping[str, Any]) -> Dict[str, str]:
    code = plan.get("inference_code")
    if not isinstance(code, Mapping):
        raise ValueError("plan lacks inference code identities")
    expected = {
        "wrapper": Path(__file__).with_name("infer_masks_repro.py"),
        "auxiliary_wrapper": Path(__file__).with_name("infer_drunet_mc_dropout.py"),
        "runner": Path(__file__).with_name("infer_masks.py"),
        "deep_utils": Path(__file__).parents[1] / "segbench" / "deep_utils.py",
        "deep_models": Path(__file__).parents[1] / "segbench" / "deep_models.py",
        "artifact_identity": Path(__file__).parents[1] / "segbench" / "artifact_identity.py",
    }
    if set(code) != set(expected):
        raise ValueError("plan inference-code identity set changed")
    output = {}
    for name, actual_path in expected.items():
        _, digest = _identity(code[name], f"{name} inference code")
        if sha256_file(actual_path) != digest:
            raise ValueError(f"planned {name} inference bytes differ from invoked code")
        output[name] = digest
    return output


def _validate_gate_and_plan(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    gate = _read_json(args.gate, "Stage 3 training-readiness gate")
    plan = _read_json(args.plan, "Stage 3 inference plan")
    prep = _read_json(args.prep_bundle, "frozen Stage 3 preparation bundle")
    expected_prep_sha = _sha(
        args.expected_prep_bundle_sha256,
        "externally registered preparation-bundle digest",
    )
    if sha256_file(args.prep_bundle) != expected_prep_sha:
        raise ValueError("Stage 3 preparation bundle differs from external registration")
    if prep.get("schema_version") != 1 or prep.get("kind") != "stage3_native_frozen_preparation_bundle":
        raise ValueError("Stage 3 preparation bundle schema or kind changed")
    if gate.get("schema_version") != 1 or gate.get("kind") != "stage3_native_training_readiness_gate":
        raise ValueError("Stage 3 gate schema or kind changed")
    if gate.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise ValueError("Stage 3 gate is not passing")
    if plan.get("schema_version") != 1 or plan.get("kind") != "stage3_native_inference_plan":
        raise ValueError("Stage 3 inference plan schema or kind changed")
    blind_claims = {
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
    }
    if any(gate.get(key) != value for key, value in blind_claims.items()):
        raise ValueError("Stage 3 gate is not ground-truth blind")
    if any(plan.get(key) != value for key, value in blind_claims.items()):
        raise ValueError("Stage 3 plan is not ground-truth blind")
    if plan.get("gate_sha256") != sha256_file(args.gate):
        raise ValueError("inference plan does not bind the training-readiness gate")
    if Path(str(plan.get("gate", ""))).resolve() != args.gate.resolve():
        raise ValueError("inference plan gate path changed")
    prep_expected = {
        "gate": {"path": str(args.gate), "sha256": sha256_file(args.gate)},
        "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
        "queue": {
            "path": str(args.inference_queue),
            "sha256": sha256_file(args.inference_queue),
        },
        "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
        "splits": {"path": str(args.splits), "sha256": sha256_file(args.splits)},
        "expected_planner_git_commit": args.expected_planner_git_commit,
        "expected_inference_git_commit": plan.get("expected_inference_git_commit"),
        "expected_finalizer_git_commit": args.expected_finalizer_git_commit,
    }
    for key, expected in prep_expected.items():
        if prep.get(key) != expected:
            raise ValueError(f"Stage 3 preparation bundle binding changed: {key}")
    _, planner_digest = _identity(
        prep.get("planner"), "preparation-bundle planner"
    )
    planned_planner = plan.get("code", {}).get("planner")
    if not isinstance(planned_planner, Mapping):
        raise ValueError("Stage 3 plan lacks its planner code identity")
    _, planned_planner_digest = _identity(planned_planner, "planned planner")
    if planner_digest != planned_planner_digest:
        raise ValueError("preparation and scoring surfaces bind different planner bytes")
    queue_payload = yaml.safe_load(args.inference_queue.read_text(encoding="utf-8"))
    runtime_policy = prep.get("runtime_policy")
    if not isinstance(queue_payload, Mapping) or runtime_policy != {
        "gpu_indices": queue_payload.get("required_gpu_indices"),
        "max_jobs_per_gpu": 1,
        "memory_fraction": 0.3,
        "max_project_ram_gib": prep.get("runtime_policy", {}).get(
            "max_project_ram_gib"
        ),
    }:
        raise ValueError("preparation bundle runtime policy differs from queue")
    ram_cap = runtime_policy.get("max_project_ram_gib")
    if not isinstance(ram_cap, (int, float)) or not float(ram_cap) > 0.0:
        raise ValueError("preparation bundle lacks a positive project RAM cap")
    require_clean_process_start(
        {"provenance": gate.get("planner_provenance")},
        _commit(args.expected_planner_git_commit, "planner commit"),
        description="Stage 3 inference planner",
    )
    identities = gate.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError("Stage 3 gate lacks identities")
    for key, path in (("manifest_sha256", args.manifest), ("splits_sha256", args.splits)):
        if identities.get(key) != sha256_file(path):
            raise ValueError(f"Stage 3 gate {key} differs")
    if int(plan.get("expected_prediction_count", -1)) <= 0:
        raise ValueError("Stage 3 plan has invalid expected prediction count")
    if gate.get("expected_inference_git_commit") != plan.get("expected_inference_git_commit"):
        raise ValueError("gate and plan inference commits differ")
    if gate.get("scorer_code") != plan.get("code") or prep.get("scorer_code") != plan.get("code"):
        raise ValueError("gate/plan/preparation scorer identities differ")
    if gate.get("inference_code") != plan.get("inference_code") or prep.get("inference_code") != plan.get("inference_code"):
        raise ValueError("gate/plan/preparation inference identities differ")
    if gate.get("reused") != plan.get("reused"):
        raise ValueError("gate and plan reused U-Net identity differ")
    return gate, plan


def _validate_queue(
    args: argparse.Namespace, plan: Mapping[str, Any]
) -> Dict[str, str]:
    queue_sha = sha256_file(args.inference_queue)
    payload = yaml.safe_load(args.inference_queue.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("launch_authorized") is not True:
        raise ValueError("inference queue is not launch-authorized")
    if payload.get("phase") != "stage3_native_test_inference":
        raise ValueError("inference queue phase changed")
    if (
        payload.get("required_max_jobs_per_gpu") != 1
        or payload.get("required_launcher_args") != ["--max-per-gpu", "1"]
    ):
        raise ValueError("inference queue lacks the registered timing-isolation policy")
    registered_gpus = payload.get("required_gpu_indices")
    if (
        not isinstance(registered_gpus, list)
        or not registered_gpus
        or any(type(value) is not int or value < 0 for value in registered_gpus)
        or len(registered_gpus) != len(set(registered_gpus))
    ):
        raise ValueError("inference queue lacks registered GPU indices")
    if payload.get("plan_sha256") != sha256_file(args.plan):
        raise ValueError("inference queue does not bind this plan")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("inference queue lacks jobs")
    runs = plan.get("runs")
    if not isinstance(runs, list):
        raise ValueError("inference plan lacks runs")
    expected_names = [f"infer.{run['method']}.seed{int(run['seed'])}" for run in runs]
    if [str(job.get("name")) for job in jobs] != expected_names:
        raise ValueError("inference queue jobs/order differ from plan")
    ledger = _read_json(args.inference_ledger, "completed inference queue ledger")
    if ledger.get("schema_version") != 2 or ledger.get("queue_sha256") != queue_sha:
        raise ValueError("inference ledger does not bind the queue")
    expected_attestation = {
        "schema_version": 1,
        "queue_phase": "stage3_native_test_inference",
        "queue_sha256": queue_sha,
        "gpus": registered_gpus,
        "max_per_gpu": 1,
        "memory_fraction": 0.3,
        "required_launcher_args": ["--max-per-gpu", "1"],
    }
    if ledger.get("launcher_attestation") != expected_attestation:
        raise ValueError("inference ledger lacks exact timing-isolation attestation")
    states = ledger.get("jobs")
    if not isinstance(states, dict) or set(states) != set(expected_names):
        raise ValueError("inference ledger job universe differs from plan")
    for job, run in zip(jobs, runs):
        name = str(job["name"])
        state = states[name]
        card = Path(str(run["card_out"]))
        recomputed_job_identity = canonical_sha256(
            {
                "name": job.get("name"),
                "argv": job.get("argv"),
                "env": job.get("env", {}),
                "completion_artifact": job.get("completion_artifact"),
                "resources": job.get("resources"),
            }
        )
        if job.get("job_identity_sha256") != recomputed_job_identity:
            raise ValueError(f"{name}: queue job identity does not reproduce")
        if not isinstance(state, Mapping) or state.get("status") != "completed":
            raise ValueError(f"{name}: inference did not complete")
        if state.get("identity_sha256") != job.get("job_identity_sha256"):
            raise ValueError(f"{name}: queue job identity changed")
        if run.get("job_identity_sha256") != job.get("job_identity_sha256"):
            raise ValueError(f"{name}: plan and queue job identities differ")
        if state.get("artifact_sha256") != sha256_file(card):
            raise ValueError(f"{name}: completion card differs from ledger")
        if Path(str(job.get("completion_artifact", ""))).resolve() != card.resolve():
            raise ValueError(f"{name}: queue completion artifact differs from plan")
    return {
        "queue_sha256": queue_sha,
        "ledger_sha256": sha256_file(args.inference_ledger),
        "launcher_attestation": expected_attestation,
    }


def _validate_auxiliary(
    args: argparse.Namespace,
    gate: Mapping[str, Any],
    deterministic_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Authenticate the isolated DRUnet uncertainty extra without scoring it."""
    for path, description in (
        (args.aux_plan, "DRUnet auxiliary plan"),
        (args.aux_queue, "DRUnet auxiliary queue"),
        (args.aux_ledger, "DRUnet auxiliary ledger"),
        (args.aux_card, "DRUnet auxiliary completion card"),
    ):
        _require_under(path, args.restricted_root, description)
    prep = _read_json(args.prep_bundle, "frozen Stage 3 preparation bundle")
    for name, path in (("aux_plan", args.aux_plan), ("aux_queue", args.aux_queue)):
        expected_path, expected_sha = _identity(prep.get(name), name)
        if expected_path.resolve() != path.resolve() or expected_sha != sha256_file(path):
            raise ValueError(f"preparation bundle {name} binding changed")
    aux_plan = _read_json(args.aux_plan, "DRUnet auxiliary plan")
    claims = {
        "schema_version": 1,
        "kind": "stage3_drunet_mc_dropout_plan",
        "ground_truth_blind": True,
        "test_inputs_hashed": True,
        "test_images_decoded": False,
        "test_labels_opened": False,
        "timing_excluded": True,
        "mask_outputs_forbidden": True,
        "mc_dropout_passes": 20,
        "selection_rule": "first_20_lexicographic_heldout_ids",
    }
    if any(aux_plan.get(key) != value for key, value in claims.items()):
        raise ValueError("DRUnet auxiliary plan claims changed")
    for key, path in (("gate", args.gate), ("deterministic_plan", args.plan)):
        identity_path, identity_sha = _identity(aux_plan.get(key), f"auxiliary {key}")
        if identity_path.resolve() != path.resolve() or identity_sha != sha256_file(path):
            raise ValueError(f"DRUnet auxiliary {key} binding changed")
    ids = aux_plan.get("test_image_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != 20
        or ids != sorted(ids)
        or len(set(ids)) != 20
        or any(not isinstance(value, str) or not value for value in ids)
        or aux_plan.get("image_id_set_sha256") != canonical_sha256(ids)
    ):
        raise ValueError("DRUnet auxiliary image universe changed")
    expected_shapes = aux_plan.get("expected_shapes")
    if (
        not isinstance(expected_shapes, Mapping)
        or set(expected_shapes) != set(ids)
        or any(
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
            for shape in expected_shapes.values()
        )
    ):
        raise ValueError("DRUnet auxiliary expected shapes changed")
    split = _read_json(args.splits, "subject split").get("splits")
    if not isinstance(split, dict):
        raise ValueError("subject split lacks splits mapping")
    frame = attach_split(load_manifest(args.manifest), split)
    selected = frame[
        (frame.subset.astype(str) == "finger_latent")
        & (frame.split.astype(str) == "test")
        & (frame.examiner_value.astype(str).str.upper() != "NONPRINT")
    ].copy()
    selected.index = selected.image_id.astype(str)
    frame = selected
    if set(ids) - set(frame.index):
        raise ValueError("DRUnet auxiliary IDs differ from held-out manifest")
    frozen_ids = sorted(map(str, frame.index))[:20]
    if ids != frozen_ids:
        raise ValueError("DRUnet auxiliary IDs differ from the frozen selection rule")
    source_records = []
    for image_id in ids:
        row = frame.loc[image_id]
        if isinstance(row, pd.DataFrame):
            raise ValueError("DRUnet auxiliary manifest contains duplicate IDs")
        expected = _sha(row.get("source_sha256"), f"{image_id} source digest")
        height, width = row.get("target_height"), row.get("target_width")
        try:
            height_value, width_value = float(height), float(width)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{image_id}: auxiliary target shape changed") from exc
        if (
            isinstance(height, (bool, np.bool_))
            or isinstance(width, (bool, np.bool_))
            or not np.isfinite([height_value, width_value]).all()
            or not height_value.is_integer()
            or not width_value.is_integer()
            or expected_shapes[image_id]
            != [int(height_value), int(width_value)]
        ):
            raise ValueError(f"{image_id}: auxiliary target shape changed")
        source = Path(str(row["image_path"]))
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"{image_id}: auxiliary source image changed")
        source_records.append({"image_id": image_id, "source_sha256": expected})
    source_digest = canonical_sha256(source_records)
    if aux_plan.get("ordered_source_images_sha256") != source_digest:
        raise ValueError("DRUnet auxiliary source-image digest changed")
    reserved_masks = Path(str(aux_plan.get("reserved_masks_root", "")))
    _require_under(reserved_masks, args.restricted_root, "reserved auxiliary masks root")
    if reserved_masks.exists() and any(path.is_file() for path in reserved_masks.rglob("*")):
        raise ValueError("DRUnet auxiliary reserved mask root is not empty")
    variance_root = Path(str(aux_plan.get("variance_out", "")))
    work_root = Path(str(aux_plan.get("work_dir", "")))
    planned_card = Path(str(aux_plan.get("card_out", "")))
    for path, description in (
        (variance_root, "DRUnet variance root"),
        (work_root, "DRUnet auxiliary work root"),
        (planned_card, "DRUnet auxiliary completion card"),
    ):
        _require_under(path, args.restricted_root, description)
    if planned_card.resolve() != args.aux_card.resolve():
        raise ValueError("DRUnet auxiliary plan completion-card path changed")
    _require_disjoint(
        reserved_masks,
        variance_root,
        "DRUnet auxiliary mask/variance",
    )
    _require_disjoint(
        reserved_masks,
        work_root,
        "DRUnet auxiliary mask/work",
    )
    _require_disjoint(
        variance_root,
        work_root,
        "DRUnet auxiliary variance/work",
    )
    run = aux_plan.get("run")
    if not isinstance(run, Mapping) or run.get("method") != "drunet":
        raise ValueError("DRUnet auxiliary run identity changed")
    seed = int(run.get("seed", -1))
    gate_seed = gate.get("methods", {}).get("drunet", {}).get("seeds", {}).get(str(seed))
    if not isinstance(gate_seed, Mapping) or gate_seed.get("status") != "valid":
        raise ValueError("DRUnet auxiliary seed is not gate-authorized")
    for field in (
        "method", "seed", "run_card", "run_card_sha256", "config",
        "config_sha256", "checkpoint", "checkpoint_sha256",
        "best_validation_dice", "selected_validation_threshold", "parameter_count",
        "training_git_commit",
    ):
        if run.get(field) != gate_seed.get(field):
            raise ValueError(f"DRUnet auxiliary run differs from gate field {field}")
    if run.get("expected_training_git_commit") != gate_seed.get("training_git_commit"):
        raise ValueError("DRUnet auxiliary training commit changed")
    if run.get("expected_inference_git_commit") != deterministic_plan.get(
        "expected_inference_git_commit"
    ):
        raise ValueError("DRUnet auxiliary inference commit changed")
    aux_identities = aux_plan.get("identities")
    if not isinstance(aux_identities, Mapping):
        raise ValueError("DRUnet auxiliary plan lacks identities")
    expected_identity_paths = {
        "config": Path(str(run.get("config", ""))),
        "checkpoint": Path(str(run.get("checkpoint", ""))),
        "run_card": Path(str(run.get("run_card", ""))),
        "manifest": args.manifest,
        "splits": args.splits,
    }
    for name, expected_path in expected_identity_paths.items():
        observed_path, observed_sha = _identity(
            aux_identities.get(name), f"DRUnet auxiliary {name}"
        )
        if (
            observed_path.resolve() != expected_path.resolve()
            or observed_sha != sha256_file(expected_path)
        ):
            raise ValueError(f"DRUnet auxiliary {name} identity changed")
    wrapper_path, wrapper_sha = _identity(
        aux_identities.get("wrapper"), "DRUnet auxiliary wrapper"
    )
    expected_wrapper = deterministic_plan.get("inference_code", {}).get(
        "auxiliary_wrapper"
    )
    planned_wrapper_path, planned_wrapper_sha = _identity(
        expected_wrapper, "deterministic-plan auxiliary wrapper"
    )
    if (
        wrapper_path.resolve() != planned_wrapper_path.resolve()
        or wrapper_sha != planned_wrapper_sha
    ):
        raise ValueError("DRUnet auxiliary wrapper identity changed")

    aux_queue = yaml.safe_load(args.aux_queue.read_text(encoding="utf-8"))
    if not isinstance(aux_queue, Mapping):
        raise ValueError("DRUnet auxiliary queue is not an object")
    for key, value in {
        "schema_version": 2,
        "phase": "stage3_drunet_mc_dropout_aux",
        "launch_authorized": True,
        "timing_excluded": True,
        "mask_outputs_forbidden": True,
        "mc_dropout_passes": 20,
        "required_max_jobs_per_gpu": 1,
        "required_launcher_args": ["--max-per-gpu", "1"],
    }.items():
        if aux_queue.get(key) != value:
            raise ValueError(f"DRUnet auxiliary queue field changed: {key}")
    if aux_queue.get("plan_sha256") != sha256_file(args.aux_plan):
        raise ValueError("DRUnet auxiliary queue does not bind its plan")
    gpus = aux_queue.get("required_gpu_indices")
    prep_gpus = prep.get("runtime_policy", {}).get("gpu_indices")
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(type(value) is not int or value < 0 for value in gpus)
        or len(gpus) != len(set(gpus))
        or gpus != prep_gpus
    ):
        raise ValueError("DRUnet auxiliary queue lacks registered GPU indices")
    jobs = aux_queue.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("DRUnet auxiliary queue must contain exactly one job")
    job = jobs[0]
    expected_job_identity = canonical_sha256(
        {
            "name": job.get("name"),
            "argv": job.get("argv"),
            "env": job.get("env", {}),
            "completion_artifact": job.get("completion_artifact"),
            "resources": job.get("resources"),
        }
    )
    if job.get("job_identity_sha256") != expected_job_identity:
        raise ValueError("DRUnet auxiliary job identity does not reproduce")
    if Path(str(job.get("completion_artifact", ""))).resolve() != args.aux_card.resolve():
        raise ValueError("DRUnet auxiliary completion-card path changed")
    argv = job.get("argv")
    expected_argv_tail = [
        str(wrapper_path),
        "--plan", str(args.aux_plan),
        "--expected-plan-sha256", sha256_file(args.aux_plan),
        "--config", str(expected_identity_paths["config"]),
        "--checkpoint", str(expected_identity_paths["checkpoint"]),
        "--manifest", str(args.manifest),
        "--splits", str(args.splits),
        "--variance-out", str(variance_root),
        "--card-out", str(args.aux_card),
        "--work-dir", str(work_root),
        "--expected-git-commit", str(run.get("expected_inference_git_commit")),
        "--expected-checkpoint-sha256", str(run.get("checkpoint_sha256")),
        "--seed", str(seed),
        "--device", "cuda",
        "--memory-fraction", "0.3",
    ]
    if (
        not isinstance(argv, list)
        or len(argv) < 2
        or not isinstance(argv[0], str)
        or argv[1:] != expected_argv_tail
    ):
        raise ValueError("DRUnet auxiliary queue command differs from its plan")
    ledger = _read_json(args.aux_ledger, "DRUnet auxiliary ledger")
    expected_attestation = {
        "schema_version": 1,
        "queue_phase": "stage3_drunet_mc_dropout_aux",
        "queue_sha256": sha256_file(args.aux_queue),
        "gpus": gpus,
        "max_per_gpu": 1,
        "memory_fraction": 0.3,
        "required_launcher_args": ["--max-per-gpu", "1"],
    }
    states = ledger.get("jobs")
    name = str(job.get("name"))
    if (
        ledger.get("schema_version") != 2
        or ledger.get("queue_sha256") != sha256_file(args.aux_queue)
        or ledger.get("launcher_attestation") != expected_attestation
        or not isinstance(states, Mapping)
        or set(states) != {name}
    ):
        raise ValueError("DRUnet auxiliary ledger binding changed")
    state = states[name]
    if (
        not isinstance(state, Mapping)
        or state.get("status") != "completed"
        or state.get("identity_sha256") != expected_job_identity
        or state.get("artifact_sha256") != sha256_file(args.aux_card)
    ):
        raise ValueError("DRUnet auxiliary job did not complete immutably")
    card = _read_json(args.aux_card, "DRUnet auxiliary completion card")
    require_clean_process_start(
        card,
        _commit(run.get("expected_inference_git_commit"), "aux inference commit"),
        description="DRUnet auxiliary inference",
    )
    if (
        card.get("schema_version") != 1
        or card.get("kind") != "stage3_drunet_mc_dropout_completion_card"
        or card.get("method") != "drunet"
        or int(card.get("seed", -1)) != seed
        or card.get("plan_sha256") != sha256_file(args.aux_plan)
        or card.get("ground_truth_opened") is not False
        or card.get("mask_outputs_written") is not False
        or card.get("timing_excluded") is not True
        or card.get("mc_dropout_passes") != 20
        or card.get("image_id_set_sha256") != canonical_sha256(ids)
        or card.get("ordered_source_images_sha256") != source_digest
    ):
        raise ValueError("DRUnet auxiliary completion card identity changed")
    for name in ("config", "checkpoint", "manifest", "splits"):
        expected_path = expected_identity_paths[name]
        if card.get("identities", {}).get(name) != {
            "path": str(expected_path),
            "sha256": sha256_file(expected_path),
        }:
            raise ValueError(f"DRUnet auxiliary card {name} identity changed")
    variance = card.get("variance_set")
    expected_paths = {
        image_id: variance_root / f"{image_id}_variance.npy" for image_id in ids
    }
    observed = sorted(path for path in variance_root.rglob("*") if path.is_file())
    if observed != sorted(expected_paths.values()):
        raise ValueError("DRUnet auxiliary variance file universe changed")
    records = []
    for image_id in sorted(ids):
        path = expected_paths[image_id]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.dtype != np.float32
            or list(array.shape) != expected_shapes[image_id]
            or not np.isfinite(array).all()
            or (array < 0).any()
        ):
            raise ValueError(f"invalid DRUnet variance artifact: {image_id}")
        records.append(
            {
                "image_id": image_id,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    reproduced = {"count": len(records), "records": records, "sha256": canonical_sha256(records)}
    if variance != reproduced:
        raise ValueError("DRUnet auxiliary variance-set identity changed")
    return {
        "plan_sha256": sha256_file(args.aux_plan),
        "queue_sha256": sha256_file(args.aux_queue),
        "ledger_sha256": sha256_file(args.aux_ledger),
        "completion_card_sha256": sha256_file(args.aux_card),
        "variance_set_sha256": reproduced["sha256"],
        "image_id_set_sha256": canonical_sha256(ids),
        "ordered_source_images_sha256": source_digest,
        "seed": seed,
        "mc_dropout_passes": 20,
        "timing_excluded": True,
        "scoring_excluded": True,
        "reserved_masks_empty": True,
    }


def _verify_prediction_card(
    run: Mapping[str, Any],
    args: argparse.Namespace,
    expected_count: int,
    ordered_source_images_sha256: str,
) -> Dict[str, Any]:
    method, seed = str(run.get("method")), int(run.get("seed", -1))
    if method not in NATIVE_METHODS or seed not in REQUIRED_SEEDS:
        raise ValueError(f"unregistered run identity {method}/seed{seed}")
    run_card_path = Path(str(run.get("run_card", "")))
    config_path = Path(str(run.get("config", "")))
    checkpoint_path = Path(str(run.get("checkpoint", "")))
    for key, path in (
        ("run_card_sha256", run_card_path),
        ("config_sha256", config_path),
        ("checkpoint_sha256", checkpoint_path),
    ):
        expected = _sha(run.get(key), f"{method}/seed{seed} {key}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{method}/seed{seed}: {key} mismatch")
    training_commit = _commit(run.get("expected_training_git_commit"), "training commit")
    inference_commit = _commit(run.get("expected_inference_git_commit"), "inference commit")
    run_card = _read_json(run_card_path, f"{method}/seed{seed} run card")
    require_clean_process_start(run_card, training_commit, description=f"{method}/seed{seed} training")
    if run_card.get("method") != method or run_card.get("seed") != seed or run_card.get("debug_limited") is not False:
        raise ValueError(f"{method}/seed{seed}: run card identity changed")
    if run_card.get("manifest_sha256") != sha256_file(args.manifest) or run_card.get("split_sha256") != sha256_file(args.splits):
        raise ValueError(f"{method}/seed{seed}: training data identity changed")
    if run_card.get("source_config_sha256") != sha256_file(config_path):
        raise ValueError(f"{method}/seed{seed}: source config identity changed")
    if run_card.get("checkpoint_sha256", {}).get("best.pt") != sha256_file(checkpoint_path):
        raise ValueError(f"{method}/seed{seed}: checkpoint differs from run card")
    card_path = Path(str(run.get("card_out", "")))
    card = _read_json(card_path, f"{method}/seed{seed} inference card")
    require_clean_process_start(card, inference_commit, description=f"{method}/seed{seed} inference")
    if (
        card.get("schema_version") != 1
        or card.get("kind") != "complete_unbiased_inference_card"
        or card.get("method") != method
        or card.get("subset") != "finger_latent"
        or card.get("split") != "test"
        or card.get("debug_limited") is not False
        or card.get("device") != "CUDA"
    ):
        raise ValueError(f"{method}/seed{seed}: inference card identity changed")
    expected_task = "patch" if method in {"lenet", "alexnet", "cnn_ae"} else "pixel"
    if card.get("task") != expected_task:
        raise ValueError(f"{method}/seed{seed}: inference task changed")
    threshold = float(run_card.get("selected_validation_threshold"))
    if float(card.get("selected_validation_threshold")) != threshold:
        raise ValueError(f"{method}/seed{seed}: inference threshold is not validation-selected")
    identities = card.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError(f"{method}/seed{seed}: inference identities absent")
    expected_identities = {
        "source_config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "authorized_checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_sha256": sha256_file(args.manifest),
        "split_sha256": sha256_file(args.splits),
        "ordered_source_images_sha256": ordered_source_images_sha256,
    }
    for key, expected in expected_identities.items():
        if identities.get(key) != expected:
            raise ValueError(f"{method}/seed{seed}: card {key} mismatch")
    prediction_set = card.get("prediction_set")
    if not isinstance(prediction_set, Mapping) or prediction_set.get("count") != expected_count:
        raise ValueError(f"{method}/seed{seed}: prediction count differs")
    if identities.get("prediction_count") != expected_count:
        raise ValueError(f"{method}/seed{seed}: card prediction count differs")
    if run.get("prediction_set_sha256") not in (None, prediction_set.get("sha256")):
        raise ValueError(f"{method}/seed{seed}: plan prediction-set digest differs")
    records = prediction_set.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"{method}/seed{seed}: prediction records incomplete")
    masks = Path(str(run.get("masks_out", "")))
    items = []
    for record in records:
        image_id = str(record.get("image_id", ""))
        path = infer_masks._safe_output(masks, image_id)
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{method}/seed{seed}: prediction changed: {image_id}")
        items.append((image_id, path))
    recomputed = prediction_set_identity(items)
    if recomputed != prediction_set or identities.get("prediction_set_sha256") != recomputed["sha256"]:
        raise ValueError(f"{method}/seed{seed}: prediction-set identity does not reproduce")
    expected_relative = {
        f"{str(record['image_id'])}.png" for record in records
    }
    # Reject every unregistered file recursively.  A nested stale mask,
    # uncertainty product, or sidecar would make the immutable prediction set
    # ambiguous even when the top-level PNG names still look correct.
    observed_relative = {
        path.relative_to(masks).as_posix()
        for path in masks.rglob("*")
        if path.is_file()
    }
    if observed_relative != expected_relative:
        raise ValueError(f"{method}/seed{seed}: prediction directory universe changed")
    timing = card.get("timing")
    if not isinstance(timing, Mapping) or timing.get("warmup_images") != 10:
        raise ValueError(f"{method}/seed{seed}: timing protocol changed")
    values = np.asarray([row.get("ms_per_img") for row in timing.get("per_image", [])], dtype=float)
    timing_ids = [str(row.get("image_id")) for row in timing.get("per_image", [])]
    prediction_ids = [str(record["image_id"]) for record in records]
    if len(timing_ids) != len(set(timing_ids)) or set(timing_ids) != set(prediction_ids):
        raise ValueError(f"{method}/seed{seed}: timing universe differs from predictions")
    if len(values) != expected_count or not np.isfinite(values).all() or not np.isclose(values.mean(), float(timing.get("mean_ms_per_img"))):
        raise ValueError(f"{method}/seed{seed}: timing evidence is incomplete")
    count = run_card.get("parameter_count")
    if not isinstance(count, Mapping) or int(count.get("total", -1)) < 1:
        raise ValueError(f"{method}/seed{seed}: parameter count absent")
    return {
        "method": method,
        "seed": seed,
        "run_card_sha256": sha256_file(run_card_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "inference_card_sha256": sha256_file(card_path),
        "prediction_set_sha256": recomputed["sha256"],
        "prediction_count": expected_count,
        "threshold": threshold,
        "params_m": int(count["total"]) / 1_000_000.0,
        "ms_per_img": float(timing["mean_ms_per_img"]),
        "training_git_commit": training_commit,
        "inference_git_commit": inference_commit,
        "run_id": sha256_file(checkpoint_path),
        "masks_out": str(masks),
    }


def _validate_plan_runs(
    plan: Mapping[str, Any], gate: Mapping[str, Any], args: argparse.Namespace,
    universes: Mapping[str, pd.DataFrame],
    ordered_source_images_sha256: str,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    runs = plan.get("runs")
    if not isinstance(runs, list):
        raise ValueError("inference plan lacks runs")
    expected_count = int(plan["expected_prediction_count"])
    if int(gate.get("expected_prediction_count", -1)) != expected_count:
        raise ValueError("gate and plan prediction counts differ")
    global_inference_commit = _commit(
        plan.get("expected_inference_git_commit"),
        "plan inference commit",
    )
    expected_ids = set(universes["A"].image_id.astype(str))
    if len(expected_ids) != expected_count:
        raise ValueError("plan prediction count differs from exact Table A universe")
    validated = [
        _verify_prediction_card(
            run, args, expected_count, ordered_source_images_sha256
        )
        for run in runs
    ]
    if any(
        item["inference_git_commit"] != global_inference_commit
        for item in validated
    ):
        raise ValueError("one run differs from the plan-wide inference commit")
    keys = [(item["method"], item["seed"]) for item in validated]
    if len(keys) != len(set(keys)) or ("unet", 0) in keys:
        raise ValueError("plan runs are duplicated or attempt to rescore U-Net seed 0")
    for run, evidence in zip(runs, validated):
        card = _read_json(Path(str(run["card_out"])), "inference card")
        ids = {str(record["image_id"]) for record in card["prediction_set"]["records"]}
        if ids != expected_ids:
            raise ValueError(f"{evidence['method']}/seed{evidence['seed']}: IDs differ from manifest")
    methods = gate.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(NATIVE_METHODS):
        raise ValueError("readiness gate method universe changed")
    for method in NATIVE_METHODS:
        record = methods[method]
        if not isinstance(record, Mapping):
            raise ValueError(f"{method}: readiness record is not an object")
        partitions = {
            "valid": {int(seed) for seed in record.get("valid_seeds", [])},
            "failed": {int(seed) for seed in record.get("failed_seeds", [])},
            "missing": {int(seed) for seed in record.get("missing_seeds", [])},
        }
        if (
            any(partitions[left] & partitions[right] for left, right in (
                ("valid", "failed"), ("valid", "missing"), ("failed", "missing")
            ))
            or set().union(*partitions.values()) != set(REQUIRED_SEEDS)
            or len(partitions["valid"]) < 2
        ):
            raise ValueError(f"{method}: readiness seed accounting is invalid")
        seed_records = record.get("seeds")
        if not isinstance(seed_records, Mapping) or set(seed_records) != {
            str(seed) for seed in REQUIRED_SEEDS
        }:
            raise ValueError(f"{method}: per-seed readiness records are incomplete")
        for status, seeds in partitions.items():
            for seed in seeds:
                expected_status = "valid" if status == "valid" else status
                if seed_records[str(seed)].get("status") != expected_status:
                    raise ValueError(
                        f"{method}/seed{seed}: readiness status disagrees with partition"
                    )
    # The preparation bundle freezes the plan after the readiness gate.  Every
    # planned scientific run must still be the *same* validation-authorized
    # gate seed, not a replacement checkpoint/card/config inserted later.
    for run in runs:
        method, seed = str(run.get("method")), int(run.get("seed", -1))
        gate_seed = methods.get(method, {}).get("seeds", {}).get(str(seed))
        if not isinstance(gate_seed, Mapping) or gate_seed.get("status") != "valid":
            raise ValueError(f"{method}/seed{seed}: plan is not gate-authorized")
        exact_fields = (
            "method",
            "seed",
            "run_card",
            "run_card_sha256",
            "config",
            "config_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "best_validation_dice",
            "selected_validation_threshold",
            "parameter_count",
        )
        for field in exact_fields:
            if run.get(field) != gate_seed.get(field):
                raise ValueError(
                    f"{method}/seed{seed}: plan differs from gate seed field {field}"
                )
        if (
            run.get("training_git_commit") != gate_seed.get("training_git_commit")
            or run.get("expected_training_git_commit")
            != gate_seed.get("training_git_commit")
        ):
            raise ValueError(f"{method}/seed{seed}: training commit differs from gate")
    reused = plan.get("reused")
    if not isinstance(reused, Mapping) or reused.get("method") != "unet" or reused.get("seed") != 0:
        raise ValueError("plan does not identify exactly reused U-Net seed 0")
    expected_keys = {
        (method, int(seed))
        for method in NATIVE_METHODS
        for seed in methods[method].get("valid_seeds", [])
    }
    if ("unet", 0) not in expected_keys:
        raise ValueError("readiness gate does not validate reused U-Net seed 0")
    if set(keys) != expected_keys - {("unet", 0)}:
        raise ValueError("inference plan run universe differs from gate-valid seeds")
    return validated, dict(reused)


def _aggregate_nan_expected(per: pd.DataFrame, metric: str, suffix: str) -> bool:
    if metric not in {"mdr", "fdr"}:
        return False
    if suffix == "_mean":
        return bool(pd.to_numeric(per[metric], errors="coerce").isna().all())
    denominator = per.tp + per.fn if metric == "mdr" else per.fp + per.tn
    return int(denominator.sum()) == 0


def _validate_per_image_metrics(per: pd.DataFrame, table: str) -> None:
    for column in ("tp", "fp", "fn", "tn"):
        numeric = pd.to_numeric(per[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{table}: invalid confusion counts")
    for index, row in per.iterrows():
        rebuilt = metrics_from_confusion(int(row.tp), int(row.fp), int(row.fn), int(row.tn))
        for metric in METRIC_KEYS:
            observed, expected = float(row[metric]), float(rebuilt[metric])
            if not (np.isnan(observed) and np.isnan(expected)) and not np.isclose(observed, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{table}: {metric} does not reconstruct at row {index}")


def _validate_seed_results(
    result_dir: Path,
    evidence: Mapping[str, Any],
    universes: Mapping[str, pd.DataFrame],
    *,
    prediction_dir: Path,
) -> Dict[str, Any]:
    summary_path = result_dir / "summary.csv"
    method, seed = str(evidence["method"]), int(evidence["seed"])
    full_summary = pd.read_csv(summary_path)
    seed_values = pd.to_numeric(full_summary.get("seed"), errors="coerce")
    summary = full_summary[
        (full_summary.method.astype(str) == method)
        & (seed_values == seed)
        & (full_summary.run_id.astype(str) == str(evidence["run_id"]))
    ].copy()
    if summary.empty:
        raise ValueError(f"{method}/seed{seed}: aggregate rows are absent")
    if set(summary.table.astype(str)) != set(TABLES):
        raise ValueError(f"{method}/seed{seed}: evaluator tables differ")
    if summary.duplicated(["table", "group_value"]).any():
        raise ValueError(f"{method}/seed{seed}: duplicate aggregate rows")
    files: Dict[str, Any] = {}
    for table, gt_column in TABLES.items():
        universe = universes[table]
        expected_groups = {"overall", *universe.examiner_value.astype(str).unique()}
        rows = summary[summary.table.astype(str) == table]
        if set(rows.group_value.astype(str)) != expected_groups:
            raise ValueError(f"{method}/seed{seed}/{table}: groups differ (including MIXED)")
        expected_counts = {
            "overall": len(universe),
            **universe.examiner_value.astype(str).value_counts().to_dict(),
        }
        if {str(r.group_value): int(r.n_images) for _, r in rows.iterrows()} != expected_counts:
            raise ValueError(f"{method}/seed{seed}/{table}: aggregate counts differ")
        identity = {
            "method": method,
            "seed": seed,
            "subset": "finger_latent",
            "split": "test",
            "ground_truth": gt_column,
            "group_by": "examiner_value",
            "run_id": evidence["run_id"],
            "training_data": "SD302-Latent train split",
            "device": "GPU",
        }
        for column, expected in identity.items():
            actual = rows[column]
            if column == "seed":
                if not (pd.to_numeric(actual, errors="coerce") == expected).all():
                    raise ValueError(f"{method}/seed{seed}/{table}: seed changed")
            elif set(actual.astype(str)) != {str(expected)}:
                raise ValueError(f"{method}/seed{seed}/{table}: {column} changed")
        if not (pd.to_numeric(rows.n_missing, errors="coerce") == 0).all():
            raise ValueError(f"{method}/seed{seed}/{table}: missing predictions")
        matches = list(result_dir.glob(f"per_image_{method}_finger_latent_{table}_seed{seed}_identity-*.csv"))
        if len(matches) != 1:
            raise ValueError(f"{method}/seed{seed}/{table}: expected one per-image result")
        # Subject identifiers are anonymous numeric strings with significant
        # leading zeroes.  Force text on CSV reload so validation does not
        # silently coerce (for example) ``00002302`` to integer ``2302``.
        per = pd.read_csv(
            matches[0],
            dtype={"image_id": str, "subject_id": str, "examiner_value": str},
        )
        expected_ids = set(universe.image_id.astype(str))
        if len(per) != len(expected_ids) or per.image_id.astype(str).duplicated().any() or set(per.image_id.astype(str)) != expected_ids:
            raise ValueError(f"{method}/seed{seed}/{table}: per-image IDs differ")
        expected_subjects = universe.set_index(universe.image_id.astype(str)).subject_id.astype(str).to_dict()
        if any(str(row.subject_id) != expected_subjects[str(row.image_id)] for row in per.itertuples()):
            raise ValueError(f"{method}/seed{seed}/{table}: subject IDs differ")
        expected_group = universe.set_index(universe.image_id.astype(str)).examiner_value.astype(str).to_dict()
        if any(str(row.examiner_value) != expected_group[str(row.image_id)] for row in per.itertuples()):
            raise ValueError(f"{method}/seed{seed}/{table}: examiner groups differ")
        if per.missing.astype(bool).any():
            raise ValueError(f"{method}/seed{seed}/{table}: missing prediction flag")
        _validate_per_image_metrics(per, table)
        # Replay the scientific primitive directly from the frozen prediction
        # and source-GT bytes.  This is deliberately not a second evaluator
        # invocation: it only authenticates the already published confusion
        # counts using the shared IO and metric primitives.
        universe_by_id = universe.set_index(universe.image_id.astype(str))
        for row in per.itertuples():
            image_id = str(row.image_id)
            source = universe_by_id.loc[image_id]
            gt = to_target_ppi(
                read_mask(str(source[gt_column])),
                float(source.ppi),
                500.0,
                is_mask=True,
            )
            pred_path = infer_masks._safe_output(prediction_dir, image_id)
            if not pred_path.is_file():
                raise ValueError(
                    f"{method}/seed{seed}/{table}: prediction disappeared: {image_id}"
                )
            pred = to_target_ppi(
                read_mask(pred_path), 500.0, 500.0, is_mask=True
            )
            pred = match_shape(pred, gt.shape)
            observed = (int(row.tp), int(row.fp), int(row.fn), int(row.tn))
            if confusion(pred, gt) != observed:
                raise ValueError(
                    f"{method}/seed{seed}/{table}: confusion differs from frozen bytes: "
                    f"{image_id}"
                )
        for metric in METRIC_KEYS:
            values = pd.to_numeric(per[metric], errors="coerce")
            finite = values.dropna()
            if not ((finite >= 0) & (finite <= 1)).all():
                raise ValueError(f"{method}/seed{seed}/{table}: metric outside [0,1]")
        for _, aggregate in rows.iterrows():
            group = str(aggregate.group_value)
            source = per if group == "overall" else per[per.examiner_value.astype(str) == group]
            rebuilt = summarise(source)
            for metric in METRIC_KEYS:
                for suffix in ("_mean", "_std", "_pooled"):
                    expected = float(rebuilt[f"{metric}{suffix}"])
                    observed = float(aggregate[f"{metric}{suffix}"])
                    if np.isnan(expected):
                        if not np.isnan(observed):
                            raise ValueError(f"{method}/seed{seed}/{table}: aggregate NaN mismatch")
                        if suffix in {"_mean", "_pooled"} and not _aggregate_nan_expected(source, metric, suffix):
                            raise ValueError(f"{method}/seed{seed}/{table}: unexpected aggregate NaN")
                    elif not np.isclose(observed, expected, rtol=1e-12, atol=1e-12):
                        raise ValueError(f"{method}/seed{seed}/{table}: aggregate mismatch")
        files[table] = {
            "per_image": matches[0],
            "per_image_sha256": sha256_file(matches[0]),
            "n_images": len(per),
            "groups": expected_counts,
        }
    return {"summary": summary, "summary_sha256": sha256_file(summary_path), "tables": files}


def _validate_stage2_reuse(
    reused: Mapping[str, Any], universes: Mapping[str, pd.DataFrame]
) -> Dict[str, Any]:
    delivery_path, delivery_sha = _identity(reused.get("delivery"), "Stage 2 delivery")
    delivery = _read_json(delivery_path, "Stage 2 delivery")
    if delivery.get("schema_version") != 1 or delivery.get("kind") != "stage2_unet_heldout_delivery":
        raise ValueError("Stage 2 delivery identity changed")
    result_dir = delivery_path.parent / "results"
    summary = result_dir / "summary.csv"
    expected_summary = delivery.get("results", {}).get("summary_sha256")
    if sha256_file(summary) != expected_summary:
        raise ValueError("Stage 2 result summary changed")
    rows = pd.read_csv(summary)
    checkpoint_sha = _sha(delivery.get("checkpoint_sha256"), "Stage 2 checkpoint digest")
    if reused.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Stage 2 plan and delivery disagree on the reused checkpoint")
    evidence = {"method": "unet", "seed": 0, "run_id": checkpoint_sha}
    card_path, card_sha = _identity(reused.get("inference_card"), "Stage 2 inference card")
    if delivery.get("inference_card_sha256") != card_sha:
        raise ValueError("Stage 2 delivery and plan disagree on inference card")
    card = _read_json(card_path, "Stage 2 inference card")
    masks = Path(str(reused.get("masks_out", "")))
    if not masks.is_dir():
        raise ValueError("Stage 2 reused prediction directory is absent")
    records = card.get("prediction_set", {}).get("records")
    if not isinstance(records, list):
        raise ValueError("Stage 2 reused prediction records are absent")
    items = []
    for record in records:
        image_id = str(record.get("image_id", ""))
        path = infer_masks._safe_output(masks, image_id)
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Stage 2 reused prediction changed: {image_id}")
        items.append((image_id, path))
    recomputed = prediction_set_identity(items)
    expected_relative = {f"{str(record['image_id'])}.png" for record in records}
    observed_relative = {
        path.relative_to(masks).as_posix()
        for path in masks.rglob("*")
        if path.is_file()
    }
    if observed_relative != expected_relative or recomputed != card["prediction_set"]:
        raise ValueError("Stage 2 reused prediction directory universe changed")
    if reused.get("prediction_set_sha256") != recomputed["sha256"]:
        raise ValueError("Stage 2 reused prediction digest changed")
    checked = _validate_seed_results(
        result_dir, evidence, universes, prediction_dir=masks
    )
    delivered_scoring = delivery.get("results", {}).get("scoring_evidence")
    if not isinstance(delivered_scoring, Mapping) or set(delivered_scoring) != {
        "summary_sha256", "A", "A_prime"
    }:
        raise ValueError("Stage 2 delivery scoring-evidence schema changed")
    if delivered_scoring.get("summary_sha256") != checked["summary_sha256"]:
        raise ValueError("Stage 2 delivery summary scoring identity changed")
    for table in TABLES:
        record = delivered_scoring.get(table)
        observed = checked["tables"][table]
        if not isinstance(record, Mapping):
            raise ValueError(f"Stage 2 delivery lacks {table} scoring evidence")
        if (
            record.get("per_image_file")
            != Path(observed["per_image"]).name
            or record.get("per_image_sha256")
            != observed["per_image_sha256"]
            or int(record.get("n_images", -1)) != observed["n_images"]
            or record.get("groups") != observed["groups"]
        ):
            raise ValueError(f"Stage 2 delivery {table} scoring evidence changed")
    reused_commit = _commit(
        reused.get("expected_inference_git_commit"),
        "reused Stage 2 inference commit",
    )
    require_clean_process_start(
        card,
        reused_commit,
        description="reused Stage 2 U-Net seed-0 inference",
    )
    if (
        card.get("schema_version") != 1
        or card.get("kind") != "complete_unbiased_inference_card"
        or card.get("method") != "unet"
        or card.get("subset") != "finger_latent"
        or card.get("split") != "test"
        or card.get("debug_limited") is not False
        or card.get("identities", {}).get("checkpoint_sha256") != checkpoint_sha
        or card.get("prediction_set", {}).get("count") != len(universes["A"])
        or reused.get("prediction_count") != len(universes["A"])
    ):
        raise ValueError("reused Stage 2 inference-card identity changed")
    if card.get("prediction_set", {}).get("sha256") != reused.get("prediction_set_sha256"):
        raise ValueError("Stage 2 reused prediction-set identity differs")
    return {
        "delivery": delivery_path,
        "delivery_sha256": delivery_sha,
        "inference_card_sha256": card_sha,
        "checkpoint_sha256": checkpoint_sha,
        "prediction_set_sha256": card["prediction_set"]["sha256"],
        "inference_git_commit": reused_commit,
        "masks_out": str(masks),
        "summary": checked["summary"],
        "result_dir": result_dir,
        "tables": checked["tables"],
    }


def _score_command(
    args: argparse.Namespace,
    evidence: Mapping[str, Any],
    output: Path,
    table: str,
    gt_column: str,
) -> list[str]:
    metadata = METHOD_METADATA[str(evidence["method"])]
    return [
        sys.executable,
        str(Path(__file__).with_name("evaluate_masks.py")),
        "--manifest", str(args.manifest),
        "--splits", str(args.splits),
        "--subsets", "finger_latent",
        "--method", str(evidence["method"]),
        "--display-name", metadata["display_name"],
        "--pred-template", str(Path(str(evidence["masks_out"])) / "{image_id}.png"),
        "--pred-ppi", "500",
        "--target-ppi", "500",
        "--split", "test",
        "--gt-column", gt_column,
        "--exclude", "examiner_value=NONPRINT",
        "--group-by", "examiner_value",
        "--seed", str(evidence["seed"]),
        "--year", metadata["year"],
        "--family", metadata["family"],
        "--training-data", "SD302-Latent train split",
        "--comment", metadata["comment"],
        "--params-m", str(evidence["params_m"]),
        "--device", "GPU",
        "--threshold", str(evidence["threshold"]),
        "--run-id", str(evidence["run_id"]),
        "--table", table,
        "--ms-per-img", str(evidence["ms_per_img"]),
        "--out", str(output),
    ]


def _invoke(command: Sequence[str]) -> None:
    completed = subprocess.run(list(command), check=False)
    if completed.returncode == 75:
        raise SystemExit(75)
    completed.check_returncode()


def _copy_seed_results(
    result: Mapping[str, Any], destination: Path, combined: list[pd.DataFrame]
) -> None:
    combined.append(result["summary"].copy())
    for table in TABLES:
        source = Path(result["tables"][table]["per_image"])
        target = destination / source.name
        if target.exists():
            raise FileExistsError(f"duplicate per-image evidence: {target.name}")
        shutil.copyfile(source, target)


def _scoring_record(
    evidence: Mapping[str, Any], checked: Mapping[str, Any]
) -> Dict[str, Any]:
    """Stable per-seed evidence, independent of the combined summary CSV."""
    return {
        **{key: evidence[key] for key in evidence if key != "masks_out"},
        "summary_sha256": checked["summary_sha256"],
        "tables": {
            table: {
                key: value
                for key, value in checked["tables"][table].items()
                if key != "per_image"
            }
            for table in TABLES
        },
    }


def _gate_evidence(
    scored: Iterable[Mapping[str, Any]], gate: Mapping[str, Any]
) -> Dict[str, Any]:
    successes = {(str(item["method"]), int(item["seed"])) for item in scored}
    methods = gate["methods"]
    rows = {}
    failures = []
    for method in NATIVE_METHODS:
        available = sorted(seed for candidate, seed in successes if candidate == method)
        missing = sorted(set(REQUIRED_SEEDS) - set(available))
        declared = methods[method]
        declared_failed = sorted(int(seed) for seed in declared.get("failed_seeds", []))
        declared_missing = sorted(int(seed) for seed in declared.get("missing_seeds", []))
        if set(declared_failed) & set(declared_missing):
            raise PostScoreFailure(f"{method}: a seed is both failed and missing")
        if sorted(set(declared_failed + declared_missing)) != missing:
            raise PostScoreFailure(f"{method}: gate missing/failed seeds do not reconcile")
        passed = len(available) >= 2
        rows[method] = {
            "successful_seeds": available,
            "failed_seeds": declared_failed,
            "missing_seeds": declared_missing,
            "n_successful": len(available),
            "minimum_required": 2,
            "pass": passed,
            "failure_details": {
                str(seed): declared.get("seeds", {}).get(str(seed), {})
                for seed in sorted(set(declared_failed + declared_missing))
            },
        }
        if not passed:
            failures.append(method)
    return {
        "criterion": "at least two successful held-out seeds per native method",
        "status": "PASS" if not failures else "FAIL",
        "methods": rows,
        "failures": failures,
    }


def _output_identities(root: Path) -> Dict[str, str]:
    output = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "delivery.json":
            output[str(path.relative_to(root))] = sha256_file(path)
    return output


def _verify_delivery(args: argparse.Namespace) -> Dict[str, Any]:
    """Revalidate every input and scientific output without running inference/eval."""
    delivery_path = args.bundle_out / "delivery.json"
    delivery = _read_json(delivery_path, "Stage 3 delivery")
    if delivery.get("schema_version") != 1 or delivery.get("kind") != "stage3_native_heldout_delivery":
        raise ValueError("published Stage 3 delivery identity changed")
    if delivery.get("gate_sha256") != sha256_file(args.gate) or delivery.get("plan_sha256") != sha256_file(args.plan):
        raise ValueError("published delivery gate/plan changed")
    expected = delivery.get("outputs")
    if not isinstance(expected, dict) or expected != _output_identities(args.bundle_out):
        raise ValueError("published Stage 3 output bytes changed")
    gate, plan = _validate_gate_and_plan(args)
    code_hashes = _validate_code_identities(plan)
    inference_code_hashes = _validate_inference_code_identities(plan)
    queue_hashes = _validate_queue(args, plan)
    auxiliary = _validate_auxiliary(args, gate, plan)
    audit_path, audit_sha = _identity(plan.get("manifest_audit"), "manifest audit")
    audit = _read_json(audit_path, "manifest audit")
    if audit.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("manifest audit does not bind the replay manifest")
    ordered_gt = _ordered_source_gt(args.manifest)
    if (
        audit.get("ordered_source_gt_sha256") != ordered_gt
        or plan.get("ordered_source_gt_sha256") != ordered_gt
    ):
        raise ValueError("source GT changed after Stage 3 publication")
    ordered_sources = _ordered_source_images(args.manifest, args.splits)
    if (
        plan.get("ordered_source_images_sha256") != ordered_sources
        or gate.get("test_universe", {}).get("ordered_source_images_sha256")
        != ordered_sources
    ):
        raise ValueError("source images changed after Stage 3 publication")
    universes = _test_universes(args.manifest, args.splits)
    validated, reused_record = _validate_plan_runs(
        plan, gate, args, universes, ordered_sources
    )
    reused = _validate_stage2_reuse(reused_record, universes)
    journal_path = _journal_path(args.bundle_out)
    journal = _read_json(journal_path, "write-once Stage 3 scoring journal")
    if delivery.get("journal_sha256") != sha256_file(journal_path):
        raise ValueError("published delivery journal changed")
    journal_expected = {
        "gate_sha256": sha256_file(args.gate),
        "plan_sha256": sha256_file(args.plan),
        "manifest_sha256": sha256_file(args.manifest),
        "splits_sha256": sha256_file(args.splits),
        "manifest_audit_sha256": audit_sha,
        "ordered_source_gt_sha256": ordered_gt,
        "ordered_source_images_sha256": ordered_sources,
        "code": code_hashes,
        "inference_code": inference_code_hashes,
        "prep_bundle_sha256": sha256_file(args.prep_bundle),
        "inference_queue": queue_hashes,
        "auxiliary": auxiliary,
        "seeds": validated,
        "planned_evaluator_invocations": 2 * len(validated),
        "test_scoring_started": True,
        "automatic_rescore_permitted": False,
        "bundle_absent_before_scoring": True,
    }
    for key, value in journal_expected.items():
        if journal.get(key) != value:
            raise ValueError(f"published Stage 3 journal binding changed: {key}")
    if journal.get("kind") != "stage3_native_heldout_scoring_journal":
        raise ValueError("published Stage 3 journal kind changed")

    results_dir = args.bundle_out / "results"
    checked_records = []
    reused_evidence = {
        "method": "unet",
        "seed": 0,
        "run_id": reused["checkpoint_sha256"],
    }
    reused_checked = _validate_seed_results(
        results_dir,
        reused_evidence,
        universes,
        prediction_dir=Path(reused["masks_out"]),
    )
    expected_rows = len(reused_checked["summary"])
    for evidence in validated:
        seed_dir = (
            args.bundle_out
            / "seed_scores"
            / f"{evidence['method']}_seed{evidence['seed']}"
        )
        seed_checked = _validate_seed_results(
            seed_dir,
            evidence,
            universes,
            prediction_dir=Path(str(evidence["masks_out"])),
        )
        # Also authenticate the copied combined per-image evidence and its
        # aggregate rows against the same frozen prediction/GT bytes.
        combined_checked = _validate_seed_results(
            results_dir,
            evidence,
            universes,
            prediction_dir=Path(str(evidence["masks_out"])),
        )
        for table in TABLES:
            if (
                seed_checked["tables"][table]["per_image_sha256"]
                != combined_checked["tables"][table]["per_image_sha256"]
            ):
                raise ValueError(
                    f"{evidence['method']}/seed{evidence['seed']}/{table}: "
                    "published copy differs from immutable seed evidence"
                )
        checked_records.append(_scoring_record(evidence, seed_checked))
        expected_rows += len(combined_checked["summary"])
    combined = pd.read_csv(results_dir / "summary.csv")
    if len(combined) != expected_rows:
        raise ValueError("published Stage 3 summary has extra or missing rows")
    if delivery.get("newly_scored_seeds") != checked_records:
        raise ValueError("published Stage 3 per-seed scoring evidence changed")
    expected_reuse = {
        "reused_not_rescored": True,
        "delivery_sha256": reused["delivery_sha256"],
        "inference_card_sha256": reused["inference_card_sha256"],
        "prediction_set_sha256": reused["prediction_set_sha256"],
    }
    if delivery.get("stage2_unet_seed0") != expected_reuse:
        raise ValueError("published Stage 2 reuse evidence changed")
    expected_gate = _gate_evidence(
        [{"method": "unet", "seed": 0}, *validated], gate
    )
    if delivery.get("gate_evidence") != expected_gate:
        raise ValueError("published Stage 3 seed gate evidence changed")
    for key, value in {
        "manifest_sha256": sha256_file(args.manifest),
        "splits_sha256": sha256_file(args.splits),
        "manifest_audit_sha256": audit_sha,
        "ordered_source_gt_sha256": ordered_gt,
        "ordered_source_images_sha256": ordered_sources,
        "code": code_hashes,
        "inference_code": inference_code_hashes,
        "prep_bundle_sha256": sha256_file(args.prep_bundle),
        "inference_queue": queue_hashes,
        "auxiliary": auxiliary,
        "evaluator_invocations": 2 * len(validated),
    }.items():
        if delivery.get(key) != value:
            raise ValueError(f"published delivery binding changed: {key}")
    return delivery


def run(args: argparse.Namespace, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute or verify the one-time transaction."""
    require_clean_process_start(
        {"provenance": provenance},
        args.expected_finalizer_git_commit,
        description="Stage 3 held-out finalizer",
    )
    _require_under(args.bundle_out, args.restricted_root, "Stage 3 bundle")
    if args.declared_delivery.resolve() != (args.bundle_out / "delivery.json").resolve():
        raise ValueError("declared finalizer output is not bundle_out/delivery.json")
    if args.bundle_out.exists():
        return _verify_delivery(args)
    journal, lock = _journal_path(args.bundle_out), _lock_path(args.bundle_out)
    if journal.exists():
        raise RuntimeError(
            "Stage 3 held-out scoring already started; automatic rescoring is forbidden"
        )
    lock_fd = _acquire_lock(lock)
    stage: Optional[Path] = None
    scoring_started = False
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.fsync(lock_fd)
        gate, plan = _validate_gate_and_plan(args)
        code_hashes = _validate_code_identities(plan)
        inference_code_hashes = _validate_inference_code_identities(plan)
        queue_hashes = _validate_queue(args, plan)
        auxiliary = _validate_auxiliary(args, gate, plan)
        audit_path, audit_sha = _identity(plan.get("manifest_audit"), "manifest audit")
        audit = _read_json(audit_path, "manifest audit")
        if audit.get("manifest_sha256") != sha256_file(args.manifest):
            raise ValueError("manifest audit does not bind the scored manifest")
        ordered_gt = _ordered_source_gt(args.manifest)
        if audit.get("ordered_source_gt_sha256") != ordered_gt or plan.get("ordered_source_gt_sha256") != ordered_gt:
            raise ValueError("source GT bytes differ from frozen manifest audit/plan")
        ordered_sources = _ordered_source_images(args.manifest, args.splits)
        if (
            plan.get("ordered_source_images_sha256") != ordered_sources
            or gate.get("test_universe", {}).get(
                "ordered_source_images_sha256"
            )
            != ordered_sources
        ):
            raise ValueError("held-out source-image identity differs from gate/plan")
        universes = _test_universes(args.manifest, args.splits)
        validated, reused_record = _validate_plan_runs(
            plan, gate, args, universes, ordered_sources
        )
        reused = _validate_stage2_reuse(reused_record, universes)
        stage = Path(tempfile.mkdtemp(prefix=f".{args.bundle_out.name}-stage3-", dir=args.bundle_out.parent))
        results_dir = stage / "results"
        results_dir.mkdir()
        combined: list[pd.DataFrame] = []
        _copy_seed_results(reused, results_dir, combined)
        bindings = {
            "schema_version": 1,
            "kind": "stage3_native_heldout_scoring_journal",
            "test_scoring_started": True,
            "automatic_rescore_permitted": False,
            "bundle_absent_before_scoring": True,
            "gate_sha256": sha256_file(args.gate),
            "plan_sha256": sha256_file(args.plan),
            "manifest_sha256": sha256_file(args.manifest),
            "splits_sha256": sha256_file(args.splits),
            "manifest_audit_sha256": audit_sha,
            "ordered_source_gt_sha256": ordered_gt,
            "ordered_source_images_sha256": ordered_sources,
            "code": code_hashes,
            "inference_code": inference_code_hashes,
            "prep_bundle_sha256": sha256_file(args.prep_bundle),
            "inference_queue": queue_hashes,
            "auxiliary": auxiliary,
            "stage2_reuse": {
                key: reused[key]
                for key in (
                    "delivery_sha256", "inference_card_sha256", "checkpoint_sha256",
                    "prediction_set_sha256",
                )
            },
            "seeds": validated,
            "planned_evaluator_invocations": 2 * len(validated),
            "staging_directory": str(stage),
            "finalizer_provenance": dict(provenance),
        }
        _exclusive_json(journal, bindings)
        scoring_started = True
        scored_evidence: list[Dict[str, Any]] = []
        for evidence in validated:
            seed_dir = stage / "seed_scores" / f"{evidence['method']}_seed{evidence['seed']}"
            seed_dir.mkdir(parents=True)
            for table, gt_column in TABLES.items():
                _invoke(_score_command(args, evidence, seed_dir, table, gt_column))
            checked = _validate_seed_results(
                seed_dir,
                evidence,
                universes,
                prediction_dir=Path(str(evidence["masks_out"])),
            )
            _copy_seed_results(checked, results_dir, combined)
            scored_evidence.append(_scoring_record(evidence, checked))
        summary = pd.concat(combined, ignore_index=True)
        key = ["method", "seed", "table", "group_value", "run_id", "training_data"]
        if summary.duplicated(key).any():
            raise PostScoreFailure("combined Stage 3 summary contains duplicate scientific rows")
        summary.to_csv(results_dir / "summary.csv", index=False)
        tables_dir = stage / "tables"
        make_tables.build_tables(summary, tables_dir, selected_tables=("A", "A_prime"))
        all_scored = [{"method": "unet", "seed": 0}, *validated]
        gate_evidence = _gate_evidence(all_scored, gate)
        if gate_evidence["status"] != "PASS":
            raise PostScoreFailure("Stage 3 seed-completeness gate failed")

        # Close the persistent-mutation window opened by the evaluator calls.
        # The evaluator and the result replay necessarily consume the same GT
        # and prediction files; without this independent post-score binding, a
        # file changed after the preflight could be internally self-consistent
        # yet disagree with the identities frozen in the journal.  Rehash all
        # scientific inputs immediately before publication and require the
        # complete card/directory and reused-Stage2 chains to reproduce.
        if _ordered_source_gt(args.manifest) != ordered_gt:
            raise PostScoreFailure("source GT changed during Stage 3 scoring")
        if _ordered_source_images(args.manifest, args.splits) != ordered_sources:
            raise PostScoreFailure("source images changed during Stage 3 scoring")
        post_validated, post_reused_record = _validate_plan_runs(
            plan, gate, args, universes, ordered_sources
        )
        if post_validated != validated or post_reused_record != reused_record:
            raise PostScoreFailure(
                "prediction identities changed during Stage 3 scoring"
            )
        post_reused = _validate_stage2_reuse(post_reused_record, universes)
        for key in (
            "delivery_sha256",
            "inference_card_sha256",
            "checkpoint_sha256",
            "prediction_set_sha256",
        ):
            if post_reused[key] != reused[key]:
                raise PostScoreFailure(
                    "reused Stage 2 evidence changed during Stage 3 scoring"
                )

        delivery = {
            "schema_version": 1,
            "kind": "stage3_native_heldout_delivery",
            "gate_sha256": sha256_file(args.gate),
            "plan_sha256": sha256_file(args.plan),
            "journal_sha256": sha256_file(journal),
            "manifest_sha256": sha256_file(args.manifest),
            "splits_sha256": sha256_file(args.splits),
            "manifest_audit_sha256": audit_sha,
            "ordered_source_gt_sha256": ordered_gt,
            "ordered_source_images_sha256": ordered_sources,
            "code": code_hashes,
            "inference_code": inference_code_hashes,
            "prep_bundle_sha256": sha256_file(args.prep_bundle),
            "inference_queue": queue_hashes,
            "auxiliary": auxiliary,
            "stage2_unet_seed0": {
                "reused_not_rescored": True,
                "delivery_sha256": reused["delivery_sha256"],
                "inference_card_sha256": reused["inference_card_sha256"],
                "prediction_set_sha256": reused["prediction_set_sha256"],
            },
            "newly_scored_seeds": scored_evidence,
            "evaluator_invocations": 2 * len(validated),
            "gate_evidence": gate_evidence,
            "provenance": dict(provenance),
        }
        delivery["outputs"] = _output_identities(stage)
        _exclusive_json(stage / "delivery.json", delivery)
        os.replace(stage, args.bundle_out)
        stage = None
        lock.unlink()
        return delivery
    except BaseException as exc:
        if scoring_started:
            failure = {
                "schema_version": 1,
                "kind": "stage3_native_postscore_failure",
                "automatic_rescore_permitted": False,
                "journal_sha256": sha256_file(journal),
                "staging_directory": str(stage) if stage is not None else "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if not _failure_path(args.bundle_out).exists():
                _exclusive_json(_failure_path(args.bundle_out), failure)
            # Deliberately retain lock, journal, and staging for audit.
        else:
            if stage is not None:
                shutil.rmtree(stage)
            lock.unlink(missing_ok=True)
        raise
    finally:
        os.close(lock_fd)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prep-bundle", required=True, type=Path)
    parser.add_argument("--expected-prep-bundle-sha256", required=True)
    parser.add_argument("--inference-queue", required=True, type=Path)
    parser.add_argument("--inference-ledger", required=True, type=Path)
    parser.add_argument("--aux-plan", required=True, type=Path)
    parser.add_argument("--aux-queue", required=True, type=Path)
    parser.add_argument("--aux-ledger", required=True, type=Path)
    parser.add_argument("--aux-card", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--bundle-out", required=True, type=Path)
    parser.add_argument("--restricted-root", required=True, type=Path)
    parser.add_argument("--declared-delivery", required=True, type=Path)
    parser.add_argument("--expected-finalizer-git-commit", required=True)
    parser.add_argument("--expected-planner-git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    from segbench.deep_utils import run_provenance

    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    print(json.dumps(run(args, provenance), sort_keys=True))


if __name__ == "__main__":
    main()
