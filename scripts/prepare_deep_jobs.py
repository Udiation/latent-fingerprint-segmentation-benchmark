#!/usr/bin/env python3
"""Prepare and select the pre-registered deep-model experiment queue.

``prepare-sweep`` materialises the two learning-rate configurations for each
model (seed 0, 30 epochs) and separates the U-Net pilot from the remaining
queue. ``select-pilot`` can freeze U-Net and emit its full seed-0 gate as soon
as those two validation runs finish. ``select-full`` retains the all-method
selection and emits separate U-Net-pilot, remaining-seed-0, and seeds-1/2
queues. Physical device choices belong to ``run_queue.py`` and never appear in
these job files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import require_clean_process_start
from segbench.deep_utils import (
    PROTOCOL_DATALOADER,
    build_queue_batch_contract,
    config_digest,
)


METHOD_PRIORITY = (
    "unet",
    "unetpp",
    "drunet",
    "ra_runet",
    "lenet",
    "alexnet",
    "cnn_ae",
    "segformer_sub",
)
LEARNING_RATES = (0.0003, 0.001)
SWEEP_EPOCHS = 30
SWEEP_PATIENCE = SWEEP_EPOCHS
FULL_EPOCHS = 100
FULL_PATIENCE = 15
FULL_SEEDS = (0, 1, 2)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
LEGACY_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
JOB_NAME_RE = re.compile(r"^full\.([a-z0-9_]+)\.seed([0-2])$")


def _production_input_identity(args: argparse.Namespace) -> Dict[str, Any]:
    """Authenticate cache/amendment inputs required by every new full job."""
    cache_dir = Path(args.cache_dir).resolve()
    receipt_path = cache_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("kind") != "uint8_500ppi_preprocessing_cache"
        or receipt.get("manifest_sha256") != file_digest(args.manifest)
        or float(receipt.get("target_ppi", math.nan)) != 500.0
        or receipt.get("mask_columns") != ["mask_path", "quality_mask_path"]
        or receipt.get("debug_limited") is not False
    ):
        raise ValueError("full-training cache receipt is not the complete 500-ppi corpus")
    amendment = Path(args.threshold_amendment).resolve()
    amendment_payload = json.loads(amendment.read_text(encoding="utf-8"))
    replacement = (
        amendment_payload.get("replacement_policy", {})
        if isinstance(amendment_payload, dict)
        else {}
    )
    if (
        amendment_payload.get("kind") != "stage4_threshold_policy_amendment"
        or replacement.get("allowed_trainable_grid") != list(THRESHOLDS)
        or replacement.get("selection_data") != "validation_only"
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(args.expected_threshold_amendment_sha256)
        )
        or file_digest(amendment)
        != str(args.expected_threshold_amendment_sha256)
    ):
        raise ValueError("full-training threshold amendment changed")
    return {
        "cache_dir": str(cache_dir),
        "cache_receipt": str(receipt_path),
        "cache_receipt_sha256": file_digest(receipt_path),
        "threshold_amendment": str(amendment),
        "threshold_amendment_sha256": file_digest(amendment),
    }


def _bind_new_training_config(
    config: Dict[str, Any], production_inputs: Mapping[str, Any]
) -> None:
    """Bind the queue batch contract and immutable production inputs in-place."""
    training = config.get("training")
    runtime = config.get("runtime")
    if not isinstance(training, dict) or not isinstance(runtime, dict):
        raise ValueError("new full-training config lacks training/runtime mappings")
    if any(training.get(key) != value for key, value in PROTOCOL_DATALOADER.items()):
        raise ValueError("new full-training config differs from the loader protocol")
    contract = build_queue_batch_contract(training)
    runtime["queue_batch_contract"] = contract
    runtime["queue_batch_contract_sha256"] = config_digest(contract)
    runtime["preprocess_cache_receipt_sha256"] = production_inputs[
        "cache_receipt_sha256"
    ]
    runtime["threshold_amendment_sha256"] = production_inputs[
        "threshold_amendment_sha256"
    ]


def _legacy_unetpp_selector(
    paths: Optional[Sequence[Path]],
    *,
    source_config: Path,
    checkpoint: Path,
    manifest_sha256: str,
    splits_sha256: str,
    amendment: Optional[Path],
) -> Dict[str, Any]:
    """Authenticate one of two expanded-grid receipts for a frozen sweep cell."""
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    training = config.get("training", {}) if isinstance(config, dict) else {}
    observed_grid = tuple(float(value) for value in training.get("threshold_candidates", ()))
    if observed_grid != LEGACY_THRESHOLDS:
        raise ValueError("U-Net++ preserved sweep config is not the registered legacy grid")
    if paths is None or len(paths) != len(LEARNING_RATES) or amendment is None:
        raise ValueError(
            "preserved U-Net++ sweeps require two expanded-grid selector receipts "
            "and the exact threshold amendment before full-run selection"
        )
    amendment_sha256 = file_digest(amendment)
    matches = []
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "kind": "expanded_validation_threshold_selector_receipt",
            "method": "unetpp",
            "seed": 0,
            "selection_split": "val",
            "selection_source": "validation_only",
            "validation_only": True,
            "test_or_heldout_opened": False,
            "source_config_sha256": file_digest(source_config),
            "checkpoint_sha256": file_digest(checkpoint),
            "manifest_sha256": manifest_sha256,
            "split_sha256": splits_sha256,
            "amendment_sha256": amendment_sha256,
            "threshold_amendment_sha256": amendment_sha256,
            "probabilities_are_pre_threshold": True,
            "derived_from_binary_mask": False,
            "debug_limited": False,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            continue
        if tuple(float(value) for value in receipt.get("allowed_grid", ())) != THRESHOLDS:
            raise ValueError("U-Net++ selector receipt does not use the expanded grid")
        if not isinstance(receipt.get("raw_validation_probability_card_sha256"), str):
            raise ValueError("U-Net++ selector lacks its raw validation card identity")
        matches.append((path, receipt))
    if len(matches) != 1:
        raise ValueError("U-Net++ sweep cell does not have exactly one matching selector")
    path, receipt = matches[0]
    return {
        "path": str(path),
        "sha256": file_digest(path),
        "raw_validation_probability_card_sha256": receipt[
            "raw_validation_probability_card_sha256"
        ],
        "selected_validation_probability_card_sha256": receipt[
            "validation_probability_card_sha256"
        ],
        "selected_threshold": float(receipt["selected_threshold"]),
        "selected_validation_dice": float(receipt["selected_validation_dice"]),
        "amendment_sha256": amendment_sha256,
    }

def expected_commit(method: str) -> str:
    """Resolve the prospective method-specific clean-commit allow-list."""
    path_value = os.environ.get("SEGBENCH_EXPECTED_COMMIT_MAP")
    if not path_value:
        raise RuntimeError("SEGBENCH_EXPECTED_COMMIT_MAP is required for selection")
    path = Path(path_value)
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or set(mapping) != set(METHOD_PRIORITY):
        raise ValueError("expected-commit map must cover exactly the native methods")
    value = mapping.get(method)
    if not isinstance(value, str):
        raise ValueError(f"expected-commit map lacks {method}")
    return value


def file_digest(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    """Atomically publish a text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable YAML atomically."""
    atomic_text(path, yaml.safe_dump(dict(payload), sort_keys=False))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON atomically."""
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_base_config(config_dir: Path, method: str) -> Dict[str, Any]:
    """Load and validate one method's committed base configuration."""
    path = config_dir / f"{method}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing base config {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("method") != method:
        raise ValueError(f"{path} does not declare method {method!r}")
    task = config.get("task")
    if task not in {"pixel", "patch"}:
        raise ValueError(f"{path} has unsupported task {task!r}")
    thresholds = tuple(float(value) for value in config["training"]["threshold_candidates"])
    if thresholds != THRESHOLDS:
        raise ValueError(f"{path} does not use the exact threshold grid {THRESHOLDS}")
    if config["data"].get("normalization") != "train_only":
        raise ValueError(f"{path} does not declare train-only normalization")
    if method == "ra_runet" and config["model"].get("alignment", {}).get("enabled"):
        raise ValueError("Dataset-1 ra_runet sweep must disable domain alignment")
    return config


def lr_slug(learning_rate: float) -> str:
    """Return the frozen filesystem/job name for a sweep rate."""
    mapping = {0.0003: "0p0003", 0.001: "0p001"}
    try:
        return mapping[learning_rate]
    except KeyError as exc:
        raise ValueError(f"unregistered learning rate {learning_rate}") from exc


def training_entrypoint(code_root: Path, task: str) -> Path:
    """Resolve the shared task-specific trainer."""
    return code_root / "scripts" / ("train_seg.py" if task == "pixel" else "train_patch.py")


def training_job(
    *,
    name: str,
    python_executable: str,
    entrypoint: Path,
    config: Path,
    manifest: Path,
    splits: Path,
    out: Path,
    seed: int,
    extra_argv: Sequence[str] = (),
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build an argv-safe resumable run_queue entry."""
    job = {
        "name": name,
        "argv": [
            python_executable,
            str(entrypoint),
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--splits",
            str(splits),
            "--out",
            str(out),
            "--seed",
            str(seed),
            "--resume",
            "auto",
            *extra_argv,
        ],
    }
    if env:
        job["env"] = dict(env)
    return job


def native_job_environment(method: str) -> Optional[Dict[str, str]]:
    """Freeze external local-only model paths into each queued job."""
    if method != "segformer_sub":
        return None
    value = os.environ.get("SEGBENCH_SEGFORMER_B2_DIR", "")
    path = Path(value)
    if not value or not path.is_absolute():
        raise RuntimeError(
            "SEGBENCH_SEGFORMER_B2_DIR must be an absolute pinned snapshot path"
        )
    return {"SEGBENCH_SEGFORMER_B2_DIR": str(path)}


def workflow_notes() -> Dict[str, str]:
    """Make separate and unsupported rows explicit rather than omitting them."""
    return {
        "sufs_ft": (
            "included through its separate audited PyFing/Keras-Torch workflow; "
            "it is intentionally not merged into the native-model trainers"
        ),
        "fingernet": (
            "official pretrained legacy inference is a separate zero-shot "
            "workflow and receives no learning-rate sweep"
        ),
    }


def prepare_sweep(args: argparse.Namespace) -> None:
    """Materialise exact seed-0, 30-epoch configurations and job queues."""
    for path in (args.manifest, args.splits):
        if not path.is_file():
            raise FileNotFoundError(path)
    production_inputs = _production_input_identity(args)
    production_argv = (
        "--cache-dir",
        production_inputs["cache_dir"],
        "--threshold-amendment",
        production_inputs["threshold_amendment"],
    )
    pilot_jobs = []
    remaining_jobs = []
    manifest_hash = file_digest(args.manifest)
    split_hash = file_digest(args.splits)
    records = []
    for method in METHOD_PRIORITY:
        base = load_base_config(args.base_config_dir, method)
        for learning_rate in LEARNING_RATES:
            slug = lr_slug(learning_rate)
            config = json.loads(json.dumps(base))
            config["seed"] = 0
            config["training"]["learning_rate"] = learning_rate
            config["training"]["epochs"] = SWEEP_EPOCHS
            # Learning-rate candidates receive exactly the same fixed budget.
            # With the trainers' epoch-boundary stopping check, patience equal
            # to the requested epoch count cannot fire inside this sweep.
            config["training"]["patience"] = SWEEP_PATIENCE
            config["experiment"] = {
                "phase": "learning_rate_sweep",
                "selection_split": "val",
                "selection_metric": "best_validation_dice",
                "learning_rate_candidates": list(LEARNING_RATES),
                "seed": 0,
                "epochs": SWEEP_EPOCHS,
                "patience": SWEEP_PATIENCE,
            }
            _bind_new_training_config(config, production_inputs)
            config_path = args.generated_config_dir / "sweep" / f"{method}_lr{slug}.yaml"
            write_yaml(config_path, config)
            out = args.runs_root / "sweep" / method / f"lr_{slug}" / "seed0"
            job = training_job(
                name=f"sweep.{method}.lr{slug}.seed0",
                python_executable=args.python_executable,
                entrypoint=training_entrypoint(args.code_root, str(base["task"])),
                config=config_path,
                manifest=args.manifest,
                splits=args.splits,
                out=out,
                seed=0,
                extra_argv=production_argv,
                env=native_job_environment(method),
            )
            (pilot_jobs if method == "unet" else remaining_jobs).append(job)
            records.append(
                {
                    "method": method,
                    "learning_rate": learning_rate,
                    "config": str(config_path),
                    "config_sha256": file_digest(config_path),
                    "run_dir": str(out),
                }
            )
    sufs_config = args.base_config_dir / "sufs_ft.yaml"
    if not sufs_config.is_file():
        raise FileNotFoundError(f"missing SUFS fine-tuning config {sufs_config}")
    for learning_rate in LEARNING_RATES:
        slug = lr_slug(learning_rate)
        out = args.runs_root / "sweep" / "sufs_ft" / f"lr_{slug}" / "seed0"
        job = training_job(
            name=f"sweep.sufs_ft.lr{slug}.seed0",
            python_executable=args.python_executable,
            entrypoint=args.code_root / "scripts" / "train_sufs.py",
            config=sufs_config,
            manifest=args.manifest,
            splits=args.splits,
            out=out,
            seed=0,
            extra_argv=(
                "--mode",
                "sweep",
                "--learning-rate",
                str(learning_rate),
                *production_argv,
            ),
            env={"KERAS_BACKEND": "torch"},
        )
        remaining_jobs.append(job)
        records.append(
            {
                "method": "sufs_ft",
                "workflow": "separate_pyfing_keras_torch",
                "learning_rate": learning_rate,
                "config": str(sufs_config),
                "config_sha256": file_digest(sufs_config),
                "run_dir": str(out),
            }
        )
    common = {
        "schema_version": 1,
        "phase": "learning_rate_sweep",
        "manifest_sha256": manifest_hash,
        "splits_sha256": split_hash,
        "learning_rates": list(LEARNING_RATES),
        "epochs": SWEEP_EPOCHS,
        "native_sweep_patience": SWEEP_PATIENCE,
        "seed": 0,
        "production_inputs": production_inputs,
        "workflow_notes": workflow_notes(),
    }
    write_yaml(args.pilot_jobs_out, {**common, "jobs": pilot_jobs})
    write_yaml(args.remaining_jobs_out, {**common, "jobs": remaining_jobs})
    write_json(
        args.generated_config_dir / "sweep_plan.json",
        {**common, "runs": records},
    )


def validate_sweep_run(
    run_dir: Path,
    *,
    method: str,
    learning_rate: float,
    source_config: Path,
    manifest_sha256: str,
    splits_sha256: str,
) -> Dict[str, Any]:
    """Verify a complete, unlimited validation-only sweep run card."""
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"incomplete sweep: missing {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    require_clean_process_start(
        run, expected_commit(method), description=f"{method} sweep {run_path}"
    )
    expected = {
        "method": method,
        "seed": 0,
        "epochs_requested": SWEEP_EPOCHS,
        "epochs_completed": SWEEP_EPOCHS,
        "manifest_sha256": manifest_sha256,
        "split_sha256": splits_sha256,
        "source_config_sha256": file_digest(source_config),
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise ValueError(
                f"{run_path}: {key}={run.get(key)!r}, expected {value!r}"
            )
    if not math.isclose(
        float(run.get("learning_rate", math.nan)), learning_rate, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"{run_path}: learning rate disagrees with sweep cell")
    if run.get("debug_limited"):
        raise ValueError(f"{run_path}: debug-limited run is ineligible for selection")
    score = float(run.get("best_validation_dice", math.nan))
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{run_path}: invalid best validation Dice {score!r}")
    for checkpoint_name, expected_hash in run.get("checkpoint_sha256", {}).items():
        checkpoint = run_dir / checkpoint_name
        if not checkpoint.is_file() or file_digest(checkpoint) != expected_hash:
            raise ValueError(f"{run_path}: checkpoint hash failure for {checkpoint_name}")
    if set(run.get("checkpoint_sha256", {})) != {"best.pt", "last.pt"}:
        raise ValueError(f"{run_path}: incomplete checkpoint provenance")
    return run


def validate_sufs_sweep_run(
    run_dir: Path,
    *,
    learning_rate: float,
    source_config: Path,
    manifest_sha256: str,
    splits_sha256: str,
) -> Path:
    """Verify SUFS sweep provenance before invoking its native selector."""
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"incomplete SUFS sweep: missing {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected = {
        "method": "sufs_ft",
        "mode": "sweep",
        "seed": 0,
        "epochs_requested": SWEEP_EPOCHS,
        "epochs_completed": SWEEP_EPOCHS,
        "manifest_sha256": manifest_sha256,
        "split_sha256": splits_sha256,
        "source_config_sha256": file_digest(source_config),
    }
    for key, value in expected.items():
        if run.get(key) != value:
            raise ValueError(
                f"{run_path}: {key}={run.get(key)!r}, expected {value!r}"
            )
    if not math.isclose(
        float(run.get("learning_rate", math.nan)), learning_rate, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"{run_path}: learning rate disagrees with SUFS sweep cell")
    if run.get("debug_limited"):
        raise ValueError(f"{run_path}: debug-limited run is ineligible for selection")
    for checkpoint_name, expected_hash in run.get("checkpoint_sha256", {}).items():
        checkpoint = run_dir / checkpoint_name
        if not checkpoint.is_file() or file_digest(checkpoint) != expected_hash:
            raise ValueError(f"{run_path}: checkpoint hash failure for {checkpoint_name}")
    if set(run.get("checkpoint_sha256", {})) != {"best.pt", "last.pt"}:
        raise ValueError(f"{run_path}: incomplete checkpoint provenance")
    return run_path


def select_native_sweep_evidence(
    sweep_cells: Sequence[Tuple[float, Path, Path]],
    *,
    method: str,
    manifest_sha256: str,
    splits_sha256: str,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Validate exactly the two registered cells and select on validation Dice.

    Each cell is ``(learning_rate, run_directory, generated_sweep_config)``.
    Keeping the registered rate attached to its fixed paths prevents an
    arbitrary run-card list from becoming a second, editable experiment plan.
    """
    if len(sweep_cells) != len(LEARNING_RATES):
        raise ValueError(
            f"{method} selection requires exactly two preregistered sweep cells"
        )
    observed_rates = [float(cell[0]) for cell in sweep_cells]
    if len(set(observed_rates)) != len(LEARNING_RATES) or set(observed_rates) != set(
        LEARNING_RATES
    ):
        raise ValueError(
            f"{method} selection requires exactly the preregistered rates "
            f"{LEARNING_RATES}"
        )
    ordered_cells = sorted(
        sweep_cells,
        key=lambda cell: LEARNING_RATES.index(float(cell[0])),
    )
    candidates = []
    for learning_rate, run_dir, source_config in ordered_cells:
        run = validate_sweep_run(
            run_dir,
            method=method,
            learning_rate=learning_rate,
            source_config=source_config,
            manifest_sha256=manifest_sha256,
            splits_sha256=splits_sha256,
        )
        run_card = run_dir / "run.json"
        checkpoint_hashes = {
            name: str(run["checkpoint_sha256"][name])
            for name in ("best.pt", "last.pt")
        }
        candidates.append(
            {
                "learning_rate": learning_rate,
                "best_validation_dice": float(run["best_validation_dice"]),
                "best_epoch": int(run["best_epoch"]),
                "selected_validation_threshold": float(
                    run["selected_validation_threshold"]
                ),
                "seed": 0,
                "epochs_requested": SWEEP_EPOCHS,
                "source_config": str(source_config),
                "source_config_sha256": file_digest(source_config),
                "config_sha256": str(run["config_sha256"]),
                "run_card": str(run_card),
                "run_card_sha256": file_digest(run_card),
                "checkpoint_sha256": checkpoint_hashes,
                "source_git_commit": run["provenance"]["git_commit"],
            }
        )
    chosen = max(
        candidates,
        key=lambda item: (
            item["best_validation_dice"],
            -LEARNING_RATES.index(item["learning_rate"]),
        ),
    )
    return candidates, chosen


def select_pilot(args: argparse.Namespace) -> None:
    """Freeze U-Net from its two validation cards and emit its full seed-0 job."""
    method = "unet"
    manifest_hash = file_digest(args.manifest)
    split_hash = file_digest(args.splits)
    production_inputs = _production_input_identity(args)
    sweep_cells = [
        (
            learning_rate,
            args.runs_root
            / "sweep"
            / method
            / f"lr_{lr_slug(learning_rate)}"
            / "seed0",
            args.generated_config_dir
            / "sweep"
            / f"{method}_lr{lr_slug(learning_rate)}.yaml",
        )
        for learning_rate in LEARNING_RATES
    ]
    candidates, chosen = select_native_sweep_evidence(
        sweep_cells,
        method=method,
        manifest_sha256=manifest_hash,
        splits_sha256=split_hash,
    )

    base_path = args.base_config_dir / f"{method}.yaml"
    base = load_base_config(args.base_config_dir, method)
    config = json.loads(json.dumps(base))
    config["seed"] = 0
    config["training"]["learning_rate"] = chosen["learning_rate"]
    config["training"]["epochs"] = FULL_EPOCHS
    config["training"]["patience"] = FULL_PATIENCE
    config["experiment"] = {
        "phase": "full_training",
        "learning_rate_selected_on": "val_best_dice",
        "selected_learning_rate": chosen["learning_rate"],
        "sweep_epochs": SWEEP_EPOCHS,
        "full_seeds": list(FULL_SEEDS),
    }
    _bind_new_training_config(config, production_inputs)
    config_path = args.generated_config_dir / "full" / f"{method}.yaml"
    write_yaml(config_path, config)

    selection: Dict[str, Any] = {
        "schema_version": 1,
        "phase": "pilot_learning_rate_selection",
        "method": method,
        "selection_split": "val",
        "selection_metric": "best_validation_dice",
        "tie_break": "lower_predeclared_learning_rate",
        "sweep_seed": 0,
        "sweep_epochs": SWEEP_EPOCHS,
        "learning_rate_candidates": list(LEARNING_RATES),
        "manifest_sha256": manifest_hash,
        "splits_sha256": split_hash,
        "production_inputs": production_inputs,
        "base_config": str(base_path),
        "base_config_sha256": file_digest(base_path),
        "candidates": candidates,
        "selected_learning_rate": chosen["learning_rate"],
        "selected_validation_dice": chosen["best_validation_dice"],
        "selected_candidate_index": candidates.index(chosen),
        "full_config": str(config_path),
        "full_config_sha256": file_digest(config_path),
        "full_seed": 0,
    }
    write_json(args.selection_out, selection)

    job = training_job(
        name=f"full.{method}.seed0",
        python_executable=args.python_executable,
        entrypoint=training_entrypoint(args.code_root, str(base["task"])),
        config=config_path,
        manifest=args.manifest,
        splits=args.splits,
        out=args.runs_root / "full" / method / "seed0",
        seed=0,
        extra_argv=(
            "--cache-dir",
            production_inputs["cache_dir"],
            "--threshold-amendment",
            production_inputs["threshold_amendment"],
        ),
        env=native_job_environment(method),
    )
    write_yaml(
        args.seed0_jobs_out,
        {
            "schema_version": 1,
            "phase": "full_training",
            "method": method,
            "seed": 0,
            "epochs": FULL_EPOCHS,
            "selection_split": "val",
            "selection": str(args.selection_out),
            "selection_sha256": file_digest(args.selection_out),
            "manifest_sha256": manifest_hash,
            "splits_sha256": split_hash,
            "config": str(config_path),
            "config_sha256": file_digest(config_path),
            "production_inputs": production_inputs,
            "jobs": [job],
        },
    )


def _native_sweep_config_index(path: Path, runs_root: Path) -> Dict[str, Path]:
    """Authenticate the mixed legacy/amended queue used for LR selection."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_names = [
        f"sweep.{method}.lr{lr_slug(rate)}.seed0"
        for method in METHOD_PRIORITY
        if method != "unet"
        for rate in LEARNING_RATES
    ]
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "stage3_concurrency_supplemental_queue"
        or payload.get("phase") != "learning_rate_sweep_remaining_native"
        or not isinstance(jobs, list)
        or [item.get("name") for item in jobs if isinstance(item, dict)]
        != expected_names
    ):
        raise ValueError("native sweep queue is not the authenticated adoption universe")
    result: Dict[str, Path] = {}
    for item in jobs:
        if not isinstance(item, dict):
            raise ValueError("native sweep queue contains a non-mapping job")
        name = str(item["name"])
        method = name.split(".")[1]
        slug = name.split(".")[2][2:]
        expected_out = runs_root / "sweep" / method / f"lr_{slug}" / "seed0"
        if (
            argv_value(item, "--resume") != "auto"
            or argv_value(item, "--seed") != "0"
            or Path(argv_value(item, "--out")).resolve() != expected_out.resolve()
            or Path(str(item.get("completion_artifact", ""))).resolve()
            != (expected_out / "run.json").resolve()
        ):
            raise ValueError(f"{name}: native sweep queue identity changed")
        config = Path(argv_value(item, "--config"))
        if not config.is_file():
            raise FileNotFoundError(f"{name}: native sweep config is absent")
        result[name] = config
    if set(result) != set(expected_names):
        raise ValueError("native sweep config index is incomplete")
    return result


def _authenticate_frozen_unet_config(
    path: Path, expected: Mapping[str, Any]
) -> Dict[str, str]:
    """Authenticate, but never rewrite, the already-run U-Net seed-0 config.

    The pilot may predate the expanded threshold grid and faster loader policy.
    Those prospective-only policy fields are allowed to differ; every other
    scientific field must match the validation-selected run.
    """
    if not path.is_file():
        raise FileNotFoundError(
            "U-Net pilot config is absent; select-pilot must complete first"
        )
    observed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(observed, dict):
        raise ValueError("frozen U-Net pilot config is not a mapping")
    observed_copy = json.loads(json.dumps(observed))
    expected_copy = json.loads(json.dumps(expected))
    observed_training = observed_copy.get("training")
    expected_training = expected_copy.get("training")
    if not isinstance(observed_training, dict) or not isinstance(expected_training, dict):
        raise ValueError("frozen U-Net pilot config lacks training settings")
    observed_grid = tuple(
        float(value) for value in observed_training.pop("threshold_candidates", ())
    )
    if observed_grid not in {LEGACY_THRESHOLDS, THRESHOLDS}:
        raise ValueError("frozen U-Net pilot config has an unregistered threshold grid")
    expected_training.pop("threshold_candidates", None)
    for key in PROTOCOL_DATALOADER:
        observed_training.pop(key, None)
        expected_training.pop(key, None)
    binding_keys = {
        "queue_batch_contract",
        "queue_batch_contract_sha256",
        "preprocess_cache_receipt_sha256",
        "threshold_amendment_sha256",
    }
    for payload in (observed_copy, expected_copy):
        runtime = payload.get("runtime")
        if isinstance(runtime, dict):
            for key in binding_keys:
                runtime.pop(key, None)
    if observed_copy != expected_copy:
        raise ValueError("frozen U-Net pilot config differs from global selection")
    return {"path": str(path), "sha256": file_digest(path)}


def select_full(args: argparse.Namespace) -> None:
    """Select rates from validation cards and prepare fresh three-seed jobs."""
    manifest_hash = file_digest(args.manifest)
    split_hash = file_digest(args.splits)
    production_inputs = _production_input_identity(args)
    production_argv = (
        "--cache-dir",
        production_inputs["cache_dir"],
        "--threshold-amendment",
        production_inputs["threshold_amendment"],
    )
    native_sweep_configs = _native_sweep_config_index(
        args.native_sweep_queue, args.runs_root
    )
    selection: Dict[str, Any] = {
        "schema_version": 1,
        "selection_split": "val",
        "selection_metric": "best_validation_dice",
        "tie_break": "lower_predeclared_learning_rate",
        "manifest_sha256": manifest_hash,
        "splits_sha256": split_hash,
        "production_inputs": production_inputs,
        "native_sweep_queue": {
            "path": str(args.native_sweep_queue),
            "sha256": file_digest(args.native_sweep_queue),
        },
        "methods": {},
        "workflow_notes": workflow_notes(),
    }
    pilot_jobs = []
    remaining_seed0_jobs = []
    seed12_jobs = []
    for method in METHOD_PRIORITY:
        candidates = []
        for learning_rate in LEARNING_RATES:
            slug = lr_slug(learning_rate)
            run_dir = args.runs_root / "sweep" / method / f"lr_{slug}" / "seed0"
            source_config = (
                args.generated_config_dir / "sweep" / f"{method}_lr{slug}.yaml"
                if method == "unet"
                else native_sweep_configs[
                    f"sweep.{method}.lr{slug}.seed0"
                ]
            )
            run = validate_sweep_run(
                run_dir,
                method=method,
                learning_rate=learning_rate,
                source_config=source_config,
                manifest_sha256=manifest_hash,
                splits_sha256=split_hash,
            )
            selector = None
            validation_dice = float(run["best_validation_dice"])
            selected_threshold = float(run["selected_validation_threshold"])
            source_payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
            source_training = (
                source_payload.get("training", {})
                if isinstance(source_payload, dict)
                else {}
            )
            source_grid = tuple(
                float(value)
                for value in source_training.get("threshold_candidates", ())
            )
            if method == "unetpp" and source_grid == LEGACY_THRESHOLDS:
                selector = _legacy_unetpp_selector(
                    getattr(args, "unetpp_selector_receipts", None),
                    source_config=source_config,
                    checkpoint=run_dir / "best.pt",
                    manifest_sha256=manifest_hash,
                    splits_sha256=split_hash,
                    amendment=getattr(args, "threshold_amendment", None),
                )
                validation_dice = selector["selected_validation_dice"]
                selected_threshold = selector["selected_threshold"]
            candidates.append(
                {
                    "learning_rate": learning_rate,
                    "best_validation_dice": validation_dice,
                    "best_epoch": int(run["best_epoch"]),
                    "selected_validation_threshold": selected_threshold,
                    "config_sha256": run["config_sha256"],
                    "best_checkpoint_sha256": run["checkpoint_sha256"]["best.pt"],
                    "run_card": str(run_dir / "run.json"),
                    **(
                        {
                            "legacy_run_metric_selection_ineligible": True,
                            "legacy_run_validation_dice": float(
                                run["best_validation_dice"]
                            ),
                            "expanded_grid_selector_receipt": selector,
                        }
                        if selector is not None
                        else {}
                    ),
                }
            )
        chosen = max(
            candidates,
            key=lambda item: (
                item["best_validation_dice"],
                -LEARNING_RATES.index(item["learning_rate"]),
            ),
        )
        base = load_base_config(args.base_config_dir, method)
        config = json.loads(json.dumps(base))
        config["training"]["learning_rate"] = chosen["learning_rate"]
        config["training"]["epochs"] = FULL_EPOCHS
        config["training"]["patience"] = FULL_PATIENCE
        config["experiment"] = {
            "phase": "full_training",
            "learning_rate_selected_on": "val_best_dice",
            "selected_learning_rate": chosen["learning_rate"],
            "sweep_epochs": SWEEP_EPOCHS,
            "full_seeds": list(FULL_SEEDS),
        }
        reused_seed0: Optional[Dict[str, str]] = None
        if method == "unet":
            legacy_path = args.generated_config_dir / "full" / "unet.yaml"
            reused_seed0 = _authenticate_frozen_unet_config(legacy_path, config)
            config_path = args.generated_config_dir / "full_amended" / "unet.yaml"
        else:
            config_path = args.generated_config_dir / "full" / f"{method}.yaml"
        _bind_new_training_config(config, production_inputs)
        write_yaml(config_path, config)
        selection["methods"][method] = {
            "candidates": candidates,
            "selected_learning_rate": chosen["learning_rate"],
            "selected_validation_dice": chosen["best_validation_dice"],
            "full_config": str(config_path),
            "full_config_sha256": file_digest(config_path),
            **(
                {
                    "reused_seed0_config": reused_seed0["path"],
                    "reused_seed0_config_sha256": reused_seed0["sha256"],
                }
                if reused_seed0 is not None
                else {}
            ),
        }
        for seed in FULL_SEEDS:
            out = args.runs_root / "full" / method / f"seed{seed}"
            job_config = (
                Path(reused_seed0["path"])
                if method == "unet" and seed == 0 and reused_seed0 is not None
                else config_path
            )
            job = training_job(
                name=f"full.{method}.seed{seed}",
                python_executable=args.python_executable,
                entrypoint=training_entrypoint(args.code_root, str(base["task"])),
                config=job_config,
                manifest=args.manifest,
                splits=args.splits,
                out=out,
                seed=seed,
                extra_argv=production_argv,
                env=native_job_environment(method),
            )
            if seed == 0 and method == "unet":
                pilot_jobs.append(job)
            elif seed == 0:
                remaining_seed0_jobs.append(job)
            else:
                seed12_jobs.append(job)
    sufs_config = args.base_config_dir / "sufs_ft.yaml"
    sufs_run_paths = [
        validate_sufs_sweep_run(
            args.runs_root / "sweep" / "sufs_ft" / f"lr_{lr_slug(rate)}" / "seed0",
            learning_rate=rate,
            source_config=sufs_config,
            manifest_sha256=manifest_hash,
            splits_sha256=split_hash,
        )
        for rate in LEARNING_RATES
    ]
    # Import the SUFS workflow's own stricter semantic validator rather than
    # reimplementing its released-weight/protocol checks here.
    from select_sufs_lr import select as select_sufs_learning_rate

    sufs_selection = select_sufs_learning_rate(sufs_run_paths)
    sufs_selection_path = args.selection_out.with_name("sufs_ft_selection.json")
    write_json(sufs_selection_path, sufs_selection)
    selection["methods"]["sufs_ft"] = {
        "workflow": "separate_pyfing_keras_torch",
        "selected_learning_rate": sufs_selection["selected_learning_rate"],
        "selected_validation_dice": sufs_selection["selected_validation_dice"],
        "selection_artifact": str(sufs_selection_path),
        "selection_artifact_sha256": file_digest(sufs_selection_path),
        "source_config": str(sufs_config),
        "source_config_sha256": file_digest(sufs_config),
    }
    for seed in FULL_SEEDS:
        out = args.runs_root / "full" / "sufs_ft" / f"seed{seed}"
        job = training_job(
            name=f"full.sufs_ft.seed{seed}",
            python_executable=args.python_executable,
            entrypoint=args.code_root / "scripts" / "train_sufs.py",
            config=sufs_config,
            manifest=args.manifest,
            splits=args.splits,
            out=out,
            seed=seed,
            extra_argv=(
                "--mode",
                "full",
                "--selection",
                str(sufs_selection_path),
                *production_argv,
            ),
            env={"KERAS_BACKEND": "torch"},
        )
        (remaining_seed0_jobs if seed == 0 else seed12_jobs).append(job)
    write_json(args.selection_out, selection)
    common = {
        "schema_version": 1,
        "phase": "full_training",
        "selection_sha256": file_digest(args.selection_out),
        "production_inputs": production_inputs,
        "epochs": FULL_EPOCHS,
        "workflow_notes": workflow_notes(),
    }
    write_yaml(args.pilot_seed0_jobs_out, {**common, "jobs": pilot_jobs})
    write_yaml(
        args.remaining_seed0_jobs_out,
        {**common, "jobs": remaining_seed0_jobs},
    )
    write_yaml(args.seed12_jobs_out, {**common, "jobs": seed12_jobs})


def load_json_mapping(path: Path, description: str) -> Dict[str, Any]:
    """Load one required JSON mapping with a useful fail-closed error."""
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def load_queue(
    path: Path,
    *,
    description: str,
    selection_sha256: str,
    expected_names: Sequence[str],
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Load a raw planner queue and require its exact registered job universe."""
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ValueError(f"{description} is not a planner queue: {path}")
    if payload.get("selection_sha256") != selection_sha256:
        raise ValueError(f"{description} does not identify the global selection")
    jobs = payload["jobs"]
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError(f"{description} contains a non-mapping job")
    names = [job.get("name") for job in jobs]
    if names != list(expected_names):
        raise ValueError(
            f"{description} job universe/order changed: {names!r}, "
            f"expected {list(expected_names)!r}"
        )
    return payload, jobs


def argv_value(job: Mapping[str, Any], option: str) -> str:
    """Return one unambiguous argv value from a queue job."""
    argv = job.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise ValueError(f"{job.get('name')}: argv must be a list of strings")
    positions = [index for index, value in enumerate(argv) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"{job.get('name')}: expected exactly one {option} value")
    return argv[positions[0] + 1]


def validate_queue_job(
    job: Mapping[str, Any],
    *,
    global_selection: Mapping[str, Any],
    runs_root: Path,
) -> Tuple[str, int]:
    """Validate method, seed, config and output identities encoded in one job."""
    name = job.get("name")
    if not isinstance(name, str):
        raise ValueError("queue job lacks a string name")
    match = JOB_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"unregistered full-training job name {name!r}")
    method, seed_text = match.groups()
    seed = int(seed_text)
    if method not in (*METHOD_PRIORITY, "sufs_ft"):
        raise ValueError(f"{name}: unregistered method")
    if int(argv_value(job, "--seed")) != seed:
        raise ValueError(f"{name}: argv seed disagrees with job name")
    if argv_value(job, "--resume") != "auto":
        raise ValueError(f"{name}: full training is not resumable")
    expected_out = runs_root / "full" / method / f"seed{seed}"
    if Path(argv_value(job, "--out")) != expected_out:
        raise ValueError(f"{name}: output path is not the registered run directory")

    production = global_selection.get("production_inputs")
    if not isinstance(production, Mapping):
        raise ValueError(f"{name}: global selection lacks production input identities")
    cache_dir = Path(str(production.get("cache_dir", "")))
    receipt = Path(str(production.get("cache_receipt", "")))
    amendment = Path(str(production.get("threshold_amendment", "")))
    if (
        Path(argv_value(job, "--cache-dir")).resolve() != cache_dir.resolve()
        or Path(argv_value(job, "--threshold-amendment")).resolve()
        != amendment.resolve()
        or not receipt.is_file()
        or file_digest(receipt) != production.get("cache_receipt_sha256")
        or not amendment.is_file()
        or file_digest(amendment)
        != production.get("threshold_amendment_sha256")
    ):
        raise ValueError(f"{name}: production cache/amendment binding changed")

    methods = global_selection.get("methods")
    if not isinstance(methods, dict) or method not in methods:
        raise ValueError(f"{name}: method absent from global selection")
    method_selection = methods[method]
    if method == "sufs_ft":
        expected_config = Path(str(method_selection["source_config"]))
        expected_digest = str(method_selection["source_config_sha256"])
        if argv_value(job, "--selection") != str(method_selection["selection_artifact"]):
            raise ValueError(f"{name}: SUFS selection artifact disagrees")
        if job.get("env") != {"KERAS_BACKEND": "torch"}:
            raise ValueError(f"{name}: SUFS backend environment changed")
    else:
        reused_unet = method == "unet" and seed == 0
        config_key = "reused_seed0_config" if reused_unet else "full_config"
        digest_key = (
            "reused_seed0_config_sha256" if reused_unet else "full_config_sha256"
        )
        expected_config = Path(str(method_selection[config_key]))
        expected_digest = str(method_selection[digest_key])
    if Path(argv_value(job, "--config")) != expected_config:
        raise ValueError(f"{name}: selected config path disagrees")
    if not expected_config.is_file() or file_digest(expected_config) != expected_digest:
        raise ValueError(f"{name}: selected config hash failure")
    return method, seed


def validate_checkpoints(run_dir: Path, run: Mapping[str, Any]) -> Dict[str, str]:
    """Require exactly the trainable run's best and last checkpoint hashes."""
    hashes = run.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {"best.pt", "last.pt"}:
        raise ValueError(f"{run_dir / 'run.json'}: incomplete checkpoint provenance")
    for name in ("best.pt", "last.pt"):
        checkpoint = run_dir / name
        if not checkpoint.is_file() or file_digest(checkpoint) != hashes[name]:
            raise ValueError(f"{run_dir / 'run.json'}: checkpoint hash failure for {name}")
    return {name: str(hashes[name]) for name in ("best.pt", "last.pt")}


def validate_completed_full_run(
    run_dir: Path,
    *,
    method: str,
    seed: int,
    manifest_sha256: str,
    splits_sha256: str,
    source_config_sha256: str,
    selected_learning_rate: float,
    selection_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Prove an omitted job already completed under the exact selected identity."""
    run_path = run_dir / "run.json"
    run = load_json_mapping(run_path, f"completed {method} seed-{seed} run card")
    expected: Dict[str, Any] = {
        "method": method,
        "seed": seed,
        "epochs_requested": FULL_EPOCHS,
        "manifest_sha256": manifest_sha256,
        "split_sha256": splits_sha256,
        "source_config_sha256": source_config_sha256,
    }
    if method == "sufs_ft":
        expected["mode"] = "full"
    for key, value in expected.items():
        if run.get(key) != value:
            raise ValueError(
                f"{run_path}: {key}={run.get(key)!r}, expected {value!r}"
            )
    completed = run.get("epochs_completed")
    if not isinstance(completed, int) or not 1 <= completed <= FULL_EPOCHS:
        raise ValueError(f"{run_path}: invalid completed epoch count {completed!r}")
    if run.get("debug_limited"):
        raise ValueError(f"{run_path}: debug-limited run cannot be omitted")
    if not math.isclose(
        float(run.get("learning_rate", math.nan)),
        selected_learning_rate,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{run_path}: learning rate disagrees with selection")
    if selection_sha256 is not None and run.get("selection_artifact_sha256") != selection_sha256:
        raise ValueError(f"{run_path}: selection artifact identity disagrees")
    checkpoint_hashes = validate_checkpoints(run_dir, run)
    return {
        "run_card": str(run_path),
        "run_card_sha256": file_digest(run_path),
        "epochs_completed": completed,
        "checkpoint_sha256": checkpoint_hashes,
    }


def canonical_native_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize pilot/global U-Net evidence without weakening any identity."""
    checkpoint = candidate.get("checkpoint_sha256")
    best_checkpoint = (
        checkpoint.get("best.pt") if isinstance(checkpoint, dict) else None
    ) or candidate.get("best_checkpoint_sha256")
    run_card = Path(str(candidate.get("run_card")))
    if not run_card.is_file():
        raise FileNotFoundError(f"missing U-Net sweep run card {run_card}")
    recorded_run_hash = candidate.get("run_card_sha256")
    observed_run_hash = file_digest(run_card)
    if recorded_run_hash is not None and recorded_run_hash != observed_run_hash:
        raise ValueError(f"U-Net sweep run-card hash failure for {run_card}")
    return {
        "learning_rate": float(candidate["learning_rate"]),
        "best_validation_dice": float(candidate["best_validation_dice"]),
        "best_epoch": int(candidate["best_epoch"]),
        "selected_validation_threshold": float(candidate["selected_validation_threshold"]),
        "config_sha256": str(candidate["config_sha256"]),
        "best_checkpoint_sha256": str(best_checkpoint),
        "run_card": str(run_card),
        "run_card_sha256": observed_run_hash,
    }


def validate_staged_selections(
    global_selection_path: Path,
    pilot_selection_path: Path,
    sufs_selection_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Require global choices to exactly reproduce both earlier audited choices."""
    global_selection = load_json_mapping(global_selection_path, "global selection")
    pilot = load_json_mapping(pilot_selection_path, "U-Net pilot selection")
    existing_sufs = load_json_mapping(sufs_selection_path, "existing SUFS selection")
    methods = global_selection.get("methods")
    expected_methods = {*METHOD_PRIORITY, "sufs_ft"}
    if not isinstance(methods, dict) or set(methods) != expected_methods:
        raise ValueError("global selection does not contain the exact registered methods")
    for artifact, description in ((pilot, "pilot"), (existing_sufs, "SUFS")):
        if artifact.get("manifest_sha256") != global_selection.get("manifest_sha256"):
            raise ValueError(f"{description} and global selections disagree on manifest")
        split_key = "splits_sha256" if description == "pilot" else "split_sha256"
        if artifact.get(split_key) != global_selection.get("splits_sha256"):
            raise ValueError(f"{description} and global selections disagree on split")

    global_unet = methods["unet"]
    if pilot.get("method") != "unet":
        raise ValueError("pilot selection is not for U-Net")
    for key in ("selected_learning_rate", "selected_validation_dice"):
        if pilot.get(key) != global_unet.get(key):
            raise ValueError(f"pilot and global U-Net selections disagree on {key}")
    if (
        pilot.get("full_config") != global_unet.get("reused_seed0_config")
        or pilot.get("full_config_sha256")
        != global_unet.get("reused_seed0_config_sha256")
    ):
        raise ValueError("pilot and global U-Net reused seed-0 config disagree")
    pilot_candidates = sorted(
        (canonical_native_candidate(item) for item in pilot.get("candidates", [])),
        key=lambda item: item["learning_rate"],
    )
    global_candidates = sorted(
        (canonical_native_candidate(item) for item in global_unet.get("candidates", [])),
        key=lambda item: item["learning_rate"],
    )
    if len(pilot_candidates) != len(LEARNING_RATES) or pilot_candidates != global_candidates:
        raise ValueError("pilot and global U-Net candidate evidence differs")

    global_sufs = methods["sufs_ft"]
    global_sufs_path = Path(str(global_sufs.get("selection_artifact")))
    if global_sufs_path.resolve() == sufs_selection_path.resolve():
        raise ValueError(
            "global SUFS selection must be written separately from the pre-existing artifact"
        )
    computed_sufs = load_json_mapping(global_sufs_path, "global SUFS selection")
    if file_digest(global_sufs_path) != global_sufs.get("selection_artifact_sha256"):
        raise ValueError("global SUFS selection artifact hash failure")
    if computed_sufs != existing_sufs:
        raise ValueError("global and existing SUFS selections differ")
    for key in ("selected_learning_rate", "selected_validation_dice"):
        if global_sufs.get(key) != existing_sufs.get(key):
            raise ValueError(f"global SUFS selection disagrees on {key}")
    return global_selection, pilot, existing_sufs


def dedupe_full(args: argparse.Namespace) -> None:
    """Emit only not-yet-run full jobs after proving staged completions."""
    raise RuntimeError(
        "dedupe-full was retired because it interleaves seed 0 with seeds 1/2; "
        "use authorize_deep_jobs.py prepare-seed0 then authorize-seed12"
    )
    global_selection, pilot, existing_sufs = validate_staged_selections(
        args.global_selection,
        args.pilot_selection,
        args.sufs_selection,
    )
    global_digest = file_digest(args.global_selection)
    expected_pilot = ["full.unet.seed0"]
    expected_remaining0 = [
        *(f"full.{method}.seed0" for method in METHOD_PRIORITY if method != "unet"),
        "full.sufs_ft.seed0",
    ]
    expected_seed12 = [
        *(f"full.{method}.seed{seed}" for method in METHOD_PRIORITY for seed in (1, 2)),
        "full.sufs_ft.seed1",
        "full.sufs_ft.seed2",
    ]
    _, pilot_jobs = load_queue(
        args.pilot_seed0_jobs,
        description="pilot seed-0 queue",
        selection_sha256=global_digest,
        expected_names=expected_pilot,
    )
    _, remaining0_jobs = load_queue(
        args.remaining_seed0_jobs,
        description="remaining seed-0 queue",
        selection_sha256=global_digest,
        expected_names=expected_remaining0,
    )
    _, seed12_jobs = load_queue(
        args.seed12_jobs,
        description="seed-1/2 queue",
        selection_sha256=global_digest,
        expected_names=expected_seed12,
    )
    all_jobs = [*pilot_jobs, *remaining0_jobs, *seed12_jobs]
    all_names = [str(job["name"]) for job in all_jobs]
    if len(all_names) != len(set(all_names)):
        raise ValueError("raw full queues contain duplicate job names")
    for job in all_jobs:
        validate_queue_job(job, global_selection=global_selection, runs_root=args.runs_root)

    manifest_hash = str(global_selection["manifest_sha256"])
    split_hash = str(global_selection["splits_sha256"])
    global_unet = global_selection["methods"]["unet"]
    completed: Dict[str, Any] = {
        "full.unet.seed0": validate_completed_full_run(
            args.runs_root / "full" / "unet" / "seed0",
            method="unet",
            seed=0,
            manifest_sha256=manifest_hash,
            splits_sha256=split_hash,
            source_config_sha256=str(global_unet["reused_seed0_config_sha256"]),
            selected_learning_rate=float(global_unet["selected_learning_rate"]),
        )
    }
    global_sufs = global_selection["methods"]["sufs_ft"]
    existing_sufs_digest = file_digest(args.sufs_selection)
    for seed in FULL_SEEDS:
        name = f"full.sufs_ft.seed{seed}"
        completed[name] = validate_completed_full_run(
            args.runs_root / "full" / "sufs_ft" / f"seed{seed}",
            method="sufs_ft",
            seed=seed,
            manifest_sha256=manifest_hash,
            splits_sha256=split_hash,
            source_config_sha256=str(global_sufs["source_config_sha256"]),
            selected_learning_rate=float(existing_sufs["selected_learning_rate"]),
            selection_sha256=existing_sufs_digest,
        )

    excluded_names = ["full.unet.seed0", *(f"full.sufs_ft.seed{seed}" for seed in FULL_SEEDS)]
    retained_jobs = [
        job for job in [*remaining0_jobs, *seed12_jobs] if job["name"] not in excluded_names
    ]
    expected_retained = [
        *(f"full.{method}.seed0" for method in METHOD_PRIORITY if method != "unet"),
        *(f"full.{method}.seed{seed}" for method in METHOD_PRIORITY for seed in (1, 2)),
    ]
    retained_names = [str(job["name"]) for job in retained_jobs]
    if retained_names != expected_retained:
        raise ValueError("filtered full queue does not equal the registered 23-job plan")

    queue = {
        "schema_version": 1,
        "phase": "full_training",
        "purpose": "staged_deduplicated_native_runs",
        "global_selection": str(args.global_selection),
        "global_selection_sha256": global_digest,
        "pilot_selection": str(args.pilot_selection),
        "pilot_selection_sha256": file_digest(args.pilot_selection),
        "sufs_selection": str(args.sufs_selection),
        "sufs_selection_sha256": existing_sufs_digest,
        "excluded_completed_jobs": excluded_names,
        "jobs": retained_jobs,
    }
    write_yaml(args.out, queue)
    provenance = {
        "schema_version": 1,
        "operation": "deduplicate_full_training_queue",
        "inputs": {
            "global_selection": {"path": str(args.global_selection), "sha256": global_digest},
            "pilot_selection": {
                "path": str(args.pilot_selection),
                "sha256": file_digest(args.pilot_selection),
            },
            "sufs_selection": {
                "path": str(args.sufs_selection),
                "sha256": existing_sufs_digest,
            },
            "pilot_seed0_jobs": {
                "path": str(args.pilot_seed0_jobs),
                "sha256": file_digest(args.pilot_seed0_jobs),
            },
            "remaining_seed0_jobs": {
                "path": str(args.remaining_seed0_jobs),
                "sha256": file_digest(args.remaining_seed0_jobs),
            },
            "seed12_jobs": {
                "path": str(args.seed12_jobs),
                "sha256": file_digest(args.seed12_jobs),
            },
        },
        "selection_agreement": {
            "unet_learning_rate": pilot["selected_learning_rate"],
            "unet_full_config_sha256": pilot["full_config_sha256"],
            "sufs_learning_rate": existing_sufs["selected_learning_rate"],
            "sufs_existing_selection_sha256": existing_sufs_digest,
            "sufs_global_selection_sha256": global_sufs["selection_artifact_sha256"],
        },
        "completed_jobs": completed,
        "excluded_jobs": excluded_names,
        "retained_jobs": retained_names,
        "output": {"path": str(args.out), "sha256": file_digest(args.out)},
    }
    write_json(args.provenance_out, provenance)


def add_common(parser: argparse.ArgumentParser) -> None:
    """Add path arguments shared by both phases."""
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--base-config-dir", required=True, type=Path)
    parser.add_argument("--generated-config-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--python-executable", default=sys.executable)


def add_production_inputs(parser: argparse.ArgumentParser) -> None:
    """Add the immutable cache/amendment inputs required by new jobs."""
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--threshold-amendment", required=True, type=Path)
    parser.add_argument("--expected-threshold-amendment-sha256", required=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the sweep and validation-selection planner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep = subparsers.add_parser("prepare-sweep")
    add_common(sweep)
    add_production_inputs(sweep)
    sweep.add_argument("--pilot-jobs-out", required=True, type=Path)
    sweep.add_argument("--remaining-jobs-out", required=True, type=Path)
    pilot = subparsers.add_parser("select-pilot")
    add_common(pilot)
    add_production_inputs(pilot)
    pilot.add_argument("--selection-out", required=True, type=Path)
    pilot.add_argument(
        "--seed0-jobs-out",
        "--pilot-seed0-jobs-out",
        dest="seed0_jobs_out",
        required=True,
        type=Path,
    )
    full = subparsers.add_parser("select-full")
    add_common(full)
    add_production_inputs(full)
    full.add_argument("--selection-out", required=True, type=Path)
    full.add_argument("--pilot-seed0-jobs-out", required=True, type=Path)
    full.add_argument("--remaining-seed0-jobs-out", required=True, type=Path)
    full.add_argument("--seed12-jobs-out", required=True, type=Path)
    full.add_argument("--native-sweep-queue", required=True, type=Path)
    full.add_argument(
        "--unetpp-selector-receipts",
        nargs=2,
        type=Path,
        help=(
            "expanded-grid validation selector receipts for the two preserved "
            "U-Net++ sweep checkpoints"
        ),
    )
    dedupe = subparsers.add_parser("dedupe-full")
    dedupe.add_argument("--global-selection", required=True, type=Path)
    dedupe.add_argument("--pilot-selection", required=True, type=Path)
    dedupe.add_argument("--sufs-selection", required=True, type=Path)
    dedupe.add_argument("--pilot-seed0-jobs", required=True, type=Path)
    dedupe.add_argument("--remaining-seed0-jobs", required=True, type=Path)
    dedupe.add_argument("--seed12-jobs", required=True, type=Path)
    dedupe.add_argument("--runs-root", required=True, type=Path)
    dedupe.add_argument("--out", required=True, type=Path)
    dedupe.add_argument("--provenance-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    if args.command == "prepare-sweep":
        prepare_sweep(args)
    elif args.command == "select-pilot":
        select_pilot(args)
    elif args.command == "select-full":
        select_full(args)
    else:
        dedupe_full(args)


if __name__ == "__main__":
    main()
