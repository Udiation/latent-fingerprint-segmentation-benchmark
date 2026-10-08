#!/usr/bin/env python3
"""Fine-tune released PyFing SUFS weights on train/validation data only.

Sweep runs are fixed to seed 0 and 30 epochs at one of the two preregistered
learning rates.  Full runs require the validation-only selection artifact made
by ``select_sufs_lr.py``.  Test rows and test masks are never opened here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

if os.environ.get("KERAS_BACKEND", "torch").lower() != "torch":
    raise RuntimeError("train_sufs.py requires KERAS_BACKEND=torch")
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from torch.utils.data import DataLoader

import _path  # noqa: F401
from segbench.deep_utils import (
    EpochRandomSampler,
    PixelCropDataset,
    atomic_json,
    atomic_torch_save,
    capture_rng_state,
    compute_train_normalization,
    config_digest,
    configure_memory_fraction,
    file_digest,
    load_config,
    load_split_rows,
    loader_generator,
    read_pair,
    restore_rng_state,
    resume_config_digest,
    run_provenance,
    seed_everything,
    seed_worker,
    select_validation_threshold,
    EpochProfiler,
    acceleration_policy,
    apply_queue_batch_contract,
    protocol_dataloader_settings,
    publish_run_with_profile,
    queue_batch_metadata,
    validate_resume_batch_transition,
)
from segbench import preprocess_cache, probability_artifacts
from segbench.sufs_finetune import (
    build_released_sufs,
    predict_sufs_probability,
    protocol_config_digest,
    state_dict_cpu,
    sufs_probability_loss,
    validate_learning_rate,
    validate_shared_training_protocol,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the auditable sweep/full training interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("sweep", "full"))
    parser.add_argument("--learning-rate", type=float, help="required only for sweep")
    parser.add_argument("--selection", type=Path, help="required only for full runs")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--resume", default="none", help="none, auto, or a last.pt path")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--acceleration-receipt", type=Path)
    parser.add_argument("--threshold-amendment", type=Path)
    parser.add_argument("--allow-unamended-debug", action="store_true")
    parser.add_argument(
        "--acceleration-evaluation",
        choices=("fp32", "bf16_channels_last"),
    )
    return parser.parse_args(argv)


def _resolved_config(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Resolve a frozen sweep or selected full-run configuration."""
    loaded = load_config(args.config)
    if loaded["method"] != "sufs_ft" or loaded["task"] != "pixel":
        raise ValueError("train_sufs.py accepts only the sufs_ft pixel config")
    validate_shared_training_protocol(loaded)
    config: Dict[str, Any] = json.loads(json.dumps(loaded))
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    config["seed"] = seed
    training = config["training"]
    selection: Optional[Dict[str, Any]] = None
    if args.mode == "sweep":
        if args.selection is not None:
            raise ValueError("sweep runs cannot consume a selection artifact")
        if args.learning_rate is None:
            raise ValueError("--learning-rate is required for a sweep run")
        if seed != 0:
            raise ValueError("learning-rate sweeps are preregistered at seed 0 only")
        training["learning_rate"] = validate_learning_rate(args.learning_rate)
        training["epochs"] = int(training["sweep_epochs"])
        if training["epochs"] != 30:
            raise ValueError("SUFS sweep budget must be exactly 30 epochs")
    else:
        if args.learning_rate is not None:
            raise ValueError("full runs obtain their rate only from --selection")
        if args.selection is None:
            raise ValueError("--selection is required for a full run")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        training["learning_rate"] = validate_learning_rate(selection.get("selected_learning_rate"))
        if int(training["epochs"]) <= int(training["sweep_epochs"]):
            raise ValueError("full training budget must exceed the sweep budget")
    runtime = config["runtime"]
    if args.device is not None:
        runtime["device"] = args.device
    memory_override = (
        args.memory_fraction
        if args.memory_fraction is not None
        else os.environ.get("SEG_GPU_MEMORY_FRACTION")
    )
    if memory_override is not None:
        runtime["memory_fraction"] = float(memory_override)
    if float(runtime["memory_fraction"]) > 0.3:
        raise ValueError("SUFS CUDA memory fraction exceeds the project ceiling")
    return config, selection


def _selection_is_compatible(
    selection: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    manifest: Path,
    splits: Path,
    released: Mapping[str, Any],
) -> None:
    """Bind a full run to the exact validation-only LR sweep evidence."""
    expected = {
        "schema_version": 1,
        "method": "sufs_ft",
        "selection_split": "val",
        "selection_metric": "best_macro_dice",
        "protocol_config_sha256": protocol_config_digest(config),
        "source_config_sha256": file_digest(config_path),
        "manifest_sha256": file_digest(manifest),
        "split_sha256": file_digest(splits),
        "released_weights_sha256": released["weights_sha256"],
    }
    disagreements = {
        key: (selection.get(key), value)
        for key, value in expected.items()
        if selection.get(key) != value
    }
    if disagreements:
        raise ValueError(f"SUFS LR selection provenance mismatch: {disagreements}")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("SUFS LR selection must contain both preregistered candidates")


def _make_loader(
    dataset: PixelCropDataset,
    *,
    batch_size: int,
    settings: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> DataLoader[Any]:
    """Create an epoch-specific deterministic loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=EpochRandomSampler(dataset, seed),
        num_workers=int(settings["num_workers"]),
        pin_memory=bool(settings["pin_memory"]),
        drop_last=False,
        persistent_workers=bool(settings["persistent_workers"]),
        prefetch_factor=int(settings["prefetch_factor"]),
        worker_init_fn=seed_worker,
        generator=loader_generator(seed),
    )


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    training: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Train one epoch on shared crops and return mean batch loss."""
    model.train()
    accumulation = int(training["gradient_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    losses = []
    profiler = EpochProfiler(device)
    for batch_index, (images, targets) in enumerate(loader):
        profiler.batch_ready()
        images = images.permute(0, 2, 3, 1).to(device, non_blocking=True)
        targets = targets.permute(0, 2, 3, 1).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=acceleration["precision"] == "bf16",
        ):
            probability = model(images, training=True)
            loss = sufs_probability_loss(
                probability,
                targets,
                bce_weight=float(training["bce_weight"]),
                dice_weight=float(training["dice_weight"]),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite SUFS loss at batch {batch_index}")
        (loss / accumulation).backward()
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        profiler.compute_done()
    if not losses:
        raise RuntimeError("SUFS training loader emitted no batches")
    return float(np.mean(losses)), profiler.finish()


def _validate(
    segmenter: Any,
    rows: Any,
    device: torch.device,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    *,
    cache_dir: Optional[Path],
    acceleration: Mapping[str, Any],
) -> Tuple[float, float, Dict[str, float]]:
    """Select threshold and checkpoint only from full validation images."""
    probabilities = []
    targets = []
    data = config["data"]
    for row in rows.to_dict("records"):
        image, target = read_pair(
            row,
            float(data["target_ppi"]),
            str(data["mask_column"]),
            cache_dir=cache_dir,
        )
        normalized = (image - float(normalization["mean"])) / float(normalization["std"])
        probabilities.append(
            predict_sufs_probability(
                segmenter,
                normalized,
                device,
                autocast_bf16=acceleration["precision"] == "bf16",
            )
        )
        targets.append(target)
    return select_validation_threshold(
        probabilities,
        targets,
        config["training"]["threshold_candidates"],
    )


def _identity(
    *,
    mode: str,
    config: Mapping[str, Any],
    manifest: Path,
    splits: Path,
    released: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return fields that must match before a checkpoint can resume."""
    return {
        "mode": mode,
        "resume_config_sha256": resume_config_digest(config),
        "manifest_sha256": file_digest(manifest),
        "split_sha256": file_digest(splits),
        "released_weights_sha256": released["weights_sha256"],
    }


def _checkpoint(
    *,
    mode: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    identity: Mapping[str, Any],
    released: Mapping[str, Any],
    epoch: int,
    threshold: float,
    validation_dice: float,
    best_validation_dice: float,
    best_epoch: int,
    stale_epochs: int,
    history: Sequence[Mapping[str, Any]],
    resumable: bool,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create a portable Keras-Torch checkpoint with audit metadata."""
    queue_batch = queue_batch_metadata(config)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "method": "sufs_ft",
        "task": "pixel",
        "framework": "pyfing-keras-torch",
        "mode": mode,
        "config": dict(config),
        "config_sha256": config_digest(config),
        "resume_config_sha256": resume_config_digest(config),
        "queue_batch_contract_sha256": queue_batch["contract_sha256"],
        "queue_batch_active_state": queue_batch["active"],
        "protocol_config_sha256": protocol_config_digest(config),
        "identity": dict(identity),
        "normalization": dict(normalization),
        "released": dict(released),
        "epoch": int(epoch),
        "selected_threshold": float(threshold),
        "validation_dice": float(validation_dice),
        "best_validation_dice": float(best_validation_dice),
        "best_epoch": int(best_epoch),
        "stale_epochs": int(stale_epochs),
        "model_state": state_dict_cpu(model),
        "history": list(history),
        "provenance": dict(provenance),
    }
    if resumable:
        payload.update(
            {
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "rng_state": capture_rng_state(),
            }
        )
    return payload


def train(
    args: argparse.Namespace,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one resumable SUFS sweep or selected full run."""
    start_provenance = dict(
        provenance
        if provenance is not None
        else run_provenance(sys.argv)
    )
    config, selection = _resolved_config(args)
    queue_batch = apply_queue_batch_contract(config)
    threshold_amendment_sha256 = None
    if getattr(args, "threshold_amendment", None) is not None:
        threshold_amendment_sha256 = probability_artifacts.validate_threshold_amendment(
            args.threshold_amendment
        )
        config["runtime"]["threshold_amendment_sha256"] = (
            threshold_amendment_sha256
        )
    elif not (
        getattr(args, "allow_unamended_debug", False)
        and str(config["runtime"].get("device", "auto")) == "cpu"
        and int(args.limit_train) > 0
        and int(args.limit_val) > 0
    ):
        raise ValueError("production training requires the exact threshold amendment")
    acceleration = acceleration_policy(
        getattr(args, "acceleration_receipt", None),
        evaluation_mode=getattr(args, "acceleration_evaluation", None),
        epochs=int(config["training"]["epochs"]),
        expected_identity={
            "method": config["method"],
            "task": config["task"],
            "source_config_sha256": file_digest(args.config),
            "manifest_sha256": file_digest(args.manifest),
            "split_sha256": file_digest(args.splits),
        },
    )
    config["runtime"]["acceleration"] = acceleration
    seed_everything(int(config["seed"]), bool(config["runtime"].get("deterministic", True)))
    requested_device = str(config["runtime"].get("device", "auto"))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    configure_memory_fraction(device, float(config["runtime"]["memory_fraction"]))
    segmenter, model, released = build_released_sufs(config, device)
    if selection is not None:
        _selection_is_compatible(
            selection,
            config=config,
            config_path=args.config,
            manifest=args.manifest,
            splits=args.splits,
            released=released,
        )
    identity = _identity(
        mode=args.mode,
        config=config,
        manifest=args.manifest,
        splits=args.splits,
        released=released,
    )
    data = config["data"]
    train_rows = load_split_rows(
        args.manifest,
        args.splits,
        subset=str(data["subset"]),
        split="train",
        mask_column=str(data["mask_column"]),
        require_mask=True,
        limit=args.limit_train,
    )
    val_rows = load_split_rows(
        args.manifest,
        args.splits,
        subset=str(data["subset"]),
        split="val",
        mask_column=str(data["mask_column"]),
        require_mask=True,
        limit=args.limit_val,
    )
    cache_receipt_sha256 = None
    if args.cache_dir is not None:
        required_ids = [
            str(value)
            for value in list(train_rows["image_id"]) + list(val_rows["image_id"])
        ]
        preprocess_cache.load_receipt(
            args.cache_dir,
            manifest_sha256=file_digest(args.manifest),
            target_ppi=float(data["target_ppi"]),
            mask_columns=[str(data["mask_column"])],
            required_ids=required_ids,
            verify_artifact_hashes=True,
        )
        cache_receipt_sha256 = file_digest(args.cache_dir / "receipt.json")
        config["runtime"]["preprocess_cache_receipt_sha256"] = cache_receipt_sha256
    normalization = compute_train_normalization(
        train_rows, float(data["target_ppi"]), cache_dir=args.cache_dir
    )
    dataset = PixelCropDataset(
        train_rows,
        target_ppi=float(data["target_ppi"]),
        mask_column=str(data["mask_column"]),
        crop_size=int(data["crop_size"]),
        normalization=normalization,
        augmentation=config["augmentation"],
        seed=int(config["seed"]),
        cache_dir=args.cache_dir,
    )
    training = config["training"]
    loader_settings = protocol_dataloader_settings(training)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training["epochs"]),
        eta_min=float(training["min_learning_rate"]),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    best_path = args.out / "best.pt"
    last_path = args.out / "last.pt"
    history_path = args.out / "history.json"
    run_path = args.out / "run.json"
    start_epoch = 0
    best_dice = -math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[Dict[str, Any]] = []
    resume_path: Optional[Path] = None
    if args.resume == "auto" and last_path.is_file():
        resume_path = last_path
    elif args.resume not in {"none", "auto"}:
        resume_path = Path(args.resume)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != identity:
            raise ValueError("SUFS resume checkpoint scientific identity differs")
        if checkpoint.get("normalization") != normalization:
            raise ValueError("SUFS resume checkpoint train normalization differs")
        validate_resume_batch_transition(checkpoint, config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_validation_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = [dict(item) for item in checkpoint.get("history", [])]
    epochs = int(training["epochs"])
    started = time.monotonic()
    if args.mode == "full" and stale_epochs >= int(training["patience"]):
        start_epoch = epochs
    loader = _make_loader(
        dataset,
        batch_size=int(training["batch_size"]),
        settings=loader_settings,
        seed=int(config["seed"]) * 1_000_003,
        device=device,
    )
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        train_loss, epoch_profile = _train_epoch(
            model, loader, device, optimizer, training, acceleration
        )
        threshold, val_dice, threshold_scores = _validate(
            segmenter,
            val_rows,
            device,
            config,
            normalization,
            cache_dir=args.cache_dir,
            acceleration=acceleration,
        )
        improved = val_dice > best_dice + 1e-12
        if improved:
            best_dice = val_dice
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_dice": val_dice,
            "selected_threshold": threshold,
            "threshold_scores": threshold_scores,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_profile": epoch_profile,
        }
        history.append(entry)
        if improved:
            atomic_torch_save(
                best_path,
                _checkpoint(
                    mode=args.mode,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    normalization=normalization,
                    identity=identity,
                    released=released,
                    epoch=epoch,
                    threshold=threshold,
                    validation_dice=val_dice,
                    best_validation_dice=best_dice,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    history=history,
                    resumable=False,
                    provenance=start_provenance,
                ),
            )
        scheduler.step()
        atomic_torch_save(
            last_path,
            _checkpoint(
                mode=args.mode,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                normalization=normalization,
                identity=identity,
                released=released,
                epoch=epoch,
                threshold=threshold,
                validation_dice=val_dice,
                best_validation_dice=best_dice,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
                resumable=True,
                provenance=start_provenance,
            ),
        )
        atomic_json(history_path, history)
        print(
            f"epoch {epoch + 1}/{epochs} loss={train_loss:.6f} "
            f"val_dice={val_dice:.6f} threshold={threshold:.3f}",
            flush=True,
        )
        preempt_file = os.environ.get("SEG_PREEMPT_FILE")
        if preempt_file and Path(preempt_file).is_file():
            raise SystemExit(75)
        if args.mode == "full" and stale_epochs >= int(training["patience"]):
            break
    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("SUFS training produced no usable checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    record: Dict[str, Any] = {
        "schema_version": 1,
        "method": "sufs_ft",
        "display_name": config["display_name"],
        "framework": "pyfing-keras-torch",
        "mode": args.mode,
        "seed": int(config["seed"]),
        "train_images": int(len(train_rows)),
        "validation_images": int(len(val_rows)),
        "normalization": normalization,
        "released": released,
        "protocol_config_sha256": protocol_config_digest(config),
        "source_config_sha256": file_digest(args.config),
        "config_sha256": config_digest(config),
        "resume_config_sha256": resume_config_digest(config),
        "queue_batch_contract": queue_batch_metadata(config)["contract"],
        "queue_batch_contract_sha256": queue_batch["contract_sha256"],
        "queue_batch_active_state": queue_batch,
        "manifest_sha256": identity["manifest_sha256"],
        "split_sha256": identity["split_sha256"],
        "learning_rate": float(training["learning_rate"]),
        "epochs_requested": epochs,
        "epochs_completed": int(last_checkpoint["epoch"]) + 1,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_dice": float(best_checkpoint["validation_dice"]),
        "selected_validation_threshold": float(best_checkpoint["selected_threshold"]),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoints": ["best.pt", "last.pt"],
        "checkpoint_sha256": {
            "best.pt": file_digest(best_path),
            "last.pt": file_digest(last_path),
        },
        "debug_limited": bool(
            args.limit_train or args.limit_val or args.allow_unamended_debug
        ),
        "device": device.type.upper(),
        "provenance": start_provenance,
        "dataloader": loader_settings,
        "acceleration": acceleration,
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
        "threshold_amendment_sha256": threshold_amendment_sha256,
    }
    if args.selection is not None:
        record["selection_artifact_sha256"] = file_digest(args.selection)
    record = publish_run_with_profile(
        run_path,
        args.out / "per_epoch_profile.json",
        run_record=record,
        method="sufs_ft",
        seed=seed,
        source_config_sha256=file_digest(args.config),
        history=history,
    )
    return record


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point with queue-compatible OOM status."""
    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    try:
        print(train(parse_args(argv), provenance), flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"CUDA OOM: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(76) from exc


if __name__ == "__main__":
    main()
