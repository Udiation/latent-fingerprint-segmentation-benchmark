#!/usr/bin/env python3
"""Train one pixel-wise segmentation method using train/validation data only."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import _path  # noqa: F401
from segbench.deep_models import RARUNet, build_model, parameter_counts
from segbench.deep_utils import (
    EpochRandomSampler,
    PixelCropDataset,
    UnlabelledPixelCropDataset,
    assert_compatible_subject_splits,
    atomic_json,
    atomic_torch_save,
    checkpoint_payload,
    compute_train_normalization,
    config_digest,
    configure_memory_fraction,
    examiner_exclusions_for_split,
    file_digest,
    final_segmentation_logits,
    full_image_validation_loss,
    load_config,
    load_split_rows,
    loader_generator,
    predict_pixel_probability,
    read_pair,
    resolve_device,
    resume_config_digest,
    restore_rng_state,
    run_provenance,
    seed_everything,
    seed_worker,
    segmentation_loss,
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--memory-fraction", type=float)
    parser.add_argument("--resume", default="none", help="none, auto, or a last.pt path")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--target-splits", type=Path)
    parser.add_argument(
        "--target-subsets",
        nargs="+",
        help="unlabelled train-domain subsets (for example SD302b and SD302g)",
    )
    parser.add_argument("--target-subset", help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--acceleration-receipt", type=Path)
    parser.add_argument("--threshold-amendment", type=Path)
    parser.add_argument("--allow-unamended-debug", action="store_true")
    parser.add_argument(
        "--acceleration-evaluation",
        choices=("fp32", "bf16_channels_last"),
    )
    return parser.parse_args(argv)


def _make_loader(
    dataset: PixelCropDataset | UnlabelledPixelCropDataset,
    *,
    batch_size: int,
    settings: Mapping[str, Any],
    seed: int,
) -> DataLoader[Any]:
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


def _forward_source(
    model: torch.nn.Module,
    image: Tensor,
    *,
    alignment_enabled: bool,
    reversal_weight: float,
) -> Tuple[Any, Optional[Tensor]]:
    if alignment_enabled:
        if not isinstance(model, RARUNet):
            raise TypeError("domain alignment is supported only by ra_runet")
        output, domain = model(
            image,
            return_domain=True,
            gradient_reversal_weight=reversal_weight,
        )
        return output, domain
    return model(image), None


def _train_epoch(
    model: torch.nn.Module,
    source_loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    training_config: Mapping[str, Any],
    *,
    target_loader: Optional[DataLoader[Any]],
    alignment_config: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    model.train()
    accumulation = int(training_config["gradient_accumulation"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    target_iterator: Optional[Iterator[Any]] = iter(target_loader) if target_loader else None
    alignment_enabled = bool(alignment_config.get("enabled", False))
    reversal_weight = float(alignment_config.get("gradient_reversal_weight", 0.0))
    domain_weight = float(alignment_config.get("domain_loss_weight", 0.0))
    profiler = EpochProfiler(device)
    bf16 = acceleration["precision"] == "bf16"
    channels_last = bool(acceleration["channels_last"])

    for batch_index, (image, target) in enumerate(source_loader):
        profiler.batch_ready()
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
            target = target.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bf16,
        ):
            output, source_domain = _forward_source(
                model,
                image,
                alignment_enabled=alignment_enabled,
                reversal_weight=reversal_weight,
            )
            loss = segmentation_loss(
                output,
                target,
                bce_weight=float(training_config["bce_weight"]),
                dice_weight=float(training_config["dice_weight"]),
            )
            if alignment_enabled:
                if target_iterator is None or source_domain is None:
                    raise RuntimeError("alignment requires a train-split target loader")
                try:
                    target_batch = next(target_iterator)
                except StopIteration:
                    if target_loader is None:
                        raise RuntimeError("alignment target loader disappeared")
                    target_iterator = iter(target_loader)
                    target_batch = next(target_iterator)
                target_image = target_batch.to(device, non_blocking=True)
                if channels_last:
                    target_image = target_image.contiguous(
                        memory_format=torch.channels_last
                    )
                if not isinstance(model, RARUNet):
                    raise TypeError("alignment model changed unexpectedly")
                _, target_domain = model(
                    target_image,
                    return_domain=True,
                    gradient_reversal_weight=reversal_weight,
                )
                source_labels = torch.zeros_like(source_domain)
                target_labels = torch.ones_like(target_domain)
                domain_loss = 0.5 * (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        source_domain, source_labels
                    )
                    + torch.nn.functional.binary_cross_entropy_with_logits(
                        target_domain, target_labels
                    )
                )
                loss = loss + domain_weight * domain_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at batch {batch_index}")
        (loss / accumulation).backward()
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(source_loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        profiler.compute_done()
    if not losses:
        raise RuntimeError("training loader emitted no batches")
    return float(np.mean(losses)), profiler.finish()


def _validate(
    model: torch.nn.Module,
    rows: Any,
    device: torch.device,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    *,
    cache_dir: Optional[Path],
    acceleration: Mapping[str, Any],
) -> Tuple[float, float, Dict[str, float], float]:
    data_config = config["data"]
    inference = config["inference"]
    probabilities = []
    targets = []
    for row in rows.to_dict("records"):
        image, target = read_pair(
            row,
            float(data_config["target_ppi"]),
            str(data_config["mask_column"]),
            cache_dir=cache_dir,
        )
        normalized = (image - float(normalization["mean"])) / float(normalization["std"])
        probability = predict_pixel_probability(
            model,
            normalized,
            device,
            pad_multiple=int(inference["pad_multiple"]),
            tile_size=int(inference["tile_size"]),
            tile_overlap=int(inference["tile_overlap"]),
            autocast_bf16=acceleration["precision"] == "bf16",
            channels_last=bool(acceleration["channels_last"]),
        )
        probabilities.append(probability)
        targets.append(target)
    threshold, dice, scores = select_validation_threshold(
        probabilities, targets, config["training"]["threshold_candidates"]
    )
    validation_loss = full_image_validation_loss(
        probabilities,
        targets,
        bce_weight=float(config["training"]["bce_weight"]),
        dice_weight=float(config["training"]["dice_weight"]),
    )
    return threshold, dice, scores, validation_loss


def train(
    args: argparse.Namespace,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one resumable pixel-model run and return its final run record."""
    start_provenance = dict(
        provenance
        if provenance is not None
        else run_provenance(sys.argv)
    )
    config = load_config(args.config)
    if config["task"] != "pixel":
        raise ValueError(f"{args.config} declares task={config['task']!r}, expected 'pixel'")
    config = dict(config)
    config["seed"] = int(args.seed if args.seed is not None else config.get("seed", 0))
    config["runtime"] = dict(config["runtime"])
    if args.device is not None:
        config["runtime"]["device"] = args.device
    memory_override = (
        args.memory_fraction
        if args.memory_fraction is not None
        else os.environ.get("SEG_GPU_MEMORY_FRACTION")
    )
    if memory_override is not None:
        config["runtime"]["memory_fraction"] = float(memory_override)
    config["training"] = dict(config["training"])
    queue_batch = apply_queue_batch_contract(config)
    config["training_data_identity"] = {
        "manifest_sha256": file_digest(args.manifest),
        "split_sha256": file_digest(args.splits),
    }
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

    seed = int(config["seed"])
    deterministic = bool(config["runtime"].get("deterministic", True))
    seed_everything(seed, deterministic)
    device = resolve_device(str(config["runtime"].get("device", "auto")))
    configure_memory_fraction(device, float(config["runtime"]["memory_fraction"]))
    data_config = config["data"]
    train_rows = load_split_rows(
        args.manifest,
        args.splits,
        subset=str(data_config["subset"]),
        split="train",
        mask_column=str(data_config["mask_column"]),
        exclude_examiner_values=examiner_exclusions_for_split(data_config, "train"),
        limit=args.limit_train,
    )
    val_rows = load_split_rows(
        args.manifest,
        args.splits,
        subset=str(data_config["subset"]),
        split="val",
        mask_column=str(data_config["mask_column"]),
        exclude_examiner_values=examiner_exclusions_for_split(data_config, "val"),
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
            target_ppi=float(data_config["target_ppi"]),
            mask_columns=[str(data_config["mask_column"])],
            required_ids=required_ids,
            verify_artifact_hashes=True,
        )
        cache_receipt_sha256 = file_digest(args.cache_dir / "receipt.json")
        config["runtime"]["preprocess_cache_receipt_sha256"] = cache_receipt_sha256
    normalization = compute_train_normalization(
        train_rows,
        float(data_config["target_ppi"]),
        cache_dir=args.cache_dir,
    )
    source_dataset = PixelCropDataset(
        train_rows,
        target_ppi=float(data_config["target_ppi"]),
        mask_column=str(data_config["mask_column"]),
        crop_size=int(data_config["crop_size"]),
        normalization=normalization,
        augmentation=config["augmentation"],
        seed=seed,
        cache_dir=args.cache_dir,
    )

    alignment_config = config["model"].get("alignment", {})
    target_dataset: Optional[UnlabelledPixelCropDataset] = None
    if bool(alignment_config.get("enabled", False)):
        if config["method"] != "ra_runet":
            raise ValueError("alignment.enabled is valid only for ra_runet")
        if args.target_subset and args.target_subsets:
            raise ValueError("use only one of --target-subset or --target-subsets")
        target_subsets = args.target_subsets or (
            [args.target_subset] if args.target_subset else None
        )
        if not (args.target_manifest and args.target_splits and target_subsets):
            raise ValueError(
                "aligned ra_runet requires --target-manifest, --target-splits and "
                "--target-subsets"
            )
        assert_compatible_subject_splits(args.splits, args.target_splits)
        config["training_data_identity"].update(
            {
                "target_manifest_sha256": file_digest(args.target_manifest),
                "target_split_sha256": file_digest(args.target_splits),
                "target_subsets": list(target_subsets),
            }
        )
        target_rows = load_split_rows(
            args.target_manifest,
            args.target_splits,
            subset=target_subsets,
            split="train",
            mask_column=str(data_config["mask_column"]),
            require_mask=False,
            exclude_examiner_values=examiner_exclusions_for_split(data_config, "train"),
            limit=args.limit_train,
        )
        target_dataset = UnlabelledPixelCropDataset(
            target_rows,
            target_ppi=float(data_config["target_ppi"]),
            crop_size=int(data_config["crop_size"]),
            normalization=normalization,
            augmentation=config["augmentation"],
            seed=seed + 1,
            cache_dir=args.cache_dir,
        )

    model = build_model(str(config["method"]), config["model"]).to(device)
    if acceleration["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    counts = parameter_counts(model)
    training_config = config["training"]
    loader_settings = protocol_dataloader_settings(training_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    if epochs < 1 or epochs > 100:
        raise ValueError("epochs must be between 1 and the frozen maximum of 100")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training_config["min_learning_rate"]),
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
    if args.resume == "auto" and last_path.exists():
        resume_path = last_path
    elif args.resume not in {"none", "auto"}:
        resume_path = Path(args.resume)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("resume_config_sha256") != resume_config_digest(config):
            raise ValueError("resume checkpoint configuration does not match this run")
        validate_resume_batch_transition(checkpoint, config)
        if checkpoint.get("method") != config["method"] or checkpoint.get("task") != "pixel":
            raise ValueError("resume checkpoint identity does not match this run")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_validation_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        if history_path.exists():
            import json

            history = json.loads(history_path.read_text(encoding="utf-8"))
        stale_epochs = max(0, start_epoch - best_epoch - 1)

    started = time.monotonic()
    source_loader = _make_loader(
        source_dataset,
        batch_size=int(training_config["batch_size"]),
        settings=loader_settings,
        seed=seed,
    )
    target_loader = (
        _make_loader(
            target_dataset,
            batch_size=int(training_config["batch_size"]),
            settings=loader_settings,
            seed=seed + 100_000,
        )
        if target_dataset is not None
        else None
    )
    for epoch in range(start_epoch, epochs):
        if stale_epochs >= int(training_config["patience"]):
            break
        source_dataset.set_epoch(epoch)
        if target_dataset is not None:
            target_dataset.set_epoch(epoch)
        train_loss, epoch_profile = _train_epoch(
            model,
            source_loader,
            device,
            optimizer,
            training_config,
            target_loader=target_loader,
            alignment_config=alignment_config,
            acceleration=acceleration,
        )
        scheduler.step()
        threshold, val_dice, threshold_scores, validation_loss = _validate(
            model,
            val_rows,
            device,
            config,
            normalization,
            cache_dir=args.cache_dir,
            acceleration=acceleration,
        )
        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                best_path,
                checkpoint_payload(
                    model=model,
                    method=str(config["method"]),
                    task="pixel",
                    config=config,
                    normalization=normalization,
                    epoch=epoch,
                    selected_threshold=threshold,
                    validation_dice=val_dice,
                    parameter_count=counts,
                    provenance=start_provenance,
                ),
            )
        else:
            stale_epochs += 1
        atomic_torch_save(
            last_path,
            checkpoint_payload(
                model=model,
                method=str(config["method"]),
                task="pixel",
                config=config,
                normalization=normalization,
                epoch=epoch,
                selected_threshold=threshold,
                validation_dice=val_dice,
                parameter_count=counts,
                optimizer=optimizer,
                scheduler=scheduler,
                best_validation_dice=best_dice,
                best_epoch=best_epoch,
                provenance=start_provenance,
            ),
        )
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_dice": val_dice,
            "selected_threshold": threshold,
            "threshold_scores": threshold_scores,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_profile": epoch_profile,
        }
        history.append(entry)
        atomic_json(history_path, history)
        print(
            f"epoch {epoch + 1}/{epochs} loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"val_dice={val_dice:.6f} threshold={threshold:.3f}",
            flush=True,
        )
        preempt_file = os.environ.get("SEG_PREEMPT_FILE")
        if preempt_file and Path(preempt_file).is_file():
            print(
                f"preemption requested after epoch {epoch + 1}; last.pt is resumable",
                flush=True,
            )
            raise SystemExit(75)
        if stale_epochs >= int(training_config["patience"]):
            break

    if not best_path.exists() or not last_path.exists():
        raise RuntimeError("training did not produce both best.pt and last.pt")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    run_record = {
        "method": config["method"],
        "task": "pixel",
        "seed": seed,
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "normalization": normalization,
        "parameter_count": counts,
        "best_epoch": best_epoch,
        "last_epoch": int(last_checkpoint["epoch"]),
        "epochs_completed": int(last_checkpoint["epoch"]) + 1,
        "best_validation_dice": best_dice,
        "validation_loss_first": float(history[0]["validation_loss"]),
        "validation_loss_last": float(history[-1]["validation_loss"]),
        "validation_loss_min": float(
            min(float(entry["validation_loss"]) for entry in history)
        ),
        "selected_validation_threshold": float(best_checkpoint["selected_threshold"]),
        "learning_rate": float(training_config["learning_rate"]),
        "epochs_requested": epochs,
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
        "source_config_sha256": file_digest(args.config),
        "config_sha256": config_digest(config),
        "resume_config_sha256": resume_config_digest(config),
        "queue_batch_contract": queue_batch_metadata(config)["contract"],
        "queue_batch_contract_sha256": queue_batch["contract_sha256"],
        "queue_batch_active_state": queue_batch,
        "manifest_sha256": file_digest(args.manifest),
        "split_sha256": file_digest(args.splits),
        "provenance": start_provenance,
        "dataloader": loader_settings,
        "acceleration": acceleration,
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
        "threshold_amendment_sha256": threshold_amendment_sha256,
    }
    if hasattr(model, "pretrained_provenance"):
        run_record["pretrained_initialization"] = model.pretrained_provenance
    if args.target_manifest is not None:
        run_record["target_manifest_sha256"] = file_digest(args.target_manifest)
    if args.target_splits is not None:
        run_record["target_split_sha256"] = file_digest(args.target_splits)
    if args.target_subsets or args.target_subset:
        run_record["target_subsets"] = args.target_subsets or [args.target_subset]
    run_record = publish_run_with_profile(
        run_path,
        args.out / "per_epoch_profile.json",
        run_record=run_record,
        method=str(config["method"]),
        seed=seed,
        source_config_sha256=file_digest(args.config),
        history=history,
    )
    return run_record


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    command = list(sys.argv) if argv is None else [sys.argv[0], *map(str, argv)]
    provenance = run_provenance(command, capture_timing="process_start")
    try:
        record = train(parse_args(argv), provenance)
        print(record)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"CUDA OOM: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(76) from exc


if __name__ == "__main__":
    main()
