#!/usr/bin/env python3
"""Train one block/patch segmentation model using train/validation data only."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import _path  # noqa: F401
from segbench.deep_models import ConvAutoencoder, build_model, parameter_counts
from segbench.deep_utils import (
    EpochRandomSampler,
    PatchDataset,
    UnlabelledPatchDataset,
    atomic_json,
    atomic_torch_save,
    binary_dice,
    blocks_to_postprocessed_mask,
    capture_rng_state,
    checkpoint_payload,
    compute_train_normalization,
    config_digest,
    configure_memory_fraction,
    examiner_exclusions_for_split,
    file_digest,
    load_config,
    load_split_rows,
    loader_generator,
    predict_patch_probability,
    read_pair,
    resolve_device,
    resume_config_digest,
    restore_rng_state,
    run_provenance,
    seed_everything,
    seed_worker,
    EpochProfiler,
    acceleration_policy,
    apply_queue_batch_contract,
    protocol_dataloader_settings,
    publish_run_with_profile,
    queue_batch_metadata,
    validate_resume_batch_transition,
)
from segbench import preprocess_cache, probability_artifacts


PRETRAIN_SEED_OFFSET = 1_000_000


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
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--acceleration-receipt", type=Path)
    parser.add_argument("--threshold-amendment", type=Path)
    parser.add_argument("--allow-unamended-debug", action="store_true")
    parser.add_argument(
        "--acceleration-evaluation",
        choices=("fp32", "bf16_channels_last"),
    )
    return parser.parse_args(argv)


def _loader(
    dataset: PatchDataset | UnlabelledPatchDataset,
    config: Mapping[str, Any],
    seed: int,
) -> DataLoader[Any]:
    settings = protocol_dataloader_settings(config)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
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


def _pretrain_autoencoder(
    model: torch.nn.Module,
    dataset: UnlabelledPatchDataset,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    *,
    checkpoint_path: Path,
    resume_identity: str,
    run_config: Mapping[str, Any],
    acceleration: Mapping[str, Any],
) -> Tuple[list[float], list[Dict[str, Any]]]:
    """Pretrain a CNN encoder on training patches without consulting labels."""
    autoencoder = ConvAutoencoder(
        in_channels=int(model_config.get("in_channels", 1)),
        latent_channels=int(model_config.get("latent_channels", 128)),
    ).to(device)
    if acceleration["channels_last"]:
        autoencoder = autoencoder.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    pretrain_epochs = int(config.get("autoencoder_pretrain_epochs", 0))
    if pretrain_epochs < 1:
        raise ValueError("cnn_ae requires autoencoder_pretrain_epochs >= 1")
    if str(config.get("autoencoder_loss", "")).lower() != "mse":
        raise ValueError("cnn_ae autoencoder_loss must be the declared 'mse'")
    losses: list[float] = []
    profiles: list[Dict[str, Any]] = []
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "autoencoder_pretrain":
            raise ValueError("invalid autoencoder pretraining checkpoint stage")
        if checkpoint.get("resume_config_sha256") != resume_identity:
            raise ValueError("autoencoder pretraining checkpoint configuration differs")
        validate_resume_batch_transition(checkpoint, run_config)
        autoencoder.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng_state(checkpoint["rng_state"])
        losses = [float(value) for value in checkpoint.get("losses", [])]
        profiles = [dict(value) for value in checkpoint.get("epoch_profiles", [])]
        start_epoch = int(checkpoint["epoch"]) + 1
    accumulation = int(config["gradient_accumulation"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    loader = _loader(dataset, config, seed + PRETRAIN_SEED_OFFSET)
    for epoch in range(start_epoch, pretrain_epochs):
        dataset.set_epoch(PRETRAIN_SEED_OFFSET + epoch)
        autoencoder.train()
        batch_losses = []
        profiler = EpochProfiler(device)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, patch in enumerate(loader):
            profiler.batch_ready()
            patch = patch.to(device, non_blocking=True)
            if acceleration["channels_last"]:
                patch = patch.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=acceleration["precision"] == "bf16",
            ):
                reconstruction = autoencoder(patch)
                loss = torch.nn.functional.mse_loss(reconstruction, patch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite autoencoder pretraining loss")
            (loss / accumulation).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_losses.append(float(loss.detach().cpu()))
            profiler.compute_done()
        losses.append(float(np.mean(batch_losses)))
        profiles.append(profiler.finish())
        queue_batch = queue_batch_metadata(run_config)
        atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": 1,
                "stage": "autoencoder_pretrain",
                "resume_config_sha256": resume_identity,
                "queue_batch_contract_sha256": queue_batch[
                    "contract_sha256"
                ],
                "queue_batch_active_state": queue_batch["active"],
                "epoch": epoch,
                "model_state": autoencoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": capture_rng_state(),
                "losses": losses,
                "epoch_profiles": profiles,
            },
        )
        preempt_file = os.environ.get("SEG_PREEMPT_FILE")
        if preempt_file and Path(preempt_file).is_file():
            print(
                f"preemption requested after autoencoder epoch {epoch + 1}; "
                "pretrain_last.pt is resumable",
                flush=True,
            )
            raise SystemExit(75)
    if not hasattr(model, "encoder"):
        raise TypeError("cnn_ae classifier does not expose its encoder")
    model.encoder.load_state_dict(autoencoder.encoder.state_dict())
    checkpoint_path.unlink(missing_ok=True)
    return losses, profiles


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    accumulation: int,
    acceleration: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    profiler = EpochProfiler(device)
    for batch_index, (patch, target) in enumerate(loader):
        profiler.batch_ready()
        patch = patch.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if acceleration["channels_last"]:
            patch = patch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=acceleration["precision"] == "bf16",
        ):
            logits = model(patch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at batch {batch_index}")
        (loss / accumulation).backward()
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
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
) -> Tuple[float, float, Dict[str, float]]:
    data_config = config["data"]
    inference = config["inference"]
    entries = []
    for row in rows.to_dict("records"):
        image, target = read_pair(
            row,
            float(data_config["target_ppi"]),
            str(data_config["mask_column"]),
            cache_dir=cache_dir,
        )
        normalized = (image - float(normalization["mean"])) / float(normalization["std"])
        block_probability = predict_patch_probability(
            model,
            normalized,
            device,
            patch_size=int(data_config["patch_size"]),
            block_size=int(data_config["block_size"]),
            batch_size=int(inference["patch_batch_size"]),
            autocast_bf16=acceleration["precision"] == "bf16",
            channels_last=bool(acceleration["channels_last"]),
        )
        entries.append((block_probability, target))
    scores: Dict[str, float] = {}
    for threshold in sorted(float(item) for item in config["training"]["threshold_candidates"]):
        dices = []
        for block_probability, target in entries:
            prediction = blocks_to_postprocessed_mask(
                block_probability,
                target.shape,
                threshold=threshold,
                min_component_blocks=int(inference["min_component_blocks"]),
            )
            dices.append(binary_dice(prediction, target))
        scores[f"{threshold:.8g}"] = float(np.mean(dices))
    selected = max(
        ((float(key), value) for key, value in scores.items()),
        key=lambda item: (item[1], -abs(item[0] - 0.5), -item[0]),
    )
    return selected[0], selected[1], scores


def train(
    args: argparse.Namespace,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one resumable patch-model run."""
    start_provenance = dict(
        provenance
        if provenance is not None
        else run_provenance(sys.argv)
    )
    config = load_config(args.config)
    if config["task"] != "patch":
        raise ValueError(f"{args.config} declares task={config['task']!r}, expected 'patch'")
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
    seed_everything(seed, bool(config["runtime"].get("deterministic", True)))
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
    dataset = PatchDataset(
        train_rows,
        target_ppi=float(data_config["target_ppi"]),
        mask_column=str(data_config["mask_column"]),
        patch_size=int(data_config["patch_size"]),
        block_size=int(data_config["block_size"]),
        blocks_per_class_per_image=int(data_config["blocks_per_class_per_image"]),
        normalization=normalization,
        augmentation=config["augmentation"],
        seed=seed,
        cache_dir=args.cache_dir,
    )
    unlabelled_dataset: Optional[UnlabelledPatchDataset] = None
    if config["method"] == "cnn_ae":
        unlabelled_dataset = UnlabelledPatchDataset(
            train_rows,
            target_ppi=float(data_config["target_ppi"]),
            patch_size=int(data_config["patch_size"]),
            block_size=int(data_config["block_size"]),
            patches_per_image=2 * int(data_config["blocks_per_class_per_image"]),
            normalization=normalization,
            augmentation=config["augmentation"],
            seed=seed,
            cache_dir=args.cache_dir,
        )
    model = build_model(str(config["method"]), config["model"]).to(device)
    if acceleration["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    counts = parameter_counts(model)
    training_config = config["training"]
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
    pretrain_path = args.out / "pretrain_last.pt"
    history_path = args.out / "history.json"
    run_path = args.out / "run.json"
    start_epoch = 0
    best_dice = -math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[Dict[str, Any]] = []
    started = time.monotonic()

    resume_path: Optional[Path] = None
    if args.resume == "auto" and last_path.exists():
        resume_path = last_path
    elif args.resume not in {"none", "auto"}:
        resume_path = Path(args.resume)
    pretrain_losses: list[float] = []
    pretrain_profiles: list[Dict[str, Any]] = []
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("resume_config_sha256") != resume_config_digest(config):
            raise ValueError("resume checkpoint configuration does not match this run")
        validate_resume_batch_transition(checkpoint, config)
        if checkpoint.get("method") != config["method"] or checkpoint.get("task") != "patch":
            raise ValueError("resume checkpoint identity does not match this run")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_validation_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        pretrain_losses = [
            float(value) for value in checkpoint.get("autoencoder_pretrain_loss", [])
        ]
        pretrain_profiles = [
            dict(value) for value in checkpoint.get("autoencoder_pretrain_profiles", [])
        ]
        stale_epochs = max(0, start_epoch - best_epoch - 1)
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    elif config["method"] == "cnn_ae":
        if unlabelled_dataset is None:
            raise RuntimeError("cnn_ae unlabelled pretraining dataset was not created")
        pretrain_losses, pretrain_profiles = _pretrain_autoencoder(
            model,
            unlabelled_dataset,
            training_config,
            config["model"],
            device,
            seed,
            checkpoint_path=pretrain_path,
            resume_identity=resume_config_digest(config),
            run_config=config,
            acceleration=acceleration,
        )

    accumulation = int(training_config["gradient_accumulation"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    loader = _loader(dataset, training_config, seed)
    for epoch in range(start_epoch, epochs):
        if stale_epochs >= int(training_config["patience"]):
            break
        dataset.set_epoch(epoch)
        train_loss, epoch_profile = _train_epoch(
            model,
            loader,
            device,
            optimizer,
            accumulation,
            acceleration,
        )
        scheduler.step()
        threshold, val_dice, threshold_scores = _validate(
            model,
            val_rows,
            device,
            config,
            normalization,
            cache_dir=args.cache_dir,
            acceleration=acceleration,
        )
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            stale_epochs = 0
            best_payload = checkpoint_payload(
                model=model,
                method=str(config["method"]),
                task="patch",
                config=config,
                normalization=normalization,
                epoch=epoch,
                selected_threshold=threshold,
                validation_dice=val_dice,
                parameter_count=counts,
                provenance=start_provenance,
            )
            best_payload["autoencoder_pretrain_loss"] = pretrain_losses
            best_payload["autoencoder_pretrain_profiles"] = pretrain_profiles
            atomic_torch_save(best_path, best_payload)
        else:
            stale_epochs += 1
        last_payload = checkpoint_payload(
            model=model,
            method=str(config["method"]),
            task="patch",
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
        )
        last_payload["autoencoder_pretrain_loss"] = pretrain_losses
        last_payload["autoencoder_pretrain_profiles"] = pretrain_profiles
        atomic_torch_save(last_path, last_payload)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_dice": val_dice,
                "selected_threshold": threshold,
                "threshold_scores": threshold_scores,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_profile": epoch_profile,
            }
        )
        atomic_json(history_path, history)
        print(
            f"epoch {epoch + 1}/{epochs} loss={train_loss:.6f} "
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
        "task": "patch",
        "seed": seed,
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "train_patches": len(dataset),
        "normalization": normalization,
        "parameter_count": counts,
        "autoencoder_pretrain_loss": pretrain_losses,
        "autoencoder_pretrain_profiles": pretrain_profiles,
        "best_epoch": best_epoch,
        "last_epoch": int(last_checkpoint["epoch"]),
        "epochs_completed": int(last_checkpoint["epoch"]) + 1,
        "best_validation_dice": best_dice,
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
        "dataloader": protocol_dataloader_settings(training_config),
        "acceleration": acceleration,
        "preprocess_cache_receipt_sha256": cache_receipt_sha256,
        "threshold_amendment_sha256": threshold_amendment_sha256,
    }
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
        print(train(parse_args(argv), provenance))
    except torch.cuda.OutOfMemoryError as exc:
        print(f"CUDA OOM: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(76) from exc


if __name__ == "__main__":
    main()
