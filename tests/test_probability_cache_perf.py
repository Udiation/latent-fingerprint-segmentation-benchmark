"""Synthetic-only tests for probability, cache, loader, and acceleration policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import yaml


os.environ.setdefault("KERAS_BACKEND", "torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_acceleration_receipt  # noqa: E402
import cache_preprocessed  # noqa: E402
import infer_masks  # noqa: E402
import select_threshold_from_probabilities  # noqa: E402
import train_seg  # noqa: E402
from segbench.artifact_identity import sha256_file  # noqa: E402
from segbench.deep_utils import (  # noqa: E402
    PROTOCOL_DATALOADER,
    PixelCropDataset,
    acceleration_policy,
    load_config,
    protocol_dataloader_settings,
    publish_run_with_profile,
    write_per_epoch_profile_card,
)
from segbench import preprocess_cache, probability_artifacts  # noqa: E402


def test_per_epoch_profile_binds_complete_contiguous_run(tmp_path: Path) -> None:
    config_digest = "a" * 64
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "method": "unet",
                "seed": 0,
                "source_config_sha256": config_digest,
                "epochs_completed": 2,
            }
        ),
        encoding="utf-8",
    )

    def history(epochs: list[int]) -> list[dict[str, object]]:
        return [
            {
                "epoch": epoch,
                "epoch_profile": {
                    "data_wait_seconds": 0.1,
                    "compute_seconds": 0.2,
                    "gpu_utilization_sample_count": 0,
                    "gpu_utilization_percent_mean": None,
                },
            }
            for epoch in epochs
        ]

    with pytest.raises(ValueError, match="every completed epoch"):
        write_per_epoch_profile_card(
            tmp_path / "gap.json",
            run_card_path=run_path,
            method="unet",
            seed=0,
            source_config_sha256=config_digest,
            history=history([0]),
        )
    card = write_per_epoch_profile_card(
        tmp_path / "profile.json",
        run_card_path=run_path,
        method="unet",
        seed=0,
        source_config_sha256=config_digest,
        history=history([0, 1]),
    )
    assert card["complete"] is True
    assert card["epochs_recorded"] == 2
    assert card["generic_device_utilisation_recorded"] is False
    assert card["run_card_sha256"] == sha256_file(run_path)
    profile_path = tmp_path / "profile.json"
    original_profile_sha = sha256_file(profile_path)
    profile_path.unlink()
    replay = json.loads(run_path.read_text(encoding="utf-8"))
    replay["elapsed_seconds"] = 999.0
    published = publish_run_with_profile(
        run_path,
        profile_path,
        run_record=replay,
        method="unet",
        seed=0,
        source_config_sha256=config_digest,
        history=history([0, 1]),
    )
    assert "elapsed_seconds" not in published
    assert sha256_file(profile_path) == original_profile_sha
    publish_run_with_profile(
        run_path,
        profile_path,
        run_record=replay,
        method="unet",
        seed=0,
        source_config_sha256=config_digest,
        history=history([0, 1]),
    )
    assert sha256_file(profile_path) == original_profile_sha


def test_resume_records_reject_duplicate_unknown_and_negative_timing() -> None:
    valid = {"image_id": "a", "ms": 0.0}
    assert probability_artifacts.authenticated_resume_records(
        [valid], allowed_ids=["a"], timing_key="ms"
    )["a"]["ms"] == 0.0
    with pytest.raises(ValueError, match="duplicate"):
        probability_artifacts.authenticated_resume_records(
            [valid, valid], allowed_ids=["a"], timing_key="ms"
        )
    with pytest.raises(ValueError, match="unknown"):
        probability_artifacts.authenticated_resume_records(
            [{"image_id": "b", "ms": 1.0}],
            allowed_ids=["a"],
            timing_key="ms",
        )
    with pytest.raises(ValueError, match="timing"):
        probability_artifacts.authenticated_resume_records(
            [{"image_id": "a", "ms": -0.1}],
            allowed_ids=["a"],
            timing_key="ms",
        )


def test_production_selector_requires_frozen_threshold_amendment() -> None:
    with pytest.raises(ValueError, match="requires the exact threshold amendment"):
        select_threshold_from_probabilities.select(
            argparse.Namespace(
                threshold_amendment=None,
                allow_unamended_debug=False,
                limit=0,
            )
        )


def _write_inputs(root: Path, count: int = 8) -> tuple[Path, Path, pd.DataFrame]:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    splits = {}
    for index in range(count):
        yy, xx = np.mgrid[:48, :52]
        image = ((xx * 3 + yy * 5 + index * 7) % 256).astype(np.uint8)
        roi = ((xx - 26) ** 2 + (yy - 24) ** 2 < (12 + index % 3) ** 2)
        quality = roi & ((xx + yy + index) % 4 != 0)
        image_path = root / f"image_{index}.png"
        roi_path = root / f"roi_{index}.png"
        quality_path = root / f"quality_{index}.png"
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(roi_path), roi.astype(np.uint8) * 255)
        assert cv2.imwrite(str(quality_path), quality.astype(np.uint8) * 255)
        subject = f"subject_{index}"
        splits[subject] = "train" if index < count - 2 else "val"
        rows.append(
            {
                "image_id": f"synthetic_{index}",
                "image_path": str(image_path),
                "source_sha256": sha256_file(image_path),
                "mask_path": str(roi_path),
                "quality_mask_path": str(quality_path) if index else "",
                "subject_id": subject,
                "subset": "finger_latent",
                "ppi": 500,
            }
        )
    manifest = root / "manifest.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest, index=False)
    split_path = root / "splits.json"
    split_path.write_text(json.dumps({"seed": 0, "splits": splits}), encoding="utf-8")
    return manifest, split_path, frame


def _cache_args(manifest: Path, out: Path, n_jobs: int) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        out=out,
        mask_columns=["mask_path", "quality_mask_path"],
        target_ppi=500.0,
        verify_count=8,
        n_jobs=n_jobs,
        limit=8,
    )


def test_parallel_cache_is_exact_resumable_and_schedule_independent(tmp_path: Path) -> None:
    manifest, _, frame = _write_inputs(tmp_path)
    cache_one = tmp_path / "cache_one"
    cache_many = tmp_path / "cache_many"
    stale = cache_many / "images" / ".partial-interrupted.npy"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"incomplete")
    one = cache_preprocessed.build(_cache_args(manifest, cache_one, 1))
    many = cache_preprocessed.build(_cache_args(manifest, cache_many, 2))
    assert one == many
    assert not stale.exists()
    assert many["verification"]["array_identity"] is True
    assert many["verification"]["verified_count"] == 8
    first = many["records"][0]
    assert first["source_mask_sha256"]["quality_mask_path"] is None
    assert first["mask_sha256"]["quality_mask_path"] is None
    # A replay safely reuses authenticated artifacts and publishes identical bytes.
    before = (cache_many / "receipt.json").read_bytes()
    assert cache_preprocessed.build(_cache_args(manifest, cache_many, 2)) == many
    assert (cache_many / "receipt.json").read_bytes() == before
    # Simulate interruption after all per-record journals were committed but
    # before the aggregate receipt rename.  Authenticated arrays are reused.
    first_artifact = preprocess_cache.image_path(
        cache_many, str(frame.iloc[0]["image_id"])
    )
    first_mtime = first_artifact.stat().st_mtime_ns
    (cache_many / "receipt.json").unlink()
    missing_id = str(frame.iloc[-1]["image_id"])
    (cache_many / ".records" / f"{missing_id}.json").unlink()
    preprocess_cache.image_path(cache_many, missing_id).unlink()
    assert cache_preprocessed.build(_cache_args(manifest, cache_many, 2)) == many
    assert first_artifact.stat().st_mtime_ns == first_mtime
    assert preprocess_cache.image_path(cache_many, missing_id).is_file()
    changed_universe = _cache_args(manifest, cache_many, 2)
    changed_universe.limit = 4
    with pytest.raises(ValueError, match="other inputs"):
        cache_preprocessed.build(changed_universe)
    for row in frame.to_dict("records"):
        image_id = str(row["image_id"])
        cached_image = np.load(preprocess_cache.image_path(cache_many, image_id))
        source_image = cv2.imread(str(row["image_path"]), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(cached_image, source_image)
        cached_roi = np.load(
            preprocess_cache.mask_path(cache_many, "mask_path", image_id)
        )
        source_roi = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(cached_roi, source_roi)
    assert set(one["implementation_sha256"]) == {
        "builder",
        "deep_utils",
        "io",
        "preprocess_cache",
    }
    receipt_path = cache_one / "receipt.json"
    altered_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    altered_receipt["implementation_sha256"]["builder"] = "0" * 64
    receipt_path.write_text(json.dumps(altered_receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="other inputs"):
        cache_preprocessed.build(_cache_args(manifest, cache_one, 1))
    receipt_path.write_text(json.dumps(one, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tampered = preprocess_cache.image_path(cache_one, str(frame.iloc[0]["image_id"]))
    value = np.load(tampered, allow_pickle=False)
    value[0, 0] ^= np.uint8(1)
    np.save(tampered, value, allow_pickle=False)
    with pytest.raises(ValueError, match="cached image bytes changed"):
        preprocess_cache.load_receipt(
            cache_one,
            manifest_sha256=sha256_file(manifest),
            target_ppi=500.0,
            mask_columns=["mask_path", "quality_mask_path"],
            required_ids=frame["image_id"].astype(str).tolist(),
            verify_artifact_hashes=True,
        )


def _tensor_set(loader: torch.utils.data.DataLoader) -> list[str]:
    result = []
    for image, target in loader:
        for index in range(len(image)):
            digest = hashlib.sha256()
            digest.update(image[index].numpy().tobytes())
            digest.update(target[index].numpy().tobytes())
            result.append(digest.hexdigest())
    return sorted(result)


@pytest.mark.skipif(sys.platform != "linux", reason="worker PID assertion is Linux-specific")
def test_persistent_workers_see_shared_epoch_and_remain_alive(tmp_path: Path) -> None:
    _, _, frame = _write_inputs(tmp_path)
    augmentation = {
        "horizontal_flip_probability": 0.5,
        "max_rotation_degrees": 15,
        "scale_min": 0.9,
        "scale_max": 1.1,
        "brightness": 0.2,
        "contrast": 0.2,
        "gaussian_noise_std": 0.02,
    }
    dataset = PixelCropDataset(
        frame,
        target_ppi=500,
        mask_column="mask_path",
        crop_size=32,
        normalization={"mean": 0.5, "std": 0.25},
        augmentation=augmentation,
        seed=123,
    )
    loader = train_seg._make_loader(
        dataset,
        batch_size=2,
        settings=PROTOCOL_DATALOADER,
        seed=456,
    )
    try:
        dataset.set_epoch(0)
        epoch_zero = _tensor_set(loader)
        first_pids = tuple(worker.pid for worker in loader._iterator._workers)
        dataset.set_epoch(1)
        epoch_one = _tensor_set(loader)
        second_pids = tuple(worker.pid for worker in loader._iterator._workers)
        dataset.set_epoch(0)
        epoch_zero_replay = _tensor_set(loader)
        third_pids = tuple(worker.pid for worker in loader._iterator._workers)
        assert first_pids == second_pids == third_pids
        assert len(first_pids) == 8 and all(pid is not None for pid in first_pids)
        assert epoch_zero != epoch_one
        assert epoch_zero == epoch_zero_replay
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_probability_artifacts_bind_sources_and_storage_roundtrip(tmp_path: Path) -> None:
    probability = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(5, 7)
    stored = probability_artifacts.storage_roundtrip(probability)
    path = probability_artifacts.safe_path(tmp_path, "safe-id")
    probability_artifacts.atomic_write(path, probability)
    assert np.array_equal(probability_artifacts.read(path), stored)
    identity = probability_artifacts.set_identity(
        [("safe-id", path)], source_sha256={"safe-id": "a" * 64}
    )
    assert identity["storage_dtype"] == "float16"
    assert identity["records"][0]["source_sha256"] == "a" * 64
    assert identity["records"][0]["bytes"] == path.stat().st_size
    assert Path(identity["records"][0]["path"]).is_absolute()


def test_loader_protocol_is_exact() -> None:
    assert protocol_dataloader_settings(PROTOCOL_DATALOADER) == PROTOCOL_DATALOADER
    drift = dict(PROTOCOL_DATALOADER, prefetch_factor=2)
    with pytest.raises(ValueError, match="exactly match"):
        protocol_dataloader_settings(drift)


def test_trainable_test_inference_requires_validation_selector_receipt() -> None:
    args = argparse.Namespace(
        split="test",
        threshold_receipt=None,
        threshold_amendment=None,
    )
    config = {"training": {"threshold_candidates": [0.1, 0.2, 0.3]}}
    with pytest.raises(ValueError, match="requires an immutable validation"):
        infer_masks._threshold_from_receipt(
            args, config, {"selected_threshold": 0.3}
        )


def _run_card(mode: str, dice: float) -> dict[str, object]:
    bf16 = mode == "bf16"
    return {
        "method": "unet",
        "task": "pixel",
        "seed": 0,
        "source_config_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "epochs_completed": 2,
        "best_validation_dice": dice,
        "acceleration": {
            "precision": "bf16" if bf16 else "fp32",
            "channels_last": bf16,
            "evaluation_only": True,
            "production_eligible": False,
        },
    }


def test_acceleration_is_fail_closed_and_receipt_requires_delta_below_limit(
    tmp_path: Path,
) -> None:
    assert acceleration_policy()["precision"] == "fp32"
    assert acceleration_policy()["channels_last"] is False
    with pytest.raises(ValueError, match="exactly 2 epochs"):
        acceleration_policy(evaluation_mode="bf16_channels_last", epochs=3)
    fp32_path, bf16_path = tmp_path / "fp32.json", tmp_path / "bf16.json"
    fp32_path.write_text(json.dumps(_run_card("fp32", 0.8)), encoding="utf-8")
    bf16_path.write_text(json.dumps(_run_card("bf16", 0.796)), encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt = build_acceleration_receipt.build(
        argparse.Namespace(
            fp32_run_card=fp32_path,
            bf16_run_card=bf16_path,
            out=receipt_path,
        )
    )
    assert receipt["absolute_validation_dice_delta"] == pytest.approx(0.004)
    policy = acceleration_policy(
        receipt_path,
        expected_identity={"method": "unet", "source_config_sha256": "a" * 64},
    )
    assert policy["precision"] == "bf16" and policy["channels_last"] is True
    bf16_path.write_text(json.dumps(_run_card("bf16", 0.795)), encoding="utf-8")
    with pytest.raises(ValueError, match="rejected"):
        build_acceleration_receipt.build(
            argparse.Namespace(
                fp32_run_card=fp32_path,
                bf16_run_card=bf16_path,
                out=tmp_path / "rejected.json",
            )
        )


def test_amendment_authorizes_legacy_config_and_expanded_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = yaml.safe_load((ROOT / "configs" / "unet.yaml").read_text())
    config["training"]["threshold_candidates"] = [0.3, 0.4, 0.5, 0.6, 0.7]
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-registered validation grid"):
        load_config(config_path)
    assert load_config(
        config_path, allow_amendment_legacy_thresholds=True
    )["training"]["threshold_candidates"] == [0.3, 0.4, 0.5, 0.6, 0.7]

    manifest, splits, frame = _write_inputs(tmp_path / "selection", count=3)
    frame = pd.read_csv(manifest)
    split_payload = json.loads(splits.read_text())
    split_payload["splits"] = {
        subject: "val" for subject in frame["subject_id"].astype(str)
    }
    splits.write_text(json.dumps(split_payload), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"method": "unet", "task": "pixel", "config": {"seed": 0}}, checkpoint
    )
    probability_dir = tmp_path / "probability"
    items = []
    sources = {}
    for row in frame.to_dict("records"):
        image_id = str(row["image_id"])
        probability = np.full((48, 52), 0.15, dtype=np.float32)
        probability[12:36, 12:40] = 0.85
        path = probability_artifacts.safe_path(probability_dir, image_id)
        probability_artifacts.atomic_write(path, probability)
        items.append((image_id, path))
        sources[image_id] = str(row["source_sha256"])
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        probability_artifacts,
        "validate_threshold_amendment",
        lambda _path: probability_artifacts.THRESHOLD_AMENDMENT_SHA256,
    )
    card = {
        "status": "complete",
        "complete": True,
        "kind": "complete_probability_map_card",
        "method": "unet",
        "task": "pixel",
        "seed": 0,
        "split": "val",
        "probabilities_are_pre_threshold": True,
        "storage_format": "float16_npy",
        "storage_dtype": "float16",
        "derived_from_binary_mask": False,
        "binary_mask_reinterpreted": False,
        "labels_opened": False,
        "selection_source": "unselected_raw",
        "preprocess_cache_receipt_sha256": None,
        "threshold_amendment_sha256": probability_artifacts.THRESHOLD_AMENDMENT_SHA256,
        "identities": {
            "source_config_sha256": sha256_file(config_path),
            "checkpoint_sha256": sha256_file(checkpoint),
            "manifest_sha256": sha256_file(manifest),
            "split_sha256": sha256_file(splits),
        },
        "probability_set": probability_artifacts.set_identity(items, source_sha256=sources),
    }
    card["probability_set"]["complete"] = True
    card_path = tmp_path / "val_card.json"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    receipt = select_threshold_from_probabilities.select(
        argparse.Namespace(
            config=config_path,
            checkpoint=checkpoint,
            manifest=manifest,
            splits=splits,
            subset="finger_latent",
            probability_dir=probability_dir,
            probability_card=card_path,
            out=tmp_path / "threshold.json",
            selected_probability_card_out=tmp_path / "selected_card.json",
            threshold_amendment=amendment,
            cache_dir=None,
            limit=0,
        )
    )
    assert receipt["threshold_candidates"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert set(receipt["threshold_scores"]) == {
        "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"
    }
