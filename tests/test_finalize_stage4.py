"""Synthetic-only tests for the fail-closed Stage 4 orchestration scaffold."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_stage4.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("finalize_stage4_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


STAGE1_METHODS = (
    "gmfs_default",
    "gmfs_tuned",
    "sufs",
    "sufs_ft",
    "fingernet",
    "svm",
    "rf",
    "welm",
    "all_background",
    "all_foreground",
)
THREE_SEED_STAGE1 = {"sufs_ft", "svm", "rf", "welm"}


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _policies() -> list[dict[str, object]]:
    rows = []
    for method in (*STAGE1_METHODS, *MODULE.NATIVE_METHODS):
        if method in THREE_SEED_STAGE1:
            policy = {
                "kind": "seeded",
                "minimum_successful": 3,
                "target_successful": 3,
                "seeds": [0, 1, 2],
            }
        elif method in MODULE.NATIVE_METHODS:
            policy = {
                "kind": "seeded",
                "minimum_successful": 2,
                "target_successful": 3,
                "seeds": [0, 1, 2],
                "partial_row_requires_delivery_declared_failure": True,
            }
        else:
            policy = {
                "kind": "unseeded",
                "minimum_successful": 1,
                "target_successful": 1,
            }
        rows.append(
            {
                "key": method,
                "role": "reference" if method.startswith("all_") else "method",
                "seed_policy": policy,
            }
        )
    return rows


def _rows(methods: tuple[str, ...], groups: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    rows = []
    for method in methods:
        if method in THREE_SEED_STAGE1 or method in MODULE.NATIVE_METHODS:
            seeds: tuple[object, ...] = (0, 1, 2)
        else:
            seeds = ("",)
        for seed in seeds:
            for table, counts in groups.items():
                for group, count in counts.items():
                    row = {
                        "method": method,
                        "seed": seed,
                        "table": table,
                        "group_value": group,
                        "n_images": count,
                        "n_missing": 0,
                        "params_m": 1.0,
                        "ms_per_img": 1.25,
                        "device": "GPU",
                        "threshold": (
                            0.5 if method in MODULE.PATCH_NEURAL_METHODS else 0.2
                        ),
                    }
                    if method in MODULE.PATCH_NEURAL_METHODS:
                        metric_values = {
                            "accuracy": 0.5,
                            "error_rate": 0.5,
                            "dice": 0.0,
                            "jaccard": 0.0,
                            "miou": 0.25,
                            "mdr": 1.0,
                            "fdr": 0.0,
                        }
                    elif method in MODULE.TRAINABLE_NEURAL_METHODS:
                        metric_values = {
                            "accuracy": 1.0,
                            "error_rate": 0.0,
                            "dice": 1.0,
                            "jaccard": 1.0,
                            "miou": 1.0,
                            "mdr": 0.0,
                            "fdr": 0.0,
                        }
                    else:
                        metric_values = {metric: 0.5 for metric in MODULE.METRICS}
                    for metric, value in metric_values.items():
                        row[f"{metric}_mean"] = value
                        row[f"{metric}_std"] = 0.0
                        row[f"{metric}_pooled"] = value
                    rows.append(row)
    return rows


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_per_image_rows(
    restricted: Path, prediction_dir: Path, table: str
) -> list[dict[str, object]]:
    gt_column = "mask_path" if table == "A" else "quality_mask_path"
    manifest = restricted / "manifests" / "latent.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        records = [
            row for row in csv.DictReader(handle) if row["image_id"].startswith("test_")
        ]
    result = []
    for record in sorted(records, key=lambda item: item["image_id"]):
        gt = MODULE.to_target_ppi(
            MODULE.read_mask(record[gt_column]),
            float(record["ppi"]),
            500.0,
            is_mask=True,
        )
        prediction = MODULE.read_mask(
            prediction_dir / f"{record['image_id']}.png"
        )
        metrics = MODULE.image_metrics(prediction, gt)
        metrics.update(
            {
                "image_id": record["image_id"],
                "subject_id": record["subject_id"],
                "missing": False,
                "examiner_value": record["examiner_value"],
            }
        )
        result.append(metrics)
    return result


def _source_image_bytes(image_id: str) -> bytes:
    value = 48 + (sum(image_id.encode("utf-8")) % 160)
    image = np.full((4, 6), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _source_record(image_id: str) -> dict[str, str]:
    return {
        "image_id": image_id,
        "source_sha256": hashlib.sha256(_source_image_bytes(image_id)).hexdigest(),
    }


def _test_source_records(count: int = 2) -> list[dict[str, str]]:
    return [
        _source_record(f"test_{index}")
        for index in range(count)
    ]


def _val_source_records() -> list[dict[str, str]]:
    return [_source_record("val_0")]


def _synthetic_cache_receipt(
    path: Path,
    *,
    restricted: Path,
    manifest_sha: str,
) -> None:
    """Create a complete semantic cache receipt for the synthetic manifest."""
    manifest = restricted / "manifests" / "latent.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["image_id"])
    records = []
    for row in rows:
        image_id = row["image_id"]
        shape = (int(row["target_height"]), int(row["target_width"]))
        cached_image = path.parent / "images" / f"{image_id}.npy"
        cached_image.parent.mkdir(parents=True, exist_ok=True)
        source_image = MODULE.read_gray(row["image_path"])
        processed_image = MODULE.to_target_ppi(
            source_image, float(row["ppi"]), 500.0, is_mask=False
        ).astype(np.uint8)
        assert processed_image.shape == shape
        np.save(cached_image, processed_image, allow_pickle=False)
        cached_masks = {}
        source_masks = {}
        for column in ("mask_path", "quality_mask_path"):
            cached_mask = path.parent / "masks" / column / f"{image_id}.npy"
            cached_mask.parent.mkdir(parents=True, exist_ok=True)
            source_mask_path = Path(row[column])
            source_mask = MODULE.read_mask(source_mask_path)
            processed_mask = MODULE.to_target_ppi(
                source_mask, float(row["ppi"]), 500.0, is_mask=True
            ).astype(np.uint8) * 255
            assert processed_mask.shape == shape
            np.save(cached_mask, processed_mask, allow_pickle=False)
            cached_masks[column] = MODULE.sha256_file(cached_mask)
            source_masks[column] = MODULE.sha256_file(source_mask_path)
        records.append(
            {
                "image_id": image_id,
                "shape": list(shape),
                "source_image_sha256": row["source_sha256"],
                "image_sha256": MODULE.sha256_file(cached_image),
                "source_mask_sha256": source_masks,
                "mask_sha256": cached_masks,
            }
        )
    image_ids = [record["image_id"] for record in records]
    verified = sorted(
        image_ids,
        key=lambda value: (hashlib.sha256(value.encode()).digest(), value),
    )[:20]
    ordered_sha = hashlib.sha256(
        "".join(f"{image_id}\n" for image_id in image_ids).encode()
    ).hexdigest()
    _json(
        path,
        {
            "schema_version": 1,
            "kind": "uint8_500ppi_preprocessing_cache",
            "manifest_sha256": manifest_sha,
            "target_ppi": 500.0,
            "mask_columns": ["mask_path", "quality_mask_path"],
            "image_format": "uint8_npy",
            "mask_format": "uint8_0_255_npy",
            "record_count": len(records),
            "ordered_image_ids_sha256": ordered_sha,
            "records": records,
            "verification": {
                "algorithm": "sha256(image_id), lexical tie-break",
                "requested_count": 20,
                "verified_count": len(verified),
                "image_ids": verified,
                "array_identity": True,
            },
            "debug_limited": False,
            "implementation_sha256": MODULE.cache_implementation_sha256(),
        },
    )


def _synthetic_scoring_receipt(
    *,
    root: Path,
    restricted: Path,
    method: str,
    seed: int,
    config_sha: str,
    checkpoint_sha: str,
    manifest_sha: str,
    splits_sha: str,
    amendment_sha: str,
    selector_sha: str,
    inference_card: Path,
    test_card: Path,
    prediction_dir: Path,
    scientific_summary_path: Path,
) -> Path:
    """Create closed-schema synthetic evaluator evidence for finalizer tests."""
    test_payload = json.loads(test_card.read_text(encoding="utf-8"))
    selected_threshold = float(test_payload["selected_validation_threshold"])
    scoring = root / "scoring.json"
    training_data = "synthetic training"
    scoring_results = root / "scoring_results"
    scoring_results.mkdir(exist_ok=True)
    with scientific_summary_path.open("r", encoding="utf-8", newline="") as handle:
        scoring_rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row["method"] == method and row["seed"] == str(seed)
        ]
    scoring_summary = scoring_results / "summary.csv"
    _csv(scoring_summary, scoring_rows)
    identity_suffix = "_identity-" + hashlib.sha256(
        json.dumps(
            {"run_id": checkpoint_sha, "training_data": training_data},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    invocations = []
    for table in MODULE.TABLES:
        per_image = scoring_results / (
            f"per_image_{method}_finger_latent_{table}_seed{seed}"
            f"{identity_suffix}.csv"
        )
        _csv(
            per_image,
            _synthetic_per_image_rows(restricted, prediction_dir, table),
        )
        row_count, row_digest = MODULE._scientific_rows_for_run(
            scoring_summary, method, seed, table
        )
        argv = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            MODULE.SCORING_EVALUATOR_BOOTSTRAP,
            str(SCRIPT.with_name("evaluate_masks.py").resolve()),
            "--manifest",
            str((restricted / "manifests" / "latent.csv").resolve()),
            "--splits",
            str((restricted / "splits.json").resolve()),
            "--subsets",
            "finger_latent",
            "--method",
            method,
            "--display-name",
            method,
            "--pred-template",
            str((prediction_dir / "{image_id}.png").resolve()),
            "--pred-ppi",
            "500",
            "--target-ppi",
            "500",
            "--split",
            "test",
            "--gt-column",
            "mask_path" if table == "A" else "quality_mask_path",
            "--exclude",
            "examiner_value=NONPRINT",
            "--group-by",
            "examiner_value",
            "--seed",
            str(seed),
            "--year",
            "2022",
            "--family",
            "synthetic",
            "--training-data",
            training_data,
            "--comment",
            "synthetic",
            "--params-m",
            "1.0",
            "--device",
            "GPU",
            "--threshold",
            str(selected_threshold),
            "--run-id",
            checkpoint_sha,
            "--table",
            table,
            "--ms-per-img",
            "1.25",
            "--out",
            str(scoring_results.resolve()),
        ]
        invocations.append(
            {
                "table": table,
                "argv": argv,
                "exit_code": 0,
                "per_image": {
                    "path": str(per_image),
                    "sha256": MODULE.sha256_file(per_image),
                },
                "scientific_row_count": row_count,
                "scientific_rows_sha256": row_digest,
                "stdout_sha256": hashlib.sha256(f"{table}-stdout".encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            }
        )
    _json(
        scoring,
        {
            "schema_version": 1,
            "kind": MODULE.SCORING_RECEIPT_KIND,
            "method": method,
            "seed": seed,
            "selected_threshold": selected_threshold,
            "selection_frozen_before_test_scoring": True,
            "test_labels_opened_only_by_evaluator": True,
            "manifest_path": str((restricted / "manifests" / "latent.csv").resolve()),
            "splits_path": str((restricted / "splits.json").resolve()),
            "identities": {
                "source_config_sha256": config_sha,
                "manifest_sha256": manifest_sha,
                "splits_sha256": splits_sha,
                "test_probability_card_sha256": MODULE.sha256_file(test_card),
                "selector_receipt_sha256": selector_sha,
                "amendment_sha256": amendment_sha,
                "inference_card_sha256": MODULE.sha256_file(inference_card),
            },
            "producer": {
                "git_commit": "c" * 40,
                "git_dirty": False,
                "script_sha256": MODULE.sha256_file(
                    SCRIPT.with_name("score_threshold_run.py")
                ),
                "evaluator_sha256": MODULE.sha256_file(
                    SCRIPT.with_name("evaluate_masks.py")
                ),
            },
            "prediction_probability_derivation_records_sha256": test_payload[
                "mask_derivation_verification"
            ]["records_sha256"],
            "prediction_count": test_payload["mask_derivation_verification"]["count"],
            "scientific_summary": {
                "path": str(scoring_summary.resolve()),
                "sha256": MODULE.sha256_file(scoring_summary),
            },
            "invocations": invocations,
        },
    )
    return scoring


def _threshold_record(
    restricted: Path,
    amendment_sha: str,
    manifest_sha: str,
    splits_sha: str,
    method: str,
    seed: int,
    *,
    legacy: bool,
    scientific_summary_sha256: str | None,
    scientific_summary_path: Path | None = None,
) -> dict[str, object]:
    root = restricted / "threshold_evidence" / method / f"seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    config = root / "config.yaml"
    grid = (
        MODULE.PRIOR_THRESHOLD_GRID if legacy else MODULE.REPLACEMENT_THRESHOLD_GRID
    )
    task = "patch" if method in MODULE.PATCH_NEURAL_METHODS else "pixel"
    selected = 0.5 if task == "patch" else 0.2
    _json(
        config,
        {
            "schema_version": 1,
            "method": method,
            "seed": seed,
            "task": task,
            "data": {
                "target_ppi": 500,
                "subset": "finger_latent",
                **({"block_size": 16} if task == "patch" else {}),
            },
            "inference": (
                {"block_size": 16, "min_component_blocks": 4}
                if task == "patch"
                else {"pad_multiple": 32}
            ),
            "training": {
                **dict(MODULE.PROTOCOL_DATALOADER),
                "threshold_candidates": list(grid),
            },
        },
    )
    config_sha = MODULE.sha256_file(config)
    checkpoint = root / "checkpoint.bin"
    checkpoint.write_bytes(f"synthetic-checkpoint-{method}-{seed}".encode())
    checkpoint_sha = MODULE.sha256_file(checkpoint)
    run_card = root / "run.json"
    run_payload = {
        "method": method,
        "seed": seed,
        "source_config_sha256": config_sha,
        **(
            {"released": {"parameter_count": 1_000_000}}
            if method == "sufs_ft"
            else {
                "parameter_count": {
                    "total": 1_000_000,
                    "trainable": 1_000_000,
                }
            }
        ),
    }
    cache_receipt = root / "cache_receipt.json"
    _synthetic_cache_receipt(
        cache_receipt,
        restricted=restricted,
        manifest_sha=manifest_sha,
    )
    cache_sha = MODULE.sha256_file(cache_receipt)
    if not legacy:
        run_payload.update(
            {
                "selected_validation_threshold": selected,
                "task": task,
                "debug_limited": False,
                "train_images": 1,
                "validation_images": 1,
                "epochs_completed": 1,
                "epochs_requested": 1,
                "manifest_sha256": manifest_sha,
                "split_sha256": splits_sha,
                "threshold_amendment_sha256": amendment_sha,
                "checkpoint_sha256": {"best.pt": checkpoint_sha},
                "dataloader": dict(MODULE.PROTOCOL_DATALOADER),
                "preprocess_cache_receipt_sha256": cache_sha,
                "acceleration": {
                    "precision": "fp32",
                    "channels_last": False,
                    "acceptance_receipt_sha256": None,
                    "evaluation_only": False,
                    "production_eligible": True,
                    "reason": "synthetic fp32 test",
                },
                "provenance": {
                    "git_commit": "c" * 40,
                    "git_dirty": False,
                    "capture_timing": "process_start",
                    "command_argv": [
                        sys.executable,
                        str(
                            SCRIPT.with_name(
                                "train_patch.py" if task == "patch" else "train_seg.py"
                            )
                        ),
                        "--config",
                        str(config),
                        "--manifest",
                        str(restricted / "manifests" / "latent.csv"),
                        "--splits",
                        str(restricted / "splits.json"),
                        "--out",
                        str(root),
                        "--seed",
                        str(seed),
                        "--cache-dir",
                        str(cache_receipt.parent),
                        "--threshold-amendment",
                        str(restricted / "reports" / "threshold_amendment.json"),
                    ],
                },
            }
        )
    _json(
        run_card,
        run_payload,
    )

    cards = {}
    source_by_id = {
        record["image_id"]: record["source_sha256"]
        for record in [*_val_source_records(), *_test_source_records()]
    }
    for split, count in (("val", 1), ("test", 2)):
        probability_records = []
        derivation_records = []
        prediction_dir = root / "predictions"
        prediction_dir.mkdir(exist_ok=True)
        for index in range(count):
            probability = root / f"{split}_{index}.npy"
            shape = (1, 1) if method in MODULE.PATCH_NEURAL_METHODS else (2, 3)
            values = (
                np.full(shape, 0.25, dtype=np.float16)
                if task == "patch"
                else np.array(
                    [[0.25, 0.25, 0.15], [0.25, 0.15, 0.15]],
                    dtype=np.float16,
                )
            )
            np.save(probability, values, allow_pickle=False)
            probability_record = {
                    "image_id": f"{split}_{index}",
                    "filename": probability.name,
                    "path": str(probability),
                    "sha256": MODULE.sha256_file(probability),
                    "bytes": probability.stat().st_size,
                    "shape": list(shape),
                    "dtype": "float16",
                    "source_sha256": source_by_id[f"{split}_{index}"],
            }
            probability_records.append(probability_record)
            if split == "test":
                prediction = MODULE._probability_to_mask(
                    np.load(probability, allow_pickle=False),
                    target_shape=(2, 3),
                    threshold=selected,
                    method=method,
                    config=yaml.safe_load(config.read_text(encoding="utf-8")),
                ).astype(np.uint8) * 255
                prediction_path = prediction_dir / f"test_{index}.png"
                assert cv2.imwrite(str(prediction_path), prediction)
                derivation_records.append(
                    {
                        "image_id": f"test_{index}",
                        "mask_sha256": MODULE.sha256_file(prediction_path),
                        "probability_sha256": MODULE.sha256_file(probability),
                    }
                )
        normalized = [
            {key: record[key] for key in ("image_id", "sha256", "bytes")}
            for record in probability_records
        ]
        card = root / f"{split}_probability_card.json"
        payload = {
            "schema_version": 1,
            "kind": MODULE.PROBABILITY_CARD_KIND,
            "method": method,
            "seed": seed,
            "split": split,
            "source_config_sha256": config_sha,
            "checkpoint_sha256": checkpoint_sha,
            "threshold_amendment_sha256": amendment_sha,
            "manifest_sha256": manifest_sha,
            "split_sha256": splits_sha,
            "threshold_candidates": list(MODULE.REPLACEMENT_THRESHOLD_GRID),
            "probabilities_are_pre_threshold": True,
            "complete": True,
            "storage_dtype": "float16",
            "storage_format": "float16_npy",
            "derived_from_binary_mask": False,
            "preprocess_cache_receipt_sha256": cache_sha if split == "val" else None,
            "cache_used_for_image_decode": split == "val",
            "paper_timing_preprocess": (
                "not_paper_timing" if split == "val" else "on_the_fly_decode_and_resample"
            ),
            "mask_derivation_verification": {
                "complete": True,
                "count": count,
                "records_sha256": MODULE.canonical_sha256(derivation_records)
                if split == "test"
                else hashlib.sha256(
                    f"{method}-{seed}-{split}-derivation".encode()
                ).hexdigest(),
                "all_masks_rederived_from_retained_probabilities": True,
            },
            "probability_set": {
                "format": "float16_npy",
                "storage_dtype": "float16",
                "complete": True,
                "count": count,
                "sha256": MODULE.canonical_sha256(normalized),
                "records": probability_records,
            },
            "status": "complete",
            "debug_limited": False,
            "preempted": False,
            "errors": [],
            "provenance": {
                "git_commit": "c" * 40,
                "git_dirty": False,
                "capture_timing": "process_start",
                "command_argv": [
                    sys.executable,
                    str(
                        SCRIPT.with_name(
                            "infer_sufs.py" if method == "sufs_ft" else "infer_masks.py"
                        )
                    ),
                    "--config",
                    str(config),
                    "--checkpoint",
                    str(checkpoint),
                    "--manifest",
                    str(restricted / "manifests" / "latent.csv"),
                    "--splits",
                    str(restricted / "splits.json"),
                    "--probability-out",
                    str(root),
                    "--subset",
                    "finger_latent",
                    "--split",
                    split,
                    "--seed",
                    str(seed),
                ],
            },
        }
        if split == "val":
            payload.update(
                {
                    "selection_source": "unselected_raw",
                    "labels_opened": False,
                    "ordered_source_images_sha256": MODULE.canonical_sha256(
                        _val_source_records()
                    ),
                }
            )
        else:
            payload.update(
                {
                    "status": "complete",
                    "task": "pixel",
                    "subset": "finger_latent",
                    "device": "GPU",
                    "mean_ms_per_img": 1.25,
                    "ms_per_img": 1.25,
                    "parameter_count": 1_000_000,
                    "warmup_images": 10,
                    "written": count,
                    "skipped_valid": 0,
                    "per_image": [
                        {
                            "image_id": item["image_id"],
                            "ms_per_img": 1.25,
                            "mask_sha256": next(
                                record["mask_sha256"]
                                for record in derivation_records
                                if record["image_id"] == item["image_id"]
                            ),
                            "probability_sha256": item["sha256"],
                        }
                        for item in probability_records
                    ],
                    "predictions": [
                        {
                            "image_id": item["image_id"],
                            "ms_per_img": 1.25,
                        }
                        for item in probability_records
                    ],
                }
            )
            payload.update(
                {
                    "labels_opened": False,
                    "selection_source": "unselected_raw",
                    "selected_validation_threshold": selected,
                    "threshold_selected_before_test": True,
                    "ordered_source_images_sha256": MODULE.canonical_sha256(
                        _test_source_records()
                    ),
                }
            )
        _json(card, payload)
        cards[split] = card
    raw_validation = cards["val"]
    selected_validation = root / "selected_val_probability_card.json"
    selected_payload = json.loads(raw_validation.read_text(encoding="utf-8"))
    validation_scores = MODULE._recompute_validation_threshold_scores(
        selected_payload,
        method=method,
        config=yaml.safe_load(config.read_text(encoding="utf-8")),
        expected_universes=MODULE._manifest_split_probability_universes(
            restricted / "manifests" / "latent.csv", restricted / "splits.json"
        ),
        restricted_root=restricted,
    )
    selected_payload.update(
        {
            "selection_source": "validation_only",
            "labels_opened": True,
            "threshold_scores": {
                str(value): validation_scores[value]
                for value in MODULE.REPLACEMENT_THRESHOLD_GRID
            },
            "selected_threshold": selected,
            "selected_validation_threshold": selected,
            "selected_validation_dice": validation_scores[selected],
        }
    )
    _json(selected_validation, selected_payload)
    selector = root / "selector.json"
    _json(
        selector,
        {
            "schema_version": 1,
            "kind": MODULE.SELECTOR_RECEIPT_KIND,
            "method": method,
            "seed": seed,
            "source_config_sha256": config_sha,
            "checkpoint_sha256": checkpoint_sha,
            "amendment_sha256": amendment_sha,
            "raw_validation_probability_card_sha256": MODULE.sha256_file(
                raw_validation
            ),
            "validation_probability_card_sha256": MODULE.sha256_file(
                selected_validation
            ),
            "allowed_grid": list(MODULE.REPLACEMENT_THRESHOLD_GRID),
            "selection_split": "val",
            "selection_metric": "macro_image_dice",
            "selection_source": "validation_only",
            "tie_break": "closest_to_0.5_then_lower",
            "threshold_scores": {
                str(value): validation_scores[value]
                for value in MODULE.REPLACEMENT_THRESHOLD_GRID
            },
            "selected_validation_dice": validation_scores[selected],
            "validation_only": True,
            "test_or_heldout_opened": False,
            "selected_threshold": selected,
            "manifest_sha256": manifest_sha,
            "split_sha256": splits_sha,
        },
    )
    selector_sha = MODULE.sha256_file(selector)
    test_payload = json.loads(cards["test"].read_text(encoding="utf-8"))
    test_payload["threshold_receipt_sha256"] = selector_sha
    _json(cards["test"], test_payload)

    inference_card = root / "inference.json"
    inference_spec = root / "inference_spec.json"
    _json(inference_spec, {"schema_version": 1, "kind": "synthetic_inference_spec"})
    identities = {
        "source_config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "selector_receipt_sha256": selector_sha,
        "raw_validation_probability_card_sha256": MODULE.sha256_file(
            raw_validation
        ),
        "validation_probability_card_sha256": MODULE.sha256_file(
            selected_validation
        ),
        "test_probability_card_sha256": MODULE.sha256_file(cards["test"]),
        "amendment_sha256": amendment_sha,
        "manifest_sha256": manifest_sha,
        "split_sha256": splits_sha,
    }
    _json(
        inference_card,
        {
            "schema_version": 1,
            "kind": MODULE.TRAINABLE_INFERENCE_CARD_KIND,
            "method": method,
            "seed": seed,
            "selected_validation_threshold": selected,
            "selection_frozen_before_test_inference": True,
            "test_labels_opened": False,
            "identities": identities,
            "producer": {
                "git_commit": "c" * 40,
                "git_dirty": False,
                "script_sha256": MODULE.sha256_file(
                    SCRIPT.with_name("build_stage4_deliveries.py")
                ),
                "spec": {
                    "path": str(inference_spec),
                    "sha256": MODULE.sha256_file(inference_spec),
                },
                "results_or_metric_values_parsed": False,
                "payload_bytes_hashed": True,
            },
        },
    )

    result = {
        "method": method,
        "seed": seed,
        "training_lineage": (
            "legacy_completed_before_amendment" if legacy else "amended_training"
        ),
        "selected_threshold": selected,
        "config": {"path": str(config), "sha256": MODULE.sha256_file(config)},
        "run_card": {"path": str(run_card), "sha256": MODULE.sha256_file(run_card)},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
        },
        "inference_card": {
            "path": str(inference_card),
            "sha256": MODULE.sha256_file(inference_card),
        },
        "selector_receipt": {
            "path": str(selector),
            "sha256": selector_sha,
        },
        "raw_validation_probability_card": {
            "path": str(raw_validation),
            "sha256": MODULE.sha256_file(raw_validation),
        },
        "validation_probability_card": {
            "path": str(selected_validation),
            "sha256": MODULE.sha256_file(selected_validation),
        },
        "test_probability_card": {
            "path": str(cards["test"]),
            "sha256": MODULE.sha256_file(cards["test"]),
        },
    }
    if legacy:
        assert scientific_summary_sha256 is not None
        assert scientific_summary_path is not None
        completion_evidence = root / "legacy_completion_evidence.json"
        _json(
            completion_evidence,
            {
                "schema_version": 1,
                "kind": MODULE.LEGACY_COMPLETION_EVIDENCE_KIND,
                "method": method,
                "seed": seed,
                "original_run_card_sha256": MODULE.sha256_file(run_card),
                "original_config_sha256": config_sha,
                "original_checkpoint_sha256": checkpoint_sha,
                "threshold_amendment_sha256": amendment_sha,
                "completed_before_threshold_amendment": True,
                "historical_run_card_modified": False,
                "evidence_basis": (
                    "independent immutable run inventory recorded before the "
                    "threshold-policy amendment"
                ),
            },
        )
        scoring = root / "scoring.json"
        invocations = []
        training_data = "synthetic training"
        run_id = checkpoint_sha
        scoring_results = root / "scoring_results"
        scoring_results.mkdir()
        with scientific_summary_path.open("r", encoding="utf-8", newline="") as handle:
            scoring_rows = [
                dict(row)
                for row in csv.DictReader(handle)
                if row["method"] == method and row["seed"] == str(seed)
            ]
        scoring_summary = scoring_results / "summary.csv"
        _csv(scoring_summary, scoring_rows)
        identity_suffix = "_identity-" + hashlib.sha256(
            json.dumps(
                {"run_id": run_id, "training_data": training_data},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        for table in MODULE.TABLES:
            per_image = scoring_results / (
                f"per_image_{method}_finger_latent_{table}_seed{seed}"
                f"{identity_suffix}.csv"
            )
            _csv(
                per_image,
                _synthetic_per_image_rows(restricted, prediction_dir, table),
            )
            row_count, row_digest = MODULE._scientific_rows_for_run(
                scoring_summary, method, seed, table
            )
            argv = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                MODULE.SCORING_EVALUATOR_BOOTSTRAP,
                str(SCRIPT.with_name("evaluate_masks.py").resolve()),
                "--manifest",
                str((restricted / "manifests" / "latent.csv").resolve()),
                "--splits",
                str((restricted / "splits.json").resolve()),
                "--subsets",
                "finger_latent",
                "--method",
                method,
                "--display-name",
                method,
                "--pred-template",
                str((prediction_dir / "{image_id}.png").resolve()),
                "--pred-ppi",
                "500",
                "--target-ppi",
                "500",
                "--split",
                "test",
                "--gt-column",
                "mask_path" if table == "A" else "quality_mask_path",
                "--exclude",
                "examiner_value=NONPRINT",
                "--group-by",
                "examiner_value",
                "--seed",
                str(seed),
                "--year",
                "2022",
                "--family",
                "synthetic",
                "--training-data",
                training_data,
                "--comment",
                "synthetic",
                "--params-m",
                "1.0",
                "--device",
                "GPU",
                "--threshold",
                "0.2",
                "--run-id",
                run_id,
                "--table",
                table,
                "--ms-per-img",
                "1.25",
                "--out",
                str(scoring_results.resolve()),
            ]
            invocations.append(
                {
                    "table": table,
                    "argv": argv,
                    "exit_code": 0,
                    "per_image": {
                        "path": str(per_image),
                        "sha256": MODULE.sha256_file(per_image),
                    },
                    "scientific_row_count": row_count,
                    "scientific_rows_sha256": row_digest,
                    "stdout_sha256": hashlib.sha256(f"{table}-stdout".encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                }
            )
        _json(
            scoring,
            {
                "schema_version": 1,
                "kind": MODULE.SCORING_RECEIPT_KIND,
                "method": method,
                "seed": seed,
                "selected_threshold": 0.2,
                "selection_frozen_before_test_scoring": True,
                "test_labels_opened_only_by_evaluator": True,
                "manifest_path": str((restricted / "manifests" / "latent.csv").resolve()),
                "splits_path": str((restricted / "splits.json").resolve()),
                "identities": {
                    "source_config_sha256": config_sha,
                    "manifest_sha256": manifest_sha,
                    "splits_sha256": splits_sha,
                    "test_probability_card_sha256": MODULE.sha256_file(cards["test"]),
                    "selector_receipt_sha256": selector_sha,
                    "amendment_sha256": amendment_sha,
                    "inference_card_sha256": MODULE.sha256_file(inference_card),
                },
                "producer": {
                    "git_commit": "c" * 40,
                    "git_dirty": False,
                    "script_sha256": MODULE.sha256_file(
                        SCRIPT.with_name("score_threshold_run.py")
                    ),
                    "evaluator_sha256": MODULE.sha256_file(
                        SCRIPT.with_name("evaluate_masks.py")
                    ),
                },
                "prediction_probability_derivation_records_sha256": test_payload[
                    "mask_derivation_verification"
                ]["records_sha256"],
                "prediction_count": test_payload["mask_derivation_verification"]["count"],
                "scientific_summary": {
                    "path": str(scoring_summary.resolve()),
                    "sha256": MODULE.sha256_file(scoring_summary),
                },
                "invocations": invocations,
            },
        )
        result.update(
            {
                "legacy_completion_evidence": {
                    "path": str(completion_evidence),
                    "sha256": MODULE.sha256_file(completion_evidence),
                },
                "scoring_receipt": {
                    "path": str(scoring),
                    "sha256": MODULE.sha256_file(scoring),
                },
            }
        )
    else:
        assert scientific_summary_sha256 is not None
        assert scientific_summary_path is not None
        scoring = _synthetic_scoring_receipt(
            root=root,
            restricted=restricted,
            method=method,
            seed=seed,
            config_sha=config_sha,
            checkpoint_sha=checkpoint_sha,
            manifest_sha=manifest_sha,
            splits_sha=splits_sha,
            amendment_sha=amendment_sha,
            selector_sha=selector_sha,
            inference_card=inference_card,
            test_card=cards["test"],
            prediction_dir=prediction_dir,
            scientific_summary_path=scientific_summary_path,
        )
        result["scoring_receipt"] = {
            "path": str(scoring),
            "sha256": MODULE.sha256_file(scoring),
        }
        profile = root / "profile.json"
        _json(
            profile,
            {
                "schema_version": 1,
                "kind": MODULE.PER_EPOCH_PROFILE_KIND,
                "method": method,
                "seed": seed,
                "source_config_sha256": config_sha,
                "run_card_sha256": MODULE.sha256_file(run_card),
                "complete": True,
                "data_wait_seconds_recorded_per_epoch": True,
                "model_compute_seconds_recorded_per_epoch": True,
                "generic_device_utilisation_recorded": True,
                "generic_device_utilisation_unavailable_reason": None,
                "epochs_recorded": 1,
                "epoch_profiles": [
                    {
                        "epoch": 0,
                        "data_wait_seconds": 0.1,
                        "compute_seconds": 1.0,
                        "gpu_utilization_sample_count": 1,
                        "gpu_utilization_percent_mean": 50.0,
                    }
                ],
            },
        )
        result["per_epoch_training_profile_card"] = {
            "path": str(profile),
            "sha256": MODULE.sha256_file(profile),
        }
        result["preprocess_cache_receipt"] = {
            "path": str(cache_receipt),
            "sha256": cache_sha,
        }
    return result


def _attestation(amendment_sha: str, records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "amendment_sha256": amendment_sha,
        "allowed_grid": list(MODULE.REPLACEMENT_THRESHOLD_GRID),
        "validation_only": True,
        "old_grid_rejected": True,
        "records": records,
    }


def _released_probability_record(
    restricted: Path,
    method: str,
    *,
    available: bool,
    manifest_sha: str,
    splits_sha: str,
) -> dict[str, object]:
    root = restricted / "released_probability_evidence" / method
    root.mkdir(parents=True, exist_ok=True)
    base = {
        "method": method,
        "seed": None,
        "released_threshold_retuned": False,
        "binary_accepted_as_probability": False,
    }
    assert available, "supported released runners must retain raw probabilities"

    selected = 0.5
    inference = root / "inference.json"
    inference_spec = root / "inference_spec.json"
    _json(inference_spec, {"schema_version": 1, "kind": "synthetic_released_spec"})
    cards = {}
    test_prediction_set_sha = ""
    source_by_id = {
        record["image_id"]: record["source_sha256"]
        for record in [*_val_source_records(), *_test_source_records()]
    }
    for split, count in (("val", 1), ("test", 2)):
        probability_records = []
        derivation_records = []
        # Keep the newly reproduced probability-derived masks separate from
        # the frozen Stage 1 output.  Equal declarations must therefore be
        # backed by equality of two independently rehashed byte sets.
        output_root = root / f"{split}_probability_masks"
        output_root.mkdir(exist_ok=True)
        prediction_digest = hashlib.sha256()
        for index in range(count):
            probability = root / f"{split}_{index}.npy"
            np.save(
                probability,
                np.linspace(0.1, 0.9, num=6, dtype=np.float16).reshape(2, 3),
                allow_pickle=False,
            )
            record = {
                "image_id": f"{split}_{index}",
                "filename": probability.name,
                "path": str(probability),
                "sha256": MODULE.sha256_file(probability),
                "bytes": probability.stat().st_size,
                "shape": [2, 3],
                "dtype": "float16",
                "source_sha256": source_by_id[f"{split}_{index}"],
            }
            probability_records.append(record)
            stored = np.load(probability, allow_pickle=False).astype(np.float32)
            if method == "sufs":
                prediction = stored >= selected
            else:
                prediction = np.zeros(stored.shape, dtype=bool)
            mask_path = output_root / f"{split}_{index}.png"
            assert cv2.imwrite(str(mask_path), prediction.astype(np.uint8) * 255)
            mask_sha = MODULE.sha256_file(mask_path)
            prediction_digest.update(
                f"{split}_{index}:{mask_sha}\n".encode("utf-8")
            )
            derivation_records.append(
                {
                    "image_id": f"{split}_{index}",
                    "mask_sha256": mask_sha,
                    "probability_sha256": record["sha256"],
                }
            )
        normalized = [
            {key: record[key] for key in ("image_id", "sha256", "bytes")}
            for record in probability_records
        ]
        payload = {
            "schema_version": 1,
            "kind": MODULE.PROBABILITY_CARD_KIND,
            "method": method,
            "seed": None,
            "split": split,
            "manifest_sha256": manifest_sha,
            "split_sha256": splits_sha,
            "threshold_source": "released_fixed",
            "released_threshold_retuned": False,
            "selected_threshold": selected,
            "probabilities_are_pre_threshold": True,
            "binary_mask_reinterpreted": False,
            "complete": True,
            "storage_dtype": "float16",
            "storage_format": "float16_npy",
            "derived_from_binary_mask": False,
            "mask_derivation_verification": {
                "complete": True,
                "count": count,
                "records_sha256": MODULE.canonical_sha256(derivation_records),
                "all_masks_rederived_from_retained_probabilities": True,
                "all_masks_match_released_runner": True,
            },
            "probability_set": {
                "count": count,
                "sha256": MODULE.canonical_sha256(normalized),
                "records": probability_records,
            },
            "prediction_count": count,
            "prediction_set_sha256": prediction_digest.hexdigest(),
            "provenance": {
                "git_commit": "c" * 40,
                "git_dirty": False,
                "capture_timing": "process_start",
                "command_argv": [
                    sys.executable,
                    str(
                        SCRIPT.with_name(
                            "run_pyfing.py" if method == "sufs" else "run_fingernet.py"
                        )
                    ),
                    *(["run", "--method", "sufs"] if method == "sufs" else []),
                    "--manifest",
                    str(restricted / "manifests" / "latent.csv"),
                    "--splits",
                    str(restricted / "splits.json"),
                    "--subset",
                    "finger_latent",
                    "--split",
                    split,
                    "--output-root",
                    str(output_root),
                    "--probability-root",
                    str(root),
                ],
            },
        }
        if split == "val":
            payload.update(
                {
                    "selection_source": "not_used_released_fixed",
                    "ordered_source_images_sha256": MODULE.canonical_sha256(
                        _val_source_records()
                    ),
                }
            )
        else:
            payload.update(
                {
                    "labels_opened": False,
                    "threshold_fixed_before_test": True,
                    "ordered_source_images_sha256": MODULE.canonical_sha256(
                        _test_source_records()
                    ),
                    "prediction_count": count,
                }
            )
        card = root / f"{split}_probability_card.json"
        _json(card, payload)
        cards[split] = card
        if split == "test":
            test_prediction_set_sha = prediction_digest.hexdigest()
    frozen = root / "frozen_stage1_run.json"
    _json(
        frozen,
        {
            "method": method,
            "prediction_count": 2,
            "prediction_set_sha256": test_prediction_set_sha,
            "command": [
                sys.executable,
                str(
                    SCRIPT.with_name(
                        "run_pyfing.py" if method == "sufs" else "run_fingernet.py"
                    )
                ),
                *(["run", "--method", "sufs"] if method == "sufs" else []),
                "--manifest",
                str(restricted / "manifests" / "latent.csv"),
                "--splits",
                str(restricted / "splits.json"),
                "--subset",
                "finger_latent",
                "--split",
                "test",
                "--output-root",
                str(root / "test_masks"),
            ],
        },
    )
    _json(
        inference,
        {
            "schema_version": 1,
            "kind": MODULE.RELEASED_INFERENCE_CARD_KIND,
            "method": method,
            "seed": None,
            "raw_probability_exposed": True,
            "binary_accepted_as_probability": False,
            "released_threshold_retuned": False,
            "threshold_source": "released_fixed",
            "selected_threshold": selected,
            "prediction_count": 2,
            "frozen_prediction_set_sha256": test_prediction_set_sha,
            "probability_derived_prediction_set_sha256": test_prediction_set_sha,
            "full_test_prediction_set_byte_identical": True,
            "identities": {
                "validation_probability_card_sha256": MODULE.sha256_file(
                    cards["val"]
                ),
                "test_probability_card_sha256": MODULE.sha256_file(cards["test"]),
                "frozen_stage1_run_card": {
                    "path": str(frozen),
                    "sha256": MODULE.sha256_file(frozen),
                },
                "frozen_stage1_run_card_sha256": MODULE.sha256_file(frozen),
            },
            "producer": {
                "git_commit": "c" * 40,
                "git_dirty": False,
                "script_sha256": MODULE.sha256_file(
                    SCRIPT.with_name("build_stage4_deliveries.py")
                ),
                "spec": {
                    "path": str(inference_spec),
                    "sha256": MODULE.sha256_file(inference_spec),
                },
                "results_or_metric_values_parsed": False,
                "payload_bytes_hashed": True,
            },
        },
    )
    return {
        **base,
        "availability": "available",
        "threshold_source": "released_fixed",
        "selected_threshold": selected,
        "frozen_stage1_reference_artifact": {
            "name": "scoring_audit",
            "sha256": "filled-by-case",
        },
        "inference_card": {
            "path": str(inference),
            "sha256": MODULE.sha256_file(inference),
        },
        "validation_probability_card": {
            "path": str(cards["val"]),
            "sha256": MODULE.sha256_file(cards["val"]),
        },
        "test_probability_card": {
            "path": str(cards["test"]),
            "sha256": MODULE.sha256_file(cards["test"]),
        },
    }


def _case(tmp_path: Path) -> SimpleNamespace:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    manifest = restricted / "manifests" / "latent.csv"
    manifest.parent.mkdir(parents=True)
    synthetic = restricted / "synthetic"
    synthetic.mkdir()
    manifest_rows = []
    for image_id, subject_id, examiner_value, source_sha in (
        ("train_0", "STRAIN", "VALUE", _source_record("train_0")["source_sha256"]),
        ("val_0", "SVAL", "VALUE", _val_source_records()[0]["source_sha256"]),
        ("test_0", "STEST", "VALUE", _test_source_records()[0]["source_sha256"]),
        ("test_1", "STEST", "NOVALUE", _test_source_records()[1]["source_sha256"]),
    ):
        image_path = synthetic / f"{image_id}.png"
        image_path.write_bytes(_source_image_bytes(image_id))
        target_mask = np.array(
            [[255, 255, 0], [255, 0, 0]], dtype=np.uint8
        )
        source_mask = np.repeat(np.repeat(target_mask, 2, axis=0), 2, axis=1)
        mask_path = synthetic / f"{image_id}_gt.png"
        quality_path = synthetic / f"{image_id}_quality.png"
        assert cv2.imwrite(str(mask_path), source_mask)
        assert cv2.imwrite(str(quality_path), source_mask)
        assert MODULE.sha256_file(image_path) == source_sha
        manifest_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "quality_mask_path": str(quality_path),
                "subject_id": subject_id,
                "subset": "finger_latent",
                "ppi": 1000,
                "source_sha256": source_sha,
                "target_height": 2,
                "target_width": 3,
                "examiner_value": examiner_value,
            }
        )
    _csv(manifest, manifest_rows)
    splits = restricted / "splits.json"
    _json(
        splits,
        {
            "seed": 42,
            "splits": {"STRAIN": "train", "SVAL": "val", "STEST": "test"},
        },
    )
    manifest_sha = MODULE.sha256_file(manifest)
    splits_sha = MODULE.sha256_file(splits)
    groups = {
        "A": {"overall": 2, "VALUE": 1, "NOVALUE": 1},
        "A_prime": {"overall": 2, "VALUE": 1, "NOVALUE": 1},
    }
    stage1_summary = restricted / "stage1" / "summary.csv"
    stage1_rows = _rows(STAGE1_METHODS, groups)
    _csv(stage1_summary, stage1_rows)
    stage1_paths = {"summary": stage1_summary}
    frozen_released_runs = {}
    for released_method in MODULE.RELEASED_NEURAL_METHODS:
        released_root = restricted / "released_probability_evidence" / released_method
        frozen_run = released_root / "frozen_stage1_run.json"
        frozen_masks = released_root / "test_masks"
        frozen_masks.mkdir(parents=True, exist_ok=True)
        frozen_digest = hashlib.sha256()
        released_probability = np.linspace(
            0.1, 0.9, num=6, dtype=np.float16
        ).reshape(2, 3)
        for index in range(2):
            released_prediction = (
                released_probability >= 0.5
                if released_method == "sufs"
                else np.zeros((2, 3), dtype=bool)
            )
            released_mask = frozen_masks / f"test_{index}.png"
            assert cv2.imwrite(
                str(released_mask), released_prediction.astype(np.uint8) * 255
            )
            frozen_digest.update(
                f"test_{index}:{MODULE.sha256_file(released_mask)}\n".encode("utf-8")
            )
        _json(
            frozen_run,
            {
                "method": released_method,
                "prediction_count": 2,
                "prediction_set_sha256": frozen_digest.hexdigest(),
                "command": [
                    sys.executable,
                    str(
                        SCRIPT.with_name(
                            "run_pyfing.py"
                            if released_method == "sufs"
                            else "run_fingernet.py"
                        )
                    ),
                    *(
                        ["run", "--method", "sufs"]
                        if released_method == "sufs"
                        else []
                    ),
                    "--manifest",
                    str(manifest),
                    "--splits",
                    str(restricted / "splits.json"),
                    "--subset",
                    "finger_latent",
                    "--split",
                    "test",
                    "--output-root",
                    str(frozen_masks),
                ],
            },
        )
        frozen_released_runs[released_method] = frozen_run
    for name in ("scoring_audit", "finalizer_audit", "journal", "table_metadata"):
        path = restricted / "stage1" / f"{name}.json"
        payload = {"kind": f"synthetic_{name}"}
        if name == "scoring_audit":
            payload["released_run_card_sha256"] = {
                method: MODULE.sha256_file(path)
                for method, path in sorted(frozen_released_runs.items())
            }
        _json(path, payload)
        stage1_paths[name] = path

    stage3_root = restricted / "stage3"
    stage3_summary = stage3_root / "results" / "summary.csv"
    _csv(stage3_summary, _rows(MODULE.NATIVE_METHODS, groups))
    stage3_success = {method: [0, 1, 2] for method in MODULE.NATIVE_METHODS}
    method_gate = {
        method: {
            "successful_seeds": seeds,
            "failed_seeds": [],
            "missing_seeds": [],
            "n_successful": 3,
            "minimum_required": 2,
            "pass": True,
        }
        for method, seeds in stage3_success.items()
    }
    newly = [
        {"method": method, "seed": seed}
        for method, seeds in stage3_success.items()
        for seed in seeds
        if not (method == "unet" and seed == 0)
    ]
    policies = _policies()
    plan = restricted / "reports" / "stage4_analysis_plan.json"
    plan_payload = {
        "schema_version": 1,
        "kind": MODULE.PLAN_KIND,
        "results_opened_during_creation": False,
        "heldout_payloads_opened_during_creation": False,
        "method_universe": {
            "methods": policies,
            "presentation_rows": len(policies),
            "real_method_rows": len(policies) - 2,
            "reference_rows": 2,
        },
        "test_universe": {"tables": groups},
        "threshold_policy": {
            "A_and_A_prime_reuse_identical_predictions": True,
            "allowed_trainable_grid": list(MODULE.PRIOR_THRESHOLD_GRID),
            "source": "frozen validation-selected run cards",
            "stage4_inference_or_rescoring_permitted": False,
            "stage4_reselection_permitted": False,
        },
        "inputs": {
            "manifest": {"sha256": manifest_sha},
            "splits": {"sha256": splits_sha},
            "stage1": {
                "scoring_audit_sha256": MODULE.sha256_file(stage1_paths["scoring_audit"]),
                "finalizer_audit_sha256": MODULE.sha256_file(stage1_paths["finalizer_audit"]),
                "journal_sha256": MODULE.sha256_file(stage1_paths["journal"]),
                "summary": {
                    "sha256": MODULE.sha256_file(stage1_summary),
                    "rows": len(stage1_rows),
                },
                "table_metadata_sha256": MODULE.sha256_file(stage1_paths["table_metadata"]),
            },
        },
        "future_input_binding": {
            "stage2": {"required_kind": MODULE.STAGE2_KIND},
            "stage3": {"required_kind": MODULE.STAGE3_KIND},
        },
    }
    _json(plan, plan_payload)
    plan_sha = MODULE.sha256_file(plan)

    amendment = restricted / "reports" / "threshold_amendment.json"
    _json(
        amendment,
        {
            "schema_version": 1,
            "kind": MODULE.AMENDMENT_KIND,
            "base_plan_sha256": plan_sha,
            "supersedes_exactly": ["threshold_policy"],
            "prior_policy": plan_payload["threshold_policy"],
            "replacement_policy": {
                "A_and_A_prime_reuse_identical_predictions": True,
                "allowed_trainable_grid": list(MODULE.REPLACEMENT_THRESHOLD_GRID),
                "binary_masks_accepted_as_probability_maps": False,
                "completed_trainable_pixel_models_reselected": True,
                "released_zero_shot_thresholds_retuned": False,
                "selection_data": "validation_only",
                "source": "frozen validation-selected run cards",
                "stage4_inference_or_rescoring_permitted": False,
                "stage4_reselection_permitted": False,
                "test_masks_regenerated_from_saved_probability_maps": True,
            },
            "required_attestations": {
                "legacy_binary_only_outputs_must_be_superseded": True,
                "probability_maps_required_for_val_and_test": True,
                "test_or_heldout_opened_for_selection": False,
                "threshold_reported_per_trainable_pixel_run": True,
                "validation_only": True,
            },
            "creation_scope": {
                "ground_truth_opened": False,
                "images_opened": False,
                "model_or_checkpoint_opened": False,
                "prediction_or_probability_maps_opened": False,
                "results_opened": False,
            },
            "generator": {
                "git_commit": "d" * 40,
                "script_sha256": "e" * 64,
            },
            "authorization": {
                "source": "operator_instruction",
                "reason": "synthetic test",
                "recorded_at_utc": "2026-10-08T00:00:00Z",
            },
        },
    )
    amendment_sha = MODULE.sha256_file(amendment)

    def delivery_producer(name: str) -> dict[str, object]:
        spec_path = restricted / "builder_specs" / f"{name}.json"
        _json(spec_path, {"schema_version": 1, "kind": f"synthetic_{name}_spec"})
        return {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "script_sha256": MODULE.sha256_file(
                SCRIPT.with_name("build_stage4_deliveries.py")
            ),
            "spec": {
                "path": str(spec_path),
                "sha256": MODULE.sha256_file(spec_path),
            },
            "results_or_metric_values_parsed": True,
            "payload_bytes_hashed": True,
        }

    data_inputs = {
        "manifest": {"path": str(manifest), "sha256": manifest_sha},
        "splits": {"path": str(splits), "sha256": splits_sha},
    }

    stage1_reselection_root = restricted / "stage1_reselection"
    stage1_reselection_summary = stage1_reselection_root / "results" / "summary.csv"
    _csv(stage1_reselection_summary, _rows(("sufs_ft",), groups))
    stage1_reselection_delivery = stage1_reselection_root / "delivery.json"
    stage1_records = [
        _threshold_record(
            restricted,
            amendment_sha,
            manifest_sha,
            splits_sha,
            "sufs_ft",
            seed,
            legacy=True,
            scientific_summary_sha256=MODULE.sha256_file(stage1_reselection_summary),
            scientific_summary_path=stage1_reselection_summary,
        )
        for seed in (0, 1, 2)
    ]
    released_probability_records = [
        _released_probability_record(
            restricted,
            "sufs",
            available=True,
            manifest_sha=manifest_sha,
            splits_sha=splits_sha,
        ),
        _released_probability_record(
            restricted,
            "fingernet",
            available=True,
            manifest_sha=manifest_sha,
            splits_sha=splits_sha,
        ),
    ]
    for record in released_probability_records:
        if record.get("availability") == "available":
            record["frozen_stage1_reference_artifact"]["sha256"] = (
                MODULE.sha256_file(stage1_paths["scoring_audit"])
            )
    _json(
        stage1_reselection_delivery,
        {
            "schema_version": 1,
            "kind": MODULE.STAGE1_RESELECTION_KIND,
            "supersedes": {
                "stage1_summary_sha256": MODULE.sha256_file(stage1_summary),
                "methods": ["sufs_ft"],
            },
            "frozen_stage1_artifacts_sha256": MODULE.canonical_sha256(
                {
                    name: MODULE.sha256_file(path)
                    for name, path in sorted(stage1_paths.items())
                }
            ),
            "ordered_source_images_sha256": MODULE.canonical_sha256(
                _test_source_records()
            ),
            "data_inputs": data_inputs,
            "threshold_policy_attestation": _attestation(amendment_sha, stage1_records),
            "probability_retention_attestation": {
                "schema_version": 1,
                "kind": MODULE.PROBABILITY_RETENTION_KIND,
                "binary_masks_accepted_as_probability_maps": False,
                "records": released_probability_records,
            },
            "outputs": {
                "results/summary.csv": MODULE.sha256_file(stage1_reselection_summary)
            },
            "producer": delivery_producer("stage1"),
        },
    )

    stage2_delivery = restricted / "stage2" / "delivery.json"
    _json(
        stage2_delivery,
        {
            "schema_version": 1,
            "kind": MODULE.STAGE2_KIND,
            "legacy_immutable": True,
        },
    )

    stage2_reselection_root = restricted / "stage2_reselection"
    stage2_reselection_summary = stage2_reselection_root / "results" / "summary.csv"
    stage2_rows = [
        row for row in _rows(("unet",), groups) if row["seed"] == 0
    ]
    _csv(stage2_reselection_summary, stage2_rows)
    stage2_reselection_delivery = stage2_reselection_root / "delivery.json"
    stage2_records = [
        _threshold_record(
            restricted,
            amendment_sha,
            manifest_sha,
            splits_sha,
            "unet",
            0,
            legacy=True,
            scientific_summary_sha256=MODULE.sha256_file(stage2_reselection_summary),
            scientific_summary_path=stage2_reselection_summary,
        )
    ]
    replacement_rows_sha = MODULE.canonical_sha256(
        [
            {
                "method": row["method"],
                "seed": row["seed"],
                "table": row["table"],
                "group_value": row["group_value"],
            }
            for row in stage2_rows
        ]
    )
    _json(
        stage2_reselection_delivery,
        {
            "schema_version": 1,
            "kind": MODULE.STAGE2_RESELECTION_KIND,
            "supersedes": {
                "stage2_delivery_sha256": MODULE.sha256_file(stage2_delivery),
                "methods": ["unet"],
                "seeds": [0],
            },
            "ordered_source_images_sha256": MODULE.canonical_sha256(
                _test_source_records()
            ),
            "data_inputs": data_inputs,
            "scientific_rows_sha256": replacement_rows_sha,
            "threshold_policy_attestation": _attestation(amendment_sha, stage2_records),
            "outputs": {
                "results/summary.csv": MODULE.sha256_file(stage2_reselection_summary)
            },
            "producer": delivery_producer("stage2"),
        },
    )

    stage3_delivery = stage3_root / "delivery.json"
    stage3_records = [
        _threshold_record(
            restricted,
            amendment_sha,
            manifest_sha,
            splits_sha,
            method,
            seed,
            legacy=False,
            scientific_summary_sha256=MODULE.sha256_file(stage3_summary),
            scientific_summary_path=stage3_summary,
        )
        for method in MODULE.TRAINABLE_NEURAL_METHODS
        if method != "sufs_ft"
        for seed in stage3_success[method]
        if not (method == "unet" and seed == 0)
    ]
    newly = [
        {
            "method": record["method"],
            "seed": record["seed"],
            "config_sha256": record["config"]["sha256"],
            "run_card_sha256": record["run_card"]["sha256"],
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "inference_card_sha256": "d" * 64,
            "training_git_commit": "c" * 40,
            "inference_git_commit": "c" * 40,
            "params_m": 1.0,
        }
        for record in stage3_records
    ]
    stage3_payload = {
        "schema_version": 1,
        "kind": MODULE.STAGE3_KIND,
        "manifest_sha256": manifest_sha,
        "splits_sha256": splits_sha,
        "ordered_source_images_sha256": MODULE.canonical_sha256(
            _test_source_records()
        ),
        "data_inputs": data_inputs,
        "stage2_unet_seed0": {
            "legacy_provenance_delivery_sha256": MODULE.sha256_file(stage2_delivery),
            "threshold_reselection_delivery_sha256": MODULE.sha256_file(
                stage2_reselection_delivery
            ),
            "replacement_scientific_rows_sha256": replacement_rows_sha,
            "reused_reselected_not_rescored": True,
        },
        "gate_evidence": {"status": "PASS", "methods": method_gate},
        "newly_scored_seeds": newly,
        "threshold_policy_attestation": _attestation(amendment_sha, stage3_records),
        "outputs": {"results/summary.csv": MODULE.sha256_file(stage3_summary)},
        "producer": delivery_producer("stage3"),
    }
    _json(stage3_delivery, stage3_payload)

    code = {
        "git_commit": "c" * 40,
        "git_dirty": False,
        "script_sha256": MODULE.sha256_file(SCRIPT),
    }
    lock_path = restricted / "reports" / "stage4_input_lock.json"
    return SimpleNamespace(
        restricted=restricted,
        manifest=manifest,
        splits=splits,
        groups=groups,
        stage1_summary=stage1_summary,
        stage1_paths=stage1_paths,
        stage1_reselection_delivery=stage1_reselection_delivery,
        stage2_delivery=stage2_delivery,
        stage2_reselection_delivery=stage2_reselection_delivery,
        stage3_delivery=stage3_delivery,
        stage3_summary=stage3_summary,
        plan=plan,
        plan_sha=plan_sha,
        amendment=amendment,
        amendment_sha=amendment_sha,
        code=code,
        lock_path=lock_path,
    )


def _lock(case: SimpleNamespace) -> dict[str, object]:
    return MODULE.build_input_lock(
        plan_path=case.plan,
        expected_plan_sha256=case.plan_sha,
        amendment_path=case.amendment,
        expected_amendment_sha256=case.amendment_sha,
        stage1_paths=case.stage1_paths,
        stage1_reselection_delivery_path=case.stage1_reselection_delivery,
        stage2_delivery_path=case.stage2_delivery,
        stage2_reselection_delivery_path=case.stage2_reselection_delivery,
        stage3_delivery_path=case.stage3_delivery,
        restricted_root=case.restricted,
        code_identity=case.code,
    )


def _first_stage3_record(case: SimpleNamespace) -> tuple[dict[str, object], dict[str, object]]:
    delivery = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    record = delivery["threshold_policy_attestation"]["records"][0]
    return delivery, record


def test_stage3_probability_producer_commit_is_frozen(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3, first = _first_stage3_record(case)
    for frozen in stage3["newly_scored_seeds"]:
        if frozen["method"] == first["method"] and frozen["seed"] == first["seed"]:
            frozen["inference_git_commit"] = "d" * 40
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="inference commit differs"):
        _lock(case)


def test_stage3_parameter_count_is_bound_to_run_card(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3, first = _first_stage3_record(case)
    for frozen in stage3["newly_scored_seeds"]:
        if frozen["method"] == first["method"] and frozen["seed"] == first["seed"]:
            frozen["params_m"] = 2.0
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="parameter count differs"):
        _lock(case)


@pytest.mark.parametrize("mutation", ["missing_record", "wrong_mean", "wrong_warmup"])
def test_test_timing_is_recomputed_from_full_probability_universe(
    tmp_path: Path, mutation: str
) -> None:
    case = _case(tmp_path)
    _, record = _first_stage3_record(case)
    card_path = Path(record["test_probability_card"]["path"])
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if mutation == "missing_record":
        card["per_image"] = card["per_image"][:-1]
    elif mutation == "wrong_mean":
        card["mean_ms_per_img"] = 99.0
    else:
        card["warmup_images"] = 9
    with pytest.raises(MODULE.Stage4Error, match="test timing"):
        MODULE._validate_trainable_inference_provenance(
            card,
            method=str(record["method"]),
            seed=int(record["seed"]),
            split="test",
            expected_commit="c" * 40,
            config_path=Path(record["config"]["path"]),
            checkpoint_path=Path(record["checkpoint"]["path"]),
            manifest_path=case.manifest,
            splits_path=case.splits,
            expected_count=2,
        )


def test_input_lock_is_results_blind_write_once_and_replayable(tmp_path: Path) -> None:
    case = _case(tmp_path)
    candidate = _lock(case)
    assert candidate["creation_scope"]["scientific_csv_content_parsed"] is True
    assert candidate["creation_scope"]["metric_values_inspected"] is True
    assert candidate["creation_scope"]["model_selection_performed"] is False
    assert (
        candidate["creation_scope"]["metric_use"]
        == "integrity_replay_only_no_selection"
    )
    assert candidate["scientific_inputs"]["stage2_summary_included"] is False
    assert "provenance only" in candidate["stage2"]["role"]
    assert candidate["stage2_reselection"]["role"].startswith("superseding")
    assert candidate["expected_universe"]["stage1_physical_summary_rows"] == 108
    assert candidate["expected_universe"]["stage1_scientific_rows"] == 90
    assert candidate["expected_universe"]["stage1_reselection_summary_rows"] == 18
    assert candidate["expected_universe"]["stage3_summary_rows"] == 144
    retention = candidate["stage1_reselection"][
        "released_probability_retention_evidence"
    ]["records"]
    assert {record["method"]: record["availability"] for record in retention} == {
        "fingernet": "available",
        "sufs": "available",
    }

    assert MODULE._publish_json_once(case.lock_path, candidate) == "created"
    assert MODULE._publish_json_once(case.lock_path, candidate) == "verified_identical"
    assert MODULE.verify_input_lock(case.lock_path, case.restricted, case.code) == candidate
    changed = dict(candidate)
    changed["automatic_heldout_rescore_permitted"] = True
    with pytest.raises(MODULE.Stage4Error, match="non-identical"):
        MODULE._publish_json_once(case.lock_path, changed)


def test_json_publication_race_never_follows_or_overwrites_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "immutable.json"
    victim = tmp_path / "victim.json"
    victim.write_bytes(b"do not replace\n")
    original_open = MODULE.os.open
    injected = False

    def racing_open(path: object, flags: int, *args: object) -> int:
        nonlocal injected
        if Path(path) == target and flags & MODULE.os.O_CREAT and not injected:
            injected = True
            target.symlink_to(victim)
        return original_open(path, flags, *args)

    monkeypatch.setattr(MODULE.os, "open", racing_open)
    with pytest.raises(MODULE.Stage4Error, match="regular file"):
        MODULE._publish_json_once(target, {"synthetic": True})
    assert injected
    assert target.is_symlink()
    assert victim.read_bytes() == b"do not replace\n"


def test_stage3_extra_file_is_rejected_before_lock(tmp_path: Path) -> None:
    case = _case(tmp_path)
    (case.stage3_delivery.parent / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(MODULE.Stage4Error, match="file universe"):
        _lock(case)


def test_stage2_is_linked_by_stage3_not_substitutable(tmp_path: Path) -> None:
    case = _case(tmp_path)
    payload = json.loads(case.stage2_delivery.read_text(encoding="utf-8"))
    payload["changed"] = True
    _json(case.stage2_delivery, payload)
    with pytest.raises(MODULE.Stage4Error, match="does not supersede"):
        _lock(case)


def test_amendment_must_be_externally_registered_and_narrow(tmp_path: Path) -> None:
    case = _case(tmp_path)
    with pytest.raises(MODULE.Stage4Error, match="external registration"):
        MODULE.build_input_lock(
            plan_path=case.plan,
            expected_plan_sha256=case.plan_sha,
            amendment_path=case.amendment,
            expected_amendment_sha256="0" * 64,
            stage1_paths=case.stage1_paths,
            stage1_reselection_delivery_path=case.stage1_reselection_delivery,
            stage2_delivery_path=case.stage2_delivery,
            stage2_reselection_delivery_path=case.stage2_reselection_delivery,
            stage3_delivery_path=case.stage3_delivery,
            restricted_root=case.restricted,
            code_identity=case.code,
        )

    payload = json.loads(case.amendment.read_text(encoding="utf-8"))
    payload["supersedes_exactly"] = ["threshold_policy", "inputs"]
    _json(case.amendment, payload)
    with pytest.raises(MODULE.Stage4Error, match="supersede another"):
        MODULE.build_input_lock(
            plan_path=case.plan,
            expected_plan_sha256=case.plan_sha,
            amendment_path=case.amendment,
            expected_amendment_sha256=MODULE.sha256_file(case.amendment),
            stage1_paths=case.stage1_paths,
            stage1_reselection_delivery_path=case.stage1_reselection_delivery,
            stage2_delivery_path=case.stage2_delivery,
            stage2_reselection_delivery_path=case.stage2_reselection_delivery,
            stage3_delivery_path=case.stage3_delivery,
            restricted_root=case.restricted,
            code_identity=case.code,
        )


def test_amended_stage3_run_cannot_claim_a_legacy_grid(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    config_record = stage3["threshold_policy_attestation"]["records"][0]["config"]
    config = Path(config_record["path"])
    config.write_text(
        "training:\n  threshold_candidates: [0.3, 0.4, 0.5, 0.6, 0.7]\n",
        encoding="utf-8",
    )
    config_record["sha256"] = MODULE.sha256_file(config)
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="training config grid differs"):
        _lock(case)


def test_legacy_exception_is_exact_and_cannot_be_claimed_by_stage3(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    record = stage3["threshold_policy_attestation"]["records"][0]
    record["training_lineage"] = "legacy_completed_before_amendment"
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="invalid training-lineage exception"):
        _lock(case)


def test_legacy_completed_run_requires_original_grid(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage2 = json.loads(case.stage2_reselection_delivery.read_text(encoding="utf-8"))
    config_record = stage2["threshold_policy_attestation"]["records"][0]["config"]
    config = Path(config_record["path"])
    config.write_text(
        "training:\n  threshold_candidates: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]\n",
        encoding="utf-8",
    )
    config_record["sha256"] = MODULE.sha256_file(config)
    _json(case.stage2_reselection_delivery, stage2)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    stage3["stage2_unet_seed0"]["threshold_reselection_delivery_sha256"] = (
        MODULE.sha256_file(case.stage2_reselection_delivery)
    )
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="training config grid differs"):
        _lock(case)


def test_legacy_run_card_stays_immutable_and_external_evidence_binds_it(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    stage2 = json.loads(case.stage2_reselection_delivery.read_text(encoding="utf-8"))
    record = stage2["threshold_policy_attestation"]["records"][0]
    run_path = Path(record["run_card"]["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert "completed_before_threshold_amendment" not in run
    lock = _lock(case)
    checked = lock["stage2_reselection"]["threshold_policy_evidence"]["records"][0]
    evidence_path = Path(record["legacy_completion_evidence"]["path"])
    assert checked["run_card_sha256"] == MODULE.sha256_file(run_path)
    assert checked["legacy_completion_evidence_sha256"] == MODULE.sha256_file(
        evidence_path
    )


def test_legacy_completion_evidence_must_bind_original_run_hash(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage2 = json.loads(case.stage2_reselection_delivery.read_text(encoding="utf-8"))
    record = stage2["threshold_policy_attestation"]["records"][0]
    evidence_record = record["legacy_completion_evidence"]
    evidence_path = Path(evidence_record["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["original_run_card_sha256"] = "0" * 64
    _json(evidence_path, evidence)
    evidence_record["sha256"] = MODULE.sha256_file(evidence_path)
    _json(case.stage2_reselection_delivery, stage2)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    stage3["stage2_unet_seed0"]["threshold_reselection_delivery_sha256"] = (
        MODULE.sha256_file(case.stage2_reselection_delivery)
    )
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="completion evidence is not bound"):
        _lock(case)


def test_probability_map_tampering_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage2 = json.loads(case.stage2_reselection_delivery.read_text(encoding="utf-8"))
    card_path = Path(
        stage2["threshold_policy_attestation"]["records"][0][
            "test_probability_card"
        ]["path"]
    )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    probability_path = Path(card["probability_set"]["records"][0]["path"])
    probability_path.write_bytes(b"tampered")
    with pytest.raises(MODULE.Stage4Error, match="probability map changed"):
        _lock(case)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("not_npy", "invalid test NPY"),
        ("float32", "2-D float16"),
        ("wrong_shape", "shape differs from manifest"),
        ("out_of_range", "outside \\[0,1\\]"),
        ("wrong_source", "source identity changed"),
    ],
)
def test_probability_semantics_and_source_identity_are_not_self_asserted(
    tmp_path: Path, mutation: str, message: str
) -> None:
    case = _case(tmp_path)
    stage2 = json.loads(case.stage2_reselection_delivery.read_text(encoding="utf-8"))
    record = stage2["threshold_policy_attestation"]["records"][0]
    card_path = Path(record["test_probability_card"]["path"])
    card = json.loads(card_path.read_text(encoding="utf-8"))
    artifact = card["probability_set"]["records"][0]
    probability_path = Path(artifact["path"])
    if mutation == "not_npy":
        probability_path.write_bytes(b"not a NumPy artifact")
    elif mutation == "float32":
        np.save(probability_path, np.full((2, 3), 0.5, dtype=np.float32), allow_pickle=False)
        artifact["dtype"] = "float32"
    elif mutation == "wrong_shape":
        np.save(probability_path, np.full((3, 3), 0.5, dtype=np.float16), allow_pickle=False)
        artifact["shape"] = [3, 3]
    elif mutation == "out_of_range":
        np.save(probability_path, np.full((2, 3), 1.5, dtype=np.float16), allow_pickle=False)
    else:
        artifact["source_sha256"] = "0" * 64
    artifact["sha256"] = MODULE.sha256_file(probability_path)
    artifact["bytes"] = probability_path.stat().st_size
    normalized = [
        {key: item[key] for key in ("image_id", "sha256", "bytes")}
        for item in card["probability_set"]["records"]
    ]
    card["probability_set"]["sha256"] = MODULE.canonical_sha256(normalized)
    with pytest.raises(MODULE.Stage4Error, match=message):
        MODULE._validate_probability_set(
            card,
            split="test",
            method="unet",
            seed=0,
            restricted_root=case.restricted,
            expected_universes=MODULE._manifest_split_probability_universes(
                case.manifest, case.splits
            ),
        )


def test_selector_requires_finite_frozen_argmax_and_tie_break() -> None:
    tied = {str(value): 0.5 for value in MODULE.REPLACEMENT_THRESHOLD_GRID}
    _, winner, score = MODULE._validated_threshold_scores(tied, "synthetic scores")
    assert winner == 0.5 and score == 0.5
    nonfinite = dict(tied)
    nonfinite["0.5"] = float("nan")
    with pytest.raises(MODULE.Stage4Error, match="non-finite"):
        MODULE._validated_threshold_scores(nonfinite, "synthetic scores")


def test_probability_universe_is_derived_from_manifest_and_split(tmp_path: Path) -> None:
    case = _case(tmp_path)
    universe = MODULE._manifest_split_probability_universes(case.manifest, case.splits)
    assert set(universe["val"]["records"]) == {"val_0"}
    assert set(universe["test"]["records"]) == {"test_0", "test_1"}
    card_path = Path(
        json.loads(case.stage2_reselection_delivery.read_text())["threshold_policy_attestation"][
            "records"
        ][0]["validation_probability_card"]["path"]
    )
    card = json.loads(card_path.read_text())
    card["probability_set"]["records"][0]["image_id"] = "test_0"
    with pytest.raises(MODULE.Stage4Error, match="unknown val probability id"):
        MODULE._validate_probability_set(
            card,
            split="val",
            method="unet",
            seed=0,
            restricted_root=case.restricted,
            expected_universes=universe,
        )


def test_amended_profile_rejects_missing_generic_utilisation(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    record = stage3["threshold_policy_attestation"]["records"][0]
    identity = record["per_epoch_training_profile_card"]
    profile_path = Path(identity["path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["generic_device_utilisation_recorded"] = False
    profile["generic_device_utilisation_unavailable_reason"] = (
        "no generic utilisation sample was available"
    )
    profile["epoch_profiles"][0]["gpu_utilization_sample_count"] = 0
    profile["epoch_profiles"][0]["gpu_utilization_percent_mean"] = None
    _json(profile_path, profile)
    identity["sha256"] = MODULE.sha256_file(profile_path)
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="per-epoch profiling"):
        _lock(case)


def test_amended_profile_epoch_count_is_bound_to_run_card(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    record = stage3["threshold_policy_attestation"]["records"][0]
    identity = record["per_epoch_training_profile_card"]
    profile_path = Path(identity["path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["epochs_recorded"] = 2
    profile["epoch_profiles"].append({**profile["epoch_profiles"][0], "epoch": 1})
    _json(profile_path, profile)
    identity["sha256"] = MODULE.sha256_file(profile_path)
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="per-epoch profiling"):
        _lock(case)


def test_amended_run_loader_and_cache_are_cryptographically_bound(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    record = stage3["threshold_policy_attestation"]["records"][0]
    run_path = Path(record["run_card"]["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["dataloader"]["num_workers"] = 7
    _json(run_path, run)
    record["run_card"]["sha256"] = MODULE.sha256_file(run_path)
    for frozen in stage3["newly_scored_seeds"]:
        if frozen["method"] == record["method"] and frozen["seed"] == record["seed"]:
            frozen["run_card_sha256"] = MODULE.sha256_file(run_path)
    profile_path = Path(record["per_epoch_training_profile_card"]["path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["run_card_sha256"] = MODULE.sha256_file(run_path)
    _json(profile_path, profile)
    record["per_epoch_training_profile_card"]["sha256"] = MODULE.sha256_file(
        profile_path
    )
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="loader/cache protocol"):
        _lock(case)


def test_amended_cache_artifact_bytes_are_reauthenticated(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    record = stage3["threshold_policy_attestation"]["records"][0]
    receipt_path = Path(record["preprocess_cache_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    image_id = receipt["records"][0]["image_id"]
    cached = receipt_path.parent / "images" / f"{image_id}.npy"
    np.save(cached, np.zeros((2, 3), dtype=np.uint8), allow_pickle=False)
    with pytest.raises(MODULE.Stage4Error, match="cached image bytes changed"):
        _lock(case)


def test_amended_run_cannot_enable_unverified_acceleration(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    record = stage3["threshold_policy_attestation"]["records"][0]
    run_path = Path(record["run_card"]["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["acceleration"] = {
        "precision": "bf16",
        "channels_last": True,
        "acceptance_receipt_sha256": None,
        "evaluation_only": False,
        "production_eligible": True,
    }
    _json(run_path, run)
    record["run_card"]["sha256"] = MODULE.sha256_file(run_path)
    for frozen in stage3["newly_scored_seeds"]:
        if frozen["method"] == record["method"] and frozen["seed"] == record["seed"]:
            frozen["run_card_sha256"] = MODULE.sha256_file(run_path)
    profile_path = Path(record["per_epoch_training_profile_card"]["path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["run_card_sha256"] = MODULE.sha256_file(run_path)
    _json(profile_path, profile)
    record["per_epoch_training_profile_card"]["sha256"] = MODULE.sha256_file(
        profile_path
    )
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="acceleration receipt"):
        _lock(case)


def test_amended_scientific_rows_require_evaluator_receipt(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    record = stage3["threshold_policy_attestation"]["records"][0]
    scoring_path = Path(record["scoring_receipt"]["path"])
    scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    scoring["invocations"][0]["exit_code"] = 1
    _json(scoring_path, scoring)
    record["scoring_receipt"]["sha256"] = MODULE.sha256_file(scoring_path)
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="evaluator exit binding"):
        _lock(case)


def test_scoring_cannot_self_declare_parameter_count(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage3, record = _first_stage3_record(case)
    scoring_path = Path(record["scoring_receipt"]["path"])
    scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    for invocation in scoring["invocations"]:
        argv = invocation["argv"]
        argv[argv.index("--params-m") + 1] = "99.0"
    _json(scoring_path, scoring)
    record["scoring_receipt"]["sha256"] = MODULE.sha256_file(scoring_path)
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="evaluator argv values changed"):
        _lock(case)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_argv", "evaluator argv schema changed"),
        ("wrong_evaluator", "committed wrapper/evaluator"),
        ("per_image_bytes", "per-image output bytes changed"),
    ],
)
def test_scoring_receipt_chain_rejects_forged_execution_evidence(
    tmp_path: Path, mutation: str, message: str
) -> None:
    case = _case(tmp_path)
    stage2 = json.loads(case.stage2_reselection_delivery.read_text(encoding="utf-8"))
    record = stage2["threshold_policy_attestation"]["records"][0]
    scoring_path = Path(record["scoring_receipt"]["path"])
    scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    if mutation == "extra_argv":
        scoring["invocations"][0]["argv"].extend(["--limit", "1"])
    elif mutation == "wrong_evaluator":
        scoring["producer"]["evaluator_sha256"] = "0" * 64
    else:
        Path(scoring["invocations"][0]["per_image"]["path"]).write_text(
            "forged\n", encoding="utf-8"
        )
    _json(scoring_path, scoring)
    record["scoring_receipt"]["sha256"] = MODULE.sha256_file(scoring_path)
    _json(case.stage2_reselection_delivery, stage2)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    stage3["stage2_unet_seed0"]["threshold_reselection_delivery_sha256"] = (
        MODULE.sha256_file(case.stage2_reselection_delivery)
    )
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match=message):
        _lock(case)


def test_finalizer_rederives_masks_despite_self_consistent_forged_receipt(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    stage2 = json.loads(case.stage2_reselection_delivery.read_text(encoding="utf-8"))
    record = stage2["threshold_policy_attestation"]["records"][0]
    card_path = Path(record["test_probability_card"]["path"])
    card = json.loads(card_path.read_text(encoding="utf-8"))
    prediction_dir = Path(record["scoring_receipt"]["path"]).parent / "predictions"
    changed_mask = prediction_dir / "test_0.png"
    assert cv2.imwrite(str(changed_mask), np.zeros((2, 3), dtype=np.uint8))
    for item in card["per_image"]:
        if item["image_id"] == "test_0":
            item["mask_sha256"] = MODULE.sha256_file(changed_mask)
    derivation_records = sorted(
        [
            {
                "image_id": item["image_id"],
                "mask_sha256": item["mask_sha256"],
                "probability_sha256": item["probability_sha256"],
            }
            for item in card["per_image"]
        ],
        key=lambda item: item["image_id"],
    )
    forged_derivation = MODULE.canonical_sha256(derivation_records)
    card["mask_derivation_verification"]["records_sha256"] = forged_derivation
    _json(card_path, card)
    record["test_probability_card"]["sha256"] = MODULE.sha256_file(card_path)

    inference_path = Path(record["inference_card"]["path"])
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    inference["identities"]["test_probability_card_sha256"] = MODULE.sha256_file(
        card_path
    )
    _json(inference_path, inference)
    record["inference_card"]["sha256"] = MODULE.sha256_file(inference_path)

    scoring_path = Path(record["scoring_receipt"]["path"])
    scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    scoring["identities"]["test_probability_card_sha256"] = MODULE.sha256_file(
        card_path
    )
    scoring["identities"]["inference_card_sha256"] = MODULE.sha256_file(
        inference_path
    )
    scoring["prediction_probability_derivation_records_sha256"] = forged_derivation
    _json(scoring_path, scoring)
    record["scoring_receipt"]["sha256"] = MODULE.sha256_file(scoring_path)
    _json(case.stage2_reselection_delivery, stage2)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    stage3["stage2_unet_seed0"]["threshold_reselection_delivery_sha256"] = (
        MODULE.sha256_file(case.stage2_reselection_delivery)
    )
    _json(case.stage3_delivery, stage3)
    with pytest.raises(
        MODULE.Stage4Error, match="not derived from probability|tp differs"
    ):
        _lock(case)


def test_supported_released_runner_cannot_claim_probability_unavailable(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    stage1 = json.loads(case.stage1_reselection_delivery.read_text(encoding="utf-8"))
    record = stage1["probability_retention_attestation"]["records"][0]
    record["availability"] = "unavailable"
    _json(case.stage1_reselection_delivery, stage1)
    with pytest.raises(MODULE.Stage4Error, match="raw probability support exists"):
        _lock(case)


def test_probability_card_cannot_substitute_another_checkpoint(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage2 = json.loads(case.stage2_reselection_delivery.read_text(encoding="utf-8"))
    card_record = stage2["threshold_policy_attestation"]["records"][0][
        "validation_probability_card"
    ]
    card_path = Path(card_record["path"])
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["checkpoint_sha256"] = "0" * 64
    _json(card_path, card)
    card_record["sha256"] = MODULE.sha256_file(card_path)
    _json(case.stage2_reselection_delivery, stage2)
    stage3 = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    stage3["stage2_unet_seed0"]["threshold_reselection_delivery_sha256"] = (
        MODULE.sha256_file(case.stage2_reselection_delivery)
    )
    _json(case.stage3_delivery, stage3)
    with pytest.raises(MODULE.Stage4Error, match="probability card checkpoint changed"):
        _lock(case)


def test_all_deliveries_must_bind_one_test_source_universe(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage1 = json.loads(case.stage1_reselection_delivery.read_text(encoding="utf-8"))
    stage1["ordered_source_images_sha256"] = "0" * 64
    _json(case.stage1_reselection_delivery, stage1)
    with pytest.raises(MODULE.Stage4Error, match="source-image universe differs"):
        _lock(case)


def test_released_binary_mask_cannot_be_reinterpreted_as_probability(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage1 = json.loads(case.stage1_reselection_delivery.read_text(encoding="utf-8"))
    sufs = next(
        record
        for record in stage1["probability_retention_attestation"]["records"]
        if record["method"] == "sufs"
    )
    card_path = Path(sufs["test_probability_card"]["path"])
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["binary_mask_reinterpreted"] = True
    _json(card_path, card)
    sufs["test_probability_card"]["sha256"] = MODULE.sha256_file(card_path)
    _json(case.stage1_reselection_delivery, stage1)
    with pytest.raises(MODULE.Stage4Error, match="binary output"):
        _lock(case)


def test_released_probability_derived_mask_bytes_are_recomputed(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage1 = json.loads(case.stage1_reselection_delivery.read_text(encoding="utf-8"))
    sufs = next(
        record
        for record in stage1["probability_retention_attestation"]["records"]
        if record["method"] == "sufs"
    )
    card = json.loads(
        Path(sufs["test_probability_card"]["path"]).read_text(encoding="utf-8")
    )
    # Resolve --output-root explicitly so this test remains insensitive to
    # option ordering.
    argv = card["provenance"]["command_argv"]
    output_root = Path(argv[argv.index("--output-root") + 1])
    mask = output_root / "test_0.png"
    assert cv2.imwrite(str(mask), np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(MODULE.Stage4Error, match="not derived from retained probability"):
        _lock(case)


def test_frozen_released_prediction_declaration_cannot_hide_byte_change(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    frozen_mask = (
        case.restricted
        / "released_probability_evidence"
        / "sufs"
        / "test_masks"
        / "test_0.png"
    )
    assert cv2.imwrite(str(frozen_mask), np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(
        MODULE.Stage4Error,
        match="frozen Stage 1 prediction-set declaration differs from bytes",
    ):
        _lock(case)


def test_metric_replay_accepts_only_the_evaluators_nan_semantics() -> None:
    assert MODULE._metric_equal("nan", float("nan"), "synthetic FDR")
    assert MODULE._metric_equal("", float("nan"), "synthetic FDR")
    assert not MODULE._metric_equal("0.0", float("nan"), "synthetic FDR")
    assert not MODULE._metric_equal("nan", 0.0, "synthetic Dice")
    with pytest.raises(MODULE.Stage4Error, match="infinite"):
        MODULE._metric_equal("inf", 0.0, "synthetic Dice")


def test_released_run_card_must_be_mentioned_by_frozen_stage1_provenance(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    stage1 = json.loads(case.stage1_reselection_delivery.read_text(encoding="utf-8"))
    sufs = next(
        record
        for record in stage1["probability_retention_attestation"]["records"]
        if record["method"] == "sufs"
    )
    sufs["frozen_stage1_reference_artifact"] = {
        "name": "journal",
        "sha256": MODULE.sha256_file(case.stage1_paths["journal"]),
    }
    _json(case.stage1_reselection_delivery, stage1)
    with pytest.raises(MODULE.Stage4Error, match="does not mention the run card"):
        _lock(case)


def test_missing_released_probability_attestation_is_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path)
    stage1 = json.loads(case.stage1_reselection_delivery.read_text(encoding="utf-8"))
    del stage1["probability_retention_attestation"]
    _json(case.stage1_reselection_delivery, stage1)
    with pytest.raises(MODULE.Stage4Error, match="lacks released probability"):
        _lock(case)


def test_registered_production_lock_rejects_synthetic_registration(tmp_path: Path) -> None:
    case = _case(tmp_path)
    MODULE._publish_json_once(case.lock_path, _lock(case))
    with pytest.raises(MODULE.Stage4Error, match="registered Stage 4 analysis plan"):
        MODULE._require_registered_production_lock(case.lock_path, case.restricted)


def test_preflight_accepts_exact_method_seed_table_group_universe(tmp_path: Path) -> None:
    case = _case(tmp_path)
    MODULE._publish_json_once(case.lock_path, _lock(case))
    receipt = MODULE.preflight_summaries(case.lock_path, case.restricted, case.code)
    assert receipt["status"] == "PASS"
    assert receipt["combined_rows"] == 252
    assert receipt["scientific_csv_scope"]["metric_values_parsed"] is True
    assert receipt["scientific_csv_scope"]["model_selection_performed"] is False
    assert {row["owner"] for row in receipt["summaries"]} == {
        "stage1",
        "stage1_reselection",
        "stage3",
    }


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("method", "unexpected", "unexpected method"),
        ("seed", "99", "unexpected seed"),
        ("group_value", "OTHER", "unexpected group"),
        ("n_missing", "1", "nonzero missing"),
    ],
)
def test_preflight_rejects_unexpected_summary_identity(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    case = _case(tmp_path)
    candidate = _lock(case)
    MODULE._publish_json_once(case.lock_path, candidate)

    rows = list(csv.DictReader(case.stage3_summary.open(encoding="utf-8")))
    rows[0][column] = value
    _csv(case.stage3_summary, rows)
    # Simulate a malicious lock edit that updates only the direct summary hash;
    # the universe validator must still reject the semantic mutation.
    altered = json.loads(case.lock_path.read_text(encoding="utf-8"))
    digest = MODULE.sha256_file(case.stage3_summary)
    altered["stage3"]["summary"]["sha256"] = digest
    altered["scientific_inputs"]["summaries"][2]["sha256"] = digest
    stage3_payload = json.loads(case.stage3_delivery.read_text(encoding="utf-8"))
    stage3_payload["outputs"]["results/summary.csv"] = digest
    _json(case.stage3_delivery, stage3_payload)
    delivery_digest = MODULE.sha256_file(case.stage3_delivery)
    altered["stage3"]["delivery"]["sha256"] = delivery_digest
    altered["stage3"]["delivery"]["bytes"] = case.stage3_delivery.stat().st_size
    altered["stage3"]["bundle_outputs_sha256"] = MODULE.canonical_sha256(stage3_payload["outputs"])
    case.lock_path.write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(MODULE.Stage4Error, match=message):
        MODULE.preflight_summaries(case.lock_path, case.restricted, case.code)


def test_bound_input_tampering_fails_before_preflight(tmp_path: Path) -> None:
    case = _case(tmp_path)
    MODULE._publish_json_once(case.lock_path, _lock(case))
    case.stage1_summary.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(MODULE.Stage4Error, match="differs from preregistration"):
        MODULE.preflight_summaries(case.lock_path, case.restricted, case.code)


def test_input_lock_semantics_are_deterministically_rederived(tmp_path: Path) -> None:
    case = _case(tmp_path)
    MODULE._publish_json_once(case.lock_path, _lock(case))
    altered = json.loads(case.lock_path.read_text(encoding="utf-8"))
    altered["expected_universe"]["combined_summary_rows"] -= 1
    _json(case.lock_path, altered)
    with pytest.raises(MODULE.Stage4Error, match="deterministic derivation"):
        MODULE.preflight_summaries(case.lock_path, case.restricted, case.code)


def test_publish_is_exact_atomic_write_once_and_verifiable(tmp_path: Path) -> None:
    case = _case(tmp_path)
    MODULE._publish_json_once(case.lock_path, _lock(case))
    receipt = MODULE.preflight_summaries(case.lock_path, case.restricted, case.code)
    receipt_path = case.restricted / "reports" / "stage4_preflight.json"
    MODULE._publish_json_once(receipt_path, receipt)

    staging = case.restricted / "stage4-staging"
    (staging / "tables").mkdir(parents=True)
    (staging / "tables" / "table_a.md").write_text("synthetic table\n", encoding="utf-8")
    outputs = MODULE._directory_inventory(staging)
    contract_path = case.restricted / "reports" / "stage4_output_contract.json"
    _json(
        contract_path,
        {
            "schema_version": 1,
            "kind": MODULE.CONTRACT_KIND,
            "input_lock_sha256": MODULE.sha256_file(case.lock_path),
            "preflight_receipt_sha256": MODULE.sha256_file(receipt_path),
            "files": outputs,
        },
    )
    bundle = case.restricted / "stage4-delivery"
    dry = MODULE.publish_outputs(
        input_lock_path=case.lock_path,
        receipt_path=receipt_path,
        contract_path=contract_path,
        staging_dir=staging,
        bundle_out=bundle,
        restricted_root=case.restricted,
        code_identity=case.code,
        dry_run=True,
    )
    assert dry["outputs"] == outputs
    assert not bundle.exists()

    delivery = MODULE.publish_outputs(
        input_lock_path=case.lock_path,
        receipt_path=receipt_path,
        contract_path=contract_path,
        staging_dir=staging,
        bundle_out=bundle,
        restricted_root=case.restricted,
        code_identity=case.code,
    )
    assert delivery["kind"] == MODULE.DELIVERY_KIND
    replay = MODULE.publish_outputs(
        input_lock_path=case.lock_path,
        receipt_path=receipt_path,
        contract_path=contract_path,
        staging_dir=staging,
        bundle_out=bundle,
        restricted_root=case.restricted,
        code_identity=case.code,
    )
    assert replay == delivery


def test_publish_does_not_replace_empty_directory_created_during_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    MODULE._publish_json_once(case.lock_path, _lock(case))
    receipt = MODULE.preflight_summaries(case.lock_path, case.restricted, case.code)
    receipt_path = case.restricted / "reports" / "stage4_preflight.json"
    MODULE._publish_json_once(receipt_path, receipt)
    staging = case.restricted / "stage4-staging"
    staging.mkdir()
    (staging / "expected.txt").write_text("expected", encoding="utf-8")
    contract_path = case.restricted / "reports" / "stage4_output_contract.json"
    _json(
        contract_path,
        {
            "schema_version": 1,
            "kind": MODULE.CONTRACT_KIND,
            "input_lock_sha256": MODULE.sha256_file(case.lock_path),
            "preflight_receipt_sha256": MODULE.sha256_file(receipt_path),
            "files": MODULE._directory_inventory(staging),
        },
    )
    bundle = case.restricted / "stage4-delivery-race"
    original_rename = MODULE._rename_directory_noreplace
    injected = False

    def racing_rename(source: Path, destination: Path) -> None:
        nonlocal injected
        assert destination == bundle
        destination.mkdir()
        injected = True
        original_rename(source, destination)

    monkeypatch.setattr(MODULE, "_rename_directory_noreplace", racing_rename)
    with pytest.raises(MODULE.Stage4Error):
        MODULE.publish_outputs(
            input_lock_path=case.lock_path,
            receipt_path=receipt_path,
            contract_path=contract_path,
            staging_dir=staging,
            bundle_out=bundle,
            restricted_root=case.restricted,
            code_identity=case.code,
        )
    assert injected
    assert bundle.is_dir() and not bundle.is_symlink()
    assert list(bundle.iterdir()) == []
    assert not (bundle.parent / f".{bundle.name}.stage4-publish.lock").exists()


def test_publish_rejects_uncontracted_output(tmp_path: Path) -> None:
    case = _case(tmp_path)
    MODULE._publish_json_once(case.lock_path, _lock(case))
    receipt = MODULE.preflight_summaries(case.lock_path, case.restricted, case.code)
    receipt_path = case.restricted / "reports" / "stage4_preflight.json"
    MODULE._publish_json_once(receipt_path, receipt)
    staging = case.restricted / "stage4-staging"
    staging.mkdir()
    (staging / "expected.txt").write_text("expected", encoding="utf-8")
    expected = MODULE._directory_inventory(staging)
    contract_path = case.restricted / "reports" / "stage4_output_contract.json"
    _json(
        contract_path,
        {
            "schema_version": 1,
            "kind": MODULE.CONTRACT_KIND,
            "input_lock_sha256": MODULE.sha256_file(case.lock_path),
            "preflight_receipt_sha256": MODULE.sha256_file(receipt_path),
            "files": expected,
        },
    )
    (staging / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(MODULE.Stage4Error, match="file universe"):
        MODULE.publish_outputs(
            input_lock_path=case.lock_path,
            receipt_path=receipt_path,
            contract_path=contract_path,
            staging_dir=staging,
            bundle_out=case.restricted / "delivery",
            restricted_root=case.restricted,
            code_identity=case.code,
            dry_run=True,
        )
