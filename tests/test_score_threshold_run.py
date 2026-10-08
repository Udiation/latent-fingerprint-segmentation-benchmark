"""Synthetic-only tests for the committed threshold-scoring wrapper."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_threshold_run.py"
EVALUATOR = ROOT / "scripts" / "evaluate_masks.py"
sys.path.insert(0, str(ROOT))

from segbench import features as block_features  # noqa: E402
from segbench.postprocess import postprocess_blocks  # noqa: E402


def _load_wrapper_module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("threshold_scorer_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_finalizer_module(path: Path):
    scripts = path.parent
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "stage4_finalizer_scoring_test", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _png(path: Path, value: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value)
    return path


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _committed_code_copy(tmp_path: Path) -> tuple[Path, str, Path, Path, Path]:
    """Commit only copied test code so the real dirty worktree is irrelevant."""
    repo = tmp_path / "committed-code"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    copied_wrapper = scripts / SCRIPT.name
    copied_evaluator = scripts / EVALUATOR.name
    shutil.copyfile(SCRIPT, copied_wrapper)
    shutil.copyfile(EVALUATOR, copied_evaluator)
    copied_finalizer = scripts / "finalize_stage4.py"
    shutil.copyfile(ROOT / "scripts" / "finalize_stage4.py", copied_finalizer)
    shutil.copyfile(ROOT / "scripts" / "_path.py", scripts / "_path.py")
    shutil.copytree(
        ROOT / "segbench",
        repo / "segbench",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _git(repo, "init", "-q")
    _git(repo, "add", "scripts", "segbench")
    _git(
        repo,
        "-c",
        "user.name=Synthetic Test",
        "-c",
        "user.email=synthetic@example.invalid",
        "commit",
        "-q",
        "-m",
        "synthetic committed scorer",
    )
    return (
        repo,
        _git(repo, "rev-parse", "HEAD"),
        copied_wrapper,
        copied_evaluator,
        copied_finalizer,
    )


def _synthetic_case(
    tmp_path: Path,
    *,
    task: str = "pixel",
    patch_config_minimum: int = 4,
    patch_mask_minimum: int | None = 4,
) -> SimpleNamespace:
    assert task in {"pixel", "patch"}
    method = "unet" if task == "pixel" else "lenet"
    side = 12 if task == "pixel" else 96
    repo, commit, wrapper, evaluator, finalizer = _committed_code_copy(tmp_path)
    restricted = tmp_path / "restricted"
    data = restricted / "data"
    predictions = restricted / "predictions"
    probabilities = restricted / "probabilities"
    results = restricted / "results"
    for directory in (data, predictions, probabilities, results):
        directory.mkdir(parents=True)

    rows = []
    probability_records = []
    per_image = []
    derivation_records = []
    split_map = {}
    for index, group in enumerate(("VALUE", "NOVALUE")):
        image_id = f"synthetic{index}"
        subject_id = f"subject{index}"
        split_map[subject_id] = "test"
        image = np.full((side, side), 90 + index * 30, dtype=np.uint8)
        if task == "pixel":
            probability = np.full((side, side), np.float16(0.1), dtype=np.float16)
            probability[2 + index : 9, 3:10] = np.float16(0.9)
            predicted = probability.astype(np.float32) >= 0.5
        else:
            probability = np.full((6, 6), np.float16(0.1), dtype=np.float16)
            probability[1:5, 1:5] = np.float16(0.9)
            probability[2, 2] = np.float16(0.1)
            raw_blocks = probability.astype(np.float32) >= 0.5
            processed_blocks = (
                raw_blocks
                if patch_mask_minimum is None
                else postprocess_blocks(
                    raw_blocks, min_component_blocks=patch_mask_minimum
                )
            )
            predicted = block_features.blocks_to_mask(
                processed_blocks, (side, side)
            )
        prediction = predicted.astype(np.uint8) * 255
        quality = prediction.copy()
        quality[2 + index, 3] = 0

        image_path = _png(data / f"{image_id}.png", image)
        roi_path = _png(data / f"{image_id}_roi.png", prediction)
        quality_path = _png(data / f"{image_id}_quality.png", quality)
        mask_path = _png(predictions / f"{image_id}.png", prediction)
        probability_path = probabilities / f"{image_id}.npy"
        with probability_path.open("wb") as handle:
            np.save(handle, probability, allow_pickle=False)
        probability_sha = _sha(probability_path)
        mask_sha = _sha(mask_path)
        probability_records.append(
            {
                "image_id": image_id,
                "filename": probability_path.name,
                "path": str(probability_path.resolve()),
                "sha256": probability_sha,
                "bytes": probability_path.stat().st_size,
                "shape": list(probability.shape),
                "dtype": "float16",
            }
        )
        per_image.append(
            {
                "image_id": image_id,
                "ms_per_img": 1.25 + index,
                "mask_sha256": mask_sha,
                "probability_sha256": probability_sha,
            }
        )
        derivation_records.append(
            {
                "image_id": image_id,
                "mask_sha256": mask_sha,
                "probability_sha256": probability_sha,
            }
        )
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "mask_path": str(roi_path),
                "quality_mask_path": str(quality_path),
                "subject_id": subject_id,
                "subset": "finger_latent",
                "ppi": 500,
                "mask_ppi": 500,
                "quality_mask_ppi": 500,
                "target_ppi": 500,
                "target_height": side,
                "target_width": side,
                "examiner_value": group,
            }
        )

    manifest = restricted / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    splits = _json(restricted / "splits.json", {"seed": 7, "splits": split_map})
    amendment = _json(
        restricted / "amendment.json",
        {
            "schema_version": 1,
            "kind": "stage4_threshold_policy_amendment",
            "replacement_policy": {
                "allowed_trainable_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            },
        },
    )
    config = _json(
        restricted / "config.json",
        {
            "schema_version": 1,
            "method": method,
            "task": task,
            "seed": 0,
            "data": {
                "target_ppi": 500,
                "subset": "finger_latent",
                **({"block_size": 16} if task == "patch" else {}),
            },
            "inference": (
                {"block_size": 16, "min_component_blocks": patch_config_minimum}
                if task == "patch"
                else {}
            ),
        },
    )
    selector = _json(
        restricted / "selector.json",
        {
            "schema_version": 1,
            "kind": "expanded_validation_threshold_selector_receipt",
            "method": method,
            "task": task,
            "seed": 0,
            "selection_split": "val",
            "selection_source": "validation_only",
            "validation_only": True,
            "test_or_heldout_opened": False,
            "selected_threshold": 0.5,
            "source_config_sha256": _sha(config),
            "manifest_sha256": _sha(manifest),
            "split_sha256": _sha(splits),
            "amendment_sha256": _sha(amendment),
        },
    )
    probability_set_records = [
        {
            "image_id": record["image_id"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        for record in probability_records
    ]
    probability_card = _json(
        restricted / "test_probability_card.json",
        {
            "schema_version": 1,
            "kind": "complete_probability_map_card",
            "status": "complete",
            "complete": True,
            "method": method,
            "task": task,
            "seed": 0,
            "subset": "finger_latent",
            "split": "test",
            "selected_validation_threshold": 0.5,
            "threshold_receipt_sha256": _sha(selector),
            "threshold_amendment_sha256": _sha(amendment),
            "threshold_selected_before_test": True,
            "labels_opened": False,
            "selection_source": "unselected_raw",
            "probabilities_are_pre_threshold": True,
            "storage_format": "float16_npy",
            "storage_dtype": "float16",
            "derived_from_binary_mask": False,
            "binary_mask_reinterpreted": False,
            "source_config_sha256": _sha(config),
            "manifest_sha256": _sha(manifest),
            "split_sha256": _sha(splits),
            "checkpoint_sha256": "c" * 64,
            "device": "CPU",
            "mean_ms_per_img": 1.75,
            "probability_set": {
                "format": "float16_npy",
                "storage_dtype": "float16",
                "complete": True,
                "count": len(probability_records),
                "sha256": _canonical_sha(probability_set_records),
                "records": probability_records,
            },
            "mask_derivation_verification": {
                "complete": True,
                "count": len(derivation_records),
                "records_sha256": _canonical_sha(derivation_records),
                "all_masks_rederived_from_retained_probabilities": True,
            },
            "per_image": per_image,
            "identities": {
                "source_config_sha256": _sha(config),
                "manifest_sha256": _sha(manifest),
                "split_sha256": _sha(splits),
                "selector_receipt_sha256": _sha(selector),
                "amendment_sha256": _sha(amendment),
            },
        },
    )
    inference = _json(
        restricted / "inference.json",
        {
            "schema_version": 1,
            "kind": "threshold_bound_inference_card",
            "method": method,
            "seed": 0,
            "selected_validation_threshold": 0.5,
            "selection_frozen_before_test_inference": True,
            "test_labels_opened": False,
            "identities": {
                "source_config_sha256": _sha(config),
                "manifest_sha256": _sha(manifest),
                "split_sha256": _sha(splits),
                "selector_receipt_sha256": _sha(selector),
                "test_probability_card_sha256": _sha(probability_card),
                "amendment_sha256": _sha(amendment),
                "checkpoint_sha256": "c" * 64,
            },
        },
    )
    receipt = restricted / "scoring_receipt.json"
    command = [
        sys.executable,
        str(wrapper),
        "--repo-root",
        str(repo),
        "--expected-git-commit",
        commit,
        "--expected-wrapper-sha256",
        _sha(wrapper),
        "--expected-evaluator-sha256",
        _sha(evaluator),
        "--restricted-root",
        str(restricted),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--config",
        str(config),
        "--test-probability-card",
        str(probability_card),
        "--selector-receipt",
        str(selector),
        "--threshold-amendment",
        str(amendment),
        "--inference-card",
        str(inference),
        "--prediction-dir",
        str(predictions),
        "--probability-dir",
        str(probabilities),
        "--results-dir",
        str(results),
        "--receipt-out",
        str(receipt),
        "--method",
        method,
        "--seed",
        "0",
        "--display-name",
        f"Synthetic {method}",
        "--year",
        "2022",
        "--family",
        "Synthetic encoder-decoder",
        "--training-data",
        "Synthetic train split",
        "--comment",
        "Synthetic validation-only threshold",
        "--params-m",
        "1.0",
    ]
    return SimpleNamespace(
        repo=repo,
        command=command,
        restricted=restricted,
        manifest=manifest,
        splits=splits,
        config=config,
        probability_card=probability_card,
        predictions=predictions,
        probabilities=probabilities,
        results=results,
        receipt=receipt,
        wrapper=wrapper,
        evaluator=evaluator,
        finalizer=finalizer,
        commit=commit,
        derivation_sha256=_canonical_sha(derivation_records),
        method=method,
        task=task,
    )


def _run(case: SimpleNamespace) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        case.command,
        cwd=case.repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _option(argv: list[str], name: str) -> str:
    assert argv.count(name) == 1
    return argv[argv.index(name) + 1]


def _rows_digest(summary: Path, table: str) -> tuple[int, str]:
    with summary.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row["method"] == "unet"
            and row["seed"] == "0"
            and row["table"] == table
        ]
    rows.sort(key=lambda row: tuple(str(row.get(key, "")) for key in sorted(row)))
    return len(rows), _canonical_sha(rows)


def test_wrapper_runs_exact_common_evaluator_and_writes_bound_receipt(
    tmp_path: Path,
) -> None:
    case = _synthetic_case(tmp_path)
    completed = _run(case)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(case.receipt.read_text(encoding="utf-8"))

    assert receipt["schema_version"] == 1
    assert receipt["kind"] == "superseding_threshold_scoring_receipt"
    assert receipt["method"] == "unet"
    assert receipt["seed"] == 0
    assert receipt["selected_threshold"] == 0.5
    assert receipt["selection_frozen_before_test_scoring"] is True
    assert receipt["test_labels_opened_only_by_evaluator"] is True
    assert receipt["prediction_count"] == 2
    assert (
        receipt["prediction_probability_derivation_records_sha256"]
        == case.derivation_sha256
    )
    assert receipt["producer"] == {
        "git_commit": case.commit,
        "git_dirty": False,
        "script_sha256": _sha(case.wrapper),
        "evaluator_sha256": _sha(case.evaluator),
    }
    assert receipt["identities"] == {
        "source_config_sha256": _sha(case.config),
        "manifest_sha256": _sha(case.manifest),
        "splits_sha256": _sha(case.splits),
        "test_probability_card_sha256": _sha(case.probability_card),
        "selector_receipt_sha256": _sha(case.restricted / "selector.json"),
        "amendment_sha256": _sha(case.restricted / "amendment.json"),
        "inference_card_sha256": _sha(case.restricted / "inference.json"),
    }

    summary = case.results / "summary.csv"
    assert receipt["scientific_summary"] == {
        "path": str(summary.resolve()),
        "sha256": _sha(summary),
    }
    assert [item["table"] for item in receipt["invocations"]] == ["A", "A_prime"]
    expected_gt = {"A": "mask_path", "A_prime": "quality_mask_path"}
    for invocation in receipt["invocations"]:
        table = invocation["table"]
        argv = invocation["argv"]
        assert argv[0] == sys.executable
        assert argv[1:5] == [
            "-I",
            "-B",
            "-c",
            _load_wrapper_module().EVALUATOR_BOOTSTRAP,
        ]
        assert Path(argv[5]).resolve() == case.evaluator.resolve()
        assert "--limit" not in argv
        assert _option(argv, "--manifest") == str(case.manifest.resolve())
        assert _option(argv, "--splits") == str(case.splits.resolve())
        assert _option(argv, "--subsets") == "finger_latent"
        assert _option(argv, "--split") == "test"
        assert _option(argv, "--target-ppi") == "500"
        assert _option(argv, "--gt-column") == expected_gt[table]
        assert _option(argv, "--exclude") == "examiner_value=NONPRINT"
        assert _option(argv, "--group-by") == "examiner_value"
        assert _option(argv, "--pred-template") == str(
            case.predictions.resolve() / "{image_id}.png"
        )
        assert _option(argv, "--out") == str(case.results.resolve())
        assert invocation["exit_code"] == 0
        assert invocation["stdout_sha256"] != ""
        assert invocation["stderr_sha256"] != ""
        per_image = Path(invocation["per_image"]["path"])
        assert invocation["per_image"]["sha256"] == _sha(per_image)
        count, digest = _rows_digest(summary, table)
        assert invocation["scientific_row_count"] == count == 3
        assert invocation["scientific_rows_sha256"] == digest

    scored = pd.read_csv(summary)
    assert set(scored.table) == {"A", "A_prime"}
    assert set(scored.group_value) == {"overall", "VALUE", "NOVALUE"}
    assert scored[scored.table == "A"].dice_mean.eq(1.0).all()

    # Exercise the consumer-side validator against the real generated evidence.
    finalizer = _load_finalizer_module(case.finalizer)
    manifest_rows = pd.read_csv(case.manifest).to_dict("records")
    test_records = {
        row["image_id"]: {
            **row,
            "target_shape": [int(row["target_height"]), int(row["target_width"])],
        }
        for row in manifest_rows
    }
    receipt_sha, checked = finalizer._validate_scoring_receipt(
        {"path": str(case.receipt), "sha256": _sha(case.receipt)},
        method="unet",
        seed=0,
        selected_threshold=0.5,
        amendment_sha256=_sha(case.restricted / "amendment.json"),
        selector_sha256=_sha(case.restricted / "selector.json"),
        inference_sha256=_sha(case.restricted / "inference.json"),
        test_probability_card_sha256=_sha(case.probability_card),
        test_probability_payload=json.loads(case.probability_card.read_text()),
        source_config_sha256=_sha(case.config),
        source_config=yaml.safe_load(case.config.read_text(encoding="utf-8")),
        expected_params_m=1.0,
        expected_ms_per_img=1.75,
        manifest_sha256=_sha(case.manifest),
        splits_sha256=_sha(case.splits),
        scientific_summary_path=summary,
        expected_universes={"test": {"records": test_records}},
        restricted_root=case.restricted,
        expected_commit=case.commit,
    )
    assert receipt_sha == _sha(case.receipt)
    assert checked["scientific_summary_sha256"] == _sha(summary)


def test_wrapper_rejects_mask_not_derived_from_retained_probability(
    tmp_path: Path,
) -> None:
    case = _synthetic_case(tmp_path)
    changed = np.zeros((12, 12), dtype=np.uint8)
    _png(case.predictions / "synthetic0.png", changed)

    completed = _run(case)
    assert completed.returncode != 0
    assert "retained mask/probability identity changed" in completed.stderr
    assert not case.receipt.exists()
    assert not (case.results / "summary.csv").exists()


def test_wrapper_rejects_prior_evaluator_output_without_receipt(
    tmp_path: Path,
) -> None:
    case = _synthetic_case(tmp_path)
    (case.results / "summary.csv").write_text(
        "method,table,dice_mean\nunet,A,1.0\n", encoding="utf-8"
    )
    completed = _run(case)
    assert completed.returncode != 0
    assert "results directory is not empty" in completed.stderr
    assert not case.receipt.exists()


def test_patch_probabilities_use_frozen_block_postprocessing(tmp_path: Path) -> None:
    case = _synthetic_case(tmp_path, task="patch")
    completed = _run(case)
    assert completed.returncode == 0, completed.stderr

    receipt = json.loads(case.receipt.read_text(encoding="utf-8"))
    assert receipt["method"] == "lenet"
    assert receipt["identities"]["source_config_sha256"] == _sha(case.config)
    assert receipt["prediction_count"] == 2
    probability = np.load(case.probabilities / "synthetic0.npy", allow_pickle=False)
    prediction = cv2.imread(
        str(case.predictions / "synthetic0.png"), cv2.IMREAD_GRAYSCALE
    )
    assert probability.shape == (6, 6)
    assert prediction.shape == (96, 96)
    assert set(pd.read_csv(case.results / "summary.csv").method) == {"lenet"}


@pytest.mark.parametrize(
    ("config_minimum", "mask_minimum"),
    [
        (4, None),  # raw threshold expansion incorrectly skips morphology
        (20, 4),  # mask used a postprocess policy other than the bound config
    ],
)
def test_patch_scoring_rejects_wrong_postprocess_or_config(
    tmp_path: Path,
    config_minimum: int,
    mask_minimum: int | None,
) -> None:
    case = _synthetic_case(
        tmp_path,
        task="patch",
        patch_config_minimum=config_minimum,
        patch_mask_minimum=mask_minimum,
    )
    completed = _run(case)
    assert completed.returncode != 0
    assert "retained probability does not reproduce mask" in completed.stderr
    assert not case.receipt.exists()
    assert not (case.results / "summary.csv").exists()


def test_scoring_receipt_is_write_once_even_after_success(tmp_path: Path) -> None:
    case = _synthetic_case(tmp_path)
    first = _run(case)
    assert first.returncode == 0, first.stderr
    altered = json.loads(case.receipt.read_text(encoding="utf-8"))
    altered["selected_threshold"] = 0.7
    _json(case.receipt, altered)

    replay = _run(case)
    assert replay.returncode != 0
    assert "refusing to execute the test evaluator twice" in replay.stderr
    assert json.loads(case.receipt.read_text(encoding="utf-8"))["selected_threshold"] == 0.7


def test_nonzero_second_evaluator_exit_never_publishes_receipt(tmp_path: Path) -> None:
    case = _synthetic_case(tmp_path)
    (case.restricted / "data" / "synthetic0_quality.png").unlink()

    completed = _run(case)
    assert completed.returncode != 0
    assert "A_prime evaluator exited" in completed.stderr
    assert not case.receipt.exists()
    partial = pd.read_csv(case.results / "summary.csv")
    assert set(partial.table) == {"A"}


def test_wrapper_rejects_uncommitted_or_wrongly_pinned_code(tmp_path: Path) -> None:
    case = _synthetic_case(tmp_path)
    index = case.command.index("--expected-wrapper-sha256") + 1
    case.command[index] = "0" * 64

    completed = _run(case)
    assert completed.returncode != 0
    assert "caller-pinned SHA-256" in completed.stderr
    assert not case.receipt.exists()
    assert not (case.results / "summary.csv").exists()


def test_receipt_publication_does_not_unlink_a_swapped_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_wrapper_module()
    receipt = tmp_path / "receipt.json"
    attacker_bytes = b"attacker-owned replacement\n"
    real_fsync = module.os.fsync
    swapped = False

    def swap_then_fsync(descriptor: int) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            receipt.unlink()
            receipt.write_bytes(attacker_bytes)
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", swap_then_fsync)
    with pytest.raises(module.ScoringError, match="replaced scoring receipt"):
        module._publish_json_once(receipt, {"schema_version": 1})
    assert receipt.read_bytes() == attacker_bytes


def test_receipt_exact_replay_never_follows_a_symlink(tmp_path: Path) -> None:
    module = _load_wrapper_module()
    target = tmp_path / "target.json"
    target.write_text("do not read through me\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.symlink_to(target)

    with pytest.raises(module.ScoringError, match="cannot open existing scoring receipt"):
        module._publish_json_once(receipt, {"schema_version": 1})
    assert target.read_text(encoding="utf-8") == "do not read through me\n"
