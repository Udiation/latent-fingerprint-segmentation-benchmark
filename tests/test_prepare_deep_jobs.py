"""CPU-only tests for the preregistered deep-run planner and selector."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "unet",
    "unetpp",
    "drunet",
    "ra_runet",
    "lenet",
    "alexnet",
    "cnn_ae",
    "segformer_sub",
)
RATES = ((0.0003, "0p0003"), (0.001, "0p001"))
TEST_COMMIT = "a" * 40


def _provenance() -> dict[str, object]:
    return {
        "capture_timing": "process_start",
        "git_commit": TEST_COMMIT,
        "git_dirty": False,
    }


@pytest.fixture(autouse=True)
def _expected_commit_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "expected_commits.json"
    path.write_text(
        json.dumps({method: TEST_COMMIT for method in METHODS}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEGBENCH_EXPECTED_COMMIT_MAP", str(path))
    snapshot = tmp_path / "segformer_snapshot"
    snapshot.mkdir()
    monkeypatch.setenv("SEGBENCH_SEGFORMER_B2_DIR", str(snapshot))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_production_inputs(tmp_path: Path, manifest: Path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "uint8_500ppi_preprocessing_cache",
                "manifest_sha256": _digest(manifest),
                "target_ppi": 500.0,
                "mask_columns": ["mask_path", "quality_mask_path"],
                "debug_limited": False,
            }
        ),
        encoding="utf-8",
    )
    amendment = tmp_path / "threshold_amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "kind": "stage4_threshold_policy_amendment",
                "replacement_policy": {
                    "allowed_trainable_grid": [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                    ],
                    "selection_data": "validation_only",
                },
            }
        ),
        encoding="utf-8",
    )
    return cache, amendment


def _write_native_run(
    *,
    generated: Path,
    runs: Path,
    manifest: Path,
    splits: Path,
    learning_rate: float,
    slug: str,
    score: float,
    method: str = "unet",
    epochs_requested: int = 30,
    epochs_completed: int | None = None,
) -> Path:
    """Write one private-run-shaped synthetic card without any image data."""
    run_dir = runs / "sweep" / "unet" / f"lr_{slug}" / "seed0"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best.pt").write_bytes(f"unet-{slug}-best".encode())
    (run_dir / "last.pt").write_bytes(f"unet-{slug}-last".encode())
    run_path = run_dir / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "method": method,
                "seed": 0,
                "epochs_requested": epochs_requested,
                "epochs_completed": (
                    epochs_requested if epochs_completed is None else epochs_completed
                ),
                "manifest_sha256": _digest(manifest),
                "split_sha256": _digest(splits),
                "source_config_sha256": _digest(
                    generated / "sweep" / f"unet_lr{slug}.yaml"
                ),
                "learning_rate": learning_rate,
                "debug_limited": False,
                "best_validation_dice": score,
                # Deliberately inverted relative to validation performance.
                # Selection must neither read nor copy this synthetic field.
                "test_dice": 1.0 - score,
                "best_epoch": 12,
                "selected_validation_threshold": 0.4,
                "config_sha256": hashlib.sha256(
                    f"resolved-unet-{slug}".encode()
                ).hexdigest(),
                "checkpoint_sha256": {
                    "best.pt": _digest(run_dir / "best.pt"),
                    "last.pt": _digest(run_dir / "last.pt"),
                },
                "provenance": _provenance(),
            }
        ),
        encoding="utf-8",
    )
    return run_path


def _load_planner_module():
    """Load the CLI module so exact-card-count validation is unit-testable."""
    path = ROOT / "scripts" / "prepare_deep_jobs.py"
    spec = importlib.util.spec_from_file_location("prepare_deep_jobs_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserved_unetpp_requires_expanded_grid_selector_receipts(
    tmp_path: Path,
) -> None:
    planner = _load_planner_module()
    source = tmp_path / "unetpp.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "threshold_candidates": list(planner.LEGACY_THRESHOLDS)
                }
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}\n", encoding="utf-8")
    manifest_sha = "a" * 64
    splits_sha = "b" * 64
    matching = tmp_path / "matching.json"
    matching.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "expanded_validation_threshold_selector_receipt",
                "method": "unetpp",
                "seed": 0,
                "selection_split": "val",
                "selection_source": "validation_only",
                "validation_only": True,
                "test_or_heldout_opened": False,
                "source_config_sha256": _digest(source),
                "checkpoint_sha256": _digest(checkpoint),
                "manifest_sha256": manifest_sha,
                "split_sha256": splits_sha,
                "amendment_sha256": _digest(amendment),
                "threshold_amendment_sha256": _digest(amendment),
                "probabilities_are_pre_threshold": True,
                "derived_from_binary_mask": False,
                "debug_limited": False,
                "allowed_grid": list(planner.THRESHOLDS),
                "raw_validation_probability_card_sha256": "c" * 64,
                "validation_probability_card_sha256": "d" * 64,
                "selected_threshold": 0.2,
                "selected_validation_dice": 0.81,
            }
        ),
        encoding="utf-8",
    )
    other = tmp_path / "other.json"
    other_payload = json.loads(matching.read_text(encoding="utf-8"))
    other_payload["checkpoint_sha256"] = "e" * 64
    other.write_text(json.dumps(other_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="require two expanded-grid"):
        planner._legacy_unetpp_selector(
            None,
            source_config=source,
            checkpoint=checkpoint,
            manifest_sha256=manifest_sha,
            splits_sha256=splits_sha,
            amendment=amendment,
        )
    selected = planner._legacy_unetpp_selector(
        [matching, other],
        source_config=source,
        checkpoint=checkpoint,
        manifest_sha256=manifest_sha,
        splits_sha256=splits_sha,
        amendment=amendment,
    )
    assert selected["selected_threshold"] == 0.2
    assert selected["selected_validation_dice"] == 0.81
    assert selected["raw_validation_probability_card_sha256"] == "c" * 64


def _write_completed_full_run(
    *,
    runs: Path,
    method: str,
    seed: int,
    manifest: Path,
    splits: Path,
    source_config_sha256: str,
    learning_rate: float,
    selection_sha256: str | None = None,
    debug_limited: bool = False,
) -> Path:
    """Write a synthetic final card with hashed checkpoints, but no data."""
    run_dir = runs / "full" / method / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best.pt").write_bytes(f"{method}-{seed}-best".encode())
    (run_dir / "last.pt").write_bytes(f"{method}-{seed}-last".encode())
    payload = {
        "method": method,
        "seed": seed,
        "epochs_requested": 100,
        # A final card after valid early stopping need not reach epoch 100.
        "epochs_completed": 19,
        "manifest_sha256": _digest(manifest),
        "split_sha256": _digest(splits),
        "source_config_sha256": source_config_sha256,
        "learning_rate": learning_rate,
        "debug_limited": debug_limited,
        "checkpoint_sha256": {
            "best.pt": _digest(run_dir / "best.pt"),
            "last.pt": _digest(run_dir / "last.pt"),
        },
        "provenance": _provenance(),
    }
    if method == "sufs_ft":
        payload["mode"] = "full"
        payload["selection_artifact_sha256"] = selection_sha256
    run_path = run_dir / "run.json"
    run_path.write_text(json.dumps(payload), encoding="utf-8")
    return run_path


def test_select_pilot_requires_exact_unet_evidence_and_emits_seed0(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic manifest\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text("{}\n", encoding="utf-8")
    generated = tmp_path / "generated"
    runs = tmp_path / "runs"
    pilot_sweep = tmp_path / "pilot_sweep.yaml"
    remaining_sweep = tmp_path / "remaining_sweep.yaml"
    selection = tmp_path / "unet_selection.json"
    seed0_jobs = tmp_path / "unet_seed0.yaml"
    cache_dir, threshold_amendment = _write_production_inputs(tmp_path, manifest)
    common = [
        "--code-root",
        str(ROOT),
        "--base-config-dir",
        str(ROOT / "configs"),
        "--generated-config-dir",
        str(generated),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--runs-root",
        str(runs),
        "--python-executable",
        "python-for-test",
        "--cache-dir",
        str(cache_dir),
        "--threshold-amendment",
        str(threshold_amendment),
        "--expected-threshold-amendment-sha256",
        _digest(threshold_amendment),
    ]
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_deep_jobs.py"),
            "prepare-sweep",
            *common,
            "--pilot-jobs-out",
            str(pilot_sweep),
            "--remaining-jobs-out",
            str(remaining_sweep),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_deep_jobs.py"),
        "select-pilot",
        *common,
        "--selection-out",
        str(selection),
        "--seed0-jobs-out",
        str(seed0_jobs),
    ]

    missing = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    assert "incomplete sweep: missing" in missing.stderr
    assert not selection.exists()
    assert not seed0_jobs.exists()

    low_path = _write_native_run(
        generated=generated,
        runs=runs,
        manifest=manifest,
        splits=splits,
        learning_rate=0.0003,
        slug="0p0003",
        score=0.81,
    )
    high_path = _write_native_run(
        generated=generated,
        runs=runs,
        manifest=manifest,
        splits=splits,
        learning_rate=0.001,
        slug="0p001",
        score=0.84,
        method="unetpp",
    )
    wrong_method = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert wrong_method.returncode != 0
    assert "method='unetpp', expected 'unet'" in wrong_method.stderr
    assert not selection.exists()

    high_path = _write_native_run(
        generated=generated,
        runs=runs,
        manifest=manifest,
        splits=splits,
        learning_rate=0.001,
        slug="0p001",
        score=0.84,
        epochs_requested=29,
    )
    unequal_budget = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert unequal_budget.returncode != 0
    assert "epochs_requested=29, expected 30" in unequal_budget.stderr
    assert not selection.exists()

    high_path = _write_native_run(
        generated=generated,
        runs=runs,
        manifest=manifest,
        splits=splits,
        learning_rate=0.001,
        slug="0p001",
        score=0.84,
        epochs_requested=30,
        epochs_completed=29,
    )
    incomplete_budget = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert incomplete_budget.returncode != 0
    assert "epochs_completed=29, expected 30" in incomplete_budget.stderr
    assert not selection.exists()

    high_path = _write_native_run(
        generated=generated,
        runs=runs,
        manifest=manifest,
        splits=splits,
        learning_rate=0.001,
        slug="0p001",
        score=0.84,
    )
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    selected = json.loads(selection.read_text(encoding="utf-8"))
    assert selected["method"] == "unet"
    assert selected["selection_split"] == "val"
    assert selected["selection_metric"] == "best_validation_dice"
    assert selected["learning_rate_candidates"] == [0.0003, 0.001]
    assert selected["selected_learning_rate"] == 0.001
    assert selected["selected_candidate_index"] == 1
    assert len(selected["candidates"]) == 2
    assert [candidate["run_card"] for candidate in selected["candidates"]] == [
        str(low_path),
        str(high_path),
    ]
    assert [
        candidate["run_card_sha256"] for candidate in selected["candidates"]
    ] == [_digest(low_path), _digest(high_path)]
    assert all(
        candidate["source_config_sha256"]
        == _digest(Path(candidate["source_config"]))
        for candidate in selected["candidates"]
    )
    assert all(
        set(candidate["checkpoint_sha256"]) == {"best.pt", "last.pt"}
        for candidate in selected["candidates"]
    )
    assert "test_dice" not in json.dumps(selected)

    full_config_path = generated / "full" / "unet.yaml"
    full_config = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))
    assert full_config["seed"] == 0
    assert full_config["training"]["epochs"] == 100
    assert full_config["training"]["patience"] == 15
    assert full_config["training"]["learning_rate"] == 0.001
    assert full_config["experiment"]["learning_rate_selected_on"] == "val_best_dice"
    assert selected["full_config_sha256"] == _digest(full_config_path)

    queue = yaml.safe_load(seed0_jobs.read_text(encoding="utf-8"))
    assert queue["selection"] == str(selection)
    assert queue["selection_sha256"] == _digest(selection)
    assert queue["config_sha256"] == _digest(full_config_path)
    assert [job["name"] for job in queue["jobs"]] == ["full.unet.seed0"]
    argv = queue["jobs"][0]["argv"]
    assert argv[0] == "python-for-test"
    assert argv[argv.index("--seed") + 1] == "0"
    assert argv[argv.index("--config") + 1] == str(full_config_path)
    assert "CUDA_VISIBLE_DEVICES" not in json.dumps(queue)
    assert not (runs / "sweep" / "unetpp").exists()

    planner = _load_planner_module()
    cells = [
        (
            rate,
            runs / "sweep" / "unet" / f"lr_{slug}" / "seed0",
            generated / "sweep" / f"unet_lr{slug}.yaml",
        )
        for rate, slug in RATES
    ]
    with pytest.raises(ValueError, match="exactly two preregistered sweep cells"):
        planner.select_native_sweep_evidence(
            [*cells, cells[0]],
            method="unet",
            manifest_sha256=_digest(manifest),
            splits_sha256=_digest(splits),
        )


def test_prepare_and_validation_only_select_deep_jobs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic manifest\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text("{}\n", encoding="utf-8")
    generated = tmp_path / "generated"
    runs = tmp_path / "runs"
    pilot_sweep = tmp_path / "pilot_sweep.yaml"
    remaining_sweep = tmp_path / "remaining_sweep.yaml"
    cache_dir, threshold_amendment = _write_production_inputs(tmp_path, manifest)
    common = [
        "--code-root",
        str(ROOT),
        "--base-config-dir",
        str(ROOT / "configs"),
        "--generated-config-dir",
        str(generated),
        "--manifest",
        str(manifest),
        "--splits",
        str(splits),
        "--runs-root",
        str(runs),
        "--python-executable",
        "python-for-test",
        "--cache-dir",
        str(cache_dir),
        "--threshold-amendment",
        str(threshold_amendment),
        "--expected-threshold-amendment-sha256",
        _digest(threshold_amendment),
    ]
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_deep_jobs.py"),
            "prepare-sweep",
            *common,
            "--pilot-jobs-out",
            str(pilot_sweep),
            "--remaining-jobs-out",
            str(remaining_sweep),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    pilot = yaml.safe_load(pilot_sweep.read_text(encoding="utf-8"))
    remaining = yaml.safe_load(remaining_sweep.read_text(encoding="utf-8"))
    assert pilot["native_sweep_patience"] == 30
    assert remaining["native_sweep_patience"] == 30
    assert [job["name"] for job in pilot["jobs"]] == [
        "sweep.unet.lr0p0003.seed0",
        "sweep.unet.lr0p001.seed0",
    ]
    assert len(remaining["jobs"]) == 16
    assert remaining["jobs"][0]["name"] == "sweep.unetpp.lr0p0003.seed0"
    assert remaining["jobs"][13]["name"] == "sweep.segformer_sub.lr0p001.seed0"
    assert remaining["jobs"][-1]["name"] == "sweep.sufs_ft.lr0p001.seed0"
    segformer_jobs = [
        job for job in remaining["jobs"] if ".segformer_sub." in job["name"]
    ]
    assert len(segformer_jobs) == 2
    assert all(
        job["env"]
        == {"SEGBENCH_SEGFORMER_B2_DIR": str(tmp_path / "segformer_snapshot")}
        for job in segformer_jobs
    )
    for job in pilot["jobs"] + remaining["jobs"]:
        assert job["argv"][0] == "python-for-test"
        assert "--resume" in job["argv"]
        assert job["argv"][job["argv"].index("--resume") + 1] == "auto"
        assert "CUDA_VISIBLE_DEVICES" not in json.dumps(job)
    for method in METHODS:
        for learning_rate, slug in RATES:
            config = yaml.safe_load(
                (generated / "sweep" / f"{method}_lr{slug}.yaml").read_text(
                    encoding="utf-8"
                )
            )
            assert config["seed"] == 0
            assert config["training"]["epochs"] == 30
            assert config["training"]["patience"] == 30
            assert config["training"]["learning_rate"] == learning_rate
            assert config["training"]["threshold_candidates"] == [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ]

            run_dir = runs / "sweep" / method / f"lr_{slug}" / "seed0"
            run_dir.mkdir(parents=True)
            (run_dir / "best.pt").write_bytes(f"{method}-{slug}-best".encode())
            (run_dir / "last.pt").write_bytes(f"{method}-{slug}-last".encode())
            # Make the lower LR win, proving selection reads validation Dice and
            # not ordering, test metrics, or a hand-edited selection list.
            score = 0.8 if learning_rate == 0.0003 else 0.7
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "method": method,
                        "seed": 0,
                        "epochs_requested": 30,
                        "epochs_completed": 30,
                        "manifest_sha256": _digest(manifest),
                        "split_sha256": _digest(splits),
                        "source_config_sha256": _digest(
                            generated / "sweep" / f"{method}_lr{slug}.yaml"
                        ),
                        "learning_rate": learning_rate,
                        "debug_limited": False,
                        "best_validation_dice": score,
                        "best_epoch": 12,
                        "selected_validation_threshold": 0.4,
                        "config_sha256": "config-proof",
                        "checkpoint_sha256": {
                            "best.pt": _digest(run_dir / "best.pt"),
                            "last.pt": _digest(run_dir / "last.pt"),
                        },
                        "provenance": _provenance(),
                    }
                ),
                encoding="utf-8",
            )

    sufs_config = ROOT / "configs" / "sufs_ft.yaml"
    for learning_rate, slug in RATES:
        run_dir = runs / "sweep" / "sufs_ft" / f"lr_{slug}" / "seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(f"sufs-{slug}-best".encode())
        (run_dir / "last.pt").write_bytes(f"sufs-{slug}-last".encode())
        score = 0.82 if learning_rate == 0.0003 else 0.72
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "method": "sufs_ft",
                    "mode": "sweep",
                    "seed": 0,
                    "epochs_requested": 30,
                    "epochs_completed": 30,
                    "manifest_sha256": _digest(manifest),
                    "split_sha256": _digest(splits),
                    "source_config_sha256": _digest(sufs_config),
                    "learning_rate": learning_rate,
                    "debug_limited": False,
                    "best_validation_dice": score,
                    "best_epoch": 14,
                    "protocol_config_sha256": "shared-protocol-proof",
                    "released": {"weights_sha256": "a" * 64},
                    "checkpoint_sha256": {
                        "best.pt": _digest(run_dir / "best.pt"),
                        "last.pt": _digest(run_dir / "last.pt"),
                    },
                    "provenance": _provenance(),
                }
            ),
            encoding="utf-8",
        )

    selection = tmp_path / "selection.json"
    full_pilot = tmp_path / "full_pilot.yaml"
    full_remaining = tmp_path / "full_remaining.yaml"
    full_seed12 = tmp_path / "full_seed12.yaml"
    pilot_selection_artifact = tmp_path / "pilot_selection_generated.json"
    pilot_job_artifact = tmp_path / "pilot_job_generated.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_deep_jobs.py"),
            "select-pilot",
            *common,
            "--selection-out",
            str(pilot_selection_artifact),
            "--seed0-jobs-out",
            str(pilot_job_artifact),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    frozen_unet_config = generated / "full" / "unet.yaml"
    frozen_unet_config_bytes = frozen_unet_config.read_bytes()
    native_sweep_queue = tmp_path / "native_sweep_queue.yaml"
    native_jobs = []
    for job in remaining["jobs"][:14]:
        copied = dict(job)
        out = Path(copied["argv"][copied["argv"].index("--out") + 1])
        copied["completion_artifact"] = str(out / "run.json")
        native_jobs.append(copied)
    native_sweep_queue.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "stage3_concurrency_supplemental_queue",
                "phase": "learning_rate_sweep_remaining_native",
                "jobs": native_jobs,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    select_command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_deep_jobs.py"),
        "select-full",
        *common,
        "--selection-out",
        str(selection),
        "--pilot-seed0-jobs-out",
        str(full_pilot),
        "--remaining-seed0-jobs-out",
        str(full_remaining),
        "--seed12-jobs-out",
        str(full_seed12),
        "--native-sweep-queue",
        str(native_sweep_queue),
    ]
    # Any post-run change to a source YAML invalidates selection, even if the
    # semantic YAML would otherwise parse identically.
    tampered_config = generated / "sweep" / "unet_lr0p0003.yaml"
    original_config = tampered_config.read_text(encoding="utf-8")
    tampered_config.write_text(original_config + "# tampered\n", encoding="utf-8")
    rejected = subprocess.run(
        select_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "source_config_sha256" in rejected.stderr
    tampered_config.write_text(original_config, encoding="utf-8")

    incomplete_run_path = (
        runs / "sweep" / "unet" / "lr_0p0003" / "seed0" / "run.json"
    )
    incomplete_run = json.loads(incomplete_run_path.read_text(encoding="utf-8"))
    incomplete_run["epochs_completed"] = 29
    incomplete_run_path.write_text(json.dumps(incomplete_run), encoding="utf-8")
    rejected = subprocess.run(
        select_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "epochs_completed=29, expected 30" in rejected.stderr
    incomplete_run["epochs_completed"] = 30
    incomplete_run_path.write_text(json.dumps(incomplete_run), encoding="utf-8")

    subprocess.run(
        select_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    selected = json.loads(selection.read_text(encoding="utf-8"))
    assert frozen_unet_config.read_bytes() == frozen_unet_config_bytes
    assert selected["methods"]["unet"]["reused_seed0_config"] == str(
        frozen_unet_config
    )
    assert selected["methods"]["unet"]["reused_seed0_config_sha256"] == _digest(
        frozen_unet_config
    )
    assert all(
        record["selected_learning_rate"] == 0.0003
        for record in selected["methods"].values()
    )
    assert selected["methods"]["sufs_ft"]["selected_learning_rate"] == 0.0003
    assert "separate audited" in selected["workflow_notes"]["sufs_ft"]
    assert "fingernet" in selected["workflow_notes"]
    assert selected["production_inputs"]["cache_dir"] == str(cache_dir.resolve())
    assert selected["production_inputs"]["threshold_amendment"] == str(
        threshold_amendment.resolve()
    )
    assert len(yaml.safe_load(full_pilot.read_text())["jobs"]) == 1
    assert len(yaml.safe_load(full_remaining.read_text())["jobs"]) == 8
    seed12 = yaml.safe_load(full_seed12.read_text())["jobs"]
    assert len(seed12) == 18
    assert [job["name"] for job in seed12[:4]] == [
        "full.unet.seed1",
        "full.unet.seed2",
        "full.unetpp.seed1",
        "full.unetpp.seed2",
    ]
    for method in METHODS:
        config_path = (
            generated / "full_amended" / "unet.yaml"
            if method == "unet"
            else generated / "full" / f"{method}.yaml"
        )
        config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        assert config["training"]["epochs"] == 100
        assert config["training"]["patience"] == 15
        assert config["training"]["learning_rate"] == 0.0003
        assert config["runtime"]["queue_batch_contract_sha256"]
    pilot_job = yaml.safe_load(full_pilot.read_text())["jobs"][0]
    assert pilot_job["argv"][pilot_job["argv"].index("--config") + 1] == str(
        frozen_unet_config
    )
    amended_seed_jobs = [
        job for job in seed12 if job["name"] in {"full.unet.seed1", "full.unet.seed2"}
    ]
    assert all(
        job["argv"][job["argv"].index("--config") + 1]
        == str(generated / "full_amended" / "unet.yaml")
        for job in amended_seed_jobs
    )
    sufs_full_jobs = [
        job
        for job in yaml.safe_load(full_remaining.read_text())["jobs"] + seed12
        if "sufs_ft" in job["name"]
    ]
    assert [job["name"] for job in sufs_full_jobs] == [
        "full.sufs_ft.seed0",
        "full.sufs_ft.seed1",
        "full.sufs_ft.seed2",
    ]
    assert all(job["env"] == {"KERAS_BACKEND": "torch"} for job in sufs_full_jobs)
    for queue_path in (full_pilot, full_remaining, full_seed12):
        for job in yaml.safe_load(queue_path.read_text())["jobs"]:
            assert job["argv"][job["argv"].index("--cache-dir") + 1] == str(
                cache_dir.resolve()
            )
            assert job["argv"][job["argv"].index("--threshold-amendment") + 1] == str(
                threshold_amendment.resolve()
            )

    # Freeze independent pre-global artifacts. The global selection must live
    # in a separate directory in production so it cannot overwrite the prior
    # SUFS evidence before equality is checked.
    global_sufs_path = tmp_path / "sufs_ft_selection.json"
    prior_sufs = tmp_path / "prior_sufs_selection.json"
    prior_sufs.write_bytes(global_sufs_path.read_bytes())
    unet_global = selected["methods"]["unet"]
    pilot_candidates = []
    for candidate in unet_global["candidates"]:
        pilot_candidate = dict(candidate)
        pilot_candidate["checkpoint_sha256"] = {
            "best.pt": candidate["best_checkpoint_sha256"]
        }
        run_card = Path(candidate["run_card"])
        pilot_candidate["run_card_sha256"] = _digest(run_card)
        pilot_candidates.append(pilot_candidate)
    pilot_selection = pilot_selection_artifact
    pilot_payload = json.loads(pilot_selection.read_text(encoding="utf-8"))

    _write_completed_full_run(
        runs=runs,
        method="unet",
        seed=0,
        manifest=manifest,
        splits=splits,
        source_config_sha256=unet_global["reused_seed0_config_sha256"],
        learning_rate=unet_global["selected_learning_rate"],
    )
    prior_sufs_digest = _digest(prior_sufs)
    for seed in (0, 1, 2):
        _write_completed_full_run(
            runs=runs,
            method="sufs_ft",
            seed=seed,
            manifest=manifest,
            splits=splits,
            source_config_sha256=selected["methods"]["sufs_ft"][
                "source_config_sha256"
            ],
            learning_rate=selected["methods"]["sufs_ft"][
                "selected_learning_rate"
            ],
            selection_sha256=prior_sufs_digest,
            debug_limited=(seed == 2),
        )

    filtered = tmp_path / "deduplicated.yaml"
    provenance = tmp_path / "deduplicated_provenance.json"
    dedupe_command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_deep_jobs.py"),
        "dedupe-full",
        "--global-selection",
        str(selection),
        "--pilot-selection",
        str(pilot_selection),
        "--sufs-selection",
        str(prior_sufs),
        "--pilot-seed0-jobs",
        str(full_pilot),
        "--remaining-seed0-jobs",
        str(full_remaining),
        "--seed12-jobs",
        str(full_seed12),
        "--runs-root",
        str(runs),
        "--out",
        str(filtered),
        "--provenance-out",
        str(provenance),
    ]

    rejected = subprocess.run(
        dedupe_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "dedupe-full was retired" in rejected.stderr
    assert not filtered.exists()
    return  # replacement barrier workflow is tested in test_stage3_authorization.py

    # Restore the valid final card, then prove checkpoint tampering also fails.
    _write_completed_full_run(
        runs=runs,
        method="sufs_ft",
        seed=2,
        manifest=manifest,
        splits=splits,
        source_config_sha256=selected["methods"]["sufs_ft"][
            "source_config_sha256"
        ],
        learning_rate=selected["methods"]["sufs_ft"]["selected_learning_rate"],
        selection_sha256=prior_sufs_digest,
    )
    seed2_best = runs / "full" / "sufs_ft" / "seed2" / "best.pt"
    valid_seed2_best = seed2_best.read_bytes()
    seed2_best.write_bytes(b"tampered")
    rejected = subprocess.run(
        dedupe_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "checkpoint hash failure for best.pt" in rejected.stderr
    seed2_best.write_bytes(valid_seed2_best)

    # A different pilot choice is never silently accepted.
    pilot_payload["selected_learning_rate"] = 0.001
    pilot_selection.write_text(json.dumps(pilot_payload), encoding="utf-8")
    rejected = subprocess.run(
        dedupe_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "disagree on selected_learning_rate" in rejected.stderr
    pilot_payload["selected_learning_rate"] = unet_global["selected_learning_rate"]
    pilot_selection.write_text(json.dumps(pilot_payload), encoding="utf-8")

    # A manual raw-queue deletion is rejected before filtering.
    valid_remaining_text = full_remaining.read_text(encoding="utf-8")
    altered_remaining = yaml.safe_load(valid_remaining_text)
    altered_remaining["jobs"] = altered_remaining["jobs"][:-1]
    full_remaining.write_text(yaml.safe_dump(altered_remaining), encoding="utf-8")
    rejected = subprocess.run(
        dedupe_command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "job universe/order changed" in rejected.stderr
    full_remaining.write_text(valid_remaining_text, encoding="utf-8")

    subprocess.run(
        dedupe_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    filtered_payload = yaml.safe_load(filtered.read_text(encoding="utf-8"))
    retained_names = [job["name"] for job in filtered_payload["jobs"]]
    expected_retained = [
        *(f"full.{method}.seed0" for method in METHODS if method != "unet"),
        *(f"full.{method}.seed{seed}" for method in METHODS for seed in (1, 2)),
    ]
    assert retained_names == expected_retained
    assert len(retained_names) == len(set(retained_names)) == 23
    assert filtered_payload["excluded_completed_jobs"] == [
        "full.unet.seed0",
        "full.sufs_ft.seed0",
        "full.sufs_ft.seed1",
        "full.sufs_ft.seed2",
    ]

    # Retained argv/env mappings are copied unchanged from the raw queues.
    raw_jobs = {
        job["name"]: job
        for queue_path in (full_remaining, full_seed12)
        for job in yaml.safe_load(queue_path.read_text())["jobs"]
    }
    assert all(job == raw_jobs[job["name"]] for job in filtered_payload["jobs"])
    proof = json.loads(provenance.read_text(encoding="utf-8"))
    assert proof["retained_jobs"] == retained_names
    assert proof["excluded_jobs"] == filtered_payload["excluded_completed_jobs"]
    assert proof["output"]["sha256"] == _digest(filtered)
    assert set(proof["completed_jobs"]) == set(proof["excluded_jobs"])
    assert proof["inputs"]["global_selection"]["sha256"] == _digest(selection)
    assert proof["inputs"]["pilot_selection"]["sha256"] == _digest(
        pilot_selection
    )
    assert proof["inputs"]["sufs_selection"]["sha256"] == _digest(prior_sufs)
