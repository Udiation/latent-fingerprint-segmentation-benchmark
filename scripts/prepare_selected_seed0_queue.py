#!/usr/bin/env python3
"""Prepare fresh native full seed-0 jobs from authenticated replay selections.

This is a metadata-only planner.  It reads validation selector/finalizer
receipts, never probabilities or test data, and emits an immutable guarded
training queue. Existing eligible U-Net/SUFS full runs are recorded as reuse.
If an old full run is ineligible, a fresh versioned seed-0 job is emitted;
SUFS gets a dedicated trainer-compatible selection receipt derived from the
authenticated expanded-grid validation evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json_once, canonical_sha256, sha256_file
from segbench.deep_utils import PROTOCOL_THRESHOLDS, build_queue_batch_contract, config_digest
from segbench.sufs_finetune import protocol_config_digest


NATIVE = {
    "unet", "unetpp", "drunet", "ra_runet", "lenet", "alexnet",
    "cnn_ae", "segformer_sub",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-summary", action="append", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--threshold-amendment", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--config-out", required=True, type=Path)
    parser.add_argument("--queue-out", required=True, type=Path)
    parser.add_argument("--proof-out", required=True, type=Path)
    parser.add_argument("--segformer-dir", type=Path)
    parser.add_argument("--job-host-ram-gib", required=True, type=float)
    parser.add_argument("--job-cpu-workers", required=True, type=int)
    return parser.parse_args(argv)


def _read_json(path: Path, description: str) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _write_yaml_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        observed = yaml.safe_load(path.read_text(encoding="utf-8"))
        if canonical_sha256(observed) != canonical_sha256(value):
            raise FileExistsError(f"refusing to overwrite changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())


def _selection_files(summaries: Sequence[Path]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    selections: Dict[str, Dict[str, Any]] = {}
    deferred: Dict[str, Any] = {}
    for summary_path in summaries:
        summary = _read_json(summary_path, "selection summary")
        if (
            summary.get("kind") != "validation_replay_selection_summary"
            or summary.get("selection_source") != "validation_only"
            or summary.get("test_or_heldout_opened") is not False
        ):
            raise ValueError("selection summary is not validation-only")
        for record in summary.get("groups", []):
            path = Path(str(record.get("path", "")))
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError("selection artifact identity changed")
            selection = _read_json(path, "learning-rate selection")
            if (
                selection.get("kind") != "expanded_grid_learning_rate_selection"
                or selection.get("selection_source") != "validation_only"
                or selection.get("test_or_heldout_opened") is not False
            ):
                raise ValueError("learning-rate selection is not validation-only")
            method = str(selection.get("method", ""))
            if method in selections:
                raise ValueError(f"duplicate method selection: {method}")
            selection["_path"] = str(path)
            selection["_sha256"] = record["sha256"]
            selections[method] = selection
        for record in summary.get("deferred_candidates", []):
            name = str(record.get("candidate", ""))
            if name in deferred:
                raise ValueError(f"duplicate deferred decision: {name}")
            deferred[name] = record
    return selections, deferred


def _completed_deferred_replay(
    selection: Mapping[str, Any], record: Mapping[str, Any]
) -> Dict[str, Any]:
    """Authenticate one eligible existing-full validation replay receipt.

    Eligibility alone says only that the historical full run used the newly
    selected learning rate.  Reuse also requires that this exact checkpoint
    has completed its expanded-grid validation probability replay.
    """
    plan_path = Path(str(selection.get("plan", "")))
    plan_sha = str(selection.get("plan_sha256", ""))
    if not plan_path.is_file() or sha256_file(plan_path) != plan_sha:
        raise ValueError("deferred replay plan identity changed")
    plan = _read_json(plan_path, "validation replay plan")
    if (
        plan.get("kind") != "dataset1_validation_probability_replay_plan"
        or plan.get("selection_split") != "val"
        or plan.get("test_or_heldout_opened") is not False
    ):
        raise ValueError("deferred replay plan is not validation-only")
    matches = [
        item for item in plan.get("candidates", [])
        if isinstance(item, Mapping) and item.get("name") == record.get("candidate")
    ]
    if len(matches) != 1:
        raise ValueError("deferred candidate is absent or duplicated in its replay plan")
    candidate = matches[0]
    if (
        candidate.get("candidate_identity_sha256")
        != record.get("candidate_identity_sha256")
        or candidate.get("method") != record.get("method")
        or candidate.get("role") != "final"
    ):
        raise ValueError("deferred candidate identity/method/role changed")
    completion_path = Path(str(candidate.get("outputs", {}).get("completion", "")))
    completion = _read_json(completion_path, "deferred replay completion")
    expected = {
        "kind": "validation_probability_replay_completion",
        "status": "complete",
        "complete": True,
        "plan_sha256": plan_sha,
        "candidate": candidate["name"],
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "method": candidate["method"],
        "role": "final",
        "selection_split": "val",
        "test_or_heldout_opened": False,
    }
    disagreements = {
        key: (completion.get(key), value)
        for key, value in expected.items()
        if completion.get(key) != value
    }
    if disagreements:
        raise ValueError(f"deferred replay completion changed: {disagreements}")
    selector = Path(str(completion.get("selector", "")))
    if not selector.is_file() or sha256_file(selector) != completion.get("selector_sha256"):
        raise ValueError("deferred replay selector identity changed")
    threshold = float(completion.get("selected_threshold", math.nan))
    dice = float(completion.get("selected_validation_dice", math.nan))
    if threshold not in set(PROTOCOL_THRESHOLDS) or not math.isfinite(dice) or not 0 <= dice <= 1:
        raise ValueError("deferred replay selected validation metrics are invalid")
    return {
        "candidate": candidate["name"],
        "completion": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "selector": str(selector),
        "selector_sha256": completion["selector_sha256"],
        "selected_threshold": threshold,
        "selected_validation_dice": dice,
    }


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    if args.job_host_ram_gib <= 0 or args.job_cpu_workers <= 0:
        raise ValueError("job resource reservations must be positive")
    for path, label in (
        (args.python, "Python"), (args.manifest, "manifest"),
        (args.splits, "split"), (args.threshold_amendment, "amendment"),
    ):
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"{label} must be an absolute file")
    if not args.cache_dir.is_absolute() or not (args.cache_dir / "receipt.json").is_file():
        raise ValueError("cache directory/receipt is absent")
    receipt = _read_json(args.cache_dir / "receipt.json", "cache receipt")
    if (
        receipt.get("manifest_sha256") != sha256_file(args.manifest)
        or receipt.get("target_ppi") != 500.0
        or receipt.get("debug_limited") is not False
    ):
        raise ValueError("cache receipt does not bind the full manifest at 500 ppi")
    amendment = _read_json(args.threshold_amendment, "threshold amendment")
    replacement = amendment.get("replacement_policy", {})
    if (
        amendment.get("kind") != "stage4_threshold_policy_amendment"
        or replacement.get("allowed_trainable_grid") != list(PROTOCOL_THRESHOLDS)
        or replacement.get("selection_data") != "validation_only"
    ):
        raise ValueError("training threshold amendment changed")
    selections, deferred = _selection_files(args.selection_summary)
    jobs = []
    decisions = []
    for method, selection in selections.items():
        selected_lr = float(selection.get("selected_learning_rate", math.nan))
        if selected_lr not in {0.0003, 0.001}:
            raise ValueError(f"{method}: selected LR is outside the frozen grid")
        method_deferred = [
            item for item in deferred.values() if item.get("method") == method
        ]
        if method_deferred and all(bool(item.get("eligible")) for item in method_deferred):
            completions = [
                _completed_deferred_replay(selection, item)
                for item in method_deferred
            ]
            decisions.append({
                "method": method,
                "action": "reuse_existing_full",
                "selected_learning_rate": selected_lr,
                "validation_replays": completions,
            })
            continue
        if method == "sufs_ft":
            base_path = args.code_root / "configs/sufs_ft.yaml"
            config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or config.get("method") != "sufs_ft":
                raise ValueError("sufs_ft: base configuration changed")
            evidence = selection.get("candidates")
            if not isinstance(evidence, list) or len(evidence) != 2:
                raise ValueError("sufs_ft: expanded-grid selection lacks both candidates")
            candidate_receipts = []
            for item in evidence:
                rate = float(item.get("learning_rate", math.nan))
                dice = float(item.get("selected_validation_dice", math.nan))
                threshold = float(item.get("selected_threshold", math.nan))
                if (
                    rate not in {0.0003, 0.001}
                    or not math.isfinite(dice)
                    or not 0 <= dice <= 1
                    or threshold not in set(PROTOCOL_THRESHOLDS)
                ):
                    raise ValueError("sufs_ft: expanded-grid candidate evidence changed")
                candidate_receipts.append({
                    "learning_rate": rate,
                    "best_validation_dice": dice,
                    "expanded_grid_threshold": threshold,
                    "checkpoint_sha256": item.get("checkpoint_sha256"),
                    "validation_selector_sha256": item.get("selector_sha256"),
                    "validation_completion_sha256": item.get("completion_sha256"),
                })
            adapter = {
                "schema_version": 1,
                "method": "sufs_ft",
                "selection_split": "val",
                # The dedicated trainer's compatibility contract uses this
                # name; the v3 fields below state the expanded-grid semantics.
                "selection_metric": "best_macro_dice",
                "protocol_config_sha256": protocol_config_digest(config),
                "source_config_sha256": sha256_file(base_path),
                "manifest_sha256": sha256_file(args.manifest),
                "split_sha256": sha256_file(args.splits),
                "released_weights_sha256": config["pretrained"]["weights_sha256"],
                "tie_rule": selection.get("tie_break"),
                "candidates": candidate_receipts,
                "selected_learning_rate": selected_lr,
                "selected_validation_dice": selection.get("selected_validation_dice"),
                "selected_threshold": selection.get("selected_threshold"),
                "v3_selection_artifact": selection["_path"],
                "v3_selection_artifact_sha256": selection["_sha256"],
                "v3_selection_metric": selection.get("selection_metric"),
                "test_or_heldout_opened": False,
            }
            adapter_path = args.config_out / "sufs_ft_selection.json"
            atomic_json_once(adapter_path, adapter)
            out = args.runs_root / "full_v3_expanded_grid" / method / "seed0"
            if (out / "run.json").exists():
                raise FileExistsError("sufs_ft: an existing v3 full seed-0 run needs audit")
            argv = [
                str(args.python), str(args.code_root / "scripts/train_sufs.py"),
                "--config", str(base_path), "--manifest", str(args.manifest),
                "--splits", str(args.splits), "--out", str(out),
                "--mode", "full", "--selection", str(adapter_path),
                "--seed", "0", "--resume", "auto", "--cache-dir",
                str(args.cache_dir), "--threshold-amendment",
                str(args.threshold_amendment),
            ]
            job = {
                "name": "full.sufs_ft.seed0",
                "argv": argv,
                "env": {"KERAS_BACKEND": "torch"},
                "completion_artifact": str(out / "run.json"),
                "resources": {
                    "host_ram_gib": float(args.job_host_ram_gib),
                    "cpu_workers": int(args.job_cpu_workers),
                },
            }
            job["job_identity_sha256"] = canonical_sha256(job)
            jobs.append(job)
            decisions.append({
                "method": method,
                "action": "fresh_full_seed0",
                "selected_learning_rate": selected_lr,
                "selected_threshold": selection.get("selected_threshold"),
                "selection_sha256": selection["_sha256"],
                "dedicated_selection": str(adapter_path),
                "dedicated_selection_sha256": sha256_file(adapter_path),
            })
            continue
        if method not in NATIVE:
            raise ValueError(f"unsupported full-training method: {method}")
        base_path = args.code_root / "configs" / f"{method}.yaml"
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict) or base.get("method") != method:
            raise ValueError(f"{method}: base configuration changed")
        config = copy.deepcopy(base)
        training = config.get("training")
        runtime = config.get("runtime")
        if not isinstance(training, dict) or not isinstance(runtime, dict):
            raise ValueError(f"{method}: base configuration lacks training/runtime")
        training.update(
            learning_rate=selected_lr,
            epochs=100,
            patience=15,
            threshold_candidates=list(PROTOCOL_THRESHOLDS),
        )
        contract = build_queue_batch_contract(training)
        runtime.update(
            queue_batch_contract=contract,
            queue_batch_contract_sha256=config_digest(contract),
            preprocess_cache_receipt_sha256=sha256_file(args.cache_dir / "receipt.json"),
            threshold_amendment_sha256=sha256_file(args.threshold_amendment),
        )
        config["experiment"] = {
            "phase": "full_training",
            "learning_rate_selected_on": "expanded_grid_validation_dice",
            "selected_learning_rate": selected_lr,
            "selection_artifact": selection["_path"],
            "selection_artifact_sha256": selection["_sha256"],
            "full_seeds": [0, 1, 2],
        }
        config_path = args.config_out / f"{method}.yaml"
        _write_yaml_once(config_path, config)
        task = str(config.get("task"))
        entrypoint = args.code_root / "scripts" / (
            "train_seg.py" if task == "pixel" else "train_patch.py"
        )
        # Never overwrite or resume a historical full run that the new LR
        # selection may have made ineligible.
        out = args.runs_root / "full_v3_expanded_grid" / method / "seed0"
        if (out / "run.json").exists():
            raise FileExistsError(f"{method}: an existing full seed-0 run needs audit")
        argv = [
            str(args.python), str(entrypoint), "--config", str(config_path),
            "--manifest", str(args.manifest), "--splits", str(args.splits),
            "--out", str(out), "--seed", "0", "--resume", "auto",
            "--cache-dir", str(args.cache_dir),
            "--threshold-amendment", str(args.threshold_amendment),
        ]
        env: Dict[str, str] = {}
        if method == "segformer_sub":
            if args.segformer_dir is None or not args.segformer_dir.is_absolute() or not args.segformer_dir.is_dir():
                raise ValueError("SegFormer full training requires a pinned local snapshot")
            env["SEGBENCH_SEGFORMER_B2_DIR"] = str(args.segformer_dir)
        resources = {
            # Preserve the operator-declared conservative full-run reservation.
            # The launcher may schedule fewer jobs when its aggregate guard
            # requires it.
            "host_ram_gib": float(args.job_host_ram_gib),
            "cpu_workers": int(training.get("num_workers", 8)),
        }
        job = {
            "name": f"full.{method}.seed0",
            "argv": argv,
            "env": env,
            "completion_artifact": str(out / "run.json"),
            "resources": resources,
        }
        job["job_identity_sha256"] = canonical_sha256(job)
        jobs.append(job)
        decisions.append({
            "method": method,
            "action": "fresh_full_seed0",
            "selected_learning_rate": selected_lr,
            "selected_threshold": selection.get("selected_threshold"),
            "selection_sha256": selection["_sha256"],
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
        })
    queue = {
        "schema_version": 1,
        "phase": "selected_full_seed0_training",
        "launch_authorized": bool(jobs),
        "selection_split": "val",
        "test_inference_authorized": False,
        "jobs": jobs,
    }
    _write_yaml_once(args.queue_out, queue)
    proof = {
        "schema_version": 1,
        "kind": "selected_full_seed0_queue_preparation",
        "status": "READY_NOT_LAUNCHED" if jobs else "NO_NEW_JOBS",
        "test_or_heldout_opened": False,
        "selection_summaries": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.selection_summary
        ],
        "queue": str(args.queue_out),
        "queue_sha256": sha256_file(args.queue_out),
        "jobs": [job["name"] for job in jobs],
        "decisions": decisions,
        "gpu_launched": False,
    }
    atomic_json_once(args.proof_out, proof)
    return proof


def main(argv: Optional[Sequence[str]] = None) -> None:
    print(json.dumps(prepare(parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
