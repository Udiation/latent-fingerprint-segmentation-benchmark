#!/usr/bin/env python3
"""Filter the registered sweep reproducibly, without rerunning completed SUFS."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

import _path  # noqa: F401
from segbench.artifact_identity import atomic_json, canonical_sha256, sha256_file
import prepare_deep_jobs as planner

METHODS = tuple(method for method in planner.METHOD_PRIORITY if method != "unet")
EXPECTED_NATIVE = tuple(
    f"sweep.{method}.lr{planner.lr_slug(rate)}.seed0"
    for method in METHODS
    for rate in planner.LEARNING_RATES
)
EXPECTED_RAW = (*EXPECTED_NATIVE, "sweep.sufs_ft.lr0p0003.seed0", "sweep.sufs_ft.lr0p001.seed0")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_stage2_delivery(path: Path) -> Dict[str, Any]:
    """Bind Stage 3 authorization to the completed one-time pilot delivery."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Stage 2 delivery schema changed")
    if payload.get("kind") != "stage2_unet_heldout_delivery":
        raise ValueError("Stage 2 held-out delivery is absent or has wrong identity")
    for key in ("gate_sha256", "inference_card_sha256", "checkpoint_sha256"):
        if not _is_sha256(payload.get(key)):
            raise ValueError(f"Stage 2 delivery lacks a valid {key}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("capture_timing") != "process_start":
        raise ValueError("Stage 2 delivery lacks process-start provenance")
    if provenance.get("git_dirty") is not False:
        raise ValueError("Stage 2 delivery was not produced from a clean repository")
    commit = provenance.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("Stage 2 delivery lacks an immutable inference commit")
    outputs = payload.get("results")
    if not isinstance(outputs, dict):
        raise ValueError("Stage 2 delivery lacks result identities")
    for relative, key in (
        ("results/summary.csv", "summary_sha256"),
        ("results/tables.md", "tables_md_sha256"),
        ("results/tables.tex", "tables_tex_sha256"),
    ):
        expected = outputs.get(key)
        target = path.parent / relative
        if not _is_sha256(expected) or not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"Stage 2 delivery output changed: {relative}")
    qa = outputs.get("qa")
    if not isinstance(qa, list) or len(qa) != 11:
        raise ValueError("Stage 2 delivery QA universe is incomplete")
    curve = path.parent / "qa/training_curves.png"
    if not curve.is_file() or sha256_file(curve) != qa[0].get("curve_sha256"):
        raise ValueError("Stage 2 delivery training curve changed")
    for record in qa[1:]:
        ordinal = int(record.get("ordinal", -1))
        overlay = path.parent / "qa/overlays" / f"overlay_{ordinal:02d}.png"
        if not 1 <= ordinal <= 10 or not overlay.is_file() or sha256_file(overlay) != record.get("sha256"):
            raise ValueError("Stage 2 delivery overlay universe changed")
    return payload


def _argv_value(job: Mapping[str, Any], option: str) -> str:
    argv = job.get("argv")
    if not isinstance(argv, list) or argv.count(option) != 1:
        raise ValueError(f"{job.get('name')}: expected one {option}")
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise ValueError(f"{job.get('name')}: {option} lacks a value")
    return str(argv[index + 1])


def _load_reservations(path: Path, max_ram_gib: float) -> Dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(EXPECTED_NATIVE):
        raise ValueError("RAM map must cover exactly the 14 native sweep jobs")
    result = {str(name): float(amount) for name, amount in value.items()}
    if any(not 0.0 < amount <= max_ram_gib for amount in result.values()):
        raise ValueError("RAM reservations exceed the runtime cap")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-queue", required=True, type=Path)
    parser.add_argument("--ram-reservations", required=True, type=Path)
    parser.add_argument("--stage2-delivery", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--proof-out", required=True, type=Path)
    parser.add_argument("--max-project-ram-gib", required=True, type=float)
    args = parser.parse_args()
    if not 0.0 < args.max_project_ram_gib:
        raise ValueError("project RAM limit must be a positive runtime cap")
    delivery = _validate_stage2_delivery(args.stage2_delivery)
    payload = yaml.safe_load(args.raw_queue.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ValueError("raw remaining-sweep artifact is not a queue")
    jobs = payload["jobs"]
    names = [job.get("name") for job in jobs]
    if names != list(EXPECTED_RAW):
        raise ValueError("raw remaining-sweep universe/order changed")
    reservations = _load_reservations(
        args.ram_reservations, args.max_project_ram_gib
    )
    guarded = []
    for job in jobs[: len(EXPECTED_NATIVE)]:
        item = copy.deepcopy(job)
        output = Path(_argv_value(item, "--out"))
        item["completion_artifact"] = str(output / "run.json")
        item["resources"] = {"host_ram_gib": reservations[item["name"]]}
        item["job_identity_sha256"] = canonical_sha256(
            {
                "name": item["name"],
                "argv": item["argv"],
                "env": item.get("env", {}),
                "completion_artifact": item["completion_artifact"],
                "resources": item["resources"],
            }
        )
        guarded.append(item)
    output = {key: value for key, value in payload.items() if key != "jobs"}
    output.update(
        schema_version=2,
        phase="learning_rate_sweep_remaining_native",
        launch_authorized=True,
        source_queue=str(args.raw_queue),
        source_queue_sha256=sha256_file(args.raw_queue),
        excluded_completed_jobs=[
            "sweep.sufs_ft.lr0p0003.seed0",
            "sweep.sufs_ft.lr0p001.seed0",
        ],
        stage2_delivery=str(args.stage2_delivery),
        stage2_delivery_sha256=sha256_file(args.stage2_delivery),
        jobs=guarded,
    )
    planner.write_yaml(args.out, output)
    atomic_json(
        args.proof_out,
        {
            "schema_version": 1,
            "operation": "filter_remaining_native_sweep",
            "input_sha256": sha256_file(args.raw_queue),
            "ram_reservations_sha256": sha256_file(args.ram_reservations),
            "stage2_delivery_sha256": sha256_file(args.stage2_delivery),
            "stage2_checkpoint_sha256": delivery["checkpoint_sha256"],
            "excluded_completed_jobs": output["excluded_completed_jobs"],
            "retained_jobs": list(EXPECTED_NATIVE),
            "output_sha256": sha256_file(args.out),
        },
    )


if __name__ == "__main__":
    main()
