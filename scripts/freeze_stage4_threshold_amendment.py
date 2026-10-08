#!/usr/bin/env python3
"""Freeze the operator-authorized Stage 4 threshold-policy amendment.

The command is deliberately results-blind: it reads only the already-frozen
analysis-plan JSON, this committed source file, and Git metadata.  Publication
is write-once; an existing byte-identical artifact is accepted, while any
different existing artifact is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


PRIOR_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
REPLACEMENT_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def build_amendment(
    *,
    base_plan_bytes: bytes,
    base_plan_sha256: str,
    generator_sha256: str,
    git_commit: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Build the amendment after validating the results-blind base policy."""
    if _sha256(base_plan_bytes) != base_plan_sha256:
        raise ValueError("base analysis-plan SHA-256 mismatch")
    plan = json.loads(base_plan_bytes)
    if plan.get("kind") != "stage4_preregistered_analysis_plan":
        raise ValueError("unexpected base analysis-plan kind")
    if plan.get("results_opened_during_creation") is not False:
        raise ValueError("base plan is not results-blind")
    if plan.get("heldout_payloads_opened_during_creation") is not False:
        raise ValueError("base plan opened held-out payloads")
    prior = plan.get("threshold_policy")
    if not isinstance(prior, dict):
        raise ValueError("base plan has no threshold policy")
    if prior.get("allowed_trainable_grid") != PRIOR_GRID:
        raise ValueError("base threshold grid is not the expected superseded grid")
    replacement = dict(prior)
    replacement["allowed_trainable_grid"] = REPLACEMENT_GRID
    replacement.update(
        {
            "selection_data": "validation_only",
            "completed_trainable_pixel_models_reselected": True,
            "test_masks_regenerated_from_saved_probability_maps": True,
            "binary_masks_accepted_as_probability_maps": False,
            "released_zero_shot_thresholds_retuned": False,
        }
    )
    return {
        "schema_version": 1,
        "kind": "stage4_threshold_policy_amendment",
        "base_plan_sha256": base_plan_sha256,
        "supersedes_exactly": ["threshold_policy"],
        "prior_policy": prior,
        "replacement_policy": replacement,
        "authorization": {
            "source": "operator_instruction",
            "recorded_at_utc": recorded_at_utc,
            "reason": "validation-only UNet++ selection reached the prior grid edge",
        },
        "required_attestations": {
            "validation_only": True,
            "test_or_heldout_opened_for_selection": False,
            "probability_maps_required_for_val_and_test": True,
            "threshold_reported_per_trainable_pixel_run": True,
            "legacy_binary_only_outputs_must_be_superseded": True,
        },
        "creation_scope": {
            "results_opened": False,
            "images_opened": False,
            "ground_truth_opened": False,
            "prediction_or_probability_maps_opened": False,
            "model_or_checkpoint_opened": False,
        },
        "generator": {
            "git_commit": git_commit,
            "script_sha256": generator_sha256,
        },
    }


def _publish_write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular(path) != payload:
            raise FileExistsError(f"refusing to replace different artifact: {path}")
        return "verified_identical"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(temporary)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.chmod(tmp, 0o400)
        os.link(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        tmp.unlink(missing_ok=True)
    if _read_regular(path) != payload:
        raise RuntimeError("published amendment failed byte verification")
    return "created"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--recorded-at-utc", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    commit = _git(args.repo, "rev-parse", "HEAD")
    if commit != args.expected_git_commit:
        raise ValueError("repository commit mismatch")
    if _git(args.repo, "status", "--porcelain"):
        raise ValueError("repository must be clean")
    script_bytes = _read_regular(Path(__file__).resolve())
    base_bytes = _read_regular(args.base_plan)
    amendment = build_amendment(
        base_plan_bytes=base_bytes,
        base_plan_sha256=args.expected_base_sha256,
        generator_sha256=_sha256(script_bytes),
        git_commit=commit,
        recorded_at_utc=args.recorded_at_utc,
    )
    encoded = _canonical(amendment)
    status = _publish_write_once(args.out, encoded)
    print(json.dumps({"status": status, "sha256": _sha256(encoded), "path": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
