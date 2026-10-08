#!/usr/bin/env python3
"""Freeze the results-blind Stage 4 analysis policy before held-out scoring.

This command reads only the source-level manifest as CSV metadata and the
frozen subject split.  Stage 1 inputs are accepted as immutable SHA-256
identities, so their result-bearing artifacts are never opened.  The command
also never opens a result CSV, source image, ground-truth mask, prediction
mask, checkpoint, or model artifact.  The output is immutable: an identical
replay is accepted, while different existing bytes are never replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
KIND = "stage4_preregistered_analysis_plan"
EXPECTED_SPLIT_SEED = 20260908
EXPECTED_TEST_SUBJECTS = 40
PRESCRIBED_TEST_SUBJECT_SHA256 = (
    "4cf381878beb5b4d8dfc12492a1803d498f067927c3f73f0bf0440be825dc684"
)
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_REPLICATES = 1000
EXPECTED_BOOTSTRAP_DRAW_SHA256 = (
    "c9beff3d21504c02c1b6f178d2392f696786a8976b7e93a413dc07a9b9727bca"
)
METRICS = ("accuracy", "error_rate", "dice", "jaccard", "miou", "mdr", "fdr")
GROUPS = ("VALUE", "LIMITED", "NOVALUE", "MIXED")
EXPECTED_GROUP_COUNTS: Mapping[str, Mapping[str, int]] = {
    "A": {"overall": 2010, "VALUE": 743, "LIMITED": 336, "NOVALUE": 922, "MIXED": 9},
    "A_prime": {
        "overall": 2009,
        "VALUE": 742,
        "LIMITED": 336,
        "NOVALUE": 922,
        "MIXED": 9,
    },
}
REAL_METHODS = (
    "gmfs_default",
    "gmfs_tuned",
    "sufs",
    "sufs_ft",
    "fingernet",
    "svm",
    "rf",
    "welm",
    "unet",
    "unetpp",
    "drunet",
    "ra_runet",
    "lenet",
    "alexnet",
    "cnn_ae",
    "segformer_sub",
)
REFERENCE_METHODS = ("all_background", "all_foreground")
STAGE1_THREE_SEED_METHODS = frozenset(("sufs_ft", "svm", "rf", "welm"))
NATIVE_THREE_SEED_METHODS = frozenset(
    ("unet", "unetpp", "drunet", "ra_runet", "lenet", "alexnet", "cnn_ae", "segformer_sub")
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}")


class PlanError(RuntimeError):
    """Raised when preregistration evidence does not match the frozen protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def require_sha256(value: object, description: str) -> str:
    require(
        isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None,
        f"{description} is not a lowercase SHA-256 digest",
    )
    return str(value)


def require_regular_file(path: Path, description: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{description} is absent, non-regular, or a symlink")


def sha256_file(path: Path) -> str:
    require_regular_file(path, str(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    require_regular_file(path, description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"could not read {description}: {exc}") from exc
    require(isinstance(payload, dict), f"{description} root is not an object")
    return payload


def canonical_subject_digest(subjects: Sequence[str]) -> str:
    ordered = sorted(subjects)
    require(len(ordered) == len(set(ordered)), "test subject list contains duplicates")
    require(all(value and "\n" not in value and "\r" not in value for value in ordered),
            "test subject identifier is empty or contains a line break")
    payload = ("\n".join(ordered) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bootstrap_draw_digest(seed: int, replicates: int, subject_count: int) -> str:
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(
        0,
        subject_count,
        size=(replicates, subject_count),
        dtype=np.int64,
    )
    encoded = draws.astype("<i8", copy=False).tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def load_split_metadata(path: Path) -> tuple[dict[str, str], int, list[str]]:
    payload = load_json_object(path, "subject split")
    seed = payload.get("seed")
    require(isinstance(seed, int) and not isinstance(seed, bool), "split seed is not an integer")
    require(seed == EXPECTED_SPLIT_SEED, f"split seed is {seed}, expected {EXPECTED_SPLIT_SEED}")
    raw = payload.get("splits")
    require(isinstance(raw, dict) and raw, "subject split mapping is absent or empty")
    splits: dict[str, str] = {}
    for subject, label in raw.items():
        key = str(subject)
        require(key and "\n" not in key and "\r" not in key, "invalid split subject identifier")
        require(isinstance(label, str) and label in {"train", "val", "test"},
                f"invalid split label for subject {key!r}")
        require(key not in splits, f"duplicate subject after string normalization: {key!r}")
        splits[key] = label
    test_subjects = sorted(subject for subject, label in splits.items() if label == "test")
    require(
        len(test_subjects) == EXPECTED_TEST_SUBJECTS,
        f"split has {len(test_subjects)} test subjects, expected {EXPECTED_TEST_SUBJECTS}",
    )
    return splits, seed, test_subjects


def inspect_manifest_metadata(path: Path, splits: Mapping[str, str]) -> dict[str, Any]:
    """Validate only CSV fields; referenced image/mask paths are never opened."""
    require_regular_file(path, "source manifest")
    required = {
        "image_id",
        "subject_id",
        "subset",
        "mask_path",
        "quality_mask_path",
        "examiner_value",
    }
    counts = {"A": Counter(), "A_prime": Counter()}
    seen_ids: set[str] = set()
    observed_test_subjects: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "source manifest has no header")
        missing = sorted(required - set(reader.fieldnames))
        require(not missing, f"source manifest lacks columns: {missing}")
        for line, row in enumerate(reader, start=2):
            image_id = str(row.get("image_id", ""))
            subject = str(row.get("subject_id", ""))
            require(image_id, f"manifest row {line} has an empty image_id")
            require(image_id not in seen_ids, f"duplicate manifest image_id: {image_id!r}")
            seen_ids.add(image_id)
            require(subject in splits, f"manifest row {line} has unknown subject {subject!r}")
            if splits[subject] != "test" or str(row.get("subset", "")) != "finger_latent":
                continue
            observed_test_subjects.add(subject)
            examiner = str(row.get("examiner_value", ""))
            if examiner == "NONPRINT" or not str(row.get("mask_path", "")):
                continue
            require(examiner in GROUPS, f"unexpected test examiner value {examiner!r}")
            counts["A"][examiner] += 1
            if str(row.get("quality_mask_path", "")):
                counts["A_prime"][examiner] += 1
    expected_test_subjects = {subject for subject, label in splits.items() if label == "test"}
    require(
        observed_test_subjects == expected_test_subjects,
        "manifest test-subject coverage differs from the frozen split",
    )
    observed: dict[str, dict[str, int]] = {}
    for table in ("A", "A_prime"):
        values = {group: int(counts[table][group]) for group in GROUPS}
        values = {"overall": sum(values.values()), **values}
        expected = dict(EXPECTED_GROUP_COUNTS[table])
        require(values == expected, f"{table} group counts differ: observed={values}, expected={expected}")
        observed[table] = values
    return {
        "unique_image_ids": len(seen_ids),
        "test_subject_coverage": len(observed_test_subjects),
        "tables": observed,
        "referenced_payloads_opened": False,
    }


def validate_stage1_identities(
    *,
    scoring_audit_sha256: str,
    finalizer_audit_sha256: str,
    journal_sha256: str,
    summary_sha256: str,
    summary_rows: int,
    table_metadata_sha256: str,
) -> dict[str, Any]:
    """Validate identity strings without opening any Stage 1 artifact."""
    scoring_audit_sha256 = require_sha256(
        scoring_audit_sha256, "Stage 1 scoring-audit digest"
    )
    finalizer_audit_sha256 = require_sha256(
        finalizer_audit_sha256, "Stage 1 finalizer-audit digest"
    )
    journal_sha256 = require_sha256(journal_sha256, "Stage 1 journal digest")
    summary_sha256 = require_sha256(summary_sha256, "Stage 1 summary digest")
    table_metadata_sha256 = require_sha256(
        table_metadata_sha256, "Stage 1 table metadata digest"
    )
    require(summary_rows == 180, "Stage 1 summary row count changed")
    return {
        "artifacts_opened_during_creation": False,
        "scoring_audit_sha256": scoring_audit_sha256,
        "finalizer_audit_sha256": finalizer_audit_sha256,
        "journal_sha256": journal_sha256,
        "summary": {"rows": summary_rows, "sha256": summary_sha256},
        "table_metadata_sha256": table_metadata_sha256,
    }


def seed_policy(method: str) -> dict[str, Any]:
    if method in STAGE1_THREE_SEED_METHODS:
        return {"kind": "seeded", "target_successful": 3, "minimum_successful": 3, "seeds": [0, 1, 2]}
    if method in NATIVE_THREE_SEED_METHODS:
        return {
            "kind": "seeded",
            "target_successful": 3,
            "minimum_successful": 2,
            "seeds": [0, 1, 2],
            "partial_row_requires_delivery_declared_failure": True,
        }
    return {"kind": "unseeded", "target_successful": 1, "minimum_successful": 1}


def analysis_policy() -> dict[str, Any]:
    draw_sha256 = bootstrap_draw_digest(
        BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES, EXPECTED_TEST_SUBJECTS
    )
    require(
        draw_sha256 == EXPECTED_BOOTSTRAP_DRAW_SHA256,
        "bootstrap implementation does not reproduce the preregistered draw digest",
    )
    methods = [
        {"key": method, "role": "method", "seed_policy": seed_policy(method)}
        for method in REAL_METHODS
    ] + [
        {"key": method, "role": "reference", "seed_policy": seed_policy(method)}
        for method in REFERENCE_METHODS
    ]
    return {
        "method_universe": {
            "presentation_rows": 18,
            "real_method_rows": 16,
            "reference_rows": 2,
            "methods": methods,
            "specification_resolution": "freeze implemented 16-real-plus-2-reference registry",
        },
        "seed_aggregation": {
            "unit": "method_by_source_image",
            "operation": "arithmetic mean of each per-image metric across successful seeds",
            "uses": [
                "point_estimates",
                "subject_bootstrap",
                "leader_selection",
                "wilcoxon",
                "paired_effects",
                "failure_ranking",
            ],
            "seeds_are_independent_observations": False,
            "missing_seed_imputation": False,
        },
        "threshold_policy": {
            "source": "frozen validation-selected run cards",
            "allowed_trainable_grid": [0.3, 0.4, 0.5, 0.6, 0.7],
            "stage4_reselection_permitted": False,
            "stage4_inference_or_rescoring_permitted": False,
            "A_and_A_prime_reuse_identical_predictions": True,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "unit": "subject",
            "subjects_per_draw": EXPECTED_TEST_SUBJECTS,
            "paired": True,
            "same_draw_matrix_for_all_methods_metrics_tables_and_contrasts": True,
            "rng": "numpy.random.Generator(numpy.random.PCG64(seed))",
            "draw_shape": [BOOTSTRAP_REPLICATES, EXPECTED_TEST_SUBJECTS],
            "draw_encoding": "little-endian int64 C-order subject indices",
            "subject_index_mapping": (
                "index i maps to the ith subject_id in ascending canonical sorted order"
            ),
            "draw_sha256": draw_sha256,
            "interval": "two-sided percentile 95%",
            "percentiles": [2.5, 97.5],
            "quantile_method": "linear",
            "estimand": "image-macro mean of method-level per-image metrics",
            "metrics": list(METRICS),
            "nonfinite_metric_rule": "metric-specific nanmean; record finite image count; fail if none finite",
        },
        "leader": {
            "table": "A",
            "view": "overall",
            "metric": "dice",
            "direction": "maximum",
            "value": "unrounded image-macro method-level Dice",
            "eligible_roles": ["method"],
            "references_excluded": list(REFERENCE_METHODS),
            "absolute_tie_tolerance": 1e-12,
            "tie_break": "lexicographically smallest method key",
        },
        "wilcoxon": {
            "table": "A",
            "metric": "method-level per-image Dice",
            "pairing": "exact 2010-source-image universe",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "continuity_correction": False,
            "method": "asymptotic",
            "all_zero_rule": {"statistic": 0.0, "raw_p": 1.0, "status": "all_zero"},
            "effects": ["mean_paired_difference", "median_paired_difference", "rank_biserial"],
        },
        "holm": {
            "alpha": 0.05,
            "family": "Table A leader versus each other preregistered real method",
            "fixed_comparisons": 15,
            "reference_rows_in_family": False,
            "raw_p_tie_break": "comparator method key",
            "adjustment": "Holm step-down with monotone adjusted p-values capped at 1",
            "unavailable_fixed_comparator": "retain fixed family denominator and report N/A with exact reason",
        },
        "qualitative_grid": {
            "images": 8,
            "quotas": {"VALUE": 3, "LIMITED": 3, "NOVALUE": 2},
            "excluded_groups": ["MIXED", "NONPRINT"],
            "score_free": True,
            "source_universe": "Table A source images",
            "selection": (
                "within each quota group, take the required first rows after the frozen order"
            ),
            "order": (
                "ascending SHA256(UTF-8 bytes of 'stage5-qualitative-v1\\0' + image_id), "
                "then image_id ascending"
            ),
            "visualization_seed": "seed 0 if successful, else smallest successful seed; sole unseeded prediction otherwise",
        },
        "failure_gallery": {
            "table": "A",
            "images_per_real_method": 3,
            "ranking": "method-level mean per-image Dice ascending",
            "tie_break": "image_id lexicographic ascending",
            "reference_rows_excluded": True,
            "visualization_seed": "seed 0 if successful, else smallest successful seed; sole unseeded prediction otherwise",
            "record_all_seed_dice": True,
        },
        "failure_policy": {
            "hard_stop": [
                "changed bound digest",
                "duplicate scientific identity",
                "changed image or group universe",
                "missing per-image row",
                "nonzero missing prediction count",
                "unexpected seed",
            ],
            "automatic_heldout_rescore_permitted": False,
            "silent_substitution_permitted": False,
            "declared_failed_seed_rule": "report exact reason and successful-seed count; never impute",
        },
        "external_tables": {
            "B1": {"status": "UNVERIFIED_COORDINATES", "reason": "independent coordinate reference unavailable"},
            "B2": {"status": "pending data", "reason": "FVC data and Thai masks unavailable"},
            "B3": {"status": "pending data", "reason": "returned SD302b human annotations unavailable"},
            "C": {"status": "pending data", "reason": "SD302c human annotations unavailable"},
            "D": {"status": "pending data", "reason": "required cross-domain inputs unavailable"},
            "downstream_matcher": {
                "status": "not_run",
                "reason": "no compatible frozen mask-aware latent-to-rolled matcher adapter",
            },
        },
        "future_input_binding": {
            "stage2": {
                "required_kind": "stage2_unet_heldout_delivery",
                "role": "provenance only; do not concatenate its summary because Stage 3 reuses U-Net seed 0",
            },
            "stage3": {
                "required_kind": "stage3_native_heldout_delivery",
                "digest_binding": "deferred until atomic publication, then frozen in a separate Stage 4 input lock",
            },
            "scientific_summary_combination": "Stage 1 summary plus Stage 3 delivery summary",
        },
    }


def git_generator_identity(repo: Path, expected_commit: str, script: Path) -> dict[str, Any]:
    require(HEX_COMMIT.fullmatch(expected_commit) is not None, "expected Git commit is not a lowercase 40-hex SHA")
    repo = repo.resolve()
    require(repo.is_dir(), "repository root is absent")

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments], cwd=repo, check=False, capture_output=True
        )
        require(completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace").strip())
        return completed.stdout

    root = Path(git("rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    require(root == repo, "--repo-root is not the Git top level")
    observed_commit = git("rev-parse", "HEAD").decode("ascii").strip()
    require(observed_commit == expected_commit, "repository HEAD differs from expected generator commit")
    require(not git("status", "--porcelain"), "repository is dirty; refuse preregistration")
    script = script.resolve()
    try:
        relative = script.relative_to(repo).as_posix()
    except ValueError as exc:
        raise PlanError("generator script is outside the repository") from exc
    git("ls-files", "--error-unmatch", relative)
    committed = git("show", f"{expected_commit}:{relative}")
    current = script.read_bytes()
    require(current == committed, "generator script bytes differ from expected commit")
    return {
        "git_commit": expected_commit,
        "git_dirty": False,
        "script": relative,
        "script_sha256": hashlib.sha256(current).hexdigest(),
        "numpy_version": np.__version__,
    }


def build_plan(
    *,
    manifest: Path,
    splits: Path,
    expected_manifest_sha256: str,
    expected_splits_sha256: str,
    expected_test_subject_sha256: str,
    stage1_scoring_audit_sha256: str,
    stage1_finalizer_audit_sha256: str,
    stage1_journal_sha256: str,
    stage1_summary_sha256: str,
    stage1_summary_rows: int,
    stage1_table_metadata_sha256: str,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    expected_manifest_sha256 = require_sha256(
        expected_manifest_sha256, "expected manifest digest"
    )
    expected_splits_sha256 = require_sha256(
        expected_splits_sha256, "expected split digest"
    )
    expected_test_subject_sha256 = require_sha256(
        expected_test_subject_sha256, "expected test-subject digest"
    )
    inputs = (manifest, splits)
    before = {str(path): sha256_file(path) for path in inputs}
    require(
        before[str(manifest)] == expected_manifest_sha256,
        "source manifest digest differs from the immutable expected identity",
    )
    require(
        before[str(splits)] == expected_splits_sha256,
        "subject split digest differs from the immutable expected identity",
    )
    split_map, split_seed, test_subjects = load_split_metadata(splits)
    test_subject_sha256 = canonical_subject_digest(test_subjects)
    require(
        test_subject_sha256 == expected_test_subject_sha256,
        "canonical test-subject digest differs from the preregistered identity",
    )
    manifest_evidence = inspect_manifest_metadata(manifest, split_map)
    stage1 = validate_stage1_identities(
        scoring_audit_sha256=stage1_scoring_audit_sha256,
        finalizer_audit_sha256=stage1_finalizer_audit_sha256,
        journal_sha256=stage1_journal_sha256,
        summary_sha256=stage1_summary_sha256,
        summary_rows=stage1_summary_rows,
        table_metadata_sha256=stage1_table_metadata_sha256,
    )
    after = {str(path): sha256_file(path) for path in inputs}
    require(before == after, "a preregistration input changed while it was being read")
    policy = analysis_policy()
    bootstrap = policy["bootstrap"]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "results_opened_during_creation": False,
        "heldout_payloads_opened_during_creation": False,
        "creation_scope": {
            "opened": [
                "source manifest CSV metadata",
                "subject split JSON",
                "Git metadata and committed generator bytes",
            ],
            "supplied_without_opening": [
                "immutable Stage 1 scoring-audit SHA-256",
                "immutable Stage 1 finalizer-audit SHA-256",
                "immutable Stage 1 journal SHA-256",
                "immutable Stage 1 summary SHA-256 and row count",
                "immutable Stage 1 table-metadata SHA-256",
            ],
            "never_opened": [
                "result CSVs",
                "source images",
                "ground-truth masks",
                "prediction masks",
                "checkpoints",
                "model artifacts",
                "Stage 1 scoring audit",
                "Stage 1 finalizer audit",
                "Stage 2 delivery",
                "Stage 3 delivery",
            ],
        },
        "generator": dict(generator),
        "inputs": {
            "manifest": {"sha256": before[str(manifest)]},
            "splits": {"sha256": before[str(splits)], "declared_seed": split_seed},
            "stage1": stage1,
        },
        "test_universe": {
            "unit": "source_image",
            "test_subject_count": len(test_subjects),
            "test_subject_ids_embedded": False,
            "subject_canonicalization": "sort subject_id as UTF-8 strings; join with LF; append final LF",
            "test_subject_sha256": test_subject_sha256,
            "expected_test_subject_sha256": expected_test_subject_sha256,
            "tables": manifest_evidence["tables"],
            "manifest_unique_image_ids": manifest_evidence["unique_image_ids"],
            "referenced_payloads_opened": False,
        },
        **policy,
    }
    require(
        bootstrap["draw_sha256"]
        == bootstrap_draw_digest(
            BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES, len(test_subjects)
        )
        == EXPECTED_BOOTSTRAP_DRAW_SHA256,
        "bootstrap draw digest is internally inconsistent",
    )
    return plan


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def publish_once(path: Path, payload: bytes) -> str:
    """Atomically create ``path``; accept only an exact-byte replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        require(path.is_file() and not path.is_symlink(), "existing plan is non-regular or a symlink")
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace changed preregistration plan: {path}")
        return "verified_identical"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.is_file() and not path.is_symlink(), "raced plan is non-regular or a symlink")
            if path.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace concurrently changed plan: {path}")
            return "verified_identical"
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--expected-splits-sha256", required=True)
    parser.add_argument(
        "--expected-test-subject-sha256",
        choices=(PRESCRIBED_TEST_SUBJECT_SHA256,),
        required=True,
    )
    parser.add_argument("--stage1-scoring-audit-sha256", required=True)
    parser.add_argument("--stage1-finalizer-audit-sha256", required=True)
    parser.add_argument("--stage1-journal-sha256", required=True)
    parser.add_argument("--stage1-summary-sha256", required=True)
    parser.add_argument("--stage1-summary-rows", type=int, required=True)
    parser.add_argument("--stage1-table-metadata-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generator = git_generator_identity(
        args.repo_root,
        args.expected_git_commit,
        Path(__file__),
    )
    plan = build_plan(
        manifest=args.manifest,
        splits=args.splits,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_splits_sha256=args.expected_splits_sha256,
        expected_test_subject_sha256=args.expected_test_subject_sha256,
        stage1_scoring_audit_sha256=args.stage1_scoring_audit_sha256,
        stage1_finalizer_audit_sha256=args.stage1_finalizer_audit_sha256,
        stage1_journal_sha256=args.stage1_journal_sha256,
        stage1_summary_sha256=args.stage1_summary_sha256,
        stage1_summary_rows=args.stage1_summary_rows,
        stage1_table_metadata_sha256=args.stage1_table_metadata_sha256,
        generator=generator,
    )
    payload = canonical_json_bytes(plan)
    status = publish_once(args.out, payload)
    print(json.dumps({
        "status": status,
        "kind": KIND,
        "output": str(args.out),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "test_subject_sha256": plan["test_universe"]["test_subject_sha256"],
        "bootstrap_draw_sha256": plan["bootstrap"]["draw_sha256"],
        "results_opened_during_creation": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
