"""Convert an existing subject-wise model split to segbench JSON format.

The input parquet may contain many rows per subject. This converter rejects any
subject assigned to more than one model split and writes one deterministic
mapping from subject identifier to ``train``, ``val``, or ``test``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd


MODEL_SPLIT_MAP: Mapping[str, str] = {
    "model_train": "train",
    "validation": "val",
    "outer_test": "test",
}


def convert_split(
    parquet_path: Path,
    subject_column: str,
    split_column: str,
) -> Dict[str, str]:
    """Read and validate a subject-wise split from a parquet file.

    Args:
        parquet_path: Source parquet containing subject and model-split columns.
        subject_column: Column containing subject identifiers.
        split_column: Column containing the existing model split labels.

    Returns:
        A subject-sorted mapping to segbench split labels.

    Raises:
        ValueError: If required columns or values are missing, labels are
            unknown, or a subject occurs in multiple splits.
    """
    frame = pd.read_parquet(parquet_path, columns=[subject_column, split_column])
    if frame.empty:
        raise ValueError("input parquet contains no rows")
    if frame[[subject_column, split_column]].isna().any().any():
        null_counts = frame[[subject_column, split_column]].isna().sum().to_dict()
        raise ValueError(f"split columns contain null values: {null_counts}")

    frame = frame.astype({subject_column: str, split_column: str})
    assignments_per_subject = frame.groupby(subject_column, sort=True)[
        split_column
    ].nunique()
    conflicts = assignments_per_subject[assignments_per_subject != 1]
    if not conflicts.empty:
        examples = conflicts.index[:10].tolist()
        raise ValueError(
            f"{len(conflicts)} subjects map to multiple model splits; "
            f"examples: {examples}"
        )

    subject_rows = frame.drop_duplicates(subset=[subject_column]).sort_values(
        subject_column
    )
    unknown = sorted(set(subject_rows[split_column]) - set(MODEL_SPLIT_MAP))
    if unknown:
        raise ValueError(f"unknown model split labels: {unknown}")

    return {
        subject: MODEL_SPLIT_MAP[model_split]
        for subject, model_split in subject_rows[
            [subject_column, split_column]
        ].itertuples(index=False, name=None)
    }


def verify_expected_counts(
    splits: Mapping[str, str],
    expected_train: Optional[int],
    expected_val: Optional[int],
    expected_test: Optional[int],
) -> Dict[str, int]:
    """Verify optional expected subject counts and return observed counts.

    Args:
        splits: Subject-to-segbench-split mapping.
        expected_train: Expected training subject count, or ``None``.
        expected_val: Expected validation subject count, or ``None``.
        expected_test: Expected test subject count, or ``None``.

    Returns:
        Observed counts keyed by segbench split label.

    Raises:
        ValueError: If any supplied expectation does not match.
    """
    observed = {
        label: sum(value == label for value in splits.values())
        for label in ("train", "val", "test")
    }
    expected = {
        "train": expected_train,
        "val": expected_val,
        "test": expected_test,
    }
    mismatches = {
        label: {"expected": count, "observed": observed[label]}
        for label, count in expected.items()
        if count is not None and observed[label] != count
    }
    if mismatches:
        raise ValueError(f"subject-count verification failed: {mismatches}")
    return observed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--subject-column", default="subject")
    parser.add_argument("--split-column", default="model_split")
    parser.add_argument("--expect-train-subjects", type=int)
    parser.add_argument("--expect-val-subjects", type=int)
    parser.add_argument("--expect-test-subjects", type=int)
    return parser.parse_args()


def main() -> None:
    """Validate, convert, and write the split file."""
    args = parse_args()
    splits = convert_split(args.parquet, args.subject_column, args.split_column)
    counts = verify_expected_counts(
        splits,
        args.expect_train_subjects,
        args.expect_val_subjects,
        args.expect_test_subjects,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": args.seed, "splits": splits}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"unique subjects: {sum(counts.values())}")
    print(
        "subject counts: "
        f"train={counts['train']} val={counts['val']} test={counts['test']}"
    )
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
