"""Idempotent aggregate-result updates for benchmark summary tables."""
import math
from numbers import Integral, Real
from pathlib import Path

import pandas as pd


# A result row is a scientific run evaluated on one declared table/GT/group.
# Keep this tuple centralized so callers and tests cannot silently diverge.
MASK_EVALUATION_KEY_COLUMNS = (
    "method",
    "subset",
    "split",
    "seed",
    "ground_truth",
    "table",
    "group_by",
    "group_value",
    "run_id",
    "training_data",
)


def _canonical_key_scalar(value):
    """Return a CSV-stable scalar representation for one key component.

    Pandas promotes an integer column containing an empty value to float on CSV
    reload.  Consequently, seed ``0`` becomes ``0.0``.  Treat integral numeric
    representations as identical while keeping non-integral and string values
    distinct.
    """
    if value is None or (not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)):
        return ""
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    return str(value)


def upsert_summary(path, row, key_columns=("method", "subset")):
    """Insert or replace one aggregate row using an explicit stable key."""
    path = Path(path)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        missing = [key for key in key_columns if key not in row]
        if missing:
            raise ValueError(f"summary row is missing key columns: {missing}")
        matches = pd.Series(True, index=old.index)
        for key in key_columns:
            if key not in old.columns:
                old[key] = ""
            expected = _canonical_key_scalar(row[key])
            actual = old[key].map(_canonical_key_scalar)
            matches &= actual == expected
        old = old[~matches]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
