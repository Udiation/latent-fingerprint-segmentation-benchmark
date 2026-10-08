"""Tests for idempotent aggregate-result updates."""
from pathlib import Path
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segbench.summary import MASK_EVALUATION_KEY_COLUMNS, upsert_summary


def test_upsert_replaces_only_the_complete_composite_key() -> None:
    keys = ("method", "subset", "seed", "ground_truth", "group_value")
    base = {
        "method": "model",
        "subset": "latent",
        "seed": 0,
        "ground_truth": "mask_path",
        "group_value": "overall",
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "summary.csv"
        upsert_summary(path, {**base, "dice_mean": 0.1}, key_columns=keys)
        upsert_summary(path, {**base, "seed": 1, "dice_mean": 0.2}, key_columns=keys)
        upsert_summary(path, {**base, "dice_mean": 0.3}, key_columns=keys)
        rows = pd.read_csv(path).sort_values("seed")
        assert len(rows) == 2
        assert rows.dice_mean.tolist() == [0.3, 0.2]


def test_upsert_seed_is_stable_after_nullable_csv_reload() -> None:
    """An unseeded row must not turn a seed-0 rerun into a duplicate."""
    base = {
        "subset": "finger_latent",
        "split": "test",
        "ground_truth": "mask_path",
        "table": "A",
        "group_by": "examiner_value",
        "group_value": "overall",
        "run_id": "frozen-model",
        "training_data": "SD302-Latent train split",
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "summary.csv"
        upsert_summary(
            path,
            {**base, "method": "gmfs_default", "seed": None, "dice_mean": 0.1},
            key_columns=MASK_EVALUATION_KEY_COLUMNS,
        )
        upsert_summary(
            path,
            {**base, "method": "rf", "seed": 0, "dice_mean": 0.2},
            key_columns=MASK_EVALUATION_KEY_COLUMNS,
        )
        # Reload occurs inside upsert_summary; pandas now represents seed 0 as 0.0.
        upsert_summary(
            path,
            {**base, "method": "rf", "seed": 0, "dice_mean": 0.3},
            key_columns=MASK_EVALUATION_KEY_COLUMNS,
        )
        rows = pd.read_csv(path)
        rf_rows = rows[rows.method == "rf"]
        assert len(rows) == 2
        assert len(rf_rows) == 1
        assert rf_rows.iloc[0].dice_mean == 0.3


def test_evaluation_key_separates_table_run_and_training_regime() -> None:
    base = {
        "method": "unet",
        "subset": "finger_latent",
        "split": "test",
        "seed": 0,
        "ground_truth": "mask_path",
        "group_by": "examiner_value",
        "group_value": "overall",
    }
    variants = [
        {**base, "table": "A", "run_id": "latent-0", "training_data": "latent"},
        {**base, "table": "D", "run_id": "latent-0", "training_data": "latent"},
        {**base, "table": "A", "run_id": "latent-1", "training_data": "latent"},
        {**base, "table": "A", "run_id": "fvc-0", "training_data": "FVC set B"},
    ]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "summary.csv"
        for index, row in enumerate(variants):
            upsert_summary(
                path,
                {**row, "dice_mean": index / 10},
                key_columns=MASK_EVALUATION_KEY_COLUMNS,
            )
        upsert_summary(
            path,
            {**variants[0], "dice_mean": 0.9},
            key_columns=MASK_EVALUATION_KEY_COLUMNS,
        )
        rows = pd.read_csv(path)
        assert len(rows) == 4
        selected = rows[
            (rows.table == "A")
            & (rows.run_id == "latent-0")
            & (rows.training_data == "latent")
        ]
        assert len(selected) == 1
        assert selected.iloc[0].dice_mean == 0.9
