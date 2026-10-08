import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "make_tables.py"
if not MODULE_PATH.exists():  # isolated development copy used by this test module
    MODULE_PATH = Path(__file__).with_name("make_tables.py")
SPEC = importlib.util.spec_from_file_location("paper_tables", MODULE_PATH)
paper_tables = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = paper_tables
SPEC.loader.exec_module(paper_tables)


def row(method, table, dice, *, seed=None, family="Classical", group="overall", **overrides):
    values = {
        "method": method,
        "display_name": overrides.pop("display_name", method),
        "table": table,
        "group_by": "examiner_value" if group != "overall" else "",
        "group_value": group,
        "training_data": overrides.pop("training_data", "SD302-Latent train"),
        "subset": "finger_latent",
        "split": "test",
        "ground_truth": "mask_path" if table == "A" else "quality_mask_path",
        "seed": seed,
        "year": overrides.pop("year", 2020),
        "family": family,
        "comment": overrides.pop("comment", "synthetic aggregate row"),
        "params_m": overrides.pop("params_m", 1.5),
        "device": overrides.pop("device", "CPU"),
        "n_images": overrides.pop("n_images", 20),
        "n_missing": 0,
        "ms_per_img": overrides.pop("ms_per_img", 10.0),
    }
    metrics = {
        "accuracy": dice + 0.05,
        "error_rate": 0.95 - dice,
        "dice": dice,
        "jaccard": dice - 0.1,
        "miou": dice - 0.05,
        "mdr": 1.0 - dice,
        "fdr": 0.5 * (1.0 - dice),
    }
    for key, value in metrics.items():
        values[f"{key}_mean"] = value
        values[f"{key}_std"] = 0.99  # must never become the seed std
    values.update(overrides)
    return values


def test_seed_means_use_between_seed_sample_std_and_single_run_omits_std():
    frame = pd.DataFrame([
        row("unet", "A", 0.70, seed=0, family="Encoder-decoder"),
        row("unet", "A", 0.80, seed=1, family="Encoder-decoder"),
        row("unet", "A", 0.90, seed=2, family="Encoder-decoder"),
        row("sufs", "A", 0.75, seed=None, family="Pretrained"),
    ])
    result = paper_tables.aggregate_seed_rows(frame).set_index("method")
    assert result.loc["unet", "n_seeds"] == 3
    assert result.loc["unet", "dice_mean"] == pytest.approx(0.8)
    assert result.loc["unet", "dice_seed_std"] == pytest.approx(0.1)
    assert result.loc["unet", "params_m_seed_std"] == pytest.approx(0.0)
    assert result.loc["unet", "ms_per_img_seed_std"] == pytest.approx(0.0)
    assert result.loc["sufs", "n_seeds"] == 1
    assert np.isnan(result.loc["sufs", "dice_seed_std"])
    assert np.isnan(result.loc["sufs", "params_m_seed_std"])
    assert np.isnan(result.loc["sufs", "ms_per_img_seed_std"])
    assert paper_tables.format_metric(result.loc["unet"], paper_tables.METRICS[2]) == "80.00 ± 10.00"
    assert paper_tables.format_metric(result.loc["sufs"], paper_tables.METRICS[2]) == "75.00"
    assert paper_tables.format_params(result.loc["unet"]) == "1.5"
    assert paper_tables.format_timing(result.loc["unet"]) == "10.0 (CPU)"


def test_seed_varying_parameter_counts_and_timings_use_mean_and_sample_std():
    params = [53.132620, 52.316740, 52.640860]
    timings = [101.0, 105.0, 99.0]
    frame = pd.DataFrame([
        row("rf", "A", 0.70 + seed * 0.01, seed=seed, params_m=params[seed],
            ms_per_img=timings[seed])
        for seed in range(3)
    ])
    result = paper_tables.aggregate_seed_rows(frame).iloc[0]
    assert result.params_m == pytest.approx(np.mean(params))
    assert result.params_m_seed_std == pytest.approx(np.std(params, ddof=1))
    assert result.ms_per_img == pytest.approx(np.mean(timings))
    assert result.ms_per_img_seed_std == pytest.approx(np.std(timings, ddof=1))
    assert paper_tables.format_params(result) == "52.697 ± 0.411"
    assert paper_tables.format_timing(result) == "101.7 ± 3.1 (CPU)"


def test_table_a_and_a_prime_are_isolated(tmp_path):
    frame = pd.DataFrame([row("rf", "A", 0.8), row("rf", "A_prime", 0.6)])
    paper_tables.build_tables(frame, tmp_path)
    a = pd.read_csv(tmp_path / "table_a.csv")
    a_prime = pd.read_csv(tmp_path / "table_a_prime.csv")
    assert list(a.table) == ["A"]
    assert list(a_prime.table) == ["A_prime"]
    assert a.dice_mean.iloc[0] == pytest.approx(0.8)
    assert a_prime.dice_mean.iloc[0] == pytest.approx(0.6)


def test_family_order_and_reference_rows_are_excluded_from_ranking():
    frame = pd.DataFrame([
        row("sufs", "A", 0.70, family="Pretrained"),
        row("unet", "A", 0.80, family="Encoder-decoder"),
        row("lenet", "A", 0.75, family="Patch CNN"),
        row("rf", "A", 0.60, family="Classical"),
        row("segformer", "A", 0.77, family="Transformer"),
        row("all_foreground", "A", 0.99, family=""),
    ])
    aggregate = paper_tables.aggregate_seed_rows(frame)
    assert list(aggregate.family) == list(paper_tables.FAMILY_ORDER)
    marks = paper_tables.ranking_marks(aggregate, paper_tables.METRICS[2])
    by_method = {aggregate.loc[index, "method"]: mark for index, mark in marks.items()}
    assert by_method == {"unet": "best", "segformer": "second"}
    markdown = paper_tables.render_markdown(aggregate, "Synthetic", "")
    assert "**80.00**" in markdown
    assert "<u>77.00</u>" in markdown
    assert "**99.00**" not in markdown


def test_legacy_summary_without_table_or_new_metadata_is_still_supported():
    legacy = row("rf", "", 0.8)
    for key in ("table", "group_by", "group_value", "year", "family", "training_data",
                "comment", "params_m", "device", "seed"):
        legacy.pop(key)
    legacy["ground_truth"] = "mask_path"
    result = paper_tables.aggregate_seed_rows(pd.DataFrame([legacy]))
    assert result.loc[0, "table"] == "A"
    assert result.loc[0, "view"] == "overall"
    assert result.loc[0, "n_seeds"] == 1


def test_examiner_values_rank_within_each_value_and_emit_separate_view(tmp_path):
    frame = pd.DataFrame([
        row("rf", "A", 0.70, group="VALUE"), row("unet", "A", 0.80, group="VALUE"),
        row("rf", "A", 0.90, group="LIMITED"), row("unet", "A", 0.60, group="LIMITED"),
        row("rf", "A", 0.75), row("unet", "A", 0.78),
    ])
    paper_tables.build_tables(frame, tmp_path)
    grouped = (tmp_path / "table_a_examiner_value.md").read_text()
    overall = (tmp_path / "table_a.md").read_text()
    assert "Examiner value" in grouped
    assert "Examiner value" not in overall
    value_unet = next(line for line in grouped.splitlines() if "| unet |" in line and "| VALUE |" in line)
    limited_rf = next(line for line in grouped.splitlines() if "| rf |" in line and "| LIMITED |" in line)
    assert "| **80.00** |" in value_unet
    assert "| **90.00** |" in limited_rf


def test_metadata_supplies_caption_facts_without_inference(tmp_path):
    frame = pd.DataFrame([row("rf", "A", 0.8)])
    metadata = {"tables": {"A": {
        "title": "Declared title",
        "dataset": "Declared dataset",
        "test_impressions": 123,
        "test_subjects": 9,
        "seeds": "0, 1, 2",
        "threshold_selection": "validation only",
    }}}
    paper_tables.build_tables(frame, tmp_path, metadata)
    output = (tmp_path / "table_a.md").read_text()
    assert "Declared title" in output
    assert "Dataset: Declared dataset" in output
    assert "Test impressions: 123" in output
    assert "Test subjects: 9" in output
    neutral = tmp_path / "neutral"
    paper_tables.build_tables(frame, neutral)
    neutral_text = (neutral / "table_a.md").read_text()
    assert "123" not in neutral_text
    assert "NIST" not in neutral_text


def test_duplicate_seed_or_inconsistent_metadata_fails_loudly():
    duplicate = pd.DataFrame([row("unet", "A", 0.7, seed=0), row("unet", "A", 0.8, seed=0)])
    with pytest.raises(ValueError, match="duplicate aggregate rows"):
        paper_tables.aggregate_seed_rows(duplicate)
    inconsistent = pd.DataFrame([
        row("unet", "A", 0.7, seed=0, family="Classical"),
        row("unet", "A", 0.8, seed=1, family="Encoder-decoder"),
    ])
    # Family is part of scientific row metadata and must not silently change across seeds.
    with pytest.raises(ValueError, match="inconsistent family"):
        paper_tables.aggregate_seed_rows(inconsistent)


def test_latex_is_booktabs_and_csv_retains_raw_seed_statistics(tmp_path):
    frame = pd.DataFrame([
        row("unet", "A", 0.7, seed=0, family="Encoder-decoder"),
        row("unet", "A", 0.9, seed=1, family="Encoder-decoder"),
    ])
    paper_tables.build_tables(frame, tmp_path, {"A": {"caption": "Supplied caption."}})
    latex = (tmp_path / "table_a.tex").read_text()
    csv = pd.read_csv(tmp_path / "table_a.csv")
    assert "\\toprule" in latex and "\\midrule" in latex and "\\bottomrule" in latex
    assert "Supplied caption." in latex
    assert csv.loc[0, "n_seeds"] == 2
    assert csv.loc[0, "dice_mean"] == pytest.approx(0.8)
    assert csv.loc[0, "dice_seed_std"] == pytest.approx(np.sqrt(0.02))
    assert csv.loc[0, "params_m_seed_std"] == pytest.approx(0.0)
    assert csv.loc[0, "ms_per_img_seed_std"] == pytest.approx(0.0)
