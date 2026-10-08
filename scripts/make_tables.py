"""Build paper-ready tables from aggregate evaluator summaries only.

Each input row is one aggregate evaluation run (normally one seed). Trained
replicates are reduced across their seed-level means; parameter counts and
timings are reduced across seeds the same way. Within-image ``*_std`` columns
are never mistaken for between-seed uncertainty. This module does not open
manifests, per-image results, images, or masks.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    scale: float
    decimals: int
    direction: str


METRICS = (
    Metric("accuracy", "Acc (%)", 100.0, 2, "max"),
    Metric("error_rate", "Error (%)", 100.0, 2, "min"),
    Metric("dice", "Dice (%)", 100.0, 2, "max"),
    Metric("jaccard", "Jaccard (%)", 100.0, 2, "max"),
    Metric("miou", "MIoU", 1.0, 3, "max"),
    Metric("mdr", "MDR (%)", 100.0, 2, "min"),
    Metric("fdr", "FDR (%)", 100.0, 2, "min"),
)
FAMILY_ORDER = (
    "Classical", "Patch CNN", "Encoder-decoder", "Transformer", "Pretrained", "Reference"
)
FAMILY_ALIASES = {
    "classical": "Classical", "patch cnn": "Patch CNN", "patch-cnn": "Patch CNN",
    "encoder decoder": "Encoder-decoder", "encoder-decoder": "Encoder-decoder",
    "encoder–decoder": "Encoder-decoder", "transformer": "Transformer",
    "pretrained": "Pretrained", "reference": "Reference", "baseline": "Reference",
}
REFERENCE_METHODS = {
    "all_background", "all-background", "all_foreground", "all-foreground",
    "hull_as_prediction", "hull-as-prediction",
}
LEGACY_TABLES = {"finger_latent": "A", "finger_livescan": "B", "palm": "C"}
IDENTITY = (
    "table", "view", "group_by", "group_value", "method", "training_data",
    "subset", "split", "ground_truth",
)
CONSISTENT = ("display_name", "year", "family", "comment", "device", "n_images")


def text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def canonical_seed(value: Any) -> str:
    value = text(value)
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if math.isfinite(number) and number.is_integer() else value


def normalize_family(value: Any) -> str:
    value = text(value)
    folded = re.sub(r"\s+", " ", value.lower()).strip()
    return FAMILY_ALIASES.get(folded, value or "Unspecified")


def normalize_table(value: Any) -> str:
    value = text(value)
    return {"a'": "A_prime", "a′": "A_prime", "a_prime": "A_prime", "aprime": "A_prime"}.get(
        value.lower(), value
    )


def derive_table(row: pd.Series) -> str:
    explicit = normalize_table(row.get("table", ""))
    if explicit:
        return explicit
    if "quality" in text(row.get("ground_truth", "")).lower():
        return "A_prime"
    subset = text(row.get("subset", ""))
    return LEGACY_TABLES.get(subset, subset or "unassigned")


def derive_view(row: pd.Series) -> str:
    value = text(row.get("group_value", ""))
    if not value or value.lower() == "overall":
        return "overall"
    return text(row.get("group_by", "")) or "grouped"


def prepare_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"method", *(f"{metric.key}_mean" for metric in METRICS)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"summary is missing required aggregate columns: {missing}")
    if frame.empty:
        raise ValueError("summary has no aggregate rows")
    out = frame.copy()
    string_columns = (
        "table", "group_by", "group_value", "training_data", "subset", "split",
        "ground_truth", "display_name", "year", "family", "comment", "device", "seed",
    )
    for column in string_columns:
        if column not in out:
            out[column] = ""
    for column in ("params_m", "ms_per_img", "n_images", "n_missing"):
        if column not in out:
            out[column] = np.nan
    out["table"] = out.apply(derive_table, axis=1)
    out["view"] = out.apply(derive_view, axis=1)
    out["group_by"] = out.group_by.map(text)
    out["group_value"] = out.group_value.map(text)
    out.loc[out.view == "overall", "group_value"] = "overall"
    out["family"] = out.family.map(normalize_family)
    out["seed"] = out.seed.map(canonical_seed)
    out["method"] = out.method.map(text)
    if (out.method == "").any():
        raise ValueError("summary contains an empty method key")
    reference = out.method.str.lower().str.replace(" ", "_", regex=False).isin(REFERENCE_METHODS)
    out.loc[reference, "family"] = "Reference"
    return out


def consistent_value(group: pd.DataFrame, column: str) -> Any:
    values = [value for value in group[column] if text(value)]
    if not values:
        return np.nan if column == "n_images" else ""
    if column == "n_images":
        numbers = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
        if not len(numbers):
            return np.nan
        if not np.allclose(numbers, numbers[0], rtol=1e-9, atol=1e-12):
            raise ValueError(f"inconsistent {column} within one aggregate group: {values}")
        return float(numbers[0])
    unique = {text(value) for value in values}
    if len(unique) != 1:
        raise ValueError(f"inconsistent {column} within one aggregate group: {sorted(unique)}")
    return next(iter(unique))


def validate_seed_rows(group: pd.DataFrame) -> list[str]:
    seeds = group.seed.tolist()
    populated = [seed for seed in seeds if seed]
    if populated and len(populated) != len(seeds):
        raise ValueError("one aggregate group mixes seeded and unseeded evaluations")
    identities = populated or ["unseeded"] * len(seeds)
    duplicates = sorted({seed for seed in identities if identities.count(seed) > 1})
    if duplicates:
        raise ValueError("duplicate aggregate rows for seed(s): " + ", ".join(duplicates))
    return identities


def family_rank(family: str) -> int:
    return FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)


def sort_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.assign(
        _family=frame.family.map(family_rank),
        _year=pd.to_numeric(frame.year, errors="coerce").fillna(9999),
    ).sort_values(["_family", "family", "_year", "display_name", "method"], kind="stable")
    return out.drop(columns=["_family", "_year"]).reset_index(drop=True)


def aggregate_seed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Group seed-level means; use sample std only when at least two seeds exist."""
    frame = prepare_summary(frame)
    records = []
    for identity, group in frame.groupby(list(IDENTITY), dropna=False, sort=False):
        seeds = validate_seed_rows(group)
        row = dict(zip(IDENTITY, identity))
        row.update({column: consistent_value(group, column) for column in CONSISTENT})
        row["display_name"] = row["display_name"] or row["method"]
        row["n_seeds"] = len(seeds)
        row["seed_ids"] = ",".join(seeds)
        row["n_missing"] = pd.to_numeric(group.n_missing, errors="coerce").max()
        for quantity in ("params_m", "ms_per_img"):
            values = pd.to_numeric(group[quantity], errors="coerce").dropna()
            row[quantity] = values.mean()
            row[f"{quantity}_seed_std"] = (
                values.std(ddof=1) if len(values) > 1 else np.nan
            )
        for metric in METRICS:
            values = pd.to_numeric(group[f"{metric.key}_mean"], errors="coerce")
            row[f"{metric.key}_mean"] = values.mean()
            row[f"{metric.key}_seed_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
        records.append(row)
    if not records:
        raise ValueError("summary produced no paper rows")
    return sort_rows(pd.DataFrame(records))


def is_reference(row: pd.Series) -> bool:
    method = text(row.get("method", "")).lower().replace(" ", "_")
    return text(row.get("family", "")) == "Reference" or method in REFERENCE_METHODS


def ranking_marks(frame: pd.DataFrame, metric: Metric) -> dict[int, str]:
    candidates = frame.loc[~frame.apply(is_reference, axis=1), f"{metric.key}_mean"].dropna()
    if candidates.empty or metric.direction not in {"min", "max"}:
        return {}
    distinct = sorted(set(candidates.astype(float)), reverse=metric.direction == "max")
    marks = {}
    for index, value in candidates.items():
        if math.isclose(value, distinct[0], rel_tol=1e-12, abs_tol=1e-15):
            marks[int(index)] = "best"
        elif len(distinct) > 1 and math.isclose(value, distinct[1], rel_tol=1e-12, abs_tol=1e-15):
            marks[int(index)] = "second"
    return marks


def all_ranking_marks(frame: pd.DataFrame) -> dict[str, dict[int, str]]:
    marks = {metric.key: {} for metric in METRICS}
    partitions = [frame] if frame.view.iloc[0] == "overall" else [g for _, g in frame.groupby("group_value")]
    for partition in partitions:
        for metric in METRICS:
            marks[metric.key].update(ranking_marks(partition, metric))
    return marks


def format_metric(row: pd.Series, metric: Metric) -> str:
    mean = row[f"{metric.key}_mean"]
    if pd.isna(mean):
        return "-"
    result = f"{mean * metric.scale:.{metric.decimals}f}"
    std = row[f"{metric.key}_seed_std"]
    if row.n_seeds > 1 and pd.notna(std):
        result += f" ± {std * metric.scale:.{metric.decimals}f}"
    return result


def format_year(value: Any) -> str:
    value = text(value)
    if not value:
        return "-"
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else value


def format_params_value(value: Any) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if value == 0:
        return "0"
    if 0 < abs(value) < 0.001:
        return "<0.001"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_params(row: pd.Series) -> str:
    result = format_params_value(row.params_m)
    std = row.params_m_seed_std
    if result != "-" and pd.notna(std) and round(float(std), 3) != 0:
        result += " ± " + format_params_value(std)
    return result


def format_timing(row: pd.Series) -> str:
    if pd.isna(row.ms_per_img):
        return "-"
    timing = f"{row.ms_per_img:.1f}"
    std = row.ms_per_img_seed_std
    if pd.notna(std) and round(float(std), 1) != 0:
        timing += f" ± {std:.1f}"
    return f"{timing} ({text(row.device)})" if text(row.device) else timing


def group_label(frame: pd.DataFrame) -> str:
    value = text(frame.group_by.iloc[0])
    return value.replace("_", " ").capitalize() if value else "Group"


def display_columns(frame: pd.DataFrame) -> list[str]:
    grouped = frame.view.iloc[0] != "overall"
    columns = ["Year", "Approach [ref]", "Family", "Training data", "Comment"]
    if grouped:
        columns.append(group_label(frame))
    return (
        columns
        + [metric.label for metric in METRICS]
        + ["Params (M)", "ms/img (device)", "Seeds"]
    )


def display_rows(frame: pd.DataFrame, style: str = "plain") -> list[list[str]]:
    grouped, marks = frame.view.iloc[0] != "overall", all_ranking_marks(frame)
    output = []
    for index, row in frame.iterrows():
        cells = [format_year(row.year), text(row.display_name), text(row.family),
                 text(row.training_data) or "-", text(row.comment) or "-"]
        if grouped:
            cells.append(text(row.group_value) or "-")
        for metric in METRICS:
            value = format_metric(row, metric)
            mark = marks[metric.key].get(int(index))
            if style == "markdown" and mark:
                value = f"**{value}**" if mark == "best" else f"<u>{value}</u>"
            elif style == "latex" and mark:
                value = (r"\textbf{" if mark == "best" else r"\underline{") + value + "}"
            cells.append(value)
        cells += [format_params(row), format_timing(row), str(int(row.n_seeds))]
        output.append(cells)
    return output


def metadata_for_view(metadata: Mapping[str, Any], table: str, view: str) -> dict[str, Any]:
    tables = metadata.get("tables", metadata)
    if not isinstance(tables, Mapping):
        raise ValueError("metadata JSON must be an object or contain an object named 'tables'")
    data = tables.get(table, {})
    if not isinstance(data, Mapping):
        raise ValueError(f"metadata for table {table!r} must be an object")
    merged = {key: value for key, value in data.items() if key != "views"}
    views = data.get("views", {})
    if not isinstance(views, Mapping):
        raise ValueError(f"metadata views for table {table!r} must be an object")
    view_data = views.get(view, {})
    if not isinstance(view_data, Mapping):
        raise ValueError(f"metadata for table {table!r} view {view!r} must be an object")
    merged.update(view_data)
    return merged


def caption_for(metadata: Mapping[str, Any], table: str, view: str) -> tuple[str, str]:
    info = metadata_for_view(metadata, table, view)
    title = text(info.get("title")) or f"Table {table.replace('_prime', '′')}"
    if view != "overall" and not text(info.get("title")):
        title += f" — {view.replace('_', ' ')}"
    if text(info.get("caption")):
        return title, text(info["caption"])
    parts = []
    for key, label in (("dataset", "Dataset"), ("ground_truth", "Ground truth"),
                       ("test_impressions", "Test impressions"),
                       ("test_subjects", "Test subjects"), ("seeds", "Seeds"),
                       ("threshold_selection", "Threshold selection")):
        if text(info.get(key)):
            parts.append(f"{label}: {text(info[key])}")
    return title, "; ".join(parts) + ("." if parts else "")


def escape_markdown(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def escape_latex(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in value)


def render_markdown(frame: pd.DataFrame, title: str, caption: str) -> str:
    columns = display_columns(frame)
    lines = [f"### {title}", ""] + ([caption, ""] if caption else [])
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(escape_markdown(v) for v in row) + " |"
              for row in display_rows(frame, "markdown")]
    return "\n".join(lines) + "\n"


def latex_cell(value: str) -> str:
    for command in (r"\textbf{", r"\underline{"):
        if value.startswith(command) and value.endswith("}"):
            escaped = escape_latex(value[len(command):-1]).replace(" ± ", r" $\pm$ ")
            return command + escaped + "}"
    return escape_latex(value).replace(" ± ", r" $\pm$ ")


def render_latex(frame: pd.DataFrame, title: str, caption: str, label: str) -> str:
    grouped = frame.view.iloc[0] != "overall"
    columns = display_columns(frame)
    align = "lllll" + ("l" if grouped else "") + "r" * (len(columns) - 5 - int(grouped))
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             rf"\caption{{{escape_latex(caption or title)}}}",
             rf"\label{{tab:{escape_latex(label)}}}", rf"\begin{{tabular}}{{{align}}}",
             r"\toprule", " & ".join(escape_latex(v) for v in columns) + r" \\", r"\midrule"]
    previous = None
    for (_, source), cells in zip(frame.iterrows(), display_rows(frame, "latex")):
        if previous is not None and source.family != previous:
            lines.append(r"\addlinespace")
        lines.append(" & ".join(latex_cell(v) for v in cells) + r" \\")
        previous = source.family
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]) + "\n"


def csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["table", "view", "group_by", "group_value", "method", "display_name", "year",
               "family", "training_data", "comment", "n_seeds", "seed_ids", "n_images", "n_missing"]
    for metric in METRICS:
        columns += [f"{metric.key}_mean", f"{metric.key}_seed_std"]
    columns += [
        "params_m",
        "params_m_seed_std",
        "ms_per_img",
        "ms_per_img_seed_std",
        "device",
        "subset",
        "split",
        "ground_truth",
    ]
    return frame.loc[:, columns].copy()


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "table"


def build_tables(summary: pd.DataFrame, out: Path, metadata: Mapping[str, Any] | None = None,
                 selected_tables: Sequence[str] = ()) -> list[Path]:
    metadata, aggregate = metadata or {}, aggregate_seed_rows(summary)
    if selected_tables:
        selected = {normalize_table(value) for value in selected_tables}
        absent = selected - set(aggregate.table)
        if absent:
            raise ValueError(f"requested tables are absent from summary: {sorted(absent)}")
        aggregate = aggregate[aggregate.table.isin(selected)]
    out.mkdir(parents=True, exist_ok=True)
    written, all_md, all_tex, all_csv = [], [], [], []
    for (table, view), part in aggregate.groupby(["table", "view"], sort=False):
        part = sort_rows(part.reset_index(drop=True))
        title, caption = caption_for(metadata, str(table), str(view))
        name = f"table_{slug(str(table))}" + (f"_{slug(str(view))}" if view != "overall" else "")
        md, tex, csv = render_markdown(part, title, caption), render_latex(
            part, title, caption, slug(f"{table}_{view}")), csv_frame(part)
        paths = [out / f"{name}.md", out / f"{name}.tex", out / f"{name}.csv"]
        paths[0].write_text(md, encoding="utf-8")
        paths[1].write_text(tex, encoding="utf-8")
        csv.to_csv(paths[2], index=False)
        written += paths
        all_md.append(md.rstrip()); all_tex.append(tex.rstrip()); all_csv.append(csv)
    legacy = [out / "tables.md", out / "tables.tex", out / "tables.csv"]
    legacy[0].write_text("\n\n".join(all_md) + "\n", encoding="utf-8")
    legacy[1].write_text("\n\n".join(all_tex) + "\n", encoding="utf-8")
    pd.concat(all_csv, ignore_index=True).to_csv(legacy[2], index=False)
    print(legacy[0].read_text(encoding="utf-8"), end="")
    return written + legacy


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--table", action="append", default=[])
    parser.add_argument("--with-std", action="store_true",
                        help="compatibility flag; multi-seed std is always emitted")
    args = parser.parse_args(argv)
    metadata = json.loads(args.metadata_json.read_text()) if args.metadata_json else {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata JSON root must be an object")
    build_tables(pd.read_csv(args.summary), args.out, metadata, args.table)


if __name__ == "__main__":
    main()
