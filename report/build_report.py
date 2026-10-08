#!/usr/bin/env python3
"""Build a self-contained benchmark report from aggregate, authenticated inputs.

The builder never reads a benchmark manifest or per-image result CSV. Public
mode accepts only aggregate tables plus images authenticated by the synthetic
demo receipt. Private mode additionally requires two explicit qualitative
evidence galleries and the low-agreement audit table.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from html import escape
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:
    from .privacy import (
        PublicReportError,
        load_synthetic_receipt,
        scan_infrastructure_text,
        scan_public_html,
        scan_public_text,
        sha256_file,
    )
except ImportError:  # Direct script execution.
    from privacy import (  # type: ignore
        PublicReportError,
        load_synthetic_receipt,
        scan_infrastructure_text,
        scan_public_html,
        scan_public_text,
        sha256_file,
    )


class ReportBuildError(RuntimeError):
    """Raised when a report input is incomplete, ambiguous, or unsafe."""


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    scale: float
    decimals: int
    direction: str | None


METRICS = (
    Metric("accuracy", "Acc (%)", 100.0, 2, "max"),
    Metric("error_rate", "Error (%)", 100.0, 2, "min"),
    Metric("dice", "Dice (%)", 100.0, 2, "max"),
    Metric("jaccard", "Jaccard (%)", 100.0, 2, "max"),
    Metric("miou", "MIoU", 1.0, 3, "max"),
    Metric("mdr", "MDR (%)", 100.0, 2, "min"),
    Metric("fdr", "FDR (%)", 100.0, 2, "min"),
    Metric("mc", "MC (%)", 100.0, 2, "max"),
    Metric("hr", "HR (%)", 100.0, 2, "max"),
    Metric("par", "PAR", 1.0, 3, None),
    Metric("ohf", "OHF (%)", 100.0, 2, None),
)
METRIC_BY_KEY = {metric.key: metric for metric in METRICS}
TABLE_REQUIRED = {"table", "view", "method", "display_name", "family", "n_seeds"}
TABLE_SAFE_COLUMNS = {
    "table", "view", "group_by", "group_value", "method", "display_name", "year",
    "family", "training_data", "comment", "n_seeds", "n_images", "n_missing",
    "params_m", "params_m_seed_std", "ms_per_img", "ms_per_img_seed_std", "device",
    "subset", "split", "ground_truth", "seed_ids",
    *(f"{metric.key}_mean" for metric in METRICS),
    *(f"{metric.key}_seed_std" for metric in METRICS),
}
DATASET_FIELDS = (
    "dataset", "status", "ground_truth", "train_images", "val_images", "test_images",
    "train_subjects", "val_subjects", "test_subjects", "notes",
)
STATISTIC_FIELDS = (
    "table", "metric", "comparison", "reference_method", "method", "estimate",
    "ci_lower", "ci_upper", "p_value", "p_holm", "significant", "n_images",
    "n_subjects", "status",
)
REPRO_FIELDS = {"git_commit", "config_hashes", "seeds", "dates", "data_manifest_sha256"}
TABLE_METADATA_FIELDS = {"table", "view", "title", "caption"}
TABLE_METADATA_DETAIL_FIELDS = {
    "title", "caption", "dataset", "ground_truth", "test_impressions",
    "test_subjects", "seeds", "threshold_selection", "views",
}
PRIVATE_GALLERY_KINDS = {"qualitative_grid", "failure_gallery"}
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


def _regular_file(path: Path, description: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReportBuildError(f"{description} is not a regular file: {path}")
    return path.resolve()


def _read_json(path: Path, description: str) -> Any:
    path = _regular_file(path, description)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportBuildError(f"cannot parse {description}: {exc}") from exc


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    path = _regular_file(path, description)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ReportBuildError(f"cannot parse {description}: {exc}") from exc
    if frame.empty:
        raise ReportBuildError(f"{description} has no rows")
    return frame


def _cell(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _public_value(value: Any, context: str, public: bool) -> str:
    rendered = _cell(value)
    scan_infrastructure_text(rendered, context)
    if public:
        scan_public_text(rendered, context)
    return rendered


def load_tables(path: Path, *, public: bool) -> pd.DataFrame:
    """Load the aggregate table produced by ``make_tables.py``."""
    frame = _read_csv(path, "aggregate tables CSV")
    missing = TABLE_REQUIRED - set(frame.columns)
    unsafe = set(frame.columns) - TABLE_SAFE_COLUMNS
    if missing:
        raise ReportBuildError(f"aggregate tables CSV is missing columns: {sorted(missing)}")
    if unsafe:
        raise ReportBuildError(f"aggregate tables CSV has unapproved columns: {sorted(unsafe)}")
    available_metrics = [
        metric for metric in METRICS if f"{metric.key}_mean" in frame.columns
        and pd.to_numeric(frame[f"{metric.key}_mean"], errors="coerce").notna().any()
    ]
    if not available_metrics:
        raise ReportBuildError("aggregate tables CSV contains no supported metrics")
    if frame.duplicated(["table", "view", "method", "group_value"], keep=False).any():
        raise ReportBuildError("aggregate tables CSV contains duplicate paper rows")
    missing_tables = {"A", "A_prime"} - set(frame["table"].astype(str))
    if missing_tables:
        raise ReportBuildError(f"aggregate tables CSV lacks required tables: {sorted(missing_tables)}")
    for column in frame.columns:
        if frame[column].dtype == object:
            for row_index, value in frame[column].items():
                _public_value(value, f"aggregate table {column} row {row_index}", public)
    for metric in available_metrics:
        values = pd.to_numeric(frame[f"{metric.key}_mean"], errors="coerce")
        if metric.key != "par" and ((values.dropna() < 0) | (values.dropna() > 1)).any():
            raise ReportBuildError(f"aggregate metric {metric.key} is outside [0, 1]")
        if metric.key == "par" and (values.dropna() < 0).any():
            raise ReportBuildError("aggregate metric par is negative")
    return frame


def load_table_metadata(
    path: Path, frame: pd.DataFrame, *, public: bool
) -> dict[tuple[str, str], tuple[str, str]]:
    value = _read_json(path, "table metadata")
    if not isinstance(value, Mapping) or set(value) != {"tables"}:
        raise ReportBuildError("table metadata root must contain only 'tables'")
    rows = value.get("tables") if isinstance(value, Mapping) else None
    if not isinstance(rows, (list, Mapping)) or not rows:
        raise ReportBuildError("table metadata must contain non-empty tables metadata")
    records: dict[tuple[str, str], tuple[str, str]] = {}
    required = {(str(table), str(view)) for table, view in frame[["table", "view"]].drop_duplicates().itertuples(index=False)}
    if isinstance(rows, list):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) - TABLE_METADATA_FIELDS:
                raise ReportBuildError(f"table metadata row {index} has an invalid schema")
            key = (_cell(row.get("table")), _cell(row.get("view")))
            title = _public_value(row.get("title"), "table metadata title", public)
            caption = _public_value(row.get("caption"), "table metadata caption", public)
            if not all(key) or not title or not caption or key in records:
                raise ReportBuildError(f"table metadata row {index} is incomplete or duplicated")
            records[key] = (title, caption)
    else:
        if set(rows) != {table for table, _ in required}:
            raise ReportBuildError("table metadata table keys differ from aggregate tables")
        for key in required:
            table, view = key
            base = rows.get(table)
            if not isinstance(base, Mapping) or set(base) - TABLE_METADATA_DETAIL_FIELDS:
                raise ReportBuildError(f"metadata for table {table} has an invalid schema")
            views = base.get("views", {})
            if not isinstance(views, Mapping):
                raise ReportBuildError(f"metadata views for table {table} is not an object")
            detail = views.get(view, {}) if view != "overall" else {}
            if not isinstance(detail, Mapping) or set(detail) - (TABLE_METADATA_DETAIL_FIELDS - {"views"}):
                raise ReportBuildError(f"metadata for table {table} view {view} is invalid")
            merged = {field: base.get(field) for field in TABLE_METADATA_DETAIL_FIELDS - {"views"}}
            merged.update(detail)
            title = _public_value(merged.get("title"), "table metadata title", public)
            caption = _cell(merged.get("caption"))
            if not caption:
                caption = "; ".join(
                    f"{label}: {_cell(merged.get(field))}"
                    for field, label in (
                        ("dataset", "Dataset"), ("ground_truth", "Ground truth"),
                        ("test_impressions", "Test impressions"),
                        ("test_subjects", "Test subjects"), ("seeds", "Seeds"),
                        ("threshold_selection", "Threshold selection"),
                    ) if _cell(merged.get(field))
                )
            caption = _public_value(caption, "table metadata caption", public)
            if not title or not caption:
                raise ReportBuildError(f"metadata for table {table} view {view} is incomplete")
            records[key] = (title, caption)
    if set(records) != required:
        raise ReportBuildError(
            "table metadata keys differ from aggregate table views: "
            f"missing={sorted(required - set(records))}, extra={sorted(set(records) - required)}"
        )
    return records


def load_dataset_summary(path: Path, *, public: bool) -> list[dict[str, Any]]:
    value = _read_json(path, "dataset summary")
    rows = value.get("datasets") if isinstance(value, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ReportBuildError("dataset summary must contain a non-empty datasets list")
    output = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ReportBuildError(f"dataset summary row {index} is not an object")
        extra = set(row) - set(DATASET_FIELDS)
        if extra:
            raise ReportBuildError(f"dataset summary row {index} has unapproved fields: {sorted(extra)}")
        if not _cell(row.get("dataset")) or not _cell(row.get("status")):
            raise ReportBuildError(f"dataset summary row {index} lacks dataset or status")
        clean = {field: _public_value(row.get(field), f"dataset summary {field}", public)
                 for field in DATASET_FIELDS}
        output.append(clean)
    return output


def load_statistics(path: Path, *, public: bool) -> pd.DataFrame:
    frame = _read_csv(path, "aggregate statistics CSV")
    unsafe = set(frame.columns) - set(STATISTIC_FIELDS)
    if unsafe:
        raise ReportBuildError(f"aggregate statistics CSV has unapproved columns: {sorted(unsafe)}")
    required = {"table", "metric", "status", "ci_lower", "ci_upper", "p_holm", "significant"}
    if not required.issubset(frame.columns):
        raise ReportBuildError(f"aggregate statistics CSV lacks columns: {sorted(required - set(frame.columns))}")
    for column in frame.columns:
        if frame[column].dtype == object:
            for row_index, value in frame[column].items():
                _public_value(value, f"statistics {column} row {row_index}", public)
    return frame


def load_reproducibility(path: Path, *, public: bool) -> dict[str, Any]:
    value = _read_json(path, "reproducibility record")
    if not isinstance(value, Mapping):
        raise ReportBuildError("reproducibility record must be an object")
    extra = set(value) - REPRO_FIELDS
    if extra:
        raise ReportBuildError(f"reproducibility record has unapproved fields: {sorted(extra)}")
    commit = value.get("git_commit")
    hashes = value.get("config_hashes")
    seeds = value.get("seeds")
    dates = value.get("dates")
    manifest_hash = value.get("data_manifest_sha256")
    if not isinstance(commit, str) or not _SHA40.fullmatch(commit):
        raise ReportBuildError("reproducibility record has an invalid Git commit")
    if not isinstance(hashes, list) or not hashes or not all(
        isinstance(item, str) and _SHA64.fullmatch(item) for item in hashes
    ):
        raise ReportBuildError("reproducibility record has invalid config hashes")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(item, int) for item in seeds):
        raise ReportBuildError("reproducibility record has invalid seeds")
    if not isinstance(dates, list) or not dates or not all(isinstance(item, str) for item in dates):
        raise ReportBuildError("reproducibility record has invalid dates")
    if not isinstance(manifest_hash, str) or not _SHA64.fullmatch(manifest_hash):
        raise ReportBuildError("reproducibility record has an invalid manifest hash")
    rendered = json.dumps(value, sort_keys=True)
    scan_infrastructure_text(rendered, "reproducibility record")
    if public:
        scan_public_text(rendered, "reproducibility record")
    return dict(value)


def markdown_to_html(markdown: str, *, public: bool) -> str:
    """Render a conservative Markdown subset without raw HTML or hyperlinks."""
    if not markdown.strip():
        raise ReportBuildError("methodology text is empty")
    if re.search(r"<\s*(?:script|iframe|object|embed|img|video|audio)\b", markdown, re.I):
        raise ReportBuildError("methodology contains forbidden raw HTML")
    scan_infrastructure_text(markdown, "methodology")
    if public:
        scan_public_text(markdown, "methodology")
    lines = markdown.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    in_list = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append("<p>" + escape(" ".join(part.strip() for part in paragraph)) + "</p>")
            paragraph = []

    for line in lines:
        stripped = line.strip()
        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            if in_list:
                out.append("</ul>"); in_list = False
            level = min(4, len(heading.group(1)) + 1)
            out.append(f"<h{level}>{escape(heading.group(2))}</h{level}>")
        elif bullet:
            flush_paragraph()
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append("<li>" + escape(bullet.group(1)) + "</li>")
        elif not stripped:
            flush_paragraph()
            if in_list:
                out.append("</ul>"); in_list = False
        elif stripped.startswith("|"):
            # Tables in the source methodology are retained as readable text;
            # result tables are rendered separately from aggregate CSV inputs.
            paragraph.append(stripped)
        else:
            paragraph.append(stripped)
    flush_paragraph()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _format_number(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _cell(value) or "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{decimals}f}"


def _reference(row: pd.Series) -> bool:
    return _cell(row.get("family")).lower() == "reference" or _cell(row.get("method")).lower() in {
        "all_background", "all_foreground", "hull_as_prediction"
    }


def _marks(part: pd.DataFrame, metric: Metric) -> dict[int, str]:
    if metric.direction is None:
        return {}
    candidates = pd.to_numeric(
        part.loc[~part.apply(_reference, axis=1), f"{metric.key}_mean"], errors="coerce"
    ).dropna()
    if candidates.empty:
        return {}
    values = sorted(set(candidates.astype(float)), reverse=metric.direction == "max")
    result: dict[int, str] = {}
    for index, value in candidates.items():
        if math.isclose(float(value), values[0], rel_tol=1e-12, abs_tol=1e-15):
            result[int(index)] = "best"
        elif len(values) > 1 and math.isclose(float(value), values[1], rel_tol=1e-12, abs_tol=1e-15):
            result[int(index)] = "second"
    return result


def render_tables(
    frame: pd.DataFrame,
    metadata: Mapping[tuple[str, str], tuple[str, str]],
) -> str:
    sections = []
    for (table, view), source in frame.groupby(["table", "view"], sort=False, dropna=False):
        part = source.reset_index(drop=True)
        metrics = [metric for metric in METRICS if f"{metric.key}_mean" in part.columns
                   and pd.to_numeric(part[f"{metric.key}_mean"], errors="coerce").notna().any()]
        marks = {metric.key: _marks(part, metric) for metric in metrics}
        grouped = _cell(view).lower() != "overall" and "group_value" in part
        raw_title, raw_caption = metadata[(_cell(table), _cell(view))]
        title = escape(raw_title)
        headers = ["Year", "Approach", "Family", "Training data", "Comment"]
        if grouped:
            headers.append(escape(_cell(part.get("group_by", pd.Series(["Group"])).iloc[0]).replace("_", " ").title()))
        headers += [metric.label for metric in metrics] + ["Params (M)", "ms/img", "Seeds"]
        rows = []
        for index, row in part.iterrows():
            cells = [
                _cell(row.get("year")) or "—", _cell(row.get("display_name")),
                _cell(row.get("family")), _cell(row.get("training_data")) or "—",
                _cell(row.get("comment")) or "—",
            ]
            if grouped:
                cells.append(_cell(row.get("group_value")) or "—")
            rendered = [escape(value) for value in cells]
            for metric in metrics:
                mean = pd.to_numeric(pd.Series([row.get(f"{metric.key}_mean")]), errors="coerce").iloc[0]
                std = pd.to_numeric(pd.Series([row.get(f"{metric.key}_seed_std")]), errors="coerce").iloc[0]
                if pd.isna(mean):
                    value = "—"
                else:
                    value = f"{float(mean) * metric.scale:.{metric.decimals}f}"
                    if int(float(row.get("n_seeds", 1))) > 1 and pd.notna(std):
                        value += f" ± {float(std) * metric.scale:.{metric.decimals}f}"
                mark = marks[metric.key].get(index)
                if mark == "best":
                    value = f"<strong>{escape(value)}</strong>"
                elif mark == "second":
                    value = f"<u>{escape(value)}</u>"
                else:
                    value = escape(value)
                rendered.append(value)
            rendered += [
                escape(_format_number(row.get("params_m"), 3)),
                escape(_format_number(row.get("ms_per_img"), 1)),
                escape(_cell(row.get("n_seeds")) or "—"),
            ]
            row_class = " class=\"reference\"" if _reference(row) else ""
            rows.append(f"<tr{row_class}>" + "".join(f"<td>{item}</td>" for item in rendered) + "</tr>")
        sections.append(
            f"<section><h2>{title}</h2><p>{escape(raw_caption)}</p><div class=\"table-wrap\"><table><thead><tr>"
            + "".join(f"<th>{escape(item)}</th>" for item in headers)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>"
        )
    return "\n".join(sections)


def render_dataset_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [field.replace("_", " ").title() for field in DATASET_FIELDS]
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(_cell(row.get(field)) or '—')}</td>"
                                      for field in DATASET_FIELDS) + "</tr>")
    return (
        "<section><h2>Datasets and splits</h2><div class=\"table-wrap\"><table><thead><tr>"
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div></section>"
    )


def render_statistics(frame: pd.DataFrame) -> str:
    columns = [column for column in STATISTIC_FIELDS if column in frame]
    headers = [column.replace("_", " ").title() for column in columns]
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in {"estimate", "ci_lower", "ci_upper", "p_value", "p_holm"}:
                values.append(_format_number(value, 4))
            else:
                values.append(_cell(value) or "—")
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>")
    return (
        "<section><h2>Statistical analysis</h2><div class=\"table-wrap\"><table><thead><tr>"
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>"
    )


def _svg_bar_chart(
    frame: pd.DataFrame, metric_key: str, title: str, *, overall_only: bool = True
) -> str:
    column = f"{metric_key}_mean"
    if column not in frame:
        return ""
    view_mask = frame["view"].astype(str) == "overall" if overall_only else pd.Series(
        True, index=frame.index
    )
    work = frame[view_mask & ~frame.apply(_reference, axis=1)].copy()
    work[metric_key] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[metric_key]).sort_values(metric_key, ascending=True).tail(20)
    if work.empty:
        return ""
    width, row_height, left, right = 920, 30, 260, 70
    height = 80 + row_height * len(work)
    usable = width - left - right
    std_column = f"{metric_key}_seed_std"
    std_values = pd.to_numeric(work.get(std_column, pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0)
    maximum = max(float((work[metric_key] + std_values).max()), 1e-9)
    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<text x="{left}" y="28" class="svg-title">{escape(title)}</text>',
    ]
    for offset, (_, row) in enumerate(work.iterrows()):
        y = 50 + offset * row_height
        value = float(row[metric_key])
        bar = usable * value / maximum
        label = _cell(row.get("display_name"))
        if not overall_only and _cell(row.get("group_value")):
            label += " — " + _cell(row.get("group_value"))
        elements += [
            f'<text x="{left - 12}" y="{y + 17}" text-anchor="end">{escape(label)}</text>',
            f'<rect x="{left}" y="{y + 3}" width="{bar:.2f}" height="18" rx="4" />',
            f'<text x="{min(left + bar + 8, width - 48):.2f}" y="{y + 17}">{value:.3f}</text>',
        ]
        std = float(std_values.loc[row.name])
        if std > 0:
            low = left + usable * max(0.0, value - std) / maximum
            high = left + usable * (value + std) / maximum
            elements += [
                f'<line x1="{low:.2f}" y1="{y + 12}" x2="{high:.2f}" y2="{y + 12}" />',
                f'<line x1="{low:.2f}" y1="{y + 8}" x2="{low:.2f}" y2="{y + 16}" />',
                f'<line x1="{high:.2f}" y1="{y + 8}" x2="{high:.2f}" y2="{y + 16}" />',
            ]
    elements.append("</svg>")
    return '<div class="chart">' + "".join(elements) + "</div>"


def _svg_scatter(frame: pd.DataFrame) -> str:
    if not {"dice_mean", "ms_per_img"}.issubset(frame.columns):
        return ""
    work = frame[(frame["view"].astype(str) == "overall") & ~frame.apply(_reference, axis=1)].copy()
    work["x"] = pd.to_numeric(work.ms_per_img, errors="coerce")
    work["y"] = pd.to_numeric(work.dice_mean, errors="coerce")
    work = work.dropna(subset=["x", "y"])
    if work.empty:
        return ""
    width, height, left, top, usable_x, usable_y = 920, 440, 80, 50, 790, 320
    xmin, xmax = float(work.x.min()), float(work.x.max())
    ymin, ymax = float(work.y.min()), float(work.y.max())
    if math.isclose(xmin, xmax): xmax = xmin + 1.0
    if math.isclose(ymin, ymax): ymax = ymin + 0.01
    items = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Dice and inference time">',
        '<text x="80" y="28" class="svg-title">Dice and inference time</text>',
        f'<line x1="{left}" y1="{top + usable_y}" x2="{left + usable_x}" y2="{top + usable_y}" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + usable_y}" />',
    ]
    for _, row in work.iterrows():
        x = left + usable_x * (float(row.x) - xmin) / (xmax - xmin)
        y = top + usable_y * (1 - (float(row.y) - ymin) / (ymax - ymin))
        label = escape(_cell(row.get("display_name")))
        items += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" />',
                  f'<text x="{x + 9:.2f}" y="{y - 8:.2f}">{label}</text>']
    items += [
        f'<text x="{left + usable_x / 2}" y="{height - 22}" text-anchor="middle">ms per image</text>',
        f'<text x="18" y="{top + usable_y / 2}" transform="rotate(-90 18 {top + usable_y / 2})" text-anchor="middle">Dice</text>',
        "</svg>",
    ]
    return '<div class="chart">' + "".join(items) + "</div>"


def render_charts(frame: pd.DataFrame) -> str:
    charts = [
        _svg_bar_chart(frame, "dice", "Dice by method"),
        _svg_scatter(frame),
        _svg_bar_chart(
            frame[frame["view"].astype(str) != "overall"], "dice",
            "Dice by examiner-value row", overall_only=False,
        ),
        _svg_bar_chart(frame, "hr", "Hull recall by method"),
    ]
    charts = [chart for chart in charts if chart]
    if not charts:
        raise ReportBuildError("aggregate tables cannot produce any required chart")
    return "<section><h2>Aggregate figures</h2>" + "".join(charts) + "</section>"


def _data_uri(path: Path, mime: str) -> str:
    payload = _regular_file(path, "gallery image").read_bytes()
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def render_synthetic_gallery(receipt_path: Path, root: Path, limit: int) -> str:
    if limit < 1 or limit > 24:
        raise ReportBuildError("synthetic example limit must be between 1 and 24")
    try:
        assets = load_synthetic_receipt(receipt_path, root)
    except PublicReportError as exc:
        raise ReportBuildError(str(exc)) from exc
    selected = [(relative, item) for relative, item in sorted(assets.items())
                if item["role"] == "example_image"][:limit]
    if not selected:
        raise ReportBuildError("synthetic receipt has no example images")
    cards = []
    for index, (_, item) in enumerate(selected, start=1):
        path = Path(item["absolute_path"])
        cards.append(
            '<figure><img alt="Synthetic fingerprint-like benchmark sample" '
            'data-provenance="synthetic" '
            f'data-asset-sha256="{item["sha256"]}" '
            f'src="{_data_uri(path, item["mime_type"])}">'
            f'<figcaption>Synthetic example {index}</figcaption></figure>'
        )
    return "<section><h2>Synthetic examples</h2><p>No biometric sample is shown.</p><div class=\"gallery\">" + "".join(cards) + "</div></section>"


def load_private_gallery(path: Path) -> dict[str, Any]:
    value = _read_json(path, "private gallery manifest")
    if not isinstance(value, Mapping) or value.get("schema_version") != 1 \
            or value.get("kind") != "private_report_galleries":
        raise ReportBuildError("private gallery manifest schema or kind is invalid")
    sections = value.get("sections")
    if not isinstance(sections, list):
        raise ReportBuildError("private gallery manifest has no sections")
    kinds = {section.get("kind") for section in sections if isinstance(section, Mapping)}
    if kinds != PRIVATE_GALLERY_KINDS:
        raise ReportBuildError("private gallery must contain qualitative_grid and failure_gallery")
    for section in sections:
        if section["kind"] == "qualitative_grid" and section.get("source_examples") != 8:
            raise ReportBuildError("qualitative grid must declare exactly eight source examples")
        if section["kind"] == "failure_gallery" and (
            not isinstance(section.get("failures_per_method"), int)
            or section["failures_per_method"] < 3
        ):
            raise ReportBuildError("failure gallery must declare at least three failures per method")
        if not isinstance(section.get("method_count"), int) or section["method_count"] < 1:
            raise ReportBuildError("private gallery section lacks a positive method count")
    return dict(value)


def render_private_gallery(value: Mapping[str, Any]) -> str:
    rendered = []
    for section in value["sections"]:
        title, items = _cell(section.get("title")), section.get("items")
        if not title or not isinstance(items, list) or not items:
            raise ReportBuildError("private gallery section is incomplete")
        scan_infrastructure_text(title, "private gallery title")
        cards = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ReportBuildError("private gallery item is not an object")
            path = Path(str(item.get("path", "")))
            mime = item.get("mime_type")
            if mime not in {"image/png", "image/jpeg", "image/webp"}:
                raise ReportBuildError("private gallery image has an invalid MIME type")
            alt = _cell(item.get("alt")) or "Private qualitative evidence"
            caption = _cell(item.get("caption")) or "Evidence image"
            scan_infrastructure_text(alt, "private gallery alt text")
            scan_infrastructure_text(caption, "private gallery caption")
            cards.append(
                f'<figure><img alt="{escape(alt)}" '
                f'src="{_data_uri(path, str(mime))}"><figcaption>'
                f'{escape(caption)}</figcaption></figure>'
            )
        rendered.append(f"<section><h2>{escape(title)}</h2><div class=\"gallery\">" + "".join(cards) + "</div></section>")
    return "".join(rendered)


def load_private_audit(path: Path) -> tuple[str, pd.DataFrame]:
    value = _read_json(path, "private audit-table manifest")
    if not isinstance(value, Mapping) or value.get("schema_version") != 1 \
            or value.get("kind") != "private_low_roi_quality_table":
        raise ReportBuildError("private audit-table manifest schema or kind is invalid")
    title = _cell(value.get("title"))
    columns = value.get("columns")
    csv_path = value.get("csv")
    if not title or not isinstance(columns, list) or not columns or not isinstance(csv_path, str):
        raise ReportBuildError("private audit-table manifest is incomplete")
    frame = _read_csv(Path(csv_path), "private low-agreement CSV")
    if any(not isinstance(column, str) or column not in frame for column in columns):
        raise ReportBuildError("private audit-table manifest names an absent column")
    if len(frame) < 20:
        raise ReportBuildError("private low-agreement table must contain at least 20 rows")
    return title, frame.loc[:, columns]


def render_private_audit(title: str, frame: pd.DataFrame) -> str:
    scan_infrastructure_text(title, "private audit title")
    for column in frame.columns:
        for value in frame[column]:
            scan_infrastructure_text(_cell(value), f"private audit {column}")
    headers = [column.replace("_", " ").title() for column in frame.columns]
    rows = ["<tr>" + "".join(f"<td>{escape(_cell(value) or '—')}</td>" for value in row) + "</tr>"
            for row in frame.itertuples(index=False, name=None)]
    return (
        f"<section><h2>{escape(title)}</h2><div class=\"table-wrap\"><table><thead><tr>"
        + "".join(f"<th>{escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>"
    )


def render_reproducibility(value: Mapping[str, Any]) -> str:
    hashes = "<br>".join(f"<code>{escape(item)}</code>" for item in value["config_hashes"])
    return (
        "<section><h2>Reproducibility</h2><dl>"
        f"<dt>Git commit</dt><dd><code>{escape(value['git_commit'])}</code></dd>"
        f"<dt>Data manifest hash</dt><dd><code>{escape(value['data_manifest_sha256'])}</code></dd>"
        f"<dt>Config hashes</dt><dd>{hashes}</dd>"
        f"<dt>Seeds</dt><dd>{escape(', '.join(str(item) for item in value['seeds']))}</dd>"
        f"<dt>Dates</dt><dd>{escape(', '.join(value['dates']))}</dd></dl></section>"
    )


CSS = """
:root{--ink:#17202a;--muted:#64748b;--line:#dbe3eb;--surface:#f7f9fb;--accent:#285a75}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1240px;margin:auto;padding:44px 28px 72px}header{border-bottom:1px solid var(--line);padding-bottom:24px}
h1{font-size:34px;letter-spacing:-.03em;margin:0 0 8px}h2{font-size:22px;margin:38px 0 12px}h3{font-size:17px}
p{max-width:86ch;color:#334155}.eyebrow{color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing:.08em;font-size:12px}
.note{padding:14px 16px;border-left:3px solid var(--accent);background:var(--surface)}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{background:var(--surface);position:sticky;top:0}
th:nth-child(-n+5),td:nth-child(-n+5){text-align:left}.reference{color:var(--muted);background:#fafafa}
.chart{border:1px solid var(--line);border-radius:10px;margin:16px 0;padding:10px;overflow:auto}.chart svg{width:100%;min-width:720px}
svg rect,svg circle{fill:var(--accent)}svg line{stroke:#94a3b8;stroke-width:1}svg text{fill:var(--ink);font:12px system-ui,sans-serif}.svg-title{font-size:16px;font-weight:700}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}.gallery figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--surface)}
.gallery img{display:block;width:100%;height:auto}.gallery figcaption{padding:9px 11px;color:var(--muted)}
dl{display:grid;grid-template-columns:180px 1fr;gap:8px 18px}dt{font-weight:700}dd{margin:0;overflow-wrap:anywhere}code{font:12px ui-monospace,monospace}
@media(max-width:700px){main{padding:28px 14px}h1{font-size:28px}dl{grid-template-columns:1fr}th,td{padding:8px}}
"""


def build_html(args: argparse.Namespace) -> str:
    public = bool(args.public)
    tables = load_tables(args.tables_csv, public=public)
    table_metadata = load_table_metadata(args.table_metadata_json, tables, public=public)
    datasets = load_dataset_summary(args.dataset_summary_json, public=public)
    statistics = load_statistics(args.statistics_csv, public=public)
    reproducibility = load_reproducibility(args.reproducibility_json, public=public)
    methodology_path = _regular_file(args.methodology, "methodology")
    methodology = markdown_to_html(methodology_path.read_text(encoding="utf-8"), public=public)
    mode_note = (
        "This public-safe page contains aggregate results and synthetic examples only."
        if public else
        "This private report contains restricted qualitative evidence and must remain on the approved network."
    )
    evidence = ""
    if public:
        if args.private_gallery or args.private_audit:
            raise ReportBuildError("public mode rejects every private evidence input")
        if not args.synthetic_receipt or not args.synthetic_root:
            raise ReportBuildError("public mode requires a synthetic root and receipt")
        evidence = render_synthetic_gallery(args.synthetic_receipt, args.synthetic_root, args.synthetic_limit)
    else:
        if args.synthetic_receipt or args.synthetic_root:
            raise ReportBuildError("private mode does not accept the public synthetic input flags")
        if not args.private_gallery or not args.private_audit:
            raise ReportBuildError("private mode requires galleries and the low-agreement audit table")
        evidence = render_private_gallery(load_private_gallery(args.private_gallery))
        audit_title, audit_frame = load_private_audit(args.private_audit)
        evidence += render_private_audit(audit_title, audit_frame)
    marker = (
        '<meta name="segbench-public-safe" content="1">'
        if public else '<meta name="segbench-private-report" content="1">'
    )
    robots = '<meta name="robots" content="noindex,nofollow">'
    title = "Segmentation benchmark — public-safe results" if public else "Segmentation benchmark — private results"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{robots}{marker}
<title>{escape(title)}</title><style>{CSS}</style></head><body><main>
<header><div class="eyebrow">Segmentation benchmark</div><h1>{escape(title)}</h1>
<p class="note">{escape(mode_note)}</p></header>
{render_dataset_summary(datasets)}
<section><h2>Methodology</h2>{methodology}</section>
{render_tables(tables, table_metadata)}
{render_charts(tables)}
{render_statistics(statistics)}
{evidence}
{render_reproducibility(reproducibility)}
</main></body></html>"""
    if public:
        # Catch unsafe semantic text before any file is published.
        scrubbed = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", "DATA_IMAGE", html)
        scan_public_text(scrubbed, "assembled public report")
    return html


def publish_once(out_dir: Path, html: str) -> tuple[Path, str]:
    """Atomically publish a write-once self-contained report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise ReportBuildError(f"output is not a regular directory: {out_dir}")
    index = out_dir / "index.html"
    encoded = html.encode("utf-8")
    existing = list(out_dir.iterdir())
    if existing:
        if existing == [index] and index.is_file() and not index.is_symlink() and index.read_bytes() == encoded:
            return index, "verified_identical"
        raise ReportBuildError("refusing to replace a non-empty or non-identical report directory")
    descriptor, temporary = tempfile.mkstemp(prefix=".index.", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, index)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return index, "created"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-csv", type=Path, required=True)
    parser.add_argument("--table-metadata-json", type=Path, required=True)
    parser.add_argument("--dataset-summary-json", type=Path, required=True)
    parser.add_argument("--statistics-csv", type=Path, required=True)
    parser.add_argument("--methodology", type=Path, required=True)
    parser.add_argument("--reproducibility-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--synthetic-root", type=Path)
    parser.add_argument("--synthetic-receipt", type=Path)
    parser.add_argument("--synthetic-limit", type=int, default=8)
    parser.add_argument("--private-gallery", type=Path)
    parser.add_argument("--private-audit", type=Path)
    parser.add_argument("--scan-report", type=Path,
                        help="public mode only: write the passing scan record outside the report directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.scan_report and not args.public:
        raise ReportBuildError("--scan-report is valid only in public mode")
    if args.scan_report:
        try:
            args.scan_report.resolve().relative_to(args.out.resolve())
        except ValueError:
            pass
        else:
            raise ReportBuildError("scan report must be outside the self-contained report directory")
    html = build_html(args)
    index, action = publish_once(args.out, html)
    result: dict[str, Any] = {
        "status": "PASS", "action": action, "index": str(index),
        "sha256": sha256_file(index), "bytes": index.stat().st_size,
        "mode": "public" if args.public else "private",
    }
    if args.public:
        scan = scan_public_html(
            index,
            synthetic_receipt=args.synthetic_receipt,
            synthetic_root=args.synthetic_root,
            generator_path=Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic_demo.py",
        )
        result["privacy_scan"] = scan
        if args.scan_report:
            if args.scan_report.exists():
                raise ReportBuildError("refusing to replace an existing scan report")
            args.scan_report.parent.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(scan, indent=2, sort_keys=True) + "\n").encode("utf-8")
            descriptor = os.open(args.scan_report, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
