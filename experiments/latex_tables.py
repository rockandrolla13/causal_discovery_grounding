"""LaTeX rendering of metric-summary and generic result tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def _latex_escape(value: Any) -> str:
    """Return a conservative LaTeX-safe representation of a scalar value."""

    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _numeric_metric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric metric column, treating booleans as 0/1 values."""

    if column not in frame.columns:
        raise ValueError(f"Missing required metric column: {column}")
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def format_latex_mean_sd(
    mean: float,
    sd: float,
    *,
    digits: int = 3,
) -> str:
    """
    Format a mean and standard deviation for a LaTeX table cell.

    Args:
        mean: Mean value to display.
        sd: Standard deviation value to display. Missing values are displayed as
            zero, which is the natural single-row group convention.
        digits: Number of decimal places.

    Returns:
        A math-mode string of the form ``$mean \\pm sd$``.
    """

    if not np.isfinite(mean):
        return ""
    display_sd = 0.0 if not np.isfinite(sd) else float(sd)
    return f"${float(mean):.{digits}f} \\pm {display_sd:.{digits}f}$"


def make_metric_summary_latex_table(
    run_summary: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    coverage_col: str = "rct_conformal_point_coverage",
    width_col: str = "conformal_weighted_width",
    digits: int = 3,
    column_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a formatted coverage/width summary table from run-summary rows.

    Args:
        run_summary: Frame containing one row per experiment run or target.
        group_columns: Columns defining rows of the output table.
        coverage_col: Numeric or boolean coverage column to summarize.
        width_col: Numeric interval-width column to summarize.
        digits: Number of decimal places in ``mean \\pm sd`` cells.
        column_labels: Optional display labels for grouping and metric columns.

    Returns:
        A display-ready DataFrame with group labels, ``N``, coverage
        ``mean \\pm sd``, and width ``mean \\pm sd``.

    Raises:
        ValueError: If the requested grouping or metric columns are missing.
    """

    if run_summary.empty:
        return pd.DataFrame()
    missing_groups = [col for col in group_columns if col not in run_summary.columns]
    if missing_groups:
        raise ValueError(f"Missing required grouping columns: {missing_groups}")

    working = run_summary.copy()
    working["_coverage_metric"] = _numeric_metric(working, coverage_col)
    working["_width_metric"] = _numeric_metric(working, width_col)
    working = working.dropna(subset=["_coverage_metric", "_width_metric"])
    if working.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for group_key, group in working.groupby(list(group_columns), dropna=False, sort=True):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        record = {
            column: _latex_escape(value)
            for column, value in zip(group_columns, key_tuple)
        }
        coverage = group["_coverage_metric"].astype(float)
        width = group["_width_metric"].astype(float)
        record["n"] = int(len(group))
        record["coverage"] = format_latex_mean_sd(
            float(coverage.mean()),
            float(coverage.std(ddof=1)) if len(coverage) > 1 else 0.0,
            digits=digits,
        )
        record["width"] = format_latex_mean_sd(
            float(width.mean()),
            float(width.std(ddof=1)) if len(width) > 1 else 0.0,
            digits=digits,
        )
        records.append(record)

    table = pd.DataFrame(records)
    labels = {
        "study": "Experiment",
        "pattern": "Confounding variable",
        "beta": "Beta",
        "target_site": "Target",
        "stage1_backend": "Backend",
        "lp_probability_backend": "LP backend",
        "n": "N",
        "coverage": "Coverage",
        "width": "Width",
    }
    if column_labels:
        labels.update(column_labels)
    return table.rename(columns={col: labels.get(col, col) for col in table.columns})


def make_multi_metric_summary_latex_table(
    run_summary: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
    digits: int = 3,
    column_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build a formatted grouped table for several numeric summary metrics.

    Args:
        run_summary: Frame containing one row per plotted stratum or run.
        group_columns: Columns defining rows of the output table.
        metrics: Pairs of ``(display_label, source_column)`` to summarize.
        digits: Number of decimal places in ``mean \\pm sd`` cells.
        column_labels: Optional display labels for grouping columns.

    Returns:
        A display-ready DataFrame with group labels, ``N``, and one
        ``mean \\pm sd`` column per requested metric.

    Raises:
        ValueError: If grouping columns or requested metric columns are missing.
    """

    if run_summary.empty:
        return pd.DataFrame()
    missing_groups = [col for col in group_columns if col not in run_summary.columns]
    if missing_groups:
        raise ValueError(f"Missing required grouping columns: {missing_groups}")
    missing_metrics = [column for _, column in metrics if column not in run_summary.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metric columns: {missing_metrics}")

    working = run_summary.copy()
    metric_work_columns: List[Tuple[str, str]] = []
    for index, (label, column) in enumerate(metrics):
        work_column = f"_metric_{index}"
        working[work_column] = _numeric_metric(working, column)
        metric_work_columns.append((label, work_column))

    records: List[Dict[str, Any]] = []
    for group_key, group in working.groupby(list(group_columns), dropna=False, sort=True):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        record = {
            column: _latex_escape(value)
            for column, value in zip(group_columns, key_tuple)
        }
        record["n"] = int(len(group))
        for label, work_column in metric_work_columns:
            values = group[work_column].dropna().astype(float)
            record[label] = format_latex_mean_sd(
                float(values.mean()) if len(values) else float("nan"),
                float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                digits=digits,
            )
        records.append(record)

    table = pd.DataFrame(records)
    labels = {
        "study": "Experiment",
        "pattern": "Confounding variable",
        "beta": "Beta",
        "target_site": "Target",
        "stage1_backend": "Backend",
        "lp_probability_backend": "LP backend",
        "n": "N",
    }
    if column_labels:
        labels.update(column_labels)
    return table.rename(columns={col: labels.get(col, col) for col in table.columns})


def render_latex_table(
    table: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    index: bool = False,
    column_format: str | None = None,
    midrule_before: Sequence[int] = (),
) -> str:
    """
    Render a display-ready DataFrame as a simple booktabs LaTeX table.

    Args:
        table: DataFrame whose cells and column labels are already formatted for
            display. Cell contents are emitted as-is so math-mode summaries such
            as ``$0.90 \\pm 0.10$`` remain intact.
        caption: Optional LaTeX caption. When provided, the tabular block is
            wrapped in a table environment.
        label: Optional LaTeX label. When provided, the tabular block is wrapped
            in a table environment.
        index: Whether to include the DataFrame index as the first column.
        column_format: Tabular column specification such as ``"llcc"``;
            defaults to one left-aligned column per displayed column.
        midrule_before: Zero-based body row positions preceded by a
            ``\\midrule``, which separates grouped blocks of rows.

    Returns:
        LaTeX source for the table without relying on Pandas Styler or Jinja2.
    """

    display = table.reset_index() if index else table
    if display.empty:
        return ""

    def _latex_cell(value: Any) -> str:
        """Return a raw LaTeX cell value, preserving preformatted math."""

        return "" if pd.isna(value) else str(value)

    lines: list[str] = []
    wrap_table = caption is not None or label is not None
    if wrap_table:
        lines.append(r"\begin{table}")
        if caption is not None:
            lines.append(f"\\caption{{{caption}}}")
        if label is not None:
            lines.append(f"\\label{{{label}}}")

    if column_format is None:
        column_format = "l" * len(display.columns)
    breaks = set(midrule_before)
    lines.extend(
        [
            f"\\begin{{tabular}}{{{column_format}}}",
            r"\toprule",
            " & ".join(str(column) for column in display.columns) + r" \\",
            r"\midrule",
        ]
    )
    for position, row in enumerate(display.itertuples(index=False, name=None)):
        if position in breaks and position > 0:
            lines.append(r"\midrule")
        lines.append(" & ".join(_latex_cell(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    if wrap_table:
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_metric_summary_latex_table(
    run_summary: pd.DataFrame,
    output_path: str | Path,
    *,
    group_columns: Sequence[str],
    coverage_col: str = "rct_conformal_point_coverage",
    width_col: str = "conformal_weighted_width",
    digits: int = 3,
    column_labels: Mapping[str, str] | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """
    Write a LaTeX coverage/width summary table and return its display frame.

    Args:
        run_summary: Frame containing run-level coverage and width metrics.
        output_path: Destination ``.tex`` path.
        group_columns: Columns defining rows of the output table.
        coverage_col: Numeric or boolean coverage column to summarize.
        width_col: Numeric interval-width column to summarize.
        digits: Number of decimal places in ``mean \\pm sd`` cells.
        column_labels: Optional display labels for output columns.
        caption: Optional LaTeX table caption.
        label: Optional LaTeX table label.

    Returns:
        The display-ready summary table that was written. If no complete metric
        rows exist, an empty file is written and an empty frame is returned.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = make_metric_summary_latex_table(
        run_summary,
        group_columns=group_columns,
        coverage_col=coverage_col,
        width_col=width_col,
        digits=digits,
        column_labels=column_labels,
    )
    if table.empty:
        output.write_text("", encoding="utf-8")
        return table
    output.write_text(
        render_latex_table(
            table,
            index=False,
            caption=caption,
            label=label,
        ),
        encoding="utf-8",
    )
    return table

