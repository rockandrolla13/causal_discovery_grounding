"""Plotting utilities for naturalness calibration experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from causal_grounding.loaders.schema import MANYLABS_COVARIATES
from experiments.backdoor_cate_error_analysis import (
    write_backdoor_cate_error_analysis_plots,
)
from experiments.utils import (
    USEFUL_COVARIATES,
    make_multi_metric_summary_latex_table,
    render_latex_table,
)


DEFAULT_EXP16_STUDIES = ("allowedforbidden", "gainloss", "reciprocity")


EXP16_PLOT_SUMMARY_METRICS = (
    ("Conformalized LP coverage", "conformalized_lp_coverage"),
    ("Conformalized LP width", "conformalized_lp_width"),
    ("Posterior mean coverage", "posterior_mean_coverage"),
    ("Posterior mean width", "posterior_mean_width"),
    ("Posterior envelope coverage", "posterior_envelope_coverage"),
    ("Posterior envelope width", "posterior_envelope_width"),
)

EXP16_PLOT_COVERAGE_SUMMARY_METRICS = (
    ("Conformalized LP", "conformalized_lp_coverage"),
    ("Posterior mean", "posterior_mean_coverage"),
    ("Posterior envelope", "posterior_envelope_coverage"),
)

EXP16_PLOT_WIDTH_SUMMARY_METRICS = (
    ("Conformalized LP", "conformalized_lp_width"),
    ("Posterior mean", "posterior_mean_width"),
    ("Posterior envelope", "posterior_envelope_width"),
)

EXP16_PLOT_SUMMARY_KEYS = (
    "stage1_backend",
    "study",
    "pattern",
    "beta",
    "target_site",
    *MANYLABS_COVARIATES,
    "z_value_str",
)


def make_method_metric_by_beta_plot(
    frame: pd.DataFrame,
    output_stem: Path,
    *,
    value_column: str,
    y_label: str,
    method_order: Sequence[str],
    method_labels: Mapping[str, str],
    method_colors: Mapping[str, str],
    method_linestyles: Mapping[str, Any] | None = None,
    emphasized_method: str | None = None,
    y_limits: tuple[float, float] | None = None,
    y_ticks: Sequence[float] | None = None,
    confidence_level: float | None = None,
) -> list[Path]:
    """Plot a method-level mean metric against assignment strength.

    The input remains at the condition level so callers can preserve the
    estimand and weighting appropriate to their experiment. This function only
    averages those condition-level values within ``(method, beta)`` and applies
    the common publication styling used by Naturalness comparison figures.

    Args:
        frame: Long condition-level table containing ``method``, ``beta``, and
            ``value_column``.
        output_stem: Destination without a suffix. Both PDF and PNG are written.
        value_column: Numeric metric to average and draw on the vertical axis.
        y_label: Vertical-axis label.
        method_order: Drawing and legend order. Missing methods are skipped.
        method_labels: Display labels keyed by method.
        method_colors: Line colours keyed by method.
        method_linestyles: Optional Matplotlib line styles keyed by method.
        emphasized_method: Optional method drawn with a thicker line and larger
            markers.
        y_limits: Optional fixed vertical-axis limits.
        y_ticks: Optional explicit vertical-axis ticks.
        confidence_level: When given, shade a Student-t confidence interval of
            that level around each mean, and note the level and the per-method
            condition counts beneath the axis. Left out, only the means are
            drawn, which is what a caller wants when the condition count is
            already fixed by the design and stated elsewhere.

    Returns:
        Paths to the vector PDF and 300-dpi PNG, in that order. An empty input
        returns an empty list without writing files.

    Raises:
        ValueError: If the input lacks a required column or contains no finite
            beta/metric pairs for the requested methods.
    """

    if frame.empty:
        return []
    required = {"method", "beta", value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Method-by-beta plot is missing columns: {sorted(missing)}")
    selected = frame.loc[frame["method"].isin(method_order)].copy()
    selected["beta"] = pd.to_numeric(selected["beta"], errors="coerce")
    selected[value_column] = pd.to_numeric(selected[value_column], errors="coerce")
    selected = selected.loc[
        np.isfinite(selected["beta"]) & np.isfinite(selected[value_column])
    ]
    if selected.empty:
        raise ValueError("Method-by-beta plot has no finite values to draw")
    # Student-t rather than normal: these means rest on of order ten to a few
    # dozen conditions, where the normal quantile is visibly too narrow.
    means = selected.groupby(["method", "beta"], observed=True, as_index=False).agg(
        **{value_column: (value_column, "mean"),
           "_n": (value_column, "size"),
           "_sd": (value_column, "std")})
    if confidence_level is not None:
        degrees = means["_n"] - 1.0
        usable = (degrees > 0) & means["_sd"].notna()
        half = pd.Series(np.nan, index=means.index, dtype=float)
        if usable.any():
            half.loc[usable] = (
                stats.t.ppf(0.5 + confidence_level / 2.0,
                            degrees.loc[usable].to_numpy(dtype=float))
                * means.loc[usable, "_sd"].to_numpy(dtype=float)
                / np.sqrt(means.loc[usable, "_n"].to_numpy(dtype=float)))
        means["_ci_low"] = means[value_column] - half
        means["_ci_high"] = means[value_column] + half
    beta_values = sorted(means["beta"].unique())
    line_styles = method_linestyles or {}

    style = {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(8.8, 4.2))
        for method in method_order:
            rows = means.loc[means["method"].eq(method)].sort_values("beta")
            if rows.empty:
                continue
            emphasized = method == emphasized_method
            if confidence_level is not None:
                band = rows.loc[rows["_ci_low"].notna() & rows["_ci_high"].notna()]
                if not band.empty:
                    axis.fill_between(
                        band["beta"], band["_ci_low"], band["_ci_high"],
                        color=method_colors.get(method, "black"),
                        alpha=0.15, linewidth=0, zorder=1,
                    )
            axis.plot(
                rows["beta"],
                rows[value_column],
                label=method_labels.get(method, method),
                color=method_colors.get(method, "black"),
                linestyle=line_styles.get(method, "-"),
                linewidth=2.6 if emphasized else 1.5,
                marker="o",
                markersize=5 if emphasized else 3.5,
                zorder=4 if emphasized else 3,
            )

        axis.set_ylabel(y_label)
        x_label = r"Selection-strength parameter $\beta$"
        if confidence_level is not None:
            # Range over every drawn point, not per method: the condition count
            # varies with beta as well as with method once a run has a deeper
            # arm than its comparators, and a single number would misstate it.
            counts = means["_n"]
            spread = (f"{int(counts.min())}" if counts.min() == counts.max()
                      else f"{int(counts.min())}-{int(counts.max())}")
            x_label += (f"\nshaded: {confidence_level:.0%} CI of the mean over "
                        f"{spread} conditions per point")
        axis.set_xlabel(x_label)
        axis.set_xticks(beta_values, [f"{beta:g}" for beta in beta_values])
        right_padding = max(0.04, 0.02 * max(beta_values))
        axis.set_xlim(0, max(beta_values) + right_padding)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        if y_ticks is not None:
            axis.set_yticks(y_ticks)
        axis.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            frameon=False,
            ncol=1,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
        figure.tight_layout()

        output_stem = Path(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        pdf_path = output_stem.with_suffix(".pdf")
        png_path = output_stem.with_suffix(".png")
        figure.savefig(pdf_path, bbox_inches="tight")
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return [pdf_path, png_path]

def _optimal_conformalized_lp_mask(bounds: pd.DataFrame) -> pd.Series:
    """
    Identify conformalized LP rows with feasible base LP intervals.

    Args:
        bounds: Target LP bounds with conformalized endpoints and feasibility
            diagnostics.

    Returns:
        Boolean mask selecting rows whose conformalized interval comes from an
        optimal base LP solve rather than the full-range fallback.
    """

    if "conformal_status" in bounds.columns:
        return bounds["conformal_status"].eq("optimal_expanded")
    if "base_interval_available" in bounds.columns:
        return bounds["base_interval_available"].astype(bool)
    if {"status", "lower", "upper"}.issubset(bounds.columns):
        return (
            bounds["status"].eq("optimal")
            & bounds["lower"].notna()
            & bounds["upper"].notna()
        )
    return pd.Series(True, index=bounds.index)

def build_exp16_plot_metric_summary_frame(
    results: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine Exp16 interval metrics at the same stratum grain used by plots.

    Args:
        results: Experiment result frames keyed like ``run_experiment`` output.
            The summary uses ``target_lp_bounds`` for conformalized LP intervals
            and ``target_gp_posterior_summary`` for posterior mean and posterior
            envelope intervals.

    Returns:
        A stratum-level frame containing coverage and width columns for the
        three plotted interval families. Missing source frames return an empty
        DataFrame or a frame with only the available metric columns.
    """

    metric_frames: list[pd.DataFrame] = []
    bounds = results.get("target_lp_bounds", pd.DataFrame()).copy()
    if not bounds.empty:
        if "conformal_width" not in bounds.columns and {
            "conformal_lower",
            "conformal_upper",
        }.issubset(bounds.columns):
            bounds["conformal_width"] = (
                pd.to_numeric(bounds["conformal_upper"], errors="coerce")
                - pd.to_numeric(bounds["conformal_lower"], errors="coerce")
            )
        if "covers_rct_cate_conformal" not in bounds.columns and {
            "conformal_lower",
            "conformal_upper",
            "rct_cate",
        }.issubset(bounds.columns):
            bounds["covers_rct_cate_conformal"] = (
                pd.to_numeric(bounds["rct_cate"], errors="coerce")
                >= pd.to_numeric(bounds["conformal_lower"], errors="coerce")
            ) & (
                pd.to_numeric(bounds["rct_cate"], errors="coerce")
                <= pd.to_numeric(bounds["conformal_upper"], errors="coerce")
            )
        if {
            "covers_rct_cate_conformal",
            "conformal_width",
        }.issubset(bounds.columns):
            bounds = bounds.loc[_optimal_conformalized_lp_mask(bounds)].copy()
            if not bounds.empty:
                columns = [
                    column
                    for column in EXP16_PLOT_SUMMARY_KEYS
                    if column in bounds.columns
                ]
                columns.extend(["covers_rct_cate_conformal", "conformal_width"])
                metric_frames.append(
                    bounds.loc[:, columns].rename(
                        columns={
                            "covers_rct_cate_conformal": (
                                "conformalized_lp_coverage"
                            ),
                            "conformal_width": "conformalized_lp_width",
                        }
                    )
                )

    posterior = results.get("target_gp_posterior_summary", pd.DataFrame()).copy()
    if not posterior.empty:
        if {"lower_mean", "upper_mean"}.issubset(posterior.columns):
            posterior["posterior_mean_width"] = (
                pd.to_numeric(posterior["upper_mean"], errors="coerce")
                - pd.to_numeric(posterior["lower_mean"], errors="coerce")
            )
        if "covers_rct_cate_posterior_mean_interval" not in posterior.columns and {
            "lower_mean",
            "upper_mean",
            "rct_cate",
        }.issubset(posterior.columns):
            posterior["covers_rct_cate_posterior_mean_interval"] = (
                pd.to_numeric(posterior["rct_cate"], errors="coerce")
                >= pd.to_numeric(posterior["lower_mean"], errors="coerce")
            ) & (
                pd.to_numeric(posterior["rct_cate"], errors="coerce")
                <= pd.to_numeric(posterior["upper_mean"], errors="coerce")
            )
        if "posterior_envelope_width" not in posterior.columns and {
            "posterior_envelope_lower",
            "posterior_envelope_upper",
        }.issubset(posterior.columns):
            posterior["posterior_envelope_width"] = (
                pd.to_numeric(posterior["posterior_envelope_upper"], errors="coerce")
                - pd.to_numeric(posterior["posterior_envelope_lower"], errors="coerce")
            )
        if "covers_rct_cate_posterior_envelope" not in posterior.columns and {
            "posterior_envelope_lower",
            "posterior_envelope_upper",
            "rct_cate",
        }.issubset(posterior.columns):
            posterior["covers_rct_cate_posterior_envelope"] = (
                pd.to_numeric(posterior["rct_cate"], errors="coerce")
                >= pd.to_numeric(posterior["posterior_envelope_lower"], errors="coerce")
            ) & (
                pd.to_numeric(posterior["rct_cate"], errors="coerce")
                <= pd.to_numeric(posterior["posterior_envelope_upper"], errors="coerce")
            )
        if {
            "covers_rct_cate_posterior_mean_interval",
            "posterior_mean_width",
            "covers_rct_cate_posterior_envelope",
            "posterior_envelope_width",
        }.issubset(posterior.columns):
            columns = [
                column
                for column in EXP16_PLOT_SUMMARY_KEYS
                if column in posterior.columns
            ]
            columns.extend(
                [
                    "covers_rct_cate_posterior_mean_interval",
                    "posterior_mean_width",
                    "covers_rct_cate_posterior_envelope",
                    "posterior_envelope_width",
                ]
            )
            metric_frames.append(
                posterior.loc[:, columns].rename(
                    columns={
                        "covers_rct_cate_posterior_mean_interval": (
                            "posterior_mean_coverage"
                        ),
                        "covers_rct_cate_posterior_envelope": (
                            "posterior_envelope_coverage"
                        ),
                    }
                )
            )

    if not metric_frames:
        return pd.DataFrame()
    combined = metric_frames[0]
    for frame in metric_frames[1:]:
        merge_keys = [
            column
            for column in EXP16_PLOT_SUMMARY_KEYS
            if column in combined.columns and column in frame.columns
        ]
        if merge_keys:
            combined = combined.merge(frame, on=merge_keys, how="outer")
        else:
            combined = pd.concat([combined, frame], ignore_index=True, sort=False)
    for _, column in EXP16_PLOT_SUMMARY_METRICS:
        if column not in combined.columns:
            combined[column] = np.nan
    metric_columns = [
        column for _, column in EXP16_PLOT_SUMMARY_METRICS
    ]
    return combined.dropna(how="all", subset=metric_columns)

def write_exp16_plot_metric_summary_tables(
    plot_metric_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """
    Write Exp16 plot-metric LaTeX summaries, one table per backend.

    Args:
        plot_metric_summary: Stratum-level frame from
            ``build_exp16_plot_metric_summary_frame``.
        output_dir: Directory where ``coverage_width_summary*.tex`` files are
            written.

    Returns:
        Paths to the written LaTeX files. The root summary contains one table
        environment per backend; backend-specific files contain only that
        backend and omit the backend column.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    root_path = output_dir / "coverage_width_summary.tex"
    if plot_metric_summary.empty:
        root_path.write_text("", encoding="utf-8")
        return [root_path]

    if "stage1_backend" in plot_metric_summary.columns:
        backend_values = sorted(
            plot_metric_summary["stage1_backend"].dropna().astype(str).unique()
        )
    else:
        backend_values = ["unknown_backend"]

    written_paths: list[Path] = []
    root_tables: list[str] = []
    for backend in backend_values:
        if "stage1_backend" in plot_metric_summary.columns:
            backend_rows = plot_metric_summary.loc[
                plot_metric_summary["stage1_backend"].astype(str).eq(backend)
            ]
        else:
            backend_rows = plot_metric_summary
        if backend_rows.empty:
            continue
        backend_title = backend.replace("_", r"\_")
        label_slug = backend.replace("_", "-")
        backend_path = output_dir / f"coverage_width_summary_{backend}.tex"
        table_specs = (
            (
                EXP16_PLOT_COVERAGE_SUMMARY_METRICS,
                (
                    f"Exp16 {backend_title} plotted target strata coverage "
                    "for conformalized LP, posterior-mean LP, and "
                    "posterior-envelope LP bounds by experiment, confounding "
                    "variable, and beta, summarized as mean $\\pm$ standard "
                    "deviation."
                ),
                f"tab:exp16-coverage-summary-{label_slug}",
            ),
            (
                EXP16_PLOT_WIDTH_SUMMARY_METRICS,
                (
                    f"Exp16 {backend_title} plotted target strata interval "
                    "width for conformalized LP, posterior-mean LP, and "
                    "posterior-envelope LP bounds by experiment, confounding "
                    "variable, and beta, summarized as mean $\\pm$ standard "
                    "deviation."
                ),
                f"tab:exp16-width-summary-{label_slug}",
            ),
        )
        backend_tables: list[str] = []
        for metrics, caption, label in table_specs:
            table = make_multi_metric_summary_latex_table(
                backend_rows,
                group_columns=("study", "pattern", "beta"),
                metrics=metrics,
            )
            if table.empty:
                continue
            backend_tables.append(
                render_latex_table(
                    table,
                    index=False,
                    caption=caption,
                    label=label,
                )
            )
        backend_path.write_text("\n\n".join(backend_tables), encoding="utf-8")
        written_paths.append(backend_path)
        root_tables.append(backend_path.read_text(encoding="utf-8"))

    root_path.write_text("\n\n".join(root_tables), encoding="utf-8")
    return [root_path, *written_paths]


def _as_bool_like(series: pd.Series, *, default: bool) -> pd.Series:
    """Coerce common CSV boolean encodings into a boolean series.

    Args:
        series: Values to interpret as booleans. Native booleans, numeric
            0/1 values, and common string encodings are supported.
        default: Value assigned when an entry cannot be interpreted.

    Returns:
        Boolean series with the same index as ``series``.
    """

    result = pd.Series(default, index=series.index, dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    result.loc[numeric_mask] = numeric.loc[numeric_mask].ne(0.0)

    text = series.astype("string").str.strip().str.lower()
    true_mask = text.isin({"true", "t", "yes", "y"})
    false_mask = text.isin({"false", "f", "no", "n"})
    result.loc[true_mask] = True
    result.loc[false_mask] = False
    return result

def _filter_target_plot_to_lp_supported_studies(bounds: pd.DataFrame) -> pd.DataFrame:
    """Drop study panels whose target LP rows have zero optimal solutions.

    Exp15 keeps ``[-1, 1]`` fallback rows in the saved CSVs so coverage and
    conformalization can still be audited. For plots, however, an all-fallback
    study visually looks like a valid but very wide interval. This helper keeps
    a study only when at least one displayed row is backed by an optimal fine LP
    solution, using the most direct diagnostic columns available.

    Args:
        bounds: Target interval rows prepared for a plot.

    Returns:
        A copy of ``bounds`` with all-fallback study groups removed. If the
        diagnostic columns are absent, the input rows are returned unchanged.
    """

    if bounds.empty:
        return bounds

    row_has_optimal: pd.Series | None = None
    if "n_base_optimal_fine_strata" in bounds.columns:
        row_has_optimal = pd.to_numeric(
            bounds["n_base_optimal_fine_strata"],
            errors="coerce",
        ).fillna(0.0).gt(0.0)
    elif "base_interval_available" in bounds.columns:
        row_has_optimal = _as_bool_like(
            bounds["base_interval_available"],
            default=False,
        )
    elif "all_fine_strata_base_optimal" in bounds.columns:
        row_has_optimal = _as_bool_like(
            bounds["all_fine_strata_base_optimal"],
            default=False,
        )
    elif "used_full_range_fallback" in bounds.columns:
        row_has_optimal = ~_as_bool_like(
            bounds["used_full_range_fallback"],
            default=True,
        )
    elif "fine_interval_status" in bounds.columns:
        row_has_optimal = ~bounds["fine_interval_status"].astype(str).eq(
            "full_range_fallback_used",
        )

    if row_has_optimal is None:
        return bounds
    if "study" not in bounds.columns:
        return bounds.loc[row_has_optimal].copy()

    keep_study = row_has_optimal.groupby(bounds["study"].astype(str)).transform("any")
    return bounds.loc[keep_study].copy()

def _filter_lp_posterior_samples_to_optimal_rows(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only posterior LP sample rows backed by optimal LP solutions.

    Args:
        samples: Sample-level useful-stratum LP rows, typically
            ``target_gp_posterior_bounds_samples``.

    Returns:
        Rows with finite endpoints and no full-range fallback contribution.
    """

    if samples.empty:
        return samples
    mask = pd.Series(True, index=samples.index, dtype=bool)
    if "status" in samples.columns:
        mask &= samples["status"].astype(str).eq("optimal")
    if "used_full_range_fallback" in samples.columns:
        mask &= ~_as_bool_like(
            samples["used_full_range_fallback"],
            default=True,
        )
    if "all_fine_strata_base_optimal" in samples.columns:
        mask &= _as_bool_like(
            samples["all_fine_strata_base_optimal"],
            default=False,
        )
    elif {
        "n_base_optimal_fine_strata",
        "n_fine_strata",
    }.issubset(samples.columns):
        n_optimal = pd.to_numeric(
            samples["n_base_optimal_fine_strata"],
            errors="coerce",
        ).fillna(0.0)
        n_fine = pd.to_numeric(samples["n_fine_strata"], errors="coerce")
        mask &= n_fine.gt(0.0) & n_optimal.ge(n_fine)
    elif "n_base_optimal_fine_strata" in samples.columns:
        mask &= pd.to_numeric(
            samples["n_base_optimal_fine_strata"],
            errors="coerce",
        ).fillna(0.0).gt(0.0)

    if {"lower", "upper"}.issubset(samples.columns):
        lower = pd.to_numeric(samples["lower"], errors="coerce")
        upper = pd.to_numeric(samples["upper"], errors="coerce")
        mask &= lower.notna() & upper.notna()
    return samples.loc[mask].copy()

def _posterior_mean_plot_data(
    posterior: pd.DataFrame,
    *,
    posterior_samples: pd.DataFrame | None = None,
    lp_support_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build posterior-mean plot rows, optionally from optimal samples only.

    Args:
        posterior: Target GP posterior LP summary carrying CATE reference
            columns.
        posterior_samples: Optional sample-level posterior LP rows. When
            provided, the plotted ``lower_mean`` and ``upper_mean`` are
            recomputed from optimal, non-fallback rows only.
        lp_support_reference: Optional target LP table used as a fallback
            study-panel filter when sample-level posterior rows are absent.

    Returns:
        Plot-ready posterior rows.
    """

    if (
        posterior_samples is None
        or posterior_samples.empty
        or not {"lp_posterior_sample_id", "lower", "upper"}.issubset(
            posterior_samples.columns,
        )
    ):
        plot_data = posterior.copy()
        if (
            lp_support_reference is not None
            and not lp_support_reference.empty
            and "study" in plot_data.columns
            and "study" in lp_support_reference.columns
        ):
            supported = _filter_target_plot_to_lp_supported_studies(
                lp_support_reference,
            )
            supported_studies = set(supported["study"].astype(str).unique())
            plot_data = plot_data.loc[
                plot_data["study"].astype(str).isin(supported_studies)
            ].copy()
        return plot_data

    optimal = _filter_lp_posterior_samples_to_optimal_rows(posterior_samples)
    if optimal.empty:
        return pd.DataFrame()

    group_columns = [
        column
        for column in [
            "study",
            "target_site",
            "selected_top_k",
            *USEFUL_COVARIATES,
        ]
        if column in optimal.columns
    ]
    if not group_columns:
        return posterior.copy()

    optimal = optimal.assign(
        _plot_lower=pd.to_numeric(optimal["lower"], errors="coerce"),
        _plot_upper=pd.to_numeric(optimal["upper"], errors="coerce"),
    ).dropna(subset=["_plot_lower", "_plot_upper"])
    if optimal.empty:
        return pd.DataFrame()

    summary = (
        optimal.groupby(group_columns, dropna=False, sort=False)
        .agg(
            lower_mean=("_plot_lower", "mean"),
            lower_q05=("_plot_lower", lambda values: values.quantile(0.05)),
            upper_mean=("_plot_upper", "mean"),
            upper_q95=("_plot_upper", lambda values: values.quantile(0.95)),
            lp_posterior_n_optimal=("lp_posterior_sample_id", "nunique"),
        )
        .reset_index()
    )
    summary["posterior_envelope_lower"] = summary["lower_q05"]
    summary["posterior_envelope_upper"] = summary["upper_q95"]
    summary["posterior_envelope_width"] = (
        summary["posterior_envelope_upper"] - summary["posterior_envelope_lower"]
    )
    total_counts = (
        posterior_samples.groupby(group_columns, dropna=False, sort=False)[
            "lp_posterior_sample_id"
        ]
        .nunique()
        .reset_index(name="lp_posterior_n_samples")
    )
    summary = summary.merge(total_counts, on=group_columns, how="left")
    summary["lp_posterior_prob_optimal"] = (
        summary["lp_posterior_n_optimal"]
        / summary["lp_posterior_n_samples"].replace(0, np.nan)
    )

    reference_columns = [
        column
        for column in posterior.columns
        if column not in summary.columns or column in group_columns
    ]
    merge_columns = [
        column for column in group_columns if column in posterior.columns
    ]
    if merge_columns and reference_columns:
        references = posterior[reference_columns].drop_duplicates(
            subset=merge_columns,
        )
        summary = summary.merge(
            references,
            on=merge_columns,
            how="left",
            validate="one_to_one",
        )
    return summary


def _target_tolerance_group_columns(
    choices: pd.DataFrame,
    bounds: pd.DataFrame,
) -> list[str]:
    """Return shared run-identity columns for target tolerance diagnostics."""

    candidate_columns = [
        "study",
        "pattern",
        "beta",
        "target_site",
        "stage1_backend",
        "lp_probability_backend",
    ]
    return [
        column
        for column in candidate_columns
        if column in choices.columns and column in bounds.columns
    ]


def _target_tolerance_plot_data(
    choices: pd.DataFrame,
    bounds: pd.DataFrame,
) -> pd.DataFrame:
    """Build long c/d rows with per-run LP fallback diagnostics."""

    required = {"covariate", "epsilon", "c", "d"}
    if choices.empty or not required.issubset(choices.columns):
        return pd.DataFrame()
    selected = choices.copy()
    if "status" in selected.columns:
        selected = selected.loc[selected["status"].astype(str).eq("selected")].copy()
    selected = selected.dropna(subset=["covariate", "epsilon", "c", "d"])
    if selected.empty:
        return pd.DataFrame()

    group_columns = _target_tolerance_group_columns(selected, bounds)
    if bounds.empty or not group_columns:
        selected["base_optimal_rate"] = np.nan
        selected["fallback_rate"] = np.nan
    else:
        diagnostic_bounds = bounds.copy()
        if "base_interval_available" in diagnostic_bounds.columns:
            base_available = pd.to_numeric(
                diagnostic_bounds["base_interval_available"],
                errors="coerce",
            )
        elif "status" in diagnostic_bounds.columns:
            base_available = diagnostic_bounds["status"].astype(str).eq("optimal")
        else:
            base_available = pd.Series(np.nan, index=diagnostic_bounds.index)
        if "used_full_range_fallback" in diagnostic_bounds.columns:
            fallback_used = pd.to_numeric(
                diagnostic_bounds["used_full_range_fallback"],
                errors="coerce",
            )
        else:
            fallback_used = 1.0 - pd.to_numeric(base_available, errors="coerce")
        diagnostic_bounds = diagnostic_bounds.assign(
            _base_available=base_available.astype(float),
            _fallback_used=fallback_used.astype(float),
        )
        summary = (
            diagnostic_bounds.groupby(group_columns, dropna=False)
            .agg(
                base_optimal_rate=("_base_available", "mean"),
                fallback_rate=("_fallback_used", "mean"),
            )
            .reset_index()
        )
        selected = selected.merge(summary, on=group_columns, how="left")

    id_columns = [
        column
        for column in [
            *group_columns,
            "covariate",
            "is_primary_choice",
            "policy_top_k",
        ]
        if column in selected.columns
    ]
    long = selected.melt(
        id_vars=id_columns + ["epsilon", "base_optimal_rate", "fallback_rate"],
        value_vars=["c", "d"],
        var_name="tolerance",
        value_name="value",
    )
    long["epsilon"] = pd.to_numeric(long["epsilon"], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["base_optimal_rate"] = pd.to_numeric(
        long["base_optimal_rate"],
        errors="coerce",
    )
    return long.dropna(subset=["epsilon", "value"])


def _plot_target_tolerance_vs_epsilon(
    choices: pd.DataFrame,
    bounds: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot selected target c/d tolerances against epsilon and LP feasibility."""

    plot_data = _target_tolerance_plot_data(choices, bounds)
    if plot_data.empty:
        output_path.unlink(missing_ok=True)
        return

    covariates = sorted(plot_data["covariate"].astype(str).unique())
    fig, axes = plt.subplots(
        1,
        len(covariates),
        figsize=(max(6.0, 3.2 * len(covariates)), 4.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    markers = {"c": "o", "d": "s"}
    colors = plot_data["base_optimal_rate"].fillna(0.5).to_numpy(dtype=float)
    color_min = float(np.nanmin(colors)) if colors.size else 0.0
    color_max = float(np.nanmax(colors)) if colors.size else 1.0
    if np.isclose(color_min, color_max):
        color_min = 0.0
        color_max = 1.0
    norm = matplotlib.colors.Normalize(vmin=color_min, vmax=color_max)
    cmap = plt.get_cmap("viridis")

    for ax, covariate in zip(axes[0], covariates):
        subset = plot_data.loc[plot_data["covariate"].astype(str).eq(covariate)]
        for tolerance, marker in markers.items():
            rows = subset.loc[subset["tolerance"].eq(tolerance)]
            if rows.empty:
                continue
            ax.scatter(
                rows["epsilon"],
                rows["value"],
                c=rows["base_optimal_rate"].fillna(0.5),
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=58,
                edgecolors="black",
                linewidths=0.45,
                alpha=0.88,
                label=tolerance,
            )
        diagonal_max = max(
            float(plot_data["epsilon"].max()),
            float(plot_data["value"].max()),
            0.5,
        )
        ax.plot(
            [0.0, diagonal_max],
            [0.0, diagonal_max],
            color="grey",
            linestyle=":",
            linewidth=1.0,
        )
        ax.set_title(covariate.removeprefix("resp_").removesuffix("_bin"))
        ax.set_xlabel("epsilon trigger")
        ax.grid(True, color="0.88", linewidth=0.8)
    axes[0, 0].set_ylabel("learned tolerance")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper right", frameon=False)
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=list(axes[0]),
        location="right",
        fraction=0.032,
        pad=0.025,
    )
    colorbar.set_label("Base LP optimal rate")
    fig.subplots_adjust(
        left=0.10,
        right=0.86,
        bottom=0.16,
        top=0.94,
        wspace=0.32,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_study_order(frame: pd.DataFrame) -> list[str]:
    """Return present studies in the preferred Exp16 display order.

    Args:
        frame: Plotting frame with an optional ``study`` column.

    Returns:
        Ordered study labels.
    """

    if "study" not in frame.columns:
        return ["target"]
    present = list(dict.fromkeys(frame["study"].dropna().astype(str)))
    preferred = [study for study in DEFAULT_EXP16_STUDIES if study in present]
    remaining = [study for study in present if study not in preferred]
    return [*preferred, *remaining]

def _prepare_study_panel_plot_data(
    frame: pd.DataFrame,
    covariates: Sequence[str],
) -> tuple[pd.DataFrame, list[str], list[tuple[Any, ...]]]:
    """Add shared x positions for Exp15-style study-panel plots.

    Args:
        frame: Plotting rows after dropping missing endpoints.
        covariates: CATE stratum columns to display on the x-axis.

    Returns:
        Plotting frame, stratum column names, and ordered stratum tuples.
    """

    plot_data = frame.copy()
    stratum_columns = [
        column for column in covariates if column in plot_data.columns
    ]
    if not stratum_columns:
        plot_data["_plot_x"] = np.arange(len(plot_data))
        return plot_data, [], []

    stratum_order = [
        tuple(row)
        for row in (
            plot_data[stratum_columns]
            .drop_duplicates()
            .sort_values(stratum_columns)
            .itertuples(index=False, name=None)
        )
    ]
    stratum_to_x = {
        stratum: index for index, stratum in enumerate(stratum_order)
    }
    plot_data["_plot_x"] = [
        stratum_to_x[tuple(row[column] for column in stratum_columns)]
        for _, row in plot_data.iterrows()
    ]
    return plot_data, stratum_columns, stratum_order

def _format_stratum_tick(stratum: tuple[Any, ...]) -> str:
    """Format a CATE stratum tuple for an x-axis tick.

    Args:
        stratum: Values identifying one CATE stratum.

    Returns:
        Compact tick label such as ``(0,1,0)``.
    """

    values: list[str] = []
    for value in stratum:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            values.append(str(value))
            continue
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            values.append(str(int(numeric_value)))
        else:
            values.append(f"{numeric_value:g}")
    return "(" + ",".join(values) + ")"

def _set_study_panel_axis_labels(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    stratum_columns: Sequence[str],
    stratum_order: Sequence[tuple[Any, ...]],
    legend_order: Sequence[str] | None = None,
    legend_ncol: int | None = None,
    legend_bbox_to_anchor: tuple[float, float] = (0.5, 0.995),
    supxlabel_y: float = 0.070,
) -> None:
    """Apply shared labels, ticks, and legend to a study-panel figure.

    Args:
        fig: Matplotlib figure to update.
        axes: Figure axes arranged as one column.
        stratum_columns: CATE stratum columns shown on the x-axis.
        stratum_order: Ordered stratum tuples used for x-axis ticks.
        legend_order: Optional preferred legend-label order.
        legend_ncol: Optional fixed number of legend columns.
        legend_bbox_to_anchor: Figure-relative legend anchor.
        supxlabel_y: Figure-relative vertical position for the shared x-label.
    """

    if stratum_order:
        for ax in axes[:, 0]:
            ax.set_xlim(-0.5, len(stratum_order) - 0.5)
            ax.set_xticks(np.arange(len(stratum_order)))
            ax.set_xticklabels(
                [_format_stratum_tick(stratum) for stratum in stratum_order],
                rotation=45,
                ha="right",
            )
    handles_by_label: dict[str, Any] = {}
    for ax in axes[:, 0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        labels = list(handles_by_label)
        if legend_order is not None:
            labels = [label for label in legend_order if label in handles_by_label]
            labels.extend(
                label for label in handles_by_label if label not in labels
            )
        handles = [handles_by_label[label] for label in labels]
        legend_kwargs: dict[str, Any] = {}
        if legend_order is not None:
            legend_kwargs.update(columnspacing=2.0, handletextpad=0.7)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=(
                legend_ncol
                if legend_ncol is not None
                else max(1, min(4, len(handles_by_label)))
            ),
            frameon=False,
            **legend_kwargs,
        )
    if stratum_columns:
        display_columns = [
            column.removeprefix("resp_").removesuffix("_bin")
            for column in stratum_columns
        ]
        fig.supxlabel(
            "Z=(" + ", ".join(display_columns) + ")",
            y=supxlabel_y,
        )
    else:
        fig.supxlabel("Z", y=supxlabel_y)


def _set_horizontal_study_panel_axis_labels(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    stratum_columns: Sequence[str],
    stratum_order: Sequence[tuple[Any, ...]],
    legend_order: Sequence[str] | None = None,
    legend_ncol: int | None = None,
    legend_bbox_to_anchor: tuple[float, float] = (0.5, 0.995),
    supxlabel_y: float = 0.055,
) -> None:
    """Apply labels and a shared legend to side-by-side CATE panels.

    Args:
        fig: Matplotlib figure containing one row of study panels.
        axes: Figure axes arranged as one row.
        stratum_columns: CATE stratum columns shown on the y-axis.
        stratum_order: Ordered stratum tuples used for y-axis ticks.
        legend_order: Optional preferred legend-label order.
        legend_ncol: Optional fixed number of legend columns.
        legend_bbox_to_anchor: Figure-relative legend anchor.
        supxlabel_y: Figure-relative vertical position for the shared x-label.
    """

    panel_axes = axes[0, :]
    if stratum_order:
        ticks = np.arange(len(stratum_order))
        tick_labels = [_format_stratum_tick(stratum) for stratum in stratum_order]
        for panel_index, ax in enumerate(panel_axes):
            ax.set_ylim(-0.5, len(stratum_order) - 0.5)
            ax.set_yticks(ticks)
            if panel_index == 0:
                ax.set_yticklabels(tick_labels)
            else:
                ax.tick_params(labelleft=False)

    handles_by_label: dict[str, Any] = {}
    for ax in panel_axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        labels = list(handles_by_label)
        if legend_order is not None:
            labels = [label for label in legend_order if label in handles_by_label]
            labels.extend(
                label for label in handles_by_label if label not in labels
            )
        handles = [handles_by_label[label] for label in labels]
        legend_kwargs: dict[str, Any] = {}
        if legend_order is not None:
            legend_kwargs.update(columnspacing=2.0, handletextpad=0.7)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=(
                legend_ncol
                if legend_ncol is not None
                else max(1, min(4, len(handles_by_label)))
            ),
            frameon=False,
            **legend_kwargs,
        )

    if stratum_columns:
        display_columns = [
            column.removeprefix("resp_").removesuffix("_bin")
            for column in stratum_columns
        ]
        panel_axes[0].set_ylabel("Z=(" + ", ".join(display_columns) + ")")
    else:
        panel_axes[0].set_ylabel("Z")
    fig.supxlabel("CATE", y=supxlabel_y)


def _save_panel_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save one panel figure as PNG and matching SVG.

    Args:
        fig: Figure to save.
        output_path: PNG output path.
    """

    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)

def _remove_panel_figure(output_path: Path) -> None:
    """
    Remove stale PNG/SVG panel outputs when no rows remain to plot.

    Args:
        output_path: PNG output path whose SVG sibling is removed too.
    """

    output_path.unlink(missing_ok=True)
    output_path.with_suffix(".svg").unlink(missing_ok=True)

def _plot_exp16_calibrated_bounds_by_study(
    bounds: pd.DataFrame,
    output_path: Path,
    covariates: Sequence[str],
) -> None:
    """Write Exp15-style panels for calibrated LP bounds versus RCT CATEs.

    Args:
        bounds: Target LP bounds with conformal endpoints and RCT labels.
        output_path: PNG output path.
        covariates: CATE stratum columns displayed on the x-axis.
    """

    required = {
        "conformal_lower",
        "conformal_upper",
        "rct_cate",
    }
    if not required.issubset(bounds.columns):
        _remove_panel_figure(output_path)
        return
    plot_data = bounds.dropna(
        subset=["conformal_lower", "conformal_upper", "rct_cate"]
    ).copy()
    plot_data = plot_data.loc[_optimal_conformalized_lp_mask(plot_data)].copy()
    if plot_data.empty:
        _remove_panel_figure(output_path)
        return
    plot_data, stratum_columns, stratum_order = _prepare_study_panel_plot_data(
        plot_data,
        covariates,
    )
    studies = _plot_study_order(plot_data)
    fig, axes = plt.subplots(
        len(studies),
        1,
        figsize=(9.0, max(3.8, 3.4 * len(studies) - 0.4)),
        squeeze=False,
    )
    for ax, study in zip(axes[:, 0], studies):
        study_rows = (
            plot_data.loc[plot_data["study"].astype(str).eq(study)]
            if "study" in plot_data.columns
            else plot_data
        ).sort_values("_plot_x")
        x = study_rows["_plot_x"].to_numpy(dtype=int)
        lower = study_rows["conformal_lower"].to_numpy(dtype=float)
        upper = study_rows["conformal_upper"].to_numpy(dtype=float)
        truth = study_rows["rct_cate"].to_numpy(dtype=float)
        ax.vlines(
            x,
            lower,
            upper,
            color="tab:blue",
            alpha=0.45,
            linewidth=5.0,
            label="Conformalized point-estimate LP interval",
            zorder=1,
        )
        if {
            "rct_cate_ci90_lower",
            "rct_cate_ci90_upper",
        }.issubset(study_rows.columns):
            ci_lower = pd.to_numeric(
                study_rows["rct_cate_ci90_lower"],
                errors="coerce",
            )
            ci_upper = pd.to_numeric(
                study_rows["rct_cate_ci90_upper"],
                errors="coerce",
            )
            has_ci = ci_lower.notna() & ci_upper.notna()
            if bool(has_ci.any()):
                ax.vlines(
                    x[has_ci.to_numpy()],
                    ci_lower.loc[has_ci].to_numpy(dtype=float),
                    ci_upper.loc[has_ci].to_numpy(dtype=float),
                    color="tab:orange",
                    linewidth=2.5,
                    label="RCT CATE 90% Wald CI",
                    zorder=2,
                )
        ax.scatter(
            x,
            truth,
            color="black",
            s=34,
            label="Held-out RCT CATE",
            zorder=3,
        )
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(str(study), fontweight="bold")
        ax.set_ylabel("CATE")
    _set_study_panel_axis_labels(
        fig,
        axes,
        stratum_columns=stratum_columns,
        stratum_order=stratum_order,
    )
    fig.tight_layout(rect=(0.0, 0.095, 1.0, 0.94))
    _save_panel_figure(fig, output_path)

def _plot_exp16_posterior_bounds_by_study(
    posterior: pd.DataFrame,
    output_path: Path,
    covariates: Sequence[str],
) -> None:
    """Write Exp15-style panels for GP-posterior LP bounds and RCT CATEs.

    Args:
        posterior: Posterior summary with endpoint means and envelopes.
        output_path: PNG output path.
        covariates: CATE stratum columns displayed on the x-axis.
    """

    required = {
        "posterior_envelope_lower",
        "posterior_envelope_upper",
        "lower_mean",
        "upper_mean",
        "rct_cate",
        "lp_posterior_prob_optimal",
    }
    if not required.issubset(posterior.columns):
        return
    plot_data = posterior.loc[
        posterior["rct_cate"].notna()
        & posterior["lower_mean"].notna()
        & posterior["upper_mean"].notna()
        & posterior["lp_posterior_prob_optimal"].ge(0.80)
    ].copy()
    if plot_data.empty:
        return
    plot_data, stratum_columns, stratum_order = _prepare_study_panel_plot_data(
        plot_data,
        covariates,
    )
    studies = _plot_study_order(plot_data)
    fig, axes = plt.subplots(
        len(studies),
        1,
        figsize=(9.0, max(3.8, 3.4 * len(studies) - 0.4)),
        squeeze=False,
    )
    for ax, study in zip(axes[:, 0], studies):
        study_rows = (
            plot_data.loc[plot_data["study"].astype(str).eq(study)]
            if "study" in plot_data.columns
            else plot_data
        ).sort_values("_plot_x")
        x = study_rows["_plot_x"].to_numpy(dtype=int)
        ax.vlines(
            x,
            study_rows["posterior_envelope_lower"].to_numpy(dtype=float),
            study_rows["posterior_envelope_upper"].to_numpy(dtype=float),
            color="#93c5fd",
            linewidth=8.0,
            alpha=0.55,
            label="Posterior endpoint envelope (5% lower, 95% upper)",
            zorder=1,
        )
        ax.vlines(
            x,
            study_rows["lower_mean"].to_numpy(dtype=float),
            study_rows["upper_mean"].to_numpy(dtype=float),
            color="tab:blue",
            linewidth=3.0,
            label="Posterior mean sampled LP interval",
            zorder=2,
        )
        ax.scatter(
            x,
            study_rows["rct_cate"].to_numpy(dtype=float),
            color="black",
            s=34,
            label="Held-out RCT CATE",
            zorder=3,
        )
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(str(study), fontweight="bold")
        ax.set_ylabel("CATE")
    _set_study_panel_axis_labels(
        fig,
        axes,
        stratum_columns=stratum_columns,
        stratum_order=stratum_order,
    )
    fig.tight_layout(rect=(0.0, 0.095, 1.0, 0.94))
    _save_panel_figure(fig, output_path)

def _plot_exp16_posterior_mean_bounds_by_study(
    posterior: pd.DataFrame,
    output_path: Path,
    covariates: Sequence[str],
    *,
    horizontal: bool = False,
) -> None:
    """Write panels for posterior-mean sampled LP bounds versus RCT CATEs.

    Args:
        posterior: Posterior summary with endpoint means and RCT labels.
        output_path: PNG output path.
        covariates: CATE stratum columns displayed on the x-axis.
        horizontal: Whether to arrange study panels side by side with CATE on
            the x-axis.
    """

    required = {
        "lower_mean",
        "upper_mean",
        "rct_cate",
    }
    if not required.issubset(posterior.columns):
        return
    plot_data = posterior.dropna(
        subset=["lower_mean", "upper_mean", "rct_cate"]
    ).copy()
    if plot_data.empty:
        return
    plot_data, stratum_columns, stratum_order = _prepare_study_panel_plot_data(
        plot_data,
        covariates,
    )
    studies = _plot_study_order(plot_data)
    if horizontal:
        fig, axes = plt.subplots(
            1,
            len(studies),
            figsize=(
                max(9.0, 5.2 * len(studies)),
                max(4.0, 0.42 * max(1, len(stratum_order)) + 1.8),
            ),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        panel_axes = axes[0, :]
    else:
        fig, axes = plt.subplots(
            len(studies),
            1,
            figsize=(9.0, max(3.8, 3.4 * len(studies) - 0.4)),
            squeeze=False,
        )
        panel_axes = axes[:, 0]
    for ax, study in zip(panel_axes, studies):
        study_rows = (
            plot_data.loc[plot_data["study"].astype(str).eq(study)]
            if "study" in plot_data.columns
            else plot_data
        ).sort_values("_plot_x")
        x = study_rows["_plot_x"].to_numpy(dtype=int)
        lower = study_rows["lower_mean"].to_numpy(dtype=float)
        upper = study_rows["upper_mean"].to_numpy(dtype=float)
        if horizontal:
            y = x
            ax.hlines(
                y,
                lower,
                upper,
                colors="tab:blue",
                alpha=0.60,
                linewidth=5.0,
                label="Posterior mean LP interval (optimal draws only)",
                zorder=1,
            )
        else:
            ax.vlines(
                x,
                lower,
                upper,
                color="tab:blue",
                alpha=0.60,
                linewidth=5.0,
                label="Posterior mean LP interval (optimal draws only)",
                zorder=1,
            )
        if {
            "rct_cate_ci90_lower",
            "rct_cate_ci90_upper",
        }.issubset(study_rows.columns):
            ci_lower = pd.to_numeric(
                study_rows["rct_cate_ci90_lower"],
                errors="coerce",
            )
            ci_upper = pd.to_numeric(
                study_rows["rct_cate_ci90_upper"],
                errors="coerce",
            )
            has_ci = ci_lower.notna() & ci_upper.notna()
            if bool(has_ci.any()):
                if horizontal:
                    ax.hlines(
                        y[has_ci.to_numpy()],
                        ci_lower.loc[has_ci].to_numpy(dtype=float),
                        ci_upper.loc[has_ci].to_numpy(dtype=float),
                        colors="tab:orange",
                        linewidth=2.5,
                        label="RCT CATE 90% Wald CI",
                        zorder=2,
                    )
                else:
                    ax.vlines(
                        x[has_ci.to_numpy()],
                        ci_lower.loc[has_ci].to_numpy(dtype=float),
                        ci_upper.loc[has_ci].to_numpy(dtype=float),
                        color="tab:orange",
                        linewidth=2.5,
                        label="RCT CATE 90% Wald CI",
                        zorder=2,
                    )
        truth = study_rows["rct_cate"].to_numpy(dtype=float)
        if horizontal:
            ax.scatter(
                truth,
                y,
                color="black",
                s=34,
                label="Held-out RCT CATE",
                zorder=3,
            )
            ax.axvline(0.0, color="grey", linewidth=0.8)
            ax.set_xlim(-1.0, 1.0)
        else:
            ax.scatter(
                x,
                truth,
                color="black",
                s=34,
                label="Held-out RCT CATE",
                zorder=3,
            )
            ax.axhline(0.0, color="grey", linewidth=0.8)
            ax.set_ylim(-1.0, 1.0)
        ax.set_title(str(study), fontweight="bold")
        if not horizontal:
            ax.set_ylabel("CATE")
    if horizontal:
        _set_horizontal_study_panel_axis_labels(
            fig,
            axes,
            stratum_columns=stratum_columns,
            stratum_order=stratum_order,
        )
        fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.90))
    else:
        _set_study_panel_axis_labels(
            fig,
            axes,
            stratum_columns=stratum_columns,
            stratum_order=stratum_order,
        )
        fig.tight_layout(rect=(0.0, 0.095, 1.0, 0.94))
    _save_panel_figure(fig, output_path)

def _plot_exp16_posterior_mean_with_envelope_by_study(
    posterior: pd.DataFrame,
    output_path: Path,
    covariates: Sequence[str],
    *,
    show_optimality_rate: bool = True,
    horizontal: bool = False,
) -> None:
    """Write posterior-mean LP bounds with endpoint envelopes.

    Args:
        posterior: Posterior summary with endpoint means, envelopes, and RCT
            labels.
        output_path: PNG output path.
        covariates: CATE stratum columns displayed on the x-axis.
        show_optimality_rate: Whether mean intervals use the posterior
            optimality color scale and the figure includes its colorbar.
        horizontal: Whether to arrange study panels side by side with CATE on
            the x-axis.
    """

    required = {
        "posterior_envelope_lower",
        "posterior_envelope_upper",
        "lower_mean",
        "upper_mean",
        "rct_cate",
    }
    if show_optimality_rate:
        required.add("lp_posterior_prob_optimal")
    if not required.issubset(posterior.columns):
        _remove_panel_figure(output_path)
        return
    required_plot_columns = [
        "posterior_envelope_lower",
        "posterior_envelope_upper",
        "lower_mean",
        "upper_mean",
        "rct_cate",
    ]
    if show_optimality_rate:
        required_plot_columns.append("lp_posterior_prob_optimal")
    plot_data = posterior.dropna(subset=required_plot_columns).copy()
    if plot_data.empty:
        _remove_panel_figure(output_path)
        return
    plot_data, stratum_columns, stratum_order = _prepare_study_panel_plot_data(
        plot_data,
        covariates,
    )
    studies = _plot_study_order(plot_data)
    if horizontal:
        fig, axes = plt.subplots(
            1,
            len(studies),
            figsize=(
                max(9.4, 5.2 * len(studies)),
                max(4.0, 0.42 * max(1, len(stratum_order)) + 1.8),
            ),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        panel_axes = axes[0, :]
    else:
        fig, axes = plt.subplots(
            len(studies),
            1,
            figsize=(9.4, max(3.8, 3.4 * len(studies) - 0.4)),
            squeeze=False,
        )
        panel_axes = axes[:, 0]
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("viridis")
    mean_label = (
        "Posterior mean LP interval"
        if show_optimality_rate
        else "Posterior mean LP interval (optimal draws only)"
    )
    envelope_label = (
        "Posterior endpoint envelope"
        if show_optimality_rate
        else "Posterior endpoint envelope (optimal draws only)"
    )
    for ax, study in zip(panel_axes, studies):
        study_rows = (
            plot_data.loc[plot_data["study"].astype(str).eq(study)]
            if "study" in plot_data.columns
            else plot_data
        ).sort_values("_plot_x")
        x = study_rows["_plot_x"].to_numpy(dtype=int)
        envelope_lower = study_rows["posterior_envelope_lower"].to_numpy(
            dtype=float
        )
        envelope_upper = study_rows["posterior_envelope_upper"].to_numpy(
            dtype=float
        )
        lower = study_rows["lower_mean"].to_numpy(dtype=float)
        upper = study_rows["upper_mean"].to_numpy(dtype=float)
        mean_colors: Any = "tab:blue"
        if show_optimality_rate:
            mean_colors = cmap(
                norm(
                    study_rows["lp_posterior_prob_optimal"].to_numpy(
                        dtype=float
                    )
                )
            )
        if horizontal:
            y = x
            ax.hlines(
                y,
                envelope_lower,
                envelope_upper,
                colors="#bfdbfe",
                linewidth=8.0,
                alpha=0.60,
                label=envelope_label,
                zorder=1,
            )
            ax.hlines(
                y,
                lower,
                upper,
                colors=mean_colors,
                linewidth=5.0,
                alpha=0.95,
                label=mean_label,
                zorder=2,
            )
        else:
            ax.vlines(
                x,
                envelope_lower,
                envelope_upper,
                color="#bfdbfe",
                linewidth=8.0,
                alpha=0.60,
                label=envelope_label,
                zorder=1,
            )
            ax.vlines(
                x,
                lower,
                upper,
                colors=mean_colors,
                linewidth=5.0,
                alpha=0.95,
                label=mean_label,
                zorder=2,
            )
        if {
            "rct_cate_ci90_lower",
            "rct_cate_ci90_upper",
        }.issubset(study_rows.columns):
            ci_lower = pd.to_numeric(
                study_rows["rct_cate_ci90_lower"],
                errors="coerce",
            )
            ci_upper = pd.to_numeric(
                study_rows["rct_cate_ci90_upper"],
                errors="coerce",
            )
            has_ci = ci_lower.notna() & ci_upper.notna()
            if bool(has_ci.any()):
                if horizontal:
                    ax.hlines(
                        y[has_ci.to_numpy()],
                        ci_lower.loc[has_ci].to_numpy(dtype=float),
                        ci_upper.loc[has_ci].to_numpy(dtype=float),
                        colors="tab:orange",
                        linewidth=2.5,
                        label="RCT CATE 90% Wald CI",
                        zorder=3,
                    )
                else:
                    ax.vlines(
                        x[has_ci.to_numpy()],
                        ci_lower.loc[has_ci].to_numpy(dtype=float),
                        ci_upper.loc[has_ci].to_numpy(dtype=float),
                        color="tab:orange",
                        linewidth=2.5,
                        label="RCT CATE 90% Wald CI",
                        zorder=3,
                    )
        truth = study_rows["rct_cate"].to_numpy(dtype=float)
        if horizontal:
            ax.scatter(
                truth,
                y,
                color="black",
                s=34,
                label="Held-out RCT CATE",
                zorder=4,
            )
            ax.axvline(0.0, color="grey", linewidth=0.8)
            ax.set_xlim(-1.0, 1.0)
        else:
            ax.scatter(
                x,
                truth,
                color="black",
                s=34,
                label="Held-out RCT CATE",
                zorder=4,
            )
            ax.axhline(0.0, color="grey", linewidth=0.8)
            ax.set_ylim(-1.0, 1.0)
        ax.set_title(str(study), fontweight="bold")
        if not horizontal:
            ax.set_ylabel("CATE")
    if horizontal:
        if show_optimality_rate:
            _set_horizontal_study_panel_axis_labels(
                fig,
                axes,
                stratum_columns=stratum_columns,
                stratum_order=stratum_order,
            )
            scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar_mappable.set_array([])
            colorbar = fig.colorbar(
                scalar_mappable,
                ax=list(panel_axes),
                location="right",
                fraction=0.025,
                pad=0.020,
            )
            colorbar.set_label("LP posterior optimal probability")
            fig.subplots_adjust(
                left=0.10,
                right=0.86,
                bottom=0.14,
                top=0.86,
                wspace=0.20,
            )
        else:
            _set_horizontal_study_panel_axis_labels(
                fig,
                axes,
                stratum_columns=stratum_columns,
                stratum_order=stratum_order,
                legend_order=(
                    "Posterior mean LP interval (optimal draws only)",
                    "Posterior endpoint envelope (optimal draws only)",
                    "RCT CATE 90% Wald CI",
                    "Held-out RCT CATE",
                ),
                legend_ncol=2,
                legend_bbox_to_anchor=(0.5, 0.995),
            )
            fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.90))
    elif show_optimality_rate:
        figure_height = fig.get_figheight()
        _set_study_panel_axis_labels(
            fig,
            axes,
            stratum_columns=stratum_columns,
            stratum_order=stratum_order,
            supxlabel_y=0.15 / figure_height,
        )
        scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(
            scalar_mappable,
            ax=list(axes[:, 0]),
            location="right",
            fraction=0.025,
            pad=0.020,
        )
        colorbar.set_label("LP posterior optimal probability")
        fig.subplots_adjust(
            left=0.10,
            right=0.86,
            bottom=0.95 / figure_height,
            top=1.0 - 0.70 / figure_height,
            hspace=0.58,
        )
    else:
        _set_study_panel_axis_labels(
            fig,
            axes,
            stratum_columns=stratum_columns,
            stratum_order=stratum_order,
            legend_order=(
                "Posterior mean LP interval (optimal draws only)",
                "Posterior endpoint envelope (optimal draws only)",
                "RCT CATE 90% Wald CI",
                "Held-out RCT CATE",
            ),
            legend_ncol=2,
            legend_bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.90))
    _save_panel_figure(fig, output_path)

def make_exp16_plots(
    output_dir: Path,
    results: Mapping[str, pd.DataFrame],
    covariates: Sequence[str],
) -> None:
    """Write Exp15-style CATE panel plots for calibrated Exp16 intervals."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    bounds = results.get("target_lp_bounds", pd.DataFrame())
    if not bounds.empty:
        _plot_exp16_calibrated_bounds_by_study(
            bounds,
            figures_dir
            / "exp16_prediction_conformalized_lp_bounds_vs_rct_cate.png",
            covariates,
        )
    choices = results.get("target_policy_choices", pd.DataFrame())
    if not choices.empty and not bounds.empty:
        _plot_target_tolerance_vs_epsilon(
            choices,
            bounds,
            figures_dir / "exp16_target_tolerance_vs_epsilon.png",
        )
    write_backdoor_cate_error_analysis_plots(
        results,
        output_dir / "error_analysis",
        covariates=covariates,
    )
    posterior = results.get("target_gp_posterior_summary", pd.DataFrame())
    if not posterior.empty:
        _plot_exp16_posterior_mean_bounds_by_study(
            posterior,
            figures_dir
            / "exp16_prediction_posterior_mean_lp_bounds_vs_rct_cate.png",
            covariates,
        )
        _plot_exp16_posterior_mean_bounds_by_study(
            posterior,
            figures_dir
            / "exp16_prediction_posterior_mean_lp_bounds_vs_rct_cate_horizontal.png",
            covariates,
            horizontal=True,
        )
        _plot_exp16_posterior_mean_with_envelope_by_study(
            posterior,
            figures_dir
            / "exp16_prediction_posterior_mean_with_envelope_lp_bounds_vs_rct_cate.png",
            covariates,
        )
        _plot_exp16_posterior_mean_with_envelope_by_study(
            posterior,
            figures_dir
            / (
                "exp16_prediction_posterior_mean_with_envelope_"
                "no_optimality_rate_lp_bounds_vs_rct_cate.png"
            ),
            covariates,
            show_optimality_rate=False,
        )
        _plot_exp16_posterior_mean_with_envelope_by_study(
            posterior,
            figures_dir
            / (
                "exp16_prediction_posterior_mean_with_envelope_"
                "no_optimality_rate_lp_bounds_vs_rct_cate_horizontal.png"
            ),
            covariates,
            show_optimality_rate=False,
            horizontal=True,
        )
        _plot_exp16_posterior_bounds_by_study(
            posterior,
            figures_dir
            / "exp16_prediction_posterior_envelope_lp_bounds_vs_rct_cate.png",
            covariates,
        )
