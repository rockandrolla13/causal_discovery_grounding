"""Merge and report the Naturalness method-comparison result trees.

The read-only half of the Exp28 pipeline: it fits nothing, and turns the trees
written by :mod:`experiments.naturalness_comparison` into figures and tables,
including the paper's Twins width table and figure. It is outside the runner's
resume fingerprint, so editing it never invalidates ``--resume``.

Subcommands:
    merge         combine result trees into one reportable directory
    report        write every figure and summary table for one merged run
    cap-blearner  copy a tree with its B-Learner bounds capped at Manski

``merge`` aborts when trees disagree on a shared row and reports only the
conditions every required method covers; ``--require-methods naturalness``
lets the arms rest on different condition counts (``method_coverage.csv``).
Widths are shown beside validity, since assuming more narrows any interval,
and containment in one realization is descriptive, not coverage.

``paper/reproduce_experiment2_twins.sh`` runs ``merge`` and ``report``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp25_hit_ratio_analysis import SIGN_TABLE_MODELS as EXP25_TABLE_MODELS
from experiments.exp25_hit_ratio_plots import (
    MODEL_COLORS as EXP25_MODEL_COLORS,
    MODEL_LABELS as EXP25_MODEL_LABELS,
    PAPER_FIGURE_WIDTH,
    PAPER_RC,
)
from experiments.naturalness_comparison import (
    MANSKI_CAPPED_METHODS,
    METHOD_LABELS as RUNNER_METHOD_LABELS,
    condition_covariates,
    rescore_with_manski_cap,
)
from experiments.naturalness_plotting import make_method_metric_by_beta_plot
from experiments.utils import render_latex_table


LOGGER = logging.getLogger(__name__)

#: Drawing order. Manski first because it is the reference rather than a rival.
METHOD_ORDER = ("manski", "naturalness", "balke_pearl", "blearner_exp_beta",
                "blearner_oracle", "cb_pi", "cb_ra", "cb_ipw", "cb_dr")

#: Exp25 analysis model each roster method is drawn as, for shared colours and
#: labels; ``blearner_oracle`` takes Gamma from the declared latent propensity.
EXP25_MODEL_FOR_METHOD = {
    "manski": "manski",
    "naturalness": "naturalness",
    "balke_pearl": "balke_pearl",
    "blearner_exp_beta": "blearner[exp_beta]",
    "blearner_oracle": "blearner[dgp_oracle]",
    "cb_ipw": "cb_ipw",
    "cb_dr": "cb_dr",
}

#: Display labels, Exp25's where Exp25 draws the arm. Set here, not in the
#: runner, because the runner's source is part of its resume fingerprint.
METHOD_LABELS = {
    **RUNNER_METHOD_LABELS,
    **{method: EXP25_MODEL_LABELS[model] for method, model in EXP25_MODEL_FOR_METHOD.items()},
}

#: Colours from the Exp25 palette; ``cb_pi`` and ``cb_ra`` have their own so a
#: run that requests them through ``--meta-learners`` is still drawn.
METHOD_COLORS = {
    "cb_pi": "#0e7490",
    "cb_ra": "#15803d",
    **{method: EXP25_MODEL_COLORS[model] for method, model in EXP25_MODEL_FOR_METHOD.items()},
}

METHOD_MARKERS = {
    "manski": "_", "naturalness": "o", "balke_pearl": "P",
    "blearner_exp_beta": "s", "blearner_oracle": "D",
    "cb_pi": "^", "cb_ra": "v", "cb_ipw": "<", "cb_dr": ">",
}

#: Exp28 methods in the optimal-draw figures. The ``exp(beta)`` B-Learner is
#: left out: its Gamma does not cover the declared assignment at nonzero beta.
EXP28_OPTIMAL_METHOD_ORDER = (
    "naturalness", "blearner_oracle", "cb_ipw", "cb_dr", "balke_pearl", "manski")

#: Labels of the Exp28 hit-ratio figure: the Exp25 labels every figure uses.
EXP28_OPTIMAL_METHOD_LABELS = dict(METHOD_LABELS)

#: Line styles of the Exp28 reference-hit-ratio figure.
EXP28_OPTIMAL_LINESTYLES = {
    "naturalness": "-",
    "blearner_oracle": "--",
    "cb_ipw": "-.",
    "cb_dr": (0, (5, 2, 1, 2)),
    "balke_pearl": ":",
    "manski": (0, (2, 2)),
}

#: Compact condition-level table that makes a merged Exp28 report independent
#: of the original sharded result trees.
OPTIMAL_ONLY_METRICS_FILENAME = "optimal_only_metrics.parquet"

#: The paper's Twins width table, the tabular counterpart of the
#: ``exp28_optimal_width_by_beta`` figure.
EXP28_OPTIMAL_WIDTH_TABLE = "exp28_optimal_width_by_beta.tex"

#: Level of every confidence interval on a mean over conditions. The intervals
#: are descriptive (the conditions are a designed grid), not hypothesis tests.
CONFIDENCE_LEVEL = 0.90

#: Per-method columns that make unequal coverage across methods visible.
COVERAGE_COLUMNS = ("n_conditions", "n_seeds")


#: Exit code of a ``merge`` that succeeded but dropped conditions; distinct from
#: ``1`` (an unhandled error) so a wrapper can tell "incomplete" from "broken".
DROPPED_CONDITIONS_EXIT = 3

#: Columns identifying one (condition, method) row. Every metrics table written
#: by the comparison runner carries all of them.
KEY_COLUMNS = ("experiment", "dataset", "dgp_seed", "seed", "beta", "target_site",
               "stage1_backend", "method")

#: The condition alone, without the arm that was fitted on it.
CONDITION_COLUMNS = tuple(c for c in KEY_COLUMNS if c != "method")

#: Ways ``merge`` can restrict conditions by the Naturalness discovery trigger;
#: see :func:`restrict_to_fired_conditions`.
TRIGGER_MODES = ("any", "fired", "consistent")

#: Columns two passes fitting the same arm on the same condition must agree on;
#: run-specific ones, such as ``fit_seconds`` and sensitivity diagnostics, are not.
AGREEMENT_COLUMNS = ("n_evaluation", "n_cells", "weighted_width",
                     "ate_lower", "ate_upper", "ate_reference")

#: Condition leaf layout written by ``run_comparison``:
#: ``<dataset>/dgp<draw>/seed<seed>/beta<beta>/target<site>``.
_LEAF_PATTERN = re.compile(
    r"(?P<dataset>[^/]+)/dgp(?P<dgp_seed>[^/]+)/seed(?P<seed>[^/]+)"
    r"/beta(?P<beta>[^/]+)/target(?P<target_site>[^/]+)$")


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    """Write one figure as PNG and PDF, cropped to its content, and close it.

    Args:
        fig: Figure to write.
        directory: Destination directory, created if absent.
        stem: File stem without an extension.

    Returns:
        The paths written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    plt.close(fig)
    return written


def _panels(datasets: Sequence[str], *, figsize: tuple[float, float] | None = None,
            layout: str | None = None) -> tuple[plt.Figure, np.ndarray]:
    """Lay out one panel per dataset on a shared grid.

    Args:
        datasets: Dataset names, one panel each.
        figsize: Figure size in inches; by default 4.6 by 3.6 per panel.
        layout: Matplotlib layout engine, for example ``"constrained"``, or
            ``None`` for none.

    Returns:
        The figure and a flat array of axes at least as long as ``datasets``.
    """
    columns = min(3, max(1, len(datasets)))
    rows = int(np.ceil(len(datasets) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=figsize or (4.6 * columns, 3.6 * rows),
                             squeeze=False, layout=layout)
    return fig, axes.ravel()


def _confidence_half_width(spread: pd.Series, count: pd.Series,
                           level: float = CONFIDENCE_LEVEL) -> pd.Series:
    """Half-width of a Student-t confidence interval on a mean.

    Student-t rather than normal because the condition counts behind these
    means can be small.

    Args:
        spread: Sample standard deviation per group (``ddof=1``).
        count: Number of conditions per group, aligned with ``spread``.
        level: Two-sided confidence level.

    Returns:
        The half-width per group, aligned with the inputs; ``NaN`` for groups
        with fewer than two conditions or a missing standard deviation.
    """
    counts = pd.to_numeric(count, errors="coerce")
    spreads = pd.to_numeric(spread, errors="coerce")
    degrees = counts - 1.0
    usable = (degrees > 0) & spreads.notna()
    quantile = pd.Series(np.nan, index=spreads.index, dtype=float)
    if usable.any():
        quantile.loc[usable] = stats.t.ppf(
            0.5 + level / 2.0, degrees.loc[usable].to_numpy(dtype=float))
    return quantile * spreads / np.sqrt(counts)


def _method_series(frame: pd.DataFrame, value: str,
                   level: float = CONFIDENCE_LEVEL) -> pd.DataFrame:
    """Average one metric over repetitions, leaving dataset, beta and method.

    Repetitions are data-generating draws, screening seeds and target sites;
    the spread and condition count travel with each mean.

    Args:
        frame: Metrics table.
        value: Column to average.
        level: Confidence level for ``ci_low`` and ``ci_high``.

    Returns:
        A long frame with ``dataset``, ``beta``, ``method``, ``value``, the
        condition count ``n``, the distinct screening seeds ``n_seeds``, the
        sample standard deviation ``sd`` and the confidence bounds ``ci_low``
        and ``ci_high`` (``NaN`` for a single-condition group).
    """
    grouped = frame.groupby(["dataset", "beta", "method"], observed=True)
    series = grouped.agg(value=(value, "mean"), n=(value, "size"),
                         sd=(value, "std"), n_seeds=("seed", "nunique")).reset_index()
    half = _confidence_half_width(series["sd"], series["n"], level)
    series["ci_low"] = series["value"] - half
    series["ci_high"] = series["value"] + half
    return series


def describe_method_coverage(metrics: pd.DataFrame) -> pd.DataFrame:
    """Report how many conditions and screening seeds stand behind each method.

    Args:
        metrics: Reported metrics table.

    Returns:
        One row per dataset and method with ``n_conditions``, ``n_seeds`` and
        ``seeds``, the sorted seed list rendered as a comma-separated string.
        An empty input returns an empty frame.
    """
    if metrics.empty:
        return pd.DataFrame()
    coverage = metrics.groupby(["dataset", "method"], observed=True).agg(
        n_conditions=("method", "size"),
        n_seeds=("seed", "nunique"),
        seeds=("seed", lambda s: ",".join(str(v) for v in sorted(set(s))))).reset_index()
    coverage["method_label"] = coverage.method.map(lambda m: METHOD_LABELS.get(m, m))
    return coverage.sort_values(["dataset", "method"]).reset_index(drop=True)


def _coverage_note(metrics: pd.DataFrame, level: float = CONFIDENCE_LEVEL) -> str:
    """Build the subtitle line naming the confidence level and any raggedness.

    Args:
        metrics: Reported metrics table.
        level: Confidence level drawn on the figure.

    Returns:
        A one-line note when every method spans the same number of screening
        seeds. When they differ, a second line names every arm at each seed
        depth, so an unequal-coverage figure cannot be read as an equal one.
    """
    percent = f"{level:.0%}"
    coverage = describe_method_coverage(metrics)
    if coverage.empty:
        return f"shaded band: {percent} CI of the mean over conditions"
    seeds = coverage.n_seeds
    if seeds.nunique() <= 1:
        return (f"shaded band: {percent} CI of the mean over conditions "
                f"({int(seeds.iloc[0])} screening seeds per method)")
    # Every arm at each depth is named: a pass with extra seeds deepens all the
    # arms it fits (the Naturalness pass also fits Manski and Balke-Pearl).
    rank = {method: position for position, method in enumerate(METHOD_ORDER)}
    groups = []
    for depth in sorted(seeds.unique(), reverse=True):
        members = coverage.loc[seeds.eq(depth)].sort_values(
            "method", key=lambda column: column.map(lambda m: rank.get(m, len(rank))))
        groups.append((int(depth), ", ".join(members.method_label)))
    if len(groups) == 2:
        (deep, labels), (shallow, _) = groups
        detail = f"{deep} screening seeds for {labels}; {shallow} for every other arm"
    else:
        detail = "; ".join(f"{depth} screening seeds for {labels}" for depth, labels in groups)
    return f"shaded band: {percent} CI of the mean over conditions\n{detail}"


def _draw_confidence_band(axis: plt.Axes, rows: pd.DataFrame, color: str) -> None:
    """Shade the confidence interval of a per-beta mean behind its line.

    Args:
        axis: Axis already holding the mean curve.
        rows: Rows for one method, sorted by beta, carrying ``ci_low`` and
            ``ci_high``.
        color: The method's line colour, reused at low alpha.
    """
    if not {"ci_low", "ci_high"}.issubset(rows.columns):
        return
    finite = rows.loc[rows.ci_low.notna() & rows.ci_high.notna()]
    if finite.empty:
        return
    axis.fill_between(finite.beta, finite.ci_low, finite.ci_high,
                      color=color, alpha=.15, linewidth=0, zorder=1)


def make_width_by_beta_plot(metrics: pd.DataFrame, directory: Path) -> list[Path]:
    """Plot population-weighted interval width against assignment strength.

    Read beside :func:`make_containment_by_beta_plot`.

    Args:
        metrics: Metrics table.
        directory: Figure directory.

    Returns:
        The paths written.
    """
    return _width_by_beta_figure(
        metrics, directory, "width_by_beta",
        "Interval width by assignment strength (narrower is better only if valid)")


def _width_by_beta_figure(metrics: pd.DataFrame, directory: Path, stem: str,
                          title: str | None, *,
                          x_label: str = r"assignment strength $\beta$",
                          y_label: str = "weighted interval width",
                          panel_titles: bool = True, coverage_note: bool = True,
                          figsize: tuple[float, float] | None = None,
                          legend_columns: int | None = None) -> list[Path]:
    """Draw mean ``weighted_width`` against beta in the report's by-beta style.

    Shared by :func:`make_width_by_beta_plot` and the Exp28 width figure on
    Naturalness-optimal draws: one panel per dataset under constrained layout,
    a confidence band behind each mean and Manski dashed. With the defaults the
    legend (right) and coverage note (below) fall outside ``figsize`` and the
    tight save extends the file to hold them; with ``legend_columns`` the
    legend sits below the panels, inside ``figsize``.

    Args:
        metrics: Condition-level table carrying ``dataset``, ``beta``,
            ``method``, ``seed`` and ``weighted_width``.
        directory: Figure directory.
        stem: File stem without an extension.
        title: Figure title, or ``None`` for none.
        x_label: Horizontal-axis label.
        y_label: Vertical-axis label.
        panel_titles: Whether each panel is titled with its dataset name.
        coverage_note: Whether to write the confidence-level and seed-count
            note beneath the axes; without it the caption must define the band.
        figsize: Figure size in inches; by default that of :func:`_panels`.
        legend_columns: Columns of a legend below the panels, read row by row
            in drawing order, or ``None`` for one column right of the figure.

    Returns:
        The paths written.
    """
    series = _method_series(metrics, "weighted_width")
    datasets = sorted(series.dataset.unique())
    fig, axes = _panels(datasets, figsize=figsize, layout="constrained")
    for axis, dataset in zip(axes, datasets):
        panel = series.loc[series.dataset.eq(dataset)]
        for method in METHOD_ORDER:
            rows = panel.loc[panel.method.eq(method)].sort_values("beta")
            if rows.empty:
                continue
            colour = METHOD_COLORS.get(method, "black")
            style = dict(color=colour, label=METHOD_LABELS.get(method, method))
            _draw_confidence_band(axis, rows, colour)
            if method == "manski":
                axis.plot(rows.beta, rows.value, linestyle="--", linewidth=1.4, **style)
            else:
                axis.plot(rows.beta, rows.value, marker=METHOD_MARKERS.get(method, "o"),
                          markersize=4.5, linewidth=1.6, **style)
        if panel_titles:
            axis.set_title(dataset, fontsize=9)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=.25)
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    # Below the axes, where a second line cannot collide with the title. It
    # hangs below the canvas, which constrained layout leaves to the panels,
    # and the tight save in _save extends the file to hold it.
    note = _coverage_note(metrics) if coverage_note else ""
    if note:
        fig.text(.5, 0, note, ha="center", va="top", fontsize=8, color="#475569")
    handles, labels = axes[0].get_legend_handles_labels()
    if legend_columns is None:
        # Beyond the right edge of the canvas, which constrained layout leaves
        # to the panels; the tight save in _save extends the file to hold it.
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1,
                   frameon=False)
    else:
        # An "outside" legend sits in a band that constrained layout sizes to
        # it. Matplotlib fills a legend column by column, so the entries are
        # put in column order for the rows to read in drawing order.
        rows = int(np.ceil(len(handles) / legend_columns))
        order = [row * legend_columns + column for column in range(legend_columns)
                 for row in range(rows) if row * legend_columns + column < len(handles)]
        fig.legend([handles[index] for index in order], [labels[index] for index in order],
                   loc="outside lower center", ncol=legend_columns, frameon=False)
    return _save(fig, directory, stem)


def make_containment_by_beta_plot(metrics: pd.DataFrame, directory: Path) -> list[Path]:
    """Plot the target mass whose reference effect lies inside the interval.

    Args:
        metrics: Metrics table.
        directory: Figure directory.

    Returns:
        The paths written.
    """
    series = _method_series(metrics, "reference_containment_mass")
    datasets = sorted(series.dataset.unique())
    fig, axes = _panels(datasets)
    for axis, dataset in zip(axes, datasets):
        panel = series.loc[series.dataset.eq(dataset)]
        for method in METHOD_ORDER:
            rows = panel.loc[panel.method.eq(method)].sort_values("beta")
            if rows.empty:
                continue
            colour = METHOD_COLORS.get(method, "black")
            _draw_confidence_band(axis, rows, colour)
            axis.plot(rows.beta, rows.value, marker=METHOD_MARKERS.get(method, "o"),
                      markersize=4.5, linewidth=1.6, color=colour,
                      linestyle="--" if method == "manski" else "-",
                      label=METHOD_LABELS.get(method, method))
        axis.set_ylim(-.03, 1.03)
        axis.set_title(dataset, fontsize=9)
        axis.set_xlabel(r"assignment strength $\beta$")
        axis.set_ylabel("target mass containing reference")
        axis.grid(alpha=.25)
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    fig.suptitle("Reference containment by assignment strength (descriptive, not coverage)", fontsize=11)
    note = _coverage_note(metrics)
    fig.text(.5, .01, note, ha="center", va="bottom", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, .05 + .04 * note.count("\n"), 1, .96))
    handles, labels = axes[0].get_legend_handles_labels()
    boxes = [axis.get_position() for axis in axes[:len(datasets)]]
    centre = (min(box.y0 for box in boxes) + max(box.y1 for box in boxes)) / 2
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, centre),
               ncol=1, frameon=False)
    return _save(fig, directory, "containment_by_beta")


def make_width_containment_plot(metrics: pd.DataFrame, directory: Path) -> list[Path]:
    """Plot mean width against mean containment, the headline figure.

    A point far left and low is narrow but wrong.

    Args:
        metrics: Metrics table.
        directory: Figure directory.

    Returns:
        The paths written.
    """
    # Only the means are joined: the dispersion columns of _method_series would
    # collide under the merge, and this figure draws points.
    width = _method_series(metrics, "weighted_width")[
        ["dataset", "beta", "method", "value"]].rename(columns={"value": "width"})
    hit = _method_series(metrics, "reference_containment_mass")[
        ["dataset", "beta", "method", "value"]].rename(columns={"value": "containment"})
    joined = width.merge(hit, on=["dataset", "beta", "method"], validate="one_to_one")
    datasets = sorted(joined.dataset.unique())
    fig, axes = _panels(datasets)
    for axis, dataset in zip(axes, datasets):
        panel = joined.loc[joined.dataset.eq(dataset)]
        for method in METHOD_ORDER:
            rows = panel.loc[panel.method.eq(method)]
            if rows.empty:
                continue
            axis.scatter(rows.width, rows.containment, s=32, alpha=.85,
                         marker=METHOD_MARKERS.get(method, "o"),
                         color=METHOD_COLORS.get(method, "black"),
                         label=METHOD_LABELS.get(method, method))
        axis.axhline(1., color="#94a3b8", linewidth=.9, linestyle=":")
        axis.set_ylim(-.03, 1.06)
        axis.set_title(dataset, fontsize=9)
        axis.set_xlabel("weighted interval width")
        axis.set_ylabel("target mass containing reference")
        axis.grid(alpha=.25)
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Width against validity: lower-left points are narrow because they are wrong", fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, .96))
    return _save(fig, directory, "width_vs_containment")


def make_gamma_coverage_plot(metrics: pd.DataFrame, directory: Path) -> list[Path]:
    """Plot the required sensitivity parameter against the assumed one.

    The ``exp(beta)`` B-Learner assumes ``Gamma=exp(beta)``; where the oracle
    requirement lies above that line, the data-generating process violates the
    assumption and the arm's width is not a valid result.

    Args:
        metrics: Metrics table carrying the sensitivity diagnostic columns.
        directory: Figure directory.

    Returns:
        The paths written, or an empty list when the diagnostic is absent.
    """
    if "oracle_finite_population_gamma_required" not in metrics:
        LOGGER.warning("No sensitivity diagnostic present; skipping the Gamma coverage figure")
        return []
    frame = metrics.drop_duplicates(["dataset", "dgp_seed", "seed", "beta", "target_site"])
    series = frame.groupby(["dataset", "beta"], observed=True).agg(
        required=("oracle_finite_population_gamma_required", "max"),
        assumed=("gamma_exp_beta", "max"),
        covered=("exp_beta_covers_declared_assignment", "mean")).reset_index()
    datasets = sorted(series.dataset.unique())
    fig, axes = _panels(datasets)
    for axis, dataset in zip(axes, datasets):
        panel = series.loc[series.dataset.eq(dataset)].sort_values("beta")
        axis.plot(panel.beta, panel.required, marker="o", markersize=5, linewidth=1.8,
                  color=METHOD_COLORS["blearner_oracle"], label="oracle requirement")
        axis.plot(panel.beta, panel.assumed, marker="s", markersize=4.5, linewidth=1.6,
                  color=METHOD_COLORS["blearner_exp_beta"], linestyle="--",
                  label=r"assumed $e^{\beta}$")
        uncovered = panel.loc[panel.covered < 1.]
        if not uncovered.empty:
            axis.scatter(uncovered.beta, uncovered.required, s=140, facecolors="none",
                         edgecolors=METHOD_COLORS["blearner_oracle"], linewidths=1.6,
                         label="assumption violated")
        axis.set_yscale("log")
        axis.set_title(dataset, fontsize=9)
        axis.set_xlabel(r"assignment strength $\beta$")
        axis.set_ylabel(r"sensitivity parameter $\Gamma$ (log)")
        axis.grid(alpha=.25, which="both")
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(r"Sensitivity parameter: where the requirement (solid) exceeds the assumption "
                 r"(dashed), the $e^{\beta}$ arm is misspecified", fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, .96))
    return _save(fig, directory, "gamma_coverage")


def make_sign_decision_plot(metrics: pd.DataFrame, directory: Path) -> list[Path]:
    """Plot the mass on which each arm resolves the sign, and how often wrongly.

    Pale bars are the decided mass and red bars the wrongly decided mass, each
    averaged over beta.

    Args:
        metrics: Metrics table.
        directory: Figure directory.

    Returns:
        The paths written.
    """
    # As in make_width_containment_plot, the means alone are joined.
    decided = _method_series(metrics, "sign_decision_mass")[
        ["dataset", "beta", "method", "value"]].rename(columns={"value": "decided"})
    wrong = _method_series(metrics, "wrong_sign_decision_mass")[
        ["dataset", "beta", "method", "value"]].rename(columns={"value": "wrong"})
    joined = decided.merge(wrong, on=["dataset", "beta", "method"], validate="one_to_one")
    summary = joined.groupby(["dataset", "method"], observed=True)[["decided", "wrong"]].mean().reset_index()
    datasets = sorted(summary.dataset.unique())
    fig, axes = _panels(datasets)
    for axis, dataset in zip(axes, datasets):
        panel = summary.loc[summary.dataset.eq(dataset)]
        order = [m for m in METHOD_ORDER if m in set(panel.method)]
        positions = np.arange(len(order))
        values = [float(panel.loc[panel.method.eq(m), "decided"].iloc[0]) for m in order]
        wrongs = [float(panel.loc[panel.method.eq(m), "wrong"].iloc[0]) for m in order]
        axis.bar(positions, values, color=[METHOD_COLORS.get(m, "black") for m in order], alpha=.55)
        axis.bar(positions, wrongs, color="#b91c1c", alpha=.95, width=.45, label="wrong sign")
        axis.set_xticks(positions)
        axis.set_xticklabels([METHOD_LABELS.get(m, m) for m in order], rotation=40,
                             ha="right", fontsize=7)
        axis.set_title(dataset, fontsize=9)
        axis.set_ylabel("target mass")
        axis.grid(alpha=.25, axis="y")
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    fig.suptitle("Sign decisions made (pale) and made wrongly (red)", fontsize=11)
    fig.tight_layout(rect=(0, .02, 1, .96))
    return _save(fig, directory, "sign_decisions")


def write_summary_tables(metrics: pd.DataFrame, directory: Path,
                         level: float = CONFIDENCE_LEVEL) -> list[Path]:
    """Write the per-dataset summary as CSV and LaTeX.

    One row per dataset, beta and method, with ``n_conditions`` and
    ``n_seeds``. ``width_sd`` is the spread across conditions, which does not
    shrink with more of them; the confidence bounds are the precision of the
    mean, which does.

    Args:
        metrics: Metrics table.
        directory: Destination directory.
        level: Confidence level for ``width_ci_low`` and ``width_ci_high``.

    Returns:
        The paths written: the summary CSV, the summary LaTeX, and the
        per-method coverage CSV.
    """
    directory.mkdir(parents=True, exist_ok=True)
    summary = metrics.groupby(["dataset", "beta", "method"], observed=True).agg(
        n_conditions=("weighted_width", "size"),
        n_seeds=("seed", "nunique"),
        width_mean=("weighted_width", "mean"),
        width_sd=("weighted_width", "std"),
        containment_mean=("reference_containment_mass", "mean"),
        sign_decision_mean=("sign_decision_mass", "mean"),
        wrong_sign_mean=("wrong_sign_decision_mass", "mean"),
        fallback_mean=("fallback_mass", "mean"),
        common_support_mean=("common_supported_mass", "mean")).reset_index()
    half = _confidence_half_width(summary["width_sd"], summary["n_conditions"], level)
    summary["width_ci_low"] = summary["width_mean"] - half
    summary["width_ci_high"] = summary["width_mean"] + half
    summary["method_label"] = summary.method.map(lambda m: METHOD_LABELS.get(m, m))
    written = [directory / "comparison_summary.csv"]
    summary.to_csv(written[0], index=False)
    latex = directory / "comparison_summary.tex"
    latex.write_text(summary.to_latex(index=False, float_format="%.3f", longtable=True))
    written.append(latex)
    coverage = describe_method_coverage(metrics)
    if not coverage.empty:
        coverage_path = directory / "method_coverage.csv"
        coverage.to_csv(coverage_path, index=False)
        written.append(coverage_path)
        if coverage.n_seeds.nunique() > 1:
            LOGGER.info(
                "Methods span unequal screening-seed counts (%s); this is what "
                "--require-methods naturalness is for, and method_coverage.csv "
                "records it per method",
                ", ".join(f"{row.method}={row.n_seeds}"
                          for row in coverage.itertuples()))
    return written


def _is_exp28(metrics: pd.DataFrame) -> bool:
    """Return whether a metrics table belongs to the Exp28 Twins comparison."""
    if metrics.empty or "experiment" not in metrics or "dataset" not in metrics:
        return False
    experiments = metrics["experiment"].dropna().astype(str)
    datasets = set(metrics["dataset"].dropna().astype(str))
    return experiments.str.startswith("exp28").any() and datasets == {"twins_reduced"}


def _restrict_optimal_metrics_to_report(
    optimal_metrics: pd.DataFrame, reported_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Restrict optimal-only rows to conditions retained by the merged report."""
    if optimal_metrics.empty:
        return optimal_metrics
    surviving = set(_keys(reported_metrics, CONDITION_COLUMNS))
    keep = _keys(optimal_metrics, CONDITION_COLUMNS).isin(surviving)
    return optimal_metrics.loc[keep].reset_index(drop=True)


def _report_source_roots(output_dir: Path) -> list[Path]:
    """Recover source result roots from merge provenance, with a local fallback."""
    provenance_path = output_dir / "merge_provenance.json"
    if not provenance_path.exists():
        return [output_dir]
    provenance = json.loads(provenance_path.read_text())
    roots = [Path(root) for root in provenance.get("roots", [])]
    return roots or [output_dir]


def load_report_optimal_only_metrics(
    output_dir: Path, reported_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Load or reconstruct the compact optimal-only table used by Exp28 plots.

    Reads the table ``merge`` wrote; when it is absent, rebuilds it from the
    source roots recorded in ``merge_provenance.json`` and saves it, so later
    reports are self-contained.

    Args:
        output_dir: Merged report directory.
        reported_metrics: Conditions retained in ``metrics.parquet``.

    Returns:
        Optimal-only condition/method metrics restricted to reported
        conditions, or an empty frame when the source run has no conditional
        posterior artifacts.
    """
    path = output_dir / OPTIMAL_ONLY_METRICS_FILENAME
    if path.exists():
        return _restrict_optimal_metrics_to_report(
            pd.read_parquet(path), reported_metrics
        )
    roots = _report_source_roots(output_dir)
    missing = [root for root in roots if not root.exists()]
    if missing:
        LOGGER.warning(
            "Cannot rebuild Exp28 optimal-only metrics; source roots are missing: %s",
            ", ".join(str(root) for root in missing),
        )
        return pd.DataFrame()
    optimal_metrics = _restrict_optimal_metrics_to_report(
        load_optimal_only_metrics(roots), reported_metrics
    )
    if not optimal_metrics.empty:
        optimal_metrics.to_parquet(path, index=False)
        LOGGER.info("Rebuilt %s optimal-only rows at %s", len(optimal_metrics), path)
    return optimal_metrics


def _collapse_exact_hit_ties(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, str], dict[str, str], dict[str, object], str]:
    """Collapse methods exactly tied with Naturalness in the hit-ratio figure."""
    means = frame.groupby(["beta", "method"], observed=True)[
        "reference_hit_ratio"
    ].mean().unstack("method")
    tied = ["naturalness"]
    if "naturalness" in means:
        reference = means["naturalness"].to_numpy(dtype=float)
        for method in EXP28_OPTIMAL_METHOD_ORDER:
            if method == "naturalness" or method not in means:
                continue
            values = means[method].to_numpy(dtype=float)
            if np.allclose(reference, values, atol=1e-12, rtol=0.0, equal_nan=False):
                tied.append(method)

    labels = dict(EXP28_OPTIMAL_METHOD_LABELS)
    colors = dict(METHOD_COLORS)
    line_styles: dict[str, object] = dict(EXP28_OPTIMAL_LINESTYLES)
    if len(tied) == 1:
        return (
            frame,
            EXP28_OPTIMAL_METHOD_ORDER,
            labels,
            colors,
            line_styles,
            "naturalness",
        )

    tie_method = "naturalness_exact_hit_tie"
    tied_labels = [EXP28_OPTIMAL_METHOD_LABELS.get(method, method) for method in tied]
    split = min(3, len(tied_labels))
    first_line = ", ".join(tied_labels[:split])
    second_line = ", ".join(tied_labels[split:])
    labels[tie_method] = first_line + (
        f",\n{second_line} (exact tie)" if second_line else " (exact tie)"
    )
    colors[tie_method] = METHOD_COLORS["naturalness"]
    line_styles[tie_method] = EXP28_OPTIMAL_LINESTYLES["naturalness"]
    collapsed = frame.loc[~frame["method"].isin(set(tied) - {"naturalness"})].copy()
    collapsed.loc[collapsed["method"].eq("naturalness"), "method"] = tie_method
    method_order = tuple(
        tie_method if method == "naturalness" else method
        for method in EXP28_OPTIMAL_METHOD_ORDER
        if method == "naturalness" or method not in tied
    )
    return collapsed, method_order, labels, colors, line_styles, tie_method


def make_exp28_optimal_only_plots(
    optimal_metrics: pd.DataFrame, directory: Path
) -> dict[str, list[Path]]:
    """Write Exp28 width and hit-ratio curves on Naturalness-optimal mass.

    The width figure shares the style of ``width_by_beta`` without its title,
    panel title or coverage note, which the paper caption carries instead. The
    hit-ratio figure keeps the publication style of
    :func:`~experiments.naturalness_plotting.make_method_metric_by_beta_plot`.

    Args:
        optimal_metrics: Output of :func:`load_optimal_only_metrics`, already
            restricted to the report's condition set.
        directory: Report figure directory.

    Returns:
        Artifact names mapped to PDF/PNG paths. Empty path lists are returned
        when no optimal-only metrics are available.
    """
    if optimal_metrics.empty:
        return {
            "exp28_optimal_width_by_beta": [],
            "exp28_optimal_hit_ratio_by_beta": [],
        }
    available = set(optimal_metrics["method"])
    missing = set(EXP28_OPTIMAL_METHOD_ORDER) - available
    if missing:
        raise ValueError(
            "Exp28 optimal-only report is missing methods: " f"{sorted(missing)}"
        )
    # Drawn like width_by_beta but bare (the caption states the band and seed
    # counts), at the printed width with the paper's typography and TrueType
    # fonts, which venues rejecting Type 3 fonts need, and with its legend
    # below the panel in two rows of three.
    with plt.rc_context(PAPER_RC):
        width_paths = _width_by_beta_figure(
            optimal_metrics, directory, "exp28_optimal_width_by_beta", None,
            x_label=r"$\beta$", y_label="Weighted interval width",
            panel_titles=False, coverage_note=False, figsize=(PAPER_FIGURE_WIDTH, 4.0),
            legend_columns=3)
    hit_frame, hit_order, hit_labels, hit_colors, hit_styles, emphasized = (
        _collapse_exact_hit_ties(optimal_metrics)
    )
    hit_paths = make_method_metric_by_beta_plot(
        hit_frame,
        directory / "exp28_optimal_hit_ratio_by_beta",
        value_column="reference_hit_ratio",
        y_label="Reference hit ratio",
        method_order=hit_order,
        method_labels=hit_labels,
        method_colors=hit_colors,
        method_linestyles=hit_styles,
        emphasized_method=emphasized,
        y_limits=(0.0, 1.02),
        y_ticks=np.linspace(0.0, 1.0, 6),
        confidence_level=CONFIDENCE_LEVEL,
    )
    return {
        "exp28_optimal_width_by_beta": width_paths,
        "exp28_optimal_hit_ratio_by_beta": hit_paths,
    }


def write_exp28_optimal_width_table(
    optimal_metrics: pd.DataFrame, directory: Path
) -> list[Path]:
    """Write the paper's Twins width table as a LaTeX tabular.

    One row per beta. Each method's cell is the mean over conditions of the
    weighted width on Naturalness-optimal mass and, in parentheses, the mean
    reference hit ratio: the values the ``exp28_optimal_width_by_beta`` and
    ``exp28_optimal_hit_ratio_by_beta`` figures draw. Columns and labels are
    those of the Exp25 tables.

    Args:
        optimal_metrics: Output of :func:`load_report_optimal_only_metrics`,
            already restricted to the report's condition set.
        directory: Report table directory.

    Returns:
        The path written, or an empty list when no optimal-only metrics are
        available. A method absent at some beta is shown as ``--``.
    """
    if optimal_metrics.empty:
        return []
    method_for_model = {model: method for method, model in EXP25_MODEL_FOR_METHOD.items()}
    columns = [(method_for_model[model], label) for model, label in EXP25_TABLE_MODELS]
    means = optimal_metrics.groupby(["beta", "method"], observed=True)[
        ["weighted_width", "reference_hit_ratio"]
    ].mean()
    rows = []
    for beta in sorted(optimal_metrics["beta"].unique()):
        row = {r"$\beta$": f"{beta:g}"}
        for method, label in columns:
            if (beta, method) in means.index:
                width, hit = means.loc[(beta, method)]
                row[label] = f"{width:.2f} ({hit:.2f})"
            else:
                row[label] = "--"
        rows.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / EXP28_OPTIMAL_WIDTH_TABLE
    path.write_text(
        render_latex_table(pd.DataFrame(rows), column_format="l" + "c" * len(columns))
    )
    return [path]


def build_report(output_dir: Path) -> dict[str, list[Path]]:
    """Produce every figure and table for one completed experiment run.

    Args:
        output_dir: The experiment's results root, containing
            ``metrics.parquet``.

    Returns:
        A mapping from artifact name to the paths written.

    Raises:
        FileNotFoundError: If the run has no metrics table.
    """
    metrics_path = output_dir / "metrics.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics table at {metrics_path}")
    metrics = pd.read_parquet(metrics_path)
    figures = output_dir / "figures"
    written = {
        "width_by_beta": make_width_by_beta_plot(metrics, figures),
        "containment_by_beta": make_containment_by_beta_plot(metrics, figures),
        "width_vs_containment": make_width_containment_plot(metrics, figures),
        "gamma_coverage": make_gamma_coverage_plot(metrics, figures),
        "sign_decisions": make_sign_decision_plot(metrics, figures),
        "tables": write_summary_tables(metrics, output_dir / "tables"),
    }
    if _is_exp28(metrics):
        optimal_metrics = load_report_optimal_only_metrics(output_dir, metrics)
        written.update(make_exp28_optimal_only_plots(optimal_metrics, figures))
        written["exp28_optimal_width_table"] = write_exp28_optimal_width_table(
            optimal_metrics, output_dir / "tables"
        )
    return written

def _discover(roots: Sequence[Path], name: str) -> list[Path]:
    """Return every file called ``name`` at or below each root.

    Args:
        roots: Directories to search. A root that is itself a run output
            directory and a root that contains many sharded ones both work.
        name: File name to look for.

    Returns:
        Sorted, de-duplicated paths.
    """
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"No such results root: {root}")
        found.update(p for p in root.rglob(name) if p.is_file())
    return sorted(found)


def load_metrics(roots: Sequence[Path]) -> pd.DataFrame:
    """Read and concatenate every ``metrics.parquet`` under the given roots.

    Args:
        roots: Results roots to search recursively.

    Returns:
        One row per (condition, method, source table), with a ``source_tree``
        column recording the file each row came from. Duplicated keys are left
        in place for :func:`merge_metrics` to adjudicate.

    Raises:
        FileNotFoundError: If no metrics table is found anywhere.
        ValueError: If a metrics table lacks the identifying columns.
    """
    paths = _discover(roots, "metrics.parquet")
    if not paths:
        raise FileNotFoundError(f"No metrics.parquet found under {[str(r) for r in roots]}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = set(KEY_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing identifying columns: {sorted(missing)}")
        frame = frame.copy()
        frame["source_tree"] = str(path.parent)
        frames.append(frame)
        LOGGER.info("Read %s rows from %s", len(frame), path)
    return pd.concat(frames, ignore_index=True)


def _normalise_key_value(value: object) -> str:
    """Render one key component so that equal conditions produce equal keys.

    A missing ``dgp_seed`` (datasets with one fixed draw, such as Twins) can
    arrive as ``None``, ``NaN`` or ``pd.NA`` depending on how the frame was
    built; all three map to one token.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return "<none>"
    return str(value)


def _keys(frame: pd.DataFrame, columns: Sequence[str] = KEY_COLUMNS) -> pd.Series:
    """Return a hashable string key per row over the named columns.

    Used instead of a MultiIndex because ``dgp_seed`` is nullable for datasets
    with a single fixed draw, and NaN keys do not group.
    """
    rendered = frame[list(columns)].map(_normalise_key_value)
    return rendered.agg("|".join, axis=1)


def _disagreements(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Return the duplicated keys whose results do not match.

    Args:
        frame: Concatenated metrics with duplicated keys still present.
        tolerance: Absolute tolerance for the numeric comparison.

    Returns:
        One row per offending (key, column) with both values, empty when the
        passes agree everywhere.
    """
    columns = [c for c in AGREEMENT_COLUMNS if c in frame.columns]
    records = []
    for key, group in frame.groupby(_keys(frame), sort=False):
        if len(group) < 2:
            continue
        first = group.iloc[0]
        for _, other in group.iloc[1:].iterrows():
            for column in columns:
                a, b = first[column], other[column]
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) or pd.isna(b) or not np.isclose(float(a), float(b), atol=tolerance,
                                                              rtol=0.):
                    records.append(dict(key=key, column=column, value=a, other_value=b,
                                        source_tree=first["source_tree"],
                                        other_source_tree=other["source_tree"]))
    return pd.DataFrame(records)


def merge_metrics(frame: pd.DataFrame, tolerance: float = 1e-9,
                  allow_disagreement: bool = False) -> tuple[pd.DataFrame, dict]:
    """Collapse duplicated (condition, method) rows into one, checking agreement.

    Every pass fits ``manski``, so duplication is expected. The first row read
    survives, with any missing column filled from later duplicates, since
    passes record different diagnostics (e.g. ``balke_pearl_*`` and
    ``blearner_*_gamma``).

    Args:
        frame: Output of :func:`load_metrics`.
        tolerance: Absolute tolerance for the agreement check.
        allow_disagreement: Keep the first row and log rather than raising; for
            inspecting a known-bad merge, not for reporting.

    Returns:
        The de-duplicated frame and a provenance dictionary.

    Raises:
        ValueError: If duplicated keys disagree and ``allow_disagreement`` is
            false.
    """
    disagreements = _disagreements(frame, tolerance)
    if not disagreements.empty:
        message = (f"{len(disagreements)} duplicated (condition, method) rows disagree across "
                   f"trees; the passes did not see the same data. First: "
                   f"{disagreements.iloc[0].to_dict()}")
        if not allow_disagreement:
            raise ValueError(message)
        LOGGER.warning("%s", message)
    keys = _keys(frame)
    merged = (frame.assign(_key=keys)
              .groupby("_key", sort=False, as_index=False)
              .first()
              .drop(columns="_key"))
    provenance = dict(rows_read=int(len(frame)), rows_after_dedup=int(len(merged)),
                      duplicate_rows=int(len(frame) - len(merged)),
                      disagreeing_rows=int(len(disagreements)),
                      source_trees=sorted(frame["source_tree"].unique().tolist()))
    LOGGER.info("Merged %s rows into %s (%s duplicates collapsed)",
                len(frame), len(merged), len(frame) - len(merged))
    return merged, provenance


def restrict_to_common_conditions(frame: pd.DataFrame,
                                  require_methods: Sequence[str] | None = None
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only conditions on which every required method was fitted.

    A failed fit is still present as a Manski-fallback row, so an absent row
    means the pass never attempted the condition.

    Args:
        frame: De-duplicated metrics.
        require_methods: Methods that must all be present. Defaults to every
            method observed anywhere in the frame.

    Returns:
        The restricted frame and a table of the dropped conditions with the
        methods each was missing.
    """
    required = tuple(require_methods) if require_methods else tuple(sorted(frame["method"].unique()))
    condition_keys = _keys(frame, CONDITION_COLUMNS)
    present = frame.assign(_condition=condition_keys).groupby("_condition")["method"].agg(set)
    missing = {key: sorted(set(required) - methods) for key, methods in present.items()}
    incomplete = {key: gaps for key, gaps in missing.items() if gaps}
    keep = ~condition_keys.isin(set(incomplete))
    dropped = pd.DataFrame([dict(condition=key, missing_methods=",".join(gaps))
                            for key, gaps in sorted(incomplete.items())])
    if not dropped.empty:
        LOGGER.warning("Dropping %s of %s conditions that lack at least one method",
                       len(dropped), len(present))
    return frame.loc[keep].reset_index(drop=True), dropped


def _leaf_metadata(leaf: Path) -> dict:
    """Recover a condition's identifying columns from its directory path.

    Args:
        leaf: A condition directory ending ``.../betaB/targetT``.

    Returns:
        The condition columns, with ``experiment`` and ``stage1_backend`` taken
        from the nearest ancestor ``identity.json`` when one exists.

    Raises:
        ValueError: If the path does not match the run's leaf layout.
    """
    match = _LEAF_PATTERN.search(leaf.as_posix())
    if match is None:
        raise ValueError(f"Not a condition leaf directory: {leaf}")
    parts = match.groupdict()
    draw = parts["dgp_seed"]
    metadata = dict(dataset=parts["dataset"],
                    dgp_seed=None if draw == "None" else int(draw),
                    seed=int(parts["seed"]), beta=float(parts["beta"]),
                    target_site=parts["target_site"],
                    experiment=None, stage1_backend=None)
    for parent in leaf.parents:
        identity = parent / "identity.json"
        if identity.exists():
            config = json.loads(identity.read_text()).get("config", {})
            metadata["experiment"] = config.get("experiment_name")
            metadata["stage1_backend"] = config.get("stage1_backend")
            break
    return metadata


def _condition_tuple(metadata: dict) -> tuple[str, ...]:
    """Return a normalized condition key for one leaf metadata dictionary."""
    return tuple(_normalise_key_value(metadata[column]) for column in CONDITION_COLUMNS)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Return a finite weighted average, or NaN when no positive weight exists."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid = np.isfinite(numeric_values) & np.isfinite(numeric_weights) & numeric_weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(numeric_values.loc[valid], weights=numeric_weights.loc[valid]))


def _cell_columns(posterior: pd.DataFrame, bounds: pd.DataFrame, path: Path) -> list[str]:
    """Infer conditioning-cell columns shared by posterior and scored bounds."""
    columns = [column for column in posterior.columns if column in bounds.columns]
    if not columns:
        raise ValueError(f"Cannot infer conditioning cells for {path}")
    return columns


def _deduplicate_condition_bounds(
    bounds: pd.DataFrame, cell_columns: Sequence[str], condition: tuple[str, ...]
) -> pd.DataFrame:
    """Collapse cross-pass duplicate cell bounds after checking agreement."""
    key = [*cell_columns, "method"]
    for group_key, group in bounds.groupby(key, observed=True, dropna=False, sort=False):
        if len(group) < 2:
            continue
        widths = pd.to_numeric(group["width"], errors="coerce").to_numpy(dtype=float)
        hits = group["contains_reference"].astype(bool).to_numpy()
        if not np.allclose(widths, widths[0], atol=1e-9, rtol=0.0, equal_nan=True) \
                or not np.all(hits == hits[0]):
            raise ValueError(
                "Duplicated optimal-only comparator cells disagree for "
                f"condition={condition}, cell={group_key}"
            )
    return bounds.drop_duplicates(key).reset_index(drop=True)


def load_optimal_only_metrics(roots: Sequence[Path]) -> pd.DataFrame:
    """Build matched Exp28 metrics conditional on optimal Naturalness draws.

    Each cell is weighted by its target evaluation mass times its posterior
    probability of an optimal Naturalness LP solve. Naturalness uses its
    posterior-mean endpoints among optimal draws; every comparator averages
    its fitted cell intervals with the same weights. Conditions with zero
    optimal mass have no conditional estimand and contribute no rows.

    Args:
        roots: Naturalness and comparator result roots searched recursively.

    Returns:
        One row per positive-mass condition and method, containing conditional
        ``weighted_width``, descriptive ``reference_hit_ratio``, and the
        condition's ``optimal_draw_mass``. Empty when no conditional posterior
        artifact is present.

    Raises:
        FileNotFoundError: If a posterior artifact has no sibling scored bounds.
        ValueError: If cell keys, evaluation weights, or duplicated comparator
            results are missing or inconsistent.
    """
    posterior_paths = _discover(roots, "posterior_conditional_on_optimal.parquet")
    if not posterior_paths:
        return pd.DataFrame()

    bounds_by_condition: dict[tuple[str, ...], list[Path]] = {}
    for path in _discover(roots, "bounds_scored.parquet"):
        condition = _condition_tuple(_leaf_metadata(path.parent))
        bounds_by_condition.setdefault(condition, []).append(path)

    rows: list[dict] = []
    seen_conditions: set[tuple[str, ...]] = set()
    comparator_methods = set(EXP28_OPTIMAL_METHOD_ORDER) - {"naturalness"}
    for posterior_path in posterior_paths:
        metadata = _leaf_metadata(posterior_path.parent)
        condition = _condition_tuple(metadata)
        if condition in seen_conditions:
            raise ValueError(f"Duplicate conditional posterior for condition {condition}")
        seen_conditions.add(condition)

        scored_path = posterior_path.parent / "bounds_scored.parquet"
        if not scored_path.exists():
            raise FileNotFoundError(
                f"Conditional posterior has no sibling scored bounds: {scored_path}"
            )
        posterior = pd.read_parquet(posterior_path)
        naturalness_bounds = pd.read_parquet(scored_path)
        cell_columns = _cell_columns(posterior, naturalness_bounds, posterior_path)
        required_posterior = {
            *cell_columns,
            "lp_posterior_prob_optimal",
            "lower_mean",
            "upper_mean",
            "width_mean",
        }
        missing_posterior = required_posterior.difference(posterior.columns)
        if missing_posterior:
            raise ValueError(
                f"{posterior_path} is missing columns: {sorted(missing_posterior)}"
            )
        weight_columns = [
            *cell_columns,
            "evaluation_weight",
            "reference_effect",
        ]
        missing_weights = set(weight_columns).difference(naturalness_bounds.columns)
        if missing_weights:
            raise ValueError(
                f"{scored_path} is missing columns: {sorted(missing_weights)}"
            )
        weights = naturalness_bounds.loc[
            naturalness_bounds["method"].eq("manski"), weight_columns
        ]
        if weights.duplicated(cell_columns).any():
            raise ValueError(f"Duplicate evaluation weights in {scored_path}")
        optimal = posterior.merge(
            weights, on=cell_columns, how="left", validate="one_to_one"
        )
        if optimal[["evaluation_weight", "reference_effect"]].isna().any().any():
            raise ValueError(f"Missing evaluation weights or references in {posterior_path}")
        optimal["selection_weight"] = (
            pd.to_numeric(optimal["evaluation_weight"], errors="coerce")
            * pd.to_numeric(optimal["lp_posterior_prob_optimal"], errors="coerce")
        )
        total_evaluation_weight = float(optimal["evaluation_weight"].sum())
        total_selection_weight = float(optimal["selection_weight"].sum())
        if total_evaluation_weight <= 0 or total_selection_weight <= 0:
            continue
        positive = optimal.loc[optimal["selection_weight"].gt(0)].copy()
        positive["contains_reference"] = (
            pd.to_numeric(positive["lower_mean"], errors="coerce")
            <= pd.to_numeric(positive["reference_effect"], errors="coerce") + 1e-10
        ) & (
            pd.to_numeric(positive["upper_mean"], errors="coerce")
            >= pd.to_numeric(positive["reference_effect"], errors="coerce") - 1e-10
        )
        common = dict(
            metadata,
            optimal_draw_mass=total_selection_weight / total_evaluation_weight,
            n_cells=int(len(optimal)),
            n_optimal_cells=int(len(positive)),
        )
        rows.append(
            dict(
                common,
                method="naturalness",
                weighted_width=_weighted_average(
                    positive["width_mean"], positive["selection_weight"]
                ),
                reference_hit_ratio=_weighted_average(
                    positive["contains_reference"].astype(float),
                    positive["selection_weight"],
                ),
            )
        )

        condition_paths = bounds_by_condition.get(condition, [])
        if not condition_paths:
            raise FileNotFoundError(f"No scored bounds found for condition {condition}")
        condition_bounds = pd.concat(
            [pd.read_parquet(path) for path in condition_paths], ignore_index=True
        )
        required_bounds = {
            *cell_columns,
            "method",
            "width",
            "contains_reference",
        }
        missing_bounds = required_bounds.difference(condition_bounds.columns)
        if missing_bounds:
            raise ValueError(
                f"Scored bounds for condition {condition} are missing columns: "
                f"{sorted(missing_bounds)}"
            )
        condition_bounds = condition_bounds.loc[
            condition_bounds["method"].isin(comparator_methods),
            [*cell_columns, "method", "width", "contains_reference"],
        ]
        condition_bounds = _deduplicate_condition_bounds(
            condition_bounds, cell_columns, condition
        )
        selected_weights = positive[cell_columns + ["selection_weight"]]
        for method in EXP28_OPTIMAL_METHOD_ORDER:
            if method == "naturalness":
                continue
            method_bounds = condition_bounds.loc[
                condition_bounds["method"].eq(method),
                [*cell_columns, "width", "contains_reference"],
            ]
            if method_bounds.empty:
                continue
            matched = selected_weights.merge(
                method_bounds,
                on=cell_columns,
                how="left",
                validate="one_to_one",
            )
            if matched[["width", "contains_reference"]].isna().any().any():
                raise ValueError(
                    f"Method {method!r} lacks an optimal-weight cell for {condition}"
                )
            rows.append(
                dict(
                    common,
                    method=method,
                    weighted_width=_weighted_average(
                        matched["width"], matched["selection_weight"]
                    ),
                    reference_hit_ratio=_weighted_average(
                        matched["contains_reference"].astype(float),
                        matched["selection_weight"],
                    ),
                )
            )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["dataset", "beta", "seed", "target_site", "method"]
    ).reset_index(drop=True)


def load_trigger_status(roots: Sequence[Path]) -> pd.DataFrame:
    """Return the Naturalness discovery outcome for every condition under ``roots``.

    An empty ``selected_choices.parquet`` in a condition leaf means no
    ``(j, epsilon, tau)`` triple satisfied ``I_j(epsilon)`` and ``D_j(tau)``,
    so the linear program can only return the Manski bound. A primary candidate
    whose measured independence deviation exceeds its tolerance ``2c`` leaves
    the program infeasible, since it constrains
    ``theta_{*.x z_j'} - theta_{*.x z_j''} <= 2c`` while pinning the observed
    arm to ``p(y | x, z)``. Both flags come from the screening output alone, so
    filtering on them does not condition on the bound coming out well.

    Args:
        roots: Results roots to search recursively. A root with no Naturalness
            arm contributes no rows.

    Returns:
        One row per condition carrying :data:`CONDITION_COLUMNS` plus
        ``trigger_fired``, ``tolerance_covers_deviation``,
        ``selected_covariate``, ``two_c`` and ``i_deviation``. Empty when no
        screening output was found.
    """
    rows = []
    for path in _discover(roots, "selected_choices.parquet"):
        choices = pd.read_parquet(path)
        record = dict(_leaf_metadata(path.parent), trigger_fired=not choices.empty,
                      tolerance_covers_deviation=False, selected_covariate=None,
                      two_c=np.nan, i_deviation=np.nan)
        if not choices.empty:
            primary = choices[choices["is_primary_choice"]] \
                if "is_primary_choice" in choices else choices
            primary = (primary if len(primary) else choices).iloc[0]
            two_c = float(primary.get("two_c", np.nan))
            deviation = float(primary.get("i_deviation", np.nan))
            # NaN on either side compares False, which excludes the condition
            # rather than silently admitting one whose tolerance is unknown.
            record.update(selected_covariate=primary.get("covariate"), two_c=two_c,
                          i_deviation=deviation,
                          tolerance_covers_deviation=bool(two_c >= deviation))
        rows.append(record)
    if rows:
        LOGGER.info("Read the Naturalness trigger for %s conditions", len(rows))
    return pd.DataFrame(rows)


def restrict_to_fired_conditions(frame: pd.DataFrame, status: pd.DataFrame,
                                 mode: str = "any") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only conditions on which the Naturalness discovery trigger fired.

    Every method is kept on a surviving condition, so the comparison stays
    paired.

    Args:
        frame: De-duplicated metrics covering any number of methods.
        status: Trigger table from :func:`load_trigger_status`.
        mode: One of :data:`TRIGGER_MODES`. ``any`` returns ``frame``
            unchanged, ``fired`` keeps conditions whose screen selected a
            candidate, and ``consistent`` additionally requires ``2c`` to cover
            the selected candidate's measured deviation.

    Returns:
        The restricted frame and a table of the excluded conditions naming why
        each was excluded.

    Raises:
        ValueError: If ``mode`` is not one of :data:`TRIGGER_MODES`, or if a
            restricting mode was asked for but no screening output was found.
    """
    if mode not in TRIGGER_MODES:
        raise ValueError(f"Unknown trigger mode {mode!r}; expected one of {TRIGGER_MODES}")
    if mode == "any":
        return frame, pd.DataFrame()
    if status.empty:
        raise ValueError("No selected_choices.parquet was found under the merged roots, so "
                         "the Naturalness trigger cannot be evaluated. Merge a tree that "
                         "includes the Naturalness arm, or pass --naturalness-trigger any.")
    status_keys = _keys(status, CONDITION_COLUMNS)
    fired = status["trigger_fired"].astype(bool)
    keep_flag = fired if mode == "fired" else fired & status["tolerance_covers_deviation"].astype(bool)
    kept, known = set(status_keys[keep_flag]), set(status_keys)
    condition_keys = _keys(frame, CONDITION_COLUMNS)
    reasons = []
    for key in sorted(set(condition_keys) - kept):
        if key not in known:
            reason = "no screening output for this condition"
        elif not fired[status_keys == key].iloc[0]:
            reason = "discovery selected no candidate"
        else:
            reason = "2c below the selected candidate's measured deviation"
        reasons.append(dict(condition=key, excluded_because=reason))
    excluded = pd.DataFrame(reasons)
    if not excluded.empty:
        LOGGER.warning("Naturalness trigger filter %r excludes %s of %s conditions",
                       mode, len(excluded), condition_keys.nunique())
    return frame.loc[condition_keys.isin(kept)].reset_index(drop=True), excluded


def merge_trees(roots: Sequence[Path], output_dir: Path,
                require_methods: Sequence[str] | None = None, keep_all: bool = False,
                tolerance: float = 1e-9, allow_disagreement: bool = False,
                naturalness_trigger: str = "any") -> dict:
    """Merge every result tree under ``roots`` into one reportable directory.

    Writes ``metrics.parquet`` (the conditions every required method covers,
    read by ``report``), ``metrics_all.parquet`` (everything merged),
    ``optimal_only_metrics.parquet`` (Exp28 conditions with positive
    Naturalness-optimal posterior mass), ``merge_provenance.json``, and CSVs of
    any dropped or excluded conditions.

    Args:
        roots: Results roots to merge.
        output_dir: Directory for the merged tables; created if absent, and
            existing merged tables are overwritten.
        require_methods: Methods that a condition must have to be reported.
        keep_all: Skip the common-condition restriction.
        tolerance: Absolute tolerance for the cross-tree agreement check.
        allow_disagreement: Log rather than raise when trees disagree.
        naturalness_trigger: One of :data:`TRIGGER_MODES`; anything but
            ``any`` keeps only conditions where Naturalness discovery fired,
            with every method, and writes ``excluded_conditions.csv``.

    Returns:
        The provenance dictionary that was written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged, provenance = merge_metrics(load_metrics(roots), tolerance, allow_disagreement)
    merged.to_parquet(output_dir / "metrics_all.parquet", index=False)
    if keep_all:
        reported, dropped = merged, pd.DataFrame()
    else:
        reported, dropped = restrict_to_common_conditions(merged, require_methods)
    reported, excluded = restrict_to_fired_conditions(
        reported, load_trigger_status(roots), naturalness_trigger)
    reported.to_parquet(output_dir / "metrics.parquet", index=False)
    if not dropped.empty:
        dropped.to_csv(output_dir / "dropped_conditions.csv", index=False)
    if not excluded.empty:
        excluded.to_csv(output_dir / "excluded_conditions.csv", index=False)
    optimal_path = output_dir / OPTIMAL_ONLY_METRICS_FILENAME
    optimal_metrics = pd.DataFrame()
    if _is_exp28(reported):
        optimal_metrics = _restrict_optimal_metrics_to_report(
            load_optimal_only_metrics(roots), reported
        )
        if not optimal_metrics.empty:
            optimal_metrics.to_parquet(optimal_path, index=False)
    if optimal_metrics.empty and optimal_path.exists():
        optimal_path.unlink()
    provenance.update(
        roots=[str(r) for r in roots], output_dir=str(output_dir),
        methods=sorted(merged["method"].unique().tolist()),
        required_methods=sorted(require_methods) if require_methods else None,
        conditions_all=int(_keys(merged, CONDITION_COLUMNS).nunique()),
        conditions_reported=int(_keys(reported, CONDITION_COLUMNS).nunique()) if len(reported) else 0,
        conditions_dropped=int(len(dropped)), rows_reported=int(len(reported)),
        naturalness_trigger=naturalness_trigger,
        conditions_excluded_by_trigger=int(len(excluded)),
        optimal_only_rows=int(len(optimal_metrics)),
        optimal_only_conditions=(
            int(_keys(optimal_metrics, CONDITION_COLUMNS).nunique())
            if not optimal_metrics.empty else 0
        ))
    (output_dir / "merge_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    return provenance


def cap_blearner_tree(source: Path, output: Path) -> dict:
    """Copy a result tree with its B-Learner arms re-scored under the Manski cap.

    For a tree written without the cap; nothing is refitted. The source is
    copied whole to ``output`` and only the copy changes: in every condition
    leaf the capped arms' rows of ``bounds_scored.parquet`` and
    ``completion.json`` are replaced through
    :func:`~experiments.naturalness_comparison.rescore_with_manski_cap`, and the
    shard's ``metrics.parquet`` rows follow. The merge's agreement check refuses
    to mix the copy with an uncapped tree holding the same arms.

    Args:
        source: A result root written by :mod:`experiments.naturalness_comparison`.
        output: Fresh directory for the re-scored copy.

    Returns:
        The provenance also written to ``manski_cap_provenance.json``: the rule
        and, per capped arm, the conditions and cells re-scored, capped and
        fallen back as disjoint.

    Raises:
        FileExistsError: If ``output`` already exists.
        ValueError: If a leaf fails the reproduction check of
            :func:`~experiments.naturalness_comparison.rescore_with_manski_cap`,
            or a shard's metrics lack exactly one row for a re-scored arm.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to write into an existing directory: {output}")
    shutil.copytree(source, output)
    records = []
    for identity_path in sorted(output.rglob("identity.json")):
        shard = identity_path.parent
        config = SimpleNamespace(**json.loads(identity_path.read_text())["config"])
        metrics_path = shard / "metrics.parquet"
        metrics = pd.read_parquet(metrics_path)
        condition_keys = _keys(metrics, CONDITION_COLUMNS)
        changed = False
        for scored_path in sorted(shard.rglob("bounds_scored.parquet")):
            leaf = scored_path.parent
            metadata = _leaf_metadata(leaf)
            covariates, _ = condition_covariates(config, metadata["dataset"])
            rescored, arm_metrics, counts = rescore_with_manski_cap(leaf, covariates)
            if not arm_metrics:
                continue
            rescored.to_parquet(scored_path, index=False)
            completion_path = leaf / "completion.json"
            completion = json.loads(completion_path.read_text())
            for entry in completion["metrics"]:
                entry.update(arm_metrics.get(entry["method"], {}))
            completion_path.write_text(json.dumps(completion, indent=2, default=str))
            key = "|".join(_condition_tuple(metadata))
            for method, metric in arm_metrics.items():
                row = condition_keys.eq(key) & metrics["method"].eq(method)
                if row.sum() != 1:
                    raise ValueError(f"{metrics_path} has {int(row.sum())} rows for {method} at {key}")
                for column, value in metric.items():
                    metrics.loc[row, column] = value
            records.extend(dict(count, **metadata) for count in counts)
            changed = True
        if changed:
            metrics.to_parquet(metrics_path, index=False)
    table = pd.DataFrame(records)
    per_method = {} if table.empty else {
        method: dict(conditions=int(len(group)), cells=int(group.cells.sum()),
                     cells_capped=int(group.cells_capped.sum()),
                     cells_disjoint=int(group.cells_disjoint.sum()))
        for method, group in table.groupby("method")}
    provenance = dict(
        source=str(source), output=str(output),
        rule=("lower = max(fitted lower, Manski lower) and upper = min(fitted upper, Manski "
              "upper) in every cell; a fitted interval disjoint from Manski falls back to Manski"),
        arms=per_method)
    (output / "manski_cap_provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance



def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for merging, reporting and capping.

    Returns:
        A parser whose ``command`` attribute is ``"merge"``, ``"report"`` or
        ``"cap-blearner"``.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser(
        "merge", help="Merge several result trees into one reportable directory")
    merge.add_argument("roots", nargs="+", help="Results roots to merge, searched recursively")
    merge.add_argument("--output-dir", required=True, help="Directory to write the merged tables")
    merge.add_argument("--require-methods", nargs="+", default=None,
                       help="Methods a condition must have to be reported "
                            "(default: every method present anywhere)")
    merge.add_argument("--keep-all", action="store_true",
                       help="Skip the common-condition restriction. Widths then average over "
                            "different condition sets per method and are not comparable.")
    merge.add_argument("--naturalness-trigger", choices=TRIGGER_MODES, default="any",
                       help="Restrict reported conditions by the Naturalness discovery "
                            "trigger. 'fired' keeps conditions whose screen selected a "
                            "candidate; 'consistent' also requires 2c to cover that "
                            "candidate's measured deviation. Every method is kept on a "
                            "surviving condition (default: any)")
    merge.add_argument("--tolerance", type=float, default=1e-9)
    merge.add_argument("--allow-disagreement", action="store_true",
                       help="Log rather than fail when two trees disagree on a shared row.")

    report = subparsers.add_parser("report", help="Build figures and tables for one run")
    report.add_argument("output_dir", help="Experiment results root containing metrics.parquet")

    cap = subparsers.add_parser(
        "cap-blearner",
        help="Copy a pre-cap result tree with its B-Learner bounds intersected with Manski")
    cap.add_argument("source", help="Result root written before the runner applied the cap")
    cap.add_argument("--output-dir", required=True, help="Fresh directory for the re-scored copy")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Merge, report or cap result trees, according to the subcommand.

    A single bare directory argument is shorthand for ``report <dir>``, the
    form recorded run manifests use. It requires exactly one argument, so a
    ``merge`` call, which always has ``--output-dir``, cannot match it.

    Args:
        argv: Command line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, or :data:`DROPPED_CONDITIONS_EXIT` when a merge
        succeeded but the common-condition restriction dropped at least one
        condition.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if (len(argv) == 1 and not argv[0].startswith("-")
            and argv[0] not in {"merge", "report", "cap-blearner"}):
        argv.insert(0, "report")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)

    if args.command == "merge":
        provenance = merge_trees([Path(r) for r in args.roots], Path(args.output_dir),
                                 args.require_methods, args.keep_all, args.tolerance,
                                 args.allow_disagreement, args.naturalness_trigger)
        LOGGER.info("Merged %s conditions (%s dropped) over methods %s into %s",
                    provenance["conditions_reported"], provenance["conditions_dropped"],
                    provenance["methods"], provenance["output_dir"])
        return DROPPED_CONDITIONS_EXIT if provenance["conditions_dropped"] else 0

    if args.command == "cap-blearner":
        provenance = cap_blearner_tree(Path(args.source), Path(args.output_dir))
        LOGGER.info("Capped B-Learner arms at Manski into %s: %s",
                    provenance["output"], provenance["arms"])
        return 0

    written = build_report(Path(args.output_dir))
    for name, paths in written.items():
        LOGGER.info("%s -> %s", name, ", ".join(str(p) for p in paths) or "skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
