"""Figures for the Exp25 sign-identification and reference-hit analysis.

:func:`make_all_plots` draws the tables of :mod:`experiments.exp25_hit_ratio_analysis`
into ``<output_dir>/figures``: the correct-sign rate against decisiveness, where each
B-Learner Gamma rule lands, the five sign-verdict cells and their cross-tabulation, the
width-containment frontiers, the rates against beta, the paired hit concordance with
Naturalness and the spread across replications. :func:`make_mechanism_arm_plots` draws
the mechanism-arm (ablation) figures. The command line runs the full analysis
(:func:`experiments.exp25_hit_ratio_analysis.run_analysis`) on one backend leaf.

Nothing here is calibrated to a containment guarantee: the abstention axis is
``decisiveness_rate``, not coverage, and the hit ratio is containment of a noisy point
estimate from a finite RCT, not a nominal interval level. Width figures use the common
support (strata every method can bound). The Monte Carlo replication is part of the
stratum key, so every rate pools all replications. ``paper/reproduce_experiment1_manylabs.sh``
and ``paper/reproduce_ablations.sh`` draw the paper's figures through this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy.stats import binomtest

from causal_grounding.paper_grids import PAPER_BETAS
from experiments.exp25_hit_ratio_analysis import (
    EXP25_KEY_COLUMNS,
    MODEL_ORDER,
    SIGN_ORDER,
    _frontier_bootstrap_intervals,
    common_support_frontier,
    run_analysis,
)
from experiments.interval_stats import wilson_interval


NATURALNESS_MODEL = "naturalness"
# Matplotlib mathtext symbols (no LaTeX needed), held as constants so f-strings that
# also contain escapes such as a newline can interpolate them.
BETA_SYMBOL = r"$\beta$"
# Mathtext ignores ``fontweight``, so a bold header needs a bold glyph.
BOLD_BETA_SYMBOL = r"$\boldsymbol{\beta}$"
GAMMA_SYMBOL = r"$\Gamma$"
BLEARNER_MODEL_PREFIX = "blearner["
# Leaf-label prefix (``benchmarks.gamma_output_label``) of a numeric Gamma on the swept
# sensitivity grid, as opposed to a named rule such as ``oracle``.
BLEARNER_FIXED_PREFIX = "fixed_"
FRONTIER_CORRECT = "sign_correct_rate"
FRONTIER_DECISIVENESS = "decisiveness_rate"
# Bootstrap bounds written by common_support_frontier, in x-then-y order.
FRONTIER_INTERVAL_COLUMNS = (
    "mean_width_ci_lower",
    "mean_width_ci_upper",
    "hit_ratio_ci_lower",
    "hit_ratio_ci_upper",
)
# Across-replication standard deviations written by common_support_frontier
# with ``replication_spread``, in x-then-y order.
FRONTIER_SD_COLUMNS = ("mean_width_sd", "hit_ratio_sd")
# Error-bar kinds on the width-containment frontier figures, and the file-stem
# suffix each kind is written under.
FRONTIER_SPREAD_SUFFIXES = {"ci": "", "sd": "_sd"}
# Confounding levels drawn in the per-study frontier, when present; below them
# Naturalness lies outside the swept Gamma grid, so no matched comparison exists.
STUDY_FRONTIER_BETAS = PAPER_BETAS
# Vector output by default: the panels are sharp at any zoom and in print.
DEFAULT_PLOT_FORMATS = ("pdf",)
# Resolution for raster output, and for the raster parts embedded in vector
# output, where a higher value costs little because only the heatmaps rasterize.
RASTER_PLOT_DPI = 220
VECTOR_RASTER_DPI = 600
VECTOR_PLOT_FORMATS = frozenset({"pdf", "svg", "eps", "ps"})
FIGURES_DIRNAME = "figures"
MODEL_LABELS = {
    "naturalness": "Naturalness",
    "blearner[dgp_oracle]": f"B-Learner ({GAMMA_SYMBOL} dgp-oracle)",
    "blearner[oracle]": f"B-Learner ({GAMMA_SYMBOL} oracle)",
    "blearner[exp_beta]": f"B-Learner ({GAMMA_SYMBOL} exp-{BETA_SYMBOL})",
    "cb_ipw": "CB-IPW",
    "cb_dr": "CB-DR",
    "balke_pearl": "Balke-Pearl",
    "manski": "Manski",
    "transport": "Transport",
}
MODEL_SHORT_LABELS = {
    "naturalness": "Nat",
    "blearner[dgp_oracle]": "BL dgp",
    "blearner[oracle]": "BL orc",
    "blearner[exp_beta]": "BL exp",
    "cb_ipw": "IPW",
    "cb_dr": "DR",
    "balke_pearl": "BP",
    "manski": "Manski",
    "transport": "Trans",
}
MODEL_COLORS = {
    "naturalness": "tab:purple",
    "blearner[dgp_oracle]": "tab:red",
    "blearner[oracle]": "#7f1d1d",
    "blearner[exp_beta]": "#f4977a",
    "cb_ipw": "tab:blue",
    "cb_dr": "tab:green",
    # Exp28 draws Balke-Pearl violet, too close to Naturalness here.
    "balke_pearl": "tab:orange",
    "manski": "#64748b",
    # Colour-blind distinguishable from Naturalness and Balke-Pearl (worst pairwise Delta E 14.4).
    "transport": "tab:cyan",
}
MODEL_MARKERS = {
    "naturalness": "o",
    "blearner[dgp_oracle]": "s",
    "blearner[oracle]": "D",
    "blearner[exp_beta]": "^",
    "cb_ipw": "v",
    "cb_dr": "P",
    "balke_pearl": "X",
    "manski": "d",
    "transport": "h",
}
FALLBACK_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")
FALLBACK_COLORS = (
    "tab:orange",
    "tab:brown",
    "tab:olive",
    "tab:cyan",
    "tab:pink",
    "0.45",
)
PAIR_OUTCOME_ORDER = (
    "both_hit",
    "naturalness_only",
    "competitor_only",
    "neither",
)
PAIR_OUTCOME_LABELS = {
    "both_hit": "both hit",
    "naturalness_only": "Naturalness only",
    "competitor_only": "competitor only",
    "neither": "neither",
}


def _save_figure(
    figure: plt.Figure,
    output_stem: Path,
    plot_formats: Sequence[str],
) -> list[Path]:
    """Save one figure in every requested format and close it.

    Vector formats embed TrueType (not Type 3) fonts and any raster content, such as
    heatmap cells, at :data:`VECTOR_RASTER_DPI`. Each file is cropped to the drawn
    content with a thin margin.

    Args:
        figure: Matplotlib figure to persist.
        output_stem: Output path without a file extension.
        plot_formats: File extensions accepted by Matplotlib.

    Returns:
        Paths written in the requested order.
    """

    paths: list[Path] = []
    for plot_format in plot_formats:
        path = output_stem.with_suffix(f".{plot_format}")
        vector = plot_format in VECTOR_PLOT_FORMATS
        with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            figure.savefig(
                path,
                dpi=VECTOR_RASTER_DPI if vector else RASTER_PLOT_DPI,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        paths.append(path)
    plt.close(figure)
    return paths
PAIR_OUTCOME_COLORS = {
    "both_hit": "#3c6e9f",
    "naturalness_only": "tab:purple",
    "competitor_only": "tab:orange",
    "neither": "0.78",
}
# Shared legends follow the Exp16 convention: across the top of the figure,
# wrapped into two rows. Figures carry no title.
LEGEND_ROW_COUNT = 2
LEGEND_BAND_INCHES = 0.68
# Grids whose panels share fixed limits carry one label per axis, sized up
# because it names the whole grid rather than one panel.
SHARED_AXIS_LABEL_FONTSIZE = 16
# Row, column and beta-panel headers of the report's frontier grids.
FRONTIER_HEADER_FONTSIZE = 14
# Legend text of the report's frontier figures.
FRONTIER_LEGEND_TEXT_STYLE = {"size": 13, "weight": "bold"}
# Manuscript figures are drawn at the printed width (inches) with PAPER_RC's print-sized
# TrueType fonts, so LaTeX includes them at about natural size.
PAPER_FIGURE_WIDTH = 6.5
PAPER_RC = {
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "figure.labelsize": 10.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
# Marker area (points squared) of a method in the manuscript's frontier figures.
PAPER_MARKER_AREA = 36.0
# Sign-verdict panels read best with the RCT verdict on rows and a fixed order.
SIGN_LABELS = {
    "negative": "neg",
    "inconclusive": "incl",
    "positive": "pos",
}
# Count column, label and colour of the five verdict cells that partition every valid,
# reference-eligible stratum, in the stacked bars' left-to-right order.
SIGN_CELL_SPECS = (
    ("n_sign_correct", "sign identified, correct", "#2f8f5b"),
    ("n_sign_reversal", "sign identified, reversed", "#b2182b"),
    ("n_sign_undetermined", "sign identified, RCT inconclusive", "#e08a3c"),
    ("n_bound_abstained", "bound contains zero, RCT decisive", "#6a8caf"),
    ("n_agree_inconclusive", "both contain zero", "0.80"),
)


def _fixed_gamma_value(rule: str) -> float | None:
    """Recover the numeric Gamma from a ``fixed_<value>`` leaf label (``fixed_2p5`` is 2.5).

    Args:
        rule: Bracketed rule name from :func:`blearner_gamma_rule`.

    Returns:
        The Gamma value, or ``None`` when the label is not a numeric one.
    """

    if not rule.startswith(BLEARNER_FIXED_PREFIX):
        return None
    try:
        return float(rule[len(BLEARNER_FIXED_PREFIX):].replace("p", "."))
    except ValueError:
        return None


def model_label(model: str) -> str:
    """Return the display label for one model column value.

    Args:
        model: Model identifier such as ``naturalness`` or
            ``blearner[exp_beta]``.

    Returns:
        The :data:`MODEL_LABELS` entry; for another B-Learner variant, a label naming
        its Gamma rule or numeric Gamma; otherwise the identifier itself.
    """

    name = str(model)
    if name in MODEL_LABELS:
        return MODEL_LABELS[name]
    rule = blearner_gamma_rule(name)
    if rule is not None:
        gamma = _fixed_gamma_value(rule)
        if gamma is not None:
            return f"B-Learner ({GAMMA_SYMBOL} = {gamma:g})"
        return f"B-Learner ({GAMMA_SYMBOL} {rule.replace('_', '-')})"
    return name


def model_short_label(model: str) -> str:
    """Return a compact model label for dense tick axes.

    Args:
        model: Model identifier such as ``blearner[exp_beta]``.

    Returns:
        The :data:`MODEL_SHORT_LABELS` entry; for another B-Learner variant, ``BL`` and
        its numeric Gamma or the start of its rule name; otherwise the first eight
        characters of the identifier.
    """

    name = str(model)
    if name in MODEL_SHORT_LABELS:
        return MODEL_SHORT_LABELS[name]
    rule = blearner_gamma_rule(name)
    if rule is not None:
        gamma = _fixed_gamma_value(rule)
        if gamma is not None:
            return f"BL {gamma:g}"
        return f"BL {rule.split('_')[0][:6]}"
    return name[:8]


def ordered_models(models: Sequence[str]) -> list[str]:
    """Order model identifiers by :data:`MODEL_ORDER`, then alphabetically.

    Args:
        models: Model identifiers, possibly with duplicates.

    Returns:
        Unique identifiers with the canonical models first and any unknown
        Gamma variants appended in sorted order.
    """

    unique = {str(model) for model in models}
    known = [model for model in MODEL_ORDER if model in unique]
    unknown = sorted(unique.difference(MODEL_ORDER))
    return [*known, *unknown]


def blearner_gamma_rule(model: str) -> str | None:
    """Extract the Gamma rule name from a B-Learner model identifier.

    Args:
        model: Model identifier such as ``blearner[oracle]`` or
            ``blearner[fixed_2p5]``.

    Returns:
        The bracketed rule name, or ``None`` when the identifier is not a
        B-Learner variant.
    """

    name = str(model)
    if name.startswith(BLEARNER_MODEL_PREFIX) and name.endswith("]"):
        return name[len(BLEARNER_MODEL_PREFIX):-1]
    return None


def is_fixed_gamma_model(model: str) -> bool:
    """Report whether a model is one point of the swept sensitivity grid.

    A fixed Gamma needs no oracle knowledge, whereas the named rules (``oracle``,
    ``dgp_oracle``, ``exp_beta``, ``width_matched``) each read part of the true
    selection mechanism.

    Args:
        model: Model identifier to classify.

    Returns:
        ``True`` for ``blearner[fixed_*]`` identifiers only.
    """

    rule = blearner_gamma_rule(model)
    return rule is not None and rule.startswith(BLEARNER_FIXED_PREFIX)


def _fallback_colors(count: int) -> list[object]:
    """Return ``count`` distinct colours for models outside :data:`MODEL_COLORS`.

    :data:`FALLBACK_COLORS` come first; any further colours are sampled from ``cividis``.
    """

    if count <= len(FALLBACK_COLORS):
        return list(FALLBACK_COLORS[:count])
    extra = plt.get_cmap("cividis")(
        np.linspace(0.15, 0.9, count - len(FALLBACK_COLORS))
    )
    return [*FALLBACK_COLORS, *extra]


def model_palette(models: Sequence[str]) -> dict[str, str]:
    """Assign a stable colour to every model, including unknown Gamma rules.

    Args:
        models: Model identifiers appearing in the tables being plotted.

    Returns:
        Mapping from model identifier to a matplotlib colour: the :data:`MODEL_COLORS`
        entry where one exists, so colours match across figures, else a
        :func:`_fallback_colors` colour.
    """

    ordered = ordered_models(models)
    unknown = [model for model in ordered if model not in MODEL_COLORS]
    spare = iter(_fallback_colors(len(unknown)))
    palette: dict[str, str] = {}
    for model in ordered:
        if model in MODEL_COLORS:
            palette[model] = MODEL_COLORS[model]
        else:
            palette[model] = next(spare, "0.35")
    return palette


def model_marker(model: str, models: Sequence[str]) -> str:
    """Return a model's marker, which keeps coincident model lines distinguishable.

    Args:
        model: Model identifier to draw.
        models: All model identifiers in the figure, used to assign fallback
            markers deterministically.

    Returns:
        A matplotlib marker code.
    """

    name = str(model)
    if name in MODEL_MARKERS:
        return MODEL_MARKERS[name]
    unknown = [value for value in ordered_models(models) if value not in MODEL_MARKERS]
    index = unknown.index(name) if name in unknown else 0
    return FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)]


def build_paired_outcomes(annotated: pd.DataFrame) -> pd.DataFrame:
    """Pair every competitor model against Naturalness on identical strata.

    One row per (stratum, competitor) eligible for both, with both hit indicators and
    widths, for a McNemar-style discordance test. Invalid intervals count as misses,
    as in the failure-penalized rates.

    Args:
        annotated: Output of
            :func:`experiments.exp25_hit_ratio_analysis.annotate_reference_agreement`.

    Returns:
        Paired rows with ``naturalness_hit``, ``competitor_hit``,
        ``naturalness_width``, ``competitor_width``, and a ``pair_outcome``
        label drawn from :data:`PAIR_OUTCOME_ORDER`.

    Raises:
        ValueError: If the table contains no Naturalness rows.
    """

    keys = list(EXP25_KEY_COLUMNS)
    base_rows = annotated.loc[annotated["model"].eq(NATURALNESS_MODEL)]
    if base_rows.empty:
        raise ValueError("Paired outcomes require Naturalness rows")
    base = base_rows.loc[
        :, [*keys, "rct_hit", "width", "is_valid", "reference_eligible"]
    ].rename(
        columns={
            "rct_hit": "naturalness_hit",
            "width": "naturalness_width",
            "is_valid": "naturalness_is_valid",
            "reference_eligible": "naturalness_eligible",
        }
    )
    others = annotated.loc[~annotated["model"].eq(NATURALNESS_MODEL)]
    merged = others.merge(base, on=keys, how="inner", validate="many_to_one")
    merged = merged.loc[
        merged["reference_eligible"] & merged["naturalness_eligible"]
    ].copy()
    merged = merged.rename(
        columns={
            "rct_hit": "competitor_hit",
            "width": "competitor_width",
            "model": "competitor",
        }
    )
    naturalness_hit = merged["naturalness_hit"].to_numpy(dtype=bool)
    competitor_hit = merged["competitor_hit"].to_numpy(dtype=bool)
    merged["pair_outcome"] = np.select(
        [
            naturalness_hit & competitor_hit,
            naturalness_hit & ~competitor_hit,
            ~naturalness_hit & competitor_hit,
        ],
        ["both_hit", "naturalness_only", "competitor_only"],
        default="neither",
    )
    return merged




def _beta_values(frame: pd.DataFrame) -> list[float]:
    """Return the sorted distinct beta levels present in a table."""

    return sorted(float(value) for value in pd.unique(frame["beta"].dropna()))


def _panel_grid(
    n_rows: int,
    n_cols: int,
    *,
    width: float,
    height: float,
    sharex: bool | str = False,
    sharey: bool | str = False,
    layout: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a subplot grid that is always indexable as a 2-D array.

    Args:
        n_rows: Rows of panels.
        n_cols: Columns of panels.
        width: Figure width per column, in inches.
        height: Figure height per row, in inches.
        sharex: Matplotlib's ``sharex``.
        sharey: Matplotlib's ``sharey``.
        layout: Matplotlib layout engine, for example ``"constrained"``, or
            ``None`` for none.

    Returns:
        The figure and its ``(n_rows, n_cols)`` array of axes.
    """

    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width * n_cols, height * n_rows),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        layout=layout,
    )
    return figure, axes


def model_handles(palette: dict[str, str]) -> list[Line2D]:
    """Build shared legend handles carrying each model's colour and marker.

    Args:
        palette: Mapping from model identifier to colour, as returned by
            :func:`model_palette`.

    Returns:
        One line handle per model, in the palette's order.
    """

    names = list(palette)
    return [
        Line2D(
            [],
            [],
            color=colour,
            marker=model_marker(model, names),
            linestyle="-",
            label=model_label(model),
        )
        for model, colour in palette.items()
    ]


def _finalize_figure(
    figure: plt.Figure,
    handles: Sequence[Any],
    *,
    legend_text_style: dict[str, Any] | None = None,
    legend_rows: int = LEGEND_ROW_COUNT,
    tight: bool = True,
) -> None:
    """Apply the shared Exp16 legend layout to a finished figure.

    The legend runs across the top in ``legend_rows`` rows, as in
    :mod:`experiments.naturalness_plotting`. Under constrained layout it sits in the
    band the layout engine reserves; otherwise, since ``tight_layout`` ignores figure
    legends, a band of :data:`LEGEND_BAND_INCHES` is reserved explicitly.

    Args:
        figure: Figure to finalize.
        handles: Shared legend handles; an empty sequence adds no legend.
        legend_text_style: Optional Matplotlib font properties for legend text.
        legend_rows: Rows the legend is wrapped into.
        tight: Whether to run ``tight_layout`` on a figure without constrained layout;
            pass ``False`` for figures with colorbars, which it cannot lay out.
    """

    constrained = isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine)
    if len(handles):
        figure.legend(
            handles=list(handles),
            # An "outside" location is the band constrained layout reserves.
            **(
                {"loc": "outside upper center"}
                if constrained
                else {"loc": "upper center", "bbox_to_anchor": (0.5, 1.0)}
            ),
            ncol=int(np.ceil(len(handles) / legend_rows)) or 1,
            frameon=False,
            columnspacing=2.0,
            handletextpad=0.7,
            prop=legend_text_style,
        )
    if constrained:
        return
    height = float(figure.get_size_inches()[1])
    legend_fraction = (
        min(LEGEND_BAND_INCHES / height, 0.26) if len(handles) else 0.0
    )
    top = max(1.0 - legend_fraction, 0.55)
    if tight:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, top))
    else:
        figure.subplots_adjust(top=top)


def _finalize_shared_axes_figure(
    figure: plt.Figure,
    axes: np.ndarray,
    handles: Sequence[Any],
    *,
    xlabel: str,
    ylabel: str | None,
    legend_text_style: dict[str, Any] | None = None,
    legend_rows: int = LEGEND_ROW_COUNT,
    label_fontsize: float | None = SHARED_AXIS_LABEL_FONTSIZE,
) -> None:
    """Finalize a grid of identically scaled panels under one label per axis.

    Switches the figure to constrained layout if needed, which keeps the shared labels
    clear of the tick labels and reserves the legend band, then centres each label on
    the span of the visible panels rather than on the figure; constrained layout keeps
    that position on later draws.

    Args:
        figure: Figure to finalize.
        axes: The grid's 2-D axes array. Panels switched off with ``axis("off")`` are
            left out of the label span, and the panel above each shows the x tick labels.
        handles: Shared legend handles, as for :func:`_finalize_figure`.
        xlabel: The one x-axis label, drawn below the grid.
        ylabel: The one y-axis label, drawn left of the grid, or ``None`` to keep the
            panels' own y labels, for panels that plot different quantities.
        legend_text_style: Optional Matplotlib font properties for legend text.
        legend_rows: Rows the legend is wrapped into.
        label_fontsize: Font size of the shared labels, or ``None`` for Matplotlib's
            ``figure.labelsize``.
    """

    if not isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine):
        figure.set_layout_engine("constrained")
    # Shared x-axes keep tick labels on the bottom row only, so a column whose
    # bottom cell is empty would otherwise show none.
    n_rows, n_cols = axes.shape
    for row_index in range(1, n_rows):
        for column_index in range(n_cols):
            if not axes[row_index][column_index].axison:
                above = axes[row_index - 1][column_index]
                above.tick_params(axis="x", labelbottom=True)
    size = {} if label_fontsize is None else {"fontsize": label_fontsize}
    x_label = figure.supxlabel(xlabel, **size)
    y_label = None if ylabel is None else figure.supylabel(ylabel, **size)
    _finalize_figure(
        figure, handles, legend_text_style=legend_text_style, legend_rows=legend_rows
    )
    # The panels' positions are only final once the layout has run.
    figure.draw_without_rendering()
    boxes = [axis.get_position() for axis in axes.flat if axis.axison]
    x_label.set_x((min(box.x0 for box in boxes) + max(box.x1 for box in boxes)) / 2)
    if y_label is not None:
        y_label.set_y((min(box.y0 for box in boxes) + max(box.y1 for box in boxes)) / 2)


def _figures_dir(output_dir: Path) -> Path:
    """Create and return the figures directory beside the analysis tables."""

    figures = Path(output_dir) / FIGURES_DIRNAME
    figures.mkdir(parents=True, exist_ok=True)
    return figures


def interpolate_frontier(
    grid: pd.DataFrame,
    x_column: str,
    y_column: str,
    target: float,
) -> float:
    """Read one swept frontier at a target value of its horizontal axis.

    There is no extrapolation: beyond the ends of the sweep the baseline reaches no
    operating point, so the result is NaN rather than a clamped value.

    Args:
        grid: Rows of one swept sensitivity grid.
        x_column: Column to interpolate against; must be monotone enough to
            sort by.
        y_column: Column to read off.
        target: Value of ``x_column`` at which to read ``y_column``.

    Returns:
        The interpolated value, or NaN when the grid has fewer than two usable
        points or the target lies outside its range.
    """

    frame = grid.loc[np.isfinite(grid[x_column]) & np.isfinite(grid[y_column])]
    frame = frame.sort_values(x_column)
    if len(frame) < 2 or not np.isfinite(target):
        return float("nan")
    x = frame[x_column].to_numpy(dtype=float)
    y = frame[y_column].to_numpy(dtype=float)
    if target < x[0] or target > x[-1]:
        return float("nan")
    return float(np.interp(target, x, y))


def width_hit_matched_table(common_support: pd.DataFrame) -> pd.DataFrame:
    """Tabulate the width-matched and containment-matched B-Learner readouts.

    For each beta, the fixed-Gamma sweep, ordered because a larger Gamma widens every
    interval, is read with :func:`interpolate_frontier` at Naturalness's mean width and
    at its hit ratio; a Naturalness point outside the swept range gives NaN.

    Args:
        common_support: Output of
            :func:`experiments.exp25_hit_ratio_analysis.common_support_frontier`.

    Returns:
        One row per beta with ``n_common``, ``naturalness_width``,
        ``naturalness_hit_ratio``, the ``blearner_hit_at_width`` reached at that width,
        the ``blearner_width_for_hit`` needed to match that hit ratio, and its ratio to
        Naturalness's width, ``width_ratio``. Empty when no beta has both a Naturalness
        row and fixed-Gamma rows.
    """

    frame = common_support.copy()
    if frame.empty:
        return pd.DataFrame()
    frame["model"] = frame["model"].astype(str)
    frame["is_grid"] = frame["model"].map(is_fixed_gamma_model)
    rows: list[dict[str, object]] = []
    for beta in _beta_values(frame):
        panel = frame.loc[np.isclose(frame["beta"], beta)]
        natural = panel.loc[panel["model"].eq(NATURALNESS_MODEL)]
        grid = panel.loc[panel["is_grid"]]
        if natural.empty or grid.empty:
            continue
        width = float(natural["mean_width"].iloc[0])
        hit = float(natural["hit_ratio"].iloc[0])
        width_for_hit = interpolate_frontier(grid, "hit_ratio", "mean_width", hit)
        rows.append(
            {
                "beta": beta,
                "n_common": int(natural["n_common"].iloc[0]),
                "naturalness_width": width,
                "naturalness_hit_ratio": hit,
                "blearner_hit_at_width": interpolate_frontier(
                    grid, "mean_width", "hit_ratio", width
                ),
                "blearner_width_for_hit": width_for_hit,
                "width_ratio": width_for_hit / width if width > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _check_frontier_spread(spread: str) -> None:
    """Reject an error-bar kind outside :data:`FRONTIER_SPREAD_SUFFIXES`."""

    if spread not in FRONTIER_SPREAD_SUFFIXES:
        raise ValueError(
            f"spread must be one of {sorted(FRONTIER_SPREAD_SUFFIXES)}, got {spread!r}"
        )


def _frontier_bar_bounds(row: pd.Series, spread: str) -> np.ndarray | None:
    """Return one point's error-bar extent on both frontier axes.

    Args:
        row: One common-support row for one model.
        spread: ``"ci"`` for the replication-bootstrap interval, ``"sd"`` for
            one across-replication standard deviation either side of the point.

    Returns:
        ``(width_lower, width_upper, hit_lower, hit_upper)``, or ``None`` when
        the row lacks the columns or any bound is not finite.
    """

    if spread == "sd":
        if not set(FRONTIER_SD_COLUMNS).issubset(row.index):
            return None
        x, y = float(row["mean_width"]), float(row["hit_ratio"])
        x_sd, y_sd = row[list(FRONTIER_SD_COLUMNS)].to_numpy(dtype=float)
        bounds = np.array([x - x_sd, x + x_sd, y - y_sd, y + y_sd])
    else:
        if not set(FRONTIER_INTERVAL_COLUMNS).issubset(row.index):
            return None
        bounds = row[list(FRONTIER_INTERVAL_COLUMNS)].to_numpy(dtype=float)
    return bounds if np.isfinite(bounds).all() else None


def _draw_width_hit_panel(
    axis: plt.Axes,
    panel: pd.DataFrame,
    point_models: Sequence[str],
    palette: dict[str, str],
    *,
    full_range_width: float | None = None,
    spread: str = "ci",
    marker_area: float = 95.0,
) -> None:
    """Draw one width-containment panel: one point per named model.

    Shared by the pooled and per-study frontier figures, with fixed axis limits so
    panels can be read against each other; fixed-Gamma rows are left out.

    Args:
        axis: Target axes; the caller sets the title and the axis labels.
        panel: Common-support rows for one cell, with an ``is_grid`` column flagging
            the fixed-Gamma rows. Points get error bars on both axes when the
            columns ``spread`` needs are present.
        point_models: Non-grid models in legend order.
        palette: Mapping from model identifier to colour.
        full_range_width: Width of the full-range fallback interval, drawn as a
            reference line.
        spread: ``"ci"`` draws the ``mean_width_ci_*`` and ``hit_ratio_ci_*`` intervals;
            ``"sd"`` draws ±1 ``mean_width_sd`` and ``hit_ratio_sd``.
        marker_area: Scatter marker area of each model, in points squared.
    """

    points = panel.loc[~panel["is_grid"]].set_index("model")
    for name in point_models:
        if name not in points.index:
            continue
        x = float(points.loc[name, "mean_width"])
        y = float(points.loc[name, "hit_ratio"])
        bounds = _frontier_bar_bounds(points.loc[name], spread)
        if bounds is not None:
            # A percentile interval can sit just off the full-sample point.
            axis.errorbar(
                x,
                y,
                xerr=[[max(x - bounds[0], 0.0)], [max(bounds[1] - x, 0.0)]],
                yerr=[[max(y - bounds[2], 0.0)], [max(bounds[3] - y, 0.0)]],
                fmt="none",
                ecolor=palette[name],
                elinewidth=1.1,
                capsize=2.5,
                zorder=3.5,
            )
        axis.scatter(
            x,
            y,
            marker=model_marker(name, point_models),
            color=palette[name],
            s=marker_area,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
    if full_range_width is not None and np.isfinite(full_range_width):
        axis.axvline(full_range_width, color="0.4", linestyle=":", linewidth=1.2)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.22)


@matplotlib.rc_context(PAPER_RC)
def make_width_hit_frontier_by_study_plot(
    common_support_by_study: pd.DataFrame,
    output_dir: Path,
    *,
    betas: Sequence[float] = STUDY_FRONTIER_BETAS,
    full_range_width: float | None = None,
    spread: str = "ci",
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Split the width-containment plane by study, one row per beta.

    The panels slice the pooled common support rather than recompute it per study, so
    they partition the pooled figure's strata and show a method that wins on some
    studies and loses on others. All panels share the pooled figure's fixed limits and
    the grid is labelled once, with studies heading the columns and betas the rows, on
    the right. A manuscript figure, drawn at :data:`PAPER_FIGURE_WIDTH` under
    :data:`PAPER_RC`.

    Args:
        common_support_by_study: Output of
            :func:`experiments.exp25_hit_ratio_analysis.common_support_frontier`
            grouped by ``("study", "beta")``.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        betas: Confounding levels to draw, one row each.
        full_range_width: Width of the full-range fallback interval, drawn as a
            reference line. Omitted when unknown.
        spread: ``"ci"`` writes ``width_hit_frontier_by_study`` with replication-bootstrap
            intervals; ``"sd"`` writes ``width_hit_frontier_by_study_sd`` with ±1
            across-replication standard deviation.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file; empty when no requested beta is present,
        or when ``spread="sd"`` and the table has no standard-deviation columns.

    Raises:
        ValueError: If ``spread`` is not a known error-bar kind.
    """

    _check_frontier_spread(spread)
    frame = common_support_by_study.copy()
    if frame.empty:
        return []
    if spread == "sd" and not set(FRONTIER_SD_COLUMNS).issubset(frame.columns):
        return []
    frame["model"] = frame["model"].astype(str)
    frame["is_grid"] = frame["model"].map(is_fixed_gamma_model)
    present = _beta_values(frame)
    rows = [
        beta
        for beta in betas
        if any(np.isclose(beta, value) for value in present)
    ]
    studies = sorted(str(value) for value in pd.unique(frame["study"].dropna()))
    if not rows or not studies:
        return []
    point_models = ordered_models(frame.loc[~frame["is_grid"], "model"])
    palette = model_palette(point_models)

    figure, axes = _panel_grid(
        len(rows),
        len(studies),
        width=PAPER_FIGURE_WIDTH / len(studies),
        height=1.9,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for row_index, beta in enumerate(rows):
        for column_index, study in enumerate(studies):
            axis = axes[row_index][column_index]
            panel = frame.loc[
                np.isclose(frame["beta"], beta) & frame["study"].eq(study)
            ]
            _draw_width_hit_panel(
                axis,
                panel,
                point_models,
                palette,
                full_range_width=full_range_width,
                spread=spread,
                marker_area=PAPER_MARKER_AREA,
            )
            if row_index == 0:
                axis.set_title(study, fontweight="bold")
            if column_index == len(studies) - 1:
                # A row header on the right, as in a faceted grid, keeps the
                # left margin for the shared y-axis label.
                axis.text(
                    1.04,
                    0.5,
                    f"{BOLD_BETA_SYMBOL} = {beta:g}",
                    transform=axis.transAxes,
                    rotation=270,
                    ha="left",
                    va="center",
                    fontweight="bold",
                )
    _finalize_shared_axes_figure(
        figure,
        axes,
        model_handles(palette),
        xlabel="Mean interval width",
        ylabel="Hit ratio",
        legend_text_style={"weight": "bold"},
        label_fontsize=None,
    )
    stem = f"width_hit_frontier_by_study{FRONTIER_SPREAD_SUFFIXES[spread]}"
    return _save_figure(figure, _figures_dir(output_dir) / stem, plot_formats)


def make_width_hit_frontier_by_beta_plot(
    common_support: pd.DataFrame,
    output_dir: Path,
    *,
    full_range_width: float | None = None,
    spread: str = "ci",
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Plot the pooled width-containment plane, one panel per beta.

    The pooled counterpart of :func:`make_width_hit_frontier_by_study_plot`,
    drawn with the same panel drawer, limits and error-bar kinds; see
    :func:`make_width_hit_frontier_plot` for how the plane is read.

    Args:
        common_support: Output of
            :func:`experiments.exp25_hit_ratio_analysis.common_support_frontier`
            grouped by ``("beta",)``.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        full_range_width: Width of the full-range fallback interval, drawn as a
            reference line. Omitted when unknown.
        spread: ``"ci"`` writes ``width_hit_frontier_by_beta`` with replication-bootstrap
            intervals; ``"sd"`` writes ``width_hit_frontier_by_beta_sd`` with ±1
            across-replication standard deviation.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file; empty when ``spread="sd"`` and the table
        has no standard-deviation columns.

    Raises:
        ValueError: If ``spread`` is not a known error-bar kind.
    """

    _check_frontier_spread(spread)
    frame = common_support.copy()
    if spread == "sd" and not set(FRONTIER_SD_COLUMNS).issubset(frame.columns):
        return []
    frame["model"] = frame["model"].astype(str)
    frame["is_grid"] = frame["model"].map(is_fixed_gamma_model)
    point_models = ordered_models(frame.loc[~frame["is_grid"], "model"])
    palette = model_palette(point_models)
    betas = _beta_values(frame)

    n_cols = min(len(betas), 2) or 1
    n_rows = int(np.ceil(len(betas) / n_cols)) or 1
    figure, axes = _panel_grid(
        n_rows, n_cols, width=5.6, height=4.8, sharex=True, sharey=True
    )
    for index, beta in enumerate(betas):
        axis = axes.flat[index]
        _draw_width_hit_panel(
            axis,
            frame.loc[np.isclose(frame["beta"], beta)],
            point_models,
            palette,
            full_range_width=full_range_width,
            spread=spread,
        )
        axis.set_title(
            f"{BOLD_BETA_SYMBOL} = {beta:g}",
            fontsize=FRONTIER_HEADER_FONTSIZE,
            fontweight="bold",
        )
    for index in range(len(betas), axes.size):
        axes.flat[index].axis("off")
    _finalize_shared_axes_figure(
        figure,
        axes,
        model_handles(palette),
        xlabel="Mean interval width",
        ylabel="Hit ratio",
        legend_text_style=FRONTIER_LEGEND_TEXT_STYLE,
    )
    stem = f"width_hit_frontier_by_beta{FRONTIER_SPREAD_SUFFIXES[spread]}"
    return _save_figure(figure, _figures_dir(output_dir) / stem, plot_formats)


def make_width_hit_frontier_plot(
    common_support: pd.DataFrame,
    output_dir: Path,
    *,
    full_range_width: float | None = None,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Plot containment against the width that bought it, on common support.

    Containment alone is maximised by an arbitrarily wide interval, so the plane is
    read jointly: the upper left is accurate and narrow, the upper right accurate only
    because uninformative. Both axes use the common support, so declining hard strata
    does not make a method look sharper; validity rates are reported separately
    (``exp25_common_support_frontier.csv``, :func:`make_dose_response_plot`).
    ``blearner[fixed_*]`` rows are not drawn; :func:`width_hit_matched_table`
    tabulates the matched readouts against them.

    Writes the per-beta panels of :func:`make_width_hit_frontier_by_beta_plot` and
    ``width_hit_frontier_trajectory``, which traces each model across beta from an
    open marker at the lowest beta to a star at the highest.

    Args:
        common_support: Output of
            :func:`experiments.exp25_hit_ratio_analysis.common_support_frontier`.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        full_range_width: Width of the full-range fallback interval, drawn as a
            reference line. Omitted when unknown.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file.
    """

    figures_dir = _figures_dir(output_dir)
    frame = common_support.copy()
    frame["model"] = frame["model"].astype(str)
    frame["is_grid"] = frame["model"].map(is_fixed_gamma_model)
    point_models = ordered_models(frame.loc[~frame["is_grid"], "model"])
    palette = model_palette(point_models)
    paths = make_width_hit_frontier_by_beta_plot(
        common_support,
        output_dir,
        full_range_width=full_range_width,
        plot_formats=plot_formats,
    )

    figure, axis = plt.subplots(figsize=(8.6, 6.2))
    for model, rows in frame.loc[~frame["is_grid"]].groupby("model", sort=False):
        rows = rows.sort_values("beta")
        x = rows["mean_width"].to_numpy(dtype=float)
        y = rows["hit_ratio"].to_numpy(dtype=float)
        axis.plot(x, y, color=palette[model], linewidth=1.4, alpha=0.85, zorder=2)
        for start in range(len(x) - 1):
            if not np.all(np.isfinite([x[start], y[start], x[start + 1], y[start + 1]])):
                continue
            axis.annotate(
                "",
                xy=(x[start + 1], y[start + 1]),
                xytext=(x[start], y[start]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": palette[model],
                    "linewidth": 1.2,
                    "alpha": 0.9,
                },
                zorder=2,
            )
        if len(x):
            axis.scatter(
                x[0],
                y[0],
                marker=model_marker(model, point_models),
                facecolor="white",
                edgecolor=palette[model],
                s=95,
                linewidth=1.6,
                zorder=4,
            )
            axis.scatter(
                x[-1], y[-1], color=palette[model], s=140, marker="*", zorder=4
            )
            for position, offset in ((0, (4, 5)), (-1, (4, -10))):
                axis.annotate(
                    f"{rows['beta'].iloc[position]:g}",
                    (x[position], y[position]),
                    textcoords="offset points",
                    xytext=offset,
                    fontsize=7,
                    color=palette[model],
                )
    if full_range_width is not None and np.isfinite(full_range_width):
        axis.axvline(full_range_width, color="0.4", linestyle=":", linewidth=1.2)
    axis.set_xlabel("Mean interval width")
    axis.set_ylabel("Hit ratio")
    axis.set_ylim(0.0, 1.02)
    axis.set_xlim(0.0, 1.0)
    axis.grid(alpha=0.22)
    _finalize_figure(
        figure,
        model_handles(palette),
    )
    paths.extend(
        _save_figure(
            figure, figures_dir / "width_hit_frontier_trajectory", plot_formats
        )
    )
    return paths


DOSE_RESPONSE_METRICS = (
    ("Reference hit", "n_hits", "hit_ratio_conditional", "hit_ratio_failure_penalized"),
    (
        "RCT CI overlap",
        "n_ci_overlaps",
        "ci_overlap_ratio_conditional",
        "ci_overlap_ratio_failure_penalized",
    ),
    (
        "Causal-sign agreement",
        "n_sign_match",
        "sign_match_conditional",
        "sign_match_failure_penalized",
    ),
)


def make_dose_response_plot(
    pooled: pd.DataFrame,
    output_dir: Path,
    *,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Plot every accuracy rate against confounding strength with Wilson 95% bands.

    Rows are the :data:`DOSE_RESPONSE_METRICS` and columns the two denominators,
    conditional on a valid interval and failure-penalized, so the drop across a row
    reflects the strata where a method returned no usable interval.

    Args:
        pooled: Beta-level summary from
            :func:`experiments.exp25_hit_ratio_analysis.summarize_agreement`.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file.
    """

    figures_dir = _figures_dir(output_dir)
    frame = pooled.copy()
    frame["model"] = frame["model"].astype(str)
    palette = model_palette(frame["model"])
    models = ordered_models(frame["model"])
    figure, axes = _panel_grid(
        len(DOSE_RESPONSE_METRICS), 2, width=5.6, height=3.6, sharex=True
    )
    for row_index, (title, count_column, conditional, penalized) in enumerate(
        DOSE_RESPONSE_METRICS
    ):
        for col_index, (rate_column, denominator, denominator_label) in enumerate(
            (
                (conditional, "n_valid_eligible", "conditional on a valid interval"),
                (penalized, "n_reference_eligible", "failures counted as misses"),
            )
        ):
            axis = axes[row_index][col_index]
            for offset_index, (model, rows) in enumerate(
                frame.groupby("model", sort=False)
            ):
                rows = rows.sort_values("beta")
                lower, upper = wilson_interval(
                    rows[count_column].to_numpy(dtype=float),
                    rows[denominator].to_numpy(dtype=float),
                )
                centre = rows[rate_column].to_numpy(dtype=float)
                # Nudge each series along beta so coincident lines (rates of one) stay visible.
                dodge = 1.0 + 0.014 * (offset_index - (len(models) - 1) / 2.0)
                beta_values = rows["beta"].to_numpy(dtype=float) * dodge
                axis.plot(
                    beta_values,
                    centre,
                    marker=model_marker(model, models),
                    markersize=4.5,
                    linewidth=1.4,
                    color=palette[model],
                    alpha=0.9,
                )
                axis.fill_between(
                    beta_values,
                    lower,
                    upper,
                    color=palette[model],
                    alpha=0.13,
                    linewidth=0.0,
                )
            axis.set_xscale("log")
            axis.set_xticks(sorted(pd.unique(frame["beta"].dropna())))
            axis.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
            )
            axis.set_ylim(0.0, 1.02)
            axis.set_title(f"{title} — {denominator_label}", fontsize=10)
            axis.grid(alpha=0.22)
            if row_index == len(DOSE_RESPONSE_METRICS) - 1:
                axis.set_xlabel(f"{BETA_SYMBOL} (confounding strength)")
            if col_index == 0:
                axis.set_ylabel("rate")
    _finalize_figure(
        figure,
        model_handles(palette),
    )
    return _save_figure(
        figure, figures_dir / "hit_ratio_dose_response", plot_formats
    )


def make_paired_outcome_plot(
    paired: pd.DataFrame,
    output_dir: Path,
    *,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Plot per-stratum hit concordance between Naturalness and each competitor.

    One stacked bar per competitor and beta, annotated with the discordant counts
    (``naturalness_only`` against ``competitor_only``) and their exact McNemar p-value.

    Args:
        paired: Output of :func:`build_paired_outcomes`.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file.
    """

    figures_dir = _figures_dir(output_dir)
    betas = _beta_values(paired)
    competitors = ordered_models(paired["competitor"])
    figure, axes = _panel_grid(
        len(betas) or 1, 1, width=10.4, height=0.62 * max(len(competitors), 1) + 1.5
    )
    for index, beta in enumerate(betas):
        axis = axes.flat[index]
        panel = paired.loc[np.isclose(paired["beta"], beta)]
        y_positions = np.arange(len(competitors))
        left = np.zeros(len(competitors), dtype=float)
        totals = np.array(
            [
                float(len(panel.loc[panel["competitor"].eq(name)]))
                for name in competitors
            ]
        )
        for outcome in PAIR_OUTCOME_ORDER:
            counts = np.array(
                [
                    float(
                        len(
                            panel.loc[
                                panel["competitor"].eq(name)
                                & panel["pair_outcome"].eq(outcome)
                            ]
                        )
                    )
                    for name in competitors
                ]
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                shares = np.where(totals > 0, counts / totals, 0.0)
            axis.barh(
                y_positions,
                shares,
                left=left,
                color=PAIR_OUTCOME_COLORS[outcome],
                edgecolor="white",
                linewidth=0.6,
            )
            left = left + shares
        for position, name in zip(y_positions, competitors):
            rows = panel.loc[panel["competitor"].eq(name)]
            naturalness_only = int(rows["pair_outcome"].eq("naturalness_only").sum())
            competitor_only = int(rows["pair_outcome"].eq("competitor_only").sum())
            discordant = naturalness_only + competitor_only
            if discordant:
                p_value = binomtest(
                    naturalness_only, discordant, 0.5, alternative="two-sided"
                ).pvalue
                annotation = (
                    f"discordant {naturalness_only} vs {competitor_only}, "
                    f"McNemar p = {p_value:.3g}"
                )
            else:
                annotation = "no discordant strata"
            axis.text(
                1.01,
                position,
                annotation,
                va="center",
                fontsize=8,
                color="0.25",
            )
        axis.set_yticks(y_positions, [model_label(name) for name in competitors])
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Share of reference-eligible strata")
        axis.set_title(f"{BETA_SYMBOL} = {beta:g}")
        axis.grid(axis="x", alpha=0.2)
    for index in range(len(betas), axes.size):
        axes.flat[index].axis("off")
    handles = [
        Patch(facecolor=PAIR_OUTCOME_COLORS[outcome], label=PAIR_OUTCOME_LABELS[outcome])
        for outcome in PAIR_OUTCOME_ORDER
    ]
    _finalize_figure(
        figure,
        handles,
    )
    return _save_figure(figure, figures_dir / "paired_hit_outcomes", plot_formats)




def bootstrap_error_bars(
    bootstrap: pd.DataFrame | None,
    *,
    beta: float,
    models: Sequence[str],
    metric: str,
    estimates: np.ndarray,
) -> np.ndarray | None:
    """Convert bootstrap percentile bounds into matplotlib error-bar offsets.

    Args:
        bootstrap: Long ``metric_summary`` from
            :func:`experiments.exp25_bootstrap_analysis.bootstrap_sign_metrics`,
            or ``None`` to draw bare markers.
        beta: Confounding level selecting one bootstrap cell.
        models: Model identifiers in the order their estimates are supplied.
        metric: Metric name to look up, for example ``sign_correct_rate``.
        estimates: Pooled point estimates the offsets are measured from.

    Returns:
        A ``(2, len(models))`` array of non-negative lower and upper offsets, or
        ``None`` when no bootstrap row matches, in which case no bars are drawn.
    """

    if bootstrap is None or bootstrap.empty or "metric" not in bootstrap.columns:
        return None
    rows = bootstrap.loc[
        bootstrap["metric"].eq(metric) & np.isclose(bootstrap["beta"], beta)
    ]
    if rows.empty:
        return None
    lookup = rows.set_index(rows["model"].astype(str))
    lower = np.full(len(models), np.nan)
    upper = np.full(len(models), np.nan)
    for index, name in enumerate(models):
        if name not in lookup.index:
            continue
        row = lookup.loc[name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        lower[index] = float(row["ci_lower"])
        upper[index] = float(row["ci_upper"])
    if not np.isfinite(lower).any():
        return None
    # Percentile intervals are not centred on the pooled estimate, so a
    # naive subtraction can produce the negative offsets errorbar rejects.
    return np.vstack(
        [
            np.clip(np.nan_to_num(estimates - lower, nan=0.0), 0.0, None),
            np.clip(np.nan_to_num(upper - estimates, nan=0.0), 0.0, None),
        ]
    )


@matplotlib.rc_context(PAPER_RC)
def make_sign_identification_frontier_plot(
    pooled: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap: pd.DataFrame | None = None,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Plot sign-identification correctness against how often a sign is identified.

    The horizontal axis is how often the identified set excludes zero (decisiveness)
    and the vertical axis how often, when it does, the excluded side is right; both
    are partial-identification quantities, not classification scores. A correct-sign
    rate is only meaningful next to its decisiveness: the upper right identifies many
    signs correctly, the upper left is correct only because it rarely commits. One
    panel per beta and one point per named model; ``blearner[fixed_*]`` rows are left
    out. A manuscript figure, drawn at :data:`PAPER_FIGURE_WIDTH` under
    :data:`PAPER_RC`.

    Args:
        pooled: Beta-level summary from
            :func:`experiments.exp25_hit_ratio_analysis.summarize_agreement`.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        bootstrap: Optional long ``metric_summary`` from
            :func:`experiments.exp25_bootstrap_analysis.bootstrap_sign_metrics`,
            used to draw replication-cluster intervals on every drawn model.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file.
    """

    figures_dir = _figures_dir(output_dir)
    frame = pooled.copy()
    frame["model"] = frame["model"].astype(str)
    frame["is_grid"] = frame["model"].map(is_fixed_gamma_model)
    point_models = ordered_models(frame.loc[~frame["is_grid"], "model"])
    palette = model_palette(point_models)
    betas = _beta_values(frame)
    n_cols = min(len(betas), 2) or 1
    n_rows = int(np.ceil(len(betas) / n_cols)) or 1
    figure, axes = _panel_grid(
        n_rows,
        n_cols,
        width=PAPER_FIGURE_WIDTH / n_cols,
        height=2.6,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for index, beta in enumerate(betas):
        axis = axes.flat[index]
        panel = frame.loc[np.isclose(frame["beta"], beta)]
        points = panel.loc[~panel["is_grid"]].set_index("model")
        drawn = [name for name in point_models if name in points.index]
        x = np.array(
            [float(points.loc[name, FRONTIER_DECISIVENESS]) for name in drawn]
        )
        y = np.array([float(points.loc[name, FRONTIER_CORRECT]) for name in drawn])
        x_error = bootstrap_error_bars(
            bootstrap,
            beta=beta,
            models=drawn,
            metric=FRONTIER_DECISIVENESS,
            estimates=x,
        )
        y_error = bootstrap_error_bars(
            bootstrap, beta=beta, models=drawn, metric=FRONTIER_CORRECT, estimates=y
        )
        for position, name in enumerate(drawn):
            axis.errorbar(
                x[position],
                y[position],
                xerr=None if x_error is None else x_error[:, position : position + 1],
                yerr=None if y_error is None else y_error[:, position : position + 1],
                marker=model_marker(name, point_models),
                markersize=np.sqrt(PAPER_MARKER_AREA),
                markeredgecolor="white",
                markeredgewidth=0.8,
                color=palette[name],
                ecolor=palette[name],
                elinewidth=1.1,
                capsize=2.0,
                linestyle="none",
                zorder=4,
            )
        axis.set_title(f"{BOLD_BETA_SYMBOL} = {beta:g}", fontweight="bold")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.22)
    for index in range(len(betas), axes.size):
        axes.flat[index].axis("off")
    # Every panel has the same fixed limits, so the grid is labelled once.
    _finalize_shared_axes_figure(
        figure,
        axes,
        model_handles(palette),
        xlabel="Decisiveness rate",
        ylabel="Correct-sign rate",
        legend_text_style={"weight": "bold"},
        label_fontsize=None,
    )
    return _save_figure(
        figure, figures_dir / "sign_identification_frontier", plot_formats
    )


def make_sign_cell_decomposition_plot(
    pooled: pd.DataFrame,
    output_dir: Path,
    *,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Split every valid stratum into the five sign-verdict outcomes.

    The stacked raw cells show what a single sign metric hides: a bar dominated by
    ``sign identified, RCT inconclusive`` answers where the reference cannot, and one
    dominated by ``both contain zero`` adds nothing over abstaining. Each bar is
    annotated with its stratum and decisive counts.

    Args:
        pooled: Beta-level summary from
            :func:`experiments.exp25_hit_ratio_analysis.summarize_agreement`.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file.
    """

    figures_dir = _figures_dir(output_dir)
    frame = pooled.copy()
    frame["model"] = frame["model"].astype(str)
    models = ordered_models(frame["model"])
    betas = _beta_values(frame)
    n_cols = min(len(betas), 2) or 1
    n_rows = int(np.ceil(len(betas) / n_cols)) or 1
    figure, axes = _panel_grid(
        n_rows,
        n_cols,
        width=6.4,
        height=0.34 * max(len(models), 1) + 1.9,
        sharex=True,
    )
    for index, beta in enumerate(betas):
        axis = axes.flat[index]
        panel = frame.loc[np.isclose(frame["beta"], beta)].set_index("model")
        positions = np.arange(len(models))
        totals = np.array(
            [
                float(panel.loc[name, "n_valid_eligible"])
                if name in panel.index
                else np.nan
                for name in models
            ]
        )
        left = np.zeros(len(models), dtype=float)
        for column, _, colour in SIGN_CELL_SPECS:
            counts = np.array(
                [
                    float(panel.loc[name, column]) if name in panel.index else np.nan
                    for name in models
                ]
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                shares = np.where(totals > 0, counts / totals, 0.0)
            shares = np.nan_to_num(shares, nan=0.0)
            axis.barh(
                positions,
                shares,
                left=left,
                color=colour,
                edgecolor="white",
                linewidth=0.6,
            )
            left = left + shares
        # Print each bar's denominators, so a high rate on a few decisive calls is not over-read.
        for position, name, total in zip(positions, models, totals):
            if name not in panel.index or not np.isfinite(total):
                continue
            axis.text(
                1.01,
                position,
                f"n={int(total)}, decisive {int(panel.loc[name, 'n_model_decisive'])}",
                va="center",
                fontsize=6.5,
                color="0.3",
            )
        axis.set_yticks(positions, [model_label(name) for name in models], fontsize=7)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_title(f"{BETA_SYMBOL} = {beta:g}", fontsize=10)
        axis.set_xlabel("Share of valid, reference-eligible strata")
        axis.grid(axis="x", alpha=0.2)
    for index in range(len(betas), axes.size):
        axes.flat[index].axis("off")
    handles = [
        Patch(facecolor=colour, label=label) for _, label, colour in SIGN_CELL_SPECS
    ]
    _finalize_figure(
        figure,
        handles,
    )
    return _save_figure(
        figure, figures_dir / "sign_cell_decomposition", plot_formats
    )




def make_sign_confusion_plot(
    confusion: pd.DataFrame,
    output_dir: Path,
    *,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Render the RCT-versus-model verdict cross-tabulation as heatmaps.

    One panel per beta and model. Cells are shaded by their share of the RCT verdict's
    row, so models with different validity rates stay comparable; the two
    anti-diagonal corners are sign reversals.

    Args:
        confusion: Output of
            :func:`experiments.exp25_hit_ratio_analysis.build_sign_confusion`.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file.
    """

    figures_dir = _figures_dir(output_dir)
    frame = confusion.copy()
    frame["model"] = frame["model"].astype(str)
    # Fixed-Gamma models get no panel (a grid would make the figure unreadably wide)
    # unless no named model is present.
    named = [name for name in pd.unique(frame["model"]) if not is_fixed_gamma_model(name)]
    models = ordered_models(named if named else frame["model"])
    betas = _beta_values(frame)
    figure, axes = _panel_grid(
        len(betas) or 1, len(models) or 1, width=2.6, height=2.9
    )
    for row_index, beta in enumerate(betas):
        for col_index, name in enumerate(models):
            axis = axes[row_index][col_index]
            panel = frame.loc[
                np.isclose(frame["beta"], beta) & frame["model"].eq(name)
            ]
            counts = (
                panel.pivot_table(
                    index="rct_sign",
                    columns="model_sign",
                    values="n",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reindex(index=list(SIGN_ORDER), fill_value=0)
                .reindex(columns=list(SIGN_ORDER), fill_value=0)
            )
            values = counts.to_numpy(dtype=float)
            row_totals = values.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                shares = np.where(row_totals > 0, values / row_totals, np.nan)
            # An equal aspect would fix the axes box, leaving tight_layout no room for titles.
            axis.imshow(shares, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    share = shares[i, j]
                    text = (
                        f"{int(values[i, j])}\n{share:.2f}"
                        if np.isfinite(share)
                        else f"{int(values[i, j])}"
                    )
                    axis.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if np.nan_to_num(share) > 0.55 else "0.15",
                    )
            axis.set_xticks(
                range(len(SIGN_ORDER)),
                [SIGN_LABELS[value] for value in SIGN_ORDER],
                fontsize=7,
            )
            axis.set_yticks(
                range(len(SIGN_ORDER)),
                [SIGN_LABELS[value] for value in SIGN_ORDER],
                fontsize=7,
            )
            if row_index == 0:
                axis.set_title(model_label(name), fontsize=8)
            if col_index == 0:
                axis.set_ylabel(
                    f"{BETA_SYMBOL} = {beta:g}\nRCT verdict", fontsize=8
                )
            if row_index == len(betas) - 1:
                axis.set_xlabel("model verdict", fontsize=8)
    _finalize_figure(
        figure,
        [],
    )
    return _save_figure(figure, figures_dir / "sign_confusion_heatmaps", plot_formats)










def make_gamma_provenance_plot(
    pooled: pd.DataFrame,
    output_dir: Path,
    *,
    metric: str = FRONTIER_CORRECT,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Plot correctness along the Gamma axis against where each rule lands.

    Left: ``metric`` along the swept fixed-Gamma grid, one curve per beta, with each
    named Gamma rule as a point at its selected Gamma and Naturalness, which has no
    Gamma, as a dashed line. The named rules each read part of the true selection
    mechanism, so the gap shows how much of a B-Learner advantage is oracle access.
    Right: each rule's selected Gamma against beta; a deployable rule should rise
    with the true confounding strength.

    Args:
        pooled: Beta-level summary from
            :func:`experiments.exp25_hit_ratio_analysis.summarize_agreement`.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        metric: Summary column drawn on the left panel's vertical axis.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file. Empty when no B-Learner variants
        are present.
    """

    frame = pooled.copy()
    frame["model"] = frame["model"].astype(str)
    blearner = frame.loc[frame["model"].str.startswith(BLEARNER_MODEL_PREFIX)]
    if blearner.empty:
        return []
    figures_dir = _figures_dir(output_dir)
    grid = blearner.loc[blearner["model"].map(is_fixed_gamma_model)]
    rules = blearner.loc[~blearner["model"].map(is_fixed_gamma_model)]
    rule_names = ordered_models(rules["model"])
    palette = model_palette(rule_names)
    betas = _beta_values(blearner)
    beta_colors = dict(
        zip(betas, plt.get_cmap("viridis")(np.linspace(0.15, 0.88, len(betas))))
    )
    naturalness = frame.loc[frame["model"].eq(NATURALNESS_MODEL)]
    figure, axes = _panel_grid(1, 2, width=6.6, height=5.2)

    axis = axes[0][0]
    for beta in betas:
        swept = grid.loc[np.isclose(grid["beta"], beta)].sort_values(
            "mean_blearner_gamma"
        )
        if not swept.empty:
            axis.plot(
                swept["mean_blearner_gamma"],
                swept[metric],
                color=beta_colors[beta],
                linewidth=1.5,
                marker="o",
                markersize=4,
                zorder=3,
            )
        # Naturalness has no Gamma, so it is a horizontal reference line.
        reference = naturalness.loc[np.isclose(naturalness["beta"], beta)]
        if not reference.empty:
            axis.axhline(
                float(reference.iloc[0][metric]),
                color=beta_colors[beta],
                linestyle="--",
                linewidth=1.1,
                alpha=0.7,
            )
    for name in rule_names:
        for beta in betas:
            row = rules.loc[
                rules["model"].eq(name) & np.isclose(rules["beta"], beta)
            ]
            if row.empty:
                continue
            gamma = float(row.iloc[0]["mean_blearner_gamma"])
            if not np.isfinite(gamma) or gamma <= 0.0:
                continue
            axis.scatter(
                gamma,
                float(row.iloc[0][metric]),
                marker=model_marker(name, rule_names),
                color=palette[name],
                edgecolor="white",
                linewidth=0.7,
                s=85,
                zorder=4,
            )
    axis.set_xscale("log")
    axis.set_xlabel(f"Selected {GAMMA_SYMBOL} (log)")
    axis.set_ylabel(f"{metric.replace('_', ' ')} (dashed line: Naturalness)")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.22)
    axis.set_title(
        f"Where each {GAMMA_SYMBOL} rule lands on the swept axis", fontsize=10
    )

    rule_axis = axes[0][1]
    for name in rule_names:
        rows = rules.loc[rules["model"].eq(name)].sort_values("beta")
        rule_axis.plot(
            rows["beta"],
            rows["mean_blearner_gamma"],
            marker=model_marker(name, rule_names),
            color=palette[name],
            linewidth=1.5,
        )
    rule_axis.set_xscale("log")
    rule_axis.set_yscale("log")
    rule_axis.set_xlabel(f"{BETA_SYMBOL} (confounding strength)")
    rule_axis.set_ylabel(f"Selected {GAMMA_SYMBOL} (log)")
    rule_axis.grid(alpha=0.22)
    rule_axis.set_title("Does the rule track the true confounding?", fontsize=10)

    beta_handles = [
        Line2D([], [], color=beta_colors[beta], marker="o", label=f"{BETA_SYMBOL} = {beta:g}")
        for beta in betas
    ]
    rule_handles = [
        Line2D(
            [],
            [],
            color=palette[name],
            marker=model_marker(name, rule_names),
            linestyle="none",
            label=(blearner_gamma_rule(name) or str(name))
            .replace("_", "-")
            .replace("beta", BETA_SYMBOL),
        )
        for name in rule_names
    ]
    _finalize_figure(
        figure,
        [*rule_handles, *beta_handles],
    )
    return _save_figure(figure, figures_dir / "gamma_provenance", plot_formats)

def make_replication_variability_plot(
    annotated: pd.DataFrame,
    output_dir: Path,
    *,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Plot the Monte Carlo spread of hit ratios across replications.

    Each point is one replication's failure-penalized hit ratio, over a box plot per
    model, so a gap between models can be judged against replication noise.

    Args:
        annotated: Row-level annotated interval table.
        output_dir: Analysis directory receiving a ``figures`` subdirectory.
        plot_formats: Matplotlib output extensions.

    Returns:
        Paths of every written figure file.
    """

    figures_dir = _figures_dir(output_dir)
    eligible = annotated.loc[annotated["reference_eligible"]].copy()
    eligible["model"] = eligible["model"].astype(str)
    per_replication = (
        eligible.groupby(["beta", "model", "replication"], observed=True)["rct_hit"]
        .mean()
        .rename("hit_ratio")
        .reset_index()
    )
    models = ordered_models(per_replication["model"])
    palette = model_palette(models)
    betas = _beta_values(per_replication)
    n_cols = min(len(betas), 2) or 1
    n_rows = int(np.ceil(len(betas) / n_cols)) or 1
    figure, axes = _panel_grid(
        n_rows, n_cols, width=5.4, height=3.9, sharey=True
    )
    rng = np.random.default_rng(20260807)
    for index, beta in enumerate(betas):
        axis = axes.flat[index]
        panel = per_replication.loc[np.isclose(per_replication["beta"], beta)]
        datasets = []
        for position, name in enumerate(models):
            values = panel.loc[panel["model"].eq(name), "hit_ratio"].to_numpy(
                dtype=float
            )
            datasets.append(values)
            if values.size == 0:
                continue
            jitter = rng.uniform(-0.16, 0.16, size=values.size)
            axis.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=12,
                alpha=0.55,
                color=palette[name],
                linewidth=0.0,
                zorder=3,
            )
        boxes = axis.boxplot(
            datasets,
            positions=np.arange(len(models)),
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            zorder=2,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("0.35")
        for whisker in boxes["whiskers"] + boxes["caps"]:
            whisker.set_color("0.35")
        for median in boxes["medians"]:
            median.set_color("black")
        axis.set_xticks(range(len(models)))
        axis.set_xticklabels(
            [model_short_label(name) for name in models], fontsize=8
        )
        axis.set_ylim(0.0, 1.02)
        axis.set_title(f"{BETA_SYMBOL} = {beta:g}")
        axis.set_ylabel("Failure-penalized hit ratio")
        axis.grid(axis="y", alpha=0.22)
    for index in range(len(betas), axes.size):
        axes.flat[index].axis("off")
    _finalize_figure(
        figure,
        model_handles(palette),
    )
    return _save_figure(
        figure, figures_dir / "replication_variability", plot_formats
    )


def infer_full_range_width(annotated: pd.DataFrame) -> float | None:
    """Infer the full-range fallback width from rows that used the fallback.

    Args:
        annotated: Interval rows carrying ``used_full_range_fallback`` and
            ``width``.

    Returns:
        The median fallback width, or ``None`` when no row used the fallback.
    """

    fallback = annotated.loc[
        annotated["used_full_range_fallback"] & np.isfinite(annotated["width"]),
        "width",
    ]
    if fallback.empty:
        return None
    return float(fallback.median())


def make_all_plots(
    annotated: pd.DataFrame,
    pooled: pd.DataFrame,
    confusion: pd.DataFrame,
    output_dir: str | Path,
    *,
    common_support: pd.DataFrame | None = None,
    common_support_by_study: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
) -> list[Path]:
    """Render the complete Exp25 figure set into ``<output_dir>/figures``.

    Args:
        annotated: Row-level table from
            :func:`experiments.exp25_hit_ratio_analysis.annotate_reference_agreement`.
        pooled: Summary grouped by beta.
        confusion: Output of
            :func:`experiments.exp25_hit_ratio_analysis.build_sign_confusion`.
        output_dir: Analysis directory receiving the ``figures`` subdirectory.
        common_support: Optional output of
            :func:`experiments.exp25_hit_ratio_analysis.common_support_frontier` for the
            width figures; recomputed from ``annotated`` when omitted.
        common_support_by_study: The same table grouped by ``("study", "beta")``, for
            the per-study frontier; recomputed when omitted.
        bootstrap: Optional long ``metric_summary`` from
            :func:`experiments.exp25_bootstrap_analysis.bootstrap_sign_metrics`; without
            it the sign-identification frontier has no uncertainty bars.
        plot_formats: Matplotlib output extensions such as ``png`` and ``pdf``.

    Returns:
        Paths of every written figure file.

    Raises:
        ValueError: If ``plot_formats`` is empty or contains a non-alphanumeric
            extension.
    """

    formats = tuple(str(value).lower().lstrip(".") for value in plot_formats)
    if not formats or any(not value.isalnum() for value in formats):
        raise ValueError("plot_formats must contain simple file extensions")
    destination = Path(output_dir).expanduser().resolve()
    paired = build_paired_outcomes(annotated)
    full_range_width = infer_full_range_width(annotated)
    paths: list[Path] = []
    paths += make_sign_identification_frontier_plot(
        pooled, destination, bootstrap=bootstrap, plot_formats=formats
    )
    paths += make_gamma_provenance_plot(pooled, destination, plot_formats=formats)
    paths += make_sign_cell_decomposition_plot(
        pooled, destination, plot_formats=formats
    )
    paths += make_sign_confusion_plot(confusion, destination, plot_formats=formats)
    # The width figures use the common support, so every model is scored on the same strata.
    if common_support is None or common_support_by_study is None:
        if common_support is None:
            common_support = common_support_frontier(
                annotated, grouping=("beta",), replication_spread=True
            )
        if common_support_by_study is None:
            common_support_by_study = common_support_frontier(
                annotated, grouping=("study", "beta"), replication_spread=True
            )
    paths += make_width_hit_frontier_plot(
        common_support,
        destination,
        full_range_width=full_range_width,
        plot_formats=formats,
    )
    paths += make_width_hit_frontier_by_beta_plot(
        common_support,
        destination,
        full_range_width=full_range_width,
        spread="sd",
        plot_formats=formats,
    )
    paths += make_width_hit_frontier_by_study_plot(
        common_support_by_study,
        destination,
        full_range_width=full_range_width,
        plot_formats=formats,
    )
    paths += make_width_hit_frontier_by_study_plot(
        common_support_by_study,
        destination,
        full_range_width=full_range_width,
        spread="sd",
        plot_formats=formats,
    )
    paths += make_dose_response_plot(pooled, destination, plot_formats=formats)
    paths += make_paired_outcome_plot(paired, destination, plot_formats=formats)
    paths += make_replication_variability_plot(
        annotated, destination, plot_formats=formats
    )
    return paths


#: Comparators drawn beside Naturalness in the mechanism-arm figures, in legend order.
ARM_COMPARATORS = ("balke_pearl", "transport", "manski")
#: Arms that modify the target: the column holding the modification size and its axis label.
ARM_SIZE_AXES = {
    "effect_shift": ("delta", r"Shift $\Delta$ added to every target stratum CATE"),
    "invalid_instrument": ("gamma", r"Direct effect $\gamma$ of the selection covariate on $Y$"),
}
#: Neutral inks of the fixed-tolerance curves, light to dark as the fixed c~1 grows.
ABLATION_FIXED_INKS = ("0.80", "0.64", "0.48", "0.32")
#: Marker, ink and legend label of the ablation's other reference rules.
ABLATION_RULE_STYLES = {
    "exact": ("v", "0.15", "Exact restrictions"),
    "prior": ("^", "0.45", "Prior only"),
}
#: Seed of the replication bootstrap behind the mechanism-arm error bars.
ARM_FIGURE_SEED = 20261008


def _parse_fixed_rule(label: str) -> tuple[float, float]:
    """Return the fixed ``(c~1, c~2)`` of an ablation rule labelled ``fixed_<c1>_<c2>``."""
    _, two_c, two_d = str(label).split("_")
    return float(two_c), float(two_d)


def _arm_reference_rows(strata: pd.DataFrame, grounding: str) -> pd.DataFrame:
    """Return the unmodified target's rows for Naturalness and the comparators, with ``model``.

    Args:
        strata: The arms' stratum table (``summary/arms_strata.parquet``).
        grounding: Naturalness grounding to keep, ``per_site`` or ``theory``.
    """
    base = strata.loc[strata["arm"].eq("effect_shift") & strata["delta"].eq(0.0)]
    keep = (base["method"].eq("naturalness") & base["grounding"].eq(grounding)) | base["method"].isin(ARM_COMPARATORS)
    return base.loc[keep].assign(model=lambda frame: frame["method"])


def _arm_intervals(
    rows: pd.DataFrame, grouping: Sequence[str], value: str, *, bootstrap_replicates: int,
    confidence_level: float, seed: int,
) -> dict[tuple, dict[str, float]]:
    """Bootstrap whole replications for the mean width and the mean of ``value``.

    Reuses :func:`_frontier_bootstrap_intervals`, whose ``hit_ratio_ci_*``
    entries then hold the interval of ``value`` (the hit ratio, or the share of
    wrong-sign intervals).
    """
    return _frontier_bootstrap_intervals(
        rows.assign(rct_hit=rows[value].astype(float)), list(grouping),
        bootstrap_replicates=bootstrap_replicates, confidence_level=confidence_level, seed=seed,
    )


def _arm_facet_title(facet: str, value: Any) -> str:
    """Return a panel header for one study or one selection strength."""
    return f"{BETA_SYMBOL} = {float(value):g}" if facet == "beta" else str(value)


def _draw_below_floor(axis: plt.Axes, x: float, y: float, floor: float, *, marker: str, color: Any,
                      size: float, name: str, placed: list[float]) -> None:
    """Pin a point that falls below a panel's floor to the floor, with its name and value.

    ``placed`` holds the x of the panel's earlier pinned points; a label near
    one of them is raised a line, so neighbouring labels never overprint.
    """
    axis.scatter(x, floor, marker=marker, color=color, s=size, edgecolor="white", linewidth=0.8,
                 zorder=5, clip_on=False)
    lift = 5 + 9 * sum(abs(x - other) < 0.12 for other in placed)
    axis.annotate(f"{name} {y:.2f}\u2193", (x, floor), xytext=(0, lift), textcoords="offset points",
                  ha="center", va="bottom", fontsize="small", color="0.25")
    placed.append(x)


@matplotlib.rc_context(PAPER_RC)
def make_grounding_ablation_frontier_plot(
    strata: pd.DataFrame,
    output_dir: Path,
    *,
    facet: str = "study",
    grounding: str = "per_site",
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = ARM_FIGURE_SEED,
    hit_floor: float = 0.7,
) -> list[Path]:
    """Place grounded Naturalness on the width-hit plane of the grounding ablation.

    One panel per value of ``facet`` (study or ``beta``), pooled over everything else.
    Grey curves are the fixed tolerances (one per fixed ``c~1``, ``c~2`` growing along
    it) and grey markers the exact-restriction and prior-only rules; Naturalness,
    Balke-Pearl and transport carry replication-bootstrap intervals on both axes, and
    Manski marks the no-assumption corner. All rules share the target discovery, so
    the panels compare how tolerances are set. Limits are shared and the hit axis
    starts at ``hit_floor``; a point below it is pinned to the floor with its value
    printed. A manuscript figure, drawn at :data:`PAPER_FIGURE_WIDTH` under
    :data:`PAPER_RC`.

    Args:
        strata: The arms' stratum table (``summary/arms_strata.parquet``).
        output_dir: Directory whose ``figures/`` subdirectory receives the files.
        facet: ``study`` or ``beta``.
        grounding: Naturalness grounding to highlight.
        plot_formats: File extensions to write.
        bootstrap_replicates: Replication resamples per interval.
        confidence_level: Level of the percentile intervals.
        seed: Bootstrap seed.
        hit_floor: Lower limit of the hit-ratio axis.

    Returns:
        Paths written; empty when the table holds no ablation rows.
    """
    named = _arm_reference_rows(strata, grounding)
    rules = strata.loc[strata["arm"].eq("grounding_ablation")].assign(model=lambda frame: frame["grounding"])
    if named.empty or rules.empty:
        return []
    points = pd.concat([named, rules]).groupby([facet, "model"]).agg(
        mean_width=("width", "mean"), hit_ratio=("hit", "mean")).reset_index()
    intervals = _arm_intervals(named, [facet], "hit", bootstrap_replicates=bootstrap_replicates,
                               confidence_level=confidence_level, seed=seed)
    values = sorted(points[facet].unique())
    fixed_c = sorted({_parse_fixed_rule(m)[0] for m in points["model"] if str(m).startswith("fixed_")})
    inks = dict(zip(fixed_c, ABLATION_FIXED_INKS))
    n_cols = len(values) if len(values) <= 3 else int(np.ceil(len(values) / 2))
    n_rows = int(np.ceil(len(values) / n_cols))
    figure, axes = _panel_grid(n_rows, n_cols, width=PAPER_FIGURE_WIDTH / n_cols, height=2.8, sharex=True,
                               sharey=True, layout="constrained")
    for axis in axes.flat[len(values):]:
        axis.axis("off")
    for axis, value in zip(axes.flat, values):
        panel = points.loc[points[facet].eq(value)].set_index("model")
        placed: list[float] = []
        for two_c in fixed_c:
            curve = sorted((_parse_fixed_rule(m)[1], m) for m in panel.index
                           if str(m).startswith("fixed_") and _parse_fixed_rule(m)[0] == two_c)
            axis.plot([panel.loc[m, "mean_width"] for _, m in curve], [panel.loc[m, "hit_ratio"] for _, m in curve],
                      color=inks[two_c], marker="o", markersize=3, linewidth=1.1, zorder=2)
        for rule, (marker, ink, _) in ABLATION_RULE_STYLES.items():
            if rule not in panel.index:
                continue
            x, y = float(panel.loc[rule, "mean_width"]), float(panel.loc[rule, "hit_ratio"])
            if y < hit_floor:
                _draw_below_floor(axis, x, y, hit_floor, marker=marker, color=ink, size=28,
                                  name=ABLATION_RULE_STYLES[rule][2].split()[0], placed=placed)
            else:
                axis.scatter(x, y, marker=marker, color=ink, s=28, edgecolor="white", linewidth=0.6, zorder=3)
        for model in (*ARM_COMPARATORS, "naturalness"):  # Naturalness last, on top
            if model not in panel.index:
                continue
            x, y = float(panel.loc[model, "mean_width"]), float(panel.loc[model, "hit_ratio"])
            if y < hit_floor:
                _draw_below_floor(axis, x, y, hit_floor, marker=MODEL_MARKERS[model], color=MODEL_COLORS[model],
                                  size=PAPER_MARKER_AREA, name=model_label(model), placed=placed)
                continue
            bounds = intervals.get((value, model))
            if bounds is not None and model != "manski":
                axis.errorbar(
                    x, y,
                    xerr=[[max(x - bounds["mean_width_ci_lower"], 0.0)], [max(bounds["mean_width_ci_upper"] - x, 0.0)]],
                    yerr=[[max(y - bounds["hit_ratio_ci_lower"], 0.0)], [max(bounds["hit_ratio_ci_upper"] - y, 0.0)]],
                    fmt="none", ecolor=MODEL_COLORS[model], elinewidth=1.0, capsize=2.0, zorder=4.5,
                )
            axis.scatter(x, y, marker=MODEL_MARKERS[model], color=MODEL_COLORS[model],
                         s=1.5 * PAPER_MARKER_AREA if model == "naturalness" else PAPER_MARKER_AREA,
                         edgecolor="white", linewidth=0.6, zorder=6 if model == "naturalness" else 5)
        axis.set_title(_arm_facet_title(facet, value))
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.set_xlim(0.0, 1.02)
        axis.set_ylim(hit_floor, 1.005)
        axis.grid(alpha=0.22)
    handles = [
        Line2D([], [], color=MODEL_COLORS["naturalness"], marker="o", linestyle="none", markersize=7,
               label=model_label("naturalness")),
        *[Line2D([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], linestyle="none", markersize=6,
                 label=model_label(m)) for m in ARM_COMPARATORS],
        *[Line2D([], [], color=ink, marker=marker, linestyle="none", markersize=5.5, label=label)
          for marker, ink, label in ABLATION_RULE_STYLES.values()],
        *[Line2D([], [], color=inks[c], marker="o", markersize=3, linewidth=1.1,
                 label=rf"Fixed $\tilde{{c}}_1$ = {c:g}") for c in fixed_c],
    ]
    _finalize_shared_axes_figure(figure, axes, handles, xlabel="Mean interval width", ylabel="Hit ratio",
                                 legend_text_style=None, label_fontsize=None)
    return _save_figure(figure, _figures_dir(output_dir) / f"grounding_ablation_frontier_by_{facet}", plot_formats)


@matplotlib.rc_context(PAPER_RC)
def make_modified_target_validity_plot(
    strata: pd.DataFrame,
    output_dir: Path,
    *,
    arm: str = "effect_shift",
    grounding: str = "per_site",
    by_study: bool = False,
    plot_formats: Sequence[str] = DEFAULT_PLOT_FORMATS,
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = ARM_FIGURE_SEED,
) -> list[Path]:
    """Draw validity against the size of a target modification, method by method.

    Pooled, panel (a) shows the hit ratio and panel (b) the share of intervals that
    exclude zero on the side opposite the reference's 90% interval; by study, only the
    hit ratio is drawn, one panel per study. Points carry replication-bootstrap
    intervals, size 0 is the unmodified target, and Manski, which never excludes
    zero, is left out of (b). The legend is one row. A manuscript figure, drawn at
    :data:`PAPER_FIGURE_WIDTH` under :data:`PAPER_RC`.

    Args:
        strata: The arms' stratum table (``summary/arms_strata.parquet``).
        output_dir: Directory whose ``figures/`` subdirectory receives the files.
        arm: A key of :data:`ARM_SIZE_AXES`.
        grounding: Naturalness grounding to draw.
        by_study: One hit-ratio panel per study instead of the pooled pair.
        plot_formats: File extensions to write.
        bootstrap_replicates: Replication resamples per interval.
        confidence_level: Level of the percentile intervals.
        seed: Bootstrap seed.

    Returns:
        Paths written; empty when the arm has no rows yet.
    """
    size_column, xlabel = ARM_SIZE_AXES[arm]
    modified = strata.loc[strata["arm"].eq(arm) & strata[size_column].gt(0)]
    if modified.empty:
        return []
    keep = (modified["method"].eq("naturalness") & modified["grounding"].eq(grounding)) | \
        modified["method"].isin(ARM_COMPARATORS)
    rows = pd.concat([_arm_reference_rows(strata, grounding), modified.loc[keep].assign(model=lambda f: f["method"])])
    rows = rows.assign(size=rows[size_column].astype(float))
    grouping = ["size", "study"] if by_study else ["size"]
    columns = sorted(rows["study"].unique()) if by_study else [None]
    sizes = sorted(rows["size"].unique())
    metrics = (("hit", "Hit ratio"), ("reversal", "Wrong-sign share"))
    if by_study:
        metrics = metrics[:1]
        figure, grid = _panel_grid(len(metrics), len(columns), width=PAPER_FIGURE_WIDTH / len(columns), height=2.4,
                                   sharex=True, sharey="row", layout="constrained")
        axes = grid
    else:
        # Pooled: the metrics side by side, so each metric is one "row" of a 1 x 2 grid.
        figure, grid = _panel_grid(1, len(metrics), width=PAPER_FIGURE_WIDTH / len(metrics), height=2.6,
                                   layout="constrained")
        axes = grid.T
    for row_index, (metric, ylabel) in enumerate(metrics):
        intervals = _arm_intervals(rows, grouping, metric, bootstrap_replicates=bootstrap_replicates,
                                   confidence_level=confidence_level, seed=seed)
        means = rows.groupby([*grouping, "model"])[metric].mean()
        top = 0.0
        for column_index, study in enumerate(columns):
            axis = axes[row_index][column_index]
            for model in ("manski", "balke_pearl", "transport", "naturalness"):
                if metric == "reversal" and model == "manski":
                    continue
                keys = [((size, study, model) if by_study else (size, model)) for size in sizes]
                y = np.array([means.get(key, np.nan) for key in keys], dtype=float)
                lower = np.array([intervals.get(key, {}).get("hit_ratio_ci_lower", np.nan) for key in keys])
                upper = np.array([intervals.get(key, {}).get("hit_ratio_ci_upper", np.nan) for key in keys])
                top = max(top, float(np.nanmax(upper)) if np.isfinite(upper).any() else 0.0)
                axis.errorbar(sizes, y, yerr=[np.clip(y - lower, 0, None), np.clip(upper - y, 0, None)],
                              color=MODEL_COLORS[model], marker=MODEL_MARKERS[model], markersize=5,
                              markeredgecolor="white", markeredgewidth=0.6, linewidth=1.5, elinewidth=1.0,
                              capsize=2.0, zorder=6 if model == "naturalness" else 4)
            if by_study:
                axis.set_title(str(study))
            else:
                axis.set_title(f"({'ab'[row_index]})", loc="left", fontweight="bold")
                # The panels plot different quantities, so each keeps its own y label.
                axis.set_ylabel(ylabel)
            axis.set_xticks(sizes)
            axis.set_xlim(sizes[0] - 0.012, sizes[-1] + 0.012)
            axis.grid(alpha=0.22)
        for axis in axes[row_index]:
            if metric == "hit":
                axis.set_ylim(0.0, 1.03)
            else:
                axis.set_ylim(-0.005, max(0.05, top * 1.12))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    handles = [
        Line2D([], [], color=MODEL_COLORS["naturalness"], marker="o", markersize=5, linewidth=1.5,
               label=model_label("naturalness")),
        *[Line2D([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], markersize=5, linewidth=1.5,
                 label=model_label(m))
          for m in ARM_COMPARATORS],
    ]
    # One x label for the grid; by study, the panels also share the hit-ratio label.
    _finalize_shared_axes_figure(figure, grid, handles, xlabel=xlabel,
                                 ylabel=metrics[0][1] if by_study else None, legend_rows=1, label_fontsize=None)
    stem = f"{arm}_validity" + ("_by_study" if by_study else "")
    return _save_figure(figure, _figures_dir(output_dir) / stem, plot_formats)


def make_mechanism_arm_plots(
    strata: pd.DataFrame,
    output_dir: Path,
    *,
    grounding: str = "per_site",
    plot_formats: Sequence[str] = ("pdf", "png"),
    bootstrap_replicates: int = 2000,
) -> list[Path]:
    """Write every mechanism-arm figure the stratum table supports.

    The grounding-ablation frontier by study and by ``beta``, and the validity
    of each target-modifying arm (:data:`ARM_SIZE_AXES`) pooled and by study.
    An arm without rows yet is skipped.

    Args:
        strata: The arms' stratum table (``summary/arms_strata.parquet``).
        output_dir: Directory whose ``figures/`` subdirectory receives the files.
        grounding: Naturalness grounding to draw.
        plot_formats: File extensions to write.
        bootstrap_replicates: Replication resamples per interval.

    Returns:
        Every path written.
    """
    paths: list[Path] = []
    for facet in ("study", "beta"):
        paths += make_grounding_ablation_frontier_plot(
            strata, output_dir, facet=facet, grounding=grounding, plot_formats=plot_formats,
            bootstrap_replicates=bootstrap_replicates)
    for arm in ARM_SIZE_AXES:
        for by_study in (False, True):
            paths += make_modified_target_validity_plot(
                strata, output_dir, arm=arm, grounding=grounding, by_study=by_study,
                plot_formats=plot_formats, bootstrap_replicates=bootstrap_replicates)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone Exp25 figure rendering."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "primary_leaf",
        help="Exp25 backend leaf containing the primary mc_intervals.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination for tables and figures (default: <leaf>/analysis)",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        default=list(DEFAULT_PLOT_FORMATS),
        help="Matplotlib output extensions, for example pdf png (default: pdf)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the Exp25 figure set from the command line."""

    args = parse_args(argv)
    result = run_analysis(
        args.primary_leaf,
        args.output_dir,
        plot_formats=args.plot_formats,
    )
    print(f"Exp25 figures: {Path(result['output_dir']) / FIGURES_DIRNAME}")
    print(f"Figure files: {len(result['figure_paths'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
