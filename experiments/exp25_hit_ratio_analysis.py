"""Summarize Exp25 reference hits and causal-sign agreement by study and beta.

Reads an Exp25 backend leaf and its ``baseline_variants/gamma_<rule>/`` leaves
(``gamma_dgp_oracle`` is the primary run; each B-Learner Gamma rule is a
separate model). A method interval *hits* when it contains the held-out RCT
CATE point estimate, and *agrees in sign* when its verdict (positive or
negative if it excludes zero, else inconclusive) matches that of the RCT Wald
90% interval. Conditional ratios use valid intervals; failure-penalized ones
count invalid intervals as misses. :func:`sign_metrics_from_matrix` splits the
verdict into ``decisiveness_rate`` and the correct, reversed and undetermined
shares of the decisive calls. These are not classification scores: each rate
is conditioned on the bound's own behaviour, never on the reference's power.

- Default (:func:`run_analysis`): summary CSVs, tables, report and figures.
- ``--failures-as-manski`` (:func:`run_failures_as_manski_analysis`): the
  paper's ManyLabs tables and frontier figures, with each stratum a method
  could not bound scored as its Manski interval.

``paper/reproduce_experiment1_manylabs.sh`` runs ``--failures-as-manski``.

This module is the orchestrator: the tables, the report and the command line. The
loaders and metrics live in :mod:`experiments.hit_ratio_metrics` and are re-exported here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd

from experiments.exp25_bootstrap_analysis import (
    CONTRAST_BOOTSTRAP_FILENAME,
    METRIC_BOOTSTRAP_FILENAME,
    bootstrap_sign_metrics,
)
from experiments.exp25_hit_ratio_plots import (
    infer_full_range_width,
    make_all_plots,
    make_sign_identification_frontier_plot,
    make_width_hit_frontier_by_study_plot,
    width_hit_matched_table,
)
# The data layer, re-exported so its public names stay importable from here.
from experiments.hit_ratio_metrics import (  # noqa: F401
    INTERVALS_FILENAME,
    EXP25_KEY_COLUMNS,
    EXP25_REQUIRED_COLUMNS,
    BLEARNER_METHOD,
    SELECTED_WIDTH_HIT_BETAS,
    SELECTED_WIDTH_HIT_MODELS,
    SIGN_TABLE_MODELS,
    PRIMARY_GAMMA_LABEL,
    BASELINE_VARIANTS_DIRNAME,
    GAMMA_LEAF_PREFIX,
    MODEL_ORDER,
    REFERENCE_BOUND_MODELS,
    FRONTIER_CONFIDENCE_LEVEL,
    SIGN_POSITIVE,
    SIGN_NEGATIVE,
    SIGN_INCONCLUSIVE,
    SIGN_ORDER,
    NEGATIVE_INDEX,
    INCONCLUSIVE_INDEX,
    POSITIVE_INDEX,
    SIGN_COUNT_COLUMNS,
    SIGN_METRIC_COLUMNS,
    POINT_IDENTIFIED_WIDTH,
    discover_exp25_gamma_leaves,
    load_exp25_models,
    interval_sign,
    annotate_reference_agreement,
    failures_as_manski,
    sign_confusion_matrix,
    sign_metrics_from_matrix,
    compute_sign_metrics,
    summarize_agreement,
    select_common_support,
    common_support_frontier,
    summarize_selected_method_width_hit,
    build_sign_confusion,
)

ANNOTATED_ROWS_FILENAME = "exp25_annotated_rows.parquet"
SELECTED_WIDTH_HIT_FILENAME = "exp25_selected_method_width_hit_summary.csv"
SIGN_TABLE_STUDY_LABELS = {
    "gainloss": "Gain-loss",
    "reciprocity": "Reciprocity",
    "allowedforbidden": "Allowed-forbidden",
}
SIGN_TABLE_FILENAMES = {
    ("beta",): "exp25_sign_table_by_beta.tex",
    ("study", "beta"): "exp25_sign_table_by_study_beta.tex",
}
# The width tables (``tab:manylabs-width`` and its by-study and by-pattern
# splits) list the studies alphabetically and name the selection covariates.
WIDTH_TABLE_STUDY_LABELS = {
    "allowedforbidden": "allowedforbidden",
    "gainloss": "gainloss",
    "reciprocity": "reciprocity",
}
SELECTION_COVARIATE_LABELS = {"age": "age", "gender": "sex", "polideo": "ideology"}
WIDTH_TABLE_FILENAMES = {
    ("beta",): "exp25_width_table_by_beta.tex",
    ("study", "beta"): "exp25_width_table_by_study_beta.tex",
    ("bias_covariate", "beta"): "exp25_width_table_by_selection_beta.tex",
}
# Default output directories of :func:`run_failures_as_manski_analysis`, keyed
# by whether Naturalness's non-optimal draws are also scored as Manski.
FAILURES_AS_MANSKI_DIRNAMES = {
    False: "analysis_failures_as_manski",
    True: "analysis_failures_as_manski_draws",
}
FAILURES_AS_MANSKI_MANIFEST = "manifest.json"



def _sign_table_cell(frame: pd.DataFrame) -> str:
    """Format one model's rows as ``decisiveness (correct-sign rate)``."""

    metrics = compute_sign_metrics(frame)
    correct = (
        "--"
        if metrics["n_model_decisive"] == 0
        else f"{metrics['sign_correct_rate']:.2f}"
    )
    return f"{metrics['decisiveness_rate']:.2f} ({correct})"


def sign_table_latex(
    annotated: pd.DataFrame,
    *,
    grouping: Sequence[str] = ("beta",),
) -> str:
    """Render decisiveness (correct-sign rate) as a paper-ready LaTeX tabular.

    Each cell is the share of strata whose interval excludes zero and, in
    parentheses, the share of those decisive intervals whose sign the RCT 90%
    interval confirms (a dash when the model is never decisive), on the common
    support of :func:`_manuscript_table_latex`. Only the tabular block is
    emitted; the caption and label belong to the paper.

    Args:
        annotated: Output of :func:`annotate_reference_agreement`.
        grouping: ``("beta",)`` for the pooled table or ``("study", "beta")``
            for the by-study split. Studies follow
            :data:`SIGN_TABLE_STUDY_LABELS` order, each block set off by a
            ``\\midrule``.

    Returns:
        LaTeX source, or an empty string when no listed model has usable
        common-support rows.
    """

    return _manuscript_table_latex(
        annotated,
        grouping=grouping,
        cell=_sign_table_cell,
        value_labels={"study": SIGN_TABLE_STUDY_LABELS},
    )


def _width_table_cell(frame: pd.DataFrame) -> str:
    """Format one model's rows as ``mean width (hit ratio)``."""

    return f"{frame['width'].mean():.2f} ({frame['rct_hit'].mean():.2f})"


def width_table_latex(
    annotated: pd.DataFrame,
    *,
    grouping: Sequence[str] = ("beta",),
) -> str:
    """Render mean width (hit ratio) as a paper-ready LaTeX tabular.

    The width counterpart of :func:`sign_table_latex`: each cell is the mean
    interval width over the row's strata and, in parentheses, the share whose
    interval contains the RCT point estimate. The paper's width tables use
    :func:`failures_as_manski` scoring, so every stratum is in the support.

    Args:
        annotated: Output of :func:`annotate_reference_agreement`.
        grouping: ``("beta",)`` for the pooled table, ``("study", "beta")`` for
            the by-study split, or ``("bias_covariate", "beta")`` for the split
            by selection covariate. Studies follow
            :data:`WIDTH_TABLE_STUDY_LABELS` order and selection covariates
            :data:`SELECTION_COVARIATE_LABELS`, each block set off by a
            ``\\midrule``.

    Returns:
        LaTeX source, or an empty string when no listed model has usable
        common-support rows.
    """

    return _manuscript_table_latex(
        annotated,
        grouping=grouping,
        cell=_width_table_cell,
        value_labels={
            "study": WIDTH_TABLE_STUDY_LABELS,
            "bias_covariate": SELECTION_COVARIATE_LABELS,
        },
    )


def _manuscript_table_latex(
    annotated: pd.DataFrame,
    *,
    grouping: Sequence[str],
    cell: Callable[[pd.DataFrame], str],
    value_labels: Mapping[str, Mapping[str, str]],
) -> str:
    """Render one paper table over the common support of its methods.

    Shared by :func:`sign_table_latex` and :func:`width_table_latex`. Unlike
    :func:`common_support_frontier`, every model of :data:`SIGN_TABLE_MODELS` in
    the run, reference bounds included, defines the common support, so all
    cells of a row share one stratum set, whose size is the ``$n$`` column.
    With several grouping columns, a ``\\midrule`` sets off each block of the
    first one.

    Args:
        annotated: Output of :func:`annotate_reference_agreement`.
        grouping: Columns defining one table row, excluding ``model``.
        cell: Formats the rows of one model in one table row.
        value_labels: Display labels per grouping column; values are shown by
            their label and ordered as the mapping lists them.

    Returns:
        LaTeX source, or an empty string when no listed model has usable
        common-support rows.
    """

    # Lazy import: experiments.utils pulls in the Naturalness grounder, which
    # would multiply this module's import time for one rendering helper.
    from experiments.utils import render_latex_table

    present = set(annotated["model"].astype(str))
    models = [(model, label) for model, label in SIGN_TABLE_MODELS if model in present]
    rows = annotated.loc[annotated["model"].astype(str).isin([m for m, _ in models])]
    common = select_common_support(rows, reference_models=())
    if common.empty:
        return ""
    keys = list(grouping)
    ranks = {
        key: {value: rank for rank, value in enumerate(labels)}
        for key, labels in value_labels.items()
    }

    def _order(item: tuple[tuple, pd.DataFrame]) -> tuple:
        """Sort labelled values in their listed order and everything else by value."""

        return tuple(
            (ranks[key].get(value, len(ranks[key])), str(value))
            if key in ranks
            else (0, value)
            for key, value in zip(keys, item[0])
        )

    groups = [
        (values if isinstance(values, tuple) else (values,), frame)
        for values, frame in common.groupby(keys, sort=True)
    ]
    headers = {"study": "Study", "beta": r"$\beta$", "bias_covariate": "Selection"}
    display_rows: list[dict[str, str]] = []
    midrule_before: list[int] = []
    previous_block = None
    for values, frame in sorted(groups, key=_order):
        condition = dict(zip(keys, values))
        if len(keys) > 1:
            if display_rows and values[0] != previous_block:
                midrule_before.append(len(display_rows))
            previous_block = values[0]
        row = {
            headers.get(key, key): (
                value_labels[key].get(value, str(value))
                if key in value_labels
                else f"{value:g}" if key == "beta" else str(value)
            )
            for key, value in condition.items()
        }
        for model, label in models:
            row[label] = cell(frame.loc[frame["model"].eq(model)])
        row["$n$"] = str(len(frame.drop_duplicates(list(EXP25_KEY_COLUMNS))))
        display_rows.append(row)
    return render_latex_table(
        pd.DataFrame(display_rows),
        column_format="l" * len(keys) + "c" * (len(models) + 1),
        midrule_before=midrule_before,
    )


def _format_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render selected columns of a summary frame as a markdown table."""

    display = frame.loc[:, list(columns)].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(3)
    return display.to_markdown(index=False)


def write_report(
    by_study: pd.DataFrame,
    pooled: pd.DataFrame,
    by_condition: pd.DataFrame,
    confusion: pd.DataFrame,
    output_dir: Path,
    *,
    contrasts: pd.DataFrame | None = None,
) -> Path:
    """Write the markdown summary of hit ratios and sign classification.

    Args:
        by_study: Summary grouped by study and beta.
        pooled: Summary grouped by beta only.
        by_condition: Summary grouped by study, bias covariate, and beta.
        confusion: Output of :func:`build_sign_confusion`.
        output_dir: Directory receiving ``exp25_hit_ratio_report.md``.
        contrasts: Optional ``contrast_summary`` from
            :func:`experiments.exp25_bootstrap_analysis.bootstrap_sign_metrics`,
            appended as a paired Naturalness-minus-competitor section.

    Returns:
        Path to the written report.
    """

    hit_columns = (
        "model", "n_strata", "valid_rate", "n_hits", "hit_ratio_conditional",
        "hit_ratio_failure_penalized", "ci_overlap_ratio_conditional",
        "n_sign_match", "sign_match_conditional",
        "sign_match_failure_penalized", "mean_valid_width", "mean_blearner_gamma",
    )
    sign_columns = (
        "model", "n_valid_eligible", "n_model_decisive", "decisiveness_rate",
        "sign_correct_rate", "sign_reversal_rate", "sign_undetermined_rate",
        "n_sign_correct", "n_sign_reversal", "n_sign_undetermined",
        "n_bound_abstained", "n_agree_inconclusive",
    )
    lines = [
        "# Exp25 reference hits and causal-sign agreement",
        "",
        "Hit: method interval contains the held-out RCT CATE point estimate.",
        "CI overlap: method interval and the RCT Wald 90% interval intersect,",
        "which does not penalize intervals narrower than the reference error.",
        "Sign agreement: method interval and the RCT Wald 90% interval imply",
        "the same verdict (positive, negative, or inconclusive).",
        "Conditional ratios use valid rows; failure-penalized ratios count",
        "invalid intervals as misses.",
        "",
        "## Pooled over studies and bias covariates",
        "",
    ]
    for beta, frame in pooled.groupby("beta", sort=True):
        lines += [f"### beta = {beta}", "", _format_table(frame, hit_columns), ""]
    lines += [
        "## Sign identification, pooled over studies and bias covariates",
        "",
        "`decisiveness_rate` is the share of valid strata whose identified set",
        "excludes zero. It is not a coverage claim: nothing here is calibrated",
        "to a containment guarantee. Conditional on those decisive calls, the",
        "three rates `sign_correct_rate`, `sign_reversal_rate`, and",
        "`sign_undetermined_rate` partition them and sum to one: the bound",
        "excluded zero on the right side, on the wrong side, or on a stratum",
        "where the RCT Wald interval cannot adjudicate. A reversal is the",
        "consequential failure -- a bound that excluded zero the wrong way.",
        "Every rate is conditioned on the bound's own behaviour, never on the",
        "reference's statistical power. Read them next to the mean width and",
        "the validity rate: a bound can raise `sign_correct_rate` by widening",
        "until it commits only on the easiest strata.",
        "",
    ]
    for beta, frame in pooled.groupby("beta", sort=True):
        lines += [f"### beta = {beta}", "", _format_table(frame, sign_columns), ""]
    lines += ["## By study", ""]
    for (study, beta), frame in by_study.groupby(["study", "beta"], sort=True):
        lines += [
            f"### {study}, beta = {beta}",
            "",
            _format_table(frame, hit_columns),
            "",
            _format_table(frame, sign_columns),
            "",
        ]
    lines += ["## Causal-verdict confusion (valid rows)", ""]
    for (model, beta), frame in confusion.groupby(["model", "beta"], observed=True):
        pivot = (
            frame.pivot_table(
                index="rct_sign",
                columns="model_sign",
                values="n",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=list(SIGN_ORDER), fill_value=0)
            .reindex(columns=list(SIGN_ORDER), fill_value=0)
            .reset_index()
        )
        lines += [
            f"### {model}, beta = {beta}",
            "",
            pivot.to_markdown(index=False),
            "",
        ]
    if contrasts is not None and not contrasts.empty:
        lines += [
            "## Paired Naturalness-minus-competitor contrasts",
            "",
            "Replication-cluster bootstrap. Every model sees an identical",
            "stratum set, so the same replication resample is applied to all of",
            "them and the difference is taken inside each draw, which removes",
            "the Monte Carlo noise both models share.",
            "`share_draws_favoring_naturalness` is the fraction of resamples in",
            "which Naturalness scores higher; it is a bootstrap share, not a",
            "frequentist p-value.",
            "",
        ]
        contrast_columns = (
            "competitor", "metric", "naturalness_estimate",
            "competitor_estimate", "difference", "ci_lower", "ci_upper",
            "share_draws_favoring_naturalness",
        )
        headline = contrasts.loc[
            contrasts["metric"].isin(
                ["decisiveness_rate", "sign_correct_rate", "sign_reversal_rate"]
            )
        ]
        for beta, frame in headline.groupby("beta", sort=True):
            lines += [
                f"### beta = {beta}",
                "",
                _format_table(frame, contrast_columns),
                "",
            ]
    lines += [
        "## By study and bias covariate",
        "",
        _format_table(
            by_condition,
            ("study", "bias_covariate", "beta", *hit_columns),
        ),
        "",
    ]
    path = output_dir / "exp25_hit_ratio_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_analysis(
    primary_leaf: str | Path,
    output_dir: str | Path | None = None,
    *,
    make_plots: bool = True,
    plot_formats: Sequence[str] = ("pdf",),
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260808,
) -> dict[str, object]:
    """Run the full Exp25 hit-ratio and sign-classification analysis.

    Args:
        primary_leaf: Backend leaf containing the primary
            ``mc_intervals.parquet``.
        output_dir: Destination for CSV tables and the markdown report.
            Defaults to ``<primary_leaf>/analysis``.
        make_plots: Whether to render the figure set from
            :mod:`experiments.exp25_hit_ratio_plots` into ``<output_dir>/figures``.
        plot_formats: Matplotlib output extensions such as ``png`` and ``pdf``.
        bootstrap_replicates: Replication-cluster bootstrap resamples for the
            sign metrics and frontier intervals; ``0`` skips the bootstrap and
            the frontier figure's uncertainty bars.
        confidence_level: Central percentile interval level for the bootstrap.
        bootstrap_seed: Deterministic bootstrap seed.

    Returns:
        Loaded tables and written paths, including ``figure_paths``,
        ``sign_table_paths`` (the LaTeX tables of :data:`SIGN_TABLE_FILENAMES`),
        ``metric_bootstrap``, and ``metric_contrasts``.
    """

    leaves = discover_exp25_gamma_leaves(primary_leaf)
    annotated = annotate_reference_agreement(load_exp25_models(leaves))
    common_rows = select_common_support(annotated)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(primary_leaf).expanduser().resolve() / "analysis"
    )
    destination.mkdir(parents=True, exist_ok=True)
    pooled = summarize_agreement(annotated, grouping=("beta",))
    by_study = summarize_agreement(annotated, grouping=("study", "beta"))
    by_condition = summarize_agreement(
        annotated,
        grouping=("study", "bias_covariate", "beta"),
    )
    by_bias = summarize_agreement(annotated, grouping=("bias_covariate", "beta"))
    common_support = common_support_frontier(
        annotated,
        grouping=("beta",),
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=FRONTIER_CONFIDENCE_LEVEL,
        seed=bootstrap_seed,
        replication_spread=True,
    )
    # The intersection itself is defined over all strata, so grouping by study
    # slices one common support rather than building three of them.
    common_support_by_study = common_support_frontier(
        annotated,
        grouping=("study", "beta"),
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=FRONTIER_CONFIDENCE_LEVEL,
        seed=bootstrap_seed,
        replication_spread=True,
    )
    selected_width_hit = summarize_selected_method_width_hit(
        common_rows,
        bootstrap_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    width_hit_matched = width_hit_matched_table(common_support)
    confusion = build_sign_confusion(annotated)
    metric_bootstrap = pd.DataFrame()
    metric_contrasts = pd.DataFrame()
    if bootstrap_replicates > 0:
        bootstrap = bootstrap_sign_metrics(
            annotated,
            grouping=("beta",),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        metric_bootstrap = bootstrap.metric_summary
        metric_contrasts = bootstrap.contrast_summary
    written = {
        "exp25_hit_ratio_pooled.csv": pooled,
        "exp25_hit_ratio_by_study.csv": by_study,
        "exp25_hit_ratio_by_bias_covariate.csv": by_bias,
        "exp25_hit_ratio_by_study_bias.csv": by_condition,
        "exp25_common_support_frontier.csv": common_support,
        "exp25_common_support_frontier_by_study.csv": common_support_by_study,
        "exp25_width_hit_matched.csv": width_hit_matched,
        SELECTED_WIDTH_HIT_FILENAME: selected_width_hit,
        "exp25_sign_confusion.csv": confusion,
        METRIC_BOOTSTRAP_FILENAME: metric_bootstrap,
        CONTRAST_BOOTSTRAP_FILENAME: metric_contrasts,
    }
    for name, frame in written.items():
        frame.to_csv(destination / name, index=False)
    sign_table_paths: list[Path] = []
    for grouping, name in SIGN_TABLE_FILENAMES.items():
        latex = sign_table_latex(annotated, grouping=grouping)
        if latex:
            path = destination / name
            path.write_text(latex, encoding="utf-8")
            sign_table_paths.append(path)
    # Persisted so the row-level figures can be redrawn without the leaves.
    annotated.to_parquet(destination / ANNOTATED_ROWS_FILENAME, index=False)
    report_path = write_report(
        by_study,
        pooled,
        by_condition,
        confusion,
        destination,
        contrasts=metric_contrasts,
    )
    figure_paths: list[Path] = []
    if make_plots:
        figure_paths = make_all_plots(
            annotated,
            pooled,
            confusion,
            destination,
            common_support=common_support,
            common_support_by_study=common_support_by_study,
            bootstrap=metric_bootstrap if not metric_bootstrap.empty else None,
            plot_formats=plot_formats,
        )
    return {
        "output_dir": destination,
        "leaves": leaves,
        "annotated": annotated,
        "common_rows": common_rows,
        "pooled": pooled,
        "by_study": by_study,
        "by_bias_covariate": by_bias,
        "by_study_bias": by_condition,
        "common_support": common_support,
        "common_support_by_study": common_support_by_study,
        "width_hit_matched": width_hit_matched,
        "selected_width_hit": selected_width_hit,
        "sign_confusion": confusion,
        "metric_bootstrap": metric_bootstrap,
        "metric_contrasts": metric_contrasts,
        "report_path": report_path,
        "sign_table_paths": sign_table_paths,
        "figure_paths": figure_paths,
    }


def run_failures_as_manski_analysis(
    primary_leaf: str | Path,
    output_dir: str | Path | None = None,
    *,
    draw_level: bool = False,
    make_plots: bool = True,
    plot_formats: Sequence[str] = ("pdf",),
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260808,
) -> dict[str, object]:
    """Write the paper's ManyLabs tables and frontier figures.

    Scores every stratum with :func:`failures_as_manski`, then writes the five
    tables (:data:`WIDTH_TABLE_FILENAMES`, :data:`SIGN_TABLE_FILENAMES`), the
    per-study width-containment frontier and the sign-identification frontier
    as :func:`run_analysis` does. The frontier's full-range reference line is
    inferred before re-scoring, which replaces the full-range fallbacks it is
    read from.

    Args:
        primary_leaf: Backend leaf whose ``baseline_variants`` hold the
            ``gamma_dgp_oracle`` and ``identification_bounds`` leaves.
        output_dir: Destination for tables, figures and the manifest; defaults
            to the :data:`FAILURES_AS_MANSKI_DIRNAMES` entry inside the leaf.
        draw_level: Also score Naturalness's non-optimal draws as Manski, the
            sensitivity variant of :func:`failures_as_manski`.
        make_plots: Whether to render the two frontier figures.
        plot_formats: Matplotlib output extensions such as ``png`` and ``pdf``.
        bootstrap_replicates: Replication-cluster bootstrap resamples for the
            frontier intervals and the sign metrics; zero skips both.
        confidence_level: Central percentile level of the sign-metric
            bootstrap; the frontier keeps :data:`FRONTIER_CONFIDENCE_LEVEL`.
        bootstrap_seed: Deterministic bootstrap seed.

    Returns:
        The output directory, the re-scored annotated rows, the summary tables,
        and the written ``table_paths``, ``figure_paths`` and ``manifest_path``.
    """

    leaves = discover_exp25_gamma_leaves(primary_leaf)
    models = load_exp25_models(leaves)
    full_range_width = infer_full_range_width(annotate_reference_agreement(models))
    annotated = annotate_reference_agreement(
        failures_as_manski(models, draw_level=draw_level)
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(primary_leaf).expanduser().resolve()
        / FAILURES_AS_MANSKI_DIRNAMES[draw_level]
    )
    destination.mkdir(parents=True, exist_ok=True)
    by_study = common_support_frontier(
        annotated,
        grouping=("study", "beta"),
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=FRONTIER_CONFIDENCE_LEVEL,
        seed=bootstrap_seed,
        replication_spread=True,
        reference_models=(),
    )
    pooled_frontier = common_support_frontier(
        annotated, grouping=("beta",), reference_models=()
    )
    pooled = summarize_agreement(annotated, grouping=("beta",))
    metric_bootstrap = pd.DataFrame()
    if bootstrap_replicates > 0:
        metric_bootstrap = bootstrap_sign_metrics(
            annotated,
            grouping=("beta",),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        ).metric_summary
    written = {
        "exp25_common_support_frontier_by_study.csv": by_study,
        "exp25_common_support_frontier.csv": pooled_frontier,
        "exp25_hit_ratio_pooled.csv": pooled,
        METRIC_BOOTSTRAP_FILENAME: metric_bootstrap,
    }
    for name, frame in written.items():
        frame.to_csv(destination / name, index=False)
    table_paths: list[Path] = []
    for render, filenames in (
        (width_table_latex, WIDTH_TABLE_FILENAMES),
        (sign_table_latex, SIGN_TABLE_FILENAMES),
    ):
        for grouping, name in filenames.items():
            latex = render(annotated, grouping=grouping)
            if latex:
                path = destination / name
                path.write_text(latex, encoding="utf-8")
                table_paths.append(path)
    figure_paths: list[Path] = []
    if make_plots:
        figure_paths += make_width_hit_frontier_by_study_plot(
            by_study,
            destination,
            full_range_width=full_range_width,
            plot_formats=plot_formats,
        )
        figure_paths += make_sign_identification_frontier_plot(
            pooled,
            destination,
            bootstrap=metric_bootstrap if not metric_bootstrap.empty else None,
            plot_formats=plot_formats,
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_leaf": str(Path(primary_leaf).expanduser().resolve()),
        "scoring": (
            "failed strata get the stratum's Manski interval"
            + (
                "; Naturalness non-optimal draws also replaced by Manski"
                if draw_level
                else ""
            )
        ),
        "strata_scored": int(len(annotated.drop_duplicates(list(EXP25_KEY_COLUMNS)))),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "generator": "experiments.exp25_hit_ratio_analysis.run_failures_as_manski_analysis",
        "tables": [path.name for path in table_paths],
        "figures": [str(path.relative_to(destination)) for path in figure_paths],
    }
    manifest_path = destination / FAILURES_AS_MANSKI_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "output_dir": destination,
        "leaves": leaves,
        "annotated": annotated,
        "full_range_width": full_range_width,
        "common_support_by_study": by_study,
        "common_support": pooled_frontier,
        "pooled": pooled,
        "metric_bootstrap": metric_bootstrap,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "manifest_path": manifest_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Exp25 hit-ratio analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "primary_leaf",
        help="Exp25 backend leaf containing baseline_variants/gamma_<rule>",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Destination for tables and report (default: <leaf>/analysis, or "
            "the failures-as-Manski directory with --failures-as-manski)"
        ),
    )
    parser.add_argument(
        "--failures-as-manski",
        choices=("stratum", "draws"),
        default=None,
        help=(
            "Write the manuscript's tables and frontier figures with every "
            "failed stratum scored as its Manski interval; 'draws' also "
            "replaces Naturalness's non-optimal draws (default: run the "
            "common-support analysis instead)"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tables and the report without rendering figures",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        default=["pdf"],
        help="Matplotlib output extensions, for example pdf png (default: pdf)",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help=(
            "Replication-cluster bootstrap resamples for the sign metrics; "
            "zero skips the bootstrap and its figure error bars (default: 2000)"
        ),
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Central percentile interval level (default: 0.95)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260808,
        help="Deterministic bootstrap seed",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Exp25 hit-ratio analysis from the command line."""

    args = parse_args(argv)
    if args.failures_as_manski is not None:
        result = run_failures_as_manski_analysis(
            args.primary_leaf,
            args.output_dir,
            draw_level=args.failures_as_manski == "draws",
            make_plots=not args.no_plots,
            plot_formats=args.plot_formats,
            bootstrap_replicates=args.bootstrap_replicates,
            confidence_level=args.confidence_level,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(f"Exp25 failures-as-Manski outputs: {result['output_dir']}")
        print(f"Table files: {len(result['table_paths'])}")
        print(f"Figure files: {len(result['figure_paths'])}")
        return 0
    result = run_analysis(
        args.primary_leaf,
        args.output_dir,
        make_plots=not args.no_plots,
        plot_formats=args.plot_formats,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"Exp25 analysis outputs: {result['output_dir']}")
    print(f"Gamma runs: {[label for label, _ in result['leaves']]}")
    print(f"Figure files: {len(result['figure_paths'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
