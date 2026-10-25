"""Replication-cluster bootstrap for the Exp25 sign-identification metrics.

Every rate reported by :mod:`experiments.exp25_hit_ratio_analysis` is a ratio
of stratum counts pooled over Monte Carlo replications, so a bare point
estimate hides two different sources of uncertainty. The stratum counts are not
independent — all strata inside one replication share a single draw of the
observational sample — and several cells rest on very few decisive calls, where
a correct-sign rate of ``2/3`` reads like a result until its denominator is shown.

This module resamples whole replications with replacement, which preserves the
within-replication dependence, and rescores the metrics on each resample. Two
design choices matter for how the output should be read:

Matrix-valued resampling
    Every metric in :func:`experiments.exp25_hit_ratio_analysis.sign_metrics_from_matrix`
    is a function of the ``3 x 3`` verdict confusion matrix alone, so a
    resample is a *sum of per-replication matrices* rather than a re-derivation
    of row-level flags. Scoring a bootstrap draw is then a handful of numpy
    operations on a ``(B, 3, 3)`` array.

Shared resamples across models
    Within one grouping cell every model is evaluated on an identical stratum
    set, so the same replication resample is applied to all of them. This keeps
    the models paired inside each draw, which is what makes the contrasts in
    :func:`bootstrap_sign_metrics` valid and what makes a bootstrap band around
    the sign-identification frontier coherent rather than a stack of
    independently jittered points.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from experiments.exp25_hit_ratio_analysis import (
    SIGN_COUNT_COLUMNS,
    SIGN_METRIC_COLUMNS,
    annotate_reference_agreement,
    discover_exp25_gamma_leaves,
    load_exp25_models,
    sign_confusion_matrix,
    sign_metrics_from_matrix,
)
from experiments.interval_stats import percentile_interval


# Kept under its old private name for external importers.
_percentile_interval = percentile_interval

NATURALNESS_MODEL = "naturalness"
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 20260808
METRIC_BOOTSTRAP_FILENAME = "exp25_sign_metric_bootstrap.csv"
CONTRAST_BOOTSTRAP_FILENAME = "exp25_sign_metric_contrasts.csv"


@dataclass(frozen=True)
class Exp25BootstrapOutputs:
    """Bootstrap tables and the settings that produced them.

    Attributes:
        metric_summary: One row per grouping level, model, and metric, with the
            pooled point estimate and its percentile interval.
        contrast_summary: One row per grouping level, competitor, and metric,
            holding the paired Naturalness-minus-competitor difference.
        grouping: Column names that defined one bootstrap cell.
        bootstrap_replicates: Number of replication resamples drawn per cell.
        confidence_level: Central percentile interval level.
        seed: Deterministic bootstrap seed.
    """

    metric_summary: pd.DataFrame
    contrast_summary: pd.DataFrame
    grouping: tuple[str, ...]
    bootstrap_replicates: int
    confidence_level: float
    seed: int


def replication_confusion_matrices(
    annotated: pd.DataFrame,
    replications: Sequence[object],
) -> np.ndarray:
    """Build one verdict confusion matrix per Monte Carlo replication.

    Replications are the bootstrap's resampling unit, so the matrices are
    aligned to a caller-supplied replication order shared by every model in the
    same cell. A replication in which the model produced no usable stratum
    contributes an all-zero matrix rather than being dropped, which keeps the
    resample weights comparable across models.

    Args:
        annotated: Rows for one model within one grouping cell, as produced by
            :func:`experiments.exp25_hit_ratio_analysis.annotate_reference_agreement`.
        replications: Replication identifiers defining the array's first axis.

    Returns:
        An array of shape ``(len(replications), 3, 3)``.
    """

    positions = {value: index for index, value in enumerate(replications)}
    matrices = np.zeros((len(replications), 3, 3), dtype=float)
    for replication, frame in annotated.groupby("replication", sort=False):
        position = positions.get(replication)
        if position is None:
            continue
        matrices[position] = sign_confusion_matrix(frame)
    return matrices


def bootstrap_sign_metrics(
    annotated: pd.DataFrame,
    *,
    grouping: Sequence[str] = ("beta",),
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Exp25BootstrapOutputs:
    """Attach replication-cluster uncertainty to every sign-identification metric.

    Within each grouping cell the replications are drawn with replacement once
    per bootstrap iteration and that single draw is applied to every model, so
    the resulting per-model draws stay paired. Contrasts are then taken inside
    each iteration, which removes the shared Monte Carlo noise that both models
    see and yields a much tighter interval than differencing two independent
    marginal intervals would.

    Args:
        annotated: Output of
            :func:`experiments.exp25_hit_ratio_analysis.annotate_reference_agreement`.
        grouping: Columns defining one bootstrap cell, excluding ``model``.
        bootstrap_replicates: Number of replication resamples per cell.
        confidence_level: Central percentile interval level in ``(0, 1)``.
        seed: Deterministic bootstrap seed.

    Returns:
        Metric and contrast tables together with the settings used.

    Raises:
        ValueError: If the resampling settings are out of range or the table
            lacks a ``replication`` column.
    """

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if "replication" not in annotated.columns:
        raise ValueError("Bootstrap requires a replication column")

    keys = list(grouping)
    metric_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for values, cell in annotated.groupby(keys, sort=True, dropna=False):
        condition = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        usable = cell.loc[cell["is_valid"] & cell["reference_eligible"]]
        replications = sorted(pd.unique(cell["replication"]))
        if not replications:
            continue
        # One resample per iteration, reused by every model in the cell so the
        # models stay paired on identical strata inside each draw.
        sampled = rng.integers(
            0, len(replications), size=(bootstrap_replicates, len(replications))
        )
        draws_by_model: dict[str, dict[str, np.ndarray]] = {}
        observed_by_model: dict[str, dict[str, np.ndarray]] = {}
        for model, frame in usable.groupby("model", sort=False, observed=True):
            matrices = replication_confusion_matrices(frame, replications)
            resampled = matrices[sampled].sum(axis=1)
            draws = sign_metrics_from_matrix(resampled)
            observed = sign_metrics_from_matrix(matrices.sum(axis=0))
            draws_by_model[str(model)] = draws
            observed_by_model[str(model)] = observed
            for metric in (*SIGN_COUNT_COLUMNS, *SIGN_METRIC_COLUMNS):
                lower, upper = percentile_interval(
                    np.asarray(draws[metric], dtype=float), confidence_level
                )
                metric_values = np.asarray(draws[metric], dtype=float)
                metric_rows.append(
                    {
                        **condition,
                        "model": str(model),
                        "metric": metric,
                        "estimate": float(observed[metric]),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "bootstrap_se": (
                            float(np.nanstd(metric_values, ddof=1))
                            if np.isfinite(metric_values).sum() > 1
                            else np.nan
                        ),
                        "n_finite_draws": int(np.isfinite(metric_values).sum()),
                        "n_replications": int(len(replications)),
                        "n_valid_eligible": int(len(frame)),
                    }
                )
        reference = draws_by_model.get(NATURALNESS_MODEL)
        if reference is None:
            continue
        reference_observed = observed_by_model[NATURALNESS_MODEL]
        for model, draws in draws_by_model.items():
            if model == NATURALNESS_MODEL:
                continue
            competitor_observed = observed_by_model[model]
            for metric in SIGN_METRIC_COLUMNS:
                difference = np.asarray(
                    reference[metric], dtype=float
                ) - np.asarray(draws[metric], dtype=float)
                lower, upper = percentile_interval(difference, confidence_level)
                finite = difference[np.isfinite(difference)]
                contrast_rows.append(
                    {
                        **condition,
                        "competitor": model,
                        "metric": metric,
                        "naturalness_estimate": float(reference_observed[metric]),
                        "competitor_estimate": float(competitor_observed[metric]),
                        "difference": float(
                            reference_observed[metric] - competitor_observed[metric]
                        ),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        # A bootstrap share, not a frequentist p-value: the
                        # fraction of resamples in which Naturalness scores
                        # higher on this metric.
                        "share_draws_favoring_naturalness": (
                            float(np.mean(finite > 0.0)) if finite.size else np.nan
                        ),
                        "n_finite_draws": int(finite.size),
                        "n_replications": int(len(replications)),
                    }
                )
    return Exp25BootstrapOutputs(
        metric_summary=pd.DataFrame(metric_rows),
        contrast_summary=pd.DataFrame(contrast_rows),
        grouping=tuple(keys),
        bootstrap_replicates=int(bootstrap_replicates),
        confidence_level=float(confidence_level),
        seed=int(seed),
    )


def metric_interval_lookup(
    metric_summary: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Pivot one metric out of the long bootstrap table for plotting.

    Args:
        metric_summary: ``metric_summary`` from :func:`bootstrap_sign_metrics`.
        metric: Metric name to extract, for example ``sign_correct_rate``.

    Returns:
        The subset of rows for that metric, indexed positionally, with the
        ``metric`` column dropped. Empty when the metric is absent, which lets
        figures degrade to point markers without a bootstrap table.
    """

    if metric_summary is None or metric_summary.empty:
        return pd.DataFrame()
    subset = metric_summary.loc[metric_summary["metric"].eq(metric)]
    return subset.drop(columns=["metric"]).reset_index(drop=True)


def run_analysis(
    primary_leaf: str | Path,
    output_dir: str | Path | None = None,
    *,
    grouping: Sequence[str] = ("beta",),
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Exp25BootstrapOutputs:
    """Bootstrap the Exp25 sign metrics straight from a set of result leaves.

    This is the standalone entry point; :func:`experiments.exp25_hit_ratio_analysis.run_analysis`
    calls :func:`bootstrap_sign_metrics` directly on the table it has already
    annotated rather than reloading the leaves.

    Args:
        primary_leaf: Backend leaf holding the ``baseline_variants/`` Gamma
            leaves, including the primary ``gamma_dgp_oracle`` run.
        output_dir: Destination for the two CSV tables. Defaults to
            ``<primary_leaf>/analysis``.
        grouping: Columns defining one bootstrap cell, excluding ``model``.
        bootstrap_replicates: Number of replication resamples per cell.
        confidence_level: Central percentile interval level.
        seed: Deterministic bootstrap seed.

    Returns:
        The bootstrap tables, which are also written to ``output_dir``.
    """

    leaves = discover_exp25_gamma_leaves(primary_leaf)
    annotated = annotate_reference_agreement(load_exp25_models(leaves))
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else Path(primary_leaf).expanduser().resolve() / "analysis"
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = bootstrap_sign_metrics(
        annotated,
        grouping=grouping,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    outputs.metric_summary.to_csv(
        destination / METRIC_BOOTSTRAP_FILENAME, index=False
    )
    outputs.contrast_summary.to_csv(
        destination / CONTRAST_BOOTSTRAP_FILENAME, index=False
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the standalone Exp25 bootstrap."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "primary_leaf",
        help="Exp25 backend leaf containing baseline_variants/gamma_<rule>",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination for the bootstrap tables (default: <leaf>/analysis)",
    )
    parser.add_argument(
        "--grouping",
        nargs="+",
        default=["beta"],
        help="Columns defining one bootstrap cell (default: beta)",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help=f"Replication resamples per cell (default: {DEFAULT_BOOTSTRAP_REPLICATES})",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help=f"Central percentile level (default: {DEFAULT_CONFIDENCE_LEVEL})",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Deterministic bootstrap seed",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Exp25 metric bootstrap from the command line."""

    args = parse_args(argv)
    outputs = run_analysis(
        args.primary_leaf,
        args.output_dir,
        grouping=args.grouping,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence_level=args.confidence_level,
        seed=args.bootstrap_seed,
    )
    print(f"Exp25 bootstrap rows: {len(outputs.metric_summary)}")
    print(f"Exp25 contrast rows: {len(outputs.contrast_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
