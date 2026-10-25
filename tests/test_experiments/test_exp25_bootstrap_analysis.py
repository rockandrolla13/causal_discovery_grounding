"""Tests for the Exp25 replication-cluster bootstrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.exp25_bootstrap_analysis as bootstrap
import experiments.hit_ratio_metrics as hit_metrics

from tests.test_experiments.test_exp25_hit_ratio_analysis import (
    _interval_fixture,
    _write_leaf,
)


def _multi_replication_annotated(n_replications: int = 6) -> pd.DataFrame:
    """Build an annotated fixture spanning several Monte Carlo replications.

    Each replication shifts the B-Learner endpoints so the resampling has
    something to vary, while Naturalness and the CB learners stay fixed. This
    keeps the expected contrast deterministic in sign while still exercising
    the bootstrap machinery.

    Args:
        n_replications: Number of replications to synthesize.

    Returns:
        A model-long annotated table with a ``replication`` column.
    """

    frames: list[pd.DataFrame] = []
    for replication in range(n_replications):
        table = _interval_fixture(blearner_shift=0.05 * replication)
        table = table.assign(replication=replication, seed=42 + replication)
        shared = table.loc[~table["method"].eq("blearner")].assign(
            gamma_run="dgp_oracle",
            model=lambda frame: frame["method"].astype(str),
        )
        blearner = table.loc[table["method"].eq("blearner")].assign(
            gamma_run="dgp_oracle", model="blearner[dgp_oracle]"
        )
        frames.append(pd.concat([shared, blearner], ignore_index=True, sort=False))
    models = pd.concat(frames, ignore_index=True, sort=False)
    return hit_metrics.annotate_reference_agreement(models)


def test_replication_confusion_matrices_align_to_the_given_order() -> None:
    """Matrices are indexed by the caller's replication order, zero when absent."""

    annotated = _multi_replication_annotated(n_replications=3)
    naturalness = annotated.loc[annotated["model"].eq("naturalness")]
    matrices = bootstrap.replication_confusion_matrices(naturalness, [0, 1, 2, 99])
    assert matrices.shape == (4, 3, 3)
    # Three usable strata per replication, and the absent replication is zero.
    assert matrices[0].sum() == pytest.approx(3.0)
    assert matrices[3].sum() == pytest.approx(0.0)
    # Summing the per-replication matrices reproduces the pooled matrix, which
    # is the identity the bootstrap relies on.
    assert matrices[:3].sum(axis=0) == pytest.approx(
        hit_metrics.sign_confusion_matrix(naturalness)
    )


def test_bootstrap_estimates_match_the_pooled_summary() -> None:
    """Point estimates equal the unresampled metric, and intervals bracket them."""

    annotated = _multi_replication_annotated()
    outputs = bootstrap.bootstrap_sign_metrics(
        annotated, grouping=("beta",), bootstrap_replicates=200, seed=7
    )
    pooled = hit_metrics.summarize_agreement(annotated, grouping=("beta",))
    metrics = outputs.metric_summary
    assert not metrics.empty
    for metric in ("sign_correct_rate", "decisiveness_rate", "sign_reversal_rate"):
        row = metrics.loc[
            metrics["metric"].eq(metric) & metrics["model"].eq("naturalness")
        ].iloc[0]
        expected = float(
            pooled.loc[pooled["model"].eq("naturalness"), metric].iloc[0]
        )
        assert row["estimate"] == pytest.approx(expected, nan_ok=True)
        if np.isfinite(expected) and np.isfinite(row["ci_lower"]):
            assert row["ci_lower"] <= expected <= row["ci_upper"]


def test_bootstrap_contrasts_pair_naturalness_against_each_competitor() -> None:
    """Contrasts cover every competitor and agree with the marginal estimates."""

    annotated = _multi_replication_annotated()
    outputs = bootstrap.bootstrap_sign_metrics(
        annotated, grouping=("beta",), bootstrap_replicates=200, seed=7
    )
    contrasts = outputs.contrast_summary
    assert set(contrasts["competitor"]) == {
        "blearner[dgp_oracle]",
        "cb_ipw",
        "cb_dr",
    }
    row = contrasts.loc[
        contrasts["competitor"].eq("cb_dr")
        & contrasts["metric"].eq("decisiveness_rate")
    ].iloc[0]
    # CB-DR spans the full range on every stratum, so it never commits while
    # Naturalness always does.
    assert row["naturalness_estimate"] == pytest.approx(1.0)
    assert row["competitor_estimate"] == pytest.approx(0.0)
    assert row["difference"] == pytest.approx(1.0)
    assert row["share_draws_favoring_naturalness"] == pytest.approx(1.0)


def test_bootstrap_uses_a_shared_resample_across_models() -> None:
    """Every model in a cell is scored on the same replication draw."""

    annotated = _multi_replication_annotated()
    first = bootstrap.bootstrap_sign_metrics(
        annotated, grouping=("beta",), bootstrap_replicates=150, seed=11
    )
    second = bootstrap.bootstrap_sign_metrics(
        annotated, grouping=("beta",), bootstrap_replicates=150, seed=11
    )
    # A fixed seed reproduces the draws exactly, which is what makes the paired
    # contrasts reproducible rather than an artifact of independent sampling.
    pd.testing.assert_frame_equal(first.metric_summary, second.metric_summary)
    pd.testing.assert_frame_equal(first.contrast_summary, second.contrast_summary)


def test_single_replication_collapses_the_interval() -> None:
    """With one cluster every resample is identical, so the interval is a point."""

    annotated = _multi_replication_annotated(n_replications=1)
    outputs = bootstrap.bootstrap_sign_metrics(
        annotated, grouping=("beta",), bootstrap_replicates=50, seed=3
    )
    row = outputs.metric_summary.loc[
        outputs.metric_summary["metric"].eq("sign_correct_rate")
        & outputs.metric_summary["model"].eq("naturalness")
    ].iloc[0]
    assert row["ci_lower"] == pytest.approx(row["ci_upper"])
    assert row["bootstrap_se"] == pytest.approx(0.0)


def test_bootstrap_rejects_invalid_settings() -> None:
    """Out-of-range resampling settings fail before any work is done."""

    annotated = _multi_replication_annotated(n_replications=2)
    with pytest.raises(ValueError, match="bootstrap_replicates must be positive"):
        bootstrap.bootstrap_sign_metrics(annotated, bootstrap_replicates=0)
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap.bootstrap_sign_metrics(annotated, confidence_level=1.5)
    with pytest.raises(ValueError, match="replication column"):
        bootstrap.bootstrap_sign_metrics(annotated.drop(columns=["replication"]))


def test_metric_interval_lookup_survives_a_missing_table() -> None:
    """Figures degrade to bare markers rather than failing without a bootstrap."""

    assert bootstrap.metric_interval_lookup(None, "sign_correct_rate").empty
    assert bootstrap.metric_interval_lookup(
        pd.DataFrame(), "sign_correct_rate"
    ).empty


def test_run_analysis_writes_both_bootstrap_tables(tmp_path: Path) -> None:
    """The standalone entry point writes the metric and contrast CSVs."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    outputs = bootstrap.run_analysis(
        primary, tmp_path / "analysis", bootstrap_replicates=25
    )
    destination = tmp_path / "analysis"
    assert (destination / bootstrap.METRIC_BOOTSTRAP_FILENAME).is_file()
    assert (destination / bootstrap.CONTRAST_BOOTSTRAP_FILENAME).is_file()
    assert not outputs.metric_summary.empty
    assert outputs.bootstrap_replicates == 25
