"""Tests for shared experiment utilities."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from causal_grounding.loaders.cohort import (
    OBSERVED_GENDER_COHORT_RULE,
    restrict_to_observed_gender,
)
from experiments.exp16_osrct_calibrated_policy import Exp16Config
from experiments.utils import (
    build_configured_naturalness_grid,
    build_lp_probability_model_kwargs,
    build_stage1_predictor_kwargs,
    render_latex_table,
    resolve_lp_probability_backend,
    resolve_target_trigger_probability_backend,
    summarize_lp_posterior_bounds,
)


def test_observed_gender_cohort_uses_original_missingness() -> None:
    """Site-mean gender imputations should be removed using ``resp_sex``."""

    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "original_study": ["a", "a", "a", "b"],
            "site": ["pi", "pi", "x", "pi"],
            "resp_sex": ["female", np.nan, "male", np.nan],
            "resp_gender": [1.0, 0.65, 0.0, 0.65],
        }
    )

    filtered, diagnostics = restrict_to_observed_gender(data)

    assert filtered["id"].tolist() == [1, 3]
    overall = diagnostics.loc[
        diagnostics["study"].eq("__all__")
        & diagnostics["site"].eq("__all__")
    ].iloc[0]
    assert overall["cohort_filter"] == OBSERVED_GENDER_COHORT_RULE
    assert overall["n_rows_before"] == 4
    assert overall["n_rows_excluded"] == 2
    assert overall["n_rows_after"] == 2
    assert overall["n_unique_ids_excluded"] == 2


def test_tabicl_v2_backend_resolution_and_cache_defaults() -> None:
    """TabICLv2 stage1 resolution uses reproducible representation caching."""
    config = SimpleNamespace(
        stage1_backend="tabicl_v2",
        lp_probability_backend="stage1",
        tabicl_device="cuda:0",
        seed=17,
    )

    backend = resolve_lp_probability_backend(config)

    assert backend == "tabicl_v2"
    assert build_stage1_predictor_kwargs(backend, config) == {
        "predictor_kwargs": {"random_state": 17, "device": "cuda:0"}
    }
    assert build_lp_probability_model_kwargs(config, backend) == {
        "random_state": 17,
        "kv_cache": "repr",
        "device": "cuda:0",
    }


def test_target_trigger_backend_resolves_independently_from_lp() -> None:
    """Target triggering can follow Stage 1 while LP uses posterior draws."""

    config = SimpleNamespace(
        stage1_backend="tabicl_v2",
        lp_probability_backend="hierarchical_dirichlet_draws",
        target_trigger_probability_backend="stage1",
    )

    assert resolve_target_trigger_probability_backend(config) == "tabicl_v2"
    config.target_trigger_probability_backend = "lp"
    assert (
        resolve_target_trigger_probability_backend(config)
        == "hierarchical_dirichlet_draws"
    )


def test_xgboost_worker_limits_are_forwarded() -> None:
    """Experiment worker budgets constrain nested XGBoost parallelism."""

    config = SimpleNamespace(
        stage1_backend="xgboost",
        lp_probability_backend="stage1",
        stage1_xgboost_n_jobs=1,
        lp_xgboost_n_jobs=2,
        seed=19,
    )

    stage1_kwargs = build_stage1_predictor_kwargs("xgboost", config)
    lp_kwargs = build_lp_probability_model_kwargs(config, "xgboost")

    assert stage1_kwargs["predictor_kwargs"]["n_jobs"] == 1
    assert lp_kwargs["n_jobs"] == 2


def test_lp_posterior_summary_tracks_optimal_frequency() -> None:
    """Posterior LP summaries keep Exp15/Exp16 target-draw diagnostics."""

    samples = pd.DataFrame({
        "Z": [0, 0],
        "z_value_str": ["(0,)", "(0,)"],
        "lp_posterior_sample_id": [0, 1],
        "status": ["optimal", "error"],
        "lower": [-0.2, np.nan],
        "upper": [0.4, np.nan],
    })
    summary = summarize_lp_posterior_bounds(samples, ["Z"])
    assert summary.loc[0, "lp_posterior_prob_optimal"] == 0.5
    assert np.isclose(summary.loc[0, "width_mean"], 0.6)


def test_render_latex_table_column_format_and_midrules() -> None:
    """Custom column specs and group midrules leave the default layout alone."""

    table = pd.DataFrame({"Group": ["a", "a", "b"], "Value": ["1", "2", "3"]})

    default = render_latex_table(table)
    grouped = render_latex_table(table, column_format="lc", midrule_before=[0, 2])

    assert r"\begin{tabular}{ll}" in default
    assert default.count(r"\midrule") == 1
    lines = grouped.splitlines()
    assert lines[0] == r"\begin{tabular}{lc}"
    # Position 0 already follows the header rule, so only row 2 adds one.
    assert grouped.count(r"\midrule") == 2
    assert lines[lines.index(r"b & 3 \\") - 1] == r"\midrule"


def test_naturalness_grid_builder_gives_extra_candidates_their_own_states() -> None:
    """Extras join the grid as candidates conditioned on the LP covariates."""

    config = Exp16Config(naturalness_extra_candidates=("noise_01",))
    grid = build_configured_naturalness_grid(config)
    legacy = build_configured_naturalness_grid(Exp16Config())

    assert grid.covariates == [*config.covariates, "noise_01"]
    assert grid.conditioning_covariates == list(config.covariates)
    assert grid.candidate_covariates == grid.covariates
    assert legacy.covariates == list(Exp16Config().covariates)
    assert legacy.conditioning_covariates is None


def test_tabicl_checkpoint_sharing_is_opt_in() -> None:
    """Default TabICL options are unchanged; sharing is added only on request."""

    config = Exp16Config()
    shared = replace(config, tabicl_share_checkpoint=True)

    assert "share_checkpoint" not in build_lp_probability_model_kwargs(config, "tabicl_v2")
    assert "share_checkpoint" not in (
        build_stage1_predictor_kwargs("tabicl_v2", config)["predictor_kwargs"]
    )
    assert build_lp_probability_model_kwargs(shared, "tabicl_v2")["share_checkpoint"] is True
    assert (
        build_stage1_predictor_kwargs("tabicl_v2", shared)["predictor_kwargs"]["share_checkpoint"]
        is True
    )
