"""
Tests for the naturalness grounding workflow.
"""

import numpy as np
import pandas as pd
import pytest

import causal_grounding.predictors as predictors_module
from causal_grounding.naturalness_grounder import (
    EmpiricalSitePMFModels,
    HierarchicalDirichletSitePMFModels,
    NATURALNESS_STAGE1_BACKENDS,
    NaturalnessGPKernel,
    NaturalnessGPSurface,
    NaturalnessGrid,
    NaturalnessGrounder,
    NaturalnessTolerance,
    PredictorBackedSitePMFModels,
    TabICLv2SitePMFModels,
    XGBoostSitePMFModels,
)


def _repeat_rows(rows):
    return pd.DataFrame(rows)


def test_xgboost_stage1_backend_is_registered():
    assert NATURALNESS_STAGE1_BACKENDS["xgboost"] is XGBoostSitePMFModels


def test_tabicl_v2_stage1_backend_is_registered():
    """TabICLv2 resolves to its predictor-backed Stage-1 adapter."""
    assert NATURALNESS_STAGE1_BACKENDS["tabicl_v2"] is TabICLv2SitePMFModels


def test_hierarchical_dirichlet_stage1_backend_is_registered():
    """Hierarchical Dirichlet resolves to the Stage-1 adapter."""
    assert (
        NATURALNESS_STAGE1_BACKENDS["hierarchical_dirichlet"]
        is HierarchicalDirichletSitePMFModels
    )


def test_predictor_backed_backends_have_clear_inheritance():
    """Model-backed naturalness adapters do not inherit through XGBoost."""
    assert issubclass(XGBoostSitePMFModels, PredictorBackedSitePMFModels)
    assert issubclass(HierarchicalDirichletSitePMFModels, PredictorBackedSitePMFModels)
    assert issubclass(TabICLv2SitePMFModels, PredictorBackedSitePMFModels)
    assert not issubclass(HierarchicalDirichletSitePMFModels, XGBoostSitePMFModels)


def test_hierarchical_dirichlet_stage1_backend_queries():
    """Hierarchical Dirichlet Stage-1 backend exposes bounded PMF queries."""
    rows = []
    for z, n_treated in [(0, 4), (1, 16)]:
        for i in range(20):
            x = 1 if i < n_treated else 0
            y = 1 if (x and i % 3 != 0) or (not x and z == 1 and i % 5 == 0) else 0
            rows.append({"Z": z, "X": x, "Y": y, "F": "idle"})

    for z in [0, 1]:
        for x in [0, 1]:
            for i in range(12):
                y = 1 if (x and i < 7 + z) or (not x and i < 3 + z) else 0
                rows.append({"Z": z, "X": x, "Y": y, "F": "on"})

    model = HierarchicalDirichletSitePMFModels(
        site_id="site",
        data=pd.DataFrame(rows),
        covariates=["Z"],
        treatment="X",
        outcome="Y",
        predictor_kwargs={
            "prior_strength": 2.0,
            "base_prior_strength": 2.0,
        },
    )

    queries = [
        model.p_idle_x_given_z(1, (0,)),
        model.p_idle_y_given_xz(1, 1, (1,)),
        model.p_star_y_given_z(1, 1, (0,)),
        model.p_star_y_given_xz(0, 1, 1, (1,)),
    ]
    assert all(0.0 <= value <= 1.0 for value in queries)


def test_xgboost_stage1_backend_queries_when_available():
    pytest.importorskip("xgboost")

    rows = []
    for z, n_treated in [(0, 8), (1, 32)]:
        for i in range(40):
            x = 1 if i < n_treated else 0
            y = 1 if (x and i % 5 != 0) or (not x and z == 1 and i % 4 == 0) else 0
            rows.append({"Z": z, "X": x, "Y": y, "F": "idle"})

    for z in [0, 1]:
        for x in [0, 1]:
            for i in range(30):
                y = 1 if (x and i < 20 + 3 * z) or (not x and i < 8 + 2 * z) else 0
                rows.append({"Z": z, "X": x, "Y": y, "F": "on"})

    data = pd.DataFrame(rows)
    model = XGBoostSitePMFModels(
        site_id="site",
        data=data,
        covariates=["Z"],
        treatment="X",
        outcome="Y",
        predictor_kwargs={"use_grid_search": False, "random_state": 0},
    )

    queries = [
        model.p_idle_x_given_z(1, (0,)),
        model.p_idle_y_given_xz(1, 1, (1,)),
        model.p_star_y_given_z(1, 1, (0,)),
        model.p_star_y_given_xz(0, 1, 1, (1,)),
    ]
    assert all(0.0 <= value <= 1.0 for value in queries)


def test_tabicl_v2_stage1_backend_queries_with_classifier_double(
    monkeypatch: pytest.MonkeyPatch,
):
    """TabICLv2 adapter exposes bounded Stage-1 PMF queries."""
    import torch

    cuda_cache_clears = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: cuda_cache_clears.append(True),
    )

    class FakeClassifier:
        """Deterministic TabICL-compatible binary classifier."""

        def __init__(self, **kwargs):
            """Retain constructor settings for assertions."""
            self.kwargs = kwargs
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            """Store target prevalence."""
            self.target_mean_ = float(np.mean(y))
            return self

        def predict_proba(self, X):
            """Return a bounded feature-dependent probability."""
            features = np.asarray(X, dtype=float)
            p1 = np.clip(
                0.2 + 0.5 * self.target_mean_ + 0.05 * features.mean(axis=1),
                0.01,
                0.99,
            )
            return np.column_stack([1.0 - p1, p1])

    monkeypatch.setattr(predictors_module, "_TABICL_AVAILABLE", True)
    monkeypatch.setattr(predictors_module, "TabICLClassifier", FakeClassifier)

    rows = []
    observed_z = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]
    for regime in ["idle", "on"]:
        for z1, z2, z3 in observed_z:
            for x in [0, 1]:
                for index in range(8):
                    rows.append({
                        "Z1": z1,
                        "Z2": z2,
                        "Z3": z3,
                        "X": x,
                        "Y": int(index < 2 + 3 * x + z1),
                        "F": regime,
                    })

    model = TabICLv2SitePMFModels(
        site_id="site",
        data=pd.DataFrame(rows),
        covariates=["Z1", "Z2", "Z3"],
        treatment="X",
        outcome="Y",
        predictor_kwargs={"n_estimators": 2, "random_state": 0},
    )

    unobserved_joint_stratum = (1, 0, 1)
    assert unobserved_joint_stratum not in observed_z
    queries = [
        model.p_idle_x_given_z(1, unobserved_joint_stratum),
        model.p_idle_y_given_xz(1, 1, unobserved_joint_stratum),
        model.p_star_y_given_z(1, 1, unobserved_joint_stratum),
        model.p_star_y_given_xz(0, 1, 1, unobserved_joint_stratum),
    ]
    assert len(model._p_idle_x1_by_z) == 8
    assert model.predictor_kwargs["kv_cache"] == "repr"
    assert model.idle_predictor_ is None
    assert model.on_predictor_ is None
    assert model.predictors_released_
    assert cuda_cache_clears == [True]
    assert all(0.0 <= value <= 1.0 for value in queries)


def test_stage1_backend_is_pluggable():
    rows = []
    for z in [0, 1]:
        for x in [0, 1]:
            rows.append({"Z": z, "X": x, "Y": x, "F": "idle"})
            rows.append({"Z": z, "X": x, "Y": x, "F": "on"})

    class MarkerBackend(EmpiricalSitePMFModels):
        def __init__(self, marker=False, **kwargs):
            super().__init__(**kwargs)
            self.marker = marker

    grid = NaturalnessGrid(
        epsilons=[1.0],
        taus=[0.0],
        covariates=["Z"],
        min_cell_count=1,
    )
    grounder = NaturalnessGrounder(
        grid=grid,
        fit_gp=False,
        stage1_backend=MarkerBackend,
        stage1_kwargs={"marker": True},
    )

    models = grounder.fit_stage1_pmfs(
        {"site": pd.DataFrame(rows)},
        treatment="X",
        outcome="Y",
    )

    assert isinstance(models["site"], MarkerBackend)
    assert isinstance(models["site"], EmpiricalSitePMFModels)
    assert models["site"].marker is True
    assert grounder.get_diagnostics()["stage1_backend"] == "MarkerBackend"


def test_reporting_covariate_conditions_without_becoming_candidate():
    """Stage 2 tests only declared candidates while retaining age in Z-minus."""

    rows = []
    for regime in ("idle", "on"):
        for age in (20, 21):
            for candidate, n_treated in ((0, 2), (1, 6)):
                for index in range(8):
                    treatment = int(index < n_treated)
                    rows.append({
                        "age": age,
                        "candidate": candidate,
                        "X": treatment,
                        "Y": treatment,
                        "F": regime,
                    })
    grid = NaturalnessGrid(
        epsilons=[1.0],
        taus=[0.0],
        covariates=["age", "candidate"],
        candidate_covariates=["candidate"],
        min_cell_count=1,
    )
    grounder = NaturalnessGrounder(grid=grid, fit_gp=False)
    observations = grounder.fit(
        {"site": pd.DataFrame(rows)},
        treatment="X",
        outcome="Y",
    ).observations_

    assert set(observations["covariate"]) == {"candidate"}
    assert set(observations["covariate_index"]) == {1}
    assert set(observations["n_z_minus_evaluated"]) == {2}


def test_off_treatment_formula_uses_subtraction():
    """Counterfactual conditioning should use the corrected subtraction formula."""
    rows = []

    # Idle data at z=0:
    # P(Y=1 | X=0,z)=0.2, P(X=0|z)=0.5, P(X=1|z)=0.5
    for i in range(10):
        rows.append({"Z": 0, "X": 0, "Y": 1 if i < 2 else 0, "F": "idle"})
    for i in range(10):
        rows.append({"Z": 0, "X": 1, "Y": 0, "F": "idle"})

    # On/RCT data:
    # P(Y(0)=1 | z)=P(Y=1 | X=0,z,F=on)=0.4
    for i in range(10):
        rows.append({"Z": 0, "X": 0, "Y": 1 if i < 4 else 0, "F": "on"})
    for i in range(10):
        rows.append({"Z": 0, "X": 1, "Y": 0, "F": "on"})

    model = EmpiricalSitePMFModels(
        site_id="site",
        data=_repeat_rows(rows),
        covariates=["Z"],
        treatment="X",
        outcome="Y",
        alpha=0.0,
    )

    estimate = model.p_star_y_given_xz(star=0, y=1, x=1, z=(0,))

    # (0.4 - 0.2 * 0.5) / 0.5 = 0.6
    assert np.isclose(estimate, 0.6)


def test_d_strength_uses_distinct_z_pairs():
    """D_j(tau) should compare distinct z_j values only."""
    rows = []
    # z=0 has P(X=1|z)=0.2; z=1 has P(X=1|z)=0.8.
    for z, n_treated in [(0, 2), (1, 8)]:
        for i in range(10):
            x = 1 if i < n_treated else 0
            rows.append({"Z": z, "X": x, "Y": x, "F": "idle"})
            rows.append({"Z": z, "X": x, "Y": x, "F": "on"})

    training_data = {"site": pd.DataFrame(rows)}
    grid = NaturalnessGrid(
        epsilons=[1.0],
        taus=[0.5],
        covariates=["Z"],
        min_cell_count=1,
    )

    grounder = NaturalnessGrounder(grid=grid, alpha=0.0, fit_gp=False)
    observations = grounder.fit(
        training_data,
        treatment="X",
        outcome="Y",
        fit_gp=False,
    ).observations_

    assert len(observations) == 1
    assert np.isclose(observations.loc[0, "d_strength"], 0.6)
    assert bool(observations.loc[0, "trigger_fired"])


def test_grounder_rows_and_predictions():
    rows = []
    for site in ["a", "b"]:
        for z1 in [0, 1]:
            for z2 in [0, 1]:
                for i in range(12):
                    x = 1 if (i + z1 + (site == "b")) % 3 else 0
                    y = 1 if x or z2 else 0
                    rows.append({"Z1": z1, "Z2": z2, "X": x, "Y": y, "F": "idle", "site": site})
                    rows.append({"Z1": z1, "Z2": z2, "X": x, "Y": y, "F": "on", "site": site})

    data = pd.DataFrame(rows)
    training_data = {site: df.copy() for site, df in data.groupby("site")}
    grid = NaturalnessGrid(
        epsilons=[0.1, 1.0],
        taus=[0.0, 0.2],
        covariates=["Z1", "Z2"],
        min_cell_count=1,
    )

    grounder = NaturalnessGrounder(grid=grid, alpha=0.5, random_seed=42, fit_gp=False)
    grounder.fit(training_data, treatment="X", outcome="Y", fit_gp=False)

    observations = grounder.observations_
    assert len(observations) == len(training_data) * len(grid.covariates) * len(grid.epsilons) * len(grid.taus)
    assert {"c_hat", "d_hat", "i_deviation", "d_strength"}.issubset(observations.columns)

    tolerance = grounder.predict_tolerance("Z1", epsilon=1.0, tau=0.0)
    assert np.isnan(tolerance.c) or 0.0 <= tolerance.c <= 0.5


def test_predict_tolerance_returns_note_scale_values():
    rows = []
    for z, n_treated in [(0, 2), (1, 8)]:
        for i in range(10):
            x = 1 if i < n_treated else 0
            rows.append({"Z": z, "X": x, "Y": x, "F": "idle"})
            rows.append({"Z": z, "X": x, "Y": x, "F": "on"})

    grid = NaturalnessGrid(
        epsilons=[1.0],
        taus=[0.5],
        covariates=["Z"],
        min_cell_count=1,
    )
    grounder = NaturalnessGrounder(grid=grid, alpha=0.0, fit_gp=False)
    grounder.fit({"site": pd.DataFrame(rows)}, treatment="X", outcome="Y", fit_gp=False)

    tolerance = grounder.predict_tolerance("Z", epsilon=1.0, tau=0.5)
    frame = grounder.predict_grid()

    assert isinstance(tolerance, NaturalnessTolerance)
    assert 0.0 <= tolerance.c <= 0.5
    assert 0.0 <= tolerance.d <= 0.5
    assert {"c_mean", "d_mean", "c_source", "d_source", "prediction_scale"}.issubset(
        frame.columns
    )


def test_naturalness_gp_kernel_uses_categorical_covariate_penalty():
    kernel = NaturalnessGPKernel(
        epsilon_lambda=0.01,
        tau_lambda=0.01,
        covariate_lambda=0.5,
    )
    features = np.array([
        [0.10, 0.20, 0],
        [0.10, 0.20, 1],
        [0.20, 0.20, 0],
    ])

    covariance, gradient = kernel(features, eval_gradient=True)

    assert np.isclose(covariance[0, 0], 1.0)
    assert np.isclose(covariance[0, 1], np.exp(-1.0 / 0.5))
    assert np.isclose(covariance[0, 2], np.exp(-0.01 / 0.01))
    assert gradient.shape == (3, 3, 3)


def test_stage3_gp_surface_exposes_logit_scale_and_diagnostics():
    pytest.importorskip("sklearn")

    observations = pd.DataFrame({
        "trigger_fired": [True] * 8,
        "covariate": ["Z1", "Z1", "Z1", "Z1", "Z2", "Z2", "Z2", "Z2"],
        "covariate_index": [0, 0, 0, 0, 1, 1, 1, 1],
        "epsilon": [0.05, 0.10, 0.20, 0.30, 0.05, 0.10, 0.20, 0.30],
        "tau": [0.05, 0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.20],
        "c_hat": [0.08, 0.10, 0.13, 0.16, 0.20, 0.22, 0.26, 0.30],
    })

    surface = NaturalnessGPSurface(
        "c_hat",
        min_observations=4,
        random_seed=0,
    ).fit(observations)

    assert surface.status == "fitted"
    prob_mean, std_logit = surface.predict(0.10, 0.10, 0)
    logit_mean, logit_std = surface.predict(0.10, 0.10, 0, scale="logit")
    upper_prob, _ = surface.predict(0.10, 0.10, 0, quantile=0.95)
    diagnostics = surface.diagnostics()

    assert 0.0 <= prob_mean <= 1.0
    assert np.isclose(prob_mean, 1.0 / (1.0 + np.exp(-logit_mean)))
    assert std_logit == logit_std
    assert upper_prob >= prob_mean
    assert diagnostics["target_scale"] == "logit"
    assert diagnostics["posterior_prediction_target"] == "latent_surface"
    assert diagnostics["log_marginal_likelihood"] is not None
    assert "k1__k2__covariate_lambda" in diagnostics["learned_hyperparameters"]

    features = pd.DataFrame({
        "epsilon": [0.10, 0.20],
        "tau": [0.10, 0.15],
        "covariate_index": [0, 1],
    })
    samples = surface.sample(features, n_samples=3, random_seed=123)
    repeat_samples = surface.sample(features, n_samples=3, random_seed=123)
    logit_samples = surface.sample(features, n_samples=3, scale="logit", random_seed=123)

    assert samples.shape == (3, 2)
    assert np.allclose(samples, repeat_samples)
    assert np.allclose(samples, 1.0 / (1.0 + np.exp(-logit_samples)))
    assert np.all((samples > 0.0) & (samples < 1.0))

    feature_array = features.to_numpy(dtype=float)
    _, latent_covariance = surface._posterior(
        feature_array,
        include_observation_noise=False,
    )
    _, observation_covariance = surface._posterior(
        feature_array,
        include_observation_noise=True,
    )
    assert np.all(
        np.diag(observation_covariance) > np.diag(latent_covariance)
    )


def test_stage3_gp_surface_can_use_observed_stat_features():
    """Observed-stat mode fits on I/D statistics instead of thresholds."""
    pytest.importorskip("sklearn")

    observations = pd.DataFrame({
        "site_id": ["a", "a", "b", "b"],
        "trigger_fired": [True, True, True, True],
        "covariate": ["Z", "Z", "Z", "Z"],
        "covariate_index": [0, 0, 0, 0],
        "epsilon": [0.50, 0.50, 0.50, 0.50],
        "tau": [0.01, 0.05, 0.01, 0.05],
        "i_deviation": [0.20, 0.20, 0.35, 0.35],
        "d_strength": [0.08, 0.08, 0.16, 0.16],
        "c_hat": [0.10, 0.10, 0.22, 0.22],
    })

    surface = NaturalnessGPSurface(
        "c_hat",
        min_observations=2,
        random_seed=0,
        feature_mode="observed_stats",
    ).fit(observations)
    diagnostics = surface.diagnostics()

    assert surface.status == "fitted"
    assert surface.feature_columns == ("i_deviation", "d_strength", "covariate_index")
    assert len(surface.training_data_) == 2
    assert diagnostics["feature_mode"] == "observed_stats"
    assert diagnostics["n_raw_training_observations"] == 4
    assert np.allclose(
        surface.training_features_,
        np.array([[0.20, 0.08, 0.0], [0.35, 0.16, 0.0]]),
    )

    observed_mean, _ = surface.predict(
        0.50,
        0.01,
        0,
        i_deviation=0.20,
        d_strength=0.08,
    )
    with pytest.raises(ValueError, match="i_deviation and d_strength"):
        surface.predict(0.50, 0.01, 0)

    assert 0.0 < observed_mean < 1.0


def test_grounder_samples_tolerances_in_long_format():
    pytest.importorskip("sklearn")

    observations = pd.DataFrame({
        "trigger_fired": [True] * 8,
        "covariate": ["Z1", "Z1", "Z1", "Z1", "Z2", "Z2", "Z2", "Z2"],
        "covariate_index": [0, 0, 0, 0, 1, 1, 1, 1],
        "epsilon": [0.05, 0.10, 0.20, 0.30, 0.05, 0.10, 0.20, 0.30],
        "tau": [0.05, 0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.20],
        "c_hat": [0.08, 0.10, 0.13, 0.16, 0.20, 0.22, 0.26, 0.30],
        "d_hat": [0.04, 0.06, 0.08, 0.10, 0.09, 0.11, 0.14, 0.18],
    })
    grid = NaturalnessGrid(
        epsilons=[0.05, 0.10, 0.20, 0.30],
        taus=[0.05, 0.10, 0.15, 0.20],
        covariates=["Z1", "Z2"],
        min_cell_count=1,
    )
    grounder = NaturalnessGrounder(grid=grid, random_seed=0, fit_gp=False)
    grounder.c_surface_ = NaturalnessGPSurface(
        "c_hat",
        min_observations=4,
        random_seed=0,
    ).fit(observations)
    grounder.d_surface_ = NaturalnessGPSurface(
        "d_hat",
        min_observations=4,
        random_seed=0,
    ).fit(observations)

    rows = pd.DataFrame({
        "covariate": ["Z1", "Z2"],
        "epsilon": [0.10, 0.20],
        "tau": [0.10, 0.15],
        "c": [0.99, 0.99],
        "d": [0.99, 0.99],
        "two_c": [1.98, 1.98],
        "two_d": [1.98, 1.98],
    })
    samples = grounder.sample_tolerances(rows, n_samples=4, random_seed=123)
    repeat_samples = grounder.sample_tolerances(rows, n_samples=4, random_seed=123)

    assert len(samples) == 8
    assert np.allclose(samples["c"], repeat_samples["c"])
    assert np.allclose(samples["d"], repeat_samples["d"])
    assert set(samples["sample_id"]) == {0, 1, 2, 3}
    assert {"c", "d", "two_c", "two_d", "c_source", "d_source"}.issubset(
        samples.columns
    )
    assert samples.columns.is_unique
    assert np.all(samples["c"].between(0.0, 0.5))
    assert np.all(samples["d"].between(0.0, 0.5))
    assert np.allclose(samples["two_c"], 2.0 * samples["c"])
    assert np.allclose(samples["two_d"], 2.0 * samples["d"])


def test_grounder_constructs_simultaneous_upper_tolerance_vector():
    """One shared maximum statistic protects every requested c/d coordinate."""
    pytest.importorskip("sklearn")

    observations = pd.DataFrame({
        "trigger_fired": [True] * 8,
        "covariate": ["Z1", "Z1", "Z1", "Z1", "Z2", "Z2", "Z2", "Z2"],
        "covariate_index": [0, 0, 0, 0, 1, 1, 1, 1],
        "epsilon": [0.05, 0.10, 0.20, 0.30, 0.05, 0.10, 0.20, 0.30],
        "tau": [0.05, 0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.20],
        "c_hat": [0.08, 0.10, 0.13, 0.16, 0.20, 0.22, 0.26, 0.30],
        "d_hat": [0.04, 0.06, 0.08, 0.10, 0.09, 0.11, 0.14, 0.18],
    })
    grid = NaturalnessGrid(
        epsilons=[0.05, 0.10, 0.20, 0.30],
        taus=[0.05, 0.10, 0.15, 0.20],
        covariates=["Z1", "Z2"],
        min_cell_count=1,
    )
    grounder = NaturalnessGrounder(grid=grid, random_seed=0, fit_gp=False)
    grounder.observations_ = observations
    grounder.c_surface_ = NaturalnessGPSurface(
        "c_hat",
        min_observations=4,
        random_seed=0,
    ).fit(observations)
    grounder.d_surface_ = NaturalnessGPSurface(
        "d_hat",
        min_observations=4,
        random_seed=0,
    ).fit(observations)
    requested = pd.DataFrame({
        "covariate": ["Z1", "Z2"],
        "epsilon": [0.10, 0.20],
        "tau": [0.10, 0.15],
    })

    upper = grounder.predict_simultaneous_upper_tolerances(
        requested,
        confidence_level=0.95,
        n_samples=512,
        random_seed=123,
    )
    repeated = grounder.predict_simultaneous_upper_tolerances(
        requested,
        confidence_level=0.95,
        n_samples=512,
        random_seed=123,
    )
    means = [
        grounder.predict_tolerance(
            row.covariate,
            epsilon=float(row.epsilon),
            tau=float(row.tau),
        )
        for row in requested.itertuples(index=False)
    ]

    assert len(upper) == 2
    assert [item.c for item in upper] == [item.c for item in repeated]
    assert [item.d for item in upper] == [item.d for item in repeated]
    assert all(item.c >= mean.c for item, mean in zip(upper, means))
    assert all(item.d >= mean.d for item, mean in zip(upper, means))
    assert {item.upper_family_size for item in upper} == {4}
    assert {item.upper_n_samples for item in upper} == {512}
    assert all(item.upper_critical_value > 0.0 for item in upper)
    assert all(
        item.c_source == "gp_simultaneous_upper_0.95"
        for item in upper
    )


def test_grounder_samples_tolerances_with_observed_stat_features():
    """Posterior tolerance sampling uses target I/D columns in observed-stat mode."""
    pytest.importorskip("sklearn")

    observations = pd.DataFrame({
        "site_id": ["a", "a", "b", "b"],
        "trigger_fired": [True, True, True, True],
        "covariate": ["Z", "Z", "Z", "Z"],
        "covariate_index": [0, 0, 0, 0],
        "epsilon": [0.50, 0.50, 0.50, 0.50],
        "tau": [0.01, 0.05, 0.01, 0.05],
        "i_deviation": [0.20, 0.20, 0.35, 0.35],
        "d_strength": [0.08, 0.08, 0.16, 0.16],
        "c_hat": [0.10, 0.10, 0.22, 0.22],
        "d_hat": [0.05, 0.05, 0.12, 0.12],
    })
    grid = NaturalnessGrid(
        epsilons=[0.50],
        taus=[0.01, 0.05],
        covariates=["Z"],
        min_cell_count=1,
    )
    grounder = NaturalnessGrounder(
        grid=grid,
        random_seed=0,
        fit_gp=False,
        gp_feature_mode="observed_stats",
    )
    grounder.c_surface_ = NaturalnessGPSurface(
        "c_hat",
        min_observations=2,
        random_seed=0,
        feature_mode="observed_stats",
    ).fit(observations)
    grounder.d_surface_ = NaturalnessGPSurface(
        "d_hat",
        min_observations=2,
        random_seed=0,
        feature_mode="observed_stats",
    ).fit(observations)

    rows = pd.DataFrame({
        "covariate": ["Z"],
        "epsilon": [0.50],
        "tau": [0.01],
        "i_deviation": [0.20],
        "d_strength": [0.08],
    })
    samples = grounder.sample_tolerances(rows, n_samples=3, random_seed=123)

    assert len(samples) == 3
    assert set(samples["sample_id"]) == {0, 1, 2}
    assert np.all(samples["c"].between(0.0, 0.5))
    assert np.all(samples["d"].between(0.0, 0.5))
    with pytest.raises(ValueError, match="i_deviation"):
        grounder.sample_tolerances(rows.drop(columns=["i_deviation"]), n_samples=1)


def test_predict_tolerance_can_use_upper_gp_quantile():
    pytest.importorskip("sklearn")

    rows = []
    for site in ["a", "b"]:
        for z, n_treated in [(0, 6), (1, 24)]:
            for i in range(30):
                x = 1 if i < n_treated else 0
                y = x
                rows.append({"Z": z, "X": x, "Y": y, "F": "idle", "site": site})
                rows.append({"Z": z, "X": x, "Y": y, "F": "on", "site": site})

    training_data = {
        site: df.copy()
        for site, df in pd.DataFrame(rows).groupby("site")
    }
    grid = NaturalnessGrid(
        epsilons=[0.5, 1.0],
        taus=[0.0, 0.1],
        covariates=["Z"],
        min_cell_count=1,
    )
    grounder = NaturalnessGrounder(grid=grid, alpha=0.5, random_seed=0, fit_gp=True)
    grounder.fit(training_data, treatment="X", outcome="Y", fit_gp=True)

    mean_tolerance = grounder.predict_tolerance("Z", epsilon=1.0, tau=0.0)
    upper_tolerance = grounder.predict_tolerance(
        "Z",
        epsilon=1.0,
        tau=0.0,
        quantile=0.95,
    )

    assert upper_tolerance.c >= mean_tolerance.c
    assert upper_tolerance.d >= mean_tolerance.d
    assert upper_tolerance.c_source.startswith("gp_quantile")
    assert upper_tolerance.d_source.startswith("gp_quantile")


def _candidate_state_training_data(seed=0, n_sites=3, n_rows=160):
    """Build site data with two real covariates and two pure-noise columns."""
    rng = np.random.default_rng(seed)
    data = {}
    for site in range(n_sites):
        frames = []
        for regime in ("idle", "on"):
            a = rng.binomial(1, 0.5, n_rows)
            b = rng.binomial(1, 0.5, n_rows)
            x = (
                rng.binomial(1, 0.2 + 0.6 * a)
                if regime == "idle"
                else rng.binomial(1, 0.5, n_rows)
            )
            frames.append(pd.DataFrame({
                "a": a,
                "b": b,
                "n1": rng.binomial(1, 0.5, n_rows),
                "n2": rng.binomial(1, 0.5, n_rows),
                "X": x,
                "Y": rng.binomial(1, 0.2 + 0.3 * x + 0.2 * b),
                "F": regime,
            }))
        data[f"site_{site}"] = pd.concat(frames, ignore_index=True)
    return data


def test_grid_candidate_states_default_to_the_full_state():
    """Without conditioning covariates every candidate shares the full Z."""
    grid = NaturalnessGrid(epsilons=[0.1], taus=[0.1], covariates=["a", "b", "c"])

    assert not grid.uses_candidate_states
    assert grid.candidate_state(1) == (["a", "b", "c"], 1)
    assert grid.candidate_state_keys == [("a", "b", "c")]


def test_grid_conditioning_covariates_define_candidate_states():
    """Each candidate is evaluated in the conditioning set plus itself."""
    grid = NaturalnessGrid(
        epsilons=[0.1],
        taus=[0.1],
        covariates=["a", "b", "n1", "n2"],
        candidate_covariates=["b", "n1", "n2"],
        conditioning_covariates=["a", "b"],
    )

    assert grid.uses_candidate_states
    assert grid.candidate_state(1) == (["a", "b"], 1)
    assert grid.candidate_state(2) == (["a", "b", "n1"], 2)
    assert grid.candidate_state(3) == (["a", "b", "n2"], 2)
    assert grid.candidate_state_keys == [("a", "b"), ("a", "b", "n1"), ("a", "b", "n2")]
    with pytest.raises(IndexError):
        grid.candidate_state(4)
    with pytest.raises(ValueError, match="subset of covariates"):
        NaturalnessGrid(epsilons=[0.1], taus=[0.1], covariates=["a"], conditioning_covariates=["z"])
    with pytest.raises(ValueError, match="duplicates"):
        NaturalnessGrid(
            epsilons=[0.1], taus=[0.1], covariates=["a", "b"], conditioning_covariates=["a", "a"]
        )


def test_candidate_states_match_single_state_grounders():
    """Stage 2 in candidate states equals legacy grounders fitted per state.

    Real candidates keep ``Z_-j`` inside the conditioning set, so they match a
    grounder over ``[a, b]``. An extra candidate is conditioned on ``[a, b]``
    only, so it matches a grounder over ``[a, b, n1]`` that tests ``n1`` alone,
    whatever other extra candidates exist.
    """
    data = _candidate_state_training_data()
    grid_kwargs = {"epsilons": [0.05, 0.5], "taus": [0.01, 0.1], "min_cell_count": 2}
    joint = NaturalnessGrounder(
        NaturalnessGrid(
            covariates=["a", "b", "n1", "n2"],
            conditioning_covariates=["a", "b"],
            **grid_kwargs,
        ),
        fit_gp=False,
        stage1_backend="empirical",
    ).fit(data, treatment="X", outcome="Y")
    base = NaturalnessGrounder(
        NaturalnessGrid(covariates=["a", "b"], **grid_kwargs),
        fit_gp=False,
        stage1_backend="empirical",
    ).fit(data, treatment="X", outcome="Y")
    single = NaturalnessGrounder(
        NaturalnessGrid(covariates=["a", "b", "n1"], candidate_covariates=["n1"], **grid_kwargs),
        fit_gp=False,
        stage1_backend="empirical",
    ).fit(data, treatment="X", outcome="Y")

    columns = [
        "site_id", "epsilon", "tau", "i_deviation", "d_strength",
        "trigger_fired", "c_hat", "d_hat", "n_z_minus_evaluated",
    ]
    observed = joint.observations_
    for covariate in ("a", "b"):
        pd.testing.assert_frame_equal(
            observed.loc[observed["covariate"].eq(covariate), columns].reset_index(drop=True),
            base.observations_.loc[
                base.observations_["covariate"].eq(covariate), columns
            ].reset_index(drop=True),
        )
    pd.testing.assert_frame_equal(
        observed.loc[observed["covariate"].eq("n1"), columns].reset_index(drop=True),
        single.observations_[columns].reset_index(drop=True),
    )
    assert observed.loc[observed["covariate"].eq("n1"), "covariate_index"].unique().tolist() == [2]
    assert sorted(joint.site_state_models_["site_0"]) == [
        ("a", "b"), ("a", "b", "n1"), ("a", "b", "n2"),
    ]
    assert joint.get_diagnostics()["n_sites"] == len(data)


def _multi_site_observations():
    """Return triggered Stage-2 rows of three sites at four grid points per covariate."""
    rows = []
    for site, offset in [("a", 0.00), ("b", 0.03), ("c", 0.06)]:
        for covariate_index, covariate in enumerate(["Z1", "Z2"]):
            for epsilon, tau in [(0.05, 0.10), (0.10, 0.10), (0.20, 0.05), (0.30, 0.05)]:
                value = 0.05 + 0.1 * covariate_index + epsilon / 4 + offset
                rows.append({
                    "site_id": site, "covariate": covariate, "covariate_index": covariate_index,
                    "epsilon": epsilon, "tau": tau, "trigger_fired": True,
                    "c_hat": value, "d_hat": value / 2,
                })
    return pd.DataFrame(rows)


def test_envelope_surface_fits_the_maximum_over_triggering_sites():
    """The envelope keeps one row per grid point: the largest triggering-site tolerance."""
    pytest.importorskip("sklearn")
    observations = _multi_site_observations()
    silent = observations.site_id.eq("c") & observations.epsilon.eq(0.30)
    observations.loc[silent, "trigger_fired"] = False

    surface = NaturalnessGPSurface(
        "c_hat", min_observations=4, random_seed=0, training_target="envelope",
    ).fit(observations)

    fired = observations[observations.trigger_fired]
    keys = ["covariate_index", "epsilon", "tau"]
    expected = fired.groupby(keys)["c_hat"].max().sort_index()
    training = surface.training_data_.set_index(keys)
    # The GP casts its feature columns to float, so compare without index dtypes.
    pd.testing.assert_frame_equal(
        training["c_hat"].sort_index().reset_index(), expected.reset_index(), check_dtype=False,
    )
    assert surface.n_raw_training_observations_ == len(fired)
    sites = training["n_triggering_sites"]
    assert sites.xs(0.30, level="epsilon").eq(2).all()
    assert sites.drop(0.30, level="epsilon").eq(3).all()
    assert surface.diagnostics()["gp_training_target"] == "envelope"
    assert surface.diagnostics()["n_training_observations"] == len(expected)


def test_tilde_scale_models_the_logit_of_twice_c_and_returns_note_scale():
    """``tilde`` fits logit(2c); predictions and draws come back as note-scale c."""
    pytest.importorskip("sklearn")
    observations = _multi_site_observations()
    surface = NaturalnessGPSurface(
        "c_hat", min_observations=4, random_seed=0, target_scale="tilde",
    ).fit(observations)

    doubled = 2.0 * surface.training_data_["c_hat"].to_numpy(float)
    np.testing.assert_allclose(surface.training_target_logit_, np.log(doubled / (1.0 - doubled)))
    prob, _ = surface.predict(0.10, 0.10, 0)
    logit, _ = surface.predict(0.10, 0.10, 0, scale="logit")
    assert np.isclose(prob, 0.5 / (1.0 + np.exp(-logit)))
    assert 0.0 < prob < 0.5

    features = pd.DataFrame({"epsilon": [0.10, 0.20], "tau": [0.10, 0.05], "covariate_index": [0, 1]})
    draws = surface.sample(features, n_samples=5, random_seed=1)
    logit_draws = surface.sample(features, n_samples=5, scale="logit", random_seed=1)
    np.testing.assert_allclose(draws, 0.5 / (1.0 + np.exp(-logit_draws)))
    assert surface.diagnostics()["gp_target_scale"] == "tilde"
    assert surface.diagnostics()["target_scale"] == "logit"


def test_grounder_passes_stage3_settings_and_matches_its_empirical_fallback():
    """Grounder settings reach both surfaces, and the no-GP fallback follows them."""
    pytest.importorskip("sklearn")
    grid = NaturalnessGrid(
        epsilons=[0.05, 0.10, 0.20, 0.30], taus=[0.05, 0.10], covariates=["Z1", "Z2"], min_cell_count=1,
    )
    observations = _multi_site_observations()
    at_point = observations[observations.covariate.eq("Z1") & observations.epsilon.eq(0.10)]

    envelope = NaturalnessGrounder(
        grid=grid, random_seed=0, gp_training_target="envelope", gp_target_scale="tilde",
    )
    envelope.observations_ = observations
    envelope.fit_stage3_gp(min_observations=4)
    for surface in (envelope.c_surface_, envelope.d_surface_):
        assert (surface.training_target, surface.target_scale) == ("envelope", "tilde")
    value, std, source = envelope._predict_surface_details("c_hat", None, "Z1", 0.10, 0.10)
    assert (source, std) == ("empirical_exact", None)
    assert np.isclose(value, at_point.c_hat.max())
    logit, _, _ = envelope._predict_surface_details("c_hat", None, "Z1", 0.10, 0.10, scale="logit")
    assert np.isclose(logit, np.log(2 * value / (1 - 2 * value)))

    per_site = NaturalnessGrounder(grid=grid, random_seed=0)
    per_site.observations_ = observations
    value, _, _ = per_site._predict_surface_details("c_hat", None, "Z1", 0.10, 0.10)
    assert np.isclose(value, at_point.c_hat.mean())


def test_stage3_settings_are_validated():
    """Unknown settings, and an envelope over observed statistics, are rejected."""
    grid = NaturalnessGrid(epsilons=[0.1], taus=[0.1], covariates=["Z"], min_cell_count=1)
    with pytest.raises(ValueError, match="gp_training_target"):
        NaturalnessGrounder(grid=grid, gp_training_target="median")
    with pytest.raises(ValueError, match="gp_target_scale"):
        NaturalnessGrounder(grid=grid, gp_target_scale="probit")
    with pytest.raises(ValueError, match="thresholds"):
        NaturalnessGPSurface("c_hat", feature_mode="observed_stats", training_target="envelope")
