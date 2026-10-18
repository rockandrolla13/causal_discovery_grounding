"""Tests for the learned-naturalness target LP solver."""

import numpy as np
import pandas as pd
import pytest

import causal_grounding.lp_solver_learned_naturalness as learned_lp
from causal_grounding.lp_solver_learned_naturalness import (
    estimate_target_observed_binary_quantities,
    solve_candidate_state_bounds_from_observed_binary_quantities,
    solve_target_bounds_from_observed_binary_quantities,
    solve_target_weighted_cate_bounds_from_observed_binary_quantities,
)
from causal_grounding.naturalness_target_policy import TargetNaturalnessChoice


def make_choice(two_c=0.0, two_d=0.0):
    """Create a deterministic target-naturalness choice for ``Z``."""
    return TargetNaturalnessChoice(
        policy="test",
        covariate="Z",
        covariate_index=0,
        epsilon=0.0,
        tau=0.0,
        i_deviation=0.0,
        d_strength=1.0,
        i_margin=0.0,
        d_margin=1.0,
        c=two_c / 2.0,
        d=two_d / 2.0,
        two_c=two_c,
        two_d=two_d,
        score=two_c + two_d,
        c_source="test",
        d_source="test",
        c_std_logit=None,
        d_std_logit=None,
        n_z_minus_evaluated=1,
        n_target_rows=200,
        is_primary_choice=True,
        reason="test choice",
    )


def add_cell(rows, z, x, n, n_y1):
    """Append one target idle cell with deterministic counts."""
    for index in range(n):
        rows.append({"F": "idle", "Z": z, "A": x, "Y": int(index < n_y1)})


def build_balancing_target_data():
    """
    Build target data whose zero-tolerance LP has the exact CATE 0.5.

    The factual outcome regressions are constant across Z:
    P(Y=1|A=0,Z)=0.2 and P(Y=1|A=1,Z)=0.7.  Treatment propensities move from
    0.2 to 0.8 across Z.  With ``2c=2d=0``, all off-treatment variables are
    forced to the same factual values, yielding mu_0=0.2 and mu_1=0.7.
    """
    rows = []
    add_cell(rows, z=0, x=0, n=80, n_y1=16)
    add_cell(rows, z=0, x=1, n=20, n_y1=14)
    add_cell(rows, z=1, x=0, n=20, n_y1=4)
    add_cell(rows, z=1, x=1, n=80, n_y1=56)
    return pd.DataFrame(rows)


def solve_target_bounds_from_data(
    data,
    choices,
    covariates,
    *,
    treatment="A",
    outcome="Y",
    min_count=5,
    return_details=True,
):
    """Estimate target probabilities, then solve the live two-step LP path."""
    quantities = estimate_target_observed_binary_quantities(
        data,
        treatment=treatment,
        outcome=outcome,
        covariates=covariates,
        min_count=min_count,
    )
    return solve_target_bounds_from_observed_binary_quantities(
        quantities,
        choices,
        return_details=return_details,
    )


def test_estimate_target_quantities_reuses_supported_cells():
    """Check target idle quantities are exact when no smoothing is requested."""
    data = build_balancing_target_data()
    quantities = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z"],
        min_count=5,
    )

    assert quantities.z_values == [(0,), (1,)]
    assert np.isclose(quantities.p_x1[(0,)], 0.2)
    assert np.isclose(quantities.p_x1[(1,)], 0.8)
    assert np.isclose(quantities.q_y0[(0,)], 0.2)
    assert np.isclose(quantities.q_y1[(1,)], 0.7)


def test_weighted_cate_query_matches_stratum_and_marginal_objectives():
    """One-hot queries reproduce cell bounds and mixtures use one joint LP."""

    quantities = estimate_target_observed_binary_quantities(
        build_balancing_target_data(),
        treatment="A",
        outcome="Y",
        covariates=["Z"],
        min_count=5,
    )
    choice = make_choice(two_c=0.0, two_d=0.0)
    stratum = solve_target_bounds_from_observed_binary_quantities(
        quantities,
        choice,
        z_values=[(0,)],
        return_details=True,
    )
    one_hot = solve_target_weighted_cate_bounds_from_observed_binary_quantities(
        quantities,
        choice,
        weighted_queries={("Z0",): {(0,): 1.0}},
        query_covariates=["group"],
        return_details=True,
    )
    marginal = solve_target_weighted_cate_bounds_from_observed_binary_quantities(
        quantities,
        choice,
        weighted_queries={("all",): {(0,): 0.5, (1,): 0.5}},
        query_covariates=["group"],
        return_details=True,
    )

    assert one_hot.loc[0, "lower"] == pytest.approx(stratum.loc[0, "lower"])
    assert one_hot.loc[0, "upper"] == pytest.approx(stratum.loc[0, "upper"])
    assert marginal.loc[0, "lower"] == pytest.approx(0.5)
    assert marginal.loc[0, "upper"] == pytest.approx(0.5)
    assert marginal.loc[0, "n_objective_strata"] == 2
    assert marginal.loc[0, "objective_weight_sum"] == pytest.approx(1.0)


def test_weighted_cate_only_mode_skips_intervention_objectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weighted CATE-only mode should retain endpoints with two LP solves."""

    quantities = estimate_target_observed_binary_quantities(
        build_balancing_target_data(),
        treatment="A",
        outcome="Y",
        covariates=["Z"],
        min_count=5,
    )
    choice = make_choice(two_c=0.0, two_d=0.0)
    weighted_queries = {("all",): {(0,): 0.5, (1,): 0.5}}
    original_solve = learned_lp._solve_linear_objective
    calls: list[np.ndarray] = []

    def counted_solve(*args, **kwargs):
        """Record each weighted objective before running the real solver."""

        calls.append(args[1])
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(learned_lp, "_solve_linear_objective", counted_solve)
    full = solve_target_weighted_cate_bounds_from_observed_binary_quantities(
        quantities,
        choice,
        weighted_queries=weighted_queries,
        query_covariates=["group"],
        return_details=True,
    )
    assert len(calls) == 6

    calls.clear()
    cate_only = (
        solve_target_weighted_cate_bounds_from_observed_binary_quantities(
            quantities,
            choice,
            weighted_queries=weighted_queries,
            query_covariates=["group"],
            return_details=True,
            include_intervention_diagnostics=False,
        )
    )

    assert len(calls) == 2
    assert cate_only.loc[0, "lower"] == pytest.approx(full.loc[0, "lower"])
    assert cate_only.loc[0, "upper"] == pytest.approx(full.loc[0, "upper"])
    assert not bool(cate_only.loc[0, "intervention_diagnostics_computed"])
    assert cate_only[
        ["mu0_lower", "mu0_upper", "mu1_lower", "mu1_upper"]
    ].isna().all().all()
    assert cate_only[
        [
            "mu0_lower_status",
            "mu0_upper_status",
            "mu1_lower_status",
            "mu1_upper_status",
        ]
    ].eq("not_requested").all().all()


def test_weighted_cate_query_rejects_invalid_probability_mass():
    """Weighted objectives require supported strata and normalized masses."""

    quantities = estimate_target_observed_binary_quantities(
        build_balancing_target_data(),
        treatment="A",
        outcome="Y",
        covariates=["Z"],
        min_count=5,
    )
    with pytest.raises(ValueError, match="sum"):
        solve_target_weighted_cate_bounds_from_observed_binary_quantities(
            quantities,
            make_choice(two_c=0.0, two_d=0.0),
            weighted_queries={("all",): {(0,): 0.4, (1,): 0.4}},
            query_covariates=["group"],
            return_details=True,
        )


def test_xgboost_target_probability_backend_when_available():
    """Check the LP can learn target idle probabilities with XGBoost."""
    pytest.importorskip("xgboost")
    data = build_balancing_target_data()

    quantities = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z"],
        min_count=5,
        probability_backend="xgboost",
        probability_model_kwargs={"use_grid_search": False, "random_state": 0},
    )

    assert quantities.z_values == [(0,), (1,)]
    for z in quantities.z_values:
        assert 0.0 <= quantities.p_x1[z] <= 1.0
        assert 0.0 <= quantities.q_y0[z] <= 1.0
        assert 0.0 <= quantities.q_y1[z] <= 1.0

    details = solve_target_bounds_from_observed_binary_quantities(
        quantities,
        make_choice(two_c=1.0, two_d=1.0),
        return_details=True,
    )

    assert set(details["status"]) == {"optimal"}


def test_tabicl_v2_target_probability_backend_routes_to_predictor(
    monkeypatch: pytest.MonkeyPatch,
):
    """Target-LP TabICLv2 backend uses package predictor contract."""
    import causal_grounding.predictors as predictor_module

    calls = []

    class FakePredictor:
        """Minimal predictor implementing target probability calls."""

        def fit_propensity(self, data, treatment, covariates):
            """Record propensity fitting and retain feature names."""
            self.covariates = list(covariates)
            calls.append(("propensity", treatment, tuple(covariates)))
            return self

        def fit_outcome(self, data, outcome, treatment, covariates):
            """Record outcome fitting."""
            calls.append(("outcome", outcome, treatment, tuple(covariates)))
            return self

        def predict_propensity(self, X):
            """Return stable treatment probabilities."""
            return np.full(len(X), 0.4)

        def predict_outcome(self, X, A):
            """Return treatment-dependent outcome probabilities."""
            return np.full(len(X), 0.25 + 0.5 * int(A))

    def fake_create_predictor(method, **kwargs):
        """Capture backend name and options."""
        calls.append(("create", method, kwargs))
        return FakePredictor()

    monkeypatch.setattr(
        predictor_module,
        "create_predictor",
        fake_create_predictor,
    )
    quantities = estimate_target_observed_binary_quantities(
        build_balancing_target_data(),
        treatment="A",
        outcome="Y",
        covariates=["Z"],
        min_count=5,
        probability_backend="tabicl_v2",
        probability_model_kwargs={"random_state": 3},
    )

    assert calls[0] == ("create", "tabicl_v2", {"random_state": 3})
    assert quantities.z_values == [(0,), (1,)]
    assert all(np.isclose(value, 0.4) for value in quantities.p_x1.values())
    assert all(np.isclose(value, 0.25) for value in quantities.q_y0.values())
    assert all(np.isclose(value, 0.75) for value in quantities.q_y1.values())


def test_model_probability_covariates_can_exclude_noise_strata():
    """Model probabilities ignore excluded features while retaining LP strata."""
    pytest.importorskip("xgboost")
    data = pd.concat(
        [
            build_balancing_target_data().assign(W=0),
            build_balancing_target_data().assign(W=1),
        ],
        ignore_index=True,
    )
    quantities = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z", "W"],
        model_covariates=["Z"],
        min_count=5,
        probability_backend="xgboost",
        probability_model_kwargs={"use_grid_search": False, "random_state": 0},
    )

    assert quantities.z_values == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for z in [0, 1]:
        assert np.isclose(
            quantities.p_x1[(z, 0)],
            quantities.p_x1[(z, 1)],
        )
        assert np.isclose(
            quantities.q_y0[(z, 0)],
            quantities.q_y0[(z, 1)],
        )
        assert np.isclose(
            quantities.q_y1[(z, 0)],
            quantities.q_y1[(z, 1)],
        )


def test_empirical_probability_covariates_can_exclude_noise_strata():
    """Empirical probabilities pool useful groups across full LP strata."""
    data = pd.concat(
        [
            build_balancing_target_data().assign(W=0),
            build_balancing_target_data().assign(W=1),
        ],
        ignore_index=True,
    )
    quantities = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z", "W"],
        model_covariates=["Z"],
        min_count=5,
        probability_backend="empirical",
    )

    assert quantities.z_values == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for z in [0, 1]:
        assert np.isclose(
            quantities.p_x1[(z, 0)],
            quantities.p_x1[(z, 1)],
        )
        assert np.isclose(
            quantities.q_y0[(z, 0)],
            quantities.q_y0[(z, 1)],
        )
        assert np.isclose(
            quantities.q_y1[(z, 0)],
            quantities.q_y1[(z, 1)],
        )

    details = solve_target_bounds_from_observed_binary_quantities(
        quantities,
        make_choice(two_c=0.0, two_d=0.0),
        return_details=True,
    )
    assert set(details["status"]) == {"optimal"}


def test_hierarchical_dirichlet_target_probability_backend():
    """Hierarchical Dirichlet backend supports target LP probability fitting."""
    data = pd.concat(
        [
            build_balancing_target_data().assign(W=0),
            build_balancing_target_data().assign(W=1),
        ],
        ignore_index=True,
    )
    quantities = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z", "W"],
        model_covariates=["Z"],
        min_count=5,
        probability_backend="hierarchical_dirichlet",
        probability_model_kwargs={
            "prior_strength": 2.0,
            "base_prior_strength": 2.0,
        },
    )

    assert quantities.z_values == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for z in [0, 1]:
        assert np.isclose(
            quantities.p_x1[(z, 0)],
            quantities.p_x1[(z, 1)],
        )
        assert np.isclose(
            quantities.q_y0[(z, 0)],
            quantities.q_y0[(z, 1)],
        )
        assert np.isclose(
            quantities.q_y1[(z, 0)],
            quantities.q_y1[(z, 1)],
        )

    details = solve_target_bounds_from_observed_binary_quantities(
        quantities,
        make_choice(two_c=1.0, two_d=1.0),
        return_details=True,
    )
    assert set(details["status"]) == {"optimal"}


def test_hierarchical_dirichlet_draws_target_probability_backend():
    """Draw backend samples target LP probability tables reproducibly."""
    data = pd.concat(
        [
            build_balancing_target_data().assign(W=0),
            build_balancing_target_data().assign(W=1),
        ],
        ignore_index=True,
    )
    kwargs = {
        "prior_strength": 2.0,
        "base_prior_strength": 2.0,
        "posterior_sample_seed": 123,
    }
    first = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z", "W"],
        model_covariates=["Z"],
        min_count=5,
        probability_backend="hierarchical_dirichlet_draws",
        probability_model_kwargs=kwargs,
    )
    second = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z", "W"],
        model_covariates=["Z"],
        min_count=5,
        probability_backend="hierarchical_dirichlet_draws",
        probability_model_kwargs=kwargs,
    )
    third = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z", "W"],
        model_covariates=["Z"],
        min_count=5,
        probability_backend="hierarchical_dirichlet_draws",
        probability_model_kwargs={**kwargs, "posterior_sample_seed": 456},
    )

    assert first.p_x1 == second.p_x1
    assert first.p_y1_given_xz == second.p_y1_given_xz
    assert first.p_x1 != third.p_x1
    assert all(0.0 <= value <= 1.0 for value in first.p_x1.values())
    assert all(0.0 <= value <= 1.0 for value in first.p_y1_given_xz.values())


def test_zero_tolerance_lp_recovers_exact_balanced_cate():
    """The direct theta LP collapses to the known CATE under zero tolerance."""
    details = solve_target_bounds_from_data(
        build_balancing_target_data(),
        make_choice(two_c=0.0, two_d=0.0),
        covariates=["Z"],
        return_details=True,
    )

    assert set(details["status"]) == {"optimal"}
    assert np.allclose(details["lower"], 0.5)
    assert np.allclose(details["upper"], 0.5)
    assert np.allclose(details["mu0_lower"], 0.2)
    assert np.allclose(details["mu1_upper"], 0.7)


def test_cate_only_mode_skips_four_intervention_objectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CATE-only mode should preserve CATE bounds with two solves per query."""

    quantities = estimate_target_observed_binary_quantities(
        build_balancing_target_data(),
        treatment="A",
        outcome="Y",
        covariates=["Z"],
        min_count=5,
    )
    original_solve = learned_lp._solve_linear_objective
    calls: list[np.ndarray] = []

    def counted_solve(*args, **kwargs):
        """Record each objective before running the real LP solver."""

        calls.append(args[1])
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(learned_lp, "_solve_linear_objective", counted_solve)
    full = solve_target_bounds_from_observed_binary_quantities(
        quantities,
        make_choice(two_c=0.0, two_d=0.0),
        z_values=[(0,)],
        return_details=True,
    )
    assert len(calls) == 6

    calls.clear()
    cate_only = solve_target_bounds_from_observed_binary_quantities(
        quantities,
        make_choice(two_c=0.0, two_d=0.0),
        z_values=[(0,)],
        return_details=True,
        include_intervention_diagnostics=False,
    )

    assert len(calls) == 2
    assert cate_only.loc[0, "lower"] == pytest.approx(full.loc[0, "lower"])
    assert cate_only.loc[0, "upper"] == pytest.approx(full.loc[0, "upper"])
    assert not bool(cate_only.loc[0, "intervention_diagnostics_computed"])
    assert cate_only[
        ["mu0_lower", "mu0_upper", "mu1_lower", "mu1_upper"]
    ].isna().all().all()
    assert cate_only[
        [
            "mu0_lower_status",
            "mu0_upper_status",
            "mu1_lower_status",
            "mu1_upper_status",
        ]
    ].eq("not_requested").all().all()


def test_looser_tolerance_cannot_narrow_zero_tolerance_bounds():
    """Relaxing learned tolerances cannot shrink feasible CATE intervals."""
    tight = solve_target_bounds_from_data(
        build_balancing_target_data(),
        make_choice(two_c=0.0, two_d=0.0),
        covariates=["Z"],
        return_details=True,
    )
    loose = solve_target_bounds_from_data(
        build_balancing_target_data(),
        make_choice(two_c=1.0, two_d=1.0),
        covariates=["Z"],
        return_details=True,
    )

    assert np.all(loose["lower"].to_numpy() <= tight["lower"].to_numpy() + 1e-9)
    assert np.all(loose["upper"].to_numpy() >= tight["upper"].to_numpy() - 1e-9)


def test_intersection_is_no_wider_than_single_choice():
    """Adding a second selected choice intersects feasible sets."""
    loose = make_choice(two_c=1.0, two_d=1.0)
    tight = make_choice(two_c=0.0, two_d=0.0)
    single = solve_target_bounds_from_data(
        build_balancing_target_data(),
        loose,
        covariates=["Z"],
        return_details=True,
    )
    intersection = solve_target_bounds_from_data(
        build_balancing_target_data(),
        [loose, tight],
        covariates=["Z"],
        return_details=True,
    )

    assert np.all(intersection["lower"].to_numpy() >= single["lower"].to_numpy() - 1e-9)
    assert np.all(intersection["upper"].to_numpy() <= single["upper"].to_numpy() + 1e-9)


def test_fixed_consistency_violation_returns_no_optimal_bounds():
    """Contradictory factual equalities and strict c constraints are infeasible."""
    data = build_balancing_target_data()
    data.loc[(data["Z"] == 1) & (data["A"] == 1), "Y"] = 0

    bounds = solve_target_bounds_from_data(
        data,
        make_choice(two_c=0.1, two_d=1.0),
        covariates=["Z"],
        return_details=False,
    )

    assert bounds == {}


def _state_target_data(seed=3, n_rows=800):
    """Build target idle rows with one relevant covariate and one noise column."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, 0.5, n_rows)
    x = rng.binomial(1, 0.25 + 0.5 * a)
    return pd.DataFrame({
        "F": "idle",
        "a": a,
        "n1": rng.binomial(1, 0.5, n_rows),
        "A": x,
        "Y": rng.binomial(1, 0.2 + 0.4 * x),
    })


def _state_quantities(data):
    """Estimate empirical target tables for the reporting and noise states."""
    return {
        state: estimate_target_observed_binary_quantities(
            data,
            treatment="A",
            outcome="Y",
            covariates=list(state),
            regime_col="F",
            min_count=5,
        )
        for state in (("a",), ("a", "n1"))
    }


STATE_MAP = {"a": ("a",), "n1": ("a", "n1")}


def _state_choice(covariate, two_c, two_d):
    """Return a choice row whose grid index the state solver must ignore."""
    return {
        "covariate": covariate,
        "covariate_index": 99,
        "epsilon": 0.1,
        "tau": 0.05,
        "two_c": two_c,
        "two_d": two_d,
    }


def _count_weights(quantities, positions):
    """Return target-count weights grouping fine strata by reporting value."""
    counts = {}
    for z in quantities.z_values:
        key = tuple(z[position] for position in positions)
        counts.setdefault(key, {})[z] = float(quantities.n_z[z])
    return {
        key: {z: value / sum(values.values()) for z, value in values.items()}
        for key, values in counts.items()
    }


def test_candidate_state_bounds_reduce_to_the_stratum_solver():
    """Choices in the reporting state give the ordinary per-stratum bounds."""
    quantities = _state_quantities(_state_target_data())
    combined = solve_candidate_state_bounds_from_observed_binary_quantities(
        quantities,
        [_state_choice("a", 0.10, 0.10)],
        candidate_states=STATE_MAP,
        reporting_covariates=["a"],
    )
    expected = solve_target_bounds_from_observed_binary_quantities(
        quantities[("a",)],
        [{"covariate": "a", "covariate_index": 0, "two_c": 0.10, "two_d": 0.10}],
        return_details=True,
    )

    columns = ["a", "lower", "upper", "status"]
    pd.testing.assert_frame_equal(
        combined[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
    )
    assert combined["n_candidate_states"].eq(1).all()
    assert combined["z_value_str"].tolist() == expected["z_value_str"].tolist()


def test_candidate_state_bounds_use_count_weighted_queries_for_extra_states():
    """An extra candidate's LP bounds the coarse CATE over its finer strata."""
    quantities = _state_quantities(_state_target_data())
    combined = solve_candidate_state_bounds_from_observed_binary_quantities(
        quantities,
        [_state_choice("n1", 0.20, 0.20)],
        candidate_states=STATE_MAP,
        reporting_covariates=["a"],
    )
    fine = quantities[("a", "n1")]
    expected = solve_target_weighted_cate_bounds_from_observed_binary_quantities(
        fine,
        [{"covariate": "n1", "covariate_index": 1, "two_c": 0.20, "two_d": 0.20}],
        _count_weights(fine, [0]),
        query_covariates=["a"],
        return_details=True,
    )

    np.testing.assert_allclose(combined["lower"], expected["lower"])
    np.testing.assert_allclose(combined["upper"], expected["upper"])
    assert combined["status"].tolist() == expected["status"].tolist()
    assert combined["binding_lower_state"].eq("a|n1").all()


def test_candidate_state_bounds_intersect_across_states():
    """Bounds from different states are intersected stratum by stratum."""
    quantities = _state_quantities(_state_target_data())
    real = solve_candidate_state_bounds_from_observed_binary_quantities(
        quantities, [_state_choice("a", 0.10, 0.10)],
        candidate_states=STATE_MAP, reporting_covariates=["a"],
    )
    noise = solve_candidate_state_bounds_from_observed_binary_quantities(
        quantities, [_state_choice("n1", 0.20, 0.20)],
        candidate_states=STATE_MAP, reporting_covariates=["a"],
    )
    both = solve_candidate_state_bounds_from_observed_binary_quantities(
        quantities,
        [_state_choice("a", 0.10, 0.10), _state_choice("n1", 0.20, 0.20)],
        candidate_states=STATE_MAP,
        reporting_covariates=["a"],
    )

    np.testing.assert_allclose(both["lower"], np.maximum(real["lower"], noise["lower"]))
    np.testing.assert_allclose(both["upper"], np.minimum(real["upper"], noise["upper"]))
    assert both["n_candidate_states"].eq(2).all()
    assert both["status"].eq("optimal").all()


def test_candidate_state_combination_rules(monkeypatch):
    """Empty intersections and failed states never yield optimal strata."""

    def fake_rows(values):
        """Build per-stratum solver rows from ``(a, lower, upper, status)``."""
        return pd.DataFrame([
            {
                "z_value": (a,), "z_value_str": str((a,)), "a": a,
                "lower": lower, "upper": upper, "status": status,
                "lower_status": status, "upper_status": status,
                "mu0_lower": np.nan, "mu0_upper": np.nan,
                "mu0_lower_status": "not_requested", "mu0_upper_status": "not_requested",
                "mu1_lower": np.nan, "mu1_upper": np.nan,
                "mu1_lower_status": "not_requested", "mu1_upper_status": "not_requested",
                "n_choices": 1, "n_strata": 2, "n_variables": 8, "n_constraints": 4,
            }
            for a, lower, upper, status in values
        ])

    monkeypatch.setattr(
        learned_lp,
        "solve_target_bounds_from_observed_binary_quantities",
        lambda *_args, **_kwargs: fake_rows([
            (0, -0.2, 0.1, "optimal"),
            (1, -0.1, 0.4, "optimal"),
            (2, 0.0, 0.5, "invalid_interval"),
        ]),
    )
    monkeypatch.setattr(
        learned_lp,
        "solve_target_weighted_cate_bounds_from_observed_binary_quantities",
        lambda *_args, **_kwargs: fake_rows([
            (0, 0.3, 0.6, "optimal"),
            (1, -0.3, 0.2, "nonoptimal"),
        ]).rename(columns={"z_value": "query_value", "z_value_str": "query_value_str"}),
    )
    quantities = {
        ("a",): learned_lp.TargetObservedBinaryQuantities(["a"], [], {}, {}, {}, {}, {}),
        ("a", "n1"): learned_lp.TargetObservedBinaryQuantities(
            ["a", "n1"], [], {}, {}, {}, {}, {}
        ),
    }
    combined = solve_candidate_state_bounds_from_observed_binary_quantities(
        quantities,
        [_state_choice("a", 0.1, 0.1), _state_choice("n1", 0.1, 0.1)],
        candidate_states=STATE_MAP,
        reporting_covariates=["a"],
        include_intervention_diagnostics=False,
    ).set_index("a")

    assert combined.loc[0, "status"] == "empty_intersection"
    assert combined.loc[0, "lower"] > combined.loc[0, "upper"]
    assert combined.loc[1, "status"] == "nonoptimal"
    assert np.isnan(combined.loc[1, "lower"])
    # A stratum returned by one state only is reported exactly as it solved.
    assert combined.loc[2, "status"] == "invalid_interval"
    assert combined.loc[2, "n_candidate_states"] == 1


def test_candidate_state_bounds_reject_unknown_states():
    """Choices must map to a state that has a matching probability table."""
    quantities = _state_quantities(_state_target_data())
    with pytest.raises(ValueError, match="No candidate state"):
        solve_candidate_state_bounds_from_observed_binary_quantities(
            quantities, [_state_choice("zz", 0.1, 0.1)],
            candidate_states=STATE_MAP, reporting_covariates=["a"],
        )
    with pytest.raises(ValueError, match="No target probability table"):
        solve_candidate_state_bounds_from_observed_binary_quantities(
            {("a",): quantities[("a",)]}, [_state_choice("n1", 0.1, 0.1)],
            candidate_states=STATE_MAP, reporting_covariates=["a"],
        )
