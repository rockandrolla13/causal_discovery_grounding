"""Tests for target-site naturalness policy selection."""

import numpy as np
import pandas as pd
import pytest

from causal_grounding.lp_solver_learned_naturalness import (
    TargetObservedBinaryQuantities,
    estimate_target_observed_binary_quantities,
)
from causal_grounding.naturalness_statistics import (
    compute_d_strength,
    compute_i_deviation,
)
from causal_grounding.naturalness_grounder import (
    NaturalnessGrid,
    NaturalnessGrounder,
    NaturalnessTolerance,
)
from causal_grounding.naturalness_target_policy import TargetNaturalnessPolicy


class FakeGrounder:
    """Minimal fitted-grounder shape needed by target policies."""

    def __init__(self, grid):
        """Store the attributes consumed by ``TargetNaturalnessPolicy``."""
        self.grid = grid
        self.treatment_ = "A"
        self.outcome_ = "Y"
        self.regime_col_ = "F"
        self.idle_value = "idle"
        self.alpha = 0.0


class FakeJointUpperGrounder(FakeGrounder):
    """Grounder double exposing simultaneous upper-vector prediction."""

    def __init__(self, grid):
        """Store grid and initialize joint-call diagnostics."""
        super().__init__(grid)
        self.joint_rows = None

    def predict_simultaneous_upper_tolerances(
        self,
        rows,
        *,
        confidence_level,
        clip,
    ):
        """Return deterministic joint-upper tolerances in input row order."""
        self.joint_rows = rows.copy()
        return [
            NaturalnessTolerance(
                covariate=str(row.covariate),
                epsilon=float(row.epsilon),
                tau=float(row.tau),
                c=0.12,
                d=0.08,
                c_source=f"gp_simultaneous_upper_{confidence_level:.2f}",
                d_source=f"gp_simultaneous_upper_{confidence_level:.2f}",
                c_std_logit=0.10,
                d_std_logit=0.20,
                upper_confidence_level=confidence_level,
                upper_critical_value=2.5,
                upper_family_size=2 * len(rows),
                upper_n_samples=4096,
            )
            for row in rows.itertuples(index=False)
        ]


class FakeToleranceProvider:
    """Deterministic provider for policy-ranking tests."""

    def get(self, covariate, epsilon, tau, i_deviation=None, d_strength=None):
        """Return small tolerances only for the intended primary row."""
        if np.isclose(epsilon, 0.20) and np.isclose(tau, 0.10):
            c, d = 0.01, 0.01
        else:
            c, d = 0.30, 0.30
        return NaturalnessTolerance(
            covariate=covariate,
            epsilon=epsilon,
            tau=tau,
            c=c,
            d=d,
            c_source="fake",
            d_source="fake",
            c_std_logit=None,
            d_std_logit=None,
        )


class RecordingToleranceProvider(FakeToleranceProvider):
    """Provider that records observed target statistics passed by the policy."""

    def __init__(self):
        """Initialize an empty call log."""
        self.calls = []

    def get(self, covariate, epsilon, tau, i_deviation=None, d_strength=None):
        """Record lookup arguments and return deterministic tolerances."""
        self.calls.append({
            "covariate": covariate,
            "epsilon": epsilon,
            "tau": tau,
            "i_deviation": i_deviation,
            "d_strength": d_strength,
        })
        return super().get(
            covariate,
            epsilon,
            tau,
            i_deviation=i_deviation,
            d_strength=d_strength,
        )


def build_target_data():
    """Create target data where Z moves treatment but not Y|A,Z."""
    rows = []
    for z, n_treated in [(0, 2), (1, 10)]:
        n_control = 12 - n_treated
        for _ in range(n_control):
            rows.append({"F": "idle", "Z": z, "A": 0, "Y": 0})
        for _ in range(n_treated):
            rows.append({"F": "idle", "Z": z, "A": 1, "Y": 1})
    return pd.DataFrame(rows)


def test_source_and_model_target_triggers_match_on_shared_support():
    """Shared trigger helpers preserve source and model-target I/D statistics."""
    rows = []
    for regime in ("idle", "on"):
        for z, n_treated in ((0, 4), (1, 16)):
            for x, count in ((0, 20 - n_treated), (1, n_treated)):
                for index in range(count):
                    rows.append({
                        "F": regime,
                        "Z": z,
                        "A": x,
                        "Y": int(index % 2 == 0),
                    })
    data = pd.DataFrame(rows)
    grid = NaturalnessGrid(
        epsilons=[0.1],
        taus=[0.1],
        covariates=["Z"],
        min_cell_count=1,
    )
    grounder = NaturalnessGrounder(grid=grid, alpha=0.5, fit_gp=False)
    grounder.fit({"site": data}, treatment="A", outcome="Y")
    source = grounder.observations_.iloc[0]
    quantities = estimate_target_observed_binary_quantities(
        data,
        treatment="A",
        outcome="Y",
        covariates=["Z"],
        regime_col="F",
        min_count=1,
        alpha=0.5,
    )
    target = TargetNaturalnessPolicy().evaluate_triggers(
        data,
        grounder,
        trigger_quantities=quantities,
    ).iloc[0]

    assert np.isclose(target.i_deviation, source.i_deviation)
    assert np.isclose(target.d_strength, source.d_strength)


def test_public_naturalness_statistics_api_computes_trigger_values():
    """Public statistic helpers expose the target-trigger definitions."""
    idle = pd.DataFrame({"Z": [0, 0, 1, 1]})
    outcome_probability = lambda y, x, z: float(z[0] == 1 and y == 1)
    treatment_probability = lambda x, z: float((z[0] == 1) == (x == 1))

    assert np.isclose(
        compute_i_deviation(["Z"], 0, [()], outcome_probability, idle),
        1.0,
    )
    assert np.isclose(
        compute_d_strength(["Z"], 0, [()], treatment_probability, idle),
        1.0,
    )


def test_all_triggered_intersection_policy_returns_all_candidates():
    grid = NaturalnessGrid(
        epsilons=[0.05, 0.20],
        taus=[0.10, 0.50],
        covariates=["Z"],
        min_cell_count=1,
    )
    result = TargetNaturalnessPolicy("all_triggered_intersection").choose(
        build_target_data(),
        FakeGrounder(grid),
        provider=FakeToleranceProvider(),
    )

    assert result.status == "selected"
    assert len(result.candidates) == 4
    assert len(result.selected_choices) == 4
    assert result.choice is not None
    assert np.isclose(result.choice.epsilon, 0.20)
    assert np.isclose(result.choice.tau, 0.10)
    assert np.isclose(result.choice.score, 0.04)
    assert result.selected_choices[0].is_primary_choice
    assert sum(choice.is_primary_choice for choice in result.selected_choices) == 1


def test_target_policy_passes_observed_stats_to_provider():
    """Candidate lookup should receive the realized target I/D statistics."""
    grid = NaturalnessGrid(
        epsilons=[0.20],
        taus=[0.10],
        covariates=["Z"],
        min_cell_count=1,
    )
    provider = RecordingToleranceProvider()

    result = TargetNaturalnessPolicy("all_triggered_intersection").choose(
        build_target_data(),
        FakeGrounder(grid),
        provider=provider,
    )

    assert result.status == "selected"
    assert provider.calls
    assert np.isfinite(provider.calls[0]["i_deviation"])
    assert np.isfinite(provider.calls[0]["d_strength"])
    assert np.isclose(
        provider.calls[0]["i_deviation"],
        result.choice.i_deviation,
    )
    assert np.isclose(
        provider.calls[0]["d_strength"],
        result.choice.d_strength,
    )


def test_target_policy_reports_no_triggered_candidates():
    grid = NaturalnessGrid(
        epsilons=[0.01],
        taus=[0.90],
        covariates=["Z"],
        min_cell_count=1,
    )
    result = TargetNaturalnessPolicy("all_triggered_intersection").choose(
        build_target_data(),
        FakeGrounder(grid),
        provider=FakeToleranceProvider(),
    )

    assert result.status == "no_triggered_candidates"
    assert result.choice is None
    assert result.selected_choices == []


def test_target_policy_exposes_rejected_trigger_diagnostics():
    """Complete diagnostics retain I/D margins for rejected target rows."""

    grid = NaturalnessGrid(
        epsilons=[0.01, 0.50],
        taus=[0.10, 0.90],
        covariates=["Z"],
        min_cell_count=1,
    )
    diagnostics = TargetNaturalnessPolicy(
        "all_triggered_intersection"
    ).evaluate_triggers(
        build_target_data(),
        FakeGrounder(grid),
    )

    assert len(diagnostics) == 4
    assert {
        "i_deviation",
        "d_strength",
        "i_margin",
        "d_margin",
        "target_triggered",
        "rejection_reason",
    }.issubset(diagnostics.columns)
    assert diagnostics["target_triggered"].any()
    assert (~diagnostics["target_triggered"]).any()
    assert set(
        diagnostics.loc[
            ~diagnostics["target_triggered"],
            "rejection_reason",
        ]
    ) - {"triggered"}


def test_target_policy_uses_model_probabilities_on_paired_support():
    """Model-backed triggers ignore unpaired cells and replace raw proportions."""

    rows = []
    for w in (0, 1):
        for z in (0, 1):
            for replicate in range(4):
                rows.append({
                    "F": "idle",
                    "W": w,
                    "Z": z,
                    "A": replicate % 2,
                    "Y": replicate % 2,
                })
    for replicate in range(4):
        rows.append({
            "F": "idle",
            "W": 2,
            "Z": 0,
            "A": replicate % 2,
            "Y": replicate % 2,
        })
    target = pd.DataFrame(rows)
    z_values = [
        (w, z)
        for w in (0, 1)
        for z in (0, 1)
    ] + [(2, 0)]
    p_x1 = {
        state: (0.10 if state[1] == 0 else 0.90)
        for state in z_values
    }
    quantities = TargetObservedBinaryQuantities(
        covariates=["W", "Z"],
        z_values=z_values,
        p_x1=p_x1,
        p_x0={state: 1.0 - value for state, value in p_x1.items()},
        p_y1_given_xz={
            (x, state): 0.40
            for state in z_values
            for x in (0, 1)
        },
        n_z={state: 4 for state in z_values},
        n_xz={
            (x, state): 2
            for state in z_values
            for x in (0, 1)
        },
    )
    grid = NaturalnessGrid(
        epsilons=[0.05],
        taus=[0.50],
        covariates=["W", "Z"],
        candidate_covariates=["Z"],
        min_cell_count=2,
    )
    grounder = FakeGrounder(grid)
    policy = TargetNaturalnessPolicy("all_triggered_intersection")

    empirical = policy.evaluate_triggers(target, grounder)
    modeled = policy.evaluate_triggers(
        target,
        grounder,
        trigger_quantities=quantities,
    )

    assert not empirical["target_triggered"].any()
    assert modeled["target_triggered"].all()
    assert modeled["n_z_minus_evaluated"].eq(2).all()
    assert modeled["target_probability_source"].eq("model").all()
    assert np.isclose(modeled["i_deviation"].iloc[0], 0.0)
    assert np.isclose(modeled["d_strength"].iloc[0], 0.8)


def test_model_triggers_intersect_supplied_probability_support():
    """Trigger comparisons should skip pairs absent from supplied support."""

    target = pd.DataFrame(
        [
            {
                "F": "idle",
                "W": w,
                "Z": z,
                "A": replicate % 2,
                "Y": replicate % 2,
            }
            for w in (0, 1)
            for z in (0, 1)
            for replicate in range(4)
        ]
    )
    z_values = [(0, 0), (0, 1), (1, 0)]
    p_x1 = {
        state: (0.10 if state[1] == 0 else 0.90)
        for state in z_values
    }
    quantities = TargetObservedBinaryQuantities(
        covariates=["W", "Z"],
        z_values=z_values,
        p_x1=p_x1,
        p_x0={state: 1.0 - value for state, value in p_x1.items()},
        p_y1_given_xz={
            (x, state): 0.40
            for state in z_values
            for x in (0, 1)
        },
        n_z={state: 4 for state in z_values},
        n_xz={
            (x, state): 2
            for state in z_values
            for x in (0, 1)
        },
    )
    grid = NaturalnessGrid(
        epsilons=[0.05],
        taus=[0.50],
        covariates=["W", "Z"],
        candidate_covariates=["Z"],
        min_cell_count=2,
    )

    diagnostics = TargetNaturalnessPolicy(
        "all_triggered_intersection"
    ).evaluate_triggers(
        target,
        FakeGrounder(grid),
        trigger_quantities=quantities,
    )

    assert diagnostics["target_triggered"].all()
    assert diagnostics["n_z_minus_evaluated"].eq(1).all()
    assert np.isclose(diagnostics["i_deviation"].iloc[0], 0.0)
    assert np.isclose(diagnostics["d_strength"].iloc[0], 0.8)


def test_target_policy_uses_one_simultaneous_upper_vector():
    """Simultaneous mode protects all triggered rows before ranking or selection."""
    grid = NaturalnessGrid(
        epsilons=[0.05, 0.20],
        taus=[0.10, 0.50],
        covariates=["Z"],
        min_cell_count=1,
    )
    grounder = FakeJointUpperGrounder(grid)

    result = TargetNaturalnessPolicy(
        "all_triggered_intersection",
        uncertainty="posterior_simultaneous_upper",
        confidence_level=0.90,
    ).choose(
        build_target_data(),
        grounder,
    )
    frame = result.to_frame(selected_only=True)

    assert result.status == "selected"
    assert grounder.joint_rows is not None
    assert len(grounder.joint_rows) == len(result.selected_choices) == 4
    assert set(frame["c_source"]) == {"gp_simultaneous_upper_0.90"}
    assert set(frame["upper_confidence_level"]) == {0.90}
    assert set(frame["upper_critical_value"]) == {2.5}
    assert set(frame["upper_family_size"]) == {8}
    assert set(frame["upper_n_samples"]) == {4096}


def _candidate_state_target_data(seed=5, n_rows=600):
    """Build target idle rows with two real covariates and two noise columns."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, 0.5, n_rows)
    b = rng.binomial(1, 0.5, n_rows)
    x = rng.binomial(1, 0.2 + 0.6 * a)
    return pd.DataFrame({
        "F": "idle",
        "a": a,
        "b": b,
        "n1": rng.binomial(1, 0.5, n_rows),
        "n2": rng.binomial(1, 0.5, n_rows),
        "A": x,
        "Y": rng.binomial(1, 0.2 + 0.3 * x + 0.2 * b),
    })


def test_target_triggers_are_evaluated_in_candidate_states():
    """Each candidate's target statistics use only its own state's covariates."""
    target = _candidate_state_target_data()
    grid_kwargs = {"epsilons": [0.05, 0.5], "taus": [0.01, 0.1], "min_cell_count": 2}
    joint_grid = NaturalnessGrid(
        covariates=["a", "b", "n1", "n2"],
        conditioning_covariates=["a", "b"],
        **grid_kwargs,
    )
    base_grid = NaturalnessGrid(covariates=["a", "b"], **grid_kwargs)
    single_grid = NaturalnessGrid(
        covariates=["a", "b", "n1"], candidate_covariates=["n1"], **grid_kwargs
    )
    policy = TargetNaturalnessPolicy()
    columns = [
        "epsilon", "tau", "i_deviation", "d_strength",
        "target_triggered", "n_z_minus_evaluated",
    ]

    def rows(frame, covariate):
        """Return one candidate's trigger rows on a fresh index."""
        return frame.loc[frame["covariate"].eq(covariate), columns].reset_index(drop=True)

    joint = policy.evaluate_triggers(target, FakeGrounder(joint_grid))
    base = policy.evaluate_triggers(target, FakeGrounder(base_grid))
    single = policy.evaluate_triggers(target, FakeGrounder(single_grid))
    for covariate in ("a", "b"):
        pd.testing.assert_frame_equal(rows(joint, covariate), rows(base, covariate))
    pd.testing.assert_frame_equal(rows(joint, "n1"), rows(single, "n1"))

    tables = {
        state: estimate_target_observed_binary_quantities(
            target,
            treatment="A",
            outcome="Y",
            covariates=list(state),
            regime_col="F",
            min_count=2,
        )
        for state in joint_grid.candidate_state_keys
    }
    mapped = policy.evaluate_triggers(
        target, FakeGrounder(joint_grid), trigger_quantities=tables
    )
    single_model = policy.evaluate_triggers(
        target,
        FakeGrounder(single_grid),
        trigger_quantities=tables[("a", "b", "n1")],
    )
    pd.testing.assert_frame_equal(rows(mapped, "n1"), rows(single_model, "n1"))
    with pytest.raises(ValueError, match="no table for candidate state"):
        policy.evaluate_triggers(
            target,
            FakeGrounder(joint_grid),
            trigger_quantities={("a", "b"): tables[("a", "b")]},
        )


def test_policy_result_carries_the_complete_trigger_grid():
    """The chosen result exposes every evaluated row, including rejections."""
    grid = NaturalnessGrid(
        epsilons=[0.20],
        taus=[0.10, 0.90],
        covariates=["Z"],
        min_cell_count=1,
    )
    result = TargetNaturalnessPolicy().choose(
        build_target_data(),
        FakeGrounder(grid),
        provider=FakeToleranceProvider(),
    )

    assert result.trigger_grid is not None
    assert len(result.trigger_grid) == 2
    assert result.trigger_grid["target_triggered"].tolist() == [True, False]
