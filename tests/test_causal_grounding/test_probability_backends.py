"""Tests for the single probability-backend registry."""

import warnings

import numpy as np
import pandas as pd
import pytest

from causal_grounding import probability_backends as registry
from causal_grounding.lp_solver_learned_naturalness import (
    HIERARCHICAL_DIRICHLET_BACKENDS,
    HIERARCHICAL_DIRICHLET_DRAW_BACKENDS,
    estimate_target_observed_binary_quantities,
)
from causal_grounding.naturalness_grounder import (
    NATURALNESS_STAGE1_BACKENDS,
    EmpiricalSitePMFModels,
    _resolve_stage1_backend,
    register_naturalness_stage1_backend,
)
from causal_grounding.predictors import (
    HierarchicalDirichletPredictor,
    _expected_xgboost_fit_warnings_ignored,
    create_predictor,
)


@pytest.fixture
def dummy_backend():
    """Register a dummy backend for one test and remove it afterwards."""
    yield registry.register_probability_backend(
        "dummy_backend",
        predictor=HierarchicalDirichletPredictor,
        stage1=EmpiricalSitePMFModels,
        target_lp=True,
    )
    registry.unregister_probability_backend("dummy_backend")


def _target_rows(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Build binary target idle rows with one binary covariate."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n)
    x = (rng.random(n) < 0.3 + 0.4 * z).astype(int)
    y = (rng.random(n) < 0.2 + 0.3 * x + 0.2 * z).astype(int)
    return pd.DataFrame({"z": z, "x": x, "y": y})


def test_builtin_lists_are_unchanged() -> None:
    """The derived lists keep the backend strings and order of the hand-written ones."""
    assert list(NATURALNESS_STAGE1_BACKENDS) == [
        "empirical",
        "xgboost",
        "hierarchical_dirichlet",
        "tabicl_v2",
    ]
    assert HIERARCHICAL_DIRICHLET_BACKENDS == {"hierarchical_dirichlet"}
    assert HIERARCHICAL_DIRICHLET_DRAW_BACKENDS == {
        "hierarchical_dirichlet_draws",
        "hierarchical_dirichlet_posterior",
        "dirichlet_posterior",
    }
    assert registry.probability_backend_names(target_lp=True) == (
        "empirical",
        "xgboost",
        "hierarchical_dirichlet",
        "hierarchical_dirichlet_draws",
        "hierarchical_dirichlet_posterior",
        "dirichlet_posterior",
        "tabicl_v2",
    )
    assert registry.probability_backend_names(stage1=True, model_based=True) == (
        "xgboost",
        "hierarchical_dirichlet",
        "tabicl_v2",
    )


def test_one_registration_reaches_stage1_predictor_and_lp(dummy_backend) -> None:
    """A single registration makes a backend available to every lookup."""
    assert "dummy_backend" in NATURALNESS_STAGE1_BACKENDS
    assert _resolve_stage1_backend("dummy_backend") == (
        "dummy_backend",
        EmpiricalSitePMFModels,
    )
    assert isinstance(create_predictor("dummy_backend"), HierarchicalDirichletPredictor)

    data = _target_rows()
    dummy = estimate_target_observed_binary_quantities(
        data, "x", "y", ["z"], probability_backend="dummy_backend"
    )
    reference = estimate_target_observed_binary_quantities(
        data, "x", "y", ["z"], probability_backend="hierarchical_dirichlet"
    )
    assert dummy.p_x1 == reference.p_x1
    assert dummy.p_y1_given_xz == reference.p_y1_given_xz


def test_unregistered_backend_is_rejected_everywhere() -> None:
    """An unknown name fails in each lookup with the established error types."""
    with pytest.raises(ValueError, match="Unknown method"):
        create_predictor("not_a_backend")
    with pytest.raises(ValueError, match="Unknown stage1_backend"):
        _resolve_stage1_backend("not_a_backend")
    with pytest.raises(ValueError, match="probability_backend must be one of"):
        estimate_target_observed_binary_quantities(
            _target_rows(), "x", "y", ["z"], probability_backend="not_a_backend"
        )


def test_stage1_registration_keeps_other_capabilities() -> None:
    """The legacy Stage-1 registration adds an adapter without dropping the predictor."""
    original = registry.get_probability_backend("hierarchical_dirichlet")
    try:
        register_naturalness_stage1_backend("hierarchical_dirichlet", EmpiricalSitePMFModels)
        entry = registry.get_probability_backend("hierarchical_dirichlet")
        assert NATURALNESS_STAGE1_BACKENDS["hierarchical_dirichlet"] is EmpiricalSitePMFModels
        assert entry.target_lp
        assert entry.predictor_factory is HierarchicalDirichletPredictor
    finally:
        registry._REGISTRY["hierarchical_dirichlet"] = original


def test_stage1_only_registration_is_not_an_lp_backend() -> None:
    """A Stage-1-only name stays out of the predictor and target-LP lookups."""
    register_naturalness_stage1_backend("stage1_only", EmpiricalSitePMFModels)
    try:
        assert "stage1_only" in NATURALNESS_STAGE1_BACKENDS
        assert "stage1_only" not in registry.probability_backend_names(target_lp=True)
        with pytest.raises(ValueError, match="Unknown method"):
            create_predictor("stage1_only")
    finally:
        del NATURALNESS_STAGE1_BACKENDS["stage1_only"]
    assert "stage1_only" not in NATURALNESS_STAGE1_BACKENDS


def test_xgboost_fit_filter_is_narrow() -> None:
    """The XGBoost fit filter hides the known CV warnings and nothing else."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _expected_xgboost_fit_warnings_ignored():
            warnings.warn(
                "The least populated class in y has only 1 members, which is less "
                "than n_splits=5.",
                UserWarning,
            )
            warnings.warn("One or more of the test scores are non-finite: [nan]", UserWarning)
            warnings.warn("something unexpected", UserWarning)
            warnings.warn("numerical trouble", RuntimeWarning)
    assert [str(w.message) for w in caught] == ["something unexpected", "numerical trouble"]
