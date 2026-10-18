"""The README's toy example runs and its intervals behave as described."""

import numpy as np

from examples.toy_naturalness import (
    COVARIATES,
    naturalness_bounds,
    reference_bounds,
    simulate_toy_data,
)


def test_toy_example_bounds_cover_the_true_cate():
    """Naturalness fires for the instrument-like Z1 and covers every true CATE."""
    sources, target, truth = simulate_toy_data(seed=0)
    bounds, conditions = naturalness_bounds(sources, target, n_draws=100, seed=0)

    assert set(conditions["covariate"]) == {"Z1"}
    table = truth.merge(bounds, on=COVARIATES).merge(reference_bounds(target), on=COVARIATES)
    assert len(table) == 4
    assert table["p_optimal"].eq(1.0).all()
    assert (table["lower"] <= table["true_cate"]).all()
    assert (table["true_cate"] <= table["upper"]).all()
    # Narrower than the assumption-free Manski bounds, whose width is one.
    assert ((table["upper"] - table["lower"]) < 0.7).all()
    np.testing.assert_allclose(table["manski_upper"] - table["manski_lower"], 1.0)
