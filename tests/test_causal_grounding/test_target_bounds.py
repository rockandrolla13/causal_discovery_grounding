"""Tests of the per-draw target program summary in ``target_bounds``."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from causal_grounding.target_bounds import (
    posterior_target_bounds,
    target_probability_draw_seeds,
)

QUANTITIES = SimpleNamespace(z_values=[(0,), (1,), (2,)])


def _draws(n_draws: int) -> pd.DataFrame:
    """Return ``n_draws`` tolerance draws of one condition."""
    return pd.DataFrame({"sample_id": range(n_draws), "c": 0.1, "d": 0.2})


def _solve(quantities, draw):
    """Solve stub: stratum 0 always optimal, 1 optimal on even draws, 2 absent."""
    sample_id = int(draw["sample_id"].iloc[0])
    return pd.DataFrame(
        {
            "Z": [0, 1],
            "status": ["optimal", "optimal" if sample_id % 2 == 0 else "infeasible"],
            "lower": [-0.1 * sample_id, -0.5],
            "upper": [0.1 * sample_id, 0.5],
        }
    )


def test_summary_averages_optimal_draws():
    """Endpoints average the optimal draws, and absent strata get p_optimal 0."""
    bounds = posterior_target_bounds(QUANTITIES, _draws(4), ["Z"], solve_draw=_solve)
    assert list(bounds.columns) == ["Z", "lower", "upper", "p_optimal"]
    assert bounds["Z"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(bounds["lower"].iloc[:2], [-0.15, -0.5])
    np.testing.assert_allclose(bounds["upper"].iloc[:2], [0.15, 0.5])
    np.testing.assert_allclose(bounds["p_optimal"], [1.0, 0.5, 0.0])
    assert bounds.loc[2, ["lower", "upper"]].isna().all()


def test_threads_give_the_serial_result():
    """``n_jobs > 1`` returns the serial bounds and samples in draw order."""
    serial = posterior_target_bounds(
        QUANTITIES, _draws(8), ["Z"], solve_draw=_solve, return_samples=True
    )
    threaded = posterior_target_bounds(
        QUANTITIES, _draws(8), ["Z"], solve_draw=_solve, n_jobs=4, return_samples=True
    )
    for left, right in zip(serial, threaded):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert serial[1]["sample_id"].tolist() == [k for k in range(8) for _ in range(2)]


def test_draw_quantities_receive_each_sample_id():
    """Each draw is solved with the quantities its ``sample_id`` maps to."""
    seen = []

    def solve(quantities, draw):
        seen.append((quantities, int(draw["sample_id"].iloc[0])))
        return _solve(quantities, draw)

    posterior_target_bounds(
        QUANTITIES,
        _draws(3),
        ["Z"],
        draw_quantities=lambda sample_id: f"table{sample_id}",
        solve_draw=solve,
    )
    assert seen == [("table0", 0), ("table1", 1), ("table2", 2)]


@pytest.mark.parametrize(
    ("draws", "n_jobs", "match"),
    [
        (pd.DataFrame({"c": [0.1]}), 1, "sample_id"),
        (_draws(0), 1, "non-empty"),
        (_draws(2), 0, "n_jobs"),
    ],
)
def test_invalid_input_is_rejected(draws, n_jobs, match):
    """Draws without ``sample_id``, no draws, or ``n_jobs < 1`` are rejected."""
    with pytest.raises(ValueError, match=match):
        posterior_target_bounds(QUANTITIES, draws, ["Z"], solve_draw=_solve, n_jobs=n_jobs)


def test_draw_seeds_are_reproducible_and_distinct():
    """Per-draw PMF seeds depend on the run seed only, and differ per draw."""
    seeds = target_probability_draw_seeds(0, 5)
    assert seeds == target_probability_draw_seeds(0, 5)
    assert len(set(seeds)) == 5
    assert seeds != target_probability_draw_seeds(1, 5)
    assert target_probability_draw_seeds(0, 3) == seeds[:3]
