"""Tests for the sharp Balke-Pearl instrumental-variable bounds.

Every case here has an answer derivable without the implementation, which is
the point: the module solves the response-type linear program from first
principles rather than evaluating a memorised closed form, so it has to be
pinned against cases whose bounds are known a priori.
"""
from itertools import product

import numpy as np
import pytest

from causal_grounding.balke_pearl import (
    OUTCOME_TYPES,
    TREATMENT_TYPES,
    balke_pearl_interval,
    intersect_intervals,
    joint_table_from_counts,
)


def table_from_types(weights):
    """Build P(Y=y, X=x | Z=z) from a distribution over latent types."""
    table = np.zeros((2, 2, 2))
    for (outcome_type, treatment_type), mass in weights.items():
        for z in (0, 1):
            x = treatment_type[z]
            table[z, outcome_type[x], x] += mass
    return table


def true_effect(weights):
    return sum(m * (ty[1] - ty[0]) for (ty, _), m in weights.items())


def test_full_compliance_point_identifies_the_effect():
    """With everyone a complier the instrument is a randomised trial."""
    weights = {((0, 1), (0, 1)): 0.5, ((0, 0), (0, 1)): 0.3, ((1, 0), (0, 1)): 0.2}
    result = balke_pearl_interval(table_from_types(weights))
    assert result.feasible and result.status == "sharp"
    assert result.upper - result.lower == pytest.approx(0.0, abs=1e-6)
    assert result.lower == pytest.approx(true_effect(weights), abs=1e-6)


def test_irrelevant_instrument_returns_the_manski_width():
    """If treatment never responds to the instrument, nothing is gained."""
    weights = {((0, 1), (1, 1)): 0.5, ((1, 0), (0, 0)): 0.5}
    result = balke_pearl_interval(table_from_types(weights))
    assert result.upper - result.lower == pytest.approx(1.0, abs=1e-6)


def test_bounds_contain_the_truth_across_random_valid_models():
    """Sharpness is worthless without validity; this is the validity check."""
    rng = np.random.default_rng(0)
    types = list(product(OUTCOME_TYPES, TREATMENT_TYPES))
    for _ in range(400):
        draw = rng.dirichlet(np.ones(len(types)))
        weights = {t: draw[i] for i, t in enumerate(types)}
        result = balke_pearl_interval(table_from_types(weights))
        assert result.feasible
        assert result.lower - 1e-8 <= true_effect(weights) <= result.upper + 1e-8


def test_never_looser_than_manski():
    """Adding a valid instrument cannot widen an identification set."""
    rng = np.random.default_rng(1)
    types = list(product(OUTCOME_TYPES, TREATMENT_TYPES))
    for _ in range(300):
        draw = rng.dirichlet(np.ones(len(types)))
        table = table_from_types({t: draw[i] for i, t in enumerate(types)})
        result = balke_pearl_interval(table)
        manski = []
        for z in (0, 1):
            treated_well, control_well = table[z, 1, 1], table[z, 1, 0]
            treated = table[z, 1, 1] + table[z, 0, 1]
            control = table[z, 1, 0] + table[z, 0, 0]
            manski.append((treated_well - control_well - treated,
                           treated_well - control_well + control))
        assert result.lower >= max(m[0] for m in manski) - 1e-8
        assert result.upper <= min(m[1] for m in manski) + 1e-8


def test_instrumental_inequality_is_reported_not_hidden():
    """A table no valid instrument can produce must be flagged, not fudged."""
    table = np.zeros((2, 2, 2))
    table[0, 1, 1], table[0, 0, 0] = 0.9, 0.1
    table[1, 0, 1], table[1, 1, 0] = 0.9, 0.1
    result = balke_pearl_interval(table)
    assert not result.feasible and result.status == "infeasible"
    assert (result.lower, result.upper) == (-1.0, 1.0)


def test_malformed_tables_are_rejected():
    with pytest.raises(ValueError, match="2,2,2"):
        balke_pearl_interval(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="sum to one"):
        balke_pearl_interval(np.zeros((2, 2, 2)))
    bad = np.zeros((2, 2, 2))
    bad[0, 0, 0] = bad[1, 0, 0] = 1.0
    bad[0, 1, 1] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        balke_pearl_interval(bad)


def test_empty_instrument_level_yields_an_unsupported_table():
    counts = np.zeros((2, 2, 2))
    counts[0, 0, 0] = 10
    assert not np.isfinite(joint_table_from_counts(counts)).all()


def test_intersection_flags_a_crossed_combination():
    """Two instruments implying disjoint sets means one of them is invalid."""
    from causal_grounding.balke_pearl import BalkePearlInterval
    crossed = intersect_intervals([BalkePearlInterval(0.4, 0.6, True, "sharp"),
                                   BalkePearlInterval(-0.6, -0.4, True, "sharp")])
    assert not crossed.feasible and crossed.status == "infeasible"
    ok = intersect_intervals([BalkePearlInterval(-0.5, 0.5, True, "sharp"),
                              BalkePearlInterval(-0.2, 0.8, True, "sharp")])
    assert (ok.lower, ok.upper) == (-0.2, 0.5)
