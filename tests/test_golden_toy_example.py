"""The toy example's bounds and conditions match the saved snapshot.

A refactor that is meant to leave results unchanged must keep this test green.
Regenerate the snapshot only for an intended change of results:

    from examples.toy_naturalness import simulate_toy_data, naturalness_bounds
    sources, target, _ = simulate_toy_data(seed=0)
    bounds, conditions = naturalness_bounds(sources, target, n_draws=100, seed=0)
    bounds.to_csv("tests/golden/toy_bounds_seed0_draws100.csv",
                  index=False, float_format="%.10g")
    conditions.to_csv("tests/golden/toy_conditions_seed0_draws100.csv",
                      index=False, float_format="%.10g")
"""

from pathlib import Path

import pandas as pd

from examples.toy_naturalness import naturalness_bounds, simulate_toy_data
from tests.conftest import assert_frame_matches_golden

GOLDEN_DIR = Path(__file__).parent / "golden"


def test_toy_example_matches_golden_snapshot(golden_round_trip):
    """Bounds and fired conditions equal the snapshot, to float tolerance.

    ``golden_round_trip`` (tests/conftest.py) writes its CSV into pytest's
    ``tmp_path``, never into ``tests/golden``.
    """
    sources, target, _ = simulate_toy_data(seed=0)
    bounds, conditions = naturalness_bounds(sources, target, n_draws=100, seed=0)

    for frame, name in [
        (bounds, "toy_bounds_seed0_draws100.csv"),
        (conditions, "toy_conditions_seed0_draws100.csv"),
    ]:
        expected = pd.read_csv(GOLDEN_DIR / name)
        assert_frame_matches_golden(golden_round_trip(frame), expected)
