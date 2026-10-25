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

GOLDEN_DIR = Path(__file__).parent / "golden"
RTOL = 1e-6
ATOL = 1e-9


def _round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame as it reads back from the snapshot's CSV format."""
    path = GOLDEN_DIR / "_round_trip.csv"
    try:
        frame.to_csv(path, index=False, float_format="%.10g")
        return pd.read_csv(path)
    finally:
        path.unlink(missing_ok=True)


def test_toy_example_matches_golden_snapshot():
    """Bounds and fired conditions equal the snapshot, to float tolerance."""
    sources, target, _ = simulate_toy_data(seed=0)
    bounds, conditions = naturalness_bounds(sources, target, n_draws=100, seed=0)

    for frame, name in [
        (bounds, "toy_bounds_seed0_draws100.csv"),
        (conditions, "toy_conditions_seed0_draws100.csv"),
    ]:
        expected = pd.read_csv(GOLDEN_DIR / name)
        pd.testing.assert_frame_equal(
            _round_trip(frame), expected, check_exact=False, rtol=RTOL, atol=ATOL
        )
