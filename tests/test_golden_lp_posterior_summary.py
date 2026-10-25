"""``summarize_lp_posterior_bounds`` reproduces a saved snapshot on fixed draws.

The draw table is synthetic and seeded: four target strata, 40 posterior
draws, a mix of optimal, infeasible and missing-endpoint rows, and one stratum
with no optimal draw. After ``experiments/utils.py`` is split, update only the
import line below. Regenerate the snapshot only for an intended change of
results, from the repository root:

    python -c "from tests.test_golden_lp_posterior_summary import write_snapshot; write_snapshot()"
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.lp_summaries import summarize_lp_posterior_bounds

SNAPSHOT = Path(__file__).parent / "golden" / "lp_posterior_summary_seed0.csv"
FLOAT_FORMAT = "%.10g"
RTOL = 1e-6
ATOL = 1e-9
COVARIATES = ("resp_age_bin", "resp_gender_bin")
N_DRAWS = 40


def synthetic_draw_table(seed: int = 0) -> pd.DataFrame:
    """Return sample-level LP rows in the layout the Exp16 posterior loop builds."""
    rng = np.random.default_rng(seed)
    rows = []
    strata = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for draw in range(N_DRAWS):
        for index, z in enumerate(strata):
            centre = 0.1 * index - 0.15
            lower = centre - rng.uniform(0.05, 0.6)
            upper = centre + rng.uniform(0.05, 0.6)
            status = "optimal" if rng.uniform() < 0.7 else "infeasible"
            if z == (1, 1):
                status = "infeasible"
            if status != "optimal":
                lower, upper = np.nan, np.nan
            elif rng.uniform() < 0.05:
                upper = np.nan
            rows.append({
                "z_value": z,
                "z_value_str": str(z),
                "resp_age_bin": z[0],
                "resp_gender_bin": z[1],
                "lower": lower,
                "upper": upper,
                "status": status,
                "lp_posterior_sample_id": draw,
            })
    return pd.DataFrame(rows)


def write_snapshot() -> None:
    """Regenerate the snapshot from the current code."""
    summary = summarize_lp_posterior_bounds(synthetic_draw_table(), COVARIATES)
    summary.to_csv(SNAPSHOT, index=False, float_format=FLOAT_FORMAT)


def test_summarize_lp_posterior_bounds_matches_golden_snapshot(golden_round_trip):
    """Endpoint quantiles, envelopes and optimal-draw rates equal the snapshot."""
    summary = summarize_lp_posterior_bounds(synthetic_draw_table(), COVARIATES)
    expected = pd.read_csv(SNAPSHOT)
    pd.testing.assert_frame_equal(
        golden_round_trip(summary), expected, check_exact=False, rtol=RTOL, atol=ATOL
    )
