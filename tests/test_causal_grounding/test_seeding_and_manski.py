"""Pinned seed schemes and the shared Manski bounds.

The pinned values were produced by the inline schemes these functions replaced.
A change to any of them changes the draws behind published results.
"""

import zlib

import numpy as np
import pandas as pd
import pytest

from causal_grounding import seeding
from causal_grounding import target_bounds
from causal_grounding.manski import manski_bounds, manski_interval


def test_seed_schemes_keep_their_pinned_values():
    """Every scheme returns the value its experiments were run with."""
    assert seeding.sha256_seed("rct-reference", 3) == 3924911482
    assert seeding.crc32_seed("twins", 1, 0.5) == 1633626763
    assert seeding.crc32_seed("anchoring1") == zlib.crc32(b"anchoring1") == 2860930317
    assert seeding.blake2b_uniform("0|1|site|7") == pytest.approx(0.33650160834350573, abs=0)
    assert seeding.posterior_sample_seeds(0, 3) == [3009447540, 4187506549, 4012810296]
    assert seeding.POSTERIOR_SAMPLE_SEED_SALT == 104729


def test_posterior_sample_seeds_match_the_target_program():
    """The named scheme is the one ``target_bounds`` draws its target PMFs with."""
    for seed in range(20):
        assert (seeding.posterior_sample_seeds(seed, 10)
                == target_bounds.target_probability_draw_seeds(seed, 10))


def test_manski_interval_has_width_one_and_handles_empty_cells():
    """A populated cell gives width one; an empty one gives [-1, 1]."""
    cell = pd.DataFrame({"iv": [0, 0, 1, 1, 1], "dv_binary": [1, 0, 1, 1, 0]})
    lower, upper = manski_interval(cell)
    assert (lower, upper) == pytest.approx((2 / 5 - 1 / 5 - 3 / 5, 2 / 5 - 1 / 5 + 2 / 5))
    assert upper - lower == pytest.approx(1.0)
    assert manski_interval(cell.iloc[:0]) == (-1., 1.)


def test_manski_bounds_cover_every_stratum():
    """Each requested stratum gets a row, with support judged on both arms."""
    idle = pd.DataFrame({"z": [0, 0, 0, 1], "iv": [0, 1, 1, 1], "dv_binary": [1, 0, 1, 1]})
    strata = pd.DataFrame({"z": [0, 1, 2]})
    result = manski_bounds(idle, strata, ("z",), 1)
    assert result.z.tolist() == [0, 1, 2]
    assert result.common_supported.tolist() == [True, False, False]
    assert result.loc[2, ["lower", "upper"]].tolist() == [-1., 1.]
    # Populated cells have width one; the empty cell keeps the full [-1, 1].
    np.testing.assert_allclose(result.upper - result.lower, [1.0, 1.0, 2.0])
