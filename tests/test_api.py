"""The ``Naturalness`` facade reproduces the toy example and its snapshot."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from causal_grounding import Naturalness
from causal_grounding.api import _site_frames
from causal_grounding.loaders.contract import SiteBundle, SiteTable
from examples.toy_naturalness import naturalness_bounds, simulate_toy_data

GOLDEN_DIR = Path(__file__).parent / "golden"
# Same snapshot format and tolerances as tests/test_golden_toy_example.py.
RTOL = 1e-6
ATOL = 1e-9
COVARIATES = ["Z1", "Z2"]
N_DRAWS = 100


def _round_trip(frame: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    """Return the frame as it reads back from the snapshot's CSV format."""
    path = tmp_path / "round_trip.csv"
    frame.to_csv(path, index=False, float_format="%.10g")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def toy_data():
    """Return the toy sources and target of seed 0."""
    sources, target, _ = simulate_toy_data(seed=0)
    return sources, target


@pytest.fixture(scope="module")
def fitted_result(toy_data):
    """Fit the facade once and return its bounds and conditions."""
    sources, target = toy_data
    model = Naturalness(COVARIATES, random_state=0).fit(sources)
    bounds, conditions = model.get_bounds(target, n_draws=N_DRAWS, return_conditions=True)
    return model, bounds, conditions


def test_facade_matches_golden_snapshot(fitted_result, tmp_path):
    """Bounds and fired conditions equal the toy example's snapshot."""
    _, bounds, conditions = fitted_result
    for frame, name in [
        (bounds, "toy_bounds_seed0_draws100.csv"),
        (conditions, "toy_conditions_seed0_draws100.csv"),
    ]:
        expected = pd.read_csv(GOLDEN_DIR / name)
        pd.testing.assert_frame_equal(
            _round_trip(frame, tmp_path), expected, check_exact=False, rtol=RTOL, atol=ATOL
        )


def test_facade_matches_example_function(toy_data, fitted_result):
    """The facade returns exactly what ``naturalness_bounds`` returns."""
    sources, target = toy_data
    model, bounds, conditions = fitted_result
    expected_bounds, expected_conditions = naturalness_bounds(
        sources, target, n_draws=N_DRAWS, seed=0
    )
    pd.testing.assert_frame_equal(bounds, expected_bounds, check_exact=True)
    pd.testing.assert_frame_equal(conditions, expected_conditions, check_exact=True)
    pd.testing.assert_frame_equal(model.conditions_, conditions, check_exact=True)
    assert model.n_sources_ == len(sources)
    assert model.policy_result_.status == "selected"


def test_long_frame_and_site_bundle_give_the_dict(toy_data):
    """A long frame with a site column and a SiteBundle unpack to the same sites.

    The grounder is deterministic given its input mapping, so equal mappings
    give equal bounds; this checks the mappings without refitting.
    """
    sources, _ = toy_data
    long_frame = pd.concat(
        [frame.assign(site=site) for site, frame in sources.items()], ignore_index=True
    )
    bundle = SiteBundle(
        dataset="toy",
        sites={
            site: SiteTable(
                site_id=site,
                frame=frame,
                treatment="X",
                outcome="Y",
                covariates=tuple(COVARIATES),
                unit_id="unit",
                source_positions=tuple(range(len(frame))),
            )
            for site, frame in sources.items()
        },
        covariates=tuple(COVARIATES),
    )
    for unpacked in (_site_frames(long_frame, "site"), _site_frames(bundle, None)):
        assert list(unpacked) == list(sources)
        for site, frame in sources.items():
            pd.testing.assert_frame_equal(unpacked[site], frame, check_exact=True)


def test_long_frame_requires_site_col(toy_data):
    """A long frame without ``site_col`` is rejected."""
    sources, _ = toy_data
    with pytest.raises(ValueError, match="site_col"):
        _site_frames(pd.concat(sources.values()), None)


def test_get_bounds_before_fit_raises(toy_data):
    """``get_bounds`` needs a fitted model."""
    _, target = toy_data
    with pytest.raises(NotFittedError):
        Naturalness(COVARIATES).get_bounds(target)


def test_conditioning_covariates_not_supported(toy_data):
    """Grids with conditioning covariates are refused before any fitting."""
    sources, _ = toy_data
    model = Naturalness(COVARIATES, grid_options={"conditioning_covariates": ["Z2"]})
    with pytest.raises(NotImplementedError, match="conditioning_covariates"):
        model.fit(sources)


def test_abstains_when_nothing_fires(toy_data):
    """No condition fires at a tiny epsilon: NaN bounds with ``p_optimal`` 0."""
    sources, target = toy_data
    model = Naturalness(COVARIATES, epsilons=[1e-6], random_state=0).fit(sources)
    bounds, conditions = model.get_bounds(target, n_draws=10, return_conditions=True)
    assert model.policy_result_.status != "selected"
    assert list(bounds.columns) == [*COVARIATES, "lower", "upper", "p_optimal"]
    assert len(bounds) == 4
    assert bounds[["lower", "upper"]].isna().all().all()
    assert np.all(bounds["p_optimal"] == 0.0)
    assert conditions.empty
