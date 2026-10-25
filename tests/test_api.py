"""The ``Naturalness`` facade reproduces the toy example's snapshot.

The golden CSVs are the independent reference: the facade and
``examples/toy_naturalness.py`` share :func:`posterior_target_bounds`, so each
is compared with the snapshot rather than with the other.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from causal_grounding import Naturalness, api
from causal_grounding.api import _as_site_mapping
from causal_grounding.loaders.contract import SiteBundle, SiteTable
from examples.toy_naturalness import simulate_toy_data

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


def _assert_matches_golden(bounds, conditions, tmp_path):
    """Assert bounds and conditions equal the toy example's snapshot."""
    for frame, name in [
        (bounds, "toy_bounds_seed0_draws100.csv"),
        (conditions, "toy_conditions_seed0_draws100.csv"),
    ]:
        expected = pd.read_csv(GOLDEN_DIR / name)
        pd.testing.assert_frame_equal(
            _round_trip(frame, tmp_path), expected, check_exact=False, rtol=RTOL, atol=ATOL
        )


def _bundle(sources, treatment="X", covariates=tuple(COVARIATES), drop=()):
    """Return the toy sources as a loader SiteBundle."""
    return SiteBundle(
        dataset="toy",
        sites={
            site: SiteTable(
                site_id=site,
                frame=frame.drop(columns=list(drop)),
                treatment=treatment,
                outcome="Y",
                covariates=tuple(covariates),
                unit_id="unit",
                source_positions=tuple(range(len(frame))),
            )
            for site, frame in sources.items()
        },
        covariates=tuple(covariates),
    )


@pytest.fixture(scope="module")
def toy_data():
    """Return the toy sources and target of seed 0."""
    sources, target, _ = simulate_toy_data(seed=0)
    return sources, target


@pytest.fixture(scope="module")
def fitted_model(toy_data):
    """Fit the facade once with its default seed."""
    sources, _ = toy_data
    return Naturalness(COVARIATES).fit(sources)


def test_facade_matches_golden_snapshot(toy_data, fitted_model, tmp_path):
    """Bounds and fired conditions equal the toy example's snapshot."""
    _, target = toy_data
    bounds, conditions, policy = fitted_model.get_bounds(
        target, n_draws=N_DRAWS, return_conditions=True, return_policy=True
    )
    _assert_matches_golden(bounds, conditions, tmp_path)
    assert policy.status == "selected"
    assert fitted_model.n_sources_ == 10


def test_threaded_draws_match_golden_snapshot(toy_data, fitted_model, tmp_path):
    """Solving the draws on threads gives the serial snapshot."""
    _, target = toy_data
    fitted_model.n_jobs = 4
    try:
        bounds, conditions = fitted_model.get_bounds(
            target, n_draws=N_DRAWS, return_conditions=True
        )
    finally:
        fitted_model.n_jobs = 1
    _assert_matches_golden(bounds, conditions, tmp_path)


def test_get_bounds_leaves_model_unchanged(toy_data, fitted_model):
    """``get_bounds`` sets no attribute on the fitted model."""
    _, target = toy_data
    before = set(vars(fitted_model))
    fitted_model.get_bounds(target, n_draws=5)
    assert set(vars(fitted_model)) == before
    assert not hasattr(fitted_model, "policy_result_")
    assert not hasattr(fitted_model, "conditions_")


def test_default_random_state_is_zero():
    """The facade is seeded by default."""
    assert Naturalness(COVARIATES).random_state == 0


def test_trigger_screen_follows_target_backend(toy_data, fitted_model, monkeypatch):
    """A model-backed target backend also feeds the trigger screen."""
    _, target = toy_data
    seen = []
    original = api.TargetNaturalnessPolicy.choose

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("trigger_quantities"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(api.TargetNaturalnessPolicy, "choose", spy)
    for target_backend, trigger_backend in [
        ("empirical", None),
        ("hierarchical_dirichlet", None),
        ("hierarchical_dirichlet", "empirical"),
    ]:
        monkeypatch.setattr(fitted_model, "target_backend", target_backend)
        monkeypatch.setattr(fitted_model, "trigger_backend", trigger_backend)
        fitted_model.get_bounds(target, n_draws=2)
    assert seen[0] is None
    assert seen[1] is not None
    assert seen[2] is None


def test_posterior_draw_backend_redraws_target_pmfs(toy_data, fitted_model, monkeypatch):
    """``hierarchical_dirichlet_draws`` draws one seeded target PMF per draw."""
    _, target = toy_data
    seeds = []
    original = api.estimate_target_observed_binary_quantities

    def spy(*args, probability_model_kwargs=None, **kwargs):
        seeds.append((probability_model_kwargs or {}).get("posterior_sample_seed"))
        return original(*args, probability_model_kwargs=probability_model_kwargs, **kwargs)

    monkeypatch.setattr(api, "estimate_target_observed_binary_quantities", spy)
    monkeypatch.setattr(fitted_model, "target_backend", "hierarchical_dirichlet_draws")
    first = fitted_model.get_bounds(target, n_draws=3)
    draw_seeds = [seed for seed in seeds if seed is not None]
    assert len(draw_seeds) == 3 and len(set(draw_seeds)) == 3
    monkeypatch.setattr(fitted_model, "n_jobs", 3)
    pd.testing.assert_frame_equal(fitted_model.get_bounds(target, n_draws=3), first)


def test_long_frame_and_site_bundle_give_the_dict(toy_data):
    """A long frame with a site column and a SiteBundle unpack to the same sites.

    The grounder is deterministic given its input mapping, so equal mappings
    give equal bounds; this checks the mappings without refitting.
    """
    sources, _ = toy_data
    long_frame = pd.concat(
        [frame.assign(site=site) for site, frame in sources.items()], ignore_index=True
    )
    for unpacked in (
        _as_site_mapping(long_frame, "site"),
        _as_site_mapping(_bundle(sources), None, covariates=COVARIATES),
    ):
        assert list(unpacked) == list(sources)
        for site, frame in sources.items():
            pd.testing.assert_frame_equal(unpacked[site], frame, check_exact=True)


def test_categorical_site_column_has_no_ghost_site(toy_data):
    """An unused category of a categorical site column is not a site."""
    sources, _ = toy_data
    long_frame = pd.concat(
        [frame.assign(site=site) for site, frame in sources.items()], ignore_index=True
    )
    long_frame["site"] = pd.Categorical(
        long_frame["site"], categories=[*sources, "unused"]
    )
    assert list(_as_site_mapping(long_frame, "site")) == list(sources)


def test_long_frame_requires_site_col(toy_data):
    """A long frame without ``site_col`` is rejected."""
    sources, _ = toy_data
    with pytest.raises(ValueError, match="site_col"):
        _as_site_mapping(pd.concat(sources.values()), None)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"drop": ["F"]}, "no regime column 'F'"),
        ({"treatment": "T"}, "treatment 'T'"),
        ({"covariates": ("Z1",)}, r"\['Z2'\] are not covariates"),
    ],
)
def test_site_bundle_must_match_the_model(toy_data, kwargs, match):
    """A bundle without the regime column or with other columns is rejected."""
    sources, _ = toy_data
    with pytest.raises(ValueError, match=match):
        Naturalness(COVARIATES).fit(_bundle(sources, **kwargs))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"covariates": "Z1"}, TypeError, "not the string"),
        ({"covariates": []}, ValueError, "at least one"),
        ({"n_draws": 0}, ValueError, "n_draws must be >= 1"),
        ({"n_draws": 2.5}, TypeError, "n_draws must be an int"),
        ({"n_jobs": 0}, ValueError, "n_jobs must be >= 1"),
        ({"stage1_backend": "nope"}, ValueError, "Unknown stage1_backend"),
        ({"target_backend": "nope"}, ValueError, "Unknown target_backend"),
        ({"trigger_backend": "nope"}, ValueError, "Unknown trigger_backend"),
        ({"grid_options": {"epsilons": [0.1]}}, ValueError, "grid_options must not set"),
        ({"grounder_options": {"random_seed": 1}}, ValueError, "grounder_options must not"),
        (
            {"target_options": {"posterior_sample_seed": 1}},
            ValueError,
            "target_options must not",
        ),
    ],
)
def test_invalid_parameters_are_rejected_before_fitting(toy_data, kwargs, error, match):
    """Invalid parameters raise a clear error before any fitting."""
    sources, _ = toy_data
    params = {"covariates": COVARIATES, **kwargs}
    with pytest.raises(error, match=match):
        Naturalness(**params).fit(sources)


def test_get_bounds_rejects_zero_draws(toy_data, fitted_model):
    """``get_bounds(n_draws=0)`` is rejected."""
    _, target = toy_data
    with pytest.raises(ValueError, match="n_draws must be >= 1"):
        fitted_model.get_bounds(target, n_draws=0)


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
    model = Naturalness(COVARIATES, epsilons=[1e-6]).fit(sources)
    bounds, conditions, policy = model.get_bounds(
        target, n_draws=10, return_conditions=True, return_policy=True
    )
    assert policy.status != "selected"
    assert list(bounds.columns) == [*COVARIATES, "lower", "upper", "p_optimal"]
    assert len(bounds) == 4
    assert bounds[["lower", "upper"]].isna().all().all()
    assert np.all(bounds["p_optimal"] == 0.0)
    assert conditions.empty
