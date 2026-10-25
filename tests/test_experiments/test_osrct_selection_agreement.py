"""Agreement between the two implementations of the OSRCT selection model.

``OSRCTSampler`` in ``osrct_benchmark_v1.0/code/osrct.py`` builds the ManyLabs
confounded datasets (Experiment I). :func:`experiments.naturalness_comparison_data.
selection_probability` builds the Twins regimes (Experiment II). The second claims
to match the first, so that ``beta`` means the same confounding strength in both.

The passing tests pin where the two agree bit for bit. The ``xfail(strict=True)``
tests document where they differ; a strict xfail turns into a failure if the
behaviour changes, so any later unification has to update this file deliberately.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from experiments import naturalness_comparison_data as data

OSRCT_PATH = Path(__file__).resolve().parents[2] / "osrct_benchmark_v1.0" / "code" / "osrct.py"

BETAS = [-5.0, -2.0, -1.0, -0.25, 0.0, 0.5, 1.0, 2.0, 5.0]
INTERCEPTS = [0.0, 0.7]
SEEDS = [0, 1, 2, 3, 4]


def _load_osrct():
    """Load ``osrct.py`` by file path; its directory name is not importable."""
    spec = importlib.util.spec_from_file_location("_osrct_under_test", OSRCT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


osrct = _load_osrct()


def _osrct_probability(z: np.ndarray, beta: float, intercept: float) -> np.ndarray:
    """Selection probability from ``OSRCTSampler`` for one biasing covariate."""
    sampler = osrct.OSRCTSampler(["z"], {"z": beta}, intercept=intercept,
                                 standardize=True, random_seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sampler.selection_probability(pd.DataFrame({"z": z}))


def _frame(kind: str, seed: int) -> np.ndarray:
    """Synthetic biasing covariate of the requested kind."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 500))
    if kind == "normal":
        return rng.normal(3.0, 2.0, n)
    if kind == "lognormal":
        return rng.lognormal(0.0, 2.0, n)
    if kind == "binary_float":
        return rng.integers(0, 2, n).astype(float)
    if kind == "uint8_codes":  # the Exp28 covariate dtype
        return rng.integers(0, 2, n).astype(np.uint8)
    if kind == "ordinal":
        return rng.integers(0, 7, n).astype(float)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["normal", "lognormal", "binary_float", "uint8_codes", "ordinal"])
@pytest.mark.parametrize("seed", SEEDS)
def test_selection_probabilities_identical_on_generic_frames(kind: str, seed: int) -> None:
    """Both implementations give bit-identical probabilities on ordinary covariates."""
    z = _frame(kind, seed)
    for beta in BETAS:
        for intercept in INTERCEPTS:
            np.testing.assert_array_equal(
                data.selection_probability(z, beta, intercept),
                _osrct_probability(z, beta, intercept),
            )


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("n", [5, 10, 37, 333])
def test_selection_probabilities_identical_on_constant_integer_codes(value: int, n: int) -> None:
    """A site whose binary code is constant gives ``expit(intercept)`` in both."""
    z = np.full(n, value, dtype=np.uint8)
    for beta in BETAS:
        np.testing.assert_array_equal(data.selection_probability(z, beta, 0.7),
                                      _osrct_probability(z, beta, 0.7))


@pytest.mark.xfail(strict=True, reason=(
    "Constant float covariate whose mean rounds (for example 4.2 repeated 10 times): "
    "numpy's std is about 1e-15, not 0, so naturalness_comparison_data divides rounding "
    "noise by it and gives probabilities of expit(+-beta), while StandardScaler treats the "
    "column as constant and gives expit(intercept). Difference up to 0.65 at beta=5."))
def test_constant_float_covariate_agrees() -> None:
    z = np.full(10, 4.2)
    np.testing.assert_array_equal(data.selection_probability(z, 2.0), _osrct_probability(z, 2.0, 0.0))


@pytest.mark.xfail(strict=True, reason=(
    "Missing values: OSRCTSampler imputes the column mean; naturalness_comparison_data "
    "gets a NaN std, falls back to zeros and returns 0.5 for every row. Exp28 drops "
    "incomplete rows first, so its inputs never contain NaN."))
def test_missing_values_agree() -> None:
    z = np.array([1.0, np.nan, 3.0, 4.0])
    np.testing.assert_array_equal(data.selection_probability(z, 1.0), _osrct_probability(z, 1.0, 0.0))


def test_large_mean_tiny_spread_differs_only_by_rounding() -> None:
    """With mean 1e6 and spread 1e-3 the two stop being bit-identical (about 2e-14 apart).

    StandardScaler and ``numpy.std`` accumulate the variance differently, so
    exact equality is not asserted here; only the size of the gap is.
    """
    z = np.random.default_rng(0).normal(1e6, 1e-3, 200)
    np.testing.assert_allclose(data.selection_probability(z, 2.0), _osrct_probability(z, 2.0, 0.0),
                               rtol=0, atol=1e-12)


@pytest.mark.xfail(strict=True, reason=(
    "Same seed, different sampled rows: OSRCTSampler draws rng.binomial(1, p) while "
    "osrct_select draws rng.random(n) < p. Both are Bernoulli(p), so the distribution "
    "matches, but the streams are not bit-identical."))
def test_same_seed_keeps_same_rows() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=200)
    treatment = rng.integers(0, 2, 200)
    sampler = osrct.OSRCTSampler(["z"], {"z": 1.0}, random_seed=7)
    observed, _ = sampler.sample(pd.DataFrame({"z": z, "iv": treatment}), verbose=False)
    keep = data.osrct_select(z, treatment, 1.0, np.random.default_rng(7))
    np.testing.assert_array_equal(np.flatnonzero(keep), observed.index.to_numpy())
