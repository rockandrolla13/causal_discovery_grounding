"""
Tests for the data-layer definitions shared by loaders and experiments.

``resp_age_bin`` and ``resp_polideo_bin`` each have one definition in
``causal_grounding.loaders``. These tests pin the binning edge cases and check
that the experiment-side names are the same objects, not copies.
"""

import numpy as np
import pandas as pd

from causal_grounding.loaders import covariates
from causal_grounding.loaders import manylabs_rct, osrct


def test_age_bin_edge_cases() -> None:
    age = pd.Series([20.0, 21.0, 21.5, np.nan, "22", "n/a", 30])
    result = covariates.derive_age_bin(age)
    assert result.tolist() == [0, 0, 1, 0, 1, 0, 1]
    assert result.dtype == int


def test_age_bin_respects_threshold() -> None:
    age = pd.Series([18.0, 25.0])
    assert covariates.derive_age_bin(age, 18.0).tolist() == [0, 1]


def test_polideo_bin_edge_cases() -> None:
    polideo = pd.Series([2.0, 3.0, 3.5, np.nan, "4", "n/a", 6])
    result = covariates.derive_polideo_bin(polideo)
    assert result.tolist() == [0, 0, 1, 0, 1, 0, 1]
    assert result.dtype == int


def test_thresholds_have_one_definition() -> None:
    from experiments import utils

    assert covariates.DEFAULT_AGE_BINARY_THRESHOLD == 21.0
    assert covariates.DEFAULT_POLIDEO_BINARY_THRESHOLD == 3.0
    assert osrct.DEFAULT_AGE_BINARY_THRESHOLD is covariates.DEFAULT_AGE_BINARY_THRESHOLD
    assert manylabs_rct.DEFAULT_AGE_BINARY_THRESHOLD is covariates.DEFAULT_AGE_BINARY_THRESHOLD
    for module in (utils, osrct, manylabs_rct):
        assert (
            module.DEFAULT_POLIDEO_BINARY_THRESHOLD
            is covariates.DEFAULT_POLIDEO_BINARY_THRESHOLD
        )


def test_experiment_utils_uses_loader_derivations() -> None:
    from experiments import utils

    assert utils.derive_age_bin is covariates.derive_age_bin
    assert utils.derive_polideo_bin is covariates.derive_polideo_bin
