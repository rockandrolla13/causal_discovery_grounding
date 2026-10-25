"""
Tests for the data-layer definitions shared by loaders and experiments.

``resp_age_bin`` and the observed-gender cohort rule each have one definition
in ``causal_grounding.loaders``. These tests pin the age edge cases and check
that the experiment-side names are the same objects, not copies.
"""

import numpy as np
import pandas as pd

from causal_grounding.loaders import cohort, covariates
from causal_grounding.loaders import manylabs_rct, osrct


def test_age_bin_edge_cases() -> None:
    age = pd.Series([20.0, 21.0, 21.5, np.nan, "22", "n/a", 30])
    result = covariates.derive_age_bin(age)
    assert result.tolist() == [0, 0, 1, 0, 1, 0, 1]
    assert result.dtype == int


def test_age_bin_respects_threshold() -> None:
    age = pd.Series([18.0, 25.0])
    assert covariates.derive_age_bin(age, 18.0).tolist() == [0, 1]


def test_age_threshold_has_one_definition() -> None:
    from experiments import utils

    assert covariates.DEFAULT_AGE_BINARY_THRESHOLD == 21.0
    assert utils.DEFAULT_AGE_BINARY_THRESHOLD is covariates.DEFAULT_AGE_BINARY_THRESHOLD
    assert osrct.DEFAULT_AGE_BINARY_THRESHOLD is covariates.DEFAULT_AGE_BINARY_THRESHOLD
    assert manylabs_rct.DEFAULT_AGE_BINARY_THRESHOLD is covariates.DEFAULT_AGE_BINARY_THRESHOLD


def test_experiment_cohort_module_re_exports_loader_rule() -> None:
    from experiments import manylabs_cohort

    assert manylabs_cohort.restrict_to_observed_gender is cohort.restrict_to_observed_gender
    assert manylabs_cohort.OBSERVED_GENDER_COHORT_RULE == cohort.OBSERVED_GENDER_COHORT_RULE
    assert (
        manylabs_cohort.OBSERVED_GENDER_SOURCE_COLUMN
        == cohort.OBSERVED_GENDER_SOURCE_COLUMN
    )
