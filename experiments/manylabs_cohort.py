"""Shared cohort restrictions for ManyLabs OSRCT experiments.

Re-exports :mod:`causal_grounding.loaders.cohort`, which owns the rule, so the
benchmark data generator does not import experiment code.
"""

from causal_grounding.loaders.cohort import (
    OBSERVED_GENDER_COHORT_RULE,
    OBSERVED_GENDER_SOURCE_COLUMN,
    restrict_to_observed_gender,
)

__all__ = [
    "OBSERVED_GENDER_COHORT_RULE",
    "OBSERVED_GENDER_SOURCE_COLUMN",
    "restrict_to_observed_gender",
]
