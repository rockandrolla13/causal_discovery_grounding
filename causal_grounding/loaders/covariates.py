"""
Binary age covariate — the one definition of ``resp_age_bin``.

Previously written three times: in :mod:`.osrct`, in :mod:`.manylabs_rct`, and
in ``experiments/utils.py`` ``add_binary_covariates``. All three were the same
expression, so they now call :func:`derive_age_bin`.
"""

import pandas as pd

DEFAULT_AGE_BINARY_THRESHOLD = 21.0


def derive_age_bin(
    age: pd.Series, threshold: float = DEFAULT_AGE_BINARY_THRESHOLD
) -> pd.Series:
    """Return ``1[age > threshold]`` as an ``int`` series.

    Args:
        age: Raw ``resp_age`` values. Non-numeric entries are coerced to NaN.
        threshold: Ages at or below this value fall in the lower bin.

    Returns:
        Integer series aligned with ``age``. An age equal to ``threshold`` maps
        to 0, and a missing or non-numeric age maps to 0, because ``NaN >
        threshold`` is False.
    """
    return (pd.to_numeric(age, errors="coerce") > threshold).astype(int)


__all__ = ["DEFAULT_AGE_BINARY_THRESHOLD", "derive_age_bin"]
