"""
Binary age and ideology covariates — the one definition of ``resp_age_bin`` and
``resp_polideo_bin``.

Each was previously written three times: in :mod:`.osrct`, in
:mod:`.manylabs_rct`, and in ``experiments/utils.py`` ``add_binary_covariates``.
All three copies were the same expression, so they now call
:func:`derive_age_bin` and :func:`derive_polideo_bin`.
"""

import pandas as pd

DEFAULT_AGE_BINARY_THRESHOLD = 21.0
DEFAULT_POLIDEO_BINARY_THRESHOLD = 3.0


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


def derive_polideo_bin(
    polideo: pd.Series, threshold: float = DEFAULT_POLIDEO_BINARY_THRESHOLD
) -> pd.Series:
    """Return ``1[polideo > threshold]`` as an ``int`` series.

    Args:
        polideo: Raw ``resp_polideo`` values. Non-numeric entries are coerced
            to NaN.
        threshold: Ideology values at or below this value fall in the lower bin.

    Returns:
        Integer series aligned with ``polideo``. A value equal to ``threshold``
        maps to 0, and a missing or non-numeric value maps to 0, because ``NaN >
        threshold`` is False.
    """
    return (pd.to_numeric(polideo, errors="coerce") > threshold).astype(int)


__all__ = [
    "DEFAULT_AGE_BINARY_THRESHOLD",
    "DEFAULT_POLIDEO_BINARY_THRESHOLD",
    "derive_age_bin",
    "derive_polideo_bin",
]
