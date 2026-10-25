"""Outcome and covariate derivation for naturalness experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from causal_grounding.discretize import discretize_polideo
from causal_grounding.loaders.covariates import (
    derive_age_bin,
    derive_polideo_bin,
)
from causal_grounding.train_target_split import (
    get_available_sites,
)


DEFAULT_GENDER_BINARY_THRESHOLD = 0.5

USEFUL_COVARIATES = (
    "resp_age_bin",
    "resp_gender_bin",
    "resp_polideo_bin",
)


def add_naturalness_columns(
    osrct_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    age_bins: int,
    outcome: str = "dv",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Add binary outcome and discrete covariates used by naturalness experiments.

    Args:
        osrct_data: Observational OSRCT rows.
        rct_data: Randomized rows from the same study.
        age_bins: Number of quantile-based age categories.
        outcome: Binary source-outcome column.

    Returns:
        Prepared OSRCT rows, prepared RCT rows, and transformation metadata.

    Raises:
        ValueError: If the combined outcome support is not binary.
    """
    osrct = osrct_data.copy()
    rct = rct_data.copy()
    outcome_values = sorted(
        set(osrct[outcome].dropna().unique().tolist())
        | set(rct[outcome].dropna().unique().tolist())
    )
    if len(outcome_values) != 2:
        raise ValueError(
            f"Naturalness experiments require a binary source outcome; "
            f"got {outcome_values}"
        )

    positive_value = outcome_values[-1]
    combined_age = pd.concat([osrct["resp_age"], rct["resp_age"]], ignore_index=True)
    _, age_edges = pd.qcut(
        combined_age,
        q=age_bins,
        retbins=True,
        duplicates="drop",
    )
    age_edges[0] = -np.inf
    age_edges[-1] = np.inf
    age_labels = list(range(len(age_edges) - 1))

    for data in [osrct, rct]:
        data["dv_binary"] = (data[outcome] == positive_value).astype(int)
        data["dv_positive_value"] = positive_value
        data["resp_age_cat"] = pd.cut(
            data["resp_age"],
            bins=age_edges,
            labels=age_labels,
            include_lowest=True,
        ).astype(int)
        data["resp_gender_bin"] = data["resp_gender"].astype(int)
        data["resp_polideo_cat"] = discretize_polideo(
            data["resp_polideo"]
        ).astype(int)

    metadata = {
        "source_outcome": outcome,
        "binary_outcome": "dv_binary",
        "positive_value": float(positive_value),
        "age_edges": age_edges.tolist(),
        "age_bins": len(age_labels),
    }
    return osrct, rct, metadata


def add_binary_covariates(
    osrct_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    age_threshold: float,
    polideo_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Add threshold-based binary age and political-ideology covariates.

    Args:
        osrct_data: Prepared OSRCT rows containing ``resp_age`` and
            ``resp_polideo``.
        rct_data: Prepared RCT rows containing the same source columns.
        age_threshold: Age value assigned to the lower bin via ``<=``.
        polideo_threshold: Ideology value assigned to the lower bin via ``<=``.

    Returns:
        Copied input frames with binary columns and threshold metadata.
    """
    osrct = osrct_data.copy()
    rct = rct_data.copy()
    for data in [osrct, rct]:
        data["resp_age_bin"] = derive_age_bin(data["resp_age"], age_threshold)
        data["resp_polideo_bin"] = derive_polideo_bin(
            data["resp_polideo"], polideo_threshold
        )

    combined = pd.concat(
        [
            osrct[["resp_age_bin", "resp_polideo_bin"]],
            rct[["resp_age_bin", "resp_polideo_bin"]],
        ],
        ignore_index=True,
    )
    metadata = {
        "age_binary_threshold": float(age_threshold),
        "polideo_binary_threshold": float(polideo_threshold),
        "age_binary_rule": f"resp_age_bin = 1[resp_age > {age_threshold:g}]",
        "polideo_binary_rule": (
            f"resp_polideo_bin = 1[resp_polideo > {polideo_threshold:g}]"
        ),
        "age_binary_share_one": float(combined["resp_age_bin"].mean()),
        "polideo_binary_share_one": float(combined["resp_polideo_bin"].mean()),
    }
    return osrct, rct, metadata


def expand_target_sites(
    idle_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    requested_sites: Sequence[str],
    max_target_sites: Optional[int],
) -> List[str]:
    """
    Expand target-site requests against sites available in both regimes.

    Args:
        idle_data: Observational rows containing a ``site`` column.
        rct_data: Randomized rows containing a ``site`` column.
        requested_sites: Requested sites, optionally containing ``all``.
        max_target_sites: Optional cap applied after expansion.

    Returns:
        Sorted eligible target-site labels.

    Raises:
        ValueError: If a requested site is unavailable or no site is selected.
    """
    idle_sites = set(get_available_sites(idle_data, "site"))
    rct_sites = set(get_available_sites(rct_data, "site"))
    eligible = sorted(idle_sites & rct_sites)

    if any(str(site).lower() == "all" for site in requested_sites):
        sites = eligible
    else:
        missing = sorted(set(str(site) for site in requested_sites) - set(eligible))
        if missing:
            raise ValueError(
                "Requested target sites not available in both idle and RCT "
                f"data: {missing}. Eligible sites include: {eligible[:20]}"
            )
        sites = sorted(str(site) for site in requested_sites)

    if max_target_sites is not None:
        sites = sites[:max_target_sites]
    if not sites:
        raise ValueError("No eligible target sites were selected")
    return sites

