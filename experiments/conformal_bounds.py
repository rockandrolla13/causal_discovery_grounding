"""Split-conformal calibration of target-LP CATE bounds."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


MAX_BINARY_CATE_ENDPOINT_RESIDUAL = 2.0


def finite_sample_conformal_quantile(
    scores: Sequence[float],
    alpha: float,
    *,
    upper_bound: float = MAX_BINARY_CATE_ENDPOINT_RESIDUAL,
) -> Tuple[float, int]:
    """
    Compute split-conformal upper quantile with bounded infinity sentinel.

    Args:
        scores: Independent nonconformity scores.
        alpha: Desired miscoverage level.
        upper_bound: Known upper bound for every score.

    Returns:
        Conformal correction and one-based quantile rank.
    """
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("At least one conformal calibration score is required")
    if not np.isfinite(values).all():
        raise ValueError("Conformal calibration scores must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")

    values = np.clip(values, 0.0, upper_bound)
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    if rank > len(values):
        return float(upper_bound), rank
    return float(np.sort(values)[rank - 1]), rank


def add_full_range_fallback(
    lp_bounds: pd.DataFrame,
    target_idle: pd.DataFrame,
    config: Any,
    *,
    empty_status: str,
) -> pd.DataFrame:
    """
    Make binary-CATE interval scoring total while retaining LP failures.

    Args:
        lp_bounds: Base LP rows, possibly empty or nonoptimal.
        target_idle: Target idle rows defining observed covariate strata.
        config: Configuration exposing ordered LP covariates.
        empty_status: Status assigned when no base LP rows exist.

    Returns:
        One row per observed target stratum, with total scoring endpoints and
        explicit fallback diagnostics. Strata omitted by the LP probability
        estimator are retained with ``lp_stratum_not_returned`` status.
    """
    result = lp_bounds.copy()
    covariates = list(config.covariates)
    strata = (
        target_idle[covariates]
        .dropna()
        .drop_duplicates()
        .sort_values(covariates)
        .reset_index(drop=True)
    )
    missing_status = empty_status
    if not result.empty:
        missing_columns = [
            covariate for covariate in covariates if covariate not in result
        ]
        if missing_columns:
            raise ValueError(
                "LP bounds are missing target stratum columns: "
                f"{missing_columns}"
            )
        returned_strata = result[covariates].drop_duplicates()
        strata = (
            strata.merge(
                returned_strata,
                on=covariates,
                how="left",
                indicator=True,
                validate="one_to_one",
            )
            .loc[lambda frame: frame["_merge"].eq("left_only"), covariates]
            .reset_index(drop=True)
        )
        missing_status = "lp_stratum_not_returned"

    if not strata.empty:
        rows: List[Dict[str, Any]] = []
        for _, stratum in strata.iterrows():
            z_value = tuple(stratum[col] for col in covariates)
            row: Dict[str, Any] = {
                "z_value": z_value,
                "z_value_str": str(z_value),
                "lower": np.nan,
                "upper": np.nan,
                "status": missing_status,
            }
            for covariate, value in zip(covariates, z_value):
                row[covariate] = value
            rows.append(row)
        result = pd.concat(
            [result, pd.DataFrame(rows)],
            ignore_index=True,
            sort=False,
        )

    base_available = (
        result["status"].eq("optimal")
        & pd.to_numeric(result["lower"], errors="coerce").notna()
        & pd.to_numeric(result["upper"], errors="coerce").notna()
    )
    result["base_interval_available"] = base_available
    result["used_full_range_fallback"] = ~base_available
    result["score_lower"] = np.where(base_available, result["lower"], -1.0)
    result["score_upper"] = np.where(base_available, result["upper"], 1.0)
    result["interval_procedure_status"] = np.where(
        base_available,
        "base_optimal",
        "full_range_fallback",
    )
    return result


def conformalize_lp_bounds(
    lp_bounds: pd.DataFrame,
    correction: float,
) -> pd.DataFrame:
    """
    Expand final binary-CATE endpoints by a conformal correction.

    Args:
        lp_bounds: Base per-stratum LP output.
        correction: Non-negative endpoint correction on CATE scale.

    Returns:
        Bounds with unchanged base columns and conformal endpoint columns.
    """
    if correction < 0.0 or not np.isfinite(correction):
        raise ValueError("Endpoint conformal correction must be finite and non-negative")

    result = lp_bounds.copy()
    if result.empty:
        return result
    result["base_lower"] = result["lower"]
    result["base_upper"] = result["upper"]
    result["base_width"] = result["upper"] - result["lower"]
    result["base_status"] = result["status"]
    if "covers_rct_cate" in result.columns:
        result["base_covers_rct_cate"] = result["covers_rct_cate"]

    optimal = (
        result["status"].eq("optimal")
        & result["lower"].notna()
        & result["upper"].notna()
    )
    result["endpoint_conformal_q"] = float(correction)
    result["conformal_lower"] = np.where(
        optimal,
        np.maximum(-1.0, result["lower"] - correction),
        -1.0,
    )
    result["conformal_upper"] = np.where(
        optimal,
        np.minimum(1.0, result["upper"] + correction),
        1.0,
    )
    result["conformal_width"] = (
        result["conformal_upper"] - result["conformal_lower"]
    )
    result["conformal_status"] = np.where(
        optimal,
        "optimal_expanded",
        "base_nonoptimal_full_range",
    )
    if "rct_cate" in result.columns:
        result["covers_rct_cate_conformal"] = (
            result["rct_cate"].notna()
            & result["conformal_lower"].notna()
            & result["conformal_upper"].notna()
            & (result["conformal_lower"] <= result["rct_cate"])
            & (result["rct_cate"] <= result["conformal_upper"])
        )
    if "rct_cate_for_score" in result.columns:
        result["covers_rct_cate_score_label_conformal"] = (
            result["rct_cate_for_score"].notna()
            & result["conformal_lower"].notna()
            & result["conformal_upper"].notna()
            & (result["conformal_lower"] <= result["rct_cate_for_score"])
            & (result["rct_cate_for_score"] <= result["conformal_upper"])
        )
    return result

