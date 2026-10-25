"""Posterior LP bound summaries and output-frame accumulation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd


def summarize_lp_posterior_bounds(
    samples: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize sampled LP endpoint distributions by target query stratum.

    Args:
        samples: Sample-level LP detail rows.
        covariates: Ordered covariate columns defining query strata.

    Returns:
        Endpoint and width summaries plus posterior optimal-solve frequency.
    """
    if samples.empty or "lp_posterior_sample_id" not in samples.columns:
        return pd.DataFrame()
    group_cols = [
        col for col in ["z_value_str", *covariates] if col in samples.columns
    ]
    if not group_cols:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for group_key, group in samples.groupby(group_cols, dropna=False, sort=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        record = dict(zip(group_cols, key_tuple))
        if "z_value" in group.columns:
            record["z_value"] = group["z_value"].iloc[0]
        n_total = int(group["lp_posterior_sample_id"].nunique())
        optimal = group[
            group["status"].eq("optimal")
            & group["lower"].notna()
            & group["upper"].notna()
        ].copy()
        n_optimal = int(optimal["lp_posterior_sample_id"].nunique())
        record["lp_posterior_n_samples"] = n_total
        record["lp_posterior_n_optimal"] = n_optimal
        record["lp_posterior_prob_optimal"] = (
            float(n_optimal / n_total) if n_total else float("nan")
        )
        if optimal.empty:
            for column in [
                "lower_mean",
                "lower_q05",
                "lower_q50",
                "lower_q95",
                "upper_mean",
                "upper_q05",
                "upper_q50",
                "upper_q95",
                "width_mean",
                "width_q05",
                "width_q50",
                "width_q95",
                "posterior_envelope_lower",
                "posterior_envelope_upper",
                "posterior_envelope_width",
            ]:
                record[column] = float("nan")
            records.append(record)
            continue

        width = optimal["upper"].astype(float) - optimal["lower"].astype(float)
        for prefix, values in [
            ("lower", optimal["lower"].astype(float)),
            ("upper", optimal["upper"].astype(float)),
            ("width", width),
        ]:
            record[f"{prefix}_mean"] = float(values.mean())
            record[f"{prefix}_q05"] = float(values.quantile(0.05))
            record[f"{prefix}_q50"] = float(values.quantile(0.50))
            record[f"{prefix}_q95"] = float(values.quantile(0.95))
        record["posterior_envelope_lower"] = record["lower_q05"]
        record["posterior_envelope_upper"] = record["upper_q95"]
        record["posterior_envelope_width"] = (
            record["posterior_envelope_upper"] - record["posterior_envelope_lower"]
        )
        records.append(record)
    return pd.DataFrame(records)


def concat_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate output frames while tolerating an empty accumulator.

    Args:
        frames: Frames to concatenate.

    Returns:
        Concatenated frame, or an empty frame when no inputs exist.
    """
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

