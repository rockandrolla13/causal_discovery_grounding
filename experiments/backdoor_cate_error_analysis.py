"""Back-door CATE correction baseline and error-analysis plots.

This module owns the deliberately non-naturalness baseline used by calibrated
Exp16. Experiment entrypoints should call the output
attachment and plotting functions here, while keeping their main naturalness
pipelines unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BACKDOOR_TRAINING_OUTPUT = "backdoor_cate_correction_training"
BACKDOOR_PREDICTION_OUTPUT = "backdoor_cate_correction_predictions"
BACKDOOR_SUMMARY_OUTPUT = "backdoor_cate_correction_summary"


def _build_empirical_rct_cate_labels(
    rct_rows: pd.DataFrame,
    *,
    covariates: Sequence[str],
    min_arm_count: int,
) -> pd.DataFrame:
    """Estimate supported empirical RCT CATE labels by stratum.

    Args:
        rct_rows: Randomized rows containing ``iv`` and ``dv_binary``.
        covariates: Covariates defining the CATE strata.
        min_arm_count: Required treated and control observations per stratum.

    Returns:
        One row per supported stratum with RCT CATE labels and support counts.
    """

    required = {"iv", "dv_binary", *covariates}
    missing = required - set(rct_rows.columns)
    if missing:
        raise ValueError(f"RCT rows are missing required columns: {sorted(missing)}")
    if int(min_arm_count) < 1:
        raise ValueError("min_arm_count must be positive.")

    rows: list[dict[str, Any]] = []
    for key, group in rct_rows.groupby(
        list(covariates),
        observed=True,
        dropna=False,
    ):
        key_tuple = key if isinstance(key, tuple) else (key,)
        treated = group.loc[group["iv"].eq(1), "dv_binary"]
        control = group.loc[group["iv"].eq(0), "dv_binary"]
        if len(treated) < int(min_arm_count) or len(control) < int(min_arm_count):
            continue
        mu1 = float(treated.mean())
        mu0 = float(control.mean())
        standard_error = float(
            np.sqrt(
                mu1 * (1.0 - mu1) / len(treated)
                + mu0 * (1.0 - mu0) / len(control)
            )
        )
        row = {
            covariate: value
            for covariate, value in zip(covariates, key_tuple)
        }
        row.update(
            {
                "rct_mu1": mu1,
                "rct_mu0": mu0,
                "rct_cate": mu1 - mu0,
                "rct_cate_se": standard_error,
                "rct_cate_ci90_lower": max(
                    -1.0,
                    mu1 - mu0 - 1.6448536269514722 * standard_error,
                ),
                "rct_cate_ci90_upper": min(
                    1.0,
                    mu1 - mu0 + 1.6448536269514722 * standard_error,
                ),
                "rct_n_treated": int(len(treated)),
                "rct_n_control": int(len(control)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_backdoor_adjusted_cate_labels(
    rows: pd.DataFrame,
    *,
    covariates: Sequence[str],
    treatment: str = "iv",
    outcome: str = "dv_binary",
    min_arm_count: int = 1,
) -> pd.DataFrame:
    """Estimate stratum-level back-door adjusted CATEs from observed rows.

    The adjustment set is the supplied binary stratum definition. Within each
    stratum, this returns the treated-minus-control difference in observed
    outcome means. Strata without at least ``min_arm_count`` treated and
    control rows are omitted so downstream correction models do not silently
    train on unsupported cells.

    Args:
        rows: Observational rows containing treatment, outcome, and covariates.
        covariates: Covariates defining the adjustment strata.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        min_arm_count: Minimum treated and control rows per stratum.

    Returns:
        One row per supported stratum with back-door arm means, arm counts, and
        the conditional CATE estimate.
    """

    required = {treatment, outcome, *covariates}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"Back-door CATE rows are missing required columns: {sorted(missing)}"
        )
    if int(min_arm_count) < 1:
        raise ValueError("min_arm_count must be positive.")

    output_columns = [
        *covariates,
        "backdoor_mu1",
        "backdoor_mu0",
        "backdoor_cate",
        "backdoor_n_treated",
        "backdoor_n_control",
    ]
    result_rows: list[dict[str, Any]] = []
    for key, group in rows.groupby(
        list(covariates),
        observed=True,
        dropna=False,
    ):
        key_tuple = key if isinstance(key, tuple) else (key,)
        treatment_values = pd.to_numeric(group[treatment], errors="coerce")
        outcome_values = pd.to_numeric(group[outcome], errors="coerce")
        treated = outcome_values.loc[treatment_values.eq(1)].dropna()
        control = outcome_values.loc[treatment_values.eq(0)].dropna()
        if len(treated) < int(min_arm_count) or len(control) < int(min_arm_count):
            continue
        mu1 = float(treated.mean())
        mu0 = float(control.mean())
        row = {
            covariate: value
            for covariate, value in zip(covariates, key_tuple)
        }
        row.update(
            {
                "backdoor_mu1": mu1,
                "backdoor_mu0": mu0,
                "backdoor_cate": mu1 - mu0,
                "backdoor_n_treated": int(len(treated)),
                "backdoor_n_control": int(len(control)),
            }
        )
        result_rows.append(row)
    return pd.DataFrame(result_rows, columns=output_columns)


def _split_backdoor_correction_regimes(
    site_data: pd.DataFrame,
    *,
    regime_col: str,
    disjoint_validation_ids: bool,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return idle and RCT rows for source-site correction training.

    Args:
        site_data: Source-site data containing idle and on regimes.
        regime_col: Column identifying the experimental regime.
        disjoint_validation_ids: Whether to drop RCT rows whose IDs also occur
            in idle rows.
        id_col: Participant identifier column for optional disjointness.

    Returns:
        Idle rows and RCT/on rows for one source site.
    """

    if regime_col not in site_data.columns:
        raise ValueError(f"Source site data must contain '{regime_col}'.")
    idle = site_data.loc[site_data[regime_col].astype(str).eq("idle")].copy()
    rct = site_data.loc[site_data[regime_col].astype(str).eq("on")].copy()
    if (
        disjoint_validation_ids
        and id_col in idle.columns
        and id_col in rct.columns
    ):
        idle_ids = set(idle[id_col].dropna())
        rct = rct.loc[~rct[id_col].isin(idle_ids)].copy()
    return idle, rct


def _fit_backdoor_cate_error_model(training: pd.DataFrame) -> dict[str, Any]:
    """Fit a one-feature linear model from back-door CATE to CATE error.

    Args:
        training: Source-site stratum table with ``backdoor_cate`` and
            ``backdoor_cate_error`` columns.

    Returns:
        Dictionary containing model status, coefficients, and training counts.
    """

    if training.empty or not {
        "backdoor_cate",
        "backdoor_cate_error",
    }.issubset(training.columns):
        return {
            "status": "no_training_rows",
            "intercept": np.nan,
            "slope": np.nan,
            "n_training_rows": 0,
        }

    x = pd.to_numeric(training["backdoor_cate"], errors="coerce")
    y = pd.to_numeric(training["backdoor_cate_error"], errors="coerce")
    valid = x.notna() & y.notna()
    n_training_rows = int(valid.sum())
    if n_training_rows == 0:
        return {
            "status": "no_training_rows",
            "intercept": np.nan,
            "slope": np.nan,
            "n_training_rows": 0,
        }
    x_valid = x.loc[valid].to_numpy(dtype=float)
    y_valid = y.loc[valid].to_numpy(dtype=float)
    if n_training_rows < 2 or pd.Series(x_valid).nunique(dropna=True) < 2:
        return {
            "status": "constant_error_model",
            "intercept": float(np.mean(y_valid)),
            "slope": 0.0,
            "n_training_rows": n_training_rows,
        }

    design = np.column_stack([np.ones(n_training_rows), x_valid])
    intercept, slope = np.linalg.lstsq(design, y_valid, rcond=None)[0]
    return {
        "status": "linear_error_model",
        "intercept": float(intercept),
        "slope": float(slope),
        "n_training_rows": n_training_rows,
    }


def _apply_backdoor_cate_error_model(
    frame: pd.DataFrame,
    model: Mapping[str, Any],
    *,
    correction_col: str,
    corrected_col: str,
) -> pd.DataFrame:
    """Attach predicted CATE correction terms to a back-door CATE table.

    Args:
        frame: Table containing ``backdoor_cate``.
        model: Fitted correction model from
            :func:`_fit_backdoor_cate_error_model`.
        correction_col: Name for the predicted additive correction column.
        corrected_col: Name for the corrected CATE column.

    Returns:
        Copied table with correction and corrected-CATE columns.
    """

    result = frame.copy()
    result[correction_col] = np.nan
    result[corrected_col] = np.nan
    if result.empty or pd.isna(model.get("intercept")) or pd.isna(model.get("slope")):
        return result

    cate = pd.to_numeric(result.get("backdoor_cate"), errors="coerce")
    valid = cate.notna()
    correction = float(model["intercept"]) + float(model["slope"]) * cate.loc[valid]
    result.loc[valid, correction_col] = correction.to_numpy(dtype=float)
    result.loc[valid, corrected_col] = np.clip(
        cate.loc[valid].to_numpy(dtype=float)
        + correction.to_numpy(dtype=float),
        -1.0,
        1.0,
    )
    return result


def _add_prediction_errors(
    frame: pd.DataFrame,
    *,
    truth_col: str,
    prediction_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Attach signed, absolute, and squared CATE prediction errors.

    Args:
        frame: Prediction table containing truth and prediction columns.
        truth_col: Column with the CATE reference.
        prediction_col: Column with the point prediction to score.
        prefix: Prefix for generated error columns.

    Returns:
        Copied table with ``*_error``, ``*_abs_error``, and
        ``*_squared_error`` columns.
    """

    result = frame.copy()
    if truth_col not in result.columns or prediction_col not in result.columns:
        result[f"{prefix}_error"] = np.nan
        result[f"{prefix}_abs_error"] = np.nan
        result[f"{prefix}_squared_error"] = np.nan
        return result

    truth = pd.to_numeric(result[truth_col], errors="coerce")
    prediction = pd.to_numeric(result[prediction_col], errors="coerce")
    valid = truth.notna() & prediction.notna()
    error = pd.Series(np.nan, index=result.index, dtype=float)
    error.loc[valid] = truth.loc[valid] - prediction.loc[valid]
    result[f"{prefix}_error"] = error
    result[f"{prefix}_abs_error"] = error.abs()
    result[f"{prefix}_squared_error"] = error.pow(2.0)
    return result


def _summarize_prediction_errors(
    frame: pd.DataFrame,
    *,
    truth_col: str,
    prediction_col: str,
    prefix: str,
) -> dict[str, Any]:
    """Summarize CATE prediction errors for one truth/prediction pair.

    Args:
        frame: Prediction table containing truth and prediction columns.
        truth_col: Column with the CATE reference.
        prediction_col: Column with the point prediction to score.
        prefix: Prefix for summary metric names.

    Returns:
        Dictionary with sample count, bias, MAE, and RMSE.
    """

    if truth_col not in frame.columns or prediction_col not in frame.columns:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_bias": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
        }
    truth = pd.to_numeric(frame[truth_col], errors="coerce")
    prediction = pd.to_numeric(frame[prediction_col], errors="coerce")
    valid = truth.notna() & prediction.notna()
    if not valid.any():
        return {
            f"{prefix}_n": 0,
            f"{prefix}_bias": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_rmse": np.nan,
        }
    error = prediction.loc[valid] - truth.loc[valid]
    return {
        f"{prefix}_n": int(valid.sum()),
        f"{prefix}_bias": float(error.mean()),
        f"{prefix}_mae": float(error.abs().mean()),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _merge_backdoor_reference_columns(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Merge available target truth/reference columns into predictions.

    Args:
        predictions: Target back-door CATE rows.
        reference: Target LP/scoring table carrying RCT and optional blackbox
            references.
        covariates: Stratum columns used as merge keys.

    Returns:
        Prediction rows outer-merged with one reference row per stratum.
    """

    if reference.empty:
        return predictions.copy()
    reference_columns = [
        *covariates,
        "n_target_rows",
        "rct_mu1",
        "rct_mu0",
        "rct_cate",
        "rct_cate_se",
        "rct_cate_ci90_lower",
        "rct_cate_ci90_upper",
        "rct_n_treated",
        "rct_n_control",
        "rct_cate_available",
        "blackbox_reference_cate",
        "blackbox_reference_mu0",
        "blackbox_reference_mu1",
    ]
    available_columns = [
        column for column in reference_columns if column in reference.columns
    ]
    if not set(covariates).issubset(available_columns):
        return predictions.copy()
    reference_subset = (
        reference[available_columns]
        .drop_duplicates(subset=list(covariates), keep="first")
        .reset_index(drop=True)
    )
    return predictions.merge(
        reference_subset,
        on=list(covariates),
        how="outer",
        validate="one_to_one",
    )


def build_backdoor_cate_correction_baseline(
    *,
    source_training_data: Mapping[str, pd.DataFrame],
    target_idle: pd.DataFrame,
    target_reference: pd.DataFrame,
    covariates: Sequence[str],
    rct_min_arm_count: int,
    backdoor_min_arm_count: int,
    treatment: str = "iv",
    outcome: str = "dv_binary",
    regime_col: str = "F",
    disjoint_validation_ids: bool = False,
    id_col: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train and score the one-feature back-door CATE correction baseline.

    The baseline pools all non-target source sites. For each source site it
    estimates a back-door adjusted CATE from idle rows, estimates the empirical
    gold CATE from the site's RCT/on rows, and fits a single linear predictor
    from ``backdoor_cate`` to ``rct_cate - backdoor_cate``. The fitted error
    predictor is then applied unchanged to the target idle back-door CATEs.

    Args:
        source_training_data: Non-target source-site data keyed by site label.
        target_idle: Target observational rows used for raw back-door CATEs.
        target_reference: Target scored table carrying held-out RCT labels and
            optional blackbox references.
        covariates: Adjustment and reporting strata.
        rct_min_arm_count: Minimum RCT treated/control rows for gold labels.
        backdoor_min_arm_count: Minimum idle treated/control rows for raw
            back-door CATEs.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        regime_col: Source-site regime indicator.
        disjoint_validation_ids: Whether to drop overlapping RCT IDs in source
            gold-label construction.
        id_col: Participant identifier column for optional disjointness.

    Returns:
        Training rows, target predictions, and one-row target/model summary.
    """

    training_rows: list[pd.DataFrame] = []
    for site, site_data in source_training_data.items():
        idle, rct = _split_backdoor_correction_regimes(
            site_data,
            regime_col=regime_col,
            disjoint_validation_ids=bool(disjoint_validation_ids),
            id_col=id_col,
        )
        if idle.empty or rct.empty:
            continue
        backdoor = estimate_backdoor_adjusted_cate_labels(
            idle,
            covariates=covariates,
            treatment=treatment,
            outcome=outcome,
            min_arm_count=int(backdoor_min_arm_count),
        )
        labels = _build_empirical_rct_cate_labels(
            rct,
            covariates=covariates,
            min_arm_count=int(rct_min_arm_count),
        )
        if backdoor.empty or labels.empty:
            continue
        merged = backdoor.merge(
            labels,
            on=list(covariates),
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            continue
        merged.insert(0, "source_site", str(site))
        training_rows.append(merged)

    training = (
        pd.concat(training_rows, ignore_index=True, sort=False)
        if training_rows
        else pd.DataFrame()
    )
    if not training.empty:
        training["backdoor_cate_error"] = (
            pd.to_numeric(training["rct_cate"], errors="coerce")
            - pd.to_numeric(training["backdoor_cate"], errors="coerce")
        )
    model = _fit_backdoor_cate_error_model(training)
    training = _apply_backdoor_cate_error_model(
        training,
        model,
        correction_col="backdoor_cate_fitted_correction_term",
        corrected_col="backdoor_cate_fitted_corrected",
    )
    training = _add_prediction_errors(
        training,
        truth_col="rct_cate",
        prediction_col="backdoor_cate",
        prefix="train_raw_backdoor",
    )
    training = _add_prediction_errors(
        training,
        truth_col="rct_cate",
        prediction_col="backdoor_cate_fitted_corrected",
        prefix="train_corrected_backdoor",
    )

    target_backdoor = estimate_backdoor_adjusted_cate_labels(
        target_idle,
        covariates=covariates,
        treatment=treatment,
        outcome=outcome,
        min_arm_count=int(backdoor_min_arm_count),
    )
    target_predictions = _merge_backdoor_reference_columns(
        target_backdoor,
        target_reference,
        covariates,
    )
    target_predictions = _apply_backdoor_cate_error_model(
        target_predictions,
        model,
        correction_col="backdoor_cate_correction_term",
        corrected_col="backdoor_cate_corrected",
    )
    target_predictions = _add_prediction_errors(
        target_predictions,
        truth_col="rct_cate",
        prediction_col="backdoor_cate",
        prefix="target_raw_backdoor_rct",
    )
    target_predictions = _add_prediction_errors(
        target_predictions,
        truth_col="rct_cate",
        prediction_col="backdoor_cate_corrected",
        prefix="target_corrected_backdoor_rct",
    )
    if "blackbox_reference_cate" in target_predictions.columns:
        target_predictions = _add_prediction_errors(
            target_predictions,
            truth_col="blackbox_reference_cate",
            prediction_col="backdoor_cate",
            prefix="target_raw_backdoor_blackbox",
        )
        target_predictions = _add_prediction_errors(
            target_predictions,
            truth_col="blackbox_reference_cate",
            prediction_col="backdoor_cate_corrected",
            prefix="target_corrected_backdoor_blackbox",
        )

    summary_row: dict[str, Any] = {
        "backdoor_cate_correction_model": "linear_error_on_backdoor_cate",
        "backdoor_cate_correction_status": model["status"],
        "backdoor_cate_correction_intercept": model["intercept"],
        "backdoor_cate_correction_slope": model["slope"],
        "backdoor_cate_correction_rct_min_arm_count": int(rct_min_arm_count),
        "backdoor_cate_correction_backdoor_min_arm_count": int(
            backdoor_min_arm_count
        ),
        "backdoor_cate_correction_n_training_rows": int(
            model["n_training_rows"]
        ),
        "backdoor_cate_correction_n_training_sites": int(
            training["source_site"].nunique() if "source_site" in training else 0
        ),
        "backdoor_cate_correction_n_target_rows": int(len(target_predictions)),
    }
    for key, value in _summarize_prediction_errors(
        training,
        truth_col="rct_cate",
        prediction_col="backdoor_cate",
        prefix="train_raw",
    ).items():
        summary_row[f"backdoor_cate_correction_{key}"] = value
    for key, value in _summarize_prediction_errors(
        training,
        truth_col="rct_cate",
        prediction_col="backdoor_cate_fitted_corrected",
        prefix="train_corrected",
    ).items():
        summary_row[f"backdoor_cate_correction_{key}"] = value
    for key, value in _summarize_prediction_errors(
        target_predictions,
        truth_col="rct_cate",
        prediction_col="backdoor_cate",
        prefix="target_raw_rct",
    ).items():
        summary_row[f"backdoor_cate_correction_{key}"] = value
    for key, value in _summarize_prediction_errors(
        target_predictions,
        truth_col="rct_cate",
        prediction_col="backdoor_cate_corrected",
        prefix="target_corrected_rct",
    ).items():
        summary_row[f"backdoor_cate_correction_{key}"] = value
    if "blackbox_reference_cate" in target_predictions.columns:
        for key, value in _summarize_prediction_errors(
            target_predictions,
            truth_col="blackbox_reference_cate",
            prediction_col="backdoor_cate",
            prefix="target_raw_blackbox",
        ).items():
            summary_row[f"backdoor_cate_correction_{key}"] = value
        for key, value in _summarize_prediction_errors(
            target_predictions,
            truth_col="blackbox_reference_cate",
            prediction_col="backdoor_cate_corrected",
            prefix="target_corrected_blackbox",
        ).items():
            summary_row[f"backdoor_cate_correction_{key}"] = value
    summary_row["backdoor_cate_correction_target_rct_mae_improvement"] = (
        summary_row["backdoor_cate_correction_target_raw_rct_mae"]
        - summary_row["backdoor_cate_correction_target_corrected_rct_mae"]
    )
    summary_row["backdoor_cate_correction_target_rct_rmse_improvement"] = (
        summary_row["backdoor_cate_correction_target_raw_rct_rmse"]
        - summary_row["backdoor_cate_correction_target_corrected_rct_rmse"]
    )
    if "backdoor_cate_correction_target_raw_blackbox_mae" in summary_row:
        summary_row["backdoor_cate_correction_target_blackbox_mae_improvement"] = (
            summary_row["backdoor_cate_correction_target_raw_blackbox_mae"]
            - summary_row[
                "backdoor_cate_correction_target_corrected_blackbox_mae"
            ]
        )

    return training, target_predictions, pd.DataFrame([summary_row])


def add_backdoor_cate_correction_outputs(
    outputs: MutableMapping[str, pd.DataFrame],
    *,
    source_training_data: Mapping[str, pd.DataFrame],
    target_idle: pd.DataFrame,
    target_reference: pd.DataFrame,
    covariates: Sequence[str],
    rct_min_arm_count: int,
    backdoor_min_arm_count: int,
    metadata: Mapping[str, Any] | None = None,
    treatment: str = "iv",
    outcome: str = "dv_binary",
    regime_col: str = "F",
    disjoint_validation_ids: bool = False,
    id_col: str = "id",
) -> MutableMapping[str, pd.DataFrame]:
    """Attach correction baseline tables and summary metrics to run outputs.

    Args:
        outputs: Experiment output dictionary to update in place.
        source_training_data: Non-target source-site data keyed by site label.
        target_idle: Target observational rows used for raw back-door CATEs.
        target_reference: Target scored table carrying held-out labels.
        covariates: Adjustment and reporting strata.
        rct_min_arm_count: Minimum RCT treated/control rows for gold labels.
        backdoor_min_arm_count: Minimum idle treated/control rows for raw CATEs.
        metadata: Optional columns to attach to baseline output tables.
        treatment: Binary treatment column.
        outcome: Binary outcome column.
        regime_col: Source-site regime indicator.
        disjoint_validation_ids: Whether to drop overlapping RCT IDs.
        id_col: Participant identifier column for optional disjointness.

    Returns:
        The same output mapping with three added baseline tables and extra
        ``run_summary`` columns.
    """

    training, predictions, summary = build_backdoor_cate_correction_baseline(
        source_training_data=source_training_data,
        target_idle=target_idle,
        target_reference=target_reference,
        covariates=covariates,
        rct_min_arm_count=int(rct_min_arm_count),
        backdoor_min_arm_count=int(backdoor_min_arm_count),
        treatment=treatment,
        outcome=outcome,
        regime_col=regime_col,
        disjoint_validation_ids=bool(disjoint_validation_ids),
        id_col=id_col,
    )
    for frame in (training, predictions, summary):
        if not frame.empty and metadata is not None:
            for key, value in metadata.items():
                frame[key] = value

    outputs[BACKDOOR_TRAINING_OUTPUT] = training
    outputs[BACKDOOR_PREDICTION_OUTPUT] = predictions
    outputs[BACKDOOR_SUMMARY_OUTPUT] = summary
    run_summary = outputs.get("run_summary", pd.DataFrame())
    if not run_summary.empty and not summary.empty:
        for column, value in summary.iloc[0].items():
            run_summary[column] = value
        outputs["run_summary"] = run_summary
    return outputs


def _run_identity_columns(frame: pd.DataFrame) -> list[str]:
    """Return metadata columns that identify runs in a plotted table.

    Args:
        frame: Output table to inspect.

    Returns:
        Ordered list of available run-identity columns.
    """

    candidates = [
        "analysis",
        "study",
        "pattern",
        "beta",
        "target_site",
        "stage1_backend",
        "lp_probability_backend",
    ]
    return [column for column in candidates if column in frame.columns]


def _stratum_label(frame: pd.DataFrame, covariates: Sequence[str]) -> pd.Series:
    """Build compact stratum labels from available covariate columns.

    Args:
        frame: Table containing stratum columns.
        covariates: Preferred covariate order.

    Returns:
        String labels like ``(0,1,0)``.
    """

    present = [column for column in covariates if column in frame.columns]
    if not present:
        return pd.Series([str(index) for index in frame.index], index=frame.index)
    return frame[present].apply(
        lambda row: "(" + ",".join(str(int(value)) for value in row) + ")",
        axis=1,
    )


def _group_label(frame: pd.DataFrame) -> pd.Series:
    """Build compact run labels from available metadata columns.

    Args:
        frame: Table with optional metadata columns.

    Returns:
        Per-row group labels.
    """

    columns = _run_identity_columns(frame)
    if not columns:
        return pd.Series(["target"] * len(frame), index=frame.index)
    return frame[columns].astype(str).agg("/".join, axis=1)


def _short_metadata_value(value: Any) -> str:
    """Format one metadata value for compact plot labels.

    Args:
        value: Metadata value from an experiment output table.

    Returns:
        Short display string suitable for axis tick labels.
    """

    text = str(value)
    replacements = {
        "hierarchical_dirichlet_draws": "HD draws",
        "hierarchical_dirichlet_posterior": "HD draws",
        "hierarchical_dirichlet": "HD",
        "xgboost": "xgb",
    }
    return replacements.get(text, text)


def _compact_group_label(frame: pd.DataFrame) -> pd.Series:
    """Build short run labels for plot axes.

    Args:
        frame: Table with optional experiment metadata columns.

    Returns:
        Per-row labels using study/pattern/beta/target first, with backend
        names included only when they vary within the plotted table.
    """

    if frame.empty:
        return pd.Series(dtype=str, index=frame.index)
    display_columns = [
        column
        for column in ["analysis", "study", "pattern", "beta", "target_site"]
        if column in frame.columns
    ]
    backend_columns = [
        column
        for column in ["stage1_backend", "lp_probability_backend"]
        if column in frame.columns and frame[column].nunique(dropna=False) > 1
    ]
    columns = [*display_columns, *backend_columns]
    if not columns:
        return _group_label(frame)

    def row_label(row: pd.Series) -> str:
        values = [
            _short_metadata_value(row[column])
            for column in columns
            if pd.notna(row[column])
        ]
        return "/".join(values) if values else "target"

    return frame.apply(row_label, axis=1)


def _overlay_panel_label(frame: pd.DataFrame) -> pd.Series:
    """Return panel labels for LP overlay plots.

    Args:
        frame: Merged prediction and interval table.

    Returns:
        One label per row, preferring study names when available.
    """

    if "study" in frame.columns:
        return frame["study"].fillna("study").astype(str)
    return _compact_group_label(frame)


def _ordered_panel_labels(labels: pd.Series) -> list[str]:
    """Return stable study-panel order for plot labels.

    Args:
        labels: Per-row panel labels.

    Returns:
        Ordered unique labels with common OSRCT studies first.
    """

    present = list(dict.fromkeys(labels.dropna().astype(str)))
    preferred_order = ["allowedforbidden", "gainloss", "reciprocity"]
    preferred = [label for label in preferred_order if label in present]
    remaining = [label for label in present if label not in preferred]
    return [*preferred, *remaining]


def _add_overlay_x_positions(
    frame: pd.DataFrame,
    covariates: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Add shared x positions for LP overlay panels.

    Args:
        frame: Plotting table with ``_panel`` and stratum columns.
        covariates: Preferred CATE-stratum covariate order.

    Returns:
        Plotting table with ``_plot_x`` plus ordered x tick labels.
    """

    plot_data = frame.copy()
    present = [column for column in covariates if column in plot_data.columns]
    if not present:
        plot_data["_plot_x"] = plot_data.groupby("_panel").cumcount()
        max_count = int(plot_data.groupby("_panel").size().max())
        return plot_data, [str(index) for index in range(max_count)]

    stratum_rows = (
        plot_data[present]
        .drop_duplicates()
        .sort_values(present)
        .reset_index(drop=True)
    )
    stratum_labels = _stratum_label(stratum_rows, present).tolist()
    stratum_keys = [
        tuple(row[column] for column in present)
        for _, row in stratum_rows.iterrows()
    ]
    stratum_to_x = {key: index for index, key in enumerate(stratum_keys)}
    keys = [
        tuple(row[column] for column in present)
        for _, row in plot_data.iterrows()
    ]
    base_x = pd.Series([stratum_to_x[key] for key in keys], index=plot_data.index)
    duplicate_group = [plot_data["_panel"], pd.Series(keys, index=plot_data.index)]
    duplicate_index = plot_data.groupby(duplicate_group).cumcount()
    duplicate_count = plot_data.groupby(duplicate_group)["_panel"].transform("size")
    jitter_width = 0.16
    jitter = (duplicate_index - (duplicate_count - 1.0) / 2.0) * jitter_width
    plot_data["_plot_x"] = base_x.astype(float) + jitter.astype(float)
    plot_data["_x_base"] = base_x.astype(int)
    return plot_data, stratum_labels


def _finalize_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    tight_rect: tuple[float, float, float, float] | None = None,
) -> None:
    """Save and close one matplotlib figure.

    Args:
        fig: Figure to save.
        output_path: PNG output path.
        tight_rect: Optional rectangle passed to ``tight_layout``.
    """

    if tight_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_rect)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _numeric_pair_frame(
    predictions: pd.DataFrame,
    *,
    truth_col: str = "rct_cate",
) -> pd.DataFrame:
    """Return prediction rows with numeric raw, corrected, and truth CATEs.

    Args:
        predictions: Baseline prediction table.
        truth_col: Reference CATE column.

    Returns:
        Filtered table with numeric helper columns.
    """

    required = {truth_col, "backdoor_cate", "backdoor_cate_corrected"}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    frame = predictions.copy()
    frame["_truth"] = pd.to_numeric(frame[truth_col], errors="coerce")
    frame["_raw"] = pd.to_numeric(frame["backdoor_cate"], errors="coerce")
    frame["_corrected"] = pd.to_numeric(
        frame["backdoor_cate_corrected"],
        errors="coerce",
    )
    return frame.dropna(subset=["_truth", "_raw", "_corrected"])


def _plot_raw_vs_corrected_scatter(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot raw and corrected CATE predictions against RCT CATE.

    Args:
        predictions: Baseline target prediction table.
        output_path: PNG output path.
    """

    frame = _numeric_pair_frame(predictions)
    if frame.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True, sharey=True)
    for ax, column, title, color in (
        (axes[0], "_raw", "Raw back-door CATE", "tab:orange"),
        (axes[1], "_corrected", "Corrected back-door CATE", "tab:green"),
    ):
        ax.scatter(frame["_truth"], frame[column], s=36, alpha=0.78, color=color)
        ax.plot([-1.0, 1.0], [-1.0, 1.0], color="black", linestyle="--", linewidth=1)
        ax.axhline(0.0, color="0.85", linewidth=0.8)
        ax.axvline(0.0, color="0.85", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Held-out RCT CATE")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
    axes[0].set_ylabel("Predicted CATE")
    _finalize_figure(fig, output_path)


def _plot_absolute_error_before_after(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot paired absolute errors before and after correction.

    Args:
        predictions: Baseline target prediction table.
        output_path: PNG output path.
    """

    frame = _numeric_pair_frame(predictions)
    if frame.empty:
        return
    raw_error = (frame["_truth"] - frame["_raw"]).abs().to_numpy(dtype=float)
    corrected_error = (
        frame["_truth"] - frame["_corrected"]
    ).abs().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for raw, corrected in zip(raw_error, corrected_error):
        color = "tab:green" if corrected <= raw else "tab:red"
        ax.plot([0, 1], [raw, corrected], color=color, alpha=0.45, linewidth=1.2)
    ax.scatter(np.zeros_like(raw_error), raw_error, color="tab:orange", s=28, zorder=3)
    ax.scatter(
        np.ones_like(corrected_error),
        corrected_error,
        color="tab:green",
        s=28,
        zorder=3,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw", "Corrected"])
    ax.set_ylabel("Absolute error vs RCT CATE")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    _finalize_figure(fig, output_path)


def _plot_source_error_model_fit(
    training: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot source-site CATE error against raw back-door CATE.

    Args:
        training: Baseline source training table.
        summary: Baseline summary table containing fitted coefficients.
        output_path: PNG output path.
    """

    required = {"backdoor_cate", "backdoor_cate_error"}
    if training.empty or not required.issubset(training.columns):
        return
    frame = training.copy()
    frame["_backdoor"] = pd.to_numeric(frame["backdoor_cate"], errors="coerce")
    frame["_error"] = pd.to_numeric(frame["backdoor_cate_error"], errors="coerce")
    frame = frame.dropna(subset=["_backdoor", "_error"])
    if frame.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.scatter(frame["_backdoor"], frame["_error"], color="tab:blue", alpha=0.70, s=34)
    x_min = min(-1.0, float(frame["_backdoor"].min()))
    x_max = max(1.0, float(frame["_backdoor"].max()))
    x_grid = np.linspace(x_min, x_max, 100)
    if not summary.empty:
        coefficient_rows = summary.dropna(
            subset=[
                "backdoor_cate_correction_intercept",
                "backdoor_cate_correction_slope",
            ],
        )
        for _, row in coefficient_rows.iterrows():
            intercept = float(row["backdoor_cate_correction_intercept"])
            slope = float(row["backdoor_cate_correction_slope"])
            ax.plot(
                x_grid,
                intercept + slope * x_grid,
                color="black",
                alpha=0.38,
                linewidth=1.4,
            )
    ax.axhline(0.0, color="0.65", linewidth=0.9)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.set_xlabel("Source idle back-door CATE")
    ax.set_ylabel("Source RCT CATE - back-door CATE")
    _finalize_figure(fig, output_path)


def _plot_mae_improvement_heatmap(
    predictions: pd.DataFrame,
    output_path: Path,
    covariates: Sequence[str],
) -> None:
    """Plot stratum-level absolute-error improvement as a heatmap.

    Args:
        predictions: Baseline target prediction table.
        output_path: PNG output path.
        covariates: Preferred stratum covariate order.
    """

    frame = _numeric_pair_frame(predictions)
    if frame.empty:
        return
    frame = frame.copy()
    frame["_improvement"] = (
        (frame["_truth"] - frame["_raw"]).abs()
        - (frame["_truth"] - frame["_corrected"]).abs()
    )
    frame["_group"] = _compact_group_label(frame)
    frame["_stratum"] = _stratum_label(frame, covariates)
    pivot = frame.pivot_table(
        index="_group",
        columns="_stratum",
        values="_improvement",
        aggfunc="mean",
    )
    if pivot.empty:
        return

    fig_width = max(7.2, 0.62 * len(pivot.columns) + 2.2)
    fig_height = max(3.4, 0.46 * len(pivot.index) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    vmax = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float))))
    vmax = max(vmax, 0.01)
    image = ax.imshow(pivot, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("CATE stratum")
    ax.set_ylabel("Run")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Abs-error improvement", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    _finalize_figure(fig, output_path)


def _plot_method_comparison_mae(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot raw versus corrected MAE from summary metrics.

    Args:
        summary: Baseline summary output table.
        output_path: PNG output path.
    """

    required = {
        "backdoor_cate_correction_target_raw_rct_mae",
        "backdoor_cate_correction_target_corrected_rct_mae",
    }
    if summary.empty or not required.issubset(summary.columns):
        return
    frame = summary.copy()
    frame["_group"] = _compact_group_label(frame)
    raw = pd.to_numeric(
        frame["backdoor_cate_correction_target_raw_rct_mae"],
        errors="coerce",
    )
    corrected = pd.to_numeric(
        frame["backdoor_cate_correction_target_corrected_rct_mae"],
        errors="coerce",
    )
    valid = raw.notna() & corrected.notna()
    if not valid.any():
        return
    frame = frame.loc[valid].reset_index(drop=True)
    raw = raw.loc[valid].reset_index(drop=True)
    corrected = corrected.loc[valid].reset_index(drop=True)
    y = np.arange(len(frame))
    height = 0.34

    fig_height = max(3.2, 0.44 * len(frame) + 1.4)
    fig, ax = plt.subplots(figsize=(7.6, fig_height))
    ax.barh(y - height / 2, raw, height, label="Raw", color="tab:orange")
    ax.barh(
        y + height / 2,
        corrected,
        height,
        label="Corrected",
        color="tab:green",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["_group"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MAE vs RCT CATE")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", color="0.90", linewidth=0.8)
    _finalize_figure(fig, output_path)


def _merge_predictions_with_bounds(
    predictions: pd.DataFrame,
    bounds: pd.DataFrame,
    covariates: Sequence[str],
) -> pd.DataFrame:
    """Merge correction predictions with LP intervals for overlay plots.

    Args:
        predictions: Baseline target prediction table.
        bounds: Target LP interval table.
        covariates: Stratum key columns.

    Returns:
        Merged plotting table.
    """

    if predictions.empty or bounds.empty:
        return pd.DataFrame()
    keys = [
        column
        for column in [*_run_identity_columns(predictions), *covariates]
        if column in predictions.columns and column in bounds.columns
    ]
    if not keys:
        return pd.DataFrame()
    interval_columns = [
        *keys,
        "conformal_lower",
        "conformal_upper",
        "rct_cate",
        "base_interval_available",
        "used_full_range_fallback",
    ]
    interval_columns = [column for column in interval_columns if column in bounds.columns]
    return predictions.merge(
        bounds[interval_columns].drop_duplicates(subset=keys),
        on=keys,
        how="inner",
        suffixes=("", "_interval"),
    )


def _plot_lp_interval_overlay(
    predictions: pd.DataFrame,
    bounds: pd.DataFrame,
    output_path: Path,
    covariates: Sequence[str],
) -> None:
    """Plot LP intervals with raw and corrected back-door CATE points.

    Args:
        predictions: Baseline target prediction table.
        bounds: Target LP interval table.
        output_path: PNG output path.
        covariates: Preferred stratum covariate order.
    """

    merged = _merge_predictions_with_bounds(predictions, bounds, covariates)
    required = {
        "conformal_lower",
        "conformal_upper",
        "rct_cate",
        "backdoor_cate",
        "backdoor_cate_corrected",
    }
    if merged.empty or not required.issubset(merged.columns):
        return
    frame = merged.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        return
    frame["_group"] = _compact_group_label(frame)
    frame["_panel"] = _overlay_panel_label(frame)
    frame["_stratum"] = _stratum_label(frame, covariates)
    frame, x_labels = _add_overlay_x_positions(frame, covariates)
    frame = frame.sort_values(["_panel", "_plot_x"]).reset_index(drop=True)
    panels = _ordered_panel_labels(frame["_panel"])
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.2, max(4.2, 3.1 * len(panels) - 0.4)),
        squeeze=False,
    )
    for ax, panel in zip(axes[:, 0], panels):
        rows = frame.loc[frame["_panel"].eq(panel)].sort_values("_plot_x")
        x = rows["_plot_x"].to_numpy(dtype=float)
        ax.vlines(
            x,
            rows["conformal_lower"],
            rows["conformal_upper"],
            color="tab:blue",
            linewidth=4.0,
            alpha=0.35,
            label="Endpoint-conformal LP interval",
            zorder=1,
        )
        ax.scatter(
            x - 0.16,
            rows["rct_cate"],
            color="black",
            s=30,
            label="RCT CATE",
            zorder=3,
        )
        ax.scatter(
            x,
            rows["backdoor_cate"],
            facecolors="none",
            edgecolors="tab:orange",
            s=44,
            linewidths=1.4,
            label="Raw back-door CATE",
            zorder=4,
        )
        ax.scatter(
            x + 0.16,
            rows["backdoor_cate_corrected"],
            color="tab:green",
            s=34,
            label="Corrected back-door CATE",
            zorder=5,
        )
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(-0.5, len(x_labels) - 0.5)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylabel("CATE")
        ax.set_title(str(panel), fontweight="bold")
        ax.grid(axis="y", color="0.90", linewidth=0.8)

    handles_by_label: dict[str, Any] = {}
    for ax in axes[:, 0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    if handles_by_label:
        fig.legend(
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
            fontsize=9,
        )
    fig.supxlabel("CATE stratum", y=0.040)
    _finalize_figure(fig, output_path, tight_rect=(0.0, 0.065, 1.0, 0.900))


def write_backdoor_cate_error_analysis_plots(
    results: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    covariates: Sequence[str],
) -> None:
    """Write all correction-baseline error-analysis plots.

    Args:
        results: Experiment results dictionary containing baseline output
            tables plus target LP intervals.
        output_dir: Folder where PNG plots should be written.
        covariates: Preferred CATE-stratum covariate order.
    """

    predictions = results.get(BACKDOOR_PREDICTION_OUTPUT, pd.DataFrame())
    training = results.get(BACKDOOR_TRAINING_OUTPUT, pd.DataFrame())
    summary = results.get(BACKDOOR_SUMMARY_OUTPUT, pd.DataFrame())
    bounds = results.get("target_lp_bounds", pd.DataFrame())
    if predictions.empty and training.empty and summary.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_raw_vs_corrected_scatter(
        predictions,
        output_dir / "raw_vs_corrected_rct_scatter.png",
    )
    _plot_absolute_error_before_after(
        predictions,
        output_dir / "absolute_error_before_after.png",
    )
    _plot_source_error_model_fit(
        training,
        summary,
        output_dir / "source_error_model_fit.png",
    )
    _plot_mae_improvement_heatmap(
        predictions,
        output_dir / "mae_improvement_heatmap.png",
        covariates,
    )
    _plot_method_comparison_mae(
        summary,
        output_dir / "method_comparison_mae.png",
    )
    _plot_lp_interval_overlay(
        predictions,
        bounds,
        output_dir / "lp_interval_overlay.png",
        covariates,
    )
