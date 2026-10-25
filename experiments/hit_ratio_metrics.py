"""Data and metrics layer of the Exp25 reference-hit and sign-agreement analysis.

Loads an Exp25 backend leaf and its ``baseline_variants/gamma_<rule>/`` leaves,
annotates every interval with its RCT reference verdicts, and computes the sign
metrics, the common-support frontier and the selected-method summaries. A method
interval *hits* when it contains the held-out RCT CATE point estimate, and
*agrees in sign* when its verdict (positive or negative if it excludes zero,
else inconclusive) matches that of the RCT Wald 90% interval.

This module imports no experiment script. The orchestrator
(:mod:`experiments.exp25_hit_ratio_analysis`), the figures
(:mod:`experiments.exp25_hit_ratio_plots`) and the bootstrap
(:mod:`experiments.exp25_bootstrap_analysis`) all build on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from causal_grounding.paper_grids import PAPER_BETAS
from experiments.interval_stats import percentile_interval

INTERVALS_FILENAME = "mc_intervals.parquet"
EXP25_KEY_COLUMNS = (
    "replication",
    "seed",
    "bias_covariate",
    "beta",
    "study",
    "target_site",
    "resp_age_bin",
    "resp_gender_bin",
    "resp_polideo_bin",
)
EXP25_REQUIRED_COLUMNS = {
    *EXP25_KEY_COLUMNS,
    "method",
    "lower",
    "upper",
    "width",
    "is_valid",
    "exclusion_reason",
    "used_full_range_fallback",
    "support_clipped",
    "blearner_gamma",
    "lp_posterior_prob_optimal",
    "rct_cate",
    "rct_cate_ci90_lower",
    "rct_cate_ci90_upper",
    "rct_cate_available",
}
BLEARNER_METHOD = "blearner"
# Reported when present; a run without some of them simply has fewer rows.
SELECTED_WIDTH_HIT_BETAS = PAPER_BETAS
SELECTED_WIDTH_HIT_MODELS = (
    ("Naturalness", "naturalness"),
    ("B-Learner Γ=2.5", "blearner[fixed_2p5]"),
    ("B-Learner Γ=3", "blearner[fixed_3]"),
    ("B-Learner dgp_oracle", "blearner[dgp_oracle]"),
    ("CB-IPW", "cb_ipw"),
    ("CB-DR", "cb_dr"),
    ("Balke-Pearl", "balke_pearl"),
    ("Manski", "manski"),
)
# The sign table (``tab:manylabs-sign``) and its by-study split. Every listed
# model the run contains defines the common support, reference bounds included.
SIGN_TABLE_MODELS = (
    ("naturalness", "Naturalness"),
    ("manski", "Manski"),
    ("balke_pearl", "Balke--Pearl"),
    ("blearner[dgp_oracle]", "Oracle B-Learner"),
    ("cb_ipw", "CB-IPW"),
    ("cb_dr", "CB-DR"),
)
PRIMARY_GAMMA_LABEL = "dgp_oracle"
BASELINE_VARIANTS_DIRNAME = "baseline_variants"
GAMMA_LEAF_PREFIX = "gamma_"
MODEL_ORDER = (
    "naturalness",
    "blearner[dgp_oracle]",
    "blearner[oracle]",
    "blearner[exp_beta]",
    "cb_ipw",
    "cb_dr",
    "balke_pearl",
    "manski",
)
# Reference bounds scored beside the compared methods but kept out of the
# common-support definition, so they cannot shrink the compared stratum set.
REFERENCE_BOUND_MODELS = ("balke_pearl", "manski")
# Level of the width and hit-ratio intervals on the pooled and per-study
# frontiers; the sign-metric bootstrap keeps its own ``--confidence-level``.
FRONTIER_CONFIDENCE_LEVEL = 0.95
# Defaults of the replication-cluster bootstrap
# (:mod:`experiments.exp25_bootstrap_analysis`), shared with Exp31's width intervals.
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 20260808
SIGN_POSITIVE = "positive"
SIGN_NEGATIVE = "negative"
SIGN_INCONCLUSIVE = "inconclusive"
SIGN_ORDER = (SIGN_NEGATIVE, SIGN_INCONCLUSIVE, SIGN_POSITIVE)
NEGATIVE_INDEX = SIGN_ORDER.index(SIGN_NEGATIVE)
INCONCLUSIVE_INDEX = SIGN_ORDER.index(SIGN_INCONCLUSIVE)
POSITIVE_INDEX = SIGN_ORDER.index(SIGN_POSITIVE)
SIGN_COUNT_COLUMNS = (
    "n_model_decisive",
    "n_rct_decisive",
    "n_sign_correct",
    "n_sign_reversal",
    "n_sign_undetermined",
    "n_bound_abstained",
    "n_agree_inconclusive",
)
# Rate columns of :func:`sign_metrics_from_matrix`, each conditioned on the
# bound's own behaviour.
SIGN_METRIC_COLUMNS = (
    "decisiveness_rate",
    "sign_correct_rate",
    "sign_reversal_rate",
    "sign_undetermined_rate",
)
POINT_IDENTIFIED_WIDTH = 1e-9


def discover_exp25_gamma_leaves(
    primary_leaf: str | Path,
) -> list[tuple[str, Path]]:
    """Collect the primary Exp25 leaf and every B-Learner Gamma variant leaf.

    Args:
        primary_leaf: Backend leaf whose ``baseline_variants/`` directory holds
            one leaf per Gamma rule, including the primary ``dgp_oracle`` run.
            If ``gamma_dgp_oracle`` has no interval table, this leaf's own
            table is the primary one.

    Returns:
        Sorted ``(gamma_run, leaf_path)`` pairs. The primary leaf is labelled
        ``dgp_oracle``; variant leaves drop the ``gamma_`` directory prefix.

    Raises:
        FileNotFoundError: If the primary compact interval table is absent.
    """

    root = Path(primary_leaf).expanduser().resolve()
    variants_dir = root / BASELINE_VARIANTS_DIRNAME
    primary = variants_dir / f"{GAMMA_LEAF_PREFIX}{PRIMARY_GAMMA_LABEL}"
    if not (primary / INTERVALS_FILENAME).is_file():
        primary = root
    if not (primary / INTERVALS_FILENAME).is_file():
        raise FileNotFoundError(
            f"Primary Exp25 leaf lacks {INTERVALS_FILENAME}: {primary}"
        )
    leaves: list[tuple[str, Path]] = [(PRIMARY_GAMMA_LABEL, primary)]
    if variants_dir.is_dir():
        discovered: list[tuple[str, Path]] = []
        for leaf in variants_dir.iterdir():
            if not leaf.is_dir() or not (leaf / INTERVALS_FILENAME).is_file():
                continue
            name = leaf.name
            label = (
                name[len(GAMMA_LEAF_PREFIX):]
                if name.startswith(GAMMA_LEAF_PREFIX)
                else name
            )
            discovered.append((label, leaf))
        labels = [label for label, _ in discovered]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Discovered duplicate Gamma labels: {labels}")
        for label, leaf in discovered:
            if label == PRIMARY_GAMMA_LABEL:
                continue
            leaves.append((label, leaf))
    return sorted(leaves, key=lambda pair: pair[0])


def load_exp25_models(leaves: Sequence[tuple[str, Path]]) -> pd.DataFrame:
    """Load one model-long table combining Naturalness, CB, and Gamma variants.

    Non-B-Learner rows are identical across Gamma leaves (B-Learner-only replay
    copies them from the primary run), so they are taken once from the primary
    leaf and validated against every variant; each variant adds its B-Learner
    rows under a distinct ``model`` label. Methods the primary lacks, such as
    the ``identification_bounds`` leaf's Manski and Balke-Pearl rows, enter
    under their method name after checking that they cover exactly the primary
    strata and come from a single leaf.

    Args:
        leaves: Gamma labels and directories from
            :func:`discover_exp25_gamma_leaves`.

    Returns:
        Interval rows with added ``gamma_run`` and ``model`` columns.

    Raises:
        ValueError: If required columns are missing, keys are duplicated, or a
            variant leaf disagrees with the primary non-B-Learner rows.
    """

    tables: dict[str, pd.DataFrame] = {}
    for gamma_run, leaf in leaves:
        intervals = pd.read_parquet(leaf / INTERVALS_FILENAME)
        missing = sorted(EXP25_REQUIRED_COLUMNS.difference(intervals.columns))
        if missing:
            raise ValueError(
                f"{leaf / INTERVALS_FILENAME} is missing columns: {missing}"
            )
        intervals = intervals.copy()
        intervals["gamma_run"] = str(gamma_run)
        intervals["result_leaf"] = str(leaf)
        if intervals.duplicated([*EXP25_KEY_COLUMNS, "method"]).any():
            raise ValueError(
                f"Exp25 interval table has duplicate method rows: {leaf}"
            )
        tables[str(gamma_run)] = intervals
    if PRIMARY_GAMMA_LABEL not in tables:
        raise ValueError(
            f"Exp25 analysis requires the {PRIMARY_GAMMA_LABEL} primary leaf"
        )
    primary = tables[PRIMARY_GAMMA_LABEL]
    shared = primary.loc[~primary["method"].eq(BLEARNER_METHOD)].copy()
    shared["model"] = shared["method"].astype(str)
    frames = [shared]
    compare_keys = [*EXP25_KEY_COLUMNS, "method"]
    primary_methods = set(shared["method"].astype(str))
    primary_keys = shared.drop_duplicates(list(EXP25_KEY_COLUMNS))
    added_methods: dict[str, str] = {}
    for gamma_run, intervals in tables.items():
        non_blearner = intervals.loc[~intervals["method"].eq(BLEARNER_METHOD)]
        added = non_blearner.loc[
            ~non_blearner["method"].astype(str).isin(primary_methods)
        ].copy()
        for method, rows in added.groupby("method", sort=True):
            if str(method) in added_methods:
                raise ValueError(
                    f"Method {method} is supplied by both "
                    f"{added_methods[str(method)]} and {gamma_run}"
                )
            added_methods[str(method)] = str(gamma_run)
            covered = primary_keys.merge(
                rows.loc[:, list(EXP25_KEY_COLUMNS)],
                on=list(EXP25_KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
            )
            if len(covered) != len(primary_keys) or len(rows) != len(primary_keys):
                raise ValueError(
                    f"Leaf {gamma_run} method {method} does not cover exactly "
                    "the primary strata"
                )
        added["model"] = added["method"].astype(str)
        frames.append(added)
        variant_shared = non_blearner.loc[
            non_blearner["method"].astype(str).isin(primary_methods)
        ]
        merged = shared.merge(
            variant_shared.loc[:, [*compare_keys, "lower", "upper", "is_valid"]],
            on=compare_keys,
            how="inner",
            suffixes=("_primary", "_variant"),
            validate="one_to_one",
        )
        if len(merged) != len(variant_shared):
            raise ValueError(
                f"Gamma leaf {gamma_run} does not align with primary strata"
            )
        endpoints_agree = np.allclose(
            merged[["lower_primary", "upper_primary"]].to_numpy(dtype=float),
            merged[["lower_variant", "upper_variant"]].to_numpy(dtype=float),
            equal_nan=True,
        )
        if not endpoints_agree or not merged["is_valid_primary"].equals(
            merged["is_valid_variant"]
        ):
            raise ValueError(
                f"Gamma leaf {gamma_run} altered reused non-B-Learner rows"
            )
        blearner = intervals.loc[intervals["method"].eq(BLEARNER_METHOD)].copy()
        blearner["model"] = f"{BLEARNER_METHOD}[{gamma_run}]"
        frames.append(blearner)
    result = pd.concat(frames, ignore_index=True, sort=False)
    for column in ("is_valid", "used_full_range_fallback", "support_clipped",
                   "rct_cate_available"):
        result[column] = result[column].astype(bool)
    for column in ("beta", "lower", "upper", "width", "rct_cate",
                   "rct_cate_ci90_lower", "rct_cate_ci90_upper",
                   "blearner_gamma", "lp_posterior_prob_optimal"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def interval_sign(lower: pd.Series, upper: pd.Series) -> pd.Series:
    """Classify each interval as a signed or inconclusive causal verdict.

    Args:
        lower: Interval lower endpoints.
        upper: Interval upper endpoints.

    Returns:
        ``positive`` when the interval lies strictly above zero, ``negative``
        when it lies strictly below zero, and ``inconclusive`` when it spans
        zero. Rows with a missing endpoint receive a missing verdict.
    """

    lower_values = pd.to_numeric(lower, errors="coerce")
    upper_values = pd.to_numeric(upper, errors="coerce")
    verdict = pd.Series(SIGN_INCONCLUSIVE, index=lower_values.index, dtype=object)
    verdict.loc[lower_values > 0.0] = SIGN_POSITIVE
    verdict.loc[upper_values < 0.0] = SIGN_NEGATIVE
    verdict.loc[~np.isfinite(lower_values) | ~np.isfinite(upper_values)] = None
    return verdict


def annotate_reference_agreement(models: pd.DataFrame) -> pd.DataFrame:
    """Attach reference-hit and causal-sign agreement flags to every row.

    Args:
        models: Model-long table from :func:`load_exp25_models`.

    Returns:
        The same table with ``rct_sign``, ``model_sign``, ``rct_hit``,
        ``sign_match``, ``reference_eligible``, ``is_point_identified``, and a
        decomposition of the sign verdict into ``sign_recall`` (both intervals
        decisive and agreeing), ``sign_reversal`` (both decisive and
        disagreeing), ``false_decisive`` (method decisive while the RCT is
        not), and ``agree_inconclusive``.
    """

    annotated = models.copy()
    annotated["reference_eligible"] = annotated["rct_cate_available"] & np.isfinite(
        annotated["rct_cate"]
    )
    annotated["rct_sign"] = interval_sign(
        annotated["rct_cate_ci90_lower"],
        annotated["rct_cate_ci90_upper"],
    )
    annotated["model_sign"] = interval_sign(
        annotated["lower"],
        annotated["upper"],
    )
    usable = annotated["is_valid"] & annotated["reference_eligible"]
    hit = (
        (annotated["rct_cate"] >= annotated["lower"])
        & (annotated["rct_cate"] <= annotated["upper"])
    )
    annotated["rct_hit"] = np.where(usable, hit, False)
    # A point hit penalizes intervals narrower than the RCT's sampling error;
    # overlap with the RCT 90% interval asks only that the two are compatible.
    overlap = (
        (annotated["lower"] <= annotated["rct_cate_ci90_upper"])
        & (annotated["upper"] >= annotated["rct_cate_ci90_lower"])
    )
    annotated["rct_ci_overlap"] = np.where(usable, overlap, False)
    annotated["sign_match"] = np.where(
        usable & annotated["model_sign"].notna() & annotated["rct_sign"].notna(),
        annotated["model_sign"].eq(annotated["rct_sign"]),
        False,
    )
    annotated["is_point_identified"] = (
        annotated["is_valid"]
        & (annotated["width"].abs() <= POINT_IDENTIFIED_WIDTH)
    )
    model_decisive = usable & annotated["model_sign"].isin(
        [SIGN_POSITIVE, SIGN_NEGATIVE]
    )
    rct_decisive = usable & annotated["rct_sign"].isin(
        [SIGN_POSITIVE, SIGN_NEGATIVE]
    )
    annotated["model_decisive"] = model_decisive
    annotated["rct_decisive"] = rct_decisive
    annotated["sign_recall"] = model_decisive & rct_decisive & annotated["sign_match"]
    annotated["sign_reversal"] = (
        model_decisive & rct_decisive & ~annotated["sign_match"]
    )
    annotated["false_decisive"] = model_decisive & ~rct_decisive
    annotated["agree_inconclusive"] = usable & ~model_decisive & ~rct_decisive
    return annotated


def failures_as_manski(
    models: pd.DataFrame,
    *,
    draw_level: bool = False,
) -> pd.DataFrame:
    """Score every stratum, giving a failed stratum its own Manski interval.

    This is the scoring rule of the paper's ManyLabs tables and frontier
    figures. A stratum a method could not bound (Naturalness with no optimal LP
    draw, Balke-Pearl with no instrument, a CB-IPW full-range fallback) gets
    the Manski interval, the bound left without the method's assumptions. With
    ``draw_level`` (the sensitivity variant), the non-optimal posterior draws
    of each valid Naturalness stratum are also scored as Manski: the stored
    endpoints are optimal-draw means, so the result is
    ``p * endpoint + (1 - p) * manski_endpoint`` with ``p`` the posterior
    probability of an optimal solve.

    Args:
        models: Model-long rows from :func:`load_exp25_models`, including the
            ``manski`` rows of the ``identification_bounds`` leaf.
        draw_level: Whether to apply the draw-level replacement as well.

    Returns:
        A copy with ``lower``, ``upper`` and ``width`` re-scored and every row
        marked valid, ready for :func:`annotate_reference_agreement`.

    Raises:
        ValueError: If a stratum has no Manski row, or, with ``draw_level``, a
            valid Naturalness row lacks its optimal-solve probability.
    """

    keys = list(EXP25_KEY_COLUMNS)
    manski = models.loc[models["model"].astype(str).eq("manski"), [*keys, "lower", "upper"]]
    manski = manski.rename(columns={"lower": "manski_lower", "upper": "manski_upper"})
    scored = models.merge(manski, on=keys, how="left", validate="many_to_one")
    if scored["manski_lower"].isna().any():
        raise ValueError("Some strata have no Manski interval")
    failed = ~scored["is_valid"].astype(bool)
    scored.loc[failed, "lower"] = scored.loc[failed, "manski_lower"]
    scored.loc[failed, "upper"] = scored.loc[failed, "manski_upper"]
    if draw_level:
        naturalness = scored["model"].astype(str).eq("naturalness") & ~failed
        probability = scored.loc[naturalness, "lp_posterior_prob_optimal"]
        if probability.isna().any():
            raise ValueError("Valid Naturalness rows lack lp_posterior_prob_optimal")
        probability = probability.clip(0.0, 1.0)
        for side in ("lower", "upper"):
            scored.loc[naturalness, side] = (
                probability * scored.loc[naturalness, side]
                + (1.0 - probability) * scored.loc[naturalness, f"manski_{side}"]
            )
    scored["width"] = scored["upper"] - scored["lower"]
    scored["is_valid"] = True
    return scored.drop(columns=["manski_lower", "manski_upper"])


def _safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
) -> np.ndarray:
    """Divide elementwise, returning NaN wherever the denominator is not positive.

    Args:
        numerator: Dividend, scalar or array.
        denominator: Divisor, scalar or array broadcastable against the
            numerator.

    Returns:
        The elementwise ratio, with NaN in every position whose denominator is
        zero, negative, or missing.
    """

    top = np.asarray(numerator, dtype=float)
    bottom = np.asarray(denominator, dtype=float)
    shape = np.broadcast_shapes(top.shape, bottom.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.divide(
            np.broadcast_to(top, shape),
            np.broadcast_to(bottom, shape),
            out=np.full(shape, np.nan),
            where=np.broadcast_to(bottom > 0.0, shape),
        )


def sign_confusion_matrix(annotated: pd.DataFrame) -> np.ndarray:
    """Cross-tabulate RCT and model verdicts over usable strata.

    Every sign metric here is a function of this matrix alone, so the
    replication-cluster bootstrap in :mod:`experiments.exp25_bootstrap_analysis`
    resamples and adds per-replication matrices.

    Args:
        annotated: Output of :func:`annotate_reference_agreement`, or any
            subset of its rows.

    Returns:
        A ``(3, 3)`` count matrix whose rows are the RCT verdict and whose
        columns are the model verdict, both ordered by :data:`SIGN_ORDER`.
        Only valid, reference-eligible rows carrying both verdicts contribute.
    """

    usable = annotated.loc[
        annotated["is_valid"]
        & annotated["reference_eligible"]
        & annotated["rct_sign"].notna()
        & annotated["model_sign"].notna()
    ]
    positions = {value: index for index, value in enumerate(SIGN_ORDER)}
    matrix = np.zeros((len(SIGN_ORDER), len(SIGN_ORDER)), dtype=float)
    if usable.empty:
        return matrix
    rows = usable["rct_sign"].map(positions)
    columns = usable["model_sign"].map(positions)
    known = rows.notna() & columns.notna()
    np.add.at(
        matrix,
        (
            rows.loc[known].to_numpy(dtype=int),
            columns.loc[known].to_numpy(dtype=int),
        ),
        1.0,
    )
    return matrix


def sign_metrics_from_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Derive the sign-identification counts and rates from a confusion matrix.

    Writing ``+/-`` for an interval that excludes zero and ``?`` for one that
    contains it, five cells partition the usable strata:

    ======================== ================================================
    Cell                     Meaning
    ======================== ================================================
    ``sign_correct``         Bound ``+/-``, RCT ``+/-``, same side
    ``sign_reversal``        Bound ``+/-``, RCT ``+/-``, opposite side
    ``sign_undetermined``    Bound ``+/-``, RCT ``?``: no reference verdict
    ``bound_abstained``      Bound ``?``, RCT ``+/-``
    ``agree_inconclusive``   Bound ``?``, RCT ``?``
    ======================== ================================================

    ``decisiveness_rate`` is the decisive share of the usable strata, and the
    first three cells split the decisive calls into rates that sum to one. A
    sign reversal is a violated bound: the identified set excluded zero on the
    wrong side.

    Args:
        matrix: Confusion counts of shape ``(..., 3, 3)`` from
            :func:`sign_confusion_matrix`; leading axes are a batch, which the
            bootstrap uses to score many resamples at once.

    Returns:
        Mapping from column name to an array of shape ``matrix.shape[:-2]``,
        covering :data:`SIGN_COUNT_COLUMNS` and :data:`SIGN_METRIC_COLUMNS`.
    """

    counts = np.asarray(matrix, dtype=float)
    negative, inconclusive, positive = (
        NEGATIVE_INDEX,
        INCONCLUSIVE_INDEX,
        POSITIVE_INDEX,
    )
    n_correct = counts[..., negative, negative] + counts[..., positive, positive]
    n_reversal = counts[..., negative, positive] + counts[..., positive, negative]
    n_undetermined = (
        counts[..., inconclusive, negative] + counts[..., inconclusive, positive]
    )
    n_abstained = (
        counts[..., negative, inconclusive] + counts[..., positive, inconclusive]
    )
    n_agree_inconclusive = counts[..., inconclusive, inconclusive]
    n_model_decisive = n_correct + n_reversal + n_undetermined
    n_rct_decisive = n_correct + n_reversal + n_abstained
    n_usable = counts.sum(axis=(-2, -1))
    return {
        "n_model_decisive": n_model_decisive,
        "n_rct_decisive": n_rct_decisive,
        "n_sign_correct": n_correct,
        "n_sign_reversal": n_reversal,
        "n_sign_undetermined": n_undetermined,
        "n_bound_abstained": n_abstained,
        "n_agree_inconclusive": n_agree_inconclusive,
        # How often the bound determines a sign at all.
        "decisiveness_rate": _safe_divide(n_model_decisive, n_usable),
        # The three-way split of the decisive calls, which sums to one.
        "sign_correct_rate": _safe_divide(n_correct, n_model_decisive),
        "sign_reversal_rate": _safe_divide(n_reversal, n_model_decisive),
        "sign_undetermined_rate": _safe_divide(n_undetermined, n_model_decisive),
    }


def compute_sign_metrics(annotated: pd.DataFrame) -> dict[str, float]:
    """Score one group of annotated rows on every sign-identification metric.

    Args:
        annotated: Output of :func:`annotate_reference_agreement`, or any
            subset of its rows forming one cell of a summary.

    Returns:
        Scalar counts (:data:`SIGN_COUNT_COLUMNS`, as integers) and rates
        (:data:`SIGN_METRIC_COLUMNS`, as floats).
    """

    scored = sign_metrics_from_matrix(sign_confusion_matrix(annotated))
    return {
        key: int(value) if key in SIGN_COUNT_COLUMNS else float(value)
        for key, value in scored.items()
    }


def summarize_agreement(
    annotated: pd.DataFrame,
    *,
    grouping: Sequence[str],
) -> pd.DataFrame:
    """Aggregate validity, width, hit, and sign classification over a grouping.

    Sign-classification columns come from :func:`compute_sign_metrics` and are
    scored on the valid, reference-eligible rows of each cell, so they share a
    denominator with ``hit_ratio_conditional`` rather than with the
    failure-penalized ratios.

    Args:
        annotated: Output of :func:`annotate_reference_agreement`.
        grouping: Column names defining one summary row, excluding ``model``.

    Returns:
        One row per grouping level and model, ordered by :data:`MODEL_ORDER`,
        carrying validity and width summaries, both hit notions, and every
        column in :data:`SIGN_COUNT_COLUMNS` and :data:`SIGN_METRIC_COLUMNS`.
    """

    keys = [*grouping, "model"]
    rows: list[dict[str, object]] = []
    for values, frame in annotated.groupby(keys, sort=True, dropna=False):
        eligible = frame.loc[frame["reference_eligible"]]
        valid = frame.loc[frame["is_valid"]]
        valid_eligible = eligible.loc[eligible["is_valid"]]
        n_eligible = int(len(eligible))
        n_valid_eligible = int(len(valid_eligible))
        n_hits = int(valid_eligible["rct_hit"].sum())
        n_sign_match = int(valid_eligible["sign_match"].sum())
        rows.append(
            {
                **dict(zip(keys, values)),
                "n_strata": int(len(frame)),
                "n_valid": int(len(valid)),
                "valid_rate": float(len(valid) / len(frame)) if len(frame) else np.nan,
                "n_reference_eligible": n_eligible,
                "n_valid_eligible": n_valid_eligible,
                "n_hits": n_hits,
                "hit_ratio_conditional": (
                    float(n_hits / n_valid_eligible) if n_valid_eligible else np.nan
                ),
                "hit_ratio_failure_penalized": (
                    float(n_hits / n_eligible) if n_eligible else np.nan
                ),
                "n_ci_overlaps": int(valid_eligible["rct_ci_overlap"].sum()),
                "ci_overlap_ratio_conditional": (
                    float(
                        valid_eligible["rct_ci_overlap"].sum() / n_valid_eligible
                    )
                    if n_valid_eligible
                    else np.nan
                ),
                "ci_overlap_ratio_failure_penalized": (
                    float(valid_eligible["rct_ci_overlap"].sum() / n_eligible)
                    if n_eligible
                    else np.nan
                ),
                "n_sign_match": n_sign_match,
                "sign_match_conditional": (
                    float(n_sign_match / n_valid_eligible)
                    if n_valid_eligible
                    else np.nan
                ),
                "sign_match_failure_penalized": (
                    float(n_sign_match / n_eligible) if n_eligible else np.nan
                ),
                "mean_valid_width": (
                    float(valid["width"].mean()) if not valid.empty else np.nan
                ),
                "median_valid_width": (
                    float(valid["width"].median()) if not valid.empty else np.nan
                ),
                "always_inconclusive_baseline": (
                    float(
                        (~valid_eligible["rct_decisive"]).sum() / n_valid_eligible
                    )
                    if n_valid_eligible
                    else np.nan
                ),
                "sign_match_lift": (
                    float(
                        (
                            n_sign_match
                            - (~valid_eligible["rct_decisive"]).sum()
                        )
                        / n_valid_eligible
                    )
                    if n_valid_eligible
                    else np.nan
                ),
                **compute_sign_metrics(valid_eligible),
                "n_point_identified": int(frame["is_point_identified"].sum()),
                "n_full_range_fallback": int(frame["used_full_range_fallback"].sum()),
                "n_support_clipped": int(frame["support_clipped"].sum()),
                "mean_blearner_gamma": (
                    float(frame["blearner_gamma"].mean())
                    if frame["blearner_gamma"].notna().any()
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["model"] = pd.Categorical(
        summary["model"],
        categories=[
            *MODEL_ORDER,
            *sorted(set(summary["model"]).difference(MODEL_ORDER)),
        ],
        ordered=True,
    )
    return summary.sort_values([*grouping, "model"]).reset_index(drop=True)


def select_common_support(
    annotated: pd.DataFrame,
    *,
    key_columns: Sequence[str] = EXP25_KEY_COLUMNS,
    reference_models: Sequence[str] = REFERENCE_BOUND_MODELS,
) -> pd.DataFrame:
    """Return strata where every compared Exp25 model has a usable interval.

    The stratum set is fixed by the compared models alone; models in
    ``reference_models`` are kept on it wherever their own row is usable, so an
    often undefined reference bound (Balke-Pearl without an instrument) reports
    a smaller ``n_common`` instead of shrinking the comparison.

    Args:
        annotated: Model-long rows from :func:`annotate_reference_agreement`.
        key_columns: Columns uniquely identifying one requested stratum.
        reference_models: Models kept out of the support definition. An empty
            sequence makes every model in ``annotated`` define it, so all of
            them are scored on one identical stratum set.

    Returns:
        Usable model rows on the complete-case strata, retaining original
        columns and row order.
    """

    working = annotated.copy()
    working["model"] = working["model"].astype(str)
    usable = working["is_valid"] & working["reference_eligible"]
    compared = ~working["model"].isin(list(reference_models))
    n_models = int(working.loc[compared, "model"].nunique())
    usable_count = (usable & compared).groupby(
        [working[column] for column in key_columns], dropna=False
    ).transform("sum")
    return working.loc[usable & usable_count.eq(n_models)].copy()


def _frontier_bootstrap_intervals(
    common: pd.DataFrame,
    grouping: Sequence[str],
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[tuple, dict[str, float]]:
    """Bootstrap common-support mean width and hit ratio by replication.

    Resamples whole Monte Carlo replications (their strata share one
    observational draw), applying each resample to every model of a cell and
    re-pooling rows, so replications are weighted as in the point estimate.

    Args:
        common: Rows from :func:`select_common_support`.
        grouping: Columns defining one summary cell, excluding ``model``.
        bootstrap_replicates: Number of replication resamples per cell.
        confidence_level: Central percentile interval level in ``(0, 1)``.
        seed: Deterministic bootstrap seed.

    Returns:
        Percentile bounds keyed by ``(*grouping values, model)``.

    Raises:
        ValueError: If ``confidence_level`` is outside ``(0, 1)``.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    intervals: dict[tuple, dict[str, float]] = {}
    for values, cell in common.groupby(list(grouping), sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        replications = sorted(pd.unique(cell["replication"]))
        sampled = rng.integers(
            0, len(replications), size=(bootstrap_replicates, len(replications))
        )
        for model, frame in cell.groupby("model", sort=True):
            clusters = (
                frame.groupby("replication")
                .agg(n=("width", "size"), width=("width", "sum"), hits=("rct_hit", "sum"))
                .reindex(replications, fill_value=0)
            )
            counts = clusters["n"].to_numpy(dtype=float)[sampled].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                width = clusters["width"].to_numpy(dtype=float)[sampled].sum(axis=1) / counts
                hits = clusters["hits"].to_numpy(dtype=float)[sampled].sum(axis=1) / counts
            width_lower, width_upper = percentile_interval(width, confidence_level)
            hit_lower, hit_upper = percentile_interval(hits, confidence_level)
            intervals[(*values, str(model))] = {
                "mean_width_ci_lower": width_lower,
                "mean_width_ci_upper": width_upper,
                "hit_ratio_ci_lower": hit_lower,
                "hit_ratio_ci_upper": hit_upper,
                "ci_level": float(confidence_level),
            }
    return intervals


def _frontier_replication_spread(
    common: pd.DataFrame,
    grouping: Sequence[str],
) -> dict[tuple, dict[str, float]]:
    """Spread of common-support width and hit ratio across replications.

    The standard deviation (``ddof=1``) of each replication's own mean, unlike
    the bootstrap interval of the pooled mean; a model seen in a single
    replication gets NaN rather than zero.

    Args:
        common: Rows from :func:`select_common_support`.
        grouping: Columns defining one summary cell, excluding ``model``.

    Returns:
        ``mean_width_sd``, ``hit_ratio_sd`` and ``n_replications`` keyed by
        ``(*grouping values, model)``.
    """

    per_replication = (
        common.groupby([*grouping, "model", "replication"], sort=True, dropna=False)
        .agg(width=("width", "mean"), hits=("rct_hit", "mean"))
        .reset_index()
    )
    spread: dict[tuple, dict[str, float]] = {}
    for values, frame in per_replication.groupby(
        [*grouping, "model"], sort=True, dropna=False
    ):
        spread[tuple(values)] = {
            "mean_width_sd": float(frame["width"].std(ddof=1)),
            "hit_ratio_sd": float(frame["hits"].astype(float).std(ddof=1)),
            "n_replications": int(len(frame)),
        }
    return spread


def common_support_frontier(
    annotated: pd.DataFrame,
    *,
    grouping: Sequence[str] = ("beta",),
    key_columns: Sequence[str] = EXP25_KEY_COLUMNS,
    bootstrap_replicates: int = 0,
    confidence_level: float = FRONTIER_CONFIDENCE_LEVEL,
    seed: int = 20260808,
    replication_spread: bool = False,
    reference_models: Sequence[str] = REFERENCE_BOUND_MODELS,
) -> pd.DataFrame:
    """Score width and containment on strata that every method can bound.

    Per-method means over each method's own valid rows would favour a method
    that declines hard strata, so every model is summarized on the strata where
    all compared methods returned a usable interval, and validity is reported
    separately. ``hit_ratio_all_valid`` (containment over the method's own
    usable rows) sits beside ``hit_ratio``; a large gap between them warns that
    the intersection is a selected sample.

    Args:
        annotated: Output of :func:`annotate_reference_agreement`.
        grouping: Column names defining one summary row, excluding ``model``.
        key_columns: Columns uniquely identifying a requested stratum (default:
            the Exp25 key).
        bootstrap_replicates: Replication resamples for percentile intervals
            on ``mean_width`` and ``hit_ratio``; zero adds no interval columns.
        confidence_level: Central level of those intervals.
        seed: Deterministic bootstrap seed.
        replication_spread: Whether to add the across-replication standard
            deviation of ``mean_width`` and ``hit_ratio``; requires a
            ``replication`` column.
        reference_models: Models kept out of the support definition, passed to
            :func:`select_common_support`; naming every model but one scores
            all of them on the strata that one model can bound.

    Returns:
        One row per grouping level and model, ordered by :data:`MODEL_ORDER`,
        with ``n_common``, ``mean_width``, ``median_width``, ``hit_ratio``,
        ``decisiveness_rate``, ``sign_correct_rate``, ``valid_rate`` and
        ``hit_ratio_all_valid``. Bootstrap replicates add
        ``mean_width_ci_lower``/``_upper``, ``hit_ratio_ci_lower``/``_upper``
        and ``ci_level``; ``replication_spread`` adds ``mean_width_sd``,
        ``hit_ratio_sd`` and ``n_replications``.
    """

    working = annotated.copy()
    working["model"] = working["model"].astype(str)
    keys = list(key_columns)
    common = select_common_support(
        working, key_columns=keys, reference_models=reference_models)
    intervals = (
        _frontier_bootstrap_intervals(
            common,
            grouping,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed,
        )
        if bootstrap_replicates > 0
        else {}
    )
    spread = _frontier_replication_spread(common, grouping) if replication_spread else {}
    rows: list[dict[str, object]] = []
    for values, frame in common.groupby([*grouping, "model"], sort=True, dropna=False):
        condition = dict(zip([*grouping, "model"], values))
        own = working.loc[
            working["model"].eq(condition["model"])
            & np.logical_and.reduce(
                [working[column].eq(condition[column]) for column in grouping]
            )
        ]
        own_usable = own.loc[own["is_valid"] & own["reference_eligible"]]
        rows.append(
            {
                **condition,
                "n_common": int(len(frame)),
                "mean_width": float(frame["width"].mean()),
                "median_width": float(frame["width"].median()),
                "hit_ratio": float(frame["rct_hit"].mean()),
                "decisiveness_rate": float(frame["model_decisive"].mean()),
                "sign_correct_rate": (
                    float(
                        frame.loc[frame["model_decisive"], "sign_recall"].mean()
                    )
                    if frame["model_decisive"].any()
                    else np.nan
                ),
                "valid_rate": float(own["is_valid"].mean()) if len(own) else np.nan,
                # Containment over the method's own usable rows.
                "hit_ratio_all_valid": (
                    float(own_usable["rct_hit"].mean())
                    if not own_usable.empty
                    else np.nan
                ),
                **intervals.get(tuple(values), {}),
                **spread.get(tuple(values), {}),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["model"] = pd.Categorical(
        summary["model"],
        categories=[
            *MODEL_ORDER,
            *sorted(set(summary["model"]).difference(MODEL_ORDER)),
        ],
        ordered=True,
    )
    return summary.sort_values([*grouping, "model"]).reset_index(drop=True)


def summarize_selected_method_width_hit(
    common_rows: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
    betas: Sequence[float] = SELECTED_WIDTH_HIT_BETAS,
) -> pd.DataFrame:
    """Summarize selected methods by beta and study with clustered SEs.

    Point estimates pool bias-covariate patterns within each beta-study cell.
    Standard errors resample whole Monte Carlo replications, keeping every
    stratum and bias pattern from a sampled replication together. The same
    sampled replication indices are used for every method and cell.

    Args:
        common_rows: Complete-case rows from :func:`select_common_support`.
        bootstrap_replicates: Number of replication-cluster bootstrap draws.
            Zero retains point estimates and reports missing standard errors.
        seed: Deterministic bootstrap seed.
        betas: Beta values retained in the output when ``common_rows`` has them.

    Returns:
        One row per requested beta, study, and selected method with mean width,
        width SE, RCT point-containment ratio, and containment-ratio SE.

    Raises:
        ValueError: If required columns are absent or the bootstrap replicate
            count is negative.
    """

    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be nonnegative")
    required = {"replication", "study", "beta", "model", "width", "rct_hit"}
    missing = sorted(required.difference(common_rows.columns))
    if missing:
        raise ValueError(
            f"Selected width-hit summary is missing columns: {missing}"
        )

    columns = [
        "beta",
        "study",
        "method",
        "mean_width",
        "se_width",
        "mean_hit_ratio",
        "se_hit_ratio",
    ]
    if common_rows.empty:
        return pd.DataFrame(columns=columns)

    working = common_rows.copy()
    working["model"] = working["model"].astype(str)
    working["study"] = working["study"].astype(str)
    numeric_beta = pd.to_numeric(working["beta"], errors="coerce")
    requested_betas = tuple(
        beta for beta in map(float, betas) if np.isclose(numeric_beta, beta).any()
    )
    studies = sorted(pd.unique(working["study"]))
    replications = sorted(pd.unique(working["replication"]))
    rng = np.random.default_rng(seed)
    sampled = (
        rng.integers(
            0,
            len(replications),
            size=(bootstrap_replicates, len(replications)),
        )
        if bootstrap_replicates > 0
        else None
    )

    rows: list[dict[str, object]] = []
    for beta in requested_betas:
        for study in studies:
            cell = working.loc[
                np.isclose(numeric_beta, beta) & working["study"].eq(study)
            ]
            for label, model in SELECTED_WIDTH_HIT_MODELS:
                frame = cell.loc[cell["model"].eq(model)]
                if frame.empty:
                    rows.append(
                        {
                            "beta": beta,
                            "study": study,
                            "method": label,
                            "mean_width": np.nan,
                            "se_width": np.nan,
                            "mean_hit_ratio": np.nan,
                            "se_hit_ratio": np.nan,
                        }
                    )
                    continue
                clusters = (
                    frame.groupby("replication", sort=False)
                    .agg(
                        n=("width", "size"),
                        sum_width=("width", "sum"),
                        n_hits=("rct_hit", "sum"),
                    )
                    .reindex(replications, fill_value=0)
                )
                counts = clusters["n"].to_numpy(dtype=float)
                width_sums = clusters["sum_width"].to_numpy(dtype=float)
                hit_sums = clusters["n_hits"].to_numpy(dtype=float)
                total_count = counts.sum()
                width_se = np.nan
                hit_se = np.nan
                if sampled is not None and bootstrap_replicates > 1:
                    draw_counts = counts[sampled].sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        width_draws = (
                            width_sums[sampled].sum(axis=1) / draw_counts
                        )
                        hit_draws = hit_sums[sampled].sum(axis=1) / draw_counts
                    width_se = float(np.nanstd(width_draws, ddof=1))
                    hit_se = float(np.nanstd(hit_draws, ddof=1))
                rows.append(
                    {
                        "beta": beta,
                        "study": study,
                        "method": label,
                        "mean_width": float(width_sums.sum() / total_count),
                        "se_width": width_se,
                        "mean_hit_ratio": float(hit_sums.sum() / total_count),
                        "se_hit_ratio": hit_se,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_sign_confusion(
    annotated: pd.DataFrame,
    *,
    grouping: Sequence[str] = ("model", "beta"),
) -> pd.DataFrame:
    """Cross-tabulate RCT and model causal verdicts by requested grouping.

    Args:
        annotated: Output of :func:`annotate_reference_agreement`.
        grouping: Columns defining one confusion-matrix cell before the RCT
            and model verdicts are appended. Defaults to Exp25's model and
            beta grouping.

    Returns:
        Long counts of ``(rct_sign, model_sign)`` pairs over valid, reference
        eligible rows, with the share of each RCT verdict.
    """

    usable = annotated.loc[
        annotated["is_valid"]
        & annotated["reference_eligible"]
        & annotated["model_sign"].notna()
    ]
    keys = list(grouping)
    counts = (
        usable.groupby([*keys, "rct_sign", "model_sign"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = (
        counts.groupby([*keys, "rct_sign"], observed=True)["n"]
        .transform("sum")
    )
    counts["share_of_rct_verdict"] = counts["n"] / totals
    return counts.sort_values(
        [*keys, "rct_sign", "model_sign"]
    ).reset_index(drop=True)
