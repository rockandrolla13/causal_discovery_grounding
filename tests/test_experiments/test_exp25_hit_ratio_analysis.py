"""Tests for the Exp25 hit-ratio and causal-sign agreement analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.exp25_hit_ratio_analysis as analysis
import experiments.hit_ratio_metrics as hit_metrics


STRATA = (
    # rct_cate, ci90_lower, ci90_upper -> RCT verdict
    (0.30, 0.10, 0.50),   # positive
    (-0.30, -0.50, -0.10),  # negative
    (0.02, -0.20, 0.24),  # inconclusive
    (0.15, 0.05, 0.25),   # positive
)
METHOD_BOUNDS = {
    # method -> (lower, upper) applied to every stratum
    "naturalness": (0.05, 0.45),
    "blearner": (-0.60, -0.20),
    "cb_ipw": (-0.90, 0.90),
    "cb_dr": (-1.00, 1.00),
}


def _interval_fixture(*, blearner_shift: float = 0.0) -> pd.DataFrame:
    """Build one replication of four strata with deterministic bounds.

    Args:
        blearner_shift: Additive shift applied to both B-Learner endpoints so
            Gamma variants differ only in their B-Learner rows.

    Returns:
        Method-long Exp25-shaped interval table.
    """

    rows: list[dict[str, object]] = []
    for index, (cate, ci_lower, ci_upper) in enumerate(STRATA):
        for method, (lower, upper) in METHOD_BOUNDS.items():
            if method == "blearner":
                lower, upper = lower + blearner_shift, upper + blearner_shift
            invalid = method == "naturalness" and index == 3
            rows.append(
                {
                    "replication": 0,
                    "seed": 42,
                    "bias_covariate": "age",
                    "beta": 1.0,
                    "study": "gainloss",
                    "target_site": "pi",
                    "resp_age_bin": index % 2,
                    "resp_gender_bin": index // 2,
                    "resp_polideo_bin": 0,
                    "method": method,
                    "lower": np.nan if invalid else lower,
                    "upper": np.nan if invalid else upper,
                    "width": np.nan if invalid else upper - lower,
                    "is_valid": not invalid,
                    "exclusion_reason": "no_optimal_lp_draw" if invalid else "",
                    "used_full_range_fallback": False,
                    "support_clipped": False,
                    "n_target_idle_rows": 40,
                    "blearner_gamma": 2.0 if method == "blearner" else np.nan,
                    "lp_posterior_prob_optimal": (
                        0.0 if invalid else 0.9 if method == "naturalness" else np.nan
                    ),
                    "rct_cate": cate,
                    "rct_cate_ci90_lower": ci_lower,
                    "rct_cate_ci90_upper": ci_upper,
                    "rct_cate_available": True,
                }
            )
    return pd.DataFrame(rows)


def _write_leaf(path: Path, frame: pd.DataFrame) -> Path:
    """Write one interval table into a result leaf directory."""

    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path / hit_metrics.INTERVALS_FILENAME, index=False)
    return path


def test_interval_sign_classifies_verdicts() -> None:
    """Signed, straddling, and missing intervals map to distinct verdicts."""

    lower = pd.Series([0.1, -0.5, -0.2, np.nan])
    upper = pd.Series([0.4, -0.1, 0.3, 0.5])
    verdicts = hit_metrics.interval_sign(lower, upper)
    assert list(verdicts[:3]) == [
        hit_metrics.SIGN_POSITIVE,
        hit_metrics.SIGN_NEGATIVE,
        hit_metrics.SIGN_INCONCLUSIVE,
    ]
    assert verdicts.isna().iloc[3]


def test_annotate_reference_agreement_flags_hits_and_signs() -> None:
    """Hits, sign matches, and the sign decomposition follow the fixture."""

    models = _interval_fixture().assign(
        gamma_run="dgp_oracle",
        model=lambda frame: np.where(
            frame["method"].eq("blearner"),
            "blearner[dgp_oracle]",
            frame["method"],
        ),
    )
    annotated = hit_metrics.annotate_reference_agreement(models)
    naturalness = annotated.loc[annotated["method"].eq("naturalness")]
    # Bounds [0.05, 0.45] cover the positive stratum 0 but not the negative
    # stratum 1 nor the inconclusive stratum 2; stratum 3 is invalid.
    assert list(naturalness["rct_hit"]) == [True, False, False, False]
    assert list(naturalness["model_sign"][:3]) == [hit_metrics.SIGN_POSITIVE] * 3
    assert list(naturalness["sign_match"]) == [True, False, False, False]
    assert list(naturalness["sign_recall"]) == [True, False, False, False]
    assert list(naturalness["sign_reversal"]) == [False, True, False, False]
    assert list(naturalness["false_decisive"]) == [False, False, True, False]

    cb_dr = annotated.loc[annotated["method"].eq("cb_dr")]
    assert cb_dr["rct_hit"].all()
    assert not cb_dr["model_decisive"].any()
    assert list(cb_dr["sign_match"]) == [False, False, True, False]


def test_summarize_agreement_counts_failures_as_misses() -> None:
    """Failure-penalized ratios use every eligible stratum as denominator."""

    models = _interval_fixture().assign(
        gamma_run="dgp_oracle",
        model=lambda frame: np.where(
            frame["method"].eq("blearner"),
            "blearner[dgp_oracle]",
            frame["method"],
        ),
    )
    annotated = hit_metrics.annotate_reference_agreement(models)
    summary = hit_metrics.summarize_agreement(annotated, grouping=("beta",))
    naturalness = summary.loc[summary["model"].eq("naturalness")].iloc[0]
    assert naturalness["n_strata"] == 4
    assert naturalness["n_valid"] == 3
    assert naturalness["n_hits"] == 1
    assert naturalness["hit_ratio_conditional"] == pytest.approx(1.0 / 3.0)
    assert naturalness["hit_ratio_failure_penalized"] == pytest.approx(0.25)
    assert naturalness["n_rct_decisive"] == 2
    # Every sign rate is conditioned on the bound's own decisive calls, of
    # which there are three: one correct, one reversed, one the RCT cannot
    # adjudicate.
    assert naturalness["sign_reversal_rate"] == pytest.approx(1.0 / 3.0)
    assert naturalness["sign_undetermined_rate"] == pytest.approx(1.0 / 3.0)


def test_selected_width_hit_table_breaks_out_beta_and_study() -> None:
    """Selected methods should receive paired cluster SEs in requested cells."""

    rows: list[dict[str, object]] = []
    for replication in (0, 1):
        for beta in (0.25, 0.5, 1.0):
            for study_index, study in enumerate(("gainloss", "reciprocity")):
                for model_index, (_, model) in enumerate(
                    hit_metrics.SELECTED_WIDTH_HIT_MODELS
                ):
                    rows.append(
                        {
                            "replication": replication,
                            "beta": beta,
                            "study": study,
                            "model": model,
                            "width": (
                                beta
                                + 0.1 * model_index
                                + 0.2 * replication
                                + 0.01 * study_index
                            ),
                            "rct_hit": (replication + model_index) % 2 == 0,
                        }
                    )
    common = pd.DataFrame(rows)

    first = hit_metrics.summarize_selected_method_width_hit(
        common, bootstrap_replicates=100, seed=19
    )
    second = hit_metrics.summarize_selected_method_width_hit(
        common, bootstrap_replicates=100, seed=19
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [
        "beta",
        "study",
        "method",
        "mean_width",
        "se_width",
        "mean_hit_ratio",
        "se_hit_ratio",
    ]
    assert set(first["beta"]) == {0.5, 1.0}
    assert len(first) == 2 * 2 * len(hit_metrics.SELECTED_WIDTH_HIT_MODELS)
    selected = first.loc[
        first["beta"].eq(0.5)
        & first["study"].eq("gainloss")
        & first["method"].eq("Naturalness")
    ].iloc[0]
    assert selected["mean_width"] == pytest.approx(0.6)
    assert selected["mean_hit_ratio"] == pytest.approx(0.5)
    assert selected["se_width"] >= 0.0
    assert selected["se_hit_ratio"] >= 0.0


def test_sign_confusion_matrix_counts_only_usable_verdicts() -> None:
    """The matrix drops invalid rows and orders axes by :data:`SIGN_ORDER`."""

    models = _interval_fixture().assign(
        gamma_run="dgp_oracle", model=lambda frame: frame["method"].astype(str)
    )
    annotated = hit_metrics.annotate_reference_agreement(models)
    naturalness = annotated.loc[annotated["model"].eq("naturalness")]
    matrix = hit_metrics.sign_confusion_matrix(naturalness)
    # Bounds [0.05, 0.45] call every stratum positive, and stratum 3 is
    # invalid, so all three usable rows land in the positive column.
    assert matrix.sum() == pytest.approx(3.0)
    assert matrix[:, hit_metrics.POSITIVE_INDEX].tolist() == [1.0, 1.0, 1.0]
    assert matrix[:, hit_metrics.NEGATIVE_INDEX].sum() == pytest.approx(0.0)


def test_sign_metrics_partition_the_usable_strata() -> None:
    """The five verdict cells sum to the usable count and drive every rate."""

    matrix = np.zeros((3, 3))
    matrix[hit_metrics.POSITIVE_INDEX, hit_metrics.POSITIVE_INDEX] = 1.0
    matrix[hit_metrics.NEGATIVE_INDEX, hit_metrics.POSITIVE_INDEX] = 1.0
    matrix[hit_metrics.INCONCLUSIVE_INDEX, hit_metrics.POSITIVE_INDEX] = 1.0
    scored = hit_metrics.sign_metrics_from_matrix(matrix)
    cells = (
        scored["n_sign_correct"]
        + scored["n_sign_reversal"]
        + scored["n_sign_undetermined"]
        + scored["n_bound_abstained"]
        + scored["n_agree_inconclusive"]
    )
    assert cells == pytest.approx(matrix.sum())
    assert scored["n_model_decisive"] == pytest.approx(3.0)
    assert scored["decisiveness_rate"] == pytest.approx(1.0)
    # The three rates split the decisive calls and must sum to one.
    assert scored["sign_correct_rate"] == pytest.approx(1.0 / 3.0)
    assert scored["sign_reversal_rate"] == pytest.approx(1.0 / 3.0)
    assert scored["sign_undetermined_rate"] == pytest.approx(1.0 / 3.0)
    assert (
        scored["sign_correct_rate"]
        + scored["sign_reversal_rate"]
        + scored["sign_undetermined_rate"]
    ) == pytest.approx(1.0)


def test_sign_metrics_exclude_classification_scores() -> None:
    """No classifier aggregate survives; a bound is not a labelling rule."""

    scored = hit_metrics.sign_metrics_from_matrix(np.diag([4.0, 5.0, 6.0]))
    for removed in (
        "sign_f1",
        "sign_mcc",
        "sign_kappa_linear",
        "sign_precision",
        "sign_recall_rate",
        "directional_accuracy",
        "false_decisive_rate",
    ):
        assert removed not in scored
    for helper in ("multiclass_mcc", "linear_weighted_kappa"):
        assert not hasattr(analysis, helper)


def test_sign_metrics_broadcast_over_stacked_matrices() -> None:
    """Leading axes are treated as a batch so the bootstrap can score in bulk."""

    stacked = np.stack([np.diag([3.0, 3.0, 3.0]), np.full((3, 3), 2.0)])
    scored = hit_metrics.sign_metrics_from_matrix(stacked)
    assert scored["sign_correct_rate"].shape == (2,)
    # A perfectly diagonal matrix has no reversals and no undetermined calls.
    assert scored["sign_correct_rate"][0] == pytest.approx(1.0)
    # A uniform matrix splits its six decisive calls evenly three ways.
    assert scored["sign_correct_rate"][1] == pytest.approx(1.0 / 3.0)


def test_summarize_agreement_reports_sign_identification_metrics() -> None:
    """Every declared metric column reaches the summary with usable values."""

    models = _interval_fixture().assign(
        gamma_run="dgp_oracle", model=lambda frame: frame["method"].astype(str)
    )
    annotated = hit_metrics.annotate_reference_agreement(models)
    summary = hit_metrics.summarize_agreement(annotated, grouping=("beta",))
    for column in (*hit_metrics.SIGN_COUNT_COLUMNS, *hit_metrics.SIGN_METRIC_COLUMNS):
        assert column in summary.columns
    naturalness = summary.loc[summary["model"].eq("naturalness")].iloc[0]
    assert naturalness["n_model_decisive"] == 3
    assert naturalness["sign_correct_rate"] == pytest.approx(1.0 / 3.0)
    assert naturalness["decisiveness_rate"] == pytest.approx(1.0)
    # CB-DR spans [-1, 1] on every stratum, so its bound never excludes zero and
    # the correct-sign rate is undefined rather than zero.
    cb_dr = summary.loc[summary["model"].eq("cb_dr")].iloc[0]
    assert cb_dr["n_model_decisive"] == 0
    assert np.isnan(cb_dr["sign_correct_rate"])
    assert cb_dr["decisiveness_rate"] == pytest.approx(0.0)


def test_discover_and_load_gamma_variants(tmp_path: Path) -> None:
    """Variant leaves contribute only their own B-Learner rows."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    _write_leaf(
        primary / hit_metrics.BASELINE_VARIANTS_DIRNAME / "gamma_exp_beta",
        _interval_fixture(blearner_shift=0.1),
    )
    leaves = hit_metrics.discover_exp25_gamma_leaves(primary)
    assert [label for label, _ in leaves] == ["dgp_oracle", "exp_beta"]
    models = hit_metrics.load_exp25_models(leaves)
    assert set(models["model"]) == {
        "naturalness",
        "cb_ipw",
        "cb_dr",
        "blearner[dgp_oracle]",
        "blearner[exp_beta]",
    }
    # Naturalness and CB rows are taken once, not duplicated per Gamma leaf.
    assert int(models["model"].eq("naturalness").sum()) == len(STRATA)
    variant = models.loc[models["model"].eq("blearner[exp_beta]")]
    assert variant["lower"].to_numpy() == pytest.approx(
        METHOD_BOUNDS["blearner"][0] + 0.1
    )


def _reference_bound_fixture() -> pd.DataFrame:
    """Build Manski and Balke-Pearl rows on the fixture strata.

    Balke-Pearl is invalid on stratum 1 (age 1, gender 0), which the core
    methods all bound, so it must drop out of its own common-support rows
    without removing that stratum from anyone else's.
    """

    base = _interval_fixture().loc[lambda frame: frame["method"].eq("naturalness")]
    base = base.assign(lp_posterior_prob_optimal=np.nan)
    manski = base.assign(method="manski", lower=-0.5, upper=0.5, width=1.0,
                         is_valid=True, exclusion_reason="")
    missing = (base["resp_age_bin"].eq(1) & base["resp_gender_bin"].eq(0)).to_numpy()
    balke_pearl = base.assign(
        method="balke_pearl",
        lower=np.where(missing, np.nan, -0.1),
        upper=np.where(missing, np.nan, 0.6),
        width=np.where(missing, np.nan, 0.7),
        is_valid=~missing,
        exclusion_reason=np.where(missing, "no_supported_instrument", ""),
    )
    return pd.concat([manski, balke_pearl], ignore_index=True)


def test_identification_leaf_adds_reference_models_without_moving_support(
    tmp_path: Path,
) -> None:
    """Manski/Balke-Pearl join as models; core common support is unchanged."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    variants = primary / hit_metrics.BASELINE_VARIANTS_DIRNAME
    core = hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))
    _write_leaf(variants / "identification_bounds", _reference_bound_fixture())
    models = hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))
    assert set(models["model"]) == set(core["model"]) | {"manski", "balke_pearl"}
    assert int(models["model"].eq("balke_pearl").sum()) == len(STRATA)

    common = hit_metrics.select_common_support(hit_metrics.annotate_reference_agreement(models))
    core_common = hit_metrics.select_common_support(hit_metrics.annotate_reference_agreement(core))
    compared = common.loc[~common["model"].isin(hit_metrics.REFERENCE_BOUND_MODELS)]
    pd.testing.assert_frame_equal(
        compared.drop(columns=["gamma_run", "result_leaf"]).reset_index(drop=True),
        core_common.drop(columns=["gamma_run", "result_leaf"]).reset_index(drop=True),
        check_like=True,
    )
    # Naturalness is invalid on stratum 3, so the core support is strata 0-2;
    # Balke-Pearl keeps the two of those it bounds, Manski all three.
    assert int(common["model"].eq("manski").sum()) == 3
    assert int(common["model"].eq("balke_pearl").sum()) == 2


def test_sign_table_uses_every_listed_model_for_support(tmp_path: Path) -> None:
    """Reference bounds shrink the sign table's support for every column."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    _write_leaf(
        primary / hit_metrics.BASELINE_VARIANTS_DIRNAME / "identification_bounds",
        _reference_bound_fixture(),
    )
    annotated = hit_metrics.annotate_reference_agreement(
        hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))
    )

    lines = analysis.sign_table_latex(annotated).splitlines()

    assert lines[0] == r"\begin{tabular}{lccccccc}"
    assert lines[2] == (
        r"$\beta$ & Naturalness & Manski & Balke--Pearl & Oracle B-Learner"
        r" & CB-IPW & CB-DR & $n$ \\"
    )
    # Naturalness drops stratum 3 and Balke-Pearl stratum 1, leaving strata 0
    # (RCT positive) and 2 (RCT inconclusive). Naturalness is positive on both,
    # the B-Learner negative on both, and every other bound contains zero.
    assert lines[4] == (
        r"1 & 1.00 (0.50) & 0.00 (--) & 0.00 (--) & 1.00 (0.00)"
        r" & 0.00 (--) & 0.00 (--) & 2 \\"
    )


def test_sign_table_by_study_follows_manuscript_order(tmp_path: Path) -> None:
    """Study blocks follow the appendix order, separated by midrules."""

    frame = pd.concat(
        [
            _interval_fixture().assign(study=study)
            for study in ("allowedforbidden", "gainloss", "reciprocity")
        ],
        ignore_index=True,
    )
    primary = _write_leaf(tmp_path / "primary", frame)
    annotated = hit_metrics.annotate_reference_agreement(
        hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))
    )

    lines = analysis.sign_table_latex(
        annotated, grouping=("study", "beta")
    ).splitlines()

    # Runs without the identification leaf simply omit those two columns.
    assert lines[0] == r"\begin{tabular}{llccccc}"
    assert lines[2] == (
        r"Study & $\beta$ & Naturalness & Oracle B-Learner & CB-IPW & CB-DR"
        r" & $n$ \\"
    )
    body = lines[4:-2]
    assert [line.split(" & ")[0] for line in body] == [
        "Gain-loss", r"\midrule", "Reciprocity", r"\midrule", "Allowed-forbidden",
    ]
    assert body[0] == (
        r"Gain-loss & 1 & 1.00 (0.33) & 1.00 (0.33) & 0.00 (--) & 0.00 (--)"
        r" & 3 \\"
    )


def test_common_support_frontier_bootstraps_width_and_hit_intervals() -> None:
    """Replication resamples bracket both axes; zero replicates adds no columns."""

    second = _interval_fixture().assign(replication=1, seed=43)
    # A wider Naturalness bound in replication 1 gives its width real spread.
    wider = second["method"].eq("naturalness") & second["is_valid"]
    second.loc[wider, "upper"] += 0.2
    second.loc[wider, "width"] += 0.2
    frames = pd.concat([_interval_fixture(), second], ignore_index=True)
    models = frames.assign(
        gamma_run="dgp_oracle",
        model=lambda frame: np.where(
            frame["method"].eq("blearner"), "blearner[dgp_oracle]", frame["method"]),
    )
    annotated = hit_metrics.annotate_reference_agreement(models)
    columns = ["mean_width_ci_lower", "mean_width_ci_upper",
               "hit_ratio_ci_lower", "hit_ratio_ci_upper"]
    plain = hit_metrics.common_support_frontier(annotated, grouping=("beta",))
    assert not set(columns).intersection(plain.columns)

    frontier = hit_metrics.common_support_frontier(
        annotated, grouping=("beta",), bootstrap_replicates=400, seed=3)
    again = hit_metrics.common_support_frontier(
        annotated, grouping=("beta",), bootstrap_replicates=400, seed=3)
    pd.testing.assert_frame_equal(frontier, again)
    pd.testing.assert_frame_equal(frontier.drop(columns=[*columns, "ci_level"]), plain)
    assert frontier["ci_level"].eq(hit_metrics.FRONTIER_CONFIDENCE_LEVEL).all()
    for metric in ("mean_width", "hit_ratio"):
        assert (frontier[f"{metric}_ci_lower"] <= frontier[metric] + 1e-12).all()
        assert (frontier[f"{metric}_ci_upper"] >= frontier[metric] - 1e-12).all()
    naturalness = frontier.loc[frontier["model"].astype(str).eq("naturalness")].iloc[0]
    assert naturalness["mean_width_ci_upper"] > naturalness["mean_width_ci_lower"]


def test_common_support_frontier_can_anchor_on_one_model() -> None:
    """Naming every other model a reference lets one model fix the support."""

    models = _interval_fixture().assign(model=lambda frame: frame["method"])
    annotated = hit_metrics.annotate_reference_agreement(models)
    default = hit_metrics.common_support_frontier(
        annotated, grouping=("beta",)).set_index("model")
    # Naturalness has no fourth-stratum interval, so the complete case drops it
    # for every model.
    assert default["n_common"].eq(3).all()
    others = sorted(set(models["model"]) - {"cb_dr"})
    anchored = hit_metrics.common_support_frontier(
        annotated, grouping=("beta",), reference_models=others).set_index("model")
    assert anchored.loc["cb_dr", "n_common"] == 4
    assert anchored.loc["cb_ipw", "n_common"] == 4
    # A model is still scored only where its own interval is usable.
    assert anchored.loc["naturalness", "n_common"] == 3


def test_common_support_frontier_reports_replication_spread() -> None:
    """The SD columns are the spread of per-replication common-support means."""

    second = _interval_fixture().assign(replication=1, seed=43)
    wider = second["method"].eq("naturalness") & second["is_valid"]
    second.loc[wider, "upper"] += 0.2
    second.loc[wider, "width"] += 0.2
    frames = pd.concat([_interval_fixture(), second], ignore_index=True)
    models = frames.assign(
        gamma_run="dgp_oracle",
        model=lambda frame: np.where(
            frame["method"].eq("blearner"), "blearner[dgp_oracle]", frame["method"]),
    )
    annotated = hit_metrics.annotate_reference_agreement(models)
    columns = ["mean_width_sd", "hit_ratio_sd", "n_replications"]
    plain = hit_metrics.common_support_frontier(annotated, grouping=("beta",))
    assert not set(columns).intersection(plain.columns)

    frontier = hit_metrics.common_support_frontier(
        annotated, grouping=("beta",), replication_spread=True)
    pd.testing.assert_frame_equal(frontier.drop(columns=columns), plain)

    common = hit_metrics.select_common_support(annotated)
    per_replication = common.groupby(["beta", "model", "replication"]).agg(
        width=("width", "mean"), hits=("rct_hit", "mean")).reset_index()
    expected = per_replication.groupby(["beta", "model"]).agg(
        mean_width_sd=("width", "std"), hit_ratio_sd=("hits", "std"),
        n_replications=("width", "size")).reset_index()
    merged = frontier.assign(model=frontier["model"].astype(str)).merge(
        expected, on=["beta", "model"], suffixes=("", "_expected"))
    assert len(merged) == len(frontier)
    for column in columns:
        np.testing.assert_allclose(
            merged[column].astype(float), merged[f"{column}_expected"].astype(float))
    assert merged["n_replications"].eq(2).all()
    # The widened replication gives Naturalness a real width spread.
    naturalness = merged.loc[merged["model"].eq("naturalness"), "mean_width_sd"]
    assert (naturalness > 0).all()


def test_identification_leaf_must_cover_primary_strata_once(tmp_path: Path) -> None:
    """A partial or duplicated reference-bound leaf is rejected."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    variants = primary / hit_metrics.BASELINE_VARIANTS_DIRNAME
    bounds = _reference_bound_fixture()
    _write_leaf(variants / "identification_bounds", bounds.iloc[1:])
    with pytest.raises(ValueError, match="does not cover exactly"):
        hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))
    _write_leaf(variants / "identification_bounds", bounds)
    _write_leaf(variants / "identification_copy", bounds)
    with pytest.raises(ValueError, match="supplied by both"):
        hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))


def test_load_exp25_models_rejects_altered_shared_rows(tmp_path: Path) -> None:
    """A variant that changed reused Naturalness or CB rows is rejected."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    altered = _interval_fixture()
    altered.loc[altered["method"].eq("cb_dr"), "upper"] = 0.5
    _write_leaf(
        primary / hit_metrics.BASELINE_VARIANTS_DIRNAME / "gamma_oracle",
        altered,
    )
    leaves = hit_metrics.discover_exp25_gamma_leaves(primary)
    with pytest.raises(ValueError, match="altered reused non-B-Learner rows"):
        hit_metrics.load_exp25_models(leaves)


def test_run_analysis_writes_tables_and_report(tmp_path: Path) -> None:
    """The end-to-end run writes every CSV table and the markdown report."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    _write_leaf(
        primary / hit_metrics.BASELINE_VARIANTS_DIRNAME / "gamma_exp_beta",
        _interval_fixture(blearner_shift=0.1),
    )
    result = analysis.run_analysis(primary, tmp_path / "analysis")
    output_dir = Path(result["output_dir"])
    for name in (
        "exp25_hit_ratio_pooled.csv",
        "exp25_hit_ratio_by_study.csv",
        "exp25_hit_ratio_by_bias_covariate.csv",
        "exp25_hit_ratio_by_study_bias.csv",
        analysis.SELECTED_WIDTH_HIT_FILENAME,
        "exp25_sign_confusion.csv",
        "exp25_hit_ratio_report.md",
        *analysis.SIGN_TABLE_FILENAMES.values(),
    ):
        assert (output_dir / name).is_file()
    assert len(result["sign_table_paths"]) == len(analysis.SIGN_TABLE_FILENAMES)
    # The fixture holds beta 1.0 only; requested betas it lacks are skipped
    # rather than written as empty rows.
    assert set(result["selected_width_hit"]["beta"]) == {1.0}
    assert "causal-sign agreement" in Path(result["report_path"]).read_text()


def test_discover_reads_primary_from_baseline_variants(tmp_path: Path) -> None:
    """The primary dgp_oracle run is discovered inside baseline_variants."""

    root = tmp_path / "tabicl_v2"
    variants = root / hit_metrics.BASELINE_VARIANTS_DIRNAME
    primary = _write_leaf(variants / "gamma_dgp_oracle", _interval_fixture())
    variant = _write_leaf(
        variants / "gamma_exp_beta",
        _interval_fixture(blearner_shift=0.1),
    )

    leaves = hit_metrics.discover_exp25_gamma_leaves(root)

    assert leaves == [("dgp_oracle", primary), ("exp_beta", variant)]
    models = hit_metrics.load_exp25_models(leaves)
    assert int(models["model"].eq("naturalness").sum()) == len(STRATA)
    assert set(models["model"]) == {
        "naturalness",
        "cb_ipw",
        "cb_dr",
        "blearner[dgp_oracle]",
        "blearner[exp_beta]",
    }


def test_discover_requires_primary_intervals(tmp_path: Path) -> None:
    """A leaf without the primary compact table raises a targeted error."""

    with pytest.raises(FileNotFoundError, match="Primary Exp25 leaf"):
        hit_metrics.discover_exp25_gamma_leaves(tmp_path)


def _leaf_with_reference_bounds(path: Path) -> Path:
    """Write the fixture run together with its identification-bounds leaf."""

    primary = _write_leaf(path, _interval_fixture())
    _write_leaf(
        primary / hit_metrics.BASELINE_VARIANTS_DIRNAME / "identification_bounds",
        _reference_bound_fixture(),
    )
    return primary


def test_failures_as_manski_scores_failed_strata_as_manski(tmp_path: Path) -> None:
    """A failed stratum takes its Manski interval; usable rows are unchanged."""

    primary = _leaf_with_reference_bounds(tmp_path / "primary")
    models = hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))

    scored = hit_metrics.failures_as_manski(models)

    assert len(scored) == len(models)
    assert scored["is_valid"].all()
    # Naturalness fails on stratum 3 and Balke-Pearl on stratum 1.
    failed = ~models["is_valid"]
    assert int(failed.sum()) == 2
    np.testing.assert_allclose(scored.loc[failed, "lower"], -0.5)
    np.testing.assert_allclose(scored.loc[failed, "upper"], 0.5)
    np.testing.assert_allclose(
        scored.loc[~failed, ["lower", "upper"]], models.loc[~failed, ["lower", "upper"]]
    )
    np.testing.assert_allclose(scored["width"], scored["upper"] - scored["lower"])


def test_failures_as_manski_draw_level_mixes_naturalness_with_manski(
    tmp_path: Path,
) -> None:
    """Valid Naturalness endpoints become their mixture with the Manski interval."""

    primary = _leaf_with_reference_bounds(tmp_path / "primary")
    models = hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))

    scored = hit_metrics.failures_as_manski(models, draw_level=True)

    naturalness = models["model"].eq("naturalness") & models["is_valid"]
    # An optimal-solve probability of 0.9 mixes [0.05, 0.45] with [-0.5, 0.5].
    np.testing.assert_allclose(scored.loc[naturalness, "lower"], 0.9 * 0.05 - 0.1 * 0.5)
    np.testing.assert_allclose(scored.loc[naturalness, "upper"], 0.9 * 0.45 + 0.1 * 0.5)
    others = ~models["model"].eq("naturalness") & models["is_valid"]
    np.testing.assert_allclose(
        scored.loc[others, ["lower", "upper"]], models.loc[others, ["lower", "upper"]]
    )


def test_failures_as_manski_requires_manski_rows(tmp_path: Path) -> None:
    """Without the identification-bounds leaf there is nothing to fall back to."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    models = hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))
    with pytest.raises(ValueError, match="no Manski interval"):
        hit_metrics.failures_as_manski(models)


def test_width_table_labels_selection_covariates(tmp_path: Path) -> None:
    """Width cells read mean width (hit ratio), in labelled selection blocks."""

    frame = pd.concat(
        [
            _interval_fixture().assign(bias_covariate=covariate)
            for covariate in ("polideo", "age", "gender")
        ],
        ignore_index=True,
    )
    primary = _write_leaf(tmp_path / "primary", frame)
    annotated = hit_metrics.annotate_reference_agreement(
        hit_metrics.load_exp25_models(hit_metrics.discover_exp25_gamma_leaves(primary))
    )

    lines = analysis.width_table_latex(
        annotated, grouping=("bias_covariate", "beta")
    ).splitlines()

    assert lines[0] == r"\begin{tabular}{llccccc}"
    assert lines[2] == (
        r"Selection & $\beta$ & Naturalness & Oracle B-Learner & CB-IPW & CB-DR"
        r" & $n$ \\"
    )
    body = lines[4:-2]
    assert [line.split(" & ")[0] for line in body] == [
        "age", r"\midrule", "sex", r"\midrule", "ideology",
    ]
    # Naturalness fails on stratum 3, so strata 0-2 form the support. Its
    # [0.05, 0.45] contains only stratum 0's reference, the B-Learner's
    # [-0.6, -0.2] only stratum 1's, and both CB bounds all three.
    assert body[0] == (
        r"age & 1 & 0.40 (0.33) & 0.40 (0.33) & 1.80 (1.00) & 2.00 (1.00) & 3 \\"
    )


def test_run_failures_as_manski_analysis_writes_manuscript_outputs(
    tmp_path: Path,
) -> None:
    """The manuscript run scores all strata and writes tables, CSVs and a manifest."""

    primary = _leaf_with_reference_bounds(tmp_path / "primary")

    result = analysis.run_failures_as_manski_analysis(
        primary, bootstrap_replicates=50, plot_formats=("png",)
    )

    output_dir = Path(result["output_dir"])
    assert output_dir == (primary / analysis.FAILURES_AS_MANSKI_DIRNAMES[False]).resolve()
    assert {path.name for path in result["table_paths"]} == {
        *analysis.WIDTH_TABLE_FILENAMES.values(),
        *analysis.SIGN_TABLE_FILENAMES.values(),
    }
    for name in (
        "exp25_common_support_frontier_by_study.csv",
        "exp25_common_support_frontier.csv",
        "exp25_hit_ratio_pooled.csv",
        analysis.FAILURES_AS_MANSKI_MANIFEST,
    ):
        assert (output_dir / name).is_file()
    # Every stratum is scored, including Naturalness's failed one.
    frontier = pd.read_csv(output_dir / "exp25_common_support_frontier.csv")
    assert frontier["n_common"].eq(len(STRATA)).all()
    manifest = json.loads((output_dir / analysis.FAILURES_AS_MANSKI_MANIFEST).read_text())
    assert manifest["strata_scored"] == len(STRATA)
    assert manifest["tables"] == [path.name for path in result["table_paths"]]
