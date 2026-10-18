"""Tests for the Exp25 hit-ratio and sign-agreement figure set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.exp25_hit_ratio_analysis as analysis
import experiments.exp25_hit_ratio_plots as plots

from tests.test_experiments.test_exp25_hit_ratio_analysis import (
    _interval_fixture,
    _write_leaf,
)


EXPECTED_FIGURE_STEMS = (
    "sign_identification_frontier",
    "gamma_provenance",
    "sign_cell_decomposition",
    "sign_confusion_heatmaps",
    "width_hit_frontier_by_beta",
    "width_hit_frontier_by_beta_sd",
    "width_hit_frontier_trajectory",
    "hit_ratio_dose_response",
    "paired_hit_outcomes",
    "replication_variability",
)


def _annotated_fixture() -> pd.DataFrame:
    """Build an annotated two-Gamma fixture without touching the filesystem."""

    primary = _interval_fixture()
    variant = _interval_fixture(blearner_shift=0.1)
    shared = primary.loc[~primary["method"].eq("blearner")].assign(
        gamma_run="dgp_oracle",
        model=lambda frame: frame["method"].astype(str),
    )
    frames = [shared]
    for gamma_run, table in (("dgp_oracle", primary), ("exp_beta", variant)):
        blearner = table.loc[table["method"].eq("blearner")].assign(
            gamma_run=gamma_run, model=f"blearner[{gamma_run}]"
        )
        frames.append(blearner)
    models = pd.concat(frames, ignore_index=True, sort=False)
    return analysis.annotate_reference_agreement(models)


def test_wilson_interval_brackets_the_proportion() -> None:
    """Wilson bounds stay inside the unit range and contain the estimate."""

    lower, upper = plots.wilson_interval([0, 5, 10], [10, 10, 10])
    assert np.all(lower >= 0.0) and np.all(upper <= 1.0)
    assert lower[0] == pytest.approx(0.0)
    assert upper[2] == pytest.approx(1.0)
    assert lower[1] < 0.5 < upper[1]
    # A zero denominator has no defined proportion.
    empty_lower, empty_upper = plots.wilson_interval([0], [0])
    assert np.isnan(empty_lower[0]) and np.isnan(empty_upper[0])


def test_build_paired_outcomes_classifies_concordance() -> None:
    """Each competitor is paired to Naturalness on identical stratum keys."""

    paired = plots.build_paired_outcomes(_annotated_fixture())
    assert set(paired["competitor"]) == {
        "blearner[dgp_oracle]",
        "blearner[exp_beta]",
        "cb_ipw",
        "cb_dr",
    }
    # Four competitors paired against each of the four strata.
    assert len(paired) == 16
    cb_dr = paired.loc[paired["competitor"].eq("cb_dr")].reset_index(drop=True)
    # CB-DR bounds [-1, 1] hit every stratum; Naturalness hits only stratum 0.
    assert list(cb_dr["pair_outcome"]) == [
        "both_hit",
        "competitor_only",
        "competitor_only",
        "competitor_only",
    ]
    assert set(paired["pair_outcome"]).issubset(set(plots.PAIR_OUTCOME_ORDER))


def test_model_ordering_and_styles_cover_unknown_gamma_rules() -> None:
    """Unknown Gamma variants receive a distinct colour, marker, and label."""

    models = ["cb_dr", "naturalness", "blearner[custom_rule]"]
    assert plots.ordered_models(models) == [
        "naturalness",
        "cb_dr",
        "blearner[custom_rule]",
    ]
    palette = plots.model_palette(models)
    assert palette["naturalness"] == plots.MODEL_COLORS["naturalness"]
    assert palette["blearner[custom_rule]"] not in {
        plots.MODEL_COLORS["naturalness"],
        plots.MODEL_COLORS["cb_dr"],
    }
    # An unknown Gamma rule still reads as a B-Learner variant rather than as a
    # raw column value, with Gamma rendered as a mathtext symbol.
    assert plots.model_label("blearner[custom_rule]") == (
        f"B-Learner ({plots.GAMMA_SYMBOL} custom-rule)"
    )
    assert plots.model_label("blearner[fixed_2p5]") == (
        f"B-Learner ({plots.GAMMA_SYMBOL} = 2.5)"
    )
    assert plots.model_short_label("blearner[custom_rule]") == "BL custom"
    assert plots.model_marker("cb_dr", models) == plots.MODEL_MARKERS["cb_dr"]
    assert plots.model_marker("blearner[custom_rule]", models) in plots.FALLBACK_MARKERS
    handles = plots.model_handles(palette)
    assert [handle.get_label() for handle in handles] == [
        plots.model_label(name) for name in plots.ordered_models(models)
    ]


def test_is_fixed_gamma_model_separates_the_sweep_from_the_rules() -> None:
    """Only numeric Gamma leaves join the swept frontier curve."""

    assert plots.is_fixed_gamma_model("blearner[fixed_2p5]")
    assert not plots.is_fixed_gamma_model("blearner[oracle]")
    assert not plots.is_fixed_gamma_model("blearner[width_matched]")
    assert not plots.is_fixed_gamma_model("naturalness")
    assert plots.blearner_gamma_rule("blearner[dgp_oracle]") == "dgp_oracle"
    assert plots.blearner_gamma_rule("cb_ipw") is None


def test_model_palette_survives_a_full_gamma_grid() -> None:
    """A nine-point grid exhausts the hand-picked fallbacks without collisions."""

    models = [
        "naturalness",
        *[f"blearner[fixed_{index}]" for index in range(9)],
        "blearner[oracle]",
        "cb_ipw",
    ]
    palette = plots.model_palette(models)
    assert len(palette) == len(models)
    colours = [str(value) for value in palette.values()]
    assert len(set(colours)) == len(colours)


def test_sign_identification_frontier_omits_the_fixed_gamma_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixed-Gamma rows draw nothing: no curve, no matched note, no legend entry."""

    named = ["naturalness", "cb_ipw", "blearner[dgp_oracle]"]
    fixed = ["blearner[fixed_1]", "blearner[fixed_2]", "blearner[fixed_4]"]
    pooled = pd.DataFrame(
        [
            {
                "model": model,
                "beta": beta,
                plots.FRONTIER_DECISIVENESS: 0.1 * (index + 1),
                plots.FRONTIER_CORRECT: 0.9 - 0.1 * index,
                "mean_blearner_gamma": float(index + 1),
            }
            for beta in (0.5, 1.0)
            for index, model in enumerate(named + fixed)
        ]
    )
    captured = []

    def capture(figure, output_stem, plot_formats):
        """Keep the figure open for inspection instead of saving it."""
        captured.append(figure)
        return []

    monkeypatch.setattr(plots, "_save_figure", capture)
    plots.make_sign_identification_frontier_plot(pooled, tmp_path)
    (figure,) = captured
    labels = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
    assert labels and not any("sweep" in label for label in labels)
    # The legend is bold, at the manuscript's legend size.
    assert all(
        text.get_fontsize() == pytest.approx(plots.PAPER_RC["legend.fontsize"])
        and text.get_fontweight() == "bold"
        for legend in figure.legends
        for text in legend.get_texts()
    )
    panels = [axis for axis in figure.axes if axis.get_title()]
    assert len(panels) == 2
    for axis in panels:
        # One errorbar container per named model; nothing else is drawn.
        assert len(axis.containers) == len(named)
        assert not axis.texts and not axis.collections
        # The grid is labelled once, at figure level, not panel by panel.
        assert axis.get_xlabel() == "" and axis.get_ylabel() == ""
        assert axis.title.get_fontweight() == "bold"
    assert figure._supxlabel.get_text() == "Decisiveness rate"
    assert figure._supylabel.get_text() == "Correct-sign rate"
    plots.plt.close(figure)


def test_width_hit_by_study_draws_interval_bars_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interval columns become two-axis bars, and the figure carries no title."""

    rows = [
        {"study": "gainloss", "beta": beta, "model": model, "mean_width": width,
         "hit_ratio": hit, "mean_width_ci_lower": width - 0.05,
         "mean_width_ci_upper": width + 0.04, "hit_ratio_ci_lower": hit - 0.1,
         "hit_ratio_ci_upper": min(hit + 0.02, 1.0), "ci_level": 0.9}
        for beta in (0.5, 1.0)
        for model, width, hit in (("naturalness", 0.4, 0.95), ("cb_ipw", 0.7, 0.9))
    ]
    captured = []

    def capture(figure, output_stem, plot_formats):
        """Keep the figure open for inspection instead of saving it."""
        captured.append(figure)
        return []

    monkeypatch.setattr(plots, "_save_figure", capture)
    plots.make_width_hit_frontier_by_study_plot(pd.DataFrame(rows), tmp_path)
    (figure,) = captured
    # One study, two betas: a single column of two panels.
    assert len(figure.axes) == 2
    for axis in figure.axes:
        assert len(axis.containers) == 2
        (bars,) = axis.containers[:1]
        assert bars.has_xerr and bars.has_yerr
    assert figure._suptitle is None
    plots.plt.close(figure)

    captured.clear()
    bare = pd.DataFrame(rows).drop(columns=[*plots.FRONTIER_INTERVAL_COLUMNS, "ci_level"])
    plots.make_width_hit_frontier_by_study_plot(bare, tmp_path)
    (figure,) = captured
    assert all(not axis.containers for axis in figure.axes)
    assert figure._suptitle is None
    plots.plt.close(figure)


def _frontier_rows(*, with_study: bool) -> pd.DataFrame:
    """Two-model frontier rows carrying both 95% intervals and SD columns."""

    rows = [
        {"beta": beta, "model": model, "mean_width": width, "hit_ratio": hit,
         "mean_width_ci_lower": width - 0.05, "mean_width_ci_upper": width + 0.04,
         "hit_ratio_ci_lower": hit - 0.1, "hit_ratio_ci_upper": min(hit + 0.02, 1.0),
         "ci_level": 0.95, "mean_width_sd": 0.03, "hit_ratio_sd": 0.07,
         "n_replications": 20}
        for beta in (0.5, 1.0)
        for model, width, hit in (("naturalness", 0.4, 0.9), ("cb_ipw", 0.7, 0.85))
    ]
    frame = pd.DataFrame(rows)
    return frame.assign(study="gainloss") if with_study else frame


@pytest.mark.parametrize(
    ("with_study", "draw", "stem"),
    [
        (False, "make_width_hit_frontier_by_beta_plot", "width_hit_frontier_by_beta"),
        (True, "make_width_hit_frontier_by_study_plot", "width_hit_frontier_by_study"),
    ],
)
def test_width_hit_frontiers_draw_intervals_or_standard_deviations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_study: bool, draw: str, stem: str
) -> None:
    """Each frontier has a 95%-interval figure and an SD twin under ``_sd``."""

    captured = {}

    def capture(figure, output_stem, plot_formats):
        """Record the bar extents and title instead of saving the figure."""
        panels = [axis for axis in figure.axes if axis.get_title()]
        captured[output_stem.name] = {
            "suptitle": figure._suptitle,
            # Horizontal then vertical bar segments of each errorbar container.
            "segments": [
                [
                    [tuple(point) for point in segment]
                    for collection in container.lines[2]
                    for segment in collection.get_segments()
                ]
                for axis in panels
                for container in axis.containers
            ],
        }
        plots.plt.close(figure)
        return [output_stem.with_suffix(".png")]

    monkeypatch.setattr(plots, "_save_figure", capture)
    frame = _frontier_rows(with_study=with_study)
    make = getattr(plots, draw)
    assert make(frame, tmp_path)[0].name == f"{stem}.png"
    assert make(frame, tmp_path, spread="sd")[0].name == f"{stem}_sd.png"

    interval, deviation = captured[stem], captured[f"{stem}_sd"]
    assert interval["suptitle"] is None and deviation["suptitle"] is None
    # Naturalness at beta 0.5 is drawn first: width 0.4 +/- 0.03, hit 0.9 +/- 0.07.
    horizontal, vertical = deviation["segments"][0]
    np.testing.assert_allclose(horizontal, [(0.37, 0.9), (0.43, 0.9)])
    np.testing.assert_allclose(vertical, [(0.4, 0.83), (0.4, 0.97)])
    horizontal, _ = interval["segments"][0]
    np.testing.assert_allclose(horizontal, [(0.35, 0.9), (0.44, 0.9)])

    # Without SD columns there is nothing for the twin to show.
    bare = frame.drop(columns=list(plots.FRONTIER_SD_COLUMNS))
    assert make(bare, tmp_path, spread="sd") == []
    with pytest.raises(ValueError, match="spread must be one of"):
        make(frame, tmp_path, spread="sem")


def test_bootstrap_error_bars_are_non_negative_offsets() -> None:
    """Percentile bounds become clipped offsets, and a missing table is skipped."""

    bootstrap = pd.DataFrame(
        {
            "beta": [1.0, 1.0],
            "model": ["naturalness", "cb_ipw"],
            "metric": ["sign_correct_rate", "sign_correct_rate"],
            "ci_lower": [0.55, 0.10],
            "ci_upper": [0.80, 0.30],
        }
    )
    offsets = plots.bootstrap_error_bars(
        bootstrap,
        beta=1.0,
        models=["naturalness", "cb_ipw"],
        metric="sign_correct_rate",
        estimates=np.array([0.70, 0.20]),
    )
    assert offsets is not None
    assert offsets.shape == (2, 2)
    assert np.all(offsets >= 0.0)
    assert offsets[0][0] == pytest.approx(0.15)
    assert offsets[1][0] == pytest.approx(0.10)
    assert (
        plots.bootstrap_error_bars(
            None,
            beta=1.0,
            models=["naturalness"],
            metric="sign_correct_rate",
            estimates=np.array([0.7]),
        )
        is None
    )
    # An absent metric leaves the markers bare rather than drawing zero bars.
    assert (
        plots.bootstrap_error_bars(
            bootstrap,
            beta=1.0,
            models=["naturalness"],
            metric="sign_reversal_rate",
            estimates=np.array([0.7]),
        )
        is None
    )


def test_infer_full_range_width_uses_fallback_rows() -> None:
    """The reference width comes from rows that actually used the fallback."""

    frame = pd.DataFrame(
        {
            "used_full_range_fallback": [False, True, True],
            "width": [0.3, 2.0, 2.0],
        }
    )
    assert plots.infer_full_range_width(frame) == pytest.approx(2.0)
    frame["used_full_range_fallback"] = False
    assert plots.infer_full_range_width(frame) is None


def test_make_all_plots_writes_every_figure(tmp_path: Path) -> None:
    """The orchestrator renders each figure stem in the requested format."""

    annotated = _annotated_fixture()
    pooled = analysis.summarize_agreement(annotated, grouping=("beta",))
    confusion = analysis.build_sign_confusion(annotated)
    paths = plots.make_all_plots(
        annotated,
        pooled,
        confusion,
        tmp_path,
        plot_formats=("png",),
    )
    figures_dir = tmp_path / plots.FIGURES_DIRNAME
    written = {path.stem for path in paths}
    for stem in EXPECTED_FIGURE_STEMS:
        assert stem in written, f"missing figure: {stem}"
        assert (figures_dir / f"{stem}.png").is_file()


GRID_GAMMAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
GRID_STRATA = ((-0.40, 0.12), (-0.15, 0.10), (-0.03, 0.11), (0.05, 0.10),
               (0.22, 0.11), (0.45, 0.13))


def _grid_interval_fixture(
    gamma: float | None,
    *,
    n_replications: int = 4,
    betas: tuple[float, ...] = (0.5, 1.0),
) -> pd.DataFrame:
    """Build one Exp25 leaf whose B-Learner width grows with Gamma.

    The sweep is what the frontier figure is for, so the fixture reproduces its
    defining property: raising Gamma widens every B-Learner interval, which
    monotonically lowers decisiveness while concentrating the surviving
    decisive calls on the largest effects. Non-B-Learner rows are identical
    across Gammas, which is what ``load_exp25_models`` validates.

    Args:
        gamma: Fixed sensitivity parameter, or ``None`` for the primary
            ``dgp_oracle`` leaf.
        n_replications: Monte Carlo replications to synthesize.
        betas: Confounding levels to synthesize.

    Returns:
        A method-long Exp25-shaped interval table.
    """

    effective = 2.75 if gamma is None else float(gamma)
    rows: list[dict[str, object]] = []
    for replication in range(n_replications):
        jitter = 0.01 * (replication - n_replications / 2.0)
        for beta in betas:
            for index, (cate, half_width) in enumerate(GRID_STRATA):
                bounds = {
                    "naturalness": (cate - 0.13 - jitter, cate + 0.13 + jitter),
                    "blearner": (
                        cate - 0.05 * effective + jitter,
                        cate + 0.05 * effective - jitter,
                    ),
                    "cb_ipw": (cate - 0.45, cate + 0.45),
                    "cb_dr": (-1.0, 1.0),
                }
                for method, (lower, upper) in bounds.items():
                    rows.append(
                        {
                            "replication": replication,
                            "seed": 42 + replication,
                            "bias_covariate": "age",
                            "beta": beta,
                            "study": "gainloss",
                            "target_site": "pi",
                            "resp_age_bin": index % 3,
                            "resp_gender_bin": index // 3,
                            "resp_polideo_bin": 0,
                            "method": method,
                            "lower": lower,
                            "upper": upper,
                            "width": upper - lower,
                            "is_valid": True,
                            "exclusion_reason": "",
                            "used_full_range_fallback": False,
                            "support_clipped": False,
                            "blearner_gamma": (
                                effective if method == "blearner" else np.nan
                            ),
                            "lp_posterior_prob_optimal": (
                                0.9 if method == "naturalness" else np.nan
                            ),
                            "rct_cate": cate,
                            "rct_cate_ci90_lower": cate - half_width,
                            "rct_cate_ci90_upper": cate + half_width,
                            "rct_cate_available": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_gamma_grid_renders_a_swept_frontier(tmp_path: Path) -> None:
    """A full Gamma grid produces a monotone pooled sweep and a width-matched readout."""

    primary = _write_leaf(tmp_path / "primary", _grid_interval_fixture(None))
    for gamma in GRID_GAMMAS:
        label = f"fixed_{gamma:g}".replace(".", "p")
        _write_leaf(
            primary / analysis.BASELINE_VARIANTS_DIRNAME / f"gamma_{label}",
            _grid_interval_fixture(gamma),
        )
    result = analysis.run_analysis(
        primary, tmp_path / "analysis", bootstrap_replicates=100
    )
    pooled = result["pooled"]
    pooled_models = set(pooled["model"].astype(str))
    assert "blearner[fixed_1]" in pooled_models
    assert "blearner[dgp_oracle]" in pooled_models

    panel = pooled.loc[np.isclose(pooled["beta"], 1.0)].copy()
    panel["model"] = panel["model"].astype(str)
    grid = panel.loc[panel["model"].map(plots.is_fixed_gamma_model)].sort_values(
        "mean_blearner_gamma"
    )
    assert len(grid) == len(GRID_GAMMAS)
    # Widening every interval can only turn a decisive verdict inconclusive, so
    # decisiveness must fall monotonically along the sweep.
    decisiveness = grid[plots.FRONTIER_DECISIVENESS].to_numpy(dtype=float)
    assert np.all(np.diff(decisiveness) <= 1e-12)

    bootstrap = result["metric_bootstrap"]
    assert not bootstrap.empty
    assert {"estimate", "ci_lower", "ci_upper"}.issubset(bootstrap.columns)
    assert not result["metric_contrasts"].empty

    matched = result["width_hit_matched"]
    assert not matched.empty
    assert {
        "n_common",
        "naturalness_width",
        "naturalness_hit_ratio",
        "blearner_hit_at_width",
        "blearner_width_for_hit",
        "width_ratio",
    }.issubset(matched.columns)
    assert (tmp_path / "analysis" / "exp25_width_hit_matched.csv").is_file()

    figures = tmp_path / "analysis" / plots.FIGURES_DIRNAME
    for stem in ("sign_identification_frontier", "gamma_provenance",
                 "sign_cell_decomposition", "sign_confusion_heatmaps"):
        assert (figures / f"{stem}.pdf").is_file(), f"missing figure: {stem}"
    assert "Paired Naturalness-minus-competitor" in Path(
        result["report_path"]
    ).read_text()


def test_width_hit_matched_table_refuses_to_extrapolate() -> None:
    """Operating points off the ends of the swept grid read NaN, not a clamp."""

    common_support = pd.DataFrame(
        [
            # Inside the sweep on both axes: both readouts are available.
            {"beta": 1.0, "model": "naturalness", "n_common": 10,
             "mean_width": 0.30, "hit_ratio": 0.80},
            {"beta": 1.0, "model": "blearner[fixed_2]", "n_common": 10,
             "mean_width": 0.20, "hit_ratio": 0.70},
            {"beta": 1.0, "model": "blearner[fixed_3]", "n_common": 10,
             "mean_width": 0.40, "hit_ratio": 0.90},
            # Wider and more contained than any Gamma in the grid.
            {"beta": 0.1, "model": "naturalness", "n_common": 10,
             "mean_width": 0.90, "hit_ratio": 0.99},
            {"beta": 0.1, "model": "blearner[fixed_2]", "n_common": 10,
             "mean_width": 0.20, "hit_ratio": 0.70},
            {"beta": 0.1, "model": "blearner[fixed_3]", "n_common": 10,
             "mean_width": 0.40, "hit_ratio": 0.90},
        ]
    )
    matched = plots.width_hit_matched_table(common_support).set_index("beta")

    inside = matched.loc[1.0]
    assert inside["blearner_hit_at_width"] == pytest.approx(0.80)
    assert inside["blearner_width_for_hit"] == pytest.approx(0.30)
    assert inside["width_ratio"] == pytest.approx(1.0)

    outside = matched.loc[0.1]
    assert np.isnan(outside["blearner_hit_at_width"])
    assert np.isnan(outside["blearner_width_for_hit"])
    assert np.isnan(outside["width_ratio"])


def test_width_hit_frontier_by_study_draws_one_panel_per_study_and_beta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-study frontier is a beta-by-study grid restricted to the drawn betas."""

    captured: dict[str, object] = {}
    original = plots._save_figure

    def _capture(figure, output_stem, plot_formats):
        # _save_figure closes the figure, so the panels are inspected first.
        captured["titles"] = [
            (axis.get_title(), axis.title.get_fontweight()) for axis in figure.axes
        ]
        captured["row_labels"] = [
            [
                (text.get_text(), text.get_fontweight(), text.get_fontsize())
                for text in axis.texts
            ]
            for axis in figure.axes
        ]
        captured["suptitle"] = figure._suptitle
        captured["legend_fonts"] = [
            (text.get_fontsize(), text.get_fontweight())
            for text in figure.legends[0].get_texts()
        ]
        captured["legend_labels"] = [
            text.get_text() for text in figure.legends[0].get_texts()
        ]
        # The fixed-Gamma rows must leave no curve, markers, or Gamma labels.
        captured["artists"] = [
            (len(axis.lines), len(axis.collections), len(axis.texts))
            for axis in figure.axes
        ]
        return original(figure, output_stem, plot_formats)

    monkeypatch.setattr(plots, "_save_figure", _capture)

    rows = []
    for study in ("alpha", "beta_study", "gamma_study"):
        # 0.25 is present in the table but outside STUDY_FRONTIER_BETAS.
        for beta in (0.25, 0.5, 1.0):
            rows.append(
                {"study": study, "beta": beta, "model": "naturalness",
                 "n_common": 40, "mean_width": 0.40, "hit_ratio": 0.90}
            )
            for gamma, width, hit in (("2", 0.30, 0.80), ("3", 0.50, 0.95)):
                rows.append(
                    {"study": study, "beta": beta,
                     "model": f"blearner[fixed_{gamma}]", "n_common": 40,
                     "mean_width": width, "hit_ratio": hit}
                )
    written = plots.make_width_hit_frontier_by_study_plot(
        pd.DataFrame(rows), tmp_path
    )
    assert len(written) == 1
    assert written[0].name == "width_hit_frontier_by_study.pdf"

    # Two drawn betas x three studies; the 0.25 rows are excluded. Only the
    # top row is titled, with the study heading its column.
    assert captured["titles"] == [
        ("alpha", "bold"),
        ("beta_study", "bold"),
        ("gamma_study", "bold"),
        ("", "normal"),
        ("", "normal"),
        ("", "normal"),
    ]
    # The beta heads each row once, in bold, on the right-hand panel.
    beta, size = plots.BOLD_BETA_SYMBOL, plots.PAPER_RC["font.size"]
    assert captured["row_labels"] == [
        [], [], [(f"{beta} = 0.5", "bold", size)],
        [], [], [(f"{beta} = 1", "bold", size)],
    ]
    assert captured["suptitle"] is None
    assert all(
        size == pytest.approx(plots.PAPER_RC["legend.fontsize"])
        and weight == "bold"
        for size, weight in captured["legend_fonts"]
    )
    assert captured["legend_labels"] == ["Naturalness"]
    # One Naturalness scatter per panel; only the right column carries a text.
    assert captured["artists"] == [(0, 1, 0), (0, 1, 0), (0, 1, 1)] * 2

    assert plots.make_width_hit_frontier_by_study_plot(pd.DataFrame(), tmp_path) == []


def test_width_hit_frontier_by_study_labels_the_grid_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One centred label per axis, tick labels only on the grid's edge, at print size."""

    captured: dict[str, object] = {}

    def _capture(figure, output_stem, _plot_formats):
        """Record the grid's labelling instead of writing the figure."""

        captured["figure"] = (
            float(figure.get_size_inches()[0]),
            type(figure.get_layout_engine()).__name__,
        )
        captured["axis_labels"] = [
            (axis.get_xlabel(), axis.get_ylabel()) for axis in figure.axes
        ]
        captured["x_ticks"] = [bool(axis.get_xticklabels()) for axis in figure.axes]
        captured["y_ticks"] = [bool(axis.get_yticklabels()) for axis in figure.axes]
        captured["shared"] = [
            (figure._supxlabel.get_text(), figure._supxlabel.get_fontsize()),
            (figure._supylabel.get_text(), figure._supylabel.get_fontsize()),
        ]
        boxes = [axis.get_position() for axis in figure.axes]
        captured["centres"] = (
            figure._supxlabel.get_position()[0],
            (min(box.x0 for box in boxes) + max(box.x1 for box in boxes)) / 2,
            figure._supylabel.get_position()[1],
            (min(box.y0 for box in boxes) + max(box.y1 for box in boxes)) / 2,
        )
        # Distance in inches from each label to the outermost tick labels.
        renderer = figure.canvas.get_renderer()
        y_ticks = figure.axes[0].yaxis.get_tightbbox(renderer)
        x_ticks = figure.axes[-1].xaxis.get_tightbbox(renderer)
        captured["gaps"] = (
            (y_ticks.x0 - figure._supylabel.get_window_extent(renderer).x1)
            / figure.dpi,
            (x_ticks.y0 - figure._supxlabel.get_window_extent(renderer).y1)
            / figure.dpi,
        )
        plots.plt.close(figure)
        return []

    monkeypatch.setattr(plots, "_save_figure", _capture)
    rows = pd.DataFrame(
        [
            {"study": study, "beta": beta, "model": "naturalness",
             "n_common": 40, "mean_width": 0.40, "hit_ratio": 0.90}
            for study in ("alpha", "beta_study")
            for beta in plots.STUDY_FRONTIER_BETAS
        ]
    )

    plots.make_width_hit_frontier_by_study_plot(rows, tmp_path)

    n_rows = len(plots.STUDY_FRONTIER_BETAS)
    assert n_rows > 2
    # A manuscript figure: drawn at the printed width, by constrained layout.
    assert captured["figure"] == (plots.PAPER_FIGURE_WIDTH, "ConstrainedLayoutEngine")
    assert captured["axis_labels"] == [("", "")] * 2 * n_rows
    size = plots.PAPER_RC["figure.labelsize"]
    assert captured["shared"] == [("Mean interval width", size), ("Hit ratio", size)]
    # Two studies per row: bottom row keeps x ticks, left column keeps y ticks.
    assert captured["x_ticks"] == [False] * 2 * (n_rows - 1) + [True] * 2
    assert captured["y_ticks"] == [True, False] * n_rows
    x_label, x_centre, y_label, y_centre = captured["centres"]
    assert x_label == pytest.approx(x_centre)
    assert y_label == pytest.approx(y_centre)
    # Constrained layout keeps each label two layout pads clear of the tick numbers.
    pads = (
        plots.matplotlib.rcParams["figure.constrained_layout.w_pad"],
        plots.matplotlib.rcParams["figure.constrained_layout.h_pad"],
    )
    assert captured["gaps"] == pytest.approx((2 * pads[0], 2 * pads[1]), abs=1e-3)


def test_save_figure_embeds_truetype_fonts_in_pdf(tmp_path: Path) -> None:
    """PDF output embeds TrueType rather than Type 3 fonts."""

    figure, axis = plots.plt.subplots()
    axis.set_xlabel("Mean interval width")

    (path,) = plots._save_figure(figure, tmp_path / "figure", ("pdf",))

    # Matplotlib embeds Type 42 fonts as a Type 0 font over a CIDFontType2.
    content = path.read_bytes()
    assert b"/Subtype /CIDFontType2" in content
    assert b"/Subtype /Type3" not in content


def test_width_hit_frontier_by_beta_has_no_figure_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The beta-panel frontier leaves its former figure-title band empty."""

    captured: dict[str, object] = {}

    def _capture(figure, output_stem, _plot_formats):
        """Record each figure title instead of writing the figure."""

        captured[output_stem.name] = {
            "suptitle": figure._suptitle,
            "xlabels": [axis.get_xlabel() for axis in figure.axes],
            # The single-panel trajectory figure passes through here too.
            "shared_xlabel": (
                figure._supxlabel.get_text() if figure._supxlabel else None
            ),
            "legend_fonts": [
                (text.get_fontsize(), text.get_fontweight())
                for text in figure.legends[0].get_texts()
            ],
            "legend_labels": [
                text.get_text() for text in figure.legends[0].get_texts()
            ],
        }
        plots.plt.close(figure)
        return []

    monkeypatch.setattr(plots, "_save_figure", _capture)
    rows = pd.DataFrame(
        [
            {
                "beta": beta,
                "model": model,
                "mean_width": width,
                "hit_ratio": hit,
            }
            for beta in (0.1, 0.25, 0.5, 1.0)
            for model, width, hit in (
                ("naturalness", 0.4, 0.9),
                ("blearner[fixed_2]", 0.3, 0.8),
            )
        ]
    )

    plots.make_width_hit_frontier_plot(rows, tmp_path)

    frontier = captured["width_hit_frontier_by_beta"]
    assert frontier["suptitle"] is None
    # The panels share one figure-level x label rather than one each.
    assert frontier["xlabels"] == [""] * 4
    assert frontier["shared_xlabel"] == "Mean interval width"
    assert all(
        size == pytest.approx(plots.FRONTIER_LEGEND_TEXT_STYLE["size"])
        and weight == "bold"
        for size, weight in frontier["legend_fonts"]
    )
    # The fixed-Gamma grid is present in the input but not in the legend.
    assert frontier["legend_labels"] == ["Naturalness"]


def test_width_hit_frontier_by_beta_keeps_ticks_above_an_empty_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an odd beta count, the panel above the empty cell keeps x ticks."""

    captured: dict[str, list[bool]] = {}

    def _capture(figure, output_stem, _plot_formats):
        """Record which panels show x tick labels instead of saving."""

        captured["x_ticks"] = [
            bool(axis.get_xticklabels()) for axis in figure.axes if axis.axison
        ]
        plots.plt.close(figure)
        return []

    monkeypatch.setattr(plots, "_save_figure", _capture)
    rows = pd.DataFrame(
        [
            {"beta": beta, "model": "naturalness", "mean_width": 0.4, "hit_ratio": 0.9}
            for beta in (0.5, 1.0, 1.5)
        ]
    )

    plots.make_width_hit_frontier_by_beta_plot(rows, tmp_path)

    # A 2x2 grid with its bottom-right cell switched off: the top-right panel
    # ends its column, so it carries the column's tick labels.
    assert captured["x_ticks"] == [False, True, True]


def test_make_all_plots_rejects_bad_formats(tmp_path: Path) -> None:
    """An empty or malformed format list is refused before any figure is drawn."""

    annotated = _annotated_fixture()
    pooled = analysis.summarize_agreement(annotated, grouping=("beta",))
    confusion = analysis.build_sign_confusion(annotated)
    with pytest.raises(ValueError, match="simple file extensions"):
        plots.make_all_plots(
            annotated, pooled, confusion, tmp_path, plot_formats=()
        )


def test_run_analysis_persists_rows_and_renders_figures(tmp_path: Path) -> None:
    """The end-to-end run writes the row table and the figure set."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    _write_leaf(
        primary / analysis.BASELINE_VARIANTS_DIRNAME / "gamma_exp_beta",
        _interval_fixture(blearner_shift=0.1),
    )
    result = analysis.run_analysis(primary, tmp_path / "analysis")
    output_dir = Path(result["output_dir"])
    assert (output_dir / analysis.ANNOTATED_ROWS_FILENAME).is_file()
    assert len(result["figure_paths"]) >= len(EXPECTED_FIGURE_STEMS)
    assert (output_dir / plots.FIGURES_DIRNAME).is_dir()


def test_run_analysis_can_skip_figures(tmp_path: Path) -> None:
    """Disabling plots leaves the tables intact and writes no figures."""

    primary = _write_leaf(tmp_path / "primary", _interval_fixture())
    result = analysis.run_analysis(
        primary, tmp_path / "analysis", make_plots=False
    )
    output_dir = Path(result["output_dir"])
    assert result["figure_paths"] == []
    assert not (output_dir / plots.FIGURES_DIRNAME).exists()
    assert (output_dir / "exp25_hit_ratio_pooled.csv").is_file()


def _arm_strata(seed: int = 0) -> pd.DataFrame:
    """Return a small synthetic arms stratum table: two studies, two betas, three replications."""
    rng = np.random.default_rng(seed)
    rows = []
    for replication in range(3):
        for study in ("s1", "s2"):
            for beta in (1.0, 2.0):
                for stratum in range(4):
                    def row(arm, method, grounding="", delta=0.0, gamma=0.0):
                        width = float(rng.uniform(0.2, 0.9))
                        rows.append({"arm": arm, "method": method, "grounding": grounding, "delta": delta,
                                     "gamma": gamma, "replication": replication, "study": study, "beta": beta,
                                     "resp_age_bin": stratum, "width": width, "hit": bool(rng.random() < 0.8),
                                     "reversal": bool(rng.random() < 0.05)})
                    for delta in (0.0, 0.1):
                        row("effect_shift", "naturalness", "per_site", delta=delta)
                        for method in ("balke_pearl", "transport", "manski"):
                            row("effect_shift", method, delta=delta)
                    for rule in ("fixed_0.1_0.02", "fixed_0.1_0.05", "fixed_0.3_0.02", "exact", "prior"):
                        row("grounding_ablation", "naturalness", rule)
    return pd.DataFrame(rows)


def test_mechanism_arm_plots_write_the_available_figures(tmp_path: Path) -> None:
    """Each arm with rows gets its figures; an arm without rows (invalid instrument) is skipped."""
    paths = plots.make_mechanism_arm_plots(_arm_strata(), tmp_path, plot_formats=("png",), bootstrap_replicates=20)

    names = sorted(path.name for path in paths)
    assert names == [
        "effect_shift_validity.png", "effect_shift_validity_by_study.png",
        "grounding_ablation_frontier_by_beta.png", "grounding_ablation_frontier_by_study.png",
    ]
    assert all(path.exists() and path.parent == tmp_path / "figures" for path in paths)


def test_ablation_frontier_pins_points_below_the_floor(tmp_path: Path) -> None:
    """A rule whose hit ratio is below the axis floor is still drawn, at the floor."""
    strata = _arm_strata()
    strata.loc[strata["grounding"].eq("exact"), "hit"] = False
    paths = plots.make_grounding_ablation_frontier_plot(
        strata, tmp_path, facet="study", plot_formats=("png",), bootstrap_replicates=20, hit_floor=0.7)
    assert [path.name for path in paths] == ["grounding_ablation_frontier_by_study.png"]
    assert plots._parse_fixed_rule("fixed_0.3_0.05") == (0.3, 0.05)
    assert plots.make_modified_target_validity_plot(strata, tmp_path, arm="invalid_instrument") == []
