from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from pytest import MonkeyPatch

from experiments.naturalness_plotting import (
    build_exp16_plot_metric_summary_frame,
    write_exp16_plot_metric_summary_tables,
)
from experiments.utils import (
    make_metric_summary_latex_table,
    make_multi_metric_summary_latex_table,
    write_metric_summary_latex_table,
)


def test_make_metric_summary_latex_table_groups_mean_and_sd() -> None:
    """Coverage and width summaries are grouped and formatted as mean plus sd."""

    run_summary = pd.DataFrame(
        {
            "study": ["a", "a", "b"],
            "pattern": ["age", "age", "gender"],
            "rct_conformal_point_coverage": [1.0, 0.8, 0.5],
            "conformal_weighted_width": [0.4, 0.6, 1.0],
        }
    )

    table = make_metric_summary_latex_table(
        run_summary,
        group_columns=("study", "pattern"),
        digits=2,
    )

    assert table.loc[0, "Experiment"] == "a"
    assert table.loc[0, "Confounding variable"] == "age"
    assert table.loc[0, "N"] == 2
    assert table.loc[0, "Coverage"] == r"$0.90 \pm 0.14$"
    assert table.loc[0, "Width"] == r"$0.50 \pm 0.14$"
    assert table.loc[1, "Coverage"] == r"$0.50 \pm 0.00$"


def test_write_metric_summary_latex_table_writes_tex(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """LaTeX writer emits a table with unescaped math-mode summary cells."""

    monkeypatch.setattr(
        pd.DataFrame,
        "to_latex",
        Mock(side_effect=ImportError("Missing optional dependency 'Jinja2'.")),
    )
    run_summary = pd.DataFrame(
        {
            "study": ["a"],
            "rct_conformal_point_coverage": [True],
            "conformal_weighted_width": [0.75],
        }
    )
    output = tmp_path / "summary.tex"

    table = write_metric_summary_latex_table(
        run_summary,
        output,
        group_columns=("study",),
        digits=2,
    )

    assert not table.empty
    text = output.read_text(encoding="utf-8")
    assert r"$1.00 \pm 0.00$" in text
    assert r"$0.75 \pm 0.00$" in text


def test_make_multi_metric_summary_latex_table_keeps_metric_columns() -> None:
    """Multiple interval metrics are summarized without dropping valid columns."""

    run_summary = pd.DataFrame(
        {
            "study": ["a", "a"],
            "conformalized_lp_coverage": [True, False],
            "conformalized_lp_width": [0.2, 0.4],
            "posterior_mean_coverage": [True, True],
            "posterior_mean_width": [0.1, 0.3],
        }
    )

    table = make_multi_metric_summary_latex_table(
        run_summary,
        group_columns=("study",),
        metrics=(
            ("Conformalized LP coverage", "conformalized_lp_coverage"),
            ("Conformalized LP width", "conformalized_lp_width"),
            ("Posterior mean coverage", "posterior_mean_coverage"),
            ("Posterior mean width", "posterior_mean_width"),
        ),
        digits=2,
    )

    assert table.loc[0, "Experiment"] == "a"
    assert table.loc[0, "N"] == 2
    assert table.loc[0, "Conformalized LP coverage"] == r"$0.50 \pm 0.71$"
    assert table.loc[0, "Conformalized LP width"] == r"$0.30 \pm 0.14$"
    assert table.loc[0, "Posterior mean coverage"] == r"$1.00 \pm 0.00$"
    assert table.loc[0, "Posterior mean width"] == r"$0.20 \pm 0.14$"


def test_build_exp16_plot_metric_summary_frame_merges_plotted_strata() -> None:
    """Exp16 plot-metric summaries align LP and posterior rows by stratum."""

    key_columns = {
        "stage1_backend": ["hierarchical_dirichlet", "hierarchical_dirichlet"],
        "study": ["a", "a"],
        "pattern": ["age", "age"],
        "beta": [0.1, 0.1],
        "target_site": ["site1", "site1"],
        "resp_age_bin": [0, 1],
        "resp_gender_bin": [0, 0],
        "resp_polideo_bin": [1, 1],
    }
    target_lp_bounds = pd.DataFrame(
        {
            **key_columns,
            "covers_rct_cate_conformal": [True, False],
            "conformal_width": [0.5, 0.7],
            "conformal_status": ["optimal_expanded", "optimal_expanded"],
        }
    )
    posterior = pd.DataFrame(
        {
            **key_columns,
            "lower_mean": [-0.1, -0.2],
            "upper_mean": [0.2, 0.3],
            "covers_rct_cate_posterior_mean_interval": [True, True],
            "posterior_envelope_width": [0.6, 0.8],
            "covers_rct_cate_posterior_envelope": [True, False],
        }
    )

    summary = build_exp16_plot_metric_summary_frame(
        {
            "target_lp_bounds": target_lp_bounds,
            "target_gp_posterior_summary": posterior,
        }
    )

    assert len(summary) == 2
    assert summary["conformalized_lp_width"].tolist() == [0.5, 0.7]
    assert summary["posterior_mean_width"].round(6).tolist() == [0.3, 0.5]
    assert summary["posterior_envelope_coverage"].tolist() == [True, False]


def test_build_exp16_plot_metric_summary_frame_ignores_fallback_lp_rows() -> None:
    """Conformalized LP summaries ignore nonoptimal full-range fallback rows."""

    key_columns = {
        "stage1_backend": ["hierarchical_dirichlet"],
        "study": ["a"],
        "pattern": ["age"],
        "beta": [1.0],
        "target_site": ["site1"],
        "resp_age_bin": [0],
        "resp_gender_bin": [0],
        "resp_polideo_bin": [1],
    }
    target_lp_bounds = pd.DataFrame(
        {
            **key_columns,
            "covers_rct_cate_conformal": [True],
            "conformal_width": [2.0],
            "conformal_status": ["base_nonoptimal_full_range"],
        }
    )
    posterior = pd.DataFrame(
        {
            **key_columns,
            "lower_mean": [-0.1],
            "upper_mean": [0.2],
            "covers_rct_cate_posterior_mean_interval": [True],
            "posterior_envelope_width": [0.6],
            "covers_rct_cate_posterior_envelope": [True],
        }
    )

    summary = build_exp16_plot_metric_summary_frame(
        {
            "target_lp_bounds": target_lp_bounds,
            "target_gp_posterior_summary": posterior,
        }
    )

    assert len(summary) == 1
    assert summary["conformalized_lp_width"].isna().all()
    assert summary["conformalized_lp_coverage"].isna().all()
    assert summary["posterior_mean_width"].round(6).tolist() == [0.3]


def test_write_exp16_plot_metric_summary_tables_omits_backend_column(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Backend-specific Exp16 tables carry backend in caption, not columns."""

    monkeypatch.setattr(
        pd.DataFrame,
        "to_latex",
        Mock(side_effect=ImportError("Missing optional dependency 'Jinja2'.")),
    )
    summary = pd.DataFrame(
        {
            "stage1_backend": ["hierarchical_dirichlet", "hierarchical_dirichlet"],
            "study": ["a", "a"],
            "pattern": ["age", "age"],
            "beta": [0.1, 0.1],
            "conformalized_lp_coverage": [True, True],
            "conformalized_lp_width": [0.5, 0.7],
            "posterior_mean_coverage": [True, False],
            "posterior_mean_width": [0.3, 0.4],
            "posterior_envelope_coverage": [True, True],
            "posterior_envelope_width": [0.8, 0.9],
        }
    )

    write_exp16_plot_metric_summary_tables(summary, tmp_path)

    text = (tmp_path / "coverage_width_summary_hierarchical_dirichlet.tex").read_text(
        encoding="utf-8"
    )
    assert r"hierarchical\_dirichlet" in text
    assert "Backend &" not in text
    assert text.count(r"\begin{table}") == 2
    assert "target strata coverage" in text
    assert "target strata interval width" in text
    assert "Experiment & Confounding variable & Beta & N & Conformalized LP" in text
    assert "Conformalized LP coverage" not in text
    assert "Conformalized LP width" not in text
