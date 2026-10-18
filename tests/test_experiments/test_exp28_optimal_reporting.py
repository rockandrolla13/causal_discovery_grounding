"""Regression tests for Exp28 optimal-only report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments import naturalness_comparison_results as results


def _write_identity(root: Path) -> None:
    """Write the minimal identity metadata required by leaf discovery."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "identity.json").write_text(
        json.dumps(
            {
                "config": {
                    "experiment_name": "exp28_twins_method_comparison",
                    "stage1_backend": "tabicl_v2",
                }
            }
        )
    )


def _write_exp28_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    """Create one two-cell Exp28 condition split across two result passes."""
    naturalness_root = tmp_path / "naturalness"
    comparator_root = tmp_path / "comparators"
    naturalness_run = naturalness_root / "shard"
    comparator_run = comparator_root / "shard"
    _write_identity(naturalness_run)
    _write_identity(comparator_run)
    relative = Path("twins_reduced/dgpNone/seed1/beta0/target0")
    naturalness_leaf = naturalness_run / relative
    comparator_leaf = comparator_run / relative
    naturalness_leaf.mkdir(parents=True)
    comparator_leaf.mkdir(parents=True)

    pd.DataFrame(
        {
            "z": [0, 1],
            "lp_posterior_prob_optimal": [0.5, 0.0],
            "lower_mean": [-0.1, -0.4],
            "upper_mean": [0.1, 0.4],
            "width_mean": [0.2, 0.8],
        }
    ).to_parquet(
        naturalness_leaf / "posterior_conditional_on_optimal.parquet", index=False
    )

    naturalness_rows = []
    for method, widths in (
        ("manski", (1.0, 1.0)),
        ("naturalness", (0.2, 0.8)),
        ("balke_pearl", (0.8, 0.9)),
    ):
        for z, width, weight in zip((0, 1), widths, (0.6, 0.4)):
            naturalness_rows.append(
                {
                    "z": z,
                    "method": method,
                    "width": width,
                    "evaluation_weight": weight,
                    "reference_effect": 0.0,
                    "contains_reference": True,
                    "contains_paired_reference": True,
                }
            )
    pd.DataFrame(naturalness_rows).to_parquet(
        naturalness_leaf / "bounds_scored.parquet", index=False
    )

    comparator_rows = []
    specifications = {
        "manski": ((1.0, 1.0), (True, True)),
        "blearner_oracle": ((0.1, 0.2), (False, True)),
        "cb_ipw": ((0.7, 0.8), (True, True)),
        "cb_dr": ((0.75, 0.85), (True, True)),
    }
    for method, (widths, hits) in specifications.items():
        for z, width, hit in zip((0, 1), widths, hits):
            comparator_rows.append(
                {
                    "z": z,
                    "method": method,
                    "width": width,
                    "contains_reference": hit,
                    "contains_paired_reference": hit,
                }
            )
    pd.DataFrame(comparator_rows).to_parquet(
        comparator_leaf / "bounds_scored.parquet", index=False
    )
    return naturalness_root, comparator_root


def _reported_metrics() -> pd.DataFrame:
    """Return a minimal merged Exp28 metrics table for the synthetic condition."""
    return pd.DataFrame(
        [
            {
                "experiment": "exp28_twins_method_comparison",
                "dataset": "twins_reduced",
                "dgp_seed": None,
                "seed": 1,
                "beta": 0.0,
                "target_site": "0",
                "stage1_backend": "tabicl_v2",
                "method": "naturalness",
            }
        ]
    )


def _write_pass_metrics(root: Path, methods: tuple[str, ...]) -> None:
    """Write condition-level method metrics for one synthetic result pass."""
    rows = []
    for method in methods:
        rows.append(
            {
                "experiment": "exp28_twins_method_comparison",
                "dataset": "twins_reduced",
                "dgp_seed": None,
                "seed": 1,
                "beta": 0.0,
                "target_site": "0",
                "stage1_backend": "tabicl_v2",
                "method": method,
                "n_evaluation": 100,
                "n_cells": 2,
                "weighted_width": 1.0 if method == "manski" else 0.5,
                "ate_lower": -0.5,
                "ate_upper": 0.5,
                "ate_reference": 0.0,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "shard" / "metrics.parquet", index=False)


def test_optimal_only_metrics_use_identical_naturalness_weights(tmp_path):
    """Naturalness and every comparator must use the same selected cell mass."""
    roots = _write_exp28_artifacts(tmp_path)
    frame = results.load_optimal_only_metrics(list(roots)).set_index("method")

    assert set(frame.index) == set(results.EXP28_OPTIMAL_METHOD_ORDER)
    assert frame.loc["naturalness", "optimal_draw_mass"] == pytest.approx(0.3)
    assert frame.loc["naturalness", "weighted_width"] == pytest.approx(0.2)
    assert frame.loc["balke_pearl", "weighted_width"] == pytest.approx(0.8)
    assert frame.loc["cb_ipw", "weighted_width"] == pytest.approx(0.7)
    assert frame.loc["blearner_oracle", "reference_hit_ratio"] == pytest.approx(0.0)
    assert frame.loc["naturalness", "reference_hit_ratio"] == pytest.approx(1.0)


def test_exp28_optimal_width_table_reports_width_and_hit_ratio(tmp_path):
    """The paper table holds each method's mean width and hit ratio per beta."""
    roots = _write_exp28_artifacts(tmp_path / "raw")
    frame = results.load_optimal_only_metrics(list(roots))

    paths = results.write_exp28_optimal_width_table(frame, tmp_path / "tables")

    assert paths == [tmp_path / "tables" / results.EXP28_OPTIMAL_WIDTH_TABLE]
    lines = paths[0].read_text().splitlines()
    assert lines[0] == r"\begin{tabular}{lcccccc}"
    assert lines[2] == (
        r"$\beta$ & Naturalness & Manski & Balke--Pearl & Oracle B-Learner"
        r" & CB-IPW & CB-DR \\"
    )
    # Only cell 0 has optimal mass, so every method is scored on it alone.
    assert lines[4] == (
        r"0 & 0.20 (1.00) & 1.00 (1.00) & 0.80 (1.00) & 0.10 (0.00)"
        r" & 0.70 (1.00) & 0.75 (1.00) \\"
    )

    without_cb_dr = frame.loc[frame["method"].ne("cb_dr")]
    results.write_exp28_optimal_width_table(without_cb_dr, tmp_path / "partial")
    partial = (tmp_path / "partial" / results.EXP28_OPTIMAL_WIDTH_TABLE).read_text()
    assert partial.splitlines()[4].endswith(r"& -- \\")
    assert results.write_exp28_optimal_width_table(frame.iloc[0:0], tmp_path / "none") == []


def test_exp28_paper_width_figure_sets_its_legend_below_in_rows(tmp_path, monkeypatch):
    """The paper's Twins figure is drawn at text width, its legend below in two rows of three."""
    frame = results.load_optimal_only_metrics(list(_write_exp28_artifacts(tmp_path / "raw")))
    captured = {}

    def capture(fig, directory, stem):
        """Record the width figure's layout instead of writing it."""
        if stem == "exp28_optimal_width_by_beta":
            fig.draw_without_rendering()
            (legend,) = fig.legends
            captured["size"] = tuple(fig.get_size_inches())
            captured["labels"] = [text.get_text() for text in legend.get_texts()]
            captured["columns"] = len({round(text.get_window_extent().x0) for text in legend.get_texts()})
            captured["below"] = legend.get_window_extent().y1 <= fig.axes[0].get_tightbbox().y0
        results.plt.close(fig)
        return []

    monkeypatch.setattr(results, "_save", capture)
    results.make_exp28_optimal_only_plots(frame, tmp_path / "figures")

    assert captured["size"] == pytest.approx((results.PAPER_FIGURE_WIDTH, 4.0))
    assert captured["columns"] == 3 and captured["below"]
    # Matplotlib fills legend columns first, so with two rows the even entries
    # are the top row; the rows read in drawing order.
    labels = captured["labels"]
    present = set(frame["method"])
    drawn = [results.METHOD_LABELS[method] for method in results.METHOD_ORDER if method in present]
    assert labels[0::2] + labels[1::2] == drawn


def test_exp28_merge_persists_the_optimal_only_table(tmp_path):
    """A new merge must be self-contained for later report invocations."""
    naturalness_root, comparator_root = _write_exp28_artifacts(tmp_path / "raw")
    _write_pass_metrics(
        naturalness_root, ("manski", "naturalness", "balke_pearl")
    )
    _write_pass_metrics(
        comparator_root,
        ("manski", "blearner_oracle", "cb_ipw", "cb_dr"),
    )

    provenance = results.merge_trees(
        [naturalness_root, comparator_root], tmp_path / "merged"
    )

    compact = pd.read_parquet(
        tmp_path / "merged" / results.OPTIMAL_ONLY_METRICS_FILENAME
    )
    assert set(compact["method"]) == set(results.EXP28_OPTIMAL_METHOD_ORDER)
    assert provenance["optimal_only_rows"] == 6
    assert provenance["optimal_only_conditions"] == 1


def test_exp28_report_rebuilds_and_plots_optimal_only_metrics(
    tmp_path, monkeypatch
):
    """The report command must emit both plots even for a pre-change merge."""
    roots = _write_exp28_artifacts(tmp_path / "raw")
    merged = tmp_path / "merged"
    merged.mkdir()
    _reported_metrics().to_parquet(merged / "metrics.parquet", index=False)
    (merged / "merge_provenance.json").write_text(
        json.dumps({"roots": [str(root) for root in roots]})
    )

    skipped = (
        "make_width_by_beta_plot",
        "make_containment_by_beta_plot",
        "make_width_containment_plot",
        "make_gamma_coverage_plot",
        "make_sign_decision_plot",
        "write_summary_tables",
    )
    for name in skipped:
        monkeypatch.setattr(results, name, lambda *args, **kwargs: [])

    written = results.build_report(merged)

    compact = pd.read_parquet(merged / results.OPTIMAL_ONLY_METRICS_FILENAME)
    assert set(compact["method"]) == set(results.EXP28_OPTIMAL_METHOD_ORDER)
    for artifact in (
        "exp28_optimal_width_by_beta",
        "exp28_optimal_hit_ratio_by_beta",
    ):
        assert len(written[artifact]) == 2
        assert all(path.exists() for path in written[artifact])
    assert written["exp28_optimal_width_table"] == [
        merged / "tables" / results.EXP28_OPTIMAL_WIDTH_TABLE
    ]
