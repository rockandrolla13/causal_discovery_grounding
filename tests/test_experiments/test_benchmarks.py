"""Tests for shared benchmark helpers."""

from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# experiments.benchmarks imports the Exp16-Exp25 cluster ML stack, which is only
# installed by requirements.txt. Skip instead of aborting collection.
pytest.importorskip("torch")
pytest.importorskip("catenets")
pytest.importorskip("sklearn_quantile")

import experiments.benchmarks as benchmarks
from experiments.benchmarks import (
    EXP_BETA_GAMMA,
    BLearnerFitTask,
    BenchmarkFitConfig,
    FitTask,
    atomic_write_parquet,
    blearner_gammas,
    build_mean_bounds_comparison,
    build_run_diagnostics,
    commit_replication,
    comparison_to_long_intervals,
    draw_width_violin_panel,
    load_existing_outputs,
    parse_gamma,
    run_blearner_fit_tasks,
    validate_blearner_repository,
)


def _comparison() -> pd.DataFrame:
    """Return one complete wide Exp18 comparison row."""

    return pd.DataFrame(
        {
            "study": ["study"],
            "target_site": ["target"],
            "stratum": [0],
            "n_target_idle_rows": [8],
            "rct_cate": [0.1],
            "rct_cate_ci90_lower": [0.0],
            "rct_cate_ci90_upper": [0.2],
            "rct_cate_available": [True],
            "blackbox_reference_cate": [0.15],
            "naturalness_lower": [-0.2],
            "naturalness_upper": [0.3],
            "lp_posterior_prob_optimal": [1.0],
            "blearner_lower": [-1.2],
            "blearner_upper": [1.1],
            "blearner_gamma": [2.0],
            "outside_binary_cate_support": [True],
            "cb_ipw_lower": [-0.4],
            "cb_ipw_upper": [0.5],
            "cb_ipw_used_full_range_fallback": [False],
            "cb_ipw_interval_inverted": [False],
            "cb_dr_lower": [-0.3],
            "cb_dr_upper": [0.4],
            "cb_dr_used_full_range_fallback": [False],
            "cb_dr_interval_inverted": [False],
        }
    )


def test_shared_normalization_preserves_method_validity() -> None:
    """Shared normalization clips B-Learner support and retains CB validity."""

    result = comparison_to_long_intervals(
        _comparison(),
        methods=("naturalness", "blearner", "cb_ipw", "cb_dr"),
        identity_columns=(
            "study",
            "stratum",
            "target_site",
            "n_target_idle_rows",
            "rct_cate",
            "rct_cate_ci90_lower",
            "rct_cate_ci90_upper",
            "rct_cate_available",
            "blackbox_reference_cate",
        ),
        output_reference_columns={
            "blackbox_reference_cate": "model_implied_cate"
        },
        replication=0,
        seed=11,
        bias_covariate="stratum",
        beta=1.0,
        config_fingerprint_value="fingerprint",
        experiment_name="test",
    )

    blearner = result.loc[result["method"].eq("blearner")].iloc[0]
    assert (blearner["lower"], blearner["upper"]) == (-1.0, 1.0)
    assert bool(blearner["support_clipped"])
    assert result["is_valid"].all()
    assert result["model_implied_cate"].eq(0.15).all()
    assert "blackbox_reference_cate" not in result

    diagnostics = build_run_diagnostics(
        result,
        grouping_columns=(
            "config_fingerprint",
            "replication",
            "seed",
            "bias_covariate",
            "beta",
            "study",
            "method",
        ),
        total_label="strata",
        runtime_seconds=2.0,
    )
    assert diagnostics["n_total_strata"].eq(1).all()
    assert diagnostics.loc[
        diagnostics["method"].eq("blearner"), "n_support_clipped"
    ].iloc[0] == 1


def test_shared_normalization_accepts_blearner_only_comparison() -> None:
    """B-Learner-only replay should not require Naturalness or CB columns."""

    comparison = _comparison().loc[
        :,
        [
            "study",
            "stratum",
            "target_site",
            "n_target_idle_rows",
            "rct_cate",
            "blearner_lower",
            "blearner_upper",
            "blearner_gamma",
            "outside_binary_cate_support",
        ],
    ]
    result = comparison_to_long_intervals(
        comparison,
        methods=("blearner",),
        identity_columns=(
            "study",
            "stratum",
            "target_site",
            "n_target_idle_rows",
            "rct_cate",
        ),
        replication=0,
        seed=11,
        bias_covariate="stratum",
        beta=1.0,
        config_fingerprint_value="fingerprint",
        experiment_name="test",
    )

    assert result["method"].tolist() == ["blearner"]
    assert result["blearner_gamma"].tolist() == [2.0]


def test_shared_gamma_parsing_and_exp_beta_resolution() -> None:
    """Shared Gamma helpers retain Exp18's accepted strategy semantics."""

    assert parse_gamma("2.5") == 2.5
    assert blearner_gammas(
        EXP_BETA_GAMMA,
        beta=1.0,
        gamma_search_grid=(1.0, 2.0),
    ) == (2.718281828459045,)


def _benchmark_fit_config(n_jobs: int) -> BenchmarkFitConfig:
    """Return minimal shared fit settings for B-Learner batch execution."""

    return BenchmarkFitConfig(
        seed=7,
        n_jobs=n_jobs,
        worker_threads=1,
        n_layers_out=1,
        n_units_out=1,
        n_iter=1,
        batch_size=1,
        validation_fraction=0.3,
        n_iter_print=1,
        blearner_repository="/unused",
    )


def _blearner_fit_tasks() -> list[BLearnerFitTask]:
    """Return two independent single-study B-Learner tasks."""

    return [
        BLearnerFitTask(
            task_id=index,
            study="study",
            gamma=gamma,
            x=np.zeros((4, 1)),
            treatment=np.array([0, 1, 0, 1]),
            outcome=np.array([0.0, 1.0, 1.0, 0.0]),
            x_target=np.zeros((2, 1)),
        )
        for index, gamma in enumerate((1.5, 3.0))
    ]


def test_blearner_fits_are_single_threaded_and_keyed_by_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batched B-Learner fits pin forest threads and keep their task ids."""

    requested: list[tuple[float, int]] = []

    class FakeBLearner:
        """Learner returning Gamma-specific endpoints without fitting."""

        def __init__(self, gamma: float) -> None:
            self.gamma = gamma

        def fit(self, x: np.ndarray, a: np.ndarray, y: np.ndarray) -> None:
            """Accept the fitting arrays without fitting a model."""

            del x, a, y

        def effect(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return one Gamma-scaled interval per target row."""

            return -self.gamma * np.ones(len(x)), self.gamma * np.ones(len(x))

    def fake_make_blearner(
        config: BenchmarkFitConfig,
        gamma: float,
        *,
        model_n_jobs: int = -1,
    ) -> FakeBLearner:
        """Record the requested Gamma and forest worker count."""

        del config
        requested.append((float(gamma), int(model_n_jobs)))
        return FakeBLearner(float(gamma))

    monkeypatch.setattr(benchmarks, "make_blearner", fake_make_blearner)

    results = run_blearner_fit_tasks(
        _blearner_fit_tasks(),
        _benchmark_fit_config(n_jobs=1),
    )

    assert sorted(requested) == [(1.5, 1), (3.0, 1)]
    assert set(results) == {0, 1}
    assert results[0].lower.tolist() == [-1.5, -1.5]
    assert results[1].upper.tolist() == [3.0, 3.0]
    assert all(result.runtime_seconds >= 0.0 for result in results.values())


def test_empty_blearner_batch_starts_no_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty B-Learner batch returns without touching worker settings."""

    def fail(*args: object, **kwargs: object) -> None:
        """Fail if the runner configures workers for an empty batch."""

        raise AssertionError("empty batch must not configure workers")

    monkeypatch.setattr(benchmarks, "_initialize_blearner_worker", fail)

    assert run_blearner_fit_tasks([], _benchmark_fit_config(n_jobs=4)) == {}


def test_fit_task_validates_learner_specific_inputs() -> None:
    """Shared task objects should reject incomplete WB and CB inputs."""

    arrays = {
        "x_left": np.zeros((2, 1)),
        "a_left": np.zeros(2),
        "y_left": np.zeros(2),
        "x_target": np.zeros((1, 1)),
    }
    with pytest.raises(ValueError, match="WB tasks require environment"):
        FitTask(
            task_id=0,
            study="study",
            kind="wb",
            endpoint="lower",
            seed=1,
            **arrays,
        )
    with pytest.raises(ValueError, match="right-environment arrays"):
        FitTask(
            task_id=1,
            study="study",
            kind="cb",
            endpoint="upper",
            seed=1,
            method="cb_dr",
            left_environment="left",
            right_environment="right",
            **arrays,
        )


def test_repository_validation_uses_one_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repeated validation should query the pinned checkout only once."""

    repository = tmp_path / "BLearner"
    module = repository / "models" / "blearner" / "BLearner.py"
    module.parent.mkdir(parents=True)
    module.touch()
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        """Return the pinned commit and record the command."""

        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=f"{benchmarks.BLEARNER_COMMIT}\n",
            stderr="",
        )

    monkeypatch.setattr(benchmarks.subprocess, "run", fake_run)
    benchmarks._VALIDATED_BLEARNER_REPOSITORIES.discard(repository.resolve())

    validate_blearner_repository(repository)
    validate_blearner_repository(repository)

    assert len(calls) == 1
    assert repository.resolve() in benchmarks._VALIDATED_BLEARNER_REPOSITORIES


def test_shared_parquet_resume_and_replication_replacement(
    tmp_path: Path,
) -> None:
    """Shared persistence should recover orphans and replace one replication."""

    pytest.importorskip("pyarrow")
    intervals_path = tmp_path / "intervals.parquet"
    orphan = pd.DataFrame(
        {"config_fingerprint": ["fingerprint"], "replication": [0]}
    )
    atomic_write_parquet(orphan, intervals_path)

    intervals, diagnostics = load_existing_outputs(
        tmp_path,
        intervals_filename=intervals_path.name,
        diagnostics_filename="diagnostics.parquet",
        fingerprint="fingerprint",
        resume=True,
        experiment_name="Test",
    )
    assert intervals.empty and diagnostics.empty
    assert not intervals_path.exists()

    existing_intervals = pd.DataFrame(
        {
            "config_fingerprint": ["fingerprint", "fingerprint"],
            "replication": [0, 1],
            "value": [1, 2],
        }
    )
    existing_diagnostics = existing_intervals.rename(
        columns={"value": "diagnostic"}
    )
    replacement_intervals = existing_intervals.iloc[[0]].assign(value=3)
    replacement_diagnostics = existing_diagnostics.iloc[[0]].assign(
        diagnostic=4
    )
    committed, committed_diagnostics = commit_replication(
        tmp_path,
        existing_intervals,
        existing_diagnostics,
        replacement_intervals,
        replacement_diagnostics,
        interval_sort_columns=("replication",),
        diagnostics_sort_columns=("replication",),
        intervals_filename=intervals_path.name,
        diagnostics_filename="diagnostics.parquet",
    )
    assert committed.set_index("replication").loc[0, "value"] == 3
    assert committed_diagnostics.set_index("replication").loc[
        0, "diagnostic"
    ] == 4

    with pytest.raises(ValueError, match="different Test config"):
        load_existing_outputs(
            tmp_path,
            intervals_filename=intervals_path.name,
            diagnostics_filename="diagnostics.parquet",
            fingerprint="other",
            resume=True,
            experiment_name="Test",
        )


def test_shared_width_panel_and_mean_aggregation() -> None:
    """Shared result helpers should retain plot and mean-bound semantics."""

    rows = []
    for replication, width in enumerate((0.4, 0.6)):
        for method in ("naturalness", "blearner"):
            rows.append(
                {
                    "replication": replication,
                    "bias_covariate": "gender",
                    "beta": 1.0,
                    "study": "study",
                    "method": method,
                    "lower": -width / 2,
                    "upper": width / 2,
                    "width": width,
                    "is_valid": True,
                    "rct_cate": 0.1,
                }
            )
    intervals = pd.DataFrame(rows)
    figure, axis = plt.subplots()
    draw_width_violin_panel(
        axis,
        intervals,
        methods=("naturalness", "blearner"),
        method_labels={"naturalness": "Naturalness", "blearner": "B-Learner"},
        method_colors={"naturalness": "tab:purple", "blearner": "tab:red"},
        maximum=2.0,
        axis_label="CATE interval width",
    )
    assert axis.get_xlabel() == "CATE interval width"
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "Naturalness",
        "B-Learner",
    ]
    assert "Naturalness: 4/4" not in axis.texts[0].get_text()
    assert "Naturalness: 2/2" in axis.texts[0].get_text()
    plt.close(figure)

    comparison = build_mean_bounds_comparison(
        intervals,
        bias_covariate="gender",
        beta=1.0,
        keys=("study",),
        reference_columns=("rct_cate",),
        reference_label="RCT reference",
        methods=("naturalness", "blearner"),
    )
    assert comparison.loc[0, "naturalness_lower"] == pytest.approx(-0.25)
    assert comparison.loc[0, "blearner_upper"] == pytest.approx(0.25)
