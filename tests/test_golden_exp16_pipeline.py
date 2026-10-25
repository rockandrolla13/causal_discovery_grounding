"""Exp16's per-target runner reproduces saved snapshots on synthetic sites.

These tests guard the numbers of ``run_one_target`` and of the per-draw target
probability tables while the code moves (the runner and ``Exp16Config`` into a
domain module, ``experiments/utils.py`` split up). After a move, update only
the import lines below. Regenerate the snapshots only for an intended change
of results, from the repository root:

    python -c "from tests.test_golden_exp16_pipeline import write_snapshots; write_snapshots()"

The fixture is a small OSRCT/ManyLabs-format frame: seven source sites and a
target ``pi``, three binary covariates and a hidden confounder. Every backend
is CPU-only (hierarchical-Dirichlet Stage 1, empirical or hierarchical-Dirichlet
target probabilities, scikit-learn GP surfaces); no TabICL, CATENets, numpyro
or network access. No number here is a paper result.

Each scenario snapshots every non-empty output frame of ``run_one_target``
under ``tests/golden/exp16_run_one_target/<scenario>/``. Text columns that
only repeat a Python ``repr`` (``z_value``, ``z_value_str``) or carry free-text
messages (``reason``, ``warnings``) are dropped; columns are sorted by name.
The scenarios with GP surfaces depend on the GP hyperparameter optimum, which
may move across BLAS builds; regenerate in the reference environment only.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from causal_grounding.lp_solver_learned_naturalness import (
    estimate_target_observed_binary_quantities,
)
from causal_grounding.train_target_split import create_train_target_split
from experiments.osrct_target_pipeline import Exp16Config, run_one_target
from experiments.run_config import build_lp_probability_model_kwargs

GOLDEN_DIR = Path(__file__).parent / "golden" / "exp16_run_one_target"
PMF_SNAPSHOT = GOLDEN_DIR / "hierarchical_dirichlet_draws_target_pmfs.csv"
FLOAT_FORMAT = "%.10g"
RTOL = 1e-6
ATOL = 1e-9
TARGET_SITE = "pi"
TARGET_INDEX = 7
DROPPED_COLUMNS = ("z_value", "z_value_str", "reason", "warnings")
PMF_DRAW_SEEDS = (0, 1, 2)
TWO_COVARIATES = ("resp_age_bin", "resp_gender_bin")

_COMMON: dict[str, Any] = dict(
    study="s", studies=("s",), pattern="p", patterns=("p",), beta=1.0, betas=(1.0,),
    target_sites=(TARGET_SITE,),
    stage1_backend="hierarchical_dirichlet",
    validation_stage1_backend="hierarchical_dirichlet",
    policy_top_k_values=(1, 2),
    policy_tuning_fraction=0.3, min_policy_tuning_sites=2,
    calibration_fraction=0.25, min_calibration_sites=2,
    conformal_alpha=0.5, rct_min_arm_count=5, min_cell_count=2,
    lp_posterior_n_jobs=1, make_plots=False, verbose=False,
)
_GP_DRAWS: dict[str, Any] = dict(
    target_trigger_probability_backend="hierarchical_dirichlet",
    lp_probability_backend="hierarchical_dirichlet_draws",
    fit_gp=True,
    epsilons=(0.05, 0.2, 0.5),
    taus=(0.01, 0.1, 0.3),
)
#: Scenario name -> Exp16Config overrides on top of ``_COMMON``.
SCENARIOS: dict[str, dict[str, Any]] = {
    # Direct LP on empirical target probabilities; top-k tuning and
    # endpoint calibration on the full paper epsilon/tau grid.
    "empirical_direct_lp": dict(
        target_trigger_probability_backend="empirical",
        lp_probability_backend="empirical",
    ),
    # GP tolerance draws joined with per-draw hierarchical-Dirichlet target
    # probabilities; top-k tuning and endpoint calibration.
    "gp_hierarchical_dirichlet_draws": dict(
        **_GP_DRAWS, lp_posterior_samples=10,
        candidate_selection_mode="best_per_covariate",
    ),
    # Extra candidate in its own state: per-state target tables and
    # intersected LPs, for the point solve and for every posterior draw.
    "gp_candidate_states": dict(
        **_GP_DRAWS, lp_posterior_samples=5,
        candidate_selection_mode="all_triggered",
        covariates=TWO_COVARIATES,
        naturalness_covariates=TWO_COVARIATES,
        lp_probability_covariates=TWO_COVARIATES,
        naturalness_extra_candidates=("resp_polideo_bin",),
    ),
}


def synthetic_osrct(
    seed: int = 0, n_sites: int = 8, n_per_site: int = 150
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return OSRCT-style idle rows and randomized rows with a hidden confounder."""
    rng = np.random.default_rng(seed)
    frames, participant = [], 0
    for site in [f"s{index:02d}" for index in range(n_sites - 1)] + [TARGET_SITE]:
        n_rows = n_per_site * (2 if site == TARGET_SITE else 1)
        hidden = rng.normal(size=n_rows)
        age = rng.binomial(1, 0.5, n_rows)
        treatment = rng.binomial(1, 0.5, n_rows)
        frames.append(pd.DataFrame({
            "site": site,
            "id": np.arange(participant, participant + n_rows, dtype=float),
            "iv": treatment,
            "dv_binary": rng.binomial(
                1, 1 / (1 + np.exp(-(-0.5 + 0.8 * treatment + 0.4 * age + 0.5 * hidden)))
            ),
            "resp_age_bin": age,
            "resp_gender_bin": rng.binomial(1, 0.5, n_rows),
            "resp_polideo_bin": rng.binomial(1, 0.4, n_rows),
            "hidden": hidden,
        }))
        participant += n_rows
    rct = pd.concat(frames, ignore_index=True)
    selection = 1 / (1 + np.exp(-(-0.2 + 1.2 * rct["resp_age_bin"] + 0.8 * rct["hidden"])))
    keep = rng.binomial(1, selection) == rct["iv"]
    rct = rct.drop(columns="hidden")
    return rct.loc[keep].reset_index(drop=True), rct


def scenario_config(name: str) -> Exp16Config:
    """Return the Exp16 configuration of one snapshot scenario."""
    return Exp16Config(**_COMMON, **SCENARIOS[name])


def scenario_frames(name: str) -> dict[str, pd.DataFrame]:
    """Run one target and return its non-empty output frames, snapshot-ready."""
    osrct, rct = synthetic_osrct()
    outputs = run_one_target(
        osrct_data=osrct, rct_data=rct, target_site=TARGET_SITE,
        target_index=TARGET_INDEX, config=scenario_config(name), base_metadata={},
    )
    frames = {}
    for frame_name, frame in outputs.items():
        if frame.empty:
            continue
        kept = frame.drop(columns=[c for c in DROPPED_COLUMNS if c in frame.columns])
        frames[frame_name] = kept[sorted(kept.columns)].reset_index(drop=True)
    return frames


def target_pmf_draws() -> pd.DataFrame:
    """Return target probability tables: posterior mean and three seeded draws.

    Uses Exp16's own model options for the hierarchical-Dirichlet backends
    (``build_lp_probability_model_kwargs``); ``draw_seed`` is -1 for the
    deterministic ``hierarchical_dirichlet`` table.
    """
    osrct, rct = synthetic_osrct()
    _, target_idle = create_train_target_split(
        osrct_data=osrct, rct_data=rct, target_site=TARGET_SITE,
        site_col="site", regime_col="F",
    )
    config = scenario_config("gp_hierarchical_dirichlet_draws")
    covariates = list(config.covariates)
    requests = [("hierarchical_dirichlet", -1)] + [
        ("hierarchical_dirichlet_draws", seed) for seed in PMF_DRAW_SEEDS
    ]
    rows = []
    for backend, seed in requests:
        kwargs = dict(build_lp_probability_model_kwargs(config, backend) or {})
        if seed >= 0:
            kwargs["posterior_sample_seed"] = seed
        table = estimate_target_observed_binary_quantities(
            target_idle, treatment="iv", outcome="dv_binary",
            covariates=covariates, model_covariates=covariates,
            probability_backend=backend, min_count=int(config.min_cell_count),
            probability_model_kwargs=kwargs,
        )
        for z in table.z_values:
            rows.append({
                "backend": backend, "draw_seed": seed,
                **{name: int(value) for name, value in zip(covariates, z)},
                "p_x1": table.p_x1[z], "p_x0": table.p_x0[z],
                "p_y1_given_x0": table.p_y1_given_xz[(0, z)],
                "p_y1_given_x1": table.p_y1_given_xz[(1, z)],
                "n_z": table.n_z[z],
                "n_x0z": table.n_xz.get((0, z), 0), "n_x1z": table.n_xz.get((1, z), 0),
            })
    return pd.DataFrame(rows)


def write_snapshots() -> None:
    """Regenerate every snapshot of this module from the current code."""
    for name in SCENARIOS:
        directory = GOLDEN_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*.csv"):
            old.unlink()
        for frame_name, frame in scenario_frames(name).items():
            frame.to_csv(directory / f"{frame_name}.csv", index=False, float_format=FLOAT_FORMAT)
    target_pmf_draws().to_csv(PMF_SNAPSHOT, index=False, float_format=FLOAT_FORMAT)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_run_one_target_matches_golden_snapshot(scenario, golden_round_trip):
    """Every output frame of one target run equals its snapshot, to float tolerance."""
    frames = scenario_frames(scenario)
    directory = GOLDEN_DIR / scenario
    expected_names = sorted(path.stem for path in directory.glob("*.csv"))
    assert sorted(frames) == expected_names, "set of non-empty output frames changed"
    for frame_name in expected_names:
        expected = pd.read_csv(directory / f"{frame_name}.csv")
        try:
            pd.testing.assert_frame_equal(
                golden_round_trip(frames[frame_name]), expected,
                check_exact=False, rtol=RTOL, atol=ATOL,
            )
        except AssertionError as error:
            raise AssertionError(f"{scenario}/{frame_name}: {error}") from error


def test_target_pmf_draws_match_golden_snapshot(golden_round_trip):
    """Per-draw hierarchical-Dirichlet target probabilities equal the snapshot."""
    expected = pd.read_csv(PMF_SNAPSHOT)
    pd.testing.assert_frame_equal(
        golden_round_trip(target_pmf_draws()), expected,
        check_exact=False, rtol=RTOL, atol=ATOL,
    )
