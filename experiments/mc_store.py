"""Read an Exp25 Monte Carlo baseline-comparison store and write its derived tables.

Ablations that rerun part of Exp25 (Exp32's noise candidates, the tau-floor regrounding)
read the scientific configuration a store recorded, reuse the Exp16 target index
of each condition, and write Parquet tables whose object columns may mix types.

:class:`~experiments.exp25_osrct_mc_baseline_comparison.Exp25Config` still lives in
the Exp25 script, so this module imports that one script for the class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.exp25_osrct_mc_baseline_comparison import Exp25Config
from experiments.osrct_data_utils import OSRCTConfig, iter_base_configs


def load_exp25_store_config(store: Path) -> Exp25Config:
    """Rebuild the scientific Exp25 configuration recorded in a store.

    Args:
        store: Exp25 backend leaf containing ``run_manifest.yml``.

    Returns:
        Exp25 configuration with the store's seeds, grid and backends. The
        RCT path is this checkout's copy of the ManyLabs pickle.

    Raises:
        FileNotFoundError: If the store has no manifest or replications.
    """

    manifest_path = store / "run_manifest.yml"
    if not manifest_path.exists() or not (store / "replications").is_dir():
        raise FileNotFoundError(
            f"{store} is not an Exp25 backend leaf (run_manifest.yml and replications/)"
        )
    recorded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["config"]

    def flag(value: Any) -> bool:
        """Parse a manifest boolean written as text."""

        return str(value).strip().lower() in {"1", "true", "yes"}

    return Exp25Config(
        n_replications=int(recorded["n_replications"]),
        base_seed=int(recorded["base_seed"]),
        studies=tuple(str(value) for value in recorded["studies"]),
        bias_covariates=tuple(str(value) for value in recorded["bias_covariates"]),
        betas=tuple(float(value) for value in recorded["betas"]),
        target_site=str(recorded["target_site"]),
        require_observed_gender=flag(recorded["require_observed_gender"]),
        stage1_backend=str(recorded["stage1_backend"]),
        target_trigger_probability_backend=str(
            recorded["target_trigger_probability_backend"]
        ),
        tabicl_device=str(recorded["tabicl_device"]),
        lp_probability_backend=str(recorded["lp_probability_backend"]),
        lp_posterior_samples=int(recorded["lp_posterior_samples"]),
        lp_posterior_n_jobs=int(recorded["lp_posterior_n_jobs"]),
    )


def exp25_target_index(
    exp25: Exp25Config,
    *,
    study: str,
    pattern: str,
    beta: float,
) -> int:
    """Return the Exp16 target index Exp25 used for one condition.

    Exp16 seeds source roles with ``seed + target_index`` and GP draws with
    ``seed + 20000 + target_index``, where the index enumerates the full
    study/pattern/beta grid. Reusing it keeps each ablation run paired with
    the stored Exp25 fit.

    Args:
        exp25: Store configuration.
        study: ManyLabs study.
        pattern: OSRCT selection pattern.
        beta: OSRCT selection strength.

    Returns:
        ``base_index * 10000`` for the single Exp25 target site.

    Raises:
        ValueError: If the condition is not part of the store's grid.
    """

    probe = OSRCTConfig(
        studies=tuple(exp25.studies),
        patterns=tuple(exp25.bias_covariates),
        betas=tuple(exp25.betas),
    )
    for base_index, base in enumerate(iter_base_configs(probe)):
        if (
            base.study == study
            and base.pattern == pattern
            and np.isclose(float(base.beta), float(beta))
        ):
            return int(base_index * 10_000)
    raise ValueError(f"Condition {study}/{pattern}/beta={beta} is not in the Exp25 store")


def parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose object columns hold only Parquet-friendly scalars.

    Object columns holding containers, such as LP stratum-key tuples, or
    mixed scalar types are written as text; single-type columns are kept.
    """

    def missing(value: Any) -> bool:
        """Return whether a cell is empty."""

        return value is None or (isinstance(value, float) and np.isnan(value))

    def kind(value: Any) -> str:
        """Classify a present cell by the Parquet type it would need."""

        if isinstance(value, str):
            return "text"
        if isinstance(value, (bool, np.bool_)):
            return "bool"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return "number"
        return "other"

    result = frame.copy()
    for column in result.columns:
        if result[column].dtype != object:
            continue
        values = result[column]
        kinds = {kind(value) for value in values if not missing(value)}
        if len(kinds) <= 1 and "other" not in kinds:
            continue
        result[column] = values.map(
            lambda value: value if missing(value) else str(value)
        )
    return result
