"""OSRCT data-preparation helpers used by the calibrated-policy experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from causal_grounding.loaders import RawDataset, load_sites
from experiments.manylabs_cohort import (
    OBSERVED_GENDER_COHORT_RULE,
    restrict_to_observed_gender,
)
from experiments.utils import (
    DEFAULT_AGE_BINARY_THRESHOLD,
    DEFAULT_POLIDEO_BINARY_THRESHOLD,
    add_binary_covariates,
    add_naturalness_columns,
    expand_target_sites,
    find_osrct_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Columns both loaders derive for their own validation (``_canonicalise``); dropped
#: after reading so ``add_naturalness_columns``/``add_binary_covariates`` append them.
_LOADER_DERIVED_COLUMNS = (
    "dv_binary",
    "dv_positive_value",
    "resp_gender_bin",
    "resp_age_bin",
    "resp_polideo_bin",
)


@dataclass
class OSRCTConfig:
    """Shared configuration fields for OSRCT/RCT naturalness experiments."""

    study: str = "gainloss"
    studies: tuple[str, ...] | None = None
    pattern: str = "age"
    patterns: tuple[str, ...] | None = None
    beta: float = 0.5
    betas: tuple[float, ...] | None = None
    seed: int = 42
    target_sites: tuple[str, ...] = ("lse",)

    data_root: str = str(
        PROJECT_ROOT / "osrct_benchmark_v1.0" / "confounded_datasets" / "by_study"
    )
    rct_path: str = str(
        PROJECT_ROOT / "ManyLabs1" / "pre-process" / "Manylabs1_data.pkl"
    )

    epsilons: tuple[float, ...] = (
        0.01,
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.50,
    )
    taus: tuple[float, ...] = (0.01, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    covariates: tuple[str, ...] = (
        "resp_age_bin",
        "resp_gender_bin",
        "resp_polideo_bin",
    )
    age_bins: int = 3
    age_binary_threshold: float = DEFAULT_AGE_BINARY_THRESHOLD
    polideo_binary_threshold: float = DEFAULT_POLIDEO_BINARY_THRESHOLD
    min_cell_count: int = 2
    max_z_minus_samples: int | None = None
    require_observed_gender: bool = False

    alpha: float = 0.5
    min_overlap: float = 0.02
    stage1_backend: str = "empirical"
    validation_stage1_backend: str | None = None
    tabicl_device: str | None = None
    fit_gp: bool = False

    target_policy_uncertainty: str = "posterior_mean"
    tolerance_uncertainty: str = "posterior_mean"
    run_prediction_lp: bool = True
    lp_solver_method: str = "highs"
    lp_probability_backend: str = "stage1"
    lp_posterior_samples: int = 0

    disjoint_validation_ids: bool = False
    id_col: str = "id"
    max_target_sites: int | None = None
    fail_fast: bool = False

    output_dir: str = str(PROJECT_ROOT / "results" / "osrct_data")
    make_plots: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        """Fill vector-valued sweep defaults and validate posterior samples."""

        if self.studies is None:
            self.studies = (self.study,)
        if self.patterns is None:
            self.patterns = (self.pattern,)
        if self.betas is None:
            self.betas = (self.beta,)
        if self.validation_stage1_backend is None:
            self.validation_stage1_backend = self.stage1_backend
        if self.tabicl_device is not None and not (
            self.tabicl_device in {"cpu", "cuda", "mps"}
            or (
                self.tabicl_device.startswith("cuda:")
                and self.tabicl_device.removeprefix("cuda:").isdigit()
            )
        ):
            raise ValueError(
                "tabicl_device must be None, 'cpu', 'cuda', 'cuda:<index>', "
                "or 'mps'"
            )
        if self.lp_posterior_samples < 0:
            raise ValueError("lp_posterior_samples must be non-negative")


def iter_base_configs(config: OSRCTConfig) -> Iterable[OSRCTConfig]:
    """Yield one concrete config for each study, confounding pattern, and beta."""

    assert config.studies is not None
    assert config.patterns is not None
    assert config.betas is not None
    for study in config.studies:
        for pattern in config.patterns:
            for beta in config.betas:
                yield replace(config, study=study, pattern=pattern, beta=float(beta))


def load_prepared_data(
    config: OSRCTConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load one OSRCT/RCT pair and add naturalness-ready covariates.

    The RCT file and ``.csv`` OSRCT files are read through the validated loaders
    (:class:`~causal_grounding.loaders.osrct.OSRCTLoader` and
    :class:`~causal_grounding.loaders.manylabs_rct.ManyLabsRCTLoader`), which
    read only csv. ``.parquet`` and ``.pkl`` OSRCT files are sampled realisation
    artifacts (Exp25 writes its regenerated OSRCT grid as parquet, and
    ``find_osrct_file`` prefers ``.parquet``); they are read directly and skip
    the loader's validation.

    The pairing lives here, not in ``loaders/``, because the age bins of
    ``add_naturalness_columns`` are computed on both files pooled. The loaders'
    derived binary columns are dropped and recomputed by
    ``add_naturalness_columns``/``add_binary_covariates``, so every derived
    column is appended in a fixed order. The observed-gender cohort filter is
    applied here rather than through the loaders, so its per-site diagnostics
    reach the returned metadata.
    """

    osrct_path = find_osrct_file(config)
    if osrct_path.suffix == ".csv":
        osrct_bundle = load_sites(
            RawDataset(
                name=f"osrct_{config.study}_{osrct_path.stem}",
                path=osrct_path,
                format="csv",
            ),
            family="osrct",
            study=config.study,
        )
        osrct_data = (
            osrct_bundle.to_frame()
            .reset_index(drop=True)
            .drop(columns=list(_LOADER_DERIVED_COLUMNS))
        )
    elif osrct_path.suffix == ".parquet":
        osrct_data = pd.read_parquet(osrct_path).reset_index(drop=True)
    elif osrct_path.suffix == ".pkl":
        osrct_data = pd.read_pickle(osrct_path).reset_index(drop=True)
    else:
        raise ValueError(
            f"Unsupported OSRCT artifact format {osrct_path.suffix!r}: "
            f"{osrct_path}. Expected '.csv' (raw, read through the Loader) or "
            f"'.parquet'/'.pkl' (sampled realisation artifacts)."
        )

    rct_path = Path(config.rct_path)
    rct_bundle = load_sites(
        RawDataset(
            name=f"manylabsrct_{config.study}",
            path=rct_path,
            format=rct_path.suffix.lstrip("."),
        ),
        family="manylabsrct",
        study=config.study,
    )
    rct_data = (
        rct_bundle.to_frame()
        .reset_index(drop=True)
        .drop(columns=list(_LOADER_DERIVED_COLUMNS))
    )

    cohort_metadata: dict[str, Any] = {
        "cohort_filter": "none",
        "require_observed_gender": bool(config.require_observed_gender),
    }
    if config.require_observed_gender:
        osrct_data, osrct_cohort = restrict_to_observed_gender(osrct_data)
        rct_data, rct_cohort = restrict_to_observed_gender(rct_data)
        # Reset the row labels so this branch returns the same RangeIndex as
        # the unfiltered one. Not done inside `restrict_to_observed_gender`:
        # the benchmark generator also calls it before OSRCT sampling, where a
        # reset could change which rows the published datasets drew.
        osrct_data = osrct_data.reset_index(drop=True)
        rct_data = rct_data.reset_index(drop=True)
        osrct_overall = osrct_cohort.loc[
            osrct_cohort["study"].eq("__all__")
            & osrct_cohort["site"].eq("__all__")
        ].iloc[0]
        rct_overall = rct_cohort.loc[
            rct_cohort["study"].eq("__all__")
            & rct_cohort["site"].eq("__all__")
        ].iloc[0]
        cohort_metadata.update(
            {
                "cohort_filter": OBSERVED_GENDER_COHORT_RULE,
                "n_osrct_rows_before_cohort_filter": int(
                    osrct_overall["n_rows_before"]
                ),
                "n_osrct_rows_excluded_by_cohort_filter": int(
                    osrct_overall["n_rows_excluded"]
                ),
                "n_osrct_rows_after_cohort_filter": int(
                    osrct_overall["n_rows_after"]
                ),
                "n_rct_rows_before_cohort_filter": int(
                    rct_overall["n_rows_before"]
                ),
                "n_rct_rows_excluded_by_cohort_filter": int(
                    rct_overall["n_rows_excluded"]
                ),
                "n_rct_rows_after_cohort_filter": int(
                    rct_overall["n_rows_after"]
                ),
            }
        )
    osrct_data, rct_data, metadata = add_naturalness_columns(
        osrct_data,
        rct_data,
        age_bins=config.age_bins,
        outcome="dv",
    )
    osrct_data, rct_data, binary_metadata = add_binary_covariates(
        osrct_data,
        rct_data,
        age_threshold=config.age_binary_threshold,
        polideo_threshold=config.polideo_binary_threshold,
    )
    metadata.update(
        {
            "osrct_path": str(osrct_path),
            "rct_path": str(config.rct_path),
            "study": config.study,
            "pattern": config.pattern,
            "beta": float(config.beta),
            "seed": int(config.seed),
            **cohort_metadata,
            **binary_metadata,
        }
    )
    return osrct_data, rct_data, metadata


__all__ = [
    "OSRCTConfig",
    "expand_target_sites",
    "iter_base_configs",
    "load_prepared_data",
]
