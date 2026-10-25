"""Compatibility re-export of the former shared experiment helpers.

This module holds no code. It re-exports every name it exported before the
split, so existing ``from experiments.utils import ...`` lines keep working.
New code should import from the module that owns the concern:

* ``experiments.osrct_files``: OSRCT file discovery and the supported
  ManyLabs studies.
* ``experiments.covariate_preparation``: binary outcome and covariate
  derivation, and target-site expansion.
* ``experiments.run_config``: probability-backend lists and resolution, the
  typed ``ProbabilityModelOptions`` contract, the predictor kwargs builders,
  and the configured naturalness grid and grounder builders.
* ``experiments.conformal_bounds``: split-conformal quantiles and conformal
  calibration of target-LP bounds.
* ``experiments.lp_summaries``: posterior LP bound summaries and output-frame
  concatenation.
* ``experiments.run_manifest``: JSON conversion and YAML run manifests.
* ``experiments.latex_tables``: LaTeX metric-summary and generic tables.
"""

from __future__ import annotations

# Names the original module imported and therefore exposed. Kept so that no
# attribute of ``experiments.utils`` disappears.
import json  # noqa: F401
import math  # noqa: F401
from dataclasses import asdict, is_dataclass  # noqa: F401
from datetime import datetime, timezone  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from causal_grounding.discretize import discretize_polideo  # noqa: F401
from causal_grounding.loaders.covariates import (  # noqa: F401  threshold re-exported
    DEFAULT_POLIDEO_BINARY_THRESHOLD,
    derive_age_bin,
    derive_polideo_bin,
)
from causal_grounding.naturalness_grounder import (  # noqa: F401
    NaturalnessGrounder,
    NaturalnessGrid,
)
from causal_grounding.probability_backends import probability_backend_names  # noqa: F401
from causal_grounding.train_target_split import get_available_sites  # noqa: F401
from experiments.conformal_bounds import (  # noqa: F401
    MAX_BINARY_CATE_ENDPOINT_RESIDUAL,
    add_full_range_fallback,
    conformalize_lp_bounds,
    finite_sample_conformal_quantile,
)
from experiments.covariate_preparation import (  # noqa: F401
    DEFAULT_GENDER_BINARY_THRESHOLD,
    USEFUL_COVARIATES,
    add_binary_covariates,
    add_naturalness_columns,
    expand_target_sites,
)
from experiments.latex_tables import (  # noqa: F401
    _latex_escape,
    _numeric_metric,
    format_latex_mean_sd,
    make_metric_summary_latex_table,
    make_multi_metric_summary_latex_table,
    render_latex_table,
    write_metric_summary_latex_table,
)
from experiments.lp_summaries import (  # noqa: F401
    concat_frames,
    summarize_lp_posterior_bounds,
)
from experiments.osrct_files import (  # noqa: F401
    SUPPORTED_MANYLABS_STUDIES,
    find_osrct_file,
)
from experiments.run_config import (  # noqa: F401
    DEFAULT_LP_POSTERIOR_N_JOBS,
    HIERARCHICAL_DIRICHLET_LP_BACKENDS,
    PROBABILITY_MODEL_OPTION_PREFIXES,
    SUPPORTED_LP_PROBABILITY_BACKENDS,
    SUPPORTED_STAGE1_BACKENDS,
    SUPPORTED_TARGET_TRIGGER_PROBABILITY_BACKENDS,
    ProbabilityModelOptions,
    build_configured_naturalness_grid,
    build_configured_naturalness_grounder,
    build_lp_probability_model_kwargs,
    build_stage1_predictor_kwargs,
    fit_configured_naturalness_grounder,
    resolve_lp_probability_backend,
    resolve_target_trigger_probability_backend,
    uses_hierarchical_dirichlet_draws,
)
from experiments.run_manifest import (  # noqa: F401
    _append_yaml_value,
    _manifest_safe_value,
    _resolve_manifest_path,
    _yaml_scalar,
    json_default,
    write_run_manifest,
)
