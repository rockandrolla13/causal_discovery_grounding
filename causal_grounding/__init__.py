"""
Causal Grounding Module
=======================

Naturalness grounding: bounds on the CATE of a binary treatment on a binary
outcome at a target site that has observational data only, learned from source
sites that hold both a randomized trial and observational data. Follows Silva's
"Causal Discovery Grounding and the Naturalness Assumption".

Modules:
    naturalness_grounder - Stages 1-3 on the sources: triggers, tolerances and
        their Gaussian-process surfaces (NaturalnessGrounder)
    naturalness_statistics - Support and trigger statistics
    naturalness_target_policy - The conditions that fire at the target
        (TargetNaturalnessPolicy)
    lp_solver_learned_naturalness - The target linear program
    bayesian_modular_naturalness - The cut-Bayesian pipeline of Experiment III
    predictors - Probability models: empirical, XGBoost, TabICLv2
    balke_pearl - Sharp Balke-Pearl bounds, the comparator
    discretize, train_target_split - Covariate discretization and the
        source/target split of the experiments
    loaders - Raw-data loading of the ManyLabs1 trials and OSRCT samples

Example:
    from causal_grounding import NaturalnessGrid, NaturalnessGrounder

    grid = NaturalnessGrid(epsilons=[0.05, 0.10], taus=[0.10, 0.20], covariates=['Z1', 'Z2'])
    grounder = NaturalnessGrounder(grid, stage1_backend='empirical')
    grounder.fit(sources, treatment='X', outcome='Y', regime_col='F')

examples/toy_naturalness.py runs the whole pipeline on simulated data.
"""

__version__ = "0.1.0"

# =============================================================================
# PROBABILITY MODELS
# =============================================================================

from .predictors import (
    BasePredictorModel,
    EmpiricalPredictor,
    TabICLv2Predictor,
    create_predictor,
)

# XGBoost predictor is optional
try:
    from .predictors import XGBoostPredictor, XGBoostBackdoorModel
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# =============================================================================
# DATA PREPARATION
# =============================================================================

from .discretize import (
    discretize_covariates,
    discretize_age,
    discretize_polideo,
    get_discretized_covariate_names,
)

from .train_target_split import (
    create_train_target_split,
    add_regime_indicator,
    get_available_sites,
)

# =============================================================================
# NATURALNESS GROUNDING
# =============================================================================

from .naturalness_grounder import (
    BaseNaturalnessPMFModel,
    EmpiricalSitePMFModels,
    HierarchicalDirichletSitePMFModels,
    NaturalnessGrid,
    NaturalnessObservation,
    NaturalnessTolerance,
    NaturalnessGPKernel,
    NaturalnessGPSurface,
    NATURALNESS_STAGE1_BACKENDS,
    NaturalnessGrounder,
    PredictorBackedSitePMFModels,
    TabICLv2SitePMFModels,
    XGBoostSitePMFModels,
    register_naturalness_stage1_backend,
)

from .naturalness_statistics import (
    compute_d_strength,
    compute_i_deviation,
)

from .naturalness_target_policy import (
    TargetNaturalnessChoice,
    TargetNaturalnessPolicy,
    TargetNaturalnessPolicyResult,
)
from .lp_solver_learned_naturalness import (
    TargetObservedBinaryQuantities,
    estimate_target_observed_binary_quantities,
    solve_target_bounds_from_observed_binary_quantities,
    solve_target_weighted_cate_bounds_from_observed_binary_quantities,
)

from .bayesian_modular_naturalness import (
    RandomIdentifiedSetResult,
    SitePMFDraw,
    ToleranceSurfacePosterior,
    compute_source_evidence,
    compute_trigger_statistics,
    draw_random_identified_sets,
    draw_raw_pmfs,
    draw_rct_reference_posterior,
    fit_tolerance_surfaces,
    posterior_outer_envelopes,
    score_random_sets_against_rct,
)

__all__ = [
    # Prediction models
    'BasePredictorModel',
    'EmpiricalPredictor',
    'TabICLv2Predictor',
    'XGBoostPredictor',
    'XGBoostBackdoorModel',
    'create_predictor',

    # Preprocessing
    'discretize_covariates',
    'discretize_age',
    'discretize_polideo',
    'get_discretized_covariate_names',

    # Train/target split
    'create_train_target_split',
    'add_regime_indicator',
    'get_available_sites',

    # Naturalness grounding
    'BaseNaturalnessPMFModel',
    'EmpiricalSitePMFModels',
    'NaturalnessGrid',
    'NaturalnessObservation',
    'NaturalnessTolerance',
    'NaturalnessGPKernel',
    'NaturalnessGPSurface',
    'NATURALNESS_STAGE1_BACKENDS',
    'NaturalnessGrounder',
    'HierarchicalDirichletSitePMFModels',
    'PredictorBackedSitePMFModels',
    'TabICLv2SitePMFModels',
    'XGBoostSitePMFModels',
    'register_naturalness_stage1_backend',
    'compute_d_strength',
    'compute_i_deviation',
    'TargetNaturalnessChoice',
    'TargetNaturalnessPolicy',
    'TargetNaturalnessPolicyResult',
    'TargetObservedBinaryQuantities',
    'estimate_target_observed_binary_quantities',
    'solve_target_bounds_from_observed_binary_quantities',
    'solve_target_weighted_cate_bounds_from_observed_binary_quantities',

    # Modular/cut-Bayesian Naturalness
    'RandomIdentifiedSetResult',
    'SitePMFDraw',
    'ToleranceSurfacePosterior',
    'compute_source_evidence',
    'compute_trigger_statistics',
    'draw_random_identified_sets',
    'draw_raw_pmfs',
    'draw_rct_reference_posterior',
    'fit_tolerance_surfaces',
    'posterior_outer_envelopes',
    'score_random_sets_against_rct',
]
