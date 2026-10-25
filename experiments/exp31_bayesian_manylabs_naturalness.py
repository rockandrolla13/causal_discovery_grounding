"""Experiment 31: modular cut-Bayesian Naturalness grounding and bounds on ManyLabs.

Runs :mod:`causal_grounding.bayesian_modular_naturalness` on Exp25's ManyLabs OSRCT design: studies allowedforbidden
and gainloss (``--studies`` adds reciprocity), OSRCT selection on age, gender or political ideology with strength
``beta`` in {0.5, 1, 1.5, 2}, target site ``pi`` with every other eligible site as a source, and 20 replications with
seeds 62-81.  Uncertainty in the observed-data PMFs (Module A: TabICLv2 draws) and in the grounded tolerances
(Module B: source evidence; Module C: NUTS tolerance surfaces) is propagated under a cut to a posterior over random
identified sets (Module D), set beside Bayesian Manski and Balke-Pearl sets built from the same target draws and
scored against the randomized ``pi`` reference (Module E).  The output is a posterior over identified sets under a
modular model and its priors: not a confidence interval, not a credible interval for the causal effect, and
reference membership is not frequentist coverage.

Subcommands::

    prepare      generate the OSRCT samples with Exp25's generator and fingerprint them
    draw-pmfs    draw Module A's TabICLv2 PMFs for one arm's conditions, on a GPU
    adopt-pmfs   reuse another PMF root's draws of conditions whose inputs are unchanged
    run          run one arm's conditions from a draw-pmfs root and finalize the run root
    finalize     validate a run root and write its completion
    balke-pearl  add the Bayesian Balke-Pearl arm to a finalized run
    regrid       replay a finalized run on its trigger grid restricted to tau >= --tau-floor
    report       combine finalized roots into the discovery-first report

``paper/reproduce_experiment3_bayesian.sh`` runs the paper's Experiment III with this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


from causal_grounding.loaders.schema import OUTCOME_COLUMN, TREATMENT_COLUMN
from causal_grounding import balke_pearl
from causal_grounding import bayesian_modular_naturalness as bayes
from causal_grounding.bayesian_modular_naturalness import (
    GP_NUGGET,
    compute_source_evidence,
    draw_random_identified_sets,
    draw_rct_reference_posterior,
    draw_tabicl_bootstrap_pmfs,
    fit_tolerance_surfaces,
    grid_trigger,
    leave_one_source_out_scores,
    posterior_predictive_tolerances,
    raw_pmf_draws_from_frame,
    raw_pmf_draws_to_frame,
)
from causal_grounding.naturalness_statistics import safe_sorted
from causal_grounding.paper_grids import PAPER_BETAS
from causal_grounding.predictors import TABICL_V2_CLASSIFIER_CHECKPOINT
from causal_grounding.seeding import sha256_seed
from causal_grounding.train_target_split import create_train_target_split
from experiments.osrct_target_pipeline import Exp16Config, split_site_regimes
from experiments.osrct_data_utils import load_prepared_data
from experiments.run_manifest import write_run_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


LOGGER = logging.getLogger(__name__)

EXPERIMENT_NAME = "exp31_bayesian_manylabs_naturalness"
SCHEMA_VERSION = 1

# --- Design ---------------------------------------------------------------------
TARGET_SITE = "pi"
#: Studies outside the default :data:`STUDIES`, run only when selected with ``--studies``.
DEFERRED_STUDIES = ("reciprocity",)
STUDIES = ("allowedforbidden", "gainloss")
#: Every study a run can select with ``--studies``.
ALL_STUDIES = STUDIES + DEFERRED_STUDIES
PATTERNS = ("age", "gender", "polideo")
#: OSRCT selection strengths: the beta grid of the Exp25 results the report's Exp25 bridge reads.
BETAS = PAPER_BETAS
N_REPLICATIONS = 20
BASE_SEED = 62
SEEDS = tuple(BASE_SEED + replication for replication in range(N_REPLICATIONS))
#: Exp25's stratum and candidate set (``EXP25_STRATUM_COLUMNS``): ``Z`` is the eight cells of these three binary
#: covariates and each is a Naturalness candidate.  Read from Exp16's configuration to avoid importing Exp25.
COVARIATES = tuple(Exp16Config().covariates)
#: A candidate triggers when any epsilon/tau pair of the trigger grid fires; its tolerance is read at the target's
#: own trigger statistics, which must lie inside the source support (except in :data:`UNGATED_ARMS`).
CANDIDATE_SELECTION = "all_triggered"
#: Exp25's trigger grid, the Exp16/OSRCT defaults it runs under.
TRIGGER_EPSILONS = tuple(float(value) for value in Exp16Config().epsilons)
TRIGGER_TAUS = tuple(float(value) for value in Exp16Config().taus)
REQUIRE_SOURCE_SUPPORT = True
PRIMARY_IMPUTATIONS = 20
PILOT_IMPUTATION_COUNTS = (10, 20)
ESCALATED_IMPUTATIONS = 40
IMPUTATION_SHIFT_THRESHOLD = 0.02
RANDOM_SET_DRAWS = 2000
TARGET_POLICY_UNCERTAINTY = "posterior_draws"
#: Module A's ``bootstrap`` scheme: each PMF draw is a stratified Bayesian bootstrap of the TabICLv2 fits
#: (:func:`bayes.draw_tabicl_bootstrap_pmfs`), made on a GPU by ``draw-pmfs``.
PMF_MODEL = "tabicl_v2_stratified_bayesian_bootstrap"
#: Source draws per condition: the escalated imputation count, so a pilot decision never needs new draws.
SOURCE_PMF_DRAWS = ESCALATED_IMPUTATIONS
#: Target draws per condition: one per random-set draw.
TARGET_PMF_DRAWS = RANDOM_SET_DRAWS
#: Module A's draw schemes: the stratified ``bootstrap`` or ``predictive`` resampling (:func:`predictive_pmfs`).
#: A PMF root's identity records the scheme only when it is not ``bootstrap``, so bootstrap fingerprints are stable.
MODULE_A_SCHEMES = ("bootstrap", "predictive")
#: Draw counts of the predictive scheme: source imputations and target draws.
PREDICTIVE_SOURCE_DRAWS = 20
PREDICTIVE_TARGET_DRAWS = 500
#: Exp25's TabICLv2 settings (pinned v2 checkpoint, 8 estimators, softmax temperature 0.9, the replication's OSRCT
#: seed as ``random_state``); sharing the loaded checkpoint only avoids reloading it and leaves predictions unchanged.
TABICL_PREDICTOR_KWARGS = {"kv_cache": "repr", "share_checkpoint": True}
#: Target idle rows each arm of an LP stratum needs: Exp16's ``min_cell_count``, as in Exp25's Naturalness LP.
LP_MIN_CELL_COUNT = int(Exp16Config().min_cell_count)
#: Reference rows each arm of a stratum needs before it is scored.
RCT_MIN_ARM_COUNT = 8
REQUIRE_OBSERVED_GENDER = True
#: Reference rows: every complete-case target randomized row (Exp25's ``disjoint_validation_ids=False``).  They
#: overlap the target's idle rows; :func:`_data_role_audit` records the overlap.
REFERENCE_ROWS = "full_target_rct"
#: The reference posterior is a scoring device with a fixed Beta(1, 1) arm prior.
REFERENCE_PRIOR_ALPHA = 1.0
#: Known invalid or placebo candidates; ManyLabs has none.
CANDIDATE_ROLES: dict[str, str] = {}
#: Handling of a tolerance surface that still fails the sampler gate after its retry.
UNCONVERGED_SURFACE_POLICY = (
    "a surface failing the gate after its retry gets one second retry; if that also fails, the candidate is "
    "excluded for that source imputation only (its paired draws record 'surface_not_converged'), never replaced "
    "by a plug-in surface, and exclusion rates are reported"
)
SOURCE_SITE_RULE = (
    "Exp16 create_train_target_split: every non-target site with at least one "
    "randomized and one OSRCT-selected complete-case row"
)
METHODS = ("bayesian_naturalness", "bayesian_manski")
NATURALNESS, MANSKI = METHODS
#: The Bayesian Balke-Pearl method, built on a finalized run's draws by the ``balke-pearl`` command; it is not in
#: :data:`METHODS`, which lists the methods a run itself computes.
BALKE_PEARL = "bayesian_balke_pearl"
#: Width gains at or below this are numerical ties: the Naturalness feasible
#: region is contained in the Manski region, so the true gain is never negative.
WIDTH_TIE_TOLERANCE = 1e-9

#: The smoke arm's (study, pattern, beta, replication) conditions.
SMOKE_CONDITIONS = (("allowedforbidden", "age", 1.0, 0),)
#: The computation-pilot design: every study and pattern at two strengths, as a Latin square over replications 0 and 1.
PILOT_DESIGN = (
    ("allowedforbidden", "age", 1.0, 0),
    ("gainloss", "gender", 1.0, 0),
    ("reciprocity", "polideo", 1.0, 0),
    ("allowedforbidden", "polideo", 0.5, 1),
    ("gainloss", "age", 0.5, 1),
    ("reciprocity", "gender", 0.5, 1),
)
#: Pilot and leave-one-source-site-out conditions: the pilot design restricted to :data:`STUDIES`.
PILOT_CONDITIONS = tuple(condition for condition in PILOT_DESIGN if condition[0] in STUDIES)
LOSO_CONDITIONS = PILOT_CONDITIONS

#: Thresholds of the report's classification (:func:`classify_families`): descriptive reporting rules, not gates, and
#: none is a feasibility requirement for success.
CLASSIFICATION_RULES = {
    "discovery_replication_threshold": 0.5,
    "stable_discovery_min_replication_share": 0.75,
    "abstention_max_any_eligible": 0.5,
    "empty_set_tension_min_empty_given_eligible": 0.5,
    "useful_min_width_gain_positive_all": 0.5,
    "material_membership_drop": 0.10,
    "material_wrong_sign_increase": 0.05,
    "imputation_endpoint_shift": IMPUTATION_SHIFT_THRESHOLD,
    "loso_target_tolerance_shift": 0.05,
    "single_cell_width_gain_share": 0.80,
}

#: Run arms: ``conditions`` fixes an arm's conditions (``None``: the full grid), and the primary arm takes its
#: imputation count from a pilot run's decision.
ARMS = {
    "smoke": {"conditions": SMOKE_CONDITIONS, "needs_pilot_decision": False},
    "pilot": {"conditions": PILOT_CONDITIONS, "needs_pilot_decision": False},
    "primary": {"conditions": None, "needs_pilot_decision": True},
    "loso": {"conditions": LOSO_CONDITIONS, "needs_pilot_decision": False},
    # Full-data fits as the only source imputation, recentred target draws, no source-support gate (MODULAR_DESIGN).
    "modular": {"conditions": None, "needs_pilot_decision": False},
    # Source imputations and target draws from a predictive-resampling PMF root, no source-support gate.
    "predictive": {"conditions": None, "needs_pilot_decision": False},
    # Leave-one-source-site-out check on any replications and studies, with the tested design's PMF root and evidence.
    "loso_grid": {"conditions": None, "needs_pilot_decision": False},
}
#: Arms that bound without the source-support gate, as Exp25's grounding does.
UNGATED_ARMS = frozenset({"modular", "predictive"})
#: Arms that refit the first imputation without each source site instead of bounding.
LOSO_ARMS = frozenset({"loso", "loso_grid"})
#: Arms that may run on a predictive-resampling PMF root.
PREDICTIVE_ROOT_ARMS = frozenset({"predictive", "loso_grid"})
#: Arms that accept Module B's evidence subset (``--evidence-min-randomized-rows``).
EVIDENCE_FLOOR_ARMS = UNGATED_ARMS | {"loso_grid"}
#: The modular arm's departures from the primary arm, recorded in its identity.
MODULAR_DESIGN = {
    "plan_section": "16.8",
    "added": "2026-10-02, after the primary outcomes; post hoc",
    "source_evidence": "one imputation from Module A's TabICLv2 models fitted once to every source site's full rows",
    "target_draws": "the PMF root's bootstrap target draws, each conditional recentred on the logit scale so their mean "
                    "is the target's full-data fit",
    "surfaces": "one NUTS envelope per candidate and tolerance kind, fitted to that evidence",
    "require_source_support": False,
    "uncertainty_cut": "source-PMF uncertainty does not reach the evidence (a cut posterior)",
}
#: The predictive arm's departures from the primary arm, recorded in its identity.
PREDICTIVE_DESIGN = {
    "plan_section": "16.8",
    "added": "2026-10-02, after the primary outcomes and the replication-0 predictive ladder; post hoc",
    "module_a": "predictive resampling: every draw simulates each site's data from its full-data TabICLv2 fit and "
                "refits Module A's models (draw-pmfs --module-a-scheme predictive)",
    "source_imputations": PREDICTIVE_SOURCE_DRAWS,
    "target_draws": PREDICTIVE_TARGET_DRAWS,
    "surfaces": "one NUTS envelope per imputation, candidate and tolerance kind, as in the primary arm",
    "require_source_support": False,
    "uncertainty_cut": None,
}

#: Files of one condition's leaf under a ``draw-pmfs`` root.
PMF_FILES = ("identity.json", "source_pmf_draws.parquet", "target_pmf_draws.parquet")
PMF_DATA_FILES = PMF_FILES[1:]
#: Module A library sources.  ``adopt-pmfs`` copies draws between PMF roots only
#: when both recorded identical fingerprints for every one of them.
MODULE_A_SOURCES = (
    "causal_grounding/bayesian_modular_naturalness.py",
    "causal_grounding/naturalness_grounder.py",
    "causal_grounding/predictors.py",
    "causal_grounding/lp_solver_learned_naturalness.py",
    "causal_grounding/naturalness_statistics.py",
)
CONDITION_FILES = (
    "identity.json",
    "data_role_audit.parquet",
    "source_evidence_draws.parquet",
    "tolerance_surface_diagnostics.parquet",
    "rct_reference_draws.parquet",
)
MIXTURE_FILES = (
    "target_trigger_draws.parquet",
    "tolerance_surface_draws.parquet",
    "random_identified_sets.parquet",
    "scored_random_sets.parquet",
    "condition_metrics.parquet",
)
LOSO_FILES = (
    "identity.json",
    "data_role_audit.parquet",
    "source_evidence_draws.parquet",
    "tolerance_surface_diagnostics.parquet",
    "loso_predictive_scores.parquet",
    "condition_metrics.parquet",
)
RUN_FILES = (
    "identity.json",
    "run_manifest.yml",
    "completion.json",
    "condition_metrics.parquet",
    "summary_by_study_pattern_beta.csv",
)
#: Pipeline sources whose checksums every manifest records; the cluster image
#: carries no ``.git``, so these are the code provenance that always exists.
SOURCE_FILES = (
    "experiments/exp31_bayesian_manylabs_naturalness.py",
    "causal_grounding/bayesian_modular_naturalness.py",
    "causal_grounding/lp_solver_learned_naturalness.py",
    "causal_grounding/naturalness_grounder.py",
    "causal_grounding/predictors.py",
    "causal_grounding/naturalness_statistics.py",
    "causal_grounding/paper_grids.py",
    "causal_grounding/balke_pearl.py",
    "causal_grounding/train_target_split.py",
    "causal_grounding/loaders/manylabs_rct.py",
    "causal_grounding/loaders/osrct.py",
    "causal_grounding/loaders/covariates.py",
    "causal_grounding/loaders/cohort.py",
    "experiments/exp16_osrct_calibrated_policy.py",
    "experiments/osrct_target_pipeline.py",
    "experiments/exp25_osrct_mc_baseline_comparison.py",
    "experiments/osrct_data_utils.py",
    "experiments/osrct_files.py",
    "experiments/covariate_preparation.py",
    "experiments/run_config.py",
    "experiments/conformal_bounds.py",
    "experiments/lp_summaries.py",
    "experiments/run_manifest.py",
    "experiments/latex_tables.py",
    "osrct_benchmark_v1.0/code/generate_confounded_datasets.py",
    "osrct_benchmark_v1.0/code/osrct.py",
    "osrct_benchmark_v1.0/code/experimental_grid.py",
)

# Denominator labels.  Every reported probability or summary names its support.
ALL_DRAWS = "all random-set draws"
NONEMPTY = "nonempty random sets of this method"
JOINT = "draws with nonempty Naturalness and Manski sets"
ELIGIBLE_DRAWS = "draws with at least one eligible candidate"
TRIGGERED = "draws in which this candidate triggered"
CANDIDATE_ELIGIBLE = "draws in which this candidate was eligible"
IMPUTATIONS = "source raw-PMF imputations"
EVIDENCE_UNITS = "source evidence units (site x candidate x imputation)"
SURFACES = "fitted tolerance surfaces"
ROWS = "rows"
HELD_OUT_UNITS = "held-out source evidence units"
BP_JOINT_MANSKI = "draws with nonempty Balke-Pearl and Manski sets"
BP_JOINT_NATURALNESS = "draws with nonempty Naturalness and Balke-Pearl sets"
USED_INSTRUMENT = "draws in which this candidate instrumented at least one LP cell"


@dataclass(frozen=True, order=True)
class ConditionKey:
    """One Exp31 condition: study, OSRCT pattern, beta and replication."""

    study: str
    pattern: str
    beta: float
    replication: int

    @property
    def seed(self) -> int:
        """Return the replication's OSRCT seed (62 for replication 0)."""

        return BASE_SEED + int(self.replication)

    def label(self) -> str:
        """Return a compact human-readable label."""

        return f"{self.study}/{self.pattern}/beta={_format_beta(self.beta)}/rep={self.replication:03d}"

    def leaf(self, root: Path) -> Path:
        """Return this condition's result directory under a run root."""

        return (
            Path(root)
            / "conditions"
            / f"study={self.study}"
            / f"pattern={self.pattern}"
            / f"beta={_format_beta(self.beta)}"
            / f"replication={self.replication:03d}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the key, with its seed, as plain JSON-ready values."""

        return {
            "study": self.study,
            "pattern": self.pattern,
            "beta": float(self.beta),
            "replication": int(self.replication),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class Exp31RunSettings:
    """Resolved compute settings of one Exp31 run root.

    Attributes:
        arm: One of :data:`ARMS`; it fixes the conditions unless they are the full grid.
        imputation_counts: Source PMF mixture sizes; the largest is fitted and smaller ones reuse its leading
            imputations with the same target draws and seeds.
        replications: Replication subset of a full-grid arm; ``None`` runs all.
        random_set_draws: Paired random-set draws per condition.
        num_chains: NUTS chains per surface.
        num_warmup: Warm-up draws per chain.
        num_samples: Retained draws per chain.
        target_accept_prob: NUTS target acceptance.
        retry_warmup: Warm-up of the first retry.
        retry_target_accept_prob: Target acceptance of the retry.
        retry_num_samples: Retained draws per chain on the retry.
        second_retry_warmup: Warm-up of the second retry, run when the first retry still fails the gate.
        second_retry_num_samples: Retained draws per chain on the second retry.
        second_retry_target_accept_prob: Target acceptance of the second retry.
        dense_mass: Adapt a full NUTS mass matrix in every attempt (sampling efficiency only).
        loso_score_draws: Retained draws per leave-one-out predictive score.
        studies: Studies of a full-grid arm from :data:`ALL_STUDIES`; ``None`` runs :data:`STUDIES` and is left out
            of the run identity.
        evidence_min_randomized_rows: Module B's evidence subset: a stratum pair counts toward a source site's ``2c``
            and ``2d`` only when both strata have this many randomized rows per arm; ``0`` keeps every pair and is
            left out of the run identity.
        cluster_manifest_path: Cluster manifest embedded in the run manifest.
    """

    arm: str
    imputation_counts: tuple[int, ...]
    replications: tuple[int, ...] | None = None
    random_set_draws: int = RANDOM_SET_DRAWS
    num_chains: int = 4
    num_warmup: int = 2000
    num_samples: int = 3000
    target_accept_prob: float = 0.90
    retry_warmup: int = 4000
    retry_target_accept_prob: float = 0.95
    retry_num_samples: int | None = 6000
    second_retry_warmup: int = 8000
    second_retry_num_samples: int = 24000
    second_retry_target_accept_prob: float = 0.95
    dense_mass: bool = False
    loso_score_draws: int = 2000
    evidence_min_randomized_rows: int = 0
    studies: tuple[str, ...] | None = None
    cluster_manifest_path: str | None = None

    def nuts_settings(self) -> dict[str, Any]:
        """Return the NUTS budget, which a pilot decision records and the primary arm must match."""

        return {
            "num_chains": int(self.num_chains),
            "num_warmup": int(self.num_warmup),
            "num_samples": int(self.num_samples),
            "target_accept_prob": float(self.target_accept_prob),
            "retry_warmup": int(self.retry_warmup),
            "retry_target_accept_prob": float(self.retry_target_accept_prob),
            "retry_num_samples": int(self.retry_num_samples or self.num_samples),
            "second_retry_warmup": int(self.second_retry_warmup),
            "second_retry_num_samples": int(self.second_retry_num_samples),
            "second_retry_target_accept_prob": float(self.second_retry_target_accept_prob),
            "dense_mass": bool(self.dense_mass),
        }

    def attempt_budgets(self) -> dict[str, tuple[int, int]]:
        """Return the ``(warm-up, retained draws)`` of every NUTS attempt."""

        return {
            "primary": (int(self.num_warmup), int(self.num_samples)),
            "retry": (int(self.retry_warmup), int(self.retry_num_samples or self.num_samples)),
            "second_retry": (int(self.second_retry_warmup), int(self.second_retry_num_samples)),
        }

    def fit_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments of :func:`fit_tolerance_surfaces`.

        Surfaces that fail the second retry are excluded, not raised (:data:`UNCONVERGED_SURFACE_POLICY`).
        """

        return {
            "num_chains": int(self.num_chains),
            "num_warmup": int(self.num_warmup),
            "num_samples": int(self.num_samples),
            "target_accept_prob": float(self.target_accept_prob),
            "retry_warmup": int(self.retry_warmup),
            "retry_target_accept_prob": float(self.retry_target_accept_prob),
            "retry_num_samples": self.retry_num_samples,
            "second_retry": {
                "num_warmup": int(self.second_retry_warmup),
                "num_samples": int(self.second_retry_num_samples),
                "target_accept_prob": float(self.second_retry_target_accept_prob),
            },
            "dense_mass": bool(self.dense_mass),
            "on_unconverged": "exclude",
        }

    def validate(self) -> None:
        """Reject settings that break the protocol.

        Raises:
            ValueError: For an unknown arm, a replication or study subset on a fixed-condition arm, invalid counts
                or an invalid sampler budget.
        """

        if self.arm not in ARMS:
            raise ValueError(f"Unknown arm {self.arm!r}; choose from {sorted(ARMS)}")
        if self.replications is not None:
            if ARMS[self.arm]["conditions"] is not None:
                raise ValueError(f"Arm {self.arm!r} has frozen conditions; --replications does not apply")
            if not self.replications or len(set(self.replications)) != len(self.replications) or any(
                not 0 <= value < N_REPLICATIONS for value in self.replications
            ):
                raise ValueError(f"replications must be distinct indices in [0, {N_REPLICATIONS})")
        counts = tuple(int(value) for value in self.imputation_counts)
        if not counts or any(value < 1 for value in counts) or tuple(sorted(set(counts))) != counts:
            raise ValueError("imputation_counts must be positive, unique and increasing")
        if self.arm in LOSO_ARMS and counts != (1,):
            raise ValueError("The leave-one-source-site-out arm uses the first imputation only")
        if self.arm == "modular" and counts != (1,):
            raise ValueError("The modular arm has one evidence imputation, the full-data fit")
        if self.arm == "predictive" and len(counts) != 1:
            raise ValueError("The predictive arm solves one mixture of its PMF root's imputations")
        if self.studies is not None:
            if ARMS[self.arm]["conditions"] is not None:
                raise ValueError(f"Arm {self.arm!r} has frozen conditions; --studies does not apply")
            if not self.studies or len(set(self.studies)) != len(self.studies) or not set(self.studies) <= set(ALL_STUDIES):
                raise ValueError(f"studies must be distinct values from {ALL_STUDIES}")
        if self.evidence_min_randomized_rows < 0:
            raise ValueError("evidence_min_randomized_rows must be non-negative")
        if self.evidence_min_randomized_rows and self.arm not in EVIDENCE_FLOOR_ARMS:
            raise ValueError(f"An evidence subset is a post hoc option of the {sorted(EVIDENCE_FLOOR_ARMS)} arms only")
        if self.arm == "pilot" and counts != PILOT_IMPUTATION_COUNTS:
            raise ValueError(f"The computation pilot compares imputation counts {PILOT_IMPUTATION_COUNTS}")
        if self.random_set_draws < 1 or any(self.random_set_draws % value for value in counts):
            raise ValueError("random_set_draws must be positive and divisible by every imputation count")
        if min(self.num_chains, self.num_warmup, self.num_samples, self.retry_warmup, self.loso_score_draws) < 1:
            raise ValueError("NUTS chain, warmup, sample and score counts must be positive")
        if self.retry_num_samples is not None and self.retry_num_samples < self.num_samples:
            raise ValueError("retry_num_samples must be at least num_samples")
        if self.second_retry_num_samples < (self.retry_num_samples or self.num_samples) or self.second_retry_warmup < 1:
            raise ValueError("The second retry must keep at least as many draws as the first")


def protocol_payload() -> dict[str, Any]:
    """Return the scientific design recorded in every identity.

    Returns:
        JSON-ready protocol: data design, estimands, trigger rule, priors, Monte Carlo rules, reference policy,
        fixed condition subsets and the classification rules.
    """

    return {
        "experiment": EXPERIMENT_NAME,
        "target_site": TARGET_SITE,
        "studies": list(STUDIES),
        "deferred_studies": {
            "studies": list(DEFERRED_STUDIES),
            "decision": "deferred by Guilherme on 2026-09-25 before any Exp31 condition finished; not outcome-driven",
        },
        "stratification_patterns": list(PATTERNS),
        "osrct_betas": [float(value) for value in BETAS],
        "osrct_betas_decision": (
            "(0.5, 1, 1.5, 2) replaced the plan's (0.1, 0.25, 0.5, 1) on 2026-09-26, after the 20260926 pilot; "
            "plan Section 16.6"
        ),
        "replications": N_REPLICATIONS,
        "base_seed": BASE_SEED,
        "seeds": list(SEEDS),
        "seed_disjointness": "disjoint from Exp25's saved seeds 202608-202627 and its default 42-61",
        "source_sites": SOURCE_SITE_RULE,
        "covariates": list(COVARIATES),
        "stratum": "eight cells of the three Exp25 binary covariates; pattern is the OSRCT selection covariate",
        "estimands": {
            "primary": "E[Y(1)-Y(0) | Z=z, S=pi] for every LP-supported cell z",
            "secondary": "site average, cells weighted by the target idle P(Z) draw normalized over LP cells",
        },
        "candidate_selection": CANDIDATE_SELECTION,
        "trigger_rule": {
            "eligibility": "triggered when any epsilon/tau grid pair fires, then inside the source support",
            "epsilons": list(TRIGGER_EPSILONS),
            "taus": list(TRIGGER_TAUS),
            "tolerance_read_at": "target's own i_deviation and d_strength",
            "require_source_support": REQUIRE_SOURCE_SUPPORT,
        },
        "target_policy_uncertainty": TARGET_POLICY_UNCERTAINTY,
        "module_a": {
            "pmf_model": PMF_MODEL,
            "estimator": (
                "Exp25's TabICLv2 Stage-1 models: idle P(X|Z) and P(Y|X,Z) at every site, randomized P(Y|X,Z) "
                "at source sites"
            ),
            "tabicl": {
                "checkpoint": TABICL_V2_CLASSIFIER_CHECKPOINT,
                **TABICL_PREDICTOR_KWARGS,
                "random_state": "the replication's OSRCT seed",
            },
            "resampling": (
                "rows resampled with replacement within every Z stratum; a source whose idle rows are nested in its "
                "randomized rows resamples its physical units separately among idle-selected and other units"
            ),
            "stratum_masses": "P(Z | idle) ~ Dirichlet(n_z), the Bayesian bootstrap marginal",
            "support": "the observed idle strata and their physical counts; TabICLv2 never adds LP support",
            "source_draws": SOURCE_PMF_DRAWS,
            "target_draws": TARGET_PMF_DRAWS,
            "decision": (
                "replaced the Dirichlet raw-cell draws on 2026-09-26, after the 20260925b pilot; "
                "plan Section 16.5"
            ),
        },
        "pmf_imputations_primary": PRIMARY_IMPUTATIONS,
        "imputation_rule": {
            "pilot_counts": list(PILOT_IMPUTATION_COUNTS),
            "threshold": IMPUTATION_SHIFT_THRESHOLD,
            "escalated_count": ESCALATED_IMPUTATIONS,
            "statistic": "largest shift of a target-mass-weighted Naturalness endpoint summary, 10 vs 20",
        },
        "random_set_draws_primary": RANDOM_SET_DRAWS,
        "lp_min_cell_count": LP_MIN_CELL_COUNT,
        "rct_min_arm_count": RCT_MIN_ARM_COUNT,
        "require_observed_gender": REQUIRE_OBSERVED_GENDER,
        "reference": {
            "rows": REFERENCE_ROWS,
            "prior": f"Beta({REFERENCE_PRIOR_ALPHA:g}, {REFERENCE_PRIOR_ALPHA:g}) per arm and cell",
            "scored_cells": f"LP cells with at least {RCT_MIN_ARM_COUNT} reference rows in each arm",
            "site_average_scored": "only when every LP cell is scored",
            "overlap": "target idle participants are also reference participants; audited",
        },
        "feasibility_gate": None,
        "candidate_roles": dict(CANDIDATE_ROLES),
        "smoke_conditions": [list(value) for value in SMOKE_CONDITIONS],
        "pilot_conditions": [list(value) for value in PILOT_CONDITIONS],
        "loso_conditions": [list(value) for value in LOSO_CONDITIONS],
        "classification_rules": dict(CLASSIFICATION_RULES),
        "tolerance_model": {
            "family": "unit_variance_probit_gaussian_process per candidate and kind",
            "gp_nugget": GP_NUGGET,
            "hyperparameters": "integrated with NUTS, never plugged in",
            "sampler_gate": "rank-normalized R-hat <= 1.01, bulk and tail ESS >= 400, no divergences, no support violations",
            "unconverged_surface_policy": UNCONVERGED_SURFACE_POLICY,
        },
        "claim_limit": (
            "posterior random identified sets under a modular cut model; not confidence intervals, "
            "not causal-effect credible intervals, and reference membership is not frequentist coverage"
        ),
    }


def all_conditions(replications: Sequence[int] | None = None,
                   studies: Sequence[str] | None = None) -> list[ConditionKey]:
    """Enumerate the full condition grid in its fixed run order.

    Args:
        replications: Optional replication subset.
        studies: Optional study selection from :data:`ALL_STUDIES`; :data:`STUDIES` by default.

    Returns:
        Conditions ordered by replication, study, pattern and beta.
    """

    selected = range(N_REPLICATIONS) if replications is None else sorted(int(value) for value in replications)
    return [
        ConditionKey(study, pattern, float(beta), int(replication))
        for replication in selected
        for study in (STUDIES if studies is None else tuple(studies))
        for pattern in PATTERNS
        for beta in BETAS
    ]


def arm_conditions(arm: str, replications: Sequence[int] | None = None,
                   studies: Sequence[str] | None = None) -> list[ConditionKey]:
    """Return the conditions of one arm.

    Args:
        arm: One of :data:`ARMS`.
        replications: Replication subset of a full-grid arm.
        studies: Study selection of a full-grid arm (:func:`all_conditions`).

    Returns:
        The arm's conditions in run order.
    """

    frozen = ARMS[arm]["conditions"]
    if frozen is None:
        return all_conditions(replications, studies)
    return [ConditionKey(study, pattern, float(beta), int(rep)) for study, pattern, beta, rep in frozen]


def _format_beta(beta: float) -> str:
    """Format beta as Exp25 does (``experiments.benchmarks.format_beta``).

    Duplicated rather than imported: ``experiments.benchmarks`` loads PyTorch, about 1 GiB per condition worker.
    """

    value = float(beta)
    return f"{value:.1f}" if value.is_integer() else f"{value:g}"


def _utc_now() -> str:
    """Return the current UTC time in ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Convert common scientific Python values to deterministic JSON."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(payload: Any, path: Path) -> None:
    """Atomically write a formatted JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(root: Path) -> dict[str, str]:
    """Return hashes for material leaf artifacts, excluding completion state."""

    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "completion.json"
    }


def _dependency_versions() -> dict[str, str | None]:
    """Record the numerical stack that defines a Bayesian run."""

    versions: dict[str, str | None] = {}
    for distribution in ("numpy", "pandas", "scipy", "jax", "jaxlib", "numpyro"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _configure_logging() -> None:
    """Configure the INFO-level log format shared by the CLI and its workers."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _wait_for_run_identity(output_dir: Path, timeout_seconds: float = 1800.0) -> None:
    """Block a non-zero shard until shard 0 has written the run identity."""

    deadline = time.monotonic() + timeout_seconds
    while not (output_dir / "identity.json").is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Shard 0 did not initialize {output_dir} within {timeout_seconds:.0f}s")
        time.sleep(10.0)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace one Parquet table.

    Mirrors ``experiments.benchmarks.atomic_write_parquet`` without importing its PyTorch stack into every worker.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False, engine="pyarrow")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the schema version and a stable fingerprint to a payload."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        **payload,
    }


def _stream_seed(key: ConditionKey, *parts: Any) -> int:
    """Derive one condition's random stream, identical across arms.

    Seeds exclude the arm, so arms share common random numbers and read the same PMF draws of a condition.
    """

    return sha256_seed(EXPERIMENT_NAME, key.study, key.pattern, _format_beta(key.beta), key.seed, *parts)


# --- Provenance -----------------------------------------------------------------


def _git_provenance() -> dict[str, Any]:
    """Return the git revision and dirty state, or why they are unavailable."""

    try:
        revision = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
        return {
            "revision": revision,
            "dirty_worktree": bool(status.strip()),
            "n_changed_paths": int(len(status.splitlines())),
            "source": "git checkout",
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "revision": os.environ.get("PREDICTIVE_SHIFT_GIT_REVISION"),
            "dirty_worktree": None,
            "source": "no git checkout (container image); see source_fingerprints",
        }


def _source_fingerprints() -> dict[str, str | None]:
    """Hash the pipeline sources so the executed code is identified exactly."""

    fingerprints: dict[str, str | None] = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        fingerprints[relative] = _sha256(path) if path.is_file() else None
    return fingerprints


def _software_environment() -> dict[str, Any]:
    """Record the interpreter, platform, numerical stack and thread settings."""

    versions = _dependency_versions()
    try:
        import importlib.metadata as metadata

        versions["pyarrow"] = metadata.version("pyarrow")
    except Exception:  # pragma: no cover - pyarrow is required to write outputs.
        versions["pyarrow"] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "environment": {
            name: os.environ.get(name)
            for name in ("XLA_FLAGS", "JAX_PLATFORM_NAME", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


# --- prepare --------------------------------------------------------------------


def _osrct_dir(prepared_root: Path, replication: int) -> Path:
    """Return one replication's generated OSRCT grid directory."""

    return Path(prepared_root) / f"replication={int(replication):03d}" / "osrct"


def _osrct_file(prepared_root: Path, key: ConditionKey) -> Path:
    """Return the generated OSRCT sample of one condition."""

    return _osrct_dir(prepared_root, key.replication) / key.study / f"{key.pattern}_beta{float(key.beta)}_seed{key.seed}.parquet"


def prepare_inputs(
    output_dir: Path,
    *,
    rct_path: Path,
    replications: Sequence[int] | None = None,
    studies: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate every Exp31 OSRCT grid with Exp25's generator and fingerprint it.

    Each replication's grid is written as Exp25 writes its own, by
    :func:`experiments.exp25_osrct_mc_baseline_comparison.generate_replication_osrct_grid`, with the design's studies,
    patterns, betas, the observed-gender cohort rule and the replication's Exp31 seed.  The prepared identity pins the
    RCT pickle, the generator sources and every generated file by SHA-256, so a run rejects inputs changed since.

    Args:
        output_dir: Fresh directory for the prepared grids.
        rct_path: ManyLabs RCT pickle, never downloaded or replaced here.
        replications: Optional subset; all 20 by default.
        studies: Studies to generate, from :data:`ALL_STUDIES`; :data:`STUDIES` by default.

    Returns:
        The prepared identity written to ``identity.json``.

    Raises:
        FileExistsError: If ``output_dir`` is not empty.
        ValueError: If a generated condition failed or breaks nesting.
    """

    # Exp25 loads Exp18 and PyTorch; it is needed here only.
    from experiments.exp25_osrct_mc_baseline_comparison import Exp25Config, generate_replication_osrct_grid

    studies = STUDIES if studies is None else tuple(studies)
    if not studies or len(set(studies)) != len(studies) or not set(studies) <= set(ALL_STUDIES):
        raise ValueError(f"studies must be distinct values from {ALL_STUDIES}")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Prepared directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(range(N_REPLICATIONS)) if replications is None else sorted(int(value) for value in replications)
    config = Exp25Config(
        n_replications=N_REPLICATIONS,
        base_seed=BASE_SEED,
        studies=studies,
        bias_covariates=PATTERNS,
        betas=BETAS,
        target_site=TARGET_SITE,
        rct_path=str(rct_path),
        require_observed_gender=REQUIRE_OBSERVED_GENDER,
        output_dir=str(output_dir / "unused_exp25_output"),
        verbose=False,
    )
    inputs: list[dict[str, Any]] = []
    for replication in selected:
        seed = BASE_SEED + replication
        LOGGER.info("Generating OSRCT grid for replication %s (seed %s)", replication, seed)
        summary = generate_replication_osrct_grid(config, seed=seed, output_dir=_osrct_dir(output_dir, replication))
        if not summary["success"].astype(bool).all():
            raise ValueError(f"OSRCT generation failed for seed {seed}: {summary.loc[~summary.success.astype(bool)]}")
        for study, pattern, beta in product(studies, PATTERNS, BETAS):
            key = ConditionKey(study, pattern, float(beta), replication)
            path = _osrct_file(output_dir, key)
            frame = pd.read_parquet(path)
            if frame["id"].duplicated().any():
                raise ValueError(f"Duplicate participant IDs in {path}")
            inputs.append({
                **key.as_dict(),
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
                "n_rows": int(len(frame)),
                "n_target_rows": int(frame["site"].astype(str).eq(TARGET_SITE).sum()),
            })
    identity = _identity({
        "experiment": EXPERIMENT_NAME,
        "stage": "prepared",
        "generator": "experiments.exp25_osrct_mc_baseline_comparison.generate_replication_osrct_grid",
        "rct_path": str(Path(rct_path).resolve()),
        "rct_sha256": _sha256(Path(rct_path)),
        "generator_sources": {
            relative: _sha256(PROJECT_ROOT / relative)
            for relative in SOURCE_FILES
            if relative.startswith("osrct_benchmark") and (PROJECT_ROOT / relative).is_file()
        },
        "replications": selected,
        "inputs": inputs,
        "design": {
            "studies": list(studies),
            "patterns": list(PATTERNS),
            "betas": [float(value) for value in BETAS],
            "seeds": [BASE_SEED + replication for replication in selected],
            "require_observed_gender": REQUIRE_OBSERVED_GENDER,
        },
    })
    _write_json(identity, output_dir / "identity.json")
    write_run_manifest(
        output_dir,
        experiment_name=EXPERIMENT_NAME,
        script_path="experiments/exp31_bayesian_manylabs_naturalness.py",
        config={"stage": "prepared", "protocol": protocol_payload()},
        argv=sys.argv,
        extra={
            "prepared_fingerprint": identity["fingerprint"],
            "git": _git_provenance(),
            "source_fingerprints": _source_fingerprints(),
            "software_environment": _software_environment(),
            "row_counts": inputs,
        },
    )
    return identity


def read_prepared_identity(prepared_root: Path) -> dict[str, Any]:
    """Read and validate an Exp31 prepared root's identity.

    Raises:
        FileNotFoundError: If the root was never prepared.
        ValueError: If it belongs to another experiment or stage.
    """

    path = Path(prepared_root) / "identity.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No prepared Exp31 data at {path}; run `python -m experiments.exp31_bayesian_manylabs_naturalness "
            f"prepare --output-dir {prepared_root}` first"
        )
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("experiment") != EXPERIMENT_NAME or identity.get("stage") != "prepared":
        raise ValueError(f"{prepared_root} is not an Exp31 prepared root")
    return identity


def verify_prepared_inputs(
    prepared_root: Path,
    identity: Mapping[str, Any],
    conditions: Sequence[ConditionKey],
    rct_path: Path,
) -> None:
    """Re-hash the inputs a run reads and compare them with the prepared identity.

    Raises:
        ValueError: If the RCT pickle or a selected OSRCT file changed, or a selected condition was never prepared.
    """

    if _sha256(Path(rct_path)) != identity["rct_sha256"]:
        raise ValueError(f"RCT pickle {rct_path} does not match the prepared fingerprint")
    recorded = {(row["study"], row["pattern"], float(row["beta"]), int(row["replication"])): row for row in identity["inputs"]}
    for key in conditions:
        row = recorded.get((key.study, key.pattern, float(key.beta), key.replication))
        if row is None:
            raise ValueError(f"Condition {key.label()} was not prepared in {prepared_root}")
        if _sha256(Path(prepared_root) / row["path"]) != row["sha256"]:
            raise ValueError(f"Prepared input {row['path']} changed after preparation")


# --- Data adapter ---------------------------------------------------------------


def _participant_ids(values: pd.Series) -> pd.Series:
    """Render ManyLabs participant IDs as stable strings (``2400853.0`` -> ``2400853``)."""

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and np.all(np.isclose(numeric, np.round(numeric))):
        return numeric.round().astype("int64").astype(str)
    return values.astype(str)


def _loader_config(key: ConditionKey, prepared_root: Path, rct_path: Path) -> Exp16Config:
    """Return the Exp16 loading configuration Exp25 uses, for one condition.

    ``disjoint_validation_ids`` stays at Exp25's ``False``, so
    :func:`split_site_regimes` returns the full target RCT as the reference.
    """

    return Exp16Config(
        study=key.study,
        studies=(key.study,),
        pattern=key.pattern,
        patterns=(key.pattern,),
        beta=float(key.beta),
        betas=(float(key.beta),),
        seed=int(key.seed),
        target_sites=(TARGET_SITE,),
        data_root=str(_osrct_dir(prepared_root, key.replication)),
        rct_path=str(rct_path),
        require_observed_gender=REQUIRE_OBSERVED_GENDER,
        disjoint_validation_ids=False,
        verbose=False,
        make_plots=False,
    )


@dataclass
class ConditionData:
    """Learner-facing rows of one condition, its reference rows and their audit.

    Attributes:
        key: The condition.
        sources: Every eligible source site's randomized and idle rows (idle nested in randomized).
        target_idle: The target's OSRCT-selected idle rows.
        reference: The target's complete-case randomized rows (scoring only).
        audit: Row-level data-role audit.
        counts: Row counts by role, checked against the audit.
        loader_metadata: Metadata returned by the Exp16 loader.
    """

    key: ConditionKey
    sources: pd.DataFrame
    target_idle: pd.DataFrame
    reference: pd.DataFrame
    audit: pd.DataFrame
    counts: dict[str, int]
    loader_metadata: dict[str, Any] = field(default_factory=dict)


def _bayes_frame(frame: pd.DataFrame, regime: str | None = None) -> pd.DataFrame:
    """Project rows onto the Bayesian module's schema with compact dtypes."""

    output = frame.copy()
    if regime is not None:
        output["F"] = regime
    output["id"] = _participant_ids(output["id"])
    output["site"] = output["site"].astype(str)
    for column in (TREATMENT_COLUMN, OUTCOME_COLUMN, *COVARIATES):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(np.int8)
    columns = ["id", "site", "F", TREATMENT_COLUMN, OUTCOME_COLUMN, *COVARIATES] if "F" in output else ["id", "site", TREATMENT_COLUMN, OUTCOME_COLUMN, *COVARIATES]
    return output[columns].reset_index(drop=True)


def load_condition_data(key: ConditionKey, prepared_root: Path, rct_path: Path) -> ConditionData:
    """Load one condition through Exp25's data path and assign every row a role.

    Follows Exp16's ``run_one_target`` under Exp25's configuration: the complete-case rule (observed ``resp_sex``)
    and binary encodings of :func:`load_prepared_data`, the source/target split of
    :func:`create_train_target_split`, and every target randomized row as the reference (:func:`split_site_regimes`).

    Args:
        key: The condition.
        prepared_root: Exp31 prepared root.
        rct_path: The ManyLabs RCT pickle.

    Returns:
        The condition's rows in the Bayesian module's schema and their audit.

    Raises:
        AssertionError: If the roles or their counts are inconsistent, or a target randomized row could reach a
            fitting stage.
    """

    config = _loader_config(key, prepared_root, rct_path)
    osrct_data, rct_data, metadata = load_prepared_data(config)
    source_training, target_idle = create_train_target_split(
        osrct_data=osrct_data,
        rct_data=rct_data,
        target_site=TARGET_SITE,
        site_col="site",
        regime_col="F",
    )
    target_rct = rct_data.loc[rct_data["site"].astype(str).eq(TARGET_SITE)].copy()
    target_combined = pd.concat(
        [target_idle.assign(F="idle"), target_rct.assign(F="on")],
        ignore_index=True,
        sort=False,
    )
    _, reference_rows, removed = split_site_regimes(target_combined, config)
    if removed:
        raise AssertionError("Exp25's reference convention keeps every target randomized row")
    source_sites = sorted(str(site) for site in source_training)
    sources = pd.concat(
        [_bayes_frame(source_training[site]) for site in source_sites],
        ignore_index=True,
    )
    target = _bayes_frame(target_idle, regime="idle")
    reference = _bayes_frame(reference_rows.drop(columns=["F"]))
    audit = _data_role_audit(key, rct_path, rct_data, osrct_data, source_sites, target, reference)
    counts = {
        "n_source_sites": len(source_sites),
        "n_source_randomized_rows": int(sources.F.eq("on").sum()),
        "n_source_idle_rows": int(sources.F.eq("idle").sum()),
        "n_target_idle_rows": int(len(target)),
        "n_target_reference_rows": int(len(reference)),
        "n_reference_rows_shared_with_target_idle": int(reference.id.isin(set(target.id)).sum()),
        "n_excluded_resp_sex_missing": int(audit.exclusion_reason.eq("resp_sex_missing").sum()),
        "n_excluded_site_without_idle_rows": int(audit.exclusion_reason.eq("site_without_idle_rows").sum()),
    }
    _check_data_roles(counts, audit, sources, target, reference, metadata)
    return ConditionData(key, sources, target, reference, audit, counts, dict(metadata))


def _data_role_audit(
    key: ConditionKey,
    rct_path: Path,
    rct_data: pd.DataFrame,
    osrct_data: pd.DataFrame,
    source_sites: Sequence[str],
    target: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Assign every study row a data role and, when excluded, a reason.

    A participant appears once per regime record: an OSRCT-selected source participant has ``source_randomized`` and
    ``source_idle`` rows, and an OSRCT-selected target participant is both ``target_idle`` and ``target_reference``
    (``shared_with_target_idle``).  Rows removed by the complete-case rule are read from the raw pickle.

    Returns:
        One row per (participant, regime record) with the condition keys.
    """

    raw = pd.read_pickle(rct_path)
    raw = raw.loc[raw["original_study"].eq(key.study)]
    frames: list[pd.DataFrame] = []
    missing_sex = raw.loc[raw["resp_sex"].isna()]
    frames.append(pd.DataFrame({
        "site": missing_sex["site"].astype(str),
        "participant_id": _participant_ids(missing_sex["id"]),
        "regime": "on",
        "role": "excluded",
        "exclusion_reason": "resp_sex_missing",
    }))
    sources = set(source_sites)
    for regime, frame in (("on", rct_data), ("idle", osrct_data)):
        site = frame["site"].astype(str)
        role = np.select(
            [site.eq(TARGET_SITE), site.isin(sources)],
            ["target_reference" if regime == "on" else "target_idle", "source_randomized" if regime == "on" else "source_idle"],
            default="excluded",
        )
        frames.append(pd.DataFrame({
            "site": site.to_numpy(),
            "participant_id": _participant_ids(frame["id"]).to_numpy(),
            "regime": regime,
            "role": role,
            "exclusion_reason": np.where(role == "excluded", "site_without_idle_rows", ""),
        }))
    audit = pd.concat(frames, ignore_index=True)
    target_ids = set(target.id)
    audit["shared_with_target_idle"] = audit.role.eq("target_reference") & audit.participant_id.isin(target_ids)
    for name, value in key.as_dict().items():
        audit[name] = value
    columns = ["study", "pattern", "beta", "replication", "seed", "site", "participant_id", "regime", "role",
               "exclusion_reason", "shared_with_target_idle"]
    audit = audit[columns]
    for column in ("study", "pattern", "site", "regime", "role", "exclusion_reason"):
        audit[column] = audit[column].astype("category")
    return audit


def _check_data_roles(
    counts: Mapping[str, int],
    audit: pd.DataFrame,
    sources: pd.DataFrame,
    target: pd.DataFrame,
    reference: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> None:
    """Assert that roles, counts and information flow are consistent.

    Raises:
        AssertionError: On any inconsistency.
    """

    role_counts = audit.role.value_counts()
    expected = {
        "source_randomized": counts["n_source_randomized_rows"],
        "source_idle": counts["n_source_idle_rows"],
        "target_idle": counts["n_target_idle_rows"],
        "target_reference": counts["n_target_reference_rows"],
    }
    for role, count in expected.items():
        if int(role_counts.get(role, 0)) != int(count):
            raise AssertionError(f"Audit has {role_counts.get(role, 0)} {role} rows, the pipeline {count}")
    if counts["n_source_sites"] < 3:
        raise AssertionError("Exp31 needs at least three eligible source sites")
    if TARGET_SITE in set(sources.site):
        raise AssertionError("Target rows reached the source fitting frame")
    if set(target.site) != {TARGET_SITE} or set(reference.site) != {TARGET_SITE}:
        raise AssertionError("Target idle or reference rows belong to another site")
    if not set(target.id) <= set(reference.id):
        raise AssertionError("Every OSRCT-selected target row must have its randomized record")
    for site, frame in sources.groupby("site", observed=True):
        if not set(frame.loc[frame.F.eq("idle"), "id"]) <= set(frame.loc[frame.F.eq("on"), "id"]):
            raise AssertionError(f"Source site {site} has idle rows outside its randomized rows")
    recorded = metadata.get("n_rct_rows_excluded_by_cohort_filter")
    if recorded is not None and int(recorded) != counts["n_excluded_resp_sex_missing"]:
        raise AssertionError("Complete-case exclusions do not match the loader's cohort diagnostics")


# --- Modules A-E ----------------------------------------------------------------


def _randomized_pair_filters(
    sources: pd.DataFrame,
    min_rows: int,
) -> dict[str, Callable[[tuple[int, ...], tuple[int, ...]], bool]]:
    """Return, per source site, the stratum-pair filter of Module B's evidence subset.

    A pair passes when both strata have at least ``min_rows`` randomized rows in each arm at the site.
    """

    covariates = list(COVARIATES)
    randomized = sources.loc[sources.F.eq("on")]
    counts = randomized.groupby([randomized.site.astype(str), TREATMENT_COLUMN, *covariates]).size().to_dict()

    def site_filter(site: str) -> Callable[[tuple[int, ...], tuple[int, ...]], bool]:
        """Return the filter of one site."""

        return lambda z, z_prime: min(int(counts.get((site, x, *stratum), 0))
                                      for x in (0, 1) for stratum in (z, z_prime)) >= min_rows

    return {str(site): site_filter(str(site)) for site in sources.site.astype(str).unique()}


def _source_evidence(
    imputations: Sequence[Mapping[str, bayes.SitePMFDraw]],
    sources: pd.DataFrame | None = None,
    min_randomized_rows: int = 0,
) -> pd.DataFrame:
    """Recompute every source site's evidence for every imputation (Module B).

    Every candidate of every site is kept, triggered or not.  With ``min_randomized_rows``, a site's ``2c`` and ``2d``
    range only over the stratum pairs :func:`_randomized_pair_filters` passes in ``sources`` (the condition's source
    rows); a unit without such a pair keeps ``NaN`` requirements, which grounding leaves out.
    """

    if min_randomized_rows and sources is None:
        raise ValueError("An evidence subset needs the condition's source rows")
    filters = _randomized_pair_filters(sources, min_randomized_rows) if min_randomized_rows else {}
    frames = []
    for imputation_index, imputation in enumerate(imputations):
        evidence = pd.concat([compute_source_evidence(draw, filters.get(str(site)))
                              for site, draw in imputation.items()], ignore_index=True)
        evidence.insert(0, "imputation_index", int(imputation_index))
        frames.append(evidence)
    evidence = _with_source_grid(pd.concat(frames, ignore_index=True), TRIGGER_TAUS)
    return evidence.rename(columns={"sample_id": "imputation_id"})


def _with_source_grid(evidence: pd.DataFrame, taus: Sequence[float]) -> pd.DataFrame:
    """Record whether each source evidence unit would itself trigger on the grid with ``taus``.

    Descriptive only: every unit is grounded, triggered or not, so the grid never changes the fitted surfaces.
    """

    triggers = [grid_trigger(i, d, TRIGGER_EPSILONS, taus) for i, d in zip(evidence.i_deviation, evidence.d_strength)]
    return evidence.assign(
        source_grid_triggered=[value[0] for value in triggers],
        source_trigger_epsilon=[value[1] for value in triggers],
        source_trigger_tau=[value[2] for value in triggers],
    )


def _quantiles(values: Any, prefix: str) -> dict[str, float]:
    """Return mean and 5/50/95 percent quantiles of a posterior sample."""

    array = np.asarray(values, dtype=float).ravel()
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_q05": float(np.quantile(array, 0.05)),
        f"{prefix}_q50": float(np.quantile(array, 0.50)),
        f"{prefix}_q95": float(np.quantile(array, 0.95)),
    }


def _surface_diagnostic_row(
    posterior: bayes.ToleranceSurfacePosterior,
    *,
    imputation_index: int,
    root_seed: int,
) -> dict[str, Any]:
    """Summarize one fitted surface: sampler diagnostics, posterior and support."""

    diagnostics = posterior.diagnostics
    attempt = diagnostics.get("attempt", "primary")
    row: dict[str, Any] = {
        "imputation_index": int(imputation_index),
        "imputation_id": posterior.imputation_id,
        "candidate": posterior.candidate,
        "tolerance_kind": posterior.tolerance_kind,
        "root_seed": int(root_seed),
        "nuts_seed": int(sha256_seed(root_seed, posterior.imputation_id, posterior.candidate, posterior.tolerance_kind, attempt)),
        **{key: value for key, value in diagnostics.items() if not isinstance(value, (dict, list))},
        "excluded": False,
    }
    row.update(_quantiles(posterior.posterior_samples["eta"], "eta"))
    if "mu" in posterior.posterior_samples:  # a learned latent location (bayes.MARGINAL_LOCATIONS)
        row.update(_quantiles(posterior.posterior_samples["mu"], "mu"))
    lengths = np.asarray(posterior.posterior_samples["length_scale"], dtype=float)
    for position, name in enumerate(posterior.feature_names):
        row.update(_quantiles(lengths[:, position], f"length_scale_{name}"))
        row[f"source_{name}_min"] = float(posterior.source_features[:, position].min())
        row[f"source_{name}_max"] = float(posterior.source_features[:, position].max())
    tolerance = np.asarray(posterior.posterior_samples["source_tolerance"], dtype=float)
    slack = tolerance.mean(axis=0) - posterior.source_omega
    row["source_tolerance_posterior_mean"] = float(tolerance.mean())
    row["source_tolerance_min_mean_slack"] = float(slack.min())
    row["source_omega_mean"] = float(posterior.source_omega.mean())
    return row


def _fit_imputation(
    evidence_r: pd.DataFrame,
    root_seed: int,
    fit_kwargs: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], bayes.ToleranceSurfacePosterior], list[dict[str, Any]]]:
    """Fit one source imputation's candidate surfaces; module level so a spawned worker can run it.

    Returns:
        The ``{(candidate, kind): posterior}`` mapping of the kept surfaces, and the diagnostics of every surface
        excluded as unconverged (returned, since a worker cannot fill the caller's ``unconverged`` list).
    """

    unconverged: list[dict[str, Any]] = []
    fitted = fit_tolerance_surfaces(evidence_r, random_seed=root_seed, unconverged=unconverged, **fit_kwargs)
    return fitted, unconverged


def fit_surface_mixture(
    evidence: pd.DataFrame,
    key: ConditionKey,
    settings: Exp31RunSettings,
    *,
    partial_path: Path | None = None,
    isolate: bool = False,
) -> tuple[list[dict[tuple[str, str], bayes.ToleranceSurfacePosterior]], pd.DataFrame]:
    """Fit every imputation's candidate surfaces with NUTS (Module C).

    Hyperparameters are integrated over, never plugged in.  Each surface's ``source_tolerance`` draws are dropped
    once its diagnostics are summarized, to save memory; prediction needs only the latents and length scales.  A
    surface failing the gate after its second retry is excluded (:data:`UNCONVERGED_SURFACE_POLICY`) and keeps a
    diagnostics row with ``excluded=True``; any other error fails the condition, after the diagnostics fitted so far
    are written to ``partial_path`` when given.

    With ``isolate`` each imputation is fitted in a fresh process, which bounds the memory growth of repeated NUTS
    fits; seeds are fixed per condition and imputation, so the posteriors are unchanged.

    Returns:
        One ``{(candidate, kind): posterior}`` mapping per imputation, and the diagnostics table of every surface,
        excluded ones included.
    """

    mixtures = []
    rows = []
    pool = None
    if isolate:
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                   max_tasks_per_child=1, initializer=_configure_logging)
    try:
        for imputation_index, evidence_r in evidence.groupby("imputation_index", sort=True):
            LOGGER.info("%s: fitting source imputation %s", key.label(), int(imputation_index) + 1)
            root_seed = _stream_seed(key, "surface", int(imputation_index))
            arguments = (evidence_r.rename(columns={"imputation_id": "sample_id"}), root_seed, settings.fit_kwargs())
            try:
                if pool is None:
                    fitted, unconverged = _fit_imputation(*arguments)
                else:
                    fitted, unconverged = pool.submit(_fit_imputation, *arguments).result()
            except RuntimeError as error:  # includes a killed worker (BrokenProcessPool)
                if partial_path is not None:
                    failed = pd.DataFrame(rows).assign(condition_failed=False)
                    failure = pd.DataFrame([{"imputation_index": int(imputation_index), "condition_failed": True,
                                             "failure": repr(error)}])
                    _write_parquet(pd.concat([failed, failure], ignore_index=True), partial_path)
                raise
            for posterior in fitted.values():
                rows.append(_surface_diagnostic_row(posterior, imputation_index=int(imputation_index), root_seed=root_seed))
                posterior.posterior_samples.pop("source_tolerance", None)
            for failure in unconverged:
                LOGGER.warning("%s: excluding unconverged %s/%s surface of imputation %s", key.label(),
                               failure["candidate"], failure["tolerance_kind"], int(imputation_index) + 1)
                rows.append({
                    "imputation_index": int(imputation_index),
                    "root_seed": int(root_seed),
                    "nuts_seed": int(sha256_seed(root_seed, failure["imputation_id"], failure["candidate"],
                                                        failure["tolerance_kind"], failure.get("attempt", "second_retry"))),
                    **{name: value for name, value in failure.items() if not isinstance(value, (dict, list))},
                    "excluded": True,
                })
            mixtures.append(fitted)
    finally:
        if pool is not None:
            pool.shutdown()
    return mixtures, pd.DataFrame(rows)


def lp_strata(target_draws: Sequence[bayes.SitePMFDraw]) -> list[tuple[int, ...]]:
    """Return the target strata the LP supports; they depend on counts only."""

    return list(target_draws[0].observed_quantities(min_cell_count=LP_MIN_CELL_COUNT).z_values)


def reference_support(
    reference: pd.DataFrame,
    target_idle: pd.DataFrame,
    strata: Sequence[tuple[int, ...]],
) -> pd.DataFrame:
    """Count reference and target idle rows by arm in every LP stratum.

    A stratum is scored only when both reference arms have at least
    :data:`RCT_MIN_ARM_COUNT` rows, so no reference value rests on its prior.
    """

    rows = []
    for z in strata:
        key = json.dumps([int(value) for value in z])
        reference_mask = np.logical_and.reduce([reference[column].eq(value) for column, value in zip(COVARIATES, z)])
        idle_mask = np.logical_and.reduce([target_idle[column].eq(value) for column, value in zip(COVARIATES, z)])
        row = {"estimand_key": key, **dict(zip(COVARIATES, (int(value) for value in z)))}
        for x in (0, 1):
            row[f"reference_n_arm{x}"] = int((reference_mask & reference.iv.eq(x)).sum())
            row[f"target_idle_n_arm{x}"] = int((idle_mask & target_idle.iv.eq(x)).sum())
        row["reference_supported"] = bool(min(row["reference_n_arm0"], row["reference_n_arm1"]) >= RCT_MIN_ARM_COUNT)
        rows.append(row)
    return pd.DataFrame(rows)


def draw_reference(
    reference: pd.DataFrame,
    target_draws: Sequence[bayes.SitePMFDraw],
    support: pd.DataFrame,
    key: ConditionKey,
    n_draws: int,
) -> pd.DataFrame:
    """Draw the randomized reference posterior on the scored strata (Module E).

    Only supported LP strata are passed to :func:`draw_rct_reference_posterior`.  Its site average is kept only when
    every LP stratum is supported, so that it averages over the same cells as the random identified set's.
    """

    scored = support.loc[support.reference_supported.astype(bool)]
    columns = ["reference_draw_index", "estimand_type", "estimand_key", "reference_value"]
    if scored.empty:
        return pd.DataFrame(columns=columns)
    cells = set(scored.estimand_key)
    keys = reference[list(COVARIATES)].apply(lambda row: json.dumps([int(value) for value in row]), axis=1)
    references = draw_rct_reference_posterior(
        reference.loc[keys.isin(cells)],
        target_draws,
        n_draws=n_draws,
        alpha=REFERENCE_PRIOR_ALPHA,
        min_arm_count=RCT_MIN_ARM_COUNT,
        random_seed=_stream_seed(key, "rct-reference"),
        covariates=COVARIATES,
    )
    if len(scored) != len(support):
        references = references.loc[references.estimand_type.ne("ate")]
    return references[columns].reset_index(drop=True)


def random_set_tables(
    result: bayes.RandomIdentifiedSetResult,
    mixtures: Sequence[Mapping[tuple[str, str], bayes.ToleranceSurfacePosterior]],
    excluded: Iterable[tuple[int, str]] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label every draw and candidate with its eligibility and LP outcome.

    ``lp_outcome`` is ``nonempty``, ``empty`` (an ``infeasible`` endpoint: the grounded restrictions contradict the
    target's observed PMF) or ``solver_failure``.  Draws with no eligible candidate are the matched Manski set and carry
    ``used_manski_fallback``; eligible draws with an empty LP stay empty (``eligible_but_lp_infeasible``).  A triggered
    candidate whose paired ``2c`` or ``2d`` surface is missing is ineligible: ``surface_not_converged`` when listed in
    ``excluded`` as ``(imputation_index, candidate)``, ``no_fitted_surface`` otherwise.

    Returns:
        The random identified sets, the target trigger draws and the grounded tolerance draws.

    Raises:
        AssertionError: If the fallback flag disagrees with eligibility.
    """

    triggers = result.triggers.copy()
    available = {
        (index, candidate)
        for index, fitted in enumerate(mixtures)
        for candidate in {name for name, _ in fitted}
        if (candidate, "c") in fitted and (candidate, "d") in fitted
    }
    pairs = list(zip(triggers.source_imputation_index.astype(int), triggers.candidate.astype(str)))
    triggers["surface_available"] = [pair in available for pair in pairs]
    unconverged = {(int(index), str(candidate)) for index, candidate in excluded}
    triggers["surface_excluded_unconverged"] = [pair in unconverged for pair in pairs]
    finite = np.isfinite(triggers.i_deviation) & np.isfinite(triggers.d_strength)
    triggers["ineligibility_reason"] = np.select(
        [
            triggers.eligible.astype(bool),
            ~finite,
            ~triggers.triggered.astype(bool),
            ~triggers.surface_available & triggers.surface_excluded_unconverged,
            ~triggers.surface_available,
            ~triggers.in_source_support.astype(bool),
        ],
        ["eligible", "nonfinite_target_statistic", "not_triggered", "surface_not_converged", "no_fitted_surface",
         "outside_source_support"],
        default="nonfinite_tolerance_draw",
    )
    n_draws = int(triggers.draw_index.max()) + 1
    eligible = triggers.loc[triggers.eligible.astype(bool)]
    per_draw = pd.DataFrame({"draw_index": np.arange(n_draws)})
    per_draw["n_eligible"] = eligible.groupby("draw_index").size().reindex(per_draw.draw_index, fill_value=0).to_numpy()
    per_draw["candidate_set"] = (
        eligible.groupby("draw_index").candidate.agg(lambda values: "|".join(sorted(map(str, values))))
        .reindex(per_draw.draw_index, fill_value="").to_numpy()
    )
    draws = result.draws.drop(columns=["trigger_payload"]).merge(per_draw, on="draw_index", how="left", validate="many_to_one")
    infeasible = draws.lower_status.eq("infeasible") | draws.upper_status.eq("infeasible")
    draws["lp_outcome"] = np.select([draws.feasible.astype(bool), infeasible], ["nonempty", "empty"], default="solver_failure")
    draws["nonempty"] = draws.lp_outcome.eq("nonempty")
    draws["width"] = (draws.upper - draws.lower).where(draws.nonempty)
    draws["no_candidate_eligible"] = draws.n_eligible.eq(0)
    naturalness = draws.method.eq(NATURALNESS)
    if not draws.loc[naturalness, "used_manski_fallback"].astype(bool).equals(draws.loc[naturalness, "no_candidate_eligible"]):
        raise AssertionError("Manski fallback must occur exactly when no candidate is eligible")
    empty_draws = set(draws.loc[naturalness & draws.lp_outcome.eq("empty"), "draw_index"])
    draws["eligible_but_lp_infeasible"] = naturalness & ~draws.no_candidate_eligible & draws.draw_index.isin(empty_draws)
    tolerances = triggers.loc[
        np.isfinite(triggers.two_c) | np.isfinite(triggers.two_d),
        ["draw_index", "sample_id", "source_imputation_index", "imputation_id", "candidate", "eligible",
         "i_deviation", "d_strength", "c_posterior_index", "d_posterior_index", "two_c", "two_d"],
    ].copy()
    tolerances["c"] = tolerances.two_c / 2.0
    tolerances["d"] = tolerances.two_d / 2.0
    trigger_columns = [
        "draw_index", "sample_id", "target_pmf_draw_index", "target_pmf_sample_id", "source_imputation_index",
        "imputation_id", "candidate", "candidate_index", "i_deviation", "d_strength", "triggered", "trigger_epsilon",
        "trigger_tau", "surface_available", "surface_excluded_unconverged", "in_source_support", "eligible",
        "ineligibility_reason",
    ]
    draw_columns = [
        "draw_index", "sample_id", "target_pmf_draw_index", "target_pmf_sample_id", "source_imputation_index",
        "imputation_id", "method", "estimand_type", "estimand_key", "estimand_weight", "lower", "upper", "width",
        "status", "lower_status", "upper_status", "feasible", "lp_outcome", "nonempty", "n_eligible", "candidate_set",
        "no_candidate_eligible", "used_manski_fallback", "eligible_but_lp_infeasible",
    ]
    return draws[draw_columns], triggers[trigger_columns], tolerances.reset_index(drop=True)


def score_random_sets(random_sets: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Pair every random set with its reference draw by draw index (Module E).

    The module's ``score_random_sets_against_rct`` is not used because its summaries count strata without a reference
    as misses; here they are ``reference_scoreable=False`` and leave every denominator.  An empty set is never a
    member and never has the wrong sign.
    """

    keys = random_sets[["draw_index", "method", "estimand_type", "estimand_key", "lower", "upper", "nonempty"]]
    scored = keys.merge(
        references.rename(columns={"reference_draw_index": "draw_index"}),
        on=["draw_index", "estimand_type", "estimand_key"],
        how="left",
        validate="many_to_one",
    )
    scoreable = scored.reference_value.notna()
    nonempty = scored.nonempty.astype(bool)
    value = scored.reference_value
    scored["reference_scoreable"] = scoreable
    scored["reference_member"] = scoreable & nonempty & value.ge(scored.lower) & value.le(scored.upper)
    scored["wrong_sign"] = scoreable & nonempty & (
        (scored.lower.gt(0.0) & value.le(0.0)) | (scored.upper.lt(0.0) & value.ge(0.0))
    )
    return scored[["draw_index", "method", "estimand_type", "estimand_key", "reference_value",
                   "reference_scoreable", "reference_member", "wrong_sign"]]


class _MetricRows:
    """Collect long-format metrics, each with its denominator and its support."""

    def __init__(self) -> None:
        """Start an empty collection."""

        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        scope: str,
        metric: str,
        value: Any,
        denominator: Any,
        label: str,
        *,
        method: str = "",
        estimand_type: str = "",
        estimand_key: str = "",
        candidate: str = "",
    ) -> None:
        """Record one metric."""

        number = float(value) if value is not None and np.isfinite(float(value)) else float("nan")
        self.rows.append({
            "scope": scope,
            "method": method,
            "estimand_type": estimand_type,
            "estimand_key": estimand_key,
            "candidate": candidate,
            "metric": metric,
            "value": number,
            "denominator": int(denominator),
            "denominator_label": label,
        })

    def frame(self) -> pd.DataFrame:
        """Return the collected metrics."""

        return pd.DataFrame(self.rows)


def _per_draw_weighted_width(cate: pd.DataFrame) -> pd.DataFrame:
    """Return each draw's target-mass-weighted CATE width, per method.

    Args:
        cate: CATE rows of the random identified sets, with ``method``,
            ``draw_index``, ``estimand_weight``, ``width`` and ``nonempty``.

    Returns:
        One row per method and draw: ``nonempty`` (every cell nonempty) and
        ``weighted_width``, NaN for a draw with an empty cell.
    """

    per_draw = cate.assign(weighted_width=cate.estimand_weight * cate.width).groupby(["method", "draw_index"]).agg(
        nonempty=("nonempty", "all"), weighted_width=("weighted_width", "sum"),
    ).reset_index()
    per_draw.loc[~per_draw.nonempty, "weighted_width"] = np.nan
    return per_draw


def _summary_statistics(values: pd.Series) -> dict[str, float]:
    """Return mean and 2.5/50/97.5 percent quantiles, or NaN when empty."""

    if values.empty:
        return {name: float("nan") for name in ("mean", "q025", "q50", "q975")}
    return {
        "mean": float(values.mean()),
        "q025": float(values.quantile(0.025)),
        "q50": float(values.quantile(0.50)),
        "q975": float(values.quantile(0.975)),
    }


def condition_metrics(
    *,
    random_sets: pd.DataFrame,
    triggers: pd.DataFrame,
    tolerances: pd.DataFrame,
    scored: pd.DataFrame,
    support: pd.DataFrame,
    evidence: pd.DataFrame,
    diagnostics: pd.DataFrame,
    counts: Mapping[str, int],
    target_mass_outside_lp: float,
    n_unobserved_target_strata: int,
) -> pd.DataFrame:
    """Compute every condition metric with its denominator.

    Scopes, in reporting order: ``data``, ``discovery`` and ``candidate`` (restriction discovery), ``grounding``,
    ``feasibility``, ``bounds``, ``weighted`` (target-mass-weighted cell summaries), ``paired`` (Naturalness against
    matched Manski) and ``reference``.  Probabilities over draws state whether they use all draws or condition on a
    nonempty set; feasibility is reported over all draws, with no LP-feasibility gate.

    Returns:
        Long table with ``scope``, ``method``, ``estimand_type``, ``estimand_key``, ``candidate``, ``metric``,
        ``value``, ``denominator`` and ``denominator_label``.
    """

    out = _MetricRows()
    n_draws = int(random_sets.draw_index.nunique())
    n_imputations = int(evidence.imputation_index.nunique())
    for name, value in counts.items():
        out.add("data", name, value, value, ROWS)
    out.add("data", "n_lp_cells", len(support), len(support), "LP cells")
    out.add("data", "n_reference_scored_cells", support.reference_supported.sum(), len(support), "LP cells")
    out.add("data", "n_unobserved_target_strata", n_unobserved_target_strata, 2 ** len(COVARIATES), "possible strata")
    out.add("data", "target_mass_outside_lp_cells", target_mass_outside_lp, n_draws, ALL_DRAWS)
    out.add(
        "data", "reference_shared_with_target_idle_share",
        counts["n_reference_rows_shared_with_target_idle"] / max(counts["n_target_reference_rows"], 1),
        counts["n_target_reference_rows"], "target reference rows",
    )

    # Restriction discovery.
    per_draw = triggers.groupby("draw_index").eligible.sum()
    out.add("discovery", "p_any_candidate_eligible", (per_draw > 0).mean(), n_draws, ALL_DRAWS)
    out.add("discovery", "p_no_candidate_eligible", (per_draw == 0).mean(), n_draws, ALL_DRAWS)
    out.add("discovery", "mean_n_eligible_candidates", per_draw.mean(), n_draws, ALL_DRAWS)
    for count in range(len(COVARIATES) + 1):
        out.add("discovery", f"p_n_eligible_eq_{count}", (per_draw == count).mean(), n_draws, ALL_DRAWS)
    sets = random_sets.loc[random_sets.method.eq(NATURALNESS)].drop_duplicates("draw_index").candidate_set
    modal = sets.value_counts()
    out.add("discovery", "modal_eligible_set_share", modal.iloc[0] / n_draws, n_draws, ALL_DRAWS,
            candidate=str(modal.index[0]) or "none")
    for candidate, group in triggers.groupby("candidate", sort=True):
        n_triggered = int(group.triggered.sum())
        n_eligible = int(group.eligible.sum())
        out.add("candidate", "p_triggered", group.triggered.mean(), n_draws, ALL_DRAWS, candidate=candidate)
        out.add("candidate", "p_eligible", group.eligible.mean(), n_draws, ALL_DRAWS, candidate=candidate)
        out.add("candidate", "p_in_source_support", group.in_source_support.mean(), n_draws, ALL_DRAWS, candidate=candidate)
        outside = group.ineligibility_reason.eq("outside_source_support")
        out.add("candidate", "p_triggered_outside_source_support", outside.mean(), n_draws, ALL_DRAWS, candidate=candidate)
        out.add("candidate", "p_outside_source_support_given_triggered",
                outside.sum() / n_triggered if n_triggered else np.nan, n_triggered, TRIGGERED, candidate=candidate)
        out.add("candidate", "p_nonfinite_target_statistic",
                group.ineligibility_reason.eq("nonfinite_target_statistic").mean(), n_draws, ALL_DRAWS, candidate=candidate)
        out.add("candidate", "p_surface_not_converged",
                group.ineligibility_reason.eq("surface_not_converged").mean(), n_draws, ALL_DRAWS, candidate=candidate)
        out.add("candidate", "target_i_deviation_q50", group.i_deviation.median(), n_draws, ALL_DRAWS, candidate=candidate)
        out.add("candidate", "target_d_strength_q50", group.d_strength.median(), n_draws, ALL_DRAWS, candidate=candidate)
        grounded = tolerances.loc[tolerances.candidate.eq(candidate) & tolerances.eligible.astype(bool)]
        for column in ("two_c", "two_d", "c", "d"):
            for statistic, value in _summary_statistics(grounded[column]).items():
                out.add("candidate", f"grounded_{column}_{statistic}", value, n_eligible, CANDIDATE_ELIGIBLE,
                        candidate=candidate)

    # Grounding: source evidence and NUTS diagnostics.
    for candidate, group in evidence.groupby("candidate", sort=True):
        finite = group.dropna(subset=["i_deviation", "d_strength", "required_two_c", "required_two_d"])
        out.add("grounding", "finite_source_units_per_imputation", len(finite) / n_imputations, n_imputations,
                IMPUTATIONS, candidate=candidate)
        for column in ("i_deviation", "d_strength", "required_two_c", "required_two_d"):
            out.add("grounding", f"source_{column}_min", finite[column].min(), len(finite), EVIDENCE_UNITS, candidate=candidate)
            out.add("grounding", f"source_{column}_q50", finite[column].median(), len(finite), EVIDENCE_UNITS, candidate=candidate)
            out.add("grounding", f"source_{column}_max", finite[column].max(), len(finite), EVIDENCE_UNITS, candidate=candidate)
        out.add("grounding", "source_grid_trigger_rate", finite.source_grid_triggered.mean(), len(finite),
                EVIDENCE_UNITS, candidate=candidate)
        out.add("grounding", "cross_pmf_projection_rate", group.cross_pmf_was_clipped.mean(), len(group),
                EVIDENCE_UNITS, candidate=candidate)
    excluded = diagnostics.excluded.astype(bool)
    used = diagnostics.loc[~excluded]
    for (candidate, kind), group in diagnostics.groupby(["candidate", "tolerance_kind"], sort=True):
        tag = f"surface_{kind}"
        converged = group.loc[~group.excluded.astype(bool)]
        # Gate statistics cover every surface, excluded ones included, so the worst mixing is visible; posterior
        # summaries cover only the converged surfaces that grounding uses.
        out.add("grounding", f"{tag}_max_r_hat", group.max_r_hat.max(), len(group), SURFACES, candidate=candidate)
        out.add("grounding", f"{tag}_min_bulk_ess", group.min_bulk_ess.min(), len(group), SURFACES, candidate=candidate)
        out.add("grounding", f"{tag}_min_tail_ess", group.min_tail_ess.min(), len(group), SURFACES, candidate=candidate)
        out.add("grounding", f"{tag}_n_divergent", group.n_divergent.sum(), len(group), SURFACES, candidate=candidate)
        out.add("grounding", f"{tag}_n_retried", group.attempt.isin(["retry", "second_retry"]).sum(), len(group), SURFACES,
                candidate=candidate)
        out.add("grounding", f"{tag}_n_second_retry", group.attempt.eq("second_retry").sum(), len(group), SURFACES,
                candidate=candidate)
        out.add("grounding", f"{tag}_n_excluded_unconverged", group.excluded.astype(bool).sum(), len(group), SURFACES,
                candidate=candidate)
        if "eta_q50" in converged:
            out.add("grounding", f"{tag}_eta_q50_mean", converged.eta_q50.mean(), len(converged), SURFACES, candidate=candidate)
            out.add("grounding", f"{tag}_min_mean_slack", converged.source_tolerance_min_mean_slack.min(), len(converged),
                    SURFACES, candidate=candidate)
    out.add("grounding", "n_surfaces", len(diagnostics), len(diagnostics), SURFACES)
    out.add("grounding", "n_surfaces_excluded_unconverged", excluded.sum(), len(diagnostics), SURFACES)
    out.add("grounding", "all_used_surfaces_passed", used.passed.astype(bool).all(), len(used), SURFACES)

    weights, per_draw_width = _method_bound_metrics(out, random_sets, n_draws)
    _paired_width_metrics(out, random_sets, per_draw_width, weights, n_draws)
    _reference_metrics(out, random_sets, scored, support, weights, n_draws)
    return out.frame()


def _method_bound_metrics(out: _MetricRows, random_sets: pd.DataFrame, n_draws: int) -> tuple[pd.Series, pd.DataFrame]:
    """Record every method's feasibility, bounds and target-mass-weighted summaries.

    Shared by :func:`condition_metrics` and :func:`balke_pearl_condition_metrics`, so every method is summarized by
    the same code.  Fallback and empty-given-eligible counts are recorded for every method but Manski.

    Args:
        out: The metric collection to extend.
        random_sets: Random identified sets of one or more methods.
        n_draws: Random-set draws per method.

    Returns:
        The LP cells' normalized mean target mass, and each draw's target-mass-weighted width per method.
    """

    # Feasibility: per-draw outcome over all draws.
    by_draw = random_sets.groupby(["method", "draw_index"]).agg(
        nonempty=("nonempty", "all"),
        is_empty=("lp_outcome", lambda values: bool((values == "empty").any())),
        n_eligible=("n_eligible", "first"),
    ).reset_index()
    for method, group in by_draw.groupby("method", sort=True):
        eligible = group.n_eligible.gt(0)
        failure = ~group.nonempty & ~group.is_empty
        out.add("feasibility", "p_nonempty", group.nonempty.mean(), n_draws, ALL_DRAWS, method=method)
        out.add("feasibility", "n_nonempty_draws", group.nonempty.sum(), n_draws, ALL_DRAWS, method=method)
        out.add("feasibility", "n_empty_draws", group.is_empty.sum(), n_draws, ALL_DRAWS, method=method)
        out.add("feasibility", "n_solver_failure_draws", failure.sum(), n_draws, ALL_DRAWS, method=method)
        if method != MANSKI:
            out.add("feasibility", "n_fallback_draws", (~eligible).sum(), n_draws, ALL_DRAWS, method=method)
            out.add("feasibility", "n_eligible_nonempty_draws", (eligible & group.nonempty).sum(), n_draws, ALL_DRAWS, method=method)
            out.add("feasibility", "n_eligible_but_lp_infeasible_draws", (eligible & group.is_empty).sum(), n_draws,
                    ALL_DRAWS, method=method)
            out.add("feasibility", "p_empty_given_eligible",
                    (eligible & group.is_empty).sum() / eligible.sum() if eligible.any() else np.nan,
                    eligible.sum(), ELIGIBLE_DRAWS, method=method)

    # Bounds per method and estimand, over nonempty draws of that method.
    for (method, estimand_type, estimand_key), group in random_sets.groupby(["method", "estimand_type", "estimand_key"], sort=True):
        keys = {"method": method, "estimand_type": estimand_type, "estimand_key": estimand_key}
        nonempty = group.loc[group.nonempty]
        n_nonempty = len(nonempty)
        out.add("bounds", "p_nonempty", n_nonempty / n_draws, n_draws, ALL_DRAWS, **keys)
        for column in ("lower", "upper", "width"):
            for statistic, value in _summary_statistics(nonempty[column]).items():
                out.add("bounds", f"{column}_{statistic}", value, n_nonempty, NONEMPTY, **keys)
        out.add("bounds", "outer_lower", nonempty.lower.quantile(0.025) if n_nonempty else np.nan, n_nonempty, NONEMPTY, **keys)
        out.add("bounds", "outer_upper", nonempty.upper.quantile(0.975) if n_nonempty else np.nan, n_nonempty, NONEMPTY, **keys)
        positive = nonempty.lower.gt(0.0)
        negative = nonempty.upper.lt(0.0)
        out.add("bounds", "p_sign_decision_all", (positive | negative).sum() / n_draws, n_draws, ALL_DRAWS, **keys)
        out.add("bounds", "p_positive_decision_all", positive.sum() / n_draws, n_draws, ALL_DRAWS, **keys)
        out.add("bounds", "p_negative_decision_all", negative.sum() / n_draws, n_draws, ALL_DRAWS, **keys)
        out.add("bounds", "p_sign_decision_given_nonempty", (positive | negative).mean() if n_nonempty else np.nan,
                n_nonempty, NONEMPTY, **keys)
        if estimand_type == "cate":
            out.add("bounds", "mean_target_mass", group.estimand_weight.mean(), n_draws, ALL_DRAWS, **keys)

    # Target-mass-weighted cell summaries.
    cate = random_sets.loc[random_sets.estimand_type.eq("cate")].copy()
    weights = cate.groupby("estimand_key").estimand_weight.mean()
    weights = weights / weights.sum()
    per_draw_width = _per_draw_weighted_width(cate)
    for method, group in per_draw_width.groupby("method", sort=True):
        nonempty = group.loc[group.nonempty, "weighted_width"]
        for statistic, value in _summary_statistics(nonempty).items():
            out.add("weighted", f"weighted_width_{statistic}", value, len(nonempty), NONEMPTY, method=method)
        cells = cate.loc[cate.method.eq(method) & cate.nonempty].groupby("estimand_key")
        summaries = pd.DataFrame({
            "outer_lower": cells.lower.quantile(0.025),
            "outer_upper": cells.upper.quantile(0.975),
            "lower_q50": cells.lower.median(),
            "upper_q50": cells.upper.median(),
        }).reindex(weights.index)
        for column in summaries:
            out.add("weighted", f"weighted_{column}", float(np.dot(weights, summaries[column])) if summaries[column].notna().all()
                    else np.nan, len(nonempty), NONEMPTY, method=method)
    return weights, per_draw_width


def _paired_width_metrics(
    out: _MetricRows,
    random_sets: pd.DataFrame,
    per_draw_width: pd.DataFrame,
    weights: pd.Series,
    n_draws: int,
    *,
    method: str = NATURALNESS,
    baseline: str = MANSKI,
    tag: str = "paired",
    narrower_metric: str = "p_naturalness_narrower_given_joint",
    joint_label: str = JOINT,
) -> None:
    """Record the paired width gain ``W_baseline - W_method`` on jointly nonempty draws.

    The defaults compare Naturalness with Manski; the Balke-Pearl arm reuses it for its pairs under its own ``tag``.
    A positive gain favors ``method``.

    Args:
        out: The metric collection to extend.
        random_sets: Random identified sets holding both methods.
        per_draw_width: Each draw's weighted width per method.
        weights: The LP cells' normalized mean target mass.
        n_draws: Random-set draws per method.
        method: The method whose narrowing is measured.
        baseline: The method it is compared against.
        tag: The ``method`` column of the paired rows.
        narrower_metric: Name of the probability that ``method`` is narrower.
        joint_label: Denominator label of the jointly nonempty draws.
    """

    wide = per_draw_width.pivot(index="draw_index", columns="method", values="weighted_width")
    joint = wide[method].notna() & wide[baseline].notna()
    gain = (wide[baseline] - wide[method])[joint]
    paired = {"method": tag}
    out.add("paired", narrower_metric, gain.gt(WIDTH_TIE_TOLERANCE).mean() if joint.any() else np.nan,
            joint.sum(), joint_label, **paired)
    out.add("paired", "p_width_gain_positive_all", gain.gt(WIDTH_TIE_TOLERANCE).sum() / n_draws, n_draws, ALL_DRAWS, **paired)
    for statistic, value in _summary_statistics(gain).items():
        out.add("paired", f"weighted_width_gain_{statistic}", value, joint.sum(), joint_label, **paired)
    cell_widths = (
        random_sets.loc[random_sets.method.isin([method, baseline])]
        .set_index(["draw_index", "estimand_type", "estimand_key", "method"]).width
        .unstack("method").reset_index()
    )
    contributions = {}
    for (estimand_type, estimand_key), group in cell_widths.groupby(["estimand_type", "estimand_key"], sort=True):
        both = group.dropna(subset=[method, baseline])
        cell_gain = both[baseline] - both[method]
        keys = {"method": tag, "estimand_type": estimand_type, "estimand_key": estimand_key}
        out.add("paired", narrower_metric, cell_gain.gt(WIDTH_TIE_TOLERANCE).mean() if len(both) else np.nan,
                len(both), joint_label, **keys)
        for statistic, value in _summary_statistics(cell_gain).items():
            out.add("paired", f"width_gain_{statistic}", value, len(both), joint_label, **keys)
        if estimand_type == "cate" and len(both):
            contributions[estimand_key] = float(weights.get(estimand_key, 0.0) * cell_gain.mean())
    total = sum(contributions.values())
    out.add("paired", "largest_cell_share_of_weighted_gain", max(contributions.values()) / total if total > 0 else np.nan,
            joint.sum(), joint_label, **paired)


def _reference_metrics(
    out: _MetricRows,
    random_sets: pd.DataFrame,
    scored: pd.DataFrame,
    support: pd.DataFrame,
    weights: pd.Series,
    n_draws: int,
) -> None:
    """Record every method's randomized-reference membership and wrong sign.

    Args:
        out: The metric collection to extend.
        random_sets: Random identified sets of one or more methods.
        scored: Their reference scoring (:func:`score_random_sets`).
        support: Reference support of the LP cells (:func:`reference_support`).
        weights: The LP cells' normalized mean target mass.
        n_draws: Random-set draws per method.
    """

    merged = scored.merge(random_sets[["draw_index", "method", "estimand_type", "estimand_key", "nonempty", "estimand_weight"]],
                          on=["draw_index", "method", "estimand_type", "estimand_key"], validate="one_to_one")
    for (method, estimand_type, estimand_key), group in merged.groupby(["method", "estimand_type", "estimand_key"], sort=True):
        keys = {"method": method, "estimand_type": estimand_type, "estimand_key": estimand_key}
        scoreable = bool(group.reference_scoreable.all())
        out.add("reference", "reference_scoreable", scoreable, n_draws, ALL_DRAWS, **keys)
        if not scoreable:
            continue
        nonempty = group.loc[group.nonempty.astype(bool)]
        out.add("reference", "membership_all", group.reference_member.mean(), n_draws, ALL_DRAWS, **keys)
        out.add("reference", "membership_given_nonempty", nonempty.reference_member.mean() if len(nonempty) else np.nan,
                len(nonempty), NONEMPTY, **keys)
        out.add("reference", "wrong_sign_all", group.wrong_sign.mean(), n_draws, ALL_DRAWS, **keys)
        out.add("reference", "wrong_sign_given_nonempty", nonempty.wrong_sign.mean() if len(nonempty) else np.nan,
                len(nonempty), NONEMPTY, **keys)
        out.add("reference", "reference_value_q50", group.reference_value.median(), n_draws, ALL_DRAWS, **keys)
    scored_cells = set(support.loc[support.reference_supported.astype(bool), "estimand_key"])
    excluded = float(weights.loc[~weights.index.isin(scored_cells)].sum())
    out.add("reference", "lp_mass_without_reference_support", excluded, n_draws, ALL_DRAWS)
    cells = merged.loc[merged.estimand_type.eq("cate") & merged.estimand_key.isin(scored_cells)].copy()
    if not cells.empty:
        cells["weighted_member"] = cells.estimand_weight * cells.reference_member
        cells["weighted_wrong"] = cells.estimand_weight * cells.wrong_sign
        per = cells.groupby(["method", "draw_index"]).agg(
            weight=("estimand_weight", "sum"), member=("weighted_member", "sum"), wrong=("weighted_wrong", "sum"),
            nonempty=("nonempty", "all"),
        ).reset_index()
        per["member"] /= per.weight
        per["wrong"] /= per.weight
        for method, group in per.groupby("method", sort=True):
            nonempty = group.loc[group.nonempty.astype(bool)]
            out.add("reference", "weighted_membership_all", group.member.mean(), n_draws, ALL_DRAWS, method=method)
            out.add("reference", "weighted_membership_given_nonempty", nonempty.member.mean() if len(nonempty) else np.nan,
                    len(nonempty), NONEMPTY, method=method)
            out.add("reference", "weighted_wrong_sign_all", group.wrong.mean(), n_draws, ALL_DRAWS, method=method)
            out.add("reference", "weighted_wrong_sign_given_nonempty", nonempty.wrong.mean() if len(nonempty) else np.nan,
                    len(nonempty), NONEMPTY, method=method)


# --- Module A: TabICLv2 PMF draws (the GPU stage) ---------------------------------


def _tabicl_versions() -> dict[str, str | None]:
    """Record the TabICLv2 inference stack that defines a PMF root."""

    import importlib.metadata as metadata

    versions: dict[str, str | None] = {}
    for distribution in ("tabicl", "torch"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def read_pmf_identity(pmf_root: Path) -> dict[str, Any]:
    """Read and validate the identity of a ``draw-pmfs`` root.

    Raises:
        FileNotFoundError: If the root was never initialized.
        ValueError: If it belongs to another experiment or stage.
    """

    path = Path(pmf_root) / "identity.json"
    if not path.is_file():
        raise FileNotFoundError(f"No Exp31 TabICLv2 PMF root at {pmf_root}; run the draw-pmfs command first")
    identity = json.loads(path.read_text(encoding="utf-8"))
    if identity.get("experiment") != EXPERIMENT_NAME or identity.get("stage") != "tabicl_pmf":
        raise ValueError(f"{pmf_root} is not an Exp31 TabICLv2 PMF root")
    return identity


def initialize_pmf_root(
    output_dir: Path,
    prepared_root: Path,
    rct_path: Path,
    conditions: Sequence[ConditionKey],
    *,
    n_source_draws: int = SOURCE_PMF_DRAWS,
    n_target_draws: int = TARGET_PMF_DRAWS,
    resume: bool = False,
    cluster_manifest_path: str | None = None,
    scheme: str = "bootstrap",
) -> dict[str, Any]:
    """Create or validate the immutable root of the TabICLv2 PMF draws.

    The identity fingerprints Module A, the draw counts, the prepared inputs and the inference stack, but not the
    device or the conditions, so one root serves every arm and is filled by resuming it.  The inputs of
    ``conditions`` are re-hashed against the prepared identity.  Only a non-bootstrap ``scheme`` (one of
    :data:`MODULE_A_SCHEMES`) enters the identity.

    Raises:
        ValueError: If the existing identity differs or resume was not given, or the scheme is unknown.
        FileExistsError: If a fresh root's directory is not empty.
    """

    if min(int(n_source_draws), int(n_target_draws)) < 1:
        raise ValueError("PMF draw counts must be positive")
    if scheme not in MODULE_A_SCHEMES:
        raise ValueError(f"Unknown Module A scheme {scheme!r}; choose from {MODULE_A_SCHEMES}")
    prepared = read_prepared_identity(prepared_root)
    verify_prepared_inputs(prepared_root, prepared, conditions, rct_path)
    payload = _identity({
        "experiment": EXPERIMENT_NAME,
        "stage": "tabicl_pmf",
        "module_a": protocol_payload()["module_a"],
        **({"module_a_scheme": scheme} if scheme != "bootstrap" else {}),
        "target_site": TARGET_SITE,
        "covariates": list(COVARIATES),
        "n_source_draws": int(n_source_draws),
        "n_target_draws": int(n_target_draws),
        "prepared_fingerprint": prepared["fingerprint"],
        "dependencies": {**_dependency_versions(), **_tabicl_versions()},
    })
    identity_path = Path(output_dir) / "identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if not resume or existing.get("fingerprint") != payload["fingerprint"]:
            raise ValueError("Existing PMF root identity differs or --resume was not supplied")
        return existing
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _write_json(payload, identity_path)
    write_run_manifest(
        output_dir,
        experiment_name=EXPERIMENT_NAME,
        script_path="experiments/exp31_bayesian_manylabs_naturalness.py",
        config={"stage": "tabicl_pmf", "module_a": protocol_payload()["module_a"], "module_a_scheme": scheme,
                "n_source_draws": int(n_source_draws), "n_target_draws": int(n_target_draws)},
        argv=sys.argv,
        cluster_manifest_path=cluster_manifest_path,
        extra={
            "schema_version": SCHEMA_VERSION,
            "pmf_fingerprint": payload["fingerprint"],
            "git": _git_provenance(),
            "source_fingerprints": _source_fingerprints(),
            "software_environment": _software_environment() | {"tabicl_stack": _tabicl_versions()},
            "input_paths": {"prepared_root": str(Path(prepared_root).resolve()), "rct_path": str(Path(rct_path).resolve())},
            "input_fingerprints": {"prepared": prepared["fingerprint"], "rct_sha256": prepared["rct_sha256"]},
        },
    )
    return payload


def _pmf_condition_identity(key: ConditionKey, pmf_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the identity one condition's leaf must carry under a PMF root."""

    return _identity({
        "experiment": EXPERIMENT_NAME,
        "stage": "tabicl_pmf",
        "condition": key.as_dict(),
        "pmf_fingerprint": pmf_identity["fingerprint"],
    })


def draw_condition_pmfs(
    *,
    key: ConditionKey,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    pmf_identity: Mapping[str, Any],
    device: str | None,
    resume: bool,
) -> None:
    """Draw and store one condition's source and target PMF draws (Module A).

    Source draws cover every source site jointly per draw; target draws cover the target's idle rows.  Both use the
    condition's own seed streams, so they do not depend on the device, shard or arm.  The completion record keeps the
    wall time and the number of TabICLv2 site fits.  Under the ``predictive`` scheme the draws come from
    :func:`predictive_pmfs`, and the full-data fits they are centred on are stored beside them.
    """

    leaf = key.leaf(pmf_root)
    if not _claim_leaf(leaf, _pmf_condition_identity(key, pmf_identity), resume=resume):
        LOGGER.info("Skipping condition with complete PMF draws %s", key.label())
        return
    started = time.monotonic()
    data = load_condition_data(key, prepared_root, rct_path)
    kwargs = {**TABICL_PREDICTOR_KWARGS, "random_state": int(key.seed), "device": device}
    n_source, n_target = int(pmf_identity["n_source_draws"]), int(pmf_identity["n_target_draws"])
    if pmf_identity.get("module_a_scheme", "bootstrap") == "predictive":
        centers, sources, target, seconds = predictive_pmfs(data, key, n_source=n_source, n_target=n_target,
                                                            device=device)
        source_seconds = time.monotonic() - started - seconds["target_draws"]
        _write_parquet(raw_pmf_draws_to_frame([centers]), leaf / "full_data_fits.parquet")
    else:
        sources = draw_tabicl_bootstrap_pmfs(data.sources, n_draws=n_source,
                                             random_seed=_stream_seed(key, "source-pmf"), predictor_kwargs=kwargs,
                                             covariates=COVARIATES)
        source_seconds = time.monotonic() - started
        target = draw_tabicl_bootstrap_pmfs(data.target_idle, n_draws=n_target,
                                            random_seed=_stream_seed(key, "target-pmf"), predictor_kwargs=kwargs,
                                            covariates=COVARIATES)
    target_seconds = time.monotonic() - started - source_seconds
    _write_parquet(raw_pmf_draws_to_frame(sources), leaf / "source_pmf_draws.parquet")
    _write_parquet(raw_pmf_draws_to_frame(target), leaf / "target_pmf_draws.parquet")
    _write_json(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "runtime_seconds": time.monotonic() - started,
            "module_seconds": {"source_draws": source_seconds, "target_draws": target_seconds},
            "device": device,
            # Three TabICLv2 classifiers per source site and draw, two per target draw.
            "n_site_fits": {"source": n_source * int(data.counts["n_source_sites"]), "target": n_target},
            "counts": data.counts,
            "artifact_sha256": _artifact_hashes(leaf),
        },
        leaf / "completion.json",
    )
    LOGGER.info("%s PMF draws complete in %.0f s (sources %.0f s, target %.0f s)", key.label(),
                time.monotonic() - started, source_seconds, target_seconds)


def draw_pmfs(
    output_dir: Path,
    prepared_root: Path,
    rct_path: Path,
    *,
    arm: str,
    device: str | None,
    n_source_draws: int = SOURCE_PMF_DRAWS,
    n_target_draws: int = TARGET_PMF_DRAWS,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    cluster_manifest_path: str | None = None,
    scheme: str = "bootstrap",
    studies: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Draw the TabICLv2 PMFs of one arm's conditions, sharded round-robin.

    Shard 0 creates the root and the others wait for its identity.  Complete leaves are skipped, so arms can fill
    one root in turn; a failed condition is recorded and the others continue.  ``scheme`` selects Module A's draws
    (:data:`MODULE_A_SCHEMES`); ``studies`` selects a full-grid arm's studies (:func:`all_conditions`).

    Returns:
        This shard's record.

    Raises:
        RuntimeError: If any condition failed.
    """

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index} of {num_shards}")
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; choose from {sorted(ARMS)}")
    conditions = arm_conditions(arm, studies=studies)
    mine = [key for position, key in enumerate(conditions) if position % num_shards == shard_index]
    if shard_index > 0:
        _wait_for_run_identity(Path(output_dir))
    identity = initialize_pmf_root(
        output_dir, prepared_root, rct_path, mine, n_source_draws=n_source_draws, n_target_draws=n_target_draws,
        resume=resume or shard_index > 0, cluster_manifest_path=cluster_manifest_path, scheme=scheme,
    )
    failures: list[dict[str, Any]] = []
    for key in mine:
        try:
            draw_condition_pmfs(key=key, prepared_root=Path(prepared_root), rct_path=Path(rct_path),
                                pmf_root=Path(output_dir), pmf_identity=identity, device=device, resume=True)
        except Exception as error:  # keep the other conditions of the shard.
            LOGGER.exception("Exp31 PMF draws failed: %s", key.label())
            failures.append({**key.as_dict(), "error": repr(error)})
            _write_json({"status": "failed", "error": repr(error), "failed_at_utc": _utc_now()},
                        key.leaf(Path(output_dir)) / "completion.json")
    record = {
        "status": "complete" if not failures else "failed",
        "arm": arm,
        "attempted_conditions": len(mine),
        "failures": failures,
        "completed_at_utc": _utc_now(),
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    if failures:
        raise RuntimeError(f"{len(failures)} Exp31 PMF conditions failed")
    return record


def _pmf_leaf_complete(key: ConditionKey, pmf_root: Path) -> bool:
    """Return whether a condition's PMF draws under ``pmf_root`` are complete."""

    path = key.leaf(Path(pmf_root)) / "completion.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"


def load_condition_pmfs(
    key: ConditionKey,
    pmf_root: Path,
    data: ConditionData,
    *,
    pmf_fingerprint: str,
    n_source_draws: int,
    n_target_draws: int,
) -> tuple[list[bayes.RawPMFImputation], list[bayes.SitePMFDraw], dict[str, str]]:
    """Read one condition's stored PMF draws for the sampling stages.

    The leaf must be complete, belong to the run's PMF root and condition, and hold files whose hashes match its
    completion record.

    Returns:
        The first ``n_source_draws`` source imputations, the first ``n_target_draws`` target draws and the SHA-256
        of both tables.

    Raises:
        ValueError: If the draws are missing, foreign, altered or too few.
    """

    leaf = key.leaf(Path(pmf_root))
    completion_path = leaf / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
    if completion.get("status") != "complete":
        raise ValueError(f"No complete TabICLv2 PMF draws for {key.label()} under {pmf_root}; run draw-pmfs first")
    identity = json.loads((leaf / "identity.json").read_text(encoding="utf-8"))
    if identity.get("pmf_fingerprint") != pmf_fingerprint or identity.get("condition") != key.as_dict():
        raise ValueError(f"PMF draws of {key.label()} belong to another PMF root or condition")
    hashes = {}
    for name in ("source_pmf_draws.parquet", "target_pmf_draws.parquet"):
        digest = _sha256(leaf / name)
        if digest != completion["artifact_sha256"].get(name):
            raise ValueError(f"{leaf / name} changed after draw-pmfs wrote it")
        hashes[name] = digest
    sources = raw_pmf_draws_from_frame(pd.read_parquet(leaf / "source_pmf_draws.parquet"), data.sources,
                                       covariates=COVARIATES)
    target = raw_pmf_draws_from_frame(pd.read_parquet(leaf / "target_pmf_draws.parquet"), data.target_idle,
                                      covariates=COVARIATES)
    if len(sources) < n_source_draws or len(target) < n_target_draws:
        raise ValueError(f"{key.label()} has {len(sources)} source and {len(target)} target PMF draws; "
                         f"the run needs {n_source_draws} and {n_target_draws}")
    if any(set(imputation) != set(data.sources.site) for imputation in sources[:n_source_draws]):
        raise ValueError(f"Stored source PMF draws of {key.label()} do not cover its source sites")
    return sources[:n_source_draws], [imputation[TARGET_SITE] for imputation in target[:n_target_draws]], hashes


def full_data_pmfs(
    data: ConditionData,
    key: ConditionKey,
    device: str | None = None,
) -> tuple[dict[str, bayes.SitePMFDraw], bayes.SitePMFDraw]:
    """Fit Module A's TabICLv2 models once to every site's full rows (the modular arm's Module A).

    :func:`draw_tabicl_bootstrap_pmfs` with ``resample=False`` and the run's predictor settings and seeds, i.e.
    Exp25's plug-in estimator on the run's support; ``P(Z | idle)`` is the observed idle share.

    Returns:
        The source imputation (site -> draw) and the target's draw.
    """

    kwargs = {**TABICL_PREDICTOR_KWARGS, "random_state": int(key.seed), "device": device}
    [sources] = draw_tabicl_bootstrap_pmfs(data.sources, n_draws=1, random_seed=_stream_seed(key, "source-pmf"),
                                           predictor_kwargs=kwargs, covariates=COVARIATES, resample=False)
    [target] = draw_tabicl_bootstrap_pmfs(data.target_idle, n_draws=1, random_seed=_stream_seed(key, "target-pmf"),
                                          predictor_kwargs=kwargs, covariates=COVARIATES, resample=False)
    return sources, target[TARGET_SITE]


def _pmf_conditionals(draw: bayes.SitePMFDraw) -> tuple[dict, dict, dict]:
    """Return a draw's ``P(X=1|z)``, ``P(Y=1|X=x,z)`` and randomized ``P(Y=1|do(x),z)``, keyed by ``z`` and ``(z, x)``."""

    p_x = {z: sum(draw.idle_xy_given_z[(z, 1, y)] for y in (0, 1)) for z in draw.z_values}
    p_y = {(z, x): draw.idle_xy_given_z[(z, x, 1)] / (p_x[z] if x else 1.0 - p_x[z])
           for z in draw.z_values for x in (0, 1)}
    p_do = {(z, x): draw.do_y_given_xz[(z, x, 1)] for z in draw.z_values for x in (0, 1)} if draw.do_y_given_xz else {}
    return p_x, p_y, p_do


def _logit_probability(value: float) -> float:
    """Return the logit of a probability floored away from 0 and 1."""

    floor = bayes.NUMERICAL_PROBABILITY_FLOOR
    clipped = float(np.clip(value, floor, 1.0 - floor))
    return float(np.log(clipped / (1.0 - clipped)))


def recentre_draws(draws: Sequence[bayes.SitePMFDraw], center: bayes.SitePMFDraw) -> list[bayes.SitePMFDraw]:
    """Shift posterior draws of one site so that their logit conditionals average to a centre's.

    Each of ``P(X=1|z)``, ``P(Y=1|X=x,z)`` and ``P(Y=1|do(x),z)`` keeps its deviation
    ``logit p_t - mean_t logit p_t`` around ``logit p_center``: the draws supply the spread and the centre the
    location.  The drawn ``P(Z | idle)`` masses are kept.  The modular arm recentres the bootstrap target draws on the
    full-data fit, because resampling raw rows centres TabICLv2's draws on rougher PMFs than the fit.
    """

    tables = [_pmf_conditionals(draw) for draw in draws]
    middle = _pmf_conditionals(center)
    shifts = [{k: _logit_probability(middle[index][k]) - float(np.mean([_logit_probability(t[index][k]) for t in tables]))
               for k in tables[0][index]} for index in range(3)]
    shifted = []
    for draw, table in zip(draws, tables):
        q_x, q_y, q_do = ({k: float(1.0 / (1.0 + np.exp(-(_logit_probability(v) + shifts[index][k]))))
                           for k, v in table[index].items()} for index in range(3))
        idle_xy = {(z, x, y): (q_x[z] if x else 1.0 - q_x[z]) * (q_y[(z, x)] if y else 1.0 - q_y[(z, x)])
                   for z in draw.z_values for x in (0, 1) for y in (0, 1)}
        do_y = {(z, x, y): q_do[(z, x)] if y else 1.0 - q_do[(z, x)]
                for z in draw.z_values for x in (0, 1) for y in (0, 1)} if q_do else {}
        shifted.append(replace(draw, idle_xy_given_z=idle_xy, do_y_given_xz=do_y))
    return shifted


def _predictive_site_fit(on, idle, site, support, overlap, key, device, sample_id, masses) -> bayes.SitePMFDraw:
    """Fit Module A's TabICLv2 models to one site's rows and read them on fixed strata.

    The per-site body of :func:`draw_tabicl_bootstrap_pmfs` with the support and stratum masses supplied, so a refit
    on simulated rows keeps the site's observed strata and counts, as a bootstrap draw does.
    """

    z_values, n_idle_z, n_idle_xz = support
    kwargs = {**TABICL_PREDICTOR_KWARGS, "random_state": int(key.seed), "device": device}
    p_x1, p_y1, p_do = bayes._tabicl_site_probabilities(
        on, idle, site_id=site, z_values=z_values, covariates=list(COVARIATES), treatment=TREATMENT_COLUMN,
        outcome=OUTCOME_COLUMN, regime_col="F", on_value="on", idle_value="idle", predictor_kwargs=kwargs,
    )
    floor = bayes.NUMERICAL_PROBABILITY_FLOOR
    idle_xy = {}
    for z in z_values:
        treated = float(np.clip(p_x1[z], floor, 1.0 - floor))
        for x in (0, 1):
            p_x = treated if x else 1.0 - treated
            for y in (0, 1):
                idle_xy[(z, x, y)] = p_x * (p_y1[(x, z)] if y else 1.0 - p_y1[(x, z)])
    do_y = {(z, x, y): p_do[(x, z)] if y else 1.0 - p_do[(x, z)]
            for z in z_values for x in (0, 1) for y in (0, 1)} if p_do else {}
    return bayes.SitePMFDraw(sample_id=sample_id, site_id=site, covariates=tuple(COVARIATES), z_values=z_values,
                             idle_xy_given_z=idle_xy, do_y_given_xz=do_y,
                             idle_z={z: float(m) for z, m in zip(z_values, masses)}, n_idle_z=n_idle_z,
                             n_idle_xz=n_idle_xz, overlap_model=overlap, alpha=float("nan"))


def _simulate_site_rows(rows: pd.DataFrame, center: bayes.SitePMFDraw, rng: np.random.Generator,
                        redraw_treatment: bool) -> pd.DataFrame:
    """Redraw one site's data from its full-data fit, keeping covariates, regimes and nesting.

    Idle units draw ``Y`` from the fit's idle ``P(Y | x, z)``; randomized units outside the idle
    regime draw it from the complementary rate that keeps the fit's randomized ``P(Y | do(x), z)``
    in expectation; an idle row nested in a randomized unit keeps that unit's outcome.  With
    ``redraw_treatment`` (the target, which has idle rows only) ``X`` is redrawn from the fit's
    ``P(X | z)`` first.  Rows in strata the fit does not cover keep their observed values.
    """

    rows = rows.copy()
    z = [tuple(int(v) for v in values) for values in rows[list(COVARIATES)].to_numpy()]
    fitted = np.array([cell in set(center.z_values) for cell in z])
    p_x, p_y, p_do = _pmf_conditionals(center)
    if redraw_treatment:
        draw = rng.random(len(rows))
        rows[TREATMENT_COLUMN] = np.array([int(u < p_x[cell]) if ok else int(x) for u, cell, ok, x in zip(draw, z, fitted, rows[TREATMENT_COLUMN])])
    x = rows[TREATMENT_COLUMN].astype(int).to_numpy()
    on = rows.F.eq("on").to_numpy()
    idle_ids = set(rows.loc[~on, "id"])
    in_idle = np.array([is_on and unit in idle_ids for is_on, unit in zip(on, rows["id"])])
    rate = np.full(len(rows), np.nan)
    for index, (cell, arm, is_on, ok) in enumerate(zip(z, x, on, fitted)):
        if ok:
            rate[index] = p_y[(cell, arm)] if (not is_on or in_idle[index] or not p_do) else np.nan
    if p_do:
        # Complementary rate of randomized units outside the idle regime, per (z, x).
        frame = pd.DataFrame({"z": z, "x": x, "on": on, "in_idle": in_idle, "fitted": fitted})
        for (cell, arm), group in frame.loc[frame.on & frame.fitted].groupby(["z", "x"]):
            n_idle, n_other = int(group.in_idle.sum()), int((~group.in_idle).sum())
            if n_other:
                total = p_do[(cell, arm)] * (n_idle + n_other) - p_y[(cell, arm)] * n_idle
                rate[group.index[~group.in_idle.to_numpy()]] = float(np.clip(total / n_other, 0.0, 1.0))
    observed = rows.dv_binary.astype(int).to_numpy()
    outcome = np.where(np.isnan(rate), observed, rng.random(len(rows)) < np.nan_to_num(rate)).astype(int)
    if on.any():
        by_unit = dict(zip(rows.loc[on, "id"], outcome[on]))
        nested = ~on & rows["id"].isin(by_unit).to_numpy()
        outcome[nested] = rows.loc[nested, "id"].map(by_unit).to_numpy()
    rows[OUTCOME_COLUMN] = outcome
    return rows


def _site_frames(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple, bool]:
    """Return a site's randomized and idle rows, its observed support and whether idle nests in randomized."""

    on = rows.loc[rows.F.eq("on")].reset_index(drop=True)
    idle = rows.loc[rows.F.eq("idle")].reset_index(drop=True)
    covariates = list(COVARIATES)
    overlap = bayes._validate_overlap(on, idle, id_col="id", columns=[TREATMENT_COLUMN, OUTCOME_COLUMN, *covariates]) if len(on) else False
    return on, idle, bayes._idle_support(idle, covariates, TREATMENT_COLUMN), overlap


def predictive_pmfs(
    data: ConditionData,
    key: ConditionKey,
    *,
    n_source: int,
    n_target: int,
    device: str | None = None,
) -> tuple[dict[str, bayes.SitePMFDraw], list[bayes.RawPMFImputation], list[bayes.RawPMFImputation], dict[str, float]]:
    """Draw Module A's PMFs by predictive resampling around the full-data TabICLv2 fit.

    Each draw simulates every site's data from its full-data fit (:func:`_simulate_site_rows`) and refits Module A's
    TabICLv2 models, so the draws centre on the smooth fit rather than on the raw rows.  This is a parametric
    bootstrap with TabICLv2 as the model: the draws stand in for posterior draws but are not the posterior of an
    explicit prior and likelihood (nor a martingale posterior).  Stratum masses are ``Dirichlet(n_z)`` draws, every
    refit is read on the site's observed strata, and source draws are joint imputations over the source sites.

    Args:
        data: The condition's loaded data.
        key: The condition, whose seed streams fix every draw.
        n_source: Source imputations.
        n_target: Target draws.
        device: TabICLv2 device, ``"cuda"`` on the cluster.

    Returns:
        The full-data fits (site -> draw, target included), the source imputations, the target draws (each
        ``{TARGET_SITE: draw}``) and the seconds spent on the fits, the source draws and the target draws.
    """

    started = time.monotonic()
    sites = {site: rows.reset_index(drop=True) for site, rows in data.sources.groupby(data.sources.site.astype(str))}
    sites[TARGET_SITE] = data.target_idle.reset_index(drop=True)
    frames = {site: _site_frames(rows) for site, rows in sites.items()}
    centers = {}
    for site, (on, idle, support, overlap) in frames.items():
        masses = np.asarray([support[1][z] for z in support[0]], dtype=float)
        centers[site] = _predictive_site_fit(on, idle, site, support, overlap, key, device, "full-data-fit",
                                             masses / masses.sum())
    seconds = {"full_data_fits": time.monotonic() - started}
    seed = _stream_seed(key, "predictive-pmf")

    def draw(site, index):
        """Return one predictive draw of one site."""

        rng = np.random.default_rng(sha256_seed(seed, site, index))
        simulated = _simulate_site_rows(sites[site], centers[site], rng, redraw_treatment=site == TARGET_SITE)
        _, _, support, overlap = frames[site]
        sim_on = simulated.loc[simulated.F.eq("on")].reset_index(drop=True)
        sim_idle = simulated.loc[simulated.F.eq("idle")].reset_index(drop=True)
        masses = rng.dirichlet(np.asarray([support[1][z] for z in support[0]], dtype=float))
        return _predictive_site_fit(sim_on, sim_idle, site, support, overlap, key, device,
                                    bayes._stable_sample_id("predictive", seed, index), masses)

    sources = []
    for index in range(int(n_source)):
        sources.append({site: draw(site, index) for site in sites if site != TARGET_SITE})
        LOGGER.info("%s: predictive source draw %d of %d", key.label(), index + 1, n_source)
    seconds["source_draws"] = time.monotonic() - started - seconds["full_data_fits"]
    targets = []
    for index in range(int(n_target)):
        targets.append({TARGET_SITE: draw(TARGET_SITE, index)})
        if (index + 1) % 50 == 0:
            LOGGER.info("%s: predictive target draw %d of %d", key.label(), index + 1, n_target)
    seconds["target_draws"] = time.monotonic() - started - seconds["full_data_fits"] - seconds["source_draws"]
    return centers, sources, targets, seconds


def _recorded_source_fingerprints(root: Path) -> dict[str, Any]:
    """Return the source fingerprints a root's run manifest recorded when it was created.

    Raises:
        ValueError: If the manifest recorded none.
    """

    import yaml

    def find(node: Any) -> Any:
        """Return the first ``source_fingerprints`` mapping in a nested manifest."""

        if isinstance(node, Mapping):
            if "source_fingerprints" in node:
                return node["source_fingerprints"]
            for value in node.values():
                found = find(value)
                if found is not None:
                    return found
        return None

    recorded = find(yaml.safe_load((Path(root) / "run_manifest.yml").read_text(encoding="utf-8")))
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{root} recorded no source fingerprints")
    return dict(recorded)


def adopt_pmfs(
    output_dir: Path,
    prepared_root: Path,
    *,
    source_root: Path,
    source_prepared_root: Path,
    arm: str = "primary",
) -> dict[str, Any]:
    """Adopt another PMF root's draws for conditions whose inputs are unchanged.

    Draws are copied only when they are what ``draw-pmfs`` would write here: both roots agree on Module A, its
    scheme, the draw counts, target site, covariates and inference stack, recorded the same fingerprint for every
    :data:`MODULE_A_SOURCES` file, and share the RCT pickle and the condition's input hash (draw seeds depend on the
    condition only).  The source leaf must be complete and unaltered; a condition this root already holds, complete
    or in progress, is never touched.  Each adopted leaf gets this root's identity and a completion record naming
    its origin.  Run it only when no ``draw-pmfs`` job writes to either root.

    Args:
        output_dir: The PMF root to adopt into, already created by ``draw-pmfs``.
        prepared_root: The prepared root this PMF root was drawn from.
        source_root: The PMF root to adopt from.
        source_prepared_root: The prepared root ``source_root`` was drawn from.
        arm: Whose conditions to consider; ``primary`` is the full grid.

    Returns:
        The adoption record, also written under ``adoptions/`` in ``output_dir``.

    Raises:
        ValueError: If the roots or a source leaf are not interchangeable.
    """

    output_dir, source_root = Path(output_dir), Path(source_root)
    target, source = read_pmf_identity(output_dir), read_pmf_identity(source_root)
    for key in ("module_a", "module_a_scheme", "n_source_draws", "n_target_draws", "target_site", "covariates",
                "dependencies"):
        if target.get(key) != source.get(key):
            raise ValueError(f"The PMF roots differ in {key!r}; their draws are not interchangeable")
    fingerprints = _recorded_source_fingerprints(output_dir)
    source_fingerprints = _recorded_source_fingerprints(source_root)
    changed = [path for path in MODULE_A_SOURCES
               if fingerprints.get(path) is None or fingerprints.get(path) != source_fingerprints.get(path)]
    if changed:
        raise ValueError(f"Module A sources differ between the PMF roots: {changed}")
    prepared, source_prepared = read_prepared_identity(prepared_root), read_prepared_identity(source_prepared_root)
    if target["prepared_fingerprint"] != prepared["fingerprint"]:
        raise ValueError(f"{output_dir} was not drawn from {prepared_root}")
    if source["prepared_fingerprint"] != source_prepared["fingerprint"]:
        raise ValueError(f"{source_root} was not drawn from {source_prepared_root}")
    if prepared["rct_sha256"] != source_prepared["rct_sha256"]:
        raise ValueError("The two prepared roots were generated from different RCT pickles")

    def input_hashes(identity: Mapping[str, Any]) -> dict[tuple[str, str, float, int], str]:
        """Index a prepared identity's input hashes by condition."""

        return {(row["study"], row["pattern"], float(row["beta"]), int(row["replication"])): row["sha256"]
                for row in identity["inputs"]}

    inputs, source_inputs = input_hashes(prepared), input_hashes(source_prepared)
    outcomes: dict[str, list[str]] = {"adopted": [], "already_in_target": [], "not_complete_in_source": [],
                                      "input_differs": []}
    for key in arm_conditions(arm):
        index = (key.study, key.pattern, float(key.beta), key.replication)
        leaf, source_leaf = key.leaf(output_dir), key.leaf(source_root)
        if (leaf / "identity.json").is_file():
            outcomes["already_in_target"].append(key.label())
            continue
        if not _pmf_leaf_complete(key, source_root):
            outcomes["not_complete_in_source"].append(key.label())
            continue
        if index not in inputs:
            raise ValueError(f"{key.label()} was not prepared in {prepared_root}")
        if source_inputs.get(index) != inputs[index]:
            outcomes["input_differs"].append(key.label())
            continue
        source_identity = json.loads((source_leaf / "identity.json").read_text(encoding="utf-8"))
        if source_identity.get("pmf_fingerprint") != source["fingerprint"] or source_identity.get("condition") != key.as_dict():
            raise ValueError(f"{source_leaf} does not belong to {source_root}")
        completion = json.loads((source_leaf / "completion.json").read_text(encoding="utf-8"))
        for name in PMF_DATA_FILES:
            if _sha256(source_leaf / name) != completion["artifact_sha256"].get(name):
                raise ValueError(f"{source_leaf / name} changed after draw-pmfs wrote it")
        leaf.mkdir(parents=True, exist_ok=True)
        for name in PMF_DATA_FILES:
            temporary = leaf / f".{name}.adopting"
            shutil.copyfile(source_leaf / name, temporary)
            os.replace(temporary, leaf / name)
        _write_json(_pmf_condition_identity(key, target), leaf / "identity.json")
        _write_json(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "runtime_seconds": completion.get("runtime_seconds"),
                "module_seconds": completion.get("module_seconds"),
                "device": completion.get("device"),
                "n_site_fits": completion.get("n_site_fits"),
                "counts": completion.get("counts"),
                "adopted_from": {
                    "pmf_root": str(source_root.resolve()),
                    "pmf_fingerprint": source["fingerprint"],
                    "drawn_at_utc": completion.get("completed_at_utc"),
                    "input_sha256": inputs[index],
                },
                "artifact_sha256": _artifact_hashes(leaf),
            },
            leaf / "completion.json",
        )
        outcomes["adopted"].append(key.label())
    record = {
        "adopted_at_utc": _utc_now(),
        "arm": arm,
        "pmf_fingerprint": target["fingerprint"],
        "source_pmf_root": str(source_root.resolve()),
        "source_pmf_fingerprint": source["fingerprint"],
        "counts": {name: len(labels) for name, labels in outcomes.items()},
        "conditions": outcomes,
    }
    _write_json(record, output_dir / "adoptions" / f"adoption_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    LOGGER.info("Adopted %d PMF conditions from %s (%s)", len(outcomes["adopted"]), source_root, record["counts"])
    return record


def _settings_payload(settings: Exp31RunSettings) -> dict[str, Any]:
    """Return the settings a run identity fingerprints.

    The cluster manifest path is blanked, and the evidence subset and study selection are recorded only when used,
    so runs without them keep their identities.
    """

    payload = asdict(settings) | {"cluster_manifest_path": None}
    if not payload["evidence_min_randomized_rows"]:
        del payload["evidence_min_randomized_rows"]
    if payload["studies"] is None:
        del payload["studies"]
    else:
        payload["studies"] = list(payload["studies"])
    return payload


def _condition_identity(key: ConditionKey, run_identity: Mapping[str, Any], settings: Exp31RunSettings) -> dict[str, Any]:
    """Return the identity a condition leaf must carry under one run."""

    return _identity({
        "experiment": EXPERIMENT_NAME,
        "arm": settings.arm,
        "condition": key.as_dict(),
        "run_fingerprint": run_identity["fingerprint"],
        "prepared_fingerprint": run_identity["prepared_fingerprint"],
        "pmf_fingerprint": run_identity["pmf_fingerprint"],
    })


def _claim_leaf(leaf: Path, identity: Mapping[str, Any], *, resume: bool) -> bool:
    """Decide whether a leaf must be computed, and never overwrite another run.

    Returns:
        ``False`` when the leaf is already complete under this identity.

    Raises:
        FileExistsError: If the leaf belongs to a different identity or exists without ``resume``.
    """

    identity_path = leaf / "identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != identity["fingerprint"]:
            raise FileExistsError(f"{leaf} belongs to a different run identity; refusing to overwrite it")
        if not resume:
            raise FileExistsError(f"Condition already exists: {leaf}")
        completion = leaf / "completion.json"
        if completion.is_file() and json.loads(completion.read_text(encoding="utf-8")).get("status") == "complete":
            return False
    leaf.mkdir(parents=True, exist_ok=True)
    _write_json(dict(identity), identity_path)
    return True


def run_condition(
    *,
    key: ConditionKey,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    output_dir: Path,
    settings: Exp31RunSettings,
    run_identity: Mapping[str, Any],
    resume: bool,
    isolate_fits: bool = True,
) -> None:
    """Run Modules A-E for one condition and persist its artifacts.

    Module A reads the condition's TabICLv2 PMF draws (made by :func:`draw_pmfs`); Module B recomputes the source
    evidence of every imputation; Module C fits the candidate surfaces with NUTS; Module D propagates paired draws
    through the Naturalness and matched Manski LPs once per mixture size, with shared target draws and seeds; and
    Module E scores the random sets against the randomized reference.  Nothing downstream feeds back upstream.  The
    leave-one-out arms (:data:`LOSO_ARMS`) replace Modules D and E by leave-one-source-site-out refits.

    Args:
        key: The condition.
        prepared_root: Exp31 prepared root.
        rct_path: ManyLabs RCT pickle.
        pmf_root: The ``draw-pmfs`` root holding the condition's PMF draws.
        output_dir: Run root.
        settings: Resolved run settings.
        run_identity: The run's identity.
        resume: Skip a leaf that is already complete under this identity.
        isolate_fits: Run each imputation's NUTS fits, or each leave-one-out refit, in its own process.
    """

    leaf = key.leaf(output_dir)
    identity = _condition_identity(key, run_identity, settings)
    if not _claim_leaf(leaf, identity, resume=resume):
        LOGGER.info("Skipping completed condition %s", key.label())
        return
    started = time.monotonic()
    timings: dict[str, float] = {}
    prepared_row = next(
        row for row in read_prepared_identity(prepared_root)["inputs"]
        if (row["study"], row["pattern"], float(row["beta"]), int(row["replication"]))
        == (key.study, key.pattern, float(key.beta), key.replication)
    )
    if _sha256(Path(prepared_root) / prepared_row["path"]) != prepared_row["sha256"]:
        raise ValueError(f"Prepared input for {key.label()} changed after preparation")

    data = load_condition_data(key, prepared_root, rct_path)
    _write_parquet(data.audit, leaf / "data_role_audit.parquet")
    timings["data"] = time.monotonic() - started

    n_fit = max(settings.imputation_counts)
    # The random sets cycle through the target draws, as the reference draws do,
    # so random set d and reference draw d always share one target draw.
    n_sets = settings.random_set_draws
    source_imputations, target_draws, pmf_hashes = load_condition_pmfs(
        key, pmf_root, data, pmf_fingerprint=run_identity["pmf_fingerprint"], n_source_draws=n_fit,
        n_target_draws=min(int(run_identity["pmf_draws"]["n_target_draws"]), n_sets),
    )
    modular = settings.arm == "modular"
    if modular:
        # Modular arm: the full-data fit is the evidence and the centre of the target draws.
        full_sources, full_target = full_data_pmfs(data, key)
        source_imputations = [full_sources]
        target_draws = recentre_draws(target_draws, full_target)
        _write_parquet(raw_pmf_draws_to_frame([{**full_sources, TARGET_SITE: full_target}]),
                       leaf / "full_data_fits.parquet")
    timings["module_a"] = time.monotonic() - started - sum(timings.values())

    evidence = _source_evidence(source_imputations, data.sources, settings.evidence_min_randomized_rows)
    _write_parquet(evidence, leaf / "source_evidence_draws.parquet")
    partial = leaf / "tolerance_surface_diagnostics.partial.parquet"
    mixtures, diagnostics = fit_surface_mixture(evidence, key, settings, partial_path=partial, isolate=isolate_fits)
    _write_parquet(diagnostics, leaf / "tolerance_surface_diagnostics.parquet")
    # A failed earlier attempt of this leaf may have left partial diagnostics.
    partial.unlink(missing_ok=True)
    timings["module_bc"] = time.monotonic() - started - sum(timings.values())

    strata = lp_strata(target_draws)
    if settings.arm in LOSO_ARMS:
        _run_loso(key, leaf, evidence, mixtures[0], target_draws, settings, data, strata,
                  isolate_sites=isolate_fits)
        timings["loso"] = time.monotonic() - started - sum(timings.values())
    else:
        support = reference_support(data.reference, data.target_idle, strata)
        references = draw_reference(data.reference, target_draws, support, key, n_sets)
        _write_parquet(references, leaf / "rct_reference_draws.parquet")
        for count in settings.imputation_counts:
            random_sets, triggers, tolerances = _mixture_random_sets(key, settings, target_draws, mixtures, diagnostics,
                                                                     count, n_draws=n_sets)
            scored = score_random_sets(random_sets, references)
            metrics = _mixture_metrics(key, settings, count, data=data, target_draws=target_draws, random_sets=random_sets,
                                       triggers=triggers, tolerances=tolerances, scored=scored, support=support,
                                       evidence=evidence, diagnostics=diagnostics)
            for name, table in zip(MIXTURE_FILES, (triggers, tolerances, random_sets, scored, metrics)):
                _write_parquet(table, leaf / f"imputations={count}" / name)
        timings["module_de"] = time.monotonic() - started - sum(timings.values())
    _write_json(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "runtime_seconds": time.monotonic() - started,
            "module_seconds": timings,
            "counts": data.counts,
            "pmf_draws_sha256": pmf_hashes,
            "artifact_sha256": _artifact_hashes(leaf),
        },
        leaf / "completion.json",
    )
    LOGGER.info("%s complete in %.0f s", key.label(), time.monotonic() - started)


def _mixture_random_sets(
    key: ConditionKey,
    settings: Exp31RunSettings,
    target_draws: Sequence[bayes.SitePMFDraw],
    mixtures: Sequence[Mapping[tuple[str, str], bayes.ToleranceSurfacePosterior]],
    diagnostics: pd.DataFrame,
    count: int,
    *,
    n_draws: int,
    taus: Sequence[float] | None = None,
    saved_tolerances: Mapping[tuple[str, str], tuple[float, int, float, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw and label one mixture size's random identified sets (Module D).

    The first ``count`` imputations' surfaces are paired with the target draws under the condition's random-set
    seed, shared by every mixture size so that the pilot comparison isolates the number of imputations.  ``taus``
    and ``saved_tolerances`` serve :func:`regrid_condition`; a run uses :data:`TRIGGER_TAUS` and samples tolerances.

    Returns:
        The random identified sets, target trigger draws and tolerance draws of :func:`random_set_tables`.
    """

    result = draw_random_identified_sets(
        target_draws,
        mixtures[:count],
        n_draws=n_draws,
        min_cell_count=LP_MIN_CELL_COUNT,
        random_seed=_stream_seed(key, "random-sets"),
        epsilons=TRIGGER_EPSILONS,
        taus=TRIGGER_TAUS if taus is None else taus,
        require_source_support=REQUIRE_SOURCE_SUPPORT and settings.arm not in UNGATED_ARMS,
        saved_tolerances=saved_tolerances,
    )
    excluded = diagnostics.loc[
        diagnostics.excluded.astype(bool) & diagnostics.imputation_index.lt(count), ["imputation_index", "candidate"]
    ].itertuples(index=False, name=None)
    return random_set_tables(result, mixtures[:count], excluded)


def _mixture_metrics(
    key: ConditionKey,
    settings: Exp31RunSettings,
    count: int,
    *,
    data: ConditionData,
    target_draws: Sequence[bayes.SitePMFDraw],
    random_sets: pd.DataFrame,
    triggers: pd.DataFrame,
    tolerances: pd.DataFrame,
    scored: pd.DataFrame,
    support: pd.DataFrame,
    evidence: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Return one mixture size's condition metrics, with its reference-support rows and condition keys."""

    lp_cells = set(lp_strata(target_draws))
    outside = [sum(mass for z, mass in draw.idle_z.items() if z not in lp_cells) for draw in target_draws]
    metrics = condition_metrics(
        random_sets=random_sets,
        triggers=triggers,
        tolerances=tolerances,
        scored=scored,
        support=support,
        evidence=evidence.loc[evidence.imputation_index.lt(count)],
        diagnostics=diagnostics.loc[diagnostics.imputation_index.lt(count)],
        counts=data.counts,
        target_mass_outside_lp=float(np.mean(outside)),
        n_unobserved_target_strata=2 ** len(COVARIATES) - len(target_draws[0].z_values),
    )
    support_rows = support.melt(id_vars=["estimand_key"], value_vars=[
        "reference_n_arm0", "reference_n_arm1", "target_idle_n_arm0", "target_idle_n_arm1", "reference_supported",
    ], var_name="metric")
    support_metrics = pd.DataFrame({
        "scope": "data", "method": "", "estimand_type": "cate", "estimand_key": support_rows.estimand_key,
        "candidate": "", "metric": support_rows.metric, "value": support_rows.value.astype(float),
        "denominator": 1, "denominator_label": "LP cell",
    })
    metrics = pd.concat([metrics, support_metrics], ignore_index=True)
    for name, value in {**key.as_dict(), "arm": settings.arm, "num_imputations": int(count)}.items():
        metrics[name] = value
    return metrics


def _run_loso(
    key: ConditionKey,
    leaf: Path,
    evidence: pd.DataFrame,
    full_fit: Mapping[tuple[str, str], bayes.ToleranceSurfacePosterior],
    target_draws: Sequence[bayes.SitePMFDraw],
    settings: Exp31RunSettings,
    data: ConditionData,
    strata: Sequence[tuple[int, ...]],
    *,
    isolate_sites: bool = True,
) -> None:
    """Refit the first imputation without each source site and score it.

    Each candidate's target query is the posterior median of its target trigger statistics; the full fit's
    predictive median there is written beside every refit's, showing each omitted site's influence on the grounded
    tolerance.  Held-out evidence is a diagnostic only and never retunes the model.

    With ``isolate_sites`` each held-out site's refit runs in a fresh process, which bounds the memory growth of
    repeated NUTS fits; seeds are fixed per site, so the scores are unchanged.
    """

    statistics = pd.concat([bayes.compute_trigger_statistics(draw) for draw in target_draws], ignore_index=True)
    queries = statistics.groupby("candidate")[["i_deviation", "d_strength"]].median()
    target_queries = {str(candidate): tuple(row) for candidate, row in queries.iterrows()}
    first = evidence.loc[evidence.imputation_index.eq(0)].rename(columns={"imputation_id": "sample_id"})
    options = {
        "target_queries": target_queries,
        "random_seed": _stream_seed(key, "loso"),
        "max_posterior_draws": settings.loso_score_draws,
        **settings.fit_kwargs(),
    }
    if isolate_sites:
        context = multiprocessing.get_context("spawn")
        frames = []
        with ProcessPoolExecutor(max_workers=1, mp_context=context, max_tasks_per_child=1,
                                 initializer=_configure_logging) as pool:
            for site in safe_sorted(first.site.astype(str).unique()):
                frames.append(pool.submit(leave_one_source_out_scores, first, held_out_sites=[site], **options).result())
        scores = pd.concat(frames, ignore_index=True)
    else:
        scores = leave_one_source_out_scores(first, **options)
    full_medians = {}
    for (candidate, kind), posterior in full_fit.items():
        query = list(target_queries[candidate])[: len(posterior.feature_names)]
        predictive = posterior_predictive_tolerances(
            posterior, query, max_posterior_draws=settings.loso_score_draws,
            random_seed=_stream_seed(key, "loso-full", candidate, kind),
        )
        full_medians[(candidate, kind)] = float(predictive.tolerance.median())
    scores["full_fit_target_predictive_q50"] = [
        full_medians.get((candidate, kind), np.nan) for candidate, kind in zip(scores.candidate, scores.tolerance_kind)
    ]
    scores["target_tolerance_shift"] = scores.get("target_predictive_q50", np.nan) - scores.full_fit_target_predictive_q50
    for name, value in key.as_dict().items():
        scores[name] = value
    _write_parquet(scores, leaf / "loso_predictive_scores.parquet")
    out = _MetricRows()
    for name, value in data.counts.items():
        out.add("data", name, value, value, ROWS)
    out.add("data", "n_lp_cells", len(strata), len(strata), "LP cells")
    for (candidate, kind), group in scores.groupby(["candidate", "tolerance_kind"], sort=True):
        scoreable = group.loc[group.scored.astype(bool)]
        supported = scoreable.loc[scoreable.in_source_support.astype(bool)]
        tags = {"candidate": candidate, "estimand_key": kind}
        out.add("loso", "mean_validity_probability", scoreable.validity_probability.mean(), len(scoreable), HELD_OUT_UNITS, **tags)
        out.add("loso", "mean_validity_probability_in_support", supported.validity_probability.mean(), len(supported),
                HELD_OUT_UNITS, **tags)
        finite = scoreable.log_predictive_density.replace(-np.inf, np.nan).dropna()
        out.add("loso", "mean_log_predictive_density_finite", finite.mean(), len(finite), HELD_OUT_UNITS, **tags)
        out.add("loso", "n_inadmissible_held_out_units", len(scoreable) - len(finite), len(scoreable), HELD_OUT_UNITS, **tags)
        out.add("loso", "share_held_out_in_support", scoreable.in_source_support.mean(), len(scoreable), HELD_OUT_UNITS, **tags)
        out.add("loso", "max_abs_target_tolerance_shift", scoreable.target_tolerance_shift.abs().max(), len(scoreable),
                HELD_OUT_UNITS, **tags)
        out.add("loso", "n_refit_failures", group.fit_error.ne("").sum(), len(group), HELD_OUT_UNITS, **tags)
    metrics = out.frame()
    for name, value in {**key.as_dict(), "arm": settings.arm, "num_imputations": 1}.items():
        metrics[name] = value
    _write_parquet(metrics, leaf / "condition_metrics.parquet")


def _run_condition_in_fresh_process(**kwargs: Any) -> None:
    """Run one condition in a newly spawned interpreter.

    Bounds resident memory to one condition, whatever JAX or glibc retain between fits; results are unchanged since
    every stream is seeded from the condition.  A worker exception, including a killed worker, is re-raised here.
    """

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_configure_logging) as pool:
        pool.submit(run_condition, **kwargs).result()


# --- Run orchestration ----------------------------------------------------------


def _read_pilot_decision(path: Path, settings: Exp31RunSettings, pmf_fingerprint: str) -> dict[str, Any]:
    """Require a pilot decision (status ``frozen``) that matches these compute settings.

    Raises:
        ValueError: If the decision is not frozen, or its imputation count, NUTS budget, draw count or PMF root
            differ from the run's.
    """

    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    if decision.get("experiment") != EXPERIMENT_NAME or decision.get("status") != "frozen":
        raise ValueError("The primary arm needs a frozen Exp31 pilot decision")
    if tuple(settings.imputation_counts) != (int(decision["selected_num_imputations"]),):
        raise ValueError(
            f"The pilot froze {decision['selected_num_imputations']} imputations; got {settings.imputation_counts}"
        )
    if decision["nuts_settings"] != settings.nuts_settings():
        raise ValueError(f"NUTS settings differ from the pilot's frozen {decision['nuts_settings']}")
    if int(decision["random_set_draws"]) != int(settings.random_set_draws):
        raise ValueError("random_set_draws differs from the pilot's")
    if decision.get("pmf_fingerprint") != pmf_fingerprint:
        raise ValueError("The pilot decision was made on the PMF draws of another draw-pmfs root")
    return decision


def initialize_run_root(
    output_dir: Path,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    settings: Exp31RunSettings,
    *,
    resume: bool,
    pilot_decision: Path | None = None,
) -> dict[str, Any]:
    """Create or validate an immutable run root and return its identity.

    The identity fingerprints the protocol, the arm and its settings, the conditions, the prepared inputs, the
    ``draw-pmfs`` root and, for the primary arm, the pilot decision; resume is refused if any differs.  Every
    condition's PMF draws must already be complete.

    Raises:
        ValueError: If the existing identity differs or resume was not given, or the PMF root does not fit the run.
        FileExistsError: If a fresh run's directory is not empty.
    """

    settings.validate()
    prepared = read_prepared_identity(prepared_root)
    conditions = arm_conditions(settings.arm, settings.replications, settings.studies)
    verify_prepared_inputs(prepared_root, prepared, conditions, rct_path)
    pmf = read_pmf_identity(pmf_root)
    if pmf["prepared_fingerprint"] != prepared["fingerprint"]:
        raise ValueError(f"PMF root {pmf_root} was drawn from other prepared inputs")
    if int(pmf["n_source_draws"]) < max(settings.imputation_counts):
        raise ValueError(f"PMF root has {pmf['n_source_draws']} source draws; the run needs {max(settings.imputation_counts)}")
    n_target = int(pmf["n_target_draws"])
    if n_target < settings.random_set_draws and settings.random_set_draws % n_target:
        raise ValueError(f"random_set_draws must be a multiple of the PMF root's {n_target} target draws")
    scheme = pmf.get("module_a_scheme", "bootstrap")
    if (scheme == "predictive" and settings.arm not in PREDICTIVE_ROOT_ARMS) or (
            scheme != "predictive" and settings.arm == "predictive"):
        raise ValueError(f"Arm {settings.arm!r} cannot run on a {scheme!r} PMF root: the predictive arm runs only on "
                         f"a root drawn with --module-a-scheme predictive, which only {sorted(PREDICTIVE_ROOT_ARMS)} use")
    decision = None
    if ARMS[settings.arm]["needs_pilot_decision"]:
        if pilot_decision is None:
            raise ValueError(f"Arm {settings.arm!r} needs --pilot-decision from a finalized pilot run")
        decision = _read_pilot_decision(pilot_decision, settings, pmf["fingerprint"])
    missing = [key.label() for key in conditions if not _pmf_leaf_complete(key, pmf_root)]
    if missing:
        raise ValueError(f"{len(missing)} conditions have no complete PMF draws in {pmf_root}, e.g. {missing[0]}")
    payload = _identity({
        "experiment": EXPERIMENT_NAME,
        "stage": "bayesian_run",
        "arm": settings.arm,
        "protocol": protocol_payload(),
        "settings": _settings_payload(settings),
        "conditions": [key.as_dict() for key in conditions],
        "prepared_fingerprint": prepared["fingerprint"],
        "pmf_fingerprint": pmf["fingerprint"],
        "pmf_draws": {"n_source_draws": int(pmf["n_source_draws"]), "n_target_draws": n_target},
        "pilot_decision_fingerprint": None if decision is None else decision["fingerprint"],
        "dependencies": _dependency_versions(),
        # Only the modular and predictive arms record a design of their own; other identities omit the key.
        **({"modular_design": MODULAR_DESIGN} if settings.arm == "modular" else {}),
        **({"predictive_design": PREDICTIVE_DESIGN} if settings.arm == "predictive" else {}),
    })
    identity_path = Path(output_dir) / "identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if not resume or existing.get("fingerprint") != payload["fingerprint"]:
            raise ValueError("Existing run identity differs or --resume was not supplied")
        return existing
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _write_json(payload, identity_path)
    write_run_manifest(
        output_dir,
        experiment_name=EXPERIMENT_NAME,
        script_path="experiments/exp31_bayesian_manylabs_naturalness.py",
        config={"protocol": protocol_payload(), "settings": asdict(settings), "arm": settings.arm},
        argv=sys.argv,
        cluster_manifest_path=settings.cluster_manifest_path,
        extra={
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": payload["fingerprint"],
            "git": _git_provenance(),
            "source_fingerprints": _source_fingerprints(),
            "software_environment": _software_environment(),
            "input_paths": {"prepared_root": str(Path(prepared_root).resolve()), "rct_path": str(Path(rct_path).resolve()),
                            "pmf_root": str(Path(pmf_root).resolve())},
            "input_fingerprints": {"prepared": prepared["fingerprint"], "rct_sha256": prepared["rct_sha256"],
                                   "pmf": pmf["fingerprint"]},
            "row_counts": [row for row in prepared["inputs"] if row["replication"] in {key.replication for key in conditions}],
            "frozen_seeds": list(SEEDS),
            "pilot_decision": decision,
            "n_conditions": len(conditions),
        },
    )
    return payload


def run_experiment(
    output_dir: Path,
    prepared_root: Path,
    rct_path: Path,
    settings: Exp31RunSettings,
    *,
    pmf_root: Path,
    resume: bool = False,
    pilot_decision: Path | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    isolate_conditions: bool = True,
) -> dict[str, Any]:
    """Run every condition of one arm, sharded round-robin, then finalize.

    Shards share one root: shard 0 creates it and the others wait for its identity; each shard records its
    completion under ``shards/``, and the shard that finds every record present finalizes the root
    (:func:`finalize_run`).  A failed condition is recorded and the others continue.

    Returns:
        This shard's completion record.

    Raises:
        RuntimeError: If any condition failed.
    """

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index} of {num_shards}")
    if shard_index > 0:
        _wait_for_run_identity(Path(output_dir))
    identity = initialize_run_root(
        output_dir, prepared_root, rct_path, pmf_root, settings, resume=resume or shard_index > 0,
        pilot_decision=pilot_decision,
    )
    conditions = arm_conditions(settings.arm, settings.replications, settings.studies)
    failures: list[dict[str, Any]] = []
    attempted = 0
    for position, key in enumerate(conditions):
        if position % num_shards != shard_index:
            continue
        attempted += 1
        try:
            (_run_condition_in_fresh_process if isolate_conditions else run_condition)(
                key=key,
                prepared_root=Path(prepared_root),
                rct_path=Path(rct_path),
                pmf_root=Path(pmf_root),
                output_dir=Path(output_dir),
                settings=settings,
                run_identity=identity,
                resume=True,
                isolate_fits=isolate_conditions,
            )
        except Exception as error:  # keep the other expensive conditions.
            LOGGER.exception("Exp31 condition failed: %s", key.label())
            failures.append({**key.as_dict(), "error": repr(error)})
            leaf = key.leaf(output_dir)
            leaf.mkdir(parents=True, exist_ok=True)
            _write_json({"status": "failed", "error": repr(error), "failed_at_utc": _utc_now()}, leaf / "completion.json")
    record = {
        "status": "complete" if not failures else "failed",
        "attempted_conditions": attempted,
        "failures": failures,
        "completed_at_utc": _utc_now(),
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    shard_path = Path(output_dir) / "shards" / f"shard={shard_index:03d}-of-{num_shards:03d}.json"
    _write_json(record, shard_path)
    paths = [Path(output_dir) / "shards" / f"shard={index:03d}-of-{num_shards:03d}.json" for index in range(num_shards)]
    if all(path.is_file() for path in paths):
        records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if all(entry["status"] == "complete" for entry in records):
            lock = Path(output_dir) / "finalize.lock"
            try:
                lock.mkdir()
            except FileExistsError:
                LOGGER.info("Another shard is finalizing %s", output_dir)
            else:
                try:
                    finalize_run(output_dir)
                finally:
                    # Finalizing is idempotent; releasing the lock lets a
                    # manual `finalize` retry after a validation failure.
                    lock.rmdir()
    if failures:
        raise RuntimeError(f"{len(failures)} Exp31 conditions failed")
    return record


def _required_columns() -> dict[str, set[str]]:
    """Return the columns each validated artifact must carry."""

    return {
        "data_role_audit.parquet": {"study", "site", "participant_id", "replication", "beta", "role", "exclusion_reason"},
        "source_evidence_draws.parquet": {"imputation_index", "imputation_id", "site", "candidate", "i_deviation", "d_strength",
                                          "required_two_c", "required_two_d", "source_grid_triggered"},
        "tolerance_surface_diagnostics.parquet": {"imputation_index", "candidate", "tolerance_kind", "num_chains", "num_warmup",
                                                  "num_samples", "nuts_seed", "n_divergent", "max_r_hat", "min_bulk_ess",
                                                  "min_tail_ess", "passed", "attempt", "excluded"},
        "rct_reference_draws.parquet": {"reference_draw_index", "estimand_type", "estimand_key", "reference_value"},
        "target_trigger_draws.parquet": {"draw_index", "sample_id", "candidate", "triggered", "in_source_support", "eligible",
                                         "ineligibility_reason"},
        "tolerance_surface_draws.parquet": {"draw_index", "candidate", "two_c", "two_d", "c", "d", "c_posterior_index",
                                            "d_posterior_index", "source_imputation_index"},
        "random_identified_sets.parquet": {"draw_index", "sample_id", "target_pmf_draw_index", "source_imputation_index", "method",
                                           "estimand_type", "estimand_key", "lower", "upper", "width", "lp_outcome", "nonempty",
                                           "candidate_set", "no_candidate_eligible", "used_manski_fallback",
                                           "eligible_but_lp_infeasible", "lower_status", "upper_status"},
        "scored_random_sets.parquet": {"draw_index", "method", "estimand_key", "reference_scoreable", "reference_member", "wrong_sign"},
        "condition_metrics.parquet": {"scope", "metric", "value", "denominator", "denominator_label", "study", "pattern", "beta",
                                      "replication", "num_imputations"},
        "loso_predictive_scores.parquet": {"held_out_site", "candidate", "tolerance_kind", "validity_probability",
                                           "log_predictive_density", "target_tolerance_shift"},
    }


def validate_condition_leaf(
    leaf: Path,
    key: ConditionKey,
    run_identity: Mapping[str, Any],
    settings: Exp31RunSettings,
) -> list[str]:
    """Check one condition leaf's completion, identity, schemas and counts.

    Returns:
        Problems found; empty when the leaf is valid.
    """

    problems: list[str] = []
    completion = leaf / "completion.json"
    if not completion.is_file() or json.loads(completion.read_text(encoding="utf-8")).get("status") != "complete":
        return [f"{key.label()}: not complete"]
    identity = json.loads((leaf / "identity.json").read_text(encoding="utf-8"))
    if identity.get("fingerprint") != _condition_identity(key, run_identity, settings)["fingerprint"]:
        problems.append(f"{key.label()}: identity does not belong to this run")
    required = _required_columns()
    names = LOSO_FILES if settings.arm in LOSO_ARMS else CONDITION_FILES
    mixture_names = () if settings.arm in LOSO_ARMS else MIXTURE_FILES
    paths = [leaf / name for name in names] + [
        leaf / f"imputations={count}" / name for count in settings.imputation_counts for name in mixture_names
    ]
    tables: dict[Path, pd.DataFrame] = {}
    for path in paths:
        if not path.is_file():
            problems.append(f"{key.label()}: missing {path.relative_to(leaf)}")
            continue
        if path.suffix == ".parquet":
            table = pd.read_parquet(path)
            missing = required.get(path.name, set()) - set(table.columns)
            if missing:
                problems.append(f"{key.label()}: {path.relative_to(leaf)} lacks {sorted(missing)}")
            tables[path] = table
    if problems:
        return problems
    audit = tables[leaf / "data_role_audit.parquet"]
    counts = json.loads(completion.read_text(encoding="utf-8"))["counts"]
    roles = audit.role.astype(str).value_counts()
    for role, name in (("source_randomized", "n_source_randomized_rows"), ("source_idle", "n_source_idle_rows"),
                       ("target_idle", "n_target_idle_rows"), ("target_reference", "n_target_reference_rows")):
        if int(roles.get(role, 0)) != int(counts[name]):
            problems.append(f"{key.label()}: audit {role} count differs from the pipeline's")
    evidence = tables[leaf / "source_evidence_draws.parquet"]
    expected_units = max(settings.imputation_counts) * int(counts["n_source_sites"]) * len(COVARIATES)
    if len(evidence) != expected_units:
        problems.append(f"{key.label()}: {len(evidence)} evidence units, expected {expected_units}")
    # Every used surface passed its gate with the chain and draw counts of its attempt, and every excluded surface
    # failed the second retry.
    surfaces = tables[leaf / "tolerance_surface_diagnostics.parquet"]
    if len(surfaces) != max(settings.imputation_counts) * len(COVARIATES) * 2:
        problems.append(f"{key.label()}: {len(surfaces)} tolerance surfaces, expected one per imputation, candidate and kind")
    budget = settings.attempt_budgets()
    for surface in surfaces.itertuples(index=False):
        warmup, samples = budget.get(str(surface.attempt), (None, None))
        frozen = int(surface.num_chains) == settings.num_chains and (int(surface.num_warmup), int(surface.num_samples)) == (
            warmup, samples)
        if bool(surface.excluded):
            valid = frozen and not bool(surface.passed) and str(surface.attempt) == "second_retry"
        else:
            valid = frozen and bool(surface.passed)
        if not valid:
            problems.append(f"{key.label()}: surface {surface.candidate}/{surface.tolerance_kind} of imputation "
                            f"{surface.imputation_index} breaks the frozen sampler budget or exclusion rule")
            break
    if settings.arm in LOSO_ARMS:
        return problems
    n_draws = int(settings.random_set_draws)
    for count in settings.imputation_counts:
        base = leaf / f"imputations={count}"
        sets = tables[base / "random_identified_sets.parquet"]
        n_estimands = sets.loc[sets.method.eq(NATURALNESS)].groupby(["estimand_type", "estimand_key"]).ngroups
        if len(sets) != len(METHODS) * n_draws * n_estimands:
            problems.append(f"{key.label()}: random-set rows are not {len(METHODS)} x {n_draws} x {n_estimands}")
        per = sets.groupby(["method", "estimand_type", "estimand_key"]).draw_index
        if not (per.nunique().eq(n_draws).all() and per.max().eq(n_draws - 1).all()):
            problems.append(f"{key.label()}: draw indices are not unique 0..{n_draws - 1} per method and estimand")
        if sets.groupby("draw_index").sample_id.nunique().ne(1).any() or sets.sample_id.nunique() != n_draws:
            problems.append(f"{key.label()}: sample IDs are not one per draw shared by both methods")
        triggers = tables[base / "target_trigger_draws.parquet"]
        if len(triggers) != n_draws * len(COVARIATES) or triggers.draw_index.nunique() != n_draws:
            problems.append(f"{key.label()}: trigger rows are not {n_draws} x {len(COVARIATES)}")
        metrics = tables[base / "condition_metrics.parquet"]
        if int(metrics.loc[metrics.metric.eq("n_target_idle_rows"), "value"].iloc[0]) != int(counts["n_target_idle_rows"]):
            problems.append(f"{key.label()}: metrics disagree with the audit on target idle rows")
    return problems


def _pilot_decision(metrics: pd.DataFrame, settings: Exp31RunSettings, run_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the imputation-count rule to the pilot conditions.

    The statistic is the largest absolute shift, between 10 and 20 source imputations, of any target-mass-weighted
    Naturalness endpoint summary (outer and median endpoints) or site-average outer endpoint; a shift above 0.02
    escalates every condition to 40 imputations.  A Monte Carlo accuracy rule, not a scientific gate.
    """

    selection = (
        (metrics.scope.eq("weighted") & metrics.method.eq(NATURALNESS)
         & metrics.metric.isin(["weighted_outer_lower", "weighted_outer_upper", "weighted_lower_q50", "weighted_upper_q50"]))
        | (metrics.scope.eq("bounds") & metrics.method.eq(NATURALNESS) & metrics.estimand_type.eq("ate")
           & metrics.metric.isin(["outer_lower", "outer_upper"]))
    )
    rows = metrics.loc[selection]
    keys = ["study", "pattern", "beta", "replication", "metric"]
    wide = rows.set_index([*keys, "num_imputations"]).value.unstack("num_imputations")
    low, high = PILOT_IMPUTATION_COUNTS
    both_missing = wide[low].isna() & wide[high].isna()
    one_missing = wide[low].isna() ^ wide[high].isna()
    shifts = (wide[high] - wide[low]).abs().where(~both_missing, 0.0).where(~one_missing, np.inf)
    per_condition = shifts.groupby(level=["study", "pattern", "beta", "replication"]).max()
    maximum = float(shifts.max()) if len(shifts) else float("inf")
    selected = PRIMARY_IMPUTATIONS if maximum <= IMPUTATION_SHIFT_THRESHOLD else ESCALATED_IMPUTATIONS
    return _identity({
        "experiment": EXPERIMENT_NAME,
        "status": "frozen",
        "rule": (
            "increase to 40 imputations for all conditions if any target-mass-weighted Naturalness endpoint "
            "moves by more than 0.02 between 10 and 20 imputations"
        ),
        "threshold": IMPUTATION_SHIFT_THRESHOLD,
        "maximum_shift": maximum,
        "per_condition_maximum_shift": [
            {"study": study, "pattern": pattern, "beta": float(beta), "replication": int(rep), "maximum_shift": float(value)}
            for (study, pattern, beta, rep), value in per_condition.items()
        ],
        "selected_num_imputations": int(selected),
        "nuts_settings": settings.nuts_settings(),
        "random_set_draws": int(settings.random_set_draws),
        "pilot_run_fingerprint": run_identity["fingerprint"],
        "pmf_fingerprint": run_identity["pmf_fingerprint"],
    })


def summarize_by_family(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate condition metrics over replications within study-pattern-beta.

    Returns:
        One row per family and metric with the replication count, the mean, standard deviation, minimum and maximum
        of the per-condition values, and the pooled denominator.
    """

    keys = ["arm", "num_imputations", "study", "pattern", "beta", "scope", "method", "estimand_type",
            "estimand_key", "candidate", "metric", "denominator_label"]
    grouped = metrics.groupby(keys, sort=True, dropna=False)
    summary = grouped.value.agg(n_conditions="count", mean="mean", sd="std", min="min", max="max").reset_index()
    summary["pooled_denominator"] = grouped.denominator.sum().to_numpy()
    return summary


def _settings_from_identity(identity: Mapping[str, Any]) -> Exp31RunSettings:
    """Rebuild the run settings a run identity recorded."""

    raw_settings = dict(identity["settings"])
    raw_settings["imputation_counts"] = tuple(raw_settings["imputation_counts"])
    for name in ("replications", "studies"):
        if raw_settings.get(name) is not None:
            raw_settings[name] = tuple(raw_settings[name])
    return Exp31RunSettings(**raw_settings)


def finalize_run(output_dir: Path) -> dict[str, Any]:
    """Validate every expected condition, then write the run's completion.

    ``completion.json`` is written only when every expected leaf is complete, belongs to this run, has every
    artifact with its schema and consistent draw and row counts; otherwise the problems go to
    ``validation_report.json`` and an error is raised.  A valid pilot also writes ``imputation_decision.json``.

    Raises:
        RuntimeError: If validation fails.
    """

    output_dir = Path(output_dir)
    identity = json.loads((output_dir / "identity.json").read_text(encoding="utf-8"))
    settings = _settings_from_identity(identity)
    conditions = [ConditionKey(row["study"], row["pattern"], float(row["beta"]), int(row["replication"]))
                  for row in identity["conditions"]]
    problems = [problem for key in conditions for problem in validate_condition_leaf(key.leaf(output_dir), key, identity, settings)]
    if problems:
        _write_json({"status": "invalid", "problems": problems, "checked_at_utc": _utc_now()}, output_dir / "validation_report.json")
        raise RuntimeError(f"{len(problems)} validation problems; see {output_dir / 'validation_report.json'}")
    metric_paths = [
        key.leaf(output_dir) / name
        for key in conditions
        for name in (["condition_metrics.parquet"] if settings.arm in LOSO_ARMS
                     else [f"imputations={count}/condition_metrics.parquet" for count in settings.imputation_counts])
    ]
    metrics = pd.concat([pd.read_parquet(path) for path in metric_paths], ignore_index=True)
    _write_parquet(metrics, output_dir / "condition_metrics.parquet")
    summarize_by_family(metrics).to_csv(output_dir / "summary_by_study_pattern_beta.csv", index=False)
    decision = None
    if settings.arm == "pilot":
        decision = _pilot_decision(metrics, settings, identity)
        _write_json(decision, output_dir / "imputation_decision.json")
    completion = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "n_conditions": len(conditions),
        "validated": True,
        "run_fingerprint": identity["fingerprint"],
        "pilot_decision": None if decision is None else {
            "selected_num_imputations": decision["selected_num_imputations"],
            "maximum_shift": decision["maximum_shift"],
            "fingerprint": decision["fingerprint"],
        },
        "run_artifact_sha256": {
            name: _sha256(output_dir / name)
            for name in ("condition_metrics.parquet", "summary_by_study_pattern_beta.csv", "imputation_decision.json")
            if (output_dir / name).is_file()
        },
    }
    _write_json(completion, output_dir / "completion.json")
    return completion


# --- Bayesian Balke-Pearl arm ---------------------------------------------------

#: Default instrument relevance threshold (Exp25's Balke-Pearl rule): a covariate whose treatment relevance ``D_j``
#: reaches the trigger grid's smallest ``tau`` is an instrument; ``--relevance-tau`` sets another, e.g. 0.075.
BP_RELEVANCE_TAU = min(TRIGGER_TAUS)
#: Arms a Balke-Pearl root can be built on: one mixture size, so random set ``d`` names one target PMF draw, and the
#: PMF root's own target draws (the modular arm recentres them).
BP_SOURCE_ARMS = ("primary", "smoke", "predictive")
#: Slack of the instrumental-inequality check that separates a refuted instrument from a solver failure.
BP_INEQUALITY_TOLERANCE = 1e-9
#: The arm's paired comparisons, as the ``method`` of their metric rows.
BP_VS_MANSKI = "paired_bp_vs_manski"
NATURALNESS_VS_BP = "paired_naturalness_vs_bp"
#: Files of one condition's leaf under a Balke-Pearl root.
BP_CONDITION_FILES = (
    "identity.json",
    "bp_random_identified_sets.parquet",
    "bp_instrument_draws.parquet",
    "bp_scored_random_sets.parquet",
    "condition_metrics.parquet",
)


def balke_pearl_protocol(relevance_tau: float = BP_RELEVANCE_TAU) -> dict[str, Any]:
    """Return the rules of the Bayesian Balke-Pearl arm.

    The arm reads the same posterior draws as Naturalness and treats them the same way, so the two differ only in
    their restriction: a draw or cell no instrument reaches takes the matched Manski set, and a restriction the
    draw contradicts leaves an empty set.

    Args:
        relevance_tau: The instrument relevance threshold on ``D_j``.
    """

    return {
        "method": BALKE_PEARL,
        "added": "2026-10-01, after the primary outcomes; post hoc, outside the prespecified classification",
        "instruments": f"every covariate with treatment relevance D_j >= {relevance_tau:g} in the draw (Exp25's rule)",
        "relevance_tau": float(relevance_tau),
        "bound": "sharp Balke-Pearl bound on E[Y(1)-Y(0) | Z_-j] from P(Y, X | Z_j, Z_-j), assigned to both cells of "
                 "the Z_-j block and intersected over each cell's instruments",
        "block_support": f"both cells of the block are LP cells (at least {LP_MIN_CELL_COUNT} target idle rows per arm)",
        "no_instrument": "the cell's matched Bayesian Manski interval; a draw without any instrument is a Manski fallback",
        "contradiction": "an empty set when the instrumental inequality fails or the instruments' bounds are disjoint",
        "site_average": "the target-mass-weighted sum of the cell bounds; empty when any cell is empty",
        "pairing": "random set d reads the target PMF draw, target-mass weights and reference draw of the run's random set d",
    }


def _instrumental_inequality_holds(joint: np.ndarray) -> bool:
    """Return whether ``P(Y=y, X=x | Z=z)``, indexed ``[z, y, x]``, meets the instrumental inequality.

    With binary instrument, treatment and outcome, ``max_x sum_y max_z P(y, x | z) <= 1`` holds exactly when the
    response-type LP of :func:`causal_grounding.balke_pearl.balke_pearl_interval` is feasible, so a failed solve
    with the inequality met is a solver failure, not a refuted instrument.
    """

    return bool(joint.max(axis=0).sum(axis=0).max() <= 1.0 + BP_INEQUALITY_TOLERANCE)


def balke_pearl_cell_bounds(
    draw: bayes.SitePMFDraw,
    cells: Sequence[tuple[int, ...]],
    instruments: Sequence[int],
) -> tuple[dict[tuple[int, ...], dict[str, Any]], dict[int, dict[str, int]]]:
    """Return one target PMF draw's Balke-Pearl interval for every LP cell.

    Instrument ``j`` bounds ``E[Y(1)-Y(0) | Z_-j]`` from ``P(Y, X | Z_j, Z_-j)`` in every block whose two cells are
    LP cells; under exogeneity that is the effect in both cells, so each cell takes the intersection of its
    instruments' bounds, as in Exp25.  A cell no instrument reaches keeps no interval; it is empty when an
    instrument is refuted (the instrumental inequality fails) or the bounds are disjoint.

    Args:
        draw: One target idle PMF draw.
        cells: The draw's LP cells.
        instruments: Covariate indices of the draw's relevant instruments.

    Returns:
        Per cell: ``outcome`` (``bounded``, ``no_instrument``, ``refuted``, ``disjoint`` or ``solver_failure``),
        ``lower``, ``upper`` (NaN unless bounded) and ``n_instruments``.  Per instrument: the blocks it bounded
        (``n_blocks``) and those that refuted it (``n_refuted``).
    """

    lp_cells = set(cells)
    per_cell: dict[tuple[int, ...], list[tuple[str, float, float]]] = {cell: [] for cell in lp_cells}
    usage = {int(j): {"n_blocks": 0, "n_refuted": 0} for j in instruments}
    for j in instruments:
        for cell in sorted(lp_cells):
            if cell[j] != 0:
                continue
            partner = cell[:j] + (1,) + cell[j + 1:]
            if partner not in lp_cells:
                continue
            joint = np.array([[[draw.idle_xy_given_z[(level, x, y)] for x in (0, 1)] for y in (0, 1)]
                              for level in (cell, partner)], dtype=float)
            interval = balke_pearl.balke_pearl_interval(joint)
            usage[j]["n_blocks"] += 1
            if interval.feasible:
                result = ("bounded", interval.lower, interval.upper)
            elif _instrumental_inequality_holds(joint):
                result = ("solver_failure", np.nan, np.nan)
            else:
                usage[j]["n_refuted"] += 1
                result = ("refuted", np.nan, np.nan)
            per_cell[cell].append(result)
            per_cell[partner].append(result)
    bounds = {}
    for cell, results in per_cell.items():
        outcomes = {outcome for outcome, _, _ in results}
        lower = max((value for _, value, _ in results), default=np.nan)
        upper = min((value for _, _, value in results), default=np.nan)
        if not results:
            outcome = "no_instrument"
        elif "refuted" in outcomes:
            outcome = "refuted"
        elif "solver_failure" in outcomes:
            outcome = "solver_failure"
        elif lower > upper:
            outcome = "disjoint"
        else:
            outcome = "bounded"
        keep = outcome == "bounded"
        bounds[cell] = {"outcome": outcome, "lower": float(lower) if keep else np.nan,
                        "upper": float(upper) if keep else np.nan, "n_instruments": len(results)}
    return bounds, usage


def balke_pearl_random_sets(
    target_draws: Sequence[bayes.SitePMFDraw],
    primary_sets: pd.DataFrame,
    relevance_tau: float = BP_RELEVANCE_TAU,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Bayesian Balke-Pearl random sets on a finalized run's draws.

    Random set ``d`` reads the target PMF draw of the run's random set ``d``, so it pairs with that draw's
    Naturalness and Manski sets, target-mass weights and reference draw.  Instruments are the covariates whose
    trigger relevance ``D_j`` (:func:`causal_grounding.bayesian_modular_naturalness.compute_trigger_statistics`)
    reaches ``relevance_tau``.  A cell no instrument reaches copies the matched Manski interval, and a draw where no
    instrument bounds any cell is a Manski fallback; the site average is the weighted sum of the cell bounds.

    Args:
        target_draws: The condition's target PMF draws, in the order of the run's ``target_pmf_draw_index``.
        primary_sets: The run's random identified sets for this condition.
        relevance_tau: The instrument relevance threshold on ``D_j``.

    Returns:
        The random sets, in the run's schema plus ``bp_cell_outcome`` and ``n_instruments``, and one row per draw
        and candidate with its relevance and use as an instrument.

    Raises:
        ValueError: If a draw's target PMF draw or LP cells differ from the run's.
    """

    manski = primary_sets.loc[primary_sets.method.eq(MANSKI)]
    rows: list[dict[str, Any]] = []
    instrument_rows: list[dict[str, Any]] = []
    for draw_index, group in manski.groupby("draw_index", sort=True):
        first = group.iloc[0]
        draw = target_draws[int(first.target_pmf_draw_index)]
        if draw.sample_id != first.target_pmf_sample_id:
            raise ValueError(f"Random set {draw_index} was built from target draw {first.target_pmf_sample_id}, "
                             f"not {draw.sample_id}")
        cate = group.loc[group.estimand_type.eq("cate")]
        cells = [tuple(json.loads(key)) for key in cate.estimand_key]
        if set(cells) != set(draw.observed_quantities(min_cell_count=LP_MIN_CELL_COUNT).z_values):
            raise ValueError(f"Random set {draw_index} has other LP cells than its target draw")
        statistics = bayes.compute_trigger_statistics(draw)
        relevant = [int(row.candidate_index) for row in statistics.itertuples(index=False)
                    if np.isfinite(row.d_strength) and float(row.d_strength) >= relevance_tau]
        bounds, usage = balke_pearl_cell_bounds(draw, cells, relevant)
        used = sorted(j for j, counts in usage.items() if counts["n_blocks"] > 0)
        for row in statistics.itertuples(index=False):
            counts = usage.get(int(row.candidate_index), {"n_blocks": 0, "n_refuted": 0})
            instrument_rows.append({
                "draw_index": int(draw_index), "target_pmf_draw_index": int(first.target_pmf_draw_index),
                "candidate": str(row.candidate), "candidate_index": int(row.candidate_index),
                "d_strength": float(row.d_strength), "relevant": int(row.candidate_index) in relevant,
                "used": counts["n_blocks"] > 0, "n_blocks": counts["n_blocks"], "n_refuted": counts["n_refuted"],
            })
        fallback = not used
        draw_rows = []
        for manski_row in group.itertuples(index=False):
            row = manski_row._asdict()
            if row["estimand_type"] == "cate":
                cell = bounds[tuple(json.loads(row["estimand_key"]))]
                outcome = cell["outcome"]
                if outcome != "no_instrument":
                    row["lower"], row["upper"] = cell["lower"], cell["upper"]
                row["n_instruments"] = cell["n_instruments"]
            else:
                row["n_instruments"] = len(used)
                outcome = "site_average"
            row["bp_cell_outcome"] = outcome
            draw_rows.append(row)
        # A cell or draw that copies the Manski set also copies its outcome.
        cate_rows = [row for row in draw_rows if row["estimand_type"] == "cate"]
        for row in cate_rows:
            outcome = row["bp_cell_outcome"]
            row["state"] = (row["lp_outcome"] if outcome == "no_instrument"
                            else "empty" if outcome in ("refuted", "disjoint")
                            else "solver_failure" if outcome == "solver_failure" else "nonempty")
        states = {row["state"] for row in cate_rows}
        empty = "empty" in states
        for row in draw_rows:
            if row["estimand_type"] != "cate":
                if fallback:
                    row["state"] = row["lp_outcome"]
                else:
                    row["state"] = "empty" if empty else "solver_failure" if "solver_failure" in states else "nonempty"
                    if row["state"] == "nonempty":
                        row["lower"] = float(sum(cell["estimand_weight"] * cell["lower"] for cell in cate_rows))
                        row["upper"] = float(sum(cell["estimand_weight"] * cell["upper"] for cell in cate_rows))
            state = row.pop("state")
            nonempty = state == "nonempty"
            status = "optimal" if nonempty else "infeasible" if state == "empty" else "failed"
            row.update({
                "method": BALKE_PEARL,
                "lower": row["lower"] if nonempty else np.nan,
                "upper": row["upper"] if nonempty else np.nan,
                "status": status, "lower_status": status, "upper_status": status,
                "feasible": nonempty, "lp_outcome": state, "nonempty": nonempty,
                "n_eligible": len(used), "candidate_set": "|".join(sorted(COVARIATES[j] for j in used)),
                "no_candidate_eligible": fallback, "used_manski_fallback": fallback,
                "eligible_but_lp_infeasible": bool(not fallback and empty),
            })
            row["width"] = row["upper"] - row["lower"] if nonempty else np.nan
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(instrument_rows)


def balke_pearl_condition_metrics(
    *,
    bp_sets: pd.DataFrame,
    primary_sets: pd.DataFrame,
    instruments: pd.DataFrame,
    scored: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize one condition's Balke-Pearl random sets as Naturalness is summarized.

    Feasibility, bounds, weighted summaries and reference scoring use :func:`_method_bound_metrics` and
    :func:`_reference_metrics`; :func:`_paired_width_metrics` compares Balke-Pearl with Manski
    (:data:`BP_VS_MANSKI`) and Naturalness with Balke-Pearl (:data:`NATURALNESS_VS_BP`, a positive gain favoring
    Naturalness) on the same draws.  Instrument use is reported under ``discovery`` and ``candidate``.

    Returns:
        Long metrics in the primary schema, every row with its denominator.
    """

    out = _MetricRows()
    n_draws = int(bp_sets.draw_index.nunique())
    per_draw = instruments.groupby("draw_index").used.sum()
    out.add("discovery", "p_any_instrument", (per_draw > 0).mean(), n_draws, ALL_DRAWS, method=BALKE_PEARL)
    out.add("discovery", "p_no_instrument", (per_draw == 0).mean(), n_draws, ALL_DRAWS, method=BALKE_PEARL)
    out.add("discovery", "mean_n_instruments", per_draw.mean(), n_draws, ALL_DRAWS, method=BALKE_PEARL)
    for candidate, group in instruments.groupby("candidate", sort=True):
        n_used = int(group.used.sum())
        keys = {"method": BALKE_PEARL, "candidate": candidate}
        out.add("candidate", "p_relevant_instrument", group.relevant.mean(), n_draws, ALL_DRAWS, **keys)
        out.add("candidate", "p_used_instrument", group.used.mean(), n_draws, ALL_DRAWS, **keys)
        out.add("candidate", "p_refuted_given_used", group.n_refuted.gt(0).sum() / n_used if n_used else np.nan,
                n_used, USED_INSTRUMENT, **keys)
        out.add("candidate", "d_strength_q50", group.d_strength.median(), n_draws, ALL_DRAWS, **keys)
    cate = bp_sets.loc[bp_sets.estimand_type.eq("cate")]
    without = (cate.estimand_weight * cate.bp_cell_outcome.eq("no_instrument")).groupby(cate.draw_index).sum()
    out.add("feasibility", "mean_target_mass_without_instrument", without.mean(), n_draws, ALL_DRAWS, method=BALKE_PEARL)
    weights, _ = _method_bound_metrics(out, bp_sets, n_draws)
    combined = pd.concat([primary_sets, bp_sets[primary_sets.columns]], ignore_index=True)
    per_draw_width = _per_draw_weighted_width(combined.loc[combined.estimand_type.eq("cate")])
    _paired_width_metrics(out, combined, per_draw_width, weights, n_draws, method=BALKE_PEARL, baseline=MANSKI,
                          tag=BP_VS_MANSKI, narrower_metric="p_narrower_given_joint", joint_label=BP_JOINT_MANSKI)
    _paired_width_metrics(out, combined, per_draw_width, weights, n_draws, method=NATURALNESS, baseline=BALKE_PEARL,
                          tag=NATURALNESS_VS_BP, narrower_metric="p_narrower_given_joint", joint_label=BP_JOINT_NATURALNESS)
    _reference_metrics(out, bp_sets, scored, support, weights, n_draws)
    return out.frame()


def _balke_pearl_condition_identity(key: ConditionKey, run_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the identity a condition leaf must carry under one Balke-Pearl root."""

    return _identity({
        "experiment": EXPERIMENT_NAME,
        "arm": "balke_pearl",
        "condition": key.as_dict(),
        "run_fingerprint": run_identity["fingerprint"],
        "source_run_fingerprint": run_identity["source_run"]["fingerprint"],
    })


def _verified_source_files(key: ConditionKey, source_root: Path, names: Sequence[str]) -> dict[str, str]:
    """Return the hashes of a finalized run's condition files, checked against its completion.

    Raises:
        ValueError: If the leaf is incomplete or a file changed after the run wrote it.
    """

    leaf = key.leaf(source_root)
    completion = json.loads((leaf / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError(f"{leaf} is not complete")
    hashes = {}
    for name in names:
        digest = _sha256(leaf / name)
        if digest != completion["artifact_sha256"].get(name):
            raise ValueError(f"{leaf / name} changed after the run wrote it")
        hashes[name] = digest
    return hashes


def run_balke_pearl_condition(
    *,
    key: ConditionKey,
    source_root: Path,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    output_dir: Path,
    run_identity: Mapping[str, Any],
    resume: bool,
) -> None:
    """Add the Balke-Pearl random sets, scores and metrics of one condition.

    Data and target PMF draws are loaded as the run loaded them; random sets, trigger and reference draws are read
    from the finalized run after a hash check; and the instruments' relevance must equal the run's trigger
    statistics draw for draw, so the arm uses exactly the run's posterior draws.

    Raises:
        ValueError: If an input differs from the run's.
    """

    leaf = key.leaf(output_dir)
    if not _claim_leaf(leaf, _balke_pearl_condition_identity(key, run_identity), resume=resume):
        LOGGER.info("Skipping completed condition %s", key.label())
        return
    started = time.monotonic()
    mixture = f"imputations={run_identity['source_run']['num_imputations']}"
    names = [f"{mixture}/random_identified_sets.parquet", f"{mixture}/target_trigger_draws.parquet",
             "rct_reference_draws.parquet"]
    input_hashes = _verified_source_files(key, source_root, names)
    source_leaf = key.leaf(source_root)
    primary_sets = pd.read_parquet(source_leaf / names[0])
    triggers = pd.read_parquet(source_leaf / names[1])
    references = pd.read_parquet(source_leaf / names[2])
    data = load_condition_data(key, prepared_root, rct_path)
    n_target = int(primary_sets.target_pmf_draw_index.max()) + 1
    _, target_draws, pmf_hashes = load_condition_pmfs(key, pmf_root, data, pmf_fingerprint=run_identity["pmf_fingerprint"],
                                                      n_source_draws=0, n_target_draws=n_target)
    bp_sets, instruments = balke_pearl_random_sets(target_draws, primary_sets,
                                                   relevance_tau=float(run_identity["protocol"]["relevance_tau"]))
    relevance = instruments.merge(triggers[["draw_index", "candidate", "d_strength"]], on=["draw_index", "candidate"],
                                  suffixes=("", "_run"), validate="one_to_one")
    if len(relevance) != len(instruments) or not np.allclose(relevance.d_strength, relevance.d_strength_run,
                                                             rtol=0.0, atol=1e-12, equal_nan=True):
        raise ValueError(f"{key.label()}: the target PMF draws do not reproduce the run's trigger statistics")
    support = reference_support(data.reference, data.target_idle, lp_strata(target_draws))
    scored = score_random_sets(bp_sets, references)
    metrics = balke_pearl_condition_metrics(bp_sets=bp_sets, primary_sets=primary_sets, instruments=instruments,
                                            scored=scored, support=support)
    for name, value in {**key.as_dict(), "arm": "balke_pearl",
                        "num_imputations": int(run_identity["source_run"]["num_imputations"])}.items():
        metrics[name] = value
    _write_parquet(bp_sets, leaf / "bp_random_identified_sets.parquet")
    _write_parquet(instruments, leaf / "bp_instrument_draws.parquet")
    _write_parquet(scored, leaf / "bp_scored_random_sets.parquet")
    _write_parquet(metrics, leaf / "condition_metrics.parquet")
    _write_json(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "runtime_seconds": time.monotonic() - started,
            "input_sha256": {**input_hashes, **{f"pmf/{name}": digest for name, digest in pmf_hashes.items()}},
            "artifact_sha256": _artifact_hashes(leaf),
        },
        leaf / "completion.json",
    )
    LOGGER.info("%s Balke-Pearl complete in %.0f s", key.label(), time.monotonic() - started)


def initialize_balke_pearl_root(
    output_dir: Path,
    source_root: Path,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    *,
    resume: bool,
    cluster_manifest_path: str | None = None,
    relevance_tau: float = BP_RELEVANCE_TAU,
) -> dict[str, Any]:
    """Create or validate the Balke-Pearl root of a finalized run and return its identity.

    The identity fingerprints the arm's rules (:func:`balke_pearl_protocol` with ``relevance_tau``), the source run,
    and its prepared inputs and PMF root, which must be the ones the run recorded; resume is refused if any differs.

    Raises:
        ValueError: If the source run is not a finalized single-mixture run, the inputs are not the run's, or the
            existing identity differs.
        FileExistsError: If a fresh root's directory is not empty.
    """

    source, _ = _read_finalized(source_root)
    if source["arm"] not in BP_SOURCE_ARMS:
        raise ValueError(f"The Balke-Pearl arm builds on a {' or '.join(BP_SOURCE_ARMS)} run, not {source['arm']!r}")
    counts = source["settings"]["imputation_counts"]
    if len(counts) != 1:
        raise ValueError(f"The source run has {len(counts)} mixture sizes; the arm needs exactly one")
    prepared = read_prepared_identity(prepared_root)
    pmf = read_pmf_identity(pmf_root)
    if prepared["fingerprint"] != source["prepared_fingerprint"] or pmf["fingerprint"] != source["pmf_fingerprint"]:
        raise ValueError("The prepared root or the PMF root is not the source run's")
    conditions = [ConditionKey(row["study"], row["pattern"], float(row["beta"]), int(row["replication"]))
                  for row in source["conditions"]]
    verify_prepared_inputs(prepared_root, prepared, conditions, rct_path)
    payload = _identity({
        "experiment": EXPERIMENT_NAME,
        "stage": "balke_pearl",
        "arm": "balke_pearl",
        "protocol": balke_pearl_protocol(relevance_tau),
        "conditions": source["conditions"],
        "source_run": {"arm": source["arm"], "fingerprint": source["fingerprint"], "num_imputations": int(counts[0])},
        "settings": {"random_set_draws": int(source["settings"]["random_set_draws"]), "imputation_counts": [int(counts[0])]},
        "prepared_fingerprint": source["prepared_fingerprint"],
        "pmf_fingerprint": source["pmf_fingerprint"],
        "dependencies": _dependency_versions(),
    })
    identity_path = Path(output_dir) / "identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if not resume or existing.get("fingerprint") != payload["fingerprint"]:
            raise ValueError("Existing Balke-Pearl identity differs or --resume was not supplied")
        return existing
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _write_json(payload, identity_path)
    write_run_manifest(
        output_dir,
        experiment_name=EXPERIMENT_NAME,
        script_path="experiments/exp31_bayesian_manylabs_naturalness.py",
        config={"stage": "balke_pearl", "protocol": balke_pearl_protocol(relevance_tau)},
        argv=sys.argv,
        cluster_manifest_path=cluster_manifest_path,
        extra={
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": payload["fingerprint"],
            "git": _git_provenance(),
            "source_fingerprints": _source_fingerprints(),
            "software_environment": _software_environment(),
            "input_paths": {"source_root": str(Path(source_root).resolve()), "prepared_root": str(Path(prepared_root).resolve()),
                            "rct_path": str(Path(rct_path).resolve()), "pmf_root": str(Path(pmf_root).resolve())},
            "n_conditions": len(conditions),
        },
    )
    return payload


def run_balke_pearl(
    output_dir: Path,
    source_root: Path,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    *,
    resume: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    cluster_manifest_path: str | None = None,
    relevance_tau: float = BP_RELEVANCE_TAU,
) -> dict[str, Any]:
    """Add the Balke-Pearl arm to every condition of a finalized run, sharded round-robin.

    Shards share one root as in :func:`run_experiment`; the last shard to finish finalizes it
    (:func:`finalize_balke_pearl`).  Every shard must get the same ``relevance_tau``, which the identity fingerprints.

    Returns:
        This shard's completion record.

    Raises:
        RuntimeError: If any condition failed.
    """

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index} of {num_shards}")
    if shard_index > 0:
        _wait_for_run_identity(Path(output_dir))
    identity = initialize_balke_pearl_root(output_dir, source_root, prepared_root, rct_path, pmf_root,
                                           resume=resume or shard_index > 0, cluster_manifest_path=cluster_manifest_path,
                                           relevance_tau=relevance_tau)
    conditions = [ConditionKey(row["study"], row["pattern"], float(row["beta"]), int(row["replication"]))
                  for row in identity["conditions"]]
    failures: list[dict[str, Any]] = []
    attempted = 0
    for position, key in enumerate(conditions):
        if position % num_shards != shard_index:
            continue
        attempted += 1
        try:
            run_balke_pearl_condition(key=key, source_root=Path(source_root), prepared_root=Path(prepared_root),
                                      rct_path=Path(rct_path), pmf_root=Path(pmf_root), output_dir=Path(output_dir),
                                      run_identity=identity, resume=True)
        except Exception as error:  # keep the other conditions.
            LOGGER.exception("Exp31 Balke-Pearl condition failed: %s", key.label())
            failures.append({**key.as_dict(), "error": repr(error)})
            leaf = key.leaf(output_dir)
            leaf.mkdir(parents=True, exist_ok=True)
            _write_json({"status": "failed", "error": repr(error), "failed_at_utc": _utc_now()}, leaf / "completion.json")
    record = {"status": "complete" if not failures else "failed", "attempted_conditions": attempted, "failures": failures,
              "completed_at_utc": _utc_now(), "shard_index": shard_index, "num_shards": num_shards}
    shards = Path(output_dir) / "shards"
    _write_json(record, shards / f"shard={shard_index:03d}-of-{num_shards:03d}.json")
    paths = [shards / f"shard={index:03d}-of-{num_shards:03d}.json" for index in range(num_shards)]
    if all(path.is_file() for path in paths) and all(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "complete" for path in paths
    ):
        lock = Path(output_dir) / "finalize.lock"
        try:
            lock.mkdir()
        except FileExistsError:
            LOGGER.info("Another shard is finalizing %s", output_dir)
        else:
            try:
                finalize_balke_pearl(output_dir)
            finally:
                lock.rmdir()
    if failures:
        raise RuntimeError(f"{len(failures)} Exp31 Balke-Pearl conditions failed")
    return record


def _validate_balke_pearl_leaf(key: ConditionKey, output_dir: Path, identity: Mapping[str, Any]) -> list[str]:
    """Return every problem of one Balke-Pearl condition leaf (empty when valid)."""

    leaf = key.leaf(output_dir)
    problems = []
    completion_path = leaf / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
    if completion.get("status") != "complete":
        return [f"{key.label()}: not complete"]
    missing = [name for name in BP_CONDITION_FILES if not (leaf / name).is_file()]
    if missing:
        return [f"{key.label()}: missing {missing}"]
    if json.loads((leaf / "identity.json").read_text(encoding="utf-8")).get("fingerprint") != \
            _balke_pearl_condition_identity(key, identity)["fingerprint"]:
        problems.append(f"{key.label()}: leaf belongs to another Balke-Pearl root")
    changed = [name for name, digest in completion["artifact_sha256"].items() if _sha256(leaf / name) != digest]
    if changed:
        problems.append(f"{key.label()}: {changed} changed after the arm wrote them")
    n_draws = int(identity["settings"]["random_set_draws"])
    sets = pd.read_parquet(leaf / "bp_random_identified_sets.parquet")
    n_estimands = sets.groupby(["estimand_type", "estimand_key"]).ngroups
    if not sets.method.eq(BALKE_PEARL).all() or len(sets) != n_draws * n_estimands:
        problems.append(f"{key.label()}: Balke-Pearl rows are not {n_draws} x {n_estimands}")
    per = sets.groupby(["estimand_type", "estimand_key"]).draw_index
    if not (per.nunique().eq(n_draws).all() and per.max().eq(n_draws - 1).all()):
        problems.append(f"{key.label()}: draw indices are not unique 0..{n_draws - 1} per estimand")
    if len(pd.read_parquet(leaf / "bp_instrument_draws.parquet")) != n_draws * len(COVARIATES):
        problems.append(f"{key.label()}: instrument rows are not {n_draws} x {len(COVARIATES)}")
    if len(pd.read_parquet(leaf / "bp_scored_random_sets.parquet")) != len(sets):
        problems.append(f"{key.label()}: scored rows do not match the random sets")
    return problems


def finalize_balke_pearl(output_dir: Path) -> dict[str, Any]:
    """Validate every condition of a Balke-Pearl root, then write its completion.

    As in :func:`finalize_run`, ``completion.json`` is written only when every leaf is complete, belongs to this
    root, is unchanged and has the expected row counts; otherwise the problems go to ``validation_report.json``.

    Raises:
        RuntimeError: If validation fails.
    """

    output_dir = Path(output_dir)
    identity = json.loads((output_dir / "identity.json").read_text(encoding="utf-8"))
    conditions = [ConditionKey(row["study"], row["pattern"], float(row["beta"]), int(row["replication"]))
                  for row in identity["conditions"]]
    problems = [problem for key in conditions for problem in _validate_balke_pearl_leaf(key, output_dir, identity)]
    if problems:
        _write_json({"status": "invalid", "problems": problems, "checked_at_utc": _utc_now()}, output_dir / "validation_report.json")
        raise RuntimeError(f"{len(problems)} validation problems; see {output_dir / 'validation_report.json'}")
    metrics = pd.concat([pd.read_parquet(key.leaf(output_dir) / "condition_metrics.parquet") for key in conditions],
                        ignore_index=True)
    _write_parquet(metrics, output_dir / "condition_metrics.parquet")
    summarize_by_family(metrics).to_csv(output_dir / "summary_by_study_pattern_beta.csv", index=False)
    completion = {
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "n_conditions": len(conditions),
        "validated": True,
        "run_fingerprint": identity["fingerprint"],
        "run_artifact_sha256": {name: _sha256(output_dir / name)
                                for name in ("condition_metrics.parquet", "summary_by_study_pattern_beta.csv")},
    }
    _write_json(completion, output_dir / "completion.json")
    return completion


# --- Trigger-grid restriction (regrid) ------------------------------------------

#: Leading draws the regrid first replays on the source run's own grid; they must reproduce the run's saved draws.
REGRID_CHECK_DRAWS = 200
#: Source-leaf files that a regridded leaf copies unchanged.
REGRID_COPIED_FILES = ("data_role_audit.parquet", "tolerance_surface_diagnostics.parquet", "rct_reference_draws.parquet")
#: Absolute tolerance of the regrid's comparisons with the run's saved draws.
REGRID_TOLERANCE = 1e-9


def _replay_mixtures(
    evidence: pd.DataFrame,
    diagnostics: pd.DataFrame,
    count: int,
) -> list[dict[tuple[str, str], bayes.ToleranceSurfacePosterior]]:
    """Rebuild a finished run's surface mappings from its saved evidence, for a replay.

    Each surface keeps what Module D reads besides its tolerance draws: the source features of its fit (units with
    finite features and requirement, as :func:`fit_tolerance_surfaces` selects them) and the imputation ID.  The
    posterior is empty, so it can be replayed through ``saved_tolerances`` but never sampled.  Surfaces the run
    excluded as unconverged stay out, as in the run.

    Args:
        evidence: The run's ``source_evidence_draws`` table.
        diagnostics: The run's ``tolerance_surface_diagnostics`` table.
        count: Mixture size; the first ``count`` imputations are rebuilt.

    Returns:
        One ``{(candidate, kind): surface}`` mapping per imputation.
    """

    excluded = {(int(row.imputation_index), str(row.candidate), str(row.tolerance_kind))
                for row in diagnostics.loc[diagnostics.excluded.astype(bool)].itertuples(index=False)}
    mixtures = []
    for index, group in evidence.loc[evidence.imputation_index.lt(count)].groupby("imputation_index", sort=True):
        surfaces = {}
        for candidate, rows in group.groupby("candidate", sort=True):
            for kind, omega_column, feature_names in bayes._SURFACE_SPECS:
                if (int(index), str(candidate), kind) in excluded:
                    continue
                units = rows.dropna(subset=[*feature_names, omega_column])
                surfaces[(str(candidate), kind)] = bayes.ToleranceSurfacePosterior(
                    candidate=str(candidate),
                    tolerance_kind=kind,
                    feature_names=tuple(feature_names),
                    source_features=units[list(feature_names)].to_numpy(dtype=float),
                    source_omega=units[omega_column].to_numpy(dtype=float),
                    posterior_samples={},
                    diagnostics={},
                    imputation_id=str(group.imputation_id.iloc[0]),
                )
        mixtures.append(surfaces)
    return mixtures


def _saved_tolerances(tolerances: pd.DataFrame) -> dict[tuple[str, str], tuple[float, int, float, int]]:
    """Key a run's ``tolerance_surface_draws`` for ``draw_random_identified_sets(saved_tolerances=...)``."""

    return {
        (str(row.sample_id), str(row.candidate)):
            (float(row.two_c), int(row.c_posterior_index), float(row.two_d), int(row.d_posterior_index))
        for row in tolerances.itertuples(index=False)
    }


def _rows_differ(replayed: pd.DataFrame, saved: pd.DataFrame, keys: Sequence[str], columns: Sequence[str]) -> list[str]:
    """Return the columns on which two tables disagree, rows matched on ``keys``.

    Floats agree within :data:`REGRID_TOLERANCE` (``NaN`` equals ``NaN``), other columns exactly; a row present in
    only one table is reported as ``"rows"``.
    """

    merged = replayed[[*keys, *columns]].merge(saved[[*keys, *columns]], on=list(keys), how="outer",
                                               suffixes=("", "_saved"), validate="one_to_one", indicator=True)
    if not merged.pop("_merge").eq("both").all():
        return ["rows"]
    differ = []
    for column in columns:
        left, right = merged[column], merged[f"{column}_saved"]
        if pd.api.types.is_float_dtype(left) or pd.api.types.is_float_dtype(right):
            same = np.allclose(left.astype(float), right.astype(float), rtol=0.0, atol=REGRID_TOLERANCE, equal_nan=True)
        else:
            same = left.astype(str).equals(right.astype(str))
        if not same:
            differ.append(column)
    return differ


def _regrid_problems(
    check: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    regridded: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    saved: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> list[str]:
    """Check a regrid against the run it replays.

    ``check``, the replay of the leading draws on the run's own grid, must reproduce the run's tables.  On the
    restricted grid every eligible candidate must have been eligible in the run, and every Manski set, and every
    Naturalness set whose draw keeps the run's eligible candidates, must be the run's.

    Args:
        check: Random sets, trigger draws and tolerance draws of the check replay.
        regridded: The same three tables on the restricted grid.
        saved: The run's three tables.

    Returns:
        Problems found; empty when the regrid is exact.
    """

    problems = []
    check_sets, check_triggers, check_tolerances = check
    sets, triggers, _ = regridded
    saved_sets, saved_triggers, saved_tolerances = saved
    leading = int(check_triggers.draw_index.max()) + 1
    set_keys = ["draw_index", "method", "estimand_type", "estimand_key"]
    set_columns = ["sample_id", "lower", "upper", "lp_outcome", "candidate_set", "used_manski_fallback"]
    for name, replayed, run, keys, columns in (
        ("random sets", check_sets, saved_sets, set_keys, set_columns),
        ("trigger draws", check_triggers, saved_triggers, ["draw_index", "candidate"],
         ["sample_id", "triggered", "trigger_tau", "in_source_support", "eligible", "ineligibility_reason"]),
        ("tolerance draws", check_tolerances, saved_tolerances, ["draw_index", "candidate"],
         ["two_c", "two_d", "c_posterior_index", "d_posterior_index"]),
    ):
        differ = _rows_differ(replayed, run.loc[run.draw_index.lt(leading)], keys, columns)
        if differ:
            problems.append(f"the replay on the run's grid does not reproduce its {name} ({differ})")
    eligible = triggers.merge(saved_triggers[["draw_index", "candidate", "eligible"]], on=["draw_index", "candidate"],
                              suffixes=("", "_run"), validate="one_to_one")
    if len(eligible) != len(triggers) or (eligible.eligible.astype(bool) & ~eligible.eligible_run.astype(bool)).any():
        problems.append("a candidate is eligible on the restricted grid but was not in the run")
    # Every row carries its draw's eligible candidates, so Manski rows are compared on their bounds only.
    differ = _rows_differ(sets.loc[sets.method.eq(MANSKI)], saved_sets.loc[saved_sets.method.eq(MANSKI)], set_keys,
                          ["sample_id", "lower", "upper", "lp_outcome"])
    if differ:
        problems.append(f"Manski sets changed ({differ})")
    naturalness = sets.loc[sets.method.eq(NATURALNESS)]
    saved_naturalness = saved_sets.loc[saved_sets.method.eq(NATURALNESS)]
    kept = naturalness[["draw_index", "candidate_set"]].drop_duplicates().merge(
        saved_naturalness[["draw_index", "candidate_set"]].drop_duplicates(), on=["draw_index", "candidate_set"])
    differ = _rows_differ(naturalness.loc[naturalness.draw_index.isin(kept.draw_index)],
                          saved_naturalness.loc[saved_naturalness.draw_index.isin(kept.draw_index)], set_keys, set_columns)
    if differ:
        problems.append(f"Naturalness sets whose draw keeps the run's eligible candidates changed ({differ})")
    return problems


def regrid_condition(
    *,
    key: ConditionKey,
    source_root: Path,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    output_dir: Path,
    run_identity: Mapping[str, Any],
    resume: bool,
) -> None:
    """Replay one condition of a finalized run on the restricted trigger grid.

    Nothing is refitted: the grid only decides which candidates are eligible at the target, and tolerance draws do
    not depend on it.  Audit, surface diagnostics and reference draws are copied; the evidence's descriptive grid
    columns are recomputed; and Modules D and E are replayed from the saved tolerance draws, the surfaces' source
    features (:func:`_replay_mixtures`) and the run's target draws and seeds.  The first :data:`REGRID_CHECK_DRAWS`
    draws are first replayed on the run's own grid, and the result is checked by :func:`_regrid_problems`.

    Raises:
        ValueError: If a run file or the PMF draws changed, or the replay is not exact.
    """

    settings = _settings_from_identity(run_identity)
    leaf = key.leaf(output_dir)
    if not _claim_leaf(leaf, _condition_identity(key, run_identity, settings), resume=resume):
        LOGGER.info("Skipping completed condition %s", key.label())
        return
    started = time.monotonic()
    source_leaf = key.leaf(source_root)
    mixture_names = [f"imputations={count}/{name}" for count in settings.imputation_counts for name in MIXTURE_FILES]
    input_hashes = _verified_source_files(key, source_root, [*CONDITION_FILES[1:], *mixture_names])
    source_completion = json.loads((source_leaf / "completion.json").read_text(encoding="utf-8"))
    data = load_condition_data(key, prepared_root, rct_path)
    if data.counts != source_completion["counts"]:
        raise ValueError(f"{key.label()}: the prepared data do not reproduce the run's row counts")
    n_sets = int(settings.random_set_draws)
    _, target_draws, pmf_hashes = load_condition_pmfs(
        key, pmf_root, data, pmf_fingerprint=run_identity["pmf_fingerprint"], n_source_draws=0,
        n_target_draws=min(int(run_identity["pmf_draws"]["n_target_draws"]), n_sets),
    )
    if pmf_hashes != source_completion["pmf_draws_sha256"]:
        raise ValueError(f"{key.label()}: the PMF draws are not the ones the run read")
    taus = [float(value) for value in run_identity["protocol"]["trigger_rule"]["taus"]]
    source_taus = [float(value) for value in run_identity["regrid"]["source_taus"]]
    for name in REGRID_COPIED_FILES:
        shutil.copyfile(source_leaf / name, leaf / name)
    evidence = _with_source_grid(pd.read_parquet(source_leaf / "source_evidence_draws.parquet"), taus)
    _write_parquet(evidence, leaf / "source_evidence_draws.parquet")
    diagnostics = pd.read_parquet(source_leaf / "tolerance_surface_diagnostics.parquet")
    references = pd.read_parquet(source_leaf / "rct_reference_draws.parquet")
    support = reference_support(data.reference, data.target_idle, lp_strata(target_draws))
    for count in settings.imputation_counts:
        source_mixture = source_leaf / f"imputations={count}"
        saved = tuple(pd.read_parquet(source_mixture / name) for name in (
            "random_identified_sets.parquet", "target_trigger_draws.parquet", "tolerance_surface_draws.parquet"))
        mixtures = _replay_mixtures(evidence, diagnostics, count)
        replay = {"saved_tolerances": _saved_tolerances(saved[2])}
        check = _mixture_random_sets(key, settings, target_draws, mixtures, diagnostics, count,
                                     n_draws=min(REGRID_CHECK_DRAWS, n_sets), taus=source_taus, **replay)
        random_sets, triggers, tolerances = _mixture_random_sets(key, settings, target_draws, mixtures, diagnostics,
                                                                 count, n_draws=n_sets, taus=taus, **replay)
        problems = _regrid_problems(check, (random_sets, triggers, tolerances), saved)
        if problems:
            raise ValueError(f"{key.label()}, {count} imputations: " + "; ".join(problems))
        scored = score_random_sets(random_sets, references)
        metrics = _mixture_metrics(key, settings, count, data=data, target_draws=target_draws, random_sets=random_sets,
                                   triggers=triggers, tolerances=tolerances, scored=scored, support=support,
                                   evidence=evidence, diagnostics=diagnostics)
        for name, table in zip(MIXTURE_FILES, (triggers, tolerances, random_sets, scored, metrics)):
            _write_parquet(table, leaf / f"imputations={count}" / name)
    _write_json(
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "runtime_seconds": time.monotonic() - started,
            "counts": data.counts,
            "pmf_draws_sha256": pmf_hashes,
            "source_sha256": input_hashes,
            "artifact_sha256": _artifact_hashes(leaf),
        },
        leaf / "completion.json",
    )
    LOGGER.info("%s regridded in %.0f s", key.label(), time.monotonic() - started)


def initialize_regrid_root(
    output_dir: Path,
    source_root: Path,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    *,
    tau_floor: float,
    resume: bool,
    cluster_manifest_path: str | None = None,
) -> dict[str, Any]:
    """Create or validate the regridded root of a finalized run and return its identity.

    The identity is the source run's with the protocol's ``taus`` restricted to ``tau >= tau_floor``, plus a
    ``regrid`` record (source run, its grid, the replay rule) and this stage's dependencies, so the root validates,
    finalizes, reports and takes a Balke-Pearl arm as a run of the source's arm.  Resume is refused if it differs.

    Raises:
        ValueError: If the source run is not a finalized run of :data:`BP_SOURCE_ARMS`, the floor drops no ``tau``
            or every one, the inputs are not the run's, or the existing identity differs.
        FileExistsError: If a fresh root's directory is not empty.
    """

    source, _ = _read_finalized(source_root)
    if source["arm"] not in BP_SOURCE_ARMS:
        raise ValueError(f"The regrid replays a {' or '.join(BP_SOURCE_ARMS)} run, not {source['arm']!r}")
    source_taus = [float(value) for value in source["protocol"]["trigger_rule"]["taus"]]
    taus = [value for value in source_taus if value >= float(tau_floor)]
    if not taus or taus == source_taus:
        raise ValueError(f"tau_floor={tau_floor:g} must drop some but not all of the run's taus {source_taus}")
    prepared = read_prepared_identity(prepared_root)
    pmf = read_pmf_identity(pmf_root)
    if prepared["fingerprint"] != source["prepared_fingerprint"] or pmf["fingerprint"] != source["pmf_fingerprint"]:
        raise ValueError("The prepared root or the PMF root is not the source run's")
    conditions = [ConditionKey(row["study"], row["pattern"], float(row["beta"]), int(row["replication"]))
                  for row in source["conditions"]]
    verify_prepared_inputs(prepared_root, prepared, conditions, rct_path)
    protocol = json.loads(json.dumps(source["protocol"]))
    protocol["trigger_rule"]["taus"] = taus
    inherited = ("experiment", "arm", "settings", "conditions", "prepared_fingerprint", "pmf_fingerprint", "pmf_draws",
                 "pilot_decision_fingerprint", "modular_design", "predictive_design")
    payload = _identity({
        **{name: source[name] for name in inherited if name in source},
        "stage": "regrid",
        "protocol": protocol,
        "dependencies": _dependency_versions(),
        "regrid": {
            "source_run": {"arm": source["arm"], "fingerprint": source["fingerprint"]},
            "tau_floor": float(tau_floor),
            "source_taus": source_taus,
            "rule": (
                "post hoc: the source run's Modules D and E replayed on its trigger grid restricted to tau >= tau_floor, "
                "from its saved tolerance draws, its surfaces' source features and its target PMF draws and seeds; "
                f"nothing is refitted, and the first {REGRID_CHECK_DRAWS} draws replayed on the run's grid reproduce "
                "the run's"
            ),
        },
    })
    identity_path = Path(output_dir) / "identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if not resume or existing.get("fingerprint") != payload["fingerprint"]:
            raise ValueError("Existing regrid identity differs or --resume was not supplied")
        return existing
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _write_json(payload, identity_path)
    write_run_manifest(
        output_dir,
        experiment_name=EXPERIMENT_NAME,
        script_path="experiments/exp31_bayesian_manylabs_naturalness.py",
        config={"stage": "regrid", "regrid": payload["regrid"], "trigger_rule": protocol["trigger_rule"]},
        argv=sys.argv,
        cluster_manifest_path=cluster_manifest_path,
        extra={
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": payload["fingerprint"],
            "git": _git_provenance(),
            "source_fingerprints": _source_fingerprints(),
            "software_environment": _software_environment(),
            "input_paths": {"source_root": str(Path(source_root).resolve()), "prepared_root": str(Path(prepared_root).resolve()),
                            "rct_path": str(Path(rct_path).resolve()), "pmf_root": str(Path(pmf_root).resolve())},
            "n_conditions": len(conditions),
        },
    )
    return payload


def run_regrid(
    output_dir: Path,
    source_root: Path,
    prepared_root: Path,
    rct_path: Path,
    pmf_root: Path,
    *,
    tau_floor: float,
    resume: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    cluster_manifest_path: str | None = None,
) -> dict[str, Any]:
    """Regrid every condition of a finalized run, sharded round-robin, then finalize.

    Shards share one root as in :func:`run_experiment`; the last shard to finish finalizes it with
    :func:`finalize_run`.  Every shard must get the same ``tau_floor``, which the identity fingerprints.

    Returns:
        This shard's completion record.

    Raises:
        RuntimeError: If any condition failed.
    """

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index} of {num_shards}")
    if shard_index > 0:
        _wait_for_run_identity(Path(output_dir))
    identity = initialize_regrid_root(output_dir, source_root, prepared_root, rct_path, pmf_root, tau_floor=tau_floor,
                                      resume=resume or shard_index > 0, cluster_manifest_path=cluster_manifest_path)
    conditions = [ConditionKey(row["study"], row["pattern"], float(row["beta"]), int(row["replication"]))
                  for row in identity["conditions"]]
    failures: list[dict[str, Any]] = []
    attempted = 0
    for position, key in enumerate(conditions):
        if position % num_shards != shard_index:
            continue
        attempted += 1
        try:
            regrid_condition(key=key, source_root=Path(source_root), prepared_root=Path(prepared_root),
                             rct_path=Path(rct_path), pmf_root=Path(pmf_root), output_dir=Path(output_dir),
                             run_identity=identity, resume=True)
        except Exception as error:  # keep the other conditions.
            LOGGER.exception("Exp31 regrid condition failed: %s", key.label())
            failures.append({**key.as_dict(), "error": repr(error)})
            leaf = key.leaf(output_dir)
            leaf.mkdir(parents=True, exist_ok=True)
            _write_json({"status": "failed", "error": repr(error), "failed_at_utc": _utc_now()}, leaf / "completion.json")
    record = {"status": "complete" if not failures else "failed", "attempted_conditions": attempted, "failures": failures,
              "completed_at_utc": _utc_now(), "shard_index": shard_index, "num_shards": num_shards}
    shards = Path(output_dir) / "shards"
    _write_json(record, shards / f"shard={shard_index:03d}-of-{num_shards:03d}.json")
    paths = [shards / f"shard={index:03d}-of-{num_shards:03d}.json" for index in range(num_shards)]
    if all(path.is_file() for path in paths) and all(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "complete" for path in paths
    ):
        lock = Path(output_dir) / "finalize.lock"
        try:
            lock.mkdir()
        except FileExistsError:
            LOGGER.info("Another shard is finalizing %s", output_dir)
        else:
            try:
                finalize_run(output_dir)
            finally:
                lock.rmdir()
    if failures:
        raise RuntimeError(f"{len(failures)} Exp31 regrid conditions failed")
    return record


# --- Report ----------------------------------------------------------------------

CONDITION_KEYS = ["study", "pattern", "beta", "replication"]
FAMILY_KEYS = ["study", "pattern", "beta"]
#: Condition-table columns: (column, scope, metric, method, estimand_type, estimand_key).
HEADLINE_METRICS = (
    ("p_any_eligible", "discovery", "p_any_candidate_eligible", "", "", ""),
    ("p_no_candidate_eligible", "discovery", "p_no_candidate_eligible", "", "", ""),
    ("mean_n_eligible", "discovery", "mean_n_eligible_candidates", "", "", ""),
    ("p_naturalness_nonempty", "feasibility", "p_nonempty", NATURALNESS, "", ""),
    ("n_fallback_draws", "feasibility", "n_fallback_draws", NATURALNESS, "", ""),
    ("n_eligible_nonempty_draws", "feasibility", "n_eligible_nonempty_draws", NATURALNESS, "", ""),
    ("n_eligible_empty_draws", "feasibility", "n_eligible_but_lp_infeasible_draws", NATURALNESS, "", ""),
    ("n_solver_failure_draws", "feasibility", "n_solver_failure_draws", NATURALNESS, "", ""),
    ("p_empty_given_eligible", "feasibility", "p_empty_given_eligible", NATURALNESS, "", ""),
    ("naturalness_weighted_width_mean", "weighted", "weighted_width_mean", NATURALNESS, "", ""),
    ("naturalness_weighted_width_q025", "weighted", "weighted_width_q025", NATURALNESS, "", ""),
    ("naturalness_weighted_width_q50", "weighted", "weighted_width_q50", NATURALNESS, "", ""),
    ("naturalness_weighted_width_q975", "weighted", "weighted_width_q975", NATURALNESS, "", ""),
    ("manski_weighted_width_mean", "weighted", "weighted_width_mean", MANSKI, "", ""),
    ("naturalness_weighted_outer_lower", "weighted", "weighted_outer_lower", NATURALNESS, "", ""),
    ("naturalness_weighted_outer_upper", "weighted", "weighted_outer_upper", NATURALNESS, "", ""),
    ("naturalness_weighted_lower_q50", "weighted", "weighted_lower_q50", NATURALNESS, "", ""),
    ("naturalness_weighted_upper_q50", "weighted", "weighted_upper_q50", NATURALNESS, "", ""),
    ("p_width_gain_positive_all", "paired", "p_width_gain_positive_all", "paired", "", ""),
    ("p_narrower_given_joint", "paired", "p_naturalness_narrower_given_joint", "paired", "", ""),
    ("weighted_width_gain_mean", "paired", "weighted_width_gain_mean", "paired", "", ""),
    ("largest_cell_share_of_gain", "paired", "largest_cell_share_of_weighted_gain", "paired", "", ""),
    ("naturalness_membership_all", "reference", "weighted_membership_all", NATURALNESS, "", ""),
    ("manski_membership_all", "reference", "weighted_membership_all", MANSKI, "", ""),
    ("naturalness_membership_given_nonempty", "reference", "weighted_membership_given_nonempty", NATURALNESS, "", ""),
    ("naturalness_wrong_sign_all", "reference", "weighted_wrong_sign_all", NATURALNESS, "", ""),
    ("manski_wrong_sign_all", "reference", "weighted_wrong_sign_all", MANSKI, "", ""),
    ("naturalness_wrong_sign_given_nonempty", "reference", "weighted_wrong_sign_given_nonempty", NATURALNESS, "", ""),
    ("lp_mass_without_reference_support", "reference", "lp_mass_without_reference_support", "", "", ""),
    ("site_average_scored", "reference", "reference_scoreable", NATURALNESS, "ate", "site_average"),
    ("site_average_naturalness_outer_lower", "bounds", "outer_lower", NATURALNESS, "ate", "site_average"),
    ("site_average_naturalness_outer_upper", "bounds", "outer_upper", NATURALNESS, "ate", "site_average"),
    ("site_average_p_sign_decision_all", "bounds", "p_sign_decision_all", NATURALNESS, "ate", "site_average"),
    ("target_mass_outside_lp_cells", "data", "target_mass_outside_lp_cells", "", "", ""),
    ("n_lp_cells", "data", "n_lp_cells", "", "", ""),
    ("n_reference_scored_cells", "data", "n_reference_scored_cells", "", "", ""),
    ("reference_shared_with_target_idle_share", "data", "reference_shared_with_target_idle_share", "", "", ""),
    ("n_source_sites", "data", "n_source_sites", "", "", ""),
    ("n_target_idle_rows", "data", "n_target_idle_rows", "", "", ""),
    ("all_used_surfaces_passed", "grounding", "all_used_surfaces_passed", "", "", ""),
    ("n_surfaces_excluded_unconverged", "grounding", "n_surfaces_excluded_unconverged", "", "", ""),
)
#: Condition-table columns of the Balke-Pearl arm, in the same spec.
BALKE_PEARL_HEADLINE_METRICS = (
    ("balke_pearl_p_any_instrument", "discovery", "p_any_instrument", BALKE_PEARL, "", ""),
    ("balke_pearl_mean_n_instruments", "discovery", "mean_n_instruments", BALKE_PEARL, "", ""),
    ("balke_pearl_p_nonempty", "feasibility", "p_nonempty", BALKE_PEARL, "", ""),
    ("balke_pearl_n_fallback_draws", "feasibility", "n_fallback_draws", BALKE_PEARL, "", ""),
    ("balke_pearl_n_eligible_nonempty_draws", "feasibility", "n_eligible_nonempty_draws", BALKE_PEARL, "", ""),
    ("balke_pearl_n_eligible_empty_draws", "feasibility", "n_eligible_but_lp_infeasible_draws", BALKE_PEARL, "", ""),
    ("balke_pearl_n_solver_failure_draws", "feasibility", "n_solver_failure_draws", BALKE_PEARL, "", ""),
    ("balke_pearl_mass_without_instrument", "feasibility", "mean_target_mass_without_instrument", BALKE_PEARL, "", ""),
    ("balke_pearl_weighted_width_mean", "weighted", "weighted_width_mean", BALKE_PEARL, "", ""),
    ("balke_pearl_weighted_width_q025", "weighted", "weighted_width_q025", BALKE_PEARL, "", ""),
    ("balke_pearl_weighted_width_q50", "weighted", "weighted_width_q50", BALKE_PEARL, "", ""),
    ("balke_pearl_weighted_width_q975", "weighted", "weighted_width_q975", BALKE_PEARL, "", ""),
    ("balke_pearl_membership_all", "reference", "weighted_membership_all", BALKE_PEARL, "", ""),
    ("balke_pearl_membership_given_nonempty", "reference", "weighted_membership_given_nonempty", BALKE_PEARL, "", ""),
    ("balke_pearl_wrong_sign_all", "reference", "weighted_wrong_sign_all", BALKE_PEARL, "", ""),
    ("balke_pearl_wrong_sign_given_nonempty", "reference", "weighted_wrong_sign_given_nonempty", BALKE_PEARL, "", ""),
    ("bp_narrower_than_manski_all", "paired", "p_width_gain_positive_all", BP_VS_MANSKI, "", ""),
    ("bp_vs_manski_width_gain_mean", "paired", "weighted_width_gain_mean", BP_VS_MANSKI, "", ""),
    ("naturalness_narrower_than_bp_all", "paired", "p_width_gain_positive_all", NATURALNESS_VS_BP, "", ""),
    ("naturalness_narrower_than_bp_given_joint", "paired", "p_narrower_given_joint", NATURALNESS_VS_BP, "", ""),
    ("naturalness_vs_bp_width_gain_mean", "paired", "weighted_width_gain_mean", NATURALNESS_VS_BP, "", ""),
)
#: Colours and markers of the selection-covariate (and candidate) series; Exp25's method colours
#: (:mod:`experiments.exp25_hit_ratio_plots`) stay reserved for methods.  Colour-vision safe, 3:1 contrast on white.
COVARIATE_COLORS = {"age": "#0097a7", "gender": "#b4531a", "polideo": "#3b5bdb"}
COVARIATE_MARKERS = {"age": "o", "gender": "s", "polideo": "^"}
#: Method labels of Exp25's plug-in Naturalness and Balke-Pearl in the width-by-beta tables.
EXP25_PLUGIN = "exp25_plugin_naturalness"
EXP25_PLUGIN_BP = "exp25_plugin_balke_pearl"
#: Central level of the posterior width interval in the width-by-beta tables and figures (the replication
#: bootstrap uses Exp25's 95%).
POSTERIOR_WIDTH_LEVEL = 0.90
#: Report figures are written as PNG, which ``report.md`` embeds, and as PDF.
FIGURE_FORMATS = ("png", "pdf")
#: As in Exp25 the figures carry no titles; ``report.md`` prints these captions.
FIGURE_CAPTIONS = {
    "02a_grounded_two_c": "Grounded 2c at the target: posterior median (markers) and 2.5-97.5% quantiles (bars) over "
                          "the candidate's eligible draws, averaged over replications and selection covariates.",
    "02b_grounded_two_d": "Grounded 2d at the target: posterior median (markers) and 2.5-97.5% quantiles (bars) over "
                          "the candidate's eligible draws, averaged over replications and selection covariates.",
    "03_feasibility_fallback_empty": "Naturalness LP outcome over all posterior draws, with no feasibility threshold: "
                                     "no eligible candidate (the matched Manski set), an eligible candidate with a "
                                     "nonempty set, or an eligible candidate with an empty set. Each bar pools every "
                                     "draw of the family's replications.",
    "04a_width_by_beta": "Target-mass-weighted CATE width against selection strength, selection covariates pooled. "
                         "Markers are the mean over replications of each condition's posterior mean width, with 95% "
                         "percentile intervals from 2,000 bootstrap resamples of the replications. The band is the "
                         f"posterior {POSTERIOR_WIDTH_LEVEL:.0%} interval of the random identified set's width over "
                         "each condition's nonempty draws, averaged over the conditions. Manski's width is one in "
                         "every draw. Exp25's plug-in Naturalness (different OSRCT seeds) is drawn hollow when its "
                         "intervals are supplied.",
    "04b_width_by_beta_by_pattern": "As the pooled width figure, with one row per OSRCT selection covariate.",
    "04c_paired_width_gain": "Paired width gain of Naturalness over the matched Bayesian Manski set (target-mass "
                             "weighted): P(narrower than Manski) over all draws, and the mean gain over jointly "
                             "nonempty draws.",
    "05_reference_membership": "Randomized-reference membership and wrong sign, posterior diagnostics rather than "
                               "coverage; dashed lines mark the prespecified classification thresholds.",
    "07_imputation_sensitivity": "Monte Carlo sensitivity of the pilot conditions: largest target-mass-weighted "
                                 "endpoint shift between 10 and 20 source imputations; the dashed line is the "
                                 "prespecified escalation threshold.",
    "08_loso_validity": "Leave-one-source-site-out: probability that the grounded tolerance is admissible for the "
                        "held-out site, per candidate and tolerance surface.",
    "04d_naturalness_vs_balke_pearl": "Post hoc: paired width comparison of Naturalness against the Bayesian "
                                      "Balke-Pearl arm on the same posterior draws (target-mass weighted): "
                                      "P(Naturalness narrower than Balke-Pearl) over jointly nonempty draws, and the "
                                      "mean gain W_BalkePearl - W_Naturalness, positive when Naturalness is narrower.",
    "05b_reference_by_method": "Post hoc: randomized-reference membership and wrong sign over all draws (an empty set "
                               "is a non-member) for Naturalness, the Bayesian Balke-Pearl arm and Manski, selection "
                               "covariates pooled; posterior diagnostics, not coverage.",
    "05c_width_membership_tradeoff": "Post hoc: width against randomized-reference membership for Naturalness and the "
                                     "Bayesian Balke-Pearl arm, one column per study. Top: the mean target-mass-"
                                     "weighted CATE width (horizontal) against membership over all draws (vertical), "
                                     "one marker per selection strength, joined in order of strength. Bottom: the "
                                     "share of draws whose set excludes zero on the side opposite the reference. "
                                     "Means over the conditions of each study and strength, with 95% percentile "
                                     "intervals from 2,000 bootstrap resamples of the replications. The values, "
                                     "Manski's included, are in `width_membership_tradeoff.csv`, and "
                                     "`width_membership_tradeoff_table.tex` sets them as the manuscript's results "
                                     "table.",
}
#: Methods of the width-membership trade-off (:func:`tradeoff_table`), keyed by their column prefix in the complete
#: condition table, with the label, colour and marker of their series in the trade-off figure.  Manski (width one,
#: never the wrong sign) is summarized for the results table but not drawn.
TRADEOFF_METHODS = {
    "naturalness": ("Naturalness", "tab:purple", "o"),
    "balke_pearl": ("Balke–Pearl", "tab:orange", "X"),
    "manski": ("Manski", "#64748b", "d"),
}
TRADEOFF_PLOTTED = ("naturalness", "balke_pearl")
TRADEOFF_BOOTSTRAP_REPLICATES = 2000
TRADEOFF_BOOTSTRAP_SEED = 20261004
#: Appended to the width figure's caption when the Balke-Pearl arm was run.
BALKE_PEARL_WIDTH_CAPTION = (
    " The post hoc Bayesian Balke-Pearl arm (orange; plan Section 16.7) is drawn the same way from the same posterior "
    "draws, and Exp25's plug-in Balke-Pearl is drawn hollow when its intervals are supplied."
)
#: Family columns of the Naturalness-against-Balke-Pearl comparison table.
BALKE_PEARL_COMPARISON_COLUMNS = (
    "naturalness_weighted_width_mean", "balke_pearl_weighted_width_mean", "manski_weighted_width_mean",
    "naturalness_narrower_than_bp_given_joint", "naturalness_vs_bp_width_gain_mean",
    "p_width_gain_positive_all", "bp_narrower_than_manski_all",
    "naturalness_membership_all", "balke_pearl_membership_all", "manski_membership_all",
    "naturalness_wrong_sign_all", "balke_pearl_wrong_sign_all",
    "p_naturalness_nonempty", "balke_pearl_p_nonempty", "balke_pearl_p_any_instrument",
    "balke_pearl_mass_without_instrument",
)


def _read_finalized(root: Path, arm: str | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return a finalized run's identity and condition metrics.

    Raises:
        ValueError: If the root is not a complete, validated Exp31 run, or not of the expected arm.
    """

    root = Path(root)
    identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    completion_path = root / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
    if identity.get("experiment") != EXPERIMENT_NAME or completion.get("status") != "complete" or not completion.get("validated"):
        raise ValueError(f"{root} is not a finalized Exp31 run")
    if arm is not None and identity["arm"] != arm:
        raise ValueError(f"{root} is a {identity['arm']!r} run, expected {arm!r}")
    return identity, pd.read_parquet(root / "condition_metrics.parquet")


def _metric_columns(metrics: pd.DataFrame, specs: Sequence[tuple[str, ...]]) -> list[pd.Series]:
    """Return one condition-indexed column per ``(column, scope, metric, method, estimand_type, estimand_key)`` spec."""

    columns = []
    for column, scope, metric, method, estimand_type, estimand_key in specs:
        mask = (
            metrics.scope.eq(scope) & metrics.metric.eq(metric) & metrics.method.eq(method)
            & metrics.estimand_type.eq(estimand_type) & metrics.estimand_key.eq(estimand_key)
        )
        columns.append(metrics.loc[mask].set_index(CONDITION_KEYS).value.rename(column))
    return columns


def balke_pearl_condition_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one row per condition with the Balke-Pearl arm's headline metrics.

    The columns (:data:`BALKE_PEARL_HEADLINE_METRICS`) are prefixed so they join the primary condition table.
    """

    table = pd.concat(_metric_columns(metrics, BALKE_PEARL_HEADLINE_METRICS), axis=1).reset_index()
    return table.sort_values(CONDITION_KEYS).reset_index(drop=True)


def condition_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one row per condition with its headline metrics.

    Candidate eligibility gets one column per candidate; the modal eligible-candidate set and its share are kept
    for the agreement summary.
    """

    frames = _metric_columns(metrics, HEADLINE_METRICS)
    for candidate in COVARIATES:
        for metric in ("p_eligible", "p_triggered", "p_outside_source_support_given_triggered", "p_surface_not_converged"):
            mask = metrics.scope.eq("candidate") & metrics.metric.eq(metric) & metrics.candidate.eq(candidate)
            frames.append(metrics.loc[mask].set_index(CONDITION_KEYS).value.rename(f"{metric}__{candidate}"))
    modal = metrics.loc[metrics.metric.eq("modal_eligible_set_share")].set_index(CONDITION_KEYS)
    frames.append(modal.value.rename("modal_eligible_set_share"))
    frames.append(modal.candidate.rename("modal_eligible_set"))
    table = pd.concat(frames, axis=1).reset_index()
    return table.sort_values(CONDITION_KEYS).reset_index(drop=True)


def _candidate_family_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize discovery and grounded tolerances by family and candidate."""

    wanted = [
        "p_triggered", "p_eligible", "p_in_source_support", "p_outside_source_support_given_triggered",
        "target_i_deviation_q50", "target_d_strength_q50", "grounded_two_c_q50", "grounded_two_c_q025",
        "grounded_two_c_q975", "grounded_two_d_q50", "grounded_two_d_q025", "grounded_two_d_q975",
        "grounded_c_q50", "grounded_d_q50",
    ]
    rows = metrics.loc[metrics.scope.eq("candidate") & metrics.metric.isin(wanted)]
    wide = rows.pivot_table(index=[*CONDITION_KEYS, "candidate"], columns="metric", values="value", aggfunc="first")
    denominators = rows.loc[rows.metric.eq("grounded_two_c_q50")].set_index([*CONDITION_KEYS, "candidate"]).denominator
    wide["n_candidate_eligible_draws"] = denominators
    grouped = wide.reset_index().groupby([*FAMILY_KEYS, "candidate"], sort=True)
    summary = grouped.mean(numeric_only=True).drop(columns=["replication"])
    summary["sd_p_eligible_across_replications"] = grouped.p_eligible.std()
    summary["n_replications"] = grouped.size()
    summary["n_candidate_eligible_draws"] = grouped.n_candidate_eligible_draws.sum()
    return summary.reset_index()


#: Column prefix of each method's posterior width interval in the condition table.
WIDTH_PREFIXES = {NATURALNESS: "naturalness", BALKE_PEARL: "balke_pearl"}


def posterior_width_intervals(
    root: Path,
    identity: Mapping[str, Any],
    level: float = POSTERIOR_WIDTH_LEVEL,
    *,
    method: str = NATURALNESS,
) -> pd.DataFrame:
    """Return each condition's central posterior interval of one method's width.

    The finalized metrics keep only the 2.5, 50 and 97.5% width quantiles, so other levels are recomputed from the
    saved random identified sets with the same per-draw weighted width (:func:`_per_draw_weighted_width`).  The
    recomputed posterior mean is returned too, for checking against the finalized metrics.

    Args:
        root: A finalized run root with a single imputation count, or a Balke-Pearl root.
        identity: The root's ``identity.json``.
        level: Central posterior level in ``(0, 1)``.
        method: The method whose width is summarized, a key of :data:`WIDTH_PREFIXES`.

    Returns:
        One row per condition with the :data:`CONDITION_KEYS` and, under the method's prefix,
        ``_width_posterior_lower``, ``_width_posterior_upper`` and ``_width_recomputed_mean``.

    Raises:
        ValueError: If ``level`` is outside ``(0, 1)`` or the root has more than one imputation count.
    """

    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    counts = identity["settings"]["imputation_counts"]
    if len(counts) != 1:
        raise ValueError(f"Expected a single imputation count, got {counts}")
    prefix = WIDTH_PREFIXES[method]
    name = ("bp_random_identified_sets.parquet" if identity["arm"] == "balke_pearl"
            else f"imputations={counts[0]}/random_identified_sets.parquet")
    columns = ["method", "draw_index", "estimand_type", "estimand_weight", "width", "nonempty"]
    rows = []
    for item in identity["conditions"]:
        key = ConditionKey(item["study"], item["pattern"], float(item["beta"]), int(item["replication"]))
        sets = pd.read_parquet(key.leaf(root) / name, columns=columns)
        per_draw = _per_draw_weighted_width(sets.loc[sets.estimand_type.eq("cate") & sets.method.eq(method)])
        widths = per_draw.loc[per_draw.nonempty, "weighted_width"]
        rows.append({
            **dict(zip(CONDITION_KEYS, (key.study, key.pattern, key.beta, key.replication))),
            f"{prefix}_width_posterior_lower": float(widths.quantile((1 - level) / 2)) if len(widths) else np.nan,
            f"{prefix}_width_posterior_upper": float(widths.quantile((1 + level) / 2)) if len(widths) else np.nan,
            f"{prefix}_width_recomputed_mean": float(widths.mean()) if len(widths) else np.nan,
        })
    return pd.DataFrame(rows)


def _report_studies(frame: pd.DataFrame) -> list[str]:
    """Return the studies present in a report table, in :data:`ALL_STUDIES` order."""

    present = set(frame.study.astype(str))
    return [study for study in ALL_STUDIES if study in present] or list(STUDIES)


def _as_roots(value: Path | str | Sequence[Path | str] | None) -> list[Path]:
    """Return one root or several as a list of paths (empty for ``None``)."""

    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


#: Settings that may differ between runs of one design that a report combines.
COMBINABLE_SETTINGS_EXEMPT = ("studies", "replications", "cluster_manifest_path")


def _design_signature(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the parts of a run identity that runs combined in one report must share.

    The arm, the protocol, the settings other than the condition selection, the PMF draw counts and any modular or
    predictive design; the inputs (prepared and PMF roots) and the conditions may differ.
    """

    settings = {name: value for name, value in identity.get("settings", {}).items()
                if name not in COMBINABLE_SETTINGS_EXEMPT}
    return {"arm": identity.get("arm"), "settings": settings, "protocol": identity.get("protocol"),
            "pmf_draws": identity.get("pmf_draws"), "modular_design": identity.get("modular_design"),
            "predictive_design": identity.get("predictive_design")}


def _read_combined(roots: Sequence[Path], arms: Sequence[str]) -> list[tuple[Path, dict[str, Any], pd.DataFrame]]:
    """Read finalized runs of one design over disjoint conditions, to be reported as one run.

    Raises:
        ValueError: If a root is not a finalized run of one of ``arms``, the runs differ in design
            (:func:`_design_signature`), or two runs share a condition.
    """

    runs = []
    for root in roots:
        identity, metrics = _read_finalized(root)
        if identity["arm"] not in arms:
            expected = ", ".join(repr(arm) for arm in arms[:-1]) + f" or {arms[-1]!r}" if len(arms) > 1 else repr(arms[0])
            raise ValueError(f"{root} is a {identity['arm']!r} run, expected {expected}")
        runs.append((Path(root), identity, metrics))
    first = _design_signature(runs[0][1])
    seen: set[tuple[str, str, float, int]] = set()
    for root, identity, _ in runs:
        if _design_signature(identity) != first:
            raise ValueError(f"{root} has another design than {runs[0][0]}; only runs with identical settings combine")
        keys = {(row["study"], row["pattern"], float(row["beta"]), int(row["replication"])) for row in identity["conditions"]}
        if keys & seen:
            raise ValueError(f"{root} repeats conditions of another combined run")
        seen |= keys
    return runs


def _merge_posterior_widths(conditions: pd.DataFrame, root: Path, identity: Mapping[str, Any], method: str) -> pd.DataFrame:
    """Join one method's posterior width intervals to the condition table, checked against its metrics.

    Raises:
        ValueError: If the saved random sets do not reproduce the method's finalized posterior-mean widths.
    """

    prefix = WIDTH_PREFIXES[method]
    merged = conditions.merge(posterior_width_intervals(root, identity, method=method), on=CONDITION_KEYS,
                              how="left", validate="one_to_one")
    recomputed = merged.pop(f"{prefix}_width_recomputed_mean")
    if not np.allclose(recomputed, merged[f"{prefix}_weighted_width_mean"], rtol=0.0, atol=1e-12, equal_nan=True):
        raise ValueError(f"The saved random identified sets do not reproduce the finalized {prefix} widths")
    return merged


def _exp25_method_rows(path: Path, method: str = "naturalness") -> pd.DataFrame:
    """Read one method's saved Exp25 intervals at the Exp31 target site.

    Adds the Exp31 family keys (``pattern`` and a float ``beta``), each cell's weight (its target idle rows) and
    whether a valid interval contains the Wald RCT estimate.  Exp25 keeps Naturalness in its main intervals and
    Balke-Pearl in its ``identification_bounds`` intervals.
    """

    intervals = pd.read_parquet(path)
    rows = intervals.loc[intervals.method.eq(method) & intervals.target_site.astype(str).eq(TARGET_SITE)].copy()
    if rows.empty:
        raise ValueError(f"{path} holds no {method!r} intervals at target site {TARGET_SITE!r}")
    rows["pattern"] = rows.bias_covariate.astype(str)
    rows["beta"] = rows.beta.astype(float)
    available = rows.is_valid.astype(bool) & rows.rct_cate_available.astype(bool)
    rows["contains"] = available & rows.rct_cate.between(rows.lower, rows.upper)
    rows["weight"] = rows.n_target_idle_rows.astype(float)
    return rows


def _exp25_condition_widths(path: Path, method: str = "naturalness") -> pd.DataFrame:
    """Return one method's Exp25 plug-in width per condition, as the bridge weights it.

    The width is the target-idle-row-weighted mean over the condition's valid cells, one row per family and
    replication.
    """

    rows = _exp25_method_rows(path, method)
    valid = rows.loc[rows.is_valid.astype(bool)].assign(weighted_width=lambda frame: frame.width * frame.weight)
    sums = valid.groupby([*FAMILY_KEYS, "replication"])[["weighted_width", "weight"]].sum()
    return (sums.weighted_width / sums.weight).rename("width").reset_index()


def _exp25_bridge(path: Path, families: pd.DataFrame) -> pd.DataFrame:
    """Set Exp25's plug-in Naturalness beside Exp31, family by family.

    Descriptive only: the two use different OSRCT seeds (Exp25 202608-202627, Exp31 62-81), Exp25 scores plug-in
    intervals for point containment of the Wald RCT estimate while Exp31 membership is posterior containment of
    reference draws, and cells are weighted by target idle rows in Exp25 and by target idle mass in Exp31.
    """

    rows = _exp25_method_rows(path)
    output = []
    for family, group in rows.groupby(FAMILY_KEYS, sort=True):
        valid_rows = group.loc[group.is_valid.astype(bool)]
        scored = group.loc[group.is_valid.astype(bool) & group.rct_cate_available.astype(bool)]
        output.append({
            **dict(zip(FAMILY_KEYS, family)),
            "exp25_n_replications": int(group.replication.nunique()),
            "exp25_valid_share": float(group.is_valid.astype(bool).mean()),
            "exp25_full_range_fallback_share": float(group.used_full_range_fallback.astype(bool).mean()),
            "exp25_weighted_width_valid": float(np.average(valid_rows.width, weights=valid_rows.weight)) if len(valid_rows) else np.nan,
            "exp25_weighted_point_containment": float(np.average(scored.contains, weights=scored.weight)) if len(scored) else np.nan,
        })
    bridge = pd.DataFrame(output).merge(
        families[[*FAMILY_KEYS, "p_any_eligible", "p_naturalness_nonempty", "naturalness_weighted_width_mean",
                  "naturalness_membership_all", "naturalness_membership_given_nonempty"]].rename(columns=lambda c: c if c in FAMILY_KEYS else f"exp31_{c}"),
        on=FAMILY_KEYS, how="outer",
    )
    bridge["note"] = "descriptive bridge; different OSRCT seeds, estimators and containment notions"
    return bridge


def classify_families(
    families: pd.DataFrame,
    conditions: pd.DataFrame,
    imputation_shifts: pd.DataFrame | None,
    loso: pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply the report's classification to every family.

    Each flag is computed from :data:`CLASSIFICATION_RULES`.  The label is the first matching category in the order
    overconfident restriction, empty-set tension, unstable grounding, valid abstention, useful example, else
    ``mixed_evidence``.  Feasibility enters only the descriptive empty-set-tension flag, never the useful-example
    label.  ``robust`` means no stability check flagged the family; the Monte Carlo (pilot) and leave-one-out checks
    cover their own condition subsets only, as ``imputation_assessed`` and ``loso_assessed`` record.

    Returns:
        The family evidence with every flag and the primary label.
    """

    rules = CLASSIFICATION_RULES
    output = families.copy()
    recurrence = conditions.assign(
        discovered=conditions.p_any_eligible.ge(rules["discovery_replication_threshold"])
    ).groupby(FAMILY_KEYS).discovered.mean().rename("discovery_recurrence_share")
    output = output.merge(recurrence.reset_index(), on=FAMILY_KEYS, how="left")
    output["stable_discovery"] = output.discovery_recurrence_share.ge(rules["stable_discovery_min_replication_share"])
    output["abstention"] = output.p_any_eligible.lt(rules["abstention_max_any_eligible"])
    output["empty_set_tension"] = output.pooled_p_empty_given_eligible.ge(rules["empty_set_tension_min_empty_given_eligible"])
    output["width_gain_for_most_mass"] = output.p_width_gain_positive_all.gt(rules["useful_min_width_gain_positive_all"])
    output["membership_drop"] = output.manski_membership_all - output.naturalness_membership_all
    output["wrong_sign_increase"] = output.naturalness_wrong_sign_all - output.manski_wrong_sign_all
    output["material_membership_drop"] = output.membership_drop.gt(rules["material_membership_drop"])
    output["material_wrong_sign_increase"] = output.wrong_sign_increase.gt(rules["material_wrong_sign_increase"])
    output["single_cell_driven"] = output.largest_cell_share_of_gain.gt(rules["single_cell_width_gain_share"])
    output["max_imputation_endpoint_shift"] = np.nan
    if imputation_shifts is not None and not imputation_shifts.empty:
        shifts = imputation_shifts.groupby(FAMILY_KEYS).maximum_shift.max()
        output["max_imputation_endpoint_shift"] = output.set_index(FAMILY_KEYS).index.map(shifts)
    output["imputation_unstable"] = output.max_imputation_endpoint_shift.gt(rules["imputation_endpoint_shift"])
    output["max_loso_target_tolerance_shift"] = np.nan
    if loso is not None and not loso.empty:
        shifts = loso.groupby(FAMILY_KEYS).max_abs_target_tolerance_shift.max()
        output["max_loso_target_tolerance_shift"] = output.set_index(FAMILY_KEYS).index.map(shifts)
    output["loso_unstable"] = output.max_loso_target_tolerance_shift.gt(rules["loso_target_tolerance_shift"])
    output["imputation_assessed"] = output.max_imputation_endpoint_shift.notna()
    output["loso_assessed"] = output.max_loso_target_tolerance_shift.notna()
    output["unstable_grounding"] = output.imputation_unstable | output.loso_unstable
    output["robust"] = ~output.unstable_grounding
    useful = (
        output.stable_discovery & output.width_gain_for_most_mass & ~output.material_membership_drop
        & ~output.material_wrong_sign_increase & output.robust & ~output.single_cell_driven
    )
    output["label"] = np.select(
        [
            output.width_gain_for_most_mass & (output.material_membership_drop | output.material_wrong_sign_increase),
            output.empty_set_tension,
            output.unstable_grounding,
            output.abstention,
            useful,
        ],
        ["overconfident_restriction", "empty_set_tension", "unstable_grounding", "valid_abstention",
         "useful_bayesian_naturalness_example"],
        default="mixed_evidence",
    )
    return output


def family_table(conditions: pd.DataFrame) -> pd.DataFrame:
    """Average the condition table within each study-pattern-beta family.

    Probabilities over draws are averaged over replications, which pools them since every condition has the same
    number of draws; the conditional empty-set probability is pooled over its own eligible draws.
    """

    numeric = conditions.select_dtypes(include=[np.number, bool]).columns.difference(CONDITION_KEYS)
    grouped = conditions.groupby(FAMILY_KEYS, sort=True)
    families = grouped[list(numeric)].mean()
    families["n_replications"] = grouped.size()
    families["sd_p_any_eligible"] = grouped.p_any_eligible.std()
    eligible = grouped.n_eligible_nonempty_draws.sum() + grouped.n_eligible_empty_draws.sum()
    families["pooled_eligible_draws"] = eligible
    families["pooled_p_empty_given_eligible"] = grouped.n_eligible_empty_draws.sum() / eligible.where(eligible > 0)
    agreement = grouped.modal_eligible_set.agg(lambda values: values.value_counts().iloc[0] / len(values))
    families["modal_set_agreement_across_replications"] = agreement
    families["family_modal_eligible_set"] = grouped.modal_eligible_set.agg(lambda values: values.value_counts().index[0])
    return families.reset_index()


def _plot_style() -> Any:
    """Return Exp25's plotting module, whose palette, layout and file output the figures reuse.

    Imported on first use, so the sampling stages, which never plot, do not load it.
    """

    from experiments import exp25_hit_ratio_plots

    return exp25_hit_ratio_plots


def _beta_axis(axis: Any) -> None:
    """Put the selection strengths on a linear x-axis under Exp25's light grid.

    Linear rather than Exp25's logarithmic axis, because the four strengths are evenly spaced.
    """

    axis.set_xticks(list(BETAS), [_format_beta(beta) for beta in BETAS])
    axis.set_xlim(BETAS[0] - 0.2, BETAS[-1] + 0.2)
    axis.grid(alpha=0.22)
    axis.set_axisbelow(True)


def _facet_headers(axes: np.ndarray, *, columns: Sequence[str], rows: Sequence[str] | None = None) -> None:
    """Head each column in bold and, when given, label each row on the right, as Exp25's grids do."""

    size = _plot_style().FRONTIER_HEADER_FONTSIZE
    for index, label in enumerate(columns):
        axes[0][index].set_title(label, fontsize=size, fontweight="bold")
    for index, label in enumerate(rows or ()):
        axes[index][-1].text(1.04, 0.5, label, transform=axes[index][-1].transAxes, rotation=270, ha="left",
                             va="center", fontsize=size, fontweight="bold")


def _series_handles(values: Sequence[str], colors: Mapping[str, str], markers: Mapping[str, str],
                    labels: Mapping[str, str]) -> list[Any]:
    """Return legend handles for line series, each with its colour, marker and label."""

    from matplotlib.lines import Line2D

    return [Line2D([], [], color=colors[value], marker=markers[value], linestyle="-", label=labels[value])
            for value in values]


def _method_series_style() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return Exp25's colours and markers, and display labels, for the Bayesian methods."""

    style = _plot_style()
    names = {NATURALNESS: "naturalness", BALKE_PEARL: "balke_pearl", MANSKI: "manski"}
    return ({method: style.MODEL_COLORS[name] for method, name in names.items()},
            {method: style.MODEL_MARKERS[name] for method, name in names.items()},
            {NATURALNESS: "Naturalness", BALKE_PEARL: "Balke-Pearl", MANSKI: "Manski"})


def _threshold_line(axis: Any, value: float) -> None:
    """Draw a classification threshold as a dashed line labelled just beneath it, at the panel's right edge.

    Beneath, so that a threshold at the top of the scale keeps its label inside the panel.
    """

    axis.axhline(value, color="0.4", linestyle="--", linewidth=1.0)
    axis.annotate(f"prespecified {value:g}", xy=(0.98, value), xycoords=axis.get_yaxis_transform(),
                  xytext=(0, -2), textcoords="offset points", ha="right", va="top", fontsize=8, color="0.3")


def _line_grid(
    plt: Any,
    families: pd.DataFrame,
    *,
    series_column: str,
    series_values: Sequence[str],
    series_prefix: str,
    y_columns: Sequence[tuple[str, str]],
    stem: Path,
    ylim: tuple[float, float] | None = None,
    reference_lines: Mapping[str, float] | None = None,
    range_columns: Mapping[str, tuple[str, str]] | None = None,
    series_style: tuple[Mapping[str, str], Mapping[str, str], Mapping[str, str]] | None = None,
) -> None:
    """Draw study columns by metric rows of beta-response lines, one per series.

    In the style of Exp25's line figures: a colour and marker per series, one y-scale per row, bold study headings,
    the legend across the top and no title (``report.md`` prints the caption).  Ranges are error bars dodged along
    the strength axis, because the series' ranges can coincide.

    Args:
        plt: The imported ``matplotlib.pyplot``.
        families: Family-level rows with the columns plotted.
        series_column: Column naming the series: a candidate, a selection covariate or a method.
        series_values: Series values, keys of :data:`COVARIATE_COLORS` unless ``series_style`` is given.
        series_prefix: Legend prefix of every covariate series, such as ``"candidate "``.
        y_columns: ``(column, y-axis label)`` for every row of panels.
        stem: Output path without an extension.
        ylim: Shared y-limits; otherwise each row runs from zero (or its negative minimum) to just above its
            highest value or threshold.
        reference_lines: Optional ``column -> classification threshold`` line.
        range_columns: Optional ``column -> (low, high)`` drawn as error bars.
        series_style: Optional ``(colors, markers, labels)`` keyed by series value, such as
            :func:`_method_series_style`.
    """

    colors, markers, labels = series_style or (
        COVARIATE_COLORS, COVARIATE_MARKERS, {value: f"{series_prefix}{value}" for value in series_values})

    style = _plot_style()
    # The legend band has a fixed height, so a single row needs a taller figure.
    studies = _report_studies(families)
    figure, axes = style._panel_grid(len(y_columns), len(studies), width=4.6, height=3.0 if len(y_columns) > 1 else 3.8,
                                     sharex=True, sharey="row")
    for row, (column, label) in enumerate(y_columns):
        tops = [families[column].max()]
        if range_columns and column in range_columns:
            tops.append(families[range_columns[column][1]].max())
        if reference_lines and column in reference_lines:
            tops.append(reference_lines[column])
        top = pd.Series(tops, dtype=float).max()
        bottom = min(float(families[column].min()), 0.0)
        for col, study in enumerate(studies):
            axis = axes[row][col]
            subset = families.loc[families.study.eq(study)]
            for index, value in enumerate(series_values):
                line = subset.loc[subset[series_column].eq(value)].sort_values("beta")
                if line.empty or line[column].isna().all():
                    continue
                color = colors[value]
                x = line.beta.to_numpy(dtype=float)
                if range_columns and column in range_columns:
                    low, high = range_columns[column]
                    x = x + 0.06 * (index - (len(series_values) - 1) / 2)
                    error = [np.maximum(line[column] - line[low], 0.0), np.maximum(line[high] - line[column], 0.0)]
                    axis.errorbar(x, line[column], yerr=error, fmt="none", ecolor=color, elinewidth=1.1, capsize=2.5,
                                  alpha=0.9)
                axis.plot(x, line[column], color=color, marker=markers[value], markersize=5.5,
                          linewidth=1.4, alpha=0.9)
            if reference_lines and column in reference_lines:
                _threshold_line(axis, reference_lines[column])
            # Probabilities, gains over Manski (whose region contains every other method's) and absolute shifts are
            # nonnegative; a gain over Balke-Pearl need not be, and then gets a zero line.
            if ylim is not None:
                axis.set_ylim(*ylim)
            elif bottom < 0.0:
                span = max(top, 0.0) - bottom
                axis.axhline(0.0, color="0.4", linewidth=0.8)
                axis.set_ylim(bottom - 0.05 * span, max(top, 0.0) + 0.05 * span)
            else:
                axis.set_ylim(0.0, 1.05 * top if top > 0 else 1.0)
            _beta_axis(axis)
            if col == 0:
                axis.set_ylabel(label)
            if row == len(y_columns) - 1:
                axis.set_xlabel(f"{style.BETA_SYMBOL} (selection strength)")
    _facet_headers(axes, columns=studies)
    style._finalize_figure(figure, _series_handles(series_values, colors, markers, labels))
    style._save_figure(figure, stem, FIGURE_FORMATS)


def _feasibility_figure(plt: Any, families: pd.DataFrame, stem: Path) -> None:
    """Stack the share of draws that fall back, are nonempty or are empty.

    One panel per study (columns) and selection covariate (rows), one bar per strength over the family's pooled
    draws.  The fallback (the matched Manski set) takes Manski's colour, a nonempty set Naturalness's, and an empty
    set Exp25's failure red.  ``report.md`` states the draws per bar.
    """

    from matplotlib.patches import Patch

    style = _plot_style()
    states = (("n_fallback_draws", "no eligible candidate (matched Manski)", style.MODEL_COLORS["manski"]),
              ("n_eligible_nonempty_draws", "eligible, nonempty set", style.MODEL_COLORS["naturalness"]),
              ("n_eligible_empty_draws", "eligible, empty set", "#b2182b"))
    studies = _report_studies(families)
    figure, axes = style._panel_grid(len(PATTERNS), len(studies), width=4.6, height=2.8, sharex=True, sharey=True)
    for row, pattern in enumerate(PATTERNS):
        for col, study in enumerate(studies):
            axis = axes[row][col]
            subset = families.loc[families.study.eq(study) & families.pattern.eq(pattern)].sort_values("beta")
            totals = subset[[name for name, _, _ in states]].sum(axis=1).to_numpy(dtype=float)
            bottom = np.zeros(len(subset))
            for name, _, color in states:
                share = subset[name].to_numpy(dtype=float) / np.where(totals > 0, totals, 1.0)
                axis.bar(subset.beta, share, bottom=bottom, width=0.3, color=color, edgecolor="white", linewidth=0.8)
                bottom += share
            axis.set_ylim(0.0, 1.0)
            _beta_axis(axis)
            axis.xaxis.grid(False)
    _facet_headers(axes, columns=studies, rows=[f"selection on {pattern}" for pattern in PATTERNS])
    style._finalize_shared_axes_figure(
        figure, axes, [Patch(facecolor=color, label=label) for _, label, color in states],
        xlabel=f"{style.BETA_SYMBOL} (selection strength)", ylabel="Share of all posterior draws",
        legend_text_style=style.FRONTIER_LEGEND_TEXT_STYLE,
    )
    style._save_figure(figure, stem, FIGURE_FORMATS)


def _bar_panel(plt: Any, table: pd.DataFrame, *, label_column: str, value_column: str, series_column: str | None,
               ylabel: str, stem: Path, threshold: float | None = None, series_prefix: str = "") -> None:
    """Draw one bar panel in Exp25's style, grouped by series when given.

    Args:
        plt: The imported ``matplotlib.pyplot``.
        table: One row per bar.
        label_column: Column of the x-axis labels, one group of bars each.
        value_column: Column of the bar heights.
        series_column: Optional column of the series, keys of :data:`COVARIATE_COLORS`; without one, every bar is
            Naturalness purple.
        ylabel: The y-axis label.
        stem: Output path without an extension.
        threshold: Optional classification threshold, drawn as a dashed line.
        series_prefix: Legend prefix of every series, such as ``"candidate "``.
    """

    from matplotlib.patches import Patch

    style = _plot_style()
    labels = list(dict.fromkeys(table[label_column]))
    series = [None] if series_column is None else list(dict.fromkeys(table[series_column]))
    figure, axes = style._panel_grid(1, 1, width=max(6.0, 0.45 * len(labels) + 2.0), height=3.8)
    axis = axes[0][0]
    width = min(0.8 / len(series), 0.3)
    for index, value in enumerate(series):
        subset = table if value is None else table.loc[table[series_column].eq(value)]
        positions = [labels.index(label) + (index - (len(series) - 1) / 2) * width for label in subset[label_column]]
        color = style.MODEL_COLORS["naturalness"] if value is None else COVARIATE_COLORS[value]
        axis.bar(positions, subset[value_column], width=width * 0.9, color=color, edgecolor="white", linewidth=0.8)
    if threshold is not None:
        _threshold_line(axis, threshold)
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.22, axis="y")
    axis.set_axisbelow(True)
    handles = [] if series_column is None else [
        Patch(facecolor=COVARIATE_COLORS[value], label=f"{series_prefix}{value}") for value in series
    ]
    style._finalize_figure(figure, handles)
    style._save_figure(figure, stem, FIGURE_FORMATS)


def width_by_beta_table(
    conditions: pd.DataFrame,
    exp25: pd.DataFrame | None = None,
    *,
    by_pattern: bool = False,
    exp25_bp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize the target-mass-weighted CATE width against selection strength.

    Two sources of uncertainty are kept apart.  ``posterior_lower`` and ``posterior_upper`` average every
    condition's central :data:`POSTERIOR_WIDTH_LEVEL` posterior width interval (:func:`posterior_width_intervals`).
    ``mean`` is the mean over replications of the posterior-mean width, and ``ci_lower``/``ci_upper`` a 95%
    percentile bootstrap over whole replications with Exp25's resample count and seed
    (:mod:`experiments.exp25_bootstrap_analysis`); unless ``by_pattern``, selection covariates are averaged within
    each replication first.  The Balke-Pearl series come last, so they leave the others' bootstrap draws unchanged.

    Args:
        conditions: Output of :func:`condition_table`, merged with :func:`posterior_width_intervals` and, when run,
            the Balke-Pearl arm's columns.
        exp25: Optional plug-in Naturalness widths (:func:`_exp25_condition_widths`) at the Exp31 studies and
            strengths; like Manski's (width one in every draw), their posterior columns are NaN.
        by_pattern: Whether to keep one row per selection covariate.
        exp25_bp: Optional plug-in Balke-Pearl widths, as ``exp25``.

    Returns:
        One row per study, selection covariate when ``by_pattern``, strength and method (:data:`METHODS`,
        :data:`BALKE_PEARL`, :data:`EXP25_PLUGIN` and :data:`EXP25_PLUGIN_BP` when present), with ``mean``,
        ``ci_lower``, ``ci_upper``, ``posterior_lower``, ``posterior_upper``, ``posterior_level``, ``n_replications``
        and ``n_conditions`` (those with a finite width).
    """

    from experiments.hit_ratio_metrics import (
        DEFAULT_BOOTSTRAP_REPLICATES,
        DEFAULT_BOOTSTRAP_SEED,
        DEFAULT_CONFIDENCE_LEVEL,
    )
    from experiments.interval_stats import percentile_interval

    keys = ["study", "pattern", "beta"] if by_pattern else ["study", "beta"]
    posterior = ["naturalness_width_posterior_lower", "naturalness_width_posterior_upper"]
    series = [(NATURALNESS, conditions, "naturalness_weighted_width_mean", posterior),
              (MANSKI, conditions, "manski_weighted_width_mean", [])]
    studies = _report_studies(conditions)
    if exp25 is not None:
        shared = exp25.study.isin(studies) & exp25.pattern.isin(PATTERNS) & exp25.beta.isin(BETAS)
        series.append((EXP25_PLUGIN, exp25.loc[shared], "width", []))
    if "balke_pearl_weighted_width_mean" in conditions:
        series.append((BALKE_PEARL, conditions, "balke_pearl_weighted_width_mean",
                       ["balke_pearl_width_posterior_lower", "balke_pearl_width_posterior_upper"]))
    if exp25_bp is not None:
        shared = exp25_bp.study.isin(studies) & exp25_bp.pattern.isin(PATTERNS) & exp25_bp.beta.isin(BETAS)
        series.append((EXP25_PLUGIN_BP, exp25_bp.loc[shared], "width", []))
    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    rows = []
    for method, frame, column, quantiles in series:
        counts = frame.groupby(keys)[column].count()
        # Pandas means skip a condition without a nonempty draw (NaN width).
        per_replication = frame.groupby([*keys, "replication"])[[column, *quantiles]].mean()
        for values, group in per_replication.groupby(level=keys, sort=True):
            means = group[column].to_numpy(dtype=float)
            means = means[np.isfinite(means)]
            lower = upper = np.nan
            if means.size:
                sampled = rng.integers(0, means.size, size=(DEFAULT_BOOTSTRAP_REPLICATES, means.size))
                lower, upper = percentile_interval(means[sampled].mean(axis=1), DEFAULT_CONFIDENCE_LEVEL)
            rows.append({
                **dict(zip(keys, values)),
                "method": method,
                "mean": float(means.mean()) if means.size else np.nan,
                "ci_lower": lower,
                "ci_upper": upper,
                "posterior_lower": float(group[quantiles[0]].mean()) if quantiles else np.nan,
                "posterior_upper": float(group[quantiles[1]].mean()) if quantiles else np.nan,
                "posterior_level": POSTERIOR_WIDTH_LEVEL if quantiles else np.nan,
                "n_replications": int(means.size),
                "n_conditions": int(counts.loc[values]),
            })
    return pd.DataFrame(rows)


def balke_pearl_comparison_table(families: pd.DataFrame, keys: Sequence[str] = tuple(FAMILY_KEYS)) -> pd.DataFrame:
    """Set Naturalness, the Balke-Pearl arm and Manski side by side, averaged within ``keys``.

    Every column (:data:`BALKE_PEARL_COMPARISON_COLUMNS`) is a family mean over replications; coarser ``keys``
    average the families, which have equal replication counts.  ``p_width_gain_positive_all`` is Naturalness's
    probability of being narrower than Manski over all draws.
    """

    return families.groupby(list(keys), as_index=False)[list(BALKE_PEARL_COMPARISON_COLUMNS)].mean()


def _reference_by_method(families: pd.DataFrame) -> pd.DataFrame:
    """Return all-draw reference membership and wrong sign per method, selection covariates pooled."""

    pooled = families.groupby(["study", "beta"], as_index=False).mean(numeric_only=True)
    prefixes = {NATURALNESS: "naturalness", BALKE_PEARL: "balke_pearl", MANSKI: "manski"}
    return pd.concat([
        pooled[["study", "beta"]].assign(method=method, membership_all=pooled[f"{prefix}_membership_all"],
                                         wrong_sign_all=pooled[f"{prefix}_wrong_sign_all"])
        for method, prefix in prefixes.items()
    ], ignore_index=True)


def tradeoff_table(conditions: pd.DataFrame) -> pd.DataFrame:
    """Summarize width, reference membership and wrong sign per study, strength and method.

    Each value is the mean over a study and strength's conditions of the target-mass-weighted CATE width, or of the
    reference membership or wrong-sign rate over all draws, with a 95% percentile interval from
    :data:`TRADEOFF_BOOTSTRAP_REPLICATES` bootstrap resamples of the replications' means.  One seeded generator is
    consumed in a fixed order (study, strength, method, then width, membership and wrong sign), so the outputs are
    reproducible.

    Args:
        conditions: The complete condition table, with the Balke-Pearl arm's columns.

    Returns:
        One row per study, strength and method, with ``width``, ``member`` and ``wrong`` and their ``_lo`` and
        ``_hi`` interval bounds.
    """

    rng = np.random.RandomState(TRADEOFF_BOOTSTRAP_SEED)
    rows = []
    for (study, beta), group in conditions.groupby(["study", "beta"]):
        for prefix in TRADEOFF_METHODS:
            row: dict[str, Any] = {"study": study, "beta": beta, "method": prefix}
            for name, column in (("width", f"{prefix}_weighted_width_mean"), ("member", f"{prefix}_membership_all"),
                                 ("wrong", f"{prefix}_wrong_sign_all")):
                per_replication = group.groupby("replication")[column].mean()
                means = [per_replication.sample(len(per_replication), replace=True, random_state=rng).mean()
                         for _ in range(TRADEOFF_BOOTSTRAP_REPLICATES)]
                row[name] = per_replication.mean()
                row[f"{name}_lo"], row[f"{name}_hi"] = np.percentile(means, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def tradeoff_results_latex(summary: pd.DataFrame) -> str:
    """Set the output of :func:`tradeoff_table` as the paper's Experiment III results table.

    One row per study and strength: the mean width of Naturalness, the Balke-Pearl arm and Manski, then the
    reference membership and wrong-sign rate of Naturalness and the Balke-Pearl arm.  Only the tabular block is
    emitted; the caption and label belong to the paper.

    Args:
        summary: Output of :func:`tradeoff_table`.

    Returns:
        LaTeX tabular source.
    """

    values = summary.set_index(["study", "beta", "method"])
    lines = [
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Mean width} & \multicolumn{2}{c}{Membership} & \multicolumn{2}{c}{Wrong sign} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(l){8-9}",
        r"Study & $\beta$ & Nat. & BP & Manski & Nat. & BP & Nat. & BP \\",
        r"\midrule",
    ]
    for study, beta in summary[["study", "beta"]].drop_duplicates().itertuples(index=False):

        def cell(method: str, name: str, digits: int) -> str:
            """Format one method's summary value for this study and strength."""

            return f"{values.loc[(study, beta, method), name]:.{digits}f}"

        lines.append(" & ".join([
            rf"\emph{{{study}}}", f"{beta:g}",
            cell("naturalness", "width", 2), cell("balke_pearl", "width", 2), cell("manski", "width", 2),
            cell("naturalness", "member", 2), cell("balke_pearl", "member", 2),
            cell("naturalness", "wrong", 3), cell("balke_pearl", "wrong", 3),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _tradeoff_figure(plt: Any, summary: pd.DataFrame, stem: Path) -> None:
    """Draw the width-membership trade-off of Naturalness and the Balke-Pearl arm, one column per study.

    Top row: each method's mean weighted CATE width against its reference membership over all draws, one marker per
    strength, joined in order of strength, with only the outer strengths labelled.  Bottom row: the wrong-sign rate
    over all draws against strength.  Bars are the bootstrap intervals of :func:`tradeoff_table`.  As the paper's
    Experiment III figure, it is drawn at Exp25's ``PAPER_FIGURE_WIDTH`` with its ``PAPER_RC`` typography, laid out
    by constrained layout and saved cropped to its content by Exp25's ``_save_figure``.

    Args:
        plt: The imported ``matplotlib.pyplot``.
        summary: Output of :func:`tradeoff_table`.
        stem: Output path without an extension; one file per :data:`FIGURE_FORMATS`.
    """

    style = _plot_style()
    studies = sorted(summary.study.unique())
    with plt.rc_context(style.PAPER_RC):
        # One grid keeps the columns of both rows aligned; each row's x-label sits under its middle panel only,
        # which centres it on the row.
        figure, axes = plt.subplots(2, len(studies), figsize=(style.PAPER_FIGURE_WIDTH, 4.8), layout="constrained",
                                    sharey="row", gridspec_kw={"height_ratios": [1.35, 1]}, squeeze=False)
        for column, study in enumerate(studies):
            top, bottom = axes[0, column], axes[1, column]
            for prefix in TRADEOFF_PLOTTED:
                label, colour, marker = TRADEOFF_METHODS[prefix]
                data = summary.loc[summary.study.eq(study) & summary.method.eq(prefix)].sort_values("beta")
                top.errorbar(data.width, data.member, xerr=[data.width - data.width_lo, data.width_hi - data.width],
                             yerr=[data.member - data.member_lo, data.member_hi - data.member], color=colour,
                             marker=marker, markersize=5, linewidth=1.4, elinewidth=0.8, capsize=0, label=label,
                             zorder=3)
                # Only the path's ends are labelled; the line joins the strengths in order between them.
                # Naturalness labels sit below its points, Balke-Pearl's above its first point and below its last,
                # clear of the other path.
                for _, point in data.loc[data.beta.isin([BETAS[0], BETAS[-1]])].iterrows():
                    if prefix == "naturalness":
                        offset = (-18, -12)
                    else:
                        offset = (-6, 5) if point.beta == BETAS[0] else (3, -11)
                    top.annotate(f"β={point.beta:g}", (point.width, point.member), textcoords="offset points",
                                 xytext=offset, fontsize="small", color=colour, fontweight="bold")
                bottom.errorbar(data.beta, data.wrong, yerr=[data.wrong - data.wrong_lo, data.wrong_hi - data.wrong],
                                color=colour, marker=marker, markersize=5, linewidth=1.4, elinewidth=0.8, label=label)
            top.set_title(study, fontweight="bold")
            top.set_xlim(0.25, 1.0)
            # Headroom for the labels above the points at full membership.
            top.set_ylim(0.4, 1.05)
            top.grid(alpha=0.3)
            bottom.set_xticks(BETAS)
            bottom.set_yticks([0.0, 0.04, 0.08, 0.12])
            bottom.set_ylim(-0.005, 0.14)
            bottom.grid(alpha=0.3)
            if column == 0:
                top.set_ylabel("Reference membership (all draws)")
                bottom.set_ylabel("Wrong-sign rate (all draws)")
        middle = len(studies) // 2
        axes[0, middle].set_xlabel("Mean weighted CATE width")
        axes[1, middle].set_xlabel("β")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=len(TRADEOFF_PLOTTED), frameon=False)
        style._save_figure(figure, stem, FIGURE_FORMATS)


def _width_by_beta_figure(plt: Any, table: pd.DataFrame, stem: Path, *, by_pattern: bool) -> None:
    """Draw the target-mass-weighted CATE width against selection strength, per study.

    Naturalness is the purple line: markers at the mean posterior-mean width with replication-bootstrap error bars,
    and a band over the average posterior :data:`POSTERIOR_WIDTH_LEVEL` width interval.  The Balke-Pearl arm, when
    present, is drawn the same way in Exp25's Balke-Pearl orange, and Manski is the slate line; Exp25's plug-in
    intervals are dashed lines with hollow markers.  Series sit at the strengths without a dodge, on a strength
    axis trimmed to just past the outer strengths.

    Args:
        plt: The imported ``matplotlib.pyplot``.
        table: Output of :func:`width_by_beta_table`.
        stem: Output path without an extension.
        by_pattern: Whether ``table`` has one row per selection covariate, each drawn as a row of panels.
    """

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    style = _plot_style()
    colors, markers = style.MODEL_COLORS, style.MODEL_MARKERS
    with_bp = bool(table.method.eq(BALKE_PEARL).any())
    # (method, legend label, colour, marker, line style, filled marker)
    specs = ((NATURALNESS, "Naturalness", colors["naturalness"], markers["naturalness"], "-", True),
             (EXP25_PLUGIN, "Exp. I plug-in Naturalness", colors["naturalness"], markers["naturalness"], "--", False),
             (BALKE_PEARL, "Balke-Pearl", colors["balke_pearl"], markers["balke_pearl"], "-", True),
             (EXP25_PLUGIN_BP, "Exp. I plug-in Balke-Pearl", colors["balke_pearl"], markers["balke_pearl"], "--", False),
             (MANSKI, "Manski", colors["manski"], markers["manski"], "-", True))
    rows = PATTERNS if by_pattern else (None,)
    # The legend band and the shared y label need a taller figure for one row.
    studies = _report_studies(table)
    figure, axes = style._panel_grid(len(rows), len(studies), width=4.6, height=3.4 if by_pattern else 4.8,
                                     sharex=True, sharey=True)
    for row, pattern in enumerate(rows):
        for col, study in enumerate(studies):
            axis = axes[row][col]
            panel = table.loc[table.study.eq(study)]
            if by_pattern:
                panel = panel.loc[panel.pattern.eq(pattern)]
            for method, _, color, marker, linestyle, filled in specs:
                line = panel.loc[panel.method.eq(method)].sort_values("beta")
                if line.empty:
                    continue
                x = line.beta.to_numpy(dtype=float)
                if line.posterior_lower.notna().any():
                    axis.fill_between(x, line.posterior_lower, line.posterior_upper, color=color, alpha=0.13,
                                      linewidth=0.0)
                # A percentile interval can sit just off the full-sample mean.
                error = [np.maximum(line["mean"] - line.ci_lower, 0.0), np.maximum(line.ci_upper - line["mean"], 0.0)]
                # Unclipped, so that the markers at the outer strengths are whole on the trimmed axis.
                axis.errorbar(x, line["mean"], yerr=error, color=color, ecolor=color, linestyle=linestyle,
                              linewidth=1.4, marker=marker, markersize=7, markerfacecolor=color if filled else "white",
                              markeredgecolor="white" if filled else color, markeredgewidth=0.8 if filled else 1.4,
                              elinewidth=1.1, capsize=2.5, zorder=3, clip_on=False)
            axis.set_ylim(0.0, 1.05)
            _beta_axis(axis)
            # Undodged series need no room beyond the outer strengths.
            axis.set_xlim(BETAS[0] - 0.02, BETAS[-1] + 0.02)
    _facet_headers(axes, columns=studies, rows=[f"selection on {pattern}" for pattern in PATTERNS] if by_pattern else None)
    present = set(table.method)
    handles = [
        Line2D([], [], color=color, marker=marker, linestyle=linestyle, label=label,
               markerfacecolor=color if filled else "white")
        for method, label, color, marker, linestyle, filled in specs if method in present
    ]
    band = f"posterior {POSTERIOR_WIDTH_LEVEL:.0%} interval"
    if with_bp:
        handles.append(Patch(facecolor="0.5", alpha=0.25, label=f"{band.capitalize()} (shaded)"))
    else:
        handles.insert(1, Patch(facecolor=colors["naturalness"], alpha=0.13, label=f"Naturalness {band}"))
    # Six bold entries in three columns do not fit two panels at Exp25's size.
    legend_style = dict(style.FRONTIER_LEGEND_TEXT_STYLE, **({"size": 11} if len(handles) > 4 else {}))
    style._finalize_shared_axes_figure(
        figure, axes, handles, xlabel=style.BOLD_BETA_SYMBOL, ylabel="Weighted CATE width",
        legend_text_style=legend_style,
    )
    style._save_figure(figure, stem, FIGURE_FORMATS)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 3) -> str:
    """Render selected columns as a GitHub-flavoured Markdown table."""

    def cell(value: Any) -> str:
        """Format one table cell."""

        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row[column]) for column in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join(lines)


def report_results(
    *,
    primary_root: Path | Sequence[Path],
    output_dir: Path,
    pilot_root: Path | None = None,
    loso_root: Path | Sequence[Path] | None = None,
    balke_pearl_root: Path | Sequence[Path] | None = None,
    exp25_intervals: Path | None = None,
    exp25_bp_intervals: Path | None = None,
) -> dict[str, Any]:
    """Write the discovery-first Exp31 report over finalized arms.

    Reads only finalized, validated roots into a fresh directory and reports every condition of the primary run, in
    order: restriction discovery, grounding, feasibility, bounds and reference scoring, the Monte Carlo and
    leave-one-out sensitivities, the Exp25 bridge, the family classification and the complete condition table.  It
    states claim limits and applies no feasibility threshold.  The Balke-Pearl arm adds its widths, paired
    comparisons and reference scores beside Naturalness's but never enters the family classification.  Several
    primary roots of one design over disjoint conditions are reported as one run (:func:`_read_combined`), each with
    its own Balke-Pearl root when any is given.

    Args:
        primary_root: The finalized primary run, or several runs of one design.
        output_dir: A fresh report directory.
        pilot_root: The finalized pilot run, for the Monte Carlo sensitivity.
        loso_root: The finalized leave-one-source-site-out run or runs (:data:`LOSO_ARMS`), pooled.
        balke_pearl_root: The finalized Balke-Pearl root built on each primary run.
        exp25_intervals: Exp25's saved Naturalness intervals, for the bridge.
        exp25_bp_intervals: Exp25's saved Balke-Pearl intervals (its ``identification_bounds`` leaf), drawn in the
            width figures.

    Returns:
        The report summary written to ``report_summary.json``.

    Raises:
        FileExistsError: If ``output_dir`` is not empty.
        ValueError: If an input root is not a finalized run of its arm, combined runs differ in design or overlap, a
            Balke-Pearl root was built on another run, or saved random sets do not reproduce their run's metrics.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PDF output subsets its fonts through fontTools, which logs every table at INFO.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Report directory is not empty: {output_dir}")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary_runs = _read_combined(_as_roots(primary_root), ("primary", "modular", "predictive"))
    primary_identity = primary_runs[0][1]
    primary_metrics = pd.concat([metrics for _, _, metrics in primary_runs], ignore_index=True)
    tables = []
    for root, identity, metrics in primary_runs:
        table = condition_table(metrics)
        if len(table) != len(identity["conditions"]):
            raise ValueError(f"Primary metrics cover {len(table)} of {len(identity['conditions'])} conditions")
        tables.append(_merge_posterior_widths(table, root, identity, NATURALNESS))
    conditions = pd.concat(tables, ignore_index=True)
    run_inputs = [{"root": str(root), "fingerprint": identity["fingerprint"]} for root, identity, _ in primary_runs]
    inputs: dict[str, Any] = {"primary": run_inputs[0] if len(run_inputs) == 1 else run_inputs}
    bp_roots = _as_roots(balke_pearl_root)
    if bp_roots:
        runs_by_fingerprint = {identity["fingerprint"]: (root, identity) for root, identity, _ in primary_runs}
        bp_tables, bp_inputs = [], []
        for bp_path in bp_roots:
            bp_identity, bp_metrics = _read_finalized(bp_path, "balke_pearl")
            source = bp_identity["source_run"]["fingerprint"]
            if source not in runs_by_fingerprint or any(entry["source"] == source for entry in bp_inputs):
                raise ValueError(f"{bp_path} was built on another run than {', '.join(str(r) for r, _, _ in primary_runs)}")
            run_root, run_identity = runs_by_fingerprint[source]
            bp_conditions = balke_pearl_condition_table(bp_metrics)
            if len(bp_conditions) != len(run_identity["conditions"]):
                raise ValueError(f"Balke-Pearl metrics cover {len(bp_conditions)} of {len(run_identity['conditions'])} "
                                 "conditions")
            bp_tables.append(_merge_posterior_widths(bp_conditions, bp_path, bp_identity, BALKE_PEARL))
            bp_inputs.append({"root": str(bp_path), "fingerprint": bp_identity["fingerprint"], "source": source,
                              "relevance_tau": float(bp_identity["protocol"]["relevance_tau"])})
        if len(bp_inputs) != len(primary_runs):
            raise ValueError("Every combined run needs its own Balke-Pearl root")
        if len({entry["relevance_tau"] for entry in bp_inputs}) != 1:
            raise ValueError("The Balke-Pearl roots use different relevance thresholds")
        conditions = conditions.merge(pd.concat(bp_tables, ignore_index=True), on=CONDITION_KEYS, how="left",
                                      validate="one_to_one")
        for entry in bp_inputs:
            del entry["source"]
        inputs["balke_pearl"] = bp_inputs[0] if len(bp_inputs) == 1 else bp_inputs
    families = family_table(conditions)
    candidates = _candidate_family_table(primary_metrics)
    imputation_shifts = None
    decision = None
    if pilot_root is not None:
        identity, _ = _read_finalized(pilot_root, "pilot")
        decision = json.loads((Path(pilot_root) / "imputation_decision.json").read_text(encoding="utf-8"))
        imputation_shifts = pd.DataFrame(decision["per_condition_maximum_shift"])
        inputs["pilot"] = {"root": str(pilot_root), "fingerprint": identity["fingerprint"]}
    loso = None
    loso_roots = _as_roots(loso_root)
    if loso_roots:
        loso_runs = _read_combined(loso_roots, sorted(LOSO_ARMS))
        loso_metrics = pd.concat([metrics for _, _, metrics in loso_runs], ignore_index=True)
        loso = loso_metrics.loc[loso_metrics.metric.eq("max_abs_target_tolerance_shift")].rename(
            columns={"value": "max_abs_target_tolerance_shift"})
        loso_table = loso_metrics.loc[loso_metrics.scope.eq("loso")].pivot_table(
            index=[*CONDITION_KEYS, "candidate", "estimand_key"], columns="metric", values="value", aggfunc="first"
        ).reset_index().rename(columns={"estimand_key": "tolerance_kind"})
        loso_table.to_csv(output_dir / "loso_grounding_diagnostics.csv", index=False)
        loso_inputs = [{"root": str(root), "fingerprint": identity["fingerprint"]} for root, identity, _ in loso_runs]
        inputs["loso"] = loso_inputs[0] if len(loso_inputs) == 1 else loso_inputs
    classified = classify_families(families, conditions, imputation_shifts, loso)

    conditions.to_csv(output_dir / "complete_condition_table.csv", index=False)
    families.to_csv(output_dir / "family_summary.csv", index=False)
    # Cell results beside the mass-weighted ones: every LP stratum and the site average, per method.
    cells = primary_metrics.loc[
        primary_metrics.scope.isin(["bounds", "paired", "reference"]) & primary_metrics.estimand_type.isin(["cate", "ate"])
    ]
    summarize_by_family(cells).to_csv(output_dir / "cell_results_by_family.csv", index=False)
    candidates.to_csv(output_dir / "discovery_and_grounding_by_candidate.csv", index=False)
    classified.to_csv(output_dir / "family_classification.csv", index=False)
    if imputation_shifts is not None:
        imputation_shifts.to_csv(output_dir / "imputation_sensitivity.csv", index=False)
    bridge = None
    if exp25_intervals is not None:
        bridge = _exp25_bridge(exp25_intervals, families)
        bridge.to_csv(output_dir / "exp25_descriptive_bridge.csv", index=False)
        inputs["exp25_intervals"] = {"path": str(exp25_intervals), "sha256": _sha256(Path(exp25_intervals))}
    if exp25_bp_intervals is not None:
        inputs["exp25_bp_intervals"] = {"path": str(exp25_bp_intervals), "sha256": _sha256(Path(exp25_bp_intervals))}
    comparison = None
    if bp_roots:
        comparison = balke_pearl_comparison_table(families)
        comparison.to_csv(output_dir / "balke_pearl_comparison_by_family.csv", index=False)

    # Figures in reporting order, discovery and grounding before any bound; report.md prints their captions.
    captions = dict(FIGURE_CAPTIONS)
    discovery = candidates.copy()
    discovery["candidate_label"] = discovery.candidate.str.replace("resp_", "").str.replace("_bin", "")
    labels = [value.replace("resp_", "").replace("_bin", "") for value in COVARIATES]
    for index, pattern in enumerate(PATTERNS):
        name = f"01{chr(97 + index)}_discovery_{pattern}"
        _line_grid(
            plt, discovery.loc[discovery.pattern.eq(pattern)], series_column="candidate_label", series_values=labels,
            series_prefix="candidate ",
            y_columns=[("p_eligible", "P(eligible), all draws"), ("p_triggered", "P(triggered), all draws")],
            stem=figures / name, ylim=(0.0, 1.02),
        )
        captions[name] = (f"Restriction discovery under OSRCT selection on {pattern}: posterior probability over all "
                          "draws that each candidate is triggered, and that it is eligible (triggered, with a "
                          "converged tolerance surface and the target's statistics inside source support).")
    pooled_patterns = discovery.groupby(["study", "beta", "candidate_label"], as_index=False).mean(numeric_only=True)
    for index, kind in enumerate(("two_c", "two_d")):
        _line_grid(
            plt, pooled_patterns, series_column="candidate_label", series_values=labels, series_prefix="candidate ",
            y_columns=[(f"grounded_{kind}_q50", f"grounded {kind.replace('two_', '2')}, eligible draws")],
            stem=figures / f"02{chr(97 + index)}_grounded_{kind}", ylim=(0.0, 1.02),
            range_columns={f"grounded_{kind}_q50": (f"grounded_{kind}_q025", f"grounded_{kind}_q975")},
        )
    _feasibility_figure(plt, families, figures / "03_feasibility_fallback_empty")
    states = conditions[["n_fallback_draws", "n_eligible_nonempty_draws", "n_eligible_empty_draws"]].sum(axis=1)
    per_bar = states.groupby([conditions[key] for key in FAMILY_KEYS]).sum()
    counts = f"{int(per_bar.min()):,}" if per_bar.min() == per_bar.max() else f"{int(per_bar.min()):,}-{int(per_bar.max()):,}"
    captions["03_feasibility_fallback_empty"] += (
        f" Draws per bar: {counts}; solver-failure draws, left out of the bars: {int(conditions.n_solver_failure_draws.sum()):,}."
    )
    exp25_widths = None if exp25_intervals is None else _exp25_condition_widths(exp25_intervals)
    exp25_bp_widths = None if exp25_bp_intervals is None else _exp25_condition_widths(exp25_bp_intervals, "balke_pearl")
    for by_pattern, name, table_name in ((False, "04a_width_by_beta", "width_by_beta.csv"),
                                         (True, "04b_width_by_beta_by_pattern", "width_by_beta_by_pattern.csv")):
        widths = width_by_beta_table(conditions, exp25_widths, by_pattern=by_pattern, exp25_bp=exp25_bp_widths)
        widths.to_csv(output_dir / table_name, index=False)
        _width_by_beta_figure(plt, widths, figures / name, by_pattern=by_pattern)
    if bp_roots:
        captions["04a_width_by_beta"] += BALKE_PEARL_WIDTH_CAPTION
        _line_grid(
            plt, families, series_column="pattern", series_values=PATTERNS, series_prefix="selection on ",
            y_columns=[("naturalness_narrower_than_bp_given_joint", "P(Naturalness narrower than\nBalke-Pearl), joint draws"),
                       ("naturalness_vs_bp_width_gain_mean", "mean width gain over\nBalke-Pearl, joint draws")],
            stem=figures / "04d_naturalness_vs_balke_pearl",
        )
        _line_grid(
            plt, _reference_by_method(families), series_column="method", series_values=[NATURALNESS, BALKE_PEARL, MANSKI],
            series_prefix="", y_columns=[("membership_all", "reference membership,\nall draws"),
                                         ("wrong_sign_all", "wrong sign, all draws")],
            stem=figures / "05b_reference_by_method", series_style=_method_series_style(),
        )
        tradeoff = tradeoff_table(conditions)
        tradeoff.to_csv(output_dir / "width_membership_tradeoff.csv", index=False)
        (output_dir / "width_membership_tradeoff_table.tex").write_text(tradeoff_results_latex(tradeoff),
                                                                        encoding="utf-8")
        _tradeoff_figure(plt, tradeoff, figures / "05c_width_membership_tradeoff")
    _line_grid(
        plt, families, series_column="pattern", series_values=PATTERNS, series_prefix="selection on ",
        y_columns=[("p_width_gain_positive_all", "P(narrower than Manski),\nall draws"),
                   ("weighted_width_gain_mean", "mean width gain,\njoint nonempty draws")],
        stem=figures / "04c_paired_width_gain",
    )
    scoring = families.assign(membership_drop=families.manski_membership_all - families.naturalness_membership_all)
    _line_grid(
        plt, scoring, series_column="pattern", series_values=PATTERNS, series_prefix="selection on ",
        y_columns=[("naturalness_membership_all", "Naturalness membership, all draws"),
                   ("membership_drop", "Manski - Naturalness membership"),
                   ("naturalness_wrong_sign_all", "Naturalness wrong sign, all draws")],
        stem=figures / "05_reference_membership",
        reference_lines={"membership_drop": CLASSIFICATION_RULES["material_membership_drop"],
                         "naturalness_wrong_sign_all": CLASSIFICATION_RULES["material_wrong_sign_increase"]},
    )
    if imputation_shifts is not None:
        shifts = imputation_shifts.assign(condition=[
            f"{row.study}/{row.pattern}/b={_format_beta(row.beta)}/r{int(row.replication)}" for row in imputation_shifts.itertuples()
        ])
        _bar_panel(plt, shifts, label_column="condition", value_column="maximum_shift", series_column=None,
                   ylabel="max weighted endpoint shift", stem=figures / "07_imputation_sensitivity",
                   threshold=IMPUTATION_SHIFT_THRESHOLD)
    if loso_roots:
        table = loso_table.assign(condition=[
            f"{row.study}/{row.pattern}/b={_format_beta(row.beta)}/{row.tolerance_kind}" for row in loso_table.itertuples()
        ])
        table["candidate"] = table.candidate.str.replace("resp_", "").str.replace("_bin", "")
        _bar_panel(plt, table, label_column="condition", value_column="mean_validity_probability", series_column="candidate",
                   ylabel="mean validity probability", stem=figures / "08_loso_validity", series_prefix="candidate ")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "generated_at_utc": _utc_now(),
        "arm": primary_identity["arm"],
        "evidence_min_randomized_rows": int(primary_identity["settings"].get("evidence_min_randomized_rows", 0)),
        "trigger_taus": [float(value) for value in
                         primary_identity.get("protocol", {}).get("trigger_rule", {}).get("taus", TRIGGER_TAUS)],
        "regrid_tau_floor": primary_identity.get("regrid", {}).get("tau_floor"),
        "inputs": inputs,
        "n_conditions": int(len(conditions)),
        "n_families": int(len(families)),
        "labels": classified.label.value_counts().to_dict(),
        "pilot_decision": decision,
        "classification_rules": CLASSIFICATION_RULES,
        "claim_limit": protocol_payload()["claim_limit"],
    }
    _write_json(summary, output_dir / "report_summary.json")
    _write_report_markdown(output_dir, summary, classified, conditions, bridge, decision, captions, comparison)
    write_run_manifest(
        output_dir,
        experiment_name=EXPERIMENT_NAME,
        script_path="experiments/exp31_bayesian_manylabs_naturalness.py",
        config={"stage": "report", "inputs": inputs},
        argv=sys.argv,
        extra={"git": _git_provenance(), "source_fingerprints": _source_fingerprints(), "schema_version": SCHEMA_VERSION},
    )
    _write_json(
        {"status": "complete", "completed_at_utc": _utc_now(), "artifact_sha256": _artifact_hashes(output_dir)},
        output_dir / "completion.json",
    )
    return summary


def _write_report_markdown(
    output_dir: Path,
    summary: Mapping[str, Any],
    classified: pd.DataFrame,
    conditions: pd.DataFrame,
    bridge: pd.DataFrame | None,
    decision: Mapping[str, Any] | None,
    captions: Mapping[str, str],
    comparison: pd.DataFrame | None = None,
) -> None:
    """Write ``report.md``: figures and tables in reporting order.

    ``captions`` maps a figure's file stem to the caption printed beneath it (the figures carry no titles).
    ``comparison`` is the Balke-Pearl arm's family table (:func:`balke_pearl_comparison_table`), when it was run.
    """

    figures = sorted(path.name for path in (output_dir / "figures").glob("*.png"))

    def embed(prefix: str) -> str:
        """Embed every figure whose name starts with ``prefix``, each with its caption."""

        blocks = [
            f"![{name}](figures/{name})" + (f"\n\n*{captions[Path(name).stem]}*" if Path(name).stem in captions else "")
            for name in figures if name.startswith(prefix)
        ]
        return "\n\n".join(blocks) or "_Not produced for these inputs._"

    lines = [
        "# Exp31: Bayesian Naturalness grounding and bounds on ManyLabs",
        "",
        f"Generated {summary['generated_at_utc']} over {summary['n_conditions']} conditions in {summary['n_families']} "
        "study-pattern-beta families.",
        "",
        *([
            "**Arm: modular (plan Section 16.8, post hoc).** The source evidence comes from one full-data TabICLv2 fit "
            "per site, the target draws are the bootstrap draws recentred on the target's full-data fit, one envelope "
            "is fitted per candidate and kind, and there is no source-support gate. Source-PMF uncertainty does not "
            "reach the evidence (a cut posterior).",
            "",
        ] if summary.get("arm") == "modular" else []),
        *([
            "**Arm: predictive (plan Section 16.8, post hoc).** Module A draws every source imputation and target draw "
            "by predictive resampling: each site's data are simulated from its full-data TabICLv2 fit and the models "
            "refitted. The evidence, one envelope per imputation, candidate and kind, and the random identified sets "
            "are the primary arm's, with no source-support gate. Source- and target-PMF uncertainty both propagate."
            + (f" Module B's evidence ranges only over stratum pairs whose two strata hold at least "
               f"{summary['evidence_min_randomized_rows']} randomized rows in each arm."
               if summary.get("evidence_min_randomized_rows") else ""),
            "",
        ] if summary.get("arm") == "predictive" else []),
        *([
            f"**Trigger grid tau >= {summary['regrid_tau_floor']:g} (post hoc regrid).** The run's Modules D and E were "
            "replayed with the trigger grid's taus restricted to "
            + ", ".join(f"{value:g}" for value in summary["trigger_taus"])
            + ", from its saved tolerance draws; nothing was refitted, and the surfaces are fitted to every source "
            "site whatever the grid.",
            "",
        ] if summary.get("regrid_tau_floor") is not None else []),
        "**Claim limits.** " + str(summary["claim_limit"]).capitalize() + ". The outer envelope "
        "[Q0.025(L), Q0.975(U)] is a posterior summary of a random identified set, not a confidence or credible "
        "interval. There is no LP-feasibility gate; every probability states its denominator. The randomized "
        "reference is the full complete-case pi RCT, about half of whose participants are also target idle rows.",
        "",
        "## 1. Restriction discovery",
        "",
        embed("01"),
        "",
        "Per-candidate values: `discovery_and_grounding_by_candidate.csv`. No prespecified invalid or placebo "
        "candidate exists in the ManyLabs candidate library, so no rejection rate is reported.",
        "",
        "## 2. Grounding",
        "",
        embed("02"),
        "",
        "Sampler diagnostics per surface are in each condition's `tolerance_surface_diagnostics.parquet`; "
        "leave-one-source-site-out scores, when run, in `loso_grounding_diagnostics.csv`.",
        "",
        embed("08"),
        "",
        "## 3. Feasibility, fallback and empty sets",
        "",
        embed("03"),
        "",
        "## 4. Bounds: Naturalness against matched Bayesian Manski",
        "",
        embed("04"),
        "",
        "Widths with both intervals, the posterior one and the replication bootstrap: `width_by_beta.csv` "
        "(selection covariates pooled) and `width_by_beta_by_pattern.csv`. "
        "Cell results, every LP stratum and the site average for both methods with pooled denominators: "
        "`cell_results_by_family.csv`. Target mass outside the LP cells and LP mass without reference support are "
        "reported per condition in `complete_condition_table.csv`.",
        "",
    ]
    if comparison is not None:
        pooled = balke_pearl_comparison_table(comparison, keys=("study", "beta"))
        # Combined runs list one Balke-Pearl root each, all with the same threshold.
        bp_inputs = summary["inputs"]["balke_pearl"]
        relevance_tau = (bp_inputs[0] if isinstance(bp_inputs, list) else bp_inputs)["relevance_tau"]
        lines += [
            "**Post hoc Bayesian Balke-Pearl (plan Section 16.7).** Added after the primary outcomes, on the same "
            "posterior draws: in each draw every covariate with treatment relevance "
            f"D_j >= {relevance_tau:g} "
            "instruments its blocks, a cell no instrument reaches takes the matched Manski interval, and a refuted "
            "instrument leaves an empty set. It does not enter the family classification. Means over replications, "
            "selection covariates pooled (per family: `balke_pearl_comparison_by_family.csv`):",
            "",
            _markdown_table(pooled, ["study", "beta", "naturalness_weighted_width_mean", "balke_pearl_weighted_width_mean",
                                     "naturalness_narrower_than_bp_given_joint", "naturalness_membership_all",
                                     "balke_pearl_membership_all", "balke_pearl_p_nonempty"]),
            "",
        ]
    lines += [
        "## 5. Randomized-reference scoring",
        "",
        embed("05"),
        "",
        "## 6. Sensitivity",
        "",
        "Module A is a stratified Bayesian bootstrap of TabICLv2 fits, which has no prior concentration, so there "
        "is no Dirichlet-prior sensitivity. The Monte Carlo check below covers the pilot conditions and the "
        "leave-one-out check (Section 2) the same subset.",
        "",
        embed("07"),
        "",
    ]
    if decision is not None:
        lines += [
            f"Pilot decision: {decision['selected_num_imputations']} source imputations "
            f"(largest 10-vs-20 shift {decision['maximum_shift']:.4f}, threshold {decision['threshold']}).",
            "",
        ]
    lines += ["## 7. Exp25 descriptive bridge", ""]
    if bridge is not None:
        lines += [
            "Descriptive only: different OSRCT seeds, a plug-in TabICLv2 estimator and point containment of the "
            "Wald estimate in Exp25, posterior reference membership in Exp31.",
            "",
            _markdown_table(bridge, [*FAMILY_KEYS, "exp25_valid_share", "exp25_weighted_width_valid",
                                     "exp25_weighted_point_containment", "exp31_p_any_eligible",
                                     "exp31_naturalness_weighted_width_mean", "exp31_naturalness_membership_all"]),
            "",
        ]
    else:
        lines += ["_No Exp25 intervals were supplied._", ""]
    lines += [
        "## 8. Family classification (prespecified rules)",
        "",
        "Rules: " + ", ".join(f"`{key}` = {value}" for key, value in CLASSIFICATION_RULES.items()) + ".",
        "",
        _markdown_table(classified, [*FAMILY_KEYS, "label", "p_any_eligible", "discovery_recurrence_share",
                                     "p_width_gain_positive_all", "membership_drop", "wrong_sign_increase",
                                     "pooled_p_empty_given_eligible", "max_imputation_endpoint_shift",
                                     "max_loso_target_tolerance_shift"]),
        "",
        "## 9. Complete condition table",
        "",
        f"`complete_condition_table.csv` has all {len(conditions)} conditions, negative results included.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# --- Command line ---------------------------------------------------------------


def _default_imputation_counts(arm: str, requested: Sequence[int] | None) -> tuple[int, ...]:
    """Return an arm's imputation counts, honouring an explicit request."""

    if requested:
        return tuple(sorted(set(int(value) for value in requested)))
    if arm == "pilot":
        return PILOT_IMPUTATION_COUNTS
    if arm in LOSO_ARMS or arm == "modular":
        return (1,)
    if arm == "predictive":
        return (PREDICTIVE_SOURCE_DRAWS,)
    return (PRIMARY_IMPUTATIONS,)


def build_parser() -> argparse.ArgumentParser:
    """Build the prepare/draw-pmfs/adopt-pmfs/run/finalize/balke-pearl/regrid/report command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    default_rct = Path(Exp16Config().rct_path)

    prepare = commands.add_parser("prepare", help="Generate and fingerprint the Exp31 OSRCT grids")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--rct-path", type=Path, default=default_rct)
    prepare.add_argument("--replications", type=int, nargs="+")
    prepare.add_argument("--studies", nargs="+", choices=ALL_STUDIES, help="Studies to prepare (default: STUDIES)")

    pmfs = commands.add_parser("draw-pmfs", help="Draw one arm's TabICLv2 bootstrap PMFs (Module A, on a GPU)")
    pmfs.add_argument("--arm", choices=sorted(ARMS), required=True, help="Whose conditions to draw")
    pmfs.add_argument("--prepared-root", type=Path, required=True)
    pmfs.add_argument("--output-dir", type=Path, required=True, help="The PMF root, shared by every arm")
    pmfs.add_argument("--rct-path", type=Path, default=default_rct)
    pmfs.add_argument("--device", default="cuda", help="TabICLv2 device, 'cuda' on the cluster")
    pmfs.add_argument("--source-draws", type=int, default=SOURCE_PMF_DRAWS)
    pmfs.add_argument("--target-draws", type=int, default=TARGET_PMF_DRAWS)
    pmfs.add_argument("--shard-index", type=int, default=0)
    pmfs.add_argument("--num-shards", type=int, default=1)
    pmfs.add_argument("--cluster-manifest-path")
    pmfs.add_argument("--resume", action="store_true")
    pmfs.add_argument("--studies", nargs="+", choices=ALL_STUDIES, help="Studies of a full-grid arm (default: STUDIES)")
    pmfs.add_argument("--module-a-scheme", choices=MODULE_A_SCHEMES, default="bootstrap",
                      help="Module A's draws: the stratified bootstrap or predictive resampling")

    adopt = commands.add_parser("adopt-pmfs", help="Reuse another PMF root's draws of conditions whose inputs are unchanged")
    adopt.add_argument("--output-dir", type=Path, required=True, help="The PMF root to adopt into")
    adopt.add_argument("--prepared-root", type=Path, required=True, help="The prepared root of --output-dir")
    adopt.add_argument("--source-pmf-root", type=Path, required=True, help="The PMF root to adopt from")
    adopt.add_argument("--source-prepared-root", type=Path, required=True, help="The prepared root of --source-pmf-root")
    adopt.add_argument("--arm", choices=sorted(ARMS), default="primary", help="Whose conditions to consider")

    run = commands.add_parser("run", help="Run one arm's conditions and finalize its root")
    run.add_argument("--arm", choices=sorted(ARMS), required=True)
    run.add_argument("--prepared-root", type=Path, required=True)
    run.add_argument("--pmf-root", type=Path, required=True, help="The draw-pmfs root with every condition's draws")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--rct-path", type=Path, default=default_rct)
    run.add_argument("--pilot-decision", type=Path, help="imputation_decision.json of a finalized pilot run")
    run.add_argument("--imputation-count", type=int, nargs="+", dest="imputation_counts")
    run.add_argument("--replications", type=int, nargs="+", help="Replication subset of a full-grid arm")
    run.add_argument("--random-set-draws", type=int, default=RANDOM_SET_DRAWS)
    run.add_argument("--num-chains", type=int, default=4)
    run.add_argument("--num-warmup", type=int, default=2000)
    run.add_argument("--num-samples", type=int, default=3000)
    run.add_argument("--target-accept-prob", type=float, default=0.90)
    run.add_argument("--retry-warmup", type=int, default=4000)
    run.add_argument("--retry-target-accept-prob", type=float, default=0.95)
    run.add_argument("--retry-num-samples", type=int, default=6000)
    run.add_argument("--second-retry-warmup", type=int, default=8000)
    run.add_argument("--second-retry-num-samples", type=int, default=24000)
    run.add_argument("--second-retry-target-accept-prob", type=float, default=0.95)
    run.add_argument("--dense-mass", action="store_true", help="Adapt a full NUTS mass matrix (sampler efficiency only)")
    run.add_argument("--loso-score-draws", type=int, default=2000)
    run.add_argument("--studies", nargs="+", choices=ALL_STUDIES, help="Studies of a full-grid arm (default: STUDIES)")
    run.add_argument("--evidence-min-randomized-rows", type=int, default=0,
                     help="Evidence subset: randomized rows per arm both strata of a pair need (0 keeps all)")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--cluster-manifest-path")
    run.add_argument("--resume", action="store_true")

    finalize = commands.add_parser("finalize", help="Validate a run root and write its completion")
    finalize.add_argument("--output-dir", type=Path, required=True)

    bp = commands.add_parser("balke-pearl", help="Add the Bayesian Balke-Pearl arm to a finalized run")
    bp.add_argument("--source-root", type=Path, required=True, help="The finalized primary (or smoke) run")
    bp.add_argument("--prepared-root", type=Path, required=True, help="The source run's prepared root")
    bp.add_argument("--pmf-root", type=Path, required=True, help="The source run's draw-pmfs root")
    bp.add_argument("--rct-path", type=Path, default=default_rct)
    bp.add_argument("--output-dir", type=Path, required=True)
    bp.add_argument("--shard-index", type=int, default=0)
    bp.add_argument("--num-shards", type=int, default=1)
    bp.add_argument("--relevance-tau", type=float, default=BP_RELEVANCE_TAU,
                    help="Instrument relevance threshold on D_j (default: the trigger grid's smallest tau)")
    bp.add_argument("--cluster-manifest-path")
    bp.add_argument("--resume", action="store_true")

    regrid = commands.add_parser("regrid", help="Replay a finalized run on a restricted trigger grid")
    regrid.add_argument("--source-root", type=Path, required=True, help="The finalized run to replay")
    regrid.add_argument("--prepared-root", type=Path, required=True, help="The source run's prepared root")
    regrid.add_argument("--pmf-root", type=Path, required=True, help="The source run's draw-pmfs root")
    regrid.add_argument("--rct-path", type=Path, default=default_rct)
    regrid.add_argument("--output-dir", type=Path, required=True)
    regrid.add_argument("--tau-floor", type=float, required=True, help="Keep the trigger grid's taus at or above this")
    regrid.add_argument("--shard-index", type=int, default=0)
    regrid.add_argument("--num-shards", type=int, default=1)
    regrid.add_argument("--cluster-manifest-path")
    regrid.add_argument("--resume", action="store_true")

    report = commands.add_parser("report", help="Write the discovery-first report over finalized arms")
    report.add_argument("--primary-root", type=Path, nargs="+", required=True,
                        help="The finalized run, or several runs of one design over disjoint conditions")
    report.add_argument("--pilot-root", type=Path)
    report.add_argument("--loso-root", type=Path, nargs="+")
    report.add_argument("--balke-pearl-root", type=Path, nargs="+", help="The finalized balke-pearl root of each run")
    report.add_argument("--exp25-intervals", type=Path, help="Frozen Exp25 mc_intervals.parquet for the descriptive bridge")
    report.add_argument("--exp25-bp-intervals", type=Path,
                        help="Frozen Exp25 identification_bounds/mc_intervals.parquet, Balke-Pearl's plug-in intervals")
    report.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the Exp31 commands."""

    _configure_logging()
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_inputs(args.output_dir, rct_path=args.rct_path, replications=args.replications, studies=args.studies)
        return 0
    if args.command == "draw-pmfs":
        draw_pmfs(
            args.output_dir, args.prepared_root, args.rct_path, arm=args.arm, device=args.device,
            n_source_draws=args.source_draws, n_target_draws=args.target_draws, shard_index=args.shard_index,
            num_shards=args.num_shards, resume=args.resume, cluster_manifest_path=args.cluster_manifest_path,
            scheme=args.module_a_scheme, studies=args.studies,
        )
        return 0
    if args.command == "adopt-pmfs":
        record = adopt_pmfs(args.output_dir, args.prepared_root, source_root=args.source_pmf_root,
                            source_prepared_root=args.source_prepared_root, arm=args.arm)
        print(json.dumps(record["counts"], sort_keys=True))
        return 0
    if args.command == "finalize":
        arm = json.loads((Path(args.output_dir) / "identity.json").read_text(encoding="utf-8")).get("arm")
        finalize = finalize_balke_pearl if arm == "balke_pearl" else finalize_run
        LOGGER.info("Exp31 finalize: %s", finalize(args.output_dir)["status"])
        return 0
    if args.command == "balke-pearl":
        run_balke_pearl(
            args.output_dir, args.source_root, args.prepared_root, args.rct_path, args.pmf_root, resume=args.resume,
            shard_index=args.shard_index, num_shards=args.num_shards, cluster_manifest_path=args.cluster_manifest_path,
            relevance_tau=args.relevance_tau,
        )
        return 0
    if args.command == "regrid":
        run_regrid(
            args.output_dir, args.source_root, args.prepared_root, args.rct_path, args.pmf_root, tau_floor=args.tau_floor,
            resume=args.resume, shard_index=args.shard_index, num_shards=args.num_shards,
            cluster_manifest_path=args.cluster_manifest_path,
        )
        return 0
    if args.command == "report":
        report_results(
            primary_root=args.primary_root,
            output_dir=args.output_dir,
            pilot_root=args.pilot_root,
            loso_root=args.loso_root,
            balke_pearl_root=args.balke_pearl_root,
            exp25_intervals=args.exp25_intervals,
            exp25_bp_intervals=args.exp25_bp_intervals,
        )
        return 0
    # One XLA host device per chain lets NUTS run its chains in parallel; it
    # must be set before JAX initializes, and spawned workers inherit it.
    os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={args.num_chains}")
    settings = Exp31RunSettings(
        arm=args.arm,
        imputation_counts=_default_imputation_counts(args.arm, args.imputation_counts),
        replications=None if args.replications is None else tuple(args.replications),
        random_set_draws=args.random_set_draws,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        target_accept_prob=args.target_accept_prob,
        retry_warmup=args.retry_warmup,
        retry_target_accept_prob=args.retry_target_accept_prob,
        retry_num_samples=args.retry_num_samples,
        second_retry_warmup=args.second_retry_warmup,
        second_retry_num_samples=args.second_retry_num_samples,
        second_retry_target_accept_prob=args.second_retry_target_accept_prob,
        dense_mass=args.dense_mass,
        loso_score_draws=args.loso_score_draws,
        evidence_min_randomized_rows=args.evidence_min_randomized_rows,
        studies=None if args.studies is None else tuple(args.studies),
        cluster_manifest_path=args.cluster_manifest_path,
    )
    run_experiment(
        args.output_dir,
        args.prepared_root,
        args.rct_path,
        settings,
        pmf_root=args.pmf_root,
        resume=args.resume,
        pilot_decision=args.pilot_decision,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
