"""Experiment 28: method comparison on Twins under a withheld confounder.

Compares Naturalness with Manski, the B-Learner at ``Gamma=exp(beta)`` and at
the oracle-required ``Gamma``, CB-IPW, CB-DR and Balke-Pearl bounds on Twins
(US twin births 1989-1991; sites are the mother's place of birth, the heavier
twin is the revealed arm and one-year mortality the binary outcome). This is
the paper's Experiment II, run by ``paper/reproduce_experiment2_twins.sh``.

Methods observe only :data:`EXP28_COVARIATES`; assignment selects on the
withheld ``meduc6`` (mother's education), an unmeasured confounder (inside the
conditioning set it would point identify the target effect). Selection is
logistic in the standardised binary selector, so the within-cell odds ratio
``exp(beta / sqrt(p(1-p)))`` at prevalence ``p`` is at least ``exp(2*beta)``
and ``Gamma=exp(beta)`` understates the confounding. Covariates and outcomes
are real and the twin pair gives an exact reference, but selection is
simulated: the benchmark is semi-synthetic.
"""

from __future__ import annotations

from typing import Sequence

from experiments.naturalness_comparison import (
    ComparisonConfig,
    NATURALNESS_META_LEARNERS,
    NATURALNESS_PER_CELL_METHODS,
    main_from_defaults,
)


#: Reporting cells. These are the only covariates any method observes.
EXP28_COVARIATES = ("birmon", "gestat10", "mager8")

#: Withheld selector. Must stay outside EXP28_COVARIATES, otherwise the design
#: has no unmeasured confounding at all.
EXP28_BIAS_COVARIATE = "meduc6"

#: Selection strengths ``beta`` on the withheld selector.
EXP28_BETAS = (0., .25, .5, 1., 1.5, 2.)

#: Twins has a single release, so repetition comes from these seeds.
EXP28_SEEDS = (20260918, 20260919, 20260920)


def default_config() -> ComparisonConfig:
    """Return the default Exp28 configuration, used when no flags are given.

    The selector is listed in ``withheld_covariates``, so the run manifest
    records the withholding.

    Returns:
        The default :class:`ComparisonConfig` for the Twins benchmark.

    Raises:
        ValueError: If the selector has been edited into the conditioning set,
            which would silently remove the unmeasured confounding.
    """
    if EXP28_BIAS_COVARIATE in EXP28_COVARIATES:
        raise ValueError(
            "The Twins selector must stay outside the conditioning set; placing it inside "
            "point identifies the target effect and tests no partial-identification method")
    return ComparisonConfig(
        experiment_name="exp28_twins_method_comparison",
        datasets=("twins_reduced",),
        dgp_seeds=(20260918,),
        seeds=EXP28_SEEDS,
        betas=EXP28_BETAS,
        target_sites=("all",),
        twins_covariates=EXP28_COVARIATES,
        twins_bias_covariate=EXP28_BIAS_COVARIATE,
        withheld_covariates=(EXP28_BIAS_COVARIATE,),
        methods=NATURALNESS_PER_CELL_METHODS,
        meta_learners=NATURALNESS_META_LEARNERS,
        cb_source_regime="on",
        output_dir="results/exp28_twins_method_comparison",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp28 and return nonzero when any arm failed to fit."""
    return main_from_defaults(default_config(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
