"""Naturalness bounds on a toy, simulated multi-site dataset.

The simulation mimics the structure of the paper's experiments. Ten *source*
sites each hold a randomized trial (``F = "on"``) and an observational sample
(``F = "idle"``); one *target* site holds observational rows only. The goal is
the target's conditional average treatment effect
``CATE(z) = E[Y(1) - Y(0) | Z = z]`` for every stratum ``z = (Z1, Z2)``.

Data-generating process, per site ``s`` (all variables binary):

* ``Z1``, ``Z2`` and an *unobserved* ``U`` are independent, with site-specific
  prevalences drawn from ``Uniform(0.3, 0.7)``.
* Observational regime: ``P(X=1 | Z1, Z2, U) = expit(-1.5 + 3 Z1 + 0.5 Z2 + (2U - 1))``.
  Trial regime: ``X ~ Bernoulli(1/2)``.
* Outcome: ``P(Y=1 | X, Z2, U) = expit(-0.5 + a_s + X + Z2 - X Z2 + (2U - 1))``,
  with a site offset ``a_s ~ Normal(0, 0.25^2)``.

``U`` confounds treatment and outcome in the observational regime, so the target
CATE is not identified from the target's rows. ``Z1`` moves treatment but has no
direct effect on the outcome, which is the pattern the Naturalness condition is
built to detect, so its conditions should fire. ``Z2`` modifies the effect and
barely moves treatment; on some seeds it fires at the loosest thresholds
(``epsilon >= 0.2``, ``tau = 0.075``), where its learned tolerances are wide.

The script runs the method with the empirical Stage-1 backend (cell counts, no
model download) and the paper's relevance and naturalness grids:

1. Stages 1-3 on the source sites: ``NaturalnessGrounder`` measures which
   ``(covariate, epsilon, tau)`` conditions fire at each source and fits
   Gaussian-process surfaces to the tolerances ``c`` and ``d``.
2. ``TargetNaturalnessPolicy`` evaluates the same triggers on the target.
3. Tolerance draws from the surfaces are propagated through the target linear
   program; the Naturalness interval is the mean of the lower and upper
   endpoints over the draws whose program is optimal, as in the paper.
4. The intervals are compared with the true CATE, the Manski bounds and the
   Balke-Pearl bounds that treat ``Z1`` as a known valid instrument.

Run it from the repository root:

    python examples/toy_naturalness.py
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_grounding import (  # noqa: E402
    PAPER_EPSILONS,
    PAPER_TAUS,
    NaturalnessGrid,
    NaturalnessGrounder,
    TargetNaturalnessPolicy,
    estimate_target_observed_binary_quantities,
    solve_target_bounds_from_observed_binary_quantities,
)
from causal_grounding.balke_pearl import balke_pearl_interval  # noqa: E402

COVARIATES = ["Z1", "Z2"]
#: Naturalness thresholds ``epsilon`` and the relevance grid ``tau >= 0.075``
#: of the paper's experiments.
EPSILONS = list(PAPER_EPSILONS)
TAUS = list(PAPER_TAUS)


def _treatment_probability(z1: np.ndarray, z2: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return ``P(X=1 | Z1, Z2, U)`` in the observational regime."""
    return expit(-1.5 + 3.0 * z1 + 0.5 * z2 + (2 * u - 1))


def _outcome_probability(x, z2, u, offset: float):
    """Return ``P(Y=1 | X, Z2, U)`` at a site with outcome offset ``offset``."""
    return expit(-0.5 + offset + x + z2 - x * z2 + (2 * u - 1))


def _draw_site(rng: np.random.Generator) -> Dict[str, float]:
    """Draw one site's covariate prevalences and outcome offset."""
    return {
        "p_z1": rng.uniform(0.3, 0.7),
        "p_z2": rng.uniform(0.3, 0.7),
        "p_u": rng.uniform(0.3, 0.7),
        "offset": rng.normal(0.0, 0.25),
    }


def _sample_site(
    rng: np.random.Generator,
    site: Dict[str, float],
    n_idle: int,
    n_on: int,
) -> pd.DataFrame:
    """Sample ``n_idle`` observational and ``n_on`` randomized rows of one site.

    Args:
        rng: Random generator.
        site: Site parameters from :func:`_draw_site`.
        n_idle: Number of observational (``F="idle"``) rows.
        n_on: Number of randomized (``F="on"``) rows.

    Returns:
        Rows with columns ``Z1``, ``Z2``, ``X``, ``Y`` and ``F``. The
        confounder ``U`` is dropped: no method observes it.
    """
    n = n_idle + n_on
    z1 = rng.binomial(1, site["p_z1"], n)
    z2 = rng.binomial(1, site["p_z2"], n)
    u = rng.binomial(1, site["p_u"], n)
    regime = np.repeat(["idle", "on"], [n_idle, n_on])
    x = np.where(
        regime == "idle",
        rng.binomial(1, _treatment_probability(z1, z2, u)),
        rng.binomial(1, 0.5, n),
    )
    y = rng.binomial(1, _outcome_probability(x, z2, u, site["offset"]))
    return pd.DataFrame({"Z1": z1, "Z2": z2, "X": x, "Y": y, "F": regime})


def _true_cate(site: Dict[str, float]) -> pd.DataFrame:
    """Return the exact CATE of every ``(Z1, Z2)`` stratum at a site.

    ``U`` is independent of ``Z``, so ``CATE(z) = sum_u P(u) [P(Y=1 | X=1, z2, u)
    - P(Y=1 | X=0, z2, u)]``, which depends on ``z2`` only.
    """
    rows = []
    for z1 in (0, 1):
        for z2 in (0, 1):
            cate = sum(
                weight * (
                    _outcome_probability(1, z2, u, site["offset"])
                    - _outcome_probability(0, z2, u, site["offset"])
                )
                for u, weight in ((0, 1.0 - site["p_u"]), (1, site["p_u"]))
            )
            rows.append({"Z1": z1, "Z2": z2, "true_cate": float(cate)})
    return pd.DataFrame(rows)


def simulate_toy_data(
    seed: int = 0,
    n_sources: int = 10,
    n_idle: int = 2000,
    n_on: int = 2000,
    n_target: int = 4000,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Simulate the source sites, the target site and the target's true CATEs.

    Args:
        seed: Seed of the whole simulation.
        n_sources: Number of source sites.
        n_idle: Observational rows per source site.
        n_on: Randomized rows per source site.
        n_target: Observational rows at the target site.

    Returns:
        ``(sources, target, truth)``: a mapping from source-site name to its
        rows, the target's observational rows, and the target's true CATE by
        ``(Z1, Z2)`` stratum.
    """
    rng = np.random.default_rng(seed)
    sources = {
        f"site{k:02d}": _sample_site(rng, _draw_site(rng), n_idle, n_on)
        for k in range(n_sources)
    }
    target_site = _draw_site(rng)
    target = _sample_site(rng, target_site, n_target, 0)
    return sources, target, _true_cate(target_site)


def naturalness_bounds(
    sources: Dict[str, pd.DataFrame],
    target: pd.DataFrame,
    n_draws: int = 1000,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bound the target CATEs with Naturalness.

    Args:
        sources: Source-site rows with both regimes, keyed by site name.
        target: Target-site observational rows.
        n_draws: Tolerance draws propagated through the target linear program
            (the paper uses 1000).
        seed: Seed of the Gaussian-process fits and the tolerance draws.

    Returns:
        ``(bounds, conditions)``: one row per target stratum with the
        Naturalness interval and the share of draws whose program is optimal
        (``NaN`` endpoints where no condition fires or no draw is optimal),
        and the target conditions that fired.
    """
    grid = NaturalnessGrid(epsilons=EPSILONS, taus=TAUS, covariates=COVARIATES)
    grounder = NaturalnessGrounder(grid, stage1_backend="empirical", random_seed=seed)
    with warnings.catch_warnings():
        # The GP hyperparameter search reports bounds it touches; harmless here.
        warnings.simplefilter("ignore", ConvergenceWarning)
        grounder.fit(sources, treatment="X", outcome="Y", regime_col="F")

    policy = TargetNaturalnessPolicy().choose(target, grounder)
    quantities = estimate_target_observed_binary_quantities(
        target, treatment="X", outcome="Y", covariates=COVARIATES, regime_col="F"
    )
    strata = pd.DataFrame(quantities.z_values, columns=COVARIATES)
    if policy.status != "selected":
        # Nothing fires at the target: Naturalness abstains, and the paper
        # scores such strata with their Manski interval.
        empty = strata.assign(lower=np.nan, upper=np.nan, p_optimal=0.0)
        return empty, pd.DataFrame()

    conditions = policy.to_frame(selected_only=True)
    draws = grounder.sample_tolerances(conditions, n_samples=n_draws, random_seed=seed)
    solved = pd.concat(
        [
            solve_target_bounds_from_observed_binary_quantities(
                quantities,
                draw,
                return_details=True,
                include_intervention_diagnostics=False,
            )
            for _, draw in draws.groupby("sample_id")
        ],
        ignore_index=True,
    )
    solved["optimal"] = solved["status"].eq("optimal")
    summary = solved.groupby(COVARIATES).agg(p_optimal=("optimal", "mean")).reset_index()
    endpoints = (
        solved[solved["optimal"]].groupby(COVARIATES)[["lower", "upper"]].mean().reset_index()
    )
    bounds = strata.merge(summary, on=COVARIATES, how="left").merge(
        endpoints, on=COVARIATES, how="left"
    )
    return bounds[[*COVARIATES, "lower", "upper", "p_optimal"]], conditions


def reference_bounds(target: pd.DataFrame) -> pd.DataFrame:
    """Return the Manski and Balke-Pearl bounds of every target stratum.

    Manski bounds use no assumption beyond a binary outcome. Balke-Pearl bounds
    treat ``Z1`` as a valid instrument within each ``Z2`` stratum, which is true
    in this simulation by construction but has to be assumed in practice.
    """
    rows = []
    for (z1, z2), cell in target.groupby(COVARIATES):
        p_x1 = cell["X"].mean()
        mean_y1 = cell.loc[cell["X"].eq(1), "Y"].mean()
        mean_y0 = cell.loc[cell["X"].eq(0), "Y"].mean()
        manski_lower = mean_y1 * p_x1 - mean_y0 * (1 - p_x1) - p_x1
        stratum = target[target["Z2"].eq(z2)]
        # joint[z1, y, x] = P(Y=y, X=x | Z1=z1, Z2=z2)
        joint = np.zeros((2, 2, 2))
        for level in (0, 1):
            arm = stratum[stratum["Z1"].eq(level)]
            for y in (0, 1):
                for x in (0, 1):
                    joint[level, y, x] = (arm["Y"].eq(y) & arm["X"].eq(x)).mean()
        balke_pearl = balke_pearl_interval(joint)
        rows.append({
            "Z1": z1,
            "Z2": z2,
            "manski_lower": manski_lower,
            "manski_upper": manski_lower + 1.0,
            "balke_pearl_lower": balke_pearl.lower,
            "balke_pearl_upper": balke_pearl.upper,
        })
    return pd.DataFrame(rows)


def _interval(lower: float, upper: float) -> str:
    """Format an interval, or ``abstains`` when it is missing."""
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return "abstains"
    return f"[{lower:+.3f}, {upper:+.3f}]"


def main() -> None:
    """Simulate the toy data, run Naturalness and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="simulation and fitting seed")
    parser.add_argument("--n-draws", type=int, default=1000, help="LP posterior draws")
    args = parser.parse_args()

    started = time.perf_counter()
    sources, target, truth = simulate_toy_data(seed=args.seed)
    regimes = next(iter(sources.values()))["F"].value_counts()
    print(
        f"Simulated {len(sources)} source sites ({regimes['idle']} observational and "
        f"{regimes['on']} trial rows each) and a target site with {len(target)} "
        "observational rows."
    )
    bounds, conditions = naturalness_bounds(
        sources, target, n_draws=args.n_draws, seed=args.seed
    )

    if conditions.empty:
        print("\nNo Naturalness condition fires at the target: Naturalness abstains.")
    else:
        print("\nTarget conditions that fire (I_j <= epsilon and D_j >= tau):")
        shown = conditions[["covariate", "epsilon", "tau", "i_deviation", "d_strength", "two_c", "two_d"]]
        renamed = {"i_deviation": "I_j", "d_strength": "D_j", "two_c": "2c", "two_d": "2d"}
        print(shown.rename(columns=renamed).to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        ))

    table = truth.merge(bounds, on=COVARIATES).merge(reference_bounds(target), on=COVARIATES)
    print(f"\nTarget CATE bounds ({args.n_draws} LP posterior draws):")
    print(pd.DataFrame({
        "Z1": table["Z1"],
        "Z2": table["Z2"],
        "true CATE": table["true_cate"].map(lambda value: f"{value:+.3f}"),
        "Naturalness": [_interval(lo, hi) for lo, hi in zip(table["lower"], table["upper"])],
        "P(optimal)": table["p_optimal"].map(lambda value: f"{value:.2f}"),
        "Manski": [_interval(lo, hi) for lo, hi in zip(table["manski_lower"], table["manski_upper"])],
        "Balke-Pearl (Z1)": [
            _interval(lo, hi) for lo, hi in zip(table["balke_pearl_lower"], table["balke_pearl_upper"])
        ],
    }).to_string(index=False))

    bounded = table.dropna(subset=["lower", "upper"])
    covered = bounded["lower"].le(bounded["true_cate"]) & bounded["true_cate"].le(bounded["upper"])
    print(
        f"\nNaturalness bounds {len(bounded)} of {len(table)} strata and covers the true CATE "
        f"in {int(covered.sum())} of them; mean width {(bounded['upper'] - bounded['lower']).mean():.3f} "
        f"(Manski 1.000, Balke-Pearl "
        f"{(table['balke_pearl_upper'] - table['balke_pearl_lower']).mean():.3f}). "
        f"Elapsed {time.perf_counter() - started:.0f} s."
    )


if __name__ == "__main__":
    main()
