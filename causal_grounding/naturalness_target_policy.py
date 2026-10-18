"""
Target-site policies for learned naturalness tolerances.

The Stage-2/Stage-3 naturalness learner estimates tolerances from training
sites. This module decides which learned tolerance(s) should be applied to a
held-out target site that usually contains only ``F='idle'`` rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from . import naturalness_statistics
from .naturalness_grounder import (
    NaturalnessGrounder,
    NaturalnessGrid,
    NaturalnessTolerance,
)


@dataclass(frozen=True)
class TargetNaturalnessChoice:
    """One target-side naturalness choice and its learned tolerance.

    Attributes:
        policy: Name of the policy that produced the choice.
        covariate: Selected covariate name.
        covariate_index: Position of the covariate in the naturalness grid.
        epsilon: Target-side ``I_j(epsilon)`` threshold.
        tau: Target-side ``D_j(tau)`` threshold.
        i_deviation: Estimated target-side ``I_j`` deviation.
        d_strength: Estimated target-side ``D_j`` strength.
        i_margin: ``epsilon - i_deviation``. Larger is more stable.
        d_margin: ``d_strength - tau``. Larger is more stable.
        c: Learned note-scale ``c`` tolerance.
        d: Learned note-scale ``d`` tolerance.
        two_c: Solver-scale ``2c`` tolerance.
        two_d: Solver-scale ``2d`` tolerance.
        score: Ranking score used by tolerance-minimizing policies.
        c_source: Source label for the learned c prediction.
        d_source: Source label for the learned d prediction.
        c_std_logit: Posterior standard deviation for c on logit scale, if any.
        d_std_logit: Posterior standard deviation for d on logit scale, if any.
        upper_confidence_level: Joint upper-band confidence level, if used.
        upper_critical_value: Shared maximum-statistic critical value, if used.
        upper_family_size: Number of c/d coordinates protected jointly.
        upper_n_samples: Monte Carlo draws used for joint-band calibration.
        n_z_minus_evaluated: Number of target support slices used.
        n_target_rows: Number of target idle rows considered.
        is_primary_choice: Whether this is the policy's one primary row.
        reason: Human-readable explanation for the choice.
    """

    policy: str
    covariate: str
    covariate_index: int
    epsilon: float
    tau: float
    i_deviation: float
    d_strength: float
    i_margin: float
    d_margin: float
    c: float
    d: float
    two_c: float
    two_d: float
    score: float
    c_source: str
    d_source: str
    c_std_logit: Optional[float]
    d_std_logit: Optional[float]
    n_z_minus_evaluated: int
    n_target_rows: int
    is_primary_choice: bool
    reason: str
    upper_confidence_level: Optional[float] = None
    upper_critical_value: Optional[float] = None
    upper_family_size: Optional[int] = None
    upper_n_samples: Optional[int] = None


@dataclass
class TargetNaturalnessPolicyResult:
    """
    Result object returned by ``TargetNaturalnessPolicy.choose``.

    ``candidates`` contains every target-triggered candidate. ``selected_choices``
    contains the candidate(s) selected by the policy. ``choice`` is the primary
    selected row and is useful for policies that produce exactly one choice.
    ``trigger_grid`` holds the complete target trigger evaluation, including
    rejected rows, when the policy computed it.
    """

    policy: str
    status: str
    choice: Optional[TargetNaturalnessChoice]
    selected_choices: List[TargetNaturalnessChoice]
    candidates: List[TargetNaturalnessChoice]
    reason: str
    trigger_grid: Optional[pd.DataFrame] = None

    def to_frame(self, selected_only: bool = False) -> pd.DataFrame:
        """
        Return candidate or selected choices as a display-friendly DataFrame.

        Args:
            selected_only: If true, return only policy-selected rows. Otherwise
                return every target-triggered candidate.

        Returns:
            DataFrame with one row per choice. Empty results include the policy
            status and reason columns so callers can write diagnostics.
        """
        rows = self.selected_choices if selected_only else self.candidates
        if not rows:
            return pd.DataFrame([{
                "policy": self.policy,
                "status": self.status,
                "reason": self.reason,
            }])
        frame = pd.DataFrame([asdict(row) for row in rows])
        frame.insert(1, "status", self.status)
        return frame


class _GrounderToleranceLookup:
    """Adapter that asks a fitted grounder for note-scale c/d predictions."""

    _supported_uncertainty = {
        "posterior_mean",
        "posterior_lower_05",
        "posterior_upper_95",
        "posterior_simultaneous_upper",
        "posterior_simultaneous_upper_95",
    }

    def __init__(
        self,
        grounder: NaturalnessGrounder,
        uncertainty: str,
        clip: bool,
        confidence_level: float,
    ) -> None:
        """Store lookup settings for a fitted naturalness grounder."""
        if uncertainty not in self._supported_uncertainty:
            raise ValueError(
                f"Unsupported uncertainty mode {uncertainty!r}. "
                f"Available modes: {sorted(self._supported_uncertainty)}"
            )
        self.grounder = grounder
        self.uncertainty = uncertainty
        self.clip = bool(clip)
        self.confidence_level = float(confidence_level)

    def get(
        self,
        covariate: str,
        epsilon: float,
        tau: float,
        i_deviation: Optional[float] = None,
        d_strength: Optional[float] = None,
    ) -> NaturalnessTolerance:
        """
        Return the requested note-scale naturalness tolerance.

        Args:
            covariate: Candidate naturalness covariate.
            epsilon: Naturalness trigger threshold.
            tau: Treatment-strength trigger threshold.
            i_deviation: Target-side ``I_j`` statistic, used by observed-stat
                GP feature mode.
            d_strength: Target-side ``D_j`` statistic, used by observed-stat
                GP feature mode.

        Returns:
            NaturalnessTolerance predicted by the fitted grounder.
        """
        if self.uncertainty in {
            "posterior_simultaneous_upper",
            "posterior_simultaneous_upper_95",
        }:
            raise RuntimeError(
                "Simultaneous upper tolerances require one vector lookup over "
                "all triggered rows."
            )
        return self.grounder.predict_tolerance(
            covariate,
            epsilon=epsilon,
            tau=tau,
            quantile=self._quantile_for_uncertainty(),
            clip=self.clip,
            i_deviation=i_deviation,
            d_strength=d_strength,
        )

    def get_many(
        self,
        rows: pd.DataFrame,
    ) -> Optional[List[NaturalnessTolerance]]:
        """
        Return a jointly protected tolerance vector when requested.

        Args:
            rows: All target-triggered rows in their eventual candidate order.

        Returns:
            Simultaneous upper tolerances, or ``None`` for scalar uncertainty
            modes so the caller can use ordinary row-wise lookup.
        """
        if self.uncertainty not in {
            "posterior_simultaneous_upper",
            "posterior_simultaneous_upper_95",
        }:
            return None
        confidence_level = (
            0.95
            if self.uncertainty == "posterior_simultaneous_upper_95"
            else self.confidence_level
        )
        return self.grounder.predict_simultaneous_upper_tolerances(
            rows,
            confidence_level=confidence_level,
            clip=self.clip,
        )

    def _quantile_for_uncertainty(self) -> Optional[float]:
        """Map the target policy's uncertainty mode to a GP quantile."""
        if self.uncertainty == "posterior_mean":
            return None
        if self.uncertainty == "posterior_lower_05":
            return 0.05
        if self.uncertainty == "posterior_upper_95":
            return 0.95
        if self.uncertainty in {
            "posterior_simultaneous_upper",
            "posterior_simultaneous_upper_95",
        }:
            raise RuntimeError(
                "Simultaneous upper uncertainty has no scalar GP quantile."
            )
        raise ValueError(f"Unsupported uncertainty mode {self.uncertainty!r}")


class TargetNaturalnessPolicy:
    """
    Choose learned naturalness tolerances for a held-out target site.

    All target-triggered choices are returned for downstream top-k selection
    (as in Exp16); the smallest-tolerance row is marked as the primary
    diagnostic choice.
    """

    SUPPORTED_POLICIES = {
        "all_triggered_intersection",
    }

    def __init__(
        self,
        policy: str = "all_triggered_intersection",
        uncertainty: Optional[str] = None,
        clip: bool = True,
        confidence_level: float = 0.95,
    ) -> None:
        """
        Initialize target selection and learned-tolerance uncertainty settings.

        Args:
            policy: Target candidate-selection policy name.
            uncertainty: GP summary mode. ``posterior_simultaneous_upper`` uses
                ``confidence_level``; its ``_95`` variant always uses 0.95.
            clip: Whether learned note-scale tolerances are clipped to
                ``[0, 0.5]``.
            confidence_level: Joint one-sided posterior level for generic
                simultaneous-upper mode.
        """
        if policy not in self.SUPPORTED_POLICIES:
            raise ValueError(
                f"Unsupported target naturalness policy {policy!r}. "
                f"Available policies: {sorted(self.SUPPORTED_POLICIES)}"
            )
        if uncertainty is None:
            uncertainty = "posterior_mean"
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1")
        self.policy = policy
        self.uncertainty = uncertainty
        self.clip = bool(clip)
        self.confidence_level = float(confidence_level)

    def choose(
        self,
        target_idle_data: pd.DataFrame,
        grounder: NaturalnessGrounder,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        regime_col: Optional[str] = None,
        provider: Optional[Any] = None,
        trigger_quantities: Optional[Any] = None,
    ) -> TargetNaturalnessPolicyResult:
        """
        Choose target-side naturalness tolerance(s).

        Args:
            target_idle_data: Target-site data. If ``regime_col`` is present,
                only rows equal to the grounder's idle value are used.
            grounder: Fitted naturalness learner supplying the grid and learned
                tolerance provider.
            treatment: Optional treatment column. Defaults to the fitted
                grounder's treatment.
            outcome: Optional outcome column. Defaults to the fitted grounder's
                outcome.
            regime_col: Optional regime column. Defaults to the fitted
                grounder's regime column.
            provider: Optional preconfigured tolerance provider. If omitted,
                one is created from ``grounder`` with this policy's uncertainty
                mode.
            trigger_quantities: Optional model-backed target probabilities with
                the ``TargetObservedBinaryQuantities`` interface. When supplied,
                target trigger statistics use these probabilities instead of
                empirical joint-cell proportions. A grid with
                ``conditioning_covariates`` takes a mapping from each
                candidate-state tuple to its table instead.

        Returns:
            ``TargetNaturalnessPolicyResult`` with candidates and selected
            choice(s). If no target-triggered candidate exists, ``status`` is
            ``"no_triggered_candidates"``.
        """
        treatment = treatment or grounder.treatment_
        outcome = outcome or grounder.outcome_
        regime_col = regime_col or grounder.regime_col_
        if treatment is None or outcome is None:
            raise ValueError("treatment and outcome must be provided or fitted on grounder")

        # The target policy is allowed to inspect only target idle-regime rows.
        idle = self._prepare_target_idle_data(
            target_idle_data,
            treatment=treatment,
            outcome=outcome,
            covariates=grounder.grid.covariates,
            regime_col=regime_col,
            idle_value=grounder.idle_value,
        )
        # Tolerances still come from the fitted source grounder; target data only
        # decides which grid row(s) are relevant.
        provider = provider or _GrounderToleranceLookup(
            grounder,
            uncertainty=self.uncertainty,
            clip=self.clip,
            confidence_level=self.confidence_level,
        )
        diagnostics = self._evaluate_trigger_grid(
            idle,
            grounder.grid,
            treatment=treatment,
            outcome=outcome,
            alpha=grounder.alpha,
            trigger_quantities=trigger_quantities,
        )
        candidates = self._build_candidates(
            grounder.grid,
            provider,
            policy=self.policy,
            diagnostics=diagnostics,
        )

        if not candidates:
            return TargetNaturalnessPolicyResult(
                policy=self.policy,
                status="no_triggered_candidates",
                choice=None,
                selected_choices=[],
                candidates=[],
                reason="No target grid point satisfied I_j(epsilon) and D_j(tau).",
                trigger_grid=diagnostics,
            )

        # Rank target-triggered rows according to the requested policy.
        ranked = self._rank_candidates(candidates)
        selected = [
            self._replace_primary_flag(choice, is_primary=(index == 0))
            for index, choice in enumerate(ranked)
        ]
        reason = (
            "Selected all target-triggered choices for downstream bound "
            "intersection; primary row has the smallest tolerance score."
        )

        return TargetNaturalnessPolicyResult(
            policy=self.policy,
            status="selected",
            choice=selected[0],
            selected_choices=selected,
            candidates=candidates,
            reason=reason,
            trigger_grid=diagnostics,
        )

    def evaluate_triggers(
        self,
        target_idle_data: pd.DataFrame,
        grounder: NaturalnessGrounder,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        regime_col: Optional[str] = None,
        trigger_quantities: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Evaluate every target candidate-grid trigger, including rejections.

        Args:
            target_idle_data: Target-site rows. When the regime column exists,
                only the fitted idle regime is retained.
            grounder: Fitted naturalness grounder defining covariates and grid.
            treatment: Optional treatment column override.
            outcome: Optional outcome column override.
            regime_col: Optional regime column override.
            trigger_quantities: Optional model-backed target probabilities used
                in place of empirical joint-cell proportions.

        Returns:
            One row per candidate, epsilon, and tau with target ``I`` and ``D``
            statistics, margins, pass indicators, and rejection reason.
        """

        treatment = treatment or grounder.treatment_
        outcome = outcome or grounder.outcome_
        regime_col = regime_col or grounder.regime_col_
        if treatment is None or outcome is None:
            raise ValueError(
                "treatment and outcome must be provided or fitted on grounder"
            )
        idle = self._prepare_target_idle_data(
            target_idle_data,
            treatment=treatment,
            outcome=outcome,
            covariates=grounder.grid.covariates,
            regime_col=regime_col,
            idle_value=grounder.idle_value,
        )
        return self._evaluate_trigger_grid(
            idle,
            grounder.grid,
            treatment=treatment,
            outcome=outcome,
            alpha=grounder.alpha,
            trigger_quantities=trigger_quantities,
        )

    @staticmethod
    def _prepare_target_idle_data(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str],
        regime_col: Optional[str],
        idle_value: str,
    ) -> pd.DataFrame:
        """Validate target data and return the idle-regime rows."""
        missing = [
            col for col in list(covariates) + [treatment, outcome]
            if col not in data.columns
        ]
        if missing:
            raise ValueError(f"Missing required target columns: {missing}")

        idle = data.copy()
        if regime_col is not None and regime_col in idle.columns:
            idle = idle[idle[regime_col] == idle_value].copy()
        if idle.empty:
            raise ValueError("target_idle_data has no usable idle rows")
        if not naturalness_statistics.is_binary_series(idle[treatment]):
            raise ValueError(f"Treatment column {treatment!r} must be binary")
        if not naturalness_statistics.is_binary_series(idle[outcome]):
            raise ValueError(f"Outcome column {outcome!r} must be binary")
        return idle

    def _build_candidates(
        self,
        grid: NaturalnessGrid,
        provider: Any,
        policy: str,
        diagnostics: pd.DataFrame,
    ) -> List[TargetNaturalnessChoice]:
        """
        Evaluate target triggers and attach learned source tolerances.

        A target candidate row ``(j, epsilon, tau)`` is retained when:

            I*_j <= epsilon
            D*_j >= tau

        where ``I*_j`` is the target idle outcome-stability statistic and
        ``D*_j`` is the target idle treatment-relevance statistic.  The target
        rows choose the grid point only; the actual LP tolerances
        ``c(j,epsilon,tau)`` and ``d(j,epsilon,tau)`` come from the fitted
        source grounder or its GP surface.
        """
        candidates: List[TargetNaturalnessChoice] = []
        triggered = diagnostics.loc[diagnostics["target_triggered"]].reset_index(
            drop=True
        )
        bulk_getter = getattr(provider, "get_many", None)
        bulk_tolerances = bulk_getter(triggered) if callable(bulk_getter) else None
        if bulk_tolerances is not None and len(bulk_tolerances) != len(triggered):
            raise RuntimeError(
                "Bulk tolerance provider returned a different number of rows "
                "than the target trigger grid."
            )

        for row_index, row in enumerate(triggered.itertuples(index=False)):
            tolerance = (
                bulk_tolerances[row_index]
                if bulk_tolerances is not None
                else provider.get(
                    row.covariate,
                    epsilon=float(row.epsilon),
                    tau=float(row.tau),
                    i_deviation=float(row.i_deviation),
                    d_strength=float(row.d_strength),
                )
            )
            two_c = self._double_tolerance(tolerance.c)
            two_d = self._double_tolerance(tolerance.d)
            score = self._choice_score(tolerance)
            reason = self._choice_reason(policy, score)
            candidates.append(TargetNaturalnessChoice(
                policy=policy,
                covariate=str(row.covariate),
                covariate_index=int(row.covariate_index),
                epsilon=float(row.epsilon),
                tau=float(row.tau),
                i_deviation=float(row.i_deviation),
                d_strength=float(row.d_strength),
                i_margin=float(row.i_margin),
                d_margin=float(row.d_margin),
                c=tolerance.c,
                d=tolerance.d,
                two_c=two_c,
                two_d=two_d,
                score=score,
                c_source=tolerance.c_source,
                d_source=tolerance.d_source,
                c_std_logit=tolerance.c_std_logit,
                d_std_logit=tolerance.d_std_logit,
                n_z_minus_evaluated=int(row.n_z_minus_evaluated),
                n_target_rows=int(row.n_target_rows),
                is_primary_choice=False,
                reason=reason,
                upper_confidence_level=tolerance.upper_confidence_level,
                upper_critical_value=tolerance.upper_critical_value,
                upper_family_size=tolerance.upper_family_size,
                upper_n_samples=tolerance.upper_n_samples,
            ))

        return candidates

    def _evaluate_trigger_grid(
        self,
        idle: pd.DataFrame,
        grid: NaturalnessGrid,
        *,
        treatment: str,
        outcome: str,
        alpha: float,
        trigger_quantities: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Return complete target trigger diagnostics over the configured grid.

        Each candidate is evaluated in its own state
        (``NaturalnessGrid.candidate_state``), which is the full covariate
        vector unless the grid declares ``conditioning_covariates``.
        """

        rows: List[Dict[str, Any]] = []
        for j, covariate in grid.candidate_items:
            state_covariates, state_index = grid.candidate_state(j)
            state_quantities = self._state_trigger_quantities(
                trigger_quantities,
                state_covariates,
            )
            z_minus_values = self._get_z_minus_values(
                idle,
                grid,
                j,
                supported_z_values=(
                    state_quantities.z_values
                    if state_quantities is not None
                    else None
                ),
            )
            i_deviation = self._compute_target_i_deviation(
                idle,
                state_covariates,
                treatment,
                outcome,
                alpha,
                state_index,
                z_minus_values,
                trigger_quantities=state_quantities,
            )
            d_strength = self._compute_target_d_strength(
                idle,
                state_covariates,
                treatment,
                alpha,
                state_index,
                z_minus_values,
                trigger_quantities=state_quantities,
            )
            finite = bool(
                np.isfinite(i_deviation) and np.isfinite(d_strength)
            )
            for epsilon in grid.epsilons:
                for tau in grid.taus:
                    outcome_passed = bool(
                        finite and i_deviation <= float(epsilon)
                    )
                    treatment_passed = bool(
                        finite and d_strength >= float(tau)
                    )
                    target_triggered = outcome_passed and treatment_passed
                    if not finite:
                        rejection_reason = "nonfinite_target_statistic"
                    elif not outcome_passed and not treatment_passed:
                        rejection_reason = (
                            "outcome_instability_and_weak_treatment_relevance"
                        )
                    elif not outcome_passed:
                        rejection_reason = "outcome_instability"
                    elif not treatment_passed:
                        rejection_reason = "weak_treatment_relevance"
                    else:
                        rejection_reason = "triggered"
                    rows.append(
                        {
                            "covariate": str(covariate),
                            "covariate_index": int(j),
                            "epsilon": float(epsilon),
                            "tau": float(tau),
                            "i_deviation": float(i_deviation),
                            "d_strength": float(d_strength),
                            "i_margin": float(epsilon - i_deviation),
                            "d_margin": float(d_strength - tau),
                            "outcome_screen_passed": outcome_passed,
                            "treatment_screen_passed": treatment_passed,
                            "target_triggered": target_triggered,
                            "rejection_reason": rejection_reason,
                            "n_z_minus_evaluated": len(z_minus_values),
                            "n_target_rows": int(len(idle)),
                            "target_probability_source": (
                                "model"
                                if trigger_quantities is not None
                                else "empirical"
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    def _rank_candidates(
        self,
        candidates: List[TargetNaturalnessChoice],
    ) -> List[TargetNaturalnessChoice]:
        """
        Rank target-triggered candidates for the diagnostic primary row.

        The main score is:

            score(j,epsilon,tau) = 2c(j,epsilon,tau) + 2d(j,epsilon,tau).

        Ties prefer lower posterior uncertainty, smaller ``epsilon``, larger
        ``tau``, larger target trigger margins, and finally stable covariate
        order. All triggered rows are kept for downstream top-k selection;
        this ranking's first row is marked primary.
        """
        # The primary diagnostic row is the target-triggered choice with the
        # smallest learned 2c + 2d tolerance score.
        return sorted(
            candidates,
            key=lambda choice: (
                self._nan_last(choice.score),
                self._nan_last(self._uncertainty_score(choice)),
                choice.epsilon,
                -choice.tau,
                -choice.i_margin,
                -choice.d_margin,
                choice.covariate_index,
            ),
        )

    @staticmethod
    def _replace_primary_flag(
        choice: TargetNaturalnessChoice,
        is_primary: bool,
    ) -> TargetNaturalnessChoice:
        """Return a copy of a choice with an updated primary-choice flag."""
        values = asdict(choice)
        values["is_primary_choice"] = bool(is_primary)
        return TargetNaturalnessChoice(**values)

    @staticmethod
    def _choice_score(tolerance: NaturalnessTolerance) -> float:
        """Score a tolerance by the solver-scale total ``2c + 2d``."""
        if np.isnan(tolerance.c) or np.isnan(tolerance.d):
            return float("nan")
        return float(2.0 * tolerance.c + 2.0 * tolerance.d)

    @staticmethod
    def _double_tolerance(value: float) -> float:
        """Return the solver-scale ``2c`` or ``2d`` RHS for a note-scale tolerance."""
        return float(2.0 * value) if not np.isnan(value) else float("nan")

    @staticmethod
    def _choice_reason(policy: str, score: float) -> str:
        """Return a short explanation for a candidate under a policy."""
        if policy == "all_triggered_intersection":
            return "Target-triggered candidate retained for downstream intersection."
        return f"Target-triggered candidate with tolerance score {score:.6g}."

    @staticmethod
    def _uncertainty_score(choice: TargetNaturalnessChoice) -> float:
        """Combine c and d logit-scale posterior uncertainties for tie-breaking."""
        values = [
            value
            for value in [choice.c_std_logit, choice.d_std_logit]
            if value is not None and np.isfinite(value)
        ]
        return float(sum(values)) if values else float("nan")

    @staticmethod
    def _nan_last(value: float) -> Tuple[int, float]:
        """Sort finite values before NaNs."""
        return (1, 0.0) if value is None or np.isnan(value) else (0, float(value))

    @staticmethod
    def _z_mask(
        data: pd.DataFrame,
        covariates: List[str],
        z: Tuple[Any, ...],
    ) -> pd.Series:
        """Return rows whose covariate tuple exactly matches ``z``."""
        z = naturalness_statistics.as_tuple(z)
        if len(z) != len(covariates):
            raise ValueError(f"Expected z with {len(covariates)} values, got {len(z)}")
        mask = pd.Series(True, index=data.index)
        for col, value in zip(covariates, z):
            mask &= data[col].eq(value)
        return mask

    @staticmethod
    def _smoothed_binary_prob(n_success: int, n_total: int, alpha: float) -> float:
        """Estimate a binary target probability with symmetric smoothing."""
        return float((n_success + alpha) / (n_total + 2.0 * alpha))

    def _p_idle_x_given_z(
        self,
        idle: pd.DataFrame,
        covariates: List[str],
        treatment: str,
        alpha: float,
        x: int,
        z: Tuple[Any, ...],
    ) -> float:
        """Return target-side empirical P(X=x | z; idle)."""
        mask_z = self._z_mask(idle, covariates, z)
        mask_x = idle[treatment].eq(x)
        return self._smoothed_binary_prob(int((mask_z & mask_x).sum()), int(mask_z.sum()), alpha)

    def _p_idle_y_given_xz(
        self,
        idle: pd.DataFrame,
        covariates: List[str],
        treatment: str,
        outcome: str,
        alpha: float,
        y: int,
        x: int,
        z: Tuple[Any, ...],
    ) -> float:
        """Return target-side empirical P(Y=y | X=x,z; idle)."""
        mask_z = self._z_mask(idle, covariates, z)
        mask_xz = mask_z & idle[treatment].eq(x)
        mask_y = idle[outcome].eq(y)
        return self._smoothed_binary_prob(
            int((mask_xz & mask_y).sum()),
            int(mask_xz.sum()),
            alpha,
        )

    @staticmethod
    def _state_trigger_quantities(
        trigger_quantities: Optional[Any],
        state_covariates: List[str],
    ) -> Optional[Any]:
        """Return the target probability table for one candidate state.

        Args:
            trigger_quantities: ``None``, one table shared by every candidate,
                or a mapping from candidate-state tuples to tables.
            state_covariates: Ordered covariates of the candidate's state.

        Returns:
            The table whose covariates are exactly ``state_covariates``, or
            ``None`` for empirical target statistics.

        Raises:
            ValueError: If no table matches the candidate's state.
        """

        if trigger_quantities is None:
            return None
        quantities = trigger_quantities
        if isinstance(trigger_quantities, Mapping):
            try:
                quantities = trigger_quantities[tuple(state_covariates)]
            except KeyError as exc:
                raise ValueError(
                    "trigger_quantities has no table for candidate state "
                    f"{tuple(state_covariates)!r}"
                ) from exc
        if list(quantities.covariates) != list(state_covariates):
            raise ValueError(
                "trigger_quantities covariates must match the naturalness grid"
            )
        return quantities

    @staticmethod
    def _get_z_minus_values(
        idle: pd.DataFrame,
        grid: NaturalnessGrid,
        j: int,
        supported_z_values: Optional[Sequence[Tuple[Any, ...]]] = None,
    ) -> List[Tuple[Any, ...]]:
        """Enumerate paired target-supported values of Z without covariate j.

        Args:
            idle: Prepared target idle-regime rows.
            grid: Naturalness grid defining the ordered covariates and minimum
                row count for each paired candidate level. ``Z`` is candidate
                ``j``'s state, so declared conditioning covariates bound
                ``Z_-j``.
            j: Index of the candidate coordinate being compared.
            supported_z_values: Optional full-state support supplied by a
                target probability table. When present, a ``Z_-j`` slice is
                retained only if every observed level of ``Z_j`` reconstructs
                a state in that table.

        Returns:
            Ordered ``Z_-j`` tuples with row-count support and, when supplied,
            complete probability-table support across candidate levels.
        """

        state_covariates, state_index = grid.candidate_state(j)
        z_j_col = state_covariates[state_index]
        z_j_values = naturalness_statistics.safe_sorted(
            idle[z_j_col].dropna().unique().tolist()
        )
        z_minus_cols = [
            c for idx, c in enumerate(state_covariates) if idx != state_index
        ]
        if not z_minus_cols:
            counts = idle.groupby(z_j_col, observed=True).size()
            values = (
                [()]
                if all(
                    int(counts.get(value, 0)) >= grid.min_cell_count
                    for value in z_j_values
                )
                else []
            )
            if supported_z_values is not None:
                supported = {
                    naturalness_statistics.as_tuple(z)
                    for z in supported_z_values
                }
                values = [
                    ()
                    for _ in values
                    if all((z_j,) in supported for z_j in z_j_values)
                ]
            return values

        grouped = (
            idle.groupby(
                [*z_minus_cols, z_j_col],
                observed=True,
            )
            .size()
            .unstack(z_j_col, fill_value=0)
        )
        for value in z_j_values:
            if value not in grouped.columns:
                grouped[value] = 0
        paired = grouped[z_j_values].ge(grid.min_cell_count).all(axis=1)
        values = [
            naturalness_statistics.as_tuple(index)
            for index in grouped.index[paired].tolist()
        ]
        values = naturalness_statistics.safe_sorted(values)

        if supported_z_values is not None:
            supported = {
                naturalness_statistics.as_tuple(z)
                for z in supported_z_values
            }
            n_covariates = len(state_covariates)
            values = [
                z_minus
                for z_minus in values
                if all(
                    naturalness_statistics.assemble_covariate_state(
                        state_index,
                        z_j,
                        z_minus,
                        n_covariates,
                    )
                    in supported
                    for z_j in z_j_values
                )
            ]

        max_values = grid.max_z_minus_samples
        if max_values is not None and len(values) > max_values:
            values = values[:max_values]

        return values

    def _compute_target_i_deviation(
        self,
        idle: pd.DataFrame,
        covariates: List[str],
        treatment: str,
        outcome: str,
        alpha: float,
        j: int,
        z_minus_values: List[Tuple[Any, ...]],
        trigger_quantities: Optional[Any] = None,
    ) -> float:
        """
        Compute target-side ``I*_j`` for candidate discovery.

        The statistic is:

            I*_j = max_{x,y,z_j,z'_j,z_-j}
                   |P_target(Y=y | X=x,Z=(z_j,z_-j); idle)
                    - P_target(Y=y | X=x,Z=(z'_j,z_-j); idle)|.

        Here ``target`` marks target-site probabilities, not potential
        outcomes. They are empirical by default and may be supplied by a
        model-backed trigger table. A row passes the target outcome-stability
        screen when ``I*_j <= epsilon``.
        """
        def probability(y: int, x: int, z: Tuple[Any, ...]) -> float:
            if trigger_quantities is None:
                return self._p_idle_y_given_xz(
                    idle,
                    covariates,
                    treatment,
                    outcome,
                    alpha,
                    y,
                    x,
                    z,
                )
            p_y1 = float(trigger_quantities.p_y1_given_xz[(x, z)])
            return 1.0 - p_y1 if y == 0 else p_y1

        return naturalness_statistics.compute_i_deviation(
            covariates,
            j,
            z_minus_values,
            probability,
            idle,
        )

    def _compute_target_d_strength(
        self,
        idle: pd.DataFrame,
        covariates: List[str],
        treatment: str,
        alpha: float,
        j: int,
        z_minus_values: List[Tuple[Any, ...]],
        trigger_quantities: Optional[Any] = None,
    ) -> float:
        """
        Compute target-side ``D*_j`` for treatment relevance.

        The statistic is:

            D*_j = min_{x,z_j,z'_j,z_-j}
                   |P_target(X=x | Z=(z_j,z_-j); idle)
                    - P_target(X=x | Z=(z'_j,z_-j); idle)|.

        A row passes the target relevance screen when ``D*_j >= tau``.
        """
        def probability(x: int, z: Tuple[Any, ...]) -> float:
            if trigger_quantities is None:
                return self._p_idle_x_given_z(
                    idle,
                    covariates,
                    treatment,
                    alpha,
                    x,
                    z,
                )
            return float(trigger_quantities.p_x(x, z))

        return naturalness_statistics.compute_d_strength(
            covariates,
            j,
            z_minus_values,
            probability,
            idle,
        )


__all__ = [
    "TargetNaturalnessChoice",
    "TargetNaturalnessPolicy",
    "TargetNaturalnessPolicyResult",
]
