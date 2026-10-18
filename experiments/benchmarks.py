"""Shared B-Learner benchmark settings and Monte Carlo result helpers."""

from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from catenets.models.torch import DRLearner, PWLearner, RALearner, TLearner
from catenets.models.torch.base import BasicNet, DEVICE
from econml.grf import RegressionForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn_quantile import RandomForestQuantileRegressor


META_LEARNER_LABELS = {
    "cb_pi": "CB-PI",
    "cb_ra": "CB-RA",
    "cb_ipw": "CB-IPW",
    "cb_dr": "CB-DR",
}
META_LEARNER_COLORS = {
    "cb_pi": "tab:blue",
    "cb_ra": "tab:green",
    "cb_ipw": "tab:cyan",
    "cb_dr": "tab:olive",
}
WIDTH_MATCHED_GAMMA = "width_matched"
WIDTH_MATCHED_BINARY_GAMMA = "width_matched_binary"
EXP_BETA_GAMMA = "exp_beta"
ORACLE_GAMMA = "oracle"
DGP_ORACLE_GAMMA = "dgp_oracle"
BLEARNER_GAMMA_STRATEGIES = (
    WIDTH_MATCHED_GAMMA,
    WIDTH_MATCHED_BINARY_GAMMA,
    EXP_BETA_GAMMA,
    ORACLE_GAMMA,
    DGP_ORACLE_GAMMA,
)
BLEARNER_GAMMA_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
BLEARNER_BINARY_SEARCH_GRID = (1.0, 2.0, 3.0, 4.0, 5.0)
BLEARNER_BINARY_SEARCH_GAMMA_TOLERANCE = 0.01
BLEARNER_BINARY_SEARCH_WIDTH_TOLERANCE = 0.001
BLEARNER_BINARY_SEARCH_MAX_ITERATIONS = 10
BLEARNER_COMMIT = "734fd7639da54554e41d944bfc2187a24e75cf81"
_VALIDATED_BLEARNER_REPOSITORIES: set[Path] = set()


@dataclass(frozen=True)
class BenchmarkFitConfig:
    """Runtime settings shared by B-Learner and CATENets benchmark fits."""

    seed: int
    n_jobs: int
    worker_threads: int
    n_layers_out: int
    n_units_out: int
    n_iter: int
    batch_size: int
    validation_fraction: float
    n_iter_print: int
    blearner_repository: str
    verbose: bool = False


@dataclass(frozen=True)
class FitTask:
    """One independent WB or CB endpoint regression."""

    task_id: int
    study: str
    kind: Literal["wb", "cb"]
    endpoint: Literal["lower", "upper"]
    seed: int
    x_left: np.ndarray
    a_left: np.ndarray
    y_left: np.ndarray
    x_target: np.ndarray
    environment: str | None = None
    method: Literal["cb_pi", "cb_ra", "cb_ipw", "cb_dr"] | None = None
    left_environment: str | None = None
    right_environment: str | None = None
    x_right: np.ndarray | None = None
    a_right: np.ndarray | None = None
    y_right: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate data required by the selected benchmark learner."""

        if self.kind == "wb":
            if self.environment is None:
                raise ValueError("WB tasks require environment")
            return
        if self.kind != "cb":
            raise ValueError("kind must be wb or cb")
        if self.method not in META_LEARNER_LABELS:
            raise ValueError(
                f"CB tasks require one of {tuple(META_LEARNER_LABELS)}"
            )
        if self.left_environment is None or self.right_environment is None:
            raise ValueError("CB tasks require both environment labels")
        if any(
            value is None
            for value in (self.x_right, self.a_right, self.y_right)
        ):
            raise ValueError("CB tasks require right-environment arrays")


@dataclass(frozen=True)
class FitResult:
    """Predictions and runtime from one completed benchmark fit."""

    task_id: int
    predictions: np.ndarray
    runtime_seconds: float


@dataclass(frozen=True)
class BLearnerFitTask:
    """One independent B-Learner fit for a single study and Gamma."""

    task_id: int
    study: str
    gamma: float
    x: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    x_target: np.ndarray


@dataclass(frozen=True)
class BLearnerFitResult:
    """Raw endpoint predictions and runtime from one B-Learner fit."""

    task_id: int
    lower: np.ndarray
    upper: np.ndarray
    runtime_seconds: float


class _PaperPWLearner(PWLearner):
    """CATENets 0.2.4 PW learner with tensor placeholders."""

    def _first_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        fit_mask: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit the environment propensity without NumPy placeholder errors."""

        del y
        propensity = self._impute_propensity(X, w, fit_mask, pred_mask).squeeze()
        unused = torch.full_like(propensity, torch.nan, device=DEVICE)
        return unused, unused.clone(), propensity.to(DEVICE)


def wb_pseudo_outcome(
    a: np.ndarray,
    y: np.ndarray,
    endpoint: Literal["lower", "upper"],
) -> np.ndarray:
    """Construct the binary-outcome WB pseudo-outcome."""

    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)
    if endpoint == "upper":
        return 1.0 - a * (1.0 - y) - (1.0 - a) * y
    if endpoint == "lower":
        return a * y - ((1.0 - a) * y + a)
    raise ValueError("endpoint must be lower or upper")


def cb_pseudo_outcome(
    a: np.ndarray,
    y: np.ndarray,
    leading: np.ndarray,
    endpoint: Literal["lower", "upper"],
) -> np.ndarray:
    """Construct the directional CB pseudo-outcome for two environments."""

    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)
    leading = np.asarray(leading, dtype=bool)
    if endpoint == "upper":
        return np.where(leading, 1.0 - a * (1.0 - y), (1.0 - a) * y)
    if endpoint == "lower":
        return np.where(leading, a * y, (1.0 - a) * y + a)
    raise ValueError("endpoint must be lower or upper")


def validate_blearner_repository(repository: Path) -> None:
    """Validate a pinned upstream B-Learner checkout once per process."""

    repository = repository.resolve()
    if repository in _VALIDATED_BLEARNER_REPOSITORIES:
        return
    if not (repository / "models" / "blearner" / "BLearner.py").exists():
        raise FileNotFoundError(
            f"Missing pinned B-Learner checkout at {repository}. "
            f"Expected commit {BLEARNER_COMMIT}."
        )
    commit = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != BLEARNER_COMMIT:
        raise ValueError(
            f"Expected B-Learner commit {BLEARNER_COMMIT}, found "
            f"{commit} at {repository}"
        )
    _VALIDATED_BLEARNER_REPOSITORIES.add(repository)


def make_blearner(
    config: BenchmarkFitConfig,
    gamma: float,
    *,
    model_n_jobs: int = -1,
) -> Any:
    """Construct the pinned notebook-faithful B-Learner for one Gamma.

    Args:
        config: Shared benchmark fit settings, including the pinned checkout
            and the seed offsets that make every nuisance reproducible.
        gamma: Sensitivity parameter at least one.
        model_n_jobs: Worker count given to every nuisance forest. The default
            ``-1`` keeps the historical single-fit behaviour of using all
            available cores. Callers that fit several B-Learners concurrently
            pass ``1`` so the outer pool owns the cores. Forest predictions are
            invariant to this setting because each tree is seeded from
            ``random_state`` before any work is dispatched.

    Returns:
        Unfitted B-Learner from the pinned upstream repository.
    """

    repository = Path(config.blearner_repository).resolve()
    validate_blearner_repository(repository)
    loaded_models = sys.modules.get("models")
    loaded_models_path = getattr(loaded_models, "__file__", None)
    if loaded_models_path is not None and repository not in Path(
        loaded_models_path
    ).resolve().parents:
        raise RuntimeError(
            "A different top-level 'models' package is already loaded from "
            f"{loaded_models_path}; restart with {repository} first on sys.path."
        )
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    if gamma < 1.0:
        raise ValueError("gamma must be at least one")
    blearner = importlib.import_module("models.blearner")
    nuisance = importlib.import_module("models.blearner.nuisance")
    tau = gamma / (1.0 + gamma)

    def forest(offset: int) -> Any:
        """Construct one seeded B-Learner nuisance forest."""

        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=0.01,
            max_depth=6,
            n_jobs=int(model_n_jobs),
            random_state=config.seed + offset,
        )

    return blearner.BLearner(
        propensity_model=LogisticRegression(
            C=1.0,
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            max_iter=10_000,
            random_state=config.seed,
        ),
        quantile_plus_model=RandomForestQuantileRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=0.01,
            n_jobs=int(model_n_jobs),
            q=tau,
            random_state=config.seed + 1,
        ),
        quantile_minus_model=RandomForestQuantileRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=0.01,
            n_jobs=int(model_n_jobs),
            q=1.0 - tau,
            random_state=config.seed + 2,
        ),
        mu_model=forest(3),
        cvar_plus_model=nuisance.KernelSuperquantileRegressor(
            nuisance.RFKernel(forest(4)), tau=tau, tail="right"
        ),
        cvar_minus_model=nuisance.KernelSuperquantileRegressor(
            nuisance.RFKernel(forest(5)), tau=1.0 - tau, tail="left"
        ),
        cate_bounds_model=RegressionForest(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=0.01,
            n_jobs=int(model_n_jobs),
            random_state=config.seed + 6,
        ),
        use_rho=True,
        gamma=gamma,
        cv=3,
        random_state=config.seed,
    )


def resolve_n_jobs(requested: int) -> int:
    """Resolve a worker count against the current CPU affinity mask."""

    if requested > 0:
        return requested
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, int(available))


def _set_worker_thread_environment(worker_threads: int) -> None:
    """Set numerical-library thread caps inherited by fit workers."""

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = str(worker_threads)


def _initialize_fit_worker(worker_threads: int) -> None:
    """Configure Torch and BLAS threading inside one fit worker."""

    _set_worker_thread_environment(worker_threads)
    torch.set_num_threads(int(worker_threads))
    torch.set_num_interop_threads(1)


def _initialize_blearner_worker(worker_threads: int) -> None:
    """Cap numerical-library threads inside one B-Learner fit worker.

    B-Learner nuisances are scikit-learn and econml estimators that never touch
    Torch, so this deliberately leaves Torch's interop pool alone: that setting
    is rejected once parallel work has started, which would abort an otherwise
    valid in-process batch.
    """

    _set_worker_thread_environment(worker_threads)


def _fit_task(task: FitTask, network: Mapping[str, Any]) -> FitResult:
    """Fit one WB/CB CATENets task and predict its target strata."""

    start = time.perf_counter()
    kwargs = {
        "n_unit_in": int(task.x_left.shape[1]),
        "binary_y": False,
        "seed": int(task.seed),
        **network,
    }
    if task.kind == "wb":
        model = BasicNet(name=f"wb_{task.endpoint}_{task.task_id}", **kwargs)
        model.fit(
            task.x_left,
            wb_pseudo_outcome(task.a_left, task.y_left, task.endpoint).astype(
                np.float32
            ),
        )
        model.eval()
        with torch.no_grad():
            prediction = model(
                torch.as_tensor(
                    task.x_target,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            )
    else:
        assert task.method is not None
        assert task.x_right is not None
        assert task.a_right is not None
        assert task.y_right is not None
        learner_class = {
            "cb_pi": TLearner,
            "cb_ra": RALearner,
            "cb_ipw": _PaperPWLearner,
            "cb_dr": DRLearner,
        }[task.method]
        x = np.concatenate((task.x_left, task.x_right), axis=0)
        a = np.concatenate((task.a_left, task.a_right), axis=0)
        y = np.concatenate((task.y_left, task.y_right), axis=0)
        leading = np.concatenate(
            (
                np.ones(len(task.x_left), dtype=bool),
                np.zeros(len(task.x_right), dtype=bool),
            )
        )
        learner = learner_class(**kwargs)
        learner.fit(
            x,
            cb_pseudo_outcome(a, y, leading, task.endpoint).astype(np.float32),
            leading.astype(np.int64),
        )
        prediction = learner.predict(task.x_target)
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    return FitResult(
        task.task_id,
        np.asarray(prediction, dtype=float).reshape(-1),
        time.perf_counter() - start,
    )


def run_fit_tasks(
    tasks: Sequence[FitTask],
    config: BenchmarkFitConfig,
) -> tuple[dict[int, FitResult], pd.DataFrame]:
    """Execute independent WB/CB fits with the configured worker limits."""

    if not tasks:
        return {}, pd.DataFrame()
    network = {
        "n_layers_out": config.n_layers_out,
        "n_units_out": config.n_units_out,
        "n_iter": config.n_iter,
        "batch_size": config.batch_size,
        "val_split_prop": config.validation_fraction,
        "n_iter_print": config.n_iter_print,
    }
    workers = min(resolve_n_jobs(config.n_jobs), len(tasks))
    _set_worker_thread_environment(config.worker_threads)
    results: dict[int, FitResult] = {}
    if workers == 1:
        _initialize_fit_worker(config.worker_threads)
        for task in tasks:
            result = _fit_task(task, network)
            results[result.task_id] = result
            if config.verbose:
                print(
                    f"Completed fit {len(results)}/{len(tasks)}: "
                    f"task={task.task_id}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_fit_worker,
            initargs=(config.worker_threads,),
        ) as executor:
            futures: dict[Future[FitResult], FitTask] = {
                executor.submit(_fit_task, task, network): task for task in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                results[result.task_id] = result
                if config.verbose:
                    print(
                        f"Completed fit {completed}/{len(tasks)}: "
                        f"task={task.task_id} ({task.study}, {task.kind})",
                        flush=True,
                    )
    diagnostics = pd.DataFrame(
        [
            {
                "task_id": task.task_id,
                "study": task.study,
                "kind": task.kind,
                "method": task.method,
                "endpoint": task.endpoint,
                "environment": task.environment,
                "left_environment": task.left_environment,
                "right_environment": task.right_environment,
                "seed": task.seed,
                "runtime_seconds": results[task.task_id].runtime_seconds,
            }
            for task in tasks
        ]
    )
    return results, diagnostics


def fit_blearner_task(
    task: BLearnerFitTask,
    config: BenchmarkFitConfig,
) -> BLearnerFitResult:
    """Fit one B-Learner and predict its target rows.

    Every nuisance forest is built with ``model_n_jobs=1`` so the result never
    depends on how many fits run concurrently, and so the enclosing pool owns
    the cores. Forests this small (hundreds of rows) fit faster single-threaded
    than with joblib dispatch anyway.

    Args:
        task: Fitting arrays, target rows, and the Gamma for one fit.
        config: Shared benchmark fit settings, including the pinned checkout.

    Returns:
        Raw lower and upper endpoint predictions with the fit runtime.
    """

    start = time.perf_counter()
    learner = make_blearner(config, float(task.gamma), model_n_jobs=1)
    learner.fit(task.x, task.treatment, task.outcome)
    lower, upper = learner.effect(task.x_target)
    return BLearnerFitResult(
        task_id=int(task.task_id),
        lower=np.asarray(lower, dtype=float).reshape(-1),
        upper=np.asarray(upper, dtype=float).reshape(-1),
        runtime_seconds=time.perf_counter() - start,
    )


def run_blearner_fit_tasks(
    tasks: Sequence[BLearnerFitTask],
    config: BenchmarkFitConfig,
) -> dict[int, BLearnerFitResult]:
    """Execute independent B-Learner fits across spawned worker processes.

    B-Learner fits over a Gamma grid are mutually independent and individually
    single-threaded, so a Gamma-by-study grid is the natural unit of work here.
    Results are keyed by task id and are invariant to the worker count: each
    fit is seeded from ``config.seed`` and always uses single-threaded forests.

    Args:
        tasks: Independent fits to execute.
        config: Shared benchmark fit settings. ``n_jobs`` caps the pool and
            ``worker_threads`` caps numerical-library threads inside workers.

    Returns:
        Completed fit results keyed by task id.
    """

    if not tasks:
        return {}
    workers = min(resolve_n_jobs(config.n_jobs), len(tasks))
    _initialize_blearner_worker(config.worker_threads)
    start = time.perf_counter()
    results: dict[int, BLearnerFitResult] = {}
    if workers == 1:
        for task in tasks:
            result = fit_blearner_task(task, config)
            results[result.task_id] = result
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_blearner_worker,
            initargs=(config.worker_threads,),
        ) as executor:
            futures: dict[Future[BLearnerFitResult], BLearnerFitTask] = {
                executor.submit(fit_blearner_task, task, config): task
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                results[result.task_id] = result
    if config.verbose:
        print(
            f"Completed {len(tasks)} B-Learner fits with n_jobs={workers} "
            f"in {time.perf_counter() - start:.1f}s.",
            flush=True,
        )
    return results


def parse_gamma(value: str) -> float | str:
    """Parse a fixed B-Learner Gamma or supported selection strategy."""

    if value in BLEARNER_GAMMA_STRATEGIES:
        return value
    try:
        gamma = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "gamma must be 'width_matched', 'width_matched_binary', "
            "'exp_beta', 'oracle', 'dgp_oracle', or a number at least one"
        ) from error
    if not np.isfinite(gamma) or gamma < 1.0:
        raise argparse.ArgumentTypeError(
            "gamma must be 'width_matched', 'width_matched_binary', "
            "'exp_beta', 'oracle', 'dgp_oracle', or a number at least one"
        )
    return gamma


def parse_gamma_search_grid(value: str) -> tuple[float, ...]:
    """Parse a comma-separated coarse Gamma search grid."""

    try:
        return tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "gamma-search-grid must be a comma-separated numeric list"
        ) from error


def blearner_gammas(
    gamma: float | str,
    *,
    beta: float,
    gamma_search_grid: Sequence[float],
) -> tuple[float, ...]:
    """Return the B-Learner Gamma values required by one strategy."""

    if gamma == WIDTH_MATCHED_GAMMA:
        return BLEARNER_GAMMA_GRID
    if gamma == WIDTH_MATCHED_BINARY_GAMMA:
        return tuple(float(value) for value in gamma_search_grid)
    if gamma == EXP_BETA_GAMMA:
        return (float(np.exp(beta)),)
    if gamma in (ORACLE_GAMMA, DGP_ORACLE_GAMMA):
        return ()
    return (float(gamma),)


def format_beta(beta: float) -> str:
    """Format one beta value for stable artifact names."""

    value = float(beta)
    return f"{value:.1f}" if value.is_integer() else f"{value:g}"


def gamma_output_label(gamma: float | str) -> str:
    """Return a filesystem-safe label for one B-Learner Gamma rule."""

    if isinstance(gamma, (int, float)):
        numeric = f"{float(gamma):g}".replace(".", "p")
        return f"fixed_{numeric}"
    return str(gamma)


def comparison_to_long_intervals(
    comparison: pd.DataFrame,
    *,
    methods: Sequence[str],
    identity_columns: Sequence[str],
    output_reference_columns: Mapping[str, str] | None = None,
    replication: int,
    seed: int,
    bias_covariate: str,
    beta: float,
    config_fingerprint_value: str,
    experiment_name: str,
) -> pd.DataFrame:
    """Convert a wide Exp18 comparison into normalized method-level rows.

    Args:
        comparison: Wide Exp18 comparison table.
        methods: Output method names, including ``naturalness`` and
            ``blearner``.
        identity_columns: Existing comparison columns retained in every row.
        output_reference_columns: Mapping from source reference columns to
            their normalized output names.
        replication: Monte Carlo replication index.
        seed: Realization seed.
        bias_covariate: OSRCT selection covariate.
        beta: OSRCT selection strength.
        config_fingerprint_value: Scientific-configuration fingerprint.
        experiment_name: Name included in missing-schema errors.

    Returns:
        One normalized row per input stratum and benchmark method.
    """

    output_reference_columns = dict(output_reference_columns or {})
    required = set(identity_columns)
    if "naturalness" in methods:
        required.update(
            {
                "naturalness_lower",
                "naturalness_upper",
                "lp_posterior_prob_optimal",
            }
        )
    if "blearner" in methods:
        required.update(
            {
                "blearner_lower",
                "blearner_upper",
                "blearner_gamma",
                "outside_binary_cate_support",
            }
        )
    for method in ("cb_ipw", "cb_dr"):
        if method in methods:
            required.update(
                {
                    f"{method}_lower",
                    f"{method}_upper",
                    f"{method}_used_full_range_fallback",
                    f"{method}_interval_inverted",
                }
            )
    missing = required.difference(comparison.columns)
    if missing:
        raise ValueError(
            f"Exp18 comparison is missing {experiment_name} fields: "
            f"{sorted(missing)}"
        )

    frames: list[pd.DataFrame] = []
    for method in methods:
        frame = comparison.loc[:, list(identity_columns)].copy()
        frame["replication"] = int(replication)
        frame["seed"] = int(seed)
        frame["bias_covariate"] = str(bias_covariate)
        frame["beta"] = float(beta)
        frame["method"] = method
        frame["config_fingerprint"] = config_fingerprint_value
        frame["blearner_gamma"] = (
            pd.to_numeric(comparison["blearner_gamma"], errors="coerce")
            if method == "blearner"
            else np.nan
        )
        frame["lp_posterior_prob_optimal"] = (
            pd.to_numeric(comparison["lp_posterior_prob_optimal"], errors="coerce")
            if method == "naturalness"
            else np.nan
        )
        if method == "naturalness":
            raw_lower = pd.to_numeric(comparison["naturalness_lower"], errors="coerce")
            raw_upper = pd.to_numeric(comparison["naturalness_upper"], errors="coerce")
            effective_lower, effective_upper = raw_lower, raw_upper
            fallback = pd.Series(False, index=comparison.index)
            no_optimal = pd.to_numeric(
                comparison["lp_posterior_prob_optimal"], errors="coerce"
            ).fillna(0.0).le(0.0)
            support_clipped = pd.Series(False, index=comparison.index)
        elif method == "blearner":
            raw_lower = pd.to_numeric(comparison["blearner_lower"], errors="coerce")
            raw_upper = pd.to_numeric(comparison["blearner_upper"], errors="coerce")
            effective_lower = raw_lower.clip(lower=-1.0, upper=1.0)
            effective_upper = raw_upper.clip(lower=-1.0, upper=1.0)
            fallback = pd.Series(False, index=comparison.index)
            no_optimal = pd.Series(False, index=comparison.index)
            support_clipped = (
                raw_lower.lt(-1.0)
                | raw_lower.gt(1.0)
                | raw_upper.lt(-1.0)
                | raw_upper.gt(1.0)
                | comparison["outside_binary_cate_support"].astype(bool)
            )
        else:
            raw_lower_column = f"{method}_lower_raw"
            raw_upper_column = f"{method}_upper_raw"
            raw_lower = pd.to_numeric(
                comparison[
                    raw_lower_column
                    if raw_lower_column in comparison
                    else f"{method}_lower"
                ],
                errors="coerce",
            )
            raw_upper = pd.to_numeric(
                comparison[
                    raw_upper_column
                    if raw_upper_column in comparison
                    else f"{method}_upper"
                ],
                errors="coerce",
            )
            effective_lower = pd.to_numeric(comparison[f"{method}_lower"], errors="coerce")
            effective_upper = pd.to_numeric(comparison[f"{method}_upper"], errors="coerce")
            fallback = comparison[f"{method}_used_full_range_fallback"].astype(bool)
            fallback |= comparison[f"{method}_interval_inverted"].astype(bool)
            no_optimal = pd.Series(False, index=comparison.index)
            support_clipped = pd.Series(False, index=comparison.index)

        finite = (
            np.isfinite(effective_lower)
            & np.isfinite(effective_upper)
            & effective_lower.le(effective_upper)
        )
        reason = pd.Series("", index=comparison.index, dtype="object")
        reason.loc[~finite] = "missing_or_nonfinite_endpoints"
        reason.loc[no_optimal] = "no_optimal_lp_draw"
        reason.loc[fallback] = "full_range_fallback"
        frame["raw_lower"] = raw_lower.to_numpy(dtype=float)
        frame["raw_upper"] = raw_upper.to_numpy(dtype=float)
        frame["lower"] = effective_lower.to_numpy(dtype=float)
        frame["upper"] = effective_upper.to_numpy(dtype=float)
        frame["width"] = (effective_upper - effective_lower).to_numpy(dtype=float)
        frame["is_valid"] = (finite & ~fallback & ~no_optimal).to_numpy(dtype=bool)
        frame["exclusion_reason"] = reason.to_numpy(dtype=object)
        frame["used_full_range_fallback"] = fallback.to_numpy(dtype=bool)
        frame["support_clipped"] = support_clipped.to_numpy(dtype=bool)
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True, sort=False)
    result = result.rename(columns=dict(output_reference_columns))
    output_identity_columns = [
        output_reference_columns.get(column, column)
        for column in identity_columns
    ]
    ordered = [
        "config_fingerprint",
        "replication",
        "seed",
        "bias_covariate",
        "beta",
        *output_identity_columns,
        "method",
        "raw_lower",
        "raw_upper",
        "lower",
        "upper",
        "width",
        "is_valid",
        "exclusion_reason",
        "used_full_range_fallback",
        "support_clipped",
        "blearner_gamma",
        "lp_posterior_prob_optimal",
    ]
    return result.loc[:, ordered + [
        column for column in result.columns if column not in ordered
    ]]


def build_run_diagnostics(
    intervals: pd.DataFrame,
    *,
    grouping_columns: Sequence[str],
    total_label: str,
    runtime_seconds: float,
) -> pd.DataFrame:
    """Summarize one completed benchmark condition by method."""

    rows: list[dict[str, Any]] = []
    for keys, frame in intervals.groupby(list(grouping_columns), sort=True, dropna=False):
        metadata = dict(zip(grouping_columns, keys))
        valid = frame.loc[frame["is_valid"].astype(bool)]
        rows.append(
            {
                **metadata,
                "status": "completed",
                "error_type": None,
                "error": None,
                f"n_total_{total_label}": int(len(frame)),
                f"n_valid_{total_label}": int(len(valid)),
                f"n_excluded_{total_label}": int(len(frame) - len(valid)),
                "n_full_range_fallbacks": int(frame["used_full_range_fallback"].astype(bool).sum()),
                "n_support_clipped": int(frame["support_clipped"].astype(bool).sum()),
                "mean_valid_width": float(valid["width"].mean()) if not valid.empty else np.nan,
                "selected_gamma": (
                    float(valid["blearner_gamma"].dropna().iloc[0])
                    if metadata["method"] == "blearner"
                    and valid["blearner_gamma"].notna().any()
                    else np.nan
                ),
                "mean_lp_posterior_prob_optimal": (
                    float(pd.to_numeric(valid["lp_posterior_prob_optimal"], errors="coerce").mean())
                    if metadata["method"] == "naturalness" and not valid.empty
                    else np.nan
                ),
                "condition_runtime_seconds": float(runtime_seconds),
            }
        )
    return pd.DataFrame(rows)


def require_parquet_engine(experiment_name: str) -> None:
    """Raise a targeted error when PyArrow is unavailable."""

    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise ImportError(
            f"{experiment_name} Parquet output requires pyarrow. Install "
            "project dependencies before running the experiment."
        ) from error


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace one Parquet table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".parquet",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=False, engine="pyarrow")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_existing_outputs(
    output_dir: Path,
    *,
    intervals_filename: str,
    diagnostics_filename: str,
    fingerprint: str,
    resume: bool,
    experiment_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and reconcile a pair of committed Monte Carlo tables."""

    intervals_path = output_dir / intervals_filename
    diagnostics_path = output_dir / diagnostics_filename
    paths_exist = (intervals_path.exists(), diagnostics_path.exists())
    if any(paths_exist) and not all(paths_exist):
        if not resume:
            raise FileExistsError(
                f"Partial {experiment_name} outputs exist and --no-resume "
                "was requested"
            )
        orphan_path = (
            intervals_path if intervals_path.exists() else diagnostics_path
        )
        orphan = pd.read_parquet(orphan_path, engine="pyarrow")
        observed = set(
            orphan.get("config_fingerprint", pd.Series(dtype=str))
            .dropna()
            .astype(str)
        )
        if observed != {fingerprint}:
            raise ValueError(
                f"Partial {experiment_name} output belongs to a different "
                f"{experiment_name} config"
            )
        orphan_path.unlink()
        return pd.DataFrame(), pd.DataFrame()
    if not all(paths_exist):
        return pd.DataFrame(), pd.DataFrame()
    if not resume:
        raise FileExistsError(
            f"{experiment_name} outputs already exist and --no-resume was "
            "requested"
        )

    intervals = pd.read_parquet(intervals_path, engine="pyarrow")
    diagnostics = pd.read_parquet(diagnostics_path, engine="pyarrow")
    for name, frame in (("intervals", intervals), ("diagnostics", diagnostics)):
        if "config_fingerprint" not in frame:
            raise ValueError(
                f"Existing {name} table lacks config_fingerprint"
            )
        observed = set(frame["config_fingerprint"].dropna().astype(str))
        if observed != {fingerprint}:
            raise ValueError(
                f"Existing {name} table belongs to a different "
                f"{experiment_name} config"
            )
    interval_reps = set(intervals["replication"].astype(int))
    diagnostic_reps = set(diagnostics["replication"].astype(int))
    common_reps = interval_reps & diagnostic_reps
    if interval_reps != diagnostic_reps:
        intervals = intervals.loc[
            intervals["replication"].astype(int).isin(common_reps)
        ].copy()
        diagnostics = diagnostics.loc[
            diagnostics["replication"].astype(int).isin(common_reps)
        ].copy()
        atomic_write_parquet(intervals, intervals_path)
        atomic_write_parquet(diagnostics, diagnostics_path)
    return intervals, diagnostics


def completed_replications(
    diagnostics: pd.DataFrame,
    *,
    condition_columns: Sequence[str],
    expected_conditions: int,
) -> set[int]:
    """Return replications containing every successful condition row."""

    if diagnostics.empty:
        return set()
    completed: set[int] = set()
    for replication, rows in diagnostics.groupby("replication", sort=True):
        conditions = rows.loc[:, list(condition_columns)].drop_duplicates()
        if (
            len(conditions) == expected_conditions
            and rows["status"].astype(str).eq("completed").all()
        ):
            completed.add(int(replication))
    return completed


def commit_replication(
    output_dir: Path,
    existing_intervals: pd.DataFrame,
    existing_diagnostics: pd.DataFrame,
    replication_intervals: pd.DataFrame,
    replication_diagnostics: pd.DataFrame,
    *,
    interval_sort_columns: Sequence[str],
    diagnostics_sort_columns: Sequence[str],
    intervals_filename: str,
    diagnostics_filename: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atomically insert or replace one complete replication."""

    replication = int(replication_intervals["replication"].iloc[0])
    if not existing_intervals.empty:
        existing_intervals = existing_intervals.loc[
            existing_intervals["replication"].astype(int).ne(replication)
        ].copy()
    if not existing_diagnostics.empty:
        existing_diagnostics = existing_diagnostics.loc[
            existing_diagnostics["replication"].astype(int).ne(replication)
        ].copy()
    intervals = pd.concat(
        [existing_intervals, replication_intervals],
        ignore_index=True,
        sort=False,
    ).sort_values(list(interval_sort_columns)).reset_index(drop=True)
    diagnostics = pd.concat(
        [existing_diagnostics, replication_diagnostics],
        ignore_index=True,
        sort=False,
    ).sort_values(list(diagnostics_sort_columns)).reset_index(drop=True)
    atomic_write_parquet(intervals, output_dir / intervals_filename)
    atomic_write_parquet(diagnostics, output_dir / diagnostics_filename)
    return intervals, diagnostics


def width_distribution_annotation(
    panel: pd.DataFrame,
    *,
    methods: Sequence[str],
    method_labels: Mapping[str, str],
) -> str:
    """Format method-specific valid counts for one plot panel."""

    lines = []
    for method in methods:
        rows = panel.loc[panel["method"].eq(method)]
        valid = int(rows["is_valid"].astype(bool).sum())
        lines.append(f"{method_labels[method]}: {valid}/{len(rows)}")
    return "valid n\n" + "\n".join(lines)


def draw_width_violin_panel(
    axis: Any,
    panel: pd.DataFrame,
    *,
    methods: Sequence[str],
    method_labels: Mapping[str, str],
    method_colors: Mapping[str, str],
    maximum: float,
    axis_label: str,
) -> None:
    """Draw method-specific horizontal width distributions on one axis."""

    for method_index, method in enumerate(methods):
        rows = panel.loc[
            panel["method"].eq(method) & panel["is_valid"].astype(bool)
        ]
        widths = pd.to_numeric(rows["width"], errors="coerce").dropna()
        if len(widths) >= 2 and widths.nunique() >= 2:
            violin = axis.violinplot(
                [widths.to_numpy()],
                positions=[method_index],
                orientation="horizontal",
                widths=0.75,
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            for body in violin["bodies"]:
                body.set_facecolor(method_colors[method])
                body.set_edgecolor(method_colors[method])
                body.set_alpha(0.6)
            for component in ("cbars", "cmins", "cmaxes", "cmedians"):
                violin[component].set_color(method_colors[method])
        elif not widths.empty:
            axis.scatter(
                widths.iloc[0],
                method_index,
                color=method_colors[method],
                marker="D",
                s=28,
                zorder=3,
            )
    axis.set_xlabel(axis_label)
    axis.set_yticks(range(len(methods)))
    axis.set_yticklabels([method_labels[method] for method in methods])
    axis.set_xlim(0.0, maximum)
    axis.text(
        0.98,
        0.96,
        width_distribution_annotation(
            panel,
            methods=methods,
            method_labels=method_labels,
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )


def constant_references(
    rows: pd.DataFrame,
    *,
    keys: Sequence[str],
    reference_columns: Sequence[str],
    reference_label: str,
) -> pd.DataFrame:
    """Validate fixed evaluation references and return unique strata."""

    records: list[dict[str, Any]] = []
    for group_key, group in rows.groupby(list(keys), sort=True, dropna=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        record = dict(zip(keys, key_tuple))
        for column in reference_columns:
            values = group[column].dropna().unique()
            if len(values) > 1:
                raise ValueError(
                    f"{reference_label} {column} changes across MC runs"
                )
            record[column] = values[0] if len(values) == 1 else np.nan
        records.append(record)
    return pd.DataFrame(records)


def build_mean_bounds_comparison(
    intervals: pd.DataFrame,
    *,
    bias_covariate: str,
    beta: float,
    keys: Sequence[str],
    reference_columns: Sequence[str],
    reference_label: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Build a wide table of valid Monte Carlo mean endpoints."""

    selected = intervals.loc[
        intervals["bias_covariate"].astype(str).eq(str(bias_covariate))
        & np.isclose(
            pd.to_numeric(intervals["beta"], errors="coerce"),
            float(beta),
        )
    ].copy()
    result = constant_references(
        selected,
        keys=keys,
        reference_columns=reference_columns,
        reference_label=reference_label,
    )
    valid = selected.loc[selected["is_valid"].astype(bool)].copy()
    means = (
        valid.groupby([*keys, "method"], as_index=False, dropna=False)
        .agg(
            lower=("lower", "mean"),
            upper=("upper", "mean"),
            n_mc_valid=("replication", "nunique"),
        )
    )
    for method in methods:
        current = means.loc[means["method"].eq(method)].drop(columns="method")
        current = current.rename(
            columns={
                "lower": f"{method}_lower",
                "upper": f"{method}_upper",
                "n_mc_valid": f"{method}_n_mc_valid",
            }
        )
        result = result.merge(
            current,
            on=list(keys),
            how="outer",
            validate="one_to_one",
        )
    for method in ("cb_ipw", "cb_dr"):
        if method in methods:
            result[f"{method}_interval_inverted"] = False
    return result
