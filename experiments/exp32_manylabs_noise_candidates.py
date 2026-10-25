"""Experiment 32: pure-noise naturalness candidates on ManyLabs.

An ablation of Exp25, reported in the paper as part of Experiment I, that adds
``n_noise`` binary candidates ``N_k`` independent of treatment, outcome,
covariates, site and OSRCT selection. Their population relevance ``D_k``,
exclusion ``I_k`` and tolerances ``c``, ``d`` are all 0, so every trigger fire
is false; such a fire can still move a bound through spurious relevance in the
target plug-in or by displacing real candidates. Each ``N_k`` is conditioned
on the three Exp25 covariates only, whose eight strata carry the bounds; adding
the other, independent, noise columns would only fragment these strata.

Routes: ``separate`` fits Exp16 once per noise candidate, as the only
candidate, and understates downstream harm (a candidate that never fires at a
source site gets no tolerance, so it cannot reach the target LP); ``joint``
fits the real and all noise candidates with one shared GP, through which
tolerances pass between the two kinds; ``baseline`` reruns Exp25 without noise.
Runs reuse an Exp25 replication's persisted OSRCT draw, Exp16 settings and
target index (the store is only read), so each is paired with the stored
noise-free result or, with ``tau_floor``, with this root's ``baseline`` route
at the same floor.
"""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from causal_grounding.paper_grids import floor_taus
from experiments.benchmarks import atomic_write_parquet, format_beta
from experiments.osrct_target_pipeline import Exp16Config, run_one_target
from experiments.exp25_osrct_mc_baseline_comparison import (
    Exp25Config,
    _exp16_config_for_replication,
)
from experiments.interval_stats import wilson_interval
from experiments.mc_store import (
    exp25_target_index,
    load_exp25_store_config,
    parquet_safe,
)
from experiments.osrct_data_utils import load_prepared_data
from experiments.utils import json_default, write_run_manifest


EXP32_NAME = "exp32_manylabs_noise_candidates"
ROUTES = ("separate", "joint", "baseline")
NOISE_PREFIX = "noise_"
#: Exp16 outputs persisted per run, under their stored names. Posterior
#: tolerance and LP draws are omitted: no summary reads them.
PERSISTED_OUTPUTS = {
    "final_source_naturalness_observations": "source_observations",
    "target_trigger_statistics": "target_trigger_statistics",
    "target_policy_candidates": "target_candidates",
    "target_policy_choices": "target_choices",
    "target_gp_posterior_summary": "target_posterior_summary",
    "target_lp_bounds": "target_lp_bounds",
    "run_summary": "run_summary",
    "gp_diagnostics": "gp_diagnostics",
    "source_site_roles": "source_site_roles",
}
COMPLETE_MARKER = "complete.json"


@dataclass(frozen=True)
class Exp32Config:
    """Configuration for the noise-candidate ablation.

    Args:
        exp25_store: Exp25 backend leaf holding ``replications/`` and
            ``run_manifest.yml``; read only.
        output_dir: Fresh result root; one subdirectory per route.
        study: ManyLabs study to ablate.
        patterns: OSRCT selection patterns to run.
        beta: OSRCT selection strength; must exist in the store.
        replications: Replication indices to run; ``None`` runs all.
        route: ``separate``, ``joint`` or ``baseline``.
        n_noise: Number of pure-noise candidates.
        noise_prevalence: ``P(N_k = 1)`` for every noise column.
        noise_seed: Salt combined with each replication seed so noise differs
            across replications but not across patterns or routes.
        tabicl_share_checkpoint: Share one loaded TabICLv2 checkpoint across
            fits. Predictions are unchanged.
        lp_posterior_samples: Posterior LP draws; ``None`` uses the store's.
        lp_posterior_n_jobs: LP posterior threads; ``None`` uses the store's.
        tabicl_device: TabICLv2 device; ``None`` uses the store's.
        tau_floor: Smallest relevance threshold ``tau`` kept in the Exp16 grid;
            ``None`` keeps the store's grid.
        resume: Skip runs whose completion marker exists.
        summarize: Write the summary tables after running.
        verbose: Print progress.
        cluster_manifest_path: Optional Kubernetes job YAML embedded in the
            run manifest.
        command_argv: Command-line arguments recorded in provenance.
    """

    exp25_store: str
    output_dir: str
    study: str = "allowedforbidden"
    patterns: tuple[str, ...] = ("age", "gender", "polideo")
    beta: float = 2.0
    replications: tuple[int, ...] | None = None
    route: str = "separate"
    n_noise: int = 15
    noise_prevalence: float = 0.5
    noise_seed: int = 32
    tabicl_share_checkpoint: bool = True
    lp_posterior_samples: int | None = None
    lp_posterior_n_jobs: int | None = None
    tabicl_device: str | None = None
    tau_floor: float | None = None
    resume: bool = True
    summarize: bool = True
    verbose: bool = True
    cluster_manifest_path: str | None = None
    command_argv: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the route, noise design and replication selection."""

        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}")
        if self.route != "baseline" and self.n_noise < 1:
            raise ValueError("n_noise must be positive for noise routes")
        if not 0.0 < float(self.noise_prevalence) < 1.0:
            raise ValueError("noise_prevalence must lie strictly between 0 and 1")
        if not self.patterns or len(set(self.patterns)) != len(self.patterns):
            raise ValueError("patterns must be nonempty and unique")
        if self.replications is not None and (
            not self.replications
            or len(set(self.replications)) != len(self.replications)
            or any(int(index) < 0 for index in self.replications)
        ):
            raise ValueError("replications must be distinct nonnegative indices")
        if self.lp_posterior_samples is not None and self.lp_posterior_samples < 1:
            raise ValueError("lp_posterior_samples must be positive")
        if self.lp_posterior_n_jobs is not None and self.lp_posterior_n_jobs < 1:
            raise ValueError("lp_posterior_n_jobs must be positive")
        if self.tau_floor is not None and not (
            np.isfinite(float(self.tau_floor)) and float(self.tau_floor) >= 0.0
        ):
            raise ValueError("tau_floor must be finite and nonnegative")


def noise_candidate_names(n_noise: int) -> tuple[str, ...]:
    """Return the ordered noise-candidate column names.

    Args:
        n_noise: Number of noise candidates.

    Returns:
        ``("noise_01", ..., "noise_{n_noise}")``.
    """

    width = max(2, len(str(int(n_noise))))
    return tuple(
        f"{NOISE_PREFIX}{index:0{width}d}" for index in range(1, int(n_noise) + 1)
    )


def _participant_key(site: Any, participant: Any) -> str:
    """Return a stable text key for one participant.

    Integral float ids, as ManyLabs stores them, are written as integers so
    the key does not depend on the column dtype.
    """

    if isinstance(participant, (float, np.floating)) and float(participant).is_integer():
        participant = int(participant)
    return f"{site}|{participant}"


def _uniform_from_key(text: str) -> float:
    """Map a text key to a reproducible uniform draw in ``[0, 1)``."""

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


def add_noise_candidates(
    frames: Sequence[pd.DataFrame],
    names: Sequence[str],
    *,
    prevalence: float,
    seed: int,
    site_col: str = "site",
    id_col: str = "id",
) -> list[pd.DataFrame]:
    """Append binary pure-noise columns keyed on participant identity.

    Column ``k`` of participant ``(site, id)`` is ``1`` when a hash of
    ``(seed, k, site, id)`` falls below ``prevalence``, so it is independent of
    every participant attribute and equal across regime frames. Any continuous
    noise cut at its ``prevalence`` quantile gives the same Bernoulli column,
    which is all a discrete-``Z`` method sees.

    Args:
        frames: Frames to augment, typically the OSRCT and RCT rows.
        names: Noise column names to create.
        prevalence: ``P(N_k = 1)``.
        seed: Noise seed.
        site_col: Site column.
        id_col: Participant id column; ``(site, id)`` must be unique per frame.

    Returns:
        Copies of ``frames`` with the noise columns appended.

    Raises:
        ValueError: If a name already exists or participant keys repeat.
    """

    augmented: list[pd.DataFrame] = []
    for frame in frames:
        clashes = sorted(set(names) & set(frame.columns))
        if clashes:
            raise ValueError(f"Noise columns already exist: {clashes}")
        if frame[id_col].isna().any():
            raise ValueError(f"Column {id_col!r} must not contain missing ids")
        keys = [
            _participant_key(site, participant)
            for site, participant in zip(frame[site_col], frame[id_col])
        ]
        if len(set(keys)) != len(keys):
            raise ValueError(f"({site_col}, {id_col}) must identify rows uniquely")
        result = frame.copy()
        for index, name in enumerate(names, start=1):
            uniforms = np.array(
                [_uniform_from_key(f"{seed}|{index}|{key}") for key in keys],
                dtype=float,
            )
            result[name] = (uniforms < float(prevalence)).astype(int)
        augmented.append(result)
    return augmented


def route_run_configs(
    route: str,
    base: Exp16Config,
    noise_names: Sequence[str],
) -> list[tuple[str, Exp16Config]]:
    """Return the labelled Exp16 configurations one condition runs.

    Args:
        route: ``separate``, ``joint`` or ``baseline``.
        base: Paired Exp25 configuration for the condition.
        noise_names: Noise-candidate columns present in the data.

    Returns:
        ``(run_label, config)`` pairs: one per noise candidate for
        ``separate``, one joint fit, or the unchanged baseline.
    """

    if route == "baseline":
        return [("baseline", base)]
    if route == "joint":
        return [(
            "joint",
            replace(
                base,
                naturalness_extra_candidates=tuple(noise_names),
                naturalness_candidate_covariates=None,
                fallback_when_gp_unfitted=True,
            ),
        )]
    if route == "separate":
        return [
            (
                name,
                replace(
                    base,
                    naturalness_extra_candidates=(name,),
                    naturalness_candidate_covariates=(name,),
                    fallback_when_gp_unfitted=True,
                ),
            )
            for name in noise_names
        ]
    raise ValueError(f"Unknown route {route!r}")


def _scientific_config(config: Exp32Config, exp25: Exp25Config) -> dict[str, Any]:
    """Return the fields that must match before resuming into a root."""

    scientific = {
        "experiment": EXP32_NAME,
        "exp25_store": str(Path(config.exp25_store).resolve()),
        "exp25_base_seed": int(exp25.base_seed),
        "study": str(config.study),
        "beta": float(config.beta),
        "route": str(config.route),
        "n_noise": int(config.n_noise) if config.route != "baseline" else 0,
        "noise_prevalence": float(config.noise_prevalence),
        "noise_seed": int(config.noise_seed),
        "lp_posterior_samples": int(
            config.lp_posterior_samples or exp25.lp_posterior_samples
        ),
    }
    # Only floored designs carry the key; always writing it would change unfloored roots' fingerprints.
    if config.tau_floor is not None:
        scientific["tau_floor"] = float(config.tau_floor)
    return scientific


def _fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable short hash of a JSON-serializable mapping."""

    text = json.dumps(payload, sort_keys=True, default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _claim_route_root(root: Path, scientific: Mapping[str, Any]) -> None:
    """Create or validate a route root so different designs never share one.

    Raises:
        ValueError: If the root already holds a different configuration.
    """

    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "exp32_identity.json"
    fingerprint = _fingerprint(scientific)
    if identity_path.exists():
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded.get("fingerprint") != fingerprint:
            raise ValueError(
                f"{root} holds a different Exp32 configuration; use a fresh output root"
            )
        return
    # Write, then rename: pods sharing a root never read a half-written file.
    temporary = identity_path.with_name(f".{identity_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {"fingerprint": fingerprint, "scientific_config": dict(scientific)},
            indent=2,
            sort_keys=True,
            default=json_default,
        ),
        encoding="utf-8",
    )
    temporary.replace(identity_path)


def run_labelled_config(
    *,
    run_label: str,
    run_config: Exp16Config,
    osrct_data: pd.DataFrame,
    rct_data: pd.DataFrame,
    base_metadata: Mapping[str, Any],
    target_index: int,
    run_dir: Path,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one Exp16 configuration and persist its compact tables.

    Args:
        run_label: Run name, a noise column for ``separate`` runs.
        run_config: Exp16 configuration for this run.
        osrct_data: Observational rows including the noise columns.
        rct_data: Randomized rows including the noise columns.
        base_metadata: Data metadata passed to Exp16.
        target_index: Paired Exp25 target index.
        run_dir: Directory receiving the run's tables.
        extra: Identifying columns added to every persisted table.

    Returns:
        The completion record written to the run's marker.
    """

    start = time.time()
    outputs = run_one_target(
        osrct_data=osrct_data,
        rct_data=rct_data,
        target_site=str(run_config.target_sites[0]),
        target_index=int(target_index),
        config=run_config,
        base_metadata=base_metadata,
    )
    seconds = float(time.time() - start)
    for output_name, stored_name in PERSISTED_OUTPUTS.items():
        frame = outputs.get(output_name, pd.DataFrame())
        frame = frame.copy()
        for column, value in extra.items():
            frame[column] = value
        frame["run_label"] = run_label
        atomic_write_parquet(parquet_safe(frame), run_dir / f"{stored_name}.parquet")
    record = {
        **dict(extra),
        "run_label": run_label,
        "runtime_seconds": seconds,
        "naturalness_extra_candidates": list(run_config.naturalness_extra_candidates),
        "naturalness_candidate_covariates": (
            None
            if run_config.naturalness_candidate_covariates is None
            else list(run_config.naturalness_candidate_covariates)
        ),
        "lp_posterior_samples": int(run_config.lp_posterior_samples),
        "completed_at_unix": time.time(),
    }
    (run_dir / COMPLETE_MARKER).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    return record


def run_experiment(config: Exp32Config) -> dict[str, Any]:
    """Run one route of the noise-candidate ablation.

    Args:
        config: Ablation configuration.

    Returns:
        Route root, completion records of the runs made by this call, and the
        summary tables when ``config.summarize`` is true.
    """

    store = Path(config.exp25_store)
    exp25 = load_exp25_store_config(store)
    if config.study not in exp25.studies:
        raise ValueError(f"Study {config.study!r} is not in the Exp25 store")
    if not any(np.isclose(float(config.beta), float(beta)) for beta in exp25.betas):
        raise ValueError(f"beta={config.beta} is not in the Exp25 store")
    unknown_patterns = sorted(set(config.patterns) - set(exp25.bias_covariates))
    if unknown_patterns:
        raise ValueError(f"Patterns not in the Exp25 store: {unknown_patterns}")
    replications = (
        tuple(int(index) for index in config.replications)
        if config.replications is not None
        else tuple(range(int(exp25.n_replications)))
    )
    if any(index >= int(exp25.n_replications) for index in replications):
        raise ValueError("replications must be below the store's n_replications")

    scientific = _scientific_config(config, exp25)
    root = Path(config.output_dir) / config.route
    _claim_route_root(root, scientific)
    write_run_manifest(
        root,
        experiment_name=EXP32_NAME,
        script_path="experiments/exp32_manylabs_noise_candidates.py",
        config=asdict(config),
        argv=config.command_argv,
        cluster_manifest_path=config.cluster_manifest_path,
        extra={"scientific_config": scientific},
    )

    noise_names = noise_candidate_names(config.n_noise) if config.route != "baseline" else ()
    records: list[dict[str, Any]] = []
    for replication in replications:
        seed = int(exp25.base_seed + replication)
        data_root = store / "replications" / f"replication={replication:03d}" / "osrct"
        replication_config = _exp16_config_for_replication(
            exp25,
            seed=seed,
            data_root=data_root,
            output_dir=root / f"replication={replication:03d}",
        )
        if config.tau_floor is not None:
            replication_config = replace(
                replication_config,
                taus=floor_taus(replication_config.taus, float(config.tau_floor)),
            )
        for pattern in config.patterns:
            leaf = root / f"replication={replication:03d}" / str(pattern)
            base = replace(
                replication_config,
                study=str(config.study),
                studies=(str(config.study),),
                pattern=str(pattern),
                patterns=(str(pattern),),
                beta=float(config.beta),
                betas=(float(config.beta),),
                tabicl_share_checkpoint=bool(config.tabicl_share_checkpoint),
                lp_posterior_samples=int(
                    config.lp_posterior_samples or replication_config.lp_posterior_samples
                ),
                lp_posterior_n_jobs=int(
                    config.lp_posterior_n_jobs or replication_config.lp_posterior_n_jobs
                ),
                tabicl_device=str(config.tabicl_device or replication_config.tabicl_device),
                verbose=bool(config.verbose),
            )
            run_configs = route_run_configs(config.route, base, noise_names)
            pending = [
                (label, run_config)
                for label, run_config in run_configs
                if not (config.resume and (leaf / "runs" / label / COMPLETE_MARKER).exists())
            ]
            if not pending:
                continue
            osrct_data, rct_data, metadata = load_prepared_data(base)
            if noise_names:
                osrct_data, rct_data = add_noise_candidates(
                    [osrct_data, rct_data],
                    noise_names,
                    prevalence=float(config.noise_prevalence),
                    seed=int(config.noise_seed) * 1_000_003 + seed,
                    id_col=str(base.id_col),
                )
            metadata = {
                **metadata,
                "exp32_route": config.route,
                "exp32_noise_candidates": list(noise_names),
                "exp32_noise_prevalence": float(config.noise_prevalence),
            }
            target_index = exp25_target_index(
                exp25,
                study=str(config.study),
                pattern=str(pattern),
                beta=float(config.beta),
            )
            for label, run_config in pending:
                if config.verbose:
                    print(
                        f"Exp32 {config.route}: replication={replication} "
                        f"pattern={pattern} run={label}",
                        flush=True,
                    )
                record = run_labelled_config(
                    run_label=label,
                    run_config=run_config,
                    osrct_data=osrct_data,
                    rct_data=rct_data,
                    base_metadata=metadata,
                    target_index=target_index,
                    run_dir=leaf / "runs" / label,
                    extra={
                        "route": config.route,
                        "replication": int(replication),
                        "seed": seed,
                        "study": str(config.study),
                        "pattern": str(pattern),
                        "beta": float(config.beta),
                    },
                )
                records.append(record)
                if config.verbose:
                    print(
                        f"  finished in {record['runtime_seconds']:.1f}s",
                        flush=True,
                    )
    summary = (
        summarize_results(
            Path(config.output_dir),
            store,
            study=str(config.study),
            beta=float(config.beta),
            tau_floor=config.tau_floor,
        )
        if config.summarize
        else {}
    )
    return {"root": root, "records": records, "summary": summary}


def _read_run_tables(output_dir: Path, stored_name: str) -> pd.DataFrame:
    """Concatenate one persisted table over every completed run and route.

    The files are small and a read is dominated by latency on cephfs, so they
    are read on a thread pool; the concatenation keeps the sorted run order.
    """

    paths = [
        marker.parent / f"{stored_name}.parquet"
        for marker in sorted(output_dir.glob(f"*/replication=*/*/runs/*/{COMPLETE_MARKER}"))
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        frames = list(pool.map(lambda path: pd.read_parquet(path) if path.exists() else None, paths))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _candidate_kind(covariates: pd.Series) -> pd.Series:
    """Label candidates as ``noise`` or ``real`` by column name."""

    return np.where(
        covariates.astype(str).str.startswith(NOISE_PREFIX),
        "noise",
        "real",
    )


def _baseline_posterior_summaries(
    store: Path,
    runs: pd.DataFrame,
    *,
    study: str,
    beta: float,
) -> pd.DataFrame:
    """Load the stored Exp25 posterior summaries paired with completed runs."""

    frames: list[pd.DataFrame] = []
    covariates = list(Exp16Config().covariates)
    for replication, pattern in (
        runs[["replication", "pattern"]].drop_duplicates().itertuples(index=False)
    ):
        path = _store_condition_leaf(store, replication, pattern, beta) / "target_gp_posterior_summary.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame.loc[frame["study"].astype(str).eq(study)]
        keep = [*covariates, "lower_mean", "upper_mean", "lp_posterior_prob_optimal"]
        frame = frame[keep].rename(columns={
            "lower_mean": "exp25_lower",
            "upper_mean": "exp25_upper",
            "lp_posterior_prob_optimal": "exp25_prob_optimal",
        })
        frame["replication"] = int(replication)
        frame["pattern"] = str(pattern)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _route_baseline_reference(
    output_dir: Path,
    bounds: pd.DataFrame,
    *,
    tau_floor: float,
) -> pd.DataFrame:
    """Return this root's ``baseline``-route intervals as the paired m = 0 reference.

    A floored design has no stored counterpart, since the Exp25 store used the
    full tau grid, so the reference is the root's own baseline route.

    Args:
        output_dir: Exp32 result root holding one directory per route.
        bounds: Posterior-summary rows of every route, as built by
            :func:`summarize_results`.
        tau_floor: Floor the noise routes were run at.

    Returns:
        Baseline rows under the ``exp25_*`` reference column names.

    Raises:
        ValueError: If the baseline route is missing or used another floor.
    """

    identity_path = output_dir / "baseline" / "exp32_identity.json"
    recorded = None
    if identity_path.exists():
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))[
            "scientific_config"].get("tau_floor")
    if recorded is None or not np.isclose(float(recorded), float(tau_floor)):
        raise ValueError(
            f"tau_floor={tau_floor} pairs runs with this root's baseline route, "
            "which must be run at the same floor"
        )
    covariates = list(Exp16Config().covariates)
    reference = bounds.loc[
        bounds["route"].astype(str).eq("baseline"),
        ["replication", "pattern", *covariates, "lower_mean", "upper_mean",
         "lp_posterior_prob_optimal"],
    ]
    return reference.rename(columns={
        "lower_mean": "exp25_lower",
        "upper_mean": "exp25_upper",
        "lp_posterior_prob_optimal": "exp25_prob_optimal",
    })


def summarize_results(
    output_dir: Path,
    store: Path,
    *,
    study: str,
    beta: float,
    tau_floor: float | None = None,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Write discovery-first summary tables over every completed run.

    Tables, in reporting order:

    * ``summary_source_triggers``: share of source-site rows that fired, by
      route, pattern, candidate kind and ``tau``;
    * ``summary_target_triggers``: share of candidates that fired at the target
      for at least one grid point, with their ``D*`` statistics;
    * ``summary_target_tolerances``: learned ``2c`` and ``2d`` of
      target-triggered rows by candidate kind;
    * ``summary_bounds``: per stratum, the run's posterior-mean interval next
      to the paired Exp25 interval, with RCT-CATE containment for both. With
      ``tau_floor`` the paired interval is this root's baseline route at the
      same floor (:func:`_route_baseline_reference`).

    Args:
        output_dir: Exp32 result root holding one directory per route.
        store: Paired Exp25 backend leaf.
        study: Ablated study.
        beta: Ablated selection strength.
        tau_floor: Floor the routes were run at; ``None`` pairs with the store.
        write: Whether to write each table as CSV into ``output_dir``.

    Returns:
        The summary tables by name.
    """

    tables: dict[str, pd.DataFrame] = {}
    source = _read_run_tables(output_dir, "source_observations")
    if not source.empty:
        source["candidate_kind"] = _candidate_kind(source["covariate"])
        tables["summary_source_triggers"] = (
            source.groupby(["route", "pattern", "candidate_kind", "tau"], sort=True)
            .agg(
                n_rows=("trigger_fired", "size"),
                fire_rate=("trigger_fired", "mean"),
                mean_d_strength=("d_strength", "mean"),
                mean_i_deviation=("i_deviation", "mean"),
            )
            .reset_index()
        )

    statistics = _read_run_tables(output_dir, "target_trigger_statistics")
    if not statistics.empty:
        statistics["candidate_kind"] = _candidate_kind(statistics["covariate"])
        tables["summary_target_triggers"] = (
            statistics.groupby(["route", "pattern", "candidate_kind"], sort=True)
            .agg(
                n_candidate_replications=("target_triggered_any", "size"),
                fire_rate=("target_triggered_any", "mean"),
                median_d_strength=("d_strength", "median"),
                max_d_strength=("d_strength", "max"),
                median_i_deviation=("i_deviation", "median"),
            )
            .reset_index()
        )

    candidates = _read_run_tables(output_dir, "target_candidates")
    if not candidates.empty and "two_c" in candidates:
        candidates = candidates.loc[candidates["covariate"].notna()].copy()
        candidates["candidate_kind"] = _candidate_kind(candidates["covariate"])
        tables["summary_target_tolerances"] = (
            candidates.groupby(["route", "pattern", "candidate_kind", "covariate"], sort=True)
            .agg(
                n_triggered_rows=("two_c", "size"),
                median_two_c=("two_c", "median"),
                median_two_d=("two_d", "median"),
                min_two_c=("two_c", "min"),
            )
            .reset_index()
        )

    posterior = _read_run_tables(output_dir, "target_posterior_summary")
    if not posterior.empty:
        covariates = list(Exp16Config().covariates)
        keep = [
            "route", "replication", "pattern", "run_label", *covariates,
            "lower_mean", "upper_mean", "lp_posterior_prob_optimal", "rct_cate",
            "rct_cate_available", "n_target_rows",
        ]
        bounds = posterior[[column for column in keep if column in posterior]].copy()
        baseline = (
            _baseline_posterior_summaries(store, bounds, study=study, beta=beta)
            if tau_floor is None
            else _route_baseline_reference(output_dir, bounds, tau_floor=tau_floor)
        )
        if not baseline.empty:
            bounds = bounds.merge(
                baseline,
                on=["replication", "pattern", *covariates],
                how="left",
                validate="many_to_one",
            )
        truth = pd.to_numeric(bounds["rct_cate"], errors="coerce")
        bounds["width"] = bounds["upper_mean"] - bounds["lower_mean"]
        bounds["contains_rct_cate"] = truth.between(bounds["lower_mean"], bounds["upper_mean"])
        if "exp25_lower" in bounds:
            bounds["exp25_width"] = bounds["exp25_upper"] - bounds["exp25_lower"]
            bounds["exp25_contains_rct_cate"] = truth.between(
                bounds["exp25_lower"], bounds["exp25_upper"]
            )
            bounds["width_minus_exp25"] = bounds["width"] - bounds["exp25_width"]
        tables["summary_bounds"] = bounds

    if write:
        for name, table in tables.items():
            table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def _store_condition_leaf(store: Path, replication: int, pattern: str, beta: float) -> Path:
    """Return the Exp25 store's persisted Naturalness leaf of one condition."""

    return store.joinpath(
        "replications",
        f"replication={int(replication):03d}",
        "naturalness",
        "all_triggered",
        "pi",
        str(pattern),
        format_beta(float(beta)),
    )


def _root_tau_floor(output_dir: Path) -> float | None:
    """Return the tau floor every route of an Exp32 root was run at.

    Raises:
        ValueError: If the root holds no route or its routes disagree.
    """

    floors = {
        json.loads(path.read_text(encoding="utf-8"))["scientific_config"].get("tau_floor")
        for path in sorted(output_dir.glob("*/exp32_identity.json"))
    }
    if not floors:
        raise ValueError(f"{output_dir} holds no Exp32 route")
    if len(floors) != 1:
        raise ValueError(f"Routes of {output_dir} use different tau floors: {sorted(floors, key=str)}")
    return floors.pop()


def _reference_choices(
    output_dir: Path,
    store: Path,
    runs: pd.DataFrame,
    *,
    study: str,
    beta: float,
    tau_floor: float | None,
) -> pd.DataFrame:
    """Return the selected LP rows of the m = 0 reference paired with ``runs``.

    The reference is the one :func:`summarize_results` pairs bounds with: this
    root's baseline route for a floored design, the Exp25 store otherwise.
    """

    if tau_floor is not None:
        choices = _read_run_tables(output_dir, "target_choices")
        if choices.empty:
            return choices
        baseline = choices.loc[choices["route"].astype(str).eq("baseline")]
        conditions = runs[["replication", "pattern"]].drop_duplicates()
        return baseline.merge(conditions, on=["replication", "pattern"])
    frames = []
    for replication, pattern in runs[["replication", "pattern"]].drop_duplicates().itertuples(index=False):
        path = _store_condition_leaf(store, replication, pattern, beta) / "target_policy_choices.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            frames.append(frame.loc[frame["study"].astype(str).eq(study)].assign(
                replication=int(replication), pattern=str(pattern)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bounds_change(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize joint-route intervals against their paired m = 0 reference."""

    width_ok = frame["width"].notna() & frame["exp25_width"].notna()
    change = frame.loc[width_ok, "width_minus_exp25"]
    scored = width_ok & frame["rct_cate_available"].astype(str).str.lower().eq("true")
    return {
        "n_strata": int(len(frame)),
        "n_both_valid": int(width_ok.sum()),
        "n_joint_only": int((frame["width"].notna() & frame["exp25_width"].isna()).sum()),
        "n_reference_only": int((frame["width"].isna() & frame["exp25_width"].notna()).sum()),
        "mean_width_change": float(change.mean()),
        "median_width_change": float(change.median()),
        "mean_relative_width_change": float((change / frame.loc[width_ok, "exp25_width"]).mean()),
        "share_narrower": float((change < -1e-6).mean()),
        "share_wider": float((change > 1e-6).mean()),
        "share_abs_change_above_0.01": float((change.abs() > 0.01).mean()),
        "max_abs_width_change": float(change.abs().max()),
        "n_scored": int(scored.sum()),
        "joint_containment": float(frame.loc[scored, "contains_rct_cate"].astype(bool).mean()),
        "reference_containment": float(frame.loc[scored, "exp25_contains_rct_cate"].astype(bool).mean()),
        "joint_prob_optimal": float(frame["lp_posterior_prob_optimal"].mean()),
        "reference_prob_optimal": float(frame["exp25_prob_optimal"].mean()),
    }


def ablation_metrics(
    output_dir: Path,
    store: Path,
    *,
    study: str,
    beta: float,
) -> tuple[dict[str, pd.DataFrame], float | None]:
    """Compute the discovery-first ablation metrics of one Exp32 root.

    In reporting order: source trigger rates by candidate kind and ``tau``;
    target trigger rates with Wilson 95% intervals; the composition of the
    target LP; the shift of the real candidates' tolerances in the joint route
    against the m = 0 reference, the channel through which noise reaching only
    the shared GP can still move a bound; and the joint route's intervals
    against the same reference, overall and by selection pattern.

    Args:
        output_dir: Exp32 result root holding one directory per route.
        store: Paired Exp25 backend leaf.
        study: Ablated study.
        beta: Ablated selection strength.

    Returns:
        The metric tables by name and the root's tau floor.
    """


    tau_floor = _root_tau_floor(output_dir)
    summary = summarize_results(output_dir, store, study=study, beta=beta, tau_floor=tau_floor, write=False)
    metrics: dict[str, pd.DataFrame] = {}

    source = summary["summary_source_triggers"].assign(
        n_fired=lambda frame: frame["fire_rate"] * frame["n_rows"])
    source = source.groupby(["route", "candidate_kind", "tau"], sort=True)[["n_rows", "n_fired"]].sum().reset_index()
    source["n_fired"] = source["n_fired"].round().astype(int)
    metrics["source_trigger_rates"] = source.assign(fire_rate=source["n_fired"] / source["n_rows"])

    statistics = _read_run_tables(output_dir, "target_trigger_statistics")
    statistics["candidate_kind"] = _candidate_kind(statistics["covariate"])
    statistics["fired"] = statistics["target_triggered_any"].astype(str).str.lower().eq("true")
    keys = ["route", "candidate_kind"]
    target = statistics.groupby(keys, sort=True).agg(
        n_candidate_conditions=("fired", "size"), n_fired=("fired", "sum")).reset_index()
    with_fire = statistics.loc[statistics["fired"]].drop_duplicates([*keys, "replication", "pattern"])
    target = target.merge(with_fire.groupby(keys).size().rename("n_conditions_with_fire").reset_index(),
                          on=keys, how="left").fillna({"n_conditions_with_fire": 0})
    target["n_conditions_with_fire"] = target["n_conditions_with_fire"].astype(int)
    target["fire_rate"] = target["n_fired"] / target["n_candidate_conditions"]
    target["fire_rate_lower95"], target["fire_rate_upper95"] = wilson_interval(
        target["n_fired"], target["n_candidate_conditions"])
    metrics["target_trigger_rates"] = target

    choices = _read_run_tables(output_dir, "target_choices")
    choices = choices.loc[choices["covariate"].notna()].copy()
    choices["candidate_kind"] = _candidate_kind(choices["covariate"])
    composition = choices.groupby(keys, sort=True).agg(
        n_lp_rows=("two_c", "size"), median_two_c=("two_c", "median"), median_two_d=("two_d", "median")).reset_index()
    composition = composition.merge(
        choices.drop_duplicates([*keys, "replication", "pattern"]).groupby(keys).size()
        .rename("n_conditions").reset_index(), on=keys)
    # Every route and kind the target saw gets a row, so a kind that never reaches the LP shows as zero.
    composition = target[keys].merge(composition, on=keys, how="left").fillna({"n_lp_rows": 0, "n_conditions": 0})
    metrics["lp_composition"] = composition.astype({"n_lp_rows": int, "n_conditions": int})

    grid = ["replication", "pattern", "covariate", "epsilon", "tau"]
    joint_real = choices.loc[choices["route"].astype(str).eq("joint") & choices["candidate_kind"].eq("real")]
    joint_conditions = statistics.loc[statistics["route"].astype(str).eq("joint"), ["replication", "pattern"]]
    reference = _reference_choices(output_dir, store, joint_conditions, study=study, beta=beta, tau_floor=tau_floor)
    if reference.empty:
        # Typed like the joint rows: an untyped empty frame cannot be merged on float keys.
        reference = joint_real.iloc[0:0]
    reference = reference.loc[reference["covariate"].notna()]
    reference = reference.loc[_candidate_kind(reference["covariate"]) == "real"]
    paired = joint_real[[*grid, "two_c", "two_d"]].merge(
        reference[[*grid, "two_c", "two_d"]], on=grid, how="outer", suffixes=("_joint", "_reference"), indicator=True)
    shift: dict[str, Any] = {
        "n_matched_rows": int(paired["_merge"].eq("both").sum()),
        "n_joint_only_rows": int(paired["_merge"].eq("left_only").sum()),
        "n_reference_only_rows": int(paired["_merge"].eq("right_only").sum()),
    }
    for tolerance in ("two_c", "two_d"):
        difference = (paired[f"{tolerance}_joint"] - paired[f"{tolerance}_reference"]).dropna()
        shift.update({
            f"mean_{tolerance}_change": float(difference.mean()),
            f"median_abs_{tolerance}_change": float(difference.abs().median()),
            f"p95_abs_{tolerance}_change": float(difference.abs().quantile(0.95)),
            f"max_abs_{tolerance}_change": float(difference.abs().max()),
        })
    metrics["real_tolerance_shift"] = pd.DataFrame([shift])

    bounds = summary["summary_bounds"]
    joint = bounds.loc[bounds["route"].astype(str).eq("joint")]
    rows = [{"pattern": "all", **_bounds_change(joint)}]
    rows += [{"pattern": str(pattern), **_bounds_change(frame)} for pattern, frame in joint.groupby("pattern", sort=True)]
    metrics["joint_bounds_change"] = pd.DataFrame(rows)
    return metrics, tau_floor


def analyze_results(
    output_dir: Path,
    store: Path,
    *,
    study: str,
    beta: float,
    destination: Path,
    reference_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Write the ablation analysis of one Exp32 root, optionally beside another design.

    Each design is labelled by the smallest tau its grid keeps. The tables of
    :func:`ablation_metrics` are stacked over designs and written as
    ``exp32_<table>.csv``, with a Markdown report and a provenance manifest.
    The roots are only read; in particular no summary CSV is rewritten.

    Args:
        output_dir: Exp32 result root to analyse.
        store: Paired Exp25 backend leaf.
        study: Ablated study.
        beta: Ablated selection strength.
        destination: Directory receiving the analysis.
        reference_dir: Optional second Exp32 root, such as the same design at
            another tau floor.

    Returns:
        The stacked tables by name.
    """

    from experiments.exp31_bayesian_manylabs_naturalness import _markdown_table

    designs = []
    stacked: dict[str, list[pd.DataFrame]] = {}
    for root in [Path(output_dir), *([Path(reference_dir)] if reference_dir is not None else [])]:
        metrics, tau_floor = ablation_metrics(root, store, study=study, beta=beta)
        smallest_tau = float(tau_floor) if tau_floor is not None else float(min(Exp16Config().taus))
        label = f"tau>={smallest_tau:g}"
        designs.append({"design": label, "root": str(root.resolve()), "tau_floor": tau_floor})
        for name, frame in metrics.items():
            stacked.setdefault(name, []).append(frame.assign(design=label)[["design", *frame.columns]])
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in stacked.items()}

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / f"exp32_{name}.csv", index=False)

    source = tables["source_trigger_rates"]
    pooled = source.groupby(["design", "route", "candidate_kind"], sort=False)[["n_rows", "n_fired"]].sum().reset_index()
    pooled["fire_rate"] = pooled["n_fired"] / pooled["n_rows"]
    noise_by_tau = source.loc[source["candidate_kind"].eq("noise") & source["route"].eq("joint")].pivot(
        index="design", columns="tau", values="fire_rate").reset_index()
    noise_by_tau.columns = [str(column) for column in noise_by_tau.columns]
    reference_text = (
        "this root's baseline route at the same floor" if designs[0]["tau_floor"] is not None
        else "the stored Exp25 result")
    sections = [
        "# Exp32 noise-candidate ablation: analysis",
        f"Real ManyLabs pipeline output: study `{study}`, beta = {beta:g}, the Exp25 OSRCT draws of `{store}`. "
        "Each noise candidate is an independent binary column, so every trigger it passes is a false fire. "
        f"The m = 0 reference for tolerances and bounds is {reference_text}; a design without a floor is "
        "paired with the stored Exp25 result.",
        _markdown_table(pd.DataFrame(designs), ["design", "tau_floor", "root"]),
        "## 1. Source triggers (share of source-site rows that fire)",
        _markdown_table(pooled, ["design", "route", "candidate_kind", "n_rows", "fire_rate"], digits=4),
        "Noise fire rate by tau, joint route:",
        _markdown_table(noise_by_tau, list(noise_by_tau.columns), digits=4),
        "## 2. Target triggers (candidate-conditions that fire at any grid point; Wilson 95% interval)",
        _markdown_table(tables["target_trigger_rates"], ["design", "route", "candidate_kind", "n_candidate_conditions",
                                                          "n_fired", "n_conditions_with_fire", "fire_rate",
                                                          "fire_rate_upper95"], digits=4),
        "## 3. Target LP composition",
        _markdown_table(tables["lp_composition"], ["design", "route", "candidate_kind", "n_lp_rows", "n_conditions",
                                                   "median_two_c", "median_two_d"]),
        "## 4. Real-candidate tolerances, joint route against the reference",
        _markdown_table(tables["real_tolerance_shift"], ["design", "n_matched_rows", "n_joint_only_rows",
                                                         "n_reference_only_rows", "mean_two_c_change",
                                                         "p95_abs_two_c_change", "mean_two_d_change",
                                                         "p95_abs_two_d_change"], digits=4),
        "## 5. Joint-route intervals against the reference",
        _markdown_table(tables["joint_bounds_change"], ["design", "pattern", "n_both_valid", "n_joint_only",
                                                        "n_reference_only", "mean_width_change",
                                                        "mean_relative_width_change", "share_narrower",
                                                        "share_wider", "max_abs_width_change", "joint_containment",
                                                        "reference_containment"], digits=4),
    ]
    (destination / "exp32_analysis_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    (destination / "exp32_analysis_manifest.json").write_text(json.dumps({
        "experiment": EXP32_NAME,
        "designs": designs,
        "exp25_store": str(Path(store).resolve()),
        "study": study,
        "beta": float(beta),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }, indent=2, default=json_default) + "\n", encoding="utf-8")
    return tables


def _pooled(frame: pd.DataFrame, weight: str, columns: Sequence[str]) -> dict[str, float]:
    """Return ``weight``-weighted means of ``columns`` over the rows of ``frame``.

    Rows of zero weight are left out, so a cell with nothing to pool (its means
    are NaN) does not turn the pooled value into NaN.
    """

    used = frame.loc[frame[weight].astype(float) > 0]
    if used.empty:
        return {column: float("nan") for column in columns}
    weights = used[weight].astype(float)
    return {column: float(np.average(used[column].astype(float), weights=weights)) for column in columns}


def analyze_grid(parent: Path, destination: Path) -> dict[str, pd.DataFrame]:
    """Write the ablation analysis of a grid of Exp32 roots, one per study and beta.

    Every subdirectory of ``parent`` that holds Exp32 routes is analysed with
    :func:`ablation_metrics`; its study, beta and store come from the routes'
    identity files. The tables are stacked with ``study`` and ``beta`` columns
    and written as ``exp32_grid_<table>.csv``, beside a Markdown report with the
    grid totals. Each root is paired with its own baseline route, so every root
    must have been run with a tau floor.

    Args:
        parent: Directory holding one Exp32 root per study and beta.
        destination: Directory receiving the analysis.

    Returns:
        The stacked tables by name.

    Raises:
        FileNotFoundError: If ``parent`` holds no Exp32 root.
        ValueError: If a root has no tau floor or the roots use different floors.
    """

    from experiments.exp31_bayesian_manylabs_naturalness import _markdown_table

    roots, stacked = [], {}
    for root in sorted(path for path in Path(parent).iterdir() if any(path.glob("*/exp32_identity.json"))):
        identity = json.loads(next(root.glob("*/exp32_identity.json")).read_text(encoding="utf-8"))
        scientific = identity["scientific_config"]
        study, beta = str(scientific["study"]), float(scientific["beta"])
        if _root_tau_floor(root) is None:
            raise ValueError(f"{root} has no tau floor; the grid analysis pairs each root with its baseline route")
        metrics, tau_floor = ablation_metrics(root, Path(scientific["exp25_store"]), study=study, beta=beta)
        roots.append({"study": study, "beta": beta, "tau_floor": tau_floor, "root": str(root.resolve())})
        for name, frame in metrics.items():
            stacked.setdefault(name, []).append(frame.assign(study=study, beta=beta)[["study", "beta", *frame.columns]])
    if not roots:
        raise FileNotFoundError(f"No Exp32 root under {parent}")
    if len({entry["tau_floor"] for entry in roots}) != 1:
        raise ValueError(f"Grid roots use different tau floors: {roots}")
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in stacked.items()}

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / f"exp32_grid_{name}.csv", index=False)

    target = tables["target_trigger_rates"]
    noise = target.loc[target["candidate_kind"].eq("noise")]
    pooled_noise = noise.groupby("route", sort=True)[["n_candidate_conditions", "n_fired", "n_conditions_with_fire"]].sum()
    pooled_noise["fire_rate"] = pooled_noise["n_fired"] / pooled_noise["n_candidate_conditions"]
    pooled_noise["fire_rate_lower95"], pooled_noise["fire_rate_upper95"] = wilson_interval(
        pooled_noise["n_fired"], pooled_noise["n_candidate_conditions"])
    pooled_noise = pooled_noise.reset_index()
    source = tables["source_trigger_rates"]
    source = source.loc[source["route"].eq("joint")].groupby(["candidate_kind", "tau"], sort=True)[
        ["n_rows", "n_fired"]].sum().reset_index()
    source["fire_rate"] = source["n_fired"] / source["n_rows"]
    lp = tables["lp_composition"]
    lp = lp.loc[lp["route"].eq("joint")].pivot_table(
        index=["study", "beta"], columns="candidate_kind", values="n_lp_rows", aggfunc="sum").reset_index()
    lp.columns = [str(column) for column in lp.columns]
    shift = tables["real_tolerance_shift"]
    bounds = tables["joint_bounds_change"]
    overall = bounds.loc[bounds["pattern"].eq("all")]
    weighted = ["mean_width_change", "mean_relative_width_change", "share_narrower", "share_wider",
                "share_abs_change_above_0.01"]
    grid_row = {
        "study": "all", "beta": "all",
        "n_both_valid": int(overall["n_both_valid"].sum()), "n_joint_only": int(overall["n_joint_only"].sum()),
        "n_reference_only": int(overall["n_reference_only"].sum()),
        **_pooled(overall, "n_both_valid", weighted),
        "max_abs_width_change": float(overall["max_abs_width_change"].max()),
        **_pooled(overall, "n_scored", ["joint_containment", "reference_containment"]),
    }
    overall = pd.concat([overall, pd.DataFrame([grid_row])], ignore_index=True)
    def labelled(frame: pd.DataFrame) -> pd.DataFrame:
        """Render ``beta`` as written in the grid rather than as a padded float."""

        return frame.assign(beta=frame["beta"].map(lambda value: value if isinstance(value, str) else f"{float(value):g}"))

    noise, lp, shift, overall = (labelled(frame) for frame in (noise, lp, shift, overall))
    bound_columns = ["study", "beta", "n_both_valid", "n_joint_only", "n_reference_only", "mean_width_change",
                     "mean_relative_width_change", "share_narrower", "share_wider", "max_abs_width_change",
                     "joint_containment", "reference_containment"]
    sections = [
        "# Exp32 noise-candidate ablation over the Exp25 grid: analysis",
        f"Real ManyLabs pipeline output: {len(roots)} study-beta cells, each with every selection pattern and "
        f"replication of its Exp25 store, at tau >= {roots[0]['tau_floor']:g}. Each noise candidate is an "
        "independent binary column, so every trigger it passes is a false fire. Tolerances and bounds are "
        "paired with each cell's own baseline route (no noise candidates) at the same floor.",
        "## 1. Target triggers of noise candidates, all cells (Wilson 95% interval)",
        _markdown_table(pooled_noise, ["route", "n_candidate_conditions", "n_fired", "n_conditions_with_fire",
                                       "fire_rate", "fire_rate_upper95"], digits=4),
        "By cell:",
        _markdown_table(noise, ["study", "beta", "route", "n_candidate_conditions", "n_fired",
                                "n_conditions_with_fire", "fire_rate_upper95"], digits=4),
        "## 2. Source triggers, joint route, all cells",
        _markdown_table(source, ["candidate_kind", "tau", "n_rows", "fire_rate"], digits=4),
        "## 3. Target LP rows by candidate kind, joint route",
        _markdown_table(lp, list(lp.columns)),
        "## 4. Real-candidate tolerances, joint route against the baseline",
        _markdown_table(shift, ["study", "beta", "n_matched_rows", "n_joint_only_rows", "n_reference_only_rows",
                                "mean_two_c_change", "p95_abs_two_c_change", "mean_two_d_change",
                                "p95_abs_two_d_change"], digits=4),
        "## 5. Joint-route intervals against the baseline",
        _markdown_table(overall, bound_columns, digits=4),
        "Per selection pattern: `exp32_grid_joint_bounds_change.csv`.",
    ]
    (destination / "exp32_grid_analysis_report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    (destination / "exp32_grid_analysis_manifest.json").write_text(json.dumps({
        "experiment": EXP32_NAME,
        "parent": str(Path(parent).resolve()),
        "roots": roots,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }, indent=2, default=json_default) + "\n", encoding="utf-8")
    return tables


def parse_args(argv: Sequence[str] | None = None) -> tuple[Exp32Config, argparse.Namespace]:
    """Parse command-line arguments.

    Args:
        argv: Arguments; ``None`` reads ``sys.argv``.

    Returns:
        The ablation configuration and the parsed arguments, which carry the
        ``summarize_only``, ``analysis_dir`` and ``grid_analysis_dir`` modes.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp25-store", default=None, help="Exp25 backend leaf; not needed for --grid-analysis-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study", default="allowedforbidden")
    parser.add_argument("--patterns", nargs="+", default=["age", "gender", "polideo"])
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--replications", type=int, nargs="+", default=None)
    parser.add_argument("--route", choices=ROUTES, default="separate")
    parser.add_argument("--n-noise", type=int, default=15)
    parser.add_argument("--noise-prevalence", type=float, default=0.5)
    parser.add_argument("--noise-seed", type=int, default=32)
    parser.add_argument("--lp-posterior-samples", type=int, default=None)
    parser.add_argument("--lp-posterior-n-jobs", type=int, default=None)
    parser.add_argument("--tabicl-device", default=None)
    parser.add_argument("--tau-floor", type=float, default=None,
                        help="smallest tau kept in the Exp16 grid; pairs runs with this root's baseline route")
    parser.add_argument("--no-share-checkpoint", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--analysis-dir", default=None,
                        help="write the ablation analysis of --output-dir here instead of running")
    parser.add_argument("--reference-dir", default=None,
                        help="analysis: a second Exp32 root to compare with, e.g. another tau floor")
    parser.add_argument("--grid-analysis-dir", default=None,
                        help="write the analysis of every Exp32 root under --output-dir (one per study and beta) here")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cluster-manifest-path", default=None)
    args = parser.parse_args(argv)
    if args.exp25_store is None and args.grid_analysis_dir is None:
        parser.error("--exp25-store is required unless --grid-analysis-dir is given")
    config = Exp32Config(
        exp25_store=args.exp25_store or "",
        output_dir=args.output_dir,
        study=args.study,
        patterns=tuple(args.patterns),
        beta=float(args.beta),
        replications=None if args.replications is None else tuple(args.replications),
        route=args.route,
        n_noise=int(args.n_noise),
        noise_prevalence=float(args.noise_prevalence),
        noise_seed=int(args.noise_seed),
        tabicl_share_checkpoint=not args.no_share_checkpoint,
        lp_posterior_samples=args.lp_posterior_samples,
        lp_posterior_n_jobs=args.lp_posterior_n_jobs,
        tabicl_device=args.tabicl_device,
        tau_floor=args.tau_floor,
        resume=not args.no_resume,
        summarize=not args.no_summary,
        verbose=not args.quiet,
        cluster_manifest_path=args.cluster_manifest_path,
        command_argv=tuple(sys.argv[1:] if argv is None else argv),
    )
    return config, args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ablation, rebuild its summaries, or write its analysis."""

    config, args = parse_args(argv)
    if args.grid_analysis_dir is not None:
        analyze_grid(Path(config.output_dir), Path(args.grid_analysis_dir))
        return 0
    if args.analysis_dir is not None:
        analyze_results(
            Path(config.output_dir),
            Path(config.exp25_store),
            study=config.study,
            beta=config.beta,
            destination=Path(args.analysis_dir),
            reference_dir=None if args.reference_dir is None else Path(args.reference_dir),
        )
        return 0
    if args.summarize_only:
        summarize_results(
            Path(config.output_dir),
            Path(config.exp25_store),
            study=config.study,
            beta=config.beta,
            tau_floor=config.tau_floor,
        )
        return 0
    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
