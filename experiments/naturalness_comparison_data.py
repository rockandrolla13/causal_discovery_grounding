"""Data preparation and regime construction for Naturalness comparisons.

Holds the dataset specifications (the reduced Twins benchmark), the declared binarisation rules
with an audit record per column, analysis-site construction, and the randomized, observational
and reference partitions that :mod:`experiments.naturalness_comparison` fits and scores for Exp28.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence
import zlib

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
MISSING_CODE = 255

def stable_seed(*parts: Any) -> int:
    """Return a reproducible seed from arbitrary parts.

    Python randomises string hashing per process, so ``hash("name")`` differs
    between runs; this uses a CRC32 checksum of the parts' text instead.

    Args:
        *parts: Values whose text form identifies the draw.

    Returns:
        A seed in ``[0, 2**32)``.
    """
    return zlib.crc32("|".join(str(part) for part in parts).encode("utf-8"))


# --------------------------------------------------------------------------
# Binarisation rules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryRule:
    """One declared rule turning a raw column into a ``{0, 1}`` indicator.

    Each rule records the construct it encodes and the justification for its cut; both are copied
    into the dataset provenance record.

    Attributes:
        kind: ``"at_least"``, ``"at_most"``, ``"equals"``, ``"in_set"``, ``"passthrough"`` for an
            already-binary column, or ``"median"`` for the data-dependent fallback.
        value: Threshold, level, or collection compared against; ignored for ``"passthrough"``
            and ``"median"``.
        label: What the indicator means when it equals one, phrased as the construct.
        justification: Why this cut rather than another.
        verified: Whether the cut's numeric boundary has been checked against the source codebook.
    """

    kind: str
    value: Any = None
    label: str = ""
    justification: str = ""
    verified: bool = True

    @property
    def prespecified(self) -> bool:
        """Whether the rule is fixed in advance rather than read off the data."""
        return self.kind != "median"


def matches_levels(series: pd.Series, wanted: Sequence[Any]) -> np.ndarray:
    """Test membership in a set of levels, tolerating archive label formatting.

    Text labels are stripped and case-folded before comparison, because real releases carry stray
    whitespace and inconsistent casing that would make a literal comparison silently match nothing.

    Args:
        series: Column to test.
        wanted: Levels counting as a match.

    Returns:
        Boolean array, one entry per row.
    """
    wanted = list(wanted)
    if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
        normalised = series.astype("string").str.strip().str.casefold()
        targets = {str(value).strip().casefold() for value in wanted}
        return normalised.isin(targets).to_numpy()
    return series.isin(wanted).to_numpy()


def apply_binary_rule(values: pd.Series, rule: BinaryRule) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one rule and return the codes alongside a full audit record.

    Args:
        values: Raw column.
        rule: The rule to apply.

    Returns:
        A ``(codes, record)`` pair. ``codes`` is ``uint8`` with
        :data:`MISSING_CODE` marking rows that could not be coded. ``record``
        carries the rule, the realised positive rate, the missing count, and for
        a ``"median"`` rule the threshold that was computed.

    Raises:
        ValueError: If the rule kind is unknown, or a ``"passthrough"`` column
            turns out not to be binary, which would silently corrupt the cell
            space.
    """
    series = pd.Series(values)
    missing = series.isna().to_numpy()
    observed = series[~missing]
    threshold: Any = rule.value

    if rule.kind == "passthrough":
        levels = set(pd.unique(observed))
        if not levels <= {0, 1, 0.0, 1.0, True, False}:
            raise ValueError("passthrough rule expects an already-binary column, saw levels "
                             f"{sorted(map(str, levels))[:8]}")
        codes = pd.to_numeric(series, errors="coerce").to_numpy(float) > 0.5
    elif rule.kind == "at_least":
        codes = pd.to_numeric(series, errors="coerce").to_numpy(float) >= float(rule.value)
    elif rule.kind == "at_most":
        codes = pd.to_numeric(series, errors="coerce").to_numpy(float) <= float(rule.value)
    elif rule.kind in ("equals", "in_set"):
        codes = matches_levels(series, [rule.value] if rule.kind == "equals" else rule.value)
    elif rule.kind == "median":
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(float)
        levels = np.unique(numeric[np.isfinite(numeric)])
        if levels.size <= 2:
            # The median of a two-level column is one of its levels, so ">= median" would mark
            # every row; split between the levels instead.
            threshold = float(levels.max()) if levels.size == 2 else float(levels[0])
            codes = numeric >= threshold
        else:
            threshold = float(np.nanmedian(numeric))
            codes = numeric >= threshold
    else:
        raise ValueError(f"Unknown binarisation rule kind: {rule.kind}")

    out = np.asarray(codes).astype(np.uint8)
    out[missing] = MISSING_CODE
    coded = out[out != MISSING_CODE]
    return out, dict(
        kind=rule.kind,
        threshold=(float(threshold) if isinstance(threshold, (int, float, np.floating))
                   else (None if threshold is None else str(threshold))),
        label=rule.label,
        justification=rule.justification,
        prespecified=rule.prespecified,
        verified=rule.verified,
        positive_rate=(float(coded.mean()) if coded.size else float("nan")),
        n_missing=int(missing.sum()),
        degenerate=bool(coded.size and (coded.min() == coded.max())),
    )


def make_bucketed_sites(frame: pd.DataFrame, column: str,
                        n_buckets: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Group a high-cardinality column into a fixed number of arbitrary sites.

    For a column with no natural grouping, such as a household identifier that
    :func:`make_artificial_sites` would pool into one site, levels are hashed into ``n_buckets``
    sites. Buckets are statistically identical by construction, so cross-site transfer on them is
    a negative control, not evidence of transfer.

    Args:
        frame: Source table.
        column: High-cardinality categorical column.
        n_buckets: How many sites to create.

    Returns:
        A ``(site_codes, description)`` pair, with the description marking the
        grouping as arbitrary.
    """
    labels = frame[column].astype("string").fillna("__MISSING__")
    codes = np.array([zlib.crc32(str(value).encode("utf-8")) % n_buckets for value in labels],
                     dtype=np.int64)
    return codes, dict(
        source_column=column,
        n_sites=int(len(np.unique(codes))),
        grouping="deterministic hash bucketing",
        arbitrary=True,
        caveat=("Buckets are statistically identical by construction, so cross-site transfer on "
                "this dataset is a negative control, not evidence that the principle transfers."),
        site_names=[str(index) for index in range(n_buckets)],
        site_rows={str(index): int((codes == index).sum()) for index in range(n_buckets)},
        n_distinct_source_levels=int(labels.nunique()),
    )


def make_artificial_sites(frame: pd.DataFrame, column: str, n_sites: int,
                          min_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Group a categorical column into a bounded number of analysis sites.

    The largest levels stay standalone; rare levels and rows with a missing label are pooled into
    ``__OTHER__`` rather than dropped, so no rows are lost.

    Args:
        frame: Source table.
        column: Column holding the categorical level, for example a state code.
        n_sites: Maximum number of sites to keep separate. The largest levels
            remain standalone; everything else is pooled into ``__OTHER__``.
        min_rows: Levels below this many rows are pooled regardless of rank.

    Returns:
        A ``(site_codes, description)`` pair. ``site_codes`` is an integer array
        of site indices; ``description`` records the mapping and row counts.
    """
    raw = frame[column]
    labels = raw.astype("string").fillna("__OTHER__")
    counts = labels.value_counts()
    eligible = [name for name in counts.index if counts[name] >= min_rows and name != "__OTHER__"]
    keep = eligible[:max(n_sites - 1, 1)] if len(eligible) >= n_sites else eligible
    labels = labels.where(labels.isin(keep), "__OTHER__")
    categories = sorted(labels.unique())
    mapping = {name: index for index, name in enumerate(categories)}
    codes = labels.map(mapping).to_numpy(np.int64)
    return codes, dict(
        source_column=column,
        n_sites=len(categories),
        site_names=[str(name) for name in categories],
        site_rows={str(name): int((codes == index).sum()) for name, index in mapping.items()},
        n_missing_labels=int(raw.isna().sum()),
    )


# --------------------------------------------------------------------------
# Constructed observational regimes
# --------------------------------------------------------------------------


def selection_probability(z_bias: np.ndarray, beta: float, intercept: float = 0.0) -> np.ndarray:
    """Return ``P(preferred treatment = 1 | z)`` for the biasing covariate.

    The covariate is standardised before the logistic transform, matching
    ``osrct_benchmark_v1.0/code/osrct.py``, so ``beta`` means the same there.
    The two agree bit for bit on complete, non-degenerate covariates. They differ
    on NaN input and on a constant covariate whose float mean rounds; see
    ``tests/test_experiments/test_osrct_selection_agreement.py``.

    Args:
        z_bias: Biasing covariate values.
        beta: Selection coefficient on the standardised covariate.
        intercept: Selection intercept, normally zero for a balanced design.

    Returns:
        Selection probabilities, one per row.
    """
    values = np.asarray(z_bias, dtype=float)
    spread = values.std()
    standardised = (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)
    return 1.0 / (1.0 + np.exp(-(intercept + beta * standardised)))


def osrct_select(z_bias: np.ndarray, treatment: np.ndarray, beta: float,
                 rng: np.random.Generator, intercept: float = 0.0,
                 propensity: np.ndarray | None = None,
                 trial_probability: np.ndarray | None = None,
                 clusters: np.ndarray | None = None) -> np.ndarray:
    """Return the OSRCT keep-mask for a trial with one potential outcome per unit.

    A preferred treatment is drawn from :func:`selection_probability` and a unit is retained only
    when its randomized assignment matches, which reproduces the preferred propensity in balanced
    trials. With known unequal allocation, arm ``a`` is accepted with probability
    ``k * g_a / pi_a``; a constant ``k`` keeps retention independent of measured and hidden causes.

    Args:
        z_bias: Biasing covariate values.
        treatment: Randomized binary assignment.
        beta: Selection coefficient.
        rng: Generator supplying the preference draws.
        intercept: Selection intercept.
        propensity: Optional explicit latent-conditioned preferred propensity.
        trial_probability: Known randomized treatment probabilities; switches to design-correct
            rejection.
        clusters: Optional cluster codes; draw retention once per cluster.

    Returns:
        Boolean mask of retained rows.
    """
    desired = selection_probability(z_bias, beta, intercept) if propensity is None else propensity
    if trial_probability is None:
        preferred = rng.random(len(z_bias)) < desired
        return preferred == treatment.astype(bool)
    pi = np.asarray(trial_probability)
    k = float(np.minimum(pi, 1 - pi).min())
    acceptance = k * np.where(treatment == 1, desired / pi, (1 - desired) / (1 - pi))
    if clusters is None:
        return rng.random(len(z_bias)) < acceptance
    unique, inverse = np.unique(clusters, return_inverse=True)
    for cluster in unique:
        if np.ptp(acceptance[clusters == cluster]) > 1e-10:
            raise ValueError("Cluster rejection requires constant acceptance within a cluster")
    return rng.random(len(unique))[inverse] < acceptance


def reveal_potential_outcome(z_bias: np.ndarray, beta: float, rng: np.random.Generator,
                             intercept: float = 0.0) -> np.ndarray:
    """Draw which arm is revealed when both potential outcomes are recorded.

    Nothing is rejected: the arm is drawn from :func:`selection_probability`, so every row is
    kept and ``P(X=1 | z)`` matches the OSRCT construction.

    Args:
        z_bias: Biasing covariate values.
        beta: Selection coefficient. Zero gives a balanced randomized reveal.
        rng: Generator supplying the draws.
        intercept: Selection intercept.

    Returns:
        Binary array of revealed arms, one per row.
    """
    return (rng.random(len(z_bias)) < selection_probability(z_bias, beta, intercept)).astype(np.uint8)


# --------------------------------------------------------------------------
# Dataset specifications and partitions
# --------------------------------------------------------------------------



@dataclass(frozen=True)
class SitePartition:
    """One site's randomized partition, constructed observational sample, and truth.

    Attributes:
        site: Site index.
        rct: Row indices of the randomized partition, which identifies the
            counterfactual marginals the tolerances are built from.
        idle: Row indices of the constructed observational sample.
        reference: Row indices supplying the ground-truth contrast; the idle rows themselves when
            all potential outcomes are known.
        exact_truth: Whether ``reference`` carries known potential outcomes
            rather than an estimated randomized contrast.
    """

    site: int
    rct: np.ndarray
    idle: np.ndarray
    reference: np.ndarray
    exact_truth: bool = False

@dataclass
class DatasetSpec:
    """Everything the comparison needs to turn a raw file into a fitted table.

    Attributes:
        name: Identifier used in outputs.
        path: File or directory to read, relative to the data root.
        treatment: Column holding the treatment.
        outcome: Column holding the outcome.
        covariates: Candidate covariate columns forming the search pool.
        site_column: Column used to construct analysis sites.
        site_buckets: When set, the site column is hashed into this many arbitrary buckets
            (:func:`make_bucketed_sites`) instead of being grouped by level size.
        rules: Binarisation rule per column, required for every covariate, treatment and outcome.
        row_filter: Optional ``(column, keep_values)`` restricting which rows enter, for example to
            reduce a multi-arm trial to a two-arm contrast.
        potential_outcomes: Optional ``(y0_column, y1_column)``; switches the comparison to direct
            reveal with exact ground truth.
        loader: Optional callable returning a DataFrame, for formats assembled from several files.
        source: Where the data comes from, recorded in the manifest.
        notes: Anything a downstream reader needs to interpret the results.
        assignment_columns: Optional candidate/hidden columns defining a latent-confounded
            assignment mechanism, kept from learners except a declared observed candidate.
        response_probabilities: Optional q0/q1 columns marginalized over latent causes, giving an
            oracle response-model reference distinct from the paired-outcome contrast.
        cluster_column: Unit kept intact in regime/reference splits and used
            for cluster-robust randomized-reference standard errors.
        trial_probability_column: Known trial allocation probability, for design-correct
            rejection under unequal randomized allocation.
    """

    name: str
    path: str
    treatment: str
    outcome: str
    covariates: tuple[str, ...]
    site_column: str
    site_buckets: int | None = None
    rules: dict[str, BinaryRule] = field(default_factory=dict)
    row_filter: tuple[str, tuple[Any, ...]] | None = None
    potential_outcomes: tuple[str, str] | None = None
    loader: Callable[[Path], pd.DataFrame] | None = None
    source: str = ""
    notes: str = ""
    assignment_columns: tuple[str, ...] | None = None
    response_probabilities: tuple[str, str] | None = None
    cluster_column: str | None = None
    trial_probability_column: str | None = None


def _load_twins(root: Path) -> pd.DataFrame:
    """Assemble the Twins benchmark from its three published CSV files.

    Reads ``twin_pairs_X``, ``twin_pairs_T`` and ``twin_pairs_Y`` (one row per pair) and keeps
    both twins' one-year mortality as ``y0`` and ``y1``. No revealed arm is picked here: the
    regime construction draws it per configuration.

    Args:
        root: Directory holding the three CSV files.

    Returns:
        One row per pair with ``y0``, ``y1`` and the covariates.

    Raises:
        FileNotFoundError: If any component file is absent.
        ValueError: If the two mortality columns cannot be identified.
    """
    frames = {}
    for key, stem in (("x", "twin_pairs_X"), ("t", "twin_pairs_T"), ("y", "twin_pairs_Y")):
        # Upstream names carry a sample suffix (twin_pairs_X_3years_samesex.csv); match the stem.
        matches = sorted(root.glob(f"{stem}*.csv")) if root.is_dir() else []
        if not matches:
            raise FileNotFoundError(f"Twins component missing: {root / (stem + '*.csv')}")
        frames[key] = pd.read_csv(matches[0])
    table = frames["x"].drop(columns=[c for c in frames["x"].columns if c.startswith("Unnamed")],
                             errors="ignore")
    outcomes = frames["y"]
    if {"mort_0", "mort_1"} <= set(outcomes.columns):
        y0, y1 = outcomes["mort_0"].to_numpy(), outcomes["mort_1"].to_numpy()
    else:
        numeric = outcomes.select_dtypes(include="number")
        numeric = numeric.drop(columns=[c for c in numeric.columns if c.startswith("Unnamed")],
                               errors="ignore")
        if numeric.shape[1] < 2:
            raise ValueError("Cannot identify the two mortality columns in "
                             f"{root / 'twin_pairs_Y.csv'}; saw {list(outcomes.columns)}")
        y0, y1 = numeric.iloc[:, 0].to_numpy(), numeric.iloc[:, 1].to_numpy()
    table["y0"], table["y1"] = y0, y1
    # Placeholders; the revealed arm is drawn per configuration in the regime step.
    table["treatment"], table["outcome"] = 0, y0
    return table


#: Binarisation rules for the reduced Twins covariates; each cut is a recognised clinical or
#: demographic threshold, not a sample quantile.
TWINS_RULES: dict[str, BinaryRule] = {
    "gestat10": BinaryRule(
        "at_most", 5, label="preterm gestation (before 37 weeks)",
        justification=("Preterm birth before 37 weeks is the standard obstetric threshold and the "
                       "dominant determinant of neonatal mortality, so it defines the clinically "
                       "meaningful stratum rather than a sample quantile. The column is the NCHS "
                       "ten-category gestation recode running 1 to 10, whose categories 1 to 5 "
                       "cover everything under 37 weeks; the observed data confirm the direction, "
                       "with mean birth weight rising monotonically from 404g in category 1 to "
                       "2574g in category 6 and mortality falling from 0.87 to 0.006 across the "
                       "same range. The cut selects 47.6 per cent of pairs, consistent with "
                       "published preterm rates for twin births. Category 10 is the not-stated "
                       "code and is counted as term by this rule, which affects 0.9 per cent of "
                       "pairs."),
        verified=True),
    "mager8": BinaryRule(
        "at_least", 6, label="advanced maternal age (35 or older)",
        justification=("Advanced maternal age of 35 years is the conventional obstetric risk "
                       "threshold in perinatal epidemiology. The column is the NCHS eight-category "
                       "maternal age recode running 1 to 8, whose category 6 begins at 35 years. "
                       "The cut selects 11.0 per cent of pairs."),
        verified=True),
    "meduc6": BinaryRule(
        "at_least", 3, label="completed high school or more",
        justification=("Completion of secondary education is the standard socioeconomic cut in "
                       "birth-outcome research and separates the two established risk groups. The "
                       "column is the NCHS maternal education recode whose category 3 is twelve "
                       "years of schooling; categories 1 and 2 are below that. The cut selects "
                       "76.1 per cent of pairs."),
        verified=True),
    "dlivord_min": BinaryRule(
        "at_most", 1, label="first live birth",
        justification=("Primiparity is a recognised obstetric risk category, distinct from later "
                       "births in both care pathway and outcome. The column runs from 1 to 17 in "
                       "the published data, so 1 is the first birth. The cut selects 34.6 per cent "
                       "of pairs."),
        verified=True),
    "dtotord_min": BinaryRule(
        "at_most", 1, label="first pregnancy overall",
        justification=("Total birth order separates first pregnancies, which differ systematically "
                       "in monitoring intensity, from subsequent ones. The column runs from 1 to "
                       "21 in the published data, so 1 is the first pregnancy."),
        verified=True),
    "nprevistq": BinaryRule(
        "at_least", 3, label="prenatal visits in the top two quintiles",
        justification=("Prenatal visit count is the input to the recognised adequacy-of-care "
                       "indices. The codebook describes this column as the quintile of the number "
                       "of prenatal visits and it runs 0 to 4, so the top two quintiles are the "
                       "finest split the source supports without inventing a visit-count "
                       "threshold the data cannot express. The cut selects 34.9 per cent of "
                       "pairs."),
        verified=True),
    "dfageq": BinaryRule(
        "at_least", 4, label="paternal age in the upper half of the distribution",
        justification=("The codebook describes this column as the octile of the father's age and "
                       "it runs 0 to 7, so the upper half begins at 4. No finer clinical "
                       "threshold, such as the conventional 40-year paternal age cut, is "
                       "recoverable from an octile coding. The cut selects 39.7 per cent of the "
                       "pairs that record it; paternal age is missing for 11,117 pairs, which "
                       "drop from any stratum using it."),
        verified=True),
    "birmon": BinaryRule(
        "in_set", (12, 1, 2), label="winter birth (December to February)",
        justification=("Season of birth is the classical near-excluded instrument in the Angrist "
                       "and Krueger tradition: it shifts exposure and care patterns while having "
                       "no plausible direct pathway to mortality given gestation and the remaining "
                       "covariates. Winter is the meteorologically standard grouping and was "
                       "chosen before any outcome was examined."),
        verified=True),
}

#: The reduced Twins covariate set used throughout the causal-inference literature.
TWINS_REDUCED_COVARIATES = (
    "birmon", "dfageq", "dlivord_min", "dtotord_min",
    "gestat10", "mager8", "meduc6", "nprevistq",
)

BUILTIN_SPECS: dict[str, DatasetSpec] = {
    "twins_reduced": DatasetSpec(
        name="twins_reduced",
        path="twins",
        treatment="treatment",
        outcome="outcome",
        covariates=TWINS_REDUCED_COVARIATES,
        site_column="mplbir",
        rules={**TWINS_RULES,
               "treatment": BinaryRule("passthrough", label="heavier twin revealed",
                                       justification="Set by the regime draw, not read from file."),
               "outcome": BinaryRule("passthrough", label="died within one year",
                                     justification="Native binary mortality indicator.")},
        potential_outcomes=("y0", "y1"),
        loader=_load_twins,
        source="NBER Linked Birth/Infant Death Cohort Data, via the standard Twins benchmark CSVs",
        notes=("Both potential outcomes are recorded, so the regime is built by direct reveal "
               "rather than rejection and the true effect of the units being bounded is known "
               "exactly. Sites come from mother's place of birth."),
    ),
}


def read_source_frame(spec: DatasetSpec, data_root: Path) -> pd.DataFrame:
    """Read one dataset's raw table, before any filtering or binarisation.

    Args:
        spec: Dataset specification.
        data_root: Directory that ``spec.path`` is relative to.

    Returns:
        The raw table exactly as the source provides it.

    Raises:
        FileNotFoundError: If the file or directory is absent.
    """
    target = data_root / spec.path
    if spec.loader is not None:
        return spec.loader(target)
    if not target.is_file():
        raise FileNotFoundError(f"Dataset file missing: {target}")
    if target.suffix == ".parquet":
        return pd.read_parquet(target)
    if target.suffix == ".dta":
        return pd.read_stata(target)
    return pd.read_csv(target)


def prepare_dataset(spec: DatasetSpec, data_root: Path, n_sites: int, min_site_rows: int,
                    allow_unspecified_rules: bool = False) -> dict[str, Any]:
    """Load one dataset and reduce it to the arrays the engine consumes.

    Args:
        spec: Dataset specification.
        data_root: Directory that ``spec.path`` is relative to.
        n_sites: Maximum number of analysis sites.
        min_site_rows: Minimum rows for a site to stay standalone.
        allow_unspecified_rules: Whether a column without a declared rule may fall back to a
            median split (off by default).

    Returns:
        Mapping with the covariate matrix and names, treatment, outcome, optional potential
        outcomes, site codes, and a provenance record carrying every binarisation rule and its
        justification.

    Raises:
        FileNotFoundError: If the dataset file or directory is absent.
        ValueError: If a required column is missing, a row filter empties the
            table, or a column has no declared rule while
            ``allow_unspecified_rules`` is off.
    """
    frame = read_source_frame(spec, data_root)
    metadata = frame.attrs.get("benchmark_metadata", {})
    sidecar = (data_root / spec.path).with_suffix(".metadata.json")
    if not metadata and sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))

    required = [spec.treatment, spec.outcome, spec.site_column, *spec.covariates]
    if spec.potential_outcomes:
        required.extend(spec.potential_outcomes)
    for columns in (spec.assignment_columns, spec.response_probabilities):
        if columns:
            required.extend(columns)
    required.extend(column for column in (spec.cluster_column, spec.trial_probability_column) if column)
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise ValueError(f"{spec.name} is missing columns: {absent}")

    filter_record = None
    if spec.row_filter is not None:
        column, keep = spec.row_filter
        observed = sorted(map(str, pd.unique(frame[column])))[:10]
        before = len(frame)
        frame = frame.loc[matches_levels(frame[column], keep)].reset_index(drop=True)
        filter_record = dict(column=column, kept_values=[str(value) for value in keep],
                             rows_before=before, rows_after=len(frame))
        if frame.empty:
            raise ValueError(f"{spec.name}: row filter on '{column}' kept nothing; the column's "
                             f"observed values begin {observed}")

    def rule_for(column: str) -> BinaryRule:
        """Look up a declared rule, allowing only an explicitly enabled fallback."""
        if column in spec.rules:
            return spec.rules[column]
        if not allow_unspecified_rules:
            raise ValueError(
                f"{spec.name}: no pre-specified binarisation rule for '{column}'. Declare one in "
                "the spec rather than falling back to a split chosen from the data.")
        return BinaryRule("median", label=f"{column} at or above its sample median",
                          justification="Fallback split; not pre-specified and not defensible as-is.",
                          verified=False)

    treatment, treatment_info = apply_binary_rule(frame[spec.treatment], rule_for(spec.treatment))
    outcome, outcome_info = apply_binary_rule(frame[spec.outcome], rule_for(spec.outcome))

    codes, descriptions, kept, dropped = [], {}, [], []
    for column in spec.covariates:
        column_codes, description = apply_binary_rule(frame[column], rule_for(column))
        descriptions[column] = description
        # A constant indicator makes every trigger and tolerance on it vacuously zero, which would
        # masquerade as a perfectly excluded covariate, so it is dropped.
        if description["degenerate"]:
            dropped.append(column)
            continue
        codes.append(column_codes)
        kept.append(column)
    if len(kept) < 2:
        raise ValueError(f"{spec.name}: only {len(kept)} non-constant covariates survived "
                         f"binarisation (dropped {dropped}); the search pool would be empty.")
    covariates = np.stack(codes, axis=1)

    complete = ((treatment != MISSING_CODE) & (outcome != MISSING_CODE)
                & (covariates != MISSING_CODE).all(axis=1))
    potential = None
    responses = None
    assignment = None
    trial_probability = None
    if spec.potential_outcomes:
        y0 = pd.to_numeric(frame[spec.potential_outcomes[0]], errors="coerce").to_numpy(float)
        y1 = pd.to_numeric(frame[spec.potential_outcomes[1]], errors="coerce").to_numpy(float)
        complete = complete & np.isfinite(y0) & np.isfinite(y1)
        if not set(np.concatenate([y0[np.isfinite(y0)], y1[np.isfinite(y1)]])) <= {0, 1}:
            raise ValueError(f"{spec.name}: potential outcomes must be binary")
        potential = (y0, y1)
    for attribute, columns in (("responses", spec.response_probabilities),
                               ("assignment", spec.assignment_columns)):
        if columns:
            values = tuple(pd.to_numeric(frame[column], errors="coerce").to_numpy(float) for column in columns)
            for value in values:
                complete &= np.isfinite(value)
                if np.any(np.isfinite(value) & ((value < 0) | (value > 1))):
                    raise ValueError(f"{spec.name}: {attribute} inputs must lie in [0,1]")
            if attribute == "responses":
                responses = values
            else:
                assignment = values
    if spec.trial_probability_column:
        trial_probability = pd.to_numeric(frame[spec.trial_probability_column], errors="coerce").to_numpy(float)
        complete &= np.isfinite(trial_probability)
        if np.any(np.isfinite(trial_probability) & ((trial_probability <= 0) | (trial_probability >= 1))):
            raise ValueError(f"{spec.name}: trial probabilities must lie in (0,1)")
    if spec.cluster_column:
        complete &= frame[spec.cluster_column].notna().to_numpy()
        groups = frame.groupby(spec.cluster_column)
        if groups[spec.site_column].nunique().gt(1).any():
            raise ValueError(f"{spec.name}: clusters cannot cross sites")
        if potential is None and groups[spec.treatment].nunique().gt(1).any():
            raise ValueError(f"{spec.name}: randomized assignment must be constant within clusters")
    if potential is not None:
        potential = tuple(values[complete] for values in potential)
    if responses is not None:
        responses = tuple(values[complete] for values in responses)
    if assignment is not None:
        assignment = tuple(values[complete] for values in assignment)

    frame = frame.loc[complete].reset_index(drop=True)
    sites, site_info = (make_bucketed_sites(frame, spec.site_column, spec.site_buckets)
                        if spec.site_buckets else
                        make_artificial_sites(frame, spec.site_column, n_sites, min_site_rows))
    unverified = sorted(name for name, record in descriptions.items() if not record["verified"])
    arbitrary = sorted(name for name, record in descriptions.items() if not record["prespecified"])
    if spec.assignment_columns:
        # Declared environments are kept as sites, never merged to meet a site-count limit.
        labels = frame[spec.site_column].astype("string")
        if labels.isna().any():
            raise ValueError(f"{spec.name}: declared environment IDs cannot be missing")
        sites, labels_unique = pd.factorize(labels, sort=True)
        site_info = dict(source_column=spec.site_column, n_sites=len(labels_unique),
                         site_names=list(labels_unique), grouping="declared environments, no pooling",
                         site_rows={str(label): int((labels == label).sum()) for label in labels_unique})

    return dict(
        covariates=covariates[complete].astype(np.uint8),
        treatment=treatment[complete].astype(np.uint8),
        outcome=outcome[complete].astype(np.uint8),
        potential=potential,
        responses=responses,
        assignment=assignment,
        clusters=(pd.factorize(frame[spec.cluster_column], sort=True)[0] if spec.cluster_column else None),
        trial_probability=(trial_probability[complete] if trial_probability is not None else None),
        sites=sites,
        provenance=dict(
            name=spec.name, source=spec.source, notes=spec.notes,
            benchmark_metadata=metadata,
            assignment_columns=spec.assignment_columns,
            response_probabilities=spec.response_probabilities,
            cluster_column=spec.cluster_column,
            reference_error_method=("village/cluster sandwich" if spec.cluster_column and potential is None
                                    else "oracle response-model reference" if responses is not None else "individual/paired"),
            n_rows_raw=int(len(complete)), n_rows_complete=int(complete.sum()),
            row_filter=filter_record,
            regime_construction=("direct reveal, all potential outcomes known"
                                 if potential is not None else "OSRCT rejection sampling"),
            treatment=treatment_info, outcome=outcome_info, covariates=descriptions,
            rules_needing_codebook_check=unverified,
            rules_not_prespecified=arbitrary,
            dropped_constant_covariates=dropped,
            sites=site_info,
        ),
        covariate_names=kept,
    )

def _shuffle_split(rows: np.ndarray, shares: Sequence[float], rng: np.random.Generator,
                   clusters: np.ndarray | None = None) -> list[np.ndarray]:
    """Split rows at cumulative shares, keeping optional cluster codes intact."""
    units = rows if clusters is None else np.unique(clusters[rows])
    shuffled = rng.permutation(units)
    cuts = [int(len(units) * share) for share in shares]
    parts = np.split(shuffled, cuts)
    return parts if clusters is None else [rows[np.isin(clusters[rows], part)] for part in parts]


def build_partitions(sites: np.ndarray, treatment: np.ndarray, covariates: np.ndarray,
                     bias_index: int, beta: float, reference_share: float,
                     rng: np.random.Generator,
                     potential: tuple[np.ndarray, np.ndarray] | None = None,
                     idle_probability: np.ndarray | None = None,
                     clusters: np.ndarray | None = None,
                     trial_probability: np.ndarray | None = None,
                     ) -> tuple[list[SitePartition], np.ndarray, np.ndarray]:
    """Split every site into its regimes, by the construction the data allows.

    With all potential outcomes known, each site is split in half: a randomized half with a
    balanced reveal, supplying the counterfactual marginals, and a confounded half that the method
    sees, whose own true effect is the reference. Otherwise a reference share is held out before
    any selection, and the rest splits into a randomized part and an OSRCT-selected part.

    Args:
        sites: Site index per row.
        treatment: Binary treatment per row, used only in the rejection path.
        covariates: ``(n, p)`` covariate matrix.
        bias_index: Column index of the biasing covariate.
        beta: Selection coefficient.
        reference_share: Fraction held out as ground truth, unless potential outcomes are known.
        rng: Generator for the splits and the selection draws.
        potential: Optional ``(y0, y1)`` arrays.
        idle_probability: Optional propensity using separately declared hidden
            causes; does not enter the observed conditioning matrix.
        clusters: Codes of units kept together between regimes/reference.
        trial_probability: Known trial probabilities for corrected rejection.

    Returns:
        ``(partitions, treatment, outcome)``. In the direct-reveal path the
        returned treatment and outcome are the newly revealed ones, so callers
        must use them in place of the inputs.
    """
    if potential is not None:
        y0, y1 = potential
        revealed = np.zeros(len(sites), dtype=np.uint8)
        partitions = []
        for site in np.unique(sites):
            rows = np.flatnonzero(sites == site)
            rct_rows, idle_rows = _shuffle_split(rows, [0.5], rng, clusters)
            # A balanced reveal in the randomized half, a biased one in the other.
            revealed[rct_rows] = reveal_potential_outcome(covariates[rct_rows, bias_index], 0.0, rng)
            revealed[idle_rows] = (reveal_potential_outcome(covariates[idle_rows, bias_index], beta, rng)
                                  if idle_probability is None else
                                  (rng.random(len(idle_rows)) < idle_probability[idle_rows]).astype(np.uint8))
            if len(rct_rows) and len(idle_rows):
                partitions.append(SitePartition(int(site), rct_rows, idle_rows,
                                                reference=idle_rows, exact_truth=True))
        observed = np.where(revealed == 1, y1, y0).astype(np.uint8)
        return partitions, revealed, observed

    partitions = []
    for site in np.unique(sites):
        rows = np.flatnonzero(sites == site)
        reference, rct, pool = _shuffle_split(rows, [reference_share, (1 + reference_share) / 2], rng, clusters)
        keep = osrct_select(covariates[pool, bias_index], treatment[pool], beta, rng,
                            propensity=idle_probability[pool] if idle_probability is not None else None,
                            trial_probability=trial_probability[pool] if trial_probability is not None else None,
                            clusters=clusters[pool] if clusters is not None else None)
        idle = pool[keep]
        if len(reference) and len(rct) and len(idle):
            partitions.append(SitePartition(int(site), rct, idle, reference=reference))
    return partitions, treatment, np.asarray([])
