"""
Public types and the loader protocol for raw-data ingestion.

Three concepts are public: :class:`RawDataset` identifies what to load,
:class:`SiteTable` is one validated per-site table, and :class:`SiteBundle` is
everything a loader returns for one dataset. :class:`SiteLoader` is the
one-method protocol every dataset implementation satisfies.

Loader reads, binarises outcome, derives covariates, splits by site and
validates. It **does not sample and does not build regimes** — those stay in
experiment scripts. Frames returned here therefore carry no regime column.
"""

from typing import (
    Protocol, Mapping, Any, Literal, Tuple, runtime_checkable,
)
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


RawFormat = Literal["parquet", "csv", "pkl"]


# =============================================================================
# WHAT TO LOAD
# =============================================================================

@dataclass(frozen=True)
class RawDataset:
    """Identifies a raw file to load. No behaviour.

    Attributes:
        name: Stable identifier, e.g. ``"manylabs_gainloss"``.
            Must be unique across bundles loaded in one process (invariant 10).
        path: Path to the raw file.
        format: File format. A closed set — an unrecognised suffix is an error
            rather than a silent fall-through to ``read_csv``.
    """

    name: str
    path: Path
    format: RawFormat


# =============================================================================
# WHAT COMES BACK
# =============================================================================

@dataclass(frozen=True)
class SiteTable:
    """One validated site's table.

    Attributes:
        site_id: Site identifier, always a ``str``.
        frame: The site's rows, in source order. See ``source_positions``.
        treatment: Column name of the binary treatment.
        outcome: Column name of the binary outcome.
        covariates: Ordered covariate column names. Identical across every
            site in a bundle (invariant 5).
        unit_id: Column name of the row identifier, unique within this site.
        source_positions: Positional offsets of these rows in the parent frame,
            before the site split.

    Note:
        ``source_positions`` is not in ``docs/loader_contract_spec.md`` §4. It
        exists because invariant 11 — preserve row order and membership — is
        otherwise not checkable: a reordered frame has the same shape, the same
        columns and the same values, so nothing else distinguishes it.

        This matters whenever downstream sampling uses positional arrays.
        Reordering a row can silently change selection probabilities.
    """

    site_id: str
    frame: pd.DataFrame
    treatment: str
    outcome: str
    covariates: Tuple[str, ...]
    unit_id: str
    source_positions: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class SiteBundle:
    """Every site from one dataset, validated as a group.

    Attributes:
        dataset: The ``RawDataset.name`` this came from.
        sites: Site id to :class:`SiteTable`. Insertion order is sorted site id.
        covariates: The shared covariate tuple, asserted equal across all sites.
        provenance: Source path, file hash, row counts before and after any
            filtering, and the reason for anything dropped.
    """

    dataset: str
    sites: Mapping[str, SiteTable]
    covariates: Tuple[str, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def site_ids(self) -> Tuple[str, ...]:
        """Site ids in bundle order."""
        return tuple(self.sites)

    @property
    def n_rows(self) -> int:
        """Total rows across every site."""
        return sum(len(table) for table in self.sites.values())

    def to_frame(self) -> pd.DataFrame:
        """Reassemble every site into one frame, in original source order.

        The inverse of the site split. Exists because several consumers want the
        flat table rather than the mapping — ``make_osrct_regimes`` does its own
        splitting, and ``fit_model_implied_reference`` fits across all sites at
        once.

        Order is restored from ``source_positions``, so this returns the rows in
        the order the source file had them. That matters: regime construction
        aligns a positional array against this frame, so a differently-ordered
        reassembly would silently change every site's numbers.
        """
        ordered = sorted(
            self.sites.values(), key=lambda table: table.source_positions[0]
        )
        return pd.concat([table.frame for table in ordered]).sort_index()


# =============================================================================
# THE PROTOCOL
# =============================================================================

@runtime_checkable
class SiteLoader(Protocol):
    """Contract for any dataset implementation.

    One method. Validation is not the loader's job — :func:`load_sites` applies
    every invariant centrally so that no implementation can forget one, and so
    that adding a dataset cannot weaken the guarantee.

    Adding a dataset means one new module implementing this protocol plus its
    canonicalisation map, and zero edits to existing files.
    """

    def read(self) -> SiteBundle:
        """Read, canonicalise, derive covariates and split by site.

        Returns:
            An unvalidated bundle. :func:`load_sites` validates it.

        Raises:
            FileNotFoundError: If the source file is absent.
        """
        ...
