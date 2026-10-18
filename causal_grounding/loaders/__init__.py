"""
Raw-data loading — the single point of contact between datasets and the grounder.

Five public symbols. Everything else is implementation.

The loader reads a raw file, canonicalises it, derives covariates, splits it by
site, and enforces eleven invariants. It **does not sample and does not build
regimes**; those stay in the experiment scripts, because they own an RNG and a
``beta``.

Adding a dataset is one new module implementing :class:`SiteLoader`, plus one
``@register_site_loader`` line. No edits to the contract, the validator, or any
existing loader.

See ``docs/loader_contract_spec.md`` for the contract.
"""

from typing import Dict, Type, Callable, Any, Optional

from .contract import (
    RawDataset,
    RawFormat,
    SiteTable,
    SiteBundle,
    SiteLoader,
)
from .validation import (
    LoaderValidationError,
    validate_bundle,
    reset_loaded_datasets,
)

__all__ = [
    "RawDataset",
    "SiteTable",
    "SiteBundle",
    "SiteLoader",
    "load_sites",
    # Supporting surface — needed to extend or to test, not to use.
    "RawFormat",
    "LoaderValidationError",
    "register_site_loader",
    "create_site_loader",
    "available_site_loaders",
    "reset_loaded_datasets",
]


# =============================================================================
# REGISTRY
# =============================================================================

_SITE_LOADER_REGISTRY: Dict[str, Type] = {}


def register_site_loader(name: str) -> Callable[[Type], Type]:
    """Register a loader implementation under ``name``.

    Args:
        name: Dataset family key, e.g. ``"osrct"``.

    Returns:
        A class decorator returning the class unchanged.

    Example:
        @register_site_loader("ihdp")
        class IHDPLoader:
            def read(self) -> SiteBundle: ...
    """
    def decorator(cls: Type) -> Type:
        if name in _SITE_LOADER_REGISTRY:
            raise ValueError(
                f"site loader {name!r} is already registered to "
                f"{_SITE_LOADER_REGISTRY[name].__name__}"
            )
        _SITE_LOADER_REGISTRY[name] = cls
        return cls
    return decorator


def create_site_loader(family: str, dataset: RawDataset, **kwargs: Any) -> SiteLoader:
    """Instantiate the loader registered for ``family``.

    Raises:
        KeyError: If no loader is registered under that name.
    """
    if family not in _SITE_LOADER_REGISTRY:
        raise KeyError(
            f"no site loader registered for {family!r}. "
            f"Available: {available_site_loaders()}"
        )
    return _SITE_LOADER_REGISTRY[family](dataset, **kwargs)


def available_site_loaders() -> tuple:
    """Registered dataset family names, sorted."""
    return tuple(sorted(_SITE_LOADER_REGISTRY))


# =============================================================================
# THE ONE PUBLIC ENTRY POINT
# =============================================================================

def load_sites(
    dataset: RawDataset,
    *,
    family: Optional[str] = None,
    min_rows: int = 1,
    strict_covariate_variation: bool = False,
    strict_dataset_names: bool = False,
    **loader_kwargs: Any,
) -> SiteBundle:
    """Load a dataset and return a fully validated bundle.

    Every invariant is applied here rather than inside each loader, so that no
    implementation can forget one and adding a dataset cannot weaken the
    guarantee.

    Args:
        dataset: What to load.
        family: Registry key. Defaults to ``dataset.name`` split on the first
            underscore, so ``"osrct_gainloss"`` resolves to ``"osrct"``.
        min_rows: Invariant 8's threshold. The default of 1 reproduces current
            behaviour; read ``validation.check_min_rows`` before raising it.
        strict_covariate_variation: Make invariant 9 fatal. Off by default
            because real data violates it — ``resp_age_bin`` is constant at 11
            of 108 OSRCT site-study combinations. The offenders are recorded in
            ``provenance["constant_covariates"]`` either way.
        **loader_kwargs: Passed through to the loader's constructor.

    Returns:
        A validated :class:`SiteBundle`, with ``constant_covariates`` added to
        its provenance.

    Raises:
        LoaderValidationError: On the first invariant violation.
        KeyError: If no loader is registered for the family.
    """
    resolved = family or dataset.name.split("_", 1)[0]
    loader = create_site_loader(resolved, dataset, **loader_kwargs)
    return validate_bundle(
        loader.read(),
        min_rows=min_rows,
        strict_covariate_variation=strict_covariate_variation,
        strict_dataset_names=strict_dataset_names,
    )


# Import for side effect: registration. Kept at the bottom so the registry
# functions above are defined first.
from . import osrct as _osrct        # noqa: E402,F401
from . import manylabs_rct as _mlrct  # noqa: E402,F401

register_site_loader("osrct")(_osrct.OSRCTLoader)
register_site_loader("manylabs")(_osrct.OSRCTLoader)
register_site_loader("manylabsrct")(_mlrct.ManyLabsRCTLoader)
