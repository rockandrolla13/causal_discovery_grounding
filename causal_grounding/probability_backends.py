"""Single registry of the probability backends that fit ``P(X | Z)`` and ``P(Y | X, Z)``.

A probability backend is a named model family. One registry entry states what the
name can do:

* ``predictor``: the :class:`~causal_grounding.predictors.BasePredictorModel`
  factory that :func:`~causal_grounding.predictors.create_predictor` returns;
* ``stage1``: the Stage-1 PMF adapter that
  :class:`~causal_grounding.naturalness_grounder.NaturalnessGrounder` fits per
  source site;
* ``target_lp``: whether the target-LP nuisance estimator
  (:func:`~causal_grounding.lp_solver_learned_naturalness.estimate_target_observed_binary_quantities`)
  accepts the name; and
* ``posterior_draws``: whether the name asks for one posterior draw of the
  fitted model instead of its point estimate.

Every other backend list in the package and in ``experiments/utils.py`` derives
from this registry. A new backend therefore needs one
:func:`register_probability_backend` call. The built-in entries name their
factories as ``"module:attribute"`` strings, so this module imports no model code,
and a backend's dependencies load only when the backend is used.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any

FactoryRef = Callable[..., Any] | str


def _resolve(ref: FactoryRef | None) -> Callable[..., Any] | None:
    """Import a ``"module:attribute"`` reference, or return a callable unchanged.

    Args:
        ref: A callable, a ``"module:attribute"`` string, or ``None``.

    Returns:
        The referenced callable, or ``None`` when ``ref`` is ``None``.
    """
    if ref is None or callable(ref):
        return ref
    module_name, _, attribute = ref.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


@dataclass(frozen=True)
class ProbabilityBackend:
    """Name and capabilities of one probability backend.

    Attributes:
        name: Backend string accepted by the factories and configurations.
        predictor_ref: Predictor factory, or its ``"module:attribute"`` path.
            ``None`` when the name only has a Stage-1 adapter.
        stage1_ref: Stage-1 PMF adapter factory, or its ``"module:attribute"``
            path. ``None`` when the name cannot drive Stage 1.
        target_lp: Whether the target-LP nuisance estimator accepts the name.
        posterior_draws: Whether the name asks for one posterior draw.
        model_based: ``False`` for the closed-form empirical cell estimator,
            which the target LP and the experiments treat separately.
        fit_backend: Predictor name the target LP fits for this name. Defaults
            to ``name``; a posterior-draw alias names its point-estimate family.
    """

    name: str
    predictor_ref: FactoryRef | None = None
    stage1_ref: FactoryRef | None = None
    target_lp: bool = False
    posterior_draws: bool = False
    model_based: bool = True
    fit_backend: str | None = None

    @property
    def stage1(self) -> bool:
        """Whether the backend can drive Stage 1."""
        return self.stage1_ref is not None

    @property
    def predictor_factory(self) -> Callable[..., Any] | None:
        """The predictor factory, imported on first access."""
        return _resolve(self.predictor_ref)

    @property
    def stage1_factory(self) -> Callable[..., Any] | None:
        """The Stage-1 adapter factory, imported on first access."""
        return _resolve(self.stage1_ref)

    @property
    def fit_name(self) -> str:
        """Predictor name the target LP fits for this backend."""
        return self.fit_backend or self.name


_REGISTRY: dict[str, ProbabilityBackend] = {}


def register_probability_backend(
    name: str,
    *,
    predictor: FactoryRef | None = None,
    stage1: FactoryRef | None = None,
    target_lp: bool = False,
    posterior_draws: bool = False,
    model_based: bool = True,
    fit_backend: str | None = None,
) -> ProbabilityBackend:
    """Register, or replace, a probability backend under ``name``.

    Args:
        name: Backend string. Lookups match it exactly.
        predictor: Predictor factory or ``"module:attribute"`` path.
        stage1: Stage-1 PMF adapter factory or ``"module:attribute"`` path.
        target_lp: Whether the target-LP nuisance estimator accepts ``name``.
            It needs ``predictor`` unless ``model_based`` is ``False``.
        posterior_draws: Whether ``name`` asks for one posterior draw.
        model_based: ``False`` only for a closed-form cell estimator.
        fit_backend: Predictor name the target LP fits. Defaults to ``name``.

    Returns:
        The registered entry.

    Raises:
        ValueError: If ``name`` is empty, or a model-based target-LP backend
            has no predictor.
        TypeError: If a factory is neither callable nor a string path.
    """
    if not name:
        raise ValueError("backend name must be non-empty")
    for ref in (predictor, stage1):
        if ref is not None and not (callable(ref) or isinstance(ref, str)):
            raise TypeError("backend must be callable")
    if target_lp and model_based and predictor is None and fit_backend is None:
        raise ValueError(f"target-LP backend {name!r} needs a predictor factory")
    entry = ProbabilityBackend(
        name=name,
        predictor_ref=predictor,
        stage1_ref=stage1,
        target_lp=bool(target_lp),
        posterior_draws=bool(posterior_draws),
        model_based=bool(model_based),
        fit_backend=fit_backend,
    )
    _REGISTRY[name] = entry
    return entry


def update_probability_backend(name: str, **changes: Any) -> ProbabilityBackend:
    """Change fields of a registered backend, or register a new one.

    Args:
        name: Backend string.
        **changes: :class:`ProbabilityBackend` field values to set.

    Returns:
        The updated entry.
    """
    entry = _REGISTRY.get(name, ProbabilityBackend(name=name))
    entry = dataclasses.replace(entry, **changes)
    _REGISTRY[name] = entry
    return entry


def unregister_probability_backend(name: str) -> None:
    """Remove ``name`` from the registry.

    Args:
        name: Backend string.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    del _REGISTRY[name]


def get_probability_backend(name: str) -> ProbabilityBackend:
    """Return the registry entry for ``name``.

    Args:
        name: Backend string.

    Returns:
        The registered entry.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    return _REGISTRY[name]


def probability_backend_names(
    *,
    predictor: bool | None = None,
    stage1: bool | None = None,
    target_lp: bool | None = None,
    posterior_draws: bool | None = None,
    model_based: bool | None = None,
    fit_backend: str | None = None,
) -> tuple[str, ...]:
    """Return registered names, in registration order, that match every filter.

    A filter left at ``None`` does not restrict the result.

    Args:
        predictor: Require (or exclude) a predictor factory.
        stage1: Require (or exclude) a Stage-1 adapter.
        target_lp: Require (or exclude) target-LP support.
        posterior_draws: Require (or exclude) posterior-draw aliases.
        model_based: Require (or exclude) model-based backends.
        fit_backend: Require this target-LP fit name.

    Returns:
        Matching backend names.
    """

    def matches(entry: ProbabilityBackend) -> bool:
        """Check one entry against every active filter."""
        checks = (
            (predictor, entry.predictor_ref is not None),
            (stage1, entry.stage1),
            (target_lp, entry.target_lp),
            (posterior_draws, entry.posterior_draws),
            (model_based, entry.model_based),
        )
        if any(want is not None and want != have for want, have in checks):
            return False
        return fit_backend is None or entry.fit_name == fit_backend

    return tuple(name for name, entry in _REGISTRY.items() if matches(entry))


class Stage1BackendMapping(MutableMapping):
    """Live ``name -> Stage-1 factory`` view of the registry.

    It backs ``naturalness_grounder.NATURALNESS_STAGE1_BACKENDS``. Reading a name
    imports its adapter. Assigning a name registers its Stage-1 adapter and
    keeps the name's other capabilities. Deleting a name removes only its
    Stage-1 adapter.
    """

    def __getitem__(self, name: str) -> Callable[..., Any]:
        """Return the Stage-1 factory of ``name``."""
        entry = _REGISTRY.get(name)
        if entry is None or not entry.stage1:
            raise KeyError(name)
        return entry.stage1_factory

    def __setitem__(self, name: str, backend: Callable[..., Any]) -> None:
        """Register ``backend`` as the Stage-1 adapter of ``name``."""
        update_probability_backend(name, stage1_ref=backend)

    def __delitem__(self, name: str) -> None:
        """Remove the Stage-1 adapter of ``name``."""
        if name not in self:
            raise KeyError(name)
        entry = _REGISTRY[name]
        if entry.predictor_ref is None and not entry.target_lp:
            del _REGISTRY[name]
        else:
            _REGISTRY[name] = dataclasses.replace(entry, stage1_ref=None)

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` has a Stage-1 adapter."""
        entry = _REGISTRY.get(name) if isinstance(name, str) else None
        return entry is not None and entry.stage1

    def __iter__(self) -> Iterator[str]:
        """Iterate over Stage-1 names in registration order."""
        return iter(probability_backend_names(stage1=True))

    def __len__(self) -> int:
        """Return the number of Stage-1 names."""
        return len(probability_backend_names(stage1=True))

    def __repr__(self) -> str:
        """Show the registered Stage-1 names."""
        return f"{type(self).__name__}({list(self)!r})"


_PREDICTORS = "causal_grounding.predictors"
_GROUNDER = "causal_grounding.naturalness_grounder"

# Built-in backends. The registration order fixes the order of every derived list.
register_probability_backend(
    "empirical",
    predictor=f"{_PREDICTORS}:EmpiricalPredictor",
    stage1=f"{_GROUNDER}:EmpiricalSitePMFModels",
    target_lp=True,
    model_based=False,
)
register_probability_backend(
    "xgboost",
    predictor=f"{_PREDICTORS}:_create_xgboost_predictor",
    stage1=f"{_GROUNDER}:XGBoostSitePMFModels",
    target_lp=True,
)
register_probability_backend(
    "hierarchical_dirichlet",
    predictor=f"{_PREDICTORS}:HierarchicalDirichletPredictor",
    stage1=f"{_GROUNDER}:HierarchicalDirichletSitePMFModels",
    target_lp=True,
)
for _alias in (
    "hierarchical_dirichlet_draws",
    "hierarchical_dirichlet_posterior",
    "dirichlet_posterior",
):
    register_probability_backend(
        _alias,
        predictor=f"{_PREDICTORS}:HierarchicalDirichletPredictor",
        target_lp=True,
        posterior_draws=True,
        fit_backend="hierarchical_dirichlet",
    )
del _alias
register_probability_backend(
    "tabicl_v2",
    predictor=f"{_PREDICTORS}:TabICLv2Predictor",
    stage1=f"{_GROUNDER}:TabICLv2SitePMFModels",
    target_lp=True,
)


__all__ = [
    "ProbabilityBackend",
    "Stage1BackendMapping",
    "get_probability_backend",
    "probability_backend_names",
    "register_probability_backend",
    "unregister_probability_backend",
    "update_probability_backend",
]
