"""OSRCT data-file discovery and the OSRCT benchmark generator code.

The generator lives in ``osrct_benchmark_v1.0/code``. The dot in that name
makes the directory unimportable as a package, so
:func:`load_osrct_benchmark_module` loads its modules by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

#: Directory of the OSRCT benchmark's generator code.
OSRCT_BENCHMARK_CODE_DIR = Path(__file__).resolve().parents[1] / "osrct_benchmark_v1.0" / "code"
#: The benchmark's modules, each listed after the siblings it imports by bare name.
OSRCT_BENCHMARK_MODULES = ("osrct", "experimental_grid", "generate_confounded_datasets")

SUPPORTED_MANYLABS_STUDIES = (
    "gainloss",
    "reciprocity",
    "allowedforbidden",
)


def find_osrct_file(config: Any) -> Path:
    """
    Find an OSRCT file for a study, pattern, beta, and seed configuration.

    Args:
        config: Object exposing ``data_root``, ``study``, ``pattern``, ``beta``,
            and ``seed`` attributes.

    Returns:
        Path to the first matching Parquet, CSV, or pickle file.

    Raises:
        FileNotFoundError: If none of the supported naming conventions exist.
    """
    data_root = Path(config.data_root)
    study_dir = data_root / config.study
    beta = config.beta
    stems = [
        f"{config.pattern}_beta{beta}_seed{config.seed}",
        f"{config.pattern}_beta{beta:g}_seed{config.seed}",
        f"{config.pattern}_beta{beta:.1f}_seed{config.seed}",
        f"{config.pattern}_beta{beta:.2f}_seed{config.seed}",
        f"{config.study}_{config.pattern}_beta_{beta}",
    ]
    candidates = [
        study_dir / f"{stem}{suffix}"
        for stem in stems
        for suffix in (".parquet", ".csv", ".pkl")
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "OSRCT file not found. Tried:\n"
        + "\n".join(f"  {path}" for path in candidates)
    )


def load_osrct_benchmark_module(name: str) -> ModuleType:
    """Load one module of the OSRCT benchmark code by its file path.

    The generator imports its siblings by bare name (``from osrct import ...``).
    The function therefore loads ``name`` and the modules listed before it in
    :data:`OSRCT_BENCHMARK_MODULES`, and registers each in ``sys.modules`` under
    its bare name. A module that is already registered from the same file is
    reused, so repeated calls return the same module object.

    Args:
        name: Module name without ``.py``, one of :data:`OSRCT_BENCHMARK_MODULES`.

    Returns:
        The loaded module.

    Raises:
        ValueError: If ``name`` is not a benchmark module.
        ImportError: If ``sys.modules`` already holds a different module of that name.
    """
    if name not in OSRCT_BENCHMARK_MODULES:
        raise ValueError(f"Unknown OSRCT benchmark module {name!r}; choose from {OSRCT_BENCHMARK_MODULES}")
    module = None
    for dependency in OSRCT_BENCHMARK_MODULES[: OSRCT_BENCHMARK_MODULES.index(name) + 1]:
        path = OSRCT_BENCHMARK_CODE_DIR / f"{dependency}.py"
        module = sys.modules.get(dependency)
        if module is not None:
            if Path(getattr(module, "__file__", "") or "").resolve() != path.resolve():
                raise ImportError(f"sys.modules[{dependency!r}] is {module!r}, not {path}")
            continue
        spec = importlib.util.spec_from_file_location(dependency, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[dependency] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[dependency]
            raise
    return module
