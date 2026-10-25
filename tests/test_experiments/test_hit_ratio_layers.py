"""Layering of the Exp25 hit-ratio analysis and the shared figure-style registry."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import experiments.exp25_bootstrap_analysis as bootstrap
import experiments.exp25_hit_ratio_plots as plots
import experiments.hit_ratio_metrics as metrics
import experiments.method_styles as styles
import experiments.naturalness_comparison_results as results

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def _experiment_imports(module: str) -> set[str]:
    """Return the ``experiments.*`` modules a file imports, at top level or lazily."""

    tree = ast.parse((EXPERIMENTS / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "experiments":
                found.update(alias.name for alias in node.names)
            elif node.module.startswith("experiments."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            found.update(
                alias.name.split(".")[1]
                for alias in node.names
                if alias.name.startswith("experiments.")
            )
    return found


@pytest.mark.parametrize("module", ["hit_ratio_metrics", "method_styles"])
def test_shared_layers_import_no_experiment_script(module: str) -> None:
    assert not {name for name in _experiment_imports(module) if name.startswith("exp")}


@pytest.mark.parametrize("module", ["exp25_hit_ratio_plots", "exp25_bootstrap_analysis"])
def test_plots_and_bootstrap_do_not_import_the_orchestrator(module: str) -> None:
    assert "exp25_hit_ratio_analysis" not in _experiment_imports(module)


def test_bootstrap_defaults_have_one_definition() -> None:
    assert bootstrap.DEFAULT_BOOTSTRAP_REPLICATES == metrics.DEFAULT_BOOTSTRAP_REPLICATES
    assert bootstrap.DEFAULT_BOOTSTRAP_SEED == metrics.DEFAULT_BOOTSTRAP_SEED
    assert bootstrap.DEFAULT_CONFIDENCE_LEVEL == metrics.DEFAULT_CONFIDENCE_LEVEL


def test_figures_share_the_registry_styles() -> None:
    assert plots.MODEL_LABELS is styles.MODEL_LABELS
    assert plots.MODEL_COLORS is styles.MODEL_COLORS
    assert results.PAPER_RC is plots.PAPER_RC is styles.PAPER_RC
    for method, model in styles.EXP25_MODEL_FOR_METHOD.items():
        assert results.METHOD_COLORS[method] == styles.MODEL_COLORS[model]
        assert results.METHOD_LABELS[method] == styles.MODEL_LABELS[model]
    for method in styles.METHOD_ORDER:
        assert method in styles.METHOD_COLORS
        assert method in styles.METHOD_MARKERS
