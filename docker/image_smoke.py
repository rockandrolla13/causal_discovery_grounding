"""Verify imports and the PyTorch flavor of a built image.

The Dockerfile runs this at build time. It also checks a local environment:

    python docker/image_smoke.py --torch-variant cpu
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch

# The repository and BLearner are separate source roots, and must be on the
# path before the imports below can resolve. The OSRCT benchmark code is
# loaded by file path (experiments.osrct_files.load_osrct_benchmark_module).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLEARNER_ROOT = Path(os.environ.get("B_LEARNER_REPOSITORY", "/opt/BLearner"))
sys.path[:0] = [
    str(BLEARNER_ROOT),
    str(PROJECT_ROOT),
]

RUNTIME_MODULES = (
    "jax",
    "numpyro",
    "catenets",
    "econml",
    "pyarrow",
    "pyreadstat",
    "pytorch_lightning",
    "sklearn_quantile",
    "tabicl",
    "xgboost",
    "yaml",
    "experiments.backdoor_cate_error_analysis",
    "experiments.naturalness_plotting",
    "experiments.exp16_osrct_calibrated_policy",
    "experiments.exp18_cb_meta_learner_comparison",
    "experiments.exp25_osrct_mc_baseline_comparison",
    "experiments.naturalness_comparison_data",
    # The comparison entry points and the arms they dispatch. Without these the
    # smoke test passes on an image whose Exp28 code cannot even import,
    # and the failure surfaces only after the job has been scheduled.
    "experiments.naturalness_comparison",
    "experiments.exp28_twins_method_comparison",
    "experiments.exp31_bayesian_manylabs_naturalness",
    # The read-only merge/report half. It pulls in matplotlib, so importing it
    # here is also the check that the image's plotting backend is usable.
    "experiments.naturalness_comparison_results",
    # The other entry points of the paper scripts.
    "experiments.exp25_tau_floor_regrounding",
    "experiments.exp25_hit_ratio_analysis",
    "experiments.exp25_hit_ratio_plots",
    "causal_grounding",
    # Sharp Balke-Pearl bounds depend on scipy.optimize.linprog with the HiGHS
    # method, so importing this also checks the SciPy wheel is usable.
    "causal_grounding.balke_pearl",
    "causal_grounding.bayesian_modular_naturalness",
)


def main() -> None:
    """Import runtime modules and validate the selected PyTorch wheel."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-variant", choices=("cpu", "cu124"), required=True)
    args = parser.parse_args()

    expect_cuda = args.torch_variant == "cu124"
    assert (torch.version.cuda is not None) == expect_cuda, (
        f"torch {torch.__version__} does not match --torch-variant {args.torch_variant}"
    )

    for module_name in RUNTIME_MODULES:
        importlib.import_module(module_name)

    from experiments.osrct_files import load_osrct_benchmark_module

    # Loaded at run time from osrct_benchmark_v1.0/code by Exp18 and Exp25.
    load_osrct_benchmark_module("generate_confounded_datasets")
    assert hasattr(load_osrct_benchmark_module("osrct").OSRCTSampler, "selection_probability")
    if BLEARNER_ROOT.is_dir():
        from models.blearner import BLearner  # noqa: F401
    elif "B_LEARNER_REPOSITORY" in os.environ:
        raise SystemExit(f"B_LEARNER_REPOSITORY={BLEARNER_ROOT} does not exist")
    else:
        print(f"note: no BLearner checkout at {BLEARNER_ROOT}; its import is not checked")
    print(f"ok: {len(RUNTIME_MODULES)} modules import with torch {torch.__version__}")


if __name__ == "__main__":
    main()
