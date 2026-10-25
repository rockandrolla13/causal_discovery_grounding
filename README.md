# Naturalness

Code for the paper *Causal Discovery Grounding and the Naturalness Assumption*: the `causal_grounding` Python package, which implements Naturalness grounding, and the scripts that reproduce every table and figure of the paper's experiments.

Naturalness bounds the conditional average treatment effect (CATE) of a binary treatment on a binary outcome at a **target site** that has observational data only. It learns from **source sites** that hold both a randomized trial and observational data. On the sources it measures when a covariate $Z_j$ behaves like a natural experiment: with treatment and the other covariates held fixed, the observed outcome barely changes across $Z_j$ (the condition $\mathcal I_j(\epsilon)$), while treatment uptake does change with $Z_j$ (the condition $\mathcal D_j(\tau)$). It then learns, with Gaussian processes, how far the unobserved potential-outcome distributions can still vary when both conditions hold (the tolerances $c$ and $d$). At the target, the conditions that fire turn these tolerances into the constraints of a linear program whose minimum and maximum bound the CATE of each covariate stratum.

## Contents

| Path | What it holds |
|---|---|
| [`causal_grounding/`](causal_grounding/) | The package. The Naturalness pipeline is in `naturalness_grounder.py` (Stages 1–3: triggers, tolerances, Gaussian-process surfaces), `naturalness_target_policy.py` (target triggers), `lp_solver_learned_naturalness.py` (the target linear program) and `bayesian_modular_naturalness.py` (the cut-Bayesian pipeline of Experiment III). `balke_pearl.py` holds the Balke–Pearl comparator. |
| [`examples/toy_naturalness.py`](examples/toy_naturalness.py) | The toy example below, on simulated data. |
| [`paper/`](paper/) | One script per part of the paper's results, their documentation ([`paper/README.md`](paper/README.md)) and the paper's figures (`paper/img/`). |
| [`experiments/`](experiments/) | The experiment modules the paper scripts run: Exp25 is Experiment I, Exp28 Experiment II and Exp31 Experiment III. |
| [`osrct_benchmark_v1.0/code/`](osrct_benchmark_v1.0/code/) | The OSRCT sampler that turns the ManyLabs1 trials into confounded observational samples. |
| [`ManyLabs1/pre-process/`](ManyLabs1/pre-process/) | Conversion of the raw ManyLabs1 file into the table the experiments read. |
| [`tests/`](tests/) | Unit tests of the package and the experiment modules, a test of the toy example and dry runs of the paper scripts. |
| [`Dockerfile`](Dockerfile), [`docker/`](docker/) | The runtime image, CPU or CUDA. |
| [`requirements.txt`](requirements.txt), [`requirements-dev.txt`](requirements-dev.txt), [`pyproject.toml`](pyproject.toml) | The pinned environment of the paper's runs, its test extras, and the package metadata. |

## Installation

The paper's results were produced with Python 3.11, torch 2.5.1 (CUDA 12.4 on GPU machines) and the pinned packages of `requirements.txt`. Install torch first from the PyTorch index for your machine, so that pip does not pick the default PyPI wheel:

```bash
conda create -n naturalness python=3.11
conda activate naturalness
# GPU machines (CUDA 12.4 build):
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
# CPU-only machines instead:
# pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
pip install -r requirements.txt      # or requirements-dev.txt to also run the tests
pip install -e .                     # optional: `import causal_grounding` from anywhere
```

The package on its own needs only numpy, pandas, scipy and scikit-learn, which is all the toy example uses: `pip install .` installs just those. The other packages back particular parts of the pipeline: torch and `tabicl` for the TabICLv2 probability model the paper uses, JAX and NumPyro for Experiment III, CATENets, EconML and BLearner for the comparators.

### Docker

The [`Dockerfile`](Dockerfile) builds the same environment, with BLearner at the commit the paper used (`734fd76`):

```bash
docker build -t naturalness:cpu .                                    # CPU only, about 2.5 GB
docker build -t naturalness:cu124 --build-arg TORCH_VARIANT=cu124 .  # NVIDIA GPUs (CUDA 12.4)

docker run --rm naturalness:cpu                                      # runs the toy example
docker run --rm naturalness:cpu python -m pytest -q                  # runs the tests
```

GPU containers need the NVIDIA Container Toolkit and `docker run --gpus all`. The image holds no data and no results. To run the paper scripts, mount your checkout over the image's copy so that data, results and logs stay on the host, and mount a Hugging Face cache so that the TabICLv2 checkpoint is downloaded only once:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD":/workspace/naturalness \
  -v "$HOME/.cache/huggingface":/tmp/huggingface \
  naturalness:cu124 \
  paper/reproduce_experiment2_twins.sh --gpus 1
```

## Quick start: a toy example

[`examples/toy_naturalness.py`](examples/toy_naturalness.py) simulates ten source sites, each with 2,000 trial rows (`F = "on"`) and 2,000 observational rows (`F = "idle"`), and a target site with 4,000 observational rows. All variables are binary. An unobserved $U$ confounds treatment $X$ and outcome $Y$ in the observational regime. $Z_1$ moves treatment but not the outcome, and $Z_2$ modifies the effect: the true CATE is exactly $0$ when $Z_2 = 1$, and $+0.189$ when $Z_2 = 0$ at the target of the default seed. The script fits Naturalness with the paper's grids, propagates 1,000 draws of the learned tolerances through the target linear program, as the paper does, and compares the result with the truth and two reference bounds. It needs only the package's own dependencies and takes one to two minutes on a laptop CPU, depending on the hardware:

```bash
python examples/toy_naturalness.py
```

```text
Simulated 10 source sites (2000 observational and 2000 trial rows each) and a target site with 4000 observational rows.

Target conditions that fire (I_j <= epsilon and D_j >= tau):
covariate  epsilon   tau   I_j   D_j    2c    2d
       Z1    0.200 0.150 0.197 0.552 0.239 0.074
       Z1    0.200 0.200 0.197 0.552 0.239 0.074
       Z1    0.200 0.100 0.197 0.552 0.239 0.074
       Z1    0.200 0.075 0.197 0.552 0.239 0.074
       Z1    0.200 0.300 0.197 0.552 0.239 0.074
       Z1    0.500 0.150 0.197 0.552 0.241 0.075
       Z1    0.500 0.200 0.197 0.552 0.241 0.075
       Z1    0.500 0.100 0.197 0.552 0.241 0.075
       Z1    0.500 0.075 0.197 0.552 0.241 0.075
       Z1    0.500 0.300 0.197 0.552 0.241 0.075

Target CATE bounds (1000 LP posterior draws):
 Z1  Z2 true CATE      Naturalness P(optimal)           Manski Balke-Pearl (Z1)
  0   0    +0.189 [-0.056, +0.472]       1.00 [-0.272, +0.728] [-0.028, +0.420]
  0   1    +0.000 [-0.290, +0.261]       1.00 [-0.461, +0.539] [-0.217, +0.231]
  1   0    +0.189 [-0.056, +0.472]       1.00 [-0.409, +0.591] [-0.028, +0.420]
  1   1    +0.000 [-0.290, +0.261]       1.00 [-0.460, +0.540] [-0.217, +0.231]

Naturalness bounds 4 of 4 strata and covers the true CATE in 4 of them; mean width 0.539 (Manski 1.000, Balke-Pearl 0.448). Elapsed 24 s.
```

Only conditions on $Z_1$ fire at the target. Naturalness is never told which covariate is instrument-like; it finds $Z_1$ from the data. The intervals contain the true CATE and are about half as wide as the Manski bounds, which use no assumption and always have width 1. The Balke–Pearl bounds are narrower because they are told that $Z_1$ is a valid instrument. That is true in this simulation by construction, but in practice it has to be assumed. Over seeds 0 to 9 of the simulation (`--seed`, 200 draws each), every stratum was bounded and covered, with mean widths between 0.53 and 0.62.

### Step by step

The same pipeline with the package's `Naturalness` class, run from the repository root (or after `pip install -e .`):

```python
from causal_grounding import Naturalness
from examples.toy_naturalness import simulate_toy_data

# Sources: {"site00": rows, ...} with covariates Z1, Z2, treatment X, outcome Y
# and regime F ("on" = randomized trial, "idle" = observational).
# Target: observational rows only. truth: the target's true CATE by stratum.
sources, target, truth = simulate_toy_data(seed=0)

model = Naturalness(covariates=["Z1", "Z2"], n_draws=1000, random_state=0)
model.fit(sources)                 # Stages 1-3 on the sources
bounds = model.get_bounds(target)  # target conditions, then the target LP
print(bounds.merge(truth, on=["Z1", "Z2"]).round(3))
```

```text
   Z1  Z2  lower  upper  p_optimal  true_cate
0   0   0 -0.056  0.472        1.0      0.189
1   0   1 -0.290  0.261        1.0      0.000
2   1   0 -0.056  0.472        1.0      0.189
3   1   1 -0.290  0.261        1.0      0.000
```

`fit` runs Stages 1-3 on the sources with `NaturalnessGrounder`: where $I_j(\epsilon)$ and $D_j(\tau)$ fire, and Gaussian-process surfaces for the learned tolerances $c$ and $d$. `get_bounds` finds the conditions that fire at the target with `TargetNaturalnessPolicy`, propagates tolerance draws through the target linear program, and averages the endpoints over the draws whose program is optimal. `p_optimal` is the share of such draws. `get_bounds(target, return_conditions=True)` also returns the target conditions. `fit` accepts a mapping of site frames, one long frame with `site_col=`, or a loader `SiteBundle`. The default grids are the paper's (`PAPER_EPSILONS`, `PAPER_TAUS`). [`examples/toy_naturalness.py`](examples/toy_naturalness.py) spells out the same steps with the lower-level classes.

scikit-learn may raise a `ConvergenceWarning` while the Gaussian processes are fitted, when a length-scale reaches its bound because a tolerance surface is flat along that axis. It is harmless here, and `fit` silences it.

To use your own data, give every site the same discrete covariate columns, a binary treatment, a binary outcome and a regime column. Source sites need both regimes; the target needs observational rows only. A few points to keep in mind:

- **Probability model.** `stage1_backend="empirical"` uses smoothed cell frequencies. The paper uses `"tabicl_v2"` (TabICLv2) for Stage 1 and for the target probabilities (`Naturalness(..., stage1_backend="tabicl_v2", target_backend="tabicl_v2")`). `"xgboost"` and `"hierarchical_dirichlet"` are also available.
- **When nothing fires.** If no condition fires at the target, `model.policy_result_.status` is `"no_triggered_candidates"` and Naturalness abstains: every endpoint is `NaN` and `p_optimal` is 0. A stratum whose program has no optimal draw also has no Naturalness interval. The paper scores both cases with the Manski interval.
- **Source sites.** In Experiment I, the paper's pipeline (`experiments/exp25_osrct_mc_baseline_comparison.py`, on top of the site split in `experiments/exp16_osrct_calibrated_policy.py`) holds out 9 of the 35 source sites for endpoint calibration and 11 more for policy tuning, and fits its grounder on the remaining 15. The 9 come from Exp25's calibration settings (`calibration_fraction=0.25`, `min_calibration_sites=9`), which replace Exp16's defaults. The example fits on every source.

## Reproducing the paper

Each script in [`paper/`](paper/) rebuilds one part of the paper's results from the public data and writes its tables (`.tex`) and figures (`.pdf`, `.png`). [`paper/README.md`](paper/README.md) documents the outputs, stages, options and costs; `--help` prints a script's full documentation.

| Script | Reproduces | GPU-hours | CPU core-hours |
|---|---|--:|--:|
| [`reproduce_experiment1_manylabs.sh`](paper/reproduce_experiment1_manylabs.sh) | Experiment I (ManyLabs): 5 tables, 2 figures | 81 | 250 |
| [`reproduce_experiment2_twins.sh`](paper/reproduce_experiment2_twins.sh) | Experiment II (Twins): 1 table, 1 figure | 16 | 370 |
| [`reproduce_experiment3_bayesian.sh`](paper/reproduce_experiment3_bayesian.sh) | Experiment III (Bayesian pipeline on ManyLabs): 1 table, 1 figure | 114 | 1,740 |
| [`reproduce_ablations.sh`](paper/reproduce_ablations.sh) | Appendix "Ablations on Naturalness conditions": 1 table, 4 figures | 0.7 | 510 |

The costs were measured on NVIDIA H100 GPUs. The ablations read Experiment I's outputs, so run that script first.

**Data.** Experiments I and III start from the ManyLabs1 file `CleanedDataset.sav`. Download it from <https://osf.io/wx7ck/> into `osrct_benchmark_v1.0/data/` (see [its README](osrct_benchmark_v1.0/data/README.md)) or pass `--manylabs-sav`. Experiment II downloads the three Twins files from the CEVAE repository and checks them against the SHA-256 checksums of the paper's copies. On first use, TabICLv2 downloads its checkpoint to `HF_HOME`, and the comparator stages clone BLearner unless `B_LEARNER_REPOSITORY` names a checkout. The image sets it to `/opt/BLearner`.

**Running.** Run the scripts from the repository root. Every stage resumes, so rerunning a command continues a stopped run. `--dry-run` prints the plan and every command without running anything:

```bash
paper/reproduce_experiment1_manylabs.sh --dry-run
paper/reproduce_experiment1_manylabs.sh --gpus 2 --cpus 32 --memory-gb 128
paper/reproduce_experiment2_twins.sh --gpus 1 --cpus 32 --memory-gb 128
paper/reproduce_experiment3_bayesian.sh --gpus 1 --cpus 64 --memory-gb 256
paper/reproduce_ablations.sh --gpus 1 --cpus 32 --memory-gb 128
```

Outputs go to `results/paper/<experiment>/`: the paper's files in `tables/` and `figures/`, next to the full results tree and logs. `--tree DIR` runs only the report stage on a finished results tree, which takes minutes. Seeds, grids and settings are the paper's; [`paper/README.md`](paper/README.md#reproducibility) lists the sources of run-to-run variation (CB-IPW and CB-DR, the NUTS fits, TabICLv2 on other hardware).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite runs on the CPU. It takes a few minutes on an idle 16-thread machine, and much longer under load. Tests that need data not included here (the ManyLabs1 file, the pre-generated OSRCT benchmark samples) are skipped.

## Not included

Data and results are not included; the scripts download or build them as described above.

## License

MIT; see [`LICENSE`](LICENSE). BLearner, TabICLv2 and the datasets come under their own terms.
