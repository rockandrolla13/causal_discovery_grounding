# Reproducing the paper's results

Each script in this directory regenerates one part of the results of the Naturalness paper from the public data. It rebuilds the experiment's results tree and writes the paper's tables as `.tex` files and its figures as `.pdf` and `.png` files. The scripts run on a single machine, with or without GPUs.

| Script | Reproduces | Experiment in this repository | Paper outputs | GPU-hours | CPU core-hours |
|---|---|---|---|--:|--:|
| [`reproduce_experiment1_manylabs.sh`](reproduce_experiment1_manylabs.sh) | Experiment I (ManyLabs) | Exp25 | 5 tables, 2 figures | 81 | 250 |
| [`reproduce_experiment2_twins.sh`](reproduce_experiment2_twins.sh) | Experiment II (Twins) | Exp28 | 1 table, 1 figure | 16 | 370 |
| [`reproduce_experiment3_bayesian.sh`](reproduce_experiment3_bayesian.sh) | Experiment III (Bayesian pipeline on ManyLabs) | Exp31 | 1 table, 1 figure | 114 | 1,740 |
| [`reproduce_ablations.sh`](reproduce_ablations.sh) | Appendix "Ablations on Naturalness conditions" | Exp25 | 1 table, 4 figures | 0.7 | 510 |

[Compute cost](#compute-cost) explains these numbers. The ablations read the outputs of Experiment I, so run `reproduce_experiment1_manylabs.sh` before `reproduce_ablations.sh`. When Experiment I's output is present, the report of Experiment III also compares the two experiments; its paper table and figure do not need it. Experiment II is independent of the others.

## Setup

1. **Python environment.** Python 3.11 with torch 2.5.1 and the pinned packages of [`requirements.txt`](../requirements.txt), which include `pyreadstat` to read the ManyLabs SPSS file:

   ```bash
   conda create -n paper python=3.11
   conda activate paper
   # GPU machines (CUDA 12.4 build):
   pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
   # CPU-only machines instead:
   # pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
   pip install -r requirements.txt
   ```

   Activate the environment before running a script, or pass `--conda-env paper` (or `--python /path/to/python`). The scripts need bash 4.4 or newer. The image built from [`Dockerfile`](../Dockerfile) holds the same environment; the [main README](../README.md#docker) shows how to run the scripts inside it.

2. **ManyLabs data** (Experiments I and III, and so the ablations). Download `CleanedDataset.sav` from <https://osf.io/wx7ck/> and save it as `osrct_benchmark_v1.0/data/CleanedDataset.sav`, or pass its path with `--manylabs-sav`. The `data` stage converts it to `ManyLabs1/pre-process/Manylabs1_data.pkl`, which later runs reuse.

3. **Network access on first use.** TabICLv2 downloads its checkpoint to `HF_HOME`. Experiments I and II clone BLearner at commit `734fd76` into `<output-dir>/deps/BLearner`, unless the environment variable `B_LEARNER_REPOSITORY` names a checkout. Experiment II downloads the Twins data, unless `--twins-dir` holds it.

## Common options and behaviour

The four scripts share their options and behaviour. Run them from the repository root.

- **Output.** Everything goes under `results/paper/<experiment>/` by default; `--output-dir DIR` changes it. The paper's files are in its `tables/` and `figures/` directories. The rest is the experiment's results tree, including the full report. Each job writes a log under `logs/`.
- **Stages.** A script runs its stages in order, and `--stages a,b` runs a subset. Every stage resumes: rerunning the same command after a stop or a failure keeps the finished work and continues from there.
- **Report only.** `--tree DIR` runs only the report stage on a finished results tree, such as a published one, without refitting anything.
- **Resources.** `--gpus`, `--jobs-per-gpu`, `--cpus` and `--memory-gb` default to what the machine has. They set how many jobs each stage runs in parallel. `--memory-gb` is only used to size the parallelism and is not enforced as a limit. With `--gpus 0`, TabICLv2 runs on the CPU, which is much slower and can change the last digits.
- **Dry run.** `--dry-run` prints the resources found, the number of parallel jobs per stage and every command, and runs nothing.
- **Help.** `--help` prints the script's full documentation, including its reproducibility checks.

## Compute cost

The costs were measured on the runs that produced the paper's results, which used NVIDIA H100 80GB GPUs. GPU-hours are device-hours of the GPU stages. CPU core-hours are core-hours of the CPU-only stages and leave out the CPU cores that GPU jobs hold.

| Script | GPU-hours | CPU core-hours | CPU cores per GPU job | Disk |
|---|--:|--:|---|---|
| Experiment I | 81 | 250 | 16 (4 in `reground`) | about 2 GB |
| Experiment II | 16 | 370 | 4 | under 1 GB |
| Experiment III | 114 | 1,740 | 1 per process, 3 processes per GPU | about 2 GB |
| Ablations | 0.7 | 510 | 4 | under 1 GB |
| **All four** | **212** | **2,870** | | |

Stages run one after another, and the jobs of a stage run in parallel. A stage therefore takes roughly its GPU-hours divided by the number of GPUs, or its core-hours divided by the number of cores, as long as `--memory-gb` allows that many parallel jobs. Each experiment's section below breaks the cost down by stage, and `--dry-run` shows how many jobs each stage will run in parallel on a given machine. On other GPU models, or on the CPU with `--gpus 0`, the GPU stages take a different time.

## Experiment I: ManyLabs

`reproduce_experiment1_manylabs.sh` rebuilds Experiment I from the raw ManyLabs1 data. Its 720 conditions are 20 replications × 3 studies × 3 selection covariates × 4 selection strengths β ∈ {0.5, 1, 1.5, 2}.

**Outputs**, under `results/paper/experiment1_manylabs/`:

| File | Paper label | Content |
|---|---|---|
| `tables/manylabs_width_hit_by_study.tex` | `tab:manylabs-width-hit-by-study` | Mean interval width (hit ratio) by study and β |
| `tables/manylabs_width.tex` | `tab:manylabs-width` | Mean width (hit ratio) by β |
| `tables/manylabs_sign.tex` | `tab:manylabs-sign` | Decisiveness (correct-sign rate) by β |
| `tables/manylabs_width_by_pattern.tex` | `tab:manylabs-width-by-pattern` | Mean width (hit ratio) by selection covariate and β |
| `tables/manylabs_sign_by_study.tex` | `tab:manylabs-sign-by-study` | Decisiveness (correct-sign rate) by study and β |
| `figures/manylabs_width_hit_frontier_by_study.{pdf,png}` | `fig:widths_manylabs` | Width against hit ratio, by study and β |
| `figures/manylabs_sign_identification_frontier.{pdf,png}` | `fig:sign_manylabs` | Sign identification frontier, by β |

The tables are booktabs tabulars to `\input` inside a table float. Their last column, n, counts the strata of each row. Every method is scored on every stratum, and a stratum that a method cannot bound gets its Manski interval. The paper's results tree is `merged_beta_0p5_2_tau_floor_0p075/`. `report/analysis_failures_as_manski/` holds the full analysis behind the tables. `report/analysis_failures_as_manski_draws/` holds a sensitivity variant that also scores the non-optimal posterior draws of Naturalness as Manski.

**Stages**

| Stage | What it does | Runs on | Cost on the paper's runs |
|---|---|---|---|
| `data` | Builds `ManyLabs1/pre-process/Manylabs1_data.pkl` from `CleanedDataset.sav`; skipped if the file exists | CPU | seconds |
| `naturalness` | OSRCT samples and Naturalness fits (TabICLv2 probabilities, Gaussian-process tolerances, a 1000-draw LP posterior), as 40 jobs: one per replication and β group | GPU, 16 CPU threads per job | about 2 h per job; 81 GPU-h |
| `comparators` | CB-IPW, CB-DR and the oracle B-Learner, then Manski and Balke-Pearl, on the saved samples | CPU | 3.5 h per β group on 24 cores; about 170 core-h |
| `merge` | One results tree over the four values of β | CPU | small, not measured |
| `reground` | Naturalness grounded again on the relevance grid τ ≥ 0.075, Manski and Balke-Pearl recomputed at τ = 0.075, and the paper's tree assembled | GPU, then CPU | 20 GPU-min, then 19 h of 4-core jobs (about 76 core-h) |
| `report` | The tables and figures above | CPU | about a minute |

The script budgets 48 GB of memory per Naturalness job, 3 GB per comparator worker, 24 GB per GPU job of `reground` and 8 GB per CPU job of `reground`. These are the requests of the paper's cluster jobs.

**Examples**

```bash
# Full run on 2 GPUs, 32 cores and 128 GB of memory
paper/reproduce_experiment1_manylabs.sh --gpus 2 --cpus 32 --memory-gb 128

# ManyLabs file elsewhere, Python from a conda environment
paper/reproduce_experiment1_manylabs.sh --manylabs-sav ~/data/CleanedDataset.sav --conda-env paper

# The GPU-heavy stage first, the rest later (same --output-dir)
paper/reproduce_experiment1_manylabs.sh --stages data,naturalness
paper/reproduce_experiment1_manylabs.sh --stages comparators,merge,reground,report

# Tables and figures only, from a finished Exp25 tree (its root or its backend leaf)
paper/reproduce_experiment1_manylabs.sh --tree /data/exp25_paper_tree
```

## Experiment II: Twins

`reproduce_experiment2_twins.sh` rebuilds Experiment II from the public Twins benchmark files. Its 720 conditions are 6 target sites × 6 selection strengths β ∈ {0, 0.25, 0.5, 1, 1.5, 2} × 20 splitting seeds (20260918 to 20260937).

**Outputs**, under `results/paper/experiment2_twins/`:

| File | Paper label | Content |
|---|---|---|
| `tables/twins_width_by_beta.tex` | `tab:twins-width-by-beta` | Mean weighted interval width (mean reference hit ratio) by β |
| `figures/twins_width_by_beta.{pdf,png}` | `fig:widths_twins` | Mean weighted width against β, with 90% Student-t bands |

Each covariate cell is weighted by its target mass times the probability that the Naturalness LP draw is optimal. The merged tree, `merged_consistent/`, also holds every other table and figure of the experiment's report in its `tables/` and `figures/` directories.

**Stages**

| Stage | What it does | Runs on | Cost on the paper's runs |
|---|---|---|---|
| `data` | Downloads the three Twins CSVs from the CEVAE repository, or copies them from `--twins-dir`, and checks them against the SHA-256 checksums of the paper's copies | CPU | small, not measured |
| `naturalness` | Naturalness, Manski and Balke-Pearl on the 720 conditions, as 23 jobs | GPU, 4 CPU threads per job | 44 to 86 s per condition; 16 GPU-h |
| `comparators` | Manski, the B-Learner (at Γ = exp(β) and at the oracle Γ), CB-IPW and CB-DR on the same conditions | CPU, 8 cores per job | about 3.8 min per condition; about 370 core-h |
| `merge` | One tree over both passes, keeping the conditions whose discovered candidate's tolerance covers its own measured deviation (404 of 720 in the paper) | CPU | a few minutes |
| `report` | Every table and figure of the merged tree, then the paper's table and figure | CPU | a few minutes |

The script budgets 48 GB of memory per job in both passes.

**Examples**

```bash
# Full run on 1 GPU, 32 cores and 128 GB of memory
paper/reproduce_experiment2_twins.sh --gpus 1 --cpus 32 --memory-gb 128

# Twins files already downloaded (twin_pairs_{X,T,Y}_3years_samesex.csv)
paper/reproduce_experiment2_twins.sh --twins-dir ~/data/twins

# Table and figure only, from a finished merged Exp28 tree (the report runs on a copy)
paper/reproduce_experiment2_twins.sh --tree /data/exp28_paper_tree
```

## Experiment III: Bayesian pipeline on ManyLabs

`reproduce_experiment3_bayesian.sh` reruns the Bayesian pipeline on the paper's 720 ManyLabs conditions, starting from the raw ManyLabs1 data. They are 20 replications (seeds 62 to 81) × 3 studies × 3 selection covariates × 4 values of β.

**Outputs**, under `results/paper/experiment3_bayesian/`:

| File | Paper label | Content |
|---|---|---|
| `tables/bayesian_results.tex` | `tab:bayesian-results` | By study and β: mean target-mass-weighted CATE width of Naturalness, Balke-Pearl and Manski, and the membership and wrong-sign rates of the first two |
| `figures/exp3_tradeoff.{pdf,png}` | `fig:exp3-tradeoff` | Width against membership, with 95% bootstrap intervals over replications |

`report/` holds the full report behind the numbers that the paper's text quotes, such as the leave-one-out check and the classification of the combinations.

**Stages**

| Stage | What it does | Runs on | Cost on the paper's runs |
|---|---|---|---|
| `data` | As in Experiment I | CPU | seconds |
| `prepare` | The OSRCT samples of the 20 replications | CPU | a few minutes |
| `pmfs` | 20 source and 500 target pmf draws per condition, by predictive resampling of TabICLv2 fits | GPU, 3 processes per GPU (`--jobs-per-gpu`), 1 CPU core each | about 25 min per condition and process; 114 GPU-h |
| `naturalness` | The source evidence, the NUTS-fitted tolerance surfaces and the random identified sets | CPU, 2 cores per process (4 NUTS chains) | about 67 min per condition; about 1,600 core-h |
| `regrid` | The same sets on the relevance grid τ ≥ 0.075, recomputed from the fitted surfaces | CPU, 1 core per process | 15 core-h |
| `balkepearl` | The Bayesian Balke-Pearl sets, relevance threshold 0.075, on the same posterior draws | CPU, 1 core per process | 3 core-h |
| `loso` | Leave-one-source-site-out check on replication 0 of the 36 combinations of study, selection covariate and β | CPU, 2 cores per process | 125 core-h |
| `report` | The report, then the paper's table and figure | CPU | a few minutes |

The script budgets 8 GB of memory per process in `pmfs`, `naturalness` and `loso`, and 2 GB in `regrid` and `balkepearl`. The `naturalness` stage dominates the cost, so it benefits most from extra cores. Each stage writes its own directory: `prepared/`, `pmf/`, `naturalness/` (relevance grid from τ = 0.01), `naturalness_tau0p075/` (the paper's grid), `balke_pearl_tau0p075/`, `loso/` and `report/`.

**Examples**

```bash
# Full run on 1 GPU, 64 cores and 256 GB of memory
paper/reproduce_experiment3_bayesian.sh --gpus 1 --cpus 64 --memory-gb 256

# The GPU stage first, then the CPU-only stages (same --output-dir)
paper/reproduce_experiment3_bayesian.sh --stages data,prepare,pmfs
paper/reproduce_experiment3_bayesian.sh --stages naturalness,regrid,balkepearl,loso,report

# Table and figure only, from finished results laid out as in the output directory,
# with Experiment I's tree outside its default location
paper/reproduce_experiment3_bayesian.sh --tree /data/exp31_paper_results --exp1-tree /data/exp25_paper_tree
```

## Ablations on Naturalness conditions

`reproduce_ablations.sh` reruns the paper's ablations on the design of Experiment I. It reads the outputs of `reproduce_experiment1_manylabs.sh`: the OSRCT samples and stored source evidence, the target probability tables, the regrounded Naturalness intervals, and the Manski and Balke-Pearl bounds. Run Experiment I first, at least up to its `reground` stage.

**Outputs**, under `results/paper/ablations/`:

| File | Paper label | Content |
|---|---|---|
| `tables/ablation_family.tex` | `tab:ablation-family` | Family mismatch: mean width (hit ratio) of Naturalness by β, grounded on matched and on mismatched sources |
| `figures/ablation_effect_shift_validity.{pdf,png}` | `fig:effect-shift` | Effect shift: hit ratio and wrong-sign share against the shift Δ |
| `figures/ablation_effect_shift_validity_by_study.{pdf,png}` | `fig:effect-shift-by-study` | Effect shift: hit ratio against Δ, by study |
| `figures/ablation_grounding_frontier_by_study.{pdf,png}` | `fig:ablation-grounding-by-study` | Grounding ablation: width against hit ratio, by study |
| `figures/ablation_grounding_frontier_by_beta.{pdf,png}` | `fig:ablation-grounding-by-beta` | Grounding ablation: width against hit ratio, by β |

Methods are scored as in Experiment I. `arms/summary/` holds the CSV tables behind the numbers that the appendix quotes, such as the realized shifts and the binding tolerances.

**Stages**

| Stage | What it does | Runs on | Cost on the paper's runs |
|---|---|---|---|
| `tables` | The target's probability tables after each effect shift Δ ∈ {0.1, 0.2, 0.3} | GPU, 4 CPU threads per job | 41 GPU-min |
| `shift` | Effect shift: the reference, Naturalness, Manski, Balke-Pearl and the transport baseline on each shifted target | CPU, 4 cores per job | 71 h of jobs, about 284 core-h |
| `family` | Family mismatch: Naturalness grounded on the sources of the condition at the farthest β, or on sources selected on another covariate | CPU, 4 cores per job | 44 h of jobs, about 176 core-h |
| `grounding` | Grounding ablation: the linear program under fixed, exact and prior-only tolerances | CPU, 4 cores per job | 13 h of jobs, about 52 core-h |
| `report` | Scores the arms, then writes the paper's table and figures | CPU | a few minutes |

The script budgets 24 GB of memory per `tables` job and 8 GB per CPU job. Naturalness is grounded as in Experiment I, with a per-site Gaussian process and the relevance grid from τ = 0.075.

**Examples**

```bash
# Full run on 1 GPU, 32 cores and 128 GB of memory, with Experiment I's output
# in its default location
paper/reproduce_ablations.sh --gpus 1 --cpus 32 --memory-gb 128

# Experiment I's output elsewhere
paper/reproduce_ablations.sh --exp1-dir /data/experiment1_manylabs

# Table and figures only, from finished arms (delta=*, d3_beta, d3_covariate, ablation).
# Experiment I's outputs are still read.
paper/reproduce_ablations.sh --tree /data/exp25_mechanism_arms
```

## Reproducibility

The scripts use the paper's seeds, grids, settings and input files. The paper's runs used the Python stack of [`Dockerfile`](../Dockerfile): Python 3.11, torch 2.5.1 with CUDA 12.4, the packages of [`requirements.txt`](../requirements.txt) (including JAX 0.10.2 and NumPyro 0.22.0) and BLearner at commit `734fd76`. Some results can still differ slightly from the paper's:

- CB-IPW and CB-DR (Experiments I and II) are not deterministic, even between two runs on one machine, because their CATENets fits draw from numpy's global random state.
- The NUTS fits of Experiment III are not bit-identical across machines and library builds, so its results agree with the paper's within Monte Carlo error.
- TabICLv2 on another GPU model or on the CPU, and other BLAS builds, can change the last digits. This can move a ManyLabs stratum that sits near an LP feasibility boundary, or a Twins condition across the filter of the `merge` stage.

The REPRODUCIBILITY section of each script's `--help` lists which parts were checked against the paper's runs and how closely they agreed.
