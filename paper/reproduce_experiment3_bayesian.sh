#!/usr/bin/env bash
#
# Reproduce Experiment III (the Bayesian pipeline on ManyLabs) of the
# Naturalness paper on one machine.
#
# Experiment III is Exp31 in this repository. Starting from the raw ManyLabs1
# data, the script reruns the Bayesian pipeline on the paper's 720 conditions
# and writes the experiment's table and figure to OUTPUT_DIR:
#
#   tables/bayesian_results.tex        tab:bayesian-results
#   figures/exp3_tradeoff.{pdf,png}    fig:exp3-tradeoff
#
# The table is a booktabs tabular to \input inside a table float: per study and
# beta, the mean weighted CATE width of Naturalness, Balke-Pearl and Manski, and
# the reference membership and wrong-sign rate of the first two. The figure
# draws the same summaries with 95% bootstrap intervals over replications.
# report/ holds the full report behind the numbers the paper's text quotes,
# such as the leave-one-out check and the classification of the combinations.
#
# STAGES, in order. --stages runs a subset. Every stage resumes, so rerunning
# the same command continues a stopped run.
#   data         ManyLabs1/pre-process/Manylabs1_data.pkl from the OSF file
#                CleanedDataset.sav (https://osf.io/wx7ck/).
#   prepare      The OSRCT samples of the 20 replications (seeds 62-81).
#   pmfs         20 source and 500 target pmf draws per condition, by
#                predictive resampling of TabICLv2 fits. Uses the GPUs.
#   naturalness  The source evidence, the NUTS-fitted tolerance surfaces and the
#                random identified sets of every condition. CPU only.
#   regrid       The same sets on the relevance grid tau >= 0.075. CPU only.
#   balkepearl   The Bayesian Balke-Pearl sets, relevance threshold 0.075, on
#                the same posterior draws. CPU only.
#   loso         The leave-one-source-site-out check on replication 0 of the
#                36 combinations of study, selection covariate and beta. CPU only.
#   report       The report, then the paper's table and figure above.
#
# Two points follow from how the paper's numbers were produced:
#   * One run for the three studies. The paper ran allowedforbidden and
#     gainloss as one run and reciprocity as a second. Every condition draws its
#     random numbers from its own study, selection covariate, beta and seed, so
#     one run gives the same conditions; preparing the three studies together
#     reproduces the paper's 720 OSRCT samples byte for byte.
#   * The regrid stage. The naturalness stage uses the module's relevance grid,
#     which starts at tau = 0.01; the paper reports the grid from tau = 0.075.
#     The grid only decides which candidates are eligible at the target, so the
#     replay recomputes the sets from the fitted surfaces and refits nothing.
#
# RESOURCES. Each defaults to what the machine has.
#   --gpus N          GPUs to use. The pmfs stage runs its processes on them; 0
#                     runs TabICLv2 on the CPU, which is much slower.
#   --jobs-per-gpu N  pmfs processes sharing one GPU (default: 3, as on the
#                     paper's cluster runs: one process holds one CPU core and
#                     uses a fraction of the GPU).
#   --cpus N          CPU cores. A naturalness or loso process runs 4 NUTS
#                     chains on 2 cores; a regrid or balkepearl process uses 1.
#   --memory-gb N     Memory budget in GB. It allows one parallel process per
#                     8 GB in the pmfs, naturalness and loso stages and per 2 GB
#                     in regrid and balkepearl, after the requests of the
#                     paper's cluster jobs. It sizes the parallelism and is not
#                     enforced as a limit.
# Suggested: one or more CUDA GPUs (the paper used NVIDIA H100 80GB), and as
# many cores as possible for the naturalness stage, which dominates the cost.
#
# COST on the paper's cluster runs: pmfs 114 GPU-hours, with two or three
# processes per GPU (about 25 min per condition and process); naturalness about
# 67 min per condition on 2 cores, 1,600 core-hours in all (100 h on 32
# cores); loso 125, regrid 15 and balkepearl 3 core-hours; prepare and report a
# few minutes. The outputs take about 2 GB of disk.
#
# REPRODUCIBILITY. Seeds, grids, settings and input files are the paper's.
# Checked against the paper's runs: the prepare stage reproduces its 720 OSRCT
# samples byte for byte; regrid and balkepearl, replayed on a paper condition,
# reproduce its outputs to within 1e-16; the report, run on the paper's
# results, reproduces tab:bayesian-results exactly. The NUTS fits of the
# naturalness and loso stages are not bit-identical across machines and
# library builds, so their outputs agree with the paper's within Monte Carlo
# error rather than exactly, and TabICLv2 on another GPU model or on the CPU
# can change the last digits of the pmf draws. The paper's runs used the Python
# stack of ./Dockerfile: Python 3.11, torch 2.5.1 with CUDA 12.4, and
# requirements.txt (JAX 0.10.2, NumPyro 0.22.0).
#
# PREREQUISITES
#   * A Python 3.11 environment with torch 2.5.1 (the CUDA 12.4 build on GPU
#     machines) and `pip install -r requirements.txt`, which includes
#     pyreadstat for the data stage. Activate it, or pass --conda-env, or run
#     the script inside the image built by ./Dockerfile.
#   * CleanedDataset.sav from https://osf.io/wx7ck/, unless
#     ManyLabs1/pre-process/Manylabs1_data.pkl exists.
#   * Network access on first use: TabICLv2 downloads its checkpoint to HF_HOME.
#
# USAGE
#   paper/reproduce_experiment3_bayesian.sh [options]
#
#   --output-dir DIR     Results root (default: results/paper/experiment3_bayesian).
#   --stages LIST        Comma-separated stages to run (default: all).
#   --tree DIR           Only run the report stage, on the finished results in
#                        DIR, laid out as in OUTPUT_DIR (naturalness_tau0p075*,
#                        balke_pearl_tau0p075* and loso*; several of a kind are
#                        reported as one run), for example a published copy.
#   --exp1-tree DIR      Experiment I's results tree (root or backend leaf), for
#                        the report's comparison with Experiment I (default:
#                        the output of paper/reproduce_experiment1_manylabs.sh,
#                        when present).
#   --manylabs-sav FILE  ManyLabs1 SPSS file
#                        (default: osrct_benchmark_v1.0/data/CleanedDataset.sav).
#   --conda-env NAME     Run Python through `conda run -n NAME`.
#   --python EXE         Python interpreter otherwise (default: python).
#   --dry-run            Print the plan and every command; run nothing.
#   -h, --help           Show this help.
#   and the RESOURCES options above.
#
# EXAMPLES
#   paper/reproduce_experiment3_bayesian.sh --gpus 1 --cpus 64 --memory-gb 256
#   paper/reproduce_experiment3_bayesian.sh --dry-run
#   paper/reproduce_experiment3_bayesian.sh --tree /data/exp31_paper_results
set -euo pipefail

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "ERROR: bash 4.4 or newer is required" >&2
  exit 1
fi

SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
ORIGINAL_ARGS=("$@")

# The paper's design. Changing any of these changes the results.
MODULE=experiments.exp31_bayesian_manylabs_naturalness
STUDIES=(allowedforbidden gainloss reciprocity)
N_CONDITIONS=720       # 20 replications x 3 studies x 3 selection covariates x 4 betas
N_LOSO_CONDITIONS=36   # replication 0 of each combination
TAU_FLOOR=0.075
PMF_ARGS=(--arm primary --module-a-scheme predictive --source-draws 20 --target-draws 500)
NUTS_ARGS=(
  --random-set-draws 500 --evidence-min-randomized-rows 3
  --num-chains 4 --num-warmup 2000 --num-samples 3000
  --retry-warmup 4000 --retry-num-samples 6000
  --second-retry-warmup 8000 --second-retry-num-samples 24000
)
NATURALNESS_ARGS=(--arm predictive --imputation-count 20 "${NUTS_ARGS[@]}")
LOSO_ARGS=(--arm loso_grid --imputation-count 1 --replications 0 "${NUTS_ARGS[@]}")
# JAX runs on the CPU in every stage; a NUTS process spreads its 4 chains over 4
# host devices, as the paper's jobs did.
JAX_ENV=(JAX_PLATFORM_NAME=cpu)
NUTS_ENV=(JAX_PLATFORM_NAME=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4)
RCT_PICKLE="${PROJECT_ROOT}/ManyLabs1/pre-process/Manylabs1_data.pkl"

# Per-process cores and memory (GB), after the requests of the paper's cluster jobs.
PMF_PROCESS_GB=8     # 16 GB per GPU, shared by two or three processes
NUTS_CORES=2
NUTS_PROCESS_GB=8
REPLAY_PROCESS_GB=2  # regrid and balkepearl, one core each

# Paper file name = file name in the report.
PAPER_TABLES=("bayesian_results.tex=width_membership_tradeoff_table.tex")
PAPER_FIGURES=("exp3_tradeoff=figures/05c_width_membership_tradeoff")
ALL_STAGES=(data prepare pmfs naturalness regrid balkepearl loso report)

# --- Helpers -----------------------------------------------------------------

# Print the header comment of this script as its help text.
usage() {
  awk 'NR > 2 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${SCRIPT_PATH}"
}

# Print an error and stop.
die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Print a warning and continue.
warn() {
  echo "WARNING: $*" >&2
}

# Print a timestamped progress message.
note() {
  echo "[$(date '+%F %T')] $*"
}

# Print PATH made absolute against the directory the script was called from.
absolute() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${PWD}/$1" ;;
  esac
}

# Print the smallest of the integer arguments.
min_of() {
  local smallest="$1" value
  shift
  for value in "$@"; do
    if (( value < smallest )); then smallest="${value}"; fi
  done
  echo "${smallest}"
}

# Print the integer argument, raised to 1 if it is smaller.
at_least_one() {
  if (( $1 >= 1 )); then echo "$1"; else echo 1; fi
}

# Print how many processes of PER_PROCESS_GB fit in the memory budget, at least one.
memory_cap() {
  at_least_one $(( MEMORY_GB / $1 ))
}

# Print the GPU ids this process may use, one per line.
visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES+set}" ]]; then
    # CUDA stops enumerating at the first invalid entry, such as -1.
    awk -F, '{ for (i = 1; i <= NF; i++) { gsub(/[[:space:]]/, "", $i);
               if ($i == "" || $i == "-1") exit; print $i } }' <<< "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null || true
  fi
}

# Succeed when the set of visible GPUs is known, rather than merely undetected.
gpus_known() {
  [[ -n "${CUDA_VISIBLE_DEVICES+set}" ]] || nvidia-smi -L > /dev/null 2>&1
}

# Succeed when STAGE is among the stages to run.
stage_selected() {
  [[ -n "${SELECTED[$1]:-}" ]]
}

# Run a command, or only print it under --dry-run.
run() {
  if (( DRY_RUN )); then
    printf '+'; printf ' %q' "$@"; printf '\n'
    return 0
  fi
  "$@"
}

# Run Python on the repository, appending its output to a log.
#   py LOG THREADS [VAR=VALUE ...] -- PYTHON_ARGS...
# THREADS caps the BLAS and OpenMP threads; the VAR=VALUE pairs are extra
# environment, such as the CUDA_VISIBLE_DEVICES of one process.
py() {
  local log="$1" threads="$2"
  shift 2
  local extra_env=()
  while [[ "$1" != "--" ]]; do
    extra_env+=("$1")
    shift
  done
  shift
  local command=(
    env "PYTHONPATH=${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" MPLBACKEND=Agg
    "OMP_NUM_THREADS=${threads}" "OPENBLAS_NUM_THREADS=${threads}"
    "MKL_NUM_THREADS=${threads}" NUMEXPR_NUM_THREADS=1
    "${extra_env[@]}" "${PY[@]}" "$@"
  )
  if (( DRY_RUN )); then
    run "${command[@]}"
    return 0
  fi
  { printf '[%s] +' "$(date '+%F %T')"; printf ' %q' "${command[@]}"; printf '\n'; } >> "${log}"
  "${command[@]}" >> "${log}" 2>&1
}

# Wait for background processes and stop if any of them failed.
#   wait_for NAME PID...
wait_for() {
  local name="$1" pid failed=0
  shift
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || die "${name}: some processes failed (see ${LOG_DIR}/${name}_shard_*.log)." \
    "Rerun the same command to retry them; finished conditions are kept."
}

# Run one stage of the Exp31 module as parallel shards, each with
# --shard-index K --num-shards SHARDS. All shards start together, because
# shard 0 initializes the output root and the others wait for it.
#   run_shards NAME SHARDS USES_GPU [VAR=VALUE ...] -- MODULE_ARGS...
# A shard that uses the GPU gets GPU K modulo --gpus; the others see none.
run_shards() {
  local name="$1" shards="$2" uses_gpu="$3"
  shift 3
  local stage_env=()
  while [[ "$1" != "--" ]]; do
    stage_env+=("$1")
    shift
  done
  shift
  local pids=() shard devices log
  for (( shard = 0; shard < shards; shard++ )); do
    devices=("${NO_GPU[@]}")
    if (( uses_gpu && GPUS > 0 )); then
      devices=("CUDA_VISIBLE_DEVICES=${GPU_IDS[shard % GPUS]}")
    fi
    log="${LOG_DIR}/${name}_shard_${shard}.log"
    if (( DRY_RUN )); then
      py "${log}" 1 "${devices[@]}" "${stage_env[@]}" -- -m "${MODULE}" "$@" \
        --shard-index "${shard}" --num-shards "${shards}"
    else
      py "${log}" 1 "${devices[@]}" "${stage_env[@]}" -- -m "${MODULE}" "$@" \
        --shard-index "${shard}" --num-shards "${shards}" &
      pids+=("$!")
    fi
  done
  wait_for "${name}" "${pids[@]}"
}

# Stop the background processes and the Python processes they started.
stop_jobs() {
  local pid
  for pid in $(jobs -p); do
    pkill -TERM -P "${pid}" 2> /dev/null || true
    kill -TERM "${pid}" 2> /dev/null || true
  done
}

# Succeed when ROOT holds a completion record that validated every condition.
root_validated() {
  [[ -f "$1/completion.json" ]] && grep -q '"validated": true' "$1/completion.json"
}

# Make sure the sharded run in ROOT is finalized: the last shard to finish
# finalizes it, and this retries once if that validation failed.
#   finalize_root NAME ROOT
finalize_root() {
  local name="$1" root="$2"
  if (( DRY_RUN )); then
    return 0
  fi
  if ! root_validated "${root}"; then
    py "${LOG_DIR}/${name}_finalize.log" 1 "${NO_GPU[@]}" -- -m "${MODULE}" finalize --output-dir "${root}" \
      || die "${name}: finalize failed; see ${LOG_DIR}/${name}_finalize.log"
  fi
  root_validated "${root}" || die "${name}: ${root} did not validate; see ${LOG_DIR}/${name}_*.log"
}

# Fail unless ROOT is a validated run, the input of stage NAME.
#   require_validated NAME ROOT STAGE
require_validated() {
  (( DRY_RUN )) || root_validated "$2" || die "$1: ${2} is unfinished; run the $3 stage."
}

# Print the number of conditions with complete pmf draws.
pmf_conditions() {
  (find "${PMF_ROOT}/conditions" -name completion.json 2> /dev/null || true) | wc -l
}

# Print the Exp25 backend leaf of an Experiment I tree given as its root or leaf.
exp1_leaf_of() {
  local tree="$1"
  if [[ -d "${tree}/baseline_variants" ]]; then
    printf '%s\n' "${tree}"
  elif [[ -d "${tree}/tabicl_v2/tabicl_v2/tabicl_v2/baseline_variants" ]]; then
    printf '%s\n' "${tree}/tabicl_v2/tabicl_v2/tabicl_v2"
  fi
}

# Check that the Python environment imports what the selected stages need.
preflight() {
  local imports=() log="${LOG_DIR}/preflight.log" stage
  if stage_selected data && [[ ! -f "${RCT_PICKLE}" ]]; then
    imports+=(pyreadstat)
  fi
  for stage in prepare pmfs naturalness regrid balkepearl loso report; do
    if stage_selected "${stage}"; then
      imports+=("${MODULE}" jax numpyro)
      break
    fi
  done
  if (( ${#imports[@]} > 0 )); then
    py "${log}" 1 "${NO_GPU[@]}" -- -c "import $(IFS=,; echo "${imports[*]}")" \
      || die "The Python environment cannot import what these stages need; see ${log}"
  fi
  if (( GPUS > 0 )) && stage_selected pmfs; then
    py "${log}" 1 "CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")" -- -c \
      "import sys, torch; n = torch.cuda.device_count(); sys.exit(0 if n >= ${GPUS} else f'PyTorch sees {n} CUDA GPU(s), not ${GPUS}')" \
      || die "PyTorch cannot use the ${GPUS} GPU(s); see ${log}, or pass --gpus 0 to run on the CPU"
  fi
}

# Append the command line, revision and resources of this run to the log.
record_invocation() {
  local revision
  revision="$(git -C "${PROJECT_ROOT}" describe --always --dirty 2> /dev/null || echo unknown)"
  {
    printf '%s %s' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${SCRIPT_PATH}"
    if (( ${#ORIGINAL_ARGS[@]} > 0 )); then printf ' %q' "${ORIGINAL_ARGS[@]}"; fi
    printf '\n  revision %s; stages %s; gpus %s [%s]; cpus %s; memory_gb %s\n' \
      "${revision}" "${RUN_STAGES[*]}" "${GPUS}" "$(IFS=,; echo "${GPU_IDS[*]}")" "${CPUS}" "${MEMORY_GB}"
  } >> "${LOG_DIR}/invocations.log"
}

# --- Stages ------------------------------------------------------------------

# data: build the ManyLabs1 pickle every later stage reads.
stage_data() {
  if [[ -f "${RCT_PICKLE}" ]]; then
    note "data: ${RCT_PICKLE} exists; skipping"
    return 0
  fi
  [[ -f "${MANYLABS_SAV}" ]] || (( DRY_RUN )) \
    || die "No ${MANYLABS_SAV}. Download CleanedDataset.sav from https://osf.io/wx7ck/ and pass --manylabs-sav."
  note "data: building ${RCT_PICKLE} from ${MANYLABS_SAV}"
  py "${LOG_DIR}/data.log" 1 "${NO_GPU[@]}" -- ManyLabs1/pre-process/ml1_preprocess.py \
    --input "${MANYLABS_SAV}" --output "$(dirname "${RCT_PICKLE}")" --quiet \
    || die "data: preprocessing failed; see ${LOG_DIR}/data.log"
}

# prepare: the OSRCT samples of every replication, in one prepared root.
stage_prepare() {
  if [[ -f "${PREPARED_ROOT}/identity.json" ]]; then
    note "prepare: ${PREPARED_ROOT} exists; skipping"
    return 0
  fi
  [[ ! -e "${PREPARED_ROOT}" ]] \
    || die "${PREPARED_ROOT} exists without identity.json, probably from an interrupted prepare; remove it and rerun."
  [[ -f "${RCT_PICKLE}" ]] || (( DRY_RUN )) || die "prepare: no ${RCT_PICKLE}; run the data stage."
  note "prepare: ${PREPARED_ROOT}"
  py "${LOG_DIR}/prepare.log" 1 "${NO_GPU[@]}" -- -m "${MODULE}" prepare \
    --output-dir "${PREPARED_ROOT}" --studies "${STUDIES[@]}" \
    || die "prepare failed; see ${LOG_DIR}/prepare.log"
}

# pmfs: the source and target pmf draws of every condition.
stage_pmfs() {
  if (( ! DRY_RUN )) && (( $(pmf_conditions) == N_CONDITIONS )); then
    note "pmfs: all ${N_CONDITIONS} conditions have their draws; skipping"
    return 0
  fi
  [[ -f "${PREPARED_ROOT}/identity.json" ]] || (( DRY_RUN )) || die "pmfs: run the prepare stage first."
  note "pmfs: ${N_CONDITIONS} conditions in ${PMF_JOBS} process(es); logs ${LOG_DIR}/pmfs_shard_*.log"
  run_shards pmfs "${PMF_JOBS}" 1 "${JAX_ENV[@]}" -- draw-pmfs "${PMF_ARGS[@]}" \
    --prepared-root "${PREPARED_ROOT}" --output-dir "${PMF_ROOT}" --device "${TABICL_DEVICE}" \
    --studies "${STUDIES[@]}" --resume
  if (( ! DRY_RUN )); then
    (( $(pmf_conditions) == N_CONDITIONS )) \
      || die "pmfs: $(pmf_conditions)/${N_CONDITIONS} conditions drawn; rerun to retry the others."
  fi
}

# naturalness: evidence, tolerance surfaces and random identified sets.
stage_naturalness() {
  if root_validated "${NATURALNESS_ROOT}"; then
    note "naturalness: ${NATURALNESS_ROOT} is finished; skipping"
    return 0
  fi
  if (( ! DRY_RUN )) && (( $(pmf_conditions) != N_CONDITIONS )); then
    die "naturalness: the pmf draws are unfinished; run the pmfs stage."
  fi
  note "naturalness: ${N_CONDITIONS} conditions in ${NUTS_JOBS} process(es); logs ${LOG_DIR}/naturalness_shard_*.log"
  run_shards naturalness "${NUTS_JOBS}" 0 "${NUTS_ENV[@]}" -- run "${NATURALNESS_ARGS[@]}" \
    --prepared-root "${PREPARED_ROOT}" --pmf-root "${PMF_ROOT}" --output-dir "${NATURALNESS_ROOT}" \
    --studies "${STUDIES[@]}" --resume
  finalize_root naturalness "${NATURALNESS_ROOT}"
}

# regrid: the sets on the relevance grid tau >= 0.075, from the fitted surfaces.
stage_regrid() {
  if root_validated "${REGRID_ROOT}"; then
    note "regrid: ${REGRID_ROOT} is finished; skipping"
    return 0
  fi
  require_validated regrid "${NATURALNESS_ROOT}" naturalness
  note "regrid: tau >= ${TAU_FLOOR} in ${REPLAY_JOBS} process(es)"
  run_shards regrid "${REPLAY_JOBS}" 0 -- regrid --source-root "${NATURALNESS_ROOT}" \
    --prepared-root "${PREPARED_ROOT}" --pmf-root "${PMF_ROOT}" --output-dir "${REGRID_ROOT}" \
    --tau-floor "${TAU_FLOOR}" --resume
  finalize_root regrid "${REGRID_ROOT}"
}

# balkepearl: the Bayesian Balke-Pearl sets on the regridded run's draws.
stage_balkepearl() {
  if root_validated "${BP_ROOT}"; then
    note "balkepearl: ${BP_ROOT} is finished; skipping"
    return 0
  fi
  require_validated balkepearl "${REGRID_ROOT}" regrid
  note "balkepearl: relevance ${TAU_FLOOR} in ${REPLAY_JOBS} process(es)"
  run_shards balkepearl "${REPLAY_JOBS}" 0 -- balke-pearl --source-root "${REGRID_ROOT}" \
    --prepared-root "${PREPARED_ROOT}" --pmf-root "${PMF_ROOT}" --output-dir "${BP_ROOT}" \
    --relevance-tau "${TAU_FLOOR}" --resume
  finalize_root balkepearl "${BP_ROOT}"
}

# loso: the leave-one-source-site-out check on replication 0.
stage_loso() {
  if root_validated "${LOSO_ROOT}"; then
    note "loso: ${LOSO_ROOT} is finished; skipping"
    return 0
  fi
  if (( ! DRY_RUN )) && (( $(pmf_conditions) != N_CONDITIONS )); then
    die "loso: the pmf draws are unfinished; run the pmfs stage."
  fi
  note "loso: ${N_LOSO_CONDITIONS} conditions in ${LOSO_JOBS} process(es)"
  run_shards loso "${LOSO_JOBS}" 0 "${NUTS_ENV[@]}" -- run "${LOSO_ARGS[@]}" \
    --prepared-root "${PREPARED_ROOT}" --pmf-root "${PMF_ROOT}" --output-dir "${LOSO_ROOT}" \
    --studies "${STUDIES[@]}" --resume
  finalize_root loso "${LOSO_ROOT}"
}

# report: the report over the finished roots, then the paper's outputs.
stage_report() {
  local source="${OUTPUT_DIR}" primary=() balke_pearl=() loso=() loso_args=() root exp1_args=() leaf
  [[ -z "${TREE}" ]] || source="${TREE}"
  for root in "${source}"/naturalness_tau0p075*/; do
    if [[ -d "${root}" ]]; then primary+=("${root%/}"); fi
  done
  for root in "${source}"/balke_pearl_tau0p075*/; do
    if [[ -d "${root}" ]]; then balke_pearl+=("${root%/}"); fi
  done
  for root in "${source}"/loso*/; do
    if [[ -d "${root}" ]]; then loso+=("${root%/}"); fi
  done
  if (( ! DRY_RUN )); then
    (( ${#primary[@]} > 0 && ${#balke_pearl[@]} == ${#primary[@]} )) \
      || die "report: ${source} needs naturalness_tau0p075* roots and one balke_pearl_tau0p075* root for each."
    for root in "${primary[@]}" "${balke_pearl[@]}" "${loso[@]}"; do
      root_validated "${root}" || die "report: ${root} is unfinished."
    done
    (( ${#loso[@]} > 0 )) || warn "report: no loso* root in ${source}; the report omits the leave-one-out check."
  else
    (( ${#primary[@]} > 0 )) || primary=("${REGRID_ROOT}")
    (( ${#balke_pearl[@]} > 0 )) || balke_pearl=("${BP_ROOT}")
    (( ${#loso[@]} > 0 )) || loso=("${LOSO_ROOT}")
  fi
  leaf="$(exp1_leaf_of "${EXP1_TREE}")"
  if [[ -n "${leaf}" ]]; then
    exp1_args=(--exp25-intervals "${leaf}/baseline_variants/gamma_dgp_oracle/mc_intervals.parquet"
               --exp25-bp-intervals "${leaf}/baseline_variants/identification_bounds/mc_intervals.parquet")
  else
    note "report: no Experiment I tree at ${EXP1_TREE}; the report omits its comparison with Experiment I"
  fi
  if (( ${#loso[@]} > 0 )); then
    loso_args=(--loso-root "${loso[@]}")
  fi

  if [[ -f "${REPORT_DIR}/report_summary.json" ]]; then
    note "report: ${REPORT_DIR} exists; skipping (remove it to rebuild the report)"
  else
    [[ ! -e "${REPORT_DIR}" ]] \
      || die "${REPORT_DIR} exists without report_summary.json, probably from an interrupted report; remove it and rerun."
    note "report: ${REPORT_DIR}"
    py "${LOG_DIR}/report.log" 1 "${NO_GPU[@]}" -- -m "${MODULE}" report --primary-root "${primary[@]}" \
      --balke-pearl-root "${balke_pearl[@]}" "${loso_args[@]}" "${exp1_args[@]}" \
      --output-dir "${REPORT_DIR}" || die "report failed; see ${LOG_DIR}/report.log"
  fi

  local entry extension
  run mkdir -p "${OUTPUT_DIR}/tables" "${OUTPUT_DIR}/figures"
  for entry in "${PAPER_TABLES[@]}"; do
    run cp "${REPORT_DIR}/${entry#*=}" "${OUTPUT_DIR}/tables/${entry%%=*}"
  done
  for entry in "${PAPER_FIGURES[@]}"; do
    for extension in pdf png; do
      run cp "${REPORT_DIR}/${entry#*=}.${extension}" "${OUTPUT_DIR}/figures/${entry%%=*}.${extension}"
    done
  done
  note "report: tables in ${OUTPUT_DIR}/tables, figures in ${OUTPUT_DIR}/figures"
}

# --- Command line --------------------------------------------------------------

OUTPUT_DIR="${PROJECT_ROOT}/results/paper/experiment3_bayesian"
STAGES=""
TREE=""
EXP1_TREE="${PROJECT_ROOT}/results/paper/experiment1_manylabs/merged_beta_0p5_2_tau_floor_0p075"
MANYLABS_SAV="${PROJECT_ROOT}/osrct_benchmark_v1.0/data/CleanedDataset.sav"
CONDA_ENV=""
PYTHON_EXE="python"
GPUS=""
JOBS_PER_GPU=3
CPUS=""
MEMORY_GB=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --output-dir|--stages|--tree|--exp1-tree|--manylabs-sav|--conda-env|--python|--gpus|--jobs-per-gpu|--cpus|--memory-gb)
      [[ $# -ge 2 ]] || die "$1 needs a value (see --help)"
      case "$1" in
        --output-dir) OUTPUT_DIR="$2" ;;
        --stages) STAGES="$2" ;;
        --tree) TREE="$2" ;;
        --exp1-tree) EXP1_TREE="$2" ;;
        --manylabs-sav) MANYLABS_SAV="$2" ;;
        --conda-env) CONDA_ENV="$2" ;;
        --python) PYTHON_EXE="$2" ;;
        --gpus) GPUS="$2" ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2" ;;
        --cpus) CPUS="$2" ;;
        --memory-gb) MEMORY_GB="$2" ;;
      esac
      shift 2 ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
done

# Stages, in pipeline order whatever order --stages lists them in.
if [[ -n "${TREE}" ]]; then
  [[ -z "${STAGES}" || "${STAGES}" == report ]] || die "--tree only applies to the report stage"
  STAGES=report
fi
[[ -n "${STAGES}" && "${STAGES}" != all ]] || STAGES="$(IFS=,; echo "${ALL_STAGES[*]}")"
declare -A SELECTED=()
IFS=',' read -r -a requested_stages <<< "${STAGES// /}"
for stage in "${requested_stages[@]}"; do
  [[ " ${ALL_STAGES[*]} " == *" ${stage} "* ]] || die "Unknown stage '${stage}'; the stages are ${ALL_STAGES[*]}"
  SELECTED["${stage}"]=1
done
RUN_STAGES=()
for stage in "${ALL_STAGES[@]}"; do
  if stage_selected "${stage}"; then RUN_STAGES+=("${stage}"); fi
done

# Resources: what the machine has, unless given.
[[ -n "${CPUS}" ]] || CPUS="$(nproc 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 8)"
if [[ -z "${MEMORY_GB}" ]]; then
  if [[ -r /proc/meminfo ]]; then
    MEMORY_GB="$(awk '/^MemTotal:/ { print int($2 / 1048576) }' /proc/meminfo)"
  else
    MEMORY_GB="$(( $(sysctl -n hw.memsize 2> /dev/null || echo 34359738368) / 1073741824 ))"
  fi
fi
mapfile -t VISIBLE_GPUS < <(visible_gpus)
[[ -n "${GPUS}" ]] || GPUS="${#VISIBLE_GPUS[@]}"
for value in "${GPUS}" "${JOBS_PER_GPU}" "${CPUS}" "${MEMORY_GB}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || die "Resource values must be whole numbers, not '${value}'"
done
(( JOBS_PER_GPU >= 1 && CPUS >= 1 && MEMORY_GB >= 1 )) \
  || die "--jobs-per-gpu, --cpus and --memory-gb must be at least 1"
if (( GPUS > ${#VISIBLE_GPUS[@]} )); then
  if gpus_known && (( ! DRY_RUN )); then
    die "--gpus ${GPUS}, but only ${#VISIBLE_GPUS[@]} GPU(s) are visible"
  fi
  # Undetected (no nvidia-smi) or a dry run: number the GPUs from 0.
  VISIBLE_GPUS=()
  for (( i = 0; i < GPUS; i++ )); do VISIBLE_GPUS+=("${i}"); done
fi
GPU_IDS=("${VISIBLE_GPUS[@]:0:${GPUS}}")
NO_GPU=("CUDA_VISIBLE_DEVICES=")
if (( GPUS > 0 )); then TABICL_DEVICE=cuda; else TABICL_DEVICE=cpu; fi

# Parallel processes per stage, within the CPU, GPU and memory budgets.
if (( GPUS > 0 )); then
  PMF_JOBS=$(( GPUS * JOBS_PER_GPU ))
else
  PMF_JOBS="${CPUS}"
fi
PMF_JOBS="$(min_of "${PMF_JOBS}" "$(memory_cap "${PMF_PROCESS_GB}")" "${N_CONDITIONS}")"
NUTS_JOBS="$(min_of "$(at_least_one $(( CPUS / NUTS_CORES )))" "$(memory_cap "${NUTS_PROCESS_GB}")" "${N_CONDITIONS}")"
LOSO_JOBS="$(min_of "${NUTS_JOBS}" "${N_LOSO_CONDITIONS}")"
REPLAY_JOBS="$(min_of "${CPUS}" "$(memory_cap "${REPLAY_PROCESS_GB}")" "${N_CONDITIONS}")"

if [[ -n "${CONDA_ENV}" ]]; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PY=("${PYTHON_EXE}")
fi

# Paths. Python runs from the repository root, so every path is absolute.
OUTPUT_DIR="$(absolute "${OUTPUT_DIR}")"
MANYLABS_SAV="$(absolute "${MANYLABS_SAV}")"
EXP1_TREE="$(absolute "${EXP1_TREE}")"
[[ -z "${TREE}" ]] || TREE="$(absolute "${TREE}")"
[[ -z "${TREE}" || -d "${TREE}" ]] || die "--tree ${TREE} is not a directory"
PREPARED_ROOT="${OUTPUT_DIR}/prepared"
PMF_ROOT="${OUTPUT_DIR}/pmf"
NATURALNESS_ROOT="${OUTPUT_DIR}/naturalness"
REGRID_ROOT="${OUTPUT_DIR}/naturalness_tau0p075"
BP_ROOT="${OUTPUT_DIR}/balke_pearl_tau0p075"
LOSO_ROOT="${OUTPUT_DIR}/loso"
REPORT_DIR="${OUTPUT_DIR}/report"
LOG_DIR="${OUTPUT_DIR}/logs"
cd "${PROJECT_ROOT}"

# --- Run -----------------------------------------------------------------------

if (( GPUS > 0 )); then
  gpu_text="${GPUS} (${GPU_IDS[*]}), TabICLv2 on cuda"
else
  gpu_text="none, TabICLv2 on the CPU"
fi
cat << EOF
Experiment III (Bayesian pipeline on ManyLabs, Exp31) reproduction
  output     ${OUTPUT_DIR}
  stages     ${RUN_STAGES[*]}
  python     ${PY[*]}
  GPUs       ${gpu_text}
  CPUs       ${CPUS}
  memory     ${MEMORY_GB} GB
  parallel   pmfs ${PMF_JOBS} process(es), naturalness ${NUTS_JOBS}, loso ${LOSO_JOBS},
             regrid and balkepearl ${REPLAY_JOBS}
EOF
if (( DRY_RUN )); then
  echo "  (dry run: the commands below are printed, not run)"
fi
if stage_selected pmfs; then
  if (( GPUS == 0 )); then
    warn "No GPU in use: TabICLv2 runs on the CPU, which is much slower and can change the last digits." \
      "Pass --gpus N if this machine has GPUs that were not detected."
  elif (( PMF_JOBS < GPUS * JOBS_PER_GPU && PMF_JOBS < N_CONDITIONS )); then
    warn "The ${MEMORY_GB} GB budget allows ${PMF_JOBS} pmfs process(es) at ${PMF_PROCESS_GB} GB each," \
      "fewer than the GPUs could run; raise --memory-gb if the machine has more."
  fi
fi

if (( ! DRY_RUN )); then
  mkdir -p "${LOG_DIR}"
  record_invocation
  trap 'stop_jobs; exit 130' INT
  trap 'stop_jobs; exit 143' TERM
  preflight
fi

for stage in "${RUN_STAGES[@]}"; do
  note "=== ${stage}"
  "stage_${stage}"
done
note "finished: ${RUN_STAGES[*]}"
