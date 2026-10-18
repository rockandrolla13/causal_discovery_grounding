#!/usr/bin/env bash
#
# Reproduce Experiment I (ManyLabs) of the Naturalness paper on one machine.
#
# Experiment I is Exp25 in this repository. Starting from the raw ManyLabs1
# data, the script rebuilds the results tree that the paper reports and writes
# the experiment's tables and figures to OUTPUT_DIR:
#
#   tables/manylabs_width_hit_by_study.tex    tab:manylabs-width-hit-by-study
#   tables/manylabs_width.tex                 tab:manylabs-width
#   tables/manylabs_sign.tex                  tab:manylabs-sign
#   tables/manylabs_width_by_pattern.tex      tab:manylabs-width-by-pattern
#   tables/manylabs_sign_by_study.tex         tab:manylabs-sign-by-study
#   figures/manylabs_width_hit_frontier_by_study.{pdf,png}   fig:widths_manylabs
#   figures/manylabs_sign_identification_frontier.{pdf,png}  fig:sign_manylabs
#
# The tables are booktabs tabulars to \input inside a table float; their last
# column, $n$, counts the strata of each row. Every method is scored on every
# stratum, and a stratum that a method cannot bound gets its Manski interval.
# report/analysis_failures_as_manski_draws/ holds the sensitivity variant in
# which Naturalness's non-optimal posterior draws are scored as Manski too.
#
# STAGES, in order. --stages runs a subset. Every stage resumes, so rerunning
# the same command continues a stopped run.
#   data         ManyLabs1/pre-process/Manylabs1_data.pkl from the OSF file
#                CleanedDataset.sav (https://osf.io/wx7ck/).
#   naturalness  For each replication (seeds 202608-202627), the OSRCT samples
#                and the Naturalness fit: TabICLv2 probabilities, Gaussian
#                process tolerances and a 1000-draw LP posterior. Uses the GPUs.
#   comparators  CB-IPW, CB-DR and the oracle B-Learner, then Manski and
#                Balke-Pearl, on the saved samples. CPU only.
#   merge        One tree over beta in {0.5, 1, 1.5, 2}.
#   reground     Naturalness grounded again on the relevance grid tau >= 0.075,
#                Manski and Balke-Pearl recomputed at tau = 0.075, and the
#                paper's tree assembled. A few GPU minutes, then CPU.
#   report       The tables and figures above.
#
# Two steps follow from how the paper's numbers were produced:
#   * Two beta groups. Exp16 seeds each condition by its position in the run's
#     study x selection covariate x beta grid, and the paper fitted beta
#     {0.5, 1} and beta {1.5, 2} as separate runs. One run over the four betas
#     would reseed most conditions.
#   * The reground stage. Exp25 fits Exp16's relevance grid, which starts at
#     tau = 0.01. The paper reports the grid from tau = 0.075, which no longer
#     selects candidates on sampling noise. Discovery (Stages 1-2) is kept and
#     only the grounding is redone (experiments/exp25_tau_floor_regrounding.py).
#     Balke-Pearl's relevance threshold is the smallest tau of the grid.
#
# RESOURCES. Each defaults to what the machine has.
#   --gpus N          GPUs to use. Each Naturalness job holds one GPU; 0 runs
#                     TabICLv2 on the CPU, which is much slower.
#   --jobs-per-gpu N  Naturalness jobs sharing one GPU (default: 1).
#   --cpus N          CPU cores. A Naturalness job runs 16 LP threads, the
#                     comparators one worker per core, and the reground stage
#                     one 4-thread job per 4 cores.
#   --memory-gb N     Memory budget in GB. It allows one parallel job per 48 GB
#                     for Naturalness, per 24 GB for the target tables and per
#                     8 GB for regrounding, and one comparator worker per 3 GB,
#                     the requests of the paper's cluster jobs. It sizes the
#                     parallelism and is not enforced as a limit.
# Suggested: one or more CUDA GPUs (the paper used NVIDIA H100 80GB), with 16
# CPU cores and 48 GB of RAM per GPU.
#
# COST on the paper's cluster runs (H100 GPUs, 16 threads per job):
# naturalness 2 h per replication and beta group, 80 GPU-hours in all (about
# 20 h with 4 GPUs and 64 cores); comparators 3.5 h per beta group on 24
# cores; reground 20 GPU-minutes and 19 h of 4-thread jobs (2.5 h on 32
# cores); report about a minute. The outputs take about 2 GB of disk.
#
# REPRODUCIBILITY. Seeds, grids and settings are the paper's. Other hardware
# or library versions can change the last digits (TabICLv2 on another GPU
# model or on the CPU, BLAS builds) and so move strata that sit near an LP
# feasibility boundary. CB-IPW and CB-DR are not deterministic, even between
# two runs on one machine: their CATENets fits draw from numpy's global random
# state (on a Twins condition of Experiment II, which fits them with the same
# code, two identical runs gave cell endpoints up to 0.09 apart), so their
# entries can differ slightly from the paper's. The paper's runs used the
# Python stack of ./Dockerfile: Python 3.11, torch 2.5.1 with CUDA 12.4,
# requirements.txt and BLearner at commit 734fd76.
#
# PREREQUISITES
#   * A Python 3.11 environment with torch 2.5.1 (the CUDA 12.4 build on GPU
#     machines) and `pip install -r requirements.txt`, which includes
#     pyreadstat for the data stage. Activate it, or pass --conda-env, or run
#     the script inside the image built by ./Dockerfile.
#   * CleanedDataset.sav from https://osf.io/wx7ck/, unless
#     ManyLabs1/pre-process/Manylabs1_data.pkl exists.
#   * Network access on first use: TabICLv2 downloads its checkpoint to
#     HF_HOME, and the comparators stage clones BLearner at the pinned commit
#     unless B_LEARNER_REPOSITORY names a checkout.
#
# USAGE
#   paper/reproduce_experiment1_manylabs.sh [options]
#
#   --output-dir DIR     Results root (default: results/paper/experiment1_manylabs).
#   --stages LIST        Comma-separated stages to run (default: all).
#   --tree DIR           Only run the report stage, on this finished Exp25 tree
#                        (root or backend leaf), for example a published one.
#   --manylabs-sav FILE  ManyLabs1 SPSS file
#                        (default: osrct_benchmark_v1.0/data/CleanedDataset.sav).
#   --conda-env NAME     Run Python through `conda run -n NAME`.
#   --python EXE         Python interpreter otherwise (default: python).
#   --dry-run            Print the plan and every command; run nothing.
#   -h, --help           Show this help.
#   and the RESOURCES options above.
#
# EXAMPLES
#   paper/reproduce_experiment1_manylabs.sh --gpus 2 --cpus 32 --memory-gb 128
#   paper/reproduce_experiment1_manylabs.sh --dry-run
#   paper/reproduce_experiment1_manylabs.sh --tree /data/exp25_paper_tree
set -euo pipefail

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "ERROR: bash 4.4 or newer is required" >&2
  exit 1
fi

SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
ORIGINAL_ARGS=("$@")

# The paper's design. Changing any of these changes the results.
N_REPLICATIONS=20
BETA_GROUPS=("0.5 1.0" "1.5 2.0")
GROUP_NAMES=("beta_0p5_1" "beta_1p5_2")
PAPER_BETAS=(0.5 1.0 1.5 2.0)
TAU_FLOOR=0.075
N_CONDITIONS=720  # 20 replications x 3 studies x 3 selection covariates x 4 betas
# Exp25 records its LP-posterior thread count in the run fingerprint, but the
# results do not depend on it (every draw is seeded before the draws are
# solved). It keeps the paper's value under any --cpus, so a run stays
# resumable.
LP_THREADS=16
EXP25_ARGS=(
  --n-replications "${N_REPLICATIONS}" --base-seed 202608
  --studies allowedforbidden gainloss reciprocity
  --bias-covariates age gender polideo
  --target-site pi --require-observed-gender
  --stage1-backend tabicl_v2
  --target-trigger-probability-backend tabicl_v2
  --lp-probability-backend tabicl_v2
  --lp-posterior-samples 1000 --lp-posterior-n-jobs "${LP_THREADS}"
  --resume --no-plots
)
BLEARNER_URL="https://github.com/CausalML/BLearner.git"
BLEARNER_COMMIT="734fd7639da54554e41d944bfc2187a24e75cf81"
BACKEND_LEAF="tabicl_v2/tabicl_v2/tabicl_v2"
RCT_PICKLE="${PROJECT_ROOT}/ManyLabs1/pre-process/Manylabs1_data.pkl"

# Memory (GB) and threads per job, as the paper's cluster jobs requested them.
NATURALNESS_JOB_GB=48
COMPARATOR_WORKER_GB=3
TABLES_JOB_GB=24
REGROUND_JOB_GB=8
REGROUND_THREADS=4
BOUNDS_THREADS=2

# Paper file name = file name in the report stage's stratum-scored analysis.
PAPER_TABLES=(
  "manylabs_width_hit_by_study.tex=exp25_width_table_by_study_beta.tex"
  "manylabs_width.tex=exp25_width_table_by_beta.tex"
  "manylabs_sign.tex=exp25_sign_table_by_beta.tex"
  "manylabs_width_by_pattern.tex=exp25_width_table_by_selection_beta.tex"
  "manylabs_sign_by_study.tex=exp25_sign_table_by_study_beta.tex"
)
PAPER_FIGURES=(
  "manylabs_width_hit_frontier_by_study=width_hit_frontier_by_study"
  "manylabs_sign_identification_frontier=sign_identification_frontier"
)
ALL_STAGES=(data naturalness comparators merge reground report)

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

# Print how many jobs of PER_JOB_GB fit in the memory budget, at least one.
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

# Print the Exp25 backend leaf of a results tree given as its root or leaf.
backend_leaf_of() {
  local tree
  tree="$(absolute "$1")"
  if [[ -d "${tree}/baseline_variants" ]]; then
    printf '%s\n' "${tree}"
  elif [[ -d "${tree}/${BACKEND_LEAF}/baseline_variants" ]]; then
    printf '%s\n' "${tree}/${BACKEND_LEAF}"
  else
    die "${tree} is not an Exp25 results tree (no baseline_variants/)"
  fi
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
# environment, such as the CUDA_VISIBLE_DEVICES of one job.
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

# Wait for background jobs and stop if any of them failed.
#   wait_for NAME PID...
wait_for() {
  local name="$1" pid failed=0
  shift
  for pid in "$@"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || die "${name}: some jobs failed (see the warnings above and ${LOG_DIR})." \
    "Rerun the same command to retry them; finished work is kept."
}

# Run a Python pass as parallel shards, each with --shard K --n-shards JOBS.
#   run_shards NAME JOBS THREADS USES_GPU -- PYTHON_ARGS...
# A shard that uses the GPU gets GPU K modulo --gpus; the others see none.
run_shards() {
  local name="$1" jobs="$2" threads="$3" uses_gpu="$4"
  shift 5
  local pids=() shard devices log
  for (( shard = 0; shard < jobs; shard++ )); do
    devices=("${NO_GPU[@]}")
    if (( uses_gpu && GPUS > 0 )); then
      devices=("CUDA_VISIBLE_DEVICES=${GPU_IDS[shard % GPUS]}")
    fi
    log="${LOG_DIR}/${name}_shard_${shard}.log"
    if (( DRY_RUN )); then
      py "${log}" "${threads}" "${devices[@]}" -- "$@" --shard "${shard}" --n-shards "${jobs}"
    else
      py "${log}" "${threads}" "${devices[@]}" -- "$@" --shard "${shard}" --n-shards "${jobs}" &
      pids+=("$!")
    fi
  done
  wait_for "${name}" "${pids[@]}"
}

# Stop the background jobs and the Python processes they started.
stop_jobs() {
  local pid
  for pid in $(jobs -p); do
    pkill -TERM -P "${pid}" 2> /dev/null || true
    kill -TERM "${pid}" 2> /dev/null || true
  done
}

# Fail unless every replication of beta group GROUP has its Naturalness bundle.
require_bundles() {
  local group="$1" leaf count
  leaf="${RUNS_DIR}/${GROUP_NAMES[group]}/${BACKEND_LEAF}"
  count="$( (find "${leaf}/replications" -maxdepth 2 -name naturalness_run_index.parquet 2> /dev/null || true) | wc -l)"
  (( count == N_REPLICATIONS )) \
    || die "${GROUP_NAMES[group]} has ${count}/${N_REPLICATIONS} Naturalness fits; run the naturalness stage."
}

# Succeed when beta group GROUP has finished its comparators stage.
comparators_done() {
  [[ -f "${RUNS_DIR}/${GROUP_NAMES[$1]}/${BACKEND_LEAF}/baseline_variants/identification_bounds/mc_intervals.parquet" ]]
}

# Export B_LEARNER_REPOSITORY, cloning BLearner at the paper's commit if needed.
ensure_blearner() {
  local head
  if [[ -n "${B_LEARNER_REPOSITORY:-}" ]]; then
    [[ -d "${B_LEARNER_REPOSITORY}" ]] || die "B_LEARNER_REPOSITORY=${B_LEARNER_REPOSITORY} does not exist"
    head="$(git -C "${B_LEARNER_REPOSITORY}" rev-parse HEAD 2> /dev/null || echo unknown)"
    [[ "${head}" == "${BLEARNER_COMMIT}" ]] \
      || warn "BLearner in ${B_LEARNER_REPOSITORY} is at ${head}; the paper used ${BLEARNER_COMMIT}"
    return 0
  fi
  local checkout="${OUTPUT_DIR}/deps/BLearner"
  if [[ ! -d "${checkout}/.git" ]]; then
    note "comparators: cloning BLearner ${BLEARNER_COMMIT:0:7} into ${checkout}"
    run git clone --quiet "${BLEARNER_URL}" "${checkout}" \
      || die "Could not clone ${BLEARNER_URL}; set B_LEARNER_REPOSITORY to a checkout at ${BLEARNER_COMMIT}"
  fi
  if (( DRY_RUN )) || [[ "$(git -C "${checkout}" rev-parse HEAD)" != "${BLEARNER_COMMIT}" ]]; then
    run git -C "${checkout}" checkout --quiet "${BLEARNER_COMMIT}"
  fi
  export B_LEARNER_REPOSITORY="${checkout}"
}

# Check that the Python environment imports what the selected stages need.
preflight() {
  local imports=() log="${LOG_DIR}/preflight.log"
  if stage_selected data && [[ ! -f "${RCT_PICKLE}" ]]; then
    imports+=(pyreadstat)
  fi
  if stage_selected naturalness || stage_selected comparators || stage_selected merge; then
    imports+=(experiments.exp25_osrct_mc_baseline_comparison)
  fi
  if stage_selected reground; then
    imports+=(experiments.exp25_tau_floor_regrounding)
  fi
  if stage_selected report; then
    imports+=(experiments.exp25_hit_ratio_analysis)
  fi
  if (( ${#imports[@]} > 0 )); then
    py "${log}" 1 "${NO_GPU[@]}" -- -c "import $(IFS=,; echo "${imports[*]}")" \
      || die "The Python environment cannot import what these stages need; see ${log}"
  fi
  if (( GPUS > 0 )) && { stage_selected naturalness || stage_selected reground; }; then
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

# Fit the Naturalness jobs of one worker: jobs WORKER, WORKER + n, WORKER + 2n,
# ... of the job list, where n is the number of workers.
#   naturalness_worker WORKER GROUP:REPLICATION...
naturalness_worker() {
  local worker="$1"
  shift
  local jobs=("$@") index group replication log status=0 betas=()
  local devices=("${NO_GPU[@]}") device_label="CPU"
  if (( GPUS > 0 )); then
    devices=("CUDA_VISIBLE_DEVICES=${GPU_IDS[worker % GPUS]}")
    device_label="GPU ${GPU_IDS[worker % GPUS]}"
  fi
  for (( index = worker; index < ${#jobs[@]}; index += NATURALNESS_JOBS )); do
    group="${jobs[index]%%:*}"
    replication="${jobs[index]##*:}"
    read -r -a betas <<< "${BETA_GROUPS[group]}"
    log="${LOG_DIR}/naturalness_${GROUP_NAMES[group]}_replication_$(printf '%03d' "${replication}").log"
    (( DRY_RUN )) || note "naturalness: start ${GROUP_NAMES[group]} replication ${replication} on ${device_label}"
    if py "${log}" 1 "${devices[@]}" -- -m experiments.exp25_osrct_mc_baseline_comparison \
        "${EXP25_ARGS[@]}" --betas "${betas[@]}" --tabicl-device "${TABICL_DEVICE}" \
        --output-dir "${RUNS_DIR}/${GROUP_NAMES[group]}" --scratch-dir "${SCRATCH_DIR}" \
        --naturalness-only --replications "${replication}"; then
      (( DRY_RUN )) || note "naturalness: done ${GROUP_NAMES[group]} replication ${replication}"
    else
      warn "naturalness: ${GROUP_NAMES[group]} replication ${replication} failed; see ${log}"
      status=1
    fi
  done
  return "${status}"
}

# naturalness: OSRCT samples and Naturalness fits, one job per replication and
# beta group, spread over the parallel workers.
stage_naturalness() {
  [[ -f "${RCT_PICKLE}" ]] || (( DRY_RUN )) || die "naturalness: no ${RCT_PICKLE}; run the data stage."
  # Replication 0 initializes a group's store and the other replications wait
  # for it, so both groups' replication 0 come first.
  local jobs=() replication group
  for (( replication = 0; replication < N_REPLICATIONS; replication++ )); do
    for group in "${!GROUP_NAMES[@]}"; do
      jobs+=("${group}:${replication}")
    done
  done
  note "naturalness: ${#jobs[@]} fits in ${NATURALNESS_JOBS} parallel job(s); logs ${LOG_DIR}/naturalness_*.log"
  local pids=() worker
  for (( worker = 0; worker < NATURALNESS_JOBS; worker++ )); do
    if (( DRY_RUN )); then
      naturalness_worker "${worker}" "${jobs[@]}"
    else
      naturalness_worker "${worker}" "${jobs[@]}" &
      pids+=("$!")
    fi
  done
  wait_for naturalness "${pids[@]}"
  if (( ! DRY_RUN )); then
    for group in "${!GROUP_NAMES[@]}"; do
      require_bundles "${group}"
    done
  fi
}

# comparators: CB-IPW, CB-DR, B-Learner, Manski and Balke-Pearl per beta group.
stage_comparators() {
  local group log betas=()
  ensure_blearner
  for group in "${!GROUP_NAMES[@]}"; do
    if comparators_done "${group}"; then
      note "comparators: ${GROUP_NAMES[group]} is finished; skipping"
      continue
    fi
    (( DRY_RUN )) || require_bundles "${group}"
    read -r -a betas <<< "${BETA_GROUPS[group]}"
    log="${LOG_DIR}/comparators_${GROUP_NAMES[group]}.log"
    note "comparators: ${GROUP_NAMES[group]} with ${COMPARATOR_WORKERS} worker(s); log ${log}"
    py "${log}" 1 "${NO_GPU[@]}" -- -m experiments.exp25_osrct_mc_baseline_comparison \
      "${EXP25_ARGS[@]}" --betas "${betas[@]}" --tabicl-device "${TABICL_DEVICE}" \
      --output-dir "${RUNS_DIR}/${GROUP_NAMES[group]}" --scratch-dir "${SCRATCH_DIR}" \
      --exp18-n-jobs "${COMPARATOR_WORKERS}" --exp18-worker-threads 1 --comparators-only \
      || die "comparators: ${GROUP_NAMES[group]} failed; see ${log}"
  done
}

# merge: one results tree over the four betas.
stage_merge() {
  if [[ -f "${MERGED_LEAF}/merge_manifest.json" ]]; then
    note "merge: ${MERGED_DIR} exists; skipping"
    return 0
  fi
  [[ ! -e "${MERGED_DIR}" ]] \
    || die "${MERGED_DIR} exists without merge_manifest.json, probably from an interrupted merge; remove it and rerun."
  local group
  for group in "${!GROUP_NAMES[@]}"; do
    comparators_done "${group}" || (( DRY_RUN )) \
      || die "merge: ${GROUP_NAMES[group]} is unfinished; run the comparators stage."
  done
  note "merge: ${GROUP_NAMES[*]} into ${MERGED_DIR}"
  py "${LOG_DIR}/merge.log" 1 "${NO_GPU[@]}" -- -m experiments.exp25_osrct_mc_baseline_comparison merge \
    "${RUNS_DIR}/${GROUP_NAMES[0]}" "${RUNS_DIR}/${GROUP_NAMES[1]}" --output-dir "${MERGED_DIR}" \
    || die "merge failed; see ${LOG_DIR}/merge.log"
}

# reground: Naturalness and Balke-Pearl on the relevance grid tau >= 0.075,
# assembled into the paper's results tree.
stage_reground() {
  if [[ -f "${PAPER_LEAF}/regrounding_manifest.json" ]]; then
    note "reground: ${PAPER_DIR} exists; skipping"
    return 0
  fi
  [[ ! -e "${PAPER_DIR}" ]] \
    || die "${PAPER_DIR} exists without regrounding_manifest.json, probably from an interrupted assembly; remove it and rerun."
  [[ -f "${MERGED_LEAF}/merge_manifest.json" ]] || (( DRY_RUN )) || die "reground: run the merge stage first."
  local stores=() group count
  for group in "${!GROUP_NAMES[@]}"; do
    stores+=("${RUNS_DIR}/${GROUP_NAMES[group]}/${BACKEND_LEAF}")
  done

  note "reground: target probability tables in ${TABLES_JOBS} job(s)"
  run_shards reground_tables "${TABLES_JOBS}" "${REGROUND_THREADS}" 1 -- \
    -m experiments.exp25_tau_floor_regrounding --mode tables --stores "${stores[@]}" \
    --output-dir "${REGROUND_ROOT}" --betas "${PAPER_BETAS[@]}" --tabicl-device "${TABICL_DEVICE}"

  note "reground: Naturalness at tau >= ${TAU_FLOOR} in ${REGROUND_JOBS} job(s)"
  run_shards reground_lp "${REGROUND_JOBS}" "${REGROUND_THREADS}" 0 -- \
    -m experiments.exp25_tau_floor_regrounding --mode reground --stores "${stores[@]}" \
    --output-dir "${REGROUND_ROOT}" --betas "${PAPER_BETAS[@]}" --floors "${TAU_FLOOR}" \
    --lp-posterior-n-jobs "${REGROUND_THREADS}"
  if (( ! DRY_RUN )); then
    count="$( (find "${REGROUND_ROOT}/floor=${TAU_FLOOR}" -name posterior_summary.parquet 2> /dev/null || true) | wc -l)"
    (( count == N_CONDITIONS )) \
      || die "reground: ${count}/${N_CONDITIONS} conditions regrounded; rerun to retry the others."
  fi

  note "reground: Manski and Balke-Pearl at tau = ${TAU_FLOOR}"
  py "${LOG_DIR}/reground_bounds.log" "${BOUNDS_THREADS}" "${NO_GPU[@]}" -- \
    -m experiments.exp25_tau_floor_regrounding --mode bounds --stores "${stores[@]}" \
    --output-dir "${BOUNDS_ROOT}" --floors "${TAU_FLOOR}" \
    || die "reground: bounds failed; see ${LOG_DIR}/reground_bounds.log"

  note "reground: assembling ${PAPER_DIR}"
  py "${LOG_DIR}/reground_assemble.log" 1 "${NO_GPU[@]}" -- \
    -m experiments.exp25_tau_floor_regrounding --mode assemble --output-dir "${REGROUND_ROOT}" \
    --merged-leaf "${MERGED_LEAF}" --floor "${TAU_FLOOR}" --destination "${PAPER_LEAF}" \
    --bounds-root "${BOUNDS_ROOT}" \
    || die "reground: assembly failed; see ${LOG_DIR}/reground_assemble.log"
}

# report: the paper's tables and figures from the assembled (or --tree) tree.
stage_report() {
  if (( ! DRY_RUN )); then
    [[ -f "${REPORT_LEAF}/baseline_variants/identification_bounds/mc_intervals.parquet" ]] \
      || die "report: no finished Exp25 tree at ${REPORT_LEAF}; run the earlier stages or pass --tree."
    grep -q "\"tau_floor\": ${TAU_FLOOR}" "${REPORT_LEAF}/regrounding_manifest.json" 2> /dev/null \
      || warn "report: ${REPORT_LEAF} was not regrounded at tau >= ${TAU_FLOOR}, so its numbers are not the paper's"
  fi
  local variant output
  local -A outputs=(
    [stratum]="${REPORT_DIR}/analysis_failures_as_manski"
    [draws]="${REPORT_DIR}/analysis_failures_as_manski_draws"
  )
  for variant in stratum draws; do
    output="${outputs[${variant}]}"
    note "report: ${output}"
    py "${LOG_DIR}/report_${variant}.log" "${CPUS}" "${NO_GPU[@]}" -- -m experiments.exp25_hit_ratio_analysis \
      "${REPORT_LEAF}" --failures-as-manski "${variant}" --plot-formats pdf png --output-dir "${output}" \
      || die "report: ${variant} analysis failed; see ${LOG_DIR}/report_${variant}.log"
  done

  local entry extension source="${outputs[stratum]}"
  run mkdir -p "${OUTPUT_DIR}/tables" "${OUTPUT_DIR}/figures"
  for entry in "${PAPER_TABLES[@]}"; do
    run cp "${source}/${entry#*=}" "${OUTPUT_DIR}/tables/${entry%%=*}"
  done
  for entry in "${PAPER_FIGURES[@]}"; do
    for extension in pdf png; do
      run cp "${source}/figures/${entry#*=}.${extension}" "${OUTPUT_DIR}/figures/${entry%%=*}.${extension}"
    done
  done
  note "report: tables in ${OUTPUT_DIR}/tables, figures in ${OUTPUT_DIR}/figures"
}

# --- Command line --------------------------------------------------------------

OUTPUT_DIR="${PROJECT_ROOT}/results/paper/experiment1_manylabs"
STAGES=""
TREE=""
MANYLABS_SAV="${PROJECT_ROOT}/osrct_benchmark_v1.0/data/CleanedDataset.sav"
CONDA_ENV=""
PYTHON_EXE="python"
GPUS=""
JOBS_PER_GPU=1
CPUS=""
MEMORY_GB=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --output-dir|--stages|--tree|--manylabs-sav|--conda-env|--python|--gpus|--jobs-per-gpu|--cpus|--memory-gb)
      [[ $# -ge 2 ]] || die "$1 needs a value (see --help)"
      case "$1" in
        --output-dir) OUTPUT_DIR="$2" ;;
        --stages) STAGES="$2" ;;
        --tree) TREE="$2" ;;
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

# Parallel jobs per stage, within the CPU, GPU and memory budgets.
if (( GPUS > 0 )); then
  NATURALNESS_JOBS=$(( GPUS * JOBS_PER_GPU ))
  TABLES_JOBS="${GPUS}"
else
  NATURALNESS_JOBS="$(at_least_one $(( CPUS / LP_THREADS )))"
  TABLES_JOBS="$(at_least_one $(( CPUS / REGROUND_THREADS )))"
fi
NATURALNESS_JOBS="$(min_of "${NATURALNESS_JOBS}" "$(memory_cap "${NATURALNESS_JOB_GB}")" $(( 2 * N_REPLICATIONS )))"
TABLES_JOBS="$(min_of "${TABLES_JOBS}" "$(memory_cap "${TABLES_JOB_GB}")")"
COMPARATOR_WORKERS="$(min_of "${CPUS}" "$(memory_cap "${COMPARATOR_WORKER_GB}")")"
REGROUND_JOBS="$(min_of "$(at_least_one $(( CPUS / REGROUND_THREADS )))" "$(memory_cap "${REGROUND_JOB_GB}")")"

if [[ -n "${CONDA_ENV}" ]]; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PY=("${PYTHON_EXE}")
fi

# Paths. Python runs from the repository root, so every path is absolute.
OUTPUT_DIR="$(absolute "${OUTPUT_DIR}")"
MANYLABS_SAV="$(absolute "${MANYLABS_SAV}")"
RUNS_DIR="${OUTPUT_DIR}/runs"
MERGED_DIR="${OUTPUT_DIR}/merged_beta_0p5_2"
MERGED_LEAF="${MERGED_DIR}/${BACKEND_LEAF}"
REGROUND_ROOT="${OUTPUT_DIR}/tau_floor_0p075/regrounding"
BOUNDS_ROOT="${OUTPUT_DIR}/tau_floor_0p075/identification_bounds"
PAPER_DIR="${OUTPUT_DIR}/merged_beta_0p5_2_tau_floor_0p075"
PAPER_LEAF="${PAPER_DIR}/${BACKEND_LEAF}"
REPORT_DIR="${OUTPUT_DIR}/report"
LOG_DIR="${OUTPUT_DIR}/logs"
SCRATCH_DIR="${OUTPUT_DIR}/scratch"
REPORT_LEAF="${PAPER_LEAF}"
if [[ -n "${TREE}" ]]; then
  REPORT_LEAF="$(backend_leaf_of "${TREE}")"
fi
cd "${PROJECT_ROOT}"

# --- Run -----------------------------------------------------------------------

if (( GPUS > 0 )); then
  gpu_text="${GPUS} (${GPU_IDS[*]}), TabICLv2 on cuda"
else
  gpu_text="none, TabICLv2 on the CPU"
fi
cat << EOF
Experiment I (ManyLabs, Exp25) reproduction
  output     ${OUTPUT_DIR}
  stages     ${RUN_STAGES[*]}
  python     ${PY[*]}
  GPUs       ${gpu_text}
  CPUs       ${CPUS}
  memory     ${MEMORY_GB} GB
  parallel   naturalness ${NATURALNESS_JOBS} job(s), comparators ${COMPARATOR_WORKERS} worker(s),
             target tables ${TABLES_JOBS} job(s), regrounding ${REGROUND_JOBS} job(s)
EOF
if (( DRY_RUN )); then
  echo "  (dry run: the commands below are printed, not run)"
fi
if stage_selected naturalness || stage_selected reground; then
  if (( GPUS == 0 )); then
    warn "No GPU in use: TabICLv2 runs on the CPU, which is much slower and can change the last digits." \
      "Pass --gpus N if this machine has GPUs that were not detected."
  elif (( NATURALNESS_JOBS < GPUS * JOBS_PER_GPU )); then
    warn "The ${MEMORY_GB} GB budget allows ${NATURALNESS_JOBS} Naturalness job(s) at ${NATURALNESS_JOB_GB} GB each," \
      "fewer than the GPUs could run; raise --memory-gb if the machine has more."
  fi
  if stage_selected naturalness && (( NATURALNESS_JOBS * LP_THREADS > CPUS )); then
    warn "${NATURALNESS_JOBS} Naturalness job(s) x ${LP_THREADS} LP threads exceed the ${CPUS} CPU cores; they will share them."
  fi
fi

if (( ! DRY_RUN )); then
  mkdir -p "${LOG_DIR}" "${SCRATCH_DIR}"
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
