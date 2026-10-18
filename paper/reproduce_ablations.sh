#!/usr/bin/env bash
#
# Reproduce the ablations on Naturalness conditions of the Naturalness paper
# (appendix "Ablations on Naturalness conditions") on one machine.
#
# The ablations reuse the design of Experiment I (Exp25 in this repository), so
# they read the outputs of paper/reproduce_experiment1_manylabs.sh: its OSRCT
# samples and stored source evidence, its target probability tables, its
# regrounded Naturalness intervals and its Manski and Balke-Pearl bounds. Run
# that script first (at least its naturalness, comparators, merge and reground
# stages). This script writes the ablations' table and figures to OUTPUT_DIR:
#
#   tables/ablation_family.tex                              tab:ablation-family
#   figures/ablation_effect_shift_validity.{pdf,png}        fig:effect-shift
#   figures/ablation_effect_shift_validity_by_study.{pdf,png}
#                                                           fig:effect-shift-by-study
#   figures/ablation_grounding_frontier_by_study.{pdf,png}  fig:ablation-grounding-by-study
#   figures/ablation_grounding_frontier_by_beta.{pdf,png}   fig:ablation-grounding-by-beta
#
# The table is a booktabs tabular to \input inside a table float. Every method
# is scored as in Experiment I: a stratum that a method cannot bound gets its
# Manski interval. arms/summary/ holds the CSV tables behind the numbers the
# appendix quotes, such as the realized shifts and the binding tolerances.
#
# STAGES, in order. --stages runs a subset. Every stage resumes, so rerunning
# the same command continues a stopped run.
#   tables     The target's probability tables after each effect shift
#              (Delta = 0.1, 0.2, 0.3). Uses the GPUs.
#   shift      Effect shift: the reference, Naturalness, Manski, Balke-Pearl
#              and the transport baseline on each shifted target. CPU only.
#   family     Family mismatch: Naturalness grounded on the sources of the
#              condition at the farthest beta, or selected on another
#              covariate. CPU only.
#   grounding  Grounding ablation: the program under fixed, exact and
#              prior-only tolerances. CPU only.
#   report     The scored arms, then the paper's table and figures above.
#
# Naturalness is grounded as in Experiment I (per-site Gaussian process,
# relevance grid from tau = 0.075). The code also offers an envelope grounding
# and an invalid-instrument arm, which the paper does not report and this
# script does not run.
#
# RESOURCES. Each defaults to what the machine has.
#   --gpus N          GPUs to use. The tables stage runs its jobs on them; 0
#                     runs TabICLv2 on the CPU, which is much slower.
#   --jobs-per-gpu N  Tables jobs sharing one GPU (default: 1).
#   --cpus N          CPU cores. Every CPU job runs on 4 cores.
#   --memory-gb N     Memory budget in GB. It allows one parallel job per 24 GB
#                     in the tables stage and per 8 GB in the others, the
#                     requests of the paper's cluster jobs. It sizes the
#                     parallelism and is not enforced as a limit.
#
# COST on the paper's cluster runs: tables 41 GPU-minutes; shift 71 h, family
# 44 h and grounding 13 h of 4-core jobs, about 510 core-hours in all (16 h on
# 32 cores); report a few minutes. The outputs take under 1 GB of disk.
#
# REPRODUCIBILITY. Seeds, grids and settings are the paper's, and each stage
# reads Experiment I's outputs exactly as the paper's runs read the published
# ones. Checked against the paper's runs on a condition each: the grounding
# ablation reproduces its outputs exactly; the effect shift reproduces the
# shifted reference, Manski, Balke-Pearl and transport exactly and Naturalness
# to 3e-6 (with TabICLv2 on the CPU); the family mismatch reproduces
# Naturalness to 0.008, the run-to-run variation of the Gaussian-process fit
# across machines that Experiment I's regrounding also shows. The table
# writer, run on the paper's scored arms, reproduces tab:ablation-family
# exactly. Inherited from Experiment I: results agree with the paper's within
# the variation of its own reproduction.
#
# PREREQUISITES
#   * The Python environment of paper/reproduce_experiment1_manylabs.sh.
#     Activate it, or pass --conda-env.
#   * Experiment I's outputs (--exp1-dir).
#
# USAGE
#   paper/reproduce_ablations.sh [options]
#
#   --output-dir DIR     Results root (default: results/paper/ablations).
#   --exp1-dir DIR       Output directory of paper/reproduce_experiment1_manylabs.sh
#                        (default: results/paper/experiment1_manylabs).
#   --stages LIST        Comma-separated stages to run (default: all).
#   --tree DIR           Only run the report stage, on the finished arms in DIR
#                        (delta=*, d3_beta, d3_covariate and ablation, as this
#                        script writes them under OUTPUT_DIR/arms), for example
#                        a published copy. They are linked, never written to.
#   --conda-env NAME     Run Python through `conda run -n NAME`.
#   --python EXE         Python interpreter otherwise (default: python).
#   --dry-run            Print the plan and every command; run nothing.
#   -h, --help           Show this help.
#   and the RESOURCES options above.
#
# EXAMPLES
#   paper/reproduce_ablations.sh --gpus 1 --cpus 32 --memory-gb 128
#   paper/reproduce_ablations.sh --dry-run
#   paper/reproduce_ablations.sh --tree /data/exp25_mechanism_arms
set -euo pipefail

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "ERROR: bash 4.4 or newer is required" >&2
  exit 1
fi

SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
ORIGINAL_ARGS=("$@")

# The paper's design. Changing any of these changes the results.
MODULE=experiments.exp25_tau_floor_regrounding
N_CONDITIONS=720  # 20 replications x 3 studies x 3 selection covariates x 4 betas
BETAS=(0.5 1.0 1.5 2.0)
TAU_FLOOR=0.075
SHIFTS=(0.1 0.2 0.3)
MISMATCHES=(beta covariate)
# Experiment I's grounding, as the paper's per-site runs passed it.
PER_SITE=(--gp-training-target per_site --gp-target-scale note --target-pairs all)
# A CPU job's BLAS and LP-posterior threads.
THREADS=4
# Experiment I's layout (paper/reproduce_experiment1_manylabs.sh).
GROUP_NAMES=(beta_0p5_1 beta_1p5_2)
BACKEND_LEAF="tabicl_v2/tabicl_v2/tabicl_v2"

# Memory (GB) per job, as the paper's cluster jobs requested it.
TABLES_JOB_GB=24
CPU_JOB_GB=8

# Paper file name = file name in arms/summary.
PAPER_TABLES=("ablation_family.tex=family_mismatch_table.tex")
PAPER_FIGURES=(
  "ablation_effect_shift_validity=figures/effect_shift_validity"
  "ablation_effect_shift_validity_by_study=figures/effect_shift_validity_by_study"
  "ablation_grounding_frontier_by_study=figures/grounding_ablation_frontier_by_study"
  "ablation_grounding_frontier_by_beta=figures/grounding_ablation_frontier_by_beta"
)
ARM_DIRS=(delta=0.1 delta=0.2 delta=0.3 d3_beta d3_covariate ablation)
ALL_STAGES=(tables shift family grounding report)

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
  (( failed == 0 )) || die "${name}: some jobs failed (see ${LOG_DIR}/${name}_shard_*.log)." \
    "Rerun the same command to retry them; finished conditions are kept."
}

# Run one mode of the regrounding module as parallel shards, each with
# --shard K --n-shards JOBS. Finished conditions are skipped by the module.
#   run_shards NAME JOBS USES_GPU -- MODULE_ARGS...
# A shard that uses the GPU gets GPU K modulo --gpus; the others see none.
run_shards() {
  local name="$1" jobs="$2" uses_gpu="$3"
  shift 4
  local pids=() shard devices log
  for (( shard = 0; shard < jobs; shard++ )); do
    devices=("${NO_GPU[@]}")
    if (( uses_gpu && GPUS > 0 )); then
      devices=("CUDA_VISIBLE_DEVICES=${GPU_IDS[shard % GPUS]}")
    fi
    log="${LOG_DIR}/${name}_shard_${shard}.log"
    if (( DRY_RUN )); then
      py "${log}" "${THREADS}" "${devices[@]}" -- -m "${MODULE}" "$@" --shard "${shard}" --n-shards "${jobs}"
    else
      py "${log}" "${THREADS}" "${devices[@]}" -- -m "${MODULE}" "$@" --shard "${shard}" --n-shards "${jobs}" &
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

# Print how many files called NAME lie under DIR.
count_files() {
  (find "$1" -name "$2" 2> /dev/null || true) | wc -l
}

# Fail unless Experiment I's outputs that the ablations read are complete.
require_exp1() {
  (( DRY_RUN )) && return 0
  local store
  for store in "${STORES[@]}"; do
    [[ -f "${store}/run_manifest.yml" && -d "${store}/replications" ]] \
      || die "No Experiment I store at ${store}; run paper/reproduce_experiment1_manylabs.sh first (see --exp1-dir)."
  done
  (( $(count_files "${EXP1_TABLES}" target_tables.pkl) == N_CONDITIONS )) \
    || die "Experiment I's target tables are incomplete in ${EXP1_TABLES}; run its reground stage."
  (( $(count_files "${EXP1_PUBLISHED}/floor=${TAU_FLOOR}" posterior_summary.parquet) == N_CONDITIONS )) \
    || die "Experiment I's regrounded intervals are incomplete in ${EXP1_PUBLISHED}; run its reground stage."
  for store in "${GROUP_NAMES[@]}"; do
    [[ -f "${EXP1_IDENTITIES}/${store}/mc_intervals.parquet" ]] \
      || die "No Manski and Balke-Pearl bounds at ${EXP1_IDENTITIES}/${store}; run Experiment I's reground stage."
  done
}

# Check that the Python environment imports what the selected stages need.
preflight() {
  local log="${LOG_DIR}/preflight.log"
  py "${log}" 1 "${NO_GPU[@]}" -- -c "import ${MODULE}, experiments.exp25_hit_ratio_plots" \
    || die "The Python environment cannot import what these stages need; see ${log}"
  if (( GPUS > 0 )) && stage_selected tables; then
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

# tables: the target's probability tables after each effect shift.
stage_tables() {
  require_exp1
  local delta
  for delta in "${SHIFTS[@]}"; do
    if (( ! DRY_RUN )) && (( $(count_files "${ARMS}/delta=${delta}/tables" target_tables.pkl) == N_CONDITIONS )); then
      note "tables: delta=${delta} is finished; skipping"
      continue
    fi
    note "tables: delta=${delta} in ${TABLES_JOBS} job(s)"
    run_shards "tables_delta${delta}" "${TABLES_JOBS}" 1 -- --mode tables --stores "${STORES[@]}" \
      --output-dir "${ARMS}/delta=${delta}" --betas "${BETAS[@]}" --effect-shift "${delta}" \
      --tabicl-device "${TABICL_DEVICE}"
    if (( ! DRY_RUN )); then
      (( $(count_files "${ARMS}/delta=${delta}/tables" target_tables.pkl) == N_CONDITIONS )) \
        || die "tables: delta=${delta} is incomplete; rerun to retry."
    fi
  done
}

# Run one regrounded variant of the arms on every condition, unless finished.
#   reground_variant NAME OUTPUT FLAGS...
reground_variant() {
  local name="$1" output="$2"
  shift 2
  if (( ! DRY_RUN )) && (( $(count_files "${output}/floor=${TAU_FLOOR}" posterior_summary.parquet) == N_CONDITIONS )); then
    note "${name}: finished; skipping"
    return 0
  fi
  note "${name}: ${N_CONDITIONS} conditions in ${CPU_JOBS} job(s)"
  run_shards "${name}" "${CPU_JOBS}" 0 -- --mode reground --stores "${STORES[@]}" --output-dir "${output}" \
    --betas "${BETAS[@]}" --floors "${TAU_FLOOR}" --lp-posterior-n-jobs "${THREADS}" "${PER_SITE[@]}" "$@"
  if (( ! DRY_RUN )); then
    (( $(count_files "${output}/floor=${TAU_FLOOR}" posterior_summary.parquet) == N_CONDITIONS )) \
      || die "${name}: incomplete; rerun to retry."
  fi
}

# shift: every method on each shifted target.
stage_shift() {
  require_exp1
  local delta
  for delta in "${SHIFTS[@]}"; do
    if (( ! DRY_RUN )) && (( $(count_files "${ARMS}/delta=${delta}/tables" target_tables.pkl) != N_CONDITIONS )); then
      die "shift: the delta=${delta} tables are unfinished; run the tables stage."
    fi
    reground_variant "shift_delta${delta}" "${ARMS}/delta=${delta}/per_site" --effect-shift "${delta}" \
      --identities-root "${EXP1_IDENTITIES}" --tables-dir "${ARMS}/delta=${delta}/tables"
  done
}

# family: Naturalness grounded on another condition's sources.
stage_family() {
  require_exp1
  local mismatch
  for mismatch in "${MISMATCHES[@]}"; do
    reground_variant "family_${mismatch}" "${ARMS}/d3_${mismatch}/per_site" --source-mismatch "${mismatch}" \
      --tables-dir "${EXP1_TABLES}"
  done
}

# grounding: the program under tolerances that ignore the source trials.
stage_grounding() {
  require_exp1
  local output="${ARMS}/ablation"
  if (( ! DRY_RUN )) && (( $(count_files "${output}/floor=${TAU_FLOOR}" ablation_summary.parquet) == N_CONDITIONS )); then
    note "grounding: finished; skipping"
    return 0
  fi
  note "grounding: ${N_CONDITIONS} conditions in ${CPU_JOBS} job(s)"
  run_shards grounding "${CPU_JOBS}" 0 -- --mode ablate --stores "${STORES[@]}" --output-dir "${output}" \
    --tables-dir "${EXP1_TABLES}" --betas "${BETAS[@]}" --floors "${TAU_FLOOR}" --lp-posterior-n-jobs "${THREADS}"
  if (( ! DRY_RUN )); then
    (( $(count_files "${output}/floor=${TAU_FLOOR}" ablation_summary.parquet) == N_CONDITIONS )) \
      || die "grounding: incomplete; rerun to retry."
  fi
}

# report: score the arms, then copy the paper's table and figures.
stage_report() {
  require_exp1
  local arm
  if [[ -n "${TREE}" ]]; then
    # The summary is written next to the arms, so a finished tree is linked
    # into a directory of its own rather than written to.
    run mkdir -p "${ARMS}"
    for arm in "${ARM_DIRS[@]}"; do
      if [[ -e "${TREE}/${arm}" ]]; then
        if [[ -e "${ARMS}/${arm}" && ! -L "${ARMS}/${arm}" ]]; then
          die "${ARMS}/${arm} exists and is not a link to ${TREE}/${arm}"
        fi
        run ln -sfn "${TREE}/${arm}" "${ARMS}/${arm}"
      fi
    done
  fi
  if (( ! DRY_RUN )); then
    for arm in "${ARM_DIRS[@]}"; do
      [[ -e "${ARMS}/${arm}" ]] || warn "report: no ${ARMS}/${arm}; its arm is left out of the summary."
    done
  fi
  note "report: ${ARMS}/summary"
  py "${LOG_DIR}/report.log" "${THREADS}" "${NO_GPU[@]}" -- -m "${MODULE}" --mode summarize-arms \
    --stores "${STORES[@]}" --output-dir "${ARMS}" --published-root "${EXP1_PUBLISHED}" \
    --identities-root "${EXP1_IDENTITIES}" --betas "${BETAS[@]}" --floors "${TAU_FLOOR}" \
    || die "report failed; see ${LOG_DIR}/report.log"

  local entry extension
  run mkdir -p "${OUTPUT_DIR}/tables" "${OUTPUT_DIR}/figures"
  for entry in "${PAPER_TABLES[@]}"; do
    run cp "${ARMS}/summary/${entry#*=}" "${OUTPUT_DIR}/tables/${entry%%=*}"
  done
  for entry in "${PAPER_FIGURES[@]}"; do
    for extension in pdf png; do
      run cp "${ARMS}/summary/${entry#*=}.${extension}" "${OUTPUT_DIR}/figures/${entry%%=*}.${extension}"
    done
  done
  note "report: tables in ${OUTPUT_DIR}/tables, figures in ${OUTPUT_DIR}/figures"
}

# --- Command line --------------------------------------------------------------

OUTPUT_DIR="${PROJECT_ROOT}/results/paper/ablations"
EXP1_DIR="${PROJECT_ROOT}/results/paper/experiment1_manylabs"
STAGES=""
TREE=""
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
    --output-dir|--exp1-dir|--stages|--tree|--conda-env|--python|--gpus|--jobs-per-gpu|--cpus|--memory-gb)
      [[ $# -ge 2 ]] || die "$1 needs a value (see --help)"
      case "$1" in
        --output-dir) OUTPUT_DIR="$2" ;;
        --exp1-dir) EXP1_DIR="$2" ;;
        --stages) STAGES="$2" ;;
        --tree) TREE="$2" ;;
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
CPU_JOBS="$(min_of "$(at_least_one $(( CPUS / THREADS )))" "$(memory_cap "${CPU_JOB_GB}")" "${N_CONDITIONS}")"
if (( GPUS > 0 )); then
  TABLES_JOBS=$(( GPUS * JOBS_PER_GPU ))
else
  TABLES_JOBS="${CPU_JOBS}"
fi
TABLES_JOBS="$(min_of "${TABLES_JOBS}" "$(memory_cap "${TABLES_JOB_GB}")" "${N_CONDITIONS}")"

if [[ -n "${CONDA_ENV}" ]]; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PY=("${PYTHON_EXE}")
fi

# Paths. Python runs from the repository root, so every path is absolute.
OUTPUT_DIR="$(absolute "${OUTPUT_DIR}")"
EXP1_DIR="$(absolute "${EXP1_DIR}")"
[[ -z "${TREE}" ]] || TREE="$(absolute "${TREE}")"
[[ -z "${TREE}" || -d "${TREE}" ]] || die "--tree ${TREE} is not a directory"
STORES=()
for group in "${GROUP_NAMES[@]}"; do STORES+=("${EXP1_DIR}/runs/${group}/${BACKEND_LEAF}"); done
EXP1_TABLES="${EXP1_DIR}/tau_floor_0p075/regrounding/tables"
EXP1_PUBLISHED="${EXP1_DIR}/tau_floor_0p075/regrounding"
EXP1_IDENTITIES="${EXP1_DIR}/tau_floor_0p075/identification_bounds/floor=${TAU_FLOOR}"
ARMS="${OUTPUT_DIR}/arms"
[[ -z "${TREE}" ]] || ARMS="${OUTPUT_DIR}/tree_arms"
LOG_DIR="${OUTPUT_DIR}/logs"
cd "${PROJECT_ROOT}"

# --- Run -----------------------------------------------------------------------

if (( GPUS > 0 )); then
  gpu_text="${GPUS} (${GPU_IDS[*]}), TabICLv2 on cuda"
else
  gpu_text="none, TabICLv2 on the CPU"
fi
cat << EOF
Ablations on Naturalness conditions (Experiment I design) reproduction
  output     ${OUTPUT_DIR}
  exp. I     ${EXP1_DIR}
  stages     ${RUN_STAGES[*]}
  python     ${PY[*]}
  GPUs       ${gpu_text}
  CPUs       ${CPUS}
  memory     ${MEMORY_GB} GB
  parallel   tables ${TABLES_JOBS} job(s), other stages ${CPU_JOBS} job(s) of ${THREADS} cores
EOF
if (( DRY_RUN )); then
  echo "  (dry run: the commands below are printed, not run)"
fi
if stage_selected tables && (( GPUS == 0 )); then
  warn "No GPU in use: TabICLv2 runs on the CPU, which is much slower and can change the last digits." \
    "Pass --gpus N if this machine has GPUs that were not detected."
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
