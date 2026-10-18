#!/usr/bin/env bash
#
# Reproduce Experiment II (Twins) of the Naturalness paper on one machine.
#
# Experiment II is Exp28 in this repository. Starting from the public Twins
# benchmark files, the script rebuilds the results tree that the paper reports
# and writes the experiment's table and figure to OUTPUT_DIR:
#
#   tables/twins_width_by_beta.tex          tab:twins-width-by-beta
#   figures/twins_width_by_beta.{pdf,png}   fig:widths_twins
#
# The table is a booktabs tabular to \input inside a table float. Each cell is
# the mean over conditions of the weighted width and, in parentheses, the mean
# reference hit ratio, with every covariate cell weighted by its target mass
# times the probability that the Naturalness LP draw is optimal; the figure
# draws the same widths with 90% Student-t bands.
#
# STAGES, in order. --stages runs a subset. Every stage resumes, so rerunning
# the same command continues a stopped run.
#   data         The three Twins CSVs, downloaded from the CEVAE repository
#                and checked against the checksums of the paper's copies.
#   naturalness  Naturalness, Manski and Balke-Pearl on all 720 conditions
#                (6 target sites x 6 betas x 20 splitting seeds). Uses the GPUs.
#   comparators  Manski, the B-Learner (at exp(beta) and the oracle Gamma),
#                CB-IPW and CB-DR on the same 720 conditions. CPU only.
#   merge        One tree over both passes, keeping the conditions whose
#                discovered candidate's tolerance covers its own measured
#                deviation (404 of 720 in the paper).
#   report       Every figure and table of the merged tree, then the paper's
#                table and figure above.
#
# Both passes run the same jobs: seeds 20260918-20260920 as two jobs each
# (target sites 0-2 and 3-5) and seeds 20260921-20260937 as one job each. A
# condition's random draws depend only on its own dataset, seed, beta and
# target, so the split does not change results.
#
# RESOURCES. Each defaults to what the machine has.
#   --gpus N          GPUs to use. Each Naturalness job holds one GPU; 0 runs
#                     TabICLv2 on the CPU, which is much slower.
#   --jobs-per-gpu N  Naturalness jobs sharing one GPU (default: 1).
#   --cpus N          CPU cores. A Naturalness job runs 4 LP threads and a
#                     comparator job 8 fitting workers, so the comparators run
#                     one job per 8 cores.
#   --memory-gb N     Memory budget in GB. It allows one parallel job per 48 GB
#                     in either pass, the request of the paper's cluster jobs.
#                     It sizes the parallelism and is not enforced as a limit.
# Suggested: one or more CUDA GPUs (the paper used NVIDIA H100 80GB), with 8
# CPU cores and 48 GB of RAM per GPU.
#
# COST on the paper's cluster runs: naturalness 44-86 s per condition on an
# H100, 16 GPU-hours in all; comparators about 3.8 min per condition on 8 cores,
# about 46 h in all, or 8 h with 48 cores; merge and report a few minutes. The
# outputs take under 1 GB of disk.
#
# REPRODUCIBILITY. Seeds, grids, settings and input files are the paper's.
# Refitting a paper condition on a CPU reproduced Manski and both B-Learner
# arms to within 1e-6. CB-IPW and CB-DR are not deterministic, even between
# two runs on one machine: their CATENets fits draw from numpy's global random
# state. On that condition their cell endpoints moved by up to 0.14 and their
# weighted widths by up to 0.005, so their table entries can differ slightly
# from the paper's. TabICLv2 on another GPU model or on the CPU can change the
# last digits of Naturalness, which can move a condition across the merge's
# tolerance-covers-deviation filter. The paper's runs used the Python stack of
# ./Dockerfile: Python 3.11, torch 2.5.1 with CUDA 12.4, requirements.txt and
# BLearner at commit 734fd76.
#
# PREREQUISITES
#   * A Python 3.11 environment with torch 2.5.1 (the CUDA 12.4 build on GPU
#     machines) and `pip install -r requirements.txt`. Activate it, or pass
#     --conda-env, or run the script inside the image built by ./Dockerfile.
#   * Network access on first use: the data stage downloads the Twins files
#     unless --twins-dir holds them, TabICLv2 downloads its checkpoint to
#     HF_HOME, and the comparators stage clones BLearner at the pinned commit
#     unless B_LEARNER_REPOSITORY names a checkout.
#
# USAGE
#   paper/reproduce_experiment2_twins.sh [options]
#
#   --output-dir DIR     Results root (default: results/paper/experiment2_twins).
#   --stages LIST        Comma-separated stages to run (default: all).
#   --tree DIR           Only run the report stage, on a copy of this finished
#                        merged Exp28 tree, for example a published one.
#   --twins-dir DIR      Directory holding twin_pairs_{X,T,Y}_3years_samesex.csv,
#                        copied instead of downloading them.
#   --conda-env NAME     Run Python through `conda run -n NAME`.
#   --python EXE         Python interpreter otherwise (default: python).
#   --dry-run            Print the plan and every command; run nothing.
#   -h, --help           Show this help.
#   and the RESOURCES options above.
#
# EXAMPLES
#   paper/reproduce_experiment2_twins.sh --gpus 1 --cpus 32 --memory-gb 128
#   paper/reproduce_experiment2_twins.sh --dry-run
#   paper/reproduce_experiment2_twins.sh --tree /data/exp28_paper_tree
set -euo pipefail

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "ERROR: bash 4.4 or newer is required" >&2
  exit 1
fi

SCRIPT_PATH="${BASH_SOURCE[0]}"
PROJECT_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
ORIGINAL_ARGS=("$@")

# The paper's design. Changing any of these changes the results.
# A job is SEED:TARGET_SITES. The jobs cover seeds 20260918-20260937 and target
# sites 0-5, and both passes run every job.
JOB_LIST=(
  20260918:012 20260918:345 20260919:012 20260919:345 20260920:012 20260920:345
  20260921:012345 20260922:012345 20260923:012345 20260924:012345 20260925:012345
  20260926:012345 20260927:012345 20260928:012345 20260929:012345 20260930:012345
  20260931:012345 20260932:012345 20260933:012345 20260934:012345 20260935:012345
  20260936:012345 20260937:012345
)
N_BETAS=6
COMMON_ARGS=(
  --datasets twins_reduced
  --betas 0 0.25 0.5 1.0 1.5 2.0
  --twins-covariates birmon gestat10 mager8
  --twins-bias-covariate meduc6 --withheld-covariates meduc6
  --stage1-backend tabicl_v2
  --max-fit-rows 5000 --min-cell-count 20
  --lp-posterior-samples 1000
  --resume
)
# --n-jobs enters each run's resume fingerprint but not its results (the LP
# draws and the forest and meta-learner fits are seeded per fit), so both
# passes keep the paper's values under any --cpus and a run stays resumable.
NATURALNESS_THREADS=4
COMPARATOR_WORKERS=8
# The B-Learner path is inert in the Naturalness pass but fingerprinted, and
# its default depends on the machine, so it is fixed to the paper's value.
NATURALNESS_ARGS=(
  --methods manski naturalness balke_pearl --n-jobs "${NATURALNESS_THREADS}"
  --blearner-repository /opt/BLearner
)
COMPARATOR_ARGS=(
  --methods manski blearner_exp_beta blearner_oracle cb_ipw cb_dr
  --meta-learners cb_ipw cb_dr --tabicl-device cpu --n-jobs "${COMPARATOR_WORKERS}"
)
MERGE_ARGS=(--naturalness-trigger consistent)
TWINS_URL="https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS"
# SHA-256 of the files the paper's runs read.
TWINS_FILES=(
  "twin_pairs_X_3years_samesex.csv=6ec6263e678f67205d584ebda083173c1ec44fb46ea7cfaafe3c87a31f612b86"
  "twin_pairs_T_3years_samesex.csv=fbe6eb1cad27794c61cbbd8b11b72a26e96549e59d64fb88b07b30aaf2bae07d"
  "twin_pairs_Y_3years_samesex.csv=d03428d23ed308673f9151e3934aa7712df27c1fed244ed44ecc80a98a60836b"
)
BLEARNER_URL="https://github.com/CausalML/BLearner.git"
BLEARNER_COMMIT="734fd7639da54554e41d944bfc2187a24e75cf81"

# Memory (GB) per job, as the paper's cluster jobs requested it.
JOB_GB=48

# Paper file name = file name in the merged tree's report.
PAPER_TABLES=("twins_width_by_beta.tex=tables/exp28_optimal_width_by_beta.tex")
PAPER_FIGURES=("twins_width_by_beta=figures/exp28_optimal_width_by_beta")
ALL_STAGES=(data naturalness comparators merge report)

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
  (( failed == 0 )) || die "${name}: some jobs failed (see the warnings above and ${LOG_DIR})." \
    "Rerun the same command to retry them; finished work is kept."
}

# Stop the background jobs and the Python processes they started.
stop_jobs() {
  local pid
  for pid in $(jobs -p); do
    pkill -TERM -P "${pid}" 2> /dev/null || true
    kill -TERM "${pid}" 2> /dev/null || true
  done
}

# Print the run directory of a job of a pass.
#   job_dir PASS SEED:TARGET_SITES
job_dir() {
  printf '%s\n' "${OUTPUT_DIR}/$1/seed${2%%:*}-targets${2##*:}"
}

# Succeed when a job's run finished every condition: one completion record per
# beta and target site, and the run-level record written after the last one.
#   job_complete PASS SEED:TARGET_SITES
job_complete() {
  local directory sites="${2##*:}" count
  directory="$(job_dir "$1" "$2")"
  [[ -f "${directory}/completion.json" ]] || return 1
  count="$( (find "${directory}" -mindepth 2 -name completion.json 2> /dev/null || true) | wc -l)"
  (( count == N_BETAS * ${#sites} ))
}

# Fail unless every job of a pass finished.
#   require_pass PASS JOB...
require_pass() {
  local pass="$1" job missing=0
  shift
  for job in "$@"; do
    job_complete "${pass}" "${job}" || missing=$(( missing + 1 ))
  done
  (( missing == 0 )) || die "${pass}: ${missing} job(s) unfinished; run the ${pass} stage."
}

# Run the jobs of one pass that fall to one worker: jobs WORKER, WORKER + n,
# WORKER + 2n, ... of the list, where n is the number of workers. A run that
# finishes every condition counts as done even when it exits non-zero, which
# it does when an arm failed to fit some cells (those cells are scored with
# their Manski interval).
#   pass_worker PASS WORKER WORKERS JOB...
pass_worker() {
  local pass="$1" worker="$2" workers="$3"
  shift 3
  local jobs=("$@") index job seed sites targets directory log status=0
  local devices=("${NO_GPU[@]}") device_label="CPU"
  local extra_args=("${COMPARATOR_ARGS[@]}" --blearner-repository "${B_LEARNER_REPOSITORY:-}")
  if [[ "${pass}" == naturalness ]]; then
    extra_args=("${NATURALNESS_ARGS[@]}" --tabicl-device "${TABICL_DEVICE}")
    if (( GPUS > 0 )); then
      devices=("CUDA_VISIBLE_DEVICES=${GPU_IDS[worker % GPUS]}")
      device_label="GPU ${GPU_IDS[worker % GPUS]}"
    fi
  fi
  for (( index = worker; index < ${#jobs[@]}; index += workers )); do
    job="${jobs[index]}"
    seed="${job%%:*}"
    sites="${job##*:}"
    read -r -a targets <<< "$(sed 's/./& /g' <<< "${sites}")"
    directory="$(job_dir "${pass}" "${job}")"
    if (( ! DRY_RUN )) && job_complete "${pass}" "${job}"; then
      note "${pass}: seed ${seed} sites ${sites} is finished; skipping"
      continue
    fi
    log="${LOG_DIR}/${pass}_seed${seed}-targets${sites}.log"
    (( DRY_RUN )) || note "${pass}: start seed ${seed} sites ${sites} on ${device_label}"
    py "${log}" 1 "${devices[@]}" -- -m experiments.exp28_twins_method_comparison \
      "${COMMON_ARGS[@]}" "${extra_args[@]}" --data-root "${DATA_ROOT}" \
      --seeds "${seed}" --target-sites "${targets[@]}" --output-dir "${directory}" || true
    if (( DRY_RUN )); then
      continue
    elif job_complete "${pass}" "${job}"; then
      note "${pass}: done seed ${seed} sites ${sites}" \
        "($(grep -o '"n_fit_failures": [0-9]*' "${directory}/completion.json" | grep -o '[0-9]*$') failed fits, scored as Manski)"
    else
      warn "${pass}: seed ${seed} sites ${sites} failed; see ${log}"
      status=1
    fi
  done
  return "${status}"
}

# Run every job of a pass on parallel workers and wait for them.
#   run_pass PASS WORKERS JOB...
run_pass() {
  local pass="$1" workers="$2"
  shift 2
  local pids=() worker
  note "${pass}: $# job(s) on ${workers} parallel worker(s); logs ${LOG_DIR}/${pass}_*.log"
  for (( worker = 0; worker < workers; worker++ )); do
    if (( DRY_RUN )); then
      pass_worker "${pass}" "${worker}" "${workers}" "$@"
    else
      pass_worker "${pass}" "${worker}" "${workers}" "$@" &
      pids+=("$!")
    fi
  done
  wait_for "${pass}" "${pids[@]}"
  if (( ! DRY_RUN )); then
    require_pass "${pass}" "$@"
  fi
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

# Fail unless the Twins files in DATA_ROOT/twins are the paper's.
require_twins_data() {
  local entry name expected actual
  for entry in "${TWINS_FILES[@]}"; do
    name="${entry%%=*}"
    expected="${entry#*=}"
    [[ -f "${DATA_ROOT}/twins/${name}" ]] || die "No ${DATA_ROOT}/twins/${name}; run the data stage."
    if command -v sha256sum > /dev/null 2>&1; then
      actual="$(sha256sum "${DATA_ROOT}/twins/${name}" | cut -d' ' -f1)"
    else
      actual="$(shasum -a 256 "${DATA_ROOT}/twins/${name}" | cut -d' ' -f1)"
    fi
    [[ "${actual}" == "${expected}" ]] \
      || die "${DATA_ROOT}/twins/${name} differs from the file the paper used (sha256 ${actual})."
  done
}

# Check that the Python environment imports what the selected stages need.
preflight() {
  local imports=() log="${LOG_DIR}/preflight.log"
  if stage_selected naturalness || stage_selected comparators; then
    imports+=(experiments.exp28_twins_method_comparison)
  fi
  if stage_selected merge || stage_selected report; then
    imports+=(experiments.naturalness_comparison_results)
  fi
  if (( ${#imports[@]} > 0 )); then
    py "${log}" 1 "${NO_GPU[@]}" -- -c "import $(IFS=,; echo "${imports[*]}")" \
      || die "The Python environment cannot import what these stages need; see ${log}"
  fi
  if (( GPUS > 0 )) && stage_selected naturalness; then
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

# data: the three Twins CSVs, from --twins-dir or the CEVAE repository.
stage_data() {
  local entry name
  run mkdir -p "${DATA_ROOT}/twins"
  for entry in "${TWINS_FILES[@]}"; do
    name="${entry%%=*}"
    if [[ -f "${DATA_ROOT}/twins/${name}" ]]; then
      continue
    elif [[ -n "${TWINS_DIR}" ]]; then
      [[ -f "${TWINS_DIR}/${name}" ]] || (( DRY_RUN )) || die "No ${TWINS_DIR}/${name}"
      note "data: copying ${name} from ${TWINS_DIR}"
      run cp "${TWINS_DIR}/${name}" "${DATA_ROOT}/twins/${name}"
    else
      note "data: downloading ${name}"
      run curl -sSfL --retry 3 -o "${DATA_ROOT}/twins/${name}.part" "${TWINS_URL}/${name}" \
        || die "Could not download ${TWINS_URL}/${name}; pass --twins-dir with the three files."
      run mv "${DATA_ROOT}/twins/${name}.part" "${DATA_ROOT}/twins/${name}"
    fi
  done
  if (( ! DRY_RUN )); then
    require_twins_data
    note "data: ${DATA_ROOT}/twins holds the paper's files"
  fi
}

# naturalness: Naturalness, Manski and Balke-Pearl on every seed.
stage_naturalness() {
  (( DRY_RUN )) || require_twins_data
  run_pass naturalness "${NATURALNESS_JOBS}" "${JOB_LIST[@]}"
}

# comparators: B-Learner, CB-IPW, CB-DR and Manski on every seed.
stage_comparators() {
  (( DRY_RUN )) || require_twins_data
  ensure_blearner
  run_pass comparators "${COMPARATOR_JOBS}" "${JOB_LIST[@]}"
}

# merge: one tree over both passes, restricted to the consistent conditions.
stage_merge() {
  if [[ -f "${MERGED_DIR}/merge_provenance.json" ]]; then
    note "merge: ${MERGED_DIR} exists; skipping"
    return 0
  fi
  [[ ! -e "${MERGED_DIR}" ]] \
    || die "${MERGED_DIR} exists without merge_provenance.json, probably from an interrupted merge; remove it and rerun."
  if (( ! DRY_RUN )); then
    require_pass naturalness "${JOB_LIST[@]}"
    require_pass comparators "${JOB_LIST[@]}"
  fi
  note "merge: into ${MERGED_DIR}"
  py "${LOG_DIR}/merge.log" 1 "${NO_GPU[@]}" -- -m experiments.naturalness_comparison_results merge \
    "${OUTPUT_DIR}/naturalness" "${OUTPUT_DIR}/comparators" "${MERGE_ARGS[@]}" \
    --output-dir "${MERGED_DIR}" \
    || die "merge failed or dropped conditions; see ${LOG_DIR}/merge.log"
}

# report: every figure and table of the merged tree, then the paper's outputs.
stage_report() {
  local tree="${MERGED_DIR}" file
  if [[ -n "${TREE}" ]]; then
    # The report writes next to its inputs, so it runs on a copy of them.
    tree="${REPORT_DIR}/tree"
    run mkdir -p "${tree}"
    for file in metrics.parquet optimal_only_metrics.parquet merge_provenance.json; do
      if [[ -f "${TREE}/${file}" ]]; then
        run cp "${TREE}/${file}" "${tree}/${file}"
      fi
    done
  fi
  if (( ! DRY_RUN )); then
    [[ -f "${tree}/metrics.parquet" && -f "${tree}/merge_provenance.json" ]] \
      || die "report: no merged Exp28 tree at ${tree}; run the earlier stages or pass --tree."
    grep -q '"naturalness_trigger": "consistent"' "${tree}/merge_provenance.json" \
      || warn "report: ${tree} was not merged with --naturalness-trigger consistent, so its numbers are not the paper's"
  fi
  note "report: ${tree}"
  py "${LOG_DIR}/report.log" "${CPUS}" "${NO_GPU[@]}" -- -m experiments.naturalness_comparison_results \
    report "${tree}" || die "report failed; see ${LOG_DIR}/report.log"

  local entry extension
  run mkdir -p "${OUTPUT_DIR}/tables" "${OUTPUT_DIR}/figures"
  for entry in "${PAPER_TABLES[@]}"; do
    run cp "${tree}/${entry#*=}" "${OUTPUT_DIR}/tables/${entry%%=*}"
  done
  for entry in "${PAPER_FIGURES[@]}"; do
    for extension in pdf png; do
      run cp "${tree}/${entry#*=}.${extension}" "${OUTPUT_DIR}/figures/${entry%%=*}.${extension}"
    done
  done
  note "report: tables in ${OUTPUT_DIR}/tables, figures in ${OUTPUT_DIR}/figures"
}

# --- Command line --------------------------------------------------------------

OUTPUT_DIR="${PROJECT_ROOT}/results/paper/experiment2_twins"
STAGES=""
TREE=""
TWINS_DIR=""
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
    --output-dir|--stages|--tree|--twins-dir|--conda-env|--python|--gpus|--jobs-per-gpu|--cpus|--memory-gb)
      [[ $# -ge 2 ]] || die "$1 needs a value (see --help)"
      case "$1" in
        --output-dir) OUTPUT_DIR="$2" ;;
        --stages) STAGES="$2" ;;
        --tree) TREE="$2" ;;
        --twins-dir) TWINS_DIR="$2" ;;
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

# Parallel jobs per pass, within the CPU, GPU and memory budgets.
MEMORY_JOBS="$(at_least_one $(( MEMORY_GB / JOB_GB )))"
if (( GPUS > 0 )); then
  NATURALNESS_JOBS=$(( GPUS * JOBS_PER_GPU ))
else
  NATURALNESS_JOBS="$(at_least_one $(( CPUS / NATURALNESS_THREADS )))"
fi
NATURALNESS_JOBS="$(min_of "${NATURALNESS_JOBS}" "${MEMORY_JOBS}" "${#JOB_LIST[@]}")"
COMPARATOR_JOBS="$(min_of "$(at_least_one $(( CPUS / COMPARATOR_WORKERS )))" "${MEMORY_JOBS}" \
  "${#JOB_LIST[@]}")"

if [[ -n "${CONDA_ENV}" ]]; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PY=("${PYTHON_EXE}")
fi

# Paths. Python runs from the repository root, so every path is absolute.
OUTPUT_DIR="$(absolute "${OUTPUT_DIR}")"
[[ -z "${TWINS_DIR}" ]] || TWINS_DIR="$(absolute "${TWINS_DIR}")"
[[ -z "${TREE}" ]] || TREE="$(absolute "${TREE}")"
DATA_ROOT="${OUTPUT_DIR}/data"
MERGED_DIR="${OUTPUT_DIR}/merged_consistent"
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
Experiment II (Twins, Exp28) reproduction
  output     ${OUTPUT_DIR}
  stages     ${RUN_STAGES[*]}
  python     ${PY[*]}
  GPUs       ${gpu_text}
  CPUs       ${CPUS}
  memory     ${MEMORY_GB} GB
  parallel   naturalness ${NATURALNESS_JOBS} job(s), comparators ${COMPARATOR_JOBS} job(s)
EOF
if (( DRY_RUN )); then
  echo "  (dry run: the commands below are printed, not run)"
fi
if stage_selected naturalness; then
  if (( GPUS == 0 )); then
    warn "No GPU in use: TabICLv2 runs on the CPU, which is much slower and can change the last digits." \
      "Pass --gpus N if this machine has GPUs that were not detected."
  elif (( NATURALNESS_JOBS < GPUS * JOBS_PER_GPU && NATURALNESS_JOBS < ${#JOB_LIST[@]} )); then
    warn "The ${MEMORY_GB} GB budget allows ${NATURALNESS_JOBS} Naturalness job(s) at ${JOB_GB} GB each," \
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
