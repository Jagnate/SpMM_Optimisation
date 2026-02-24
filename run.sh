#!/usr/bin/env bash
# run.sh
#
# Run experiments (plain python or under ncu). You can choose to:
#   - only generate data (--generate)
#   - only run (--run)
#   - both (--generate --run)
# Choose profiler mode (--profile with ncu) or plain run (--no-profile).
# Choose mode: dense, sparse, both.
#
# Examples:
#   ./run.sh --generate --run                    # generate then run both modes (default)
#   ./run.sh --run --mode dense --profile        # run dense under ncu
#   MODE=sparse FP16=1 ./run.sh --run --no-profile
#   ./run.sh --generate --m 2048 --k 2048 --n 1024 --density 0.02

set -euo pipefail

print_help() {
  cat <<HEREDOC
run.sh - run Tensor Core experiments (dense & sparse) and optionally profile with ncu.

Options:
  --generate            generate data first (calls generate_data.sh)
  --run                 run experiment (default if neither generate/run provided => both)
  --profile / --no-profile   whether to run under ncu (default: profile if ncu exists)
  --mode MODE           dense | sparse | both (default: both)
  --repeat R            number of repeats for timing (default 3)
  --tile T              tile size for sparse path (default 16)
  --out BASE            base path for outputs (default results/C_out)
  --help

Environment overrides (optional):
  M, K, N, DENSITY, OUT_DIR, FP16 (0/1), WARMUP (0/1), TOL_ABS, TOL_REL

Examples:
  ./run.sh --generate --run
  ./run.sh --run --mode dense --profile
  MODE=dense FP16=1 ./run.sh --run --no-profile

HEREDOC
}

# defaults / env overrides
M=${M:-1024}
K=${K:-1024}
N=${N:-1024}
DENSITY=${DENSITY:-0.05}
OUT_DIR=${OUT_DIR:-"data_case"}     # where generate_data.sh will write
OUT_BASE=${OUT_BASE:-"results/C_out"} # base output path
REPEAT=${REPEAT:-3}
TILE=${TILE:-16}
MODE=${MODE:-"both"}    # dense|sparse|both
FP16=${FP16:-1}         # 1 use fp16 path
WARMUP=${WARMUP:-1}
TOL_ABS=${TOL_ABS:-1e-2}
TOL_REL=${TOL_REL:-1e-2}
RESULTS_DIR=${RESULTS_DIR:-"results"}

# parse CLI args
DO_GENERATE=0
DO_RUN=0
FORCE_PROFILE=""   # if set to "1" force profile; if "0" force no profile; if empty autodetect
while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate) DO_GENERATE=1; shift;;
    --run) DO_RUN=1; shift;;
    --profile) FORCE_PROFILE=1; shift;;
    --no-profile) FORCE_PROFILE=0; shift;;
    --mode) MODE="$2"; shift 2;;
    --repeat) REPEAT="$2"; shift 2;;
    --tile) TILE="$2"; shift 2;;
    --out) OUT_BASE="$2"; shift 2;;
    --m) M="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --density) DENSITY="$2"; shift 2;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1"; print_help; exit 1;;
  esac
done

# default behavior: if neither --generate nor --run specified, do both
if [[ "$DO_GENERATE" -eq 0 && "$DO_RUN" -eq 0 ]]; then
  DO_GENERATE=1
  DO_RUN=1
fi

mkdir -p "${RESULTS_DIR}"

# decide profiling
if [[ "${FORCE_PROFILE}" == "1" ]]; then
  USE_NCU=1
elif [[ "${FORCE_PROFILE}" == "0" ]]; then
  USE_NCU=0
else
  # autodetect ncu
  if command -v ncu >/dev/null 2>&1; then
    USE_NCU=1
  else
    USE_NCU=0
  fi
fi

echo "Config:"
echo "  M=${M} K=${K} N=${N} density=${DENSITY}"
echo "  mode=${MODE} repeat=${REPEAT} tile=${TILE} fp16=${FP16} warmup=${WARMUP}"
echo "  data_dir=${OUT_DIR} out_base=${OUT_BASE}"
echo "  use_ncu=${USE_NCU}"
echo ""

# Step A: generate
if [[ "$DO_GENERATE" -eq 1 ]]; then
  echo ">>> Generating data..."
  # forward env vars to generator, but also allow CLI override in generate_data.sh
  SAVE_DENSE=${SAVE_DENSE:-0}
  SAVE_FP16=${SAVE_FP16:-0}
  ./generate_data.sh --m "${M}" --k "${K}" --n "${N}" --density "${DENSITY}" --out "${OUT_DIR}"
fi

# helper to run one mode (dense/sparse)
run_mode() {
  local mode="$1"   # dense | sparse
  local timestamp
  timestamp="$(date +%Y%m%d-%H%M%S)"
  local ncu_report="${RESULTS_DIR}/ncu_${mode}_${timestamp}.ncu-rep"
  local ncu_prefix="${RESULTS_DIR}/ncu_${mode}_${timestamp}"
  local args=(--input_dir "${OUT_DIR}" --mode "${mode}" --repeat "${REPEAT}" --tile "${TILE}" --out "${OUT_BASE}" )
  if [[ "${FP16}" -eq 1 ]]; then args+=(--fp16); fi
  if [[ "${WARMUP}" -eq 1 ]]; then args+=(--warmup); fi
  args+=(--compare --tol_abs "${TOL_ABS}" --tol_rel "${TOL_REL}")

  echo ""
  echo ">>> Running mode=${mode} (fp16=${FP16}) repeat=${REPEAT} ..."

  if [[ "${USE_NCU}" -eq 1 ]]; then
    echo "    Running under ncu (collecting Tensor Core metric)..."
    # Collect only Tensor Core utilization metric to keep time reasonable
    ncu --target-processes all \
        --metrics sm__pipe_tensor_active.avg.pct_of_peak_sustained_active \
        --export "${ncu_prefix}" \
        python cuda/tensor_core_test.py "${args[@]}"
    echo "    ncu report: ${ncu_report}"
  else
    echo "    Running without profiler..."
    python cuda/tensor_core_test.py "${args[@]}"
  fi
}

# Step B: run
if [[ "$DO_RUN" -eq 1 ]]; then
  if [[ "${MODE}" == "dense" || "${MODE}" == "both" ]]; then
    run_mode "dense"
  fi
  if [[ "${MODE}" == "sparse" || "${MODE}" == "both" ]]; then
    run_mode "sparse"
  fi
fi

echo ""
echo "Done. Results under ${RESULTS_DIR} and outputs like ${OUT_BASE}.dense.npy / ${OUT_BASE}.sparse.npy"
if [[ "${USE_NCU}" -eq 1 ]]; then
  echo "Open ncu UI with: ncu-ui ${RESULTS_DIR}/ncu_*"
fi