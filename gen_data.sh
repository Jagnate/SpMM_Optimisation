#!/usr/bin/env bash
# generate_data.sh
#
# Generate test data using data_gen/generate_case.py
#
# Usage examples:
#   ./generate_data.sh                      # use defaults
#   ./generate_data.sh --m 2048 --k 2048 --n 1024 --density 0.05 --out my_case
#   M=4096 K=4096 N=1024 DENSITY=0.01 ./generate_data.sh

set -euo pipefail

# defaults (can be overridden by env or CLI args)
M=${M:-1024}
K=${K:-1024}
N=${N:-1024}
DENSITY=${DENSITY:-0.05}
SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-"data_case"}
SAVE_DENSE=${SAVE_DENSE:-0}   # 1 to also save A_dense.npy (careful: large!)
SAVE_FP16=${SAVE_FP16:-0}     # 1 to also save FP16 versions

print_help() {
  cat <<HEREDOC
generate_data.sh - wrapper to call data_gen/generate_case.py

Environment-overridable defaults:
  M K N          matrix dims (default 1024)
  DENSITY        density of sparse A (default 0.05)
  SEED           RNG seed (default 42)
  OUT_DIR        output directory (default data_case)
  SAVE_DENSE     1 to also save dense A (A_dense.npy) (default 0)
  SAVE_FP16      1 to also save FP16 versions (default 0)

CLI usage:
  ./generate_data.sh [--m M] [--k K] [--n N] [--density D] [--out PATH] [--save-dense] [--save-fp16]
HEREDOC
}

# parse simple args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --m) M="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --density) DENSITY="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --out|--out-dir) OUT_DIR="$2"; shift 2;;
    --save-dense) SAVE_DENSE=1; shift 1;;
    --save-fp16) SAVE_FP16=1; shift 1;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown arg: $1"; print_help; exit 1;;
  esac
done

echo "Generating data with:"
echo "  M=$M K=$K N=$N density=$DENSITY seed=$SEED out=$OUT_DIR save_dense=$SAVE_DENSE save_fp16=$SAVE_FP16"
echo ""

cmd=(python data_gen/generate_case.py \
     --m "$M" --k "$K" --n "$N" --density "$DENSITY" --seed "$SEED" --out "$OUT_DIR")

if [[ "$SAVE_DENSE" == "1" ]]; then
  cmd+=(--save-dense)
fi
if [[ "$SAVE_FP16" == "1" ]]; then
  cmd+=(--save-fp16)
fi

# run
"${cmd[@]}"

echo "Data generated in: ${OUT_DIR}"