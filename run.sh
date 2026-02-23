#!/bin/bash

# ================================
# Sparse + Tensor Core Experiment
# ================================

set -e

# ---- Configurable parameters ----
M=1024
K=1024
N=1024
DENSITY=0.05
TILE=16
DATA_DIR="data_case"
OUT_FILE="results/C_out.npy"
NCU_REPORT="results/ncu_report"

echo "=============================="
echo "Generating test case..."
echo "=============================="

python data_gen/generate_case.py \
    --m $M \
    --k $K \
    --n $N \
    --density $DENSITY \
    --out $DATA_DIR

echo ""
echo "=============================="
echo "Running Tensor Core test with Nsight Compute..."
echo "=============================="

ncu \
  --set full \
  --metrics sm__pipe_tensor_active.avg.pct_of_peak_sustained_active \
  --target-processes all \
  --export $NCU_REPORT \
  python cuda/tensor_core_test.py \
    --input_dir $DATA_DIR \
    --out $OUT_FILE \
    --tile $TILE \
    --fp16 \
    --warmup

echo ""
echo "=============================="
echo "Checking correctness..."
echo "=============================="

python results/compare_results.py \
    --ref $DATA_DIR/C_ref.npy \
    --out $OUT_FILE \
    --tol_abs 1e-2 \
    --tol_rel 1e-2

echo ""
echo "=============================="
echo "Experiment complete."
echo "Nsight report saved as: $NCU_REPORT.ncu-rep"
echo "=============================="