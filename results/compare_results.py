#!/usr/bin/env python3
"""
compare_results.py

Compare CUDA output matrix against reference C_ref.

Example:
  python compare_results.py \
      --ref ../case_out/C_ref.npy \
      --out C_out.npy \
      --tol_abs 1e-4 \
      --tol_rel 1e-3

The script prints error metrics and returns non-zero exit code if
the result exceeds tolerance (useful for CI).
"""

import argparse
import numpy as np
import sys
import os


def compare(ref_path, out_path, tol_abs, tol_rel):
    """
    Compare two matrices and compute:
      - Max absolute error
      - Mean absolute error
      - Relative Frobenius norm error
    """

    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Output file not found: {out_path}")

    C_ref = np.load(ref_path)
    C_out = np.load(out_path)

    if C_ref.shape != C_out.shape:
        raise ValueError(f"Shape mismatch: ref {C_ref.shape} vs out {C_out.shape}")

    diff = C_ref.astype(np.float64) - C_out.astype(np.float64)

    max_abs_error = float(np.max(np.abs(diff)))
    mean_abs_error = float(np.mean(np.abs(diff)))
    relative_error = float(
        np.linalg.norm(diff) / (np.linalg.norm(C_ref) + 1e-12)
    )

    print("===== Comparison Result =====")
    print(f"Matrix shape: {C_ref.shape}")
    print(f"Max absolute error: {max_abs_error:.6g}")
    print(f"Mean absolute error: {mean_abs_error:.6g}")
    print(f"Relative Frobenius error: {relative_error:.6g}")
    print(f"Absolute tolerance: {tol_abs}")
    print(f"Relative tolerance: {tol_rel}")

    passed = (max_abs_error <= tol_abs) or (relative_error <= tol_rel)

    if passed:
        print("PASS: Result is within tolerance.")
        return 0
    else:
        print("FAIL: Result exceeds tolerance.")
        return 1


def parse_args():
    parser = argparse.ArgumentParser(description="Compare result matrix with reference")
    parser.add_argument("--ref", required=True, help="Path to C_ref.npy")
    parser.add_argument("--out", required=True, help="Path to C_out.npy")
    parser.add_argument("--tol_abs", type=float, default=1e-4, help="Absolute error tolerance")
    parser.add_argument("--tol_rel", type=float, default=1e-3, help="Relative error tolerance")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = compare(args.ref, args.out, args.tol_abs, args.tol_rel)
    sys.exit(exit_code)