#!/usr/bin/env python3
"""
compare_results_bin.py

Compare two raw float32 row-major binary matrices:
  ref_file (C_ref.bin) and out_file (C_out.bin).

Usage:
  python compare_results_bin.py --ref case_out/C_ref.bin --out results/C_out.bin --m 1024 --n 1024 --tol_abs 1e-4 --tol_rel 1e-3
Returns exit code 0 if passed, 1 if failed.
"""
import argparse
import numpy as np
import os
import sys

def load_raw_f32(path, shape):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    expected = shape[0] * shape[1]
    data = np.fromfile(path, dtype=np.float32, count=expected)
    if data.size != expected:
        raise ValueError(f"File {path} contains {data.size} elements, expected {expected}")
    return data.reshape(shape)

def compare(C_ref, C_out, tol_abs, tol_rel):
    diff = C_ref.astype(np.float64) - C_out.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(C_ref) + 1e-12))
    print("===== Comparison =====")
    print(f"Shape: {C_ref.shape}")
    print(f"Max abs error: {max_abs:.6g}")
    print(f"Mean abs error: {mean_abs:.6g}")
    print(f"Relative Frobenius error: {rel:.6g}")
    print(f"Abs tol: {tol_abs}, Rel tol: {tol_rel}")
    passed = (max_abs <= tol_abs) or (rel <= tol_rel)
    if passed:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="Reference C_ref.bin (raw float32 row-major)")
    p.add_argument("--out", required=True, help="Output C_out.bin (raw float32 row-major)")
    p.add_argument("--m", type=int, required=True, help="Rows (M)")
    p.add_argument("--n", type=int, required=True, help="Cols (N)")
    p.add_argument("--tol_abs", type=float, default=1e-4)
    p.add_argument("--tol_rel", type=float, default=1e-3)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    shape = (args.m, args.n)
    C_ref = load_raw_f32(args.ref, shape)
    C_out = load_raw_f32(args.out, shape)
    rc = compare(C_ref, C_out, args.tol_abs, args.tol_rel)
    sys.exit(0 if rc==0 else 1)