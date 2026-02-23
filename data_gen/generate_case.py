#!/usr/bin/env python3
"""
generate_case.py

Generate a reproducible sparse-dense matrix multiplication test case:

    A (sparse CSR) : [m x k]
    B (dense)      : [k x n]
    C_ref = A @ B  : [m x n]

Output directory structure (args.out):

out_dir/
  A.npz              (scipy CSR format)
  A_data.npy         (CSR values)
  A_indices.npy      (CSR column indices)
  A_indptr.npy       (CSR row pointers)
  B.npy              (dense matrix)
  C_ref.npy          (ground truth result)

Example:
  python generate_case.py --m 512 --k 512 --n 256 --density 0.1 --seed 42 --out case1
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp


def generate_case(m, k, n, density, seed, out_dir, dtype=np.float32):
    """
    Generate sparse matrix A, dense matrix B,
    and compute reference result C_ref = A @ B.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Generate sparse matrix A in CSR format
    A = sp.random(
        m,
        k,
        density=density,
        data_rvs=lambda s: rng.standard_normal(s).astype(dtype),
        format="csr",
        dtype=dtype,
    )

    # 2. Generate dense matrix B
    B = rng.standard_normal((k, n)).astype(dtype)

    # 3. Compute reference result
    C_ref = (A @ B).astype(dtype)

    # 4. Save CSR matrix in both .npz and raw CSR arrays
    sp.save_npz(os.path.join(out_dir, "A.npz"), A)
    np.save(os.path.join(out_dir, "A_data.npy"), A.data)
    np.save(os.path.join(out_dir, "A_indices.npy"), A.indices)
    np.save(os.path.join(out_dir, "A_indptr.npy"), A.indptr)

    # 5. Save dense input and reference output
    np.save(os.path.join(out_dir, "B.npy"), B)
    np.save(os.path.join(out_dir, "C_ref.npy"), C_ref)

    # Print summary
    print("===== Test Case Generated =====")
    print(f"A shape: ({m}, {k})")
    print(f"A nnz: {A.nnz}")
    print(f"A density: {A.nnz / (m * k):.6f}")
    print(f"B shape: ({k}, {n})")
    print(f"C_ref shape: ({m}, {n})")
    print(f"Saved to directory: {os.path.abspath(out_dir)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sparse-dense multiplication test case")
    parser.add_argument("--m", type=int, default=512, help="Number of rows of A")
    parser.add_argument("--k", type=int, default=512, help="Number of columns of A / rows of B")
    parser.add_argument("--n", type=int, default=512, help="Number of columns of B")
    parser.add_argument("--density", type=float, default=0.1, help="Density of sparse matrix A")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="case_out", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_case(args.m, args.k, args.n, args.density, args.seed, args.out)