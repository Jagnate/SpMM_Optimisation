#!/usr/bin/env python3
"""
generate_case.py

Generate a reproducible sparse-dense matrix multiplication test case:

    A (sparse CSR) : [m x k]
    B (dense)      : [k x n]
    C_ref = A @ B  : [m x n]

Outputs (default):
  out_dir/
    A.npz           (scipy CSR)
    A_data.npy
    A_indices.npy
    A_indptr.npy
    B.npy
    C_ref.npy
    meta.json       (metadata)

Optional outputs (use flags):
  A_dense.npy         (dense representation of A; large memory!)
  B_fp16.npy          (B as float16)
  A_dense_fp16.npy    (A_dense as float16; only if A_dense is saved)

Usage examples:
  python generate_case.py --m 512 --k 512 --n 256 --density 0.1 --seed 42 --out case1
  python generate_case.py --m 4096 --k 4096 --n 1024 --density 0.05 --save-dense --save-fp16 --out case_big
"""

import os
import json
import argparse
import numpy as np
import scipy.sparse as sp


def generate_case(m, k, n, density, seed, out_dir, dtype=np.float32, save_dense=False, save_fp16=False):
    """
    Generate sparse matrix A (CSR), dense matrix B, and C_ref = A @ B.
    Optionally save dense A and FP16 variants.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Generate sparse matrix A in CSR format
    A = sp.random(
        m,
        k,
        density=density,
        data_rvs=lambda s: rng.standard_normal(s).astype(dtype),
        format="csr",
        dtype=dtype,
    )

    # 2) Generate dense matrix B
    B = rng.standard_normal((k, n)).astype(dtype)

    # 3) Compute reference result
    C_ref = (A @ B).astype(dtype)

    # 4) Save CSR matrix in both .npz and raw CSR arrays
    sp.save_npz(os.path.join(out_dir, "A.npz"), A)
    np.save(os.path.join(out_dir, "A_data.npy"), A.data)
    np.save(os.path.join(out_dir, "A_indices.npy"), A.indices)
    np.save(os.path.join(out_dir, "A_indptr.npy"), A.indptr)

    # 5) Save dense input and reference output
    np.save(os.path.join(out_dir, "B.npy"), B)
    np.save(os.path.join(out_dir, "C_ref.npy"), C_ref)

    # 6) Optionally save dense A (be careful: can be very large)
    if save_dense:
        # materialize dense A
        A_dense = A.toarray().astype(dtype)
        np.save(os.path.join(out_dir, "A_dense.npy"), A_dense)
        print(f"Saved dense A to {os.path.join(out_dir, 'A_dense.npy')} (size: {A_dense.nbytes} bytes)")
        if save_fp16:
            # save fp16 variant
            A_dense_fp16 = A_dense.astype(np.float16)
            np.save(os.path.join(out_dir, "A_dense_fp16.npy"), A_dense_fp16)
            print(f"Saved dense A FP16 to {os.path.join(out_dir, 'A_dense_fp16.npy')}")
    else:
        if save_fp16:
            # We can still save B_fp16 even if A_dense is not saved.
            pass

    # 7) Optionally save FP16 versions of B (and A_dense if saved)
    if save_fp16:
        B_fp16 = B.astype(np.float16)
        np.save(os.path.join(out_dir, "B_fp16.npy"), B_fp16)
        print(f"Saved B FP16 to {os.path.join(out_dir, 'B_fp16.npy')}")

    # 8) Save metadata for later automation
    meta = {
        "m": int(m),
        "k": int(k),
        "n": int(n),
        "density": float(density),
        "nnz": int(A.nnz),
        "dtype": str(dtype),
        "seed": int(seed),
        "files": os.listdir(out_dir),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    print("===== Test Case Generated =====")
    print(f"A shape: ({m}, {k})")
    print(f"A nnz: {A.nnz}")
    print(f"A density: {A.nnz / (m * k):.6f}")
    print(f"B shape: ({k}, {n})")
    print(f"C_ref shape: ({m}, {n})")
    print(f"Saved to directory: {os.path.abspath(out_dir)}")
    print(f"Files: {meta['files']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sparse-dense multiplication test case")
    parser.add_argument("--m", type=int, default=512, help="Number of rows of A")
    parser.add_argument("--k", type=int, default=512, help="Number of columns of A / rows of B")
    parser.add_argument("--n", type=int, default=512, help="Number of columns of B")
    parser.add_argument("--density", type=float, default=0.1, help="Density of sparse matrix A")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="case_out", help="Output directory")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="Data type to generate")
    parser.add_argument("--save-dense", action="store_true", help="Also save dense representation of A (A_dense.npy). WARNING: can be large")
    parser.add_argument("--save-fp16", action="store_true", help="Also save FP16 versions of B (and A_dense if saved)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float16
    generate_case(args.m, args.k, args.n, args.density, args.seed, args.out, dtype=dtype, save_dense=args.save_dense, save_fp16=args.save_fp16)