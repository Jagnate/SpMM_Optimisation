#!/usr/bin/env python3
"""
generate_case_bin.py

Generate reproducible test cases for dense & sparse matrix multiply,
and write outputs in FP32 row-major raw binary format so CUDA/C++ programs
can read them easily.

Outputs (out_dir/):
  - A_dense.bin       # float32, row-major, shape (m,k)
  - B.bin             # float32, row-major, shape (k,n)
  - C_ref.bin         # float32, row-major, shape (m,n)
  - A.npz             # scipy CSR saved (for Python)
  - A_data.npy, A_indices.npy, A_indptr.npy
  - A_dense.npy, B.npy, C_ref.npy
  - meta.json         # JSON with shapes, dtype, nnz

Usage example:
  python data_gen/generate_case_bin.py --m 1024 --k 1024 --n 1024 --density 0.05 --seed 42 --out case_out

Notes:
  - All numeric outputs use float32.
  - Raw binary files are written in C-order (row-major), packed float32 values.
  - C/C++ code can read them with fread into float buffer and then reshape.
"""

import os
import argparse
import json
import numpy as np
import scipy.sparse as sp


def write_raw_bin(path: str, arr: np.ndarray):
    """
    Write numpy array in row-major float32 raw binary for easy C/C++ reading.
    """
    assert arr.dtype == np.float32, "Array must be float32"
    # ensure C-order
    c_arr = np.ascontiguousarray(arr)
    with open(path, "wb") as f:
        f.write(c_arr.tobytes())


def generate_unstructured(m, k, n, density, seed, out_dir):
    rng = np.random.default_rng(seed)
    # generate sparse CSR A with float32 values
    A = sp.random(
        m,
        k,
        density=density,
        data_rvs=lambda s: rng.standard_normal(s).astype(np.float32),
        format="csr",
        dtype=np.float32,
    )
    # dense copy of A (row-major)
    A_dense = A.toarray().astype(np.float32)
    # generate dense B
    B = rng.standard_normal((k, n)).astype(np.float32)
    # compute reference
    C_ref = (A_dense @ B).astype(np.float32)
    return A, A_dense, B, C_ref


def save_all(out_dir, A_csr, A_dense, B, C_ref):
    os.makedirs(out_dir, exist_ok=True)
    # 1) save raw binaries for C/C++ consumption
    write_raw_bin(os.path.join(out_dir, "A_dense.bin"), A_dense)
    write_raw_bin(os.path.join(out_dir, "B.bin"), B)
    write_raw_bin(os.path.join(out_dir, "C_ref.bin"), C_ref)

    # 2) save numpy arrays (python-friendly)
    np.save(os.path.join(out_dir, "A_dense.npy"), A_dense)
    np.save(os.path.join(out_dir, "B.npy"), B)
    np.save(os.path.join(out_dir, "C_ref.npy"), C_ref)

    # 3) save CSR arrays and .npz
    sp.save_npz(os.path.join(out_dir, "A.npz"), A_csr)
    np.save(os.path.join(out_dir, "A_data.npy"), A_csr.data)
    np.save(os.path.join(out_dir, "A_indices.npy"), A_csr.indices)
    np.save(os.path.join(out_dir, "A_indptr.npy"), A_csr.indptr)

    # 4) save meta info
    meta = {
        "m": int(A_dense.shape[0]),
        "k": int(A_dense.shape[1]),
        "n": int(B.shape[1]),
        "dtype": "float32",
        "A_nnz": int(A_csr.nnz)
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Generate test case and write binary outputs (FP32 row-major).")
    p.add_argument("--m", type=int, default=512, help="Number of rows of A")
    p.add_argument("--k", type=int, default=512, help="Number of columns of A / rows of B")
    p.add_argument("--n", type=int, default=512, help="Number of columns of B")
    p.add_argument("--density", type=float, default=0.1, help="Density for sparse A (0..1).")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out", type=str, default="case_out", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    A_csr, A_dense, B, C_ref = generate_unstructured(args.m, args.k, args.n, args.density, args.seed, args.out)
    save_all(args.out, A_csr, A_dense, B, C_ref)
    print("Generated case:")
    print(f"  A shape: ({args.m}, {args.k}), nnz: {A_csr.nnz}, density actual: {A_csr.nnz / (args.m * args.k):.6f}")
    print(f"  B shape: ({args.k}, {args.n})")
    print(f"  Output written to directory: {os.path.abspath(args.out)}")
    print("Files written (examples):")
    print("  - A_dense.bin  (raw float32 row-major)")
    print("  - B.bin        (raw float32 row-major)")
    print("  - C_ref.bin    (raw float32 row-major)")
    print("  - A.npz, A_data.npy, A_indices.npy, A_indptr.npy")
    print("  - meta.json")

if __name__ == "__main__":
    main()