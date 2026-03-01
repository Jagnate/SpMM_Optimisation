#!/usr/bin/env python3
"""
generate_case_bin.py

Generate reproducible test cases for dense & sparse matrix multiply,
and write outputs in FP32 row-major raw binary format so CUDA/C++ programs
can read them easily. Also write raw CSR arrays in binary:
  - A_data.bin    (float32)  length = nnz
  - A_indices.bin (int32)    length = nnz
  - A_indptr.bin  (int32)    length = m+1

And keep python-friendly copies (A_data.npy etc) for convenience.
"""

import os
import argparse
import json
import numpy as np
import scipy.sparse as sp


def write_raw_bin(path: str, arr: np.ndarray):
    """
    Write numpy array in raw binary (C-order) for easy C/C++ reading.
    """
    # ensure contiguous
    c_arr = np.ascontiguousarray(arr)
    with open(path, "wb") as f:
        f.write(c_arr.tobytes())


def generate_unstructured(m, k, n, density, seed):
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

    # 1) raw dense binaries
    write_raw_bin(os.path.join(out_dir, "A_dense.bin"), A_dense)
    write_raw_bin(os.path.join(out_dir, "B.bin"), B)
    write_raw_bin(os.path.join(out_dir, "C_ref.bin"), C_ref)

    # 2) numpy copies (python-friendly)
    np.save(os.path.join(out_dir, "A_dense.npy"), A_dense)
    np.save(os.path.join(out_dir, "B.npy"), B)
    np.save(os.path.join(out_dir, "C_ref.npy"), C_ref)

    # 3) CSR .npz and CSR raw binary arrays
    sp.save_npz(os.path.join(out_dir, "A.npz"), A_csr)
    np.save(os.path.join(out_dir, "A_data.npy"), A_csr.data)
    np.save(os.path.join(out_dir, "A_indices.npy"), A_csr.indices)
    np.save(os.path.join(out_dir, "A_indptr.npy"), A_csr.indptr)

    # write raw CSR binaries (A_data.bin float32, A_indices.bin int32, A_indptr.bin int32)
    write_raw_bin(os.path.join(out_dir, "A_data.bin"), A_csr.data.astype(np.float32))
    write_raw_bin(os.path.join(out_dir, "A_indices.bin"), A_csr.indices.astype(np.int32))
    write_raw_bin(os.path.join(out_dir, "A_indptr.bin"), A_csr.indptr.astype(np.int32))

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
    A_csr, A_dense, B, C_ref = generate_unstructured(args.m, args.k, args.n, args.density, args.seed)
    save_all(args.out, A_csr, A_dense, B, C_ref)
    print("Generated case:")
    print(f"  A shape: ({args.m}, {args.k}), nnz: {A_csr.nnz}, density actual: {A_csr.nnz / (args.m * args.k):.6f}")
    print(f"  B shape: ({args.k}, {args.n})")
    print(f"  Output written to directory: {os.path.abspath(args.out)}")
    print("Files written (examples):")
    print("  - A_dense.bin  (raw float32 row-major)")
    print("  - B.bin        (raw float32 row-major)")
    print("  - C_ref.bin    (raw float32 row-major)")
    print("  - A_data.bin   (float32 raw CSR values)")
    print("  - A_indices.bin (int32 raw CSR col indices)")
    print("  - A_indptr.bin  (int32 raw CSR row ptrs)")
    print("  - meta.json")

if __name__ == "__main__":
    main()