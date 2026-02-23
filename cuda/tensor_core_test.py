#!/usr/bin/env python3
"""
tensor_core_test.py

Prototype: perform sparse(A CSR) x dense(B) using tile-based packing
and FP16 matmul to leverage Tensor Cores via PyTorch.

Workflow:
  1. Load CSR A (A.npz or A_data/indices/indptr) and dense B.npy from input_dir
  2. Tile A into (tile x tile) blocks; for each block that contains any non-zero,
     convert the block to a dense FP16 tensor and multiply with the corresponding
     B block (FP16) on CUDA using torch.matmul.
  3. Accumulate results into a float32 output matrix C_out (to reduce accumulation error)
  4. Save results/C_out.npy

Important notes:
 - This script is a correctness-oriented prototype that also exercises Tensor Cores
   (if your GPU and PyTorch setup support FP16 Tensor Core math).
 - For real high-performance comparison use cuBLASLt/CUTLASS or a WMMA kernel and
   measure end-to-end timings carefully.
"""

import argparse
import os
import time
import numpy as np
import scipy.sparse as sp
import torch
import subprocess
from typing import Optional


def load_csr_from_dir(input_dir: str):
    """
    Load CSR either from A.npz or from A_data.npy/A_indices.npy/A_indptr.npy.
    Returns a scipy.sparse.csr_matrix.
    """
    npz_path = os.path.join(input_dir, "A.npz")
    data_path = os.path.join(input_dir, "A_data.npy")
    idx_path = os.path.join(input_dir, "A_indices.npy")
    indptr_path = os.path.join(input_dir, "A_indptr.npy")

    if os.path.exists(npz_path):
        A = sp.load_npz(npz_path).tocsr()
        return A
    elif os.path.exists(data_path) and os.path.exists(idx_path) and os.path.exists(indptr_path):
        data = np.load(data_path)
        indices = np.load(idx_path)
        indptr = np.load(indptr_path)
        # Note: we need to know shape; try to load B to infer k if necessary
        # Here we cannot always infer m,k; assume user used generate_case.py (so shapes consistent)
        # We will try to construct shape from indptr length.
        m = len(indptr) - 1
        # For columns (k) infer from max index + 1
        if indices.size > 0:
            k = int(indices.max()) + 1
        else:
            k = 0
        A = sp.csr_matrix((data, indices, indptr), shape=(m, k))
        return A
    else:
        raise FileNotFoundError(f"Could not find CSR files in {input_dir}. Expected A.npz or A_data/indices/indptr.")


def run_tile_tensorcore_spmm(
    A: sp.csr_matrix,
    B_np: np.ndarray,
    tile: int = 16,
    use_fp16: bool = True,
    verbose: bool = True,
    warmup: bool = False,
):
    """
    Tile-based sparse x dense multiply using FP16 matmuls on CUDA.

    Args:
      A: scipy csr matrix (m x k)
      B_np: numpy dense matrix (k x n)
      tile: tile size (e.g., 16 to map to Tensor Core favorable tiles)
      use_fp16: if True, convert tiles to float16 for matmul
      warmup: if True, perform one warmup matmul
    Returns:
      C_out as numpy float32 array (m x n), and elapsed compute time (seconds)
    """
    assert sp.isspmatrix_csr(A), "A must be CSR"
    m, k = A.shape
    k_b, n = B_np.shape
    assert k_b == k, f"Inner dims mismatch A:{A.shape} vs B:{B_np.shape}"

    device = torch.device("cuda")
    # Move B to GPU in chosen precision
    if use_fp16:
        B = torch.from_numpy(B_np).to(device=device, dtype=torch.float16)
    else:
        B = torch.from_numpy(B_np).to(device=device, dtype=torch.float32)

    # Output buffer on GPU (float32 accumulate)
    C_out = torch.zeros((m, n), device=device, dtype=torch.float32)

    # Optionally warm up a dummy operation to stabilize timing & Tensor Core usage
    if warmup:
        dummy = torch.randn((tile, tile), device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
        _ = torch.matmul(dummy, dummy)
        torch.cuda.synchronize()

    # iterate tile rows and tile cols of A
    t0 = time.time()
    # We'll iterate block rows of A, and block cols of A (over k)
    for br in range(0, m, tile):
        br_end = min(br + tile, m)
        # Pre-create a float32 slice view for accumulation (to minimize device<->host chatter)
        for bk in range(0, k, tile):
            bk_end = min(bk + tile, k)
            # Extract sub-block A_block (dense) from CSR (on CPU)
            A_block = A[br:br_end, bk:bk_end].toarray()  # small tile as ndarray
            if not np.any(A_block):
                # skip empty blocks
                continue
            # Convert to chosen precision and move to device
            if use_fp16:
                A_block_t = torch.from_numpy(A_block.astype(np.float16)).to(device=device, dtype=torch.float16)
            else:
                A_block_t = torch.from_numpy(A_block.astype(np.float32)).to(device=device, dtype=torch.float32)

            # B block: rows bk:bk_end, all columns
            B_block = B[bk:bk_end, :]  # shape (bk_len, n)

            # matmul: (br_len x bk_len) x (bk_len x n) => (br_len x n)
            # FP16 matmul will use Tensor Cores on supported hardware/config
            prod = torch.matmul(A_block_t, B_block)  # dtype=float16 if use_fp16 else float32

            # Accumulate (convert to float32)
            if prod.dtype == torch.float16:
                C_out[br:br_end, :] += prod.to(dtype=torch.float32)
            else:
                C_out[br:br_end, :] += prod

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Move result back to CPU as float32 numpy array
    C_out_np = C_out.cpu().numpy().astype(np.float32)
    return C_out_np, elapsed


def main():
    parser = argparse.ArgumentParser(description="Tile-based Tensor Core prototype for SpMM (uses PyTorch FP16 matmul).")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory produced by generate_case.py (contains A.npz or CSR arrays, and B.npy and C_ref.npy)")
    parser.add_argument("--out", type=str, default="results/C_out.npy", help="Path to save output numpy result")
    parser.add_argument("--tile", type=int, default=16, help="Tile size (e.g., 16)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 matmul (Tensor Core path)")
    parser.add_argument("--warmup", action="store_true", help="Run a small warmup op on GPU before timing")
    parser.add_argument("--compare", action="store_true", help="Run compare_results.py after computing")
    parser.add_argument("--compare_script", type=str, default="../results/compare_results.py", help="Path to compare_results.py")
    parser.add_argument("--tol_abs", type=float, default=1e-3, help="Absolute tolerance passed to compare script")
    parser.add_argument("--tol_rel", type=float, default=1e-2, help="Relative tolerance passed to compare script")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load data
    print("Loading CSR A and B...")
    A = load_csr_from_dir(args.input_dir)
    B_path = os.path.join(args.input_dir, "B.npy")
    Cref_path = os.path.join(args.input_dir, "C_ref.npy")
    if not os.path.exists(B_path):
        raise FileNotFoundError(f"B.npy not found in {args.input_dir}")
    B_np = np.load(B_path)

    if os.path.exists(Cref_path):
        print("Reference C_ref found; will optionally compare at the end.")
    else:
        print("Warning: C_ref.npy not found; skip automatic compare if requested.")

    print(f"A shape: {A.shape}, nnz={A.nnz}")
    print(f"B shape: {B_np.shape}")
    print(f"Tile size: {args.tile}, FP16: {args.fp16}")

    # Run tiled Tensor Core path
    print("Running tile-based spmm on GPU...")
    C_out_np, elapsed = run_tile_tensorcore_spmm(A, B_np, tile=args.tile, use_fp16=args.fp16, warmup=args.warmup)
    print(f"Done. Elapsed compute time: {elapsed:.6f} s")

    # Save result
    np.save(args.out, C_out_np)
    print(f"Saved output to {args.out}")

    # Optionally compare using your compare_results.py
    if args.compare and os.path.exists(Cref_path):
        cmd = [
            "python",
            args.compare_script,
            "--ref",
            Cref_path,
            "--out",
            args.out,
            "--tol_abs",
            str(args.tol_abs),
            "--tol_rel",
            str(args.tol_rel),
        ]
        print("Running compare script:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("Compare script failed with return code", e.returncode)
            # do not raise; we report failure above

if __name__ == "__main__":
    main()