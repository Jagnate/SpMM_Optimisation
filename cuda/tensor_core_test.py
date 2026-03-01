#!/usr/bin/env python3
"""
tensor_core_test.py

Extended prototype to exercise Tensor Cores (via PyTorch) for both:

  - dense path:    A_dense @ B  (single large GEMM)
  - sparse path:   tile-packed sparse (CSR) -> many small dense tiles -> batched matmuls

The script can run "dense", "sparse", or "both" and compares numeric results
against data_dir/C_ref.npy when present.

Notes:
 - The dense path materializes A as dense (A.toarray()) and performs a single matmul
   on the GPU (can use FP16 Tensor Core path if requested).
 - The sparse path reuses your tile-based packing approach (materialize sparse tiles
   on CPU then run many small matmuls on GPU). This is a correctness + prototype
   builder; it is not an optimized production implementation.
 - Timings are for end-to-end execution of each path (including CPU tile formation
   for the sparse path). You can increase --repeat to amortize overhead and get
   more stable timings.
"""

import argparse
import os
import time
import json
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp
import torch
import subprocess


def load_csr_from_dir(input_dir: str) -> sp.csr_matrix:
    """Load CSR either from A.npz or A_data/indices/indptr arrays."""
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
        m = len(indptr) - 1
        k = int(indices.max()) + 1 if indices.size > 0 else 0
        A = sp.csr_matrix((data, indices, indptr), shape=(m, k))
        return A
    else:
        raise FileNotFoundError(f"Could not find CSR files in {input_dir}. Expected A.npz or A_data/indices/indptr.")


def compute_error_metrics(C_ref: np.ndarray, C_out: np.ndarray) -> Tuple[float, float, float]:
    """Compute max abs, mean abs, relative Frobenius error."""
    diff = C_ref.astype(np.float64) - C_out.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(C_ref) + 1e-12))
    return max_abs, mean_abs, rel


def run_dense_tensorcore_gemm(
    A_dense: np.ndarray,
    B: np.ndarray,
    use_fp16: bool = True,
    warmup: bool = False,
    repeat: int = 3,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Dense path: do a single large GEMM A_dense @ B using PyTorch (FP16 or FP32).
    Returns (C_out_cpu_float32, avg_elapsed_seconds)
    """
    device = torch.device("cuda")
    # Convert numpy -> torch and move to device in chosen precision
    if use_fp16:
        A_t = torch.from_numpy(A_dense.astype(np.float16)).to(device=device, dtype=torch.float16)
        B_t = torch.from_numpy(B.astype(np.float16)).to(device=device, dtype=torch.float16)
    else:
        A_t = torch.from_numpy(A_dense.astype(np.float32)).to(device=device, dtype=torch.float32)
        B_t = torch.from_numpy(B.astype(np.float32)).to(device=device, dtype=torch.float32)

    # output accumulation in float32 for better numeric stability
    if use_fp16:
        # do matmul then cast to float32
        pass

    # optional warmup
    if warmup:
        # small warmup matmul
        dummy = torch.randn((16, 16), device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
        _ = torch.matmul(dummy, dummy)
        torch.cuda.synchronize()

    # repeat timed runs
    times = []
    C_out = None
    for i in range(repeat):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # perform matmul
        prod = torch.matmul(A_t, B_t)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)  # milliseconds
        times.append(elapsed_ms / 1000.0)
        if verbose:
            print(f"[dense] iter {i}: {elapsed_ms:.3f} ms")

        # accumulate result to float32 on CPU after last iter
        if i == repeat - 1:
            # cast to float32 for output
            C_out = prod.to(dtype=torch.float32).cpu().numpy()

    avg_time = float(np.mean(times))
    return C_out, avg_time


from typing import Tuple

def run_tile_tensorcore_spmm(
    A: sp.csr_matrix,
    B_np: np.ndarray,
    tile: int = 16,
    use_fp16: bool = True,
    warmup: bool = False,
    repeat: int = 3,
    verbose: bool = True,
    batch_size: int = 256,
) -> Tuple[np.ndarray, float]:
    """
    Batched tile-based sparse x dense multiply using FP16 matmuls on CUDA.

    This version collects many tiles into a batch (pads to 'tile' size),
    stacks them into contiguous host arrays, transfers the batch to GPU once,
    performs a single batched matmul (torch.bmm) and scatters results back.
    It prints separate timings for packing (host work + H2D) and GPU compute.

    Returns (C_out_cpu_float32, avg_elapsed_seconds) where elapsed includes
    packing + GPU compute for each repeat.
    """
    assert sp.isspmatrix_csr(A), "A must be CSR"
    m, k = A.shape
    k_b, n = B_np.shape
    assert k_b == k, f"Inner dims mismatch A:{A.shape} vs B:{B_np.shape}"

    device = torch.device("cuda")

    # prepare B as numpy in target dtype for stacking (we will convert to torch later)
    dtype_np = np.float16 if use_fp16 else np.float32
    B_cast = B_np.astype(dtype_np, copy=False)

    # warmup single small op to stabilize kernels / Tensor Core usage
    if warmup:
        dummy = torch.randn((tile, tile), device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
        _ = torch.matmul(dummy, dummy)
        torch.cuda.synchronize()

    times = []
    last_C = None

    # Precompute tile counts
    n_tile_rows = (m + tile - 1) // tile
    n_tile_cols = (k + tile - 1) // tile

    for r in range(repeat):
        # output buffer on GPU (float32 accumulate)
        C_out = torch.zeros((m, n), device=device, dtype=torch.float32)

        # batching buffers on host
        A_tiles = []  # list of numpy arrays shape (tile, tile) dtype_np
        B_tiles = []  # list of numpy arrays shape (tile, n) dtype_np
        coords = []   # list of tuples (br, br_end, m_t)

        pack_time = 0.0
        gpu_time = 0.0

        t0_run = time.time()

        # scan tiles and collect non-empty ones into batches
        for br in range(0, m, tile):
            br_end = min(br + tile, m)
            m_t = br_end - br
            for bk in range(0, k, tile):
                bk_end = min(bk + tile, k)
                k_t = bk_end - bk

                # pack on host (timed)
                t_pack0 = time.time()
                A_block = A[br:br_end, bk:bk_end].toarray()  # small dense block on CPU
                t_pack1 = time.time()
                pack_time += (t_pack1 - t_pack0)

                if not np.any(A_block):
                    continue

                # pad to full tile so stacking is easy; store original m_t to scatter later
                if m_t != tile or k_t != tile:
                    a_p = np.zeros((tile, tile), dtype=dtype_np)
                    a_p[:m_t, :k_t] = A_block.astype(dtype_np, copy=False)
                    b_p = np.zeros((tile, n), dtype=dtype_np)
                    b_p[:k_t, :] = B_cast[bk:bk_end, :]
                    A_tiles.append(a_p)
                    B_tiles.append(b_p)
                else:
                    A_tiles.append(A_block.astype(dtype_np, copy=False))
                    B_tiles.append(B_cast[bk:bk_end, :])

                coords.append((br, br_end, m_t))

                # if batch full -> flush
                if len(A_tiles) >= batch_size:
                    # pack batch to device (time counted as part of pack_time)
                    t_h2d0 = time.time()
                    A_batch_np = np.stack(A_tiles)       # shape (B, tile, tile)
                    B_batch_np = np.stack(B_tiles)       # shape (B, tile, n)
                    # convert to torch and move in one shot
                    A_batch = torch.from_numpy(A_batch_np).to(device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
                    B_batch = torch.from_numpy(B_batch_np).to(device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
                    torch.cuda.synchronize()
                    t_h2d1 = time.time()
                    pack_time += (t_h2d1 - t_h2d0)

                    # GPU batched matmul (time counted separately)
                    t_gpu0 = time.time()
                    C_batch = torch.bmm(A_batch, B_batch)  # (B, tile, n)
                    torch.cuda.synchronize()
                    t_gpu1 = time.time()
                    gpu_time += (t_gpu1 - t_gpu0)

                    # scatter results back to C_out
                    for i, (br0, br1, mm) in enumerate(coords):
                        C_out[br0:br1, :] += C_batch[i, :mm, :].to(dtype=torch.float32)

                    # reset batch buffers
                    A_tiles, B_tiles, coords = [], [], []

        # flush remaining
        if len(A_tiles) > 0:
            t_h2d0 = time.time()
            A_batch_np = np.stack(A_tiles)
            B_batch_np = np.stack(B_tiles)
            A_batch = torch.from_numpy(A_batch_np).to(device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
            B_batch = torch.from_numpy(B_batch_np).to(device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
            torch.cuda.synchronize()
            t_h2d1 = time.time()
            pack_time += (t_h2d1 - t_h2d0)

            t_gpu0 = time.time()
            C_batch = torch.bmm(A_batch, B_batch)
            torch.cuda.synchronize()
            t_gpu1 = time.time()
            gpu_time += (t_gpu1 - t_gpu0)

            for i, (br0, br1, mm) in enumerate(coords):
                C_out[br0:br1, :] += C_batch[i, :mm, :].to(dtype=torch.float32)

        elapsed = time.time() - t0_run
        times.append(elapsed)
        last_C = C_out.cpu().numpy().astype(np.float32)

        if verbose:
            print(f"[sparse] iter {r}: total_elapsed={elapsed:.6f}s pack_time={pack_time:.6f}s gpu_time={gpu_time:.6f}s "
                  f"tiles_computed={(len(A_tiles) if len(A_tiles)>0 else 'flushed')}")
    avg_time = float(np.mean(times))
    return last_C, avg_time


def flop_count_dense(m: int, k: int, n: int) -> float:
    """Return number of floating-point operations for dense GEMM (2 * m * n * k)."""
    return 2.0 * float(m) * float(n) * float(k)


def flop_count_sparse(nnz: int, n: int) -> float:
    """Return number of floating-point operations for sparse-dense multiply: 2 * nnz * n."""
    return 2.0 * float(nnz) * float(n)


def print_error_and_pass(C_ref: Optional[np.ndarray], C_out: np.ndarray, tol_abs: float, tol_rel: float, label: str):
    if C_ref is None:
        print(f"[{label}] No reference provided, skipping numeric comparison.")
        return
    max_abs, mean_abs, rel = compute_error_metrics(C_ref, C_out)
    print(f"[{label}] Max abs error: {max_abs:.6g}")
    print(f"[{label}] Mean abs error: {mean_abs:.6g}")
    print(f"[{label}] Relative Frobenius error: {rel:.6g}")
    passed = (max_abs <= tol_abs) or (rel <= tol_rel)
    print(f"[{label}] PASS = {passed} (abs_tol={tol_abs}, rel_tol={tol_rel})")


def parse_args():
    p = argparse.ArgumentParser(description="Tile-based Tensor Core prototype (dense/sparse) using PyTorch.")
    p.add_argument("--input_dir", type=str, required=True, help="Directory produced by generate_case.py")
    p.add_argument("--out", type=str, default="results/C_out.npy", help="Base path to save outputs (will append .dense.npy/.sparse.npy)")
    p.add_argument("--tile", type=int, default=16, help="Tile size (e.g., 16)")
    p.add_argument("--fp16", action="store_true", help="Use FP16 matmul (Tensor Core path)")
    p.add_argument("--warmup", action="store_true", help="Run a small warmup op on GPU before timing")
    p.add_argument("--compare", action="store_true", help="Compare outputs to reference C_ref.npy if present")
    p.add_argument("--tol_abs", type=float, default=1e-3, help="Absolute tolerance for compare")
    p.add_argument("--tol_rel", type=float, default=1e-2, help="Relative tolerance for compare")
    p.add_argument("--mode", choices=["sparse", "dense", "both"], default="both", help="Which path(s) to run")
    p.add_argument("--repeat", type=int, default=3, help="Number of repeated runs for averaging")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for stacking tiles")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("Loading data...")
    A = load_csr_from_dir(args.input_dir)
    B_path = os.path.join(args.input_dir, "B.npy")
    Cref_path = os.path.join(args.input_dir, "C_ref.npy")
    if not os.path.exists(B_path):
        raise FileNotFoundError(f"B.npy not found in {args.input_dir}")
    B = np.load(B_path)
    C_ref = np.load(Cref_path) if os.path.exists(Cref_path) else None

    print(f"A shape: {A.shape}, nnz={A.nnz}")
    print(f"B shape: {B.shape}")
    print(f"Mode: {args.mode}, tile={args.tile}, fp16={args.fp16}, repeat={args.repeat}")

    results = {}

    # DENSE path (materialize A as dense)
    if args.mode in ("dense", "both"):
        print("Running dense path (A_dense @ B) ...")
        # materialize A dense (beware memory)
        A_dense = A.toarray()
        C_dense, t_dense = run_dense_tensorcore_gemm(
            A_dense, B, use_fp16=args.fp16, warmup=args.warmup, repeat=args.repeat, verbose=True
        )
        flops = flop_count_dense(A_dense.shape[0], A_dense.shape[1], B.shape[1])
        gflops = flops / (t_dense * 1e9) if t_dense > 0 else float("nan")
        print(f"[dense] avg time: {t_dense:.6f} s, GFLOPS: {gflops:.3f}")
        out_dense_path = args.out + ".dense.npy"
        np.save(out_dense_path, C_dense)
        print(f"[dense] saved: {out_dense_path}")
        results["dense"] = {"time": t_dense, "gflops": gflops, "out": out_dense_path}
        # compare if requested
        if args.compare and C_ref is not None:
            print("Comparing dense result to reference...")
            print_error_and_pass(C_ref, C_dense, args.tol_abs, args.tol_rel, label="dense")

    # SPARSE path
    if args.mode in ("sparse", "both"):
        print("Running sparse path (tile-packed) ...")
        C_sparse, t_sparse = run_tile_tensorcore_spmm(
            A, B, tile=args.tile, use_fp16=args.fp16, warmup=args.warmup, repeat=args.repeat, verbose=True, batch_size=args.batch_size
        )
        flops_s = flop_count_sparse(int(A.nnz), B.shape[1])
        gflops_s = flops_s / (t_sparse * 1e9) if t_sparse > 0 else float("nan")
        print(f"[sparse] avg time: {t_sparse:.6f} s, effective GFLOPS (counting 2*nnz*n): {gflops_s:.3f}")
        out_sparse_path = args.out + ".sparse.npy"
        np.save(out_sparse_path, C_sparse)
        print(f"[sparse] saved: {out_sparse_path}")
        results["sparse"] = {"time": t_sparse, "gflops": gflops_s, "out": out_sparse_path}
        if args.compare and C_ref is not None:
            print("Comparing sparse result to reference...")
            print_error_and_pass(C_ref, C_sparse, args.tol_abs, args.tol_rel, label="sparse")

    # Summarize
    print("\n=== SUMMARY ===")
    if "dense" in results:
        print(f"Dense: time={results['dense']['time']:.6f}s, GFLOPS={results['dense']['gflops']:.3f}, out={results['dense']['out']}")
    if "sparse" in results:
        print(f"Sparse: time={results['sparse']['time']:.6f}s, GFLOPS={results['sparse']['gflops']:.3f}, out={results['sparse']['out']}")

    # Save summary json
    summary = {
        "input_dir": args.input_dir,
        "mode": args.mode,
        "tile": args.tile,
        "fp16": bool(args.fp16),
        "repeat": int(args.repeat),
        "results": results,
    }
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {args.out}.summary.json")


if __name__ == "__main__":
    main()