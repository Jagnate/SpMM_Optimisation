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

from concurrent.futures import ThreadPoolExecutor, as_completed

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




def run_tile_tensorcore_spmm(
    A: sp.csr_matrix,
    B_np: np.ndarray,
    tile: int = 16,
    use_fp16: bool = True,
    warmup: bool = False,
    repeat: int = 1,
    verbose: bool = True,
    batch_size: int = 256,
    dense_baseline_time: Optional[float] = None,
    packing_threads: int = None,
) -> Tuple[np.ndarray, float]:
    """
    Batched tile-based sparse x dense multiply with parallel host packing and
    an automatic decision to fall back to dense GEMM when sparse-case isn't beneficial.

    Args:
      A: scipy csr matrix (m x k)
      B_np: numpy dense matrix (k x n)
      tile: tile size
      use_fp16: use FP16 multiply on device
      warmup: run a small warmup
      repeat: number of repeats to average
      verbose: print per-iteration stats
      batch_size: how many tiles to stack per device batch
      dense_fallback_threshold: multiplier; if estimated sparse_total_time >=
        dense_baseline_time * dense_fallback_threshold, we run dense GEMM instead.
      dense_baseline_time: optionally supply a measured dense FFT baseline (seconds).
      packing_threads: number of host threads for packing (default: min(32, cpu_count)).
    Returns:
      (C_out_cpu_float32, avg_elapsed_seconds)
    """
    import multiprocessing
    if packing_threads is None:
        packing_threads = min(32, max(1, multiprocessing.cpu_count()))

    assert sp.isspmatrix_csr(A), "A must be CSR"
    m, k = A.shape
    k_b, n = B_np.shape
    assert k_b == k, f"Inner dims mismatch A:{A.shape} vs B:{B_np.shape}"

    device = torch.device("cuda")

    dtype_np = np.float16 if use_fp16 else np.float32

    # Convert B once to chosen dtype for packing uses (keeps host copy)
    B_cast = B_np.astype(dtype_np, copy=False)

    # Optional warmup
    if warmup:
        dummy = torch.randn((tile, tile), device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
        _ = torch.matmul(dummy, dummy)
        torch.cuda.synchronize()

    # Quick scan: list coords of non-empty tiles (fast using .nnz on submatrix)
    tile_coords = []
    for br in range(0, m, tile):
        br_end = min(br + tile, m)
        for bk in range(0, k, tile):
            bk_end = min(bk + tile, k)
            sub = A[br:br_end, bk:bk_end]
            if sub.nnz != 0:
                tile_coords.append((br, br_end, bk, bk_end))
    num_nonempty = len(tile_coords)
    total_tiles = ((m + tile - 1) // tile) * ((k + tile - 1) // tile)
    if verbose:
        print(f"Tile grid: {total_tiles} tiles, non-empty tiles: {num_nonempty}")

    # If no non-empty tiles -> return zero matrix quickly
    if num_nonempty == 0:
        if verbose:
            print("No non-empty tiles found. Returning zero matrix.")
        return np.zeros((m, n), dtype=np.float32), 0.0

    # If dense baseline time not given, measure a single-shot dense FP16 matmul time (fast)
    if dense_baseline_time is None:
        # quick measurement (not averaged heavily, just an estimate)
        A_dense = A.toarray().astype(np.float32)
        # Do a single FP16 matmul timed with CUDA events
        A_t = torch.from_numpy(A_dense.astype(np.float16)).to(device=device, dtype=torch.float16)
        B_t = torch.from_numpy(B_np.astype(np.float16)).to(device=device, dtype=torch.float16)
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        tmp = torch.matmul(A_t, B_t)
        end_ev.record()
        torch.cuda.synchronize()
        dense_baseline_time = start_ev.elapsed_time(end_ev) * 1e-3  # seconds
        if verbose:
            print(f"[auto] measured dense FP16 matmul gpu_time ~ {dense_baseline_time:.6f}s (single run)")

    # Estimate cost per tile for host packing by sampling up to S tiles
    sample_S = min(8, max(1, num_nonempty))
    sample_coords = tile_coords[:sample_S]
    t_pack_sample0 = time.time()
    # perform small sample extraction serially (cheap) to measure average host cost
    for (br, br_end, bk, bk_end) in sample_coords:
        _ = A[br:br_end, bk:bk_end].toarray()
    t_pack_sample1 = time.time()
    avg_pack_cost_sample = (t_pack_sample1 - t_pack_sample0) / sample_S
    # estimate total pack cost (rough)
    est_total_pack = avg_pack_cost_sample * num_nonempty
    # estimate gpu compute cost: assume each tile does a small matmul; we use dense_baseline_time scaled:
    est_gpu_per_tile = max(1e-6, dense_baseline_time * ( (tile*tile* n) / (A.shape[0]*A.shape[1]*n) ))  # heuristic
    est_total_gpu = est_gpu_per_tile * num_nonempty
    est_total_sparse = est_total_pack + est_total_gpu
    if verbose:
        print(f"[auto] est pack total {est_total_pack:.6f}s, est gpu total {est_total_gpu:.6f}s, est sparse total {est_total_sparse:.6f}s, dense baseline {dense_baseline_time:.6f}s")

    # Otherwise run sparse batched path with parallel packing
    times = []
    last_C = None

    # Helper for threaded extraction
    def _extract_block(args):
        br, br_end, bk, bk_end = args
        blk = A[br:br_end, bk:bk_end].toarray()
        return (br, br_end, bk, bk_end, blk)

    for r in range(repeat):
        C_out = torch.zeros((m, n), device=device, dtype=torch.float32)
        t_run0 = time.time()
        pack_time = 0.0
        gpu_time = 0.0

        # Process in batches of batch_size
        idx = 0
        while idx < num_nonempty:
            batch_coords = tile_coords[idx: idx + batch_size]
            idx += batch_size

            # Parallel extraction
            t_p0 = time.time()
            extracted = []
            with ThreadPoolExecutor(max_workers=packing_threads) as ex:
                futures = [ex.submit(_extract_block, coords) for coords in batch_coords]
                for f in as_completed(futures):
                    extracted.append(f.result())
            t_p1 = time.time()
            pack_time += (t_p1 - t_p0)

            # Prepare lists preserving the original order (sort by input order)
            # extracted may be out-of-order; sort by br then bk to have deterministic behavior
            extracted.sort(key=lambda x: (x[0], x[2]))

            A_batch_list = []
            B_batch_list = []
            coords_local = []
            for (br, br_end, bk, bk_end, A_block) in extracted:
                m_t = br_end - br
                k_t = bk_end - bk
                # if block already full-size tile -> use directly (no pad)
                if m_t == tile and k_t == tile:
                    A_batch_list.append(A_block.astype(dtype_np, copy=False))
                    B_batch_list.append(B_cast[bk:bk_end, :])
                else:
                    # pad to full tile shape (tile, tile) and (tile, n)
                    a_p = np.zeros((tile, tile), dtype=dtype_np)
                    a_p[:m_t, :k_t] = A_block.astype(dtype_np, copy=False)
                    b_p = np.zeros((tile, n), dtype=dtype_np)
                    b_p[:k_t, :] = B_cast[bk:bk_end, :]
                    A_batch_list.append(a_p)
                    B_batch_list.append(b_p)
                coords_local.append((br, br_end, m_t))

            # Stack and H2D in one shot
            t_h2d0 = time.time()
            A_batch_np = np.stack(A_batch_list)   # (B, tile, tile)
            B_batch_np = np.stack(B_batch_list)   # (B, tile, n)
            A_batch = torch.from_numpy(A_batch_np).to(device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
            B_batch = torch.from_numpy(B_batch_np).to(device=device, dtype=(torch.float16 if use_fp16 else torch.float32))
            torch.cuda.synchronize()
            t_h2d1 = time.time()
            pack_time += (t_h2d1 - t_h2d0)

            # GPU batched matmul
            t_gpu0 = time.time()
            C_batch = torch.bmm(A_batch, B_batch)  # (B, tile, n)
            torch.cuda.synchronize()
            t_gpu1 = time.time()
            gpu_time += (t_gpu1 - t_gpu0)

            # scatter results
            for i, (br0, br1, mm) in enumerate(coords_local):
                C_out[br0:br1, :] += C_batch[i, :mm, :].to(dtype=torch.float32)

        torch.cuda.synchronize()
        t_run1 = time.time()
        elapsed = t_run1 - t_run0
        times.append(elapsed)
        last_C = C_out.cpu().numpy().astype(np.float32)
        if verbose:
            print(f"[sparse] iter {r}: total_elapsed={elapsed:.6f}s pack_time={pack_time:.6f}s gpu_time={gpu_time:.6f}s tiles={num_nonempty}")
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
    p.add_argument("--packing_threads", type=int, default=None, help="Number of threads for host packing (None = auto)")
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
            A, B, tile=args.tile, use_fp16=args.fp16, warmup=args.warmup, repeat=args.repeat, verbose=True, batch_size=args.batch_size, packing_threads=args.packing_threads
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