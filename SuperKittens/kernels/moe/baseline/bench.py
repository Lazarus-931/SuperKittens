"""bench.py — correctness + micro-bench for SuperKittens MoE primitives.

Compares router / dispatch / combine against numpy ref (and MLX where useful).
Runs on Gemma-4-26B-A4B-style shapes.

Usage:
    SK_METALLIB=$(pwd)/build/libsk.metallib SK_DYLIB=$(pwd)/build/libsk.dylib \
      python3 SuperKittens/kernels/moe/baseline/bench.py
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np

from SuperKittens.kernels.moe import moe as sk_moe
from SuperKittens.kernels.moe.baseline.moe_ref import (
    router_ref, dispatch_ref, combine_ref,
)

try:
    import mlx.core as mx
    HAS_MLX = True
except Exception:
    HAS_MLX = False


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) * 1e6 / iters  # us


def correctness_router(T, D, N, K, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, D)).astype(np.float16) * 0.1
    W = rng.standard_normal((D, N)).astype(np.float16) * 0.1

    idx_sk, sc_sk = sk_moe.router(x, W, K)
    idx_ref, sc_ref = router_ref(x, W, K)

    # top-k indices may differ in tie-breaking but should mostly match.
    match = (idx_sk == idx_ref).mean()
    score_diff = float(np.max(np.abs(sc_sk.astype(np.float32) - sc_ref.astype(np.float32))))
    return match, score_diff


def correctness_dispatch_combine(T, D, N, K, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, D)).astype(np.float16) * 0.1
    # Use ref router so top_idx is deterministic for this test.
    W = rng.standard_normal((D, N)).astype(np.float16) * 0.1
    top_idx, top_score = router_ref(x, W, K)
    # renormalize across kept experts:
    s = top_score.astype(np.float32)
    s = s / np.maximum(s.sum(-1, keepdims=True), 1e-6)
    top_score = s.astype(np.float16)

    xd_sk, off_sk, dest_sk, inv_sk = sk_moe.dispatch(x, top_idx, N)
    xd_rf, off_rf, dest_rf, inv_rf = dispatch_ref(x, top_idx, N)

    # offsets must match
    off_match = np.array_equal(off_sk, off_rf)
    # x_dispatched rows may be permuted within an expert bin (atomic order),
    # so compare the per-bin SET of source rows.
    bin_match = True
    T_ = T; K_ = K
    # For each expert bin, gather source token ids on each side.
    src_sk = inv_sk // K_
    src_rf = inv_rf // K_
    for e in range(N):
        a = sorted(src_sk[off_sk[e]:off_sk[e+1]].tolist())
        b = sorted(src_rf[off_rf[e]:off_rf[e+1]].tolist())
        if a != b:
            bin_match = False
            break

    # Round-trip combine: if expert_out = x_dispatched, then
    # combine(...) should equal sum_k top_score[t,k] * x[t] = (sum_k score)*x[t]
    out_sk = sk_moe.combine(xd_sk, dest_sk, top_score, T_)
    expected = (top_score.astype(np.float32).sum(-1, keepdims=True) * x.astype(np.float32)).astype(np.float16)
    diff = float(np.max(np.abs(out_sk.astype(np.float32) - expected.astype(np.float32))))
    return off_match, bin_match, diff


def main():
    print("=" * 72)
    print("SuperKittens MoE — correctness + bench")
    print("=" * 72)
    if HAS_MLX:
        print("MLX available")
    else:
        print("MLX NOT available (skipping MLX comparison)")

    shapes = [
        # (label, T, D, N, K)
        ("decode-step",   1, 4608, 128, 2),
        ("decode-128",  128, 4608, 128, 2),
        ("medium-512", 512, 4608, 128, 2),
        ("topk8-128",   128, 4608, 128, 8),
    ]

    print("\n--- Correctness (router) ---")
    print(f"{'shape':<14} {'top-k match':>12} {'max|score|diff':>16}")
    for label, T, D, N, K in shapes:
        m, sd = correctness_router(T, D, N, K)
        print(f"{label:<14} {m*100:>11.1f}% {sd:>16.4e}")

    print("\n--- Correctness (dispatch + combine round-trip) ---")
    print(f"{'shape':<14} {'offsets':>9} {'bins':>6} {'max_abs_diff':>14}")
    for label, T, D, N, K in shapes:
        ok_off, ok_bin, diff = correctness_dispatch_combine(T, D, N, K)
        print(f"{label:<14} {str(ok_off):>9} {str(ok_bin):>6} {diff:>14.4e}")

    print("\n--- Latency (per-primitive, M2) ---")
    print(f"{'shape':<14} {'router_us':>10} {'dispatch_us':>12} {'combine_us':>11}")
    for label, T, D, N, K in shapes:
        rng = np.random.default_rng(42)
        x = (rng.standard_normal((T, D)).astype(np.float16) * 0.1)
        W = (rng.standard_normal((D, N)).astype(np.float16) * 0.1)
        top_idx, top_score = sk_moe.router(x, W, K)

        t_router = bench(lambda: sk_moe.router(x, W, K), iters=5, warmup=2)
        t_disp   = bench(lambda: sk_moe.dispatch(x, top_idx, N), iters=5, warmup=2)
        xd, off, dest, inv = sk_moe.dispatch(x, top_idx, N)
        # synthetic expert_out same shape
        eo = (rng.standard_normal(xd.shape).astype(np.float16) * 0.1)
        t_comb   = bench(lambda: sk_moe.combine(eo, dest, top_score, T), iters=5, warmup=2)

        print(f"{label:<14} {t_router:>10.1f} {t_disp:>12.1f} {t_comb:>11.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
