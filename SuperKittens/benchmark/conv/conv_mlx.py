#!/usr/bin/env python3
"""MLX baseline benchmarks — conv1d, conv2d, conv3d"""
import time, statistics
import mlx.core as mx

WARMUP, ITERS = 5, 20

def bench(name, fn, *args):
    for _ in range(WARMUP):
        mx.eval(fn(*args))
    mx.synchronize()
    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.median(times)

# ── conv1d: causal depthwise ──

def conv1d_fn(x, w, b):
    B, L, C = x.shape
    K = w.shape[1]
    x_pad = mx.pad(x, [(0,0),(K-1,0),(0,0)])
    y = mx.zeros((B, L, C), dtype=mx.float16)
    for k in range(K):
        y = y + x_pad[:, k:k+L, :] * w[:, k].reshape(1,1,C)
    return y + b.reshape(1,1,C)

def bench_conv1d():
    print("\n── conv1d ──")
    print(f"{'config':<40}  {'us':>8}  {'GB/s':>8}")
    for B, L, C, K in [(1,128,256,4), (1,1024,256,4), (1,128,512,4), (1,128,256,8)]:
        x = mx.random.normal((B,L,C), dtype=mx.float16) * 0.5
        w = mx.random.normal((C,K), dtype=mx.float16) * 0.1
        b = mx.random.normal((C,), dtype=mx.float16) * 0.01
        mx.eval(x,w,b)
        us = bench("conv1d", conv1d_fn, x, w, b)
        bw = (2 * B * L * C * 2) / (us * 1e3)  # GB/s
        print(f"B={B} L={L} C={C} K={K:<30}  {us:>8.1f}  {bw:>8.1f}")

# ── conv2d ──

def bench_conv2d():
    print("\n── conv2d ──")
    print(f"{'config':<40}  {'us':>8}  {'GFLOPS':>8}")
    for H, W, C_in, C_out, K in [(56,56,64,64,3), (28,28,128,128,3), (14,14,256,256,3), (56,56,64,128,1)]:
        x = mx.random.normal((1,H,W,C_in), dtype=mx.float16)
        w = mx.random.normal((C_out,K,K,C_in), dtype=mx.float16)
        b = mx.random.normal((C_out,), dtype=mx.float16)
        mx.eval(x,w,b)
        us = bench("conv2d", lambda: mx.conv2d(x,w,stride=(1,1),padding=(0,0))+b)
        H_out, W_out = H-K+1, W-K+1
        flops = 2.0 * H_out * W_out * K * K * C_in * C_out
        gflops = flops / (us * 1e3)
        print(f"H={H} W={W} C_in={C_in} C_out={C_out} K={K:<10}  {us:>8.1f}  {gflops:>8.1f}")

# ── conv3d ──

def bench_conv3d():
    print("\n── conv3d ──")
    print(f"{'config':<50}  {'us':>8}  {'GFLOPS':>8}")
    for D, H, W, C_in, C_out, K in [(16,32,32,64,64,3), (8,56,56,32,64,3), (32,16,16,128,128,3)]:
        x = mx.random.normal((1,D,H,W,C_in), dtype=mx.float16)
        w = mx.random.normal((C_out,K,K,K,C_in), dtype=mx.float16)
        b = mx.random.normal((C_out,), dtype=mx.float16)
        mx.eval(x,w,b)
        us = bench("conv3d", lambda: mx.conv3d(x,w,stride=(1,1,1),padding=(0,0,0))+b)
        D_out, H_out, W_out = D-K+1, H-K+1, W-K+1
        flops = 2.0 * D_out * H_out * W_out * K * K * K * C_in * C_out
        gflops = flops / (us * 1e3)
        print(f"D={D} H={H} W={W} C_in={C_in} C_out={C_out} K={K:<4}  {us:>8.1f}  {gflops:>8.1f}")

if __name__ == "__main__":
    print("=" * 65)
    print("Conv — MLX Baseline")
    print("=" * 65)
    bench_conv1d()
    bench_conv2d()
    bench_conv3d()
