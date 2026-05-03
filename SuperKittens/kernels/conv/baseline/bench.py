#!/usr/bin/env python3
"""conv bench.py — SuperKittens vs MLX conv2d benchmark"""
import sys, os, time, statistics, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "baseline"
sys.path.insert(0, str(BASELINE))
from conv2d import bench_conv2d as mlx_bench

def metal_bench(H=56, W=56, C_in=64, C_out=64, K=3):
    """Run pre-built Metal benchmark and parse output."""
    binary = ROOT.parent / "build" / "conv2d_full_bench"
    metallib = ROOT.parent / "build" / "conv_full.metallib"
    if not binary.exists() or not metallib.exists():
        print("  [Metal] build not found — run: ./build_conv.sh")
        return None
    out = subprocess.check_output([str(binary), str(metallib)], text=True)
    for line in out.splitlines():
        if "Total:" in line:
            # "  Total:    589.8 us  (364.5 GFLOPS)"
            parts = line.strip().split()
            return float(parts[1])
    return None

def main():
    print("=" * 60)
    print("Conv2d — SuperKittens vs MLX")
    print("=" * 60)

    mlx_us = mlx_bench()
    metal_us = metal_bench()

    print()
    if metal_us:
        print(f"  Metal:  {metal_us:.1f} us")
        print(f"  MLX:    {mlx_us:.1f} us")
        speedup = mlx_us / metal_us
        print(f"  vs MLX: {speedup:.2f}×")
        if speedup < 1:
            print(f"  MLX is {1/speedup:.2f}× faster")
    else:
        print(f"  MLX:    {mlx_us:.1f} us")
        print(f"  (Metal binary not found)")

if __name__ == "__main__":
    main()
