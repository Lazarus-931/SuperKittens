"""Mamba-2 MLX benchmark."""
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[3] / "kernels" / "mamba" / "mamba2" / "baseline"
sys.path.insert(0, str(BASELINE))

from bench import bench_ssm, bench_block


def main():
    print("=" * 55)
    print("Mamba-2 — MLX Baseline")
    print("=" * 55)
    t_ssm = bench_ssm()
    t_blk = bench_block()
    print(f"  SSM:       {t_ssm:.3f}ms  (Metal: 1.49ms)")
    print(f"  Block:     {t_blk:.3f}ms  (Metal: —)")


if __name__ == "__main__":
    main()
