"""Mamba-3 MLX benchmark."""
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[3] / "kernels" / "mamba" / "mamba3" / "baseline"
sys.path.insert(0, str(BASELINE))

from bench import bench_ssm, bench_block


def main():
    print("=" * 55)
    print("Mamba-3 — MLX Baseline")
    print("=" * 55)

    print("\n── MIMO  (DQ=64, DV=64, H=4) ──")
    t_ssm = bench_ssm(L=128, DQ=64, DV=64, H=4)
    t_blk = bench_block(L=128, D=128, DQ=64, DV=64, H=4)
    print(f"  SSM:       {t_ssm:.3f}ms  (Metal: 0.91ms)")
    print(f"  Block:     {t_blk:.3f}ms  (Metal: 1.1ms)")

    print("\n── SISO  (DQ=4, DV=4, H=4) ──")
    t_ssm_s = bench_ssm(L=128, DQ=4, DV=4, H=4)
    t_blk_s = bench_block(L=128, D=128, DQ=4, DV=4, H=4)
    print(f"  SSM:       {t_ssm_s:.3f}ms")
    print(f"  Block:     {t_blk_s:.3f}ms")


if __name__ == "__main__":
    main()
