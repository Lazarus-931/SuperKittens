from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.harness import BenchHarness, bench
from kernels.gemm.baseline.mlx.gemm import gemm_nn

_INPUTS_KEY = "_cached_inputs"


def _make_inputs(cfg: dict) -> tuple[mx.array, mx.array, mx.array]:
    seed = int(cfg.get("seed", 0))
    mx.random.seed(seed)

    m = int(cfg["m"])
    n = int(cfg["n"])
    k = int(cfg["k"])
    dtype = cfg.get("dtype", mx.float16)
    a = mx.random.normal((m, k), dtype=dtype)
    b = mx.random.normal((k, n), dtype=dtype)
    c_in = mx.random.normal((m, n), dtype=dtype)
    return a, b, c_in


def _get_inputs(cfg: dict) -> tuple[mx.array, mx.array, mx.array]:
    cached = cfg.get(_INPUTS_KEY)
    if cached is None:
        cached = _make_inputs(cfg)
        mx.eval(*cached)
        cfg[_INPUTS_KEY] = cached
    return cached


def _flops(cfg: dict) -> float:
    m = int(cfg["m"])
    n = int(cfg["n"])
    k = int(cfg["k"])
    return 2.0 * m * n * k


@bench("gemm_mlx_nn", flops=_flops)
def gemm_mlx_nn(cfg: dict):
    a, b, c_in = _get_inputs(cfg)
    return gemm_nn(a, b, c_in, float(cfg.get("alpha", 1.0)), float(cfg.get("beta", 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, nargs="?", default=256)
    parser.add_argument("n", type=int, nargs="?", default=256)
    parser.add_argument("k", type=int, nargs="?", default=256)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    harness = BenchHarness(iters=args.iters)
    result = harness.run(
        gemm_mlx_nn,
        m=args.m,
        n=args.n,
        k=args.k,
        alpha=1.0,
        beta=0.0,
        seed=args.seed,
        dtype=mx.float16,
    )

    if args.csv:
        print(f"{result.median_ms * 1000.0:.0f},{result.gflops:.1f}")
    else:
        harness.print_result(result)


if __name__ == "__main__":
    main()
