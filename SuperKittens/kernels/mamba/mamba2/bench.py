from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.harness import BenchHarness, bench
from kernels.mamba.mamba2.baseline.baseline_m2 import ssd


def _make_inputs(cfg: dict) -> tuple[mx.array, ...]:
    seed = int(cfg.get("seed", 0))
    mx.random.seed(seed)

    batch = int(cfg["batch"])
    length = int(cfg["length"])
    heads = int(cfg["heads"])
    d_state = int(cfg["d_state"])
    d_value = int(cfg["d_value"])
    block_len = int(cfg["block_len"])
    dtype = cfg.get("dtype", mx.float32)

    q = mx.random.normal((batch, length, heads, d_state), dtype=dtype)
    k = mx.random.normal((batch, length, heads, d_state), dtype=dtype)
    v = mx.random.normal((batch, length, heads, d_value), dtype=dtype)
    a = mx.random.normal((batch, length, heads), dtype=dtype)
    init = mx.zeros((batch, heads, d_state, d_value), dtype=dtype)
    return q, k, v, a, block_len, init


@bench("mamba2_mlx_ssd")
def mamba2_mlx(cfg: dict):
    return ssd(*_make_inputs(cfg))[0]


def main() -> None:
    harness = BenchHarness()
    harness.sweep(
        mamba2_mlx,
        batch=[1],
        length=[512, 1024],
        heads=[8],
        d_state=[64],
        d_value=[64],
        block_len=[64],
        seed=[0],
        dtype=[mx.float32],
    )


if __name__ == "__main__":
    main()
