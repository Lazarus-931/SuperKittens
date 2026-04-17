from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.harness import BenchHarness, bench
from kernels.mamba.mamba3.baseline.baseline_m3 import (
    apply_rotary_qk,
    rotary_angles,
    trap_discretization,
)


def _make_trap_inputs(cfg: dict) -> tuple[mx.array, mx.array]:
    seed = int(cfg.get("seed", 0))
    mx.random.seed(seed)

    batch = int(cfg["batch"])
    length = int(cfg["length"])
    heads = int(cfg["heads"])
    dtype = cfg.get("dtype", mx.float32)

    a = mx.random.normal((batch, length, heads), dtype=dtype)
    b = mx.random.normal((batch, length, heads), dtype=dtype)
    return a, b


def _make_rotary_inputs(cfg: dict) -> tuple[mx.array, mx.array, mx.array]:
    seed = int(cfg.get("seed", 0))
    mx.random.seed(seed)

    batch = int(cfg["batch"])
    length = int(cfg["length"])
    heads = int(cfg["heads"])
    dim = int(cfg["dim"])
    dtype = cfg.get("dtype", mx.float32)

    q = mx.random.normal((batch, length, heads, dim), dtype=dtype)
    k = mx.random.normal((batch, length, heads, dim), dtype=dtype)
    a = mx.random.normal((batch, length, heads), dtype=dtype)
    angles = mx.random.normal((batch, length, heads, dim // 2), dtype=dtype)
    angle_state = mx.random.normal((batch, heads, dim // 2), dtype=dtype)
    rotary = rotary_angles(a, angles, angle_state)
    return q, k, rotary


@bench("mamba3_mlx_trap")
def mamba3_trap(cfg: dict):
    return trap_discretization(*_make_trap_inputs(cfg))


@bench("mamba3_mlx_rotary")
def mamba3_rotary(cfg: dict):
    return apply_rotary_qk(*_make_rotary_inputs(cfg))[0]


def main() -> None:
    harness = BenchHarness()
    harness.sweep(
        mamba3_trap,
        batch=[1],
        length=[512, 1024],
        heads=[8],
        seed=[0],
        dtype=[mx.float32],
    )
    harness.sweep(
        mamba3_rotary,
        batch=[1],
        length=[512, 1024],
        heads=[8],
        dim=[64],
        seed=[0],
        dtype=[mx.float32],
    )


if __name__ == "__main__":
    main()
