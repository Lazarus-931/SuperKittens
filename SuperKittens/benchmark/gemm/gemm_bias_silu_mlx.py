from __future__ import annotations

from pathlib import Path
import sys

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.harness import BenchHarness
from kernels.gemm.baseline.mlx.bench_bias_silu import gemm_mlx_bias_silu


def main() -> None:
    harness = BenchHarness()
    shapes = [
        (256, 256, 256),
        (1024, 1024, 1024),
        (2048, 3072, 4096),
        (3072, 2048, 4096),
        (4096, 4096, 4096),
    ]
    for m, n, k in shapes:
        result = harness.run(
            gemm_mlx_bias_silu,
            m=m,
            n=n,
            k=k,
            alpha=1.0,
            beta=0.0,
            seed=42,
            dtype=mx.float16,
        )
        harness.print_result(result)


if __name__ == "__main__":
    main()
