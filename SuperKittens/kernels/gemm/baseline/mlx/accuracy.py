from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from gemm import err_stats, gemm_bias_silu, gemm_nn, load_float, load_half, parse_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    args = parser.parse_args()

    dump_dir = args.dump_dir
    meta = parse_meta(dump_dir / "meta.txt")
    mode = meta.get("mode", "plain")
    m = int(meta["M"])
    n = int(meta["N"])
    k = int(meta["K"])
    alpha = float(meta.get("alpha", "1.0"))
    beta = float(meta.get("beta", "0.0"))

    a = mx.array(load_half(dump_dir / "a.bin", (m, k)))
    b = mx.array(load_half(dump_dir / "b.bin", (k, n)))
    c_in = mx.array(load_half(dump_dir / "c_in.bin", (m, n)))
    y_gpu = load_half(dump_dir / "y_gpu.bin", (m, n)).astype(np.float32)
    y_cpu = load_float(dump_dir / "y_cpu.bin", (m, n))

    if mode == "bias_silu":
        bias = mx.array(load_half(dump_dir / "bias.bin", (n,)))
        out = gemm_bias_silu(a, b, bias, c_in, alpha=alpha, beta=beta)
    else:
        out = gemm_nn(a, b, c_in, alpha=alpha, beta=beta)

    mx.eval(out)
    y_mlx = np.array(out, copy=False).astype(np.float32)

    cpu_max, cpu_mean, cpu_l2 = err_stats(y_mlx, y_cpu)
    gpu_max, gpu_mean, gpu_l2 = err_stats(y_mlx, y_gpu)

    print(f"mode={mode} M={m} N={n} K={k}")
    print(f"mlx_vs_cpu max_abs={cpu_max:.5f} mean_abs={cpu_mean:.5f} l2_rel={cpu_l2:.5f}")
    print(f"mlx_vs_gpu max_abs={gpu_max:.5f} mean_abs={gpu_mean:.5f} l2_rel={gpu_l2:.5f}")


if __name__ == "__main__":
    main()
