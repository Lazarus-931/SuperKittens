"""Paged attention MLX benchmark."""
import sys, time, statistics, numpy as np
from pathlib import Path
import mlx.core as mx

BASELINE = Path(__file__).resolve().parents[2] / "kernels" / "paged_attn" / "baseline"
sys.path.insert(0, str(BASELINE))
from paged_attn import paged_attention


def bench(fn, args=(), iters=20):
    for _ in range(5):
        mx.eval(fn(*args))
    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return statistics.median(times)


def main():
    print("=" * 55)
    print("Paged Attention — MLX Baseline")
    print("=" * 55)

    num_seqs, num_heads, head_dim = 8, 32, 128
    num_kv_heads, block_size = 8, 16
    seq_len = 256
    num_blocks = (seq_len + block_size - 1) // block_size

    Q = mx.random.normal((num_seqs, num_heads, head_dim), dtype=mx.float16) * 0.5
    total_blocks = num_seqs * num_blocks  # enough for all sequences
    K_cache = mx.random.normal((total_blocks, block_size, num_kv_heads, head_dim), dtype=mx.float16) * 0.1
    V_cache = mx.random.normal((total_blocks, block_size, num_kv_heads, head_dim), dtype=mx.float16) * 0.1
    bt_np = np.zeros((num_seqs, num_blocks), dtype=np.int32)
    for s in range(num_seqs):
        for b in range(num_blocks):
            bt_np[s, b] = s * num_blocks + b
    block_table = mx.array(bt_np)

    seq_lens_arr = mx.array(np.full(num_seqs, seq_len, dtype=np.int32))
    mx.eval(Q, K_cache, V_cache, block_table, seq_lens_arr)

    us = bench(paged_attention, (Q, K_cache, V_cache, block_table, seq_lens_arr, block_size))
    total_tokens = num_seqs * seq_len
    flops = 2.0 * num_seqs * num_heads * seq_len * head_dim * 2
    gflops = flops / (us * 1e3)

    print(f"  seqs={num_seqs} heads={num_heads}/{num_kv_heads} d={head_dim} "
          f"L={seq_len} blocks={num_blocks}")
    print(f"  time: {us:.1f}us  ({gflops:.1f} GFLOPS)")
    print(f"  MLX reference — Python loop, not optimized")


if __name__ == "__main__":
    main()
