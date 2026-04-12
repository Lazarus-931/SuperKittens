#
#//  bench.py
#//  SuperKittens
#//
#//  Created by Alazar Manakelew on 4/6/26.
#//

from __future__ import annotations

import time

import mlx.core as mx

from baselines.mlx_ssd import ssd

# Accuracy + Latency Benchmark
## Measures the speed of both the minimal ssm as well as the speed of a full fwd pass as well as returns the output for accuracy comparison against metal




def mlx_barrier(*arrays: object, stream: mx.Stream | None = None) -> None:
    """Force lazy MLX work to complete before or after timing."""
    if arrays:
        mx.eval(*arrays)
    mx.synchronize(stream)


def time_mlx_call(
    fn,
    *args,
    stream: mx.Stream | None = None,
    **kwargs,
) -> tuple[object, float]:
    """Time an MLX call with the required eval/synchronize barriers."""
    mx.synchronize(stream)
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    mlx_barrier(out, stream=stream)
    elapsed = time.perf_counter() - start
    return out, elapsed


class LatencyBench:
    def __init__(
        self,
        seq_len: int,
        batch_size: int = 1,
        n_heads: int = 8,
        d_state: int = 64,
        d_value: int = 64,
        block_len: int = 64,
        dtype: mx.Dtype = mx.float32,
        seed: int = 0,
        use_compiled: bool = False,
        stream: mx.Stream | None = None,
    ) -> None:
        if seq_len <= 0 or block_len <= 0:
            raise ValueError("seq_len and block_len must be positive")
        if seq_len % block_len != 0:
            raise ValueError("seq_len must be divisible by block_len")

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_value = d_value
        self.block_len = block_len
        self.dtype = dtype
        self.seed = seed
        self.stream = stream

        mx.random.seed(seed)

        self.q = mx.random.normal(
            (batch_size, seq_len, n_heads, d_state), dtype=dtype
        )
        self.k = mx.random.normal(
            (batch_size, seq_len, n_heads, d_state), dtype=dtype
        )
        self.v = mx.random.normal(
            (batch_size, seq_len, n_heads, d_value), dtype=dtype
        )
        self.a = mx.random.normal((batch_size, seq_len, n_heads), dtype=dtype)
        self.initial_states = mx.zeros(
            (batch_size, n_heads, d_state, d_value), dtype=dtype
        )

        self.ssd_fn = mx.compile(ssd) if use_compiled else ssd
        self.fwd_fn = (
            mx.compile(lambda *args: ssd(*args)[0]) if use_compiled else lambda *args: ssd(*args)[0]
        )
        self.ssd_args = (
            self.q,
            self.k,
            self.v,
            self.a,
            self.block_len,
            self.initial_states,
        )

        # Pre-materialize constructor inputs so setup cost stays out of timing.
        mlx_barrier(
            self.q,
            self.k,
            self.v,
            self.a,
            self.initial_states,
            stream=stream,
        )

    def ssd_bench(self):
        fn = self.ssd_fn
        args = self.ssd_args
        stream = self.stream
        (y, final_state), elapsed = time_mlx_call(fn, *args, stream=stream)
        return {
            "elapsed_s": elapsed,
            "output_shape": y.shape,
            "final_state_shape": final_state.shape,
        }

    def fwd_bench(self):
        fn = self.fwd_fn
        args = self.ssd_args
        stream = self.stream
        y, elapsed = time_mlx_call(fn, *args, stream=stream)
        return {
            "elapsed_s": elapsed,
            "output_shape": y.shape,
        }
    
    def output(self, type: str):
        if type == "ssd":
            y, _ = self.ssd_fn(*self.ssd_args)
            mlx_barrier(y, stream=self.stream)
            return y
        

if __name__ == "__main__":
    SEQ_LENS = [512, 1024, 2048, 4096, 8192]
    for i in SEQ_LENS:
        time_bench = LatencyBench(i).ssd_bench
        print(f"Time for {i}: {time_bench}")
