#!/usr/bin/env python3
"""MLX layernorm baseline."""
import sys, time, math
import mlx.core as mx

rows  = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
d     = int(sys.argv[2]) if len(sys.argv) > 2 else 128
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20

x = mx.random.normal((rows, d)).astype(mx.float16)
gamma = mx.random.normal((d,)).astype(mx.float16)
beta = mx.random.normal((d,)).astype(mx.float16)
mx.eval(x, gamma, beta)

def run():
    y = mx.fast.layer_norm(x, gamma, beta, 1e-5)
    mx.eval(y)

for _ in range(5): run()
times = []
for _ in range(iters):
    t0 = time.perf_counter()
    run()
    times.append((time.perf_counter() - t0) * 1e6)
times.sort()
us = times[len(times) // 2]
print(f"{us:.0f}")
