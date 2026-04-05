#!/usr/bin/env python3
import mlx.core as mx, time, sys

seq = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
d = int(sys.argv[2]) if len(sys.argv) > 2 else 128
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20

Q = mx.random.normal((1, 1, seq, d)).astype(mx.float16)
K = mx.random.normal((1, 1, seq, d)).astype(mx.float16)
V = mx.random.normal((1, 1, seq, d)).astype(mx.float16)
mx.eval(Q, K, V)

for _ in range(5):
    O = mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0/d**0.5)
    mx.eval(O)

times = []
for _ in range(iters):
    t0 = time.perf_counter()
    O = mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0/d**0.5)
    mx.eval(O)
    times.append((time.perf_counter() - t0) * 1e6)

times.sort()
t = times[len(times) // 2]
flops = 4 * seq * seq * d + 5 * seq * seq
print(f"{t:.0f},{flops / (t * 1e3):.1f}")
