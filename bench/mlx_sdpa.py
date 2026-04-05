import mlx.core as mx, time, sys
seq, d = int(sys.argv[1]), int(sys.argv[2])
Q = mx.random.normal((1, 1, seq, d)).astype(mx.float16)
K = mx.random.normal((1, 1, seq, d)).astype(mx.float16)
V = mx.random.normal((1, 1, seq, d)).astype(mx.float16)
mx.eval(Q, K, V)
for _ in range(5):
    O = mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0/d**0.5)
    mx.eval(O)
t0 = time.perf_counter()
O = mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0/d**0.5)
mx.eval(O)
print(f"{(time.perf_counter()-t0)*1e6:.1f}")
