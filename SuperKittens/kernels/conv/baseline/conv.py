"""conv2d_mlx.py — MLX conv2d baseline for SuperKittens comparison"""
import time, statistics
import mlx.core as mx



def conv1d(B=1, L=128, C=256, K=4, iters=20):
    # Causal depthwise Conv1D: x(B,L,C), w(C,K), b(C)
    x = mx.random.normal(shape=(B, L, C)).astype(mx.float16) * 0.5
    w = mx.random.normal(shape=(C, K)).astype(mx.float16) * 0.1
    b = mx.random.normal(shape=(C,)).astype(mx.float16) * 0.01

    def causal_conv1d(x, w, b_):
        B, L, C = x.shape
        K_ = w.shape[1]
        x_pad = mx.pad(x, [(0,0),(K_-1,0),(0,0)])
        y = mx.zeros((B, L, C), dtype=mx.float16)
        for k in range(K_):
            y = y + x_pad[:, k:k+L, :] * w[:, k].reshape(1,1,C)
        return y + b_.reshape(1,1,C)

    for _ in range(5):
        mx.eval(causal_conv1d(x, w, b))

    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(causal_conv1d(x, w, b))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    med = statistics.median(times)
    print(f"MLX conv1d:  B={B} L={L} C={C} K={K}")
    print(f"  median={med:.1f}us  min={min(times):.1f}us  max={max(times):.1f}us")
    return med


def conv2d(H=56, W=56, C_in=64, C_out=64, K=3, iters=20):
    N = 1
    # MLX uses NHWC: input (N,H,W,C), weight (C_out, KH, KW, C_in)
    x = mx.random.normal(shape=(N, H, W, C_in)).astype(mx.float16)
    w = mx.random.normal(shape=(C_out, K, K, C_in)).astype(mx.float16)
    b = mx.random.normal(shape=(C_out,)).astype(mx.float16)

    for _ in range(5):
        y = mx.conv2d(x, w, stride=(1,1), padding=(0,0)) + b
        mx.eval(y)

    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        y = mx.conv2d(x, w, stride=(1,1), padding=(0,0)) + b
        mx.eval(y)
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    med = statistics.median(times)
    H_out, W_out = H - K + 1, W - K + 1
    flops = 2 * N * H_out * W_out * K * K * C_in * C_out
    gflops = flops / (med * 1e3)

    print(f"MLX conv2d:  H={H} W={W} C_in={C_in} C_out={C_out} K={K}")
    print(f"  median={med:.1f}us  min={min(times):.1f}us  max={max(times):.1f}us")
    print(f"  {gflops:.1f} GFLOPS")
    return med

def conv3d(D=56, H=56, W=56, C_in=64, C_out=64, K=3, iters=20):
    N = 1
    # MLX 3D conv: input (N, D, H, W, C), weight (C_out, KD, KH, KW, C_in)
    x = mx.random.normal(shape=(N, D, H, W, C_in)).astype(mx.float16)
    w = mx.random.normal(shape=(C_out, K, K, K, C_in)).astype(mx.float16)
    b = mx.random.normal(shape=(C_out,)).astype(mx.float16)

    for _ in range(5):
        y = mx.conv3d(x, w, stride=(1,1,1), padding=(0,0,0)) + b
        mx.eval(y)

    mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        y = mx.conv3d(x, w, stride=(1,1,1), padding=(0,0,0)) + b
        mx.eval(y)
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    med = statistics.median(times)
    D_out, H_out, W_out = D - K + 1, H - K + 1, W - K + 1
    flops = 2 * N * D_out * H_out * W_out * K * K * K * C_in * C_out
    gflops = flops / (med * 1e3)

    print(f"MLX conv3d:  D={D} H={H} W={W} C_in={C_in} C_out={C_out} K={K}")
    print(f"  median={med:.1f}us  {gflops:.1f} GFLOPS")
    return med
    


