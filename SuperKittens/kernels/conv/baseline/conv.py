"""conv2d_mlx.py — MLX conv2d baseline for SuperKittens comparison"""
import time, statistics
import mlx.core as mx

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

if __name__ == "__main__":
    bench_conv2d()
