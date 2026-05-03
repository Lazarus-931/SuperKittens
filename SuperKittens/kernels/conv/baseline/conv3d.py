"""conv3d MLX baseline for SuperKittens comparison."""
import time, statistics
import mlx.core as mx

def bench_conv3d(D=56, H=56, W=56, C_in=64, C_out=64, K=3, iters=20):
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

if __name__ == "__main__":
    bench_conv3d()
