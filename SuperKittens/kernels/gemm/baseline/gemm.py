"""gemm MLX baselines — fp16 and fp8."""
import mlx.core as mx


def gemm_fp16(a: mx.array, b: mx.array, c_in: mx.array | None = None,
              alpha: float = 1.0, beta: float = 0.0) -> mx.array:
    out = alpha * mx.matmul(a, b)
    if c_in is not None and beta != 0.0:
        out = out + beta * c_in
    return out


def gemm_fp8(a: mx.array, b: mx.array, c_in: mx.array | None = None,
             alpha: float = 1.0, beta: float = 0.0) -> mx.array:
    out = alpha * mx.matmul(a.astype(mx.float16), b.astype(mx.float16))
    if c_in is not None and beta != 0.0:
        out = out + beta * c_in
    return out


def gemm_bias_silu(a: mx.array, b: mx.array, bias: mx.array,
                   c_in: mx.array | None = None,
                   alpha: float = 1.0, beta: float = 0.0) -> mx.array:
    out = alpha * mx.matmul(a, b)
    if c_in is not None and beta != 0.0:
        out = out + beta * c_in
    out = out + bias
    return out * mx.sigmoid(out)
