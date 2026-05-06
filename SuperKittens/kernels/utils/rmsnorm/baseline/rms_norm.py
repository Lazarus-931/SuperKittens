"""RMSNorm — MLX reference."""
import mlx.core as mx

def rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    return x * mx.rsqrt((x * x).mean(axis=-1, keepdims=True) + eps) * weight
