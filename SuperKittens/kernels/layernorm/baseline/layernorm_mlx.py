"""MLX layernorm/rmsnorm baselines for SuperKittens comparison."""
import mlx.core as mx

def layernorm(x, weight, bias, eps=1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / mx.sqrt(var + eps) + bias

def rmsnorm(x, weight, eps=1e-5):
    rrms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rrms * weight
