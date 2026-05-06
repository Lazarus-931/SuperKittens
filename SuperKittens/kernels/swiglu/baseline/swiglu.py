"""
SwiGLU — MLX reference.
y = silu(x_gate) * x_up   where x is (rows, 2*d), first d = gate, second d = up
"""
import mlx.core as mx


def swiglu(x: mx.array) -> mx.array:
    """x: (..., 2*d) → (..., d)"""
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    return up * (gate * mx.sigmoid(gate))
