"""
MLX reference implementations for activation functions.
"""

import mlx.core as mx


def gelu(x: mx.array) -> mx.array:
    """GELU: x * Φ(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))"""
    a = 0.044715 * x * x * x
    b = 0.7978845608028654 * (x + a)  # sqrt(2/π)
    return 0.5 * x * (1.0 + mx.tanh(b))


def silu(x: mx.array) -> mx.array:
    """SiLU: x * sigmoid(x) = x / (1 + exp(-x))"""
    return x * mx.sigmoid(x)


def relu(x: mx.array) -> mx.array:
    """ReLU: max(0, x)"""
    return mx.maximum(x, 0.0)
