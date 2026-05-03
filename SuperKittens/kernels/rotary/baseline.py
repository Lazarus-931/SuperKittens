"""
RoPE — MLX reference.
"""

import math
import mlx.core as mx


def rope(q: mx.array, k: mx.array, base: float = 10000.0) -> tuple[mx.array, mx.array]:
    """Apply rotary position embedding to Q and K.

    Args:
        q, k: (batch*heads, seq, head_dim) or (batch, heads, seq, head_dim)
        base: frequency base (default 10000.0)
    Returns:
        q_rot, k_rot
    """
    *_, seq, dim = q.shape
    half = dim // 2

    # frequencies: 1.0 / (base^(2*i/dim))
    freq = 1.0 / (base ** (mx.arange(0, half, dtype=mx.float32) * 2.0 / dim))
    # positions
    pos = mx.arange(seq, dtype=mx.float32)
    # theta: (seq, half)
    theta = pos[:, None] * freq[None, :]

    cos = mx.cos(theta)
    sin = mx.sin(theta)

    q0, q1 = q[..., :half], q[..., half:]
    k0, k1 = k[..., :half], k[..., half:]

    q_rot0 = q0 * cos - q1 * sin
    q_rot1 = q0 * sin + q1 * cos
    k_rot0 = k0 * cos - k1 * sin
    k_rot1 = k0 * sin + k1 * cos

    return (mx.concatenate([q_rot0, q_rot1], axis=-1),
            mx.concatenate([k_rot0, k_rot1], axis=-1))
