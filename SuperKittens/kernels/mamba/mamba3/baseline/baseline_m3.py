import mlx.core as mx


def trap_discretization(a: mx.array, b: mx.array) -> mx.array:
    a_cumsum = mx.cumsum(a, axis=1)
    trap_scale = 1.0 + b * mx.exp(-a_cumsum)
    return mx.exp(a_cumsum) * trap_scale


def rotary_angles(
    a: mx.array,
    angles: mx.array,
    angle_state: mx.array,
) -> mx.array:
    a_cumsum = mx.cumsum(a, axis=1)[..., None]
    return angle_state[:, None, :, :] + a_cumsum * angles * 3.141592653589793


def apply_rotary_qk(
    q: mx.array,
    k: mx.array,
    rotary_angle: mx.array,
) -> tuple[mx.array, mx.array]:
    half_dim = q.shape[-1] // 2
    q0, q1 = q[..., :half_dim], q[..., half_dim:]
    k0, k1 = k[..., :half_dim], k[..., half_dim:]

    c = mx.cos(rotary_angle)
    s = mx.sin(rotary_angle)

    q_rot = mx.concatenate([q0 * c - q1 * s, q0 * s + q1 * c], axis=-1)
    k_rot = mx.concatenate([k0 * c - k1 * s, k0 * s + k1 * c], axis=-1)
    return q_rot, k_rot
