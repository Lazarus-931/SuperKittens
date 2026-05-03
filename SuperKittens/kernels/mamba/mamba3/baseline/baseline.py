"""
Mamba-3 SSM block — MLX reference implementation.

Corresponds to the mamba3_fwd Metal kernel.
Implements the selective scan with trapezoidal discretization,
rotary embeddings, and chunked recurrent state update.
"""

import mlx.core as mx

PI = 3.14159265358979323846


def rotate_qk(q: mx.array, k: mx.array, theta: mx.array) -> tuple[mx.array, mx.array]:
    """Apply rotary embedding to Q and K by angle theta.

    Args:
        q, k: (..., d) where d is even
        theta: (..., d//2)
    Returns:
        q_rot, k_rot: (..., d)
    """
    half = q.shape[-1] // 2
    q0, q1 = q[..., :half], q[..., half:]
    k0, k1 = k[..., :half], k[..., half:]
    c = mx.cos(theta)
    s = mx.sin(theta)
    q_rot = mx.concatenate([q0 * c - q1 * s, q0 * s + q1 * c], axis=-1)
    k_rot = mx.concatenate([k0 * c - k1 * s, k0 * s + k1 * c], axis=-1)
    return q_rot, k_rot


def mamba3_ssm(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    angle: mx.array,
    chunk_size: int = 64,
) -> mx.array:
    """Mamba-3 SSM forward pass.

    Implements the selective scan with trapezoidal discretization and
    rotary embeddings, processing the sequence in chunks with a recurrent
    KV state carried across chunks.

    Args:
        q:      (batch, heads, seq_len, d_qk)
        k:      (batch, heads, seq_len, d_qk)
        v:      (batch, heads, seq_len, d_v)
        a:      (batch, heads, seq_len)          — dt
        b:      (batch, heads, seq_len)          — dt bias
        angle:  (batch, heads, seq_len, d_qk//2) — raw rotary angles
        chunk_size: int

    Returns:
        output: (batch, heads, seq_len, d_v)
    """
    B, H, L, DQ = q.shape
    DV = v.shape[-1]
    n_chunks = (L + chunk_size - 1) // chunk_size

    # Flatten batch+heads for batched matmul
    q = q.reshape(B * H, L, DQ)
    k = k.reshape(B * H, L, DQ)
    v = v.reshape(B * H, L, DV)
    a = a.reshape(B * H, L)
    b = b.reshape(B * H, L)
    angle = angle.reshape(B * H, L, DQ // 2)

    kv_state = mx.zeros((B * H, DQ, DV), dtype=mx.float32)
    outputs = []

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(L, start + chunk_size)
        clen = end - start

        qc = q[:, start:end]          # (BH, clen, DQ)
        kc = k[:, start:end]
        vc = v[:, start:end]
        ac = a[:, start:end]
        bc = b[:, start:end]
        ang_c = angle[:, start:end]    # (BH, clen, DQ//2)

        # 1. Cumulative sum of A (dt) — continuous-time position
        a_cs = mx.cumsum(ac, axis=1)                                    # (BH, clen)

        # 2. Trapezoidal discretization: b_scale = 1 + B * exp(-a_cs)
        b_scale = 1.0 + bc * mx.exp(-a_cs)                              # (BH, clen)

        # 3. Rotary angles: theta = a_cs * raw_angle * pi
        theta = a_cs[:, :, None] * ang_c * PI                           # (BH, clen, DQ//2)

        # 4. Apply rotary to Q, K
        q_rot, k_rot = rotate_qk(qc, kc, theta)                         # (BH, clen, DQ)

        # 5. Recurrent state update: kv_state *= chunk_decay + K^T @ V
        chunk_decay = mx.exp(a_cs[:, -1]) * b_scale[:, -1]              # (BH,)
        kv_state = kv_state * chunk_decay[:, None, None]                 # (BH, DQ, DV)
        kv_state = kv_state + mx.matmul(
            mx.transpose(k_rot, (0, 2, 1)), vc)                         # K^T @ V

        # 6. Intra-chunk causal attention
        scores = q_rot @ mx.transpose(k_rot, (0, 2, 1))                 # (BH, clen, clen)
        decay = mx.exp(a_cs[:, :, None] - a_cs[:, None, :])             # (BH, clen, clen)
        causal = mx.tril(mx.ones((clen, clen), dtype=mx.float32))
        intra = (scores * decay * causal) @ vc                           # (BH, clen, DV)

        # 7. Inter-chunk contribution from recurrent state
        q_decay = (mx.exp(a_cs) * b_scale)[:, :, None]                  # (BH, clen, 1)
        inter = q_decay * (q_rot @ kv_state)                             # (BH, clen, DV)

        outputs.append(intra + inter)

    out = mx.concatenate(outputs, axis=1)  # (BH, L, DV)
    return out.reshape(B, H, L, DV)
