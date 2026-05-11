#
#  baseline_m2.py
#  SuperKittens — MLX reference for Mamba-2 selective scan
#


import mlx.core as mx


def selective_scan(
    Q: mx.array,       # (batch, length, n_heads, d_state)
    K: mx.array,       # (batch, length, n_heads, d_state)
    V: mx.array,       # (batch, length, n_heads, d_value)
    A_log: mx.array,   # (batch, length, n_heads) — log-space decay
    chunk_len: int = 64,
) -> mx.array:
    """
    Mamba-2 selective scan (SSD).
    Per (batch, head): h_t = exp(A[t])*h_{t-1} + K[t]^T @ V[t];  y_t = Q[t] @ h_t
    Returns y: (batch, length, n_heads, d_value)
    """
    B, L, H, Ds = Q.shape
    Dv = V.shape[-1]

    pad = (chunk_len - L % chunk_len) % chunk_len
    if pad:
        Q = mx.pad(Q, [(0,0),(0,pad),(0,0),(0,0)])
        K = mx.pad(K, [(0,0),(0,pad),(0,0),(0,0)])
        V = mx.pad(V, [(0,0),(0,pad),(0,0),(0,0)])
        A_log = mx.pad(A_log, [(0,0),(0,pad),(0,0)])
    Lp = Q.shape[1]
    C = Lp // chunk_len

    # (B, chunks, heads, chunk, d)
    Qc = Q.reshape(B, C, chunk_len, H, Ds).transpose(0,1,3,2,4)
    Kc = K.reshape(B, C, chunk_len, H, Ds).transpose(0,1,3,2,4)
    Vc = V.reshape(B, C, chunk_len, H, Dv).transpose(0,1,3,2,4)
    Ac = A_log.reshape(B, C, chunk_len, H).transpose(0,1,3,2)  # (B,C,H,chunk)

    # ── Cross-chunk state via sequential scan ──
    state = mx.zeros((B, H, Ds, Dv), dtype=Q.dtype)
    outputs = []

    for c in range(C):
        A_c = Ac[:, c]       # (B, H, chunk)
        Q_c = Qc[:, c]       # (B, H, chunk, Ds)
        K_c = Kc[:, c]       # (B, H, chunk, Ds)
        V_c = Vc[:, c]       # (B, H, chunk, Dv)

        decay = mx.exp(mx.cumsum(A_c, axis=-1))  # (B, H, chunk)

        state_cur = state  # (B, H, Ds, Dv)
        for i in range(chunk_len):
            d_i = mx.exp(A_c[..., i])[..., None, None]  # (B, H, 1, 1)
            kv_i = mx.einsum("bhd,bhD->bhdD", K_c[..., i, :], V_c[..., i, :])
            state_cur = d_i * state_cur + kv_i
            y_i = mx.einsum("bhd,bhdD->bhD", Q_c[..., i, :], state_cur)
            outputs.append(y_i)

        # Final state carries to next chunk
        state = state_cur

    y = mx.stack(outputs, axis=2)  # (B, H, Lp, Dv)
    y = y.transpose(0, 2, 1, 3)    # (B, Lp, H, Dv)
    return y[:, :L] if pad else y
