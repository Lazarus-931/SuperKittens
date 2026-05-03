"""
Mamba-3 full block — MLX reference (SISO + MIMO modes).

Modes:
  SISO: DQ=H, DV=H  — one scalar Q/K/V per head, no rotary
  MIMO: DQ>H, DV>H  — multi-dimensional per head, rotary applied
"""

import mlx.core as mx
from ssm import mamba3_ssm


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    rrms = mx.rsqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x * rrms * weight


def mamba3_block(
    x: mx.array,              # (B, L, D)
    in_proj_w: mx.array,      # (D, DV + 2*DQ + H)
    in_proj_b: mx.array,      # (DV + 2*DQ + H,)
    norm_w_q: mx.array,       # (DQ,)
    norm_w_k: mx.array,       # (DQ,)
    out_proj_w: mx.array,     # (DV, D)
    out_proj_b: mx.array,     # (D,)
    angle_w: mx.array | None, # (DQ//2,) — None for SISO
    DQ: int = 64,
    DV: int = 64,
    H: int = 2,
    CS: int = 32,
) -> mx.array:
    B, L, D = x.shape
    dq_head = DQ // H
    dv_head = DV // H

    # ── 1. in_proj ──
    proj = x @ in_proj_w + in_proj_b             # (B, L, DV + 2*DQ + H)
    z     = proj[..., :DV]                        # gate
    q_raw = proj[..., DV:DV+DQ]                   # Q input
    k_raw = proj[..., DV+DQ:DV+2*DQ]             # K input
    dt    = proj[..., DV+2*DQ:]                   # (B, L, H)

    # ── 2. pre_ssm: RMSNorm + rotary ──
    q = rms_norm(q_raw, norm_w_q)
    k = rms_norm(k_raw, norm_w_k)
    v = q_raw  # V from same source, no norm

    # A = softplus(dt), B = softplus(k_proj)
    a = mx.log(1.0 + mx.exp(dt))                 # (B, L, H)
    b = mx.log(1.0 + mx.exp(k.mean(axis=-1)))    # (B, L)
    b = mx.broadcast_to(b[..., None], (B, L, H)) # (B, L, H)

    # ── 3. Rotary angles (SISO: dq_head=1 → skip rotary) ──
    if dq_head >= 2 and angle_w is not None:
        pos = mx.arange(L, dtype=mx.float32)
        freq = angle_w[:dq_head // 2]
        theta = pos[:, None] * freq[None, :]                     # (L, dq_head//2)
        angle = mx.broadcast_to(theta[None, None, :, :], (B, H, L, dq_head // 2))
    else:
        angle = mx.zeros((B, H, L, 1), dtype=mx.float16)         # dummy

    # Reshape for SSM: (B, H, L, dq_head)
    q = q.reshape(B, L, H, dq_head).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, H, dq_head).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, H, dv_head).transpose(0, 2, 1, 3)
    a = a.transpose(0, 2, 1)  # (B, H, L)
    b = b.transpose(0, 2, 1)

    # ── 4. SSM ──
    ssm_out = mamba3_ssm(q, k, v, a, b, angle, chunk_size=CS)  # (B, H, L, dv_head)

    # ── 5. post_ssm: silu gate ──
    z_h = z.reshape(B, L, H, dv_head).transpose(0, 2, 1, 3)
    gated = ssm_out * (z_h * mx.sigmoid(z_h))

    # ── 6. out_proj ──
    gated_flat = gated.transpose(0, 2, 1, 3).reshape(B, L, DV)
    out = gated_flat @ out_proj_w + out_proj_b

    return out


def _make_block_inputs(B, L, D, DQ, DV, H):
    """Create random weights for a mamba3 block."""
    proj_dim = DV + 2 * DQ + H
    return (
        mx.random.normal((B, L, D), dtype=mx.float16) * 0.5,
        mx.random.normal((D, proj_dim), dtype=mx.float16) * 0.02,
        mx.random.normal((proj_dim,), dtype=mx.float16) * 0.01,
        mx.ones((DQ,), dtype=mx.float16),                    # norm_q
        mx.ones((DQ,), dtype=mx.float16),                    # norm_k
        mx.random.normal((DV, D), dtype=mx.float16) * 0.02,  # out_proj_w
        mx.random.normal((D,), dtype=mx.float16) * 0.01,     # out_proj_b
        mx.ones((DQ // 2,), dtype=mx.float16) if DQ // 2 > 0 else None,  # angle_w
    )
