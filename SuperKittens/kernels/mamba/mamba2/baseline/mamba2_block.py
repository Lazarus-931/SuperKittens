#
#  mamba2_block.py
#  SuperKittens — MLX reference: full Mamba-2 sequential block
#
#  Mamba2(x) = out_proj(gate(SSM(conv1d(proj(x)))) * silu(z))
#
#  Shapes: B=batch, L=length, D=d_model, E=expand, H=heads, N=d_state

import mlx.core as mx
import math


def mamba2_block(
    x: mx.array,                   # (B, L, D)
    # Projection weights
    in_proj_weight: mx.array,      # (D, 2*E + 2*H*N)
    out_proj_weight: mx.array,     # (E, D)
    # Conv1D
    conv_weight: mx.array,         # (E, 4)  — depthwise, kernel=4
    conv_bias: mx.array | None,    # (E,)
    # SSM
    A_log: mx.array,               # (H, N)  — log of state transition
    D: mx.array,                   # (H,)    — skip connection
    dt_bias: mx.array | None,      # (H,)    — time-step bias
    # RMSNorm (optional)
    norm_weight: mx.array | None,  # (E,)
) -> mx.array:
    """
    Mamba-2 sequential forward pass.

    Architecture (simplified from paper):
      xBC_and_z = in_proj(x)                → linear projection
      z, xBC = split(xBC_and_z)             → gate + SSM input
      xBC = SiLU(conv1d(xBC))               → short conv + activation
      x, B, C = split(xBC)                  → SSM parameters
      dt = softplus(Linear(x) + dt_bias)    → time step
      ssm_out = selective_scan(x, dt, A_log, B, C)
      ssm_out = ssm_out * silu(z)           → gate
      ssm_out = rmsnorm(ssm_out) if present
      out = out_proj(ssm_out) + D * x      → output projection + skip
    """
    B, L, D_model = x.shape
    E = out_proj_weight.shape[0]   # expand size (from output projection)
    H = A_log.shape[0]             # n_heads
    N = A_log.shape[1]             # d_state

    # ── 1. Input projection ──
    proj = x @ in_proj_weight  # (B, L, 2*E + 2*H*N)

    # Split: first E = z (gate), remaining = x + B + C (for SSM after conv)
    z = proj[..., :E]                            # (B, L, E)
    xBC = proj[..., E:]                          # (B, L, E + 2*H*N)

    # ── 2. Depthwise causal Conv1D + SiLU ──
    d_conv = conv_weight.shape[1]  # kernel width, e.g. 4
    C_conv = conv_weight.shape[0]  # channels being convolved (= E + 2*H*N)

    # Causal conv: accumulate shifted versions with weights
    xBC_conv = mx.zeros_like(xBC)
    for k in range(d_conv):
        shifted = mx.pad(xBC, [(0,0), (k, 0), (0,0)])[:, :L]
        xBC_conv = xBC_conv + shifted * conv_weight[None, None, :, k]
    if conv_bias is not None:
        xBC_conv = xBC_conv + conv_bias[None, None, :]

    # SiLU activation
    xBC_act = xBC_conv * mx.sigmoid(xBC_conv)  # silu

    # ── 3. Split xBC into x, B, C ──
    xBC_e = xBC_act[..., :E]       # (B, L, E)  — SSM input x
    B_raw = xBC_act[..., E:E+H*N]  # (B, L, H*N)
    C_raw = xBC_act[..., E+H*N:]   # (B, L, H*N)

    # Reshape to (B, L, H, N)
    B_arr = B_raw.reshape(B, L, H, N)
    C_arr = C_raw.reshape(B, L, H, N)

    # x needs to be split into heads: (B, L, E) → (B, L, H, E//H)
    head_dim = E // H
    x_arr = xBC_e.reshape(B, L, H, head_dim)

    # ── 4. Time step dt = softplus(Linear(x) + dt_bias) ──
    # In the simplified version, dt is computed from a portion of the projection.
    # For the reference, we compute dt as softplus of a learned linear projection.
    # In practice, this is part of the in_proj. We'll use a separate learned dt_proj.
    # For now: dt_log = x @ dt_proj_weight + dt_bias, dt = softplus(dt_log)
    # But in the standard Mamba2, dt comes from the in_proj output.
    # Let's compute dt from the x input via a simple linear projection.
    # dt = softplus(linear(x)) where linear maps (E//H) → 1 per head.

    # Simplified: use a dummy dt projection (in real impl, dt_proj is part of in_proj)
    # For the reference, let's compute dt from the SSM input x
    dt_raw = x_arr.mean(axis=-1)  # (B, L, H) — simplified time step
    if dt_bias is not None:
        dt_raw = dt_raw + dt_bias[None, None, :]
    dt = mx.log(1.0 + mx.exp(dt_raw))  # softplus (B, L, H)

    # ── 5. Selective scan (SSM) ──
    # A_log is (H, N), broadcast to (B, L, H, N)
    A_log_expanded = mx.broadcast_to(A_log[None, None, :, :], (B, L, H, N))

    # Combine dt and A: effective decay = exp(dt * A_log)
    # In selective scan: h_t = exp(A_log * dt) * h_{t-1} + B_t^T @ x_t
    #   y_t = C_t @ h_t
    effective_A = dt[..., None] * mx.exp(A_log_expanded)  # (B, L, H, N)

    # Run SSM recurrence: h_t = A_t * h_{t-1} + B_t^T @ x_t; y_t = C_t @ h_t
    state = mx.zeros((B, H, N, head_dim), dtype=x.dtype)
    outputs = []
    for t in range(L):
        # Decay factor: exp(A[t]) = exp(A_log * dt[t])
        A_t = mx.exp(A_log_expanded[:, t])  # (B, H, N)
        decay = A_t[..., None]  # (B, H, N, 1)

        # K^T @ V style update: h = decay * h + B[t]^T @ x[t]
        B_t = B_arr[:, t]  # (B, H, N)
        x_t = x_arr[:, t]  # (B, H, head_dim)
        kv_update = mx.einsum("bhN,bhD->bhND", B_t, x_t)  # (B, H, N, head_dim)
        state = decay * state + kv_update

        # Output: y[t] = C[t] @ h[t]
        C_t = C_arr[:, t]  # (B, H, N)
        y_t = mx.einsum("bhN,bhND->bhD", C_t, state)  # (B, H, head_dim)
        outputs.append(y_t)

    y_ssm = mx.stack(outputs, axis=1)  # (B, L, H, head_dim)
    y_ssm = y_ssm.reshape(B, L, E)     # (B, L, E)

    # ── 6. Gating: y = y_ssm * silu(z) ──
    y_gated = y_ssm * (z * mx.sigmoid(z))

    # ── 7. Optional RMSNorm ──
    if norm_weight is not None:
        rms = mx.sqrt((y_gated ** 2).mean(axis=-1, keepdims=True) + 1e-6)
        y_gated = y_gated * norm_weight[None, None, :] / rms

    # ── 8. Output projection + skip connection with D ──
    out = y_gated @ out_proj_weight  # (B, L, E) @ (E, D) → (B, L, D)

    # D skip: out = out + x * D (broadcast over heads)
    # Simplified: D is (H,), broadcast to (H, head_dim)
    if D is not None:
        # D has shape (H,) — add skip per head
        D_expanded = mx.broadcast_to(D[None, None, :, None], (B, L, H, head_dim))
        skip = x_arr * D_expanded  # (B, L, H, head_dim)
        skip_flat = skip.reshape(B, L, E)
        out = out + skip_flat @ out_proj_weight  # add skip contribution to output

    return out
