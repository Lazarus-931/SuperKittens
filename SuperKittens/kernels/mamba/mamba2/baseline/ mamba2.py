"""
Mamba-2 full block — MLX reference.

Architecture:
  in_proj → split(z, xBC) → conv1d + SiLU → split(x, B, C)
  → dt = softplus(proj(x)) → selective_scan → gate(silu(z)) → out_proj

Calls ssm.selective_scan for the SSM recurrence.
"""

import mlx.core as mx
import math
from ssm import selective_scan


def mamba2_block(
    x: mx.array,                   # (B, L, D)
    in_proj_w: mx.array,           # (D, 2*E + 2*H*N)
    out_proj_w: mx.array,          # (E, D)
    conv_weight: mx.array,         # (E, 4) depthwise conv, kernel=4
    conv_bias: mx.array | None,    # (E,)
    A_log: mx.array,               # (H, N) state transition
    D: mx.array,                   # (E,) skip connection
    dt_proj_w: mx.array,           # (E//H, 1) per-head dt projection
    dt_bias: mx.array | None,      # (H,) dt bias
    norm_weight: mx.array | None,  # (E,) optional RMSNorm
) -> mx.array:
    B, L, D_model = x.shape
    E = out_proj_w.shape[0]
    H = A_log.shape[0]
    N = A_log.shape[1]
    head_dim = E // H

    # ── 1. in_proj ──
    proj = x @ in_proj_w                       # (B, L, 2*E + 2*H*N)
    z    = proj[..., :E]                        # gate
    xBC  = proj[..., E:]                        # conv + SSM input

    # ── 2. Split xBC → x (SSM input), B, C ──
    x_raw = xBC[..., :E]                        # (B, L, E) SSM input
    B_c   = xBC[..., E:E+H*N]                   # (B, L, H*N)
    C_c   = xBC[..., E+H*N:]                    # (B, L, H*N)

    # ── 3. Depthwise causal conv1d on x only + SiLU ──
    d_conv = conv_weight.shape[1]
    x_conv = mx.zeros_like(x_raw)
    for k in range(d_conv):
        shifted = mx.pad(x_raw, [(0, 0), (k, 0), (0, 0)])[:, :L]
        x_conv = x_conv + shifted * conv_weight[None, None, :, k]
    if conv_bias is not None:
        x_conv = x_conv + conv_bias[None, None, :]
    x_bc = x_conv * mx.sigmoid(x_conv)           # SiLU

    B_arr = B_c.reshape(B, L, H, N)             # (B, L, H, N)
    C_arr = C_c.reshape(B, L, H, N)             # (B, L, H, N)
    x_arr = x_bc.reshape(B, L, H, head_dim)     # (B, L, H, head_dim)

    # ── 4. dt = softplus(linear(x) + bias) ──
    dt_raw = (x_arr @ dt_proj_w).squeeze(-1)    # (B, L, H)
    if dt_bias is not None:
        dt_raw = dt_raw + dt_bias[None, None, :]
    dt = mx.log(1.0 + mx.exp(dt_raw))           # softplus

    # ── 5. Selective scan via ssm.py ──
    # Mapping: Q=C (output), K=B (input), V=x (values)
    # A_log for ssm: per-head dt * mean(A_log)
    A_effective = dt * A_log[None, None, :, :].mean(axis=-1)  # (B, L, H)
    ssm_out = selective_scan(C_arr, B_arr, x_arr, A_effective)  # (B, L, H, head_dim)

    # ── 6. Gate: silu(z) * ssm_out ──
    z_h = z.reshape(B, L, H, head_dim)
    gated = ssm_out * (z_h * mx.sigmoid(z_h))

    # ── 7. Optional RMSNorm ──
    if norm_weight is not None:
        gated_flat = gated.reshape(B, L, E)
        rrms = mx.rsqrt((gated_flat ** 2).mean(axis=-1, keepdims=True) + 1e-6)
        gated_flat = gated_flat * rrms * norm_weight[None, None, :]
        gated = gated_flat.reshape(B, L, H, head_dim)

    # ── 8. out_proj + skip ──
    gated_flat = gated.reshape(B, L, E)
    out = gated_flat @ out_proj_w                   # (B, L, D)

    # D skip connection: D is (E,) → reshape to (H, head_dim)
    D_h = D.reshape(H, head_dim)                      # (H, head_dim)
    skip = x_arr * D_h[None, None, :, :]              # broadcast over B,L
    skip_flat = skip.reshape(B, L, E)
    out = out + skip_flat @ out_proj_w

    return out


def bench_block(L=128, D=128, E=64, H=2, N=64, iters=20):
    """Quick smoke test for the full block."""
    B = 1
    head_dim = E // H  # = 32

    x = mx.random.normal((B, L, D), dtype=mx.float16) * 0.5
    in_proj_w  = mx.random.normal((D, 2*E + 2*H*N), dtype=mx.float16) * 0.02
    out_proj_w = mx.random.normal((E, D), dtype=mx.float16) * 0.02
    conv_w     = mx.random.normal((E, 4), dtype=mx.float16) * 0.1
    conv_b     = mx.random.normal((E,), dtype=mx.float16) * 0.01
    A_log      = mx.random.normal((H, N), dtype=mx.float32) * 0.01
    D_skip     = mx.ones((E,), dtype=mx.float16) * 0.5
    dt_proj_w  = mx.random.normal((head_dim, 1), dtype=mx.float16) * 0.1
    dt_b       = mx.random.normal((H,), dtype=mx.float16) * 0.01
    norm_w     = mx.ones((E,), dtype=mx.float16)
    mx.eval(x, in_proj_w, out_proj_w, conv_w, conv_b, A_log, D_skip, dt_proj_w, dt_b, norm_w)

    for _ in range(5):
        mx.eval(mamba2_block(x, in_proj_w, out_proj_w, conv_w, conv_b,
                             A_log, D_skip, dt_proj_w, dt_b, norm_w))

    mx.synchronize()
    import time, statistics
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(mamba2_block(x, in_proj_w, out_proj_w, conv_w, conv_b,
                             A_log, D_skip, dt_proj_w, dt_b, norm_w))
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    med = statistics.median(times)
    print(f"MLX Mamba-2 full block: L={L} D={D} E={E} H={H} N={N}")
    print(f"  median={med:.3f}ms")
    return med

if __name__ == "__main__":
    bench_block()
