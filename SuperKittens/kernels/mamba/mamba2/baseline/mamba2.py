"""
Mamba-2 — MLX reference.

Architecture:
  in_proj → split(z, xBC) → conv1d + SiLU → split(x, B, C)
  → dt = softplus(proj(x)) → selective_scan → gate(silu(z)) → out_proj + skip
"""

import mlx.core as mx
from ssm import selective_scan


class Mamba2:
    """Single Mamba-2 block. All weights as named mlx arrays."""

    def __init__(self, *,
                 d_model:  int = 128,
                 expand:   int = 64,
                 d_state:  int = 64,
                 n_heads:  int = 2,
                 conv_kernel: int = 4):
        self.D    = d_model
        self.E    = expand
        self.N    = d_state
        self.H    = n_heads
        self.K    = conv_kernel
        self.head_dim = expand // n_heads
        self.proj_dim = 2 * expand + 2 * n_heads * d_state

        
        self.in_proj:  mx.array = None   # (D, proj_dim)
        self.out_proj: mx.array = None   # (E, D)
        self.conv_w:   mx.array = None   # (E, K) depthwise
        self.conv_b:   mx.array = None   # (E,)
        self.A_log:    mx.array = None   # (H, N) state transition
        self.D_skip:   mx.array = None   # (E,) skip connection
        self.dt_proj:  mx.array = None   # (head_dim, 1) per-head dt
        self.dt_bias:  mx.array = None   # (H,) dt bias
        self.norm_w:   mx.array = None   # (E,) optional RMSNorm

  

    def from_random(self):
        D, E, N, H, K = self.D, self.E, self.N, self.H, self.K
        hd = self.head_dim
        self.in_proj  = mx.random.normal((D, self.proj_dim), dtype=mx.float16) * 0.02
        self.out_proj = mx.random.normal((E, D), dtype=mx.float16) * 0.02
        self.conv_w   = mx.random.normal((E, K), dtype=mx.float16) * 0.1
        self.conv_b   = mx.random.normal((E,), dtype=mx.float16) * 0.01
        self.A_log    = mx.random.normal((H, N), dtype=mx.float32) * 0.01
        self.D_skip   = mx.ones((E,), dtype=mx.float16) * 0.5
        self.dt_proj  = mx.random.normal((hd, 1), dtype=mx.float16) * 0.1
        self.dt_bias  = mx.random.normal((H,), dtype=mx.float16) * 0.01
        self.norm_w   = mx.ones((E,), dtype=mx.float16)
        return self

   

    def _in_proj(self, x: mx.array) -> tuple:
        """x: (B, L, D) → z(E), xBC(E + 2*H*N)"""
        proj = x @ self.in_proj
        return proj[..., :self.E], proj[..., self.E:]

    def _conv1d(self, xBC: mx.array) -> tuple:
        """Causal conv1d + SiLU on x portion. Returns x(E), B_arr(B,L,H,N), C_arr(B,L,H,N)."""
        B, L, _ = xBC.shape
        E, K, H, N = self.E, self.K, self.H, self.N
        hd = self.head_dim

        x_raw = xBC[..., :E]
        B_c   = xBC[..., E:E+H*N]
        C_c   = xBC[..., E+H*N:]

   
        x_conv = mx.zeros_like(x_raw)
        for k in range(K):
            shifted = mx.pad(x_raw, [(0,0), (k,0), (0,0)])[:, :L]
            x_conv = x_conv + shifted * self.conv_w[None, None, :, k]
        if self.conv_b is not None:
            x_conv = x_conv + self.conv_b[None, None, :]
        x_silu = x_conv * mx.sigmoid(x_conv)

        return (x_silu.reshape(B, L, H, hd),
                B_c.reshape(B, L, H, N),
                C_c.reshape(B, L, H, N))

    def _dt(self, x_arr: mx.array) -> mx.array:
        """dt = softplus(linear(x) + bias) → (B, L, H)"""
        dt_raw = (x_arr @ self.dt_proj).squeeze(-1)
        if self.dt_bias is not None:
            dt_raw = dt_raw + self.dt_bias[None, None, :]
        return mx.log(1.0 + mx.exp(dt_raw))

    def _ssm(self, C_arr, B_arr, x_arr, dt):
        A_eff = dt * self.A_log[None, None, :, :].mean(axis=-1)
        return selective_scan(C_arr, B_arr, x_arr, A_eff)

    def _gate(self, z, ssm_out):
        B, L, _ = z.shape
        H, hd = self.H, self.head_dim
        z_h = z.reshape(B, L, H, hd)
        gated = ssm_out * (z_h * mx.sigmoid(z_h))
        return gated.reshape(B, L, self.E)

    def _out_proj(self, gated, x_arr):
        out = gated @ self.out_proj
        D_h = self.D_skip.reshape(self.H, self.head_dim)
        skip = (x_arr * D_h[None, None, :, :]).reshape(x_arr.shape[0], x_arr.shape[1], self.E)
        return out + skip @ self.out_proj

    

    def forward(self, x: mx.array) -> mx.array:
        """x: (B, L, D) → output: (B, L, D)"""
        z, xBC = self._in_proj(x)
        x_arr, B_arr, C_arr = self._conv1d(xBC)
        dt = self._dt(x_arr)
        ssm_out = self._ssm(C_arr, B_arr, x_arr, dt)
        gated = self._gate(z, ssm_out)
        # optional
        if self.norm_w is not None:
            rrms = mx.rsqrt((gated ** 2).mean(axis=-1, keepdims=True) + 1e-6)
            gated = gated * rrms * self.norm_w[None, None, :]
        return self._out_proj(gated, x_arr)

    @property
    def param_count(self) -> int:
        return (self.D * self.proj_dim +
                self.E * self.D +
                self.E * self.K + self.E +
                self.H * self.N +
                self.E + self.head_dim + self.H + self.E)
