"""
Mamba-3 — MLX reference (SISO + MIMO).

SISO: DQ=H, DV=H  — scalar per head, no rotary
MIMO: DQ>H, DV>H  — multi-dimensional per head, rotary applied
"""

import mlx.core as mx
from ssm import mamba3_ssm


class Mamba3:
    """Single Mamba-3 block. All weights as named mlx arrays."""

    def __init__(self, *,
                 d_model: int = 128,
                 expand:  int = 64,
                 d_state: int = 64,
                 n_heads: int = 4,
                 chunk:   int = 32):
        self.D    = d_model
        self.DV   = expand
        self.DQ   = d_state
        self.H    = n_heads
        self.CS   = chunk
        self.proj_dim = expand + 2 * d_state + n_heads
        self._dq_head = d_state // n_heads
        self._dv_head = expand // n_heads
        self._siso = self._dq_head == 1

        # Weights — set by from_random() or directly
        self.in_proj:  mx.array = None   # (D, proj_dim)
        self.in_bias:  mx.array = None   # (proj_dim,)
        self.out_proj: mx.array = None   # (DV, D)
        self.out_bias: mx.array = None   # (D,)
        self.gamma_q:  mx.array = None   # (DQ,)
        self.gamma_k:  mx.array = None   # (DQ,)
        self.angle:    mx.array = None   # (DQ//2,) or None for SISO

    

    def from_random(self):
        """Fill weights with random values (fp16-safe scale)."""
        D, DV, DQ, H = self.D, self.DV, self.DQ, self.H
        pd = self.proj_dim
        self.in_proj  = mx.random.normal((D, pd), dtype=mx.float16) * 0.02
        self.in_bias  = mx.zeros((pd,), dtype=mx.float16)
        self.out_proj = mx.random.normal((DV, D), dtype=mx.float16) * 0.02
        self.out_bias = mx.zeros((D,), dtype=mx.float16)
        self.gamma_q  = mx.ones((DQ,), dtype=mx.float16)
        self.gamma_k  = mx.ones((DQ,), dtype=mx.float16)
        self.angle    = mx.ones((DQ//2,), dtype=mx.float16) if not self._siso else None
        return self

 

    def _in_proj(self, x: mx.array) -> tuple:
        """x: (B, L, D) → z(DV), q_raw(DQ), k_raw(DQ), dt(H)"""
        proj = x @ self.in_proj + self.in_bias
        return (proj[..., :self.DV],
                proj[..., self.DV:self.DV+self.DQ],
                proj[..., self.DV+self.DQ:self.DV+2*self.DQ],
                proj[..., self.DV+2*self.DQ:])

    def _pre_ssm(self, q_raw, k_raw, dt):
        """RMSNorm + rotary → Q, K, V, A, B for SSM."""
        B, L, _ = q_raw.shape
        H, dq_head, dv_head = self.H, self._dq_head, self._dv_head

        q = self._rms_norm(q_raw, self.gamma_q)
        k = self._rms_norm(k_raw, self.gamma_k)
        v = q_raw

        a = mx.log(1.0 + mx.exp(dt))                            # softplus (B, L, H)
        b = mx.log(1.0 + mx.exp(k.mean(axis=-1)))               # (B, L)
        b = mx.broadcast_to(b[..., None], (B, L, H))            # (B, L, H)

        angle = self._make_angle(B, L) if not self._siso else mx.zeros((B, H, L, 1), dtype=mx.float16)

        return (
            q.reshape(B, L, H, dq_head).transpose(0, 2, 1, 3),  # (B, H, L, dq_head)
            k.reshape(B, L, H, dq_head).transpose(0, 2, 1, 3),
            v.reshape(B, L, H, dv_head).transpose(0, 2, 1, 3),
            a.transpose(0, 2, 1),                                 # (B, H, L)
            b.transpose(0, 2, 1),
            angle,
        )

    def _ssm(self, q, k, v, a, b, angle):
        return mamba3_ssm(q, k, v, a, b, angle, chunk_size=self.CS)

    def _post_ssm(self, z, ssm_out):
        B, L, _ = z.shape
        H, dv_head = self.H, self._dv_head
        z_h = z.reshape(B, L, H, dv_head).transpose(0, 2, 1, 3)
        gated = ssm_out * (z_h * mx.sigmoid(z_h))
        return gated.transpose(0, 2, 1, 3).reshape(B, L, self.DV)

    def _out_proj(self, gated):
        return gated @ self.out_proj + self.out_bias


    @staticmethod
    def _rms_norm(x, weight, eps=1e-5):
        return x * mx.rsqrt((x * x).mean(axis=-1, keepdims=True) + eps) * weight

    def _make_angle(self, B, L):
        dq_head = self._dq_head
        pos  = mx.arange(L, dtype=mx.float32)
        freq = self.angle[:dq_head // 2]
        theta = pos[:, None] * freq[None, :]                     # (L, dq_head//2)
        return mx.broadcast_to(theta[None, None, :, :], (B, self.H, L, dq_head // 2))

    @property
    def is_siso(self) -> bool:
        return self._siso



    def forward(self, x: mx.array) -> mx.array:
        """x: (B, L, D) → output: (B, L, D)"""
        z, q_raw, k_raw, dt = self._in_proj(x)
        q, k, v, a, b, angle = self._pre_ssm(q_raw, k_raw, dt)
        ssm_out = self._ssm(q, k, v, a, b, angle)
        gated = self._post_ssm(z, ssm_out)
        return self._out_proj(gated)

    @property
    def param_count(self) -> int:
        return (self.D * self.proj_dim + self.proj_dim +
                self.DV * self.D + self.D +
                self.DQ * 2 + (self.DQ // 2 if not self._siso else 0))
