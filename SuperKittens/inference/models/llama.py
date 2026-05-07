"""
LLaMA forward pass — chains SuperKittens kernels end-to-end.

A LLaMA block:
    residual = x
    x = rmsnorm(x, attn_norm_w)
    q, k, v = q_proj(x), k_proj(x), v_proj(x)    # 3 GEMMs
    q, k = rope(q, k)                              # RoPE
    x = attention(q, k, v, causal=True)            # FA / MHA
    x = out_proj(x)                               # GEMM
    x = x + residual                              # residual

    residual = x
    x = rmsnorm(x, ffn_norm_w)
    gate = gate_proj(x)                           # GEMM
    up   = up_proj(x)                             # GEMM
    x = silu(gate) * up                           # SwiGLU gate
    x = down_proj(x)                              # GEMM
    x = x + residual                              # residual
"""

from dataclasses import dataclass
from typing import List
import numpy as np

# ── model config ────────────────────────────────────────────────────

@dataclass
class LLaMAConfig:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8       # GQA
    head_dim: int = 64
    hidden_dim: int = 5632    # intermediate FFN size
    vocab_size: int = 128256
    max_seq_len: int = 2048
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5

    @classmethod
    def llama_1b(cls):
        """LLaMA 3.2 1B"""
        return cls(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8,
                   head_dim=64, hidden_dim=5632)

    @classmethod
    def llama_3b(cls):
        """LLaMA 3.2 3B"""
        return cls(dim=3072, n_layers=28, n_heads=24, n_kv_heads=8,
                   head_dim=128, hidden_dim=8192)

# ── forward pass (dispatch via tensor handles) ──────────────────────

class LLaMABlock:
    """One transformer block. All tensors are GPU handles from sk_tensor_alloc."""

    def __init__(self, layer_idx: int, cfg: LLaMAConfig):
        self.idx = layer_idx
        self.cfg = cfg
        self._weights: dict[str, object] = {}  # handle → tensor handle

    def load_weight(self, name: str, data: np.ndarray):
        """Store weight as numpy for later upload. Call .upload() to send to GPU."""
        self._weights[name] = data

    def forward(self, x_handle, cos_handle, sin_handle):
        """Run one block. All tensors are GPU handles (void*).

        This is the template for the C++ inference loop.
        Currently in Python for readability — move to C++ for perf.
        """
        raise NotImplementedError("Python forward pass is reference only — "
            "the real inference loop runs in C++ via the dylib dispatch functions.")


def llama_forward_pass(cfg: LLaMAConfig, token_ids: List[int]):
    """High-level pseudocode for the inference loop.

    This shows the kernel sequence. Actual implementation dispatches
    via sk_dispatch_* functions in the dylib.

    Phase 1: Prefill (process prompt, populate KV-cache)
    ─────────────────────────────────────────────────
    x = embedding_lookup(token_ids)              # gather: vocab[ids] → (1, L, dim)
    for layer in layers:
        residual = x
        x = rmsnorm(x, attn_norm_w)
        q = gemm(x, q_proj_w)                    # M×K × K×(H*d)
        k = gemm(x, k_proj_w)
        v = gemm(x, v_proj_w)
        q, k = rope(q, k, cos, sin)              # RoPE
        x = attention(q, k, v, causal=True)       # FA d=64 / MHA d=128
        x = gemm(x, out_proj_w)                  # (H*d) × dim
        x = add(x, residual)

        residual = x
        x = rmsnorm(x, ffn_norm_w)
        gate = gemm(x, gate_proj_w)
        up   = gemm(x, up_proj_w)
        x = silu(gate) * up                      # gated activation
        x = gemm(x, down_proj_w)
        x = add(x, residual)

    logits = gemm(x, lm_head_w)                  # dim × vocab
    next_token = argmax(logits[:, -1, :])

    Phase 2: Decode (one token per step, KV-cache hit)
    ──────────────────────────────────────────────────
    for step in range(max_new_tokens):
        x = embedding_lookup([next_token])       # (1, 1, dim)
        for layer in layers:
            ... same as above but with KV-cache append ...
        next_token = sample(logits)
    """
    pass


# ── weight manifest ─────────────────────────────────────────────────

def llama_weight_shapes(cfg: LLaMAConfig) -> dict:
    """Return the shape of every weight tensor for memory planning."""
    d, h, n = cfg.dim, cfg.hidden_dim, cfg.n_layers
    return {
        "tok_embeddings.weight":     (cfg.vocab_size, d),
        **{f"layers.{i}.attention.wq.weight": (cfg.n_heads * cfg.head_dim, d) for i in range(n)},
        **{f"layers.{i}.attention.wk.weight": (cfg.n_kv_heads * cfg.head_dim, d) for i in range(n)},
        **{f"layers.{i}.attention.wv.weight": (cfg.n_kv_heads * cfg.head_dim, d) for i in range(n)},
        **{f"layers.{i}.attention.wo.weight": (d, cfg.n_heads * cfg.head_dim) for i in range(n)},
        **{f"layers.{i}.attention_norm.weight": (d,) for i in range(n)},
        **{f"layers.{i}.feed_forward.w1.weight": (h, d) for i in range(n)},   # gate
        **{f"layers.{i}.feed_forward.w2.weight": (d, h) for i in range(n)},   # down
        **{f"layers.{i}.feed_forward.w3.weight": (h, d) for i in range(n)},   # up
        **{f"layers.{i}.ffn_norm.weight": (d,) for i in range(n)},
        "norm.weight":                (d,),
        "lm_head.weight":             (cfg.vocab_size, d),
    }
