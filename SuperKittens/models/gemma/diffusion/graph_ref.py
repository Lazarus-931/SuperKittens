# pyright: reportMissingImports=false
"""graph_ref.py — host math shared by the CPU reference forward and the Metal
driver's glue, plus the full CPU-f32 reference forward.

Every op mirrors llama.cpp PR #24423 (src/models/diffusion-gemma.cpp +
gemma4-common.h + llm_graph_context::build_moe_ffn at c84e85af). The CPU
forward is the parity bisect oracle: dump(name, il, arr) taps match the GPU
driver's taps one-for-one, so first-divergence is a numpy diff per layer.
"""
from __future__ import annotations

import numpy as np

from .config import DiffusionGemmaConfig
from .gguf_io import GGUFFile

F32 = np.float32


# -- primitive ops (all f32 in / f32 out) -------------------------------------

def rms_norm(x: np.ndarray, eps: float, w: np.ndarray | None = None) -> np.ndarray:
    inv = 1.0 / np.sqrt((x.astype(F32) ** 2).mean(axis=-1, keepdims=True) + F32(eps))
    y = x * inv
    if w is not None:
        y = y * w
    return y.astype(F32)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    # ggml_gelu / HF gelu_pytorch_tanh
    x = x.astype(F32)
    return (0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))).astype(F32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp((x - m).astype(F32))
    return (e / e.sum(axis=axis, keepdims=True)).astype(F32)


def rope_neox(x: np.ndarray, pos: np.ndarray, base: float,
              freq_factors: np.ndarray | None) -> np.ndarray:
    """x [T, H, D] -> rotated. ggml NEOX: pair (j, j+D/2), angle =
    pos * base^(-2j/D) / ff[j]. Full-dim rotation (n_rot == head_dim here)."""
    T, H, D = x.shape
    j = np.arange(D // 2, dtype=np.float64)
    inv = np.power(float(base), -2.0 * j / D)
    if freq_factors is not None:
        inv = inv / freq_factors.astype(np.float64)
    ang = pos.astype(np.float64)[:, None] * inv[None, :]        # [T, D/2]
    c = np.cos(ang).astype(F32)[:, None, :]
    s = np.sin(ang).astype(F32)[:, None, :]
    x0 = x[..., :D // 2]
    x1 = x[..., D // 2:]
    out = np.empty_like(x, dtype=F32)
    out[..., :D // 2] = x0 * c - x1 * s
    out[..., D // 2:] = x0 * s + x1 * c
    return out


def build_mask(P: int, C: int, swa: bool, n_swa: int, n_cols: int | None = None) -> np.ndarray:
    """Additive mask [P+C, n_cols] f32 (0 / -inf), PR rules. Prompt queries:
    causal over prompt only (SWA-clipped when swa). Canvas queries:
    bidirectional — global sees all; SWA sees last (n_swa-1) prompt + canvas.
    Columns >= P+C (pad) stay -inf."""
    N = P + C
    n_cols = n_cols or N
    mask = np.full((N, n_cols), -np.inf, dtype=F32)
    q = np.arange(N)[:, None]
    k = np.arange(N)[None, :]
    q_canvas = q >= P
    k_canvas = k >= P
    canvas_prompt_lo = P - n_swa + 1
    if swa:
        allow_canvas_q = k_canvas | (k >= canvas_prompt_lo)
    else:
        allow_canvas_q = np.ones((1, N), dtype=bool)
    allow_prompt_q = (~k_canvas) & (k <= q)
    if swa:
        allow_prompt_q = allow_prompt_q & (q - k < n_swa)  # is_masked_swa STANDARD
    allow = np.where(q_canvas, allow_canvas_q, allow_prompt_q)
    mask[:, :N][allow] = 0.0
    return mask


# -- weight access -------------------------------------------------------------

class Weights:
    """Thin named access over the GGUF (dequant cached only for small F32)."""

    def __init__(self, gg: GGUFFile):
        self.gg = gg
        self._f32: dict[str, np.ndarray] = {}

    def f32(self, name: str) -> np.ndarray:
        a = self._f32.get(name)
        if a is None:
            ti = self.gg.tensors[name]
            assert ti.type_name == "F32", name
            a = self.gg.dequant(name)
            self._f32[name] = a
        return a

    def dq(self, name: str, rows=None) -> np.ndarray:
        return self.gg.dequant(name, rows)


# -- region-aware unified forward (CPU f32 reference) -------------------------

def embed_tokens(w: Weights, cfg: DiffusionGemmaConfig, ids: np.ndarray, P: int) -> np.ndarray:
    x = w.dq("token_embd.weight", rows=np.asarray(ids, np.int64))
    x = x * F32(np.sqrt(F32(cfg.d_model)))
    x[P:] = rms_norm(x[P:], cfg.eps)        # canvas rows: rmsnorm no-scale (zero-SC)
    return x.astype(F32)


def moe_route(w: Weights, cfg: DiffusionGemmaConfig, attn_out: np.ndarray, il: int):
    """Router (operates on the UNNORMED residual): rms_noscale -> /sqrt(d) ->
    * gate_inp scale -> logits -> softmax -> top-8 -> renorm weights."""
    t = rms_norm(attn_out, cfg.eps)
    t = t * F32(1.0 / np.sqrt(F32(cfg.d_model)))
    t = t * w.f32(f"blk.{il}.ffn_gate_inp.scale")
    logits = t @ w.f32(f"blk.{il}.ffn_gate_inp.weight").T          # [T, 128]
    probs = softmax(logits)
    sel = np.argsort(-probs, axis=-1, kind="stable")[:, :cfg.n_expert_used]  # [T, 8]
    wts = np.take_along_axis(probs, sel, axis=-1)
    wts = wts / np.maximum(wts.sum(-1, keepdims=True), F32(6.103515625e-5))
    return sel, wts.astype(F32)


def forward_cpu(gg: GGUFFile, cfg: DiffusionGemmaConfig, ids: np.ndarray,
                P: int, dump=None) -> np.ndarray:
    """Unified [prompt|canvas] zero-SC forward; returns canvas logits f32 [C, V]."""
    w = Weights(gg)
    N = len(ids)
    C = N - P
    pos = np.arange(N, dtype=np.int64)
    dmp = dump or (lambda name, il, arr: None)

    x = embed_tokens(w, cfg, np.asarray(ids), P)
    dmp("inp_region", -1, x)

    rope_ff = w.f32("rope_freqs.weight")

    for il in range(cfg.n_layers):
        swa = cfg.is_swa[il]
        hd = cfg.head_dim(il)
        n_kv = cfg.n_kv_heads[il]
        base, use_ff = cfg.rope_params(il)
        ff = rope_ff if use_ff else None
        gqa = cfg.n_heads // n_kv

        h = rms_norm(x, cfg.eps, w.f32(f"blk.{il}.attn_norm.weight"))

        q = (h @ w.dq(f"blk.{il}.attn_q.weight").T).reshape(N, cfg.n_heads, hd)
        k_raw = h @ w.dq(f"blk.{il}.attn_k.weight").T
        vname = f"blk.{il}.attn_v.weight"
        v_raw = (h @ w.dq(vname).T) if vname in gg.tensors else k_raw  # global: V = raw k_proj
        k = k_raw.reshape(N, n_kv, hd)
        v = v_raw.reshape(N, n_kv, hd)

        q = rms_norm(q, cfg.eps, w.f32(f"blk.{il}.attn_q_norm.weight"))
        k = rms_norm(k, cfg.eps, w.f32(f"blk.{il}.attn_k_norm.weight"))
        v = rms_norm(v, cfg.eps)                                  # v-norm: no scale
        q = rope_neox(q, pos, base, ff)
        k = rope_neox(k, pos, base, ff)
        dmp("q_pos", il, q); dmp("k_pos", il, k); dmp("v_normed", il, v)

        mask = build_mask(P, C, swa, cfg.window)
        o = np.empty((N, cfg.n_heads, hd), dtype=F32)
        for hh in range(cfg.n_heads):
            g = hh // gqa
            s = (q[:, hh, :] @ k[:, g, :].T) * F32(cfg.attn_scale) + mask
            a = softmax(s)
            o[:, hh, :] = a @ v[:, g, :]
        attn = o.reshape(N, cfg.n_heads * hd) @ w.dq(f"blk.{il}.attn_output.weight").T
        attn = rms_norm(attn, cfg.eps, w.f32(f"blk.{il}.post_attention_norm.weight"))
        attn_out = (attn + x).astype(F32)
        dmp("attn_out", il, attn_out)

        # dense MLP (shared expert): geglu, then post_ffw_norm_1
        m = rms_norm(attn_out, cfg.eps, w.f32(f"blk.{il}.ffn_norm.weight"))
        g_ = gelu_tanh(m @ w.dq(f"blk.{il}.ffn_gate.weight").T)
        u_ = m @ w.dq(f"blk.{il}.ffn_up.weight").T
        mlp = (g_ * u_) @ w.dq(f"blk.{il}.ffn_down.weight").T
        mlp = rms_norm(mlp, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm_1.weight"))
        dmp("ffn_mlp", il, mlp)

        # MoE
        e_in = rms_norm(attn_out, cfg.eps, w.f32(f"blk.{il}.pre_ffw_norm_2.weight"))
        sel, wts = moe_route(w, cfg, attn_out, il)
        dmp("moe_sel", il, sel.astype(np.int32)); dmp("moe_wts", il, wts)
        down_s = w.f32(f"blk.{il}.ffn_down_exps.scale")
        moe = np.zeros((N, cfg.d_model), dtype=F32)
        gu_ti = gg.tensors[f"blk.{il}.ffn_gate_up_exps.weight"]
        dn_ti = gg.tensors[f"blk.{il}.ffn_down_exps.weight"]
        for e in np.unique(sel):
            tok, slot = np.nonzero(sel == e)
            r0 = e * 2 * cfg.n_ff_exp
            wgu = w.dq(gu_ti.name, rows=slice(r0, r0 + 2 * cfg.n_ff_exp))
            gu = e_in[tok] @ wgu.T                                # [m, 1408]
            act = gelu_tanh(gu[:, :cfg.n_ff_exp]) * gu[:, cfg.n_ff_exp:]
            r0 = e * cfg.d_model
            wdn = w.dq(dn_ti.name, rows=slice(r0, r0 + cfg.d_model))
            d_ = (act @ wdn.T) * down_s[e]                        # [m, 2816]
            moe[tok] += d_ * wts[tok, slot][:, None]
        moe = rms_norm(moe, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm_2.weight"))
        dmp("ffn_moe", il, moe)

        f = rms_norm(mlp + moe, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm.weight"))
        cur = (f + attn_out).astype(F32)

        enc_s = w.f32(f"blk.{il}.enc_layer_output_scale.weight")[0]
        dec_s = w.f32(f"blk.{il}.layer_output_scale.weight")[0]
        cur[:P] *= enc_s
        cur[P:] *= dec_s
        dmp("l_out", il, cur)
        x = cur

    x = rms_norm(x, cfg.eps, w.f32("output_norm.weight"))
    dmp("result_norm", -1, x)

    # tied head on canvas rows only, vocab-chunked (full f32 dequant is ~3 GB)
    xc = x[P:]
    V = cfg.vocab_size
    logits = np.empty((C, V), dtype=F32)
    step = 16384
    for v0 in range(0, V, step):
        wchunk = w.dq("token_embd.weight", rows=slice(v0, min(v0 + step, V)))
        logits[:, v0:v0 + wchunk.shape[0]] = xc @ wchunk.T
    cap = F32(cfg.final_logit_softcap)
    logits = (np.tanh(logits / cap) * cap).astype(F32)
    dmp("result_output", -1, logits)
    return logits
