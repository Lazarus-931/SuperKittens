"""gemma4_unified.py — adapter for google/gemma-4-12B-it (model_type "gemma4_unified").

Structurally distinct from the E2B/E4B "gemma4" arch (see temp/gemma4-unified/STATUS.md
for the full delta table). The unified text model is a dense gemma transformer:

  * NO PLE (hidden_size_per_layer_input=0), NO KV-sharing.
  * 16 Q heads. Sliding layers: 8 KV heads, head_dim 256. Full layers: 1 KV head,
    head_dim 512 (full_attention at L%6==5).
  * FULL RoPE on full-attention layers (the proportional rotary spans the whole
    global head_dim) — the E-variant uses partial 0.25.
  * Per-layer learned output scale `layer_scalar` applied at the END of every layer
    (E-variant only applies it inside PLE inject).
  * V RMSNorm without scale, 4-norm sandwich, embed-scale sqrt(d_model), final
    logit softcap 30 — all already covered by the shared gemma4 kernels.

CLEAN DIVISION: this adapter reuses the shared gemma4 C launcher + kernel library and
subclasses the shared :class:`Gemma4` handle, but does NOT mutate the E-variant adapter
and adds no cross-arch conditionals to the E2B/E4B path. The unified-only behaviour is
driven entirely by config flags (full_rope_global, apply_layer_scalar) the launcher reads.

Fit: the HF bf16 safetensors are 23.9 GB and do not fit a 16 GB mini; the canonical
artifact is a Q4_K_M GGUF (~7.4 GB on disk), loaded via sk_gemma4_load_gguf which
dequantizes the K-quant body to bf16 (and re-packs Q8 body when SK_GEMMA4_BODY_Q8 is on).
"""
from __future__ import annotations
import ctypes, json
import numpy as np
from pathlib import Path

from SuperKittens.models.gemma.gemma4.gemma4 import (
    Gemma4, Gemma4Config, _bake_rope, _load,
)


def unified_12b_config() -> Gemma4Config:
    # Verified against google/gemma-4-12B-it config.json (text_config) and the
    # Q4_K_M GGUF metadata + per-layer tensor shapes.
    return Gemma4Config(
        n_layers=48, local_period=6,
        d_model=3840, n_int=15360,
        n_heads=16,
        n_kv_heads_local=8, n_kv_heads_global=1,
        head_dim_local=256, head_dim_global=512,
        vocab_size=262144,
        window=1024,
        has_ple=False,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        eps=1e-6,
        rope_theta_global=1_000_000.0,
        rope_theta_local=10_000.0,
        final_logit_softcap=30.0,
        # HF "proportional" rope on global layers rotates only the first
        # partial_rotary_factor*head_dim = 0.25*512 = 128 dims (rope_angles=64
        # nonzero inv_freq pairs; the rest are zeroed). inv_freq there is
        # 1/(1e6^(arange(0,128,2)/512)) == 1/(1e6^(arange(0,64)/256)), which is
        # exactly the first 64 entries of _bake_rope(512, 1e6). So the E-variant
        # default (rot_dims = head_dim*0.25 = 128) is HF-correct; full_rope_global
        # (rotate all 512) over-rotates and breaks coherence.
        full_rope_global=0,
        apply_layer_scalar=1,
    )


class Gemma4Unified(Gemma4):
    """Stateful gemma4_unified (12B) inference handle. Reuses the shared gemma4 core."""

    def load_gguf(self, path: str) -> None:
        lib = _load()
        fn = getattr(lib, "sk_gemma4_load_gguf", None)
        if fn is None:
            raise RuntimeError("libsk.dylib has no sk_gemma4_load_gguf symbol; rebuild dylib")
        rc = fn(self._h, str(path).encode())
        if rc:
            raise RuntimeError(f"sk_gemma4_load_gguf failed: {rc}")

    @classmethod
    def from_spec(cls, spec, **overrides) -> "Gemma4Unified":
        sk_root = Path(__file__).resolve().parents[4]
        snap = Path(overrides.pop("snapshot", None)
                    or (sk_root / "SuperKittens" / "model_weights" / spec.weight_dir))

        cfg = unified_12b_config()
        # Prefer on-disk config.json (text_config) when present; the GGUF-only
        # snapshots won't have it, so the verified preset stands in.
        cfg_path = snap / "config.json"
        if cfg_path.exists():
            hf = json.loads(cfg_path.read_text())
            t = hf.get("text_config", hf) if isinstance(hf.get("text_config"), dict) else hf
            def g(k, d):
                return t.get(k, hf.get(k, d))
            cfg.n_layers   = int(g("num_hidden_layers", cfg.n_layers))
            cfg.d_model    = int(g("hidden_size", cfg.d_model))
            cfg.n_int      = int(g("intermediate_size", cfg.n_int))
            cfg.n_heads    = int(g("num_attention_heads", cfg.n_heads))
            cfg.n_kv_heads_local  = int(g("num_key_value_heads", cfg.n_kv_heads_local))
            cfg.n_kv_heads_global = int(g("num_global_key_value_heads", cfg.n_kv_heads_global))
            cfg.head_dim_local    = int(g("head_dim", cfg.head_dim_local))
            cfg.head_dim_global   = int(g("global_head_dim", cfg.head_dim_global))
            cfg.window     = int(g("sliding_window", cfg.window))
            cfg.vocab_size = int(g("vocab_size", cfg.vocab_size))
            cfg.eps        = float(g("rms_norm_eps", cfg.eps))
            cfg.final_logit_softcap = float(g("final_logit_softcapping", cfg.final_logit_softcap) or 0.0)
            lt = g("layer_types", None)
            if isinstance(lt, list) and lt:
                cfg.layer_types = tuple(lt)

        for k in ("seq_max", "cache_max"):
            if k in overrides:
                setattr(cfg, k, overrides.pop(k))
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        m = cls(cfg)

        # Load weights: GGUF if present (canonical fit-16GB artifact), else
        # safetensors (HF tensor names) for the bf16 path.
        gguf = None
        if spec.gguf_name:
            cand = snap / spec.gguf_name
            if not cand.exists():
                alt = sk_root / "SuperKittens" / "model_weights" / spec.gguf_name
                cand = alt if alt.exists() else cand
            gguf = cand
        if gguf is None or not gguf.exists():
            # accept any *.gguf in the snapshot dir
            gg = sorted(snap.glob("*.gguf")) if snap.exists() else []
            if gg:
                gguf = gg[0]

        if gguf and gguf.exists():
            m.load_gguf(str(gguf))
        else:
            idx = snap / "model.safetensors.index.json"
            single = snap / "model.safetensors"
            target = idx if idx.exists() else single
            if not target.exists():
                raise FileNotFoundError(f"no GGUF or safetensors in {snap}")
            rc = (_load().sk_gemma4_load_safetensors_index(m._h, str(idx).encode())
                  if idx.exists()
                  else _load().sk_gemma4_load_safetensors(m._h, str(single).encode()))
            if rc:
                raise RuntimeError(f"gemma4_unified safetensors load failed: {rc}")

        # Dual-theta RoPE tables (local theta 1e4 over head_dim_local; global
        # theta 1e6 over head_dim_global — full rotary).
        cos_l, sin_l = _bake_rope(cfg.cache_max, cfg.head_dim_local,  cfg.rope_theta_local)
        cos_g, sin_g = _bake_rope(cfg.cache_max, cfg.head_dim_global, cfg.rope_theta_global)
        m.set_rope_tables(cos_l, sin_l, cos_g, sin_g)

        m.tokenizer = None
        if spec.tokenizer_family:
            try:
                from SuperKittens.models.load.tokenizer import Tokenizer
                jp = snap / "tokenizer.json"
                sp = snap / "tokenizer.model"
                if jp.exists():
                    m.tokenizer = Tokenizer.from_hf_json(str(jp), family=spec.tokenizer_family)
                elif sp.exists():
                    m.tokenizer = Tokenizer.from_sentencepiece(str(sp), family=spec.tokenizer_family)
            except Exception as e:
                print(f"[gemma4_unified] tokenizer attach failed: {e}")
        return m
