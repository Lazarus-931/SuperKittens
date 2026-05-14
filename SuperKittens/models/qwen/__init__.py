"""Qwen3 model package. Imports register the model with sk.load."""
from __future__ import annotations
import json
from pathlib import Path

from .qwen import Qwen, Config


def _build_cfg_from_snapshot(snap: Path, **overrides) -> Config:
    cfgj = json.loads((snap / "config.json").read_text())
    cfg = Config(
        n_layers   = cfgj["num_hidden_layers"],
        d_model    = cfgj["hidden_size"],
        n_heads    = cfgj["num_attention_heads"],
        n_kv_heads = cfgj["num_key_value_heads"],
        head_dim   = cfgj.get("head_dim", cfgj["hidden_size"] // cfgj["num_attention_heads"]),
        n_int      = cfgj["intermediate_size"],
        vocab_size = cfgj["vocab_size"],
        eps        = cfgj.get("rms_norm_eps", 1e-6),
        rope_freq_base = cfgj.get("rope_theta", 1_000_000.0),
        seq_max    = overrides.pop("seq_max", 128),
        cache_max  = overrides.pop("cache_max", 512),
        tie_word_embeddings = int(cfgj.get("tie_word_embeddings", True)),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


_VARIANT_TO_DIR = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-8b":   "Qwen3-8B",
}
_VARIANT_TO_GGUF = {
    "qwen3-0.6b": "Qwen3-0.6B-Q8_0.gguf",
    "qwen3-8b":   "Qwen3-8B-Q8_0.gguf",
}


def _from_pretrained(variant: str = "qwen3-0.6b", quant: str | None = None,
                     snapshot: str | None = None, gguf_path: str | None = None,
                     **cfg_overrides) -> Qwen:
    spec = variant
    dir_name = _VARIANT_TO_DIR.get(spec.lower(), spec)
    sk_root = Path(__file__).resolve().parents[3]
    snap = Path(snapshot) if snapshot else (sk_root / "SuperKittens" / "model_weights" / dir_name)
    if not snap.exists():
        raise FileNotFoundError(f"snapshot dir not found: {snap}")

    cfg = _build_cfg_from_snapshot(snap, **cfg_overrides)
    m = Qwen(cfg)

    if quant in ("q8_0", "Q8_0", "gguf"):
        gpath = Path(gguf_path) if gguf_path else (
            sk_root / "SuperKittens" / "model_weights" / _VARIANT_TO_GGUF.get(spec.lower(), "Qwen3-0.6B-Q8_0.gguf"))
        if not gpath.exists():
            raise FileNotFoundError(f"gguf file not found: {gpath}")
        m.load_gguf(str(gpath))
    else:
        # fp16 safetensors path
        from .qwen import _load
        idx_path = snap / "model.safetensors.index.json"
        single = snap / "model.safetensors"
        lib = _load()
        if idx_path.exists():
            if not hasattr(lib, "sk_qwen_load_safetensors_index"):
                raise RuntimeError("libsk.dylib has no sk_qwen_load_safetensors_index symbol; rebuild dylib")
            rc = lib.sk_qwen_load_safetensors_index(m._h, str(idx_path).encode())
            if rc:
                raise RuntimeError(f"sk_qwen_load_safetensors_index failed: {rc}")
        elif single.exists():
            rc = lib.sk_qwen_load_safetensors(m._h, str(single).encode())
            if rc:
                raise RuntimeError(f"sk_qwen_load_safetensors failed: {rc}")
        else:
            raise FileNotFoundError(f"no safetensors in {snap}")

    # RoPE tables
    m.bake_and_set_rope()

    # Tokenizer
    try:
        from SuperKittens.models.load.tokenizer import Tokenizer
        json_path = snap / "tokenizer.json"
        sp_path   = snap / "tokenizer.model"
        if json_path.exists():
            m.tokenizer = Tokenizer.from_hf_json(str(json_path), family="qwen3")
        elif sp_path.exists():
            m.tokenizer = Tokenizer.from_sentencepiece(str(sp_path))
        else:
            print(f"[qwen] no tokenizer.json or tokenizer.model in {snap}")
    except Exception as e:
        print(f"[qwen] tokenizer attach failed: {e}")

    return m


# Adapter so MODEL_REGISTRY's cls.from_pretrained(**defaults, **kwargs) routes here.
class _QwenFactory:
    @staticmethod
    def from_pretrained(**kwargs):
        return _from_pretrained(**kwargs)


from SuperKittens.api import register
for _spec in ("qwen3-0.6b", "qwen3-8b"):
    try:
        register(_spec, _QwenFactory, variant=_spec)
    except ValueError:
        pass

__all__ = ["Qwen", "Config"]
