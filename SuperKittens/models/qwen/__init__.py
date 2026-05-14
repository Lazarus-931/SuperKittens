"""Qwen3 model package. Imports register the model with sk.load."""
from __future__ import annotations
import json
from pathlib import Path

from .qwen import Qwen, Config, _load


_QWEN3_VARIANTS = {
    "qwen3-8b": {
        "hf_repo":   "Qwen/Qwen3-8B",
        "weight_dir": "Qwen3-8B-GGUF",
        "gguf_name":  "Qwen3-8B-Q8_0.gguf",
        "default_quant": "q8_0",
        "dims": dict(
            n_layers=36, d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128,
            n_int=12288, vocab_size=151936, eps=1e-6, rope_freq_base=1_000_000.0,
            tie_word_embeddings=0,
        ),
    },
}


def _cfg_from_snapshot(snap: Path, **overrides) -> Config:
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


def _cfg_from_dims(variant: str, **overrides) -> Config:
    dims = dict(_QWEN3_VARIANTS[variant]["dims"])
    dims["seq_max"]   = overrides.pop("seq_max", 128)
    dims["cache_max"] = overrides.pop("cache_max", 512)
    cfg = Config(**dims)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _resolve_tokenizer(snap: Path, variant: str):
    from SuperKittens.models.load.tokenizer import Tokenizer
    for cand in (snap / "tokenizer.json", *snap.glob("tokenizer.json")):
        if cand.exists():
            return Tokenizer.from_hf_json(str(cand), family="qwen3")
    repo = _QWEN3_VARIANTS[variant]["hf_repo"]
    try:
        from huggingface_hub import hf_hub_download
        tok_path = hf_hub_download(repo_id=repo, filename="tokenizer.json")
        return Tokenizer.from_hf_json(tok_path, family="qwen3")
    except Exception as e:
        print(f"[qwen] hf_hub_download tokenizer failed: {e}")
    return None


def _from_pretrained(variant: str = "qwen3-8b", quant: str | None = None,
                     snapshot: str | None = None, gguf_path: str | None = None,
                     **cfg_overrides) -> Qwen:
    spec = variant.lower()
    if spec not in _QWEN3_VARIANTS:
        raise ValueError(f"unknown qwen3 variant {spec!r}; known: {list(_QWEN3_VARIANTS)}")
    meta = _QWEN3_VARIANTS[spec]

    sk_root = Path(__file__).resolve().parents[3]
    snap = Path(snapshot) if snapshot else (sk_root / "SuperKittens" / "model_weights" / meta["weight_dir"])

    if (snap / "config.json").exists():
        cfg = _cfg_from_snapshot(snap, **cfg_overrides)
    else:
        cfg = _cfg_from_dims(spec, **cfg_overrides)

    m = Qwen(cfg)
    quant = quant or meta["default_quant"]

    if quant in ("q8_0", "Q8_0", "gguf"):
        gpath = Path(gguf_path) if gguf_path else (snap / meta["gguf_name"])
        if not gpath.exists():
            ggs = list(snap.glob("*.gguf")) if snap.exists() else []
            if not ggs:
                raise FileNotFoundError(f"gguf file not found: {gpath}")
            gpath = ggs[0]
        m.load_gguf(str(gpath))
    else:
        lib = _load()  # ABI bound via QWEN_ABI; argtypes/restype set centrally
        idx_path = snap / "model.safetensors.index.json"
        single = snap / "model.safetensors"
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

    m.bake_and_set_rope()

    tok = _resolve_tokenizer(snap, spec)
    if tok is not None:
        m.tokenizer = tok
    else:
        print(f"[qwen] no tokenizer available for {spec}")

    return m


class _QwenFactory:
    @staticmethod
    def from_pretrained(**kwargs):
        return _from_pretrained(**kwargs)


from SuperKittens.api import register
for _spec in _QWEN3_VARIANTS:
    try:
        register(_spec, _QwenFactory, variant=_spec)
    except ValueError:
        pass

__all__ = ["Qwen", "Config"]
