"""inference/registry.py — single source of truth for per-model knowledge.

Each row in ``SPECS`` describes everything a family-specific adapter needs to
turn a model name (e.g. ``"qwen3-0.6b"``) into a working ``Model``: HF repo,
on-disk weight directory, optional GGUF artifact, default quant, tokenizer
family, and hardcoded dimensional fallbacks for when ``config.json`` is
missing.

Adding a new model means adding ONE row to this table.

The narrow per-family seam is :pymeth:`from_spec` on each adapter class. The
adapter receives a :class:`ModelSpec` plus runtime overrides (``seq_max``,
``cache_max``, etc.) and is responsible for constructing the model, loading
weights, baking RoPE tables, and attaching a tokenizer.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    family: str                       # "qwen3" | "gemma4" | "mamba2" | "deepseek"
    adapter: str                      # "pkg.mod:ClassName"
    hf_repo: str                      # canonical HuggingFace repo
    weight_dir: str                   # subdir under SuperKittens/model_weights/
    gguf_name: str | None = None      # filename inside weight_dir if GGUF is canonical
    default_quant: str | None = None  # "q8_0" | None
    tokenizer_family: str | None = "qwen3"  # routes through Tokenizer._FAMILY_SPECIALS
    config_path: str | None = None    # subkey in config.json (e.g. "text_config" for gemma4)
    dims: dict[str, Any] = field(default_factory=dict)  # hardcoded fallback dims


SPECS: dict[str, ModelSpec] = {
    "qwen3-0.6b": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-0.6B",
        weight_dir="Qwen3-0.6B",
        gguf_name="Qwen3-0.6B-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="qwen3",
        dims=dict(n_layers=28, d_model=1024, n_heads=16, n_kv_heads=8,
                  head_dim=128, n_int=3072, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=1),
    ),
    "gemma4-e2b": ModelSpec(
        family="gemma4",
        adapter="SuperKittens.models.gemma.gemma4.gemma4:Gemma4",
        hf_repo="google/gemma-4-E2B-it",
        weight_dir="gemma-4-E2B-it",
        default_quant=None,
        tokenizer_family="gemma",
        config_path="text_config",
        dims=dict(variant="e2b"),
    ),
    "gemma4-e4b": ModelSpec(
        family="gemma4",
        adapter="SuperKittens.models.gemma.gemma4.gemma4:Gemma4",
        hf_repo="google/gemma-4-E4B-it",
        weight_dir="gemma-4-E4B-it",
        default_quant=None,
        tokenizer_family="gemma",
        config_path="text_config",
        dims=dict(variant="e4b"),
    ),
    "gemma4-26b": ModelSpec(
        family="gemma4",
        adapter="SuperKittens.models.gemma.gemma4.gemma4:Gemma4",
        hf_repo="google/gemma-4-26B-it",
        weight_dir="gemma-4-26B-it",
        default_quant=None,
        tokenizer_family="gemma",
        config_path="text_config",
        dims=dict(variant="26b"),
    ),
    "gemma4-31b": ModelSpec(
        family="gemma4",
        adapter="SuperKittens.models.gemma.gemma4.gemma4:Gemma4",
        hf_repo="google/gemma-4-31B-it",
        weight_dir="gemma-4-31B-it",
        default_quant=None,
        tokenizer_family="gemma",
        config_path="text_config",
        dims=dict(variant="31b"),
    ),
    "mamba2-130m": ModelSpec(
        family="mamba2",
        adapter="SuperKittens.models.mamba2.mamba2:Mamba2Model",
        hf_repo="AntonV/mamba2-130m-hf",
        weight_dir="mamba2-130m-hf",
        default_quant=None,
        tokenizer_family=None,  # pre-instruct: no chat template, no specials needed
        dims=dict(n_layers=24, d_model=768, intermediate=1536, n_heads=24,
                  head_dim=64, state_size=128, n_groups=1, conv_kernel=4,
                  chunk_size=256, vocab_size=50288, rms_eps=1e-5,
                  tie_word_embeddings=1),
    ),
    "qwen3-8b": ModelSpec(
        family="qwen3",
        adapter="SuperKittens.models.qwen.qwen:Qwen",
        hf_repo="Qwen/Qwen3-8B",
        weight_dir="Qwen3-8B-GGUF",
        gguf_name="Qwen3-8B-Q8_0.gguf",
        default_quant="q8_0",
        tokenizer_family="qwen3",
        dims=dict(n_layers=36, d_model=4096, n_heads=32, n_kv_heads=8,
                  head_dim=128, n_int=12288, vocab_size=151936,
                  eps=1e-6, rope_freq_base=1_000_000.0, tie_word_embeddings=0),
    ),
}


def get_spec(name: str) -> ModelSpec:
    if name not in SPECS:
        raise ValueError(f"unknown model {name!r}; known: {list(SPECS)}")
    return SPECS[name]


def list_specs() -> list[str]:
    return list(SPECS)


def _import_adapter(path: str):
    mod_path, cls_name = path.split(":")
    return getattr(importlib.import_module(mod_path), cls_name)


def load(name: str, **overrides):
    """Generic loader: lookup spec, import adapter, build model via from_spec."""
    spec = get_spec(name)
    cls = _import_adapter(spec.adapter)
    return cls.from_spec(spec, **overrides)
