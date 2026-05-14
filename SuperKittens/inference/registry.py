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


SPECS: dict[str, ModelSpec] = {}


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
