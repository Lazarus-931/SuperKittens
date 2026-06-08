"""Shared dense-transformer inference core.

`DenseDecoder` is the family-agnostic Python handle over the `sk_qwen_*` C-ABI
launcher. Per-family adapters (Qwen, Nemotron) subclass it and configure the
core (Q/K-norm on/off, RoPE bake, generation prelude).
"""
from __future__ import annotations

from .dense_decoder import DenseDecoder, Config, DENSE_ABI

__all__ = ["DenseDecoder", "Config", "DENSE_ABI"]
