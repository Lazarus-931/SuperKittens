"""DeepSeek V2/V3/V4 inference package.

Per-model knowledge lives in ``SuperKittens.inference.registry.SPECS``.
Loading is centralised: ``sk.load("deepseek-v2-lite")`` -> ``registry.load`` ->
``DeepSeek.from_spec(spec, ...)``.
"""
from __future__ import annotations

from .deepseek import DeepSeek, Config

__all__ = ["DeepSeek", "Config"]
