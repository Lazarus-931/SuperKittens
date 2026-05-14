"""Qwen3 model package.

Per-model knowledge lives in ``SuperKittens.inference.registry.SPECS``.
Loading is centralised: ``sk.load("qwen3-0.6b")`` -> ``registry.load`` ->
``Qwen.from_spec(spec, ...)``.
"""
from __future__ import annotations

from .qwen import Qwen, Config

__all__ = ["Qwen", "Config"]
