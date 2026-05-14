"""Gemma 4 model package.

Per-model knowledge lives in ``SuperKittens.inference.registry.SPECS``.
Loading is centralised: ``sk.load("gemma4-e2b")`` -> ``registry.load`` ->
``Gemma4.from_spec(spec, ...)``.
"""
from __future__ import annotations

from .gemma4 import Gemma4, Gemma4Config

__all__ = ["Gemma4", "Gemma4Config"]
