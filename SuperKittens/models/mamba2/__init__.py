"""Mamba2 model package.

Per-model knowledge lives in ``SuperKittens.inference.registry.SPECS``.
Loading is centralised: ``sk.load("mamba2-130m")`` -> ``registry.load`` ->
``Mamba2Model.from_spec(spec, ...)``.
"""
from __future__ import annotations

from .mamba2 import Mamba2Model, Mamba2Config

__all__ = ["Mamba2Model", "Mamba2Config"]
