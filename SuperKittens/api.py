from __future__ import annotations
from typing import Type, Any

from .inference import registry as _registry

MODEL_REGISTRY: dict[str, tuple[Type[Any], dict]] = {}


def register(spec: str, cls: Type[Any], **defaults) -> None:
    if spec in MODEL_REGISTRY:
        raise ValueError(f"already registered: {spec}")
    MODEL_REGISTRY[spec] = (cls, defaults)


def list_models() -> list[str]:
    """All known model specs (central registry + legacy MODEL_REGISTRY)."""
    return sorted(set(_registry.list_specs()) | set(MODEL_REGISTRY))


def load(spec: str, **kwargs):
    # Prefer the central registry; fall back to legacy MODEL_REGISTRY adapters.
    if spec in _registry.SPECS:
        return _registry.load(spec, **kwargs)
    if spec not in MODEL_REGISTRY:
        known = ", ".join(sorted(set(_registry.list_specs()) | set(MODEL_REGISTRY))) or "(none)"
        raise KeyError(f"unknown model spec {spec!r}. known: {known}")
    cls, defaults = MODEL_REGISTRY[spec]
    return cls.from_pretrained(**{**defaults, **kwargs})
