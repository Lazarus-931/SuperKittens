from __future__ import annotations
from typing import Type, Any

MODEL_REGISTRY: dict[str, tuple[Type[Any], dict]] = {}


def register(spec: str, cls: Type[Any], **defaults) -> None:
    if spec in MODEL_REGISTRY:
        raise ValueError(f"already registered: {spec}")
    MODEL_REGISTRY[spec] = (cls, defaults)


def load(spec: str, **kwargs):
    if spec not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY)) or "(none)"
        raise KeyError(f"unknown model spec {spec!r}. known: {known}")
    cls, defaults = MODEL_REGISTRY[spec]
    return cls.from_pretrained(**{**defaults, **kwargs})
