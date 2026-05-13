from .gemma4 import Gemma4, Gemma4Config

from SuperKittens.api import register
for v in ("e2b", "e4b", "26b", "31b"):
    try:
        register(f"gemma4-{v}", Gemma4, variant=v)
    except ValueError:
        pass

__all__ = ["Gemma4", "Gemma4Config"]
