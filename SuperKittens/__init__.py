import os as _os
from pathlib import Path as _Path

# Resolve bundled native artifacts before any kernel module imports.
# Existing loader sites read SK_DYLIB / SK_METALLIB env vars; pointing those
# at the wheel-installed _libs/ directory makes `pip install`ed users work
# without touching every CDLL site.
_LIBS_DIR = _Path(__file__).resolve().parent / "_libs"
_BUNDLED_DYLIB = _LIBS_DIR / "libsk.dylib"
_BUNDLED_METALLIB = _LIBS_DIR / "libsk.metallib"
if _BUNDLED_DYLIB.exists():
    _os.environ.setdefault("SK_DYLIB", str(_BUNDLED_DYLIB))
if _BUNDLED_METALLIB.exists():
    _os.environ.setdefault("SK_METALLIB", str(_BUNDLED_METALLIB))

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PNF
    try:
        __version__ = _pkg_version("superkittens")
    except _PNF:
        __version__ = "0.0.0+unknown"
except Exception:
    __version__ = "0.0.0+unknown"

from .api import load, register, list_models, MODEL_REGISTRY
from .inference.generation import Model

__all__ = ["load", "register", "list_models", "Model", "MODEL_REGISTRY"]
