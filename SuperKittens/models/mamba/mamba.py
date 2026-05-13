"""Python wrapper for SK Mamba 1. Scaffold — raises until kernels land.

Mirrors the shape of SuperKittens.models.mamba2.mamba2 but targets the
selective-SSM architecture used by state-spaces/mamba-2.8b-hf.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path


class Mamba:
    """SK end-to-end Mamba 1 model. Not yet functional; see STATUS.md."""

    def __init__(self, weights_dir: str | os.PathLike, variant: str = "2.8b"):
        self.weights_dir = str(weights_dir)
        self.variant = variant
        self._ctx = None
        self._lib = self._load_lib()

    @staticmethod
    def _load_lib():
        dylib = os.environ.get(
            "SK_DYLIB",
            str(Path(__file__).resolve().parents[3] / "build" / "libsk.dylib"),
        )
        if not Path(dylib).exists():
            return None
        try:
            return ctypes.CDLL(dylib)
        except OSError:
            return None

    def load(self):
        raise NotImplementedError(
            "SK Mamba 1 kernels are not yet implemented. "
            "See SuperKittens/models/mamba/STATUS.md."
        )

    def chat(self, prompt: str, max_new_tokens: int = 16) -> str:
        raise NotImplementedError(
            "SK Mamba 1 kernels are not yet implemented. "
            "See SuperKittens/models/mamba/STATUS.md."
        )
