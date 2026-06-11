# pyright: reportMissingImports=false
"""adapter.py — registry seam for the DiffusionGemma family.

Stage-1 scope: loading + the unified zero-SC forward (logits parity). The
denoising sampler loop / generation entrypoint land in Stage 2, so this
adapter intentionally exposes `forward(ids, P)` and not `generate`.
"""
from __future__ import annotations

from pathlib import Path

from .config import config_from_gguf
from .gguf_io import GGUFFile


class DiffusionGemma:
    @classmethod
    def from_spec(cls, spec, **overrides):
        from .forward_metal import DiffusionGemmaMetal

        sk_root = Path(__file__).resolve().parents[3]
        snap = Path(overrides.pop("snapshot", None)
                    or (sk_root / "model_weights" / spec.weight_dir))
        gguf = overrides.pop("gguf", None) or (snap / spec.gguf_name)
        if not Path(gguf).exists():
            raise FileNotFoundError(f"DiffusionGemma GGUF not found: {gguf}")
        cfg = config_from_gguf(GGUFFile(str(gguf)).meta)
        return DiffusionGemmaMetal(str(gguf), cfg)
