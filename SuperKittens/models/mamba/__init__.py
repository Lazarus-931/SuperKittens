"""SK Mamba 1 (selective SSM) family — state-spaces/mamba-*-hf.

This package is a scaffold; the Metal kernels (selective_scan, conv1d_silu,
gated_rmsnorm, in/out_proj GEMMs) are not yet wired. See STATUS.md.

Registration is intentionally guarded so that importing SuperKittens does
not fail while the port is incomplete.
"""
from __future__ import annotations

try:
    from .mamba import Mamba  # noqa: F401

    import SuperKittens as _sk

    if hasattr(_sk, "register"):
        _sk.register("mamba-2.8b", Mamba, variant="2.8b")
except Exception:  # pragma: no cover
    # Scaffold phase: do not break `import SuperKittens` if pieces are missing.
    pass
