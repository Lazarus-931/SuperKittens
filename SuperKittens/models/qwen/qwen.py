"""qwen.py — Qwen3 (dense) adapter over the shared dense-transformer core.

Qwen3 configures the shared :class:`DenseDecoder` with per-head Q/K RMSNorm ON
(``use_qk_norm=1``) and plain RoPE. All the launcher/kernel plumbing lives in
``SuperKittens.models.dense.dense_decoder``; this adapter only adds the Qwen3
preset table and identity.

Usage:
    from SuperKittens.models.qwen.qwen import Qwen
    with Qwen("32b") as m:
        m.load_random_weights(seed=0)
        toks = m.generate([1,2,3], max_new_tokens=8)
"""
from __future__ import annotations

from SuperKittens.models.dense.dense_decoder import DenseDecoder, Config

# WHY: re-export so existing `from ...qwen import Config` / ABI imports keep
# working after the shared-core extraction.
from SuperKittens.models.dense.dense_decoder import (  # noqa: F401
    DENSE_ABI as QWEN_ABI, _Config, _Weights, _WEIGHT_FIELDS,
)


def _qwen_preset(name: str) -> Config:
    n = name.lower().replace("-", "_")
    if n in ("32b", "qwen3_32b", "v3_32b"):
        return Config()
    if n in ("test", "tiny"):
        # Smallest config that exercises every dispatch step.
        return Config(n_layers=2, d_model=512, n_heads=4, n_kv_heads=2,
                      head_dim=128, n_int=512, vocab_size=1024,
                      seq_max=16, cache_max=64,
                      rope_freq_base=1_000_000.0, rope_n_ctx_orig=64)
    raise ValueError(f"unknown Qwen preset: {name!r}")


class Qwen(DenseDecoder):
    """Stateful Qwen3 (dense) inference handle (per-head Q/K-norm, plain RoPE)."""

    @classmethod
    def _config_preset(cls, name: str) -> Config:
        return _qwen_preset(name)

    @classmethod
    def test_config(cls) -> "Qwen":
        return cls(_qwen_preset("test"))
