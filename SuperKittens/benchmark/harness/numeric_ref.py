"""numeric_ref.py — numpy reference implementations for kernel correctness.

Drop-in CPU references that bench scripts can call to validate GPU output.
Each ref takes plain numpy arrays (fp32 unless noted) and returns the
expected output. Use these instead of writing fresh `ref = ...` inline in
every bench — that's how subtle shape bugs (e.g. q2_K dequant lane order)
get reinvented every lab.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# bf16 round-trip helpers (numpy has no native bf16)
# ---------------------------------------------------------------------------


def bf16_from_f32(arr_f32: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 → bf16, packed as uint16."""
    u = np.ascontiguousarray(arr_f32, dtype=np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def f32_from_bf16(arr_u16: np.ndarray) -> np.ndarray:
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# Linear ops
# ---------------------------------------------------------------------------


def matvec_fp32(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """y = x @ W where x is [K] (or [M,K]) and W is [K,N]. Returns fp32."""
    x32 = x.astype(np.float32)
    W32 = W.astype(np.float32)
    return x32 @ W32


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """y = x / sqrt(mean(x^2) + eps) * gamma, last-axis reduction."""
    x32 = x.astype(np.float32)
    g32 = gamma.astype(np.float32)
    ms = np.mean(x32 * x32, axis=-1, keepdims=True)
    return x32 / np.sqrt(ms + eps) * g32


# ---------------------------------------------------------------------------
# Softmax (numerically-stable online form — same as fused-attn paths)
# ---------------------------------------------------------------------------


def softmax_online(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax, last-axis by default."""
    x32 = x.astype(np.float32)
    m = np.max(x32, axis=axis, keepdims=True)
    e = np.exp(x32 - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def silu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    return x32 / (1.0 + np.exp(-x32))


def silu_mul(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU body: silu(gate) * up."""
    return silu(gate) * up.astype(np.float32)


def geglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """GeGLU body: gelu(gate) * up (tanh approx, matching most kernels)."""
    g = gate.astype(np.float32)
    # tanh-approx gelu (matches Apple Metal's standard impl)
    c = np.sqrt(2.0 / np.pi)
    gelu = 0.5 * g * (1.0 + np.tanh(c * (g + 0.044715 * g * g * g)))
    return gelu * up.astype(np.float32)


# ---------------------------------------------------------------------------
# Dequantizers — STUBS. Implement carefully and unit-test them; the q2_K
# agent's bug came from a swapped scale/min lane in dequant. Don't trust an
# inline dequant — write it here, validate against a known-good output, then
# use it everywhere.
# ---------------------------------------------------------------------------


def q8_0_dequant(blocks: np.ndarray) -> np.ndarray:
    """q8_0 dequant.

    block layout (34 bytes per 32 weights):
      - 2 bytes: fp16 scale
      - 32 bytes: int8 weights
    Returns fp32 array of shape (n_blocks * 32,).
    """
    raise NotImplementedError(
        "q8_0_dequant: TODO. Use the kernel-side reference layout and write a "
        "unit test before relying on this. Don't inline a dequant in a bench."
    )


def q4_K_dequant(blocks: np.ndarray) -> np.ndarray:
    """q4_K dequant (ggml format)."""
    raise NotImplementedError(
        "q4_K_dequant: TODO. See ggml-quants.c::dequantize_row_q4_K for the "
        "exact super-block layout."
    )


def q2_K_dequant(blocks: np.ndarray) -> np.ndarray:
    """q2_K dequant (ggml format)."""
    raise NotImplementedError(
        "q2_K_dequant: TODO. The q2_K agent's bug was a swapped scales/mins "
        "lane — implement here, unit-test against ggml, then reuse."
    )


def iq2_xxs_dequant(blocks: np.ndarray) -> np.ndarray:
    """iq2_xxs dequant (ggml format)."""
    raise NotImplementedError(
        "iq2_xxs_dequant: TODO. Validate against ggml ref before use."
    )
