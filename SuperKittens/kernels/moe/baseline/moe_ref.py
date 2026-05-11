"""moe_ref.py — numpy reference for MoE router/dispatch/combine.

The reference implementation is intentionally simple and slow; it is the
correctness oracle for the SuperKittens kernels.
"""
from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x.astype(np.float32)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def router_ref(x: np.ndarray, W: np.ndarray, top_k: int):
    """x:(T,D), W:(D,N) -> (top_idx (T,K) i32, top_score (T,K) fp16 normalized)."""
    logits = x.astype(np.float32) @ W.astype(np.float32)
    probs = softmax(logits, axis=-1)
    # top-k by descending score; ties broken by lower index (matches our argmax sweep).
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :top_k].astype(np.int32)
    score = np.take_along_axis(probs, idx, axis=-1).astype(np.float16)
    return idx, score


def dispatch_ref(x: np.ndarray, top_idx: np.ndarray, n_expert: int):
    """Returns x_dispatched, expert_offsets, dest_row, inverse_perm.

    Tokens routed to expert e occupy rows [offsets[e], offsets[e+1]) of
    x_dispatched. The order within a bin is the order they appear when we
    sweep tokens t=0..T-1, k=0..K-1 (matches the atomic-fetch-add order on
    GPU for a deterministic single-threaded oracle).
    """
    T, D = x.shape
    T_, K = top_idx.shape
    TK = T * K
    counts = np.zeros(n_expert, dtype=np.int32)
    for tk in range(TK):
        counts[top_idx[tk // K, tk % K]] += 1
    offsets = np.zeros(n_expert + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    x_dispatched = np.zeros((TK, D), dtype=np.float16)
    inverse_perm = np.full((TK,), -1, dtype=np.int32)
    dest_row     = np.zeros((TK,), dtype=np.int32)
    cursor = offsets[:-1].copy()
    for t in range(T):
        for k in range(K):
            e = int(top_idx[t, k])
            slot = cursor[e]; cursor[e] += 1
            x_dispatched[slot] = x[t]
            inverse_perm[slot] = t * K + k
            dest_row[t * K + k] = slot
    return x_dispatched, offsets, dest_row, inverse_perm


def combine_ref(expert_out: np.ndarray, dest_row: np.ndarray,
                top_score: np.ndarray, T: int) -> np.ndarray:
    T_, K = top_score.shape
    D_out = expert_out.shape[1]
    out = np.zeros((T, D_out), dtype=np.float32)
    for t in range(T):
        for k in range(K):
            dst = int(dest_row[t * K + k])
            if dst < 0: continue
            w = float(top_score[t, k])
            out[t] += w * expert_out[dst].astype(np.float32)
    return out.astype(np.float16)
