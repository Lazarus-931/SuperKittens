#!/usr/bin/env python3

import math
import mlx.core as mx


def fast_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    d = q.shape[-1]
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(d))


def reference_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    d = q.shape[-1]
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(d))
    probs = mx.softmax(scores, axis=-1)
    return mx.matmul(probs, v)


def causal_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    d = q.shape[-1]
    scores = mx.matmul(q, mx.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(d))
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    q_idx = mx.arange(q_len)[:, None]
    k_idx = mx.arange(k_len)[None, :]
    mask = q_idx >= k_idx
    scores = mx.where(mask, scores, -1e30)
    probs = mx.softmax(scores, axis=-1)
    return mx.matmul(probs, v)
