# pyright: reportMissingImports=false
"""Probe: for a single (head,row) in a failing chunk case, find which key range
the kernel actually attended to (by zeroing V beyond a cutoff and seeing when O
stops changing) — diagnostic only."""
from __future__ import annotations
import sys, os
import numpy as np
import Metal

HERE = os.path.dirname(os.path.abspath(__file__))
SK_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SK_ROOT)
from SuperKittens.benchmark.harness.bench_harness import BenchHarness  # noqa: E402

METALLIB = os.path.join(SK_ROOT, "build", "libsk.metallib")
D = 128


def dispatch(h, pso, Q, Kc, Vc, n_heads, n_kv_heads, seq, kv_len, cache_stride):
    Hg = n_heads // n_kv_heads
    O = np.zeros((n_heads, seq, D), np.float16)
    bufs = [h.make_buf(Q), h.make_buf(Kc), h.make_buf(Vc), h.make_buf(O),
            h.make_buf(np.array([seq], np.uint32)),
            h.make_buf(np.array([n_heads], np.uint32)),
            h.make_buf(np.array([n_kv_heads], np.uint32)),
            h.make_buf(np.array([kv_len], np.uint32)),
            h.make_buf(np.array([cache_stride], np.uint32))]
    cb = h.queue.commandBuffer(); enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, b in enumerate(bufs):
        enc.setBuffer_offset_atIndex_(b, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(n_kv_heads, (seq + 1) // 2, 1),
        Metal.MTLSizeMake(Hg * 2 * 32, 1, 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    return h.read_buf(bufs[3], np.float16, (n_heads, seq, D)).astype(np.float32)


def main():
    h = BenchHarness(metallib_path=METALLIB)
    pso = h.pso("mha_causal")
    n_heads, n_kv_heads = 16, 8
    seq, cur = 4, 15
    kv_len = cur + seq  # 19
    cache = 1024
    rng = np.random.default_rng(15)
    Q = (rng.standard_normal((n_heads, seq, D)).astype(np.float32) * 0.5).astype(np.float16)
    Kc = np.zeros((n_kv_heads, cache, D), np.float16)
    Vc = np.zeros((n_kv_heads, cache, D), np.float16)
    Kc[:, :kv_len] = (rng.standard_normal((n_kv_heads, kv_len, D)) * 0.5).astype(np.float16)
    Vc[:, :kv_len] = (rng.standard_normal((n_kv_heads, kv_len, D)) * 0.5).astype(np.float16)

    O = dispatch(h, pso, Q, Kc, Vc, n_heads, n_kv_heads, seq, kv_len, cache)

    # For head 0, each row i should attend keys [0..cur+i]. Make a one-hot V so
    # the output directly reveals the attention weights: set V[k] = e_k basis
    # (V[:, k, k]=1). Then O[i] = attention weights over k (first kv_len comps).
    Vh = np.zeros((n_kv_heads, cache, D), np.float16)
    for k in range(min(kv_len, D)):
        Vh[:, k, k] = 1.0
    Oh = dispatch(h, pso, Q, Kc, Vh, n_heads, n_kv_heads, seq, kv_len, cache)
    scale = 1.0 / np.sqrt(D)
    for i in range(seq):
        qpos = cur + i
        scores = (Q[0, i].astype(np.float32) @ Kc[0, :kv_len].astype(np.float32).T) * scale
        scores[qpos+1:] = -1e30
        w = np.exp(scores - scores.max()); w /= w.sum()
        kw = Oh[0, i, :kv_len]  # kernel's attention weights
        # which key positions does kernel weight that ref says should be ~0 (causal future)?
        print(f"row {i} (qpos={qpos}): ref_attends<= {qpos}; "
              f"kernel nonzero key idxs = {np.where(np.abs(kw)>1e-3)[0].tolist()}")
        print(f"        ref top keys  = {np.argsort(-w)[:5].tolist()}")
        print(f"        kern top keys = {np.argsort(-kw)[:5].tolist()}")


if __name__ == "__main__":
    main()
