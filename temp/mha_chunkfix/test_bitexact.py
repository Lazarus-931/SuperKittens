# pyright: reportMissingImports=false
"""Bit-exactness check: fixed mha_causal must produce byte-identical output to the
pre-fix baseline on the WORKING paths (current_pos=0 prefill, seq=1 decode).
The chunk path is expected to DIFFER (baseline was wrong there)."""
from __future__ import annotations
import sys, os
import numpy as np
import Metal

HERE = os.path.dirname(os.path.abspath(__file__))
SK_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SK_ROOT)
from SuperKittens.benchmark.harness.bench_harness import BenchHarness  # noqa: E402

BASE = "/tmp/libsk_baseline.metallib"
FIX = "/tmp/libsk_fixed.metallib"
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
    return h.read_buf(bufs[3], np.float16, (n_heads, seq, D))


def gen(n_heads, n_kv_heads, seq, current_pos, cache, seed):
    rng = np.random.default_rng(seed)
    kv_len = current_pos + seq
    Q = (rng.standard_normal((n_heads, seq, D)) * 0.5).astype(np.float16)
    Kc = np.zeros((n_kv_heads, cache, D), np.float16)
    Vc = np.zeros((n_kv_heads, cache, D), np.float16)
    Kc[:, :kv_len] = (rng.standard_normal((n_kv_heads, kv_len, D)) * 0.5).astype(np.float16)
    Vc[:, :kv_len] = (rng.standard_normal((n_kv_heads, kv_len, D)) * 0.5).astype(np.float16)
    return Q, Kc, Vc, kv_len


def main():
    hb = BenchHarness(metallib_path=BASE); pb = hb.pso("mha_causal")
    hf = BenchHarness(metallib_path=FIX);  pf = hf.pso("mha_causal")
    cache = 1024
    configs = [("Hg=2", 16, 8), ("Hg=4", 32, 8)]
    ok = True
    for label, nh, nkv in configs:
        print(f"--- {label} ---")
        # WORKING paths: must be BIT-EXACT.
        for name, seq, cur in [("prefill seq=8 @0", 8, 0), ("prefill seq=17 @0", 17, 0),
                                ("prefill seq=64 @0", 64, 0), ("prefill seq=100 @0", 100, 0),
                                ("decode seq=1 @8", 1, 8), ("decode seq=1 @17", 1, 17),
                                ("decode seq=1 @63", 1, 63), ("decode seq=1 @200", 1, 200)]:
            Q, Kc, Vc, kv = gen(nh, nkv, seq, cur, cache, hash((name, label)) & 0xffff)
            ob = dispatch(hb, pb, Q, Kc, Vc, nh, nkv, seq, kv, cache)
            of = dispatch(hf, pf, Q, Kc, Vc, nh, nkv, seq, kv, cache)
            same = np.array_equal(ob.view(np.uint16), of.view(np.uint16))
            tag = "BIT-EXACT" if same else "DIFFERS!!"
            print(f"  [{tag}] {name}")
            if not same:
                ok = False
        # CHUNK paths: expected to differ (baseline buggy). Just report.
        for name, seq, cur in [("chunk seq=4 @15", 4, 15), ("chunk seq=4 @17", 4, 17)]:
            Q, Kc, Vc, kv = gen(nh, nkv, seq, cur, cache, hash((name, label)) & 0xffff)
            ob = dispatch(hb, pb, Q, Kc, Vc, nh, nkv, seq, kv, cache)
            of = dispatch(hf, pf, Q, Kc, Vc, nh, nkv, seq, kv, cache)
            same = np.array_equal(ob.view(np.uint16), of.view(np.uint16))
            print(f"  [{'same' if same else 'CHANGED (expected)'}] {name}")
    print("\nBIT-EXACT on working paths:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
