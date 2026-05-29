# pyright: reportMissingImports=false
"""Kernel-level repro for the mha_causal chunked-forward correctness bug.

Dispatches mha_causal (fa_d128<true>) with synthetic Q/K/V exactly as the qwen
launcher does (head-major Q/O (H,seq,D); K/V cache (n_kv_heads,cache_stride,D);
grid (n_kv_heads,(seq+1)/2,batch); TG (Hg*2*32,1,1)), then compares every output
row against a numpy causal SDPA reference over the full kv_len.

Scenarios:
  - prefill : current_pos=0, seq=S            (kv_len=S)   -- must be correct
  - decode  : current_pos=P, seq=1            (kv_len=P+1) -- must be correct
  - chunk   : current_pos=P, seq=K            (kv_len=P+K) -- the BUG
"""
from __future__ import annotations
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SK_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # repo root (has SuperKittens/)
sys.path.insert(0, SK_ROOT)
from SuperKittens.benchmark.harness.bench_harness import BenchHarness  # noqa: E402

METALLIB = os.path.join(SK_ROOT, "build", "libsk.metallib")

D = 128


def sdpa_ref(q, k, v, q_pos):
    """Causal SDPA reference. q:(S,D) k,v:(kv,D). q_pos[i]=abs pos of query row i.
    Query i attends keys [0 .. q_pos[i]]. Returns (S,D) fp32."""
    S = q.shape[0]
    kv = k.shape[0]
    scale = 1.0 / np.sqrt(D)
    out = np.zeros((S, D), np.float32)
    for i in range(S):
        lim = q_pos[i] + 1
        scores = (q[i:i+1].astype(np.float32) @ k[:lim].astype(np.float32).T) * scale  # (1,lim)
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        out[i] = (w @ v[:lim].astype(np.float32))
    return out


def run_case(h, pso, name, n_heads, n_kv_heads, seq, current_pos, cache_stride, seed=0):
    rng = np.random.default_rng(seed)
    Hg = n_heads // n_kv_heads
    kv_len = current_pos + seq
    assert kv_len <= cache_stride

    # Q head-major (H, seq, D); K/V cache (n_kv_heads, cache_stride, D).
    Q = (rng.standard_normal((n_heads, seq, D)).astype(np.float32) * 0.5).astype(np.float16)
    Kc = np.zeros((n_kv_heads, cache_stride, D), np.float16)
    Vc = np.zeros((n_kv_heads, cache_stride, D), np.float16)
    Kc[:, :kv_len] = (rng.standard_normal((n_kv_heads, kv_len, D)).astype(np.float32) * 0.5).astype(np.float16)
    Vc[:, :kv_len] = (rng.standard_normal((n_kv_heads, kv_len, D)).astype(np.float32) * 0.5).astype(np.float16)
    O = np.zeros((n_heads, seq, D), np.float16)

    bQ = h.make_buf(Q); bK = h.make_buf(Kc); bV = h.make_buf(Vc); bO = h.make_buf(O)
    bSeq = h.make_buf(np.array([seq], np.uint32))
    bNh  = h.make_buf(np.array([n_heads], np.uint32))
    bNkv = h.make_buf(np.array([n_kv_heads], np.uint32))
    bKvl = h.make_buf(np.array([kv_len], np.uint32))
    bCs  = h.make_buf(np.array([cache_stride], np.uint32))

    grid = (n_kv_heads, (seq + 1) // 2, 1)
    tg = (Hg * 2 * 32, 1, 1)
    # single dispatch
    cb = h.queue.commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, b in enumerate([bQ, bK, bV, bO, bSeq, bNh, bNkv, bKvl, bCs]):
        enc.setBuffer_offset_atIndex_(b, 0, i)
    import Metal
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(*tg))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    Oout = h.read_buf(bO, np.float16, (n_heads, seq, D)).astype(np.float32)

    # Reference per head.
    q_pos = np.array([current_pos + i for i in range(seq)], np.int32)
    worst = 0.0
    bad_rows = []
    for head in range(n_heads):
        kv_head = head * n_kv_heads // n_heads
        ref = sdpa_ref(Q[head].astype(np.float32), Kc[kv_head, :kv_len].astype(np.float32),
                       Vc[kv_head, :kv_len].astype(np.float32), q_pos)
        for i in range(seq):
            num = np.linalg.norm(Oout[head, i] - ref[i])
            den = np.linalg.norm(ref[i]) + 1e-6
            rel = num / den
            if rel > worst:
                worst = rel
            if rel > 1e-2 and (head, i) not in [(b[0], b[1]) for b in bad_rows]:
                bad_rows.append((head, i, float(rel)))
    status = "OK " if worst < 1e-2 else "BAD"
    print(f"[{status}] {name:28s} n_heads={n_heads} n_kv={n_kv_heads} seq={seq} "
          f"cur_pos={current_pos} kv_len={kv_len}  max_rel={worst:.4e}")
    if bad_rows:
        # report a few representative bad (head,row) pairs
        rows_only = sorted(set(i for (_h, i, _r) in bad_rows))
        print(f"        bad rows (q index): {rows_only}   e.g. {bad_rows[:4]}")
    return worst


def main():
    h = BenchHarness(metallib_path=METALLIB)
    pso = h.pso("mha_causal")
    cache = 1024
    print("=== mha_causal correctness ===")
    # qwen3-0.6B-like: n_heads=16, n_kv_heads=8 (Hg=2). Also test Hg=4 (4B-like 32/8).
    configs = [
        ("qwen0.6B Hg=2", 16, 8),
        ("qwen4B   Hg=4", 32, 8),
    ]
    worst_all = 0.0
    for label, nh, nkv in configs:
        print(f"--- {label} ---")
        # prefill (must be correct)
        worst_all = max(worst_all, run_case(h, pso, "prefill seq=8", nh, nkv, 8, 0, cache, seed=1))
        worst_all = max(worst_all, run_case(h, pso, "prefill seq=17", nh, nkv, 17, 0, cache, seed=2))
        # decode seq=1 (must be correct)
        worst_all = max(worst_all, run_case(h, pso, "decode seq=1 @8", nh, nkv, 1, 8, cache, seed=3))
        worst_all = max(worst_all, run_case(h, pso, "decode seq=1 @17", nh, nkv, 1, 17, cache, seed=4))
        # chunk (the bug): seq=4 @ cur=8  (the task's exact repro)
        worst_all = max(worst_all, run_case(h, pso, "chunk seq=4 @8", nh, nkv, 4, 8, cache, seed=5))
        worst_all = max(worst_all, run_case(h, pso, "chunk seq=4 @15", nh, nkv, 4, 15, cache, seed=6))
        worst_all = max(worst_all, run_case(h, pso, "chunk seq=4 @17", nh, nkv, 4, 17, cache, seed=7))
        worst_all = max(worst_all, run_case(h, pso, "chunk seq=5 @60", nh, nkv, 5, 60, cache, seed=8))
        worst_all = max(worst_all, run_case(h, pso, "chunk seq=8 @100", nh, nkv, 8, 100, cache, seed=9))
    print(f"\nWORST max_rel over all cases: {worst_all:.4e}")
    sys.exit(0 if worst_all < 1e-2 else 1)


if __name__ == "__main__":
    main()
