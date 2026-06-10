"""T=256 variant: lane-match (vs token-by-token lockstep) + TTFT A/B + decode
aggregate. ONE batch=8 handle (seq_max=64, cache_max=320). Chunk-variant and
identical-lane probes already covered at T=128.
Writes artifacts/gates_lane_t256.json.
"""
import os, sys, time, json
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-bprefill-k9")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load
from gates_lane import TEXTS

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")
N, T, CONT = 8, 256, 32
SEQ_MAX, CACHE_MAX = 64, 320
REPS, WARMUPS, GAP = 7, 2, 0.3


def main():
    t0 = time.perf_counter()
    m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF,
             batch=N, seq_max=SEQ_MAX, cache_max=CACHE_MAX)
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    lanes = []
    for tx in TEXTS:
        ids = m.tokenizer.encode(" ".join([tx] * 3), bos=True)
        assert len(ids) >= T, f"prompt too short: {len(ids)}"
        lanes.append(np.asarray(ids[:T], dtype=np.int32))
    ids_mat = np.stack(lanes)

    def tbt_prefill(mat):
        cur = np.zeros(N, dtype=np.int32)
        nxt = None
        for s in range(mat.shape[1]):
            cur[:] = mat[:, s]
            nxt = m._forward_batched(cur)
        return np.asarray(nxt, dtype=np.int32)

    def cont_steps(first, n=CONT):
        rows = [np.asarray(first, dtype=np.int32).copy()]
        cur = rows[0].copy()
        for _ in range(n):
            cur = np.asarray(m._forward_batched(cur), dtype=np.int32).copy()
            rows.append(cur)
        return np.stack(rows, axis=1)

    res = {"config": {"N": N, "T": T, "CONT": CONT, "seq_max": SEQ_MAX,
                      "cache_max": CACHE_MAX}}

    print("[gate] A: token-by-token lockstep prefill + cont", flush=True)
    m.reset()
    a_next = tbt_prefill(ids_mat)
    ta = time.perf_counter(); a_tok = cont_steps(a_next); a_dec = time.perf_counter() - ta

    print("[gate] B: prefill_batched chunk=64 + cont", flush=True)
    m.reset()
    b_next = np.asarray(m.prefill_batched(ids_mat, chunk_size=64), dtype=np.int32)
    tb = time.perf_counter(); b_tok = cont_steps(b_next); b_dec = time.perf_counter() - tb

    res["A_tokens"] = a_tok.tolist(); res["B64_tokens"] = b_tok.tolist()
    res["gates"] = {"lane_match_33tok": bool((a_tok == b_tok).all()),
                    "in_vocab": bool((b_tok >= 0).all() and (b_tok < m.cfg.vocab_size).all())}
    res["per_lane_match"] = [bool((a_tok[i] == b_tok[i]).all()) for i in range(N)]
    print(f"[gates] {res['gates']} per-lane {res['per_lane_match']}", flush=True)

    def time_A():
        m.reset(); t = time.perf_counter(); tbt_prefill(ids_mat)
        return (time.perf_counter() - t) * 1e3

    def time_B():
        m.reset(); t = time.perf_counter(); m.prefill_batched(ids_mat, chunk_size=64)
        return (time.perf_counter() - t) * 1e3

    A_ms, B_ms = [], []
    for rep in range(WARMUPS + REPS):
        a = time_A(); time.sleep(GAP)
        b = time_B(); time.sleep(GAP)
        tag = "warm" if rep < WARMUPS else "rep"
        print(f"[ttft {tag} {rep}] A={a:.1f}ms B64={b:.1f}ms", flush=True)
        if rep >= WARMUPS:
            A_ms.append(a); B_ms.append(b)
    res["ttft"] = {"A_ms": A_ms, "B64_ms": B_ms,
                   "A_med": float(np.median(A_ms)), "B64_med": float(np.median(B_ms))}
    sp = res["ttft"]["A_med"] / res["ttft"]["B64_med"]
    print(f"[ttft] A_med={res['ttft']['A_med']:.1f}ms B64_med={res['ttft']['B64_med']:.1f}ms "
          f"speedup={sp:.2f}x improvement={(1-1/sp)*100:.1f}%", flush=True)

    res["decode32_after_A_s"] = [a_dec]; res["decode32_after_B_s"] = [b_dec]
    print(f"[decode32] after-A {a_dec:.3f}s after-B {b_dec:.3f}s ratio={b_dec/a_dec:.4f}", flush=True)

    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    out = os.path.join(ROOT, "artifacts", "gates_lane_t256.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"GATES_T256_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
