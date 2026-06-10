"""gemma4-12B-unified batched-prefill gates + serving-TTFT A/B (single process, one handle).

A = baseline serving prefill: token-by-token lockstep (forward_batched seq=1, T steps)
B = new sk_gemma4_prefill_batched (chunked M=batch*seq prefill)

Gate 1 (lane isolation): per-lane first token + 32-token greedy continuation after B
must match A's, per lane; chunk=64 == single-chunk; identical prompts -> identical lanes.
Gate 4 (TTFT): reset -> all-lanes-first-token, 2 warmups + 7 reps median, 0.3s gaps.
Gate 5 (decode aggregate): 32 lockstep steps after A-prefill vs after B-prefill, +-2%.

Env: SK_DYLIB, SK_METALLIB=/nonexistent, SK_METAL_SRC_FALLBACK, SK_TREE,
     SK_GEMMA4_BODY_Q4K=1, SK_GEMMA4_EMBED_Q8=1.
"""
import os, sys, time, json, argparse, statistics

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-gemma-bprefill-r2")))
import numpy as np
import SuperKittens as sk

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seq_max", type=int, default=128)
ap.add_argument("--cache_max", type=int, default=512)
ap.add_argument("--window", type=int, default=0)  # 0 = model default (1024)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--cont", type=int, default=32)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--skip_gates", action="store_true")
ap.add_argument("--json_out", default="")
args = ap.parse_args()

over = dict(batch=args.batch, seq_max=args.seq_max, cache_max=args.cache_max)
if args.window:
    over["window"] = args.window
t0 = time.perf_counter()
m = sk.load("gemma4-12b-unified", **over)
print(f"[setup] loaded in {time.perf_counter()-t0:.1f}s batch={args.batch} "
      f"seq_max={args.seq_max} cache_max={args.cache_max} window={m.cfg.window}", flush=True)

rng = np.random.default_rng(7)
def make_prompts(T):
    return np.ascontiguousarray(rng.integers(10, 200000, size=(args.batch, T)).astype(np.int32))

def prefill_tbt(ids):
    out = None
    for t in range(ids.shape[1]):
        out = m.forward_batched(np.ascontiguousarray(ids[:, t]))
    return out

def decode_n(first, n):
    toks = [np.array(first, dtype=np.int32).copy()]
    cur = toks[0]
    for _ in range(n - 1):
        cur = m.forward_batched(cur).astype(np.int32)
        toks.append(cur.copy())
    return np.stack(toks, axis=1)  # (batch, n)

results = {"model": "gemma4-12b-unified", "batch": args.batch,
           "window": int(m.cfg.window), "T": args.T, "chunk": args.chunk}

if not args.skip_gates:
    # ---- Gate 1: lane isolation ---------------------------------------------
    ids = make_prompts(args.T)
    print(f"[gate1] T={args.T} chunk={args.chunk} cont={args.cont}", flush=True)

    m.reset()
    t0 = time.perf_counter()
    base_next = prefill_tbt(ids)
    print(f"[gate1] tbt prefill done in {time.perf_counter()-t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    base_cont = decode_n(base_next, args.cont)
    dt_dec_a = time.perf_counter() - t0
    print(f"[gate1] A continuation done in {dt_dec_a:.1f}s", flush=True)

    m.reset()
    t0 = time.perf_counter()
    new_next = m.prefill_batched(ids, chunk_size=args.chunk)
    print(f"[gate1] batched prefill done in {time.perf_counter()-t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    new_cont = decode_n(new_next, args.cont)
    dt_dec_b = time.perf_counter() - t0

    m.reset()
    new_next_1c = m.prefill_batched(ids, chunk_size=0)  # single chunk (<= seq_max)

    lane_match = [bool((base_cont[b] == new_cont[b]).all()) for b in range(args.batch)]
    first_match = [int(base_next[b]) == int(new_next[b]) for b in range(args.batch)]
    chunk_vs_1c = [int(new_next[b]) == int(new_next_1c[b]) for b in range(args.batch)]
    tok_ok = bool((new_cont >= 0).all() and (new_cont < m.cfg.vocab_size).all())
    print(f"[gate1] first-token match (B vs A) per lane: {first_match}")
    print(f"[gate1] {args.cont}-token continuation match per lane: {lane_match}")
    print(f"[gate1] chunked({args.chunk}) vs single-chunk next-token match: {chunk_vs_1c}")
    print(f"[gate1] all tokens in-vocab: {tok_ok}", flush=True)
    for b in range(args.batch):
        if not lane_match[b]:
            a, c = base_cont[b].tolist(), new_cont[b].tolist()
            d = next(i for i in range(len(a)) if a[i] != c[i])
            print(f"[gate1] lane {b} diverges at cont idx {d}: A={a[max(0,d-2):d+3]} B={c[max(0,d-2):d+3]}")
    # gate 5: decode aggregate tok/s after prefill (same lockstep path both sides)
    dec_a = args.batch * (args.cont - 1) / dt_dec_a
    dec_b = args.batch * (args.cont - 1) / dt_dec_b
    print(f"[gate5] decode aggregate after A: {dec_a:.2f} tok/s; after B: {dec_b:.2f} tok/s; "
          f"ratio B/A={dec_b/dec_a:.3f}", flush=True)
    results["gate1"] = {"first_match": first_match, "cont_match": lane_match,
                        "chunk_vs_single": chunk_vs_1c, "tokens_in_vocab": tok_ok,
                        "base_next": [int(x) for x in base_next],
                        "new_next": [int(x) for x in new_next],
                        "base_cont": base_cont.tolist(), "new_cont": new_cont.tolist()}
    results["gate5"] = {"dec_a_tok_s": dec_a, "dec_b_tok_s": dec_b, "ratio": dec_b / dec_a}

    ids_same = np.tile(ids[0], (args.batch, 1))
    m.reset()
    same_next = m.prefill_batched(ids_same, chunk_size=args.chunk)
    print(f"[gate1b] identical prompts -> identical next tokens: "
          f"{bool((same_next == same_next[0]).all())} ({same_next.tolist()})", flush=True)
    results["gate1b_same_prompt_next"] = [int(x) for x in same_next]

# ---- Gate 4: TTFT A/B --------------------------------------------------------
# One interleaved triple per rep (A tbt, B single-chunk, B chunk=N) so the slow
# A side (T lockstep steps) is measured once and both B configs share its
# thermal/contention window.
def ttft_ab(T, chunk, reps):
    p = make_prompts(T)
    for _ in range(2):  # warmups
        m.reset(); prefill_tbt(p); time.sleep(0.3)
        m.reset(); m.prefill_batched(p, chunk_size=0); time.sleep(0.3)
        m.reset(); m.prefill_batched(p, chunk_size=chunk); time.sleep(0.3)
    a, b1, bc = [], [], []
    for r in range(reps):
        m.reset(); t0 = time.perf_counter(); prefill_tbt(p)
        a.append(time.perf_counter() - t0); time.sleep(0.3)
        m.reset(); t0 = time.perf_counter(); m.prefill_batched(p, chunk_size=0)
        b1.append(time.perf_counter() - t0); time.sleep(0.3)
        m.reset(); t0 = time.perf_counter(); m.prefill_batched(p, chunk_size=chunk)
        bc.append(time.perf_counter() - t0); time.sleep(0.3)
        print(f"[ttft rep {r}] A={a[-1]*1e3:.1f}ms B(1chunk)={b1[-1]*1e3:.1f}ms "
              f"B(chunk{chunk})={bc[-1]*1e3:.1f}ms", flush=True)
    return a, b1, bc

ra, rb1, rbc = ttft_ab(args.T, args.chunk, args.reps)
results["ttft"] = []
ma = statistics.median(ra)
for chunk, rb in ((0, rb1), (args.chunk, rbc)):
    mb = statistics.median(rb)
    sp = ma / mb
    print(f"[ttft] T={args.T:4d} chunk={chunk or 'seq_max'}: A(tbt)={ma*1e3:8.1f}ms  "
          f"B(batched)={mb*1e3:8.1f}ms  speedup={sp:5.2f}x  "
          f"improvement={(1-mb/ma)*100:5.1f}%", flush=True)
    results["ttft"].append(dict(T=args.T, chunk=chunk, A_ms=ma*1e3, B_ms=mb*1e3,
                                speedup=sp, A_all=[x*1e3 for x in ra],
                                B_all=[x*1e3 for x in rb]))

if args.json_out:
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=1)
print("[done]", flush=True)
