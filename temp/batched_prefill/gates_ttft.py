"""Batched-prefill gates + serving-TTFT A/B (single process, one model handle).

A = baseline serving prefill: token-by-token lockstep (forward_batched seq=1, T steps)
B = new sk_qwen_prefill_batched (chunked M=batch*seq MMA prefill)

Gate 1: per-lane greedy 32-token continuation after B must match A's, per lane.
Gate 3: TTFT (reset -> all-lanes-first-token), 2 warmups + 7 reps median, 0.3s gaps.

Env: SK_DYLIB, SK_METALLIB=/nonexistent, SK_METAL_SRC_FALLBACK.
Args: --gguf PATH --model {1.7b,8b} --T 128 [--T2 256] --chunk 64
"""
import os, sys, time, json, argparse, statistics

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-batched-prefill")))
import numpy as np
from SuperKittens.models.dense.dense_decoder import DenseDecoder, Config

DIMS = {
    "1.7b": dict(n_layers=28, d_model=2048, n_heads=16, n_kv_heads=8, head_dim=128,
                 n_int=6144, vocab_size=151936, eps=1e-6,
                 rope_freq_base=1_000_000.0, tie_word_embeddings=1, use_qk_norm=1),
    "8b":   dict(n_layers=36, d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128,
                 n_int=12288, vocab_size=151936, eps=1e-6,
                 rope_freq_base=1_000_000.0, tie_word_embeddings=0, use_qk_norm=1),
}

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--model", default="1.7b", choices=list(DIMS))
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seq_max", type=int, default=256)
ap.add_argument("--cache_max", type=int, default=512)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--T2", type=int, default=0)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--cont", type=int, default=32)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--json_out", default="")
args = ap.parse_args()

cfg = Config(batch=args.batch, seq_max=args.seq_max, cache_max=args.cache_max,
             **DIMS[args.model])
m = DenseDecoder(cfg)
m.load_gguf(args.gguf)
m.bake_and_set_rope()
print(f"[setup] model={args.model} batch={args.batch} seq_max={args.seq_max} "
      f"cache_max={args.cache_max}", flush=True)

rng = np.random.default_rng(7)
def make_prompts(T):
    return np.ascontiguousarray(rng.integers(10, 100000, size=(args.batch, T)).astype(np.int32))

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

results = {"model": args.model, "batch": args.batch}

# ---- Gate 1: lane isolation ------------------------------------------------
ids = make_prompts(args.T)
print(f"[gate1] T={args.T} chunk={args.chunk} cont={args.cont}", flush=True)

m.reset()
base_next = prefill_tbt(ids)
base_cont = decode_n(base_next, args.cont)

m.reset()
new_next = m.prefill_batched(ids, chunk_size=args.chunk)
new_cont = decode_n(new_next, args.cont)

m.reset()
new_next_1c = m.prefill_batched(ids, chunk_size=0)  # single chunk (<= seq_max)

lane_match = [bool((base_cont[b] == new_cont[b]).all()) for b in range(args.batch)]
first_match = [int(base_next[b]) == int(new_next[b]) for b in range(args.batch)]
chunk_vs_1c = [int(new_next[b]) == int(new_next_1c[b]) for b in range(args.batch)]
tok_ok = bool((new_cont >= 0).all() and (new_cont < cfg.vocab_size).all())
print(f"[gate1] first-token match (B vs A) per lane: {first_match}")
print(f"[gate1] {args.cont}-token continuation match per lane: {lane_match}")
print(f"[gate1] chunked({args.chunk}) vs single-chunk next-token match: {chunk_vs_1c}")
print(f"[gate1] all tokens in-vocab/finite-argmax: {tok_ok}", flush=True)
for b in range(args.batch):
    if not lane_match[b]:
        a, c = base_cont[b].tolist(), new_cont[b].tolist()
        d = next(i for i in range(len(a)) if a[i] != c[i])
        print(f"[gate1] lane {b} diverges at cont idx {d}: A={a[max(0,d-2):d+3]} B={c[max(0,d-2):d+3]}")
results["gate1"] = {"first_match": first_match, "cont_match": lane_match,
                    "chunk_vs_single": chunk_vs_1c, "tokens_in_vocab": tok_ok,
                    "base_next": [int(x) for x in base_next],
                    "new_next": [int(x) for x in new_next],
                    "base_cont": base_cont.tolist(), "new_cont": new_cont.tolist()}

# lane-permutation cross-check: distinct prompts should give distinct lanes,
# identical prompts identical outputs.
ids_same = np.tile(ids[0], (args.batch, 1))
m.reset()
same_next = m.prefill_batched(ids_same, chunk_size=args.chunk)
print(f"[gate1b] identical prompts -> identical next tokens: "
      f"{bool((same_next == same_next[0]).all())} ({same_next.tolist()})", flush=True)
results["gate1b_same_prompt_next"] = [int(x) for x in same_next]

# ---- Gate 3: TTFT A/B ------------------------------------------------------
def ttft_ab(T, chunk, reps):
    p = make_prompts(T)
    for _ in range(2):  # warmups
        m.reset(); prefill_tbt(p); time.sleep(0.3)
        m.reset(); m.prefill_batched(p, chunk_size=chunk); time.sleep(0.3)
    a, b = [], []
    for _ in range(reps):
        m.reset(); t0 = time.perf_counter(); prefill_tbt(p)
        a.append(time.perf_counter() - t0); time.sleep(0.3)
        m.reset(); t0 = time.perf_counter(); m.prefill_batched(p, chunk_size=chunk)
        b.append(time.perf_counter() - t0); time.sleep(0.3)
    return statistics.median(a), statistics.median(b), a, b

Ts = [args.T] + ([args.T2] if args.T2 else [])
results["ttft"] = []
for T in Ts:
    for chunk in (0, args.chunk):
        ma, mb, ra, rb = ttft_ab(T, chunk, args.reps)
        sp = ma / mb
        print(f"[ttft] T={T:4d} chunk={chunk or 'seq_max'}: A(tbt)={ma*1e3:8.1f}ms  "
              f"B(batched)={mb*1e3:8.1f}ms  speedup={sp:5.2f}x  "
              f"improvement={(1-mb/ma)*100:5.1f}%", flush=True)
        results["ttft"].append(dict(T=T, chunk=chunk, A_ms=ma*1e3, B_ms=mb*1e3,
                                    speedup=sp, A_all=[x*1e3 for x in ra],
                                    B_all=[x*1e3 for x in rb]))

if args.json_out:
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=1)
print("[done]", flush=True)
