"""Gate 2 (single-stream byte-identity) + gate-1 cross-reference.

Runs batch=1 paths only; execute once with SK_DYLIB=<baseline build> and once
with SK_DYLIB=<patched build>, then diff the JSON outputs. Also emits each lane
prompt's single-stream chunked-prefill + greedy continuation, the
apples-to-apples (same GEMM paths) reference for batched-lane isolation.

Args: --gguf PATH --model {1.7b,8b} --T 128 --chunk 64 --json_out f.json
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-batched-prefill")))
import numpy as np
from SuperKittens.models.dense.dense_decoder import DenseDecoder, Config

# Same dims table as gates_ttft.py (inlined: importing that module would
# execute its whole bench under this process's dylib).
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
ap.add_argument("--batch_lanes", type=int, default=8)
ap.add_argument("--seq_max", type=int, default=256)
ap.add_argument("--cache_max", type=int, default=512)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--cont", type=int, default=32)
ap.add_argument("--json_out", required=True)
args = ap.parse_args()

cfg = Config(batch=1, seq_max=args.seq_max, cache_max=args.cache_max,
             **DIMS[args.model])
m = DenseDecoder(cfg)
m.load_gguf(args.gguf)
m.bake_and_set_rope()

out = {}

# Single-stream greedy generate (the production batch=1 path: chunked MMA
# prefill + T=1 decode loop). Must be identical between baseline and patched
# dylibs.
fixed = np.array([3000 + 17 * i for i in range(24)], dtype=np.int32) % 50000
toks = m.generate(fixed, max_new_tokens=48, temperature=0.0)
out["single_stream_generate"] = [int(t) for t in toks]
print("[gate2] generate(48):", toks, flush=True)

# Single-stream plain forward (one seq=T prefill) + 16 decode steps.
rng = np.random.default_rng(7)
lane_prompts = rng.integers(10, 100000, size=(args.batch_lanes, args.T)).astype(np.int32)
m.reset()
nt = m.forward(lane_prompts[0])
seq_fw = [int(nt)]
for _ in range(15):
    nt = m.forward(np.array([nt], dtype=np.int32))
    seq_fw.append(int(nt))
out["single_stream_forward_cont"] = seq_fw
print("[gate2] forward+decode(16):", seq_fw, flush=True)

# Per-lane single-stream reference: chunked prefill (same chunk) + greedy cont.
refs = []
for b in range(args.batch_lanes):
    m.reset()
    first = m.prefill_chunked(lane_prompts[b], chunk_size=args.chunk)
    cont = [int(first)]
    cur = first
    for _ in range(args.cont - 1):
        cur = int(m.forward(np.array([cur], dtype=np.int32)))
        cont.append(cur)
    refs.append(cont)
    print(f"[ref] lane {b}: {cont[:8]}...", flush=True)
out["lane_refs_chunked"] = refs

with open(args.json_out, "w") as f:
    json.dump(out, f, indent=1)
print("[done]", flush=True)
