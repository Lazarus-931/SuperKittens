"""Inherited-divergence demo on the BASE dylib (batch=1, ZERO new code).

For each fluent prompt: path1 = one forward(T) (q_seq>1 attention path),
path2 = T x forward(seq=1) (decode fast path), then 32 greedy continuations.
Any divergence here is pre-existing seq>1-vs-seq=1 reduction-order numerics —
the same numeric gap the batched-prefill A/B comparison crosses.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-gemma-bprefill-r4")))
import numpy as np
import SuperKittens as sk

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=120)
ap.add_argument("--cont", type=int, default=32)
ap.add_argument("--json_out", default="")
args = ap.parse_args()

m = sk.load("gemma4-12b-unified", seq_max=128, cache_max=512, window=512)
print(f"[setup] batch=1 window={m.cfg.window} dylib={os.environ.get('SK_DYLIB','?')}",
      flush=True)

from prompts_fluent import TEXTS

def cont32(n):
    toks = [int(n[0])]
    for _ in range(args.cont - 1):
        toks.append(int(m.forward(np.array([toks[-1]], dtype=np.int32))[0]))
    return toks

results = []
for i, t in enumerate(TEXTS):
    ids = np.asarray(m.tokenizer.encode(t)[: args.T], dtype=np.int32)
    m.reset()
    n1 = m.forward(ids)               # one seq>1 chunk
    c1 = cont32(n1)
    m.reset()
    n2 = None
    for k in range(len(ids)):         # token-by-token
        n2 = m.forward(ids[k:k+1])
    c2 = cont32(n2)
    match = c1 == c2
    d = next((j for j in range(args.cont) if c1[j] != c2[j]), -1)
    print(f"[sp] prompt {i} match={match} first_div_idx={d}", flush=True)
    results.append(dict(prompt=i, match=match, div_idx=d, c1=c1, c2=c2))

n_match = sum(r["match"] for r in results)
print(f"[sp] forward(T) vs T x forward(1): {n_match}/8 32-tok matches", flush=True)
if args.json_out:
    json.dump(results, open(args.json_out, "w"), indent=1)
print("[done]", flush=True)
