"""Pre-existing-path divergence demo (run on the BASE dylib, batch=1).

Compares two EXISTING prefill paths for the same prompt:
  path1 = one forward(T) call   (seq>1 'original' attention path)
  path2 = T x forward(seq=1)    (decode fast-path attention)
then 32 greedy continuations each. Divergence here is inherited seq>1-vs-seq=1
reduction-order numerics — present without any batched-prefill code.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-gemma-bprefill-r2")))
import numpy as np
import SuperKittens as sk

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--cont", type=int, default=32)
ap.add_argument("--random", action="store_true")
ap.add_argument("--json_out", default="")
args = ap.parse_args()

m = sk.load("gemma4-12b-unified", seq_max=128, cache_max=512, window=512)
print(f"[setup] batch=1 window={m.cfg.window}", flush=True)

if args.random:
    rng = np.random.default_rng(7)
    ids = rng.integers(10, 200000, size=args.T).astype(np.int32)
else:
    txt = ("The history of bread baking stretches back over ten thousand years, "
           "beginning with flat unleavened cakes cooked on hot stones. ") * 6
    ids = np.asarray(m.tokenizer.encode(txt)[:args.T], dtype=np.int32)

def cont_from(first, n):
    toks = [int(first)]
    for _ in range(n - 1):
        toks.append(int(m.forward(np.array([toks[-1]], dtype=np.int32))[0]))
    return toks

m.reset()
n1 = m.forward(ids)            # one seq=T call
c1 = cont_from(n1[0], args.cont)

m.reset()
n2 = None
for t in range(args.T):        # T x seq=1 calls
    n2 = m.forward(ids[t:t+1])
c2 = cont_from(n2[0], args.cont)

match = c1 == c2
print(f"[seqpaths] first token equal: {int(n1[0]) == int(n2[0])}")
print(f"[seqpaths] {args.cont}-token continuation equal: {match}")
if not match:
    d = next(i for i in range(len(c1)) if c1[i] != c2[i])
    print(f"[seqpaths] diverges at idx {d}: seqT={c1[max(0,d-2):d+3]} tbt={c2[max(0,d-2):d+3]}")
if args.json_out:
    json.dump(dict(seqT_first=int(n1[0]), tbt_first=int(n2[0]), seqT_cont=c1, tbt_cont=c2),
              open(args.json_out, "w"), indent=1)
print("[done]", flush=True)
