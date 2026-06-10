"""Gate 1 with REAL text prompts (peaked logit distributions) + bitwise
lane-permutation test.

Random-token prompts give degenerate newline-heavy continuations where the
pre-existing seq1-vs-seqN attention reduction-order noise flips argmax ties;
real prompts are the representative serving case.
"""
import os, sys, time, json, argparse

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-gemma-bprefill-r2")))
import numpy as np
import SuperKittens as sk

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--T", type=int, default=128)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--cont", type=int, default=32)
ap.add_argument("--json_out", default="")
args = ap.parse_args()

m = sk.load("gemma4-12b-unified", batch=args.batch, seq_max=128, cache_max=512,
            window=512)
print(f"[setup] batch={args.batch} window={m.cfg.window}", flush=True)

TEXTS = [
    "The history of bread baking stretches back over ten thousand years, beginning with flat unleavened cakes cooked on hot stones. ",
    "Metal compute shaders allow a programmer to dispatch thousands of threadgroups across the GPU, each cooperating through threadgroup memory. ",
    "In the deep ocean, hydrothermal vents support entire ecosystems that never see sunlight, powered instead by chemosynthetic bacteria. ",
    "A well-tuned sourdough starter doubles in volume within four to six hours of feeding, producing a pleasant aroma of ripe fruit and yogurt. ",
    "The transformer architecture replaced recurrence with attention, letting every token attend directly to every other token in the sequence. ",
    "Glaciers carve valleys over millennia, grinding bedrock into fine flour that turns meltwater lakes a striking shade of turquoise blue. ",
    "Compilers translate high level source code into machine instructions through stages of parsing, optimization, and code generation. ",
    "The annual monsoon arrives on the southwest coast in early June, bringing weeks of heavy rain that replenish rivers and aquifers. ",
]

rows = []
for t in TEXTS:
    ids = m.tokenizer.encode(t * 10)
    assert len(ids) >= args.T, f"prompt too short: {len(ids)}"
    rows.append(ids[:args.T])
ids = np.ascontiguousarray(np.array(rows, dtype=np.int32))

def prefill_tbt(p):
    out = None
    for t in range(p.shape[1]):
        out = m.forward_batched(np.ascontiguousarray(p[:, t]))
    return out

def decode_n(first, n):
    toks = [np.array(first, dtype=np.int32).copy()]
    for _ in range(n - 1):
        toks.append(m.forward_batched(toks[-1]).astype(np.int32).copy())
    return np.stack(toks, axis=1)

m.reset()
base_next = prefill_tbt(ids)
base_cont = decode_n(base_next, args.cont)
print("[real] A done", flush=True)

m.reset()
new_next = m.prefill_batched(ids, chunk_size=args.chunk)
new_cont = decode_n(new_next, args.cont)
print("[real] B done", flush=True)

first_match = [int(base_next[b]) == int(new_next[b]) for b in range(args.batch)]
lane_match = [bool((base_cont[b] == new_cont[b]).all()) for b in range(args.batch)]
print(f"[real] first-token match per lane: {first_match}")
print(f"[real] {args.cont}-token continuation match per lane: {lane_match}", flush=True)
for b in range(args.batch):
    if not lane_match[b]:
        a, c = base_cont[b].tolist(), new_cont[b].tolist()
        d = next(i for i in range(len(a)) if a[i] != c[i])
        print(f"[real] lane {b} diverges at idx {d}: A={a[max(0,d-2):d+3]} B={c[max(0,d-2):d+3]}")
        print(f"[real] lane {b} A text: {m.tokenizer.decode(base_cont[b].tolist())!r}")
        print(f"[real] lane {b} B text: {m.tokenizer.decode(new_cont[b].tolist())!r}")

# Bitwise lane-permutation test (same code path both runs): permuting the
# prompts must permute next tokens AND short continuations exactly — any
# cross-lane leakage (KV slice / row indexing) breaks this.
perm = np.array([3, 1, 4, 0, 7, 5, 2, 6])
m.reset()
pnext = m.prefill_batched(np.ascontiguousarray(ids[perm]), chunk_size=args.chunk)
pcont = decode_n(pnext, 8)
m.reset()
qnext = m.prefill_batched(ids, chunk_size=args.chunk)
qcont = decode_n(qnext, 8)
perm_next_ok = bool((np.array(pnext) == np.array(qnext)[perm]).all())
perm_cont_ok = bool((pcont == qcont[perm]).all())
print(f"[perm] next tokens permute exactly: {perm_next_ok}")
print(f"[perm] 8-token continuations permute exactly: {perm_cont_ok}", flush=True)

res = dict(first_match=first_match, cont_match=lane_match,
           perm_next_ok=perm_next_ok, perm_cont_ok=perm_cont_ok,
           base_next=[int(x) for x in base_next], new_next=[int(x) for x in new_next],
           base_cont=base_cont.tolist(), new_cont=new_cont.tolist())
if args.json_out:
    json.dump(res, open(args.json_out, "w"), indent=1)
print("[done]", flush=True)
