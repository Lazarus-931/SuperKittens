"""Gate 2b: chat-templated FLUENT prompts (sharp argmax — the it-model's regime).

Same A/B as gate_fluent (A = token-by-token lockstep, B = sk_gemma4_prefill_batched)
but each lane's prompt is a full gemma chat turn ("continue this passage") spliced
to EXACTLY T tokens with the model-turn cue intact, so greedy continuations are
fluent prose, not repetition loops with knife-edge argmax.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-gemma-bprefill-r4")))
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

from prompts_fluent import TEXTS

def chat_ids(text):
    return list(m.tokenizer.chat(
        [{"role": "user", "content": "Continue this passage:\n\n" + text}], bos=True))

# Common suffix length S of the template (model-turn cue) — splice point.
a, b = chat_ids(TEXTS[0]), chat_ids(TEXTS[1])
S = 0
while S < min(len(a), len(b)) and a[-1 - S] == b[-1 - S]:
    S += 1
print(f"[setup] template suffix tokens S={S}", flush=True)

rows = []
for t in TEXTS:
    full = chat_ids(t)
    assert len(full) >= args.T, f"templated prompt too short: {len(full)}"
    rows.append(full[: args.T - S] + full[-S:])
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
print("[chat] A done", flush=True)

m.reset()
new_next = m.prefill_batched(ids, chunk_size=args.chunk)
new_cont = decode_n(new_next, args.cont)
print("[chat] B done", flush=True)

first_match = [int(base_next[b_]) == int(new_next[b_]) for b_ in range(args.batch)]
lane_match = [bool((base_cont[b_] == new_cont[b_]).all()) for b_ in range(args.batch)]
print(f"[chat] first-token match per lane: {first_match}")
print(f"[chat] {args.cont}-token continuation match per lane: {lane_match}", flush=True)
for b_ in range(args.batch):
    tag = "OK " if lane_match[b_] else "DIV"
    print(f"[chat] lane {b_} {tag} A text: {m.tokenizer.decode(base_cont[b_].tolist())!r}")
    if not lane_match[b_]:
        x, y = base_cont[b_].tolist(), new_cont[b_].tolist()
        d = next(i for i in range(len(x)) if x[i] != y[i])
        print(f"[chat] lane {b_} diverges at idx {d}: A={x[max(0,d-2):d+3]} B={y[max(0,d-2):d+3]}")
        print(f"[chat] lane {b_} B text: {m.tokenizer.decode(new_cont[b_].tolist())!r}")

# identical-prompt + permutation invariants on the templated prompts.
ids_same = np.tile(ids[0], (args.batch, 1))
m.reset()
same_next = m.prefill_batched(ids_same, chunk_size=args.chunk)
same_ok = bool((same_next == same_next[0]).all())
print(f"[chat] identical prompts -> identical next tokens: {same_ok}", flush=True)

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
print(f"[perm] 8-tok continuations permute exactly: {perm_cont_ok}", flush=True)

res = dict(first_match=first_match, cont_match=lane_match, same_prompt_ok=same_ok,
           perm_next_ok=perm_next_ok, perm_cont_ok=perm_cont_ok,
           A_next=[int(x) for x in base_next], B_next=[int(x) for x in new_next],
           A_cont=base_cont.tolist(), B_cont=new_cont.tolist(),
           A_text=[m.tokenizer.decode(base_cont[b_].tolist()) for b_ in range(args.batch)])
if args.json_out:
    json.dump(res, open(args.json_out, "w"), indent=1)
print("[done]", flush=True)
