"""Gate 0/3 (single-stream, batch=1): 12B coherence + old-path token identity.

Run once with SK_DYLIB=<base> and once with SK_DYLIB=<patched>; JSONs must match
token-for-token (the patched paths are additive: batched_prefill defaults 0).
"""
import os, sys, time, json, argparse

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-gemma-bprefill-r2")))
import numpy as np
import SuperKittens as sk

ap = argparse.ArgumentParser()
ap.add_argument("--json_out", default="")
args = ap.parse_args()

t0 = time.perf_counter()
m = sk.load("gemma4-12b-unified", seq_max=128, cache_max=512)
print(f"[setup] loaded in {time.perf_counter()-t0:.1f}s (batch=1)", flush=True)

results = {"dylib": os.environ.get("SK_DYLIB", "?")}

# Gate 0: coherence (greedy, 32+ tokens).
ids = np.asarray(m.tokenizer.chat([{"role": "user",
                                    "content": "Generate a poem about pizza dough"}],
                                  bos=True), dtype=np.int32)
m.reset()
out = m.generate(ids, max_new_tokens=48, temperature=0.0)
text = m.tokenizer.decode(out, skip_special=True)
print(f"[coherence] {text!r}", flush=True)
results["coherence_ids"] = [int(x) for x in out]
results["coherence_text"] = text

# Plain forward + 16 decode steps from a fixed token prompt.
pids = np.asarray(m.tokenizer.encode("The capital of France is"), dtype=np.int32)
m.reset()
nxt = m.forward(pids)
dec = [int(nxt[0])]
for _ in range(16):
    nxt = m.forward(np.array([dec[-1]], dtype=np.int32))
    dec.append(int(nxt[0]))
print(f"[decode_ids] {dec}", flush=True)
results["decode_ids"] = dec

if args.json_out:
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=1)
print("[done]", flush=True)
