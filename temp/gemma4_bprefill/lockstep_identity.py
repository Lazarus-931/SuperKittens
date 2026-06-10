"""Gate 3 (lockstep, batch=8): forward_batched old-path token identity.

Run once with SK_DYLIB=<base> and once with SK_DYLIB=<patched>; JSONs must match.
T kept small (32) — this only proves the old lockstep path is untouched.
"""
import os, sys, time, json, argparse

sys.path.insert(0, os.path.expanduser(os.environ.get("SK_TREE", "~/sk-gemma-bprefill-r2")))
import numpy as np
import SuperKittens as sk

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--T", type=int, default=32)
ap.add_argument("--cont", type=int, default=16)
ap.add_argument("--window", type=int, default=0)  # 0 = model default
ap.add_argument("--json_out", default="")
args = ap.parse_args()

over = dict(batch=args.batch, seq_max=128, cache_max=512)
if args.window:
    over["window"] = args.window
t0 = time.perf_counter()
m = sk.load("gemma4-12b-unified", **over)
print(f"[setup] loaded in {time.perf_counter()-t0:.1f}s (batch={args.batch})", flush=True)

rng = np.random.default_rng(11)
ids = np.ascontiguousarray(rng.integers(10, 200000, size=(args.batch, args.T)).astype(np.int32))

m.reset()
out = None
for t in range(args.T):
    out = m.forward_batched(np.ascontiguousarray(ids[:, t]))
toks = [np.array(out, dtype=np.int32).copy()]
for _ in range(args.cont):
    toks.append(m.forward_batched(toks[-1]).astype(np.int32).copy())
seqs = np.stack(toks, axis=1)
print(f"[lockstep] per-lane tokens: {seqs.tolist()}", flush=True)

results = {"dylib": os.environ.get("SK_DYLIB", "?"), "lockstep": seqs.tolist()}
if args.json_out:
    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=1)
print("[done]", flush=True)
