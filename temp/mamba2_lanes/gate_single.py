"""Gate 2: single-stream path unchanged. Run with SK_DYLIB=<base|new> and
TAG=<name>; saves last-logits bytes + greedy tokens for byte compare."""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from SuperKittens.inference import registry

SNAP = os.environ["SNAP"]
TAG = os.environ["TAG"]
OUT = os.path.dirname(os.path.abspath(__file__))

m = registry.load("mamba2-130m", snapshot=SNAP)  # batch=1 default
ids = [510, 5347, 273, 6181, 310]  # "The capital of France is"-ish fixed ids
m.reset()
toks = m.generate(ids, max_new_tokens=24)
logits = m.get_last_logits()
np.save(os.path.join(OUT, f"single_{TAG}_logits.npy"), logits)
with open(os.path.join(OUT, f"single_{TAG}_toks.txt"), "w") as f:
    f.write(" ".join(map(str, toks)))
print(f"{TAG}: toks={toks}", flush=True)
print("SINGLE_DONE", flush=True)
