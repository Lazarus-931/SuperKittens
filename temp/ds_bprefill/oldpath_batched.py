"""Old-path byte-identity probe, lockstep-serving half. Run against the
pristine-main dylib and the patched dylib (SK_DYLIB env decides); outputs must
match exactly. batch=8 handle, token-by-token lockstep prefill (T=64) + 32
lockstep decode steps — the existing serving path, untouched by this branch.

Writes artifacts/oldpath_batched_<tag>.json (tag = $TAG).
"""
import os, sys, time, json
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-bprefill-k9")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load
from gates_lane import TEXTS

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")
TAG = os.environ.get("TAG", "untagged")
N, T = 8, 64


def main():
    t0 = time.perf_counter()
    m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF,
             batch=N, seq_max=T, cache_max=192)
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    ids_mat = np.stack([np.asarray(m.tokenizer.encode(tx + " " + tx, bos=True)[:T],
                                   dtype=np.int32) for tx in TEXTS])
    m.reset()
    cur = np.zeros(N, dtype=np.int32)
    nxt = None
    for s in range(T):
        cur[:] = ids_mat[:, s]
        nxt = m._forward_batched(cur)
    rows = [np.asarray(nxt, dtype=np.int32).copy()]
    cur = rows[0].copy()
    for _ in range(32):
        cur = np.asarray(m._forward_batched(cur), dtype=np.int32).copy()
        rows.append(cur)
    toks = np.stack(rows, axis=1)  # [N, 33]

    res = {"tag": TAG, "tokens": toks.tolist()}
    print(f"[lockstep] next per lane = {toks[:,0].tolist()}", flush=True)
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    out = os.path.join(ROOT, "artifacts", f"oldpath_batched_{TAG}.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"OLDPATH_BATCHED_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
