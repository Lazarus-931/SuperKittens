"""Old-path byte-identity probe, single-stream half. Run once against the
pristine-main dylib and once against the patched dylib (SK_DYLIB env decides);
outputs must match exactly. batch=1 handle.

  (a) generate: short prompt, one T>1 prefill forward + 47 greedy steps
  (b) T=128 T>1 prefill + 32 greedy continuation

Writes artifacts/oldpath_single_<tag>.json (tag = $TAG).
"""
import os, sys, time, json
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-bprefill-k9")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load
from gates_lane import TEXTS, T

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")
TAG = os.environ.get("TAG", "untagged")


def main():
    t0 = time.perf_counter()
    m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF,
             batch=1, seq_max=T, cache_max=192)
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)
    res = {"tag": TAG}

    # (a) generate-style: T>1 prompt forward + greedy steps
    ids = np.asarray(m.tokenizer.encode(
        "The capital of France is a city known for", bos=True), dtype=np.int32)
    m.reset()
    toks = [int(m.forward(ids))]
    for _ in range(47):
        toks.append(m.forward(np.asarray([toks[-1]], dtype=np.int32)))
    res["gen48"] = toks
    print(f"[gen48] {toks[:12]}...", flush=True)
    print(f"[gen48 text] {m.tokenizer.decode(toks)[:160]!r}", flush=True)

    # (b) T=128 prefill + 32 cont (prompt 2)
    p = np.asarray(m.tokenizer.encode(TEXTS[2] + " " + TEXTS[2], bos=True)[:T],
                   dtype=np.int32)
    m.reset()
    toks = [int(m.forward(p))]
    for _ in range(32):
        toks.append(m.forward(np.asarray([toks[-1]], dtype=np.int32)))
    res["prefill128_cont"] = toks
    print(f"[prefill128] next={toks[0]}", flush=True)

    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    out = os.path.join(ROOT, "artifacts", f"oldpath_single_{TAG}.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"OLDPATH_SINGLE_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
