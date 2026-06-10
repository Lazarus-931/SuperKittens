"""Per-lane batch=1 references: single-stream T>1 prefill + 32 greedy cont for
each of the 8 gate prompts (the now-correct post-row-grid-fix path), plus
token-by-token batch=1 spot checks on prompts 0 and 1, plus the secondary
TTFT comparison number (sum of 8 sequential single-stream T>1 prefills).

ONE batch=1 handle. Writes artifacts/gates_single.json.
"""
import os, sys, time, json
import numpy as np

ROOT = os.path.expanduser("~/sk-ds-bprefill-k9")
sys.path.insert(0, ROOT)
from SuperKittens.inference.registry import load
from gates_lane import TEXTS, T, CONT

SNAP = os.path.expanduser("~/ds_fit/DeepSeek-V2-Lite")
GGUF = os.path.expanduser("~/qwen-gguf/DeepSeek-V2-Lite.Q4_K_M.gguf")


def main():
    t0 = time.perf_counter()
    m = load("deepseek-v2-lite", snapshot=SNAP, gguf=GGUF,
             batch=1, seq_max=T, cache_max=192)
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    lanes = []
    for tx in TEXTS:
        ids = m.tokenizer.encode(tx + " " + tx, bos=True)
        assert len(ids) >= T
        lanes.append(np.asarray(ids[:T], dtype=np.int32))

    res = {"T": T, "CONT": CONT}
    seq_tokens, ttfts = [], []
    for i, ids in enumerate(lanes):
        m.reset()
        t = time.perf_counter()
        nxt = m.forward(ids)
        ttfts.append((time.perf_counter() - t) * 1e3)
        toks = [int(nxt)]
        for _ in range(CONT):
            toks.append(m.forward(np.asarray([toks[-1]], dtype=np.int32)))
        seq_tokens.append(toks)
        print(f"[lane {i}] T>1 next={toks[0]} ttft={ttfts[-1]:.1f}ms", flush=True)
    res["seq_tokens"] = seq_tokens
    res["seq_ttft_ms"] = ttfts
    res["seq_ttft_sum_ms"] = float(np.sum(ttfts))
    print(f"[secondary] 8x sequential single-stream T>1 prefill sum = "
          f"{res['seq_ttft_sum_ms']:.1f}ms", flush=True)

    # token-by-token batch=1 spot checks (prompts 0, 1)
    tbt_tokens = []
    for i in (0, 1):
        ids = lanes[i]
        m.reset()
        nxt = None
        for s in range(T):
            nxt = m.forward(np.asarray([ids[s]], dtype=np.int32))
        toks = [int(nxt)]
        for _ in range(CONT):
            toks.append(m.forward(np.asarray([toks[-1]], dtype=np.int32)))
        tbt_tokens.append(toks)
        match = toks == seq_tokens[i]
        print(f"[lane {i}] tbt-vs-T>1 33-token match = {match}", flush=True)
    res["tbt_tokens_lane01"] = tbt_tokens
    res["tbt_match_lane01"] = [tbt_tokens[k] == seq_tokens[k] for k in range(2)]

    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    out = os.path.join(ROOT, "artifacts", "gates_single.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"GATES_SINGLE_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
