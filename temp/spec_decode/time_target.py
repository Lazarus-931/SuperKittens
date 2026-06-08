"""Per-seq target forward timing: the load-bearing spec-decode metric.

A spec step costs ~ K draft-forwards + 1 target seq=(K+1) verify. It only beats
baseline if that verify forward approaches a seq=1 target decode (amortization),
since a win needs cost(verify) < accepted x cost(seq=1 decode). Measures the
target's seq=1..8 forward time and the implied break-even mean-accept per K.
"""
import argparse, time
import numpy as np
import SuperKittens as sk
import SuperKittens.models.qwen.qwen  # noqa


def time_seq(m, prompt, seq, iters=15):
    chunk = np.random.default_rng(seq).integers(10, 4000, size=seq).astype(np.int32)
    m.reset(); m.forward(prompt); m.forward(chunk); m.get_logits_rows(seq)  # warm
    ts = []
    for _ in range(iters):
        m.reset(); m.forward(prompt)
        t0 = time.perf_counter()
        m.forward(chunk); m.get_logits_rows(seq)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b-q4km")
    ap.add_argument("--draft", default="qwen3-0.6b")
    ap.add_argument("--cache-max", type=int, default=1024)
    a = ap.parse_args()
    import os
    tkw = dict(cache_max=a.cache_max); dkw = dict(cache_max=a.cache_max)
    if os.environ.get("SK_TGT_SNAP"): tkw["snapshot"] = os.environ["SK_TGT_SNAP"]
    if os.environ.get("SK_DRF_SNAP"): dkw["snapshot"] = os.environ["SK_DRF_SNAP"]
    tgt = sk.load(a.model, **tkw)
    drf = sk.load(a.draft, **dkw)
    tgt.set_lm_head_all_rows(True)
    prompt = np.random.default_rng(0).integers(10, 4000, size=16).astype(np.int32)
    print(f"=== target={a.model} draft={a.draft} ===", flush=True)
    t1 = time_seq(tgt, prompt, 1)
    d1 = time_seq(drf, prompt, 1)
    print(f"target seq=1: {t1:.2f} ms   draft seq=1: {d1:.2f} ms", flush=True)
    for K in (2, 4, 6):
        tv = time_seq(tgt, prompt, K + 1)
        # spec step cost: K draft seq=1 + 1 target seq=(K+1) verify
        step_cost = K * d1 + tv
        # baseline cost to emit `accept+1` tokens = (accept+1) * t1
        # break-even accept: step_cost == (accept+1)*t1  ->  accept_be
        accept_be = step_cost / t1 - 1.0
        print(f"K={K}: target verify seq={K+1} = {tv:.2f} ms  "
              f"(vs {(K+1)*t1:.2f} ms = {K+1}x seq1)  "
              f"step_cost={step_cost:.2f} ms  break-even accept={accept_be:.2f}", flush=True)
    tgt.close(); drf.close()


if __name__ == "__main__":
    main()
