"""Speculative decoding — draft (small) proposes K, target (big) verifies all K
in ONE seq=(K+1) forward THROUGH gemm_mma (the batched MMA path landed on
dev-sk-batched-gemm / PR #55). Accept longest matching greedy prefix.

Premise now satisfied (was 0.54x in the May spike):
  - mha_causal chunked-forward interior logits FIXED (PR #54).
  - seq>1 forward routes QKV/V/O/gate/up/down through gemm_mma, so a seq=K
    verify costs << K x a seq=1 decode (sub-linear; commit 53da587).

Greedy accept: align target's next-token argmax to the draft proposals, accept
the longest matching prefix, take target's own token at the first mismatch (the
free correction), rewind the KV cursor to drop rejected positions.
"""
from __future__ import annotations
import argparse, time
import numpy as np

import SuperKittens as sk
import SuperKittens.models.qwen.qwen  # register adapter


def _argmax_row(r) -> int:
    return int(np.argmax(r.astype(np.float32)))


def step(model, ids) -> int:
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)).reshape(-1)
    model.forward(ids)
    return _argmax_row(model.get_logits_rows(len(ids))[-1])


def baseline_decode(target, prompt_ids, n_tokens, eos_ids):
    target.reset()
    nxt = step(target, prompt_ids)
    t0 = time.perf_counter()
    out = [nxt]
    while len(out) < n_tokens and nxt not in eos_ids:
        nxt = step(target, [nxt])
        out.append(nxt)
    dt = time.perf_counter() - t0
    return out, dt


def spec_decode(target, draft, prompt_ids, n_tokens, K, eos_ids):
    prompt = np.asarray(prompt_ids, dtype=np.int32)
    target.reset(); draft.reset()
    draft.forward(prompt)
    cur = step(target, prompt)

    out = [cur]
    target_fwds = 0
    accept_counts = []
    t0 = time.perf_counter()

    while len(out) < n_tokens and cur not in eos_ids:
        # 1. Draft proposes K tokens autoregressively from `cur`.
        proposals = []
        d_in = cur
        for _ in range(K):
            d_in = step(draft, [d_in])
            proposals.append(d_in)

        # 2. Target verifies [cur, p0..p_{K-1}] in ONE forward (K+1 rows).
        verify_in = np.array([cur] + proposals, dtype=np.int32)
        pos_before = target.get_pos()
        target.forward(verify_in)
        target_fwds += 1
        trows = target.get_logits_rows(len(verify_in))
        tgt_arg = [_argmax_row(trows[i]) for i in range(len(verify_in))]

        # 3. Accept longest matching greedy prefix.
        accepted = 0
        stop = False
        for j in range(K):
            if proposals[j] == tgt_arg[j]:
                out.append(proposals[j]); accepted += 1
                if proposals[j] in eos_ids or len(out) >= n_tokens:
                    stop = True
                    break
            else:
                break
        accept_counts.append(accepted)
        if stop:
            break

        next_cur = tgt_arg[accepted]   # correction at first mismatch, or bonus token
        committed = 1 + accepted
        target.set_pos(pos_before + committed)
        draft.set_pos(pos_before + committed)
        cur = next_cur
        out.append(cur)

    dt = time.perf_counter() - t0
    return out[:n_tokens], dt, target_fwds, accept_counts


def run(target, draft, ids, n_tokens, K, eos_ids, label, reps=5):
    base_reps, base_text = [], None
    for _ in range(reps):
        bt, bdt = baseline_decode(target, ids, n_tokens, eos_ids)
        base_reps.append(len(bt) / bdt); base_text = bt
    base_tps = float(np.median(base_reps))

    spec_reps, accept_all, spec_toks, spec_fwds = [], [], None, None
    for _ in range(reps):
        st, sdt, tf, ac = spec_decode(target, draft, ids, n_tokens, K, eos_ids)
        spec_reps.append(len(st) / sdt); accept_all.extend(ac)
        spec_toks = st; spec_fwds = tf
    spec_tps = float(np.median(spec_reps))
    mean_accept = float(np.mean(accept_all)) if accept_all else 0.0
    eff = (len(spec_toks) / spec_fwds) if spec_fwds else 0.0
    # Lossless: spec ids must equal baseline ids exactly (greedy, same target).
    n = min(len(base_text), len(spec_toks))
    lossless = base_text[:n] == spec_toks[:n]
    first_div = next((i for i in range(n) if base_text[i] != spec_toks[i]), -1)

    print(f"[{label}] K={K}  baseline={base_tps:7.2f} t/s  spec={spec_tps:7.2f} t/s  "
          f"x{spec_tps/base_tps:.3f}  accept/{K}={mean_accept:.2f}  tok/fwd={eff:.2f}  "
          f"lossless={lossless}" + ("" if lossless else f" (first_div@{first_div})"))
    return dict(label=label, K=K, base_tps=base_tps, spec_tps=spec_tps,
                ratio=spec_tps / base_tps, mean_accept=mean_accept, eff=eff,
                lossless=lossless, first_div=first_div)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qwen3-4b-q4km")
    ap.add_argument("--draft", default="qwen3-0.6b")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--K", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--cache-max", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--prompt-len", type=int, default=16,
                    help="synthetic prompt length when no tokenizer is available")
    args = ap.parse_args()

    import _env  # noqa: F401  (only matters for local no-install runs)
    target = sk.load(args.target, cache_max=args.cache_max)
    draft = sk.load(args.draft, cache_max=args.cache_max)
    tok = getattr(target, "tokenizer", None)
    eos_ids = set()
    if tok is not None:
        eos_ids = set(getattr(tok, "eos_ids", None)
                      or ([tok.eos_id] if getattr(tok, "eos_id", None) is not None else []))

    prompts = []
    if tok is not None and hasattr(tok, "chat"):
        for label, p in (("creative", "Generate a poem about pizza dough"),
                         ("low-entropy", "Recite the first 20 prime numbers")):
            ids = np.asarray(tok.chat([{"role": "user", "content": p}],
                                      add_generation_prompt=True, bos=False),
                             dtype=np.int32)
            prompts.append((label, ids))
    else:
        # Tokenizer-free: synthetic prompt of real-ish ids (deterministic).
        rng = np.random.default_rng(0)
        ids = rng.integers(10, 4000, size=args.prompt_len).astype(np.int32)
        prompts.append((f"synthetic(len={args.prompt_len})", ids))

    results = []
    for label, ids in prompts:
        for K in args.K:
            results.append(run(target, draft, ids, args.n, K, eos_ids, label, args.reps))
    target.close(); draft.close()
    return results


if __name__ == "__main__":
    main()
