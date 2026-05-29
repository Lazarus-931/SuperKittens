"""Speculative decoding spike — draft (small) proposes, target (big) verifies.

Greedy accept rule: with the target's next-token argmax sequence aligned to the
draft proposals, accept the longest matching prefix; on first mismatch take the
target's own token (the "free" correction), discard the rest. One target forward
verifies K draft tokens; KV cursor is rewound to drop rejected positions.

Standalone Python orchestration over two SK Qwen handles. No kernel work.
"""
from __future__ import annotations
import argparse, time, sys, os
import numpy as np

import SuperKittens as sk  # rely on PYTHONPATH / cwd to locate the package
import SuperKittens.models.qwen.qwen  # register adapter


def _argmax_row(logits_row: np.ndarray) -> int:
    return int(np.argmax(logits_row.astype(np.float32)))


def step(model, ids) -> int:
    """Forward `ids` (advances KV by len(ids)); return argmax of the LAST row.

    WHY not use model.forward()'s int return: for a multi-token forward SK's
    output_id holds row-0's argmax (the per-row argmax kernel writes past the
    1-int buffer), so the "next token" must be read from the last logits row.
    """
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)).reshape(-1)
    model.forward(ids)
    rows = model.get_logits_rows(len(ids))
    return _argmax_row(rows[-1])


def baseline_decode(target, prompt_ids, n_tokens, eos_ids):
    """Plain autoregressive greedy decode on the target. Returns (toks, dt, fwds)."""
    target.reset()
    nxt = step(target, prompt_ids)        # prefill -> last-row argmax
    t0 = time.perf_counter()
    out = [nxt]
    fwds = 0
    while len(out) < n_tokens and nxt not in eos_ids:
        nxt = step(target, [nxt])
        fwds += 1
        out.append(nxt)
    dt = time.perf_counter() - t0
    return out, dt, fwds


def spec_decode(target, draft, prompt_ids, n_tokens, K, eos_ids):
    """Speculative greedy decode. Returns (toks, dt, target_fwds, accept_counts).

    accept_counts[i] = number of draft tokens accepted in round i (0..K).
    target_fwds counts only the K-token verify forwards (the amortized unit).
    """
    prompt = np.asarray(prompt_ids, dtype=np.int32)
    target.reset(); draft.reset()
    # Prefill both; draft prefill primes its KV, target prefill gives first cur.
    draft.forward(prompt)
    cur = step(target, prompt)           # committed next token after prompt

    out = [cur]
    target_fwds = 0
    accept_counts = []
    t0 = time.perf_counter()

    while len(out) < n_tokens and cur not in eos_ids:
        # 1. Draft proposes K tokens autoregressively, continuing from `cur`.
        #    proposals[j] is the draft's guess for the (j+1)-th token after `cur`.
        proposals = []
        d_in = cur
        for _ in range(K):
            d_in = step(draft, [d_in])
            proposals.append(d_in)

        # 2. Target verifies in ONE forward over [cur, p0, ..., p_{K-1}] (K+1).
        #    Row i predicts the token AFTER input position i:
        #      row 0   -> target token after `cur`        (compare vs proposals[0])
        #      row j   -> target token after proposals[j-1] (compare vs proposals[j])
        #      row K   -> bonus token after proposals[K-1] (free, if all accepted)
        verify_in = np.array([cur] + proposals, dtype=np.int32)
        pos_before = target.get_pos()
        target.forward(verify_in)
        target_fwds += 1
        trows = target.get_logits_rows(len(verify_in))
        tgt_arg = [_argmax_row(trows[i]) for i in range(len(verify_in))]

        # 3. Accept longest matching prefix; first mismatch -> target's token.
        accepted = 0
        for j in range(K):
            if proposals[j] == tgt_arg[j]:
                out.append(proposals[j]); accepted += 1
                if proposals[j] in eos_ids or len(out) >= n_tokens:
                    accepted = -1   # sentinel: stop (eos / budget)
                    break
            else:
                break
        accept_counts.append(accepted if accepted >= 0 else K)

        if accepted == -1:
            break
        # next `cur` = target's token after the last accepted position. With the
        # K+1-row forward this is always tgt_arg[accepted] (the correction at the
        # first mismatch, or the free bonus token when all K accepted).
        next_cur = tgt_arg[accepted]

        # 4. Commit: target KV must cover `cur` + accepted proposals; drop rest.
        committed_this_fwd = 1 + accepted
        target.set_pos(pos_before + committed_this_fwd)
        draft.set_pos(pos_before + committed_this_fwd)   # keep draft aligned

        cur = next_cur
        out.append(cur)

    dt = time.perf_counter() - t0
    return out[:n_tokens], dt, target_fwds, accept_counts


def run(spec_target, spec_draft, prompt, n_tokens, K, cache_max, label):
    target = sk.load(spec_target, cache_max=cache_max)
    draft = sk.load(spec_draft, cache_max=cache_max)
    tok = target.tokenizer
    eos_ids = set(getattr(tok, "eos_ids", None) or ([tok.eos_id] if getattr(tok, "eos_id", None) is not None else []))
    ids = tok.chat([{"role": "user", "content": prompt}], add_generation_prompt=True, bos=False) \
        if hasattr(tok, "chat") else tok.encode(prompt)
    ids = np.asarray(ids, dtype=np.int32)
    print(f"[{label}] prompt={prompt!r}  prompt_len={len(ids)}  K={K}  n={n_tokens}")

    # Baseline (median of reps).
    base_reps = []
    base_text = None
    for _ in range(5):
        bt, bdt, _ = baseline_decode(target, ids, n_tokens, eos_ids)
        base_reps.append(len(bt) / bdt)
        base_text = tok.decode(bt, skip_special=True)
    base_tps = float(np.median(base_reps))

    # Spec-decode (median of reps).
    spec_reps = []
    accept_all = []
    spec_text = None
    spec_fwds = None
    for _ in range(5):
        st, sdt, tf, ac = spec_decode(target, draft, ids, n_tokens, K, eos_ids)
        spec_reps.append(len(st) / sdt)
        accept_all.extend(ac)
        spec_text = tok.decode(st, skip_special=True)
        spec_fwds = tf
        spec_ntok = len(st)
    spec_tps = float(np.median(spec_reps))
    mean_accept = float(np.mean(accept_all)) if accept_all else 0.0
    # Effective amortization: accepted tokens per target verify forward.
    eff = (spec_ntok / spec_fwds) if spec_fwds else 0.0

    print(f"[{label}] baseline   : {base_tps:7.2f} tok/s")
    print(f"[{label}] spec-decode : {spec_tps:7.2f} tok/s   (x{spec_tps/base_tps:.2f})")
    print(f"[{label}] mean accept/round (of K={K}): {mean_accept:.2f}   tokens/target-fwd: {eff:.2f}")
    print(f"[{label}] baseline text: {base_text[:90]!r}")
    print(f"[{label}] spec     text: {spec_text[:90]!r}")
    print(f"[{label}] texts match : {base_text[:n_tokens]==spec_text[:n_tokens]}")
    target.close(); draft.close()
    return dict(label=label, base_tps=base_tps, spec_tps=spec_tps,
                mean_accept=mean_accept, eff=eff, K=K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qwen3-4b-q4km")
    ap.add_argument("--draft", default="qwen3-0.6b")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--cache-max", type=int, default=1024)
    args = ap.parse_args()

    prompts = [
        ("creative", "Generate a poem about pizza dough"),
        ("low-entropy", "Recite the first 20 prime numbers"),
    ]
    for label, p in prompts:
        run(args.target, args.draft, p, args.n, args.K, args.cache_max, label)
        print()


if __name__ == "__main__":
    main()
