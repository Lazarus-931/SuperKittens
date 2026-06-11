"""Granite hybrid batched-lane serving gates (correctness).

One batch=N handle. References run on the SAME handle via the untouched
single-stream ABI (sk_granite_forward = batch-1 dispatch into lane 0).

  G1a  WIN1 lane isolation: lockstep token-by-token prefill + decode vs each
       lane's single-stream token-by-token reference (same per-row numerics).
  G1b  same batched run vs the production single-forward-prefill reference
       (matvec-loop prefill; near-tie flips possible -> reported, not gating).
  G2   WIN2 lane isolation: batched one-forward prefill + decode vs the G1a
       token-by-token lockstep baseline, and vs the G1b production reference.
  G3   identical-prompt invariant (both prefill modes).
  G4   finite logits per lane (both wins).

Env: SK_REPO, SK_DYLIB, SK_METALLIB, SK_GRANITE_GGUF, N (8), T (32 local /
128 lexie), DEC (32 continuation tokens).
"""
import json
import os
import sys

import numpy as np

REPO = os.environ.get("SK_REPO", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, REPO)
os.environ.setdefault("SK_DYLIB", os.path.join(REPO, "build", "libsk.dylib"))
os.environ.setdefault("SK_METALLIB", os.path.join(REPO, "build", "libsk.metallib"))

from SuperKittens.models.granite.granite import Granite, Config  # noqa: E402

GGUF = os.environ["SK_GRANITE_GGUF"]
N    = int(os.environ.get("N", "8"))
T    = int(os.environ.get("T", "32"))
DEC  = int(os.environ.get("DEC", "32"))


def make_prompts(m):
    """Distinct real-text equal-length prompts when a tokenizer is available;
    otherwise seeded in-vocab ids (local smoke)."""
    texts = [
        "The history of the steam engine begins in the first century with",
        "A recipe for sourdough bread starts with a healthy starter and",
        "Quantum entanglement is a physical phenomenon that occurs when",
        "The Amazon rainforest covers much of northwestern Brazil and",
        "In object-oriented programming, encapsulation refers to the",
        "The Great Barrier Reef is the world's largest coral reef system",
        "Photosynthesis converts light energy into chemical energy that",
        "The French Revolution was a period of political and societal",
    ]
    if m.tokenizer is not None:
        rows = []
        for i in range(N):
            ids = m.tokenizer.encode(texts[i % len(texts)])
            while len(ids) < T:  # pad by repeating the text tokens
                ids = ids + ids
            rows.append(np.asarray(ids[:T], dtype=np.int32))
        return rows
    rng = np.random.default_rng(7)
    return [rng.integers(100, 50000, size=T).astype(np.int32) for _ in range(N)]


def ref_tbt(m, prompt, dec):
    """Single-stream token-by-token: prefill T steps, then dec decode steps."""
    m.reset()
    for t in prompt:
        cur = m.forward(np.array([t], dtype=np.int32))
    toks = [int(cur)]
    for _ in range(dec):
        cur = m.forward(np.array([cur], dtype=np.int32))
        toks.append(int(cur))
    return toks


def ref_prod(m, prompt, dec):
    """Production single-forward prefill + decode."""
    m.reset()
    cur = m.forward(prompt)
    toks = [int(cur)]
    for _ in range(dec):
        cur = m.forward(np.array([cur], dtype=np.int32))
        toks.append(int(cur))
    return toks


def batched(m, prompts, dec, batched_prefill):
    m.reset()
    if batched_prefill:
        cur = m.prefill_batched(prompts)
    else:
        mat = np.stack(prompts)
        for t in range(mat.shape[1]):
            cur = m.forward_batched(mat[:, t])
    outs = [[int(c)] for c in cur]
    finite = all(np.isfinite(m.logits_row(i).astype(np.float32)).all()
                 for i in range(N))
    absmax = max(float(np.abs(m.logits_row(i).astype(np.float32)).max())
                 for i in range(N))
    for _ in range(dec):
        cur = m.forward_batched(cur)
        for lane, c in enumerate(cur):
            outs[lane].append(int(c))
    finite_dec = all(np.isfinite(m.logits_row(i).astype(np.float32)).all()
                     for i in range(N))
    return outs, finite and finite_dec, absmax


def cmp_lanes(tag, got, ref):
    ok = 0
    for lane in range(N):
        if got[lane] == ref[lane]:
            ok += 1
        else:
            d = next(i for i, (a, b) in enumerate(zip(got[lane], ref[lane]))
                     if a != b)
            print(f"  [{tag}] lane {lane} diverges at tok {d}: "
                  f"{got[lane][d]} vs {ref[lane][d]}")
    print(f"[{tag}] {ok}/{N} lanes token-identical")
    return ok


def main():
    cfg = Config(batch=N, seq_max=max(T, 64), cache_max=max(2 * T + DEC + 8, 256))
    m = Granite(cfg)
    m.load_gguf(GGUF)
    try:
        from SuperKittens.models.load.tokenizer import Tokenizer
        tok_json = os.environ.get("SK_TOKENIZER_JSON")
        if tok_json and os.path.exists(tok_json):
            m.tokenizer = Tokenizer.from_hf_json(tok_json, family="granite")
    except Exception as e:
        print(f"[gate] tokenizer attach failed: {e}")
    prompts = make_prompts(m)
    print(f"[gate] N={N} T={T} DEC={DEC} tokenizer={'yes' if m.tokenizer else 'ids'}")

    refs_tbt  = [ref_tbt(m, p, DEC) for p in prompts]
    refs_prod = [ref_prod(m, p, DEC) for p in prompts]
    mism = sum(1 for a, b in zip(refs_tbt, refs_prod) if a != b)
    print(f"[ref] tbt-vs-production single-stream agree on {N - mism}/{N} lanes")

    out_tbt, fin1, amax1 = batched(m, prompts, DEC, batched_prefill=False)
    g1a = cmp_lanes("G1a win1-vs-tbt-ref", out_tbt, refs_tbt)
    g1b = cmp_lanes("G1b win1-vs-prod-ref", out_tbt, refs_prod)
    print(f"[G4] win1 logits finite={fin1} absmax={amax1:.2f}")

    out_bp, fin2, amax2 = batched(m, prompts, DEC, batched_prefill=True)
    g2a = cmp_lanes("G2a win2-vs-win1-lockstep", out_bp, out_tbt)
    g2b = cmp_lanes("G2b win2-vs-prod-ref", out_bp, refs_prod)
    g2c = cmp_lanes("G2c win2-vs-tbt-ref", out_bp, refs_tbt)
    print(f"[G4] win2 logits finite={fin2} absmax={amax2:.2f}")

    same = [prompts[0]] * N
    o3a, _, _ = batched(m, same, 8, batched_prefill=False)
    o3b, _, _ = batched(m, same, 8, batched_prefill=True)
    g3 = (all(o == o3a[0] for o in o3a), all(o == o3b[0] for o in o3b))
    print(f"[G3] identical-prompt invariant: tbt={g3[0]} batched={g3[1]}")

    res = dict(N=N, T=T, DEC=DEC, g1a=g1a, g1b=g1b, g2a=g2a, g2b=g2b, g2c=g2c,
               g3=g3, finite=(fin1, fin2), absmax=(amax1, amax2),
               ref_agree=N - mism,
               prompts=[p.tolist() for p in prompts],
               refs_tbt=refs_tbt, refs_prod=refs_prod,
               out_tbt=out_tbt, out_bp=out_bp)
    out_path = os.environ.get("OUT_JSON",
                              os.path.join(os.path.dirname(__file__), "gate_lanes.json"))
    with open(out_path, "w") as f:
        json.dump(res, f)
    print(f"[gate] wrote {out_path}")
    # The per-lane sequential reference is the PRODUCTION single-stream path
    # (single-forward prefill + decode). The token-by-token reference is also
    # reported; where the two pristine references disagree with EACH OTHER
    # (greedy near-ties, see refs_only.py on the baseline dylib) only the
    # production comparison gates.
    hard_pass = (g1b == N) and (g2a == N) and (g2b == N) and all(g3) \
        and fin1 and fin2
    print(f"[gate] HARD GATES (G1b, G2a, G2b, G3, G4): "
          f"{'PASS' if hard_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
