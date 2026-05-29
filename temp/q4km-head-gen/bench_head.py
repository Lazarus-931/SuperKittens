#!/usr/bin/env python3
"""Decode tok/s bench for a qwen3 Q4_K_M variant + coherence guardrail.

Protocol: 2 warmup gens, 30s cooldown, 5 timed reps, report median tok/s.
Greedy (temperature=0) so output is deterministic for the coherence check.
Decode-only tok/s: time the per-token loop after the prompt forward.

Usage: SK_DYLIB=... SK_METALLIB=... python3 bench_head.py <spec> [snapshot_dir]
"""
import sys, os, time, statistics, math
import numpy as np

sys.path.insert(0, os.environ["SK_REPO"])  # repo root holding the SuperKittens pkg

from SuperKittens.inference.registry import SPECS
from SuperKittens.models.qwen.qwen import Qwen

PROMPT = "Generate a poem about pizza dough"
N_NEW = 64
WARMUP = 2
REPS = 5
COOLDOWN_S = 30.0


def build(spec_name, snap):
    spec = SPECS[spec_name]
    over = {}
    if snap:
        over["snapshot"] = snap
    m = Qwen.from_spec(spec, **over)
    if m.tokenizer is None:
        raise RuntimeError("no tokenizer attached")
    return m


def timed_decode(m, ids, n_new):
    """Run a greedy decode of n_new tokens, return (token_ids, decode_tok_per_s).

    Times only the decode steps (post-prompt), matching how tok/s is reported
    for interactive decode. The prompt forward is excluded.
    """
    m.reset()
    arg = m._forward(np.asarray(ids, dtype=np.int32).reshape(-1))
    last = int(arg[0])
    out = [last]
    eos = set(getattr(m.tokenizer, "eos_ids", None) or [])
    t0 = time.perf_counter()
    steps = 0
    for _ in range(n_new - 1):
        arg = m._forward(np.array([last], dtype=np.int32))
        last = int(arg[0])
        out.append(last)
        steps += 1
        if last in eos:
            break
    dt = time.perf_counter() - t0
    return out, (steps / dt if dt > 0 else 0.0)


def main():
    spec_name = sys.argv[1]
    snap = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"[bench] spec={spec_name} snap={snap}", flush=True)

    m = build(spec_name, snap)
    ids = m.tokenizer.chat([{"role": "user", "content": PROMPT}], bos=True)
    print(f"[bench] prompt_len={len(ids)}", flush=True)

    # Warmup.
    coherent_text = None
    for w in range(WARMUP):
        toks, tps = timed_decode(m, ids, N_NEW)
        coherent_text = m.tokenizer.decode(toks, skip_special=True)
        print(f"[bench] warmup {w}: {tps:.2f} tok/s", flush=True)

    print(f"[bench] cooldown {COOLDOWN_S}s ...", flush=True)
    time.sleep(COOLDOWN_S)

    rates = []
    last_toks = None
    for r in range(REPS):
        toks, tps = timed_decode(m, ids, N_NEW)
        rates.append(tps)
        last_toks = toks
        print(f"[bench] rep {r}: {tps:.2f} tok/s", flush=True)

    med = statistics.median(rates)
    text = m.tokenizer.decode(last_toks, skip_special=True)

    # Coherence guardrail: no NaN logits (a NaN would make argmax degenerate;
    # also re-check the last logits vector directly), printable ASCII-ish text.
    logits = m._last_logits().astype(np.float32)
    has_nan = bool(np.isnan(logits).any() or np.isinf(logits).any())
    printable = sum(1 for ch in text if ch.isprintable() or ch.isspace())
    frac_print = printable / max(1, len(text))

    print("=" * 60, flush=True)
    print(f"[RESULT] spec={spec_name} median={med:.2f} tok/s "
          f"(reps={[f'{x:.2f}' for x in rates]})", flush=True)
    print(f"[RESULT] nan_or_inf_logits={has_nan} frac_printable={frac_print:.3f}", flush=True)
    print(f"[RESULT] sample_text<<<\n{text}\n>>>", flush=True)


if __name__ == "__main__":
    main()
