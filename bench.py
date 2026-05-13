"""Side-by-side decode bench: SK vs llama.cpp on the same model + prompt.

Usage:
    python bench.py --spec gemma4-e2b --gguf <path-to-gguf> [--n 64]

Prints SK tok/s and llama.cpp tok/s on the same hardware for the same model.
Both run pure decode (after a prefill warm-up).
"""
from __future__ import annotations
import argparse, subprocess, time, sys, os
import numpy as np


def bench_sk(spec: str, prompt: str, n_tokens: int) -> tuple[float, str]:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import SuperKittens as sk
    family = spec.split("-")[0]
    if family == "gemma4":
        import SuperKittens.models.gemma.gemma4
    elif family == "qwen3":
        import SuperKittens.models.qwen.qwen
    elif family == "mamba2":
        import SuperKittens.models.mamba2.mamba2
    m = sk.load(spec)
    if not getattr(m, "tokenizer", None):
        raise RuntimeError("no tokenizer attached; SK side cannot bench without tokens")
    ids = np.asarray(m.tokenizer.encode(prompt), dtype=np.int32)
    _ = m.forward(ids)
    t0 = time.perf_counter()
    out_ids = m.generate(ids, max_new_tokens=n_tokens, temperature=0.0)
    dt = time.perf_counter() - t0
    text = m.tokenizer.decode(out_ids)
    return n_tokens / dt, text


def bench_llamacpp(gguf: str, prompt: str, n_tokens: int) -> tuple[float, str]:
    out = subprocess.run(
        ["llama-bench", "-m", gguf, "-p", "0", "-n", str(n_tokens), "-ngl", "999"],
        capture_output=True, text=True, check=True,
    )
    tps = None
    for line in out.stdout.splitlines():
        if f"tg{n_tokens}" in line:
            tps = float(line.split("|")[-2].split("±")[0].strip())
            break
    return tps or 0.0, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="SK model spec, e.g. gemma4-e2b")
    ap.add_argument("--gguf", required=True, help="path to matching .gguf file for llama-bench")
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sk-only", action="store_true")
    ap.add_argument("--lcpp-only", action="store_true")
    args = ap.parse_args()

    sk_tps = lcpp_tps = None
    if not args.lcpp_only:
        sk_tps, sk_text = bench_sk(args.spec, args.prompt, args.n)
        print(f"SK         : {sk_tps:7.2f} tok/s   text={sk_text[:60]!r}")
    if not args.sk_only:
        lcpp_tps, _ = bench_llamacpp(args.gguf, args.prompt, args.n)
        print(f"llama.cpp  : {lcpp_tps:7.2f} tok/s")
    if sk_tps and lcpp_tps:
        ratio = sk_tps / lcpp_tps
        verdict = "SK FASTER" if ratio > 1 else "llama.cpp faster"
        print(f"ratio      : {ratio:7.2f}x   ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
