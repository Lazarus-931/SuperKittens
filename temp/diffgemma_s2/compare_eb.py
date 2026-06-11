"""compare_eb.py — drive the SK EntropyBoundSampler with the same per-step
logits a reference run consumed and diff every decision field against the
reference's .dec records (eb_ref_harness locally, or the instrumented
llama.cpp diffusion.cpp on amelia — identical record layout).

  python3 compare_eb.py --logits-dir d --dec-dir d2 --seed s --C 256 \
      --S 16 --n-vocab 262144 [--params t_min t_max bound stab conf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SuperKittens.models.gemma.diffusion.sampler import EBParams, EntropyBoundSampler  # noqa: E402


def read_dec(path: Path, C: int) -> dict:
    b = path.read_bytes()
    off = 0

    def take(dtype, n):
        nonlocal off
        a = np.frombuffer(b, dtype=dtype, count=n, offset=off)
        off += a.nbytes
        return a

    d = {}
    d["step_idx"] = int(take(np.int32, 1)[0])
    d["cur_step"] = int(take(np.int32, 1)[0])
    d["t"] = float(take(np.float32, 1)[0])
    d["canvas_in"] = take(np.int32, C)
    d["u"] = take(np.float32, C)
    d["renoise"] = take(np.int32, C)
    d["entropy"] = take(np.float32, C)
    d["argmax"] = take(np.int32, C)
    d["denoiser"] = take(np.int32, C)
    d["accepted"] = take(np.uint8, C).astype(bool)
    d["canvas_next"] = take(np.int32, C)
    d["held"] = int(take(np.int32, 1)[0])
    d["finished"] = bool(take(np.uint8, 1)[0])
    d["entropy_sum"] = float(take(np.float32, 1)[0])
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits-dir", required=True)
    ap.add_argument("--dec-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--n-vocab", type=int, default=262144)
    ap.add_argument("--params", nargs=5, type=float, default=None,
                    metavar=("TMIN", "TMAX", "BOUND", "STAB", "CONF"))
    args = ap.parse_args()

    p = EBParams(max_steps=args.S, seed=args.seed)
    if args.params:
        p.t_min, p.t_max, p.entropy_bound = args.params[0], args.params[1], args.params[2]
        p.stability_threshold = int(args.params[3])
        p.confidence_threshold = args.params[4]

    smp = EntropyBoundSampler(p, args.n_vocab, args.C)
    dec_dir = Path(args.dec_dir)
    ldir = Path(args.logits_dir)

    n_steps = len(sorted(dec_dir.glob("step_*.dec")))
    total = mism = 0
    fields = ["canvas_in", "u", "renoise", "entropy", "argmax", "denoiser",
              "accepted", "canvas_next", "held", "finished"]
    per_field = {f: 0 for f in fields}

    for s in range(n_steps):
        ref = read_dec(dec_dir / f"step_{s:03d}.dec", args.C)
        canvas_before = smp.canvas.copy()
        logits = np.fromfile(ldir / f"step_{s:03d}.f32", dtype=np.float32).reshape(
            args.C, args.n_vocab)
        r = smp.step(logits)
        mine = {"canvas_in": canvas_before, "u": r.u, "renoise": r.renoise,
                "entropy": r.entropy, "argmax": r.argmax, "denoiser": r.sampled,
                "accepted": r.accepted, "canvas_next": r.canvas_next,
                "held": r.held, "finished": r.finished}
        line = [f"step {s:3d} t={r.t:.4f}"]
        for f in fields:
            a, b = mine[f], ref[f]
            if f == "entropy":
                d = float(np.abs(a - b).max())
                ok = d < 5e-4
                line.append(f"H~{d:.2e}")
            elif isinstance(a, (int, bool)):
                ok = a == b
            else:
                ok = np.array_equal(a, b)
                if not ok:
                    n_bad = int((np.asarray(a) != np.asarray(b)).sum())
                    line.append(f"{f}:{n_bad}bad")
            total += 1
            if not ok:
                mism += 1
                per_field[f] += 1
        naccept = int(r.accepted.sum())
        line.append(f"acc={naccept} held={r.held} Hbar={r.entropy_mean:.4f}"
                    f" fin={r.finished}")
        print(" ".join(line))
        if r.finished:
            break

    print(f"\nfields compared: {total}, mismatched: {mism}")
    for f, n in per_field.items():
        if n:
            print(f"  {f}: {n} steps mismatched")
    print("TOKEN-IDENTICAL" if mism == 0 else "MISMATCH")
    return 0 if mism == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
