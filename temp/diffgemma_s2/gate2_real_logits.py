"""gate2_real_logits.py — Stage-2 Gate 2: SK sampler parity on REAL reference
logits. Spawns the instrumented llama-diffusion-cli (DG_EB_DUMP +
DG_EB_DUMP_THROTTLE, see instrument_diffusion.py) and streams its per-step
dumps: for every step, replay the SK EntropyBoundSampler on the exact logits
the reference sampler consumed and diff every decision field; delete each
268 MB logits file once consumed (disk stays <= 2 steps deep).

  python3 gate2_real_logits.py --cli BIN --gguf G --prompt "..." \
      --steps 10 --seed 1234 --dump-dir d --log out.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SuperKittens.models.gemma.diffusion.sampler import (  # noqa: E402
    EBParams, EntropyBoundSampler)
from tools.compare_eb import read_dec  # noqa: E402

F32 = np.float32


def wait_for(path: Path, proc, timeout: float = 600.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists() and path.stat().st_size > 0:
            time.sleep(0.5)   # writer is not atomic; settle
            return True
        if proc.poll() is not None:
            return path.exists()
        time.sleep(1.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--n-predict", type=int, default=256)
    args = ap.parse_args()

    dump = Path(args.dump_dir)
    dump.mkdir(parents=True, exist_ok=True)
    for f in dump.glob("step_*"):
        f.unlink()
    (dump / "header.bin").unlink(missing_ok=True)

    env = dict(os.environ, DG_EB_DUMP=str(dump), DG_EB_DUMP_THROTTLE="1")
    cli_log = open(dump / "cli.log", "w")
    proc = subprocess.Popen(
        [args.cli, "-m", args.gguf, "-p", args.prompt,
         "--diffusion-eb-max-steps", str(args.steps),
         "--seed", str(args.seed), "-n", str(args.n_predict), "-st"],
        stdout=cli_log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env=env)

    if not wait_for(dump / "header.bin", proc, timeout=900):
        print("FATAL: no header.bin (cli died?)", flush=True)
        proc.kill()
        return 2
    hdr = np.fromfile(dump / "header.bin", dtype=np.int32, count=4)
    n_input, C, S, seed = (int(v) for v in hdr)
    fl = np.fromfile(dump / "header.bin", dtype=np.float32, offset=16, count=5)
    p = EBParams(max_steps=S, t_min=float(fl[0]), t_max=float(fl[1]),
                 entropy_bound=float(fl[2]), stability_threshold=int(fl[3]),
                 confidence_threshold=float(fl[4]), seed=seed)
    print(f"header: n_input={n_input} C={C} S={S} seed={seed} "
          f"t=[{fl[0]:.3f},{fl[1]:.3f}] bound={fl[2]} stab={int(fl[3])} "
          f"conf={fl[4]}", flush=True)

    n_vocab = 262144
    smp = EntropyBoundSampler(p, n_vocab, C)
    fields = ["canvas_in", "u", "renoise", "entropy", "argmax", "denoiser",
              "accepted", "canvas_next", "held", "finished"]
    total = mism = 0
    per_field: dict[str, int] = {f: 0 for f in fields}

    for s in range(S):
        fdec = dump / f"step_{s:03d}.dec"
        flog = dump / f"step_{s:03d}.f32"
        if not wait_for(fdec, proc):
            print(f"step {s}: no dec record (cli exit={proc.poll()}) — stop",
                  flush=True)
            break
        ref = read_dec(fdec, C)
        canvas_before = smp.canvas.copy()
        logits = np.fromfile(flog, dtype=np.float32).reshape(C, n_vocab)
        r = smp.step(logits)
        flog.unlink()   # unblock the throttled producer
        mine = {"canvas_in": canvas_before, "u": r.u, "renoise": r.renoise,
                "entropy": r.entropy, "argmax": r.argmax, "denoiser": r.sampled,
                "accepted": r.accepted, "canvas_next": r.canvas_next,
                "held": r.held, "finished": r.finished}
        line = [f"step {s:3d} t={r.t:.4f}"]
        for f in fields:
            a, b = mine[f], ref[f]
            if f == "entropy":
                d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
                ok = d < 5e-4
                line.append(f"H~{d:.2e}")
            elif isinstance(a, (int, bool, np.bool_)):
                ok = bool(a == b)
                if not ok:
                    line.append(f"{f}:{a}!={b}")
            else:
                ok = bool(np.array_equal(a, b))
                if not ok:
                    bad = np.nonzero(np.asarray(a) != np.asarray(b))[0]
                    line.append(f"{f}:{len(bad)}bad@{bad[:4].tolist()}")
            total += 1
            if not ok:
                mism += 1
                per_field[f] += 1
        line.append(f"acc={int(r.accepted.sum())} held={r.held} "
                    f"Hbar={r.entropy_mean:.4f} fin={r.finished}")
        print(" ".join(line), flush=True)
        if ref["finished"]:
            break

    proc.wait(timeout=600)
    print(f"\nfields compared: {total}, mismatched: {mism}", flush=True)
    for f, n in per_field.items():
        if n:
            print(f"  {f}: {n} steps mismatched", flush=True)
    print("TOKEN-IDENTICAL" if mism == 0 else "MISMATCH", flush=True)
    return 0 if mism == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
