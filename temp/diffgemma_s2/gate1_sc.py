"""gate1_sc.py — Stage-2 Gate 1: SC subgraph verification on the CPU oracle
against the instrumented reference (llama-diffusion-gemma-server with
DG_DUMP_TENSORS=sc_sig,inp_region).

Runs ON amelia. Sequence (one process at a time):
  reqA: [prompt|canvas0] use_sc=0 temp=t0 -> L0 + sc_sig_r000 (must be zeros)
  SK sampler(seed) consumes L0           -> canvas1, t1
  reqB: [prompt|canvas1] use_sc=1 temp=t1 -> L1_sc + sc_sig_r001 + inp_region_r001
  reqC: [prompt|canvas1] use_sc=0 temp=t1 -> L1_nosc (envelope control)
then compares the SK CPU oracle's sc_signal / embed_tokens (and optionally a
detached forward_cpu) against the dumps.

  python3 gate1_sc.py --server BIN --gguf G --prompt p1_prompt.i32 \
      --out-dir ~/sk-diffg-s2b/gate1 [--S 16 --seed 1234] [--skip-server]
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
from SuperKittens.models.gemma.diffusion.config import config_from_gguf  # noqa: E402
from SuperKittens.models.gemma.diffusion.gguf_io import GGUFFile  # noqa: E402
from SuperKittens.models.gemma.diffusion.graph_ref import (  # noqa: E402
    Weights, embed_tokens, sc_signal)
from SuperKittens.models.gemma.diffusion.sampler import (  # noqa: E402
    EBParams, EntropyBoundSampler)

F32 = np.float32


def stats(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """a = candidate, b = reference; per-element f32 arrays, same shape."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.sqrt((b ** 2).mean()) or 1.0
    rel_rms = float(np.sqrt(((a - b) ** 2).mean()) / denom)
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {"name": name, "rel_rms": round(rel_rms, 6), "cos": round(cos, 8),
            "max_abs_diff": round(float(np.abs(a - b).max()), 6)}


def argmax_agree(la: np.ndarray, lb: np.ndarray) -> float:
    return float((la.argmax(axis=1) == lb.argmax(axis=1)).mean())


class Server:
    def __init__(self, binary: str, gguf: str, dump_dir: str):
        env = dict(os.environ, DG_DUMP_TENSORS="sc_sig,inp_region",
                   DG_DUMP_DIR=dump_dir)
        self.proc = subprocess.Popen([binary, gguf], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env, text=True)
        line = self.proc.stdout.readline().strip()
        assert line.startswith("READY"), line
        self.n_vocab = int(line.split()[1])
        print(f"[server] {line}", flush=True)

    def forward(self, path: Path, P: int, C: int, ids: np.ndarray,
                use_sc: int, temp: float) -> np.ndarray:
        hdr = np.empty(4, np.int32)
        hdr[0], hdr[1], hdr[2] = P, C, use_sc
        hdr[3:4].view(np.float32)[0] = temp
        with open(path, "wb") as f:
            hdr.tofile(f)
            ids.astype(np.int32).tofile(f)
        t0 = time.time()
        self.proc.stdin.write(str(path) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        assert line == f"OK {C}", line
        print(f"[server] {path.name}: {line} ({time.time() - t0:.1f}s)", flush=True)
        out = np.fromfile(str(path) + ".resp", dtype=np.float32).reshape(C, -1)
        os.unlink(str(path) + ".resp")   # 268 MB each; the caller persists what it needs
        return out

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-server", action="store_true",
                    help="reuse existing dumps/resp files in out-dir")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt = np.fromfile(args.prompt, dtype=np.int32)
    P = len(prompt)

    gg = GGUFFile(args.gguf)
    cfg = config_from_gguf(gg.meta)
    C, V = cfg.canvas_length, cfg.vocab_size

    smp = EntropyBoundSampler(EBParams(max_steps=args.S, seed=args.seed), V, C)
    canvas0 = smp.canvas.copy()
    t0_temp = float(smp.temperature(0))
    t1_temp = float(smp.temperature(1))

    if not args.skip_server:
        srv = Server(args.server, args.gguf, str(out))
        assert srv.n_vocab == V
        ids0 = np.concatenate([prompt, canvas0])
        L0 = srv.forward(out / "reqA.bin", P, C, ids0, use_sc=0, temp=t0_temp)
        L0.tofile(out / "L0.f32")
        step = smp.step(np.ascontiguousarray(L0, F32))
        canvas1 = step.canvas_next
        canvas1.astype(np.int32).tofile(out / "canvas1.i32")
        ids1 = np.concatenate([prompt, canvas1])
        L1_sc = srv.forward(out / "reqB.bin", P, C, ids1, use_sc=1, temp=t1_temp)
        L1_sc.tofile(out / "L1_sc.f32")
        # reqC resets the server's sc_cache to L1_nosc, so it must come last
        L1_nosc = srv.forward(out / "reqC.bin", P, C, ids1, use_sc=0, temp=t1_temp)
        L1_nosc.tofile(out / "L1_nosc.f32")
        srv.close()
    else:
        L0 = np.fromfile(out / "L0.f32", dtype=np.float32).reshape(C, V)
        step = smp.step(np.ascontiguousarray(L0, F32))
        canvas1 = step.canvas_next
        L1_sc = np.fromfile(out / "L1_sc.f32", dtype=np.float32).reshape(C, V)
        L1_nosc = np.fromfile(out / "L1_nosc.f32", dtype=np.float32).reshape(C, V)
        ids1 = np.concatenate([prompt, canvas1])

    report: list[dict] = []

    # 1. zero-SC request must have sc_sig == 0 exactly (the sc_use gate)
    sig0 = np.fromfile(out / "sc_sig_r000.f32", dtype=np.float32)
    report.append({"name": "ref sc_sig@use_sc=0 all-zero",
                   "ok": bool((sig0 == 0).all()),
                   "max_abs": float(np.abs(sig0).max())})

    # 2. oracle sc_signal vs reference sc_sig (the SC subgraph in isolation)
    w = Weights(gg)
    sc_ti = F32(1.0 / t0_temp)
    sig_oracle = sc_signal(w, cfg, np.ascontiguousarray(L0, F32), sc_ti, 1.0)
    sig_ref = np.fromfile(out / "sc_sig_r001.f32", dtype=np.float32).reshape(C, cfg.d_model)
    report.append(stats("sc_sig oracle-vs-ref", sig_oracle, sig_ref))
    report.append({"name": "sc_sig scale", "ref_rms": float(np.sqrt((sig_ref ** 2).mean())),
                   "oracle_rms": float(np.sqrt((sig_oracle ** 2).mean()))})

    # 3. oracle embed (region + SC + rms) vs reference inp_region canvas rows
    x_oracle = embed_tokens(w, cfg, ids1, P, sc_sig=sig_oracle)
    inp_ref = np.fromfile(out / "inp_region_r001.f32", dtype=np.float32).reshape(P + C, cfg.d_model)
    report.append(stats("inp_region[P:] oracle-vs-ref", x_oracle[P:], inp_ref[P:]))
    report.append(stats("inp_region[:P] oracle-vs-ref", x_oracle[:P], inp_ref[:P]))

    # 4. SC-active vs zero-SC end-logits: how big is the SC effect in the ref?
    report.append({"name": "ref SC effect (L1_sc vs L1_nosc)",
                   "argmax_agree": argmax_agree(L1_sc, L1_nosc),
                   "rel_rms": stats("", L1_sc, L1_nosc)["rel_rms"]})

    for r in report:
        print(json.dumps(r), flush=True)
    (out / "gate1_report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
