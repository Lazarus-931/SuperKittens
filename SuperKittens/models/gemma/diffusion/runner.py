# pyright: reportMissingImports=false
"""runner.py — Stage-1 parity CLI for DiffusionGemma.

Feeds golden token ids [prompt | canvas] through one unified zero-SC forward
and writes the canvas logits as raw f32 — byte-compatible with the llama.cpp
PR #24423 `llama-diffusion-gemma-eval` harness, so the two outputs diff
directly.

  python -m SuperKittens.models.gemma.diffusion.runner \
      --gguf model.gguf --prompt-ids p.i32 --canvas-ids c.i32 \
      --out logits.bin [--mode gpu|cpu] [--dump-dir d --dump-names l_out,...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import config_from_gguf
from .gguf_io import GGUFFile


def make_dump(dump_dir: str | None, names: set[str]):
    if not dump_dir:
        return None
    d = Path(dump_dir)
    d.mkdir(parents=True, exist_ok=True)

    def dump(name: str, il: int, arr: np.ndarray):
        if names and name not in names:
            return
        try:
            np.save(d / f"{name}.{il}.npy", np.asarray(arr))
        except OSError as e:  # diagnostics must not kill a long forward
            print(f"[dump] skipped {name}.{il}: {e}", file=sys.stderr)
    return dump


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt-ids", required=True)
    ap.add_argument("--canvas-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--layers", type=int, default=0, help="truncate to first N layers (debug)")
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--dump-names", default="l_out")
    args = ap.parse_args(argv)

    prompt = np.fromfile(args.prompt_ids, dtype=np.int32)
    canvas = np.fromfile(args.canvas_ids, dtype=np.int32)
    ids = np.concatenate([prompt, canvas])
    P = len(prompt)

    gg = GGUFFile(args.gguf)
    cfg = config_from_gguf(gg.meta)
    if len(canvas) != cfg.canvas_length:
        print(f"canvas len {len(canvas)} != model canvas_length {cfg.canvas_length}",
              file=sys.stderr)
        return 1
    if args.layers:
        cfg.n_layers = args.layers
    dump = make_dump(args.dump_dir, set(filter(None, args.dump_names.split(","))))

    t0 = time.time()
    if args.mode == "cpu":
        from .graph_ref import forward_cpu
        logits = forward_cpu(gg, cfg, ids, P, dump=dump)
    else:
        from .forward_metal import DiffusionGemmaMetal
        m = DiffusionGemmaMetal(args.gguf, cfg)
        m.dump = dump
        logits = m.forward(ids, P)
    dt = time.time() - t0

    logits.astype(np.float32).tofile(args.out)
    print(f"wrote {logits.shape[0]} x {logits.shape[1]} f32 logits to {args.out} "
          f"({args.mode}, P={P}, {dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
