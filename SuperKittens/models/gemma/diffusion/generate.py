# pyright: reportMissingImports=false
"""generate.py — DiffusionGemma end-to-end block generation (Stage 2).

One denoising block: EntropyBound sampler (sampler.py, reference
decision-mirror) driving the unified [prompt|canvas] forward with
self-conditioning. Per-step telemetry to stdout + a JSONL log; final output =
the argmax canvas, trimmed at the first end-of-generation token / repetition
loop like the reference CLI.

  python -m SuperKittens.models.gemma.diffusion.generate \
      --gguf model.gguf --prompt-ids p.i32 --out-dir gen/ \
      [--steps 16 --seed 1234 --mode gpu --sc-embt dg_embT_f16.bin --no-sc]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import config_from_gguf
from .gguf_io import GGUFFile
from .sampler import EBParams, EntropyBoundSampler

F32 = np.float32


def detokenize(meta: dict, ids) -> str:
    """Minimal SPM detok (ggml llama-style vocab): ▁ -> space, <0xXX> bytes."""
    tokens = meta["tokenizer.ggml.tokens"]
    out = bytearray()
    for tid in ids:
        piece = tokens[int(tid)]
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            out += bytes([int(piece[3:5], 16)])
        else:
            out += piece.replace("▁", " ").encode("utf-8")
    return out.decode("utf-8", "replace")


def eog_ids(meta: dict) -> set[int]:
    ids = set()
    for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if k in meta:
            ids.add(int(meta[k]))
    return ids


def trim_canvas(canvas: np.ndarray, eog: set[int]) -> int:
    """Reference CLI trim: cut at the first EOG token, else at the onset of a
    stride-1/2 repetition loop (>= 6 reps)."""
    n = len(canvas)
    cut = n
    for i in range(n):
        if int(canvas[i]) in eog:
            cut = i
            break
    for i in range(cut - 1):
        for stride in (1, 2):
            reps = 0
            j = i
            while j + stride < n and canvas[j] == canvas[j + stride]:
                reps += 1
                j += stride
            if reps >= 6:
                return i
    return cut


def run_block(model, cfg, prompt_ids: np.ndarray, params: EBParams,
              use_sc: bool, log, mode: str = "gpu"):
    """One denoising block; returns (argmax_canvas, steps_run, timings)."""
    C = cfg.canvas_length
    P = len(prompt_ids)
    smp = EntropyBoundSampler(params, cfg.vocab_size, C)
    prev_logits = None
    fw_s = smp_s = 0.0
    step = None
    while not smp.finished:
        ids = np.concatenate([prompt_ids, smp.canvas.astype(np.int32)])
        t0 = time.time()
        if use_sc:
            sc = prev_logits if prev_logits is not None else np.zeros((C, cfg.vocab_size), F32)
            sc_use = 0.0 if smp.step_idx == 0 else 1.0
            logits = model.forward(ids, P, sc_logits=sc,
                                   sc_temp_inv=float(smp.prev_temp_inv), sc_use=sc_use)
        else:
            logits = model.forward(ids, P)
        t1 = time.time()
        if not np.isfinite(logits).all():
            raise RuntimeError(f"non-finite logits at step {smp.step_idx}")
        step = smp.step(np.ascontiguousarray(logits, dtype=F32))
        t2 = time.time()
        fw_s += t1 - t0
        smp_s += t2 - t1
        prev_logits = logits
        rec = {"step": step.step_idx, "t": round(step.t, 4),
               "accepted": int(step.accepted.sum()), "held": step.held,
               "H_mean": round(step.entropy_mean, 5), "finished": step.finished,
               "fw_s": round(t1 - t0, 1), "smp_s": round(t2 - t1, 1)}
        log(rec)
    return smp.argmax_canvas, (step.step_idx + 1 if step else 0), (fw_s, smp_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt-ids", required=True, help="chat-templated prompt, raw i32")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sc-embt", default=None, help="dg_embT_f16.bin (make_embt.py)")
    ap.add_argument("--no-sc", action="store_true", help="zero-SC forward (Stage-1 config)")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt = np.fromfile(args.prompt_ids, dtype=np.int32)
    gg = GGUFFile(args.gguf)
    cfg = config_from_gguf(gg.meta)
    params = EBParams(max_steps=args.steps, seed=args.seed)

    if args.mode == "cpu":
        from .graph_ref import forward_cpu

        class _CpuModel:
            def forward(self, ids, P, **sc):
                return forward_cpu(gg, cfg, ids, P, **sc)
        model = _CpuModel()
    else:
        from .forward_metal import DiffusionGemmaMetal
        model = DiffusionGemmaMetal(args.gguf, cfg, sc_embt_path=args.sc_embt)

    jl = open(out / "steps.jsonl", "w")

    def log(rec):
        print(json.dumps(rec), flush=True)
        jl.write(json.dumps(rec) + "\n")
        jl.flush()

    t0 = time.time()
    canvas, steps, (fw_s, smp_s) = run_block(model, cfg, prompt, params,
                                             use_sc=not args.no_sc, log=log,
                                             mode=args.mode)
    wall = time.time() - t0

    canvas.astype(np.int32).tofile(out / "canvas.i32")
    cut = trim_canvas(canvas, eog_ids(gg.meta))
    text = detokenize(gg.meta, canvas[:cut])
    (out / "output.txt").write_text(text)
    n_mask = int((canvas == cfg.mask_token_id).sum())
    summary = {"steps": steps, "wall_s": round(wall, 1), "fw_s": round(fw_s, 1),
               "smp_s": round(smp_s, 1), "trim": cut, "mask_tokens_in_canvas": n_mask,
               "seed": args.seed, "S": args.steps, "sc": not args.no_sc}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    print("---- output ----")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
