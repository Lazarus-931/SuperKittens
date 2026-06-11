"""make_embt.py — one-time build of the SC soft-embedding weight: token_embd
dequantized + transposed to [d_model, vocab] f16 on disk (the per-step SC
matmul then streams it like any other weight). Mirrors the reference's
dg_ensure_sc_embT (f16 transpose of the dequantized embed).

  python3 make_embt.py --gguf model.gguf --out dg_embT_f16.bin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SuperKittens.models.gemma.diffusion.config import config_from_gguf  # noqa: E402
from SuperKittens.models.gemma.diffusion.gguf_io import GGUFFile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vchunk", type=int, default=8192)
    args = ap.parse_args()

    gg = GGUFFile(args.gguf)
    cfg = config_from_gguf(gg.meta)
    d, V = cfg.d_model, cfg.vocab_size
    mm = np.memmap(args.out, dtype=np.float16, mode="w+", shape=(d, V))
    for v0 in range(0, V, args.vchunk):
        v1 = min(v0 + args.vchunk, V)
        chunk = gg.dequant("token_embd.weight", rows=slice(v0, v1))  # [rows, d] f32
        mm[:, v0:v1] = chunk.T.astype(np.float16)
    mm.flush()
    del mm
    print(f"wrote [{d}, {V}] f16 -> {args.out} ({d * V * 2 / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
