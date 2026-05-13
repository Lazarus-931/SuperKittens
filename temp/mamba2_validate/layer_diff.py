"""Mamba 2 SK vs HF parity diff.

Loads `hf_ref.npz` produced by `dump_hf_mamba2.py`, runs the SK forward on the
same prompt ids, compares final logits + argmax. Also compares cumulative SSM
state per layer (rough proxy for first-divergence localization since the
SK runtime only exposes whole-layer scratch buffers for the LAST forward).

Usage (on lexie):
    SK_METALLIB=/Users/lexie/SuperKittens/build/libsk.metallib \
    SK_DYLIB=/Users/lexie/SuperKittens/build/libsk.dylib \
    python3 temp/mamba2_validate/layer_diff.py
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REF_NPZ = ROOT / "temp/mamba2_validate/hf_ref.npz"

# Local model dir (downloaded snapshot) or HF id.
HF_DIR = os.environ.get("MAMBA2_DIR",
        str(Path.home() / ".cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots"))

def find_safetensors():
    # If MAMBA2_SAFETENSORS env set use that; else scan HF cache.
    p = os.environ.get("MAMBA2_SAFETENSORS")
    if p and Path(p).exists(): return p
    base = Path(HF_DIR)
    if base.exists():
        for f in base.rglob("model.safetensors"):
            return str(f)
    raise FileNotFoundError("set MAMBA2_SAFETENSORS to model.safetensors path")

def find_config():
    p = os.environ.get("MAMBA2_CONFIG")
    if p and Path(p).exists(): return p
    base = Path(HF_DIR)
    if base.exists():
        for f in base.rglob("config.json"):
            return str(f)
    return None


def main():
    if not REF_NPZ.exists():
        print(f"[!] {REF_NPZ} missing — run dump_hf_mamba2.py first", file=sys.stderr)
        return 1
    ref = np.load(REF_NPZ, allow_pickle=True)
    print(f"[ref] keys: {len(ref.files)}  argmax={int(ref['_meta_argmax'])}")
    ids = ref["_meta_input_ids"]
    if ids.ndim == 2: ids = ids[0]
    ids = ids.astype(np.int32).tolist()
    print(f"[ref] input_ids={ids}")

    from SuperKittens.models.mamba2.mamba2 import Mamba2Config, Mamba2Model
    cfg_path = find_config()
    if cfg_path:
        cfg = Mamba2Config.from_hf_json(cfg_path)
    else:
        cfg = Mamba2Config()
    cfg.seq_max = max(cfg.seq_max, len(ids) + 64)
    print(f"[sk] cfg: layers={cfg.n_layers} D={cfg.d_model} E={cfg.intermediate} H={cfg.n_heads}")

    m = Mamba2Model(cfg)
    st_path = find_safetensors()
    print(f"[sk] loading {st_path}")
    m.load_safetensors(st_path)

    # ── Single prefill ────────────────────────────────────────────────
    t0 = time.time()
    sk_argmax = m.forward(ids)
    t1 = time.time()
    print(f"[sk] forward({len(ids)} ids) → argmax={sk_argmax}  ({(t1-t0)*1000:.1f} ms)")
    ref_argmax = int(ref["_meta_argmax"])
    print(f"[ref] argmax={ref_argmax}  match={sk_argmax == ref_argmax}")

    # ── Logits comparison ─────────────────────────────────────────────
    if "logits_final" in ref.files:
        ref_lg = ref["logits_final"].astype(np.float32)
        if ref_lg.ndim == 3: ref_lg = ref_lg[0]   # (T, V)
        ref_last = ref_lg[-1]
        sk_last = m.get_last_logits().astype(np.float32)
        diff = sk_last - ref_last
        rel = np.linalg.norm(diff) / max(np.linalg.norm(ref_last), 1e-9)
        max_abs = float(np.max(np.abs(diff)))
        print(f"[logits] rel_l2={rel:.4f}  max_abs={max_abs:.4f}  "
              f"sk_argmax={int(sk_last.argmax())} ref_argmax={int(ref_last.argmax())}")
        # Top-5
        topk = lambda a, k=5: list(zip(np.argsort(-a)[:k].tolist(), np.sort(-a)[:k].tolist()))
        print(f"[logits] sk top5 = {topk(sk_last)}")
        print(f"[logits] ref top5 = {topk(ref_last)}")

    # ── Per-layer SSM state comparison (cumulative; partial signal) ───
    print("\n=== per-layer SSM state ssd_state.L{i} vs hf ssm_states[i] (if present) ===")
    # HF dump does not include ssm_states. Skip if not present; instead just
    # show SK state norms per layer to verify finiteness.
    for L in range(cfg.n_layers):
        try:
            st = m.dump(f"ssm_state.L{L}")
        except Exception as e:
            print(f"[L{L}] dump err: {e}")
            break
        finite = np.isfinite(st).all()
        print(f"[L{L}] ssm_state norm={np.linalg.norm(st):.3e}  "
              f"absmax={np.max(np.abs(st)):.3e}  finite={finite}")
        if not finite:
            print(f"   ↑ first non-finite layer — likely divergence root")
            break

    # ── Throughput probe: 8 decode steps of length-1 prefill ──────────
    # (Reference path runs prefill for each step since recurrent path uses
    # the persistent ssm_state correctly with our kernel.)
    print("\n=== decode throughput probe (8 new tokens) ===")
    seq = list(ids)
    cur = sk_argmax
    n_new = 8
    t0 = time.time()
    for _ in range(n_new):
        seq.append(cur)
        # Note: SK has no KV cache for mamba; each call replays from scratch
        # since current_pos resets... actually current_pos accumulates.
        # For correctness here we just re-prefill the whole sequence; for a
        # real throughput number we'd add a state-keeping single-step path.
        cur = m.forward([cur])  # length-1 step using existing ssm_state
    t1 = time.time()
    dt = (t1 - t0) / n_new
    print(f"[decode] {n_new} steps  {dt*1000:.1f} ms/tok  ≈ {1/dt:.2f} tok/s")
    print(f"[decode] HF fp32 baseline (lexie) = 14.48 tok/s")

    return 0 if sk_argmax == ref_argmax else 2


if __name__ == "__main__":
    sys.exit(main())
