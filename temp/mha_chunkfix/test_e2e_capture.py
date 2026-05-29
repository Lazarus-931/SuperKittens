# pyright: reportMissingImports=false
"""End-to-end confirm through the real qwen3-0.6B stack.

A chunked forward (prefill P tokens, then a seq=K continuation at current_pos=P)
must produce the SAME post-attention-layer residual at interior positions as a
single full prefill of all P+K tokens at current_pos=0. Uses the C capture hook
(sk_qwen_set_capture_layer + sk_qwen_get_capture), which snapshots the post-layer
residual for every position — so it exercises the real attention through the
whole stack, not synthetic Q/K/V.

BEFORE the fix the chunk's interior rows diverge; AFTER they match prefill.
"""
from __future__ import annotations
import os, sys, ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SK_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SK_ROOT)
import SuperKittens as sk  # noqa: E402
from SuperKittens.models.qwen.qwen import _load  # noqa: E402

SNAP = os.path.join(HERE, "Qwen3-0.6B")


def capture_residual(m, lib, ids, capture_layer, chunk_split=None):
    """Run ids through the model capturing post-`capture_layer` residual.
    If chunk_split is None: single prefill (current_pos=0).
    Else: prefill ids[:chunk_split] then forward ids[chunk_split:] at current_pos.
    Returns (n_captured_rows, d_model) fp16 of the LAST forward's capture."""
    d_model = m.cfg.d_model
    lib.sk_qwen_set_capture_layer(m._h, ctypes.c_int32(capture_layer))
    m.reset()
    ids = np.asarray(ids, np.int32)
    if chunk_split is None:
        m.forward(ids)
        rows = len(ids)
    else:
        m.forward(ids[:chunk_split])           # prefill, current_pos 0 -> split
        m.forward(ids[chunk_split:])           # CHUNK at current_pos=split
        rows = len(ids) - chunk_split
    out = np.zeros((rows, d_model), np.float16)
    rc = lib.sk_qwen_get_capture(m._h, out.ctypes.data_as(ctypes.c_void_p))
    if rc:
        raise RuntimeError(f"get_capture rc={rc}")
    return out.astype(np.float32)


def main():
    m = sk.load("qwen3-0.6b", snapshot=SNAP, cache_max=512)
    lib = _load()
    if not (hasattr(lib, "sk_qwen_set_capture_layer") and hasattr(lib, "sk_qwen_get_capture")):
        print("capture ABI unavailable; skipping e2e capture test"); return 0
    lib.sk_qwen_set_capture_layer.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.sk_qwen_get_capture.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    cap_layer = m.cfg.n_layers - 1  # last layer: full stack of attention applied
    # A plausible token id prompt (prompt echo from the smoke test) extended.
    full = np.array([785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 1879, 374],
                    np.int32)
    worst = 0.0
    for split in [8, 15, 17, 4]:
        ids = full.copy()
        # pad/truncate so split < len
        if split >= len(ids):
            ids = np.concatenate([ids, ids])[: split + 4]
        ref = capture_residual(m, lib, ids, cap_layer, chunk_split=None)   # full prefill
        chk = capture_residual(m, lib, ids, cap_layer, chunk_split=split)  # chunk @split
        # chunk rows correspond to prefill rows [split:]
        ref_slice = ref[split:]
        n = min(len(ref_slice), len(chk))
        for i in range(n):
            num = np.linalg.norm(chk[i] - ref_slice[i])
            den = np.linalg.norm(ref_slice[i]) + 1e-6
            rel = num / den
            worst = max(worst, rel)
        max_rel = max((np.linalg.norm(chk[i]-ref_slice[i])/(np.linalg.norm(ref_slice[i])+1e-6))
                      for i in range(n))
        status = "OK " if max_rel < 1e-2 else "BAD"
        print(f"[{status}] chunk@{split} (K={n}) vs full-prefill residual  max_rel={max_rel:.4e}")
    print(f"\nWORST residual max_rel: {worst:.4e}")
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    os.environ.setdefault("SK_DYLIB", os.path.join(SK_ROOT, "build", "libsk.dylib"))
    os.environ.setdefault("SK_METALLIB", os.path.join(SK_ROOT, "build", "libsk.metallib"))
    sys.exit(main())
