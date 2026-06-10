# pyright: reportMissingImports=false
"""Synthetic op tests for the DiffusionGemma Stage-1 driver.

Cross-validates: gemm_mma_{f16,q8_0,q4k,q6k} dispatch plumbing vs the numpy
dequant in gguf_io (random quant blocks — both sides must agree), the
dg_softmax_mask kernel vs numpy, and the GEMM-composed attention vs a pure
numpy reference. No model weights needed; runs on any Apple Silicon box.
"""
import sys
from pathlib import Path

import numpy as np

SK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SK_ROOT))

from SuperKittens.models.gemma.diffusion.gguf_io import dequant_rows  # noqa: E402
from SuperKittens.models.gemma.diffusion.graph_ref import build_mask, softmax  # noqa: E402
from SuperKittens.models.gemma.diffusion import forward_metal as fm  # noqa: E402

rng = np.random.default_rng(0)
ctx = fm.MetalCtx()


def gemm(kernel, a_f16, w_buf_arr, M, N, K, ldc=None):
    a_buf = ctx.buf_from(a_f16)
    w_buf = ctx.buf_from(w_buf_arr)
    c_buf = ctx.buf_empty(M * (ldc or N) * 2)
    b = fm.GemmBatch(ctx)
    b.gemm(kernel, a_buf, 0, w_buf, 0, c_buf, 0, M, N, K, ldc)
    b.run()
    return ctx.read(c_buf, np.float16, (M, ldc or N))


def check(name, got, want, tol):
    got = got.astype(np.float64)
    want = want.astype(np.float64)
    rms_rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
    max_abs = np.abs(got - want).max()
    status = "OK " if rms_rel < tol else "FAIL"
    print(f"{status} {name}: rms rel {rms_rel:.3e} max abs {max_abs:.3e} (tol {tol})")
    return rms_rel < tol


def rand_q8_rows(n, k):
    nb = k // 32
    raw = np.zeros((n, nb, 34), np.uint8)
    d = (rng.uniform(0.001, 0.02, (n, nb)).astype(np.float16))
    raw[:, :, :2] = d.view(np.uint8).reshape(n, nb, 2)
    raw[:, :, 2:] = rng.integers(0, 256, (n, nb, 32), dtype=np.uint8)
    return raw.reshape(n, nb * 34)


def rand_q4k_rows(n, k):
    nb = k // 256
    raw = np.zeros((n, nb, 144), np.uint8)
    d = rng.uniform(0.001, 0.02, (n, nb)).astype(np.float16)
    dmin = rng.uniform(0.0005, 0.01, (n, nb)).astype(np.float16)
    raw[:, :, 0:2] = d.view(np.uint8).reshape(n, nb, 2)
    raw[:, :, 2:4] = dmin.view(np.uint8).reshape(n, nb, 2)
    raw[:, :, 4:] = rng.integers(0, 256, (n, nb, 140), dtype=np.uint8)
    return raw.reshape(n, nb * 144)


def rand_q6k_rows(n, k):
    nb = k // 256
    raw = rng.integers(0, 256, (n, nb, 210), dtype=np.uint8)
    d = rng.uniform(0.0005, 0.005, (n, nb)).astype(np.float16)
    raw[:, :, 208:210] = d.view(np.uint8).reshape(n, nb, 2)
    return raw.reshape(n, nb * 210)


ok = True

# f16 GEMM, ragged M/N + ldc band
M, K, N = 37, 512, 96
a = rng.standard_normal((M, K)).astype(np.float16)
w = rng.standard_normal((N, K)).astype(np.float16)
got = gemm("gemm_mma_f16", a, w, M, N, K)
want = a.astype(np.float32) @ w.astype(np.float32).T
ok &= check("gemm_mma_f16", got, want, 2e-2)

# quant GEMMs vs numpy dequant (mutual validation of kernel + gguf_io)
for tname, kern, gen in [("Q8_0", "gemm_mma_q8_0", rand_q8_rows),
                         ("Q4_K", "gemm_mma_q4k", rand_q4k_rows),
                         ("Q6_K", "gemm_mma_q6k", rand_q6k_rows)]:
    K = 768 if tname != "Q8_0" else 704
    N, M = 95, 33
    raw = gen(N, K)
    wf = dequant_rows(raw, tname, K)
    a = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    got = gemm(kern, a, raw.reshape(-1), M, N, K)
    want = a.astype(np.float32) @ wf.T
    ok &= check(f"gemm_mma {tname}", got, want, 2e-2)

# masked softmax kernel vs numpy (incl pad cols)
Ntok, Np, H = 69, 96, 4
s = (rng.standard_normal((H * Ntok, Np)) * 4).astype(np.float16)
mask = build_mask(13, Ntok - 13, True, 24, n_cols=Np)
s_buf = ctx.buf_from(s)
m_buf = ctx.buf_from(mask)
b = fm.GemmBatch(ctx)
b.softmax_mask(s_buf, m_buf, rows=H * Ntok, ncols=Np, ntok=Ntok, scale=1.0)
b.run()
got = ctx.read(s_buf, np.float16, (H * Ntok, Np))
want = softmax(np.tile(mask, (H, 1)) + s.astype(np.float32))
ok &= check("dg_softmax_mask", got, want, 2e-2)

# GEMM-composed attention vs numpy (GQA, dual dims, region mask)
class FakeCfg:
    attn_scale = 1.0

for hd, n_kv in [(256, 8), (512, 2)]:
    P, C = 11, 53
    N = P + C
    Hq = 16
    q = rng.standard_normal((N, Hq, hd)).astype(np.float32) * 0.3
    k = rng.standard_normal((N, n_kv, hd)).astype(np.float32) * 0.3
    v = rng.standard_normal((N, n_kv, hd)).astype(np.float32) * 0.3
    mask = build_mask(P, C, n_kv == 8, 1024, n_cols=fm._pad32(N))
    drv = object.__new__(fm.DiffusionGemmaMetal)
    drv.cfg = FakeCfg()
    drv.ctx = ctx
    got = drv._attention(0, q, k, v, mask)
    want = np.empty((N, Hq, hd), np.float32)
    for h in range(Hq):
        g = h // (Hq // n_kv)
        sc = q[:, h] @ k[:, g].T + mask[:, :N]
        want[:, h] = softmax(sc) @ v[:, g]
    ok &= check(f"attention hd={hd} kv={n_kv}", got, want.reshape(N, Hq * hd), 3e-2)

print("ALL OK" if ok else "FAILURES", flush=True)
sys.exit(0 if ok else 1)
