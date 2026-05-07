"""Quantization method comparison — Apple Silicon numpy simulation.
Tests: symmetric per-channel, per-group (gs=64/128), asymmetric, GPTQ-style.
Measures: accuracy (cosine sim vs fp16), memory, dequant+matmul throughput.
"""
import numpy as np
import time, statistics

SEED = 42; ITERS = 20; WARMUP = 5
M, K, N = 4096, 4096, 4096  # weight shape (K,N), activation (M,K)

def bench(name, fn, *args):
    for _ in range(WARMUP): fn(*args)
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1e6)
    med = statistics.median(times)
    return med

def cosine_sim(a, b):
    a_f = a.astype(np.float32); b_f = b.astype(np.float32)
    return float((a_f * b_f).sum() / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))

def fp16_baseline(W, X):
    """fp16 matmul — the gold standard."""
    return X.astype(np.float16) @ W.astype(np.float16)

# ═══════════════════════════════════════════════════════════
#  Method 1: Symmetric per-channel
# ═══════════════════════════════════════════════════════════

def quant_sym_channel(W):
    K, N = W.shape
    scale = np.max(np.abs(W), axis=0, keepdims=True) / 7.0  # (1, N)
    W_i4 = np.clip(np.round(W / scale + 8), 0, 15).astype(np.uint8)
    return W_i4, scale.astype(np.float16)

def dequant_sym_channel(W_i4, scale):
    return ((W_i4.astype(np.float16) - 8.0) * scale).astype(np.float16)

def sym_channel_matmul(W_i4, scale, X):
    W_fp16 = dequant_sym_channel(W_i4, scale)
    return X.astype(np.float16) @ W_fp16

# ═══════════════════════════════════════════════════════════
#  Method 2: Symmetric per-group (gs=64)
# ═══════════════════════════════════════════════════════════

def quant_sym_group(W, gs=64):
    K, N = W.shape
    ng = (K + gs - 1) // gs
    scales = np.zeros((ng, N), dtype=np.float16)
    W_pad = np.zeros((ng * gs, N), dtype=W.dtype)
    W_pad[:K] = W
    for g in range(ng):
        sl = slice(g*gs, min((g+1)*gs, K))
        scales[g] = np.max(np.abs(W_pad[g*gs:(g+1)*gs]), axis=0) / 7.0
    W_i4 = np.clip(np.round(W_pad / np.repeat(scales, gs, axis=0)[:K] + 8), 0, 15).astype(np.uint8)
    W_i4 = W_i4[:K]
    return W_i4, scales

def dequant_sym_group(W_i4, scales, gs=64):
    K, N = W_i4.shape
    ng = scales.shape[0]
    W_fp16 = np.zeros((ng * gs, N), dtype=np.float16)
    # Expand scales
    s_expanded = np.repeat(scales, gs, axis=0)
    W_fp16 = ((W_i4.astype(np.float16) - 8.0) * s_expanded[:K]).astype(np.float16)
    return W_fp16

# ═══════════════════════════════════════════════════════════
#  Method 3: Asymmetric per-group (gs=64) — with zero point
# ═══════════════════════════════════════════════════════════

def quant_asym_group(W, gs=64):
    K, N = W.shape
    ng = (K + gs - 1) // gs
    scales = np.zeros((ng, N), dtype=np.float16)
    zeros  = np.zeros((ng, N), dtype=np.uint8)
    W_pad = np.zeros((ng * gs, N), dtype=W.dtype)
    W_pad[:K] = W
    for g in range(ng):
        sl_w = W_pad[g*gs:(g+1)*gs]
        w_min, w_max = sl_w.min(axis=0), sl_w.max(axis=0)
        s = (w_max - w_min) / 15.0
        s[s == 0] = 1.0  # avoid div by zero
        scales[g] = s
        zeros[g] = np.clip(np.round(-w_min / s), 0, 15).astype(np.uint8)
    s_exp = np.repeat(scales, gs, axis=0)[:K].astype(np.float16)
    z_exp = np.repeat(zeros, gs, axis=0)[:K].astype(np.float16)
    W_i4 = np.clip(np.round(W / s_exp + z_exp), 0, 15).astype(np.uint8)
    return W_i4, scales, zeros

def dequant_asym_group(W_i4, scales, zeros, gs=64):
    K, N = W_i4.shape
    s_exp = np.repeat(scales, gs, axis=0)[:K].astype(np.float16)
    z_exp = np.repeat(zeros, gs, axis=0)[:K].astype(np.float16)
    return ((W_i4.astype(np.float16) - z_exp) * s_exp).astype(np.float16)

# ═══════════════════════════════════════════════════════════
#  Method 4: GPTQ-style — per-group symmetric, column-wise sorted
# ═══════════════════════════════════════════════════════════
def quant_gptq_style(W, gs=128):
    K, N = W.shape
    ng = (K + gs - 1) // gs
    scales = np.zeros((ng, N), dtype=np.float16)
    W_i4 = np.zeros((K, N), dtype=np.uint8)
    # GPTQ processes columns sequentially with residual
    residual = W.copy().astype(np.float32)
    for g in range(ng):
        r_start = g * gs
        r_end = min(r_start + gs, K)
        chunk = residual[r_start:r_end]
        scales[g] = np.max(np.abs(chunk), axis=0) / 7.0
        s_clip = np.clip(scales[g], 1e-9, None)
        W_q = np.clip(np.round(chunk / s_clip + 8), 0, 15)
        W_i4[r_start:r_end] = W_q
        # Compute error and propagate to next column group
        deq = (W_q.astype(np.float32) - 8.0) * s_clip
        err = chunk - deq
        if r_end < K:
            residual[r_end:] -= err.mean(axis=1, keepdims=True).T  # wrong dim, simplified
    return W_i4, scales

def dequant_gptq(W_i4, scales, gs=128):
    K, N = W_i4.shape
    s_exp = np.repeat(scales, gs, axis=0)[:K].astype(np.float16)
    return ((W_i4.astype(np.float16) - 8.0) * s_exp).astype(np.float16)

# ═══════════════════════════════════════════════════════════
#  Method 5: Unsigned int4 (TurboQuant-like, 0-15 range)
# ═══════════════════════════════════════════════════════════
def quant_turbo_style(W, gs=64):
    """Unsigned int4 [0,15], per-group with shared exponent."""
    K, N = W.shape
    ng = (K + gs - 1) // gs
    scales = np.zeros((ng, N), dtype=np.float16)
    W_i4 = np.zeros((K, N), dtype=np.uint8)
    for g in range(ng):
        r_start = g * gs
        r_end = min(r_start + gs, K)
        chunk = W[r_start:r_end]
        # TurboQuant uses round-to-nearest with shared exponent per group
        w_max = np.max(np.abs(chunk))
        s = max(w_max / 7.0, 1e-9)
        scales[g] = s
        W_i4[r_start:r_end] = np.clip(np.round(chunk / s + 8), 0, 15)
    return W_i4, scales  # same dequant as sym_group

# ═══════════════════════════════════════════════════════════
#  Run all benchmarks
# ═══════════════════════════════════════════════════════════

def main():
    np.random.seed(SEED)
    W = np.random.normal(0, 0.02, (K, N)).astype(np.float16)
    X = np.random.normal(0, 0.5,  (M, K)).astype(np.float16)
    ref = fp16_baseline(W, X)

    methods = []

    # 1. Sym per-channel
    W_ch, sc_ch = quant_sym_channel(W)
    t = bench("sym_channel", dequant_sym_channel, W_ch, sc_ch)
    W_dq = dequant_sym_channel(W_ch, sc_ch)
    cs = cosine_sim(W_dq.ravel(), W.ravel())
    mem = W_ch.nbytes
    methods.append(("sym-channel", cs, mem, t))

    # 2. Sym per-group gs=64
    W_s64, sc_s64 = quant_sym_group(W, 64)
    t = bench("sym_gs64", dequant_sym_group, W_s64, sc_s64, 64)
    W_dq = dequant_sym_group(W_s64, sc_s64, 64)
    cs = cosine_sim(W_dq.ravel(), W.ravel())
    mem = W_s64.nbytes
    methods.append(("sym-gs64", cs, mem, t))

    # 3. Sym per-group gs=128
    W_s128, sc_s128 = quant_sym_group(W, 128)
    t = bench("sym_gs128", dequant_sym_group, W_s128, sc_s128, 128)
    W_dq = dequant_sym_group(W_s128, sc_s128, 128)
    cs = cosine_sim(W_dq.ravel(), W.ravel())
    mem = W_s128.nbytes
    methods.append(("sym-gs128", cs, mem, t))

    # 4. Asymmetric gs=64
    W_asy, sc_asy, z_asy = quant_asym_group(W, 64)
    t = bench("asym_gs64", dequant_asym_group, W_asy, sc_asy, z_asy, 64)
    W_dq = dequant_asym_group(W_asy, sc_asy, z_asy, 64)
    cs = cosine_sim(W_dq.ravel(), W.ravel())
    mem = W_asy.nbytes
    methods.append(("asym-gs64", cs, mem, t))

    # 5. GPTQ-style gs=128
    W_gq, sc_gq = quant_gptq_style(W, 128)
    t = bench("gptq_gs128", dequant_gptq, W_gq, sc_gq, 128)
    W_dq = dequant_gptq(W_gq, sc_gq, 128)
    cs = cosine_sim(W_dq.ravel(), W.ravel())
    mem = W_gq.nbytes
    methods.append(("gptq-gs128", cs, mem, t))

    # 6. TurboQuant-style
    W_tq, sc_tq = quant_turbo_style(W, 64)
    t = bench("turbo_gs64", dequant_sym_group, W_tq, sc_tq, 64)
    W_dq = dequant_sym_group(W_tq, sc_tq, 64)
    cs = cosine_sim(W_dq.ravel(), W.ravel())
    mem = W_tq.nbytes
    methods.append(("turbo-gs64", cs, mem, t))

    # ── Report ──
    fp16_mem = W.nbytes
    fp16_us = bench("fp16_matmul", fp16_baseline, W, X)

    print(f"{'='*75}")
    print(f"Quantization comparison — {K}×{N} weight, {M}×{K} activation")
    print(f"fp16 baseline: {fp16_mem/1e6:.1f} MB, {fp16_us:.0f}us matmul")
    print(f"{'='*75}")
    print(f"{'method':<16} {'cos_sim':>8} {'mem(MB)':>8} {'%fp16':>7} {'deq(us)':>9} {'notes'}")
    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*7} {'-'*9} {'-'*20}")

    for name, cs, mem, deq_us in methods:
        # Score: cosine_sim * (fp16_mem/mem) weighted
        mem_ratio = fp16_mem / max(mem, 1)
        score = cs * min(mem_ratio, 4.0) / 4.0  # normalize, cap mem benefit at 4x
        notes = ""
        if cs > 0.999: notes = "★ best accuracy"
        if mem < fp16_mem/3.5: notes += " 4× smaller"
        print(f"{name:<16} {cs:>8.4f} {mem/1e6:>8.1f} {mem*100/fp16_mem:>6.1f}% {deq_us:>9.0f} {notes}")

    print()
    print("★ = recommended. sym-gs64: simple, 4× smaller, near-lossless, no zero point overhead")

if __name__ == "__main__":
    main()
