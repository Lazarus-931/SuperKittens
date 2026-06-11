# Prefill-attention share profile — qwen, single-stream TTFT (PROFILE-ONLY)

**Question:** at realistic prompt lengths (T=512), what fraction of single-stream
prefill wall time is the attention stage (Q@K^T + softmax + @V over T×T) vs the
MMA projections — is a prefill-attention optimization a ≥10% e2e TTFT lever?

**Answer: NO-GO. Attention is 5.1% of TTFT at T=512 on 4B (9.3% at T=1024,
2.0% on 14B at T=512). A hypothetical 2× faster prefill attention buys 2.6%
e2e at T=512 on 4B, 1.0% on 14B.**

## Setup
- Host: derek (M4 base, 16 GB), CLT-only — clang++ dylib + `SK_METAL_SRC_FALLBACK`
  runtime compile. Fresh `~/sk-prof-pfattn`, tree = local `main` @ 096c4d6.
- Model: `qwen3-4b-q4km` (`~/qwen-gguf/Qwen3-4B-Q4_K_M.gguf`), seq_max=1024,
  cache_max=2048; 14B spot check seq_max=512, cache_max=1024.
- Method: single-chunk `sk_qwen_forward(T)` (one command buffer; stages not
  interleaved). GPU-busy via `SK_QWEN_GPUPROF` (GPUEnd−GPUStart). 2 warmups +
  5 reps median, 0.3 s gaps. Rep spread ≤0.2% for T≥256 (≤1.7% overall).
- Attribution: stage-skip gates already in `models/qwen/qwen_model.h`
  (`SK_PROF_SKIP_ATTN` drops only the `mha_causal_prefill` dispatch at seq>1;
  `SK_PROF_SKIP_XPOSE` drops the 4 seq<->head transposes; decode untouched).
  Separate process per mode (gates latch as static consts).
- NOTE: first chain attempt was poisoned by a concurrent gemma-12B gate run on
  derek (T=32 read 20.5 s gpu-busy under swap thrash). Killed my chain, waited
  for the box to clear, reran clean. All numbers below are from the clean run.

## TTFT(T) sweep — qwen3-4B-Q4_K_M (gpu-busy median, ms)

| T    | base    | skip_attn | attn (Δ) | attn share | skip_xpose Δ |
|------|---------|-----------|----------|------------|--------------|
| 32   | 145.2   | 145.6     | −0.3     | (noise)    | +0.96 (0.7%) |
| 64   | 251.6   | 248.5     | 3.1      | 1.2%       | +0.43 (0.2%) |
| 128  | 474.9   | 470.6     | 4.3      | 0.9%       | +1.62 (0.3%) |
| 256  | 904.5   | 878.9     | 25.6     | 2.8%       | +1.25 (0.1%) |
| 512  | 1849.8  | 1756.1    | 93.6     | **5.1%**   | +4.29 (0.2%) |
| 1024 | 3870.3  | 3510.0    | 360.4    | **9.3%**   | +11.9 (0.3%) |

Wall ≈ gpu_busy + 2-4 ms at every T — prefill is fully GPU-bound; host overhead
is irrelevant. Transposes are free (skip_xpose Δ < 0.5% everywhere).

## Fit: TTFT(T) = c + a·T + b·T²  (relative-error-weighted LSQ on base gpu_med)

c = 38.9 ms, a = 3.311 ms/tok, b = 0.4207 µs/tok². Fit error ≤1.1% at all six T.

| T    | quad share (fit) | ablation share |
|------|------------------|----------------|
| 128  | 1.5%             | 0.9%           |
| 256  | 3.0%             | 2.8%           |
| 512  | 6.0%             | 5.1%           |
| 1024 | 11.4%            | 9.3%           |

Fit and ablation agree; the fit's quadratic term runs slightly high because it
also absorbs the KV-write growth and MMA tail effects. TTFT is dominated by the
linear term: 3.31 ms/token of projection/MLP GEMM work (≈277 tok/s prefill;
the projections are the compute-bound cost, exactly as the gemm_mma design
intends — weights are streamed once per chunk, so there is no bandwidth-bound
weight re-read to hide).

## 14B spot check (T=128/512, gpu-busy median, ms)

| T   | base   | skip_attn | attn (Δ) | attn share |
|-----|--------|-----------|----------|------------|
| 128 | 1636.9 | 1625.6    | 11.3     | 0.7%       |
| 512 | 6549.8 | 6417.2    | 132.6    | **2.0%**   |

Attention share SHRINKS with model size: attention grew 1.42× from 4B→14B
(93.6→132.6 ms at T=512, ≈ the heads·layers ratio 40·40/32·36 = 1.39×) while
the projection term grew ~3.5× (≈ the param ratio). Bigger models are even
less attention-bound at prefill.

## Ceiling math
- Attention-FREE TTFT(512) on 4B = 1756 ms vs 1850 ms → only **5.1%** better.
- 2× faster prefill attention: 0.5 × share → **2.6% e2e at T=512**, 4.7% at T=1024.
- For a 2× attention win to clear 10% e2e, attention share must be ≥20%.
  From the fit, share(T)=0.20 at **T ≈ 2000 tokens** — beyond typical prompts
  and the default seq_max envelope. Even there a 2× win is exactly at the 10% bar.

## GO/NO-GO: **NO-GO**
Attention is 5% of single-stream prefill at T=512 and ~9% at T=1024; the
projections' linear term (3.31 ms/tok) owns TTFT. The BR=8 `mha_causal_prefill`
already amortizes K/V re-streaming well enough that prefill-attention work is a
<3% e2e lever at realistic prompt lengths. Prefill optimization effort should
target the MMA projection path (the a-term), not attention.

## Artifacts
- `prof_prefill.py` (sweep driver, fd2-redirect gpuprof capture), `analyze.py`
  (fit + ablation cross-check), `build_dylib.sh`, `run.sh` (derek env wrapper).
- Raw JSONs: `base_4b.json`, `skipattn_4b.json`, `skipxpose_4b.json`,
  `base_14b.json`, `skipattn_14b.json` (5 reps each, wall + gpu).
- derek: `~/sk-prof-pfattn/` (tree, dylib, logs).
