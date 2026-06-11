# DiffusionGemma Stage 1 — SK logits parity vs llama.cpp PR #24423

Goal: load diffusiongemma-26B-A4B-it-Q4_K_M in SK and match the llama.cpp
reference canvas logits on the unified bidirectional zero-SC forward
(correctness only). Blueprint: temp/diffgemma_feas/STATUS.md.

VERDICT: **PARITY within the reference's own implementation-noise envelope.**
Mask-canvas prompts (the real step-0 inputs): argmax-canvas **100%** (gate
≥95%). Random-canvas prompt: SK-vs-reference argmax 68.8% — but llama.cpp
vs ITSELF (FA on/off, same binary/weights) agrees on only **60.9%** there, so
the gate is unachievable between any two non-bit-identical implementations on
that input class. On every prompt, SK's distance to the reference ≈ the
reference's distance to itself. Cause: the zero-SC forward over an
uninformative canvas produces diffuse, near-tied distributions, and the MoE
router (top-8 of 128 on a rms-normed residual) chaotically amplifies ANY
arithmetic difference via expert-selection flips. See "Noise calibration".

## Reference

- llama.cpp PR #24423 head: `c84e85af61011f9fbfcf41479381d5ed1661a564`
  (branch `diffusion-visual-updates`), built on amelia at `~/llamacpp-diffg`.
- Build: CPU-only Release, `-DGGML_METAL=OFF -DGGML_CPU_REPACK=OFF`.
  REPACK MATTERS on a 16 GB host: the default build repacks all Q4_K/Q8_0
  tensors into ~9 GB of anonymous RAM (no mmap). Repack-free stays on clean
  mmap pages (rss ~8 GB but evictable; swap stable ~1.5 GB; ~30 s/forward).
- Harness: the PR ships `examples/diffusion-gemma-eval` — exactly the Stage-1
  contract (raw i32 [prompt|canvas] ids in → single no-cache zero-SC unified
  forward → raw f32 canvas logits). Used unpatched.
- Inputs (amelia `~/sk-diffg-s1/inputs/`): 3 chat-templated prompts using the
  GGUF's own template (`<|turn>user\n{msg}<turn|>\n<|turn>model\n` + no-think
  channel stub `<|turn>... <|channel>thought\n<channel|>`; BOS auto). NOTE the
  vocab does NOT contain `<start_of_turn>` — this family uses `<|turn>`(105) /
  `<turn|>`(106) / `<|channel>`(100) / `<channel|>`(101).
  1. "What is the capital of France?" (P=20), canvas = 256 x <mask>(4)
  2. "Write a haiku about the ocean." (P=21), canvas = 256 x <mask>(4)
  3. "Explain gravity to a child."   (P=19), canvas = 256 random ids (seed 1234)

## SK implementation (models/gemma/diffusion/)

- `gguf_io.py` — dependency-free GGUF reader + ggml-exact numpy dequant
  (Q4_K/Q5_0/Q6_K/Q8_0/F16/F32).
- `config.py` — arch config from `diffusion-gemma` GGUF keys (per-layer
  kv-heads 8/2, SWA pattern [5xSWA,global]x6, dual head/rope dims 256/512,
  thetas 1e4/1e6, canvas_length 256, softcap 30, eps 1e-6).
- `graph_ref.py` — CPU f32 oracle mirroring the PR graph op-for-op
  (region embedding, masks, qk/v norms, NEOX rope w/ freq factors, router
  softmax→top8→renorm(clamp 6.1e-5), fused gate_up geglu + per-expert down
  scale, 4-norm sandwich, enc/dec layer scalars, tied Q6_K head + softcap).
- `forward_metal.py` — Metal driver: every weight matmul on SK quant GEMM
  kernels (kernels/gemm/gemm_mma.metal: q4k/q6k/q8_0/f16, fp16 activations,
  f32 accum; family dg_gemm_mma_q5_0 reuses the GEMM_MMA_BODY macro);
  attention GEMM-composed per spec: QK^T (dg_gemm_qkt_f32, f32 C — ggml
  forces PREC_F32 for kq and fp16 C can overflow w/o 1/sqrt(d) prescale) →
  dg_softmax_mask (f32 scores + f32 additive region mask → f16 probs) →
  @V gemm_mma_f16. K/N padded to 32; GQA via per-head dispatch offsets.
  Host glue f32 numpy shared with the oracle (same dump taps → bisectable).
- `dg_kernels.metal` — family kernels, concatenated AFTER gemm_mma.metal at
  runtime compile so the skmma loaders/macro are reused, not copied.
- `runner.py` — CLI emitting eval-compatible raw f32 canvas logits.
- Registry row `diffgemma-26b` + `adapter.py` (forward-only; sampler = Stage 2).

Arch facts pinned during the port (PR source + GGUF):
- This Q4_K_M mix puts ffn_down AND ffn_down_exps at **Q5_0 on 16/30 layers**
  (Q8_0 on the rest); attn_v is Q6_K(13)/Q4_K(12); blueprint's "Q4_K/Q6_K/Q8_0"
  was incomplete → native Q5_0 GEMM + dequant required.
- rope_freqs.weight [256] = 1.0 x64 then 1e30 x192 → global layers rotate only
  the first 64 NEOX pairs (proportional rope == partial 0.25 via freq-factor
  poisoning); SWA layers full-dim theta 1e4.
- Global layers have NO attn_v: V = rms_noscale(raw k_proj) (pre-k-norm, no
  rope). kq_scale = 1.0 (qk-norm carries scaling).
- Router input is the UNNORMED residual: rms_noscale(x)/sqrt(d) *
  ffn_gate_inp.scale; expert input is rms(x, pre_ffw_norm_2). Top-8 weights
  renormalized; per-expert ffn_down_exps.scale on down output before weighting.
- Canvas embedding rms_noscale(embed*sqrt(2816)); enc/dec layer-output scalars
  split at P; masks: prompt rows causal-over-prompt (SWA-clipped), canvas rows
  bidirectional (global: all; SWA: last n_swa-1 prompt + canvas). With
  P ≤ 1023 the SWA and global masks coincide.

## Parity table (canvas positions = 256, n_vocab = 262144)

SK GPU vs llama.cpp reference (CPU, FA off):

| prompt | rms rel/pos (mean/worst) | mean rel/pos | ARGMAX | argmax (mask-suppressed) | top5 overlap |
|---|---|---|---|---|---|
| p1 (mask canvas) | 0.085 / 0.297 | 0.251 | **256/256 = 100%** | 146/256 = 57.0% | 4.09/5 |
| p2 (mask canvas) | 0.057 / 0.260 | 0.165 | **256/256 = 100%** | 205/256 = 80.1% | 4.36/5 |
| p3 (random canvas) | 0.115 / 0.284 | 0.750 | 176/256 = 68.8% | 176/256 = 68.8% | 4.50/5 |

Reference self-noise on the same inputs (llama.cpp FA=0 vs FA=1, same
binary/weights/host — the floor any implementation comparison sits on):

| prompt | rms rel/pos mean | ARGMAX | top5 overlap |
|---|---|---|---|
| p1 (mask canvas) | 0.046 | 100% (nomask 71.1%) | 4.20/5 |
| p3 (random canvas) | 0.106 | **60.9%** | 4.51/5 |

SK-vs-reference ≈ reference-vs-itself on both input classes; on p3 SK agrees
with the FA=0 reference BETTER than the FA=1 reference does (68.8% vs 60.9%).

## Noise calibration (all on p1 — why logit rel err CANNOT be ~1e-3 here)

| pair | rms rel/pos mean | argmax | argmax nomask | max abs |
|---|---|---|---|---|
| llama.cpp FA=0 vs FA=1 (same binary/weights) | 0.046 | 100% | 71.1% | 12.6 |
| SK CPU-f32 oracle vs llama.cpp | 0.075 | 100% | 68.8% | 13.7 |
| SK GPU vs llama.cpp | 0.085 | 100% | 57.0% | 13.6 |
| SK GPU vs SK CPU-f32 oracle (identical graph) | 0.032 | 100% | 87.1% | 12.7 |

Mechanism (from per-layer GPU-vs-oracle dumps): layer-0 divergence is pure
fp16 rounding (rms 3e-4), but router expert-selection flips grow 0.09% → ~8%
by layer 8 and each flip injects an O(1) local error → rms ~2-3% by layer 10,
6.6% at result_norm. llama.cpp's own Q8_K-quantized-activation matmuls are a
~10x larger per-op noise source than SK's fp16 hops, with the same chaotic
amplifier — hence even a perfect fp32 mirror lands at ~7% rms. The all-mask
step-0 canvas maximizes near-ties (diffuse distributions), making
mask-suppressed argmax intrinsically unstable across ANY two implementations.

Per-layer l_out rms-rel drift (GPU vs oracle, p1): 3.3e-4 (L0) → 2.1e-2 (L9)
→ 6.3e-2 (L14) → ~7-9.5e-2 plateau (L17-28) → 4.7e-2 (L29). Smooth growth,
no step-change at any layer — chaotic accumulation, not a localized op bug.
First-divergence is L0 at fp16-rounding scale, i.e. zero graph-math delta.

## Memory war stories (16 GB host, 15.65 GiB model — READ BEFORE STAGE 2)

The GPU forward took amelia down HARD twice (full reboots) before stabilizing:
1. MAP_PRIVATE+PROT_WRITE no-copy mmap windows (pyobjc requires writable
   buffers for newBufferWithBytesNoCopy): GPU access converts touched pages
   to anonymous memory → ~15 GB un-evictable → box death. True no-copy needs
   MAP_SHARED PROT_READ (llama.cpp-style), which the Python bridge can't
   express.
2. pyobjc WITHOUT autorelease pools pins every MTLBuffer proxy until process
   exit (~0.85 GB/layer); per-layer `objc.autorelease_pool()` is mandatory.
3. Even with pools + per-layer eviction, per-tensor Metal buffer churn
   (~0.85 GB/layer outside process rss) swap-stormed the host. Fix that
   stuck: persistent per-ROLE scratch MTLBuffers (~1.2 GB once), refilled by
   memcpy per layer + madvise(DONTNEED) on streamed file pages.
4. Watchdog rules that work: kill on system swap > 5.5 GB or root-disk free
   < 400 MB; do NOT kill on rss (dominated by evictable clean file pages).
   macOS swapfiles eat root disk — the first parity run died on a 3 MB dump
   write with 15 GB of swapfiles around.

## Perf observations

- llama.cpp CPU reference: ~30 s/forward (P+256 tokens, warm).
- SK GPU forward: 42-47 s warm (~1.5 s/layer + head). Stage-1 shape:
  per-op CPU round trips, per-expert GEMM loop, memcpy weight streaming
  (15.6 GB/forward). Cold-cache (post-reboot) ~6-35 s/layer, dominated by
  paging.
- GGUF parse (dependency-free reader): 0.5 s for 692 tensors + tokenizer.

## Stage 2 needs

1. Sampler loop (EntropyBoundSampler) in Python per blueprint; the unified
   forward is the verify path, cached PREFILL/DECODE phases the fast path.
2. Self-conditioning subgraph (sc_embT soft-embedding @ prev logits softmax →
   gated MLP into canvas embedding) — tensors already in the GGUF (Q5_0/Q4_K).
3. Seed-reproducible parity vs the reference SAMPLER (token-level), which is
   robust to the logit noise above (sampler operates on temperature-scaled
   distributions; adjacent steps re-randomize rejected positions anyway).
4. Perf: fold host glue (norms/rope/router/geglu) on-device; one command
   buffer per layer; keep persistent scratch slots (also the right structure
   for the 2-host layer split); batch expert GEMMs via mul_mat_id-style
   grouped kernel; double-buffer weight memcpy against GPU compute.
5. DISK: amelia has ~4-8 GB free; swapfiles + 268 MB logit files collide.
   Clean as you go.

## Files

- amelia `~/sk-diffg-s1/`: inputs/, ref_p{1,2,3}.bin (llama.cpp canvas
  logits), ref_fa_p1.bin (FA variant), sk_gpu_p{1,2,3}.bin, sk_cpu_p1.bin
  (oracle), dump_{gpu,cpu}_p1/ (per-layer taps), eval_{norepack,repack}
  binaries, logs.
- Repo: SuperKittens/models/gemma/diffusion/ (family), registry row,
  temp/diffgemma_s1/ (tests + compare tools + this STATUS).
