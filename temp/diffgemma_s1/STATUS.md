# DiffusionGemma Stage 1 — SK logits parity vs llama.cpp PR #24423

Goal: load diffusiongemma-26B-A4B-it-Q4_K_M in SK and match the llama.cpp
reference canvas logits on the unified bidirectional zero-SC forward
(correctness only). Blueprint: temp/diffgemma_feas/STATUS.md.

## Reference

- llama.cpp PR #24423 head: `c84e85af61011f9fbfcf41479381d5ed1661a564`
  (branch `diffusion-visual-updates`), built on amelia at `~/llamacpp-diffg`.
- Build: CPU-only Release, `-DGGML_METAL=OFF -DGGML_CPU_REPACK=OFF`.
  REPACK MATTERS: the default build repacks all Q4_K tensors into anonymous
  RAM (no mmap) — ~8.6 GB resident before compute on a 16 GB box with colima.
  First attempt was killed at rss 7.7G/swap 2.2G; repack-free rerun stays on
  clean mmap pages.
- Harness: the PR ships `examples/diffusion-gemma-eval` — exactly the Stage-1
  contract: raw i32 [prompt|canvas] ids in, single no-cache zero-SC unified
  forward, raw f32 canvas logits out. No patches needed.
- Inputs (amelia `~/sk-diffg-s1/inputs/`): 3 chat-templated prompts
  (template from the GGUF: `<|turn>user\n{msg}<turn|>\n<|turn>model\n` +
  no-thinking channel stub `<|channel>thought\n<channel|>`, BOS auto):
  1. "What is the capital of France?" (P=20), canvas = 256 x <mask>(4)
  2. "Write a haiku about the ocean." (P=21), canvas = 256 x <mask>(4)
  3. "Explain gravity to a child."   (P=19), canvas = 256 random ids (seed 1234)

## SK implementation (models/gemma/diffusion/)

- `gguf_io.py` — dependency-free GGUF reader + ggml-exact numpy dequant
  (Q4_K/Q6_K/Q8_0/F16/F32). Weights are used NATIVE-quant on GPU.
- `config.py` — arch config from `diffusion-gemma` GGUF keys (per-layer
  kv-heads 8/2, SWA pattern [5xSWA,global]x6, dual head/rope dims 256/512,
  thetas 1e4/1e6, canvas_length 256, softcap 30, eps 1e-6).
- `graph_ref.py` — CPU f32 oracle mirroring the PR graph op-for-op
  (region embedding, masks, qk/v norms, NEOX rope w/ freq-factors, router
  softmax→top8→renorm, fused gate_up geglu + per-expert down scale, 4-norm
  sandwich, enc/dec layer scalars, tied Q6_K head + softcap). Dump taps.
- `forward_metal.py` — Metal driver: all weight matmuls via
  kernels/gemm/gemm_mma.metal (q4k/q6k/q8_0/f16, fp16 activations, f32
  accum); attention GEMM-composed per the spec (QK^T gemm → dg_softmax_mask
  additive-mask kernel → @V gemm; K/N padded to 32, GQA via per-head
  dispatch offsets); host glue f32 numpy shared with the oracle. Weights
  bound as no-copy mmap MTLBuffers (page-aligned windows; OS pager =
  streaming layer), copy fallback auto-probed.
- `dg_kernels.metal` — masked row softmax (family-local; D=128 production
  attn kernels don't apply at head_dim 256/512).
- `runner.py` — CLI emitting eval-compatible raw f32 canvas logits.
- Registry row `diffgemma-26b` + `adapter.py` (forward-only; sampler is
  Stage 2).

Key arch facts pinned during port (from PR source + GGUF):
- rope_freqs.weight [256] = 1.0 x64 then 1e30 x192 → global layers rotate
  only the first 64 pairs (proportional rope == partial 0.25 via freq-factor
  poisoning); SWA layers full-dim theta 1e4. NEOX split-half pairing.
- Global layers have NO attn_v: V = rms_noscale(raw k_proj) (pre-k-norm,
  no rope). kq_scale = 1.0 (qk-norm carries scaling).
- Router input is the UNNORMED residual: rms_noscale(x)/sqrt(d) *
  ffn_gate_inp.scale; expert input is rms(x, pre_ffw_norm_2).
- MoE: softmax over 128 → top-8 → weights renormalized (clamp 6.1e-5);
  fused gate_up [gate|up] split on output dim; gelu_tanh; per-expert
  ffn_down_exps.scale applied to down output before weighting.
- Canvas embedding: rms_noscale(embed*sqrt(2816)); prompt: embed*sqrt(2816).
- Masks (one per type): prompt rows causal-over-prompt (SWA-clipped);
  canvas rows bidirectional (global: all; SWA: last n_swa-1 prompt + canvas).
  With P ≤ 1023 the SWA and global masks coincide.

## Validation ladder

1. Synthetic op tests (test_ops.py, local M4): gemm_mma f16/q8_0/q4k/q6k vs
   numpy dequant (mutual), dg_softmax_mask vs numpy, GEMM-composed attention
   vs numpy for both (hd=256,kv=8) and (hd=512,kv=2) with region masks +
   padding: ALL OK (rms rel ≤ 7.5e-4).
2. CPU oracle vs llama.cpp ref (prompt 1): pending
3. SK GPU vs llama.cpp ref (3 prompts): pending

## Parity table

(pending)

## Perf observations

(pending)

## Stage 2 needs

(pending)
