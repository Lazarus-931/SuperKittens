# DiffusionGemma-26B-A4B-it → SK port blueprint (Phase A)

Sources: unsloth/diffusiongemma-26B-A4B-it-GGUF Q4_K_M (downloaded, amelia
~/diffgemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf, 16,806,810,336 B;
metadata dump at ~/diffgemma-gguf/dump.txt) + llama.cpp draft PR
ggml-org#24423 (full diff at /tmp/diffgemma_pr.diff on the laptop; key files
examples/diffusion/diffusion.cpp, src/models/diffusion-gemma.cpp).

## Architecture (verified from GGUF header)

| field | value |
|---|---|
| arch | `diffusion-gemma`, 30 layers, d_model 2816, vocab 262144 |
| ISWA | pattern [SWA×5, global]×6 (`sliding_window_pattern`), window 1024 |
| heads | 16 q-heads all layers; kv-heads 8 (SWA) / 2 (global) |
| head_dim | **256 (SWA) / 512 (global)** — dual dims; rope dims match (256/512) |
| rope | theta 1e6 (global) / 1e4 (SWA) |
| MoE | 128 routed experts (ff 704) + dense/shared ffn 2112, 8 used; expert tensors Q4_K/Q6_K/Q8_0 mix |
| head | tied embed (no separate output tensor beyond 0.56GiB embed), final softcap 30.0 |
| attention.causal | **false** |
| diffusion | canvas_length 256, mask_token_id 4 (`<mask>`) |

Byte budget (Q4_K_M, total 15.64 GiB): routed experts **14.09 GiB (90%)**;
attention 0.60; embed 0.56; dense ffn 0.37; norms/scales+head ~0.02.
Non-expert backbone ≈ **1.55 GiB**.

## Reference runtime contract (from PR #24423)

Graph (src/models/diffusion-gemma.cpp):
- Backbone "identical to gemma4" (shared weights; gemma4-common). ONE unified
  forward over [prompt | canvas], split P = n_tokens − C. Region-aware:
  1. embeddings: prompt = embed·sqrt(n_embd); canvas = rmsnorm_noscale(embed·sqrt(n_embd))
  2. per-layer scalars: `blk.N.enc_layer_output_scale` (prompt rows) vs
     `blk.N.layer_output_scale` (canvas rows) — tiny F32 scalars
  3. additive mask: prompt queries causal over prompt only (SWA-clipped);
     canvas queries bidirectional — global layers see ALL prompt+canvas,
     SWA layers see last (n_swa−1) prompt positions + all canvas
- "A single no-cache bidirectional forward over [prompt|canvas] reproduces
  the two-pass (causal encoder prefill + bidirectional decoder, zero
  self-conditioning) result" → Stage-1 needs NO KV machinery.
- Cached mode (perf): prefill prompt once into prefix KV; per step decode
  canvas-only with rectangular mask [P+C, C].
- Self-conditioning (SC): previous step's RAW canvas logits [n_vocab, C]
  uploaded as an input; SC subgraph feeds canvas embedding (gated off at
  step 0 via scale 0; uses softmax at prev step's 1/t).

Sampler (examples/diffusion/diffusion.cpp, EntropyBoundSampler):
- Canvas RANDOM-initialized (uniform over vocab, NOT mask tokens).
- Loop cur_step = S..1: t = t_min + (t_max−t_min)·(cur_step/S); per position:
  argmax, entropy H of softmax(logits/t), one multinomial sample; stash raw
  row for SC.
- Accept the lowest-entropy positions while the cumulative entropy of
  strictly-earlier accepted positions ≤ entropy_bound (MI budget); accepted
  positions take their SAMPLED token, the rest are RE-RANDOMIZED.
- Output = argmax canvas (not the working canvas). Adaptive stop: argmax
  unchanged for stability_threshold steps AND mean entropy <
  confidence_threshold. suppress_mask_token: logit[mask]=−inf.
- Defaults seen in PR flags: ~48 max steps, t 1.0→0.6.

## Fit plan (16GB M4 minis, wired ceiling ~12.7 GiB)

- Single-host RESIDENT: impossible (15.64 GiB).
- **2-host layer split (target)**: ~15 layers/host ≈ 7.0 GiB experts + share
  of backbone → ~7.8 GiB resident/host. Comfortable. Needs full GGUF on both
  hosts' disk (amelia has it; derek needs ~7 GiB freed — candidate
  Qwen3-14B-Q4_K_M 8.5 GiB, USER AUTHORIZATION required).
- Single-host degraded (Stages 1–2 correctness): mmap + paging on amelia.
  Expected expert touch per step: 8/128 per token × 256 tokens → essentially
  all 128 experts per layer per step → full-file reads when paging; slow but
  correct for small canvases/prompts. Use C=64 canvas + short prompts for
  validation to bound the working set.
- Per-step estimate (2-host resident): expert-read bound ≈ 7 GiB/host /
  ~110 GB/s ≈ 65 ms + attention/compute + hop → 150–500 ms/step realistic
  band; 48 steps/256-tok block → **~10–35 tok/s 26B-class** (vs ~5 tok/s AR
  if it fit). 3090 reference 326 ms/step at 8× bandwidth ⇒ temper to the low
  end until measured; adaptive early-stop often cuts steps well below S.

## SK port surface

REUSES: gemma4 launcher skeleton (ISWA layout, rope pair, softcap, qk-norm),
deepseek MoE kernels (moe_group/mul_mv_id/down_scatter) + Q4_K MoE lab port,
gemm_mma (M = P+C rows ≈ prefill shapes), loader GGUF plumbing.

NEW (layout: models/gemma/diffusion/ family package; kernels only there or
kernels/<op>/):
1. Loader rows for `diffusion-gemma` arch keys + enc/dec scale tensors.
2. Masked attention: additive-mask (or predicate) attention kernel for
   head_dim 256 and 512 at M=P+C — the D=128 dense kernels do NOT apply;
   simplest correct path = QK^T GEMM + mask add + softmax + V GEMM via
   existing gemm_mma pieces (prefill-style, no streaming KV) for Stage 1.
3. Region-aware embedding + per-layer scalar plumbing.
4. Sampler host loop (Python first; mirrors the reference exactly, incl.
   seed-reproducible RNG and SC buffer).
5. SC subgraph (softmax(prev logits/t_prev) → canvas embedding mix) — can be
   STUBBED OFF for Stage 1 (reference gates it off step 0 / zero-SC unified
   forward is the documented equivalence).
6. Stage 3: prefix-KV cached mode + 2-host layer-range split (pipeline.py
   generalization for this family).

## Stages

- **Stage 1 — logits parity**: unified no-cache forward, zero SC, small
  prompt + C=64..256 canvas on amelia (mmap OK). Validate per-position canvas
  logits vs llama.cpp PR build (build llama-diffusion-cli on amelia from the
  PR branch — llama.cpp runtime-compiles Metal, CLT-only OK; or CPU eval).
  Gate: max rel err small + argmax canvas identical on ≥3 seeds/prompts.
- **Stage 2 — e2e coherent**: sampler loop + adaptive stop + chat template;
  greedy/argmax output coherent; reproduce a reference generation
  token-for-token at fixed seed (CPU RNG is ours = exact match possible).
- **Stage 3 — perf**: prefix-KV cached canvas decode; 2-host expert split;
  measure ms/step + e2e tok/s; tune (expert batching at M=256 is
  prefill-like — MMA grouped path).

## Risks
1. Dual head_dim (256/512) attention — new territory; mitigated by
  GEMM-composed attention for Stage 1 (correctness first).
2. SC subgraph semantics (exact mixing op) — extract precisely from the PR
   model file before Stage 2; Stage 1 doesn't need it.
3. Paging thrash on single-host validation — bound with C=64 + short prompts;
   colima resident on amelia shrinks headroom.
4. The PR is a moving draft — pin the commit SHA used for parity.
