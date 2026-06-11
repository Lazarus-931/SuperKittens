# Granite-4 hybrid — Stage 1: PORTED-COHERENT (2026-06-11)

Model: **granite-4.0-h-1b** (Q8_0, arch `granitehybrid`) — 40 layers: 36 mamba2
+ 4 attention (A at 5/15/25/35, per-layer type from
`granitehybrid.attention.head_count_kv`), dense SwiGLU FFN every layer, NoPE
attention (attention_multiplier 1/128), granite scalar multipliers
(embed 12 / residual 0.22 / logit 1/6), tied Q8_0 head, vocab 100352.
h-1b over h-micro because h-micro is head_dim=64 (SK dense attention is D=128-only).

Reuses mamba2 family kernels (conv1d_silu/_step, conv_state_capture,
mamba2_ssd, gate_norm) and shared dense kernels (q8_0_matvec, mha_causal D=128,
kv_cache_write, rmsnorm, silu_mul) unchanged; `granite_ops.metal` adds the two
elementwise kernels that carry the granite multipliers.

## Gates (lexie M4 base, greedy)
- **Load**: every dim cross-checked against GGUF metadata, fail-loud; config
  table printed at load.
- **Coherence**: QA prompt is **48/48 token-identical to llama.cpp CPU greedy**
  (GGML_METAL=OFF, GGML_CPU_REPACK=OFF). Pizza-dough poem diverges at token 8
  on an fp16 near-tie (SK top-2 gap 0.125: ' cr' 36.4062 vs ' mound' 36.2812);
  both continuations fluent. Logits finite. Outputs byte-identical across
  M-series hosts.
- **No-regression**: Qwen3-1.7B-Q8 32-tok greedy token-identical (32/32),
  branch build vs pristine main build.
- **Decode**: median **51.2 tok/s** (2 warmup + 5 reps × 64 tok; spread
  50.99–51.30; indicative — lexie is thermal-drifty).

## Stage-1 limits
- Prompt must fit one prefill forward (chunked mamba prefill would re-zero-pad
  the conv left edge); decode is O(1)-state.
- `generate_n` demands a fresh sequence (mamba state); the adapter resets per
  call.
- fp16 logits end-to-end: greedy near-ties (gap ≲ 0.13) may flip vs f32
  references.
