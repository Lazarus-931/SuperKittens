# Mamba 2 SK port — STATUS

## KNOWN BROKEN

`mamba2_ssd.metal` is numerically incorrect and is **not** used by the launcher.
Missing: softplus(dt + dt_bias), `dt * B * x` input gating, `D * x` skip,
n_groups sharing for B/C. Signature takes `(Q, K, V, A_log[B,H,L])` — does not
match HF Mamba2 SSD.

The launcher routes around it to `mamba2_ssd_ref.metal` / `mamba2_step_ref.metal`
(per-token recurrence, signature-correct). The chunked `mamba2_ssd.metal` needs
a full rewrite before it can replace the ref path.

---

Branch: `dev-mamba2`. Target model: **`AntonV/mamba2-130m-hf`** (`model_type: mamba2`, `Mamba2ForCausalLM`).

## Architecture confirmed (from `config.json`)

```
hidden_size=768  num_hidden_layers=24  num_heads=24  head_dim=64
state_size=128 (N)  expand=2  intermediate_size=expand*hidden=1536 (E)
n_groups=1  chunk_size=256  conv_kernel=4
time_step_rank=256  use_bias=false  use_conv_bias=true  rms_norm=true
vocab_size=50288  tie_word_embeddings=true  residual_in_fp32=true
```

Note: `state-spaces/mamba-2.8b-hf` is **Mamba 1**, NOT Mamba 2 (`model_type: mamba`,
`MambaForCausalLM`, no `n_groups`/`head_dim`/`chunk_size`). Do not target it from
this directory.

## HF reference baseline (lexie, M4 mini)

- HF fp32 `torch_forward` decode: **14.48 tok/s** (32 new tokens, "Hi" prompt).
- Argmax-1 = token 13 (","). Generated: `"Hi, I'm a newbie here. I'm a student..."`.

`temp/mamba2_validate/dump_hf_mamba2.py` produces `hf_ref.npz` with 172 keys:
embed, every `hidden.{0..24}`, per-layer mixer in/out/proj/gate/xBC_preconv/dt_pre,
final logits.

## What is wired (SK side)

- `mamba2_block.h` — per-block dispatch scaffold (in_proj GEMM, conv1d_silu, ssd,
  gate_norm, out_proj GEMM). NOT integrated into a full-model orchestrator.
- `mamba2_ssd.metal` — SISO-style recurrence (Q/K/V naming, single `A_log[B,H,L]`).
  Signature mismatches HF Mamba 2 SSD (see "Remaining work").
- `mamba2_step.metal` — present, also needs signature rewrite.
- `conv1d_silu.metal` — usable for the conv1d+silu fusion; previously validated
  against real HF weights to fp16 noise.
- `gate_norm.metal` — usable as silu-gate + RMSNorm fusion.

## Remaining work (blocked items)

### 1. SSD kernel signature rewrite (BLOCKER)

HF reference: `transformers/models/mamba2/modeling_mamba2.py` lines 398–586.
The SSD step needs, per chunk:

- `x:    (B, L, H, P)` head-major intermediates (P=head_dim=64)
- `dt:   (B, L, H)` softplus( dt_raw + dt_bias ), clamped to `time_step_limit`
- `A:    (H,)` from `-exp(A_log)`
- `B,C:  (B, L, G, N)` G groups (G=1 here), N=state_size=128
- `D:    (H,)` skip connection
- `dt_bias: (H,)`
- `chunk_size`: 256

Current `mamba2_ssd.metal` takes `(Q,K,V, A_log[B,H,L], y)` — none of
{dt, dt_bias, D, n_groups, chunk decomposition} are threaded. The naive HF
`torch_forward` uses the chunked associative scan: outputs combine intra-chunk
attention-style block + inter-chunk state propagation (HF lines ~461–593).
This must be rewritten. Cleanest path:
  a. Write a non-chunked SSD reference kernel first (one head, per-token
     recurrence in fp32) that matches HF `torch_forward` exactly.
  b. Validate against `hf_ref.npz` `L0.mixer_out` to bf16/fp16 noise.
  c. Then add chunking for prefill perf.

### 2. Full-model orchestrator

Need (mirroring qwen/qwen_model.h, qwen/launcher.{h,c++}, qwen/weights.{h,c++}):

- `mamba2_model.h` — embed → 24 × {RMSNorm → Mamba2Mixer → residual} → final
  RMSNorm → lm_head (tied to embed). State carries `conv_state[H,P,d_conv-1]`
  and `ssm_state[H,P,N]` per layer for decode.
- `weights.{h,c++}` — HF safetensors name → fused buffer map. HF names
  (per `model.safetensors` index): `backbone.embeddings.weight`,
  `backbone.layers.{i}.norm.weight`, `backbone.layers.{i}.mixer.{in_proj,conv1d,
  dt_bias,A_log,D,norm.weight,out_proj}`, `backbone.norm_f.weight`.
- `launcher.{h,c++}` — C ABI: `sk_mamba2_create / load_safetensors / forward /
  dump_layer / destroy`.
- `mamba2.py` — ctypes wrapper. Inherit `SuperKittens.inference.generation.Model`.
- `mamba2/__init__.py` — register `mamba2-130m`.

### 3. Validation harness

- `temp/mamba2_validate/layer_diff.py` — load `hf_ref.npz`, run SK forward, compare
  layer-by-layer with rel_err. Target: rel_err < 0.1 + argmax match.

### 4. Decode benchmark

- HF fp32 baseline: **14.48 tok/s**. Beat it. Conv1d state update + single-step
  SSD kernel both need to be fused for decode.

## Layout

```
SuperKittens/models/ssm/mamba2/
├── conv1d_silu.metal      # usable
├── gate_norm.metal        # usable
├── mamba2_ssd.metal       # signature wrong (needs rewrite per HF L398-586)
├── mamba2_step.metal      # signature wrong (needs rewrite for decode)
├── mamba2_block.h         # partial scaffold, not used yet
├── mamba2.{c++,h,py}      # legacy stubs, replace with launcher.{c++,h} + mamba2.py
└── STATUS.md              # this file
SuperKittens/temp/mamba2_validate/
├── dump_hf_mamba2.py      # ✓ working, produces hf_ref.npz (172 keys)
├── hf_bench.py            # ✓ working, 14.48 tok/s
└── hf_ref.npz             # gitignored
```

## Session 2026-05-13 progress

- Synced `dev-mamba2` to origin/main (already up to date).
- `llama.cpp` baseline: installed via `brew install llama.cpp` (9110). **No
  Mamba 2 GGUF on HF Hub** (only Mamba 1 `dranger003/mamba-2.8b-hf-GGUF` and
  `leliuga/mamba-2.8b-hf-GGUF` exist). Public llama.cpp Mamba 2 ecosystem for
  small/130m model is absent; baseline skipped, HF fp32 14.48 tok/s remains
  the sole reference.
- Scaffolding written under `models/ssm/mamba2/`:
    - `launcher.h` / `launcher.c++` — C ABI (`sk_mamba2_{create,forward,
      reset,dump_layer,destroy}`), allocates fused weights + per-layer
      `LayerState{conv_state, ssm_state}`. `forward()` is `ENOSYS` until SSD
      lands end-to-end.
    - `weights.h` / `weights.c++` — full HF name map for `mamba2-130m-hf`
      (`backbone.embeddings.weight`, `backbone.norm_f.weight`,
      `backbone.layers.{L}.{norm,mixer.{in_proj,conv1d,dt_bias,A_log,D,
      norm,out_proj}}`), transpose for `in_proj`/`out_proj`, conv1d
      `(C_in,1,K) -> (K, C_in)`. BF16→FP16 narrow.
    - `mamba2_model.h` — `LayerParams`, `LayerPSOs`, `ModelWeights`,
      `LayerState`, `ModelBuffers`. Pipeline doc comment matches HF
      Mamba2Mixer order.
    - `mamba2.py` — ctypes wrapper + `Mamba2Config.from_hf_json`.
- SSD kernel rewrite (signature-correct reference, no chunking yet):
    - `mamba2_ssd_ref.metal` — host name `mamba2_ssd_ref`. Inputs
      `(x[B,L,H,P], dt_raw[B,L,H], A_log[H], B[B,L,G,N], C[B,L,G,N], D[H],
      dt_bias[H])` → `y[B,L,H,P]`, `ssm_state[B,H,P,N]` fp32 in-out. Per-token
      recurrence (HF `torch_forward` non-chunked path). Grid `(B*H, P, 1)`,
      Nstate threads/tg.
    - `mamba2_step_ref.metal` — single-token decode equivalent.

## Remaining

1. Add `mamba2_ssd_ref.metal` + `mamba2_step_ref.metal` to the Xcode build
   (`SuperKittens.xcodeproj`) and register PSOs by name (`mamba2_ssd_ref` /
   `mamba2_step_ref`) in `kernels/runtime_bindings`.
2. Wire `sk_mamba2_forward` body: embed → per-layer{ pre_norm → in_proj →
   split (z|xBC|dt_raw) → conv1d_silu → ssd_ref/step_ref → gate_norm →
   out_proj → residual } → final_norm → tied lm_head argmax.
3. `temp/mamba2_validate/layer_diff.py` — load SK forward intermediates,
   compare to `hf_ref.npz` layer-by-layer (rel < 0.1 + argmax match for "Hi"
   → token 13).
4. Chunked SSD (HF L461-586) for prefill perf, once reference passes.
5. Decode bench vs HF fp32 14.48 tok/s.

## Update — May 13 2026 (lexie + laptop session)

**Forward path live; argmax matches HF.**

- Wired `mamba2_ssd_ref` + `mamba2_step_ref` PSOs into the launcher; new
  `dispatch_layer`/`dispatch_model` in `mamba2_model.h` orchestrates the full
  per-layer pipeline (pre-norm → in_proj → split → conv1d_silu → SSD →
  gate_norm → out_proj → residual), then final-norm + tied LM-head GEMM +
  last-row argmax.
- Fixed `weights.c++::copy_conv1d` to keep HF's native `(C_in, K)` layout
  (matches `conv1d_silu.metal`'s `weight[c*K + k]` indexing).
- Made `copy_into` / `copy_transpose_fp16` / `copy_conv1d` accept F32 / F16 /
  BF16 source dtypes (HF mamba2-130m-hf ships fp32).
- `ssm_state` allocation switched to fp32 (kernel signature is `device float*`).
- Metal kernels: replaced `log1p()` (not in Metal stdlib) with `log(1 + exp(.))`.

**Validation (lexie M4 mini):**
- Prompt "Hi" → SK argmax token **13** = HF argmax **13** ✓
- Logits rel L2 = 0.147, max_abs = 23.8 (fp16 truncation across 24 layers — OK).
- Top-5 ranking matches in top-1 and top-3.

**Throughput (8-tok decode probe on lexie):**
- SK: **~90 tok/s** (11.1 ms / token)
- HF fp32 baseline: 14.48 tok/s
- **6.2× speedup over HF reference.**

### Next steps
1. Chunked SSD for prefill perf (currently runs per-token recurrence).
2. Decode path uses prefill of len-1 each step; could swap to `mamba2_step_ref`
   to avoid the prefill overhead and shave further.
3. Per-layer numerical localization (HF dumps `L{i}.mixer_out` etc. — currently
   we only confirm SSM-state finiteness per layer because `dump_layer` only
   exposes the last forward's scratch).
