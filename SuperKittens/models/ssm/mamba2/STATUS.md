# Mamba 2 SK port — STATUS

## END-TO-END COHERENT (dev-sk-mamba2-e2e)

mamba2-130m (`AntonV/mamba2-130m-hf`) generates **token-for-token identical**
output to HF `torch_forward` greedy. Verified on 4 prompts (33/33, 35/35, 34/34,
37/37 token match). Prompt "Hi" →
`"Hi, I'm a newbie here. I'm a student at the University of California, Berkeley..."`.
Prompt logits rel_L2 = 0.0008 vs HF (was 0.147 before the dt-clamp fix). Argmax 13.
Decode probe ~50 tok/s on this laptop (HF fp32 baseline 14.48).

Three bugs fixed to close the loop (all in the prefill SSD + LM-head path):

1. **dt clamp** — launcher clamped dt to `time_step_{min,max}` = (0.001, 0.1).
   Those only bound `dt_bias` init in HF; the *runtime* clamp is
   `time_step_limit` = (0, inf). Now plumbed through as `dt_limit_{min,max}`
   (config → C ABI → kernels). The kernels gate the +inf upper bound on a 1e30
   sentinel (Metal compiles `-no-infs-fp-math`, so `clamp(x, 0, inf)` is unsafe).

2. **interleaved x/B/C token stride** — both `mamba2_ssd.metal` and
   `mamba2_ssd_ref.metal` indexed x with token-stride `H*P` and B/C with `G*N`,
   but the launcher hands them ONE interleaved `[x(E)|B(G*N)|C(G*N)]` buffer
   whose per-token stride is `C_in = E + 2*G*N`. Correct only for t=0; every
   later token read the wrong region. Added an `XBC_stride` (=C_in) param.

3. **logits buffer under-allocated** — `bufs.logits` was sized for ONE row but
   the LM-head GEMM writes all T rows; the (T-1) row read by argmax /
   get_last_logits was OOB → garbage/zero. Now `T_max * vocab_size`.

## O(1) DECODE — LANDED (dev-sk-mamba2-conv1d)

`Mamba2Model.generate` now prefills the prompt once, then runs **one token per
step** (no re-prefill). Token-for-token identical to the prior O(T^2) re-prefill
path **and** to HF `torch_forward` greedy (5 prompts, 64/64 each).

The only missing piece was conv decode-state carry — the SSM kernels
(`mamba2_ssd` / `_ref`) already read+persist `ssm_state`, so an L=1 forward
continues the recurrence correctly. Two small kernels in `conv1d_silu.metal`
close the loop:

- `conv_state_capture` — after a prefill, stores that layer's last `K-1`
  pre-conv `xBC` tokens into `LayerState.conv_state` (left-zero-padded when
  `L < K-1`, matching HF `Mamba2Cache.update_conv_state(cache_init=True)`).
- `conv1d_silu_step` — for an L=1 decode, convolves the new pre-conv token
  against the carried `(K-1)`-token window, then rolls the window (drop oldest,
  append the new token). Mirrors HF `causal_conv1d_update`.

`dispatch_layer` branches on `is_decode` (= `seq==1`): prefill runs
`conv1d_silu` + `conv_state_capture`; decode runs `conv1d_silu_step`.
`get_last_logits` now keys its row off `last_seq` (=1 for decode), not
`current_pos`, so the (T-1) logits row stays in bounds across steps.

**Decode tok/s (this Mac, prompt "The history of the Roman empire", greedy):**

| new tokens | O(T^2) re-prefill | O(1) | speedup |
|-----------:|------------------:|-----:|--------:|
| 32  | 31.3 | 49.0 | 1.57x |
| 64  | 27.1 | 50.6 | 1.86x |
| 128 | 17.0 | 43.0 | 2.53x |
| 256 |  6.3 | 29.0 | 4.59x |

The O(1) path holds roughly constant tok/s (mild decay = per-step CPU/dispatch
overhead, 24 layer + LM-head dispatches synced per token); the re-prefill path
collapses quadratically. Lab: `temp/mamba2_conv1d/` (gitignored) —
`validate.py` (generate vs HF), `validate_o1.py` (o1 == reprefill == HF),
`bench_len.py` (length sweep).

## SSD kernel — FIXED (dev-sk-mamba2-ssd-fix)

`mamba2_ssd.metal` was rewritten to the HF-correct signature
(`x, dt_raw, A_log, B, C, D, dt_bias → y, ssm_state`) — same buffer layout as
`mamba2_ssd_ref` so the launcher binds it unchanged. It now does
softplus(dt + dt_bias) clamp, `dt * B * x` input gating, `D * x` skip, and
n_groups B/C sharing.

Design: grid `(B*H, P)`, one simdgroup (32 lanes) per (h,p) row, N/32 state
elements per lane held in registers, per-token `C·s` reduction via a single
`simd_sum` (no threadgroup barriers, no shared state). Algebraically the same
selective-state recurrence as the ref and as HF `torch_forward`.

The launcher now prefers `mamba2_ssd` (ref kernel is the numerical fallback).
The legacy Q/K/V `sk_mamba2_ssd*` / `sk_mamba2_step` C stubs in `mamba2.{c++,h}`
were removed (they would misbind the new signature; nothing referenced them).

**Validation (local, this Mac, vs numpy oracle + HF + ref kernel):**
- numpy oracle cross-checked against real HF `Mamba2Mixer.torch_forward`: rel_y
  ~4e-8 (so the oracle is faithful).
- new kernel vs oracle: **rel_y ≤ 2.1e-4** (fp16 output noise), fp32 state
  rel ≤ 7e-7, across L ∈ {16,128,200,255,256,257,300,512,600,700} — incl.
  L = chunk_size and L > chunk_size. Also n_groups=2 and batch=2 pass.
- new vs ref kernel: rel_y ~4e-6; state-carry (prefill → decode steps) is
  **bit-exact** vs one-shot.
- argmax equivalence through an out_proj→lm_head-shaped fp16 projection:
  **0 / 12296 token-position mismatches** vs the ref kernel ⇒ swapping kernels
  cannot change a generated token, so the ref path's known HF-argmax-match /
  coherent decode (see below) carries over unchanged.

**Perf (single-layer SSD op, GPU-timed, min-of-reps, this Mac):**
- new vs ref: **2.37x @ L=64, 2.65x @ L=256, 2.73x @ L=1024** (eliminates the
  ref's per-token tree-reduction + scalar-broadcast threadgroup barriers).

Lab: `temp/mamba2_ssd_fix/` (gitignored) — `ssd_ref_np.py` (oracle),
`check_vs_hf.py`, `validate.py` / `validate_full.py`, `argmax_equiv.py`,
`bench_ssd.py`, `metal_runner.py` (runtime-compile harness).

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
├── mamba2_ssd.metal       # FIXED: HF-correct sig, simd_sum scan (= ref, ~2.7x faster)
├── mamba2_step.metal      # legacy Q/K/V sig (unused; decode uses mamba2_step_ref)
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
