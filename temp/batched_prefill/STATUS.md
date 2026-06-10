# Batched (N-lane) seq>1 prefill for the qwen/dense shared core

Branch: `dev-sk-batched-prefill` (off origin/main @6931da9). Bench host: amelia
(M4 base, 16 GB, CLT-only; dylib clang++-built, kernels runtime-compiled via
SK_METAL_SRC_FALLBACK). NOTE: colima VM resident (~4 GB) during all runs.

## Problem
`sk_qwen_forward_batched` returned -6 for seq>1: the prefill transposes and the
LM-head tail only covered lane 0's rows, so serving callers drove prompts
token-by-token at seq==1 — weight-amortized across lanes but paying T full
weight-read passes for a T-token prompt.

## Design (additive; existing paths byte-identical)
The batch-indexing audit showed the kernels were ALREADY batch-correct:
- `mha_causal`/`mha_causal_prefill`/`mha_causal_q8` index Q/O at
  `(batch*nheads+head)*seq*D` → expect per-lane head-major `[B][H][seq][D]`.
- `kv_cache_write{,_q8}` read input as `(B,H_kv,seq,D)` and write per-lane
  cache slices `(B,H_kv,cache,D)`; grid z already spans `batch*H_kv`.
- `qwen_rope_qk` rotates row `r` at position `r % seq` (batch-aware since the
  lockstep-decode work) — correct for `[batch*seq,H,D]` at any seq.
- Projections/norms/splits operate on flat `T=batch*seq` rows (M-agnostic MMA).

What was actually broken: the four `(T,H,D)<->(H,T,D)` transposes were
dispatched with `T=p.seq`, covering only lane 0. Fix:

1. `qwen_model.h dispatch_layer`: when `batch>1 && seq>1` (previously
   unreachable), per-lane transpose loops (same byte offset `b*seq*H*D*2` for
   src and dst) so lanes stay independent `(H,seq,D)` slabs. Attention/KV-write
   dispatches unchanged — their grids already carried `z=batch`.
2. `ModelParams.batched_prefill` (default 0): `1` = interior chunk, skip final
   norm + head + argmax entirely; `2` = final chunk, project ONLY each lane's
   last prompt row (`b*seq+seq-1`) → logits row `b` (per-lane matvec: quant
   head via `quant_matvec_pso`, tied-fp16 via `gemv_t`), then per-lane argmax →
   `output_id[b]`. Avoids the decode_all_rows tail's vocab×batch×seq dead work
   and its seq==1 argmax indexing.
3. New ABI `sk_qwen_prefill_batched(h, ids[batch*seq] request-major, seq,
   chunk_size, out_next[batch])`: chunk loop (<= seq_max) carrying
   positions/KV across chunks; per chunk one M=batch*chunk GEMM pass.
4. Python: `DenseDecoder.prefill_batched(ids, chunk_size=)` —
   optional-symbol-gated (`hasattr`), raises cleanly on older dylibs.

Proven-dead respected: zero changes to `kernels/gemm/gemm_mma.metal`; no
KV-quant; no decode-path changes.

## Gates
1. **Lane isolation — PASS (8/8 lanes).** Qwen3-1.7B-Q8_0, N=8 distinct
   128-token prompts, chunk=64: every lane's greedy next token AND 32-token
   continuation after batched-chunked prefill match that lane's continuation
   from the baseline token-by-token lockstep prefill, token-for-token.
   Also: chunk=64 next tokens == single-chunk next tokens (8/8); identical
   prompts in all lanes -> identical outputs; all tokens in-vocab (argmax of
   non-finite logits would be degenerate; cross-checked below). BONUS: each
   batched lane's 32-token continuation is token-identical to the batch=1
   `sk_qwen_prefill_chunked` + decode reference for the same prompt — batched
   prefill reproduces the single-stream chunked-prefill path per lane exactly.
2. **Single-stream / lockstep byte-identity — PASS.** Baseline dylib (3 files
   at origin/main) vs patched dylib, same host/env: `generate` (48 greedy
   tokens), plain `forward`+16 decode steps, and 8× `prefill_chunked`+31
   decode continuations all token-identical across the two builds. (The new
   code is additive: `batched_prefill` defaults 0 and the per-lane transpose
   branch requires batch>1 && seq>1, previously unreachable — forward_batched
   returned -6.)
3. **TTFT >= 8% — PASS** (61.8–63.0% improvement, table below).

## A/B: serving TTFT, N=8 lanes (median of 7, 2 warmups, 0.3s gaps, same process/handle)
A = token-by-token lockstep prefill (`forward_batched` seq=1 ×T); B = `sk_qwen_prefill_batched`.
TTFT = wall from reset to all-lanes-first-token.

### Qwen3-1.7B-Q8_0 (amelia M4 base, seq_max=256, cache_max=512)
| T | chunk | A (tbt) | B (batched) | speedup | improvement |
|---|---|---|---|---|---|
| 128 | 256 (1 chunk) | 5781.6 ms | 2188.0 ms | 2.64x | 62.2% |
| 128 | 64            | 5781.4 ms | 2206.6 ms | 2.62x | 61.8% |
| 256 | 256 (1 chunk) | 11854.6 ms | 4402.0 ms | 2.69x | 62.9% |
| 256 | 64            | 12085.1 ms | 4478.2 ms | 2.70x | 63.0% |

Rep spread is tight (A within ±0.2%, B within ±0.3% for T=128; two A-side
outliers at T=256 chunk=0 — 12.7s/16.9s — from a brief concurrent gate-2 run,
median robust). Chunking at 64 costs ~1% vs one chunk — the chunk-boundary
overhead is small; attention-per-lane looping is a non-issue (attention is one
batch-grid dispatch, only the transposes loop per lane).

### Qwen3-8B-Q4_K_M (amelia M4 base, seq_max=128, cache_max=512; colima VM ~4 GB resident)
Clean re-run in a uniquely-named single-writer dir (`~/sk-bp-a7c31`) after the
shared-dir collision drained and the box was otherwise idle (78-82% memory
free at start). JSON provenance verified (`artifacts_priv/gates_8b_priv.json`).
**Gate 1 PASS (8/8 lanes: first token + 32-token continuation vs the
token-by-token baseline; chunk=64 == single-chunk; identical-prompt lanes
identical)** — exercises the Q4_K MMA body, the split-V Q6_K per-layer path,
and the untied Q6_K head via the per-lane matvec tail. T scoped to 128 only
(T=256 was attempted in the earlier tainted runs; the long protocol could not
be landed cleanly within the session — T=128 covers the headline).

| T | chunk | A (tbt) | B (batched) | speedup | improvement |
|---|---|---|---|---|---|
| 128 | 128 (1 chunk) | 37912.0 ms | 7314.5 ms | 5.18x | 80.7% |
| 128 | 64            | 37630.5 ms | 7243.0 ms | 5.20x | 80.8% |

Reps tight (A ±0.7%, B ±0.2%). A's 296 ms/lockstep-step implies partial mmap
weight eviction per step (4.7 GB weights + 4 GB colima on 16 GB); without
colima A would improve and the ratio would compress somewhat — on this host's
honest steady state the win is ~5.2x. An earlier tainted-shared-dir
observation (5.12x) independently reproduces it; an earlier two-8B-handle
private attempt was killed by me at swap 4.9/6 GB (hard-reboot risk) before
this clean window opened.

## Bench-host notes (honesty caveats)
- amelia ran a colima VM (~4 GB resident) throughout. For the 8B (4.7 GB mmap
  weights) this creates memory pressure (swap ~1.3 GB): the token-by-token A
  side re-faults evicted weight pages every step while B reads weights 1-2x,
  so pressure AMPLIFIES the 8B speedup vs. a quiet box. The 1.7B (2 GB) was
  fully resident; its 2.6-2.7x is the clean number.
- A TWIN concurrent agent session was given the same task and used the SAME
  `~/sk-batched-prefill` dir on amelia (the dir name came from the task spec):
  logs in the shared dir were doubly-written/truncated and its GPU runs
  overlapped mine. Provenance kept:
  * 1.7B numbers = `gates_17b.json`, written by MY traced PID (it held a
    log inode I had deleted — unique to my launch; args match my run exactly).
  * gate-2 comparisons ran inline in my own ssh sessions (stdout captured).
  * 8B runs in the shared dir were unattributable -> discarded; the 8B table
    below comes from a RE-RUN in a uniquely-named private dir
    (`~/sk-bp-a7c31`), single-writer.
  Twin GPU contention still affects absolute TTFTs at unknowable points; A
  and B run interleaved in the same process so contention hits both sides.
- amelia kept system-sleeping mid-bench unless an ACTIVE ssh session held
  `caffeinate` (detached caffeinate did not stick; the box's clock pauses).
  Python `perf_counter` (mach_absolute_time) excludes asleep time, so the
  measured reps stay valid; runs were driven under chained caffeinate ssh
  sessions.

## Verdict
**POSITIVE — all three gates pass; serving TTFT improves 61.8–63.0% (2.62–2.70x)
at N=8 on Qwen3-1.7B-Q8_0 (clean, provenance-verified), far above the 8%
gate; Qwen3-8B-Q4_K_M (clean private re-run): 5.18-5.20x (80.7-80.8%) at
T=128 — the win GROWS with model size as the weight-read amortization
predicts.** Lane continuations after batched prefill are token-identical both
to the token-by-token lockstep baseline AND to the per-prompt batch=1 chunked
single-stream reference; single-stream and lockstep paths are output-identical
between baseline and patched dylibs.

## Extension to other launchers (note only, not implemented)
- **gemma4**: `gemma4_model.h` already runs batch-aware rope/qkv-norm variants
  and dispatches attention with grid z=batch and NO seq<->head transposes, so
  the lane-transpose half of this fix has no direct analogue; the missing
  pieces are the chunked-prefill ABI + a per-lane last-row head/argmax tail
  (gemma adds final-logit softcapping + SWA kv addressing to audit). Similar
  shape, separate audit — not a copy-paste.
- **deepseek**: NOT trivially extensible — MLA prefill is decode-only today
  (`mla_decode_v2`; T>1 prefill hangs per models/deepseek status), so batched
  seq>1 prefill needs single-stream T>1 MLA prefill to exist first.

## Files
- `SuperKittens/models/qwen/qwen_model.h` — per-lane transposes (batch>1 &&
  seq>1), `ModelParams.batched_prefill`, lane-head/argmax tail.
- `SuperKittens/models/qwen/launcher.c++` — `run_step_prefill_batched`,
  `sk_qwen_prefill_batched` chunk loop.
- `SuperKittens/models/qwen/launcher.h` — ABI decl.
- `SuperKittens/models/dense/dense_decoder.py` — `prefill_batched()` wrapper
  (optional-symbol-gated).
- `temp/batched_prefill/{gates_ttft.py,single_stream.py}` — gate/bench
  drivers; `amelia/` — CLT-only build + runtime-compile env;
  `artifacts_shared_dir/` — run-written JSONs (1.7B gates+TTFT, gate-2 pair);
  `artifacts_priv/gates_8b_priv.json` — clean 8B gates+TTFT (private dir).
