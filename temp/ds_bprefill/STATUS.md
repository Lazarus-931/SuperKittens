# DeepSeek-V2-Lite batched N-lane seq>1 chunked serving prefill

Branch: `dev-sk-ds-bprefill` (off local main @35cd485, which carries the
gemm_fp16 row-grid fix — single-stream T>1 prefill is correct there).
Host: amelia (M4 base, 16 GB, CLT-only; clang++ dylib + SK_METAL_SRC_FALLBACK
runtime compile). Lab: ~/sk-ds-bprefill-k9 (fresh, single-writer).
Model: DeepSeek-V2-Lite Q4_K_M (~10 GB), ONE handle per process, detached runs
under caffeinate -is + perl-alarm hard timeouts.

## Problem
Serving prefill drove prompts token-by-token at seq==1 per lane
(`generate_batched` -> `forward_batched`): weight-amortized across the N lanes
but paying T full weight-read passes for a T-token prompt. Single-stream T>1
prefill became correct today (row-grid fix); this branch adds the batched
(batch=N, seq>1) chunked-prefill ABI on top of it.

## Audit findings (batch>1 AND seq>1 simultaneously)
qwen's lesson held in reverse: deepseek's kernels were ALREADY batch x seq
ready — the gaps were launcher-level only.

Batch-ready as-is (verified in source):
- `kernel_mla_decode_v2` (mla_v2.metal): grid (seq, heads, batch); Q strides
  token-major with nb03=nb01*seq; K/V caches [batch, head, cache, dk|dv] via
  nb13/nb23 = nb12/nb22 * n_heads; dst rid = b*seq*H + s*H + h (token-major,
  matches attn_out_f32); mask indexed by q-row only (lanes lockstep, equal
  lengths -> shared mask is correct).
- `deepseek_mla_kv_write_b` (mla_glue.metal): b = t/seq, s = t%seq, per-lane
  cache region at b*H*cache_max*dim, pos = write_pos + s. Exactly the batched
  prefill contract.
- `rope_interleave_f32`: reads pos[i3*ne01 + i1] = pos[b*seq+s] for Q
  (ne01=seq, ne03=batch) and pos[r], r=b*seq+s for k_pe (ne01=batch*seq).
  Kernel correct — but the HOST fill in sk_deepseek_forward writes only seq
  entries (lane 0). This was gap #1.
- All dense projections: encode_proj routes T>1 to gemm_mma (grid
  (M+31)/32 rows — handles M=batch*seq); quant matvec loops M rows; gemm_fp16
  row grid is (M+31)/32 post-fix (the exact bug class flagged in the task: every
  GEMM dispatch this path touches was checked against its kernel's BM tile —
  gemm_fp16 BM=32 ✓, gemm_mma BM=32 ✓, kv_up_pair BM=32 with (T+31)/32 ✓,
  moe MMA BM=32 with mtiles=(T*top_k+31)/32 ✓).
- MoE at M=batch*seq: router grid (T,*); mul_mv_id grid z=T*top_k with
  dst [T, top_k, n_int] (the T>1 stride bugs were fixed in the earlier MoE
  campaign); moe_group_build counting-sort loops n_slots=T*top_k (no fixed cap
  beyond n_expert=64); MMA grouped path default-ON handles M=T*top_k tiles.
  Scratch (moe_gate/up/mid/down_f32, group_slots) all sized T_max =
  batch*seq_max at create.
- rmsnorm/add/add_rmsnorm/silu_mul/split_packed/embedding_lookup/causal_mask:
  flat-T or per-row; lane-agnostic.

Launcher-level gaps (the actual port):
1. rope_pos fill: batched seq>1 needs pos[b*seq+s] = current_pos+s for ALL
   lanes (forward fills lane 0's seq entries only; forward_batched fills one
   per lane at seq==1 — both special cases of the same pattern).
2. LM-head tail: the T>1 prefill tail projects only the GLOBAL last row
   (= lane N-1's last row); decode_all_rows would project all batch*seq rows
   (vocab-wide dead work) AND its argmax writes T entries into the batch-sized
   output_id (OOB). Added ModelParams.batched_prefill: 1 = interior chunk
   (skip final norm + head + argmax entirely), 2 = final chunk (per-lane
   last-row head matvec into logits row b + argmax over batch rows —
   crib of the qwen tail + the existing decode_all_rows argmax pattern).
3. New ABI `sk_deepseek_prefill_batched(h, ids[batch*seq] request-major, seq,
   chunk_size, out_next[batch])`: chunk loop (<= seq_max) carrying positions +
   KV across chunks; per chunk one M=batch*chunk pass.
4. Python `DeepSeek.prefill_batched(ids, chunk_size=)`, optional-symbol-gated.

Bans respected: zero changes to kernels/ (gemm_mma.metal untouched); all new
code gated on batched_prefill != 0 / the new ABI; decode + serving-decode
paths dispatch byte-identically (batched_prefill defaults 0 everywhere else).

## Files
- `SuperKittens/models/deepseek/deepseek_model.h` — ModelParams.batched_prefill
  + interior-chunk skip + final-chunk per-lane head/argmax tail.
- `SuperKittens/models/deepseek/launcher.c++` — sk_deepseek_prefill_batched
  (chunk loop, per-lane rope_pos fill, command-buffer error check).
- `SuperKittens/models/deepseek/launcher.h` — ABI decl.
- `SuperKittens/models/deepseek/deepseek.py` — prefill_batched wrapper.
- `temp/ds_bprefill/{gates_lane,gates_lane_t256,gate_singlestream,oldpath_single,oldpath_batched}.py`
  — gate/bench drivers (run on amelia in ~/sk-ds-bprefill-k9/lab; dylibs built
  there via build_both.sh: patched `build/` + pristine-main `build_main/`).
- `temp/ds_bprefill/artifacts/` — run-written JSONs (tokens, timings, gates).

## Gates
Protocol: amelia, ONE handle per process, processes sequential; N=8 DISTINCT
real-text prompts (paragraphs, doubled to clear T), T=128, 32-token lockstep
continuations; 2 warmups + 7 reps median, 0.3s gaps, A/B interleaved in the
SAME process/handle. A = token-by-token lockstep prefill (forward_batched
seq=1 x T); B = sk_deepseek_prefill_batched. Batched handle: batch=8,
seq_max=64, cache_max=192 (T=256 run: cache_max=320). colima VM (~4 GB)
resident throughout (the box's standing tenant); swap 2.2-2.8 GB, stable.

1. **Baseline coherence — PASS.** Patched build, batch=1: greedy 48-token
   generation fluent ("...Paris is a city that has something for everyone,
   from the Eiffel Tower to the Louvre Museum..."); all gate probes sane.

2. **Lane isolation — PASS (8/8 lanes).** Batched-prefill first tokens + full
   32-token lockstep continuations token-identical, per lane, to BOTH
   (a) the token-by-token lockstep baseline (same handle/process) and
   (b) the per-lane batch=1 single-stream T>1 prefill + greedy continuation
   (separate process; the now-correct post-row-grid-fix path) — and (a)==(b).
   batch=1 token-by-token spot checks (lanes 0,1) == the T>1 reference too.
   chunk=32 next tokens == chunk=64; T=64 single-chunk == 2x32-chunked
   (chunk-boundary positions/KV exact); identical-prompt lanes -> identical
   outputs; all tokens in-vocab. (artifacts/gates_lane.json, gates_single.json)

3. **Old paths byte-identical — PASS.** Pristine local-main dylib vs patched
   dylib (same host/env/inputs, separate processes): single-stream 48-token
   generate IDENTICAL; T=128 single-stream T>1 prefill + 32-token continuation
   IDENTICAL; batch=8 lockstep token-by-token prefill (T=64) + 32-step lockstep
   decode IDENTICAL (all 8 lanes). The new code is additive: batched_prefill
   defaults 0; grids of all pre-existing dispatches untouched.
   (artifacts/oldpath_{single,batched}_{MAIN,PATCHED}.json)

4. **Serving TTFT >= 8% — PASS (88.4% improvement, 8.63x).** Table below.

5. **Decode aggregate after prefill — PASS.** 32 lockstep decode steps after
   B-prefill vs after A-prefill: median ratio 0.998 (3 reps each side,
   T=128 run) — well within +/-2%.

## A/B: serving TTFT, N=8 lanes (median of 7, 2 warmups, same process/handle)
A = token-by-token lockstep (forward_batched seq=1 x T); B = prefill_batched.
TTFT = wall from reset to all-lanes-first-token.

| T | chunk | A (tbt lockstep) | B (batched seq>1) | speedup | improvement |
|---|---|---|---|---|---|
| 128 | 64 | 38360.5 ms | 4445.5 ms | 8.63x | 88.4% |
| 128 | 32 | 38360.5 ms | 4997.0 ms | 7.68x | 87.0% |
| 256 | 64 | 79249.3 ms | 9301.7 ms | 8.52x | 88.3% |

Rep spread tight: A within +0.2/+2.1% (one tail rep 39.2s), B64 within +-0.4%.
T=256 run (cache_max=320): lane match 8/8 (33 tokens vs tbt lockstep), decode32
ratio 0.9999, swap peaked ~2.8 GB / settled 2.1 GB (within budget).

Secondary (honest cross-process comparison): B64 = 4445.5 ms vs 8 x sequential
batch=1 single-stream T>1 prefills = 7954.4 ms (sum of 8, same patched build,
separate batch=1 process; first rep cold 2384 ms, steady ~795 ms/prompt) →
batched-lane prefill is still 1.79x faster than the best existing sequential
T>1 path, on top of being 8.63x over the actual serving baseline.

Memory-pressure caveat (same class as the qwen-8B note): with ~10 GB weights +
colima ~4 GB on a 16 GB box, the A side's 300 ms/lockstep-step implies weight
re-faulting every step, which amplifies A's cost; B reads weights 2 chunks x 1
pass. On a quiet box the 8.63x would compress toward the weight-read ratio; the
1.79x-vs-sequential-T>1 number is the conservative bound and is itself >> 8%.

## Verdict
**WIN — all five gates pass; serving TTFT improves 88.4% (8.63x) at N=8
T=128 chunk=64 and 88.3% (8.52x) at T=256 on DeepSeek-V2-Lite Q4_K_M
(amelia M4 base), with lane continuations token-identical to both the lockstep
token-by-token baseline and the single-stream T>1 references, old paths
byte-identical across builds, and decode aggregate unchanged (0.998 / 0.9999).
The conservative secondary bound — batched-lane prefill vs 8 sequential
single-stream T>1 prefills — is 1.79x, itself far above the 8% gate.**
