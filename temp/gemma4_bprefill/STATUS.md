# Batched (N-lane) seq>1 chunked prefill — gemma4_unified (12B)

Branch: `dev-sk-gemma-bprefill` (off local main @71df9ea, which carries the
qwen batched prefill). Bench host: derek (M4 base, 16 GB, CLT-only; dylib
clang++-built, kernels runtime-compiled via SK_METAL_SRC_FALLBACK; colima
default+k3s VMs resident throughout). Model: gemma-4-12B-it Q4_K_M GGUF,
SK_GEMMA4_BODY_Q4K=1 SK_GEMMA4_EMBED_Q8=1 (the PR #78 fit-16GB knobs).

## Problem
`sk_gemma4_forward_batched` rejects seq!=1 (-6): serving callers ingest
prompts token-by-token at seq==1 (weight-amortized across lanes but paying T
full weight-read passes for a T-token prompt). The qwen fix (PR'd earlier)
does one M=batch*chunk GEMM pass per chunk instead.

## Audit findings (what gemma4 needed vs qwen)
The qwen breakage was lane-0-only seq<->head transposes. gemma4 has NO such
transposes, and the batch-aware kernels added for lockstep decode (667c2eb)
turn out to be seq-generic already:

- `gemma4_qkv_norm_batched` maps flat row t -> lane b=t/seq, pos s=t%seq and
  writes (B,H,seq,D) — any seq.
- `rope_qk_bf16_batched` / `gemma4_rope_qk_partial_batched` rotate row r at
  write_pos + (r%seq) over the (B,H,seq,D) layout — any seq.
- `kv_cache_write_bf16` reads (B,H_kv,seq,D), writes per-lane ring slices
  (B,H_kv,csize,D) at (pos+t)%csize — any seq, any batch.
- `gemma4_attn_local_d256`/`gemma4_attn_global_d512` index Q at
  (batch*nheads+head)*seq*D, per-lane KV slice at
  (batch*n_kv_heads+kv_head)*csize*D, grid z=batch; the seq>1 "original path"
  handles causal masking per query row. O is written (B,seq,H*D) = flat
  request-major rows for the o_proj GEMM.
- Projections/norms/MLP/layer_scalar operate on flat T=batch*seq rows
  (M-agnostic `gemma4_gemm_mma_*` body GEMM at M>1).

What was actually missing (all additive, default-off):
1. A tail that doesn't project all batch*seq rows: the existing T>1 head
   either GEMMs all T rows (bf16 embed) or loops the quant matvec per row —
   vocab(262144) x T dead work — and `decode_all_rows` argmax requires seq==1.
2. A chunk-loop ABI carrying positions/KV across chunks.

### SWA audit (the flagged silent-corruption spot)
Per-lane KV slices + ring addressing are batch-correct at seq>1 (above). The
real SWA finding is an EXACTNESS BOUND inherited from the single-stream seq>1
path, not a batch bug:

- Local layers use a ring buffer of size `window` (csize==window), so the
  in-kernel window mask (`lower = upper>window ? upper-window : 0`) is dead
  code there — kv_len <= window always; SWA is enforced by ring EVICTION.
- When a whole chunk is written before attention reads (the dispatch order),
  the chunk's tail overwrites the oldest in-window keys of its own interior
  rows once current_pos+seq > window: query row r of the chunk loses
  (seq-1-r) of its oldest window keys. The mask accounts for exactly the
  retained keys (no garbage reads) — outputs are a slightly-shorter-window
  approximation, diverging from the token-by-token reference.
- Bound: chunked seq>1 prefill is token-EXACT iff current_pos+seq <= window
  (then nothing is evicted: kv_len = total tokens, lower=0, full causal
  visibility). Token-by-token (seq=1) is exact at ANY length — each step's
  single query row is the chunk's last row.
- This is a pre-existing property of `sk_gemma4_forward` at seq>1 (same
  write-then-attend order, same ring); batching does not widen it. Documented
  on the new ABI rather than "fixed": an exact >window chunked prefill needs
  attend-before-overwrite or scratch K/V — out of scope here.
- Global layers (every 6th, csize=cache_max) are exact for all T <= cache_max
  (enforced by the -4 bounds check).

## Row-grid audit (deepseek 31ff3c4 bug class) — 6 real sites, ALL DORMANT for 12B
`gemma4_model.h` had 8 dispatches dividing the row grid by 64. Audited each
against the kernel it actually binds (a /64 grid against a BM=32 kernel
silently skips rows 32-63 of every 64-block at M>32; M<=32 unaffected):

| site (pre-fix line) | binds | kernel BM | grid.y was | verdict |
|---|---|---|---|---|
| L432 QKV proj fallback | `gemm_bf16` (kernels/gemm/bf16/gemm.metal) | 32 | (M+63)/64 | **BUG** — only when `!use_q8` (bf16-weight fallback); not exercised with BODY_Q4K |
| L768 o_proj fallback | `gemm_bf16` | 32 | (M+63)/64 | **BUG** — same fallback-only |
| L908 fused MLP | `gated_mlp_bf16` | 64 | (M+63)/64 | correct |
| L1039 PLE gate | `gemm_bf16` | 32 | (M+63)/64 | **BUG** at T>32 — but PLE only |
| L1087 PLE proj | `gemma4_gemm_bf16_fp32_out` (ple_inject.metal) | 32 | (M+63)/64 | **BUG** at T>32 — PLE only |
| L1457 PLE ctx proj | `gemm_bf16` | 32 | (M_v+63)/64 | **BUG** at T>32 — PLE only |
| L1839 batched lane head | `gemm_bf16` | 32 | 1 (M=1) | correct |
| L1945 LM head bf16 T>1 | `gemm_bf16` | 32 | (M_v+63)/64 | **BUG** at T>32 — only when quant head absent AND w_embed present |
| enc_body (all body projs) | `gemma4_gemm_mma_*_t64n` / `_t32` | 64 / 32 | (M+BM-1)/BM, BM matched to variant | correct |

All six buggy sites fixed to `(M + 31) / 32` on this branch. **Decisive for
this model: NONE of them fire for gemma4-12b-unified** — `gemma4_unified.py`
sets `has_ple=False` (kills L1039/L1087/L1457), BODY_Q4K=1 routes projections
through the correctly-dispatched mma kernels (kills L432/L768), and
EMBED_Q8=1 frees w_embed so the T>1 head loops the quant matvec per row
(kills L1945). Verified empirically: the fixed build reproduces the pre-fix
gate-1 divergence pattern bit-for-bit (same 6/8 lanes, same indices) — the
12B lane divergence is NOT the row-grid bug. The fixes are kept as a latent
correctness repair for the PLE-bearing E-variants (E2B/E4B run T>32 prefill
through L1039/L1087/L1457 and silently lose PLE injection on rows 32+ of
every 64-row block) and for the bf16-weight fallbacks; M<=32 and decode
dispatch grids are unchanged by construction ((M+63)/64 == (M+31)/32 == 1).

## Design (mirrors the qwen shape; existing paths byte-identical)
1. `ModelParams.batched_prefill` (default 0): 1 = interior chunk — skip final
   norm + head + descale + softcap + argmax entirely; 2 = final chunk —
   project ONLY each lane's last prompt row (b*seq+seq-1) -> logits row b
   (per-lane matvec: Q4_K head via `q4k_matvec_bf16`, else Q8_0, else bf16
   GEMM M=1), then `gemma4_logit_descale` (1/sqrt(d_model)) +
   `gemma4_logit_softcap` (cap=30) over the batch*vocab lane rows, then
   per-lane argmax -> output_id[b]. Gemma's softcapped-logits semantics are
   preserved bit-for-bit with the decode tail (same kernels, same shapes).
2. New ABI `sk_gemma4_prefill_batched(h, ids[batch*seq] request-major, seq,
   chunk_size, out_next[batch])`: chunk loop (<= seq_max), positions derived
   from current_pos (gemma4 has no rope_pos buffer), KV carried across chunks.
3. Python `Gemma4.prefill_batched(ids, chunk_size=0)` — symbol-gated.

Bans respected: zero changes to kernels/gemm/gemm_mma.metal (the gemma body
GEMM is models/gemma/gemma4/gemma4_gemm_mma.metal, also untouched); decode and
M=1 paths byte-identical (batched_prefill defaults 0 everywhere).

## Gates
All runs: derek, one handle per process, detached + polled; batch=8,
seq_max=128, cache_max=512, window OVERRIDDEN 1024->512 for memory headroom
(KV ring halves; with every total ≤ 512 tokens nothing is ever evicted, so
compute/tokens are identical to window=1024 — by the ring analysis above).

1. **Baseline coherence — PASS.** Base dylib, batch=1, greedy 48 tokens on the
   chat-templated "Generate a poem about pizza dough": coherent verse
   ("A mound of flour, like fallen snow, / In a wooden bowl where the shadows
   flow...").
2. **Lane isolation — PASS with a documented numerical caveat (see below).**
   N=8 distinct prompts, T=128, chunk=64:
   - first token after batched prefill == token-by-token baseline: 8/8;
   - chunk=64 == single-chunk next tokens: 8/8;
   - identical prompts in all lanes -> identical outputs: yes;
   - bitwise lane-permutation test (same code path both runs): permuting the
     prompts permutes next tokens AND 8-token continuations exactly — no
     cross-lane leakage, no lane-slot dependence (PASS on random-token,
     raw-fluent, and chat-templated prompt sets);
   - 32-token continuation vs the tbt baseline depends on argmax sharpness
     (all on the post-row-grid-fix build):
     * random-token prompts: first tokens 8/8, continuations 2/8
       (gates_12b_fixed.json, window=1024) — A-side streams are degenerate
       newline/comma junk, knife-edge ties everywhere;
     * RAW fluent passages (no chat template, T=120): first tokens 8/8,
       continuations 3/8 (gate_fluent_r4.log) — the it-model greedy-parrots
       untemplated text into repetition loops ('. Today laser altimetry.
       Today.\n\n111...'), again knife-edge;
     * CHAT-TEMPLATED fluent passages (sharp argmax, the it-model's regime;
       full gemma chat turn "Continue this passage: ..." spliced to exactly
       T=128 with the model-turn cue intact, gate_chat.py): first tokens
       8/8; A-side continuations genuinely fluent prose on all 8 lanes;
       32-tok continuations 6/8 — lanes 1 and 5 flip mid-continuation
       (both sides fluent paraphrases of each other). Deterministic: a
       second run (gate_chat2.json) reproduces every token.
   - vs the SINGLE-STREAM seq>1 prefill reference (mission 3b; same q_seq>1
     attention path as the chunked prefill; seq_paths_chat.json c1 vs
     gate_chat2.json): per-lane FIRST TOKEN 8/8 exact; 32-tok continuations
     7/8 exact EVEN THOUGH the continuation engines differ (batched M=8
     lockstep decode vs batch=1 M=1 decode) — the single flip is lane 1, the
     same knife-edge prompt where the two EXISTING engines flip against each
     other (demo below). The batched chunked prefill tracks its own
     reference class (seq>1 prefill) more tightly than either tracks
     token-by-token.
     Root cause of the residual flips is PRE-EXISTING numerics, not the new
     code: the A side prefills through the attention q_seq==1 decode fast
     path (split-Bc + cross-simd merge), the B side through the q_seq>1
     original path — same online softmax, different reduction order, low-bit
     bf16 noise (see the base-dylib demo below). The earlier "real-text" run
     (gates_real.log, both sides DEGENERATE on diverging lanes) is explained:
     gate_real_prompts.py tiled each sentence x10 to reach T=128, and the
     12B-it model legitimately degenerates on 10x-repeated text — harness
     prompt construction, not a model or kernel bug.
3. **Old paths byte-identical — PASS.** Same host/env, base (branch-base
   71df9ea files) vs patched dylib: batch=1 chat coherence ids + greedy
   forward+16-step decode ids token-identical; batch=8 lockstep tbt T=32 +
   16-step continuation token-identical. Held on ALL THREE builds: pre-fix
   (r2 ls_base.json == ls_new.json), post-row-grid-fix (r3
   ls_base_r3.json == ls_new_r3.json), and the FINAL r4 build (ss_new_r4.json
   coherence_ids + decode_ids == r2 ss_base.json) — expected by construction:
   every dispatch the 12B config reaches has M<=32 at decode, where
   (M+63)/64 == (M+31)/32 == 1.
4. **TTFT >= 8% — PASS** (80.4-81.9% improvement, table below).
5. **Decode aggregate after prefill — PASS:** 10.918 tok/s (after A) vs
   10.918 tok/s (after B), ratio 1.0000 (within +-2%) — post-fix build,
   window=1024 (gates_12b_fixed.json).

### Base-dylib inherited-divergence demo (zero new code)
seq_paths_fluent.py / seq_paths_chat.py, batch=1, BASE dylib (71df9ea files):
for the same prompts, path1 = one forward(T) (the EXISTING validated
single-stream seq>1 prefill, q_seq>1 attention) vs path2 = T x forward(seq=1)
(decode fast path), 32 greedy continuations each. Results: RAW fluent
prompts (T=120): 4/8 match — 4/8 diverge, one (prompt 5) at continuation
index 0, i.e. the two EXISTING paths disagree on the very first generated
token. Chat-templated (T=128): 6/8 match — prompts {1,7} diverge (idx 14 /
deeper). Compare the batched gate: 3/8 (raw) and 6/8 (chat, lanes {1,5},
sharing prompt 1) — same regime-dependent rates, overlapping prompts. The
continuation-vs-tbt flip is a property of gemma4's seq>1-vs-seq=1 attention
numerics that predates and is untouched by the batched prefill — an 8/8
32-tok-continuation-vs-tbt gate is unattainable for ANY seq>1 prefill on this
model, including the production single-stream one.

SWA honesty (gate-2 clause): window=512 (overridden; production 1024) and
T=128 < window, so the SWA ring-wrap regime is UNEXERCISED by these gates —
deliberately: per the audit above, seq>1 chunked prefill beyond the window is
not token-exact BY CONSTRUCTION (pre-existing ring-eviction bound, documented
on the ABI), so an exercised-wrap equality gate would be vacuous. Callers must
keep context+prompt <= window for exact prefill (12B production window 1024).

## A/B: serving TTFT, N=8 lanes (median of 7, 2 warmups, 0.3s gaps, same process/handle)
A = token-by-token lockstep prefill (`forward_batched` seq=1 x T); B =
`sk_gemma4_prefill_batched`. TTFT = wall from reset to all-lanes-first-token.
gemma-4-12B-it Q4_K_M, derek M4 base 16 GB, colima VMs resident.

Final (post-row-grid-fix) build, window=1024 (production), gates_12b_fixed.json:

| T | chunk | A (tbt) | B (batched) | speedup | improvement |
|---|---|---|---|---|---|
| 128 | 128 (1 chunk) | 93360.3 ms | 18275.0 ms | 5.11x | 80.4% |
| 128 | 64            | 93360.3 ms | 16856.9 ms | 5.54x | 81.9% |

(The pre-fix r2 run at window=512 measured 5.13x/5.56x — the fix is
perf-neutral for 12B since none of the fixed sites fire in this config.)
T=256 not run: logits scratch is T_max*vocab(262144) bf16 — batch=8
seq_max=256 would add ~1.07 GB on a box already near its ceiling with colima
VMs resident (qwen-8B precedent: headline scoped to T=128).

chunk=64 measured ~7.6% faster than one 128-chunk (interior chunks attend to
shorter KV prefixes; the body GEMM is already weight-amortized at M=512).

## Verdict
**WIN — promote.** Perf: serving TTFT 5.11x (single chunk) / 5.54x (chunk=64)
at N=8 T=128, decode aggregate ratio 1.0000, old paths byte-identical on the
final build. Correctness: every check that isolates the NEW code is exact —
chunk-split == single-chunk 8/8; identical prompts -> identical lanes;
lane-permutation bitwise-exact (3 prompt regimes); deterministic across
re-runs; first token 8/8 vs BOTH references (token-by-token AND single-stream
forward(T)) in every regime; 32-tok continuations 7/8 vs the single-stream
seq>1 reference even across different decode engines.

The one gate that is NOT 8/8 — 32-tok continuations vs the token-by-token
reference (6/8 chat, 3/8 raw-fluent, 2/8 random) — is shown by the zero-new-
code base-dylib demo to be a pre-existing property of gemma4's seq>1-vs-seq=1
attention numerics: the EXISTING production single-stream prefill diverges
from token-by-token at the same rates on the same prompts (6/8 chat sharing
prompt 1; 4/8 raw, one at continuation index 0). No seq>1 prefill on this
model can pass that gate as stated; the batched prefill is exactly as faithful
as the validated single-stream prefill it generalizes. No evidence of any
lane defect remains; the row-grid audit's six /64-vs-BM=32 fixes are kept as
latent repairs for the PLE E-variants and bf16-weight fallbacks (none fire in
the 12B config).

NOT root-caused to a fixable site (and out of scope): which of the q_seq==1
split-Bc merge vs q_seq>1 sequential softmax orderings is "righter" — they are
both valid online-softmax reductions; making them bit-identical would mean
rewriting the decode fast path as a degenerate seq>1 tile pass and paying its
latency.

## Files
- `SuperKittens/models/gemma/gemma4/gemma4_model.h` — `ModelParams.batched_prefill`,
  interior-chunk early-out, final-chunk per-lane head/descale/softcap/argmax tail.
- `SuperKittens/models/gemma/gemma4/launcher.c++` — `sk_gemma4_prefill_batched`
  chunk loop.
- `SuperKittens/models/gemma/gemma4/launcher.h` — ABI decl.
- `SuperKittens/models/gemma/gemma4/gemma4.py` — `prefill_batched()` wrapper.
- `temp/gemma4_bprefill/{gates_ttft.py,single_stream.py,lockstep_identity.py}` —
  gate/bench drivers; `derek/{build_dylib.sh,skenv.sh}` — CLT-only build +
  runtime-compile env; `base_files/` — branch-base copies for the A/B dylib.
- `temp/gemma4_bprefill/{gate_chat.py,seq_paths_fluent.py,seq_paths_chat.py}` —
  chat-templated lane gate (exact-T template splice) + zero-new-code
  inherited-divergence demos; `artifacts/` — result JSONs (gate_chat2,
  gates_12b_fixed, seq_paths_fluent/chat, ss identity).
- derek lab dirs: ~/sk-gemma-bprefill-r2 (pre-fix gates), -r3 (post-fix
  gates/TTFT), -r4 (final build, fluent/chat gates, demos, ss identity).
