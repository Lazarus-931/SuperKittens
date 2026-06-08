# SuperKittens Deep-Redesign Campaign — PROGRESS

Single source of truth for the autonomous optimization campaign. Relentless mode:
chase improvements, iterate on wins, never stop until the user says stop.

## Operating model
- **Branches:** feature branches OK, push to remote OK. **NEVER touch `main` directly — main is PR-only.** Integration branch: `dev-sk-redesign`. Per-effort branches: `dev-sk-<area>`.
- **Agents:** each deep effort runs worktree-isolated. Build locally (only this Mac has `xcrun metal`), serialized under `lockf -t 600 /tmp/sk_build.lock` (flock absent on macOS; concurrent `xcrun metal` crashes the host). Bench on minis under `lockf -t 1200 /tmp/sk_bench.lock`.
- **Minis (Tailscale):** derek (`dereks-mac-mini` 100.64.169.42) UP; amelia (`amelias-mac-mini` 100.102.119.75) UP; lexie (`github-lexies-mac-mini`) DOWN (ssh timeout). minis are CommandLineTools-only → build .metallib/.dylib locally, copy over.
- **Bench protocol:** prompt "Generate a poem about pizza dough"; 2 warmup + 30s cooldown + 5-rep median; same host/build A/B; report tok/s + coherence + GB/s where relevant. Compare own same-config Q4_K_M baseline (tok/s is cache_max-dependent).
- **Validated win → open a PR to main** (do not merge without orchestrator/user review). Negatives get logged here, not integrated.
- **PR descriptions: SHORT + human-readable** (a few lines: what changed, the key numbers, validation). NOT essays.
- **development/PROGRESS.md is LOCAL-ONLY — do NOT push it** (user directive 2026-06-08). Commit locally for durability; never `git reset --hard` when it has uncommitted edits. Work branches + PRs still push.
- **No Co-Authored-By trailers** (classifier blocks them). WHY-only comments.

## Current decode baselines (Q4_K_M, M4 base, cache_max-dependent)
4B 38.48 · 8B 21.09 · 14B 11.28 tok/s. Decode sits ~71% of M4 roofline; kernels hit 78–95% BW → ~20–25% is inter-dispatch overhead.

## Settled findings (do not re-attempt)
- **Q2_K @4B**: net loss 0.68× + garbage coherence + Q3_K-tensor host-dequant byte-explosion.
- **Q3_K decode @4B/8B/14B**: 0.77×/0.91×/0.62× — q3k matvec is COMPUTE-bound (30–56 GB/s vs q4k 106–110); q5k also (52). Coherence clean; speed regresses. **Q4_K_M is the M4 decode sweet spot.** q3k/q5k kernels kept as FIT enablers only.
- **Spec-decode** regresses (0.54×) without batched GEMM — verify forward re-pays per-token compute.
- **mha_causal chunked-forward fix** merged (PR #54, main 49f9174).
- **Decode is GPU-BW-bound (~99% GPU-wait); CPU-encode is 0.4-0.7%.** ICB cut CPU-encode 4.5-5.2× → tok/s FLAT. The whole CPU/dispatch/fusion class (§A #1-7) is DEAD for decode tok/s. Decode lever = fewer bytes/token only (Q4_K_M sweet spot). Real frontiers: prefill/TTFT, multi-stream throughput, breadth.
- **Q6_K-LM-head routing** win merged (PR #53): 4B/8B/14B +8.5–15.4%.

## Status legend
`TODO` · `WIP <agentid>` · `BENCH` (validated, PR pending) · `WIN` (PR open/merged) · `NEG` (negative, logged) · `BLOCKED`

## Backlog (prioritized)

### A. Pipelining / inter-dispatch overhead (the ~20–25% gap)
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 1 | Fuse qwen q-norm+k-norm+rope-q+rope-k → 1 T1 kernel (precedent: gemma4_qkv_norm_rope_partial_t1) | HIGH | **NEG** | caps ~2-3% (measured 0.6%); per-layer encoder already shared + buffer barriers cheap → not worth it |
| 2 | Inter-dispatch fence audit (qwen decode) — find more redundant barriers | HIGH | TODO | +90% win was this class |
| 3 | ICB / multi-layer command-buffer batching — measure the "expected but unmeasured" ICB win | HIGH | **NEG (decode)** | byte-identical, CPU-encode −4.5-5.2× but tok/s FLAT (decode ~99% GPU-wait). Recorder kept for T>1 prefill/verify (dev-sk-icb-batch@750783e) |
| 4 | Megakernel: fuse whole decode layer into one persistent kernel | MED | TODO | ambitious |
| 5 | Fuse split_packed (2 dispatches) into QKV matvec epilogue | MED | TODO | |
| 6 | Fuse kv_cache_write into rope-k kernel | MED | TODO | |
| 7 | Fold final RMSNorm into last layer encoder | LOW | TODO | |

### B. Batched GEMM / seq>1 (the big architectural lever)
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 8 | simdgroup_matrix (MMA 8×8) batched GEMM for seq>1 | HIGHEST | **WIN PR#55** | gemm_mma f16/q8_0/q4k; prefill 1.5-2.6× (Q8_0), sub-linear in seq; 72/72 numerics; decode unchanged (M==1→matvec) |
| 9 | Prefill attention path T>1 (qwen + MLA) | HIGH | TODO | DeepSeek T>1 hangs today |
| 10 | Spec-decode e2e (draft+target) after #8 | HIGH | **NEG (final)** | re-benched on small-M kernel: 0.36-0.64×→0.56-0.88×, still <1.0. Verify floor fixed but DRAFT TAX (K×6.92ms) + accept ceiling (3.85) now dominate. Settled-NEG on M4 4B+0.6B |
| 42 | Reduce gemm_mma small-M (seq 2-8) fixed cost | HIGH | **WIN PR#59→merged dev-sk-batched-gemm** | gemm_mma_smallm bit-exact (diff=0.0); verify floor 4.6-5.8× → seq2 1.0-1.1×, seq4 1.3-1.5×, seq8 2.3-2.8× (Q8_0); M-threshold dispatch. UNBLOCKS spec-decode re-bench |
| 11 | Chunked prefill using fixed mha_causal (PR #54) | MED | TODO | |
| 8b | gemm_mma ext: Q6_K+BF16 loaders + prefill LM-head=last-row (fixes OOB write) | HIGH | **WIN PR#56→merged into dev-sk-batched-gemm** | TTFT q8_0 2.53×/q4k 5.81× (T=200); 5/5 dtypes ≤3e-4 |
| 41 | Latent: get_last_logits uses absolute current_pos-1 vs head step-local row idx | MED | **WIN PR#64** | fixed 2 LIVE main bugs: prefill OOB+wrong-first-token (1479 vs 264) + get_last_logits stale row. Validated 0.6b |

### C. DeepSeek V2-Lite (stated focus)
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 12 | Fix kv_b de-interleave numerics (vs HF ref) | HIGH | TODO | dev-deepseek-e2e incoherent |
| 13 | Fix RoPE interleave pairing vs HF | HIGH | TODO | |
| 14 | Localize GPU OOB (Metal validation layer) → fix 30–40s/tok | HIGH | TODO | |
| 15 | Promote Q4_K MoE in-tree (~4× proven) | MED | TODO | |
| 16 | DeepSeek prefill path | MED | TODO | depends #9 |

### D. KV-cache
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 17 | KV-cache quantization (Q8_0 K/V) | HIGH | **WIN-mem PR#57** | decode parity (not a speed win — weight-bound), but ~2× cache_max on 14B; opt-in SK_KV_Q8, default byte-identical |
| 18 | KV-cache Q4 + per-channel scales | MED | TODO | |
| 19 | Paged/ring KV cache for long context | MED | TODO | |

### E. Quant kernels / fit
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 20 | Consolidate 3 q3k_matvec ports → 1 canonical (fit enabler) | MED | **WIN PR#63 (open, review)** | + q5k; lane=pos, byte-assembly UB fix, full Q3_K/Q5_K plumbing; median_rel 1.7e-4 |
| 21 | Canonicalize q5k_matvec (fit enabler) | MED | **WIN PR#63** | folded into #20 |
| 22 | Q3_K SoA-repack (4-aligned, vectorized) — parity experiment | LOW | TODO | best-case ~parity |
| 23 | IQ2_XXS matvec for extreme fit | LOW | TODO | |

### F. SSM
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 24 | mamba2_ssd.metal numeric fix (softplus/dt·B·x/D·x/n_groups) | MED | **WIN PR#60 (open, review)** | full rewrite, matches HF ref ≤2.1e-4, argmax-equiv 0/12296, 2.37-2.73× over ref-fallback; launcher prefers it. dt-clamp(0.001,0.1)-vs-HF(0,inf) flagged follow-up |
| 43 | mamba2 conv1d O(1) decode-state-carry (replace O(T²) re-prefill) | MED | **WIN PR#66** | conv_state_capture + conv1d_silu_step; O(1) decode token-id to HF; 1.57× @32tok → 4.59× @256tok |
| 25 | mamba3 seqlen>1024 hang fix | LOW | **WIN PR#61 (open, review)** | "hang" was H100 codebase (red herring); found+fixed REAL bug — intra-chunk Q@K^T scores wrong all seqlens (acc mis-read, 4/16 tiles, tail drop). Now ≤1e-3 to L=32768 |

### G. Structure / design / organizing
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 26 | Unify per-family launcher pattern (shared encode_* lib) | MED | TODO | qwen/gemma/deepseek |
| 27 | Centralize encode_quant_gemm PSO-dispatch across families | MED | TODO | |
| 28 | docs/kernels.md architecture index refresh | LOW | **WIN PR#65** | + CLAUDE.md open-state; corrected stale gemm/conv1d/delta_net/mamba2 claims |
| 29 | best.md scoreboard refresh | LOW | **WIN PR#65** | folded into #28 |
| 30 | registry.py dtype-aware variant auto-derivation | LOW | TODO | |
| 31 | Unify mixed-quant detection across families (loader) | MED | TODO | |
| 32 | Generalize ICB recorder across families | MED | TODO | depends #3 |

### H. Sampling / head / norm / misc
| # | Item | Pri | Status | Notes |
|---|------|-----|--------|-------|
| 33 | Fused argmax+sampling on-device (extend ICB) | LOW | TODO | |
| 34 | LM-head split-K for huge vocab | MED | TODO | |
| 35 | rmsnorm_t1 fusion variants for more dtypes | LOW | TODO | |
| 36 | RoPE table precompute/caching audit | LOW | TODO | |
| 37 | memory_aware_cache_max per-model headroom refinement | LOW | TODO | |
| 38 | fp16-fallback elimination audit (anywhere host-dequant streams 16bpw) | **HIGH** | WIP a-ds-fit | THE DeepSeek-coherence blocker: keep DeepSeek dense/attn/embed/head QUANTIZED + native matvec so it fits 16GB |
| 39 | split-K decode gating tuning (extend the landed win) | LOW | TODO | |
| 40 | GQA amortization audit (mha_causal) — guard against regressions | LOW | TODO | |

## Log (newest first)
- **[CRASH RECOVERY — computer crashed, 4 agents lost]** Preserved+pushed: dev-sk-deepseek-fit@6e014ec (DeepSeek coherence-debug mid-RoPE, had debug hooks + was checking poem2.log), dev-sk-chunked-prefill@1fd3023. gemma4 worktree gone (no commits — deprioritized). Q4-KV was on WRONG base (7bc4325, 277 dirty) — discarded, relaunched fresh off dev-sk-kv-quant. Relaunched 3: adee3629daeabe647 (DeepSeek-coher resume = headline), a79f5ab1e0e7b8e68 (chunked-prefill resume), aa152956cbb0112b8 (Q4-KV fresh). All agents now told to update temp/STATUS.md after EACH milestone (crashes recur → cheap resume).
- **[#43 mamba2 conv1d O(1) decode = WIN (PR #66)]** `af2aa287190300c55`. Added conv_state_capture + conv1d_silu_step (carry conv decode-state across L=1); dispatch_layer branches on is_decode; get_last_logits keys off last_seq. O(1) decode token-identical to re-prefill AND HF (64/64 ×5 prompts + edge cases). **Decode 1.57× @32tok → 4.59× @256tok** (O(1) holds constant, O(T²) re-prefill collapses). PR #66 (base dev-sk-mamba2-e2e, stacks #62). mamba2 now COMPLETE: #60 ssd-fix → #62 coherent → #66 O(1) decode. Replenishing with #18 Q4-KV.
- **[#28/#29 docs refresh = WIN (PR #65)]** `a4a2f68ab45299f9f`. docs/kernels.md + best.md + CLAUDE.md open-state refreshed (all new kernels tagged [PR #N]/not-on-main; settled findings: decode BW-wall, Q4_K_M sweet spot, spec-decode dead, KV-quant=memory). Corrected stale claims: gemm/ dir listing (ships q8/q4k/q6k/q2k/iq2/swiglu, not just fp16/fp8), conv1d_silu is in fusion/ not conv/, undocumented delta_net/ scaffold, stale mamba2-broken note. Docs-only, PR #65→main. Replenishing with gemma4-e2e (verify repo exists first).
- **[spec-decode 8B = NEG (0.860×) — definitively CLOSED]** `a6b5c3015973b7e53`. 8B target refuted the ~1.1-1.3× projection: best 0.860× (low-entropy K=8), lossless. Draft tax DID halve (15% of 8B decode) but break-even accept(K=6)=3.44 not 2.84 because the verify forward stays **2.5-3.5× a decode** (multi-row verify dequant/compute-bound, doesn't collapse to ~1×) + all-rows-head + Python-loop overhead. Spec-decode loses at BOTH 4B (0.88×) and 8B (0.86×) → SETTLED-NEG on M4, bigger targets are NOT the lever. Branch dev-sk-spec-decode-8b pushed, no PR. (Infra win: downloaded 8B GGUF directly on lexie via curl 80MB/s vs 40MB/min inter-host — use this pattern.) Replenishing with #11 chunked-prefill.
- **[#41 logits-fix = WIN (PR #64) — fixed 2 LIVE bugs on main]** `ae159d96bed2a8db5`. Bug1: qwen prefill projected all T rows, argmax wrote output_id[0..T-1] into a batch=1 buffer → OOB write + greedy read row0 = argmax of FIRST prompt token not last (empirically 1479 vs correct 264); the #56 last-row fix was only on the gemm branch, never on main. Bug2 (#41): get_last_logits read absolute current_pos-1 vs the head's step-local row → stale after T=1 decode / 2nd prefill chunk. Fix: prefill projects only row T-1; get_last_logits reads logits[last_seq-1]. Validated on 0.6b: prefill-T>1 first token 264==264, get_last_logits correct after prefill+chunked, greedy C decode byte-identical. PR #64→main. **NOTE: T>1 prefill generation on main returns the WRONG first token — real correctness bug, fixed here.** Replenishing with #28/#29 docs refresh (local).
- **[#20/#21 q3k/q5k canonical = WIN (PR #63, housekeeping)]** `a5caf67c702ab063d`. One canonical q3k_matvec + q5k_matvec in kernels/gemm/ (lane=pos, matches q6k sibling) + byte-assembly scale-unpack (UB fix: scales @offset96/110B-stride only 2-byte-aligned) + full Dtype::Q3_K/Q5_K plumbing (weight_store/gguf codes 11,13/qwen_model/launcher/weight_loader.py) — none was on main. De-duped 3 divergent q3k ports. Numerics median_rel 1.7e-4, cosine 1.0 (incl 14B dims). PR #63→main (open, review). Replenishing with #41 logits-OOB audit (local).
- **[spec-decode formal NEG confirmed; 8B win-test launched]** `a4da0c` completed: 4B+0.6B = 0.56-0.88× (best 0.882× low-entropy K=6), lossless. Added `lm_head_all_rows` ABI (PR#56 last-row broke spec-verify per-position logits). **Lead: 8B target projected ~1.1-1.3× at low entropy** (break-even accept 2.84 < measured 3.85) — launched spec-8B test `a6b5c3015973b7e53` (dev-sk-spec-decode-8b, download models directly on target mini). ROOT CAUSE of branch-flips CONFIRMED: this agent's `git reset` hit the shared checkout → "stay in own worktree" is THE fix (in all prompts now).
- **[mamba2-130m e2e = WIN (PR #62); spec-decode re-bench = NEG]**
  - **mamba2-130m COHERENT** `a09a38f1439ff6bbb`. dt-clamp was the dominant bug (config time_step_{min,max} bound dt_bias INIT only; HF runtime clamp = time_step_limit (0,inf)) → re-plumbed dt_limit_{min,max}, +inf via 1e30 sentinel (Metal -no-infs-fp-math). **Logits rel_L2 0.147→0.0008 vs HF.** +2 more bugs: interleaved x/B/C token-stride (wrong region for t>0; added XBC_stride) and logits-buffer OOB (sized 1 row, GEMM writes T → T_max*vocab). Generation token-identical to HF (33/33,35/35,34/34,37/37), ~50 tok/s. PR #62 (→main, stacks on #60). Follow-up: conv1d_silu has no decode-state carry → single-step decode O(T²) re-prefill workaround; O(1) fix = thread conv_state + route L=1 to mamba2_step_ref.
  - **spec-decode RE-BENCH on small-M = NEG (0.56-0.88×)** `a4da0cebc0d307a45`. Verify floor cut (0.36-0.64×→0.56-0.88×) but still <1.0: bottleneck moved to DRAFT TAX (K×6.92ms for 0.6B) + accept ceiling (max 3.85 < break-even). Settled-NEG on M4 for 4B+0.6B; needs a far cheaper draft. dev-sk-spec-decode-smallm@d7dee8b, no PR.
  - RECURRING BUG CLASS noted: "logits buffer sized 1 row but LM-head GEMM writes T rows" hit in #8b (qwen), mamba2 (#62), and is #41 (qwen get_last_logits). Audit all families.
- **[DeepSeek FIT solved; coherence = separate decode-dynamics bug]** `a7caf49dd596e824c` done. Quantized loader (keep q/kv_a/o/shared/LM-head Q4_K/Q6_K, no dequant) → **resident 10654MB (was OOM >16GB), FITS** with KV headroom; loader numerically faithful 4.98e-4 vs fp16; runs 5.36 tok/s, ttft 32.7s. Fixed moe_quant=2 config bug (was layer-1 OOB). **But full-model output still INCOHERENT** — plausible start then collapse to newline/space/period. Divergence analysis: loader/matvec swap is faithful → the collapse is a PRE-EXISTING decode-dynamics bug (RoPE-interleave across positions / MoE routing / KV), NOT this change; single-layer MLA was 6e-4 but multi-layer/token compounds. Branch dev-sk-deepseek-fit@73dbad5, no PR (gate=coherent). HOST NOTE: derek was saturated by a lima/Virtualization VM (swap 16GB full) — A/B couldn't finish; check host health. Launching DeepSeek full-model coherence debug (per-layer+per-token vs HF).
- **[#25 mamba3 = WIN (PR #61, open for review)]** `a7bc4c15ba2379e50`. "hang>1024" was the H100 codebase (red herring; Metal runs to 32K). REAL bug: intra-chunk Q@K^T scores wrong ALL seqlens (acc re-read via thread_elements()[], 4/16 tiles at CS=32, tail drop). Fixed (register acc + per-simdgroup tile stride + ceil8 tail). ≤1e-3 to L=32768. Follow-up: MLX baseline not a valid oracle. Launched mamba2-130m e2e closure (real generate + dt-clamp fix) a09a38f1439ff6bbb. NOTE: a `git reset --hard` to self-heal a branch-flip wiped uncommitted edits this turn — ONLY reset when branch is actually wrong AND no uncommitted edits.
- **[#24 mamba2_ssd = WIN (PR #60, open for review)]** `a678f9656535bf38b` done. Full rewrite of the KNOWN-BROKEN kernel → matches HF transformers ref ≤2.1e-4 (L∈{16..700}, n_groups=2, batch=2), argmax-equiv 0/12296, prefill→decode state bit-exact. **2.37× @L=64 → 2.73× @L=1024** over the slow ref-fallback; launcher now prefers it (ref = fallback). PR #60 → main MERGEABLE but auto-merge to main was correctly BLOCKED by the classifier (PR-only/don't-touch-main) → held for user review alongside #55 (batched GEMM) + #57 (KV-quant). Note: dt-clamp(0.001,0.1) vs HF(0,inf) is a flagged orthogonal follow-up. Replenishing with #25 mamba3-hang. NOTE: agent git-op flipped local dev-sk-redesign ref to 2fca67f again — reset to origin (d735700); the "stay in own worktree" violation keeps recurring.
- **[#42 small-M = WIN → PR#59 merged into dev-sk-batched-gemm]** `a0fec52cfa0ad30aa` done. gemm_mma_smallm Q8_0/Q4_K **bit-exact (diff=0.0)** vs matvec M∈{1,2,4,8}. Verify floor (vs M=1 decode): **seq2 4.6-5.8×→1.0-1.1×, seq4→1.3-1.5×, seq8→2.3-2.8× (Q8_0)**. M-threshold dispatch (sm 2≤M≤8 Q8_0 / 2≤M≤4 Q4_K; M=1 matvec; M>8 MMA) wired on all 6 projections + fixed dead PSOs. dev-sk-batched-gemm now = #8+#8b+#42 (PR #55→main carries all). **Break-even mean-accept now below K → spec-decode should flip to a win.** Launching the e2e spec-decode RE-BENCH on this kernel (the agent left it to avoid entangling PR lines).
- **[wave-5 — KV-quant done; process-restart recovery; small-M win-in-progress]**
  - **#17 KV-quant Q8_0 = MEMORY WIN (PR #57), not a speed win.** `a5d0b0494b2c34187` done. Q8_0 K/V (kv_cache_write_q8 + mha_causal_q8 + mha_decode_split_q8, gated SK_KV_Q8, fp16 fallback). Numerics ≤1e-2 vs fp16 across kv∈{128..4096}/GQA/prefill; coherence identical. Bench (lexie 4B): decode **parity** at short AND long ctx (kv=2048/4096) — NO BW win because decode is weight-bound, KV is a small slice (confirms the BW-wall finding). **But ~1.9-2.0× larger cache_max on 14B** (memory feature). Opt-in, default byte-identical. PR #57 open (NOT merged).
  - **Process restart killed DeepSeek-fit + gemm_mma-small-M.** Both PRESERVED+pushed: dev-sk-deepseek-fit@f698308 (quantized-loader fix mid-impl, 4 deepseek files dirty), dev-sk-gemm-mma-smallm@1ad9709. **small-M had GOOD progress: gemm_mma_smallm.metal cut the floor — seq2-4 ~1.1-2× seq1 (GOAL MET), seq8 ~3× (was 4.6-5.8×)** → should resurrect spec-decode. Relaunched both: a7caf49dd596e824c (DeepSeek-fit), a0fec52cfa0ad30aa (small-M). +launched #24 mamba2-ssd-fix a678f9656535bf38b (local correctness).
- **[#3 ICB = NEG for decode; STRATEGIC finding]** `a104b1dcefb1505af` done. ICB-replay byte-identical (4B/8B/0.6B token-ids + fp16 logits) and cut **CPU-encode 4.5-5.2× (180→40µs)** — but tok/s FLAT (0.994×/1.011×): decode is **~99% GPU-wait, bandwidth-bound** (reads full weights/token @~120GB/s); CPU-encode is only 0.41-0.69%. Async-commit can't help (autoregressive data dependency = GPU is the serializing floor; confirmed empirically). **CONCLUSION: the whole CPU/dispatch/fusion lever class (section A: #1,#3,#2,#4-7) is DEAD for decode tok/s — single-stream M4 decode is at the GPU-BW wall.** Recorder kept on dev-sk-icb-batch@750783e for T>1 prefill/spec-verify (its real payoff). No PR. Frontier now = prefill/TTFT (done #8/#8b), throughput (spec-decode, blocked by #42 small-M floor), breadth (DeepSeek).
- **[#10 spec-decode = NEG (lossless but 0.36-0.64×)]** `a2794ddadeb89b75b` done (on lexie — amelia went OOM-tight ~58MB, derek load 8; **lexie is BACK UP ~7.3GB free, has both GGUFs**). Lossless PASS all 6 (greedy spec == target greedy). Best 0.644× (low-entropy K=6), still <1.0. ROOT CAUSE: target verify forward at seq=K+1 costs **4.6-5.8× a seq=1 decode** — gemm_mma has a large FIXED-COST FLOOR at small M; #55's win was at seq=128, but spec-decode's seq≤7 verifies are cheaper as per-row matvec. Break-even mean-accept exceeds K at every K → unreachable until the kernel floor drops. Pushed dev-sk-spec-decode a21fd25..f722ab7, NO PR. **New lever (backlog #42): reduce gemm_mma small-M (seq 2-8) fixed cost** → would resurrect spec-decode + help short prefill.
- **[DeepSeek: numerics FIXED, coherence BLOCKED on fp16-dequant OOM]** `a25cf46220d96f4cb` done. Single-layer MLA rel err ~6e-4 (validated, all fixes on dev-sk-deepseek-fix@9e2f74f). BUT every 16GB mini OOM-killed the run: the DeepSeek loader **host-dequantizes dense/attn/embed/LM-head to fp16** (only routed MoE stays Q4_K/Q8_0) → resident ≫ 10.4GB GGUF → swap 100%, jetsam SIGKILL @ layer 26/27. cache_max NOT the lever (KV ≈140MB). **Same host-dequant trap the Q6_K-head fix solved for qwen.** No PR (gate=coherent poem). Anomaly: token0 argmax differed by cache_max (likely paged-out mmap weights mid-dispatch under pressure, not a stride bug; recheck on a host where it fits). **UNBLOCK = keep DeepSeek dense/attn/embed/head QUANTIZED + native matvec (= item #38 applied to DeepSeek / #15).** Launching that as the redirect.
- **[wave-3 RECOVERY — process restart killed 3 agents; all committed first]** Orchestrator process restarted → DeepSeek/spec-decode/ICB agents lost in-process state, but ALL had committed + are at "locally-validated, needs final mini bench" stage. Preserved to remote: dev-sk-deepseek-fix@9e2f74f (**numerics FIXED: attn_out/o_proj rel err ~6e-4, was 0.4-0.8 — target met**; needs coherence generate on derek), dev-sk-spec-decode@a21fd25 (**local correctness GREEN, lossless+verify-logits fixed**; needs host bench vs old 0.54×), dev-sk-icb-batch@750783e (**byte-identical ICB-replay done**; needs host tok/s bench, may need async-commit to convert CPU-encode saving). Relaunched hardened (final-bench-focused): DeepSeek a25cf46220d96f4cb (derek), spec-decode a2794ddadeb89b75b (amelia), ICB a104b1dcefb1505af (amelia). New agent output dir: .../f80f6186-b09a-4083-b92c-8f14df16587e/tasks/.
- **[#8b WIN — gemm_mma extended + prefill productionized]** `a724311a0b40baf55` (HARDENED — no watchdog death). Q6_K+BF16 MMA loaders (5/5 dtypes ≤3e-4) + prefill LM-head=last-row-only (fixes OOB write of output_id[1..T-1] past batch=1; drops T-1 dead vocab projections). **TTFT T=200: q8_0 1791→708ms (2.53×), q4k 3686→634ms (5.81×).** PR #56 merged into dev-sk-batched-gemm → PR #55 now carries full batched-GEMM package. NOTE: an agent git-op flipped the MAIN tree to dev-sk-gemm-mma-ext-work@bed017d — restored to dev-sk-redesign (worktree isolation leak; watch for recurrence). Replenishing with #3 ICB.
- **[POST-MORTEM — wave-1 ALL 6 watchdog-killed]** Root cause: 6 agents each doing SYNCHRONOUS blocking-SSH e2e benches on 2 OOM-tight minis (derek ~94MB free, load 13) → benches stalled → 600s stream watchdog killed all 6 in a cascade. NOT task failures. ALL branches preserved+pushed: dev-sk-deepseek-fix@ca25293 (MLA RoPE numerics fix!), dev-sk-spec-decode, dev-sk-gemm-mma-ext, dev-sk-icb-batch, dev-sk-qkv-rope-fusion, worktree-agent-a8c90d39fb2f2408a (KV-quant). **POLICY (mandatory for all future agents):** (a) do ALL kernel dev + numerical validation LOCALLY (this Mac has Metal); (b) mini benches via `nohup bench >log 2>&1 &` + poll the log with SHORT ssh every 60-90s EMITTING a status line each poll (resets the 600s watchdog); NEVER one ssh that blocks >300s; (c) prefer amelia (derek OOM-tight), check free-mem before model load, clamp cache_max if tight; (d) hold /tmp/sk_bench.lock only around the bench; (e) if no host slot, mark READY-TO-BENCH + push + report — don't stall to death; (f) cap concurrency ~3. HyperQueue installed (server up) but NO workers attached → stand up workers on minis to use it (deferred). #1 qkv-rope = NEG. Relaunching 3 HARDENED: #12 DeepSeek (resume), #8-ext gemm-mma-ext (resume), #10 spec-decode (resume).
- **[wave-1.1 follow-ups on #8]** `a0c01433283a43002` (#10 spec-decode re-test on dev-sk-batched-gemm → branch dev-sk-spec-decode) + `aa44764c981d38d20` (#8-ext: Q6_K/bf16 MMA loaders + Q4_K scale-cache + prefill-e2e/TTFT → branch dev-sk-gemm-mma-ext). Fleet now 6 in flight: #1 ae697b, #3 a629b5, #12 a65832, #17 a8c90d, #10 a0c014, #8-ext aa4476.
- **[#8 WIN — batched GEMM]** `a0494e7e14793d0b5` done → `gemm_mma.metal` (f16/q8_0/q4k, simdgroup_float8x8, BM8×BN32×BK32, weight tile reused M-fold). Wired into qwen prefill (M>1→MMA, M==1→matvec; SK_NO_GEMM_MMA flag). **Numerics 72/72 ≤1e-3** (M=1..100). **Prefill amelia: Q8_0 1.5-2.6×, Q4_K 1.25-1.59×; per-position sub-linear (seq=8≈seq=1 cost, seq=128≈8-13× not 128×).** Branch `dev-sk-batched-gemm` pushed, **PR #55** open (not merged — for review). Unblocks: #10 spec-decode (should flip 0.54×), #9/#16 DeepSeek+qwen T>1 prefill e2e, Q6_K/bf16 MMA loaders, last-row-only LM head. Launching follow-ups: spec-decode re-test + gemm_mma dtype-extension/prefill-e2e.
- **[wave-1 launch]** 5 worktree-isolated agents in flight (push own branch to remote; PR-only to main; bench derek/amelia, lexie down):
  - #1 Fused qkv-norm-rope T1 kernel — `ae697bf8e4d955340` — branch `dev-sk-qkv-rope-fusion`
  - #8 Batched GEMM (simdgroup_matrix MMA, seq>1; prefill/spec/DeepSeek-T>1 unblock) — `a0494e7e14793d0b5` — `dev-sk-batched-gemm`
  - #12–14 DeepSeek-V2-Lite numerics→coherence + GPU-OOB (base dev-deepseek-e2e) — `a6583284df8beb960` — `dev-sk-deepseek-fix`
  - #17 KV-cache quantization Q8_0 (short+long-ctx bench) — `a8c90d39fb2f2408a` — `dev-sk-kv-quant`
  - #3 ICB multi-layer batching (measure the unmeasured ICB win) — `a629b5aecd7b79de6` — `dev-sk-icb-batch`
- **[init]** Campaign branch `dev-sk-redesign` created off main 49f9174. PROGRESS.md seeded with operating model + 40-item backlog.
- **[wave-2 HARDENED relaunch]** 3 agents, decoupled-bench protocol (local dev+validate, nohup+poll mini bench, prefer amelia, cap 3): ac1b4d265d0791526 (#12 DeepSeek coherence, dev-sk-deepseek-fix), a724311a0b40baf55 (#8-ext gemm-mma-ext, dev-sk-gemm-mma-ext), a0a45bfc6b45fe0d8 (#10 spec-decode payoff, dev-sk-spec-decode). Deferred (branches preserved): #3 ICB, #17 KV-quant. #1 qkv = NEG.
