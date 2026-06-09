# Best measured decode tok/s

All on lexie M4 base (16 GB unified, no VMs).
Bench protocol: 2 warmup passes + 30s cooldown + 5 reps × 100 tokens (thermal-stable).
Decode on M4 base is GPU-bandwidth-bound (~99% GPU-wait) — tok/s tracks weight bytes moved.

## Current best — Qwen3 Q4_K_M (on `main`)

Q4_K_M is the decode sweet spot: 8B/14B only fit 16 GB at Q4_K_M, and sub-Q4
weight-quant (Q2_K/Q3_K) is a compute-bound net loss. Q6_K LM-head routed natively.

| Model      | SK tok/s   |
|------------|------------|
| Qwen3-4B   | **38.48**  |
| Qwen3-8B   | **21.09**  |
| Qwen3-14B  | **11.28**  |

(Cross-host check: a clean-`main` sweep on derek read Qwen3-4B 35.94 / 8B 21.18 — consistent;
lexie is the canonical perf host so the lexie numbers above are the headline.)

## Families added — SK now loads 9 (all coherent on `main`)

Decode tok/s below are per-host (lexie canonical; derek/amelia noted) — indicative, not all
under the strict lexie protocol. All validated coherent (32+ tok); qwen3/nemotron token-for-token
vs llama.cpp on a clean-`main` regression sweep.

| Model | SK tok/s | host | coherent | notes |
|---|---|---|---|---|
| DeepSeek-V2-Lite (MLA + 64-expert MoE) | **36.15** | derek | yes | 0.76→27.73→36 via NULL-buffer coherence fix + Q5_0 routed-down + native shared-down matvec ([#71]/[#76]/[#83]); fits 16 GB, no swap |
| Llama-3.2-3B-Instruct | **31.80** | lexie | yes | config-only over the shared DenseDecoder (tied embed, interleaved RoPE) ([#85]) |
| nemotron-Nano-8B (Llama-3.1) | **21.58** | derek | yes | interleaved/NORM RoPE for Llama GGUFs fixed the degeneration ([#79]); shared dense core |
| gemma4-E2B (PLE, KV-share) | **~21** | derek | yes | Q8 body+lmhead ([#69]/[#72]) |
| Mistral-7B-Instruct-v0.3 | **19.93** | amelia | yes | config-only over DenseDecoder + interleaved RoPE ([#84]) |
| gemma4-E4B (own ckpt, 42L) | **9.86** | — | yes | PLE-table + embed Q8 fixed the 1.44→9.86 paging cliff ([#74]) |
| gemma4-unified-12B (distinct arch) | **8.2–8.4** | derek | yes | Q4_K body (q4k/q6k_matvec_bf16) fits under the ~12 GB Metal wired limit + fp16-subnormal embed-dequant fix ([#78]) |

Decode-perf finding: M4 small-N M=1 matvecs are occupancy/latency-bound (~33% of the ~120 GB/s
peak), so dispatch-fusion and sub-Q4 weight-quant are exhausted body-decode levers — the wins came
from cutting resident below the swap line, routing host-dequant'd projections through native quant
matvecs, and large-N LM-head quant. Tooling: `tools/gputracer.py` (.gputrace parser + capture, [#81]/[#87]).

## Prefill / TTFT — batched GEMM (`gemm_mma`, landed [#56]/[#59])

Wiring seq>1 prefill through `gemm_mma` (f16/Q8_0/Q4_K MMA) gives **2.5–5.8× TTFT**;
seq=2–8 verify cost drops from 4.6–5.8× to 1.0–2.8× of a single decode. Decode (M=1)
unchanged. Spec-decode still net-negative on M4 4B (draft tax + accept ceiling); 8B under test.

## SSM

| Model      | SK tok/s | HF fp32 baseline | Notes |
|------------|----------|------------------|-------|
| Mamba2-130m| **~99** (derek sweep; ~50 earlier probe) | 14.48 | coherent e2e, = HF token-for-token; SSD-rewrite + e2e + O(1) conv1d decode all landed via [#66] |

## dev-perf-iter @ 30ff88b (2026-05-26) — Q8_0, historical

Decode tok/s, 5-rep median (Q8_0 GGUF):

| Model       | SK tok/s   | kernel-only ceiling | % of ceiling |
|-------------|------------|---------------------|--------------|
| Qwen3-0.6B  | **127.63** | ~129                | 99%          |
| Qwen3-1.7B  | **52.63**  | —                   | —            |
| Qwen3-4B    | **17.71**  | ~20                 | 89%          |

Net journey on dev-perf-iter: 4B 7.16 → 17.71 tok/s (+147%). Biggest single lever was item C of iter 11 — removing two redundant `memoryBarrierWithScope(Buffers)` calls in `dispatch_layer` that were forcing full L2 fence drains without serving any producer-consumer dependency (+90% on 4B from that alone). The 4B 17.71 → 38.48 jump above is the Q4_K_M weight-quant (fewer bytes moved on a bandwidth-bound decode).

## Historical (pre-perf-iter, retained for context)

| Model | SK tok/s | llama.cpp | uzu (claimed) | PR stack |
|---|---|---|---|---|
| Qwen3-0.6B | 272 (different harness) | 117 | ~150-165 | #7 #12-15 #21-24 #26-27 |
| Qwen3-8B | 11.26 | 12.79 | not supported | #13 #15 |
| Gemma4-E2B | 25.01 (bf16 + Q8 lm_head) | 54.9 | not supported | #19 #21 |
| Gemma-3-4B | not ported | 24.50 | ~32-34 (projected) | — |
| Mamba2-130m | 90 | 14.5 (HF fp32) | not supported | #4 |
