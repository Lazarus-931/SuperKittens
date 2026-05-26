# Best measured decode tok/s

All on lexie M4 base (16 GB unified, no VMs), Q8_0 GGUF unless noted.
Bench protocol: 2 warmup passes + 30s cooldown + 5 reps × 100 tokens (thermal-stable).

## dev-perf-iter @ 30ff88b (2026-05-26)

Decode tok/s, 5-rep median:

| Model       | SK tok/s   | kernel-only ceiling | % of ceiling |
|-------------|------------|---------------------|--------------|
| Qwen3-0.6B  | **127.63** | ~129                | 99%          |
| Qwen3-1.7B  | **52.63**  | —                   | —            |
| Qwen3-4B    | **17.71**  | ~20                 | 89%          |

Net journey on dev-perf-iter: 4B 7.16 → 17.71 tok/s (+147%). Biggest single lever was item C of iter 11 — removing two redundant `memoryBarrierWithScope(Buffers)` calls in `dispatch_layer` that were forcing full L2 fence drains without serving any producer-consumer dependency (+90% on 4B from that alone).

## Historical (pre-perf-iter, retained for context)

| Model | SK tok/s | llama.cpp | uzu (claimed) | PR stack |
|---|---|---|---|---|
| Qwen3-0.6B | 272 (different harness) | 117 | ~150-165 | #7 #12-15 #21-24 #26-27 |
| Qwen3-8B | 11.26 | 12.79 | not supported | #13 #15 |
| Gemma4-E2B | 25.01 (bf16 + Q8 lm_head) | 54.9 | not supported | #19 #21 |
| Gemma-3-4B | not ported | 24.50 | ~32-34 (projected) | — |
| Mamba2-130m | 90 | 14.5 (HF fp32) | not supported | #4 |
