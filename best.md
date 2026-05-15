# Best measured decode tok/s

All on lexie M4 (16 GB unified, no VMs), Q8_0 GGUF unless noted.

| Model | SK tok/s | llama.cpp | uzu (claimed) | PR stack |
|---|---|---|---|---|
| Qwen3-0.6B | **272** | 117 | ~150-165 | #7 #12-15 #21-24 #26-27 |
| Qwen3-8B | 11.26 | 12.79 | not supported | #13 #15 |
| Gemma4-E2B | 25.01 (bf16 + Q8 lm_head) | 54.9 | not supported | #19 #21 |
| Gemma-3-4B | not ported | 24.50 | ~32-34 (projected) | — |
| Mamba2-130m | 90 | 14.5 (HF fp32) | not supported | #4 |
