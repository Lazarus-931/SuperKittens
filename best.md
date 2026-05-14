# Best measured decode tok/s

All on lexie M4 (16 GB unified, no VMs).

| Model | Quant | SK tok/s | llama.cpp | PR |
|---|---|---|---|---|
| Qwen3-0.6B | Q8_0 GGUF | **137** | 119 | #7 + #12-15 |
| Qwen3-8B | Q8_0 GGUF | **11.26** | 12.79 | #13 + #15 |
| Gemma4-E2B | bf16 + Q8_0 lm_head | **25.01** (64-tok) | 54.9 | #19 + #21 |
| Mamba2-130m | fp16 safetensors | **90** | 14.5 (HF fp32) | #4 |
