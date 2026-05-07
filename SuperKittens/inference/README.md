# Inference

Model-specific fused kernels and memory management for edge inference.

## Structure

```
inference/
├── kernels/           # model-specific fused kernels
│   ├── llama/         # LLaMA attention block, MLP block (RMSNorm+RoPE+FA fused)
│   ├── mamba/         # Mamba pre_ssm + ssm + post_ssm fused variants
│   └── gemma/         # Gemma GeGLU MLP, d=256 attention
├── models/            # model configs, weight loading, memory plans
│   ├── llama_1b.py    # LLaMA 3.2 1B: layers, dims, KV-cache sizing
│   └── mamba_130m.py  # Mamba-130M: d_model=768, n_layers=24
└── runtime/           # token loop, KV-cache, scheduler, state management
```

## Design principles

- **kernels/inference/** are composed from DSL primitives (tile.h, mma.h) but fused for a specific model's tensor shapes. They assume fixed dimensions known at compile time.
- **models/** owns memory layout — KV-cache sizing, intermediate buffer allocation, weight format. Each model is a config + allocator, not a kernel.
- **runtime/** is model-agnostic — it schedules kernel launches, manages the token generation loop, and owns the KV-cache state across decode steps.
