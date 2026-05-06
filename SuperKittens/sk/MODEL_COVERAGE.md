# Model Coverage — May 2026

## What every model needs

| Primitive | Llama 4 | Mistral | Gemma 3 | Phi-4 | DeepSeek V3 | Qwen 3 | Command-R | Jamba | Granite |
|-----------|---------|---------|---------|-------|-------------|--------|-----------|-------|---------|
| RMSNorm | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RoPE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| SwiGLU | ✓ | ✓ | GeGLU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GEMM fp16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Attention (causal) | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| GQA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| MoE | — | ✓ | — | — | ✓ | ✓ | — | ✓ | — |
| Sliding window | — | ✓ | — | — | — | — | — | — | — |
| MLA | — | — | — | — | ✓ | — | — | — | — |
| Conv1D | — | — | — | — | — | — | — | ✓ | — |
| SSM (Mamba) | — | — | — | — | — | — | — | ✓ | — |
| Cross-attention | — | — | — | — | — | — | — | — | — |

## What we have vs what's missing

| Kernel | Have | Priority | Blocks which models |
|--------|------|----------|---------------------|
| GEMM fp16 | ✓ | — | — |
| GEMM fp8 | ✓ | — | — |
| RMSNorm | ✓ | — | — |
| LayerNorm | ✓ | — | — |
| RoPE | ✓ | — | — |
| SwiGLU | ✓ | — | — |
| Causal MHA | ✓ | — | — |
| **GQA (grouped query)** | ✗ | **P0** | Llama 4, Mistral, Gemma, Phi, Qwen, DeepSeek |
| **MoE (expert routing)** | ✗ | **P0** | DeepSeek V3/R1, Mixtral, Qwen-MoE |
| **Sliding window attn** | ✗ | **P1** | Mistral |
| **MLA (latent attn)** | ✗ | **P1** | DeepSeek V3/R1 |
| **Sparse MoE** | ✗ | **P2** | DeepSeek (shared experts + routed) |
| Paged attention | ✗ | **P0** | All (inference only) |
| KV cache ops | ✗ | **P0** | All (inference only) |

## What to build — ranked

### P0: GQA (1 file, 90% model coverage)
GQA is MHA where `num_kv_heads < num_q_heads`. Q heads share KV heads.
**How**: our existing MHA kernel. Just change the grid and Q/K/V indexing.
Each threadgroup processes one Q head, KV heads are replicated or indexed via `kv_head = q_head * n_kv / n_q`.
Change: 5 lines in `attn.metal` + dispatch. Not a new kernel — a parameter.

### P0: Paged attention (1 kernel, unlocks inference)
Block-table indirection. The kernel iterates `block_table[seq_id]` instead of `0..seq`.
**How**: `paged_attn.metal` — same attention math, just block-table indexed K/V loads.
paws already provides the block table.

### P0: MoE (2 kernels, unlocks DeepSeek, Mixtral, Qwen-MoE)
Two ops: `gather` (route tokens to experts) and `scatter_add` (combine expert outputs).
Actually simpler: each expert is a FFN (swiglu + gemm). The routing is:
```python
scores = softmax(router(h))        # (B, L, n_experts)
top_k_scores, top_k_idx = top_k(scores, k=2)  # pick top 2 experts
for each expert e:
    tokens_e = h[top_k_idx == e]   # gather tokens routed to this expert
    out_e = expert_ffn(tokens_e)   # SwiGLU + gemm
    output[top_k_idx == e] += scores_e * out_e  # scatter-add weighted output
```
CPU scheduling + GPU gemms. No special kernel needed.

### P1: Sliding window (1 kernel variant)
Same attention, but limit K to `[pos - window, pos]` instead of `[0, pos]`.
Change: 2 lines in the causal mask logic in `attn.metal`.

### P1: MLA (1 kernel, complex)
DeepSeek's MLA compresses KV into a latent space. Needs a different attention kernel.
Lowest priority — only DeepSeek uses it.

## Architecture — keep it clean

Each primitive is ONE `.metal` file. Models compose them:

```
kernels/
├── attn/
│   ├── attn.metal       # MHA + GQA (same kernel, different grid)
│   └── paged_attn.metal # block-table attention
├── gemm/
│   └── fp16/gemm.metal  # + fp8
├── norm/
│   ├── layernorm.metal
│   └── rmsnorm.metal
├── rope/rope.metal
├── swiglu/swiglu.metal
├── conv/conv1d.metal
├── mamba/               # mamba2 + mamba3
└── moe/                 # NEW: routing kernels (thin)
```

A model is just a Python class that calls these kernels in sequence. Llama 4 = RMSNorm + GQA + RoPE + SwiGLU + GEMM. DeepSeek V3 = same + MoE routing. Mistral = same + sliding window.

## The 3 things to build today

1. **GQA support in attn.metal** — 5 lines, unlocks Llama 4, Mistral, Gemma, Phi, Qwen
2. **paged_attn.metal** — inference-only, uses paws block tables
3. **models/llama.py** — the reference model runner everyone copies
