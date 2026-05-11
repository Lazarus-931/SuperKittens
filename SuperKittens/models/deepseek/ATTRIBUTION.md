# Attribution

The Metal kernels in `kernels/` are sourced from antirez/ds4
(https://github.com/antirez/ds4), MIT licensed. ds4 in turn adapted
several from llama.cpp/ggml.

## Build glue

The kernel files were copied verbatim from ds4. ds4 builds them as one
concatenated source string with a small preamble (`using namespace metal;`,
macros, `block_q8_0`, `ds4_metal_args_mul_mv` struct, function constants).
SK's per-file `xcrun metal` flow needs that preamble too, so we
`-include kernels/ds4_preamble.h` from build.sh for any file under
`models/deepseek/kernels/`. See build.sh.

## Files in `kernels/` (all compile clean)

DeepSeek V4 specials:
- `dsv4_hc.metal` — head-compression: sinkhorn, weighted-sum, expand
- `dsv4_kv.metal` — fp8 KV quantize / store, ratio-4 shift
- `dsv4_misc.metal` — top-k mask, indexed attention, softmax-pool, compressor store, sort
- `dsv4_rope.metal` — p-RoPE tail-only RoPE

Attention:
- `flash_attn.metal` — MLA-shaped flash attention (dk=512 / dv=512), with `_blk`, `_pad`, `_vec`, `_vec_reduce` variants

Utilities:
- `argsort.metal`, `bin.metal`, `concat.metal`, `cpy.metal`,
  `repeat.metal`, `set_rows.metal`, `softmax.metal`, `sum_rows.metal`

## Files NOT copied

- `dense.metal` — SK has its own `kernels/gemm/fp16/gemm.metal`
- `get_rows.metal` — SK has `kernels/utils/embedding/embedding.metal`
- `glu.metal` — SK has `kernels/swiglu/swiglu.metal`
- `unary.metal` — SK has `kernels/activation/activation.metal`
- `norm.metal` — SK has `kernels/utils/rmsnorm/rms_norm.metal`
- `moe.metal` — purely quantized variants (Q8_0/Q2_K/Q4_K/IQ2_XXS) that
  depend on `kernel_mul_mv_q8_0_f32_impl` from `dense.metal`. SK's fp16
  MoE FFN is covered by `kernels/moe/{router,swiglu_pair,down_scatter}.metal`
  + the new `models/deepseek/moe_ffn.h` orchestrator. If we ship the
  quantized DS4 path later, we'd port `moe.metal` + needed dense.metal
  bits into a separate quantized subtree.
