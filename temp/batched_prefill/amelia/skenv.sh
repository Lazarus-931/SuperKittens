# Source for SK runtime-compile env on amelia (CLT-only host, no xcrun metal).
# SK_DYLIB defaults to the patched build; export SK_DYLIB=$HOME/sk-batched-prefill/build/libsk_base.dylib for the baseline A side.
export SK_DYLIB=${SK_DYLIB:-$HOME/sk-batched-prefill/build/libsk.dylib}
export SK_METALLIB=/nonexistent
SKK=$HOME/sk-batched-prefill/SuperKittens/kernels
SKM=$HOME/sk-batched-prefill/SuperKittens/models
FILES="
$SKK/attn/attn.metal
$SKK/gemm/gemm_mma.metal
$SKK/gemm/gemm_mma_smallm.metal
$SKK/gemm/q4k_matvec.metal
$SKK/gemm/q6k_matvec.metal
$SKK/gemm/q8_0_matvec.metal
$SKK/gemm/q8_0_matvec_addres.metal
$SKK/gemm/fp16/gemm.metal
$SKK/gemm/fp16/gemv.metal
$SKK/gemm/fp16/gemv_t.metal
$SKK/gemm/q4k_matvec_bf16.metal
$SKK/gemm/q8_0_swiglu_m1.metal
$SKK/ops/add/add.metal
$SKK/fusion/gemv_swiglu_m1.metal
$SKK/fusion/add_rmsnorm.metal
$SKK/fusion/silu_mul.metal
$SKK/fusion/gated_mlp.metal
$SKK/fusion/q8_0_swiglu_prenorm_m1.metal
$SKK/utils/rmsnorm/rms_norm.metal
$SKK/utils/rmsnorm/rmsnorm_t1.metal
$SKK/ops/split/split_packed.metal
$SKK/ops/transpose/seq_head.metal
$SKK/ops/kv_cache/kv_cache.metal
$SKK/ops/sample/argmax_2pass.metal
$SKK/ops/sample/sample.metal
$SKK/utils/embedding/embedding.metal
$SKM/qwen/qwen_rope.metal
"
export SK_METAL_SRC_FALLBACK=$(echo $FILES | tr " \n" "::" | sed "s/^://; s/:*$//; s/::*/:/g")
