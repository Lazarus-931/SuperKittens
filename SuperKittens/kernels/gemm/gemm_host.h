//
//  gemm_host.h
//  SuperKittens
//
//  Host-side helpers for GEMM kernel selection and launch geometry.
//

#ifndef SUPERKITTENS_GEMM_HOST_H
#define SUPERKITTENS_GEMM_HOST_H

#include "gemm_impl.h"

namespace meow {
namespace gemm {

struct GemmLaunchConfig {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t threads_x;
    uint32_t threads_y;
    uint32_t block_m;
    uint32_t block_n;
    uint32_t block_k;
};

inline uint32_t ceil_div_u32(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

inline stride_t default_lda(int m, int k, uint32_t op_a) {
    return (op_a == GEMM_OP_N) ? stride_t(k) : stride_t(m);
}

inline stride_t default_ldb(int k, int n, uint32_t op_b) {
    return (op_b == GEMM_OP_N) ? stride_t(n) : stride_t(k);
}

inline GemmParams make_gemm_params(
    int m, int n, int k,
    uint32_t op_a = GEMM_OP_N,
    uint32_t op_b = GEMM_OP_N,
    float alpha = 1.0f,
    float beta = 0.0f,
    uint32_t specialization = GEMM_SPEC_GENERIC,
    uint32_t epilogue = GEMM_EPILOGUE_NONE,
    stride_t lda = 0,
    stride_t ldb = 0,
    stride_t ldc = 0)
{
    GemmParams p = {};
    p.m = m;
    p.n = n;
    p.k = k;
    p.lda = lda ? lda : default_lda(m, k, op_a);
    p.ldb = ldb ? ldb : default_ldb(k, n, op_b);
    p.ldc = ldc ? ldc : stride_t(n);
    p.alpha = alpha;
    p.beta = beta;
    p.op_a = op_a;
    p.op_b = op_b;
    p.specialization = specialization;
    p.epilogue = epilogue;
    return p;
}

inline bool is_supported(const GemmParams& p) {
    return p.op_a == GEMM_OP_N && p.op_b == GEMM_OP_N;
}

inline const char* kernel_name_for(const GemmParams& p) {
    if (!is_supported(p)) return nullptr;

    if (p.epilogue == GEMM_EPILOGUE_BIAS_SILU) {
        switch (p.specialization) {
            case GEMM_SPEC_2048_3072_4096:
                return "gemm_fp16_nn_bias_silu_2048x3072x4096";
            case GEMM_SPEC_3072_2048_4096:
                return "gemm_fp16_nn_bias_silu_3072x2048x4096";
            case GEMM_SPEC_4096_4096_4096:
                return "gemm_fp16_nn_bias_silu_4096x4096x4096";
            case GEMM_SPEC_GENERIC:
            default:
                return "gemm_fp16_nn_bias_silu_generic";
        }
    }

    switch (p.specialization) {
        case GEMM_SPEC_2048_3072_4096:
            return "gemm_fp16_nn_2048x3072x4096";
        case GEMM_SPEC_3072_2048_4096:
            return "gemm_fp16_nn_3072x2048x4096";
        case GEMM_SPEC_4096_4096_4096:
            return "gemm_fp16_nn_4096x4096x4096";
        case GEMM_SPEC_GENERIC:
        default:
            return "gemm_fp16_nn_generic";
    }
}

inline GemmLaunchConfig launch_config_for(const GemmParams& p) {
    GemmLaunchConfig cfg = {};
    cfg.block_m = GemmShape<GEMM_SPEC_GENERIC>::Config::BLOCK_M;
    cfg.block_n = GemmShape<GEMM_SPEC_GENERIC>::Config::BLOCK_N;
    cfg.block_k = GemmShape<GEMM_SPEC_GENERIC>::Config::BLOCK_K;
    cfg.threads_x = GemmShape<GEMM_SPEC_GENERIC>::Config::THREADS_X;
    cfg.threads_y = GemmShape<GEMM_SPEC_GENERIC>::Config::THREADS_Y;

    cfg.grid_x = ceil_div_u32(static_cast<uint32_t>(p.n), cfg.block_n);
    cfg.grid_y = ceil_div_u32(static_cast<uint32_t>(p.m), cfg.block_m);
    return cfg;
}

} // namespace gemm
} // namespace meow

#endif // SUPERKITTENS_GEMM_HOST_H
