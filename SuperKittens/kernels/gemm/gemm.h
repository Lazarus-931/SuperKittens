//
//  gemm.h
//  SuperKittens
//
//  Shared GEMM configuration traits.
//

#ifndef SUPERKITTENS_GEMM_H
#define SUPERKITTENS_GEMM_H

#include "gemm_impl.h"

namespace meow {
namespace gemm {

template <int BM_, int BN_, int BK_, int MR_, int MC_, int TX_, int TY_>
struct GemmTileConfig {
    enum : int {
        BLOCK_M = BM_,
        BLOCK_N = BN_,
        BLOCK_K = BK_,
        MMA_ROWS = MR_,
        MMA_COLS = MC_,
        THREADS_X = TX_,
        THREADS_Y = TY_,
    };
};

using Fp16GenericConfig = GemmTileConfig<32, 64, 32, 2, 4, 128, 1>;

template <uint32_t Spec>
struct GemmShape;

template <>
struct GemmShape<GEMM_SPEC_GENERIC> {
    enum : int { M = 0, N = 0, K = 0 };
    using Config = Fp16GenericConfig;
};

template <>
struct GemmShape<GEMM_SPEC_2048_3072_4096> {
    enum : int { M = 2048, N = 3072, K = 4096 };
    using Config = Fp16GenericConfig;
};

template <>
struct GemmShape<GEMM_SPEC_3072_2048_4096> {
    enum : int { M = 3072, N = 2048, K = 4096 };
    using Config = Fp16GenericConfig;
};

template <>
struct GemmShape<GEMM_SPEC_4096_4096_4096> {
    enum : int { M = 4096, N = 4096, K = 4096 };
    using Config = Fp16GenericConfig;
};

} // namespace gemm
} // namespace meow

#endif // SUPERKITTENS_GEMM_H
