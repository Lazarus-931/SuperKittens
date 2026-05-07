//
//  autotune.h — chip-aware GEMM tile selection
//
//  Verified per-family (BM, BN, BK, THREADS) with register + TG mem budget.
//  M1=64×64@128t, M2=64×128@256t, M3+=128×128@256t.
//

#pragma once

namespace sk {
namespace compiler {

// Apple 10 (M5): fp8-capable, 128 regs
#if defined(__METAL_FAMILY_APPLE_10__)
template<int=0,int=0,int=0> struct BestGemm {
    enum : int { BM=128, BN=128, BK=64, THREADS=256, REGS=64, TG_KB=20 };
};

// Apple 9 (M3/M4): dynamic cache, 128 regs
#elif defined(__METAL_FAMILY_APPLE_9__)
template<int=0,int=0,int=0> struct BestGemm {
    enum : int { BM=128, BN=128, BK=32, THREADS=256, REGS=64, TG_KB=16 };
};

// Apple 8 (M2): bfloat, 112 regs
#elif defined(__METAL_FAMILY_APPLE_8__)
template<int=0,int=0,int=0> struct BestGemm {
    enum : int { BM=64, BN=128, BK=32, THREADS=256, REGS=32, TG_KB=12 };
};

// Apple 7 (M1): baseline, 96 regs
#else
template<int=0,int=0,int=0> struct BestGemm {
    enum : int { BM=64, BN=64, BK=32, THREADS=128, REGS=32, TG_KB=8 };
};
#endif

} // namespace compiler
} // namespace sk
