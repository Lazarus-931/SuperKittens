//
//  test_autotune.metal — verify chip-aware tile selection
//

#include "../../../sk/compiler/autotune.h"
#include "../../../sk/src/cpp/tile.h"
#include "../../../sk/src/cpp/mma.h"

using namespace sk::compiler;
using namespace sk::dsl;

// Compile-time check: verify BM, BN, BK are non-zero
using Tune = BestGemm<>;
static_assert(Tune::BM >= 64, "tile too small");
static_assert(Tune::BN >= 64, "tile too small");

[[host_name("test_autotune_gemm")]]
[[kernel, max_total_threads_per_threadgroup(Tune::THREADS)]]
void test_autotune_gemm(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C       [[buffer(2)]],
    constant uint& M     [[buffer(3)]],
    constant uint& N     [[buffer(4)]],
    constant uint& K     [[buffer(5)]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint lid  [[thread_index_in_threadgroup]])
{
    using GEMM = GemmConfig<Tune::BM, Tune::BN, Tune::BK, Tune::THREADS>;

    threadgroup Fp16Tile<GEMM::BM, GEMM::BK> As;
    threadgroup Fp16Tile<GEMM::BK, GEMM::BN> Bs;
    auto acc = GEMM::make_acc();

    for (uint bk = 0; bk < K; bk += GEMM::BK) {
        load_tile<GEMM::BM, GEMM::BK, GEMM::THREADS>(A, K, 0, bk, M, K, &As, lid);
        load_tile<GEMM::BK, GEMM::BN, GEMM::THREADS>(B, N, bk, 0, K, N, &Bs, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        GEMM::mma(acc, As.data, Bs.data, simd);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    GEMM::store(C, N, 0, 0, M, N, acc, simd, lane);
}
