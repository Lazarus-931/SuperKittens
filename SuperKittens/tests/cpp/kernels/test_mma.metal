//
//  test_mma.metal — GEMM via user-facing GemmConfig (8×8 hidden)
//

#include "../../../sk/src/cpp/tile.h"
#include "../../../sk/src/cpp/mma.h"

using namespace sk::dsl;

[[host_name("test_mma_gemm")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void test_mma_gemm(
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
    // User only specifies: <BM, BN, BK, Threads>
    using GEMM = GemmConfig<64, 64, 32, 128>;  // BM=64, BN=64, BK=32, 128 threads

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
