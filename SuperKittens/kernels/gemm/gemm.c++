#include "gemm.h"
#include "../runtime_bindings.h"

static int dispatch(const char* kname, void* A, void* B, void* C, void* bias,
                    uint32_t M, uint32_t N, uint32_t K, uint32_t ldA, uint32_t ldB, uint32_t ldC,
                    int transA, int transB, int has_bias) {
    auto* pso = sk::bindings_pso(kname);
    if (!pso) return -1;

    size_t ab = (size_t)M * K * 2, bb = (size_t)K * N * 2, cb = (size_t)M * N * 2, bbb = (size_t)N * 2;
    auto* bA = sk::bindings_device()->newBuffer(ab, MTL::ResourceStorageModeShared);
    auto* bB = sk::bindings_device()->newBuffer(bb, MTL::ResourceStorageModeShared);
    auto* bC = sk::bindings_device()->newBuffer(cb, MTL::ResourceStorageModeShared);
    memcpy(bA->contents(), A, ab); memcpy(bB->contents(), B, bb);

    auto* bBias = has_bias ? sk::bindings_device()->newBuffer(bbb, MTL::ResourceStorageModeShared) : nullptr;
    if (bBias) memcpy(bBias->contents(), bias, bbb);

    uint32_t gx = (N + 63) / 64, gy = (M + 31) / 32;  // gemm_fp16 BM=32 rows

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bA, 0, 0); enc->setBuffer(bB, 0, 1); enc->setBuffer(bC, 0, 2);
    enc->setBytes(&M, 4, 3); enc->setBytes(&N, 4, 4); enc->setBytes(&K, 4, 5);
    enc->setBytes(&ldA, 4, 6); enc->setBytes(&ldB, 4, 7); enc->setBytes(&ldC, 4, 8);
    enc->setBytes(&transA, 4, 9); enc->setBytes(&transB, 4, 10); enc->setBytes(&has_bias, 4, 11);
    enc->setBuffer(bBias ? bBias : bC, 0, 12);
    enc->dispatchThreadgroups(MTL::Size(gx, gy, 1), MTL::Size(64, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(C, bC->contents(), cb);

    cmd->release(); bA->release(); bB->release(); bC->release(); if (bBias) bBias->release();
    return 0;
}

// M=1 specialized matvec (decode hot path). 2–6× faster than the tile-MMA
// GEMM at the typical decode shapes (proven on Qwen3-32B benches: QKV-proj
// 5.97×, O-proj 5.21×, MLP-down 5.08×).
static int dispatch_gemv_m1(void* x, void* W, void* y,
                            uint32_t N, uint32_t K) {
    auto* pso = sk::bindings_pso("gemv_fp16_m1");
    if (!pso) return -1;
    auto* dev = sk::bindings_device();

    const size_t xb = (size_t)K * 2, Wb = (size_t)K * N * 2, yb = (size_t)N * 2;
    auto* bX = dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = dev->newBuffer(Wb, MTL::ResourceStorageModeShared);
    auto* bY = dev->newBuffer(yb, MTL::ResourceStorageModeShared);
    memcpy(bX->contents(), x, xb);
    memcpy(bW->contents(), W, Wb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bX, 0, 0); enc->setBuffer(bW, 0, 1); enc->setBuffer(bY, 0, 2);
    enc->setBytes(&N, 4, 3); enc->setBytes(&K, 4, 4);
    const uint32_t BN = 128;
    enc->dispatchThreadgroups(MTL::Size((N + BN - 1) / BN, 1, 1),
                              MTL::Size(BN, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    memcpy(y, bY->contents(), yb);
    cmd->release(); bX->release(); bW->release(); bY->release();
    return 0;
}

int sk_gemm_fp16(void* A, void* B, void* C, void* bias,
                 uint32_t M, uint32_t N, uint32_t K, uint32_t ldA, uint32_t ldB, uint32_t ldC,
                 int transA, int transB, int has_bias)
{
    // Decode hot path: route M=1, no-bias, non-transposed → specialized matvec.
    // ldA/ldB/ldC must be standard (K/N/N) for the matvec kernel — the tile
    // GEMM is the fallback for strided or transposed cases.
    if (M == 1 && !transA && !transB && !has_bias
        && ldA == K && ldB == N && ldC == N) {
        return dispatch_gemv_m1(A, B, C, N, K);
    }
    return dispatch("gemm_fp16", A, B, C, bias, M, N, K, ldA, ldB, ldC,
                    transA, transB, has_bias);
}

int sk_gemm_fp8(void* A, void* B, void* C, void* bias,
                uint32_t M, uint32_t N, uint32_t K, uint32_t ldA, uint32_t ldB, uint32_t ldC,
                int transA, int transB, int has_bias)
    { return dispatch("gemm_fp8", A, B, C, bias, M, N, K, ldA, ldB, ldC, transA, transB, has_bias); }
