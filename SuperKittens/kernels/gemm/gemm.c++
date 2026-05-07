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

    uint32_t gx = (N + 63) / 64, gy = (M + 63) / 64;

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

int sk_gemm_fp16(void* A, void* B, void* C, void* bias,
                 uint32_t M, uint32_t N, uint32_t K, uint32_t ldA, uint32_t ldB, uint32_t ldC,
                 int transA, int transB, int has_bias)
    { return dispatch("gemm_fp16", A, B, C, bias, M, N, K, ldA, ldB, ldC, transA, transB, has_bias); }

int sk_gemm_fp8(void* A, void* B, void* C, void* bias,
                uint32_t M, uint32_t N, uint32_t K, uint32_t ldA, uint32_t ldB, uint32_t ldC,
                int transA, int transB, int has_bias)
    { return dispatch("gemm_fp8", A, B, C, bias, M, N, K, ldA, ldB, ldC, transA, transB, has_bias); }
