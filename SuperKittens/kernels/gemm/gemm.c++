//
//  gemm.c++ — C binding implementation for GEMM
//

#include "gemm.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>

static MTL::Device*      g_dev = nullptr;
static MTL::CommandQueue* g_q  = nullptr;
static MTL::Library*     g_lib = nullptr;

static void ensure() {
    if (g_dev) return;
    g_dev = MTL::CreateSystemDefaultDevice();
    g_q   = g_dev->newCommandQueue();
    const char* path = getenv("SK_METALLIB") ? getenv("SK_METALLIB") : "build/libsk.metallib";
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
    g_lib = g_dev->newLibrary(url, &err);
}

static int dispatch(const char* name,
                    void* A, void* B, void* C, void* bias,
                    uint32_t M, uint32_t N, uint32_t K,
                    uint32_t ldA, uint32_t ldB, uint32_t ldC,
                    int transA, int transB, int has_bias,
                    size_t elem_size) {
    ensure();
    if (!g_lib) return -1;

    auto* fn = g_lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return -2;
    NS::Error* err = nullptr;
    auto* pso = g_dev->newComputePipelineState(fn, &err); fn->release();
    if (!pso) { if (err) err->release(); return -3; }

    size_t a_bytes = (size_t)M * K * elem_size;
    size_t b_bytes = (size_t)K * N * elem_size;
    size_t c_bytes = (size_t)M * N * elem_size;

    auto* bA = g_dev->newBuffer(a_bytes, MTL::ResourceStorageModeShared);
    auto* bB = g_dev->newBuffer(b_bytes, MTL::ResourceStorageModeShared);
    auto* bC = g_dev->newBuffer(c_bytes, MTL::ResourceStorageModeShared);
    memcpy(bA->contents(), A, a_bytes);
    memcpy(bB->contents(), B, b_bytes);

    auto* bM   = g_dev->newBuffer(&M, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bN   = g_dev->newBuffer(&N, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bK   = g_dev->newBuffer(&K, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bLdA = g_dev->newBuffer(&ldA, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bLdB = g_dev->newBuffer(&ldB, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bLdC = g_dev->newBuffer(&ldC, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    bool tA = transA, tB = transB, hb = has_bias;
    auto* bTA = g_dev->newBuffer(&tA, sizeof(bool), MTL::ResourceStorageModeShared);
    auto* bTB = g_dev->newBuffer(&tB, sizeof(bool), MTL::ResourceStorageModeShared);
    auto* bHB = g_dev->newBuffer(&hb, sizeof(bool), MTL::ResourceStorageModeShared);

    size_t bias_bytes = (size_t)N * elem_size;
    auto* bBias = has_bias ? g_dev->newBuffer(bias_bytes, MTL::ResourceStorageModeShared) : nullptr;
    if (bBias && bias) memcpy(bBias->contents(), bias, bias_bytes);

    uint32_t gx = (N + 63) / 64;
    uint32_t gy = (M + 31) / 32;

    auto* cmd = g_q->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bA,    0, 0);
    enc->setBuffer(bB,    0, 1);
    enc->setBuffer(bC,    0, 2);
    enc->setBuffer(bM,    0, 3);
    enc->setBuffer(bN,    0, 4);
    enc->setBuffer(bK,    0, 5);
    enc->setBuffer(bLdA,  0, 6);
    enc->setBuffer(bLdB,  0, 7);
    enc->setBuffer(bLdC,  0, 8);
    enc->setBuffer(bTA,   0, 9);
    enc->setBuffer(bTB,   0, 10);
    enc->setBuffer(bHB,   0, 11);
    enc->setBuffer(bBias ? bBias : bC, 0, 12);
    enc->dispatchThreadgroups(MTL::Size(gx, gy, 1), MTL::Size(64, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    memcpy(C, bC->contents(), c_bytes);

    cmd->release(); pso->release();
    bA->release(); bB->release(); bC->release();
    bM->release(); bN->release(); bK->release();
    bLdA->release(); bLdB->release(); bLdC->release();
    bTA->release(); bTB->release(); bHB->release();
    if (bBias) bBias->release();
    return 0;
}

int sk_gemm_fp16(void* A, void* B, void* C, void* bias,
                 uint32_t M, uint32_t N, uint32_t K,
                 uint32_t ldA, uint32_t ldB, uint32_t ldC,
                 int transA, int transB, int has_bias) {
    return dispatch("gemm_fp16", A, B, C, bias, M, N, K, ldA, ldB, ldC, transA, transB, has_bias, 2);
}

int sk_gemm_fp8(void* A, void* B, void* C, void* bias,
                uint32_t M, uint32_t N, uint32_t K,
                uint32_t ldA, uint32_t ldB, uint32_t ldC,
                int transA, int transB, int has_bias) {
    return dispatch("gemm_fp8", A, B, C, bias, M, N, K, ldA, ldB, ldC, transA, transB, has_bias, 1);
}
