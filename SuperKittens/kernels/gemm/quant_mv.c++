//  quant_mv.c++ — launchers for Q2_K and Q4_K fp16-activation matvecs.

#include "quant_mv.h"
#include "../runtime_bindings.h"
#include <cstring>

namespace {

#pragma pack(push, 4)
struct QMvArgs {
    int32_t D;
    int32_t N;
};
#pragma pack(pop)

static constexpr size_t Q2K_BLOCK_BYTES    = 84;
static constexpr size_t Q4K_BLOCK_BYTES    = 144;
static constexpr size_t IQ2XXS_BLOCK_BYTES = 66;

static int dispatch_quant_mv(const char* pso_name,
                             size_t block_bytes,
                             void* x, void* W, void* y,
                             uint32_t D, uint32_t N)
{
    auto* pso = sk::bindings_pso(pso_name);
    if (!pso) return -1;
    auto* dev = sk::bindings_device();
    if (D % 256) return -4;

    const size_t n_blocks_per_row = D / 256;
    const size_t xb = (size_t)D * 2;
    const size_t wb = (size_t)N * n_blocks_per_row * block_bytes;
    const size_t yb = (size_t)N * 2;

    auto* bA = dev->newBuffer(sizeof(QMvArgs), MTL::ResourceStorageModeShared);
    QMvArgs args{(int32_t)D, (int32_t)N};
    std::memcpy(bA->contents(), &args, sizeof(args));

    auto* bX = dev->newBuffer(xb, MTL::ResourceStorageModeShared);
    auto* bW = dev->newBuffer(wb, MTL::ResourceStorageModeShared);
    auto* bY = dev->newBuffer(yb, MTL::ResourceStorageModeShared);
    std::memcpy(bX->contents(), x, xb);
    std::memcpy(bW->contents(), W, wb);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bA, 0, 0);
    enc->setBuffer(bX, 0, 1);
    enc->setBuffer(bW, 0, 2);
    enc->setBuffer(bY, 0, 3);
    enc->dispatchThreadgroups(MTL::Size(N, 1, 1), MTL::Size(32, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    std::memcpy(y, bY->contents(), yb);

    cmd->release();
    bA->release(); bX->release(); bW->release(); bY->release();
    return 0;
}

}  // namespace

extern "C" int sk_gemm_q2k_mv(void* x, void* W, void* y, uint32_t D, uint32_t N) {
    return dispatch_quant_mv("gemm_q2k_mv", Q2K_BLOCK_BYTES, x, W, y, D, N);
}

extern "C" int sk_gemm_q4k_mv(void* x, void* W, void* y, uint32_t D, uint32_t N) {
    return dispatch_quant_mv("gemm_q4k_mv", Q4K_BLOCK_BYTES, x, W, y, D, N);
}

extern "C" int sk_gemm_iq2xxs_mv(void* x, void* W, void* y, uint32_t D, uint32_t N) {
    return dispatch_quant_mv("gemm_iq2xxs_mv", IQ2XXS_BLOCK_BYTES, x, W, y, D, N);
}
