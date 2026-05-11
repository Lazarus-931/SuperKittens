//  rope_tail.c++ — host launcher for DS4's kernel_dsv4_rope_tail_f32.

#include "rope_tail.h"
#include "../runtime_bindings.h"
#include <cstring>

namespace {

// 1:1 with kernel_dsv4_rope_tail's `constant ds4_metal_args_dsv4_rope_tail&`.
// Field order and types must match metal/dsv4_rope.metal::ds4_metal_args_dsv4_rope_tail.
struct alignas(8) ArgsRopeTail {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_dims;
    int32_t  mode;
    int32_t  n_ctx_orig;
    int32_t  inverse;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    bool     src2;
    char     _pad[7];   // align to 8 for tail
};

}  // namespace

extern "C" int sk_deepseek_rope_tail(
    void* x, void* pos, void* freq, void* out,
    uint32_t ne03, uint32_t ne02, uint32_t ne01, uint32_t ne00,
    int32_t  n_dims, int32_t mode, int32_t n_ctx_orig,
    float    freq_base, float freq_scale, float ext_factor,
    float    attn_factor, float beta_fast, float beta_slow,
    int32_t  inverse)
{
    auto* pso = sk::bindings_pso("kernel_dsv4_rope_tail_f32");
    if (!pso) return -1;
    auto* dev = sk::bindings_device();

    const size_t nbytes = (size_t)ne03 * ne02 * ne01 * ne00 * sizeof(float);

    auto* bX   = dev->newBuffer(nbytes, MTL::ResourceStorageModeShared);
    auto* bPos = dev->newBuffer((size_t)ne02 * sizeof(int32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* bFreq = nullptr;
    if (freq) {
        bFreq = dev->newBuffer((size_t)(n_dims / 2) * sizeof(float),
                               MTL::ResourceStorageModeShared);
        std::memcpy(bFreq->contents(), freq, (size_t)(n_dims / 2) * sizeof(float));
    }
    auto* bO = dev->newBuffer(nbytes, MTL::ResourceStorageModeShared);
    std::memcpy(bX->contents(),   x,   nbytes);
    std::memcpy(bPos->contents(), pos, (size_t)ne02 * sizeof(int32_t));

    ArgsRopeTail args{};
    args.ne00 = ne00; args.ne01 = ne01; args.ne02 = ne02; args.ne03 = ne03;
    args.nb00 = sizeof(float);
    args.nb01 = ne00 * args.nb00;
    args.nb02 = ne01 * args.nb01;
    args.nb03 = ne02 * args.nb02;
    args.nb0  = args.nb00; args.nb1 = args.nb01; args.nb2 = args.nb02; args.nb3 = args.nb03;
    args.n_dims = n_dims; args.mode = mode; args.n_ctx_orig = n_ctx_orig;
    args.inverse = inverse;
    args.freq_base = freq_base; args.freq_scale = freq_scale;
    args.ext_factor = ext_factor; args.attn_factor = attn_factor;
    args.beta_fast = beta_fast; args.beta_slow = beta_slow;
    args.src2 = (freq != nullptr);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBytes(&args, sizeof(args), 0);
    enc->setBuffer(bX,   0, 1);
    enc->setBuffer(bPos, 0, 2);
    enc->setBuffer(bFreq ? bFreq : bX, 0, 3);   // src2 unused if !args.src2
    enc->setBuffer(bO,   0, 4);

    const uint32_t TGSZ = 256;
    enc->dispatchThreadgroups(MTL::Size(ne01, ne02, ne03), MTL::Size(TGSZ, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    std::memcpy(out, bO->contents(), nbytes);

    cmd->release();
    bX->release(); bPos->release(); bO->release();
    if (bFreq) bFreq->release();
    return 0;
}
