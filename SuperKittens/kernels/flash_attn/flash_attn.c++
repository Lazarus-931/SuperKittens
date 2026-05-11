//
//  flash_attn.c++ — host launcher for ds4's kernel_flash_attn_ext_vec_*.
//
//  Resolves the PSO with MTL::FunctionConstantValues (the kernel's behavioral
//  toggles like has_mask are compile-time function constants, not setBytes).
//  Mirrors the 192-byte ds4_metal_args_flash_attn_ext_vec POD 1:1.
//

#include "flash_attn.h"
#include "../runtime_bindings.h"

#include <cstring>
#include <cstdio>
#include <cmath>

namespace {

#pragma pack(push, 8)
struct ArgsFAVec {
    int32_t  ne01, ne02, ne03;
    char     _p1[4];                 // align u64 to 8
    uint64_t nb01, nb02, nb03;
    int32_t  ne11, ne_12_2, ne_12_3, ns10;
    uint64_t nb11, nb12, nb13;
    int32_t  ns20;
    char     _p2[4];                 // align u64 to 8
    uint64_t nb21, nb22, nb23;
    int32_t  ne31, ne32, ne33;
    char     _p3[4];
    uint64_t nb31, nb32, nb33;
    int32_t  ne1, ne2, ne3;
    float    scale, max_bias, m0, m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};
#pragma pack(pop)

static_assert(sizeof(ArgsFAVec) == 192, "ArgsFAVec must be 192 bytes (matches Metal struct)");

}  // namespace

extern "C" int sk_flash_attn_ext_vec(
    void* Q, void* K, void* V, void* mask, void* O,
    uint32_t B, uint32_t H, uint32_t H_kv,
    uint32_t S_q, uint32_t S_kv,
    uint32_t D_k, uint32_t D_v,
    int has_mask, float scale,
    int32_t nsg, int32_t nwg)
{
    auto* dev = sk::bindings_device();
    auto* lib = sk::bindings_library();
    if (!dev || !lib) return -1;

    char name[64];
    std::snprintf(name, sizeof(name),
                  "kernel_flash_attn_ext_vec_f16_dk%u_dv%u", D_k, D_v);

    auto* fcv = MTL::FunctionConstantValues::alloc()->init();
    bool has_mask_b   = (has_mask != 0);
    bool has_sinks_b  = false;
    bool has_bias_b   = false;
    bool has_scap_b   = false;
    bool has_kvpad_b  = false;
    int32_t ns10 = (int32_t)D_k;
    int32_t ns20 = (int32_t)D_v;

    fcv->setConstantValue(&has_mask_b,  MTL::DataTypeBool, NS::UInteger(400));
    fcv->setConstantValue(&has_sinks_b, MTL::DataTypeBool, NS::UInteger(401));
    fcv->setConstantValue(&has_bias_b,  MTL::DataTypeBool, NS::UInteger(402));
    fcv->setConstantValue(&has_scap_b,  MTL::DataTypeBool, NS::UInteger(403));
    fcv->setConstantValue(&has_kvpad_b, MTL::DataTypeBool, NS::UInteger(404));
    fcv->setConstantValue(&ns10,        MTL::DataTypeInt,  NS::UInteger(420));
    fcv->setConstantValue(&ns20,        MTL::DataTypeInt,  NS::UInteger(421));
    fcv->setConstantValue(&nsg,         MTL::DataTypeInt,  NS::UInteger(422));
    fcv->setConstantValue(&nwg,         MTL::DataTypeInt,  NS::UInteger(423));

    NS::Error* err = nullptr;
    auto* fn_name = NS::String::string(name, NS::UTF8StringEncoding);
    auto* fn = lib->newFunction(fn_name, fcv, &err);
    fcv->release();
    if (!fn) return -2;
    auto* pso = dev->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) return -3;

    // ── Buffers (Q is fp32, K/V are fp16, mask is fp16, O is fp32) ──
    const size_t qb   = (size_t)B * H    * S_q  * D_k * sizeof(float);
    const size_t kb   = (size_t)B * H_kv * S_kv * D_k * sizeof(uint16_t);
    const size_t vb   = (size_t)B * H_kv * S_kv * D_v * sizeof(uint16_t);
    const size_t mb   = has_mask ? (size_t)S_q * S_kv * sizeof(uint16_t) : 0;
    const size_t ob   = (size_t)B * S_q * H * D_v * sizeof(float);

    auto* bQ = dev->newBuffer(qb, MTL::ResourceStorageModeShared);
    auto* bK = dev->newBuffer(kb, MTL::ResourceStorageModeShared);
    auto* bV = dev->newBuffer(vb, MTL::ResourceStorageModeShared);
    MTL::Buffer* bM = mb ? dev->newBuffer(mb, MTL::ResourceStorageModeShared) : nullptr;
    auto* bO = dev->newBuffer(ob, MTL::ResourceStorageModeShared);
    std::memcpy(bQ->contents(), Q, qb);
    std::memcpy(bK->contents(), K, kb);
    std::memcpy(bV->contents(), V, vb);
    if (bM) std::memcpy(bM->contents(), mask, mb);

    // ── Args struct ──
    ArgsFAVec a{};
    a.ne01 = (int32_t)S_q;       a.ne02 = (int32_t)H;       a.ne03 = (int32_t)B;
    a.nb01 = (uint64_t)D_k * sizeof(float);
    a.nb02 = a.nb01 * S_q;
    a.nb03 = a.nb02 * H;
    a.ne11 = (int32_t)S_kv;      a.ne_12_2 = (int32_t)H_kv; a.ne_12_3 = (int32_t)B;
    a.nb11 = (uint64_t)D_k * sizeof(uint16_t);
    a.nb12 = a.nb11 * S_kv;
    a.nb13 = a.nb12 * H_kv;
    a.ns10 = (int32_t)a.nb11;          // K row-stride (bytes), per ds4 convention
    a.nb21 = (uint64_t)D_v * sizeof(uint16_t);
    a.nb22 = a.nb21 * S_kv;
    a.nb23 = a.nb22 * H_kv;
    a.ns20 = (int32_t)a.nb21;          // V row-stride (bytes)
    a.ne31 = has_mask ? (int32_t)S_q : 0;
    a.ne32 = has_mask ? 1 : 0;
    a.ne33 = has_mask ? 1 : 0;
    a.nb31 = has_mask ? (uint64_t)S_kv * sizeof(uint16_t) : 0;
    a.nb32 = has_mask ? a.nb31 * S_q : 0;
    a.nb33 = has_mask ? a.nb32 : 0;
    // Output is (B, S, H, D) — ne1, ne2, ne3 are H, S, B.
    a.ne1 = (int32_t)H; a.ne2 = (int32_t)S_q; a.ne3 = (int32_t)B;
    a.scale = scale;
    a.max_bias = 0.f; a.m0 = 1.f; a.m1 = 1.f;
    a.n_head_log2 = 0;
    a.logit_softcap = 0.f;

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBytes(&a, sizeof(a), 0);
    enc->setBuffer(bQ,         0, 1);
    enc->setBuffer(bK,         0, 2);
    enc->setBuffer(bV,         0, 3);
    enc->setBuffer(bM ? bM : bQ, 0, 4);   // mask  — unused if !has_mask
    enc->setBuffer(bQ,         0, 5);     // sinks — unused
    enc->setBuffer(bQ,         0, 6);     // pad   — unused
    enc->setBuffer(bO,         0, 7);     // dst

    // Threadgroup memory for shmem_f16 (Q tile + reductions). Allocate enough
    // for the largest dk we currently instantiate (512 → ~24 KB worth);
    // 32 KB is the M2 per-tg limit.
    enc->setThreadgroupMemoryLength(32 * 1024, 0);

    // Grid: (S_q, H, B * NWG). Threads: 32 * NSG.
    enc->dispatchThreadgroups(
        MTL::Size(S_q, H, (NS::UInteger)B * nwg),
        MTL::Size(32u * nsg, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    std::memcpy(O, bO->contents(), ob);

    cmd->release(); pso->release();
    bQ->release(); bK->release(); bV->release(); bO->release();
    if (bM) bM->release();
    return 0;
}
