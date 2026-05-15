//
//  mamba3_block.h
//  SuperKittens — host-side Mamba-3 block dispatch
//
//  Dispatches 5 kernels from one MTLCommandBuffer:
//    1. in_proj   — GEMM + bias  → z, xBC, dt
//    2. pre_ssm   — norm + rotary → Q, K, V, A, B
//    3. ssm       — selective scan → ssm_out
//    4. post_ssm  — silu(z)*ssm_out → gated
//    5. out_proj  — GEMM + bias → output
//
//  Kernel names:
//    gemm_fp16        (gemm/fp16/gemm.metal)
//    mamba3_pre_ssm   (mamba/mamba3/pre_ssm.metal)
//    mamba3_ssm       (mamba/mamba3/mamba3_ssm.metal)
//    mamba3_post_ssm  (mamba/mamba3/post_ssm.metal)

#ifndef SUPERKITTENS_MAMBA3_BLOCK_H
#define SUPERKITTENS_MAMBA3_BLOCK_H

#include <Metal/Metal.hpp>

namespace meow {
namespace mamba3 {

struct BlockParams {
    unsigned int batch    = 1;
    unsigned int length   = 128;
    unsigned int d_model  = 128;   // D
    unsigned int expand   = 64;    // DV (gate dim = SSM output dim)
    unsigned int n_heads  = 4;     // H
    unsigned int d_state  = 64;    // DQ
    unsigned int chunk    = 32;    // CS
};

inline void dispatch_block(
    MTL::CommandBuffer* cmd,
    MTL::ComputePipelineState* gemm,
    MTL::ComputePipelineState* pre_ssm,
    MTL::ComputePipelineState* ssm,
    MTL::ComputePipelineState* post_ssm,
    // Input
    MTL::Buffer* x,
    // Weights
    MTL::Buffer* in_proj_w,  MTL::Buffer* in_proj_b,
    MTL::Buffer* norm_w,
    MTL::Buffer* out_proj_w, MTL::Buffer* out_proj_b,
    // Intermediate buffers (caller allocates)
    MTL::Buffer* proj_buf,   // (B, L, DV+2*DQ+H)
    MTL::Buffer* q_buf,      // (BH, L, DQ)
    MTL::Buffer* k_buf,      // (BH, L, DQ)
    MTL::Buffer* v_buf,      // (BH, L, DQ)
    MTL::Buffer* a_buf,      // (BH, L)
    MTL::Buffer* b_buf,      // (BH, L, DQ)
    MTL::Buffer* dt_buf,     // (BH, L)
    MTL::Buffer* ang_buf,    // (BH, L, DQ/2)
    MTL::Buffer* ssm_buf,    // (BH, L, DV)
    MTL::Buffer* gate_buf,   // (BH, L, DV)
    // Output
    MTL::Buffer* out,
    // Params + constant bufs
    const BlockParams& p,
    MTL::Buffer* cB, MTL::Buffer* cL, MTL::Buffer* cD,
    MTL::Buffer* cH, MTL::Buffer* cDQ, MTL::Buffer* cDV, MTL::Buffer* cCS, MTL::Buffer* cBH,
    MTL::Buffer* cEps,
    MTL::Buffer* cFalse, MTL::Buffer* cTrue, MTL::Buffer* cHasBias,
    MTL::Buffer* cLdA_in, MTL::Buffer* cLdB_in, MTL::Buffer* cLdC_in,
    MTL::Buffer* cLdA_out, MTL::Buffer* cLdB_out, MTL::Buffer* cLdC_out,
    MTL::Buffer* cK1, MTL::Buffer* cK2)
{
    const uint BH = p.batch * p.n_heads;
    const uint proj_dim = p.expand + 2 * p.d_state + p.n_heads;  // DV + 2*DQ + H

    // 1. in_proj: x @ in_proj_w + bias → proj_buf
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(gemm);
        e->setBuffer(x,          0, 0);
        e->setBuffer(in_proj_w,  0, 1);
        e->setBuffer(proj_buf,   0, 2);
        e->setBuffer(cB,         0, 3);
        e->setBuffer(cLdC_in,    0, 4);
        e->setBuffer(cK1,        0, 5);
        e->setBuffer(cLdA_in,    0, 6);
        e->setBuffer(cLdB_in,    0, 7);
        e->setBuffer(cLdC_in,    0, 8);
        e->setBuffer(cFalse,     0, 9);
        e->setBuffer(cFalse,     0, 10);
        e->setBuffer(cHasBias,   0, 11);
        e->setBuffer(in_proj_b,  0, 12);
        e->dispatchThreadgroups(
            MTL::Size((proj_dim + 63) / 64, (p.batch * p.length + 31) / 32, 1),
            MTL::Size(64, 1, 1));
        e->endEncoding();
    }

    // 2. pre_ssm: norm(Q/K/B) + rotary → q/k/v/a/b bufs
    //    proj_buf layout: [DV gate | DQ x_ssm | DQ B_raw | H dt]
    //    dt and angle need to be prepared from proj_buf
    //    For now: skip pre_ssm, use pre-filled q/k/v (caller must pre-fill or run pre_ssm separately)
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(pre_ssm);
        e->setBuffer(proj_buf,  0, 0);   // xBC (DV:DV+2*DQ portion, plus dt/angle from elsewhere)
        e->setBuffer(dt_buf,    0, 1);   // dt
        e->setBuffer(ang_buf,   0, 2);   // angle
        e->setBuffer(norm_w,    0, 3);   // norm weight
        e->setBuffer(q_buf,     0, 4);
        e->setBuffer(k_buf,     0, 5);
        e->setBuffer(v_buf,     0, 6);
        e->setBuffer(a_buf,     0, 7);
        e->setBuffer(b_buf,     0, 8);
        e->setBuffer(cBH,       0, 9);
        e->setBuffer(cL,        0, 10);
        e->setBuffer(cDQ,       0, 11);
        e->setBuffer(cEps,      0, 12);
        e->dispatchThreadgroups(
            MTL::Size(BH, (p.length + 3) / 4, 1),
            MTL::Size(128, 1, 1));
        e->endEncoding();
    }

    // 3. SSM
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(ssm);
        e->setBuffer(q_buf,     0, 0);
        e->setBuffer(k_buf,     0, 1);
        e->setBuffer(v_buf,     0, 2);
        e->setBuffer(a_buf,     0, 3);
        e->setBuffer(b_buf,     0, 4);
        e->setBuffer(ang_buf,   0, 5);
        e->setBuffer(ssm_buf,   0, 6);
        e->setBuffer(cL,        0, 7);
        e->setBuffer(cDQ,       0, 8);
        e->setBuffer(cDV,       0, 9);
        e->setBuffer(cCS,       0, 10);
        e->dispatchThreadgroups(
            MTL::Size(BH, 1, 1),
            MTL::Size(128, 1, 1));
        e->endEncoding();
    }

    // 4. post_ssm: gate = silu(z) * ssm_out
    //    z is first DV channels of proj_buf
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(post_ssm);
        e->setBuffer(proj_buf,  0, 0);   // z (first DV channels)
        e->setBuffer(ssm_buf,   0, 1);
        e->setBuffer(gate_buf,  0, 2);
        e->setBuffer(cL,        0, 3);
        e->setBuffer(cDV,       0, 4);
        e->dispatchThreadgroups(
            MTL::Size(BH, (p.length + 3) / 4, 1),
            MTL::Size(128, 1, 1));
        e->endEncoding();
    }

    // 5. out_proj: gate @ out_proj_w + bias → out
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(gemm);
        e->setBuffer(gate_buf,   0, 0);
        e->setBuffer(out_proj_w, 0, 1);
        e->setBuffer(out,        0, 2);
        e->setBuffer(cB,         0, 3);
        e->setBuffer(cD,         0, 4);
        e->setBuffer(cK2,        0, 5);
        e->setBuffer(cLdA_out,   0, 6);
        e->setBuffer(cLdB_out,   0, 7);
        e->setBuffer(cLdC_out,   0, 8);
        e->setBuffer(cFalse,     0, 9);
        e->setBuffer(cFalse,     0, 10);
        e->setBuffer(cHasBias,   0, 11);
        e->setBuffer(out_proj_b, 0, 12);
        e->dispatchThreadgroups(
            MTL::Size((p.d_model + 63) / 64, (p.batch * p.length + 31) / 32, 1),
            MTL::Size(64, 1, 1));
        e->endEncoding();
    }
}

} // namespace mamba3
} // namespace meow

#endif
