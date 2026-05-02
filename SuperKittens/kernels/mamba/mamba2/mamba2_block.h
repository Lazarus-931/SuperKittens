//
//  mamba2_block.h
//  SuperKittens — host-side Mamba-2 block dispatch
//
//  Dispatches 5 kernels from one MTLCommandBuffer:
//    1. in_proj  — GEMM + bias
//    2. conv1d   — depthwise causal Conv1D + SiLU
//    3. ssm      — selective scan recurrence
//    4. gate_norm — silu gate + RMSNorm
//    5. out_proj — GEMM + bias
//
//  Kernel names:
//    gemm_fp16       (fp16/gemm.metal)
//    conv1d_silu     (mamba/conv1d_silu.metal)
//    mamba2_ssd      (mamba/mamba2/mamba2_ssd.metal)
//    gate_norm       (mamba/gate_norm.metal)

#ifndef SUPERKITTENS_MAMBA2_BLOCK_H
#define SUPERKITTENS_MAMBA2_BLOCK_H

#include <Metal/Metal.hpp>

namespace meow {
namespace mamba2 {

struct BlockParams {
    unsigned int batch    = 1;
    unsigned int length   = 128;
    unsigned int d_model  = 128;   // D
    unsigned int expand   = 256;   // E
    unsigned int n_heads  = 4;     // H
    unsigned int d_state  = 64;    // N
    unsigned int d_conv   = 4;     // kernel width
};

// Dispatch full Mamba-2 block on a single command buffer.
// Caller provides all pre-allocated buffers and weights.
// Returns immediately after commit — caller must waitUntilCompleted.
inline void dispatch_block(
    MTL::CommandBuffer* cmd,
    MTL::ComputePipelineState* gemm,
    MTL::ComputePipelineState* conv1d,
    MTL::ComputePipelineState* ssm,
    MTL::ComputePipelineState* gate_norm,
    // Input
    MTL::Buffer* x,            // (B, L, D)
    // Weights
    MTL::Buffer* in_proj_w,    // (D, 2E + 2HN)
    MTL::Buffer* in_proj_b,    // (2E + 2HN)
    MTL::Buffer* conv_w,       // (C_in, 4)  where C_in = E + 2HN
    MTL::Buffer* conv_b,       // (C_in)
    MTL::Buffer* A_log,        // (H, N)
    MTL::Buffer* out_proj_w,   // (E, D)
    MTL::Buffer* out_proj_b,   // (D)
    // Intermediate buffers (caller allocates)
    MTL::Buffer* proj_buf,     // (B, L, 2E + 2HN)
    MTL::Buffer* conv_buf,     // (B, L, C_in)
    MTL::Buffer* ssm_buf,      // (B, L, E)
    MTL::Buffer* gate_buf,     // (B, L, E)
    // Output
    MTL::Buffer* out,          // (B, L, D)
    // Params
    const BlockParams& p,
    // Constant buffers (caller creates once)
    MTL::Buffer* cb_B, MTL::Buffer* cb_L, MTL::Buffer* cb_D,
    MTL::Buffer* cb_E, MTL::Buffer* cb_H, MTL::Buffer* cb_N,
    MTL::Buffer* cb_Ds, MTL::Buffer* cb_Dv, MTL::Buffer* cb_Cin,
    MTL::Buffer* cb_eps,
    MTL::Buffer* cb_false, MTL::Buffer* cb_true, MTL::Buffer* cb_has_bias,
    MTL::Buffer* cb_ldA_in, MTL::Buffer* cb_ldB_in, MTL::Buffer* cb_ldC_in,
    MTL::Buffer* cb_ldA_out, MTL::Buffer* cb_ldB_out, MTL::Buffer* cb_ldC_out,
    MTL::Buffer* cb_K1, MTL::Buffer* cb_K2)
{
    const uint C_in = p.expand + 2 * p.n_heads * p.d_state;

    // 1. in_proj: GEMM (B*L, D) @ (D, 2E+2HN) → (B*L, 2E+2HN) + bias
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(gemm);
        e->setBuffer(x,          0, 0);
        e->setBuffer(in_proj_w,  0, 1);
        e->setBuffer(proj_buf,   0, 2);
        e->setBuffer(cb_B,       0, 3);
        e->setBuffer(cb_ldC_in,  0, 4);
        e->setBuffer(cb_K1,      0, 5);
        e->setBuffer(cb_ldA_in,  0, 6);
        e->setBuffer(cb_ldB_in,  0, 7);
        e->setBuffer(cb_ldC_in,  0, 8);
        e->setBuffer(cb_false,   0, 9);
        e->setBuffer(cb_false,   0, 10);
        e->setBuffer(cb_has_bias,0, 11);
        e->setBuffer(in_proj_b,  0, 12);
        e->dispatchThreadgroups(
            MTL::Size((p.d_model*2+p.expand*2+p.n_heads*p.d_state*2+63)/64, (p.batch*p.length+31)/32, 1),
            MTL::Size(64, 1, 1));
        e->endEncoding();
    }

    // 2. conv1d_silu
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(conv1d);
        e->setBuffer(proj_buf, 0, 0);
        e->setBuffer(conv_w,   0, 1);
        e->setBuffer(conv_b,   0, 2);
        e->setBuffer(conv_buf, 0, 3);
        e->setBuffer(cb_B,     0, 4);
        e->setBuffer(cb_L,     0, 5);
        e->setBuffer(cb_Cin,   0, 6);
        e->dispatchThreadgroups(
            MTL::Size(p.batch, (p.length + 3) / 4, 1),
            MTL::Size(128, 1, 1));
        e->endEncoding();
    }

    // 3. SSM
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(ssm);
        e->setBuffer(conv_buf, 0, 0); // Q
        e->setBuffer(conv_buf, 0, 1); // K
        e->setBuffer(conv_buf, 0, 2); // V
        e->setBuffer(A_log,   0, 3);
        e->setBuffer(ssm_buf, 0, 4);
        e->setBuffer(cb_L,    0, 5);
        e->setBuffer(cb_Ds,   0, 6);
        e->setBuffer(cb_Dv,   0, 7);
        e->setBuffer(cb_H,    0, 8);
        e->dispatchThreadgroups(
            MTL::Size(1, p.n_heads, 1),
            MTL::Size(128, 1, 1));
        e->endEncoding();
    }

    // 4. gate_norm
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(gate_norm);
        e->setBuffer(ssm_buf,  0, 0);
        e->setBuffer(proj_buf, 0, 1); // z (gate, first E channels of projection)
        e->setBuffer(out_proj_w, 0, 2); // norm weight
        e->setBuffer(gate_buf, 0, 3);
        e->setBuffer(cb_L,     0, 4);
        e->setBuffer(cb_E,     0, 5);
        e->setBuffer(cb_eps,   0, 6);
        e->dispatchThreadgroups(
            MTL::Size(1, (p.length + 3) / 4, 1),
            MTL::Size(128, 1, 1));
        e->endEncoding();
    }

    // 5. out_proj: GEMM (B*L, E) @ (E, D) → (B*L, D) + bias
    {
        auto* e = cmd->computeCommandEncoder();
        e->setComputePipelineState(gemm);
        e->setBuffer(gate_buf,   0, 0);
        e->setBuffer(out_proj_w, 0, 1);
        e->setBuffer(out,        0, 2);
        e->setBuffer(cb_B,       0, 3);
        e->setBuffer(cb_D,       0, 4);
        e->setBuffer(cb_K2,      0, 5);
        e->setBuffer(cb_ldA_out, 0, 6);
        e->setBuffer(cb_ldB_out, 0, 7);
        e->setBuffer(cb_ldC_out, 0, 8);
        e->setBuffer(cb_false,   0, 9);
        e->setBuffer(cb_false,   0, 10);
        e->setBuffer(cb_has_bias,0, 11);
        e->setBuffer(out_proj_b, 0, 12);
        e->dispatchThreadgroups(
            MTL::Size((p.d_model + 63) / 64, (p.batch * p.length + 31) / 32, 1),
            MTL::Size(64, 1, 1));
        e->endEncoding();
    }
}

} // namespace mamba2
} // namespace meow

#endif
