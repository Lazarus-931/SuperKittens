//
//  mamba2_step_ref.metal
//  Reference Mamba 2 decode (single-token) selective_state_update.
//
//  Matches HF Mamba2Mixer.cuda_kernels_forward decode path
//  (transformers/models/mamba2/modeling_mamba2.py) — but as a pure Metal
//  recurrence, equivalent to the L=1 case of mamba2_ssd_ref.
//
//  Inputs are all per-step (L removed from shape):
//     x        : (B, H, P)
//     dt_raw   : (B, H)
//     A_log    : (H,)
//     B_in     : (B, G, N)
//     C_in     : (B, G, N)
//     D        : (H,)
//     dt_bias  : (H,)
//  Outputs:
//     y_out    : (B, H, P)
//     ssm_state: (B, H, P, N)  in-out, fp32
//
//  Grid: (B*H, P, 1). Threads/tg: Nstate (128). Each thread owns one (h,p,n).

#include <metal_stdlib>
using namespace metal;

[[host_name("mamba2_step_ref")]]
[[kernel]]
void mamba2_step_ref(
    device const half*  x         [[buffer(0)]],   // (B,H,P)
    device const half*  dt_raw    [[buffer(1)]],   // (B,H)
    device const half*  A_log     [[buffer(2)]],   // (H,)
    device const half*  B_in      [[buffer(3)]],   // (B,G,N)
    device const half*  C_in      [[buffer(4)]],   // (B,G,N)
    device const half*  D_in      [[buffer(5)]],   // (H,)
    device const half*  dt_bias   [[buffer(6)]],   // (H,)
    device       half*  y_out     [[buffer(7)]],   // (B,H,P)
    device       float* ssm_state [[buffer(8)]],   // (B,H,P,N) fp32
    constant uint& Batch          [[buffer(9)]],
    constant uint& H              [[buffer(10)]],
    constant uint& P              [[buffer(11)]],
    constant uint& G              [[buffer(12)]],
    constant uint& Nstate         [[buffer(13)]],
    constant float& dt_min        [[buffer(14)]],
    constant float& dt_max        [[buffer(15)]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint3 gid  [[threadgroup_position_in_grid]])
{
    const uint bh = gid.x;
    const uint p  = gid.y;
    const uint n  = lid;
    if (n >= Nstate) return;

    const uint b   = bh / H;
    const uint h   = bh % H;
    const uint hpg = H / G;
    const uint g   = h / hpg;

    const size_t st_idx =
        ((size_t)b * H + h) * (size_t)P * (size_t)Nstate
        + (size_t)p * (size_t)Nstate + (size_t)n;

    threadgroup float dt_t;
    threadgroup float dA_t;
    threadgroup float x_t;

    if (n == 0) {
        float dtr = (float)dt_raw[(size_t)b * H + h] + (float)dt_bias[h];
        float dt  = (dtr > 20.0f) ? dtr : log(1.0f + exp(dtr));
        dt   = max(dt, dt_min);
        if (dt_max < 1e30f) dt = min(dt, dt_max);
        dt_t = dt;
        dA_t = exp(dt * (-exp((float)A_log[h])));
        x_t  = (float)x[((size_t)b * H + h) * P + p];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float Bv = (float)B_in[((size_t)b * G + g) * Nstate + n];
    const float Cv = (float)C_in[((size_t)b * G + g) * Nstate + n];

    float s = ssm_state[st_idx];
    s = dA_t * s + dt_t * Bv * x_t;
    ssm_state[st_idx] = s;

    threadgroup float partial[256];
    partial[n] = Cv * s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = Nstate / 2; stride > 0; stride >>= 1) {
        if (n < stride) partial[n] += partial[n + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (n == 0) {
        float y_v = partial[0] + (float)D_in[h] * x_t;
        y_out[((size_t)b * H + h) * P + p] = (half)y_v;
    }
}
