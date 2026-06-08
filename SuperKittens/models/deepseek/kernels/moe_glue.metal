// moe_glue.metal — DeepSeek V2-Lite MoE decode glue between mul_mv_id matvecs.
//
// mul_mv_id writes fp32 gate/up/down outputs and reads fp32 activations, so the
// SwiGLU and the weighted scatter-add that bracket the down-projection run in
// fp32 here.

#include <metal_stdlib>
using namespace metal;

// SwiGLU over routed-expert slots: mid[s, j] = silu(gate[s, j]) * up[s, j].
// gate/up/mid are [S, n_int] fp32 where S = top_k (decode T=1).
[[host_name("deepseek_moe_swiglu_f32")]]
kernel void deepseek_moe_swiglu_f32(
        device const float * gate [[buffer(0)]],
        device const float * up   [[buffer(1)]],
        device       float * mid  [[buffer(2)]],
        constant uint &      n     [[buffer(3)]],   // S * n_int
        uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const float g = gate[gid];
    const float silu = g / (1.0f + exp(-g));
    mid[gid] = silu * up[gid];
}

// Weighted scatter-add over T tokens: out[t,d] = residual[t,d] + sum_s score[t,s]*down[t,s,d].
// down is [T, S, D] fp32, score [T, S] fp16, residual/out [T, D] fp16. gid.x=d, gid.y=t.
[[host_name("deepseek_moe_scatter_add_f32")]]
kernel void deepseek_moe_scatter_add_f32(
        device const float * down     [[buffer(0)]],   // [T, S, D] fp32
        device const half  * score    [[buffer(1)]],   // [T, S] fp16
        device const half  * residual [[buffer(2)]],   // [T, D] fp16
        device       half  * out      [[buffer(3)]],   // [T, D] fp16
        constant uint &      D         [[buffer(4)]],
        constant uint &      S         [[buffer(5)]],
        constant float &     scale     [[buffer(6)]],   // routed_scaling_factor
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, t = gid.y;
    if (d >= D) return;
    const device float * down_t = down + (size_t)t * S * D;
    const device half  * score_t = score + (size_t)t * S;
    float acc = float(residual[(size_t)t * D + d]);
    for (uint s = 0; s < S; ++s) {
        acc += scale * float(score_t[s]) * down_t[(size_t)s * D + d];
    }
    out[(size_t)t * D + d] = half(acc);
}
