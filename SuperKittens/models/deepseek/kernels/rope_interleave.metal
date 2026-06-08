#include <metal_stdlib>
using namespace metal;

struct ArgsRopeInterleave {
    int32_t  ne00, ne01, ne02, ne03;
    uint64_t nb01, nb02, nb03;
    int32_t  n_dims;
    int32_t  n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    float    mscale;
};

static inline float yarn_ramp(float low, float high, int i) {
    float y = ((float)i - low) / max(0.001f, high - low);
    return 1.0f - clamp(y, 0.0f, 1.0f);
}

static inline void rope_freq(int i, int n_dims, float base, float ext_factor,
                             float attn_factor, float beta_fast, float beta_slow,
                             int n_ctx_orig, float freq_scale,
                             thread float& cos_out, thread float& sin_out,
                             float pos, float mscale) {
    float theta = pos * pow(base, -(float)i / (float)n_dims);
    float theta_interp = freq_scale * theta;
    float theta_v = theta_interp;
    if (ext_factor != 0.0f) {
        // HF find_correction_range(truncate=True): floor(low), ceil(high),
        // clamped to [0, n_dims/2 - 1]; ramp indexed by the pair index i/2.
        float lo = floor((float)n_dims * log((float)n_ctx_orig / (beta_fast * 6.2831853f)) /
                         (2.0f * log(base)));
        float hi = ceil((float)n_dims * log((float)n_ctx_orig / (beta_slow * 6.2831853f)) /
                        (2.0f * log(base)));
        lo = max(lo, 0.0f);
        hi = min(hi, (float)(n_dims / 2 - 1));
        float ramp_mix = yarn_ramp(lo, hi, i / 2) * ext_factor;
        theta_v = theta_interp * (1.0f - ramp_mix) + theta * ramp_mix;
    }
    float a = attn_factor * mscale;
    cos_out = cos(theta_v) * a;
    sin_out = sin(theta_v) * a;
}

[[host_name("rope_interleave_f32")]]
kernel void rope_interleave_f32(
    constant ArgsRopeInterleave& args [[buffer(0)]],
    device const float*          src  [[buffer(1)]],
    device const int*            pos  [[buffer(2)]],
    device float*                dst  [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tpitg [[thread_index_in_threadgroup]])
{
    const int i1 = (int)tgpig.x;
    const int i2 = (int)tgpig.y;
    const int i3 = (int)tgpig.z;
    if (i1 >= args.ne01 || i2 >= args.ne02 || i3 >= args.ne03) return;

    const float p = (float)pos[i3 * args.ne01 + i1];
    const int   n = args.n_dims;
    const size_t row = ((size_t)i3 * args.ne02 + i2) * args.ne01 + i1;
    device const float* s = src + row * args.ne00;
    device float*       d = dst + row * args.ne00;

    // DeepSeek-V2 partial RoPE: q is [nope(ne00-n_dims) ++ pe(n_dims)] per head,
    // RoPE acts only on the pe tail (apply_rotary_emb operates on q_pe after the
    // [qk_nope, qk_rope] split). Frequencies index the pe sub-block (0-based),
    // not the absolute 192-position.
    const int n_nope = args.ne00 - n;

    // Copy the un-rotated nope prefix.
    for (int i0 = (int)tpitg; i0 < n_nope; i0 += 256) {
        d[i0] = s[i0];
    }

    // Interleaved (view_as_complex) pairs within the pe tail: (s[n_nope+2k], s[n_nope+2k+1]).
    // y0 = x0*c - x1*s ; y1 = x0*s + x1*c
    for (int r = (int)tpitg * 2; r < n; r += 256 * 2) {
        float c, sn;
        rope_freq(r, n, args.freq_base, args.ext_factor, args.attn_factor,
                  args.beta_fast, args.beta_slow, args.n_ctx_orig,
                  args.freq_scale, c, sn, p, args.mscale);
        const int i0 = n_nope + r;
        const float x0 = s[i0];
        const float x1 = s[i0 + 1];
        d[i0]     = x0 * c - x1 * sn;
        d[i0 + 1] = x0 * sn + x1 * c;
    }
}
