// quantize_q4k.c++ — host-side bf16/fp32 → Q4_K weight quantizer (load-time).
//
// Bit-format-exact vs gguf.quants.Q4_K dequant and kernels/gemm/q4k_matvec.metal:
// 144 B / 256 weights, 6-bit packed sub-scale/sub-min pairs, 4-bit quants. Port
// of llama.cpp ggml-quants.c make_qkx2_quants + quantize_row_q4_K_ref. Used to
// quantize the gemma4 LM head so decode's bandwidth-bound head matvec runs on
// half the bytes of Q8_0 (the head is the one large-N matvec where fewer bytes
// = faster decode at the M4 ceiling).

#include "quantize.h"

#include <cmath>
#include <cstring>

namespace {

constexpr int QK_K = 256;

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t mant = x & 0x7fffff;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    if (exp >= 31) {
        if (mant && ((x >> 23 & 0xff) == 0xff)) return (uint16_t)(sign | 0x7e00);
        return (uint16_t)(sign | 0x7c00);
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t r = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (r & 1))) r += 1;
        return (uint16_t)(sign | r);
    }
    uint32_t r = mant >> 13;
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) {
        r += 1;
        if (r == 0x400) { r = 0; exp += 1; if (exp >= 31) return (uint16_t)(sign | 0x7c00); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | r);
}

// fp16 bits -> fp32, for round-trip (matches the kernel/dequant fp16 read of d/dmin).
inline float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) & 1u;
    uint32_t e = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) { bits = s << 31; }
        else { int32_t ee = 1; while ((m & 0x400u) == 0u) { m <<= 1; --ee; } m &= 0x3FFu;
               bits = (s << 31) | ((uint32_t)(ee + 112) << 23) | (m << 13); }
    } else if (e == 31) { bits = (s << 31) | (0xFFu << 23) | (m << 13); }
    else { bits = (s << 31) | ((e + 112u) << 23) | (m << 13); }
    float f; std::memcpy(&f, &bits, 4); return f;
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = ((uint32_t)b) << 16;
    float f; std::memcpy(&f, &bits, 4); return f;
}

// llama.cpp make_qkx2_quants: pick (scale, min) for one 32-element group so that
// q = round((x - min) / scale) clamped to [0, nmax] minimizes the weighted
// squared error. rmin<0, rdelta>0 sweep the min around the group's true min; the
// best scale is the least-squares fit of x onto the chosen integer quants L.
// nmax=15 for Q4_K. weights = x*x (rmse_type semantics from the reference).
float make_qkx2_quants(int n, int nmax, const float* x, const float* weights,
                       uint8_t* L, float* the_min, uint8_t* Laux,
                       float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = (int)std::nearbyintf(iscale * (x[i] - min));
        L[i] = (uint8_t)(nmax < (l > 0 ? l : 0) ? nmax : (l > 0 ? l : 0));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = (int)std::nearbyintf(iscale * (x[i] - min));
            l = (l < 0 ? 0 : (l > nmax ? nmax : l));
            Laux[i] = (uint8_t)l;
            float w = weights[i];
            sum_l  += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? std::fabs(diff) : diff * diff;
                float w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// get_scale_min_k4 packing (inverse of gguf-py Q4_K.get_scale_min): pack 8
// 6-bit sub-scales (sc[]) and 8 6-bit sub-mins (m[]) into the 12-byte scales[]:
//   j<4:  scales[j]   = sc[j] | high2(?)   ... see below
// This is the exact byte layout the kernel's sc16 extraction + gguf dequant read.
inline void pack_scales_q4k(const uint8_t* sc, const uint8_t* m, uint8_t* out) {
    for (int j = 0; j < QK_K / 32; ++j) {  // 8 sub-blocks
        if (j < 4) {
            out[j]     = sc[j] & 63;
            out[j + 4] = m[j]  & 63;
        } else {
            out[j + 4] = (sc[j] & 0xF) | ((m[j] & 0xF) << 4);
            out[j - 4] |= ((sc[j] >> 4) << 6);
            out[j - 0] |= ((m[j]  >> 4) << 6);
        }
    }
}

}  // namespace

namespace sk {

void quantize_q4_k(const float* x, size_t n_elems, uint8_t* out) {
    const size_t nb = n_elems / QK_K;
    #pragma omp parallel for if(nb > 256)
    for (size_t b = 0; b < nb; ++b) {
        const float* xb = x + b * QK_K;
        uint8_t* ob = out + b * 144;

        uint8_t L[QK_K];
        uint8_t Laux[32];
        float   weights[32];
        float   scales8[QK_K / 32];
        float   mins8[QK_K / 32];

        float max_scale = 0;
        float max_min   = 0;
        for (int j = 0; j < QK_K / 32; ++j) {
            const float* xj = xb + 32 * j;
            float sum_x2 = 0;
            for (int i = 0; i < 32; ++i) sum_x2 += xj[i] * xj[i];
            float av_x = std::sqrt(sum_x2 / 32);
            for (int i = 0; i < 32; ++i) weights[i] = av_x + std::fabs(xj[i]);
            float the_min;
            scales8[j] = make_qkx2_quants(32, 15, xj, weights, L + 32 * j,
                                          &the_min, Laux, -1.f, 0.1f, 20, false);
            mins8[j] = the_min;
            if (scales8[j] > max_scale) max_scale = scales8[j];
            if (mins8[j]   > max_min)   max_min   = mins8[j];
        }

        float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f / max_min   : 0.f;
        uint8_t sc[QK_K / 32];
        uint8_t mm[QK_K / 32];
        for (int j = 0; j < QK_K / 32; ++j) {
            int ls = (int)std::nearbyintf(inv_scale * scales8[j]);
            int lm = (int)std::nearbyintf(inv_min   * mins8[j]);
            ls = ls < 0 ? 0 : (ls > 63 ? 63 : ls);
            lm = lm < 0 ? 0 : (lm > 63 ? 63 : lm);
            sc[j] = (uint8_t)ls;
            mm[j] = (uint8_t)lm;
        }

        // Super-block d / dmin are fp16-rounded before requantizing the nibbles,
        // matching the reference (the kernel reads d/dmin back as fp16).
        float d_f    = max_scale / 63.f;
        float dmin_f = max_min   / 63.f;
        uint16_t d_h    = f32_to_f16(d_f);
        uint16_t dmin_h = f32_to_f16(dmin_f);
        std::memcpy(ob,     &d_h,    2);
        std::memcpy(ob + 2, &dmin_h, 2);
        pack_scales_q4k(sc, mm, ob + 4);

        const float d    = f16_to_f32(d_h);
        const float dmin = f16_to_f32(dmin_h);
        uint8_t* qs = ob + 16;
        std::memset(qs, 0, QK_K / 2);
        for (int j = 0; j < QK_K / 32; ++j) {
            const float dl = d * sc[j];
            const float ml = dmin * mm[j];
            if (dl == 0.f) continue;
            const float* xj = xb + 32 * j;
            uint8_t* qb = qs + (j / 2) * 32;
            const int shift = (j & 1) ? 4 : 0;
            for (int i = 0; i < 32; ++i) {
                int l = (int)std::nearbyintf((xj[i] + ml) / dl);
                l = l < 0 ? 0 : (l > 15 ? 15 : l);
                qb[i] |= (uint8_t)(l << shift);
            }
        }
    }
}

void quantize_q4_k_bf16(const uint16_t* x_bf16, size_t n_elems, uint8_t* out) {
    const size_t nb = n_elems / QK_K;
    #pragma omp parallel for if(nb > 256)
    for (size_t b = 0; b < nb; ++b) {
        float block[QK_K];
        const uint16_t* src = x_bf16 + b * QK_K;
        for (int i = 0; i < QK_K; ++i) block[i] = bf16_to_f32(src[i]);
        quantize_q4_k(block, QK_K, out + b * 144);
    }
}

}  // namespace sk
