// quantize.c++ — vDSP-backed Q8_0 quantizer (load-time).
//
// Bit-exact vs gguf.quants.Q8_0. Throughput ~5 GB/s on M-series (single thread);
// ~150 ms for a 800 MB bf16 LM-head → 400 MB Q8_0 (gemma4-E2B).
//
// Ported from temp/quantization_lab/_q8_0_vdsp.c with bf16 input wrapper.

#include "quantize.h"

#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstring>

namespace {

constexpr size_t QK8 = 32;

// fp32 (bit pattern) -> fp16 (round-to-nearest-even, IEEE 754).
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

// bf16 (top half of fp32) -> fp32.
inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = ((uint32_t)b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

namespace sk {

void quantize_q8_0(const float* x, size_t n_elems, uint8_t* out) {
    const size_t nb = n_elems / QK8;
    #pragma omp parallel for if(nb > 1024)
    for (size_t i = 0; i < nb; ++i) {
        const float* b = x + i * QK8;
        float amax;
        vDSP_maxmgv(b, 1, &amax, QK8);
        float d  = amax / 127.0f;
        float id = (d == 0.0f) ? 0.0f : 1.0f / d;
        uint8_t* op = out + i * (2 + QK8);
        uint16_t dh = f32_to_f16(d);
        op[0] = dh & 0xff;
        op[1] = (uint8_t)(dh >> 8);
        float scaled[QK8];
        vDSP_vsmul(b, 1, &id, scaled, 1, QK8);
        for (int j = 0; j < QK8; ++j) {
            int32_t q = (int32_t)std::lrintf(scaled[j]);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            op[2 + j] = (uint8_t)(int8_t)q;
        }
    }
}

void quantize_q8_0_bf16(const uint16_t* x_bf16, size_t n_elems, uint8_t* out) {
    const size_t nb = n_elems / QK8;
    #pragma omp parallel for if(nb > 1024)
    for (size_t i = 0; i < nb; ++i) {
        // Materialize the 32-element block to fp32 (cheap; bf16→fp32 is a shift).
        float block[QK8];
        const uint16_t* src = x_bf16 + i * QK8;
        for (int j = 0; j < (int)QK8; ++j) {
            block[j] = bf16_to_f32(src[j]);
        }
        float amax;
        vDSP_maxmgv(block, 1, &amax, QK8);
        float d  = amax / 127.0f;
        float id = (d == 0.0f) ? 0.0f : 1.0f / d;
        uint8_t* op = out + i * (2 + QK8);
        uint16_t dh = f32_to_f16(d);
        op[0] = dh & 0xff;
        op[1] = (uint8_t)(dh >> 8);
        float scaled[QK8];
        vDSP_vsmul(block, 1, &id, scaled, 1, QK8);
        for (int j = 0; j < (int)QK8; ++j) {
            int32_t q = (int32_t)std::lrintf(scaled[j]);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            op[2 + j] = (uint8_t)(int8_t)q;
        }
    }
}

}  // namespace sk
