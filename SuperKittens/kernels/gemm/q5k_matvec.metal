// q5k_matvec.metal — Q5_K weight × fp16 activation matvec (decode-time, M=1).
//
// Fit-enabler, not a decode-speed win: K-quant recipes bump a few early-layer
// attn_v / ffn_down tensors to Q5_K (llama.cpp "use_more_bits"). The split-V
// dispatch has no fp16 path, so V must be a native-kernel dtype — this keeps
// those Q5_K tensors packed (176 B / 256 ≈ 5.5 bpw) instead of erroring or
// host-dequant'ing to fp16. Like q3k, it exists to fit, not to win tok/s.
//
// Block layout (GGML block_q5_K, 176 B / 256 weights; per llama.cpp
// dequantize_row_q5_K, validated against gguf.quants Q5_K dequant):
//     half  d            // super-block scale for the 6-bit sub-scales
//     half  dmin         // super-block scale for the 6-bit sub-mins
//     uint8 scales[12]   // 8×(6-bit scale, 6-bit min), packed (get_scale_min_k4)
//     uint8 qh[32]       // 5th (high) bit of each of the 256 quants
//     uint8 qs[128]      // low 4 bits, 2 quants per byte
//   q = (qs_nibble) | (((qh >> bit) & 1) << 4) ; w = d*sc*q - dmin*m
//
// ggml processes 4 chunks of 64 (j=0,64,128,192), ql+=32 / is+=2 / qh-bit<<2
// each chunk. So lane l in [0,32) owns 8 taps; for chunk c in [0,4):
//   y[l + 64c]      : qs[l+32c]&0xF , scale/min (2c+0) , qh bit (2c+0)
//   y[l + 64c + 32] : qs[l+32c]>>4  , scale/min (2c+1) , qh bit (2c+1)
//   element = (low4 | (qh_bit<<4)) ; w = d*sc*element - dmin*min.
//
// Binding + dispatch mirror q4k/q6k/q3k_matvec so the launcher picks the PSO by
// dtype with one uniform call shape: NW=32 lanes, NSG=4 simdgroups tile K,
// NR0=2 rows/TG.
//   Bindings: 0 B act fp16[K]  1 A weights q5_K[N,K]  2 C out fp16[N]  3 uint K
//             4 uint N

#include <metal_stdlib>
using namespace metal;

#ifndef QK_K
#define QK_K 256
#endif

struct __attribute__((packed)) block_q5_K {
    half    d;
    half    dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];   // 32
    uint8_t qs[QK_K / 2];   // 128
};

constant constexpr short Q5_NW  = 32;
constant constexpr short Q5_NSG = 4;
constant constexpr short Q5_NR0 = 2;

// llama.cpp get_scale_min_k4: unpack the j-th 6-bit (scale, min) pair from the
// 12 packed scale bytes. Scalar byte loads — scales sits at struct offset 4 with
// a 176 B stride, so any wide load there would be misaligned for odd blocks.
static inline void get_scale_min_k4(int j, device const uint8_t *q, thread uint8_t &d, thread uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

kernel void q5k_matvec(
    device const half       *B      [[buffer(0)]],
    device const uchar      *A_raw  [[buffer(1)]],
    device       half       *C      [[buffer(2)]],
    constant uint            &K     [[buffer(3)]],
    constant uint            &N     [[buffer(4)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb   = K / QK_K;
    const uint row0 = tgpig.x * Q5_NR0;
    const short l   = tiisg;               // each lane owns one l in [0,32)

    device const block_q5_K *ax[Q5_NR0];
    #pragma unroll
    for (short r = 0; r < Q5_NR0; ++r) {
        const uint row = row0 + r;
        ax[r] = (device const block_q5_K *)(A_raw + (size_t)row * nb * sizeof(block_q5_K));
    }

    float sumf[Q5_NR0] = {0.f};

    for (uint ib = sgitg; ib < nb; ib += Q5_NSG) {
        device const half *yb = B + ib * QK_K;
        const float y0 = (float)yb[l +   0];
        const float y1 = (float)yb[l +  32];
        const float y2 = (float)yb[l +  64];
        const float y3 = (float)yb[l +  96];
        const float y4 = (float)yb[l + 128];
        const float y5 = (float)yb[l + 160];
        const float y6 = (float)yb[l + 192];
        const float y7 = (float)yb[l + 224];

        const float yv[8] = {y0, y1, y2, y3, y4, y5, y6, y7};

        #pragma unroll
        for (short r = 0; r < Q5_NR0; ++r) {
            device const uint8_t *qs = ax[r][ib].qs;
            device const uint8_t *qh = ax[r][ib].qh;
            device const uint8_t *sc = ax[r][ib].scales;
            const float d    = (float)ax[r][ib].d;
            const float dmin = (float)ax[r][ib].dmin;

            const uint8_t hb = qh[l];             // all 8 high bits for this l
            float acc_d = 0.f;   // Σ sc·element
            float acc_m = 0.f;   // Σ min  (×Σ activation)

            #pragma unroll
            for (short c = 0; c < 4; ++c) {       // 4 chunks of 64
                const uint8_t b = qs[l + 32 * c];
                uint8_t sd, sm;
                const int qlo = (int)(b & 0x0F) | (int)(((hb >> (2 * c + 0)) & 1) << 4);
                const int qhi = (int)(b >>   4) | (int)(((hb >> (2 * c + 1)) & 1) << 4);
                get_scale_min_k4(2 * c + 0, sc, sd, sm);
                acc_d += yv[2 * c + 0] * (float)sd * (float)qlo; acc_m += yv[2 * c + 0] * (float)sm;
                get_scale_min_k4(2 * c + 1, sc, sd, sm);
                acc_d += yv[2 * c + 1] * (float)sd * (float)qhi; acc_m += yv[2 * c + 1] * (float)sm;
            }
            sumf[r] += d * acc_d - dmin * acc_m;
        }
    }

    threadgroup float shmem[Q5_NR0 * Q5_NSG];
    #pragma unroll
    for (short r = 0; r < Q5_NR0; ++r) {
        float v = simd_sum(sumf[r]);
        if (tiisg == 0) shmem[r * Q5_NSG + sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        #pragma unroll
        for (short r = 0; r < Q5_NR0; ++r) {
            float v = (tiisg < Q5_NSG) ? shmem[r * Q5_NSG + tiisg] : 0.0f;
            v = simd_sum(v);
            if (tiisg == 0) {
                const uint out_row = row0 + r;
                if (out_row < N) C[out_row] = (half)v;
            }
        }
    }
}
