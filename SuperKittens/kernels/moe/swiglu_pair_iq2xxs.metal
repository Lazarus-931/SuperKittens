// moe_swiglu_pair_iq2xxs.metal — fused MoE swiglu-pair matvec with IQ2_XXS
// quantized expert weights (gate + up).
//
// For each (token t, slot s), expert E = exp_ids[t, s], compute:
//     g = x[t] @ dequant(W_gate[E])     // (D,) · (D, N_int) → (N_int,)
//     u = x[t] @ dequant(W_up[E])
//     out[t, s, n] = silu(g[n]) * u[n]
//
// Layout:
//   x       : (T, D)                                 fp16
//   W_gate  : (E, N_int, D/256) block_iq2_xxs        IQ2_XXS, 66 B / 256-wt block
//   W_up    : (E, N_int, D/256) block_iq2_xxs
//   exp_ids : (T, top_k)                             int32
//   out     : (T, top_k, N_int)                      fp16
//
// Grid: (ceil(N_int / COLS_PER_TG), T*top_k, 1). TG: 256 threads / 8 simdgroups.
// Each simdgroup processes COLS_PER_SG=2 output cols; uses simd_sum for reduction.

#include <metal_stdlib>
using namespace metal;

#define QK_K 256

struct block_iq2_xxs {
    half   d;
    ushort qs[QK_K/8];   // 32 ushorts = 8 sub-blocks of 4 ushorts
};

constant constexpr uint COLS_PER_SG = 2;
constant constexpr uint N_SG        = 8;
constant constexpr uint COLS_PER_TG = COLS_PER_SG * N_SG;   // 16
constant constexpr uint TGSZ        = 32 * N_SG;            // 256

static constant uchar kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

// Compute dot(x[k0..k0+256], dequant(block)) for a single 256-weight IQ2_XXS
// block — using ALL 32 lanes of the simdgroup. 32 = 8 sub-blocks * 4 `l` chunks.
// Per-lane partial; caller does simd_sum.
static inline float iq2xxs_block_dot_lane(
    device const block_iq2_xxs* blk,
    threadgroup const half* xs,        // pointer to xs[k0]
    ushort lane)
{
    const ushort ib = lane >> 2;           // 0..7  sub-block
    const ushort l  = lane & 3;            // 0..3  l-chunk
    device const uint16_t* q2 = blk->qs + 4 * ib;
    const float db = (float)blk->d;

    device const uchar* aux8 = (device const uchar*)q2;
    const uint32_t aux32 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    const float d = db * (0.5f + (float)(aux32 >> 28));

    threadgroup const half* y4 = xs + (uint)ib * 32 + l * 8;
    constant uchar* grid = (constant uchar*)(iq2xxs_grid + aux8[l]);
    const uchar signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
    float sum = 0.f;
    for (short j = 0; j < 8; ++j) {
        const float s = (signs & kmask_iq2xs[j]) ? -1.f : 1.f;
        sum += (float)y4[j] * (float)grid[j] * s;
    }
    return d * sum;
}

[[host_name("moe_swiglu_pair_iq2xxs")]]
[[kernel, max_total_threads_per_threadgroup(TGSZ)]]
void moe_swiglu_pair_iq2xxs(
    device const half*    x         [[buffer(0)]],
    device const char*    W_gate    [[buffer(1)]],   // block_iq2_xxs bytes
    device const char*    W_up      [[buffer(2)]],
    device const int*     exp_ids   [[buffer(3)]],
    device half*          out       [[buffer(4)]],
    constant uint&        T         [[buffer(5)]],
    constant uint&        top_k     [[buffer(6)]],
    constant uint&        D         [[buffer(7)]],
    constant uint&        N_int     [[buffer(8)]],
    uint3   gid                     [[threadgroup_position_in_grid]],
    ushort  sg                      [[simdgroup_index_in_threadgroup]],
    ushort  lane                    [[thread_index_in_simdgroup]])
{
    const uint t_slot   = gid.y;
    const uint t        = t_slot / top_k;
    const uint slot     = t_slot % top_k;
    const uint col_base = gid.x * COLS_PER_TG;
    if (t >= T) return;

    const int  eid = exp_ids[t * top_k + slot];
    const uint nb  = D / QK_K;                          // 256-weight blocks per row
    const uint base_blk = (uint)eid * N_int * nb;       // row-blocks offset for expert

    device const block_iq2_xxs* Wg = (device const block_iq2_xxs*)W_gate + base_blk;
    device const block_iq2_xxs* Wu = (device const block_iq2_xxs*)W_up   + base_blk;

    const uint tid = sg * 32 + lane;

    // We stage 256 (= QK_K) activations at a time into shared memory.
    threadgroup half xs[QK_K * 2];  // double buffer not needed; single is fine.

    float g_acc[COLS_PER_SG] = {0.f, 0.f};
    float u_acc[COLS_PER_SG] = {0.f, 0.f};

    for (uint kb = 0; kb < nb; ++kb) {
        // Coop-load x[t, kb*256 : kb*256+256] into xs.
        for (uint i = tid; i < QK_K; i += TGSZ) {
            xs[i] = x[t * D + kb * QK_K + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint cs = 0; cs < COLS_PER_SG; ++cs) {
            const uint col = col_base + sg * COLS_PER_SG + cs;
            if (col >= N_int) continue;

            device const block_iq2_xxs* bg = Wg + col * nb + kb;
            device const block_iq2_xxs* bu = Wu + col * nb + kb;

            // Per-lane partial: lanes 0..7 cover the 8 sub-blocks.
            const float gp = iq2xxs_block_dot_lane(bg, xs, lane);
            const float up = iq2xxs_block_dot_lane(bu, xs, lane);

            g_acc[cs] += gp;
            u_acc[cs] += up;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce + activate + write.
    for (uint cs = 0; cs < COLS_PER_SG; ++cs) {
        const uint col = col_base + sg * COLS_PER_SG + cs;
        if (col >= N_int) continue;
        const float g = simd_sum(g_acc[cs]) * 0.25f;
        const float u = simd_sum(u_acc[cs]) * 0.25f;
        if (lane == 0) {
            const float silu_g = g / (1.f + metal::precise::exp(-g));
            out[t_slot * N_int + col] = half(silu_g * u);
        }
    }
}
