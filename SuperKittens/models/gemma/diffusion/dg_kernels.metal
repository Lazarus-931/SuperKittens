// dg_kernels.metal — DiffusionGemma family kernels.
//
// Runtime-compiled CONCATENATED AFTER kernels/gemm/gemm_mma.metal (see
// forward_metal.MetalCtx), so the skmma tile loaders and GEMM_MMA_BODY macro
// are in scope — family-only variants live here without touching the shared
// kernel tree.
//
// dg_gemm_mma_q5_0 : gemm_mma for Q5_0 weights (this GGUF's Q4_K_M mix puts
//                    ffn_down / ffn_down_exps at Q5_0 on 16 of 30 layers).
// dg_gemm_qkt_f32  : f16 GEMM with f32 C. QK^T scores need fp32 range —
//                    ggml forces GGML_PREC_F32 for kq; a half store can
//                    overflow (no 1/sqrt(d) pre-scale in this family) and
//                    costs softmax precision.
// dg_softmax_mask  : masked row softmax, f32 scores in -> f16 probs out.

// ── Q5_0 tile loader. block = 32 weights, 22 B: half d, u32 qh, 16 nibble
// bytes. w = d * (((qs nibble) | (qh bit << 4)) - 16); element wk uses qh
// bit wk in both halves (ggml dequantize_row_q5_0).
struct __attribute__((packed)) dg_q5_0_block {
    half    d;
    uint8_t qh[4];
    uint8_t qs[16];
};

inline void load_W_q5_0(threadgroup half* Ws,
                        device const uchar* W,
                        uint bc, uint k0, uint N, uint K,
                        uint lid)
{
    const uint nb = K / 32;
    const uint kb = k0 / 32;
    for (uint i = lid; i < BN * BK; i += 64) {
        const uint wr = i / BK;
        const uint wk = i % BK;
        const uint gn = bc + wr;
        half v = half(0);
        if (gn < N) {
            device const dg_q5_0_block* blk =
                (device const dg_q5_0_block*)(W + ((size_t)gn * nb + kb) * sizeof(dg_q5_0_block));
            const uint qh = (uint)blk->qh[0] | ((uint)blk->qh[1] << 8)
                          | ((uint)blk->qh[2] << 16) | ((uint)blk->qh[3] << 24);
            const uint lo = (wk < 16) ? (blk->qs[wk] & 0x0F) : (blk->qs[wk - 16] >> 4);
            const int  q  = (int)(lo | (((qh >> wk) & 1u) << 4));
            v = (half)((float)blk->d * (float)(q - 16));
        }
        Ws[wk * BN + wr] = v;
    }
}

[[host_name("dg_gemm_mma_q5_0")]]
[[kernel]]
void dg_gemm_mma_q5_0(
    device const half*  A   [[buffer(0)]],
    device const uchar* W   [[buffer(1)]],
    device half*        C   [[buffer(2)]],
    constant uint&      M   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    GEMM_MMA_BODY(load_W_q5_0)
}

// ── GEMM_MMA_BODY with a float C store (only the final cast differs).
#define DG_GEMM_MMA_BODY_F32OUT(LOAD_W)                                        \
    const uint br = gid.y * BM;                                                \
    const uint bc = gid.x * BN;                                                \
    threadgroup half As[BM * BK];                                              \
    threadgroup half Ws[BK * BN];                                              \
    const uint lid = simd * 32 + lane;                                         \
    const uint c0  = simd * MC * 8;                                            \
    simdgroup_float8x8 acc[MR][MC] = {};                                       \
    for (uint k0 = 0; k0 < K; k0 += BK) {                                      \
        load_A(As, A, br, k0, M, K, K, lid);                                   \
        LOAD_W(Ws, W, bc, k0, N, K, lid);                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        for (uint k = 0; k < BK / 8; ++k) {                                    \
            simdgroup_half8x8 a[MR];                                           \
            for (uint r = 0; r < MR; ++r)                                      \
                simdgroup_load(a[r], As + (r * 8) * BK + k * 8, BK);           \
            for (uint c = 0; c < MC; ++c) {                                    \
                simdgroup_half8x8 b;                                           \
                simdgroup_load(b, Ws + (k * 8) * BN + c0 + c * 8, BN);         \
                for (uint r = 0; r < MR; ++r)                                  \
                    simdgroup_multiply_accumulate(acc[r][c], a[r], b, acc[r][c]); \
            }                                                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    threadgroup float Cs[BM * BN];                                             \
    for (uint r = 0; r < MR; ++r)                                              \
        for (uint c = 0; c < MC; ++c)                                          \
            simdgroup_store(acc[r][c], Cs + (r * 8) * BN + c0 + c * 8, BN);    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint i = lid; i < BM * BN; i += 64) {                                 \
        const uint r = i / BN, cc = i % BN;                                    \
        const uint gr = br + r, gc = bc + cc;                                  \
        if (gr < M && gc < N) C[(size_t)gr * ldC + gc] = Cs[i];                \
    }

[[host_name("dg_gemm_qkt_f32")]]
[[kernel]]
void dg_gemm_qkt_f32(
    device const half*  A   [[buffer(0)]],
    device const half*  W   [[buffer(1)]],
    device float*       C   [[buffer(2)]],
    constant uint&      M   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    constant uint&      ldC [[buffer(6)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  simd [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    DG_GEMM_MMA_BODY_F32OUT(load_W_f16)
}

// ── Masked row softmax for the GEMM-composed unified attention.
// S    : f32 [R, ncols] scores, R = n_heads * n_tok (head-major)
// P    : f16 [R, ncols] probs out (separate buffer; feeds the @V GEMM)
// mask : f32 [n_tok, ncols] additive (0 / -inf); row r uses mask row
//        r % n_tok (one mask per layer, shared across heads; pad cols -inf)
// scale: kq scale applied before the mask (1.0 here — qk-norm)
kernel void dg_softmax_mask(
    device const float *S      [[buffer(0)]],
    device half        *P      [[buffer(1)]],
    device const float *mask   [[buffer(2)]],
    constant uint      &ncols  [[buffer(3)]],
    constant uint      &ntok   [[buffer(4)]],
    constant float     &scale  [[buffer(5)]],
    uint3  tgpig  [[threadgroup_position_in_grid]],
    uint3  tid3   [[thread_position_in_threadgroup]],
    uint3  tptg3  [[threads_per_threadgroup]])
{
    const uint tid  = tid3.x;
    const uint tptg = tptg3.x;
    const uint r = tgpig.x;
    device const float *row  = S + (size_t)r * ncols;
    device half        *prow = P + (size_t)r * ncols;
    device const float *mrow = mask + (size_t)(r % ntok) * ncols;

    threadgroup float red[256];

    float mx = -INFINITY;
    for (uint c = tid; c < ncols; c += tptg) {
        mx = max(mx, row[c] * scale + mrow[c]);
    }
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tptg / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rmax = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint c = tid; c < ncols; c += tptg) {
        sum += exp(row[c] * scale + mrow[c] - rmax);
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tptg / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < ncols; c += tptg) {
        prow[c] = (half)(exp(row[c] * scale + mrow[c] - rmax) * inv);
    }
}
