//  mla_v2.metal — DeepSeek V2-Lite MLA decode attention (T=1).
//
//  V2-Lite MLA is per-head decompressed attention, NOT GQA: each of the 16 Q
//  heads owns a distinct dk=192 K row and dv=128 V row in the cache (SPEC.md:23).
//  The launcher does the qk_nope(128)+qk_rope(64) concat into dk=192 before this
//  kernel runs, so the kernel only sees dense fp16 K/V cache rows. Decode T=1.
//
//  Arg struct + buffer bindings mirror kernel_flash_attn_ext_vec so this is a
//  drop-in alternate for the shared dk192/dv128 vec instance:
//    Q  fp32, row stride nb01, head stride nb02, batch stride nb03  (dk wide)
//    K  fp16, row stride nb11, head stride nb12, batch stride nb13  (dk wide)
//    V  fp16, row stride nb21, head stride nb22, batch stride nb23  (dv wide)
//    mask fp16 (-inf above diagonal), row stride nb31
//    dst fp32, [batch, q_seq, n_heads, dv]
//
//  Indexing matches the vec kernel's ne_12_2/ne_12_3 head-fan-out so n_heads<KV
//  GQA layouts would still work, but V2-Lite uses ne_12_2 == n_heads (1:1).

#ifndef SK_MLA_V2_DK
#define SK_MLA_V2_DK 192
#endif
#ifndef SK_MLA_V2_DV
#define SK_MLA_V2_DV 128
#endif

// Number of simdgroups cooperating on one (q_row, head, batch). The single-SG
// form (NSPLIT==1) walks the whole KV cache serially per head → only n_heads
// simdgroups for the whole decode, badly under-filling the GPU. NSPLIT>1 stripes
// the KV positions across NSPLIT simdgroups and merges their partial
// online-softmax states (associative, so bit-equivalent ordering aside the math
// is identical), raising occupancy to n_heads*NSPLIT simdgroups.
#ifndef SK_MLA_V2_NSPLIT
#define SK_MLA_V2_NSPLIT 4
#endif

struct ds4_metal_args_mla_decode_v2 {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
};

// NSPLIT simdgroups per (q_row, head, batch). DK4=48 / DV4=32 are both lane
// (32) -multiples-of-friendly via the strided loops below, so no padding needed.
template<short DK, short DV, short NSPLIT>
kernel void kernel_mla_decode_v2(
        constant ds4_metal_args_mla_decode_v2 & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,   // unused (kept for binding parity)
        device const char * pad,     // unused (kept for binding parity)
        device       char * dst,
        threadgroup char  * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short ACCN = DV4/NW + 1;

    const ushort iq3 = tgpig[2];   // batch
    const ushort iq2 = tgpig[1];   // head
    const ushort iq1 = tgpig[0];   // q row

    if (iq1 >= (ushort) args.ne01) {
        return;
    }

    // KV head fan-out (1:1 for V2-Lite, but keep the general form).
    const short ikv2 = iq2/(args.ne02/args.ne_12_2);
    const short ikv3 = iq3/(args.ne03/args.ne_12_3);

    device const float4 * q4 = (device const float4 *)
        (q + iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03);

    device const char * kbase = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vbase = v + ikv2*args.nb22 + ikv3*args.nb23;

    device const half * pm = (device const half *)
        (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

    // Online-softmax running state; per-lane partial V accumulators.
    float M = -FLT_MAX/2;
    float S = 0.0f;
    float4 acc[ACCN];
    FOR_UNROLL (short i = 0; i < ACCN; ++i) {
        acc[i] = 0.0f;
    }

    // Each simdgroup strides over the KV positions: sgitg, sgitg+NSPLIT, ...
    for (int ic = sgitg; ic < args.ne11; ic += NSPLIT) {
        const float msk = (float) pm[ic];
        if (msk <= -MAXHALF) {
            continue;
        }

        // QK^T for this key row: lane-strided dot over dk, then simd_sum.
        // K cache is fp16 (nb11 = DK*2); read as half4 and widen.
        device const half4 * k4 =
            (device const half4 *) (kbase + ic*args.nb11);

        float qk = 0.0f;
        for (short i = tiisg; i < DK4; i += NW) {
            qk += dot((float4) q4[i], (float4) k4[i]);
        }
        qk = simd_sum(qk);
        qk = fma(qk, args.scale, msk);

        // Online softmax update (broadcast scalars across the simdgroup).
        const float m_new = max(M, qk);
        const float ms = exp(M - m_new);
        const float vs = exp(qk - m_new);
        M = m_new;
        S = S*ms + vs;

        // Rescale running V accumulators and add vs * V_row.
        // V cache is fp16 (nb21 = DV*2); read as half4 and widen.
        device const half4 * v4 =
            (device const half4 *) (vbase + ic*args.nb21);
        short slot = 0;
        for (short i = tiisg; i < DV4; i += NW, ++slot) {
            acc[slot] = acc[slot]*ms + (float4) v4[i]*vs;
        }
    }

    // ── Cross-simdgroup merge of the NSPLIT partial (M, S, acc) states ──
    // Single-SG fast path: no merge, no barrier, no shared memory traffic.
    if (NSPLIT == 1) {
        if (S != 0.0f) {
            const float inv = 1.0f/S;
            const int64_t rid =
                (int64_t)iq3*args.ne2*args.ne1 + iq2 + (int64_t)iq1*args.ne1;
            device float4 * dst4 = (device float4 *) dst + rid*DV4;
            short slot = 0;
            for (short i = tiisg; i < DV4; i += NW, ++slot) {
                dst4[i] = acc[slot]*inv;
            }
        } else {
            const int64_t rid =
                (int64_t)iq3*args.ne2*args.ne1 + iq2 + (int64_t)iq1*args.ne1;
            device float4 * dst4 = (device float4 *) dst + rid*DV4;
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = 0.0f;
            }
        }
        return;
    }

    // Shared scratch: NSPLIT*(M,S) scalars + NSPLIT*DV4 float4 partial accumulators.
    threadgroup float  * sM = (threadgroup float *) shmem;
    threadgroup float  * sS = sM + NSPLIT;
    threadgroup float4 * sAcc = (threadgroup float4 *)(sS + NSPLIT);

    // Lane 0 of each SG writes its M/S; the per-lane acc[] are written gathered
    // into the SG's DV4-wide row of sAcc (lane i owns dim i, i+NW, ...).
    if (tiisg == 0) { sM[sgitg] = M; sS[sgitg] = S; }
    {
        short slot = 0;
        for (short i = tiisg; i < DV4; i += NW, ++slot) {
            sAcc[(int)sgitg*DV4 + i] = acc[slot];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 merges all NSPLIT partials into a single online-softmax state and
    // writes the result. Combine is the standard flash-attn two-state merge.
    if (sgitg == 0) {
        float Mg = -FLT_MAX/2;
        FOR_UNROLL (short s = 0; s < NSPLIT; ++s) Mg = max(Mg, sM[s]);

        float Sg = 0.0f;
        float4 accg[ACCN];
        FOR_UNROLL (short i = 0; i < ACCN; ++i) accg[i] = 0.0f;

        for (short s = 0; s < NSPLIT; ++s) {
            const float w = exp(sM[s] - Mg);   // rescale this partial onto Mg
            Sg += sS[s] * w;
            short slot = 0;
            for (short i = tiisg; i < DV4; i += NW, ++slot) {
                accg[slot] += sAcc[(int)s*DV4 + i] * w;
            }
        }

        const int64_t rid =
            (int64_t)iq3*args.ne2*args.ne1 + iq2 + (int64_t)iq1*args.ne1;
        device float4 * dst4 = (device float4 *) dst + rid*DV4;
        if (Sg != 0.0f) {
            const float inv = 1.0f/Sg;
            short slot = 0;
            for (short i = tiisg; i < DV4; i += NW, ++slot) {
                dst4[i] = accg[slot]*inv;
            }
        } else {
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[i] = 0.0f;
            }
        }
    }
}

typedef decltype(kernel_mla_decode_v2<SK_MLA_V2_DK, SK_MLA_V2_DV, SK_MLA_V2_NSPLIT>) mla_decode_v2_t;

template [[host_name("kernel_mla_decode_v2_f16_dk192_dv128")]]
kernel mla_decode_v2_t kernel_mla_decode_v2<192, 128, SK_MLA_V2_NSPLIT>;
