#include "weights.h"
#include "deepseek_model.h"
#include "../../inference/weight_store.h"
#include "../load/gguf/gguf.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

namespace meow { namespace deepseek {
struct Handle {
    sk_deepseek_config cfg;
    uint32_t           current_pos;
    ModelPSOs          psos;
    ModelWeights       weights;
    ModelBuffers       bufs;
    std::vector<LayerCache>   layer_caches;
    std::vector<MTL::Buffer*> k_caches;
    std::vector<MTL::Buffer*> v_caches;
};
}}

namespace {

inline std::string blk_key(uint32_t L, const char* suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, suffix);
    return std::string(buf);
}

inline uint16_t fp32_bits_to_fp16(uint32_t f) {
    uint32_t sign = (f >> 16) & 0x8000u;
    uint32_t mant = f & 0x007fffffu;
    int32_t  exp  = (int32_t)((f >> 23) & 0xffu) - 127 + 15;
    if (((f >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000u) >> (uint32_t)(1 - exp);
        if (mant & 0x00001000u) mant += 0x00002000u;
        return (uint16_t)(sign | (mant >> 13));
    }
    if (mant & 0x00001000u) {
        mant += 0x00002000u;
        if (mant & 0x00800000u) { mant = 0; exp += 1; }
        if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

inline float fp16_bits_to_f32(uint16_t s) {
    uint32_t sign = (s & 0x8000u) << 16;
    uint32_t exp  = (s >> 10) & 0x1fu;
    uint32_t mant = s & 0x3ffu;
    float out;
    if (exp == 0) {
        if (mant == 0) { uint32_t v = sign; std::memcpy(&out, &v, 4); }
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
            mant &= 0x3ffu;
            uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
            std::memcpy(&out, &v, 4);
        }
    } else if (exp == 31) {
        uint32_t v = sign | 0x7f800000u | (mant << 13);
        std::memcpy(&out, &v, 4);
    } else {
        uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
        std::memcpy(&out, &v, 4);
    }
    return out;
}

inline uint16_t f32_to_fp16_bits(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    return fp32_bits_to_fp16(fb);
}

// Q5_0 dequant (22 B / 32 weights). Layout: half d, uint8 qh[4] (high bits),
// uint8 qs[16] (low 4 bits, 2 weights/byte). q = (lo | hi<<4) - 16.
void dequant_q5_0_to_f32(float* dst, const uint8_t* src, size_t n_elems) {
    const size_t nb = n_elems / 32;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 22;
        uint16_t dh; std::memcpy(&dh, p, 2);
        const float d = fp16_bits_to_f32(dh);
        uint32_t qh; std::memcpy(&qh, p + 2, 4);
        const uint8_t* qs = p + 6;
        for (int i = 0; i < 16; ++i) {
            const uint8_t xh0 = ((qh >> (i +  0)) << 4) & 0x10;
            const uint8_t xh1 = ((qh >> (i + 12))     ) & 0x10;
            const int q0 = (int)((qs[i] & 0x0F) | xh0) - 16;
            const int q1 = (int)((qs[i] >>   4) | xh1) - 16;
            dst[b * 32 + i]      = d * (float)q0;
            dst[b * 32 + i + 16] = d * (float)q1;
        }
    }
}

// Quantize a contiguous fp32 row to Q8_0 blocks (34 B / 32 weights).
void quantize_row_q8_0(uint8_t* dst, const float* src, size_t n_elems) {
    const size_t nb = n_elems / 32;
    for (size_t b = 0; b < nb; ++b) {
        const float* x = src + b * 32;
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
        const float d = amax / 127.f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        uint8_t* p = dst + b * 34;
        const uint16_t dh = f32_to_fp16_bits(d);
        std::memcpy(p, &dh, 2);
        int8_t* qs = (int8_t*)(p + 2);
        for (int i = 0; i < 32; ++i) {
            int q = (int)std::lround(x[i] * id);
            qs[i] = (int8_t)std::max(-127, std::min(127, q));
        }
    }
}

void dequant_q8_0_to_f32(float* dst, const uint8_t* src, size_t n_elems) {
    const size_t nb = n_elems / 32;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 34;
        uint16_t dh; std::memcpy(&dh, p, 2);
        const float scale = fp16_bits_to_f32(dh);
        const int8_t* qs = (const int8_t*)(p + 2);
        for (int i = 0; i < 32; ++i) dst[b * 32 + i] = (float)qs[i] * scale;
    }
}

void dequant_q4_k_to_f32(float* dst, const uint8_t* src, size_t n_elems) {
    const size_t nb = n_elems / 256;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 144;
        uint16_t dh, dminh; std::memcpy(&dh, p, 2); std::memcpy(&dminh, p + 2, 2);
        const float d = fp16_bits_to_f32(dh), dmin = fp16_bits_to_f32(dminh);
        const uint8_t* sc = p + 4;
        const uint8_t* qs = p + 4 + 12;
        float* out = dst + b * 256;
        for (int j = 0; j < 8; ++j) {
            uint8_t scl, mn;
            if (j < 4) { scl = sc[j] & 63; mn = sc[j + 4] & 63; }
            else {
                scl = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
                mn  = (sc[j + 4] >>   4) | ((sc[j    ] >> 6) << 4);
            }
            const float d1 = d * (float)scl, m1 = dmin * (float)mn;
            const uint8_t* qbase = qs + (j / 2) * 32;
            const int shift = (j & 1) ? 4 : 0;
            for (int i = 0; i < 32; ++i) {
                const int q = (qbase[i] >> shift) & 0x0F;
                out[j * 32 + i] = d1 * (float)q - m1;
            }
        }
    }
}

void dequant_q6_k_to_f32(float* dst, const uint8_t* src, size_t n_elems) {
    const size_t nb = n_elems / 256;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 210;
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t*  sc = (const int8_t*)(p + 128 + 64);
        uint16_t dh; std::memcpy(&dh, p + 128 + 64 + 16, 2);
        const float d = fp16_bits_to_f32(dh);
        float* out = dst + b * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[n + l +  0] = d * (float)sc[is + 0] * (float)q1;
                out[n + l + 32] = d * (float)sc[is + 2] * (float)q2;
                out[n + l + 64] = d * (float)sc[is + 4] * (float)q3;
                out[n + l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
}

// Dequantize any tensor to a flat fp32 scratch (row-major, as GGUF stores it:
// shape [out_features, in_features]).
bool dequant_to_f32(std::vector<float>& out, const sk::TensorView* v, size_t n_elems) {
    if (!v) return false;
    out.resize(n_elems);
    switch (v->dtype) {
        case sk::Dtype::F32:
            std::memcpy(out.data(), v->data, n_elems * 4); return true;
        case sk::Dtype::F16: {
            const uint16_t* s = (const uint16_t*)v->data;
            for (size_t i = 0; i < n_elems; ++i) out[i] = fp16_bits_to_f32(s[i]);
            return true;
        }
        case sk::Dtype::BF16: {
            const uint16_t* s = (const uint16_t*)v->data;
            for (size_t i = 0; i < n_elems; ++i) {
                uint32_t f = ((uint32_t)s[i]) << 16; std::memcpy(&out[i], &f, 4);
            }
            return true;
        }
        case sk::Dtype::Q8_0: dequant_q8_0_to_f32(out.data(), (const uint8_t*)v->data, n_elems); return true;
        case sk::Dtype::Q5_0: dequant_q5_0_to_f32(out.data(), (const uint8_t*)v->data, n_elems); return true;
        case sk::Dtype::Q4_K: dequant_q4_k_to_f32(out.data(), (const uint8_t*)v->data, n_elems); return true;
        case sk::Dtype::Q6_K: dequant_q6_k_to_f32(out.data(), (const uint8_t*)v->data, n_elems); return true;
        default:
            std::fprintf(stderr, "ds weights: unsupported dtype for dequant\n"); return false;
    }
}

// Read a 1-D norm/embed tensor as fp16 (no transpose). n_elems entries.
bool copy_into(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
               const std::string& name, size_t expect_fp16_bytes)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "ds weights: missing tensor '%s'\n", name.c_str()); return false; }
    const size_t n = expect_fp16_bytes / 2;
    std::vector<float> tmp;
    if (!dequant_to_f32(tmp, v, n)) return false;
    uint16_t* d = (uint16_t*)((char*)dst->contents() + dst_off);
    for (size_t i = 0; i < n; ++i) d[i] = f32_to_fp16_bits(tmp[i]);
    return true;
}

// Dequantize a [out_cols, out_rows] GGUF matrix and write its transpose
// [out_rows, out_cols] as fp16 — matching the dense GEMM's expected layout.
bool copy_transpose_fp16(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
                         const std::string& name, size_t out_rows, size_t out_cols)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "ds weights: missing tensor '%s'\n", name.c_str()); return false; }
    std::vector<float> src;   // [out_cols, out_rows] row-major
    if (!dequant_to_f32(src, v, out_rows * out_cols)) return false;
    uint16_t* d = (uint16_t*)((char*)dst->contents() + dst_off);
    for (size_t i = 0; i < out_rows; ++i)
        for (size_t j = 0; j < out_cols; ++j)
            d[i * out_cols + j] = f32_to_fp16_bits(src[j * out_rows + i]);
    return true;
}

bool is_block_quant(sk::Dtype t) {
    return t == sk::Dtype::Q2_K || t == sk::Dtype::Q4_K || t == sk::Dtype::IQ2_XXS;
}

}  // namespace

extern "C" int sk_deepseek_load_from_store(sk_deepseek_handle* hp, sk::WeightStore* store) {
    if (!hp || !store) return -1;
    auto* h = reinterpret_cast<meow::deepseek::Handle*>(hp);
    const auto& c = h->cfg;
    const size_t fp16 = 2;
    const size_t dm   = c.d_model;
    const size_t qra  = c.q_lora_rank;
    const size_t kvr  = c.kv_lora_rank;
    const size_t dkn  = c.qk_nope_dim;
    const size_t dkr  = c.qk_rope_dim;
    const size_t dv   = c.v_head_dim;
    const size_t nh   = c.n_heads;
    const size_t dk   = dkn + dkr;
    const size_t qbN  = nh * dk;
    const size_t kvbN = nh * (dkn + dv);
    const size_t kva_cols = kvr + dkr;
    const size_t ni   = c.n_int;
    const size_t sni  = c.shared_n_int;
    const size_t E    = c.n_expert;
    const bool   has_qlora  = (c.has_q_lora != 0);

    if (!copy_into(h->weights.w_embed, 0, store,
                   "token_embd.weight", (size_t)c.vocab_size * dm * fp16)) return -10;
    if (!copy_into(h->weights.w_final_norm, 0, store,
                   "output_norm.weight", dm * fp16)) return -11;
    // Untied LM head (output.weight); dequant to fp16 [vocab, d_model].
    if (store->get("output.weight")) {
        if (!copy_into(h->weights.w_lm_head, 0, store,
                       "output.weight", (size_t)c.vocab_size * dm * fp16)) return -12;
    }

    // q_in for the q_proj: q_lora_rank if present, else d_model (V2-Lite direct).
    const size_t q_in = has_qlora ? qra : dm;
    const size_t dint = c.dense_n_int ? c.dense_n_int : sni;

    // Per-layer routed-expert slab byte sizes (gate/up Q4_K, down Q8_0).
    const size_t gate_bytes_per_layer = (size_t)E * ni * (dm / 256) * 144;
    const size_t down_bytes_per_layer = (size_t)E * dm * (ni / 32) * 34;

    for (uint32_t L = 0; L < c.n_layers; ++L) {
        std::fprintf(stderr, "ds load: layer %u/%u\n", L, c.n_layers); std::fflush(stderr);
        const bool   dense = (L < (uint32_t)c.first_k_dense_replace);
        const size_t pre_off    = (size_t)L * dm * fp16;
        const size_t kvan_off   = (size_t)L * kvr * fp16;
        const size_t q_b_off    = (size_t)L * q_in * qbN * fp16;
        const size_t kv_a_off   = (size_t)L * dm * kva_cols * fp16;
        const size_t kv_b_off   = (size_t)L * kvr * kvbN * fp16;
        const size_t o_off      = (size_t)L * nh * dv * dm * fp16;
        const size_t sh_gate_off= (size_t)L * dm * sni * fp16;
        const size_t sh_down_off= (size_t)L * sni * dm * fp16;
        const size_t router_off = (size_t)L * dm * E * fp16;

        if (!copy_into(h->weights.w_pre_attn_norm, pre_off, store,
                       blk_key(L, "attn_norm.weight"), dm * fp16)) return -20;
        if (!copy_into(h->weights.w_pre_mlp_norm, pre_off, store,
                       blk_key(L, "ffn_norm.weight"), dm * fp16)) return -21;
        if (!copy_into(h->weights.w_kv_a_norm, kvan_off, store,
                       blk_key(L, "attn_kv_a_norm.weight"), kvr * fp16)) return -23;

        // Q projection. V2-Lite: direct q_proj (attn_q). V3: q_a/q_a_norm/q_b.
        if (has_qlora) {
            const size_t qan_off = (size_t)L * qra * fp16;
            const size_t q_a_off = (size_t)L * dm * qra * fp16;
            if (!copy_into(h->weights.w_q_a_norm, qan_off, store,
                           blk_key(L, "attn_q_a_norm.weight"), qra * fp16)) return -22;
            if (!copy_transpose_fp16(h->weights.w_q_a, q_a_off, store,
                           blk_key(L, "attn_q_a.weight"), qra, dm)) return -24;
            if (!copy_transpose_fp16(h->weights.w_q_b, q_b_off, store,
                           blk_key(L, "attn_q_b.weight"), qbN, qra)) return -25;
        } else {
            // attn_q: GGUF [in=dm, out=qbN] → dense GEMM wants [in=dm, out=qbN].
            if (!copy_transpose_fp16(h->weights.w_q_b, q_b_off, store,
                           blk_key(L, "attn_q.weight"), dm, qbN)) return -25;
        }

        {
            // kv_a: GEMM wants [K=d_model, N=kva_cols]; GGUF logical [kva_cols, d_model].
            const std::string nm = blk_key(L, "attn_kv_a_mqa.weight");
            auto* v = store->get(nm);
            const std::string nm2 = v ? nm : blk_key(L, "attn_kv.weight");
            if (!copy_transpose_fp16(h->weights.w_kv_a, kv_a_off, store,
                           nm2, dm, kva_cols)) return -26;
        }

        // kv_b: GGUF logical W is [kvbN, R] with the kvbN rows interleaved per
        // head as [h0_knope(128), h0_v(128), h1_knope, h1_v, ...]. kv_up_pair
        // wants two contiguous blocks w_k_up[R, n_heads*qk_nope] and
        // w_v_up[R, n_heads*v_head], so de-interleave + transpose here.
        {
            std::vector<float> src;   // [kvbN, R] row-major
            if (!dequant_to_f32(src, store->get(blk_key(L, "attn_kv_b.weight")),
                                kvbN * kvr)) return -27;
            const size_t k_out = nh * dkn, v_out = nh * dv;
            uint16_t* dst = (uint16_t*)((char*)h->weights.w_kv_b->contents() + kv_b_off);
            uint16_t* dk_up = dst;                 // [R, k_out]
            uint16_t* dv_up = dst + kvr * k_out;   // [R, v_out]
            for (size_t hh = 0; hh < nh; ++hh) {
                for (size_t j = 0; j < dkn; ++j) {       // k_nope cols
                    const size_t out_row = hh * (dkn + dv) + j;   // row in [kvbN,R]
                    const size_t kcol = hh * dkn + j;             // col in [R,k_out]
                    for (size_t r = 0; r < kvr; ++r)
                        dk_up[r * k_out + kcol] = f32_to_fp16_bits(src[out_row * kvr + r]);
                }
                for (size_t j = 0; j < dv; ++j) {        // v cols
                    const size_t out_row = hh * (dkn + dv) + dkn + j;
                    const size_t vcol = hh * dv + j;
                    for (size_t r = 0; r < kvr; ++r)
                        dv_up[r * v_out + vcol] = f32_to_fp16_bits(src[out_row * kvr + r]);
                }
            }
        }
        // o_proj: dense GEMM wants [K=nh*dv, N=d_model]; GGUF logical is
        // [out=d_model, in=nh*dv] → transpose to [nh*dv, d_model].
        if (!copy_transpose_fp16(h->weights.w_o, o_off, store,
                       blk_key(L, "attn_output.weight"), nh * dv, dm)) return -28;

        if (dense) {
            // Leading dense layer: wide gated MLP (dequant fp16) into w_dense_*.
            const size_t dg_off = (size_t)L * dm * dint * fp16;
            const size_t dd_off = (size_t)L * dint * dm * fp16;
            if (!copy_transpose_fp16(h->weights.w_dense_gate, dg_off, store,
                           blk_key(L, "ffn_gate.weight"), dm, dint)) return -40;
            if (!copy_transpose_fp16(h->weights.w_dense_up, dg_off, store,
                           blk_key(L, "ffn_up.weight"), dm, dint)) return -41;
            if (!copy_transpose_fp16(h->weights.w_dense_down, dd_off, store,
                           blk_key(L, "ffn_down.weight"), dint, dm)) return -42;
            continue;
        }

        // Shared experts (dequant to fp16) + router. gated_mlp wants
        // w_gate/up [K=d_model, N_int=sni] and w_down [N_int=sni, N=d_model];
        // GGUF gate/up logical [sni, dm], down logical [dm, sni] → transpose.
        if (!copy_transpose_fp16(h->weights.w_shared_gate, sh_gate_off, store,
                       blk_key(L, "ffn_gate_shexp.weight"), dm, sni)) return -30;
        if (!copy_transpose_fp16(h->weights.w_shared_up, sh_gate_off, store,
                       blk_key(L, "ffn_up_shexp.weight"), dm, sni)) return -31;
        if (!copy_transpose_fp16(h->weights.w_shared_down, sh_down_off, store,
                       blk_key(L, "ffn_down_shexp.weight"), sni, dm)) return -32;
        // Router W: moe_router reads W[D,N] (logit[e]=Σ_d x[d]·W[d,e]); GGUF
        // ffn_gate_inp logical [N=E, D] → transpose to [D, E].
        if (!copy_transpose_fp16(h->weights.w_router, router_off, store,
                       blk_key(L, "ffn_gate_inp.weight"), dm, E)) return -33;

        // Routed experts: gate/up Q4_K, down Q8_0 — copy raw block bytes.
        {
            auto* gv  = store->get(blk_key(L, "ffn_gate_exps.weight"));
            auto* uv  = store->get(blk_key(L, "ffn_up_exps.weight"));
            auto* dv2 = store->get(blk_key(L, "ffn_down_exps.weight"));
            if (!gv || !uv || !dv2) {
                std::fprintf(stderr, "ds weights: missing routed expert tensors L%u\n", L);
                return -34;
            }
            if (gv->dtype != sk::Dtype::Q4_K || uv->dtype != sk::Dtype::Q4_K) {
                std::fprintf(stderr, "ds weights: gate/up_exps L%u not Q4_K\n", L);
                return -35;
            }
            const size_t gate_off = (size_t)L * gate_bytes_per_layer;
            const size_t down_off = (size_t)L * down_bytes_per_layer;
            if (gv->nbytes != gate_bytes_per_layer || uv->nbytes != gate_bytes_per_layer) {
                std::fprintf(stderr, "ds weights: gate/up_exps size L%u got %zu/%zu expect %zu\n",
                             L, gv->nbytes, uv->nbytes, gate_bytes_per_layer);
                return -37;
            }
            std::memcpy((char*)h->weights.w_gate->contents() + gate_off, gv->data, gate_bytes_per_layer);
            std::memcpy((char*)h->weights.w_up->contents()   + gate_off, uv->data, gate_bytes_per_layer);

            // down_exps dtype varies per layer (Q8_0 / Q5_0 in Q4_K_M). Requantize
            // to a uniform Q8_0 so the q8_0 per-expert matvec sees one layout.
            {
                const size_t n_down = (size_t)E * dm * ni;
                std::vector<float> f32(n_down);
                bool ok = true;
                if (dv2->dtype == sk::Dtype::Q8_0)      dequant_q8_0_to_f32(f32.data(), (const uint8_t*)dv2->data, n_down);
                else if (dv2->dtype == sk::Dtype::Q5_0) dequant_q5_0_to_f32(f32.data(), (const uint8_t*)dv2->data, n_down);
                else ok = dequant_to_f32(f32, dv2, n_down);
                if (!ok) { std::fprintf(stderr, "ds weights: down_exps L%u dequant failed\n", L); return -36; }
                uint8_t* d = (uint8_t*)h->weights.w_down->contents() + down_off;
                // Each of E*dm rows (length ni) -> 34*(ni/32) bytes Q8_0.
                const size_t rows = (size_t)E * dm;
                const size_t row_bytes = (ni / 32) * 34;
                for (size_t r = 0; r < rows; ++r)
                    quantize_row_q8_0(d + r * row_bytes, f32.data() + r * ni, ni);
            }
        }
    }

    return 0;
}

extern "C" int sk_deepseek_load_gguf(sk_deepseek_handle* h, const char* path) {
    if (!h || !path) return -1;
    auto* store = new sk::WeightStore();
    int rc = sk::gguf::load_gguf(path, *store);
    if (rc != 0) { delete store; return rc; }
    rc = sk_deepseek_load_from_store(h, store);
    delete store;
    return rc;
}
