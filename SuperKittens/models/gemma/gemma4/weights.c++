#include "weights.h"
#include "gemma4_model.h"
#include "../../../inference/weight_store.h"
#include "../../../inference/quantize.h"
#include "../../load/safetensor/safetensor.h"
#include "../../load/gguf/gguf.h"
#include "../../../kernels/runtime_bindings.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace meow { namespace gemma4 {
struct Handle {
    sk_gemma4_config cfg;
    uint32_t         current_pos;
    ModelPSOs        psos;
    ModelWeights     weights;
    ModelBuffers     bufs;
    std::vector<LayerCache> layer_caches;
    std::vector<uint32_t> n_int_per_layer;
    std::vector<size_t>   mlp_gate_off_e;
    std::vector<size_t>   mlp_down_off_e;
    std::vector<int32_t>  kv_source_layer;
    std::vector<float>    layer_scalar_host;   // gemma4_unified per-layer output scale

    // Layout must match launcher.c++'s Handle (the pointer is reinterpret_cast
    // across both TUs); these per-layer Q8_0 byte-offset tables are populated
    // by the launcher's alloc and read here when quantizing the body weights.
    std::vector<size_t>   q_q8_off;
    std::vector<size_t>   k_q8_off;
    std::vector<size_t>   v_q8_off;
    std::vector<size_t>   out_q8_off;
    std::vector<size_t>   gate_q8_off;
    std::vector<size_t>   up_q8_off;
    std::vector<size_t>   down_q8_off;

    bool dump_enabled = false;
};
}}

namespace {

inline std::string layer_key(uint32_t L, const char* suffix) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "model.language_model.layers.%u.%s", L, suffix);
    return std::string(buf);
}

// fp32 -> bf16 (top half of word, round-to-nearest-even).
inline uint16_t fp32_bits_to_bf16(uint32_t bits) {
    if (((bits >> 23) & 0xff) == 0xff) {
        // NaN/Inf: preserve top bits.
        return (uint16_t)((bits >> 16) & 0xffffu) | (uint16_t)((bits & 0x7fffffu) ? 0x40 : 0);
    }
    uint32_t rounding_bias = 0x00007fffu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding_bias) >> 16);
}

// fp16 -> fp32 bits.
inline uint32_t fp16_to_fp32_bits(uint16_t h16) {
    uint32_t s = (uint32_t)(h16 >> 15) & 1u;
    uint32_t e = (uint32_t)(h16 >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h16 & 0x3FFu;
    if (e == 0)       return (s << 31);
    if (e == 31)      return (s << 31) | (0xFFu << 23) | (m << 13);
    return (s << 31) | ((e + 112u) << 23) | (m << 13);
}

// bf16 (top half) -> fp32 bits.
inline uint32_t bf16_to_fp32_bits(uint16_t b) {
    return ((uint32_t)b) << 16;
}

// Copy 'expect_elems' bf16 elems. If source is bf16, memcpy. If source is fp16,
// convert to bf16.
inline void copy_bf16_from(uint16_t* dst, const void* src_v, sk::Dtype dt, size_t n) {
    const uint16_t* src = (const uint16_t*)src_v;
    if (dt == sk::Dtype::BF16) {
        std::memcpy(dst, src, n * 2);
    } else if (dt == sk::Dtype::F16) {
        for (size_t i = 0; i < n; ++i) {
            uint32_t f = fp16_to_fp32_bits(src[i]);
            dst[i] = fp32_bits_to_bf16(f);
        }
    } else {
        // assume fp32 -> bf16
        const uint32_t* s32 = (const uint32_t*)src_v;
        for (size_t i = 0; i < n; ++i) dst[i] = fp32_bits_to_bf16(s32[i]);
    }
}

bool copy_into(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
               const std::string& name, size_t expect_elems)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "gemma4 weights: missing tensor '%s'\n", name.c_str()); return false; }
    const size_t elem_bytes = 2;
    const size_t expect_bytes = expect_elems * elem_bytes;
    if (v->nbytes != expect_bytes) {
        std::fprintf(stderr, "gemma4 weights: size mismatch '%s' got %zu expect %zu\n",
                     name.c_str(), v->nbytes, expect_bytes);
        return false;
    }
    char* out = (char*)dst->contents() + dst_off;
    copy_bf16_from((uint16_t*)out, v->data, v->dtype, expect_elems);
    return true;
}

bool copy_partial(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
                  const std::string& name, size_t expect_elems)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "gemma4 weights: missing tensor '%s'\n", name.c_str()); return false; }
    const size_t expect_bytes = expect_elems * 2;
    if (v->nbytes < expect_bytes) {
        std::fprintf(stderr, "gemma4 weights: tensor too small '%s' got %zu expect>=%zu\n",
                     name.c_str(), v->nbytes, expect_bytes);
        return false;
    }
    char* out = (char*)dst->contents() + dst_off;
    copy_bf16_from((uint16_t*)out, v->data, v->dtype, expect_elems);
    return true;
}

bool copy_transpose_bf16(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
                         const std::string& name, size_t out_rows, size_t out_cols)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "gemma4 weights: missing tensor '%s'\n", name.c_str()); return false; }
    if (v->nbytes != out_rows * out_cols * 2) {
        std::fprintf(stderr, "gemma4 weights: tx size mismatch '%s' got %zu expect %zu\n",
                     name.c_str(), v->nbytes, out_rows * out_cols * 2);
        return false;
    }
    const uint16_t* src = (const uint16_t*)v->data;
    uint16_t* d = (uint16_t*)((char*)dst->contents() + dst_off);
    const bool is_bf16 = (v->dtype == sk::Dtype::BF16);
    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            uint16_t s = src[j * out_rows + i];
            if (is_bf16) {
                d[i * out_cols + j] = s;
            } else {
                uint32_t f = fp16_to_fp32_bits(s);
                d[i * out_cols + j] = fp32_bits_to_bf16(f);
            }
        }
    }
    return true;
}

// bf16 -> fp32, multiply, bf16
inline void scale_bf16_inplace(uint16_t* p, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits = bf16_to_fp32_bits(p[i]);
        float f; std::memcpy(&f, &bits, 4);
        f *= scale;
        uint32_t out; std::memcpy(&out, &f, 4);
        p[i] = fp32_bits_to_bf16(out);
    }
}

// Host-quantize an HF body tensor to Q8_0 at dst[byte_off]. The tensor is
// stored HF-native [N, K] row-major; q8_0_matvec_bf16 reads that layout
// directly, so no transpose. Materializes to a bf16 staging buffer first
// (handles bf16/fp16/fp32 source) then runs the vDSP block quantizer.
bool quantize_into_q8(MTL::Buffer* dst, size_t byte_off, sk::WeightStore* store,
                      const std::string& name, size_t n_elems)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "gemma4 weights(q8): missing tensor '%s'\n", name.c_str()); return false; }
    if (v->nbytes / 2 != n_elems && v->dtype != sk::Dtype::F32) {
        // bf16/fp16 are 2 bytes/elem; fp32 is 4.
        if (!((v->dtype == sk::Dtype::F32) && (v->nbytes / 4 == n_elems))) {
            std::fprintf(stderr, "gemma4 weights(q8): size mismatch '%s' got %zu expect %zu elems\n",
                         name.c_str(), v->nbytes / 2, n_elems);
            return false;
        }
    }
    std::vector<uint16_t> staging(n_elems);
    copy_bf16_from(staging.data(), v->data, v->dtype, n_elems);
    uint8_t* out = (uint8_t*)dst->contents() + byte_off;
    sk::quantize_q8_0_bf16(staging.data(), n_elems, out);
    return true;
}

// ── GGUF K-quant → bf16 dequant (gemma4_unified loads a Q4_K_M GGUF; the HF
// bf16 safetensors are 23.9 GB and do not fit 16 GB). Block layouts match
// llama.cpp; output is bf16 (the gemma launcher's native body dtype). ──

inline float gh_fp16_to_f32(uint16_t s) {
    return (float)0 + ([&]{ uint32_t b = fp16_to_fp32_bits(s); float f; std::memcpy(&f,&b,4); return f; }());
}

void dequant_q8_0_bf16(uint16_t* dst, const uint8_t* src, size_t n) {
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 34;
        uint16_t sh; std::memcpy(&sh, p, 2);
        const float scale = gh_fp16_to_f32(sh);
        const int8_t* qs = (const int8_t*)(p + 2);
        for (int i = 0; i < 32; ++i) {
            float f = (float)qs[i] * scale;
            uint32_t fb; std::memcpy(&fb, &f, 4);
            dst[b*32+i] = fp32_bits_to_bf16(fb);
        }
    }
}

void dequant_q4_k_bf16(uint16_t* dst, const uint8_t* src, size_t n) {
    const size_t nb = n / 256;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 144;
        uint16_t dh, dminh; std::memcpy(&dh, p, 2); std::memcpy(&dminh, p+2, 2);
        const float d = gh_fp16_to_f32(dh), dmin = gh_fp16_to_f32(dminh);
        const uint8_t* sc = p + 4;
        const uint8_t* qs = p + 16;
        uint16_t* out = dst + b * 256;
        for (int j = 0; j < 8; ++j) {
            uint8_t scl, mn;
            if (j < 4) { scl = sc[j] & 63; mn = sc[j+4] & 63; }
            else { scl = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
                   mn  = (sc[j+4] >>   4) | ((sc[j  ] >> 6) << 4); }
            const float d1 = d * (float)scl, m1 = dmin * (float)mn;
            const uint8_t* qb = qs + (j/2)*32;
            const int shift = (j & 1) ? 4 : 0;
            for (int i = 0; i < 32; ++i) {
                const int q = (qb[i] >> shift) & 0x0F;
                float f = d1 * (float)q - m1;
                uint32_t fb; std::memcpy(&fb, &f, 4);
                out[j*32+i] = fp32_bits_to_bf16(fb);
            }
        }
    }
}

void dequant_q6_k_bf16(uint16_t* dst, const uint8_t* src, size_t n) {
    const size_t nb = n / 256;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 210;
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t*  sc = (const int8_t*)(p + 192);
        uint16_t dh; std::memcpy(&dh, p + 208, 2);
        const float d = gh_fp16_to_f32(dh);
        uint16_t* out = dst + b * 256;
        for (int nn = 0; nn < 256; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l   ] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l+32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l   ] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l+32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                auto put = [&](int idx, float f){ uint32_t fb; std::memcpy(&fb,&f,4); out[idx]=fp32_bits_to_bf16(fb); };
                put(nn+l+ 0, d * (float)sc[is+0] * (float)q1);
                put(nn+l+32, d * (float)sc[is+2] * (float)q2);
                put(nn+l+64, d * (float)sc[is+4] * (float)q3);
                put(nn+l+96, d * (float)sc[is+6] * (float)q4);
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
}

// Dequant any supported GGUF tensor to a bf16 staging vector (n_elems entries).
bool gguf_to_bf16(std::vector<uint16_t>& out, const sk::TensorView* v, size_t n_elems) {
    if (!v) return false;
    out.resize(n_elems);
    switch (v->dtype) {
        case sk::Dtype::BF16: std::memcpy(out.data(), v->data, n_elems*2); return true;
        case sk::Dtype::F16:  copy_bf16_from(out.data(), v->data, v->dtype, n_elems); return true;
        case sk::Dtype::F32:  copy_bf16_from(out.data(), v->data, v->dtype, n_elems); return true;
        case sk::Dtype::Q8_0: dequant_q8_0_bf16(out.data(), (const uint8_t*)v->data, n_elems); return true;
        case sk::Dtype::Q4_K: dequant_q4_k_bf16(out.data(), (const uint8_t*)v->data, n_elems); return true;
        case sk::Dtype::Q6_K: dequant_q6_k_bf16(out.data(), (const uint8_t*)v->data, n_elems); return true;
        default: return false;
    }
}

}

extern "C" int sk_gemma4_load_from_store(sk_gemma4_handle* hp, sk::WeightStore* store) {
    if (!hp || !store) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    const auto& c = h->cfg;
    const size_t dm   = c.d_model;
    const size_t nh   = c.n_heads;
    const uint32_t first_kv_shared = (c.num_kv_shared_layers >= c.n_layers)
                                     ? 0u
                                     : (c.n_layers - c.num_kv_shared_layers);

    const size_t hd_max     = c.head_dim_global > c.head_dim_local ? c.head_dim_global : c.head_dim_local;
    const size_t n_kv_max   = c.n_kv_heads_local > c.n_kv_heads_global ? c.n_kv_heads_local : c.n_kv_heads_global;
    const size_t qkv_slots_max = nh + 2u * n_kv_max;
    const size_t qkv_layer_stride = dm * qkv_slots_max * hd_max;
    const size_t qhd_max    = nh * hd_max;
    const size_t w_out_layer_stride = qhd_max * dm;

    if (!copy_into(h->weights.w_embed, 0, store,
                   "model.language_model.embed_tokens.weight",
                   (size_t)c.vocab_size * dm)) return -10;
    if (!copy_into(h->weights.w_final_norm, 0, store,
                   "model.language_model.norm.weight", dm)) return -11;

    // Embed scale: multiply embed table by sqrt(d_model) — in bf16.
    {
        const float embed_scale = std::sqrt((float)dm);
        uint16_t* eb = (uint16_t*)h->weights.w_embed->contents();
        const size_t n = (size_t)c.vocab_size * dm;
        scale_bf16_inplace(eb, n, embed_scale);
    }

    // Q8_0 LM-head packing: gemma4 ties lm_head = embed_tokens, so when the
    // launcher allocated a Q8_0 LM-head buffer we quantize the already-scaled
    // bf16 embed table into it. dispatch_model then routes the decode-time
    // (T=1) lm_head matvec through q8_0_matvec_bf16. Logit descale stays the
    // same (still divides by sqrt(d_model)).
    if (h->weights.w_lm_head_q8) {
        const uint16_t* src = (const uint16_t*)h->weights.w_embed->contents();
        uint8_t* dst        = (uint8_t*)h->weights.w_lm_head_q8->contents();
        const size_t n_elems = (size_t)c.vocab_size * dm;
        sk::quantize_q8_0_bf16(src, n_elems, dst);

        // Embed-Q8 (opt-in, SK_GEMMA4_EMBED_Q8=1): the lm_head Q8 buffer is the
        // tied (scaled) token embed in Q8 — so once it's built we can free the
        // resident bf16 embed table (~1.3 GB at E4B). dispatch_model then runs
        // embedding_lookup_q8_bf16 (gather+dequant from w_lm_head_q8) and loops
        // the Q8 matvec for the T>1 prefill LM-head. Gated off by default.
        const char* eenv = std::getenv("SK_GEMMA4_EMBED_Q8");
        const bool embed_q8 = eenv && (eenv[0] == '1');
        if (embed_q8) {
            h->weights.w_embed->release();
            h->weights.w_embed = nullptr;
        }
    }

    // per_layer_model_projection: HF tensor shape (n_layers*ple_dim, d_model)
    // (out_features=n_layers*ple_dim, in_features=d_model). SK GEMM
    // computes `B.x_a @ W` with W laid out (d_model, n_layers*ple_dim), i.e.
    // the transpose of HF. copy_transpose_bf16 treats src as
    // (out_cols, out_rows) and writes dst as (out_rows, out_cols); so to
    // transpose HF (8960, 1536) into SK (1536, 8960), we pass
    // out_rows=d_model, out_cols=n_layers*ple_dim.
    if (c.has_ple && h->weights.w_per_layer_model_projection) {
        const size_t out_cols = (size_t)c.n_layers * c.ple_dim;
        if (!copy_transpose_bf16(h->weights.w_per_layer_model_projection, 0, store,
                                 "model.language_model.per_layer_model_projection.weight",
                                 c.d_model, out_cols)) {
            std::fprintf(stderr, "gemma4 weights: missing per_layer_model_projection\n");
        }
    }
    if (c.has_ple && h->weights.w_per_layer_projection_norm) {
        if (!copy_into(h->weights.w_per_layer_projection_norm, 0, store,
                       "model.language_model.per_layer_projection_norm.weight", c.ple_dim)) {
            std::fprintf(stderr, "gemma4 weights: missing per_layer_projection_norm\n");
        }
    }

    if (c.has_ple && (h->weights.w_ple_table || h->weights.w_ple_table_q8)) {
        const char* nm = "model.language_model.embed_tokens_per_layer.weight";
        auto* v = store->get(nm);
        if (!v) {
            std::fprintf(stderr, "gemma4 weights: PLE tensor '%s' not found; leaving zero\n", nm);
        } else {
            const size_t ple_elems = (v->dtype == sk::Dtype::F32)
                                     ? v->nbytes / 4 : v->nbytes / 2;
            const float ple_scale = std::sqrt((float)c.ple_dim);
            if (h->weights.w_ple_table) {
                uint16_t* dst = (uint16_t*)h->weights.w_ple_table->contents();
                copy_bf16_from(dst, v->data, v->dtype, ple_elems);
                scale_bf16_inplace(dst, ple_elems, ple_scale);
            } else {
                // Q8 path: stage scaled bf16 then block-quantize into w_ple_table_q8.
                std::vector<uint16_t> staging(ple_elems);
                copy_bf16_from(staging.data(), v->data, v->dtype, ple_elems);
                scale_bf16_inplace(staging.data(), ple_elems, ple_scale);
                uint8_t* dst = (uint8_t*)h->weights.w_ple_table_q8->contents();
                sk::quantize_q8_0_bf16(staging.data(), ple_elems, dst);
            }
        }
    }

    const bool body_q8 = (h->weights.w_q_q8 != nullptr);
    auto* qkv_base   = h->weights.w_qkv ? (char*)h->weights.w_qkv->contents() : nullptr;
    auto* w_out_base = h->weights.w_out ? (char*)h->weights.w_out->contents() : nullptr;

    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const bool     is_global = ((L % c.local_period) == (c.local_period - 1));
        const size_t   n_kv      = is_global ? c.n_kv_heads_global : c.n_kv_heads_local;
        const size_t   hd        = is_global ? c.head_dim_global   : c.head_dim_local;
        const size_t   Nq        = nh  * hd;
        const size_t   Nkv       = n_kv * hd;
        const size_t   qkvN      = Nq + 2 * Nkv;

        const size_t ni           = (size_t)h->n_int_per_layer[L];
        const bool   is_kv_shared = (c.num_kv_shared_layers > 0) && (L >= first_kv_shared);
        const size_t pre_off      = (size_t)L * dm * 2;
        const size_t gamma_off    = (size_t)L * hd_max * 2;
        const size_t o_off        = (size_t)L * w_out_layer_stride * 2;
        const size_t gate_off     = h->mlp_gate_off_e[L] * 2;
        const size_t down_off     = h->mlp_down_off_e[L] * 2;
        const size_t qkv_off      = (size_t)L * qkv_layer_stride * 2;

        if (!copy_into(h->weights.w_pre_attn_norm,  pre_off, store,
                       layer_key(L, "input_layernorm.weight"),           dm)) return -20;
        if (!copy_into(h->weights.w_post_attn_norm, pre_off, store,
                       layer_key(L, "post_attention_layernorm.weight"),  dm)) return -21;
        if (!copy_into(h->weights.w_pre_feedforward_layernorm,   pre_off, store,
                       layer_key(L, "pre_feedforward_layernorm.weight"), dm)) return -22;
        if (!copy_into(h->weights.w_post_feedforward_layernorm,  pre_off, store,
                       layer_key(L, "post_feedforward_layernorm.weight"),dm)) return -23;

        if (!copy_partial(h->weights.gamma_q, gamma_off, store,
                          layer_key(L, "self_attn.q_norm.weight"), hd)) return -24;
        if (!is_kv_shared) {
            if (!copy_partial(h->weights.gamma_k, gamma_off, store,
                              layer_key(L, "self_attn.k_norm.weight"), hd)) return -25;
        }

        if (body_q8) {
            // HF-native [N, K]: o_proj [dm, Nq]; gate/up [ni, dm]; down [dm, ni].
            if (!quantize_into_q8(h->weights.w_out_q8,  h->out_q8_off[L],  store,
                                  layer_key(L, "self_attn.o_proj.weight"), Nq * dm)) return -26;
            if (!quantize_into_q8(h->weights.w_gate_q8, h->gate_q8_off[L], store,
                                  layer_key(L, "mlp.gate_proj.weight"), dm * ni)) return -27;
            if (!quantize_into_q8(h->weights.w_up_q8,   h->up_q8_off[L],   store,
                                  layer_key(L, "mlp.up_proj.weight"),   dm * ni)) return -28;
            if (!quantize_into_q8(h->weights.w_down_q8, h->down_q8_off[L], store,
                                  layer_key(L, "mlp.down_proj.weight"), ni * dm)) return -29;
        } else {
            if (!copy_transpose_bf16(h->weights.w_out, o_off, store,
                                     layer_key(L, "self_attn.o_proj.weight"), Nq, dm)) return -26;

            if (!copy_transpose_bf16(h->weights.w_gate, gate_off, store,
                                     layer_key(L, "mlp.gate_proj.weight"), dm, ni)) return -27;
            if (!copy_transpose_bf16(h->weights.w_up,   gate_off, store,
                                     layer_key(L, "mlp.up_proj.weight"),   dm, ni)) return -28;
            if (!copy_transpose_bf16(h->weights.w_down, down_off, store,
                                     layer_key(L, "mlp.down_proj.weight"), ni, dm)) return -29;
        }

        if (body_q8) {
            // Q/K/V stay HF-native [N, K] (no transpose). KV-shared layers have
            // no k/v_proj tensors; their Q8 K/V slabs stay zero (the K/V matvec
            // output is discarded — KV comes from the source layer's cache).
            if (!quantize_into_q8(h->weights.w_q_q8, h->q_q8_off[L], store,
                                  layer_key(L, "self_attn.q_proj.weight"), Nq * dm)) return -30;
            if (!is_kv_shared) {
                if (!quantize_into_q8(h->weights.w_k_q8, h->k_q8_off[L], store,
                                      layer_key(L, "self_attn.k_proj.weight"), Nkv * dm)) return -32;
                if (!quantize_into_q8(h->weights.w_v_q8, h->v_q8_off[L], store,
                                      layer_key(L, "self_attn.v_proj.weight"), Nkv * dm)) return -33;
            }
        } else {
            auto* q_v = store->get(layer_key(L, "self_attn.q_proj.weight"));
            if (!q_v) {
                std::fprintf(stderr, "gemma4 weights: missing q_proj for layer %u\n", L);
                return -30;
            }
            const size_t qb = Nq  * dm * 2;
            if (q_v->nbytes != qb) {
                std::fprintf(stderr, "gemma4 weights: q_proj size mismatch layer %u\n", L);
                return -31;
            }
            uint16_t* layer = (uint16_t*)(qkv_base + qkv_off);
            auto tx_into = [&](const void* src_v, sk::Dtype dt, size_t out_rows, size_t out_cols, size_t col_off) {
                const uint16_t* s = (const uint16_t*)src_v;
                const bool is_bf16 = (dt == sk::Dtype::BF16);
                for (size_t i = 0; i < out_rows; ++i) {
                    for (size_t j = 0; j < out_cols; ++j) {
                        uint16_t x = s[j * out_rows + i];
                        if (is_bf16) {
                            layer[i * qkvN + col_off + j] = x;
                        } else {
                            uint32_t f = fp16_to_fp32_bits(x);
                            layer[i * qkvN + col_off + j] = fp32_bits_to_bf16(f);
                        }
                    }
                }
            };
            tx_into(q_v->data, q_v->dtype, dm, Nq,  0);
            if (!is_kv_shared) {
                auto* k_v = store->get(layer_key(L, "self_attn.k_proj.weight"));
                auto* v_v = store->get(layer_key(L, "self_attn.v_proj.weight"));
                if (!k_v || !v_v) {
                    std::fprintf(stderr, "gemma4 weights: missing k/v_proj for layer %u\n", L);
                    return -32;
                }
                const size_t kb = Nkv * dm * 2;
                const size_t vb = Nkv * dm * 2;
                if (k_v->nbytes != kb || v_v->nbytes != vb) {
                    std::fprintf(stderr, "gemma4 weights: kv size mismatch layer %u\n", L);
                    return -33;
                }
                tx_into(k_v->data, k_v->dtype, dm, Nkv, Nq);
                tx_into(v_v->data, v_v->dtype, dm, Nkv, Nq + Nkv);
            }
        }

        (void)w_out_base; (void)qkvN;

        if (c.has_ple) {
            const size_t ple_dim = (size_t)c.ple_dim;
            if (h->weights.w_per_layer_input_gate) {
                const size_t off = (size_t)L * ple_dim * dm * 2;
                if (!copy_transpose_bf16(h->weights.w_per_layer_input_gate, off, store,
                                         layer_key(L, "per_layer_input_gate.weight"),
                                         dm, ple_dim)) return -40;
            }
            if (h->weights.w_per_layer_projection) {
                const size_t off = (size_t)L * dm * ple_dim * 2;
                if (!copy_transpose_bf16(h->weights.w_per_layer_projection, off, store,
                                         layer_key(L, "per_layer_projection.weight"),
                                         ple_dim, dm)) return -41;
            }
            if (h->weights.w_post_per_layer_input_norm) {
                const size_t off = (size_t)L * dm * 2;
                if (!copy_into(h->weights.w_post_per_layer_input_norm, off, store,
                               layer_key(L, "post_per_layer_input_norm.weight"), dm)) return -42;
            }
            if (h->weights.w_layer_scalar) {
                auto key = layer_key(L, "layer_scalar");
                auto* v = store->get(key);
                if (v) {
                    float scalar_f = 0.f;
                    if (v->dtype == sk::Dtype::BF16) {
                        uint32_t bits = bf16_to_fp32_bits(((const uint16_t*)v->data)[0]);
                        std::memcpy(&scalar_f, &bits, 4);
                    } else if (v->dtype == sk::Dtype::F16) {
                        uint32_t bits = fp16_to_fp32_bits(((const uint16_t*)v->data)[0]);
                        std::memcpy(&scalar_f, &bits, 4);
                    } else {
                        std::memcpy(&scalar_f, v->data, 4);
                    }
                    float* dst = (float*)h->weights.w_layer_scalar->contents();
                    dst[L] = scalar_f;
                } else {
                    float* dst = (float*)h->weights.w_layer_scalar->contents();
                    dst[L] = 1.0f;
                }
            }
        }
    }

    return 0;
}

extern "C" int sk_gemma4_load_safetensors(sk_gemma4_handle* h, const char* path) {
    if (!h || !path) return -1;
    auto store = new sk::WeightStore();
    int rc = sk::load_safetensors(path, *store);
    if (rc != 0) { delete store; return rc; }
    rc = sk_gemma4_load_from_store(h, store);
    delete store;
    return rc;
}

extern "C" int sk_gemma4_load_safetensors_index(sk_gemma4_handle* h, const char* index_json_path) {
    if (!h || !index_json_path) return -1;
    auto store = new sk::WeightStore();
    int rc = sk::load_safetensors_index(index_json_path, *store);
    if (rc != 0) { delete store; return rc; }
    rc = sk_gemma4_load_from_store(h, store);
    delete store;
    return rc;
}

// ── GGUF loader for gemma4_unified (blk.N.* names, K-quant body). Writes the
// same slab buffers as sk_gemma4_load_from_store. GGUF body tensors are
// [out,in] row-major (== HF [N,K]); transpose into SK [K,N] is identical to the
// HF path. has_ple is expected 0 (unified has no PLE tensors). ──
namespace {
// Transpose a bf16 staging matrix laid out [out_cols, out_rows] (HF/GGUF [N,K])
// into dst at dst_off as [out_rows, out_cols] (SK [K,N]).
void tx_bf16_staging(uint16_t* dst, const uint16_t* src, size_t out_rows, size_t out_cols) {
    for (size_t i = 0; i < out_rows; ++i)
        for (size_t j = 0; j < out_cols; ++j)
            dst[i * out_cols + j] = src[j * out_rows + i];
}
}

extern "C" int sk_gemma4_load_gguf(sk_gemma4_handle* hp, const char* path) {
    if (!hp || !path) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    const auto& c = h->cfg;

    sk::WeightStore store;
    sk::gguf::Model gmodel;
    int rc = sk::gguf::load_gguf(path, store, &gmodel);
    if (rc != 0) { std::fprintf(stderr, "gemma4 gguf: parse failed rc=%d\n", rc); return rc; }

    const size_t dm = c.d_model;
    const size_t nh = c.n_heads;
    const size_t hd_max   = c.head_dim_global > c.head_dim_local ? c.head_dim_global : c.head_dim_local;
    const size_t n_kv_max = c.n_kv_heads_local > c.n_kv_heads_global ? c.n_kv_heads_local : c.n_kv_heads_global;
    const size_t qkv_slots_max = nh + 2u * n_kv_max;
    const size_t qkv_layer_stride = dm * qkv_slots_max * hd_max;
    const size_t w_out_layer_stride = nh * hd_max * dm;

    std::vector<uint16_t> stg, tx;

    // token_embd → w_embed (then *= sqrt(d_model)); tied lm_head packs Q8.
    {
        auto* v = store.get("token_embd.weight");
        if (!v) { std::fprintf(stderr, "gemma4 gguf: missing token_embd.weight\n"); return -10; }
        if (!gguf_to_bf16(stg, v, (size_t)c.vocab_size * dm)) return -11;
        std::memcpy(h->weights.w_embed->contents(), stg.data(), stg.size()*2);
        scale_bf16_inplace((uint16_t*)h->weights.w_embed->contents(), stg.size(), std::sqrt((float)dm));
    }
    {
        auto* v = store.get("output_norm.weight");
        if (!v) { std::fprintf(stderr, "gemma4 gguf: missing output_norm.weight\n"); return -12; }
        if (!gguf_to_bf16(stg, v, dm)) return -13;
        std::memcpy(h->weights.w_final_norm->contents(), stg.data(), dm*2);
    }
    if (h->weights.w_lm_head_q8) {
        const size_t n_elems = (size_t)c.vocab_size * dm;
        sk::quantize_q8_0_bf16((const uint16_t*)h->weights.w_embed->contents(), n_elems,
                               (uint8_t*)h->weights.w_lm_head_q8->contents());
    }

    auto* qkv_base = h->weights.w_qkv ? (char*)h->weights.w_qkv->contents() : nullptr;
    const bool body_q8 = (h->weights.w_q_q8 != nullptr);

    char nm[160];
    auto get_layer = [&](uint32_t L, const char* suf) -> const sk::TensorView* {
        std::snprintf(nm, sizeof(nm), "blk.%u.%s", L, suf);
        return store.get(nm);
    };

    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const bool   is_global = ((L % c.local_period) == (c.local_period - 1));
        const size_t n_kv = is_global ? c.n_kv_heads_global : c.n_kv_heads_local;
        const size_t hd   = is_global ? c.head_dim_global   : c.head_dim_local;
        const size_t Nq   = nh   * hd;
        const size_t Nkv  = n_kv * hd;
        const size_t qkvN = Nq + 2 * Nkv;
        const size_t ni   = (size_t)h->n_int_per_layer[L];

        const size_t pre_off   = (size_t)L * dm * 2;
        const size_t gamma_off = (size_t)L * hd_max * 2;
        const size_t o_off     = (size_t)L * w_out_layer_stride * 2;
        const size_t gate_off  = h->mlp_gate_off_e[L] * 2;
        const size_t down_off  = h->mlp_down_off_e[L] * 2;
        const size_t qkv_off   = (size_t)L * qkv_layer_stride * 2;

        // 4-norm sandwich + per-head Q/K norm gammas.
        struct { const char* suf; MTL::Buffer* buf; size_t off; size_t n; } norms[] = {
            {"attn_norm.weight",            h->weights.w_pre_attn_norm,              pre_off,   dm},
            {"post_attention_norm.weight",  h->weights.w_post_attn_norm,             pre_off,   dm},
            {"ffn_norm.weight",             h->weights.w_pre_feedforward_layernorm,  pre_off,   dm},
            {"post_ffw_norm.weight",        h->weights.w_post_feedforward_layernorm, pre_off,   dm},
            {"attn_q_norm.weight",          h->weights.gamma_q,                      gamma_off, hd},
            {"attn_k_norm.weight",          h->weights.gamma_k,                      gamma_off, hd},
        };
        for (auto& e : norms) {
            auto* v = get_layer(L, e.suf);
            if (!v) { std::fprintf(stderr, "gemma4 gguf: missing blk.%u.%s\n", L, e.suf); return -20; }
            if (!gguf_to_bf16(stg, v, e.n)) return -21;
            std::memcpy((char*)e.buf->contents() + e.off, stg.data(), e.n*2);
        }

        // layer_output_scale → host scalar.
        {
            auto* v = get_layer(L, "layer_output_scale.weight");
            float scl = 1.0f;
            if (v) {
                if (v->dtype == sk::Dtype::F32) std::memcpy(&scl, v->data, 4);
                else { std::vector<uint16_t> s1; gguf_to_bf16(s1, v, 1);
                       uint32_t b = bf16_to_fp32_bits(s1[0]); std::memcpy(&scl, &b, 4); }
            }
            h->layer_scalar_host[L] = scl;
        }

        // Body projections. GGUF [out,in] (== HF [N,K]).
        auto load_proj_tx = [&](const char* suf, MTL::Buffer* dst, size_t off, size_t N, size_t K) -> bool {
            auto* v = get_layer(L, suf);
            if (!v) { std::fprintf(stderr, "gemma4 gguf: missing blk.%u.%s\n", L, suf); return false; }
            if (!gguf_to_bf16(stg, v, N * K)) return false;
            tx.resize(N * K);
            tx_bf16_staging(tx.data(), stg.data(), K, N);  // src [N,K] -> dst [K,N]
            std::memcpy((char*)dst->contents() + off, tx.data(), N*K*2);
            return true;
        };
        auto load_proj_q8 = [&](const char* suf, MTL::Buffer* dst, size_t byte_off, size_t n_elems) -> bool {
            auto* v = get_layer(L, suf);
            if (!v) { std::fprintf(stderr, "gemma4 gguf: missing blk.%u.%s\n", L, suf); return false; }
            if (!gguf_to_bf16(stg, v, n_elems)) return false;
            sk::quantize_q8_0_bf16(stg.data(), n_elems, (uint8_t*)dst->contents() + byte_off);
            return true;
        };

        // attention_k_eq_v: gemma4_unified's full-attention (global) layers have
        // NO attn_v tensor — V shares the K projection (HF v_proj is None,
        // value_states = key_states pre-norm). The K weight doubles as the V
        // weight; the qkv_norm kernel already norms V without scale, and V is
        // never RoPE'd (kv_cache_write stores the un-rotated v_tmp). Local (SWA)
        // layers have a real attn_v.
        const char* v_src = get_layer(L, "attn_v.weight") ? "attn_v.weight" : "attn_k.weight";

        if (body_q8) {
            if (!load_proj_q8("attn_output.weight", h->weights.w_out_q8, h->out_q8_off[L], Nq * dm)) return -26;
            if (!load_proj_q8("ffn_gate.weight",    h->weights.w_gate_q8, h->gate_q8_off[L], dm * ni)) return -27;
            if (!load_proj_q8("ffn_up.weight",      h->weights.w_up_q8,   h->up_q8_off[L],   dm * ni)) return -28;
            if (!load_proj_q8("ffn_down.weight",    h->weights.w_down_q8, h->down_q8_off[L], ni * dm)) return -29;
            if (!load_proj_q8("attn_q.weight",      h->weights.w_q_q8, h->q_q8_off[L], Nq  * dm)) return -30;
            if (!load_proj_q8("attn_k.weight",      h->weights.w_k_q8, h->k_q8_off[L], Nkv * dm)) return -32;
            if (!load_proj_q8(v_src,                h->weights.w_v_q8, h->v_q8_off[L], Nkv * dm)) return -33;
        } else {
            if (!load_proj_tx("attn_output.weight", h->weights.w_out, o_off, Nq, dm)) return -26;
            if (!load_proj_tx("ffn_gate.weight",    h->weights.w_gate, gate_off, dm, ni)) return -27;
            if (!load_proj_tx("ffn_up.weight",      h->weights.w_up,   gate_off, dm, ni)) return -28;
            if (!load_proj_tx("ffn_down.weight",    h->weights.w_down, down_off, ni, dm)) return -29;
            // Q/K/V transposed into the packed qkv slab.
            auto qkv_tx = [&](const char* suf, size_t N, size_t col_off) -> bool {
                auto* v = get_layer(L, suf);
                if (!v) { std::fprintf(stderr, "gemma4 gguf: missing blk.%u.%s\n", L, suf); return false; }
                if (!gguf_to_bf16(stg, v, N * dm)) return false;
                uint16_t* layer = (uint16_t*)(qkv_base + qkv_off);
                for (size_t i = 0; i < dm; ++i)
                    for (size_t j = 0; j < N; ++j)
                        layer[i * qkvN + col_off + j] = stg[j * dm + i];
                return true;
            };
            if (!qkv_tx("attn_q.weight", Nq,  0))         return -30;
            if (!qkv_tx("attn_k.weight", Nkv, Nq))        return -32;
            if (!qkv_tx(v_src,           Nkv, Nq + Nkv))  return -33;
        }
    }
    return 0;
}

extern "C" int sk_gemma4_set_rope_tables(sk_gemma4_handle* hp,
                                         const void* cos_local, const void* sin_local,
                                         const void* cos_global, const void* sin_global)
{
    if (!hp) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    const auto& c = h->cfg;
    const size_t local_bytes  = (size_t)c.cache_max * (c.head_dim_local  / 2) * 2;
    const size_t global_bytes = (size_t)c.cache_max * (c.head_dim_global / 2) * 2;
    if (cos_local)  std::memcpy(h->weights.cos_local->contents(),  cos_local,  local_bytes);
    if (sin_local)  std::memcpy(h->weights.sin_local->contents(),  sin_local,  local_bytes);
    if (cos_global) std::memcpy(h->weights.cos_global->contents(), cos_global, global_bytes);
    if (sin_global) std::memcpy(h->weights.sin_global->contents(), sin_global, global_bytes);
    return 0;
}
