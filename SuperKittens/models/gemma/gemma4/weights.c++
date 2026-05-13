#include "weights.h"
#include "gemma4_model.h"
#include "../../../inference/weight_store.h"
#include "../../load/safetensor/safetensor.h"

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
};
}}

namespace {

inline std::string layer_key(uint32_t L, const char* suffix) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "model.language_model.layers.%u.%s", L, suffix);
    return std::string(buf);
}

inline uint16_t fp32_bits_to_fp16(uint32_t f) {
    uint32_t sign = (f >> 16) & 0x8000u;
    uint32_t mant = f & 0x007fffffu;
    int32_t  exp  = (int32_t)((f >> 23) & 0xffu) - 127 + 15;
    if (((f >> 23) & 0xffu) == 0xffu) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
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

inline void bf16_to_fp16(uint16_t* dst, const uint16_t* src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t f = ((uint32_t)src[i]) << 16;
        dst[i] = fp32_bits_to_fp16(f);
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
    if (v->dtype == sk::Dtype::BF16) {
        bf16_to_fp16((uint16_t*)out, (const uint16_t*)v->data, expect_elems);
    } else {
        std::memcpy(out, v->data, expect_bytes);
    }
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
    if (v->dtype == sk::Dtype::BF16) {
        bf16_to_fp16((uint16_t*)out, (const uint16_t*)v->data, expect_elems);
    } else {
        std::memcpy(out, v->data, expect_bytes);
    }
    return true;
}

bool copy_transpose_fp16(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
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
                uint32_t f = ((uint32_t)s) << 16;
                d[i * out_cols + j] = fp32_bits_to_fp16(f);
            } else {
                d[i * out_cols + j] = s;
            }
        }
    }
    return true;
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

    {
        const float embed_scale = std::sqrt((float)dm);
        uint16_t* eb = (uint16_t*)h->weights.w_embed->contents();
        const size_t n = (size_t)c.vocab_size * dm;
        for (size_t i = 0; i < n; ++i) {
            uint16_t h16 = eb[i];
            uint32_t s = (uint32_t)(h16 >> 15) & 1;
            uint32_t e = (uint32_t)(h16 >> 10) & 0x1F;
            uint32_t m = (uint32_t)h16 & 0x3FF;
            uint32_t bits;
            if (e == 0)       bits = (s << 31);
            else if (e == 31) bits = (s << 31) | (0xFFu << 23) | (m << 13);
            else              bits = (s << 31) | ((e + 112u) << 23) | (m << 13);
            float f; std::memcpy(&f, &bits, 4);
            f *= embed_scale;
            eb[i] = fp32_bits_to_fp16(*(uint32_t*)&f);
        }
    }

    if (c.has_ple && h->weights.w_ple_table) {
        const char* nm = "model.language_model.embed_tokens_per_layer.weight";
        auto* v = store->get(nm);
        if (!v) {
            std::fprintf(stderr, "gemma4 weights: PLE tensor '%s' not found; leaving zero\n", nm);
        } else {
            const size_t ple_elems = v->nbytes / 2;
            uint16_t* dst = (uint16_t*)h->weights.w_ple_table->contents();
            if (v->dtype == sk::Dtype::BF16) {
                bf16_to_fp16(dst, (const uint16_t*)v->data, ple_elems);
            } else {
                std::memcpy(dst, v->data, v->nbytes);
            }
            const float ple_scale = std::sqrt((float)c.ple_dim);
            for (size_t i = 0; i < ple_elems; ++i) {
                uint16_t h16 = dst[i];
                uint32_t s = (uint32_t)(h16 >> 15) & 1;
                uint32_t e = (uint32_t)(h16 >> 10) & 0x1F;
                uint32_t m = (uint32_t)h16 & 0x3FF;
                uint32_t bits;
                if (e == 0)       bits = (s << 31);
                else if (e == 31) bits = (s << 31) | (0xFFu << 23) | (m << 13);
                else              bits = (s << 31) | ((e + 112u) << 23) | (m << 13);
                float f; std::memcpy(&f, &bits, 4);
                f *= ple_scale;
                dst[i] = fp32_bits_to_fp16(*(uint32_t*)&f);
            }
        }
    }

    auto* qkv_base   = (char*)h->weights.w_qkv->contents();
    auto* w_out_base = (char*)h->weights.w_out->contents();

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

        if (!copy_transpose_fp16(h->weights.w_out, o_off, store,
                                 layer_key(L, "self_attn.o_proj.weight"), Nq, dm)) return -26;

        if (!copy_transpose_fp16(h->weights.w_gate, gate_off, store,
                                 layer_key(L, "mlp.gate_proj.weight"), dm, ni)) return -27;
        if (!copy_transpose_fp16(h->weights.w_up,   gate_off, store,
                                 layer_key(L, "mlp.up_proj.weight"),   dm, ni)) return -28;
        if (!copy_transpose_fp16(h->weights.w_down, down_off, store,
                                 layer_key(L, "mlp.down_proj.weight"), ni, dm)) return -29;

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
                        uint32_t f = ((uint32_t)x) << 16;
                        layer[i * qkvN + col_off + j] = fp32_bits_to_fp16(f);
                    } else {
                        layer[i * qkvN + col_off + j] = x;
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

        (void)w_out_base;

        if (c.has_ple) {
            const size_t ple_dim = (size_t)c.ple_dim;
            if (h->weights.w_per_layer_input_gate) {
                const size_t off = (size_t)L * ple_dim * dm * 2;
                if (!copy_transpose_fp16(h->weights.w_per_layer_input_gate, off, store,
                                         layer_key(L, "per_layer_input_gate.weight"),
                                         ple_dim, dm)) return -40;
            }
            if (h->weights.w_per_layer_projection) {
                const size_t off = (size_t)L * dm * ple_dim * 2;
                if (!copy_transpose_fp16(h->weights.w_per_layer_projection, off, store,
                                         layer_key(L, "per_layer_projection.weight"),
                                         dm, ple_dim)) return -41;
            }
            if (h->weights.w_post_per_layer_input_norm) {
                const size_t off = (size_t)L * dm * 2;
                if (!copy_into(h->weights.w_post_per_layer_input_norm, off, store,
                               layer_key(L, "post_per_layer_input_norm.weight"), dm)) return -42;
            }
            if (h->weights.w_layer_scalar) {
                const char* nm_full = layer_key(L, "layer_scalar").c_str();
                auto key = layer_key(L, "layer_scalar");
                auto* v = store->get(key);
                if (v) {
                    float scalar_f = 0.f;
                    if (v->dtype == sk::Dtype::BF16) {
                        uint32_t bits = ((uint32_t)((const uint16_t*)v->data)[0]) << 16;
                        std::memcpy(&scalar_f, &bits, 4);
                    } else if (v->dtype == sk::Dtype::F16) {
                        uint16_t h16 = ((const uint16_t*)v->data)[0];
                        uint32_t s = (uint32_t)(h16 >> 15) & 1;
                        uint32_t e = (uint32_t)(h16 >> 10) & 0x1F;
                        uint32_t m = (uint32_t)h16 & 0x3FF;
                        uint32_t out;
                        if (e == 0)       out = (s << 31);
                        else if (e == 31) out = (s << 31) | (0xFFu << 23) | (m << 13);
                        else              out = (s << 31) | ((e + 112u) << 23) | (m << 13);
                        std::memcpy(&scalar_f, &out, 4);
                    } else {
                        std::memcpy(&scalar_f, v->data, 4);
                    }
                    float* dst = (float*)h->weights.w_layer_scalar->contents();
                    dst[L] = scalar_f;
                } else {
                    (void)nm_full;
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
