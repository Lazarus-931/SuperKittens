#include "weights.h"
#include "gemma4_model.h"
#include "../../../inference/weight_store.h"
#include "../../load/safetensor/safetensor.h"

#include <cstdio>
#include <cstring>
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
};
}}

namespace {

inline std::string layer_key(uint32_t L, const char* suffix) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "model.layers.%u.%s", L, suffix);
    return std::string(buf);
}

bool copy_into(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
               const std::string& name, size_t expect_bytes)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "gemma4 weights: missing tensor '%s'\n", name.c_str()); return false; }
    if (v->nbytes != expect_bytes) {
        std::fprintf(stderr, "gemma4 weights: size mismatch '%s' got %zu expect %zu\n",
                     name.c_str(), v->nbytes, expect_bytes);
        return false;
    }
    std::memcpy((char*)dst->contents() + dst_off, v->data, expect_bytes);
    return true;
}

bool copy_partial(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
                  const std::string& name, size_t expect_bytes)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "gemma4 weights: missing tensor '%s'\n", name.c_str()); return false; }
    if (v->nbytes < expect_bytes) {
        std::fprintf(stderr, "gemma4 weights: tensor too small '%s' got %zu expect>=%zu\n",
                     name.c_str(), v->nbytes, expect_bytes);
        return false;
    }
    std::memcpy((char*)dst->contents() + dst_off, v->data, expect_bytes);
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
    for (size_t i = 0; i < out_rows; ++i)
        for (size_t j = 0; j < out_cols; ++j)
            d[i * out_cols + j] = src[j * out_rows + i];
    return true;
}

}

extern "C" int sk_gemma4_load_from_store(sk_gemma4_handle* hp, sk::WeightStore* store) {
    if (!hp || !store) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    const auto& c = h->cfg;
    const size_t fp16 = 2;
    const size_t dm   = c.d_model;
    const size_t ni   = c.n_int;
    const size_t nh   = c.n_heads;

    const size_t hd_max     = c.head_dim_global > c.head_dim_local ? c.head_dim_global : c.head_dim_local;
    const size_t n_kv_max   = c.n_kv_heads_local > c.n_kv_heads_global ? c.n_kv_heads_local : c.n_kv_heads_global;
    const size_t qkv_slots_max = nh + 2u * n_kv_max;
    const size_t qkv_layer_stride = dm * qkv_slots_max * hd_max;
    const size_t qhd_max    = nh * hd_max;
    const size_t w_out_layer_stride = qhd_max * dm;

    if (!copy_into(h->weights.w_embed, 0, store,
                   "model.embed_tokens.weight", (size_t)c.vocab_size * dm * fp16)) return -10;
    if (!copy_into(h->weights.w_final_norm, 0, store,
                   "model.norm.weight", dm * fp16)) return -11;

    if (c.has_ple && h->weights.w_ple) {
        const char* candidates[] = {
            "model.per_layer_token_embedding.weight",
            "model.embed_tokens_per_layer.weight",
            "model.per_layer_embeddings.weight",
        };
        bool found = false;
        const size_t ple_bytes = (size_t)c.n_layers * c.vocab_size * dm * fp16;
        for (const char* nm : candidates) {
            auto* v = store->get(nm);
            if (v) {
                if (v->nbytes != ple_bytes) {
                    std::fprintf(stderr, "gemma4 weights: PLE size mismatch '%s' got %zu expect %zu\n",
                                 nm, v->nbytes, ple_bytes);
                    return -12;
                }
                std::memcpy(h->weights.w_ple->contents(), v->data, ple_bytes);
                found = true;
                break;
            }
        }
        if (!found) {
            std::fprintf(stderr, "gemma4 weights: PLE tensor not found; leaving zero\n");
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

        const size_t pre_off      = (size_t)L * dm * fp16;
        const size_t gamma_off    = (size_t)L * hd_max * fp16;
        const size_t o_off        = (size_t)L * w_out_layer_stride * fp16;
        const size_t gate_off     = (size_t)L * dm * ni * fp16;
        const size_t down_off     = (size_t)L * ni * dm * fp16;
        const size_t qkv_off      = (size_t)L * qkv_layer_stride * fp16;

        if (!copy_into(h->weights.w_pre_attn_norm,  pre_off, store,
                       layer_key(L, "input_layernorm.weight"),           dm * fp16)) return -20;
        if (!copy_into(h->weights.w_post_attn_norm, pre_off, store,
                       layer_key(L, "post_attention_layernorm.weight"),  dm * fp16)) return -21;
        if (!copy_into(h->weights.w_pre_mlp_norm,   pre_off, store,
                       layer_key(L, "pre_feedforward_layernorm.weight"), dm * fp16)) return -22;
        if (!copy_into(h->weights.w_post_mlp_norm,  pre_off, store,
                       layer_key(L, "post_feedforward_layernorm.weight"),dm * fp16)) return -23;

        if (!copy_partial(h->weights.gamma_q, gamma_off, store,
                          layer_key(L, "self_attn.q_norm.weight"), hd * fp16)) return -24;
        if (!copy_partial(h->weights.gamma_k, gamma_off, store,
                          layer_key(L, "self_attn.k_norm.weight"), hd * fp16)) return -25;

        // o_proj: HF (d_model, n_heads*head_dim) -> (n_heads*head_dim, d_model).
        // Written contiguous at layer offset; unused tail of slab stays zero.
        if (!copy_transpose_fp16(h->weights.w_out, o_off, store,
                                 layer_key(L, "self_attn.o_proj.weight"), Nq, dm)) return -26;

        if (!copy_transpose_fp16(h->weights.w_gate, gate_off, store,
                                 layer_key(L, "mlp.gate_proj.weight"), dm, ni)) return -27;
        if (!copy_transpose_fp16(h->weights.w_up,   gate_off, store,
                                 layer_key(L, "mlp.up_proj.weight"),   dm, ni)) return -28;
        if (!copy_transpose_fp16(h->weights.w_down, down_off, store,
                                 layer_key(L, "mlp.down_proj.weight"), ni, dm)) return -29;

        // QKV: HF q/k/v_proj each (rows, d_model). Pack transposed into one
        // (d_model, qkvN) contiguous block at this layer's slab offset.
        auto* q_v = store->get(layer_key(L, "self_attn.q_proj.weight"));
        auto* k_v = store->get(layer_key(L, "self_attn.k_proj.weight"));
        auto* v_v = store->get(layer_key(L, "self_attn.v_proj.weight"));
        if (!q_v || !k_v || !v_v) {
            std::fprintf(stderr, "gemma4 weights: missing q/k/v_proj for layer %u\n", L);
            return -30;
        }
        const size_t qb = Nq  * dm * fp16;
        const size_t kb = Nkv * dm * fp16;
        const size_t vb = Nkv * dm * fp16;
        if (q_v->nbytes != qb || k_v->nbytes != kb || v_v->nbytes != vb) {
            std::fprintf(stderr, "gemma4 weights: qkv size mismatch layer %u\n", L);
            return -31;
        }
        uint16_t* layer = (uint16_t*)(qkv_base + qkv_off);
        auto tx_into = [&](const void* src_v, size_t out_rows, size_t out_cols, size_t col_off) {
            const uint16_t* s = (const uint16_t*)src_v;
            for (size_t i = 0; i < out_rows; ++i)
                for (size_t j = 0; j < out_cols; ++j)
                    layer[i * qkvN + col_off + j] = s[j * out_rows + i];
        };
        tx_into(q_v->data, dm, Nq,  0);
        tx_into(k_v->data, dm, Nkv, Nq);
        tx_into(v_v->data, dm, Nkv, Nq + Nkv);

        (void)w_out_base;
    }

    // cos_local/sin_local/cos_global/sin_global are precomputed host-side
    // and are not loaded from safetensors. Buffers remain zero-initialized.
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
