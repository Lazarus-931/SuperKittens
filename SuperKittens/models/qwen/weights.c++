#include "weights.h"
#include "qwen_model.h"
#include "../../inference/weight_store.h"
#include "../load/safetensor/safetensor.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace meow { namespace qwen {
struct Handle {
    sk_qwen_config cfg;
    uint32_t       current_pos;
    ModelPSOs      psos;
    ModelWeights   weights;
    ModelBuffers   bufs;
    std::vector<LayerCache> layer_caches;
    std::vector<MTL::Buffer*> k_caches;
    std::vector<MTL::Buffer*> v_caches;
};
}}

namespace {

inline std::string layer_key(uint32_t L, const char* suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "model.layers.%u.%s", L, suffix);
    return std::string(buf);
}

bool copy_into(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
               const std::string& name, size_t expect_bytes)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "qwen weights: missing tensor '%s'\n", name.c_str()); return false; }
    if (v->nbytes != expect_bytes) {
        std::fprintf(stderr, "qwen weights: size mismatch '%s' got %zu expect %zu\n",
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
    if (!v) { std::fprintf(stderr, "qwen weights: missing tensor '%s'\n", name.c_str()); return false; }
    if (v->nbytes != out_rows * out_cols * 2) {
        std::fprintf(stderr, "qwen weights: tx size mismatch '%s' got %zu expect %zu\n",
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

extern "C" int sk_qwen_load_from_store(sk_qwen_handle* hp, sk::WeightStore* store) {
    if (!hp || !store) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    const auto& c = h->cfg;
    const size_t fp16 = 2;
    const size_t dm   = c.d_model;
    const size_t hd   = c.head_dim;
    const size_t Nq   = (size_t)c.n_heads    * hd;
    const size_t Nkv  = (size_t)c.n_kv_heads * hd;
    const size_t qkvN = Nq + 2 * Nkv;
    const size_t ni   = c.n_int;

    if (!copy_into(h->weights.w_embed, 0, store,
                   "model.embed_tokens.weight", (size_t)c.vocab_size * dm * fp16)) return -10;
    if (!copy_into(h->weights.w_final_norm, 0, store,
                   "model.norm.weight", dm * fp16)) return -11;

    auto* qkv_base = (char*)h->weights.w_qkv->contents();

    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const size_t pre_off = (size_t)L * dm * fp16;
        const size_t mlp_pre_off = pre_off;
        const size_t qnorm_off = (size_t)L * hd * fp16;
        const size_t o_off   = (size_t)L * Nq * dm * fp16;
        const size_t gate_off = (size_t)L * dm * ni * fp16;
        const size_t down_off = (size_t)L * ni * dm * fp16;
        const size_t qkv_layer_off = (size_t)L * dm * qkvN * fp16;

        if (!copy_into(h->weights.w_pre_attn_norm, pre_off, store,
                       layer_key(L, "input_layernorm.weight"), dm * fp16)) return -20;
        if (!copy_into(h->weights.w_pre_mlp_norm, mlp_pre_off, store,
                       layer_key(L, "post_attention_layernorm.weight"), dm * fp16)) return -21;
        if (!copy_into(h->weights.w_q_norm, qnorm_off, store,
                       layer_key(L, "self_attn.q_norm.weight"), hd * fp16)) return -22;
        if (!copy_into(h->weights.w_k_norm, qnorm_off, store,
                       layer_key(L, "self_attn.k_norm.weight"), hd * fp16)) return -23;
        if (!copy_transpose_fp16(h->weights.w_o, o_off, store,
                       layer_key(L, "self_attn.o_proj.weight"), Nq, dm)) return -24;
        if (!copy_transpose_fp16(h->weights.w_gate, gate_off, store,
                       layer_key(L, "mlp.gate_proj.weight"), dm, ni)) return -25;
        if (!copy_transpose_fp16(h->weights.w_up, gate_off, store,
                       layer_key(L, "mlp.up_proj.weight"), dm, ni)) return -26;
        if (!copy_transpose_fp16(h->weights.w_down, down_off, store,
                       layer_key(L, "mlp.down_proj.weight"), ni, dm)) return -27;

        auto* q_v = store->get(layer_key(L, "self_attn.q_proj.weight"));
        auto* k_v = store->get(layer_key(L, "self_attn.k_proj.weight"));
        auto* v_v = store->get(layer_key(L, "self_attn.v_proj.weight"));
        if (!q_v || !k_v || !v_v) {
            std::fprintf(stderr, "qwen weights: missing q/k/v_proj for layer %u\n", L);
            return -28;
        }
        const size_t qb = Nq  * dm * fp16;
        const size_t kb = Nkv * dm * fp16;
        const size_t vb = Nkv * dm * fp16;
        if (q_v->nbytes != qb || k_v->nbytes != kb || v_v->nbytes != vb) return -29;
        uint16_t* layer = (uint16_t*)(qkv_base + qkv_layer_off);
        auto tx_into = [&](const void* src_v, size_t out_rows /*=dm*/, size_t out_cols, size_t col_off) {
            const uint16_t* s = (const uint16_t*)src_v;
            for (size_t i = 0; i < out_rows; ++i)
                for (size_t j = 0; j < out_cols; ++j)
                    layer[i * qkvN + col_off + j] = s[j * out_rows + i];
        };
        tx_into(q_v->data, dm, Nq,  0);
        tx_into(k_v->data, dm, Nkv, Nq);
        tx_into(v_v->data, dm, Nkv, Nq + Nkv);
    }

    return 0;
}

extern "C" int sk_qwen_load_safetensors(sk_qwen_handle* h, const char* path) {
    if (!h || !path) return -1;
    auto store = new sk::WeightStore();
    int rc = sk::load_safetensors(path, *store);
    if (rc != 0) { delete store; return rc; }
    rc = sk_qwen_load_from_store(h, store);
    delete store;
    return rc;
}

extern "C" int sk_qwen_load_safetensors_index(sk_qwen_handle* h, const char* index_json_path) {
    if (!h || !index_json_path) return -1;
    auto store = new sk::WeightStore();
    int rc = sk::load_safetensors_index(index_json_path, *store);
    if (rc != 0) { delete store; return rc; }
    rc = sk_qwen_load_from_store(h, store);
    delete store;
    return rc;
}
