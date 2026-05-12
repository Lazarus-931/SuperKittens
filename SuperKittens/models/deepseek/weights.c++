#include "weights.h"
#include "deepseek_model.h"
#include "../../inference/weight_store.h"
#include "../load/gguf/gguf.h"

#include <cstdio>
#include <cstring>
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

bool copy_into(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
               const std::string& name, size_t expect_bytes)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "ds4 weights: missing tensor '%s'\n", name.c_str()); return false; }
    if (v->nbytes != expect_bytes) {
        std::fprintf(stderr, "ds4 weights: size mismatch '%s' got %zu expect %zu\n",
                     name.c_str(), v->nbytes, expect_bytes);
        return false;
    }
    std::memcpy((char*)dst->contents() + dst_off, v->data, expect_bytes);
    return true;
}

bool copy_raw(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
              const std::string& name, size_t expect_bytes)
{
    return copy_into(dst, dst_off, store, name, expect_bytes);
}

bool copy_transpose_fp16(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
                         const std::string& name, size_t out_rows, size_t out_cols)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "ds4 weights: missing tensor '%s'\n", name.c_str()); return false; }
    if (v->nbytes != out_rows * out_cols * 2) {
        std::fprintf(stderr, "ds4 weights: tx size mismatch '%s' got %zu expect %zu\n",
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
    const bool   moe_q = (c.moe_quant == 1);

    if (!copy_into(h->weights.w_embed, 0, store,
                   "token_embd.weight", (size_t)c.vocab_size * dm * fp16)) return -10;
    if (!copy_into(h->weights.w_final_norm, 0, store,
                   "output_norm.weight", dm * fp16)) return -11;

    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const size_t pre_off    = (size_t)L * dm * fp16;
        const size_t qan_off    = (size_t)L * qra * fp16;
        const size_t kvan_off   = (size_t)L * kvr * fp16;
        const size_t q_a_off    = (size_t)L * dm * qra * fp16;
        const size_t q_b_off    = (size_t)L * qra * qbN * fp16;
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
        if (!copy_into(h->weights.w_q_a_norm, qan_off, store,
                       blk_key(L, "attn_q_a_norm.weight"), qra * fp16)) return -22;
        if (!copy_into(h->weights.w_kv_a_norm, kvan_off, store,
                       blk_key(L, "attn_kv_a_norm.weight"), kvr * fp16)) return -23;

        if (!copy_transpose_fp16(h->weights.w_q_a, q_a_off, store,
                       blk_key(L, "attn_q_a.weight"), qra, dm)) return -24;
        if (!copy_transpose_fp16(h->weights.w_q_b, q_b_off, store,
                       blk_key(L, "attn_q_b.weight"), qbN, qra)) return -25;

        {
            const std::string nm = blk_key(L, "attn_kv_a_mqa.weight");
            auto* v = store->get(nm);
            const std::string nm2 = v ? nm : blk_key(L, "attn_kv.weight");
            if (!copy_transpose_fp16(h->weights.w_kv_a, kv_a_off, store,
                           nm2, kva_cols, dm)) return -26;
        }

        if (!copy_transpose_fp16(h->weights.w_kv_b, kv_b_off, store,
                       blk_key(L, "attn_kv_b.weight"), kvbN, kvr)) return -27;
        if (!copy_transpose_fp16(h->weights.w_o, o_off, store,
                       blk_key(L, "attn_output.weight"), dm, nh * dv)) return -28;

        if (!copy_transpose_fp16(h->weights.w_shared_gate, sh_gate_off, store,
                       blk_key(L, "ffn_gate_shexp.weight"), sni, dm)) return -30;
        if (!copy_transpose_fp16(h->weights.w_shared_up, sh_gate_off, store,
                       blk_key(L, "ffn_up_shexp.weight"), sni, dm)) return -31;
        if (!copy_transpose_fp16(h->weights.w_shared_down, sh_down_off, store,
                       blk_key(L, "ffn_down_shexp.weight"), dm, sni)) return -32;

        if (!copy_transpose_fp16(h->weights.w_router, router_off, store,
                       blk_key(L, "ffn_gate_inp.weight"), E, dm)) return -33;

        {
            const std::string gname = blk_key(L, "ffn_gate_exps.weight");
            const std::string uname = blk_key(L, "ffn_up_exps.weight");
            const std::string dname = blk_key(L, "ffn_down_exps.weight");
            auto* gv = store->get(gname);
            auto* uv = store->get(uname);
            auto* dv2 = store->get(dname);
            if (!gv || !uv || !dv2) {
                std::fprintf(stderr, "ds4 weights: missing routed expert tensors layer %u\n", L);
                return -34;
            }
            const bool gv_q = is_block_quant(gv->dtype);
            const bool uv_q = is_block_quant(uv->dtype);
            const bool dv_q = is_block_quant(dv2->dtype);

            const size_t gate_bytes_per_layer = (size_t)h->weights.w_gate->length() / c.n_layers;
            const size_t up_bytes_per_layer   = (size_t)h->weights.w_up->length()   / c.n_layers;
            const size_t down_bytes_per_layer = (size_t)h->weights.w_down->length() / c.n_layers;
            const size_t gate_off = (size_t)L * gate_bytes_per_layer;
            const size_t up_off   = (size_t)L * up_bytes_per_layer;
            const size_t down_off = (size_t)L * down_bytes_per_layer;

            if (moe_q || gv_q) {
                if (gv->nbytes != gate_bytes_per_layer) {
                    std::fprintf(stderr, "ds4 weights: gate_exps size mismatch L%u got %zu expect %zu\n",
                                 L, gv->nbytes, gate_bytes_per_layer);
                    return -35;
                }
                std::memcpy((char*)h->weights.w_gate->contents() + gate_off, gv->data, gate_bytes_per_layer);
            } else {
                if (gv->nbytes != (size_t)E * ni * dm * fp16) return -35;
                const uint16_t* s = (const uint16_t*)gv->data;
                uint16_t* d = (uint16_t*)((char*)h->weights.w_gate->contents() + gate_off);
                std::memcpy(d, s, gv->nbytes);
            }
            if (moe_q || uv_q) {
                if (uv->nbytes != up_bytes_per_layer) return -36;
                std::memcpy((char*)h->weights.w_up->contents() + up_off, uv->data, up_bytes_per_layer);
            } else {
                if (uv->nbytes != (size_t)E * ni * dm * fp16) return -36;
                std::memcpy((char*)h->weights.w_up->contents() + up_off, uv->data, uv->nbytes);
            }
            if (moe_q || dv_q) {
                if (dv2->nbytes != down_bytes_per_layer) return -37;
                std::memcpy((char*)h->weights.w_down->contents() + down_off, dv2->data, down_bytes_per_layer);
            } else {
                if (dv2->nbytes != (size_t)E * dm * ni * fp16) return -37;
                std::memcpy((char*)h->weights.w_down->contents() + down_off, dv2->data, dv2->nbytes);
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
