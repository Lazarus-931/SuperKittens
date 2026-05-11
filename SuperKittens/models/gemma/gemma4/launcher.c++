//
//  launcher.c++ — Gemma 4 inference launcher implementation.
//

#include "launcher.h"
#include "gemma4_model.h"
#include "../../../kernels/runtime_bindings.h"

#include <cstring>
#include <cstdlib>
#include <vector>

namespace meow { namespace gemma4 {

struct Handle {
    sk_gemma4_config cfg;
    uint32_t         current_pos = 0;

    ModelPSOs    psos;
    ModelWeights weights;
    ModelBuffers bufs;

    std::vector<LayerCache> layer_caches; // owned MTL::Buffer pairs
};

static MTL::Buffer* alloc_zero(MTL::Device* dev, size_t bytes) {
    auto* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (b) std::memset(b->contents(), 0, bytes);
    return b;
}

static bool resolve_psos(ModelPSOs& P) {
    P.layer.rmsnorm        = sk::bindings_pso("rmsnorm");
    P.layer.gemm           = sk::bindings_pso("gemm_fp16");
    P.layer.qkv_norm       = sk::bindings_pso("gemma4_qkv_norm");
    P.layer.rope           = sk::bindings_pso("rope_qk");
    P.layer.prope          = sk::bindings_pso("gemma4_prope_qk");
    P.layer.attn_local     = sk::bindings_pso("gemma4_attn_local_d256");
    P.layer.attn_global    = sk::bindings_pso("gemma4_attn_global_d512");
    P.layer.gated_mlp_gelu = sk::bindings_pso("gated_mlp_gelu");
    P.layer.add            = sk::bindings_pso("add_f16");
    P.layer.add_rmsnorm    = sk::bindings_pso("add_rmsnorm");
    P.layer.kv_cache_write = sk::bindings_pso("kv_cache_write");
    P.embedding_lookup     = sk::bindings_pso("embedding_lookup");
    P.ple_add              = sk::bindings_pso("gemma4_ple_add");
    P.argmax               = sk::bindings_pso("argmax");

    return P.layer.rmsnorm    && P.layer.gemm     && P.layer.qkv_norm
        && P.layer.rope       && P.layer.prope    && P.layer.attn_local
        && P.layer.attn_global && P.layer.gated_mlp_gelu && P.layer.add
        && P.layer.add_rmsnorm
        && P.layer.kv_cache_write
        && P.embedding_lookup && P.ple_add        && P.argmax;
}

}}  // namespace meow::gemma4

extern "C" sk_gemma4_handle* sk_gemma4_create(const sk_gemma4_config* cfg) {
    if (!cfg) return nullptr;
    auto* dev = sk::bindings_device();
    if (!dev) return nullptr;

    auto* h = new meow::gemma4::Handle();
    h->cfg = *cfg;
    h->current_pos = 0;

    if (!meow::gemma4::resolve_psos(h->psos)) {
        delete h;
        return nullptr;
    }

    using namespace meow::gemma4;

    const uint32_t T_max          = cfg->batch * cfg->seq_max;
    const size_t   x_bytes        = (size_t)T_max * cfg->d_model * 2;
    const size_t   logits_bytes   = (size_t)T_max * cfg->vocab_size * 2;

    const uint32_t hd_max         = cfg->head_dim_global > cfg->head_dim_local
                                    ? cfg->head_dim_global : cfg->head_dim_local;
    const uint32_t n_kv_max       = cfg->n_kv_heads_local > cfg->n_kv_heads_global
                                    ? cfg->n_kv_heads_local : cfg->n_kv_heads_global;
    const uint32_t qkv_slots_max  = cfg->n_heads + 2u * n_kv_max;

    const size_t   qkv_bytes      = (size_t)T_max * qkv_slots_max * hd_max * 2;
    const size_t   q_bytes        = (size_t)T_max * cfg->n_heads  * hd_max * 2;
    const size_t   kv_tmp_bytes   = (size_t)T_max * n_kv_max      * hd_max * 2;
    const size_t   attn_out_bytes = (size_t)T_max * cfg->n_heads  * hd_max * 2;

    // Weight buffers — sized for the slab layout dispatch_layer expects
    // (uniform per-layer stride using head_dim_max / n_kv_max).
    h->weights.w_embed         = alloc_zero(dev, (size_t)cfg->vocab_size * cfg->d_model * 2);
    h->weights.w_ple           = cfg->has_ple
                                  ? alloc_zero(dev, (size_t)cfg->n_layers * cfg->vocab_size * cfg->d_model * 2)
                                  : nullptr;
    h->weights.w_pre_attn_norm = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_post_attn_norm= alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_pre_mlp_norm  = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_post_mlp_norm = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_final_norm    = alloc_zero(dev, (size_t)cfg->d_model * 2);
    h->weights.w_qkv           = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model
                                                  * qkv_slots_max * hd_max * 2);
    h->weights.w_out           = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_heads * hd_max * cfg->d_model * 2);
    h->weights.gamma_q         = alloc_zero(dev, (size_t)cfg->n_layers * hd_max * 2);
    h->weights.gamma_k         = alloc_zero(dev, (size_t)cfg->n_layers * hd_max * 2);
    h->weights.w_gate          = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * cfg->n_int * 2);
    h->weights.w_up            = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * cfg->n_int * 2);
    h->weights.w_down          = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_int * cfg->d_model * 2);
    h->weights.cos_local       = alloc_zero(dev, (size_t)cfg->cache_max * (cfg->head_dim_local / 2) * 2);
    h->weights.sin_local       = alloc_zero(dev, (size_t)cfg->cache_max * (cfg->head_dim_local / 2) * 2);
    h->weights.cos_global      = alloc_zero(dev, (size_t)cfg->cache_max * (cfg->head_dim_global / 2) * 2);
    h->weights.sin_global      = alloc_zero(dev, (size_t)cfg->cache_max * (cfg->head_dim_global / 2) * 2);

    // Per-layer K/V cache buffers (sized per layer-type).
    h->layer_caches.resize(cfg->n_layers);
    for (uint32_t L = 0; L < cfg->n_layers; ++L) {
        const bool is_global = ((L % cfg->local_period) == (cfg->local_period - 1));
        const uint32_t n_kv  = is_global ? cfg->n_kv_heads_global : cfg->n_kv_heads_local;
        const uint32_t hd    = is_global ? cfg->head_dim_global   : cfg->head_dim_local;
        const uint32_t csize = is_global ? cfg->cache_max         : cfg->window;
        const size_t kv_bytes = (size_t)cfg->batch * n_kv * csize * hd * 2;
        h->layer_caches[L].k = alloc_zero(dev, kv_bytes);
        h->layer_caches[L].v = alloc_zero(dev, kv_bytes);
    }
    h->weights.layer_caches = h->layer_caches.data();

    // Scratch buffers
    h->bufs.input_ids  = alloc_zero(dev, (size_t)T_max * sizeof(int32_t));
    h->bufs.output_id  = alloc_zero(dev, (size_t)cfg->batch * sizeof(int32_t));
    h->bufs.x_a        = alloc_zero(dev, x_bytes);
    h->bufs.x_b        = alloc_zero(dev, x_bytes);
    h->bufs.logits     = alloc_zero(dev, logits_bytes);
    h->bufs.x_norm     = alloc_zero(dev, x_bytes);
    h->bufs.qkv_packed = alloc_zero(dev, qkv_bytes);
    h->bufs.q_norm     = alloc_zero(dev, q_bytes);
    h->bufs.k_tmp      = alloc_zero(dev, kv_tmp_bytes);
    h->bufs.v_tmp      = alloc_zero(dev, kv_tmp_bytes);
    h->bufs.attn_out   = alloc_zero(dev, attn_out_bytes);
    h->bufs.o_proj     = alloc_zero(dev, x_bytes);
    h->bufs.y_attn     = alloc_zero(dev, x_bytes);
    h->bufs.m_in       = alloc_zero(dev, x_bytes);
    h->bufs.m_out      = alloc_zero(dev, x_bytes);
    h->bufs.y_out      = alloc_zero(dev, x_bytes);

    return reinterpret_cast<sk_gemma4_handle*>(h);
}

extern "C" int sk_gemma4_load_weights(sk_gemma4_handle* hp, const sk_gemma4_weights* w) {
    if (!hp || !w) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);

    auto cp = [](MTL::Buffer* dst, const void* src) {
        if (dst && src) std::memcpy(dst->contents(), src, dst->length());
    };

    cp(h->weights.w_embed,         w->w_embed);
    if (h->cfg.has_ple) cp(h->weights.w_ple, w->w_ple);
    cp(h->weights.w_pre_attn_norm, w->w_pre_attn_norm);
    cp(h->weights.w_post_attn_norm,w->w_post_attn_norm);
    cp(h->weights.w_pre_mlp_norm,  w->w_pre_mlp_norm);
    cp(h->weights.w_post_mlp_norm, w->w_post_mlp_norm);
    cp(h->weights.w_final_norm,    w->w_final_norm);
    cp(h->weights.w_qkv,           w->w_qkv);
    cp(h->weights.w_out,           w->w_out);
    cp(h->weights.gamma_q,         w->gamma_q);
    cp(h->weights.gamma_k,         w->gamma_k);
    cp(h->weights.w_gate,          w->w_gate);
    cp(h->weights.w_up,            w->w_up);
    cp(h->weights.w_down,          w->w_down);
    cp(h->weights.cos_local,       w->cos_local);
    cp(h->weights.sin_local,       w->sin_local);
    cp(h->weights.cos_global,      w->cos_global);
    cp(h->weights.sin_global,      w->sin_global);
    return 0;
}

extern "C" void sk_gemma4_reset(sk_gemma4_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    h->current_pos = 0;
}

extern "C" int sk_gemma4_forward(sk_gemma4_handle* hp,
                                 const int* input_ids, uint32_t seq,
                                 int* output_id) {
    if (!hp || !input_ids || !output_id) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    if (seq == 0 || seq > h->cfg.seq_max) return -2;
    if (h->current_pos + seq > h->cfg.cache_max) return -4;

    auto* dev = sk::bindings_device();
    auto* q   = sk::bindings_queue();
    if (!dev || !q) return -3;

    std::memcpy(h->bufs.input_ids->contents(), input_ids,
                (size_t)h->cfg.batch * seq * sizeof(int32_t));

    meow::gemma4::ModelParams mp;
    mp.batch              = h->cfg.batch;
    mp.seq                = seq;
    mp.n_layers           = h->cfg.n_layers;
    mp.local_period       = h->cfg.local_period;
    mp.d_model            = h->cfg.d_model;
    mp.n_int              = h->cfg.n_int;
    mp.n_heads            = h->cfg.n_heads;
    mp.n_kv_heads_local   = h->cfg.n_kv_heads_local;
    mp.n_kv_heads_global  = h->cfg.n_kv_heads_global;
    mp.head_dim_local     = h->cfg.head_dim_local;
    mp.head_dim_global    = h->cfg.head_dim_global;
    mp.window             = h->cfg.window;
    mp.cache_max          = h->cfg.cache_max;
    mp.prope_p_pairs      = h->cfg.prope_p_pairs;
    mp.vocab_size         = h->cfg.vocab_size;
    mp.has_ple            = (h->cfg.has_ple != 0);
    mp.eps                = h->cfg.eps;
    mp.current_pos        = h->current_pos;

    auto* cmd = q->commandBuffer();
    meow::gemma4::dispatch_model(cmd, h->psos, h->weights, h->bufs, mp);
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    h->current_pos += seq;

    std::memcpy(output_id, h->bufs.output_id->contents(),
                (size_t)h->cfg.batch * sizeof(int32_t));
    return 0;
}

extern "C" void sk_gemma4_destroy(sk_gemma4_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);

    auto rel = [](MTL::Buffer* b) { if (b) b->release(); };
    rel(h->weights.w_embed); rel(h->weights.w_ple);
    rel(h->weights.w_pre_attn_norm); rel(h->weights.w_post_attn_norm);
    rel(h->weights.w_pre_mlp_norm);  rel(h->weights.w_post_mlp_norm);
    rel(h->weights.w_final_norm);
    rel(h->weights.w_qkv); rel(h->weights.w_out);
    rel(h->weights.gamma_q); rel(h->weights.gamma_k);
    rel(h->weights.w_gate); rel(h->weights.w_up); rel(h->weights.w_down);
    rel(h->weights.cos_local); rel(h->weights.sin_local);
    rel(h->weights.cos_global); rel(h->weights.sin_global);

    for (auto& lc : h->layer_caches) { rel(lc.k); rel(lc.v); }

    rel(h->bufs.input_ids); rel(h->bufs.output_id);
    rel(h->bufs.x_a); rel(h->bufs.x_b); rel(h->bufs.logits);
    rel(h->bufs.x_norm); rel(h->bufs.qkv_packed);
    rel(h->bufs.q_norm); rel(h->bufs.k_tmp); rel(h->bufs.v_tmp);
    rel(h->bufs.attn_out); rel(h->bufs.o_proj); rel(h->bufs.y_attn);
    rel(h->bufs.m_in); rel(h->bufs.m_out); rel(h->bufs.y_out);

    delete h;
}
