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

    std::vector<uint32_t> n_int_per_layer;
    std::vector<size_t>   mlp_gate_off_e;
    std::vector<size_t>   mlp_down_off_e;
    std::vector<int32_t>  kv_source_layer;

    bool dump_enabled = false;
};

static MTL::Buffer* alloc_zero(MTL::Device* dev, size_t bytes) {
    auto* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (b) std::memset(b->contents(), 0, bytes);
    return b;
}

static bool resolve_psos(ModelPSOs& P) {
    P.layer.rmsnorm        = sk::bindings_pso("gemma4_rmsnorm_bf16");
    P.layer.gemm           = sk::bindings_pso("gemma4_gemm_bf16");
    P.layer.qkv_norm       = sk::bindings_pso("gemma4_qkv_norm");
    P.layer.rope           = sk::bindings_pso("gemma4_rope_qk_bf16");
    P.layer.prope          = sk::bindings_pso("gemma4_prope_qk");
    P.layer.rope_partial   = sk::bindings_pso("gemma4_rope_qk_partial");
    P.layer.attn_local     = sk::bindings_pso("gemma4_attn_local_d256");
    P.layer.attn_global    = sk::bindings_pso("gemma4_attn_global_d512");
    P.layer.gated_mlp_gelu = sk::bindings_pso("gemma4_gated_mlp_bf16");
    P.layer.add            = sk::bindings_pso("add_bf16");
    P.layer.add_rmsnorm    = sk::bindings_pso("gemma4_add_rmsnorm_bf16");
    P.layer.kv_cache_write = sk::bindings_pso("gemma4_kv_cache_write_bf16");
    P.layer.ple_gate_act   = sk::bindings_pso("gemma4_ple_gate_act");
    P.layer.ple_inject     = sk::bindings_pso("gemma4_ple_inject");
    P.embedding_lookup     = sk::bindings_pso("gemma4_embedding_lookup_bf16");
    P.ple_lookup           = sk::bindings_pso("gemma4_ple_lookup");
    P.ple_context_mix      = sk::bindings_pso("gemma4_ple_context_mix");
    P.argmax               = sk::bindings_pso("gemma4_argmax_bf16");
    P.logit_softcap        = sk::bindings_pso("gemma4_logit_softcap");
    P.logit_descale = sk::bindings_pso("gemma4_logit_descale");

    #define _CK(x) if (!(x)) return false;
    _CK(P.layer.rmsnorm);
    _CK(P.layer.gemm);
    _CK(P.layer.qkv_norm);
    _CK(P.layer.rope);
    _CK(P.layer.prope);
    _CK(P.layer.rope_partial);
    _CK(P.layer.attn_local);
    _CK(P.layer.attn_global);
    _CK(P.layer.gated_mlp_gelu);
    _CK(P.layer.add);
    _CK(P.layer.add_rmsnorm);
    _CK(P.layer.kv_cache_write);
    _CK(P.layer.ple_gate_act);
    _CK(P.layer.ple_inject);
    _CK(P.embedding_lookup);
    _CK(P.ple_lookup);
    _CK(P.ple_context_mix);
    _CK(P.argmax);
    _CK(P.logit_softcap);
    _CK(P.logit_descale);
    #undef _CK
    return true;
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

    // Pre-compute per-layer tables.
    const uint32_t nL = cfg->n_layers;
    const uint32_t first_kv_shared = (cfg->num_kv_shared_layers >= nL)
                                     ? 0u
                                     : (nL - cfg->num_kv_shared_layers);
    h->n_int_per_layer.resize(nL);
    h->mlp_gate_off_e.resize(nL);
    h->mlp_down_off_e.resize(nL);
    h->kv_source_layer.assign(nL, -1);
    {
        size_t cum_gate = 0, cum_down = 0;
        for (uint32_t L = 0; L < nL; ++L) {
            const bool is_shared = (cfg->num_kv_shared_layers > 0) && (L >= first_kv_shared);
            const uint32_t mul   = (cfg->use_double_wide_mlp && is_shared) ? 2u : 1u;
            h->n_int_per_layer[L] = cfg->n_int * mul;
            h->mlp_gate_off_e[L]  = cum_gate;
            h->mlp_down_off_e[L]  = cum_down;
            cum_gate += (size_t)cfg->d_model * h->n_int_per_layer[L];
            cum_down += (size_t)h->n_int_per_layer[L] * cfg->d_model;
        }
    }
    // Determine kv_source_layer using same layer-type rule as elsewhere
    // (local_period: last layer of each window is global). For each shared
    // layer L, find the latest non-shared layer < first_kv_shared with the
    // same layer-type.
    auto is_global_l = [&](uint32_t L) -> bool {
        return (L % cfg->local_period) == (cfg->local_period - 1);
    };
    if (cfg->num_kv_shared_layers > 0 && first_kv_shared > 0) {
        for (uint32_t L = first_kv_shared; L < nL; ++L) {
            const bool want_global = is_global_l(L);
            int32_t src = -1;
            for (int32_t s = (int32_t)first_kv_shared - 1; s >= 0; --s) {
                if (is_global_l((uint32_t)s) == want_global) { src = s; break; }
            }
            h->kv_source_layer[L] = src;
        }
    }

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
    h->weights.w_ple_table     = cfg->has_ple
                                  ? alloc_zero(dev, (size_t)cfg->vocab_size * cfg->n_layers * cfg->ple_dim * 2)
                                  : nullptr;
    h->weights.w_per_layer_input_gate      = cfg->has_ple
                                  ? alloc_zero(dev, (size_t)cfg->n_layers * cfg->ple_dim * cfg->d_model * 2)
                                  : nullptr;
    h->weights.w_per_layer_projection      = cfg->has_ple
                                  ? alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * cfg->ple_dim * 2)
                                  : nullptr;
    h->weights.w_layer_scalar              = cfg->has_ple
                                  ? alloc_zero(dev, (size_t)cfg->n_layers * sizeof(float))
                                  : nullptr;
    h->weights.w_post_per_layer_input_norm = cfg->has_ple
                                  ? alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2)
                                  : nullptr;
    // per_layer_model_projection: (n_layers*ple_dim, d_model) — projects
    // inputs_embeds to (n_layers, ple_dim) per token; combined with token-
    // identity PLE per HF modeling_gemma4.py:1779-1790.
    h->weights.w_per_layer_model_projection = cfg->has_ple
                                  ? alloc_zero(dev, (size_t)cfg->n_layers * cfg->ple_dim * cfg->d_model * 2)
                                  : nullptr;
    h->weights.w_per_layer_projection_norm  = cfg->has_ple
                                  ? alloc_zero(dev, (size_t)cfg->ple_dim * 2)
                                  : nullptr;
    h->weights.w_pre_attn_norm = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_post_attn_norm= alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_pre_feedforward_layernorm  = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_post_feedforward_layernorm = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_final_norm    = alloc_zero(dev, (size_t)cfg->d_model * 2);
    h->weights.w_qkv           = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model
                                                  * qkv_slots_max * hd_max * 2);
    h->weights.w_out           = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_heads * hd_max * cfg->d_model * 2);
    h->weights.gamma_q         = alloc_zero(dev, (size_t)cfg->n_layers * hd_max * 2);
    h->weights.gamma_k         = alloc_zero(dev, (size_t)cfg->n_layers * hd_max * 2);
    {
        size_t total_gate_e = h->mlp_gate_off_e.back() + (size_t)cfg->d_model * h->n_int_per_layer.back();
        size_t total_down_e = h->mlp_down_off_e.back() + (size_t)h->n_int_per_layer.back() * cfg->d_model;
        h->weights.w_gate = alloc_zero(dev, total_gate_e * 2);
        h->weights.w_up   = alloc_zero(dev, total_gate_e * 2);
        h->weights.w_down = alloc_zero(dev, total_down_e * 2);
    }
    h->weights.cos_local       = alloc_zero(dev, (size_t)cfg->cache_max * (cfg->head_dim_local / 2) * 2);
    h->weights.sin_local       = alloc_zero(dev, (size_t)cfg->cache_max * (cfg->head_dim_local / 2) * 2);
    h->weights.cos_global      = alloc_zero(dev, (size_t)cfg->cache_max * (cfg->head_dim_global / 2) * 2);
    h->weights.sin_global      = alloc_zero(dev, (size_t)cfg->cache_max * (cfg->head_dim_global / 2) * 2);

    // Per-layer K/V cache buffers (sized per layer-type).
    h->layer_caches.assign(cfg->n_layers, LayerCache{nullptr, nullptr});
    for (uint32_t L = 0; L < cfg->n_layers; ++L) {
        if (h->kv_source_layer[L] >= 0) continue; // shared: alias source below
        const bool is_global = ((L % cfg->local_period) == (cfg->local_period - 1));
        const uint32_t n_kv  = is_global ? cfg->n_kv_heads_global : cfg->n_kv_heads_local;
        const uint32_t hd    = is_global ? cfg->head_dim_global   : cfg->head_dim_local;
        const uint32_t csize = is_global ? cfg->cache_max         : cfg->window;
        const size_t kv_bytes = (size_t)cfg->batch * n_kv * csize * hd * 2;
        h->layer_caches[L].k = alloc_zero(dev, kv_bytes);
        h->layer_caches[L].v = alloc_zero(dev, kv_bytes);
    }
    for (uint32_t L = 0; L < cfg->n_layers; ++L) {
        if (h->kv_source_layer[L] >= 0) {
            uint32_t s = (uint32_t)h->kv_source_layer[L];
            h->layer_caches[L] = h->layer_caches[s];
        }
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

    // Dump stash: (1 + 4*nL + 1) * d_model + vocab_size fp16 elements,
    // plus L0-extra: 3*H*hd_max + 3*n_kv_max*hd_max for q/k pre+post RoPE and attn_pre.
    {
        const size_t per_dm = (size_t)(2 + 4 * cfg->n_layers) * cfg->d_model;
        const size_t hd_max_e = (cfg->head_dim_local > cfg->head_dim_global)
                                ? cfg->head_dim_local : cfg->head_dim_global;
        const size_t n_kv_max_e = (cfg->n_kv_heads_local > cfg->n_kv_heads_global)
                                  ? cfg->n_kv_heads_local : cfg->n_kv_heads_global;
        const size_t l0_extra = (size_t)3 * cfg->n_heads * hd_max_e
                              + (size_t)3 * n_kv_max_e * hd_max_e;
        const size_t total_elems = per_dm + cfg->vocab_size + l0_extra;
        h->bufs.dump_stash = alloc_zero(dev, total_elems * 2);
    }

    if (cfg->has_ple) {
        h->bufs.per_layer_inputs = alloc_zero(dev, (size_t)T_max * cfg->n_layers * cfg->ple_dim * 2);
        h->bufs.ple_ctx_proj     = alloc_zero(dev, (size_t)T_max * cfg->n_layers * cfg->ple_dim * 2);
        h->bufs.ple_gate_out     = alloc_zero(dev, (size_t)T_max * cfg->ple_dim * 2);
        h->bufs.ple_gated        = alloc_zero(dev, (size_t)T_max * cfg->ple_dim * 2);
        h->bufs.ple_proj_back    = alloc_zero(dev, x_bytes);
    } else {
        h->bufs.per_layer_inputs = nullptr;
        h->bufs.ple_ctx_proj     = nullptr;
        h->bufs.ple_gate_out     = nullptr;
        h->bufs.ple_gated        = nullptr;
        h->bufs.ple_proj_back    = nullptr;
    }

    return reinterpret_cast<sk_gemma4_handle*>(h);
}

extern "C" int sk_gemma4_load_weights(sk_gemma4_handle* hp, const sk_gemma4_weights* w) {
    if (!hp || !w) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);

    auto cp = [](MTL::Buffer* dst, const void* src) {
        if (dst && src) std::memcpy(dst->contents(), src, dst->length());
    };

    cp(h->weights.w_embed,         w->w_embed);
    if (h->cfg.has_ple) {
        cp(h->weights.w_ple_table,                 w->w_ple_table);
        cp(h->weights.w_per_layer_input_gate,      w->w_per_layer_input_gate);
        cp(h->weights.w_per_layer_projection,      w->w_per_layer_projection);
        cp(h->weights.w_layer_scalar,              w->w_layer_scalar);
        cp(h->weights.w_post_per_layer_input_norm, w->w_post_per_layer_input_norm);
        cp(h->weights.w_per_layer_model_projection, w->w_per_layer_model_projection);
        cp(h->weights.w_per_layer_projection_norm,  w->w_per_layer_projection_norm);
    }
    cp(h->weights.w_pre_attn_norm, w->w_pre_attn_norm);
    cp(h->weights.w_post_attn_norm,w->w_post_attn_norm);
    cp(h->weights.w_pre_feedforward_layernorm,  w->w_pre_feedforward_layernorm);
    cp(h->weights.w_post_feedforward_layernorm, w->w_post_feedforward_layernorm);
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
    mp.ple_dim            = h->cfg.ple_dim;
    mp.has_ple            = (h->cfg.has_ple != 0);
    mp.eps                = h->cfg.eps;
    mp.final_logit_softcap = h->cfg.final_logit_softcap;
    mp.current_pos        = h->current_pos;
    mp.n_int_per_layer    = h->n_int_per_layer.data();
    mp.mlp_gate_off_e     = h->mlp_gate_off_e.data();
    mp.mlp_down_off_e     = h->mlp_down_off_e.data();
    mp.kv_source_layer    = h->kv_source_layer.data();
    mp.dump_enabled       = h->dump_enabled;

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

extern "C" int sk_gemma4_get_last_logits(sk_gemma4_handle* hp, void* out_fp16) {
    if (!hp || !out_fp16) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    const size_t V = h->cfg.vocab_size;
    const size_t row_bytes = V * sizeof(uint16_t);
    if (h->current_pos == 0) return -2;
    const size_t last_row = (size_t)(h->current_pos - 1);
    const char* src = (const char*)h->bufs.logits->contents() + last_row * row_bytes;
    std::memcpy(out_fp16, src, row_bytes);
    return 0;
}

extern "C" void sk_gemma4_set_dump_enabled(sk_gemma4_handle* hp, int enabled) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    h->dump_enabled = (enabled != 0);
}

extern "C" int sk_gemma4_dump_layer(sk_gemma4_handle* hp, const char* name, void* out_fp16) {
    if (!hp || !name || !out_fp16) return -1;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);
    if (!h->bufs.dump_stash) return -2;
    const size_t dm  = h->cfg.d_model;
    const size_t V   = h->cfg.vocab_size;
    const size_t nL  = h->cfg.n_layers;
    const char* base = (const char*)h->bufs.dump_stash->contents();

    auto copy_row = [&](size_t slot_elems, size_t count_elems) {
        std::memcpy(out_fp16, base + slot_elems * 2, count_elems * 2);
    };

    if (std::strcmp(name, "embed") == 0) {
        copy_row(0, dm); return 0;
    }
    if (std::strcmp(name, "final_norm") == 0) {
        const size_t slot = (1 + 4 * nL) * dm;
        copy_row(slot, dm); return 0;
    }
    if (std::strcmp(name, "logits") == 0) {
        const size_t slot = (1 + 4 * nL + 1) * dm;
        copy_row(slot, V); return 0;
    }
    // L0-extra slots (pre/post-RoPE q/k, attn-pre): laid out after logits.
    {
        const size_t base_extra = (1 + 4 * nL + 1) * dm + V;
        const size_t hd_max_e   = (h->cfg.head_dim_local > h->cfg.head_dim_global)
                                  ? h->cfg.head_dim_local : h->cfg.head_dim_global;
        const size_t n_kv_max_e = (h->cfg.n_kv_heads_local > h->cfg.n_kv_heads_global)
                                  ? h->cfg.n_kv_heads_local : h->cfg.n_kv_heads_global;
        const size_t H_n  = h->cfg.n_heads;
        const size_t off_qn = base_extra;
        const size_t off_kn = off_qn + H_n * hd_max_e;
        const size_t off_qr = off_kn + n_kv_max_e * hd_max_e;
        const size_t off_kr = off_qr + H_n * hd_max_e;
        const size_t off_ap = off_kr + n_kv_max_e * hd_max_e;
        // For L0 we know layer-type is local → head_dim_local, n_kv_heads_local.
        // (E2B/E4B have L0 sliding.) Return contiguous H*hd_local or n_kv*hd_local.
        const size_t hd_l   = h->cfg.head_dim_local;
        const size_t n_kv_l = h->cfg.n_kv_heads_local;
        if (std::strcmp(name, "L0.q_normed") == 0) { copy_row(off_qn, H_n  * hd_l); return 0; }
        if (std::strcmp(name, "L0.k_normed") == 0) { copy_row(off_kn, n_kv_l * hd_l); return 0; }
        if (std::strcmp(name, "L0.q_rope")   == 0) { copy_row(off_qr, H_n  * hd_l); return 0; }
        if (std::strcmp(name, "L0.k_rope")   == 0) { copy_row(off_kr, n_kv_l * hd_l); return 0; }
        if (std::strcmp(name, "L0.attn_pre") == 0) { copy_row(off_ap, H_n  * hd_l); return 0; }
    }
    // Parse "L{L}.<tag>"
    if (name[0] != 'L') return -3;
    const char* dot = std::strchr(name, '.');
    if (!dot) return -3;
    char numbuf[16] = {0};
    size_t numlen = (size_t)(dot - (name + 1));
    if (numlen == 0 || numlen >= sizeof(numbuf)) return -3;
    std::memcpy(numbuf, name + 1, numlen);
    uint32_t L = (uint32_t)std::atoi(numbuf);
    if (L >= nL) return -4;
    const char* tag = dot + 1;
    size_t off = 1 + 4 * (size_t)L;
    if (std::strcmp(tag, "x_norm") == 0) { copy_row((off + 0) * dm, dm); return 0; }
    if (std::strcmp(tag, "attn")   == 0) { copy_row((off + 1) * dm, dm); return 0; }
    if (std::strcmp(tag, "mlp")    == 0) { copy_row((off + 2) * dm, dm); return 0; }
    if (std::strcmp(tag, "out")    == 0) { copy_row((off + 3) * dm, dm); return 0; }
    return -5;
}

extern "C" void sk_gemma4_destroy(sk_gemma4_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::gemma4::Handle*>(hp);

    auto rel = [](MTL::Buffer* b) { if (b) b->release(); };
    rel(h->weights.w_embed); rel(h->weights.w_ple_table);
    rel(h->weights.w_per_layer_input_gate);
    rel(h->weights.w_per_layer_projection);
    rel(h->weights.w_layer_scalar);
    rel(h->weights.w_post_per_layer_input_norm);
    rel(h->weights.w_per_layer_model_projection);
    rel(h->weights.w_per_layer_projection_norm);
    rel(h->weights.w_pre_attn_norm); rel(h->weights.w_post_attn_norm);
    rel(h->weights.w_pre_feedforward_layernorm);
    rel(h->weights.w_post_feedforward_layernorm);
    rel(h->weights.w_final_norm);
    rel(h->weights.w_qkv); rel(h->weights.w_out);
    rel(h->weights.gamma_q); rel(h->weights.gamma_k);
    rel(h->weights.w_gate); rel(h->weights.w_up); rel(h->weights.w_down);
    rel(h->weights.cos_local); rel(h->weights.sin_local);
    rel(h->weights.cos_global); rel(h->weights.sin_global);

    for (uint32_t L = 0; L < h->layer_caches.size(); ++L) {
        if (L < h->kv_source_layer.size() && h->kv_source_layer[L] >= 0) continue;
        rel(h->layer_caches[L].k);
        rel(h->layer_caches[L].v);
    }

    rel(h->bufs.input_ids); rel(h->bufs.output_id);
    rel(h->bufs.x_a); rel(h->bufs.x_b); rel(h->bufs.logits);
    rel(h->bufs.x_norm); rel(h->bufs.qkv_packed);
    rel(h->bufs.q_norm); rel(h->bufs.k_tmp); rel(h->bufs.v_tmp);
    rel(h->bufs.attn_out); rel(h->bufs.o_proj); rel(h->bufs.y_attn);
    rel(h->bufs.m_in); rel(h->bufs.m_out); rel(h->bufs.y_out);
    rel(h->bufs.per_layer_inputs); rel(h->bufs.ple_ctx_proj); rel(h->bufs.ple_gate_out);
    rel(h->bufs.ple_gated); rel(h->bufs.ple_proj_back);
    rel(h->bufs.dump_stash);

    delete h;
}
