//
//  launcher.c++ — Qwen3-32B (dense) inference launcher.
//
//  Allocates buffers, resolves PSOs, wires them into qwen_model.h::dispatch_model.
//  Mirrors models/deepseek/launcher.c++ but simpler — no MoE, no MLA, no quant
//  routing on routed-experts.
//
//  Audit (temp/qwen/) confirmed: zero Qwen-specific kernels needed; every op
//  composes from existing SK kernels. Per-head Q/K RMSNorm (a Qwen3 quirk that
//  Qwen2 didn't have) is handled by reusing the standard `rmsnorm` kernel on a
//  reshaped (T·n_heads, head_dim) view.
//

#include "launcher.h"
#include "qwen_model.h"
#include "../../kernels/runtime_bindings.h"

#include <cstring>
#include <cstdlib>
#include <vector>

namespace meow { namespace qwen {

struct Handle {
    sk_qwen_config cfg;
    uint32_t       current_pos = 0;

    ModelPSOs    psos;
    ModelWeights weights;
    ModelBuffers bufs;
    std::vector<LayerCache> layer_caches;
    std::vector<MTL::Buffer*> k_caches;
    std::vector<MTL::Buffer*> v_caches;
};

static MTL::Buffer* alloc_zero(MTL::Device* dev, size_t bytes) {
    auto* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (b) std::memset(b->contents(), 0, bytes);
    return b;
}

static bool resolve_psos(ModelPSOs& P) {
    P.layer.rmsnorm        = sk::bindings_pso("rmsnorm");
    P.layer.gemm           = sk::bindings_pso("gemm_fp16");
    P.layer.split_packed   = sk::bindings_pso("split_packed");
    P.layer.rope_qk        = sk::bindings_pso("rope_qk");
    P.layer.attn           = sk::bindings_pso("mha_causal");
    P.layer.kv_cache_write = sk::bindings_pso("kv_cache_write");
    P.layer.add            = sk::bindings_pso("add_f16");
    P.layer.add_rmsnorm    = sk::bindings_pso("add_rmsnorm");
    P.layer.gated_mlp      = sk::bindings_pso("gated_mlp");
    P.embedding_lookup     = sk::bindings_pso("embedding_lookup");
    P.argmax               = sk::bindings_pso("argmax");

    #define _CK(name, val) if (!(val)) { std::fprintf(stderr, "qwen launcher: missing PSO " name "\n"); return false; }
    _CK("rmsnorm",          P.layer.rmsnorm);
    _CK("gemm_fp16",        P.layer.gemm);
    _CK("split_packed",     P.layer.split_packed);
    _CK("rope_qk",          P.layer.rope_qk);
    _CK("mha_causal",       P.layer.attn);
    _CK("kv_cache_write",   P.layer.kv_cache_write);
    _CK("add_f16",          P.layer.add);
    _CK("add_rmsnorm",      P.layer.add_rmsnorm);
    _CK("gated_mlp",        P.layer.gated_mlp);
    _CK("embedding_lookup", P.embedding_lookup);
    _CK("argmax",           P.argmax);
    #undef _CK
    return true;
}

}}  // namespace meow::qwen

extern "C" sk_qwen_handle* sk_qwen_create(const sk_qwen_config* cfg) {
    if (!cfg) return nullptr;
    auto* dev = sk::bindings_device();
    if (!dev) return nullptr;

    auto* h = new meow::qwen::Handle();
    h->cfg = *cfg;
    if (!meow::qwen::resolve_psos(h->psos)) { delete h; return nullptr; }

    using namespace meow::qwen;
    const uint32_t T_max = cfg->batch * cfg->seq_max;
    const uint32_t hd    = cfg->head_dim;
    const uint32_t qkv_N = (cfg->n_heads + 2 * cfg->n_kv_heads) * hd;

    // Weights
    h->weights.w_embed         = alloc_zero(dev, (size_t)cfg->vocab_size * cfg->d_model * 2);
    h->weights.w_pre_attn_norm = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_qkv           = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * qkv_N * 2);
    h->weights.w_q_norm        = alloc_zero(dev, (size_t)cfg->n_layers * hd * 2);
    h->weights.w_k_norm        = alloc_zero(dev, (size_t)cfg->n_layers * hd * 2);
    h->weights.w_o             = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_heads * hd * cfg->d_model * 2);
    h->weights.w_pre_mlp_norm  = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_final_norm    = alloc_zero(dev, (size_t)cfg->d_model * 2);
    h->weights.w_gate          = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * cfg->n_int * 2);
    h->weights.w_up            = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * cfg->n_int * 2);
    h->weights.w_down          = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_int * cfg->d_model * 2);

    // Per-layer K, V caches (full cache; GQA → n_kv_heads not n_heads)
    h->layer_caches.resize(cfg->n_layers);
    h->k_caches.resize(cfg->n_layers);
    h->v_caches.resize(cfg->n_layers);
    for (uint32_t L = 0; L < cfg->n_layers; ++L) {
        const size_t kv_bytes = (size_t)cfg->batch * cfg->n_kv_heads * cfg->cache_max * hd * 2;
        h->k_caches[L] = alloc_zero(dev, kv_bytes);
        h->v_caches[L] = alloc_zero(dev, kv_bytes);
        h->layer_caches[L].k = h->k_caches[L];
        h->layer_caches[L].v = h->v_caches[L];
    }
    h->weights.layer_caches = h->layer_caches.data();

    // Scratch
    h->bufs.input_ids  = alloc_zero(dev, (size_t)T_max * sizeof(int32_t));
    h->bufs.output_id  = alloc_zero(dev, (size_t)cfg->batch * sizeof(int32_t));
    h->bufs.x_a        = alloc_zero(dev, (size_t)T_max * cfg->d_model * 2);
    h->bufs.x_b        = alloc_zero(dev, (size_t)T_max * cfg->d_model * 2);
    h->bufs.logits     = alloc_zero(dev, (size_t)T_max * cfg->vocab_size * 2);
    h->bufs.rope_pos   = alloc_zero(dev, (size_t)T_max * sizeof(int32_t));

    h->bufs.x_norm     = alloc_zero(dev, (size_t)T_max * cfg->d_model * 2);
    h->bufs.qkv_packed = alloc_zero(dev, (size_t)T_max * qkv_N * 2);
    h->bufs.q          = alloc_zero(dev, (size_t)T_max * cfg->n_heads    * hd * 2);
    h->bufs.kv_pack    = alloc_zero(dev, (size_t)T_max * 2 * cfg->n_kv_heads * hd * 2);
    h->bufs.k_tmp      = alloc_zero(dev, (size_t)T_max * cfg->n_kv_heads * hd * 2);
    h->bufs.v_tmp      = alloc_zero(dev, (size_t)T_max * cfg->n_kv_heads * hd * 2);
    h->bufs.attn_out   = alloc_zero(dev, (size_t)T_max * cfg->n_heads    * hd * 2);
    h->bufs.o_proj     = alloc_zero(dev, (size_t)T_max * cfg->d_model * 2);
    h->bufs.y_attn     = alloc_zero(dev, (size_t)T_max * cfg->d_model * 2);
    h->bufs.m_in       = alloc_zero(dev, (size_t)T_max * cfg->d_model * 2);
    h->bufs.mlp_out    = alloc_zero(dev, (size_t)T_max * cfg->d_model * 2);

    return reinterpret_cast<sk_qwen_handle*>(h);
}

extern "C" int sk_qwen_load_weights(sk_qwen_handle* hp, const sk_qwen_weights* w) {
    if (!hp || !w) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    auto cp = [](MTL::Buffer* dst, const void* src) {
        if (dst && src) std::memcpy(dst->contents(), src, dst->length());
    };
    cp(h->weights.w_embed,         w->w_embed);
    cp(h->weights.w_pre_attn_norm, w->w_pre_attn_norm);
    cp(h->weights.w_qkv,           w->w_qkv);
    cp(h->weights.w_q_norm,        w->w_q_norm);
    cp(h->weights.w_k_norm,        w->w_k_norm);
    cp(h->weights.w_o,             w->w_o);
    cp(h->weights.w_pre_mlp_norm,  w->w_pre_mlp_norm);
    cp(h->weights.w_final_norm,    w->w_final_norm);
    cp(h->weights.w_gate,          w->w_gate);
    cp(h->weights.w_up,            w->w_up);
    cp(h->weights.w_down,          w->w_down);
    return 0;
}

extern "C" void sk_qwen_reset(sk_qwen_handle* hp) {
    if (!hp) return;
    reinterpret_cast<meow::qwen::Handle*>(hp)->current_pos = 0;
}

extern "C" int sk_qwen_forward(sk_qwen_handle* hp,
                               const int* input_ids, uint32_t seq,
                               int* output_id) {
    if (!hp || !input_ids || !output_id) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    if (seq == 0 || seq > h->cfg.seq_max) return -2;
    if (h->current_pos + seq > h->cfg.cache_max) return -4;

    auto* q = sk::bindings_queue();
    if (!q) return -3;

    std::memcpy(h->bufs.input_ids->contents(), input_ids,
                (size_t)h->cfg.batch * seq * sizeof(int32_t));

    int32_t* pos = (int32_t*)h->bufs.rope_pos->contents();
    for (uint32_t i = 0; i < seq; ++i) pos[i] = (int32_t)(h->current_pos + i);

    meow::qwen::ModelParams mp;
    mp.batch          = h->cfg.batch;
    mp.seq            = seq;
    mp.n_layers       = h->cfg.n_layers;
    mp.d_model        = h->cfg.d_model;
    mp.n_heads        = h->cfg.n_heads;
    mp.n_kv_heads     = h->cfg.n_kv_heads;
    mp.head_dim       = h->cfg.head_dim;
    mp.n_int          = h->cfg.n_int;
    mp.cache_max      = h->cfg.cache_max;
    mp.vocab_size     = h->cfg.vocab_size;
    mp.eps            = h->cfg.eps;
    mp.current_pos    = h->current_pos;
    mp.rope_n_ctx_orig = h->cfg.rope_n_ctx_orig;
    mp.rope_freq_base  = h->cfg.rope_freq_base;
    mp.rope_freq_scale = h->cfg.rope_freq_scale;
    mp.rope_ext_factor = h->cfg.rope_ext_factor;
    mp.rope_attn_factor = h->cfg.rope_attn_factor;
    mp.rope_beta_fast  = h->cfg.rope_beta_fast;
    mp.rope_beta_slow  = h->cfg.rope_beta_slow;

    auto* cmd = q->commandBuffer();
    meow::qwen::dispatch_model(cmd, h->psos, h->weights, h->bufs, mp);
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    h->current_pos += seq;
    std::memcpy(output_id, h->bufs.output_id->contents(),
                (size_t)h->cfg.batch * sizeof(int32_t));
    return 0;
}

extern "C" void sk_qwen_destroy(sk_qwen_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    auto rel = [](MTL::Buffer* b) { if (b) b->release(); };

    rel(h->weights.w_embed); rel(h->weights.w_pre_attn_norm);
    rel(h->weights.w_qkv); rel(h->weights.w_q_norm); rel(h->weights.w_k_norm); rel(h->weights.w_o);
    rel(h->weights.w_pre_mlp_norm); rel(h->weights.w_final_norm);
    rel(h->weights.w_gate); rel(h->weights.w_up); rel(h->weights.w_down);
    for (auto* b : h->k_caches) rel(b);
    for (auto* b : h->v_caches) rel(b);
    rel(h->bufs.input_ids); rel(h->bufs.output_id);
    rel(h->bufs.x_a); rel(h->bufs.x_b); rel(h->bufs.logits); rel(h->bufs.rope_pos);
    rel(h->bufs.x_norm); rel(h->bufs.qkv_packed); rel(h->bufs.q);
    rel(h->bufs.kv_pack); rel(h->bufs.k_tmp); rel(h->bufs.v_tmp);
    rel(h->bufs.attn_out); rel(h->bufs.o_proj); rel(h->bufs.y_attn);
    rel(h->bufs.m_in); rel(h->bufs.mlp_out);
    delete h;
}
