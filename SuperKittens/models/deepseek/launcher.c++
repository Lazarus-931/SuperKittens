//  launcher.c++ — DeepSeek V4 Flash inference launcher.

#include "launcher.h"
#include "deepseek_model.h"
#include "../../kernels/runtime_bindings.h"

#include <cstring>
#include <cstdlib>
#include <vector>

namespace meow { namespace deepseek {

struct Handle {
    sk_deepseek_config cfg;
    uint32_t           current_pos = 0;

    ModelPSOs    psos;
    ModelWeights weights;
    ModelBuffers bufs;
    std::vector<LayerCache> layer_caches;

    // K/V caches (we use the "cache full K, V" path here, not compressed).
    std::vector<MTL::Buffer*> k_caches;
    std::vector<MTL::Buffer*> v_caches;
};

static MTL::Buffer* alloc_zero(MTL::Device* dev, size_t bytes) {
    auto* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (b) std::memset(b->contents(), 0, bytes);
    return b;
}

static MTL::ComputePipelineState* resolve_fa_vec_pso(MTL::Library* lib,
                                                    MTL::Device*  dev,
                                                    uint32_t dk, uint32_t dv)
{
    char name[64];
    std::snprintf(name, sizeof(name),
                  "kernel_flash_attn_ext_vec_f16_dk%u_dv%u", dk, dv);
    auto* fcv = MTL::FunctionConstantValues::alloc()->init();
    // Metal bool is 1 byte; use uint8_t to avoid the C++ bool-size gotcha.
    uint8_t t = 1, f = 0;
    int32_t ns10 = (int32_t)(dk * sizeof(uint16_t));
    int32_t ns20 = (int32_t)(dv * sizeof(uint16_t));
    int32_t nsg = 4, nwg = 1;
    fcv->setConstantValue(&t,    MTL::DataTypeBool, NS::UInteger(400));   // has_mask
    fcv->setConstantValue(&f,    MTL::DataTypeBool, NS::UInteger(401));   // has_sinks
    fcv->setConstantValue(&f,    MTL::DataTypeBool, NS::UInteger(402));   // has_bias
    fcv->setConstantValue(&f,    MTL::DataTypeBool, NS::UInteger(403));   // has_scap
    fcv->setConstantValue(&f,    MTL::DataTypeBool, NS::UInteger(404));   // has_kvpad
    fcv->setConstantValue(&ns10, MTL::DataTypeInt, NS::UInteger(420));
    fcv->setConstantValue(&ns20, MTL::DataTypeInt, NS::UInteger(421));
    fcv->setConstantValue(&nsg,  MTL::DataTypeInt, NS::UInteger(422));
    fcv->setConstantValue(&nwg,  MTL::DataTypeInt, NS::UInteger(423));

    NS::Error* err = nullptr;
    auto* fn = lib->newFunction(
        NS::String::string(name, NS::UTF8StringEncoding), fcv, &err);
    fcv->release();
    if (!fn) {
        std::fprintf(stderr, "ds4: flash_attn newFunction failed (%s): %s\n",
                     name, err ? err->localizedDescription()->utf8String() : "?");
        return nullptr;
    }
    auto* pso = dev->newComputePipelineState(fn, &err);
    fn->release();
    return pso;
}

// PSO factory for the q4_K per-expert matvec + shared_swiglu, which read
// FC slot 600 (NSG, short) and 601 (nxpsg, short).
static MTL::ComputePipelineState* resolve_mvid_pso(MTL::Library* lib,
                                                   MTL::Device* dev,
                                                   const char* name) {
    auto* fcv = MTL::FunctionConstantValues::alloc()->init();
    int16_t nsg = 4, nxpsg = 4;
    fcv->setConstantValue(&nsg,   MTL::DataTypeShort, NS::UInteger(600));
    fcv->setConstantValue(&nxpsg, MTL::DataTypeShort, NS::UInteger(601));
    NS::Error* err = nullptr;
    auto* fn = lib->newFunction(
        NS::String::string(name, NS::UTF8StringEncoding), fcv, &err);
    fcv->release();
    if (!fn) {
        std::fprintf(stderr, "ds: mvid newFunction failed (%s): %s\n",
                     name, err ? err->localizedDescription()->utf8String() : "?");
        return nullptr;
    }
    auto* pso = dev->newComputePipelineState(fn, &err);
    fn->release();
    return pso;
}

static bool resolve_psos(ModelPSOs& P, uint32_t dk, uint32_t dv) {
    P.layer.rmsnorm        = sk::bindings_pso("rmsnorm");
    P.layer.rmsnorm_t1     = sk::bindings_pso("rmsnorm_t1");  // optional T=1 fast path
    P.layer.gemm           = sk::bindings_pso("gemm_fp16");
    P.layer.rope_tail      = sk::bindings_pso("kernel_dsv4_rope_tail_f32");
    P.layer.rope_interleave = sk::bindings_pso("rope_interleave_f32");
    P.layer.router_v3      = sk::bindings_pso("moe_router_v3");
    P.layer.kv_cache_write = sk::bindings_pso("kv_cache_write");
    P.layer.add            = sk::bindings_pso("add_f16");
    P.layer.add_rmsnorm    = sk::bindings_pso("add_rmsnorm");
    P.layer.cast_h2f       = sk::bindings_pso("cast_f16_to_f32");
    P.layer.cast_f2h       = sk::bindings_pso("cast_f32_to_f16");
    P.layer.causal_mask_fill = sk::bindings_pso("causal_mask_fill");
    P.layer.gated_mlp      = sk::bindings_pso("gated_mlp");
    P.layer.silu_mul       = sk::bindings_pso("silu_mul_f16");
    P.layer.kv_up_pair     = sk::bindings_pso("kv_up_pair");
    P.layer.split_packed       = sk::bindings_pso("split_packed");

    P.layer.moe.router              = sk::bindings_pso("moe_router");
    P.layer.moe.swiglu_pair         = sk::bindings_pso("moe_swiglu_pair");
    P.layer.moe.down_scatter        = sk::bindings_pso("moe_down_scatter");
    P.layer.moe.swiglu_pair_iq2xxs  = sk::bindings_pso("moe_swiglu_pair_iq2xxs");
    P.layer.moe.down_scatter_q2k    = sk::bindings_pso("moe_down_scatter_q2k");

    P.layer.flash_attn_vec = resolve_fa_vec_pso(
        sk::bindings_library(), sk::bindings_device(), dk, dv);

    // V2-Lite MLA + Q4_K MoE path.
    P.layer.router_v2     = sk::bindings_pso("moe_router");
    P.layer.mla_decode_v2 = sk::bindings_pso("kernel_mla_decode_v2_f16_dk192_dv128");
    P.layer.mla_kv_write  = sk::bindings_pso("deepseek_mla_kv_write");
    P.layer.moe_mv_gate   = resolve_mvid_pso(
        sk::bindings_library(), sk::bindings_device(), "deepseek_mul_mv_id_q4_K");
    P.layer.moe_mv_down   = resolve_mvid_pso(
        sk::bindings_library(), sk::bindings_device(), "deepseek_mul_mv_id_q8_0");
    P.layer.moe_swiglu_f32  = sk::bindings_pso("deepseek_moe_swiglu_f32");
    P.layer.moe_scatter_add = sk::bindings_pso("deepseek_moe_scatter_add_f32");

    // Native K-quant matvec (decode) so dense/attn/shared/LM-head stay quantized.
    P.layer.q4k_matvec  = sk::bindings_pso("q4k_matvec");
    P.layer.q6k_matvec  = sk::bindings_pso("q6k_matvec");
    P.layer.q8_0_matvec = sk::bindings_pso("q8_0_matvec");

    P.embedding_lookup = sk::bindings_pso("embedding_lookup");
    P.argmax           = sk::bindings_pso("argmax");
    P.argmax_partial   = sk::bindings_pso("argmax_partial");
    P.argmax_reduce    = sk::bindings_pso("argmax_reduce");

    #define _CK(name, val) if (!(val)) { std::fprintf(stderr, "ds4 launcher: missing PSO " name "\n"); return false; }
    _CK("rmsnorm",        P.layer.rmsnorm);
    _CK("gemm_fp16",      P.layer.gemm);
    _CK("rope_tail",      P.layer.rope_tail);
    _CK("rope_interleave_f32", P.layer.rope_interleave);
    _CK("moe_router_v3",  P.layer.router_v3);
    _CK("kv_cache_write", P.layer.kv_cache_write);
    _CK("add_f16",        P.layer.add);
    _CK("add_rmsnorm",    P.layer.add_rmsnorm);
    _CK("cast_f16_to_f32", P.layer.cast_h2f);
    _CK("cast_f32_to_f16", P.layer.cast_f2h);
    _CK("causal_mask_fill", P.layer.causal_mask_fill);
    _CK("gated_mlp",      P.layer.gated_mlp);
    _CK("silu_mul_f16",   P.layer.silu_mul);
    _CK("kv_up_pair",     P.layer.kv_up_pair);
    _CK("split_packed",       P.layer.split_packed);
    _CK("moe_router",            P.layer.moe.router);
    _CK("moe_router(v2)",        P.layer.router_v2);
    _CK("mla_decode_v2",         P.layer.mla_decode_v2);
    _CK("mla_kv_write",          P.layer.mla_kv_write);
    _CK("moe_mul_mv_id_q4_K",    P.layer.moe_mv_gate);
    _CK("moe_mul_mv_id_q8_0",    P.layer.moe_mv_down);
    _CK("moe_swiglu_f32",        P.layer.moe_swiglu_f32);
    _CK("moe_scatter_add_f32",   P.layer.moe_scatter_add);
    _CK("q4k_matvec",            P.layer.q4k_matvec);
    _CK("q6k_matvec",            P.layer.q6k_matvec);
    _CK("q8_0_matvec",           P.layer.q8_0_matvec);
    _CK("embedding_lookup", P.embedding_lookup);
    _CK("argmax",         P.argmax);
    #undef _CK
    return true;
}

}}  // namespace meow::deepseek

extern "C" sk_deepseek_handle* sk_deepseek_create(const sk_deepseek_config* cfg) {
    if (!cfg) return nullptr;
    auto* dev = sk::bindings_device();
    if (!dev) return nullptr;

    auto* h = new meow::deepseek::Handle();
    h->cfg = *cfg;

    const uint32_t dk = cfg->qk_nope_dim + cfg->qk_rope_dim;
    if (!meow::deepseek::resolve_psos(h->psos, dk, cfg->v_head_dim)) {
        delete h;
        return nullptr;
    }

    using namespace meow::deepseek;

    const uint32_t T_max = cfg->batch * cfg->seq_max;

    // Sizes (fp16 = 2 bytes per scalar)
    const size_t x_bytes      = (size_t)T_max * cfg->d_model * 2;
    const size_t logits_bytes = (size_t)T_max * cfg->vocab_size * 2;

    // Weight buffers
    h->weights.w_embed         = alloc_zero(dev, (size_t)cfg->vocab_size * cfg->d_model * 2);
    // Patch F: separate buffer for lm_head. Loader aliases (sets to w_embed) for tied case.
    h->weights.w_lm_head       = alloc_zero(dev, (size_t)cfg->vocab_size * cfg->d_model * 2);
    h->weights.w_pre_attn_norm = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    // V2-Lite has no Q-LoRA: q_proj maps d_model -> n_heads*dk directly and the
    // loader writes it into w_q_b. Size w_q_b for max(q_lora_rank, d_model) as
    // the input dim so both paths fit. w_q_a / w_q_a_norm get a 1-elem stub.
    const uint32_t q_in = cfg->q_lora_rank ? cfg->q_lora_rank : cfg->d_model;
    const uint32_t q_lr1 = cfg->q_lora_rank ? cfg->q_lora_rank : 1u;
    h->weights.w_q_a           = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * q_lr1 * 2);
    h->weights.w_q_a_norm      = alloc_zero(dev, (size_t)cfg->n_layers * q_lr1 * 2);
    h->weights.w_q_b           = alloc_zero(dev, (size_t)cfg->n_layers * q_in * cfg->n_heads * dk * 2);
    h->weights.w_kv_a          = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * (cfg->kv_lora_rank + cfg->qk_rope_dim) * 2);
    h->weights.w_kv_a_norm     = alloc_zero(dev, (size_t)cfg->n_layers * cfg->kv_lora_rank * 2);
    h->weights.w_kv_b          = alloc_zero(dev, (size_t)cfg->n_layers * cfg->kv_lora_rank
                                             * cfg->n_heads * (cfg->qk_nope_dim + cfg->v_head_dim) * 2);
    h->weights.w_o             = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_heads * cfg->v_head_dim * cfg->d_model * 2);
    h->weights.w_pre_mlp_norm  = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_final_norm    = alloc_zero(dev, (size_t)cfg->d_model * 2);

    h->weights.w_shared_gate = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * cfg->shared_n_int * 2);
    h->weights.w_shared_up   = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * cfg->shared_n_int * 2);
    h->weights.w_shared_down = alloc_zero(dev, (size_t)cfg->n_layers * cfg->shared_n_int * cfg->d_model * 2);

    // Leading-dense-layer MLP (fp16), sized for the first_k_dense_replace layers.
    {
        const uint32_t fkd  = cfg->first_k_dense_replace ? cfg->first_k_dense_replace : 1u;
        const uint32_t dint = cfg->dense_n_int ? cfg->dense_n_int : cfg->shared_n_int;
        h->weights.w_dense_gate = alloc_zero(dev, (size_t)fkd * cfg->d_model * dint * 2);
        h->weights.w_dense_up   = alloc_zero(dev, (size_t)fkd * cfg->d_model * dint * 2);
        h->weights.w_dense_down = alloc_zero(dev, (size_t)fkd * dint * cfg->d_model * 2);
    }

    h->weights.w_router = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * cfg->n_expert * 2);
    // Patch G: per-layer e_score_correction_bias (fp32). Allocated unconditionally;
    // loader leaves zeros for V2-Lite, kernel ignores when has_bias=0.
    h->weights.router_bias = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_expert * 4);
    // Routed-expert weights. V2-Lite Q4_K_M (moe_quant==2): gate/up are Q4_K
    // (144 B / 256 weights), DOWN is Q8_0 (34 B / 32 weights). INT2_DS4 keeps
    // the ds4 V4-Flash layout. Layer 0 is dense (leading_dense_block_count=1)
    // and stores its wide MLP in the shared-expert buffers instead — the routed
    // buffers reserve a full per-layer slab for every layer for simple indexing.
    const bool   int2     = (cfg->moe_quant == 1);
    const bool   q4k      = (cfg->moe_quant == 2);
    const size_t gate_blk = int2 ? 66 : (q4k ? 144 : 512);   // per 256 weights
    const size_t down_q8_blk = 34;                            // per 32 weights
    const size_t n_blocks_gate_per_e = (size_t)cfg->n_int * (cfg->d_model / 256);
    const size_t n_blocks_down_per_e = q4k
        ? (size_t)cfg->d_model * (cfg->n_int / 32)            // Q8_0: 32-wide blocks
        : (size_t)cfg->d_model * (cfg->n_int / 256);
    const size_t down_blk = q4k ? down_q8_blk : (int2 ? 84 : 512);
    h->weights.w_gate = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_expert *
                                        n_blocks_gate_per_e * gate_blk);
    h->weights.w_up   = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_expert *
                                        n_blocks_gate_per_e * gate_blk);
    h->weights.w_down = alloc_zero(dev, (size_t)cfg->n_layers * cfg->n_expert *
                                        n_blocks_down_per_e * down_blk);

    // Per-layer K, V caches (cache the full decompressed K/V).
    h->layer_caches.resize(cfg->n_layers);
    h->k_caches.resize(cfg->n_layers);
    h->v_caches.resize(cfg->n_layers);
    for (uint32_t L = 0; L < cfg->n_layers; ++L) {
        const size_t kb = (size_t)cfg->batch * cfg->n_heads * cfg->cache_max * dk * 2;
        const size_t vb = (size_t)cfg->batch * cfg->n_heads * cfg->cache_max * cfg->v_head_dim * 2;
        h->k_caches[L] = alloc_zero(dev, kb);
        h->v_caches[L] = alloc_zero(dev, vb);
        // LayerCache wires c_kv/k_pe into the model's compressed-cache slots,
        // but in the "cache full K, V" path we just reuse them as K/V cache.
        h->layer_caches[L].c_kv = h->k_caches[L];
        h->layer_caches[L].k_pe = h->v_caches[L];
    }
    h->weights.layer_caches = h->layer_caches.data();

    // Scratch buffers
    h->bufs.input_ids   = alloc_zero(dev, (size_t)T_max * sizeof(int32_t));
    h->bufs.output_id   = alloc_zero(dev, (size_t)cfg->batch * sizeof(int32_t));
    h->bufs.x_a         = alloc_zero(dev, x_bytes);
    h->bufs.x_b         = alloc_zero(dev, x_bytes);
    h->bufs.logits      = alloc_zero(dev, logits_bytes);
    h->bufs.rope_pos    = alloc_zero(dev, (size_t)T_max * sizeof(int32_t));

    h->bufs.x_norm      = alloc_zero(dev, x_bytes);
    h->bufs.q_a         = alloc_zero(dev, (size_t)T_max * (cfg->q_lora_rank ? cfg->q_lora_rank : 1u) * 2);
    h->bufs.q_packed    = alloc_zero(dev, (size_t)T_max * cfg->n_heads * dk * 2);
    h->bufs.q_packed_f32 = alloc_zero(dev, (size_t)T_max * cfg->n_heads * dk * 4);
    h->bufs.k_pe_f32     = alloc_zero(dev, (size_t)T_max * cfg->qk_rope_dim * 4);
    h->bufs.attn_out_f32 = alloc_zero(dev, (size_t)T_max * cfg->n_heads * cfg->v_head_dim * 4);
    h->bufs.causal_mask  = alloc_zero(dev, (size_t)cfg->seq_max * cfg->cache_max * 2);
    h->bufs.kv_a_packed = alloc_zero(dev, (size_t)T_max * (cfg->kv_lora_rank + cfg->qk_rope_dim) * 2);
    // split_packed writes the compressed-KV latent (c_kv) and the RoPE part
    // (k_pe) out of kv_a_packed; both are read downstream (kv_a_norm, RoPE,
    // kv_up). They were never allocated → NULL bindings (undefined GPU reads).
    h->bufs.c_kv        = alloc_zero(dev, (size_t)T_max * cfg->kv_lora_rank * 2);
    h->bufs.k_pe        = alloc_zero(dev, (size_t)T_max * cfg->qk_rope_dim * 2);
    h->bufs.k_no_pe     = alloc_zero(dev, (size_t)T_max * cfg->n_heads * cfg->qk_nope_dim * 2);
    h->bufs.v           = alloc_zero(dev, (size_t)T_max * cfg->n_heads * cfg->v_head_dim * 2);
    h->bufs.attn_out    = alloc_zero(dev, (size_t)T_max * cfg->n_heads * cfg->v_head_dim * 2);
    h->bufs.o_proj      = alloc_zero(dev, x_bytes);
    h->bufs.y_attn      = alloc_zero(dev, x_bytes);
    h->bufs.m_in        = alloc_zero(dev, x_bytes);
    h->bufs.shared_mid  = alloc_zero(dev, (size_t)T_max * cfg->shared_n_int * 2);
    h->bufs.shared_out  = alloc_zero(dev, x_bytes);
    // Dense/shared MLP scratch sized for the widest n_int (dense MLP, e.g. 10944).
    {
        const uint32_t dni = cfg->dense_n_int ? cfg->dense_n_int : cfg->shared_n_int;
        const uint32_t max_n_int = (dni > cfg->shared_n_int) ? dni : cfg->shared_n_int;
        h->bufs.mlp_gate = alloc_zero(dev, (size_t)T_max * max_n_int * 2);
        h->bufs.mlp_up   = alloc_zero(dev, (size_t)T_max * max_n_int * 2);
    }
    h->bufs.moe_top_idx   = alloc_zero(dev, (size_t)T_max * cfg->top_k * sizeof(int32_t));
    h->bufs.moe_top_score = alloc_zero(dev, (size_t)T_max * cfg->top_k * 2);
    h->bufs.moe_hidden    = alloc_zero(dev, (size_t)T_max * cfg->top_k * cfg->n_int * 2);
    // fp32 scratch for the per-expert mul_mv_id MoE path.
    h->bufs.moe_x_f32     = alloc_zero(dev, (size_t)T_max * cfg->d_model * 4);
    h->bufs.moe_gate_f32  = alloc_zero(dev, (size_t)T_max * cfg->top_k * cfg->n_int * 4);
    h->bufs.moe_up_f32    = alloc_zero(dev, (size_t)T_max * cfg->top_k * cfg->n_int * 4);
    h->bufs.moe_mid_f32   = alloc_zero(dev, (size_t)T_max * cfg->top_k * cfg->n_int * 4);
    h->bufs.moe_down_f32  = alloc_zero(dev, (size_t)T_max * cfg->top_k * cfg->d_model * 4);

    // 2-pass argmax scratch.
    {
        constexpr uint32_t ELTS_PER_TG = 16384u;
        const uint32_t n_blocks = (cfg->vocab_size + ELTS_PER_TG - 1u) / ELTS_PER_TG;
        h->bufs.argmax_val_buf = alloc_zero(dev, (size_t)n_blocks * sizeof(float));
        h->bufs.argmax_idx_buf = alloc_zero(dev, (size_t)n_blocks * sizeof(int32_t));
    }

    // c_kv / k_pe scratch live in the same memory budget as q_packed / etc.
    // We allocate fresh:
    // (LayerBuffers exposes c_kv/k_pe but in ModelBuffers wiring they were
    // not added — we add them here as members of the temporary LayerBuffers
    // built inside dispatch_model. Since dispatch_model only references the
    // ModelBuffers fields, c_kv and k_pe in LayerBuffers default to nullptr
    // unless wired up. We wire them via the unused-slot trick: reuse v as
    // intermediate. Cleaner: add c_kv/k_pe scratch to ModelBuffers in a
    // future revision.)

    return reinterpret_cast<sk_deepseek_handle*>(h);
}

extern "C" int sk_deepseek_load_weights(sk_deepseek_handle* hp,
                                        const sk_deepseek_weights* w) {
    if (!hp || !w) return -1;
    auto* h = reinterpret_cast<meow::deepseek::Handle*>(hp);

    auto cp = [](MTL::Buffer* dst, const void* src) {
        if (dst && src) std::memcpy(dst->contents(), src, dst->length());
    };
    cp(h->weights.w_embed,         w->w_embed);
    if (w->w_lm_head) {
        cp(h->weights.w_lm_head,   w->w_lm_head);
    } else {
        // Tied case: release the unused lm_head allocation and alias to embed.
        if (h->weights.w_lm_head) h->weights.w_lm_head->release();
        h->weights.w_lm_head = h->weights.w_embed;
    }
    cp(h->weights.w_pre_attn_norm, w->w_pre_attn_norm);
    cp(h->weights.w_q_a,           w->w_q_a);
    cp(h->weights.w_q_a_norm,      w->w_q_a_norm);
    cp(h->weights.w_q_b,           w->w_q_b);
    cp(h->weights.w_kv_a,          w->w_kv_a);
    cp(h->weights.w_kv_a_norm,     w->w_kv_a_norm);
    cp(h->weights.w_kv_b,          w->w_kv_b);
    cp(h->weights.w_o,             w->w_o);
    cp(h->weights.w_pre_mlp_norm,  w->w_pre_mlp_norm);
    cp(h->weights.w_final_norm,    w->w_final_norm);
    cp(h->weights.w_shared_gate,   w->w_shared_gate);
    cp(h->weights.w_shared_up,     w->w_shared_up);
    cp(h->weights.w_shared_down,   w->w_shared_down);
    cp(h->weights.w_router,        w->w_router);
    if (w->router_bias) cp(h->weights.router_bias, w->router_bias);
    cp(h->weights.w_gate,          w->w_gate);
    cp(h->weights.w_up,            w->w_up);
    cp(h->weights.w_down,          w->w_down);
    return 0;
}

extern "C" void sk_deepseek_reset(sk_deepseek_handle* hp) {
    if (!hp) return;
    reinterpret_cast<meow::deepseek::Handle*>(hp)->current_pos = 0;
}

extern "C" int sk_deepseek_forward(sk_deepseek_handle* hp,
                                   const int* input_ids, uint32_t seq,
                                   int* output_id) {
    if (!hp || !input_ids || !output_id) return -1;
    auto* h = reinterpret_cast<meow::deepseek::Handle*>(hp);
    if (seq == 0 || seq > h->cfg.seq_max) return -2;
    if (h->current_pos + seq > h->cfg.cache_max) return -4;

    auto* q = sk::bindings_queue();
    if (!q) return -3;

    std::memcpy(h->bufs.input_ids->contents(), input_ids,
                (size_t)h->cfg.batch * seq * sizeof(int32_t));

    // Fill positions 0..seq-1 + current_pos for RoPE.
    int32_t* pos = (int32_t*)h->bufs.rope_pos->contents();
    for (uint32_t i = 0; i < seq; ++i) pos[i] = (int32_t)(h->current_pos + i);

    meow::deepseek::ModelParams mp;
    mp.batch          = h->cfg.batch;
    mp.seq            = seq;
    mp.n_layers       = h->cfg.n_layers;
    mp.d_model        = h->cfg.d_model;
    mp.n_int          = h->cfg.n_int;
    mp.shared_n_int   = h->cfg.shared_n_int;
    mp.dense_n_int    = h->cfg.dense_n_int ? h->cfg.dense_n_int : h->cfg.shared_n_int;
    mp.n_heads        = h->cfg.n_heads;
    mp.qk_nope_dim    = h->cfg.qk_nope_dim;
    mp.qk_rope_dim    = h->cfg.qk_rope_dim;
    mp.v_head_dim     = h->cfg.v_head_dim;
    mp.q_lora_rank    = h->cfg.q_lora_rank;
    mp.kv_lora_rank   = h->cfg.kv_lora_rank;
    mp.n_expert       = h->cfg.n_expert;
    mp.top_k          = h->cfg.top_k;
    mp.cache_max      = h->cfg.cache_max;
    mp.moe_quant      = (h->cfg.moe_quant == 1) ? meow::deepseek::MoeQuant::INT2_DS4
                                                : meow::deepseek::MoeQuant::FP16;
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
    mp.has_q_lora            = (h->cfg.has_q_lora != 0);
    mp.router_has_bias       = (h->cfg.router_has_bias != 0);
    mp.rope_interleave       = (h->cfg.rope_interleave != 0);
    mp.norm_topk_prob        = (h->cfg.norm_topk_prob != 0);
    mp.n_group               = h->cfg.n_group;
    mp.topk_group            = h->cfg.topk_group;
    mp.routed_scaling        = h->cfg.routed_scaling_factor;
    mp.mscale_all_dim        = h->cfg.mscale_all_dim;
    mp.rope_scaling_factor   = h->cfg.rope_scaling_factor;
    mp.first_k_dense_replace = h->cfg.first_k_dense_replace;

    auto* cmd = q->commandBuffer();
    meow::deepseek::dispatch_model(cmd, h->psos, h->weights, h->bufs, mp);
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    h->current_pos += seq;

    if (getenv("SK_DS_DEBUG")) {
        const uint32_t V = h->cfg.vocab_size;
        const uint32_t T = seq;
        const uint16_t* lg = (const uint16_t*)h->bufs.logits->contents();
        const uint16_t* row = lg + (size_t)(T - 1) * V;   // last position
        auto h2f = [](uint16_t h) {
            uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,bits;
            if(e==0){ if(m==0)bits=s<<31; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; bits=(s<<31)|(e<<23)|(m<<13);} }
            else if(e==0x1f) bits=(s<<31)|(0xff<<23)|(m<<13);
            else bits=(s<<31)|((e-15+127)<<23)|(m<<13);
            float f; std::memcpy(&f,&bits,4); return f; };
        int top[10]; float topv[10];
        for(int k=0;k<10;k++){top[k]=-1;topv[k]=-1e30f;}
        for(uint32_t v=0; v<V; ++v){ float f=h2f(row[v]);
            for(int k=0;k<10;k++){ if(f>topv[k]){ for(int j=9;j>k;j--){topv[j]=topv[j-1];top[j]=top[j-1];} topv[k]=f; top[k]=(int)v; break; } } }
        std::fprintf(stderr, "[SK_DS_DEBUG] pos=%u logits top10:", h->current_pos-1);
        for(int k=0;k<10;k++) std::fprintf(stderr, " %d=%.3f", top[k], topv[k]);
        std::fprintf(stderr, "\n");
    }

    std::memcpy(output_id, h->bufs.output_id->contents(),
                (size_t)h->cfg.batch * sizeof(int32_t));
    return 0;
}

extern "C" void sk_deepseek_destroy(sk_deepseek_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::deepseek::Handle*>(hp);
    auto rel = [](MTL::Buffer* b) { if (b) b->release(); };

    rel(h->weights.w_embed);
    if (h->weights.w_lm_head && h->weights.w_lm_head != h->weights.w_embed)
        rel(h->weights.w_lm_head);
    rel(h->weights.router_bias);
    rel(h->weights.w_pre_attn_norm);
    rel(h->weights.w_q_a); rel(h->weights.w_q_a_norm); rel(h->weights.w_q_b);
    rel(h->weights.w_kv_a); rel(h->weights.w_kv_a_norm); rel(h->weights.w_kv_b);
    rel(h->weights.w_o); rel(h->weights.w_pre_mlp_norm); rel(h->weights.w_final_norm);
    rel(h->weights.w_shared_gate); rel(h->weights.w_shared_up); rel(h->weights.w_shared_down);
    rel(h->weights.w_dense_gate); rel(h->weights.w_dense_up); rel(h->weights.w_dense_down);
    rel(h->weights.w_router); rel(h->weights.w_gate); rel(h->weights.w_up); rel(h->weights.w_down);
    for (auto* b : h->k_caches) rel(b);
    for (auto* b : h->v_caches) rel(b);

    rel(h->bufs.input_ids); rel(h->bufs.output_id);
    rel(h->bufs.x_a); rel(h->bufs.x_b); rel(h->bufs.logits); rel(h->bufs.rope_pos);
    rel(h->bufs.x_norm); rel(h->bufs.q_a); rel(h->bufs.q_packed);
    rel(h->bufs.q_packed_f32); rel(h->bufs.k_pe_f32); rel(h->bufs.attn_out_f32);
    rel(h->bufs.causal_mask);
    rel(h->bufs.kv_a_packed); rel(h->bufs.k_no_pe); rel(h->bufs.v);
    rel(h->bufs.c_kv); rel(h->bufs.k_pe);
    rel(h->bufs.attn_out); rel(h->bufs.o_proj); rel(h->bufs.y_attn);
    rel(h->bufs.m_in); rel(h->bufs.shared_mid); rel(h->bufs.shared_out);
    rel(h->bufs.mlp_gate); rel(h->bufs.mlp_up);
    rel(h->bufs.moe_top_idx); rel(h->bufs.moe_top_score); rel(h->bufs.moe_hidden);
    rel(h->bufs.moe_x_f32); rel(h->bufs.moe_gate_f32); rel(h->bufs.moe_up_f32);
    rel(h->bufs.moe_mid_f32); rel(h->bufs.moe_down_f32);
    rel(h->bufs.argmax_val_buf); rel(h->bufs.argmax_idx_buf);

    delete h;
}
