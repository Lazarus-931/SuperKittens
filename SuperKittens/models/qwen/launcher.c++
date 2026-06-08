//  launcher.c++ — Qwen3-32B (dense) inference launcher.

#include "launcher.h"
#include "qwen_model.h"
#include "../../kernels/runtime_bindings.h"
#include "../../inference/silicon/mmap_buffer.h"

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

    uint32_t layers_run     = 0;
    int32_t  capture_layer  = -1;
    uint32_t last_seq       = 0;  // seq used at most recent forward (for get_capture sizing)

    // Zero-copy mmap of the GGUF file, when used (otherwise nullptr).
    // Owns the MTL::Buffer that w_lm_head (and future mmap-backed weights)
    // alias into via per-weight byte offsets.
    sk::silicon::MmapBuffer* gguf_mmap = nullptr;

    // Per-layer zero-copy mmap ranges for bulk Q8_0 weights (qkv/o/gate/up/down).
    // Each layer's slab is a separate file-range mmap; this vector owns the
    // MmapBuffer objects so they can be destroyed alongside the handle.
    std::vector<sk::silicon::MmapBuffer*> tensor_mmaps;
};

static MTL::Buffer* alloc_zero(MTL::Device* dev, size_t bytes) {
    auto* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (b) std::memset(b->contents(), 0, bytes);
    return b;
}

static bool resolve_psos(ModelPSOs& P) {
    P.layer.rmsnorm        = sk::bindings_pso("rmsnorm");
    // Optional T=1 fast path (nullable). 256-thread single-row variant.
    P.layer.rmsnorm_t1     = sk::bindings_pso("rmsnorm_t1");
    P.layer.gemm           = sk::bindings_pso("gemm_fp16");
    P.layer.gemv_m1        = sk::bindings_pso("gemv_fp16_m1");
    P.layer.gemv_swiglu_m1 = sk::bindings_pso("gemv_swiglu_fp16_m1");
    P.layer.gemv_t_m1      = sk::bindings_pso("gemv_t_fp16_m1");
    P.layer.gemv_t_2dtile_m1 = sk::bindings_pso("gemv_t_fp16_2dtile_m1");  // optional; nullptr OK
    P.layer.q8_0_matvec    = sk::bindings_pso("q8_0_matvec");  // optional; nullptr OK
    P.layer.q4k_matvec     = sk::bindings_pso("q4k_matvec");   // optional; nullptr OK
    P.layer.q6k_matvec     = sk::bindings_pso("q6k_matvec");   // optional; nullptr OK
    // Batched (seq>1) MMA GEMM for prefill. All optional; absence falls back to
    // the per-row matvec loop in encode_quant_gemm. SK_NO_GEMM_MMA=1 forces the
    // matvec-loop prefill (A/B bench of the batched path against the per-row one).
    if (getenv("SK_NO_GEMM_MMA")) {
        P.layer.gemm_mma_f16 = P.layer.gemm_mma_q8_0 = P.layer.gemm_mma_q4k = nullptr;
    } else {
        P.layer.gemm_mma_f16   = sk::bindings_pso("gemm_mma_f16");
        P.layer.gemm_mma_q8_0  = sk::bindings_pso("gemm_mma_q8_0");
        P.layer.gemm_mma_q4k   = sk::bindings_pso("gemm_mma_q4k");
    }
    P.layer.q8_0_swiglu_m1 = sk::bindings_pso("q8_0_swiglu_m1");  // optional; nullptr OK
    P.layer.q8_0_swiglu_prenorm_m1 = sk::bindings_pso("q8_0_swiglu_prenorm_m1");  // optional
    P.layer.q8_0_matvec_addres = sk::bindings_pso("q8_0_matvec_addres");  // optional
    P.layer.split_packed   = sk::bindings_pso("split_packed");
    P.layer.rope_qk        = sk::bindings_pso("qwen_rope_qk");
    P.layer.attn           = sk::bindings_pso("mha_causal");
    // SK_NO_SPLIT_ATTN=1 forces the mha_causal path everywhere (A/B + bisection).
    if (getenv("SK_NO_SPLIT_ATTN")) {
        P.layer.attn_split = nullptr;
        P.layer.attn_combine = nullptr;
    } else {
        P.layer.attn_split   = sk::bindings_pso("mha_decode_split");    // optional; nullptr OK
        P.layer.attn_combine = sk::bindings_pso("mha_decode_combine");  // optional; nullptr OK
    }
    P.layer.kv_cache_write = sk::bindings_pso("kv_cache_write");
    P.layer.add            = sk::bindings_pso("add_f16");
    P.layer.add_rmsnorm    = sk::bindings_pso("add_rmsnorm");
    P.layer.gated_mlp      = sk::bindings_pso("gated_mlp");
    P.layer.silu_mul       = sk::bindings_pso("silu_mul_f16");
    P.layer.t_seq_to_head  = sk::bindings_pso("transpose_seq_to_head_f16");
    P.layer.t_head_to_seq  = sk::bindings_pso("transpose_head_to_seq_f16");
    P.embedding_lookup     = sk::bindings_pso("embedding_lookup");
    P.argmax               = sk::bindings_pso("argmax");
    // Optional 2-pass argmax (nullable; T==1 fast path).
    P.argmax_partial       = sk::bindings_pso("argmax_partial");
    P.argmax_reduce        = sk::bindings_pso("argmax_reduce");
    // ICB-compatible copies (separate cache; setSupportIndirectCommandBuffers=true).
    // Nullable: if the ICB recorder fails to allocate (e.g. on a platform that
    // doesn't support compute ICBs), dispatch_model falls back to the
    // non-ICB 2-pass path.
    P.argmax_partial_icb   = sk::bindings_pso_icb("argmax_partial");
    P.argmax_reduce_icb    = sk::bindings_pso_icb("argmax_reduce");

    #define _CK(name, val) if (!(val)) { std::fprintf(stderr, "qwen launcher: missing PSO " name "\n"); return false; }
    _CK("rmsnorm",          P.layer.rmsnorm);
    _CK("gemm_fp16",        P.layer.gemm);
    _CK("gemv_fp16_m1",     P.layer.gemv_m1);
    _CK("split_packed",     P.layer.split_packed);
    _CK("rope_qk",          P.layer.rope_qk);
    _CK("mha_causal",       P.layer.attn);
    _CK("kv_cache_write",   P.layer.kv_cache_write);
    _CK("add_f16",          P.layer.add);
    _CK("add_rmsnorm",      P.layer.add_rmsnorm);
    _CK("gated_mlp",        P.layer.gated_mlp);
    _CK("silu_mul_f16",     P.layer.silu_mul);
    _CK("t_seq_to_head",    P.layer.t_seq_to_head);
    _CK("t_head_to_seq",    P.layer.t_head_to_seq);
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
    h->weights.w_q_norm        = alloc_zero(dev, (size_t)cfg->n_layers * hd * 2);
    h->weights.w_k_norm        = alloc_zero(dev, (size_t)cfg->n_layers * hd * 2);
    h->weights.w_pre_mlp_norm  = alloc_zero(dev, (size_t)cfg->n_layers * cfg->d_model * 2);
    h->weights.w_final_norm    = alloc_zero(dev, (size_t)cfg->d_model * 2);

    // Bulk Q8_0 weights: by default we allocate one big shared buffer per type
    // (matching the legacy single-buffer + strided-offset layout) so that
    // sk_qwen_load_weights() and sk_qwen_load_from_store() can keep their
    // existing one-memcpy-per-tensor logic. sk_qwen_load_gguf() replaces this
    // with per-layer mmap ranges. The per-layer vectors fan out to the SAME
    // shared buffer with strided offsets in this default path.
    auto fan_out = [&](size_t per_layer_bytes,
                       std::vector<MTL::Buffer*>& vbuf,
                       std::vector<size_t>& voff) {
        auto* big = alloc_zero(dev, (size_t)cfg->n_layers * per_layer_bytes);
        vbuf.assign(cfg->n_layers, big);
        voff.resize(cfg->n_layers);
        for (uint32_t L = 0; L < cfg->n_layers; ++L) voff[L] = (size_t)L * per_layer_bytes;
    };
    fan_out((size_t)cfg->d_model * qkv_N * 2,
            h->weights.w_qkv, h->weights.w_qkv_off);
    fan_out((size_t)cfg->n_heads * hd * cfg->d_model * 2,
            h->weights.w_o, h->weights.w_o_off);
    fan_out((size_t)cfg->d_model * cfg->n_int * 2,
            h->weights.w_gate, h->weights.w_gate_off);
    fan_out((size_t)cfg->d_model * cfg->n_int * 2,
            h->weights.w_up, h->weights.w_up_off);
    fan_out((size_t)cfg->n_int * cfg->d_model * 2,
            h->weights.w_down, h->weights.w_down_off);
    h->weights.w_lm_head       = cfg->tie_word_embeddings ? nullptr
                                  : alloc_zero(dev, (size_t)cfg->vocab_size * cfg->d_model * 2);

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
    h->bufs.cos_tbl    = alloc_zero(dev, (size_t)cfg->cache_max * (hd / 2) * 2);
    h->bufs.sin_tbl    = alloc_zero(dev, (size_t)cfg->cache_max * (hd / 2) * 2);

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
    h->bufs.capture    = alloc_zero(dev, (size_t)T_max * cfg->d_model * 2);
    h->bufs.gate_buf   = alloc_zero(dev, (size_t)T_max * cfg->n_int * 2);
    h->bufs.up_buf     = alloc_zero(dev, (size_t)T_max * cfg->n_int * 2);
    h->bufs.q_th       = alloc_zero(dev, (size_t)T_max * cfg->n_heads    * hd * 2);
    h->bufs.k_th       = alloc_zero(dev, (size_t)T_max * cfg->n_kv_heads * hd * 2);
    h->bufs.v_th       = alloc_zero(dev, (size_t)T_max * cfg->n_kv_heads * hd * 2);
    h->bufs.attn_out_seq = alloc_zero(dev, (size_t)T_max * cfg->n_heads  * hd * 2);

    // Flash-decoding split-K partials (decode only). SPLITS must match the
    // compile-time cap in attn.metal (mha_decode_split<…,SPLITS>).
    {
        constexpr uint32_t SPLITS = 8u;
        const size_t n_part = (size_t)cfg->batch * cfg->n_heads * SPLITS;
        h->bufs.attn_pm = alloc_zero(dev, n_part * sizeof(float));
        h->bufs.attn_ps = alloc_zero(dev, n_part * sizeof(float));
        h->bufs.attn_po = alloc_zero(dev, n_part * hd * sizeof(float));
    }

    // 2-pass argmax scratch (one entry per 16384-elt tile of vocab_size).
    {
        constexpr uint32_t ELTS_PER_TG = 16384u;
        const uint32_t n_blocks = (cfg->vocab_size + ELTS_PER_TG - 1u) / ELTS_PER_TG;
        h->bufs.argmax_val_buf = alloc_zero(dev, (size_t)n_blocks * sizeof(float));
        h->bufs.argmax_idx_buf = alloc_zero(dev, (size_t)n_blocks * sizeof(int32_t));

        // ICB-tail wiring. Three preconditions: both ICB PSOs resolved, the
        // IcbRecorder allocates, and the args buffer allocates. If any fails,
        // dispatch_model silently falls back to the non-ICB 2-pass path.
        if (h->psos.argmax_partial_icb && h->psos.argmax_reduce_icb) {
            h->bufs.argmax_args = alloc_zero(dev, 2 * sizeof(uint32_t));
            uint32_t* args = (uint32_t*)h->bufs.argmax_args->contents();
            args[0] = cfg->vocab_size;  // read by argmax_partial @ buffer(3)
            args[1] = n_blocks;         // read by argmax_reduce  @ buffer(3)

            auto* rec = sk::silicon::IcbRecorder::create(dev,
                /*max_slots=*/2u, /*max_buffer_bindings=*/4u);
            if (rec && h->bufs.argmax_args) {
                // Slot 0: argmax_partial(logits, val_buf, idx_buf, args[vocab_size])
                {
                    const MTL::Buffer* bufs[4] = {
                        h->bufs.logits, h->bufs.argmax_val_buf,
                        h->bufs.argmax_idx_buf, h->bufs.argmax_args };
                    NS::UInteger offs[4] = { 0, 0, 0, /*vocab_size at byte 0*/ 0 };
                    MTL::Size grid(n_blocks, 1, 1);
                    MTL::Size tg(1024, 1, 1);
                    rec->record(0, h->psos.argmax_partial_icb,
                                bufs, offs, 4, grid, tg, /*barrier_before=*/false);
                }
                // Slot 1: argmax_reduce(val_buf, idx_buf, output_id, args[n_blocks])
                {
                    const MTL::Buffer* bufs[4] = {
                        h->bufs.argmax_val_buf, h->bufs.argmax_idx_buf,
                        h->bufs.output_id, h->bufs.argmax_args };
                    // n_blocks lives at byte 4 of argmax_args.
                    NS::UInteger offs[4] = { 0, 0, 0, sizeof(uint32_t) };
                    MTL::Size grid(1, 1, 1);
                    MTL::Size tg(1024, 1, 1);
                    rec->record(1, h->psos.argmax_reduce_icb,
                                bufs, offs, 4, grid, tg, /*barrier_before=*/true);
                }
                rec->mark_resource(h->bufs.logits);
                rec->mark_resource(h->bufs.argmax_val_buf);
                rec->mark_resource(h->bufs.argmax_idx_buf);
                rec->mark_resource(h->bufs.output_id);
                rec->mark_resource(h->bufs.argmax_args);
                h->bufs.argmax_icb = rec;
            } else if (rec) {
                delete rec;
            }
        }
    }

    return reinterpret_cast<sk_qwen_handle*>(h);
}

extern "C" int sk_qwen_set_layers_run(sk_qwen_handle* hp, uint32_t n) {
    if (!hp) return -1;
    reinterpret_cast<meow::qwen::Handle*>(hp)->layers_run = n;
    return 0;
}
extern "C" int sk_qwen_set_capture_layer(sk_qwen_handle* hp, int32_t layer) {
    if (!hp) return -1;
    reinterpret_cast<meow::qwen::Handle*>(hp)->capture_layer = layer;
    return 0;
}
extern "C" int sk_qwen_get_capture(sk_qwen_handle* hp, void* out_fp16) {
    if (!hp || !out_fp16) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    // Copy only the captured slice (last_seq * d_model fp16), not the whole T_max buffer.
    const size_t bytes = (size_t)h->last_seq * h->cfg.d_model * 2;
    if (bytes == 0) return -2;
    std::memcpy(out_fp16, h->bufs.capture->contents(), bytes);
    return 0;
}

extern "C" int sk_qwen_load_weights(sk_qwen_handle* hp, const sk_qwen_weights* w) {
    if (!hp || !w) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    auto cp = [](MTL::Buffer* dst, const void* src) {
        if (dst && src) std::memcpy(dst->contents(), src, dst->length());
    };
    // Per-weight fan-out vectors share a single underlying buffer in the
    // non-GGUF loader path, so copying into [0] populates all layers.
    cp(h->weights.w_embed,         w->w_embed);
    cp(h->weights.w_pre_attn_norm, w->w_pre_attn_norm);
    cp(h->weights.w_qkv[0],        w->w_qkv);
    cp(h->weights.w_q_norm,        w->w_q_norm);
    cp(h->weights.w_k_norm,        w->w_k_norm);
    cp(h->weights.w_o[0],          w->w_o);
    cp(h->weights.w_pre_mlp_norm,  w->w_pre_mlp_norm);
    cp(h->weights.w_final_norm,    w->w_final_norm);
    cp(h->weights.w_gate[0],       w->w_gate);
    cp(h->weights.w_up[0],         w->w_up);
    cp(h->weights.w_down[0],       w->w_down);
    if (h->weights.w_lm_head && w->w_lm_head) cp(h->weights.w_lm_head, w->w_lm_head);
    return 0;
}

extern "C" void sk_qwen_reset(sk_qwen_handle* hp) {
    if (!hp) return;
    reinterpret_cast<meow::qwen::Handle*>(hp)->current_pos = 0;
}

namespace meow { namespace qwen {
// WHY: one-step driver shared between sk_qwen_forward and the in-C decode
// loop. Caller has already memcpy'd input_ids + rope_pos for `seq` tokens.
static int run_step(Handle* h, MTL::CommandQueue* q, uint32_t seq) {
    ModelParams mp;
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
    mp.layers_run      = h->layers_run;
    mp.capture_layer   = h->capture_layer;
    h->last_seq        = seq;

    auto* cmd = q->commandBuffer();
    dispatch_model(cmd, h->psos, h->weights, h->bufs, mp);
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();
    h->current_pos += seq;
    return 0;
}
}}  // namespace meow::qwen

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

    int rc = meow::qwen::run_step(h, q, seq);
    if (rc) return rc;

    std::memcpy(output_id, h->bufs.output_id->contents(),
                (size_t)h->cfg.batch * sizeof(int32_t));
    return 0;
}

extern "C" int sk_qwen_generate_n(sk_qwen_handle* hp,
                                  const int* prompt_ids, uint32_t prompt_seq,
                                  int* out_tokens, uint32_t n_tokens,
                                  int32_t eos_id) {
    if (!hp || !prompt_ids || !out_tokens) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    if (prompt_seq == 0 || prompt_seq > h->cfg.seq_max) return -2;
    if (n_tokens == 0) return 0;
    if (h->current_pos + prompt_seq > h->cfg.cache_max) return -4;
    if (h->cfg.batch != 1) return -5;  // greedy multi-token loop is batch=1

    auto* q = sk::bindings_queue();
    if (!q) return -3;

    // Prefill on the prompt: produces argmax for next token in output_id.
    std::memcpy(h->bufs.input_ids->contents(), prompt_ids,
                (size_t)prompt_seq * sizeof(int32_t));
    {
        int32_t* pos = (int32_t*)h->bufs.rope_pos->contents();
        for (uint32_t i = 0; i < prompt_seq; ++i)
            pos[i] = (int32_t)(h->current_pos + i);
    }
    if (int rc = meow::qwen::run_step(h, q, prompt_seq)) return rc;

    int32_t* in_ids  = (int32_t*)h->bufs.input_ids->contents();
    int32_t* rope    = (int32_t*)h->bufs.rope_pos->contents();
    const int32_t* out_id_buf = (const int32_t*)h->bufs.output_id->contents();

    int32_t first = out_id_buf[0];
    out_tokens[0] = first;
    if (eos_id >= 0 && first == eos_id) return 1;
    uint32_t written = 1;
    int32_t last = first;

    while (written < n_tokens) {
        if (h->current_pos + 1 > h->cfg.cache_max) break;
        in_ids[0] = last;
        rope[0]   = (int32_t)h->current_pos;
        if (int rc = meow::qwen::run_step(h, q, 1)) return rc;
        last = out_id_buf[0];
        out_tokens[written++] = last;
        if (eos_id >= 0 && last == eos_id) break;
    }
    return (int)written;
}

extern "C" int sk_qwen_set_rope_tables(sk_qwen_handle* hp, const void* cos, const void* sin) {
    if (!hp || !cos || !sin) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    const size_t n = (size_t)h->cfg.cache_max * (h->cfg.head_dim / 2) * 2;
    std::memcpy(h->bufs.cos_tbl->contents(), cos, n);
    std::memcpy(h->bufs.sin_tbl->contents(), sin, n);
    return 0;
}

extern "C" int sk_qwen_get_last_logits(sk_qwen_handle* hp, void* out_fp16) {
    if (!hp || !out_fp16) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    const size_t V = h->cfg.vocab_size;
    const size_t row_bytes = V * sizeof(uint16_t);
    if (h->current_pos == 0) return -2;
    const size_t last_row = (size_t)(h->current_pos - 1);
    const char* src = (const char*)h->bufs.logits->contents() + last_row * row_bytes;
    std::memcpy(out_fp16, src, row_bytes);
    return 0;
}

extern "C" void sk_qwen_destroy(sk_qwen_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    auto rel = [](MTL::Buffer* b) { if (b) b->release(); };

    rel(h->weights.w_embed); rel(h->weights.w_pre_attn_norm);
    rel(h->weights.w_q_norm); rel(h->weights.w_k_norm);
    rel(h->weights.w_pre_mlp_norm); rel(h->weights.w_final_norm);
    // Bulk Q8_0 vectors may contain duplicates (default fan-out path: all
    // entries point at the same big buffer) or unique entries (GGUF mmap
    // path: each entry is its own MmapBuffer-owned MTL::Buffer; those are
    // released via h->tensor_mmaps cleanup, NOT here). Walk each vector and
    // release each *unique* MTL::Buffer at most once, skipping any pointer
    // that an MmapBuffer owns.
    {
        std::vector<MTL::Buffer*> mmap_owned;
        mmap_owned.reserve(h->tensor_mmaps.size());
        for (auto* mb : h->tensor_mmaps) if (mb) mmap_owned.push_back(mb->buffer());
        auto is_mmap = [&](MTL::Buffer* b) {
            for (auto* o : mmap_owned) if (o == b) return true;
            return false;
        };
        auto release_vec_unique = [&](const std::vector<MTL::Buffer*>& v) {
            std::vector<MTL::Buffer*> seen;
            for (auto* b : v) {
                if (!b || is_mmap(b)) continue;
                bool dup = false;
                for (auto* s : seen) if (s == b) { dup = true; break; }
                if (!dup) { seen.push_back(b); b->release(); }
            }
        };
        release_vec_unique(h->weights.w_qkv);
        release_vec_unique(h->weights.w_o);
        release_vec_unique(h->weights.w_gate);
        release_vec_unique(h->weights.w_up);
        release_vec_unique(h->weights.w_down);
    }
    for (auto* mb : h->tensor_mmaps) delete mb;
    h->tensor_mmaps.clear();
    // w_lm_head may be a borrowed pointer into h->gguf_mmap when zero-copy is
    // active. Only release when it is NOT the mmap-owned buffer.
    if (h->weights.w_lm_head &&
        (!h->gguf_mmap || h->weights.w_lm_head != h->gguf_mmap->buffer())) {
        h->weights.w_lm_head->release();
    }
    if (h->gguf_mmap) { delete h->gguf_mmap; h->gguf_mmap = nullptr; }
    for (auto* b : h->k_caches) rel(b);
    for (auto* b : h->v_caches) rel(b);
    rel(h->bufs.input_ids); rel(h->bufs.output_id);
    rel(h->bufs.x_a); rel(h->bufs.x_b); rel(h->bufs.logits); rel(h->bufs.rope_pos);
    rel(h->bufs.cos_tbl); rel(h->bufs.sin_tbl);
    rel(h->bufs.x_norm); rel(h->bufs.qkv_packed); rel(h->bufs.q);
    rel(h->bufs.kv_pack); rel(h->bufs.k_tmp); rel(h->bufs.v_tmp);
    rel(h->bufs.attn_out); rel(h->bufs.o_proj); rel(h->bufs.y_attn);
    rel(h->bufs.m_in); rel(h->bufs.mlp_out); rel(h->bufs.capture);
    rel(h->bufs.gate_buf); rel(h->bufs.up_buf);
    rel(h->bufs.q_th); rel(h->bufs.k_th); rel(h->bufs.v_th); rel(h->bufs.attn_out_seq);
    rel(h->bufs.attn_pm); rel(h->bufs.attn_ps); rel(h->bufs.attn_po);
    rel(h->bufs.argmax_val_buf); rel(h->bufs.argmax_idx_buf);
    rel(h->bufs.argmax_args);
    if (h->bufs.argmax_icb) { delete h->bufs.argmax_icb; h->bufs.argmax_icb = nullptr; }
    delete h;
}
