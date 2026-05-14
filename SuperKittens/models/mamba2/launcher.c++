// launcher.c++ — Mamba 2 (mamba2-130m-hf) inference launcher.
//
// Allocates fused weight + state buffers and resolves PSOs. forward() currently
// returns -ENOSYS until mamba2_ssd / mamba2_step kernels are rewritten to match
// the HF Mamba2 SSD signature (see STATUS.md and mamba2_model.h).

#include "launcher.h"
#include "mamba2_model.h"
#include "../../kernels/runtime_bindings.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace meow { namespace mamba2 {

struct Handle {
    sk_mamba2_config cfg;
    uint32_t       current_pos = 0;
    ModelPSOs    psos;
    ModelWeights weights;
    ModelBuffers bufs;
    std::vector<LayerState> layer_states;
};

static MTL::Buffer* alloc_zero(MTL::Device* dev, size_t bytes) {
    auto* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (b) std::memset(b->contents(), 0, bytes);
    return b;
}

static bool resolve_psos(ModelPSOs& P) {
    P.layer.rmsnorm      = sk::bindings_pso("rmsnorm");
    P.layer.rmsnorm_t1   = sk::bindings_pso("rmsnorm_t1");  // optional T=1 fast path
    P.layer.gemm         = sk::bindings_pso("gemm_fp16");
    P.layer.split_packed = sk::bindings_pso("split_packed");
    P.layer.conv1d_silu  = sk::bindings_pso("conv1d_silu");
    // Prefer the reference (HF-signature-correct) kernels; fall back to legacy.
    P.layer.mamba2_ssd   = sk::bindings_pso("mamba2_ssd_ref");
    if (!P.layer.mamba2_ssd) P.layer.mamba2_ssd = sk::bindings_pso("mamba2_ssd");
    P.layer.mamba2_step  = sk::bindings_pso("mamba2_step_ref");
    if (!P.layer.mamba2_step) P.layer.mamba2_step = sk::bindings_pso("mamba2_step");
    P.layer.gate_norm    = sk::bindings_pso("gate_norm");
    P.layer.add          = sk::bindings_pso("add_f16");
    P.embedding_lookup   = sk::bindings_pso("embedding_lookup");
    P.argmax             = sk::bindings_pso("argmax");
    P.argmax_partial     = sk::bindings_pso("argmax_partial");
    P.argmax_reduce      = sk::bindings_pso("argmax_reduce");

    #define _CK(name, val) if (!(val)) { std::fprintf(stderr, "mamba2 launcher: missing PSO %s\n", name); return false; }
    _CK("rmsnorm",          P.layer.rmsnorm);
    _CK("gemm_fp16",        P.layer.gemm);
    _CK("conv1d_silu",      P.layer.conv1d_silu);
    _CK("gate_norm",        P.layer.gate_norm);
    _CK("embedding_lookup", P.embedding_lookup);
    _CK("argmax",           P.argmax);
    // ssd/step/split_packed/add are nice-to-have; warn only.
    if (!P.layer.mamba2_ssd)  std::fprintf(stderr, "mamba2 launcher: WARN no mamba2_ssd PSO\n");
    if (!P.layer.mamba2_step) std::fprintf(stderr, "mamba2 launcher: WARN no mamba2_step PSO\n");
    #undef _CK
    return true;
}

}}  // namespace meow::mamba2

extern "C" sk_mamba2_handle* sk_mamba2_create(const sk_mamba2_config* cfg) {
    if (!cfg) return nullptr;
    auto* dev = sk::bindings_device();
    if (!dev) return nullptr;

    auto* h = new meow::mamba2::Handle();
    h->cfg = *cfg;
    if (!meow::mamba2::resolve_psos(h->psos)) { delete h; return nullptr; }

    using namespace meow::mamba2;
    const size_t fp16 = 2;
    const size_t D = cfg->d_model;
    const size_t E = cfg->intermediate;
    const size_t H = cfg->n_heads;
    const size_t P = cfg->head_dim;
    const size_t G = cfg->n_groups;
    const size_t N = cfg->state_size;
    const size_t K = cfg->conv_kernel;
    const size_t IN_OUT = 2 * E + 2 * G * N + H;
    const size_t C_in   = E + 2 * G * N;
    const size_t T_max  = (size_t)cfg->batch * cfg->seq_max;

    // Weights
    h->weights.w_embed       = alloc_zero(dev, (size_t)cfg->vocab_size * D * fp16);
    h->weights.w_final_norm  = alloc_zero(dev, D * fp16);
    h->weights.w_pre_norm    = alloc_zero(dev, (size_t)cfg->n_layers * D * fp16);
    h->weights.w_in_proj     = alloc_zero(dev, (size_t)cfg->n_layers * D * IN_OUT * fp16);
    h->weights.w_conv        = alloc_zero(dev, (size_t)cfg->n_layers * K * C_in * fp16);
    h->weights.w_conv_b      = alloc_zero(dev, (size_t)cfg->n_layers * C_in * fp16);
    h->weights.w_dt_bias     = alloc_zero(dev, (size_t)cfg->n_layers * H * fp16);
    h->weights.w_A_log       = alloc_zero(dev, (size_t)cfg->n_layers * H * fp16);
    h->weights.w_D           = alloc_zero(dev, (size_t)cfg->n_layers * H * fp16);
    h->weights.w_norm        = alloc_zero(dev, (size_t)cfg->n_layers * E * fp16);
    h->weights.w_out_proj    = alloc_zero(dev, (size_t)cfg->n_layers * E * D * fp16);

    // Per-layer recurrent state for decode.
    h->layer_states.resize(cfg->n_layers);
    for (uint32_t L = 0; L < cfg->n_layers; ++L) {
        h->layer_states[L].conv_state = alloc_zero(dev,
            (size_t)cfg->batch * (K - 1) * C_in * fp16);
        // SSM state is fp32 (kernel signature: device float*).
        h->layer_states[L].ssm_state = alloc_zero(dev,
            (size_t)cfg->batch * H * P * N * sizeof(float));
    }
    h->bufs.layer_states = h->layer_states.data();

    // Scratch
    h->bufs.tok_ids       = alloc_zero(dev, T_max * sizeof(int32_t));
    h->bufs.x             = alloc_zero(dev, T_max * D * fp16);
    h->bufs.x_norm        = alloc_zero(dev, T_max * D * fp16);
    h->bufs.in_proj_out   = alloc_zero(dev, T_max * IN_OUT * fp16);
    h->bufs.z             = alloc_zero(dev, T_max * E * fp16);
    h->bufs.xBC           = alloc_zero(dev, T_max * C_in * fp16);
    h->bufs.dt_raw        = alloc_zero(dev, T_max * H * fp16);
    h->bufs.xBC_post      = alloc_zero(dev, T_max * C_in * fp16);
    h->bufs.ssd_out       = alloc_zero(dev, T_max * E * fp16);
    h->bufs.gated         = alloc_zero(dev, T_max * E * fp16);
    h->bufs.out_proj_out  = alloc_zero(dev, T_max * D * fp16);
    h->bufs.logits        = alloc_zero(dev, (size_t)cfg->vocab_size * fp16);

    // 2-pass argmax scratch.
    {
        constexpr uint32_t ELTS_PER_TG = 16384u;
        const uint32_t n_blocks = (cfg->vocab_size + ELTS_PER_TG - 1u) / ELTS_PER_TG;
        h->bufs.argmax_val_buf = alloc_zero(dev, (size_t)n_blocks * sizeof(float));
        h->bufs.argmax_idx_buf = alloc_zero(dev, (size_t)n_blocks * sizeof(int32_t));
    }

    h->current_pos = 0;
    return reinterpret_cast<sk_mamba2_handle*>(h);
}

extern "C" void sk_mamba2_reset(sk_mamba2_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::mamba2::Handle*>(hp);
    h->current_pos = 0;
    for (auto& s : h->layer_states) {
        if (s.conv_state) std::memset(s.conv_state->contents(), 0, s.conv_state->length());
        if (s.ssm_state)  std::memset(s.ssm_state->contents(),  0, s.ssm_state->length());
    }
}

extern "C" int sk_mamba2_forward(sk_mamba2_handle* hp,
                                 const int* input_ids, uint32_t seq, int* output_id) {
    if (!hp || !input_ids || !output_id || seq == 0) return -1;
    auto* h = reinterpret_cast<meow::mamba2::Handle*>(hp);
    if (seq > h->cfg.seq_max) return -2;

    auto* q = sk::bindings_queue();
    if (!q) return -3;

    // Copy token ids into device buffer.
    std::memcpy(h->bufs.tok_ids->contents(), input_ids,
                (size_t)h->cfg.batch * seq * sizeof(int32_t));

    meow::mamba2::ModelParams mp;
    mp.batch        = h->cfg.batch;
    mp.seq          = seq;
    mp.n_layers     = h->cfg.n_layers;
    mp.d_model      = h->cfg.d_model;
    mp.intermediate = h->cfg.intermediate;
    mp.n_heads      = h->cfg.n_heads;
    mp.head_dim     = h->cfg.head_dim;
    mp.state_size   = h->cfg.state_size;
    mp.n_groups     = h->cfg.n_groups;
    mp.conv_kernel  = h->cfg.conv_kernel;
    mp.chunk_size   = h->cfg.chunk_size;
    mp.vocab_size   = h->cfg.vocab_size;
    mp.eps          = h->cfg.rms_eps;
    mp.dt_min       = h->cfg.time_step_min;
    mp.dt_max       = h->cfg.time_step_max;

    // Use a single shared device buffer for output_id (1 int32).
    static MTL::Buffer* s_out_id = nullptr;
    if (!s_out_id) {
        auto* dev = sk::bindings_device();
        s_out_id = dev->newBuffer(sizeof(int32_t), MTL::ResourceStorageModeShared);
    }
    std::memset(s_out_id->contents(), 0, sizeof(int32_t));

    auto* cmd = q->commandBuffer();
    meow::mamba2::dispatch_model(cmd, h->psos, h->weights, h->bufs,
                                 h->layer_states.data(), mp, s_out_id);
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();

    h->current_pos += seq;
    int32_t v;
    std::memcpy(&v, s_out_id->contents(), sizeof(int32_t));
    *output_id = (int)v;
    return 0;
}

extern "C" int sk_mamba2_get_last_logits(sk_mamba2_handle* hp, void* out_fp16) {
    if (!hp || !out_fp16) return -1;
    auto* h = reinterpret_cast<meow::mamba2::Handle*>(hp);
    const size_t V = h->cfg.vocab_size;
    // The last forward dispatched argmax on row (T-1). We need to know T.
    // We don't track T across calls explicitly; in single-prompt usage T is
    // the most recent seq. Conservatively the caller should know — they pass
    // out_fp16 sized V*2 and we copy from offset (current_pos - 1) * V * 2.
    if (h->current_pos == 0) return -2;
    const size_t row = (size_t)(h->current_pos - 1);
    const char* src = (const char*)h->bufs.logits->contents() + row * V * 2;
    std::memcpy(out_fp16, src, V * 2);
    return 0;
}

// Dump a named intermediate tensor for HF parity testing. Tags supported
// (only the LAST forward()'s scratch buffers — for prefill of full prompt
// this is what HF dumps with use_cache=False):
//   "embed"          → (T, D) fp16  from B.x  (overwritten by layer 0)
//   "x"              → (T, D) fp16  current residual
//   "x_norm"         → (T, D) fp16  pre_norm of last layer, or final_norm
//   "in_proj_out"    → (T, IN_OUT) fp16
//   "z"              → (T, E) fp16
//   "xBC"            → (T, C_in) fp16
//   "dt_raw"         → (T, H) fp16
//   "xBC_post"       → (T, C_in) fp16  (after conv1d+silu)
//   "ssd_out"        → (T, E) fp16
//   "gated"          → (T, E) fp16  (norm_gated)
//   "out_proj_out"   → (T, D) fp16
//   "logits"         → (T, V) fp16
// State (cumulative, current values):
//   "ssm_state.L{i}" → (B, H, P, N) fp32
//   "conv_state.L{i}"→ (B, K-1, C_in) fp16
extern "C" int sk_mamba2_dump_layer(sk_mamba2_handle* hp, const char* tag,
                                    void* out, size_t out_bytes) {
    if (!hp || !tag || !out) return -1;
    auto* h = reinterpret_cast<meow::mamba2::Handle*>(hp);
    auto* b = &h->bufs;
    auto cp = [&](MTL::Buffer* src) {
        if (!src) return -10;
        size_t n = std::min(out_bytes, (size_t)src->length());
        std::memcpy(out, src->contents(), n);
        return 0;
    };
    std::string t(tag);
    if (t == "embed" || t == "x")        return cp(b->x);
    if (t == "x_norm")                   return cp(b->x_norm);
    if (t == "in_proj_out")              return cp(b->in_proj_out);
    if (t == "z")                        return cp(b->z);
    if (t == "xBC")                      return cp(b->xBC);
    if (t == "dt_raw")                   return cp(b->dt_raw);
    if (t == "xBC_post")                 return cp(b->xBC_post);
    if (t == "ssd_out")                  return cp(b->ssd_out);
    if (t == "gated")                    return cp(b->gated);
    if (t == "out_proj_out")             return cp(b->out_proj_out);
    if (t == "logits")                   return cp(b->logits);

    auto parse_layer = [&](const char* prefix) -> int {
        size_t plen = std::strlen(prefix);
        if (t.size() <= plen) return -1;
        if (std::strncmp(t.c_str(), prefix, plen) != 0) return -1;
        return std::atoi(t.c_str() + plen);
    };
    int L = parse_layer("ssm_state.L");
    if (L >= 0 && L < (int)h->layer_states.size())
        return cp(h->layer_states[L].ssm_state);
    L = parse_layer("conv_state.L");
    if (L >= 0 && L < (int)h->layer_states.size())
        return cp(h->layer_states[L].conv_state);

    std::fprintf(stderr, "sk_mamba2_dump_layer: unknown tag '%s'\n", tag);
    return -39;
}

extern "C" void sk_mamba2_destroy(sk_mamba2_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::mamba2::Handle*>(hp);
    auto rel = [](MTL::Buffer* b) { if (b) b->release(); };
    rel(h->weights.w_embed);
    rel(h->weights.w_final_norm);
    rel(h->weights.w_pre_norm);
    rel(h->weights.w_in_proj);
    rel(h->weights.w_conv);
    rel(h->weights.w_conv_b);
    rel(h->weights.w_dt_bias);
    rel(h->weights.w_A_log);
    rel(h->weights.w_D);
    rel(h->weights.w_norm);
    rel(h->weights.w_out_proj);
    for (auto& s : h->layer_states) { rel(s.conv_state); rel(s.ssm_state); }
    rel(h->bufs.tok_ids); rel(h->bufs.x); rel(h->bufs.x_norm);
    rel(h->bufs.in_proj_out); rel(h->bufs.z); rel(h->bufs.xBC);
    rel(h->bufs.dt_raw); rel(h->bufs.xBC_post); rel(h->bufs.ssd_out);
    rel(h->bufs.gated); rel(h->bufs.out_proj_out); rel(h->bufs.logits);
    rel(h->bufs.argmax_val_buf); rel(h->bufs.argmax_idx_buf);
    delete h;
}
