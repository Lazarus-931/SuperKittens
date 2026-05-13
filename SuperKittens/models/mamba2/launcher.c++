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
    P.layer.gemm         = sk::bindings_pso("gemm_fp16");
    P.layer.split_packed = sk::bindings_pso("split_packed");
    P.layer.conv1d_silu  = sk::bindings_pso("conv1d_silu");
    P.layer.mamba2_ssd   = sk::bindings_pso("mamba2_ssd");
    P.layer.mamba2_step  = sk::bindings_pso("mamba2_step");
    P.layer.gate_norm    = sk::bindings_pso("gate_norm");
    P.layer.add          = sk::bindings_pso("add_f16");
    P.embedding_lookup   = sk::bindings_pso("embedding_lookup");
    P.argmax             = sk::bindings_pso("argmax");

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
        h->layer_states[L].ssm_state = alloc_zero(dev,
            (size_t)cfg->batch * H * P * N * fp16);
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
    // TODO: dispatch embed -> per-layer{pre_norm, in_proj, split, conv1d_silu,
    //   ssd or step, gate_norm, out_proj, residual} -> final_norm -> argmax.
    // Blocked on mamba2_ssd/mamba2_step kernel rewrite (HF signature) — see STATUS.md.
    (void)input_ids; (void)seq; (void)output_id;
    std::fprintf(stderr, "sk_mamba2_forward: not yet implemented (SSD kernel pending)\n");
    return -38;  // ENOSYS-ish
}

extern "C" int sk_mamba2_dump_layer(sk_mamba2_handle* hp, const char* tag,
                                    void* out, size_t out_bytes) {
    (void)hp; (void)tag; (void)out; (void)out_bytes;
    return -38;
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
    delete h;
}
