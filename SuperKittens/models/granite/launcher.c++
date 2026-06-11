//  launcher.c++ — Granite-4.x hybrid inference launcher.

#include "launcher.h"
#include "granite_model.h"
#include "../../kernels/runtime_bindings.h"
#include "../load/gguf/gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace meow { namespace granite {

struct Handle {
    sk_granite_config cfg;
    uint32_t current_pos = 0;
    uint32_t last_seq    = 0;
    bool     loaded      = false;

    PSOs    psos;
    Weights weights;
    std::vector<LayerState> states;
    Buffers bufs;
};

static MTL::Buffer* alloc_zero(MTL::Device* dev, size_t bytes) {
    auto* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (b) std::memset(b->contents(), 0, bytes);
    return b;
}

static bool resolve_psos(PSOs& P) {
    P.rmsnorm            = sk::bindings_pso("rmsnorm");
    P.rmsnorm_t1         = sk::bindings_pso("rmsnorm_t1");          // optional
    P.q8_0_matvec        = sk::bindings_pso("q8_0_matvec");
    P.split_packed       = sk::bindings_pso("split_packed");
    P.conv1d_silu        = sk::bindings_pso("conv1d_silu");
    P.conv1d_silu_step   = sk::bindings_pso("conv1d_silu_step");
    P.conv_state_capture = sk::bindings_pso("conv_state_capture");
    P.mamba2_ssd         = sk::bindings_pso("mamba2_ssd");
    if (!P.mamba2_ssd) P.mamba2_ssd = sk::bindings_pso("mamba2_ssd_ref");
    P.gate_norm          = sk::bindings_pso("gate_norm");
    P.silu_mul           = sk::bindings_pso("silu_mul_f16");
    P.attn               = sk::bindings_pso("mha_causal");
    P.kv_cache_write     = sk::bindings_pso("kv_cache_write");
    P.t_seq_to_head      = sk::bindings_pso("transpose_seq_to_head_f16");
    P.t_head_to_seq      = sk::bindings_pso("transpose_head_to_seq_f16");
    P.scale              = sk::bindings_pso("granite_scale_f16");
    P.add_scaled         = sk::bindings_pso("granite_add_scaled_f16");
    P.embedding_lookup   = sk::bindings_pso("embedding_lookup");
    P.argmax             = sk::bindings_pso("argmax");

    #define _CK(name, val) if (!(val)) { std::fprintf(stderr, "granite launcher: missing PSO " name "\n"); return false; }
    _CK("rmsnorm",            P.rmsnorm);
    _CK("q8_0_matvec",        P.q8_0_matvec);
    _CK("split_packed",       P.split_packed);
    _CK("conv1d_silu",        P.conv1d_silu);
    _CK("conv1d_silu_step",   P.conv1d_silu_step);
    _CK("conv_state_capture", P.conv_state_capture);
    _CK("mamba2_ssd(_ref)",   P.mamba2_ssd);
    _CK("gate_norm",          P.gate_norm);
    _CK("silu_mul_f16",       P.silu_mul);
    _CK("mha_causal",         P.attn);
    _CK("kv_cache_write",     P.kv_cache_write);
    _CK("transpose_seq_to_head_f16", P.t_seq_to_head);
    _CK("transpose_head_to_seq_f16", P.t_head_to_seq);
    _CK("granite_scale_f16",      P.scale);
    _CK("granite_add_scaled_f16", P.add_scaled);
    _CK("embedding_lookup",   P.embedding_lookup);
    _CK("argmax",             P.argmax);
    #undef _CK
    return true;
}

// ── dtype conversion (subnormal-correct fp32->fp16; K-quant scales are tiny) ──

static uint16_t fp32_bits_to_fp16(uint32_t f) {
    uint32_t sign = (f >> 16) & 0x8000u;
    uint32_t mant = f & 0x007fffffu;
    int32_t  exp  = (int32_t)((f >> 23) & 0xffu) - 127 + 15;
    if (((f >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000u) >> (uint32_t)(1 - exp);
        if (mant & 0x00001000u) mant += 0x00002000u;
        return (uint16_t)(sign | (mant >> 13));
    }
    if (mant & 0x00001000u) {
        mant += 0x00002000u;
        if (mant & 0x00800000u) { mant = 0; exp += 1; }
        if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static uint16_t f32_to_fp16(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    return fp32_bits_to_fp16(fb);
}

static float fp16_bits_to_f32(uint16_t s) {
    uint32_t sign = (s & 0x8000u) << 16;
    uint32_t exp  = (s >> 10) & 0x1fu;
    uint32_t mant = s & 0x3ffu;
    uint32_t v;
    if (exp == 0) {
        if (mant == 0) v = sign;
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
            mant &= 0x3ffu;
            v = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        v = sign | 0x7f800000u | (mant << 13);
    } else {
        v = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out; std::memcpy(&out, &v, 4);
    return out;
}

static void dequant_q8_0_to_fp16(uint16_t* dst, const uint8_t* src, size_t n_elems) {
    const size_t n_blocks = n_elems / 32;
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* p = src + b * 34;
        uint16_t scale_h; std::memcpy(&scale_h, p, 2);
        const float scale = fp16_bits_to_f32(scale_h);
        const int8_t* qs = (const int8_t*)(p + 2);
        for (int i = 0; i < 32; ++i)
            dst[b * 32 + i] = f32_to_fp16((float)qs[i] * scale);
    }
}

// ── GGUF load ─────────────────────────────────────────────────────────────────

static bool copy_f32_as_fp16(MTL::Buffer* dst, size_t dst_off,
                             sk::WeightStore& store, const std::string& name,
                             size_t n_elems) {
    auto* v = store.get(name);
    if (!v) { std::fprintf(stderr, "granite gguf: missing %s\n", name.c_str()); return false; }
    if (v->dtype != sk::Dtype::F32 || v->nbytes != n_elems * 4) {
        std::fprintf(stderr, "granite gguf: %s dtype/size unexpected (dtype=%d nbytes=%zu want F32 x%zu)\n",
                     name.c_str(), (int)v->dtype, v->nbytes, n_elems);
        return false;
    }
    const float* s = (const float*)v->data;
    uint16_t* d = (uint16_t*)((char*)dst->contents() + dst_off);
    for (size_t i = 0; i < n_elems; ++i) d[i] = f32_to_fp16(s[i]);
    return true;
}

static MTL::Buffer* copy_q8_0(MTL::Device* dev, sk::WeightStore& store,
                              const std::string& name, size_t n_elems) {
    auto* v = store.get(name);
    if (!v) { std::fprintf(stderr, "granite gguf: missing %s\n", name.c_str()); return nullptr; }
    const size_t expect = (n_elems / 32) * 34;
    if (v->dtype != sk::Dtype::Q8_0 || v->nbytes != expect) {
        std::fprintf(stderr, "granite gguf: %s dtype/size unexpected (dtype=%d nbytes=%zu want Q8_0 %zu)\n",
                     name.c_str(), (int)v->dtype, v->nbytes, expect);
        return nullptr;
    }
    auto* b = dev->newBuffer(expect, MTL::ResourceStorageModeShared);
    if (!b) return nullptr;
    std::memcpy(b->contents(), v->data, expect);
    return b;
}

}}  // namespace meow::granite

extern "C" sk_granite_handle* sk_granite_create(const sk_granite_config* cfg) {
    if (!cfg) return nullptr;
    auto* dev = sk::bindings_device();
    if (!dev) return nullptr;

    auto* h = new meow::granite::Handle();
    h->cfg = *cfg;
    if (!meow::granite::resolve_psos(h->psos)) { delete h; return nullptr; }

    using namespace meow::granite;
    const uint32_t T_max = cfg->seq_max;
    const uint32_t D     = cfg->d_model;
    const uint32_t E     = cfg->d_inner;
    const uint32_t C_in  = E + 2 * cfg->ssm_n_groups * cfg->ssm_state;
    const uint32_t IN_OUT = 2 * E + 2 * cfg->ssm_n_groups * cfg->ssm_state + cfg->ssm_n_heads;
    const uint32_t qN  = cfg->n_heads * cfg->head_dim;
    const uint32_t kvN = cfg->n_kv_heads * cfg->head_dim;

    h->weights.embed      = alloc_zero(dev, (size_t)cfg->vocab_size * D * 2);
    h->weights.attn_norm  = alloc_zero(dev, (size_t)cfg->n_layers * D * 2);
    h->weights.ffn_norm   = alloc_zero(dev, (size_t)cfg->n_layers * D * 2);
    h->weights.final_norm = alloc_zero(dev, (size_t)D * 2);

    auto& b = h->bufs;
    b.input_ids   = alloc_zero(dev, (size_t)T_max * sizeof(int32_t));
    b.output_id   = alloc_zero(dev, sizeof(int32_t));
    b.x_a         = alloc_zero(dev, (size_t)T_max * D * 2);
    b.x_b         = alloc_zero(dev, (size_t)T_max * D * 2);
    b.x_norm      = alloc_zero(dev, (size_t)T_max * D * 2);
    b.ffn_inp     = alloc_zero(dev, (size_t)T_max * D * 2);
    b.mixer_out   = alloc_zero(dev, (size_t)T_max * D * 2);
    b.logits      = alloc_zero(dev, (size_t)cfg->vocab_size * 2);
    b.in_proj_out = alloc_zero(dev, (size_t)T_max * IN_OUT * 2);
    b.z           = alloc_zero(dev, (size_t)T_max * E * 2);
    b.xBC         = alloc_zero(dev, (size_t)T_max * C_in * 2);
    b.dt_raw      = alloc_zero(dev, (size_t)T_max * cfg->ssm_n_heads * 2);
    b.xBC_post    = alloc_zero(dev, (size_t)T_max * (C_in + cfg->ssm_n_heads) * 2);
    b.ssd_out     = alloc_zero(dev, (size_t)T_max * E * 2);
    b.gated       = alloc_zero(dev, (size_t)T_max * E * 2);
    b.q           = alloc_zero(dev, (size_t)T_max * qN * 2);
    b.k_tmp       = alloc_zero(dev, (size_t)T_max * kvN * 2);
    b.v_tmp       = alloc_zero(dev, (size_t)T_max * kvN * 2);
    b.attn_out    = alloc_zero(dev, (size_t)T_max * qN * 2);
    b.q_th        = alloc_zero(dev, (size_t)T_max * qN * 2);
    b.k_th        = alloc_zero(dev, (size_t)T_max * kvN * 2);
    b.v_th        = alloc_zero(dev, (size_t)T_max * kvN * 2);
    b.attn_out_seq = alloc_zero(dev, (size_t)T_max * qN * 2);
    b.gate_buf    = alloc_zero(dev, (size_t)T_max * cfg->n_int * 2);
    b.up_buf      = alloc_zero(dev, (size_t)T_max * cfg->n_int * 2);
    b.mlp_out     = alloc_zero(dev, (size_t)T_max * D * 2);

    return reinterpret_cast<sk_granite_handle*>(h);
}

extern "C" int sk_granite_load_gguf(sk_granite_handle* hp, const char* path) {
    if (!hp || !path) return -1;
    auto* h = reinterpret_cast<meow::granite::Handle*>(hp);
    auto* dev = sk::bindings_device();
    if (!dev) return -2;
    const auto& c = h->cfg;

    sk::WeightStore store;
    sk::gguf::Model gm;
    int rc = sk::gguf::load_gguf(path, store, &gm);
    if (rc != 0) {
        std::fprintf(stderr, "granite gguf: parse failed rc=%d\n", rc);
        return rc;
    }

    std::string arch;
    sk::gguf::meta_string(gm, "general.architecture", &arch);
    if (arch != "granitehybrid") {
        std::fprintf(stderr, "granite gguf: arch '%s' != granitehybrid\n", arch.c_str());
        return -10;
    }

    // Layer types: head_count_kv is a per-layer U32 array; 0 = mamba (recurrent).
    const auto* kvh = sk::gguf::meta_find(gm, "granitehybrid.attention.head_count_kv");
    if (!kvh || kvh->type != sk::gguf::V_ARRAY || kvh->arr_type != sk::gguf::V_U32
        || kvh->arr_len != c.n_layers) {
        std::fprintf(stderr, "granite gguf: head_count_kv array missing/mismatched\n");
        return -11;
    }
    std::vector<uint32_t> kv_heads(c.n_layers);
    std::memcpy(kv_heads.data(), gm.map_base + kvh->raw_pos, (size_t)c.n_layers * 4);

    // Dim cross-check against metadata (fail loud on mismatch, not garbage).
    auto check_u32 = [&](const char* key, uint32_t want) -> bool {
        uint32_t v = 0;
        if (!sk::gguf::meta_u32(gm, key, &v)) {
            std::fprintf(stderr, "granite gguf: missing meta %s\n", key);
            return false;
        }
        if (v != want) {
            std::fprintf(stderr, "granite gguf: meta %s=%u != cfg %u\n", key, v, want);
            return false;
        }
        return true;
    };
    if (!check_u32("granitehybrid.block_count",            c.n_layers))   return -12;
    if (!check_u32("granitehybrid.embedding_length",       c.d_model))    return -12;
    if (!check_u32("granitehybrid.feed_forward_length",    c.n_int))      return -12;
    if (!check_u32("granitehybrid.attention.head_count",   c.n_heads))    return -12;
    if (!check_u32("granitehybrid.vocab_size",             c.vocab_size)) return -12;
    if (!check_u32("granitehybrid.ssm.inner_size",         c.d_inner))    return -12;
    if (!check_u32("granitehybrid.ssm.state_size",         c.ssm_state))  return -12;
    if (!check_u32("granitehybrid.ssm.time_step_rank",     c.ssm_n_heads))return -12;
    if (!check_u32("granitehybrid.ssm.group_count",        c.ssm_n_groups))return -12;
    if (!check_u32("granitehybrid.ssm.conv_kernel",        c.ssm_conv))   return -12;

    float m_attn = 0, m_embd = 0, m_res = 0, m_logit = 0;
    sk::gguf::meta_f32(gm, "granitehybrid.attention.scale", &m_attn);
    sk::gguf::meta_f32(gm, "granitehybrid.embedding_scale", &m_embd);
    sk::gguf::meta_f32(gm, "granitehybrid.residual_scale",  &m_res);
    sk::gguf::meta_f32(gm, "granitehybrid.logit_scale",     &m_logit);

    // Gate A config table.
    std::fprintf(stderr, "granite config: layers=%u d_model=%u ffn=%u vocab=%u\n",
                 c.n_layers, c.d_model, c.n_int, c.vocab_size);
    std::fprintf(stderr, "granite attn: heads=%u kv_heads=%u head_dim=%u NoPE scale=%g\n",
                 c.n_heads, c.n_kv_heads, c.head_dim, m_attn);
    std::fprintf(stderr, "granite ssm: E=%u H=%u P=%u N=%u G=%u K=%u\n",
                 c.d_inner, c.ssm_n_heads, c.ssm_head_dim, c.ssm_state,
                 c.ssm_n_groups, c.ssm_conv);
    std::fprintf(stderr, "granite scales: embed=%g residual=%g attn=%g logit=%g\n",
                 m_embd, m_res, m_attn, m_logit);
    std::fprintf(stderr, "granite layer types: ");
    for (uint32_t L = 0; L < c.n_layers; ++L)
        std::fprintf(stderr, "%c", kv_heads[L] > 0 ? 'A' : 'm');
    std::fprintf(stderr, "  (A=attention, m=mamba2)\n");

    using namespace meow::granite;
    const uint32_t D    = c.d_model;
    const uint32_t E    = c.d_inner;
    const uint32_t H    = c.ssm_n_heads;
    const uint32_t G    = c.ssm_n_groups;
    const uint32_t N    = c.ssm_state;
    const uint32_t K    = c.ssm_conv;
    const uint32_t C_in = E + 2 * G * N;
    const uint32_t IN_OUT = 2 * E + 2 * G * N + H;
    const uint32_t qN   = c.n_heads * c.head_dim;
    const uint32_t kvN  = c.n_kv_heads * c.head_dim;

    // Embedding: fp16 dequant for the lookup; raw Q8_0 copy for the tied head.
    {
        auto* v = store.get("token_embd.weight");
        if (!v) { std::fprintf(stderr, "granite gguf: missing token_embd.weight\n"); return -20; }
        const size_t n_elems = (size_t)c.vocab_size * D;
        if (v->dtype != sk::Dtype::Q8_0 || v->nbytes != (n_elems / 32) * 34) {
            std::fprintf(stderr, "granite gguf: token_embd not Q8_0 (dtype=%d)\n", (int)v->dtype);
            return -20;
        }
        dequant_q8_0_to_fp16((uint16_t*)h->weights.embed->contents(),
                             (const uint8_t*)v->data, n_elems);
        h->weights.head_q8 = copy_q8_0(dev, store, "token_embd.weight", n_elems);
        if (!h->weights.head_q8) return -21;
    }
    if (!copy_f32_as_fp16(h->weights.final_norm, 0, store, "output_norm.weight", D))
        return -22;

    h->weights.layers.assign(c.n_layers, LayerWeights{});
    h->states.assign(c.n_layers, LayerState{});

    char nm[128];
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        LayerWeights& lw = h->weights.layers[L];
        LayerState&   ls = h->states[L];
        lw.is_attn = kv_heads[L] > 0;
        if (lw.is_attn && kv_heads[L] != c.n_kv_heads) {
            std::fprintf(stderr, "granite gguf: layer %u kv_heads=%u != cfg %u\n",
                         L, kv_heads[L], c.n_kv_heads);
            return -23;
        }

        const size_t off_norm = (size_t)L * D * 2;
        std::snprintf(nm, sizeof(nm), "blk.%u.attn_norm.weight", L);
        if (!copy_f32_as_fp16(h->weights.attn_norm, off_norm, store, nm, D)) return -30;
        std::snprintf(nm, sizeof(nm), "blk.%u.ffn_norm.weight", L);
        if (!copy_f32_as_fp16(h->weights.ffn_norm, off_norm, store, nm, D)) return -31;

        if (lw.is_attn) {
            std::snprintf(nm, sizeof(nm), "blk.%u.attn_q.weight", L);
            lw.wq = copy_q8_0(dev, store, nm, (size_t)qN * D);
            std::snprintf(nm, sizeof(nm), "blk.%u.attn_k.weight", L);
            lw.wk = copy_q8_0(dev, store, nm, (size_t)kvN * D);
            std::snprintf(nm, sizeof(nm), "blk.%u.attn_v.weight", L);
            lw.wv = copy_q8_0(dev, store, nm, (size_t)kvN * D);
            std::snprintf(nm, sizeof(nm), "blk.%u.attn_output.weight", L);
            lw.wo = copy_q8_0(dev, store, nm, (size_t)D * qN);
            if (!lw.wq || !lw.wk || !lw.wv || !lw.wo) return -40;

            ls.k_cache = alloc_zero(dev, (size_t)c.cache_max * kvN * 2);
            ls.v_cache = alloc_zero(dev, (size_t)c.cache_max * kvN * 2);
        } else {
            std::snprintf(nm, sizeof(nm), "blk.%u.ssm_in.weight", L);
            lw.ssm_in = copy_q8_0(dev, store, nm, (size_t)IN_OUT * D);
            std::snprintf(nm, sizeof(nm), "blk.%u.ssm_out.weight", L);
            lw.ssm_out = copy_q8_0(dev, store, nm, (size_t)D * E);
            if (!lw.ssm_in || !lw.ssm_out) return -41;

            // conv1d weight: GGUF layout [C_in rows][K] matches conv1d_silu's
            // w[c*K+k] indexing — straight F32->fp16, no transpose.
            lw.conv_w = alloc_zero(dev, (size_t)C_in * K * 2);
            std::snprintf(nm, sizeof(nm), "blk.%u.ssm_conv1d.weight", L);
            if (!copy_f32_as_fp16(lw.conv_w, 0, store, nm, (size_t)C_in * K)) return -42;
            lw.conv_b = alloc_zero(dev, (size_t)C_in * 2);
            std::snprintf(nm, sizeof(nm), "blk.%u.ssm_conv1d.bias", L);
            if (!copy_f32_as_fp16(lw.conv_b, 0, store, nm, C_in)) return -43;

            lw.dt_bias = alloc_zero(dev, (size_t)H * 2);
            std::snprintf(nm, sizeof(nm), "blk.%u.ssm_dt.bias", L);
            if (!copy_f32_as_fp16(lw.dt_bias, 0, store, nm, H)) return -44;

            // GGUF stores A = -exp(A_log) (llama.cpp conversion); SK's SSD
            // kernels compute dA = exp(dt * -exp(A_log)), so invert here.
            {
                std::snprintf(nm, sizeof(nm), "blk.%u.ssm_a", L);
                auto* v = store.get(nm);
                if (!v || v->dtype != sk::Dtype::F32 || v->nbytes != (size_t)H * 4) {
                    std::fprintf(stderr, "granite gguf: %s missing/unexpected\n", nm);
                    return -45;
                }
                lw.A_log = alloc_zero(dev, (size_t)H * 2);
                const float* a = (const float*)v->data;
                uint16_t* d = (uint16_t*)lw.A_log->contents();
                for (uint32_t i = 0; i < H; ++i) {
                    if (!(a[i] < 0.0f)) {
                        std::fprintf(stderr, "granite gguf: %s[%u]=%g not negative\n", nm, i, a[i]);
                        return -45;
                    }
                    d[i] = f32_to_fp16(std::log(-a[i]));
                }
            }

            lw.ssm_D = alloc_zero(dev, (size_t)H * 2);
            std::snprintf(nm, sizeof(nm), "blk.%u.ssm_d", L);
            if (!copy_f32_as_fp16(lw.ssm_D, 0, store, nm, H)) return -46;
            lw.ssm_norm = alloc_zero(dev, (size_t)E * 2);
            std::snprintf(nm, sizeof(nm), "blk.%u.ssm_norm.weight", L);
            if (!copy_f32_as_fp16(lw.ssm_norm, 0, store, nm, E)) return -47;

            ls.conv_state = alloc_zero(dev, (size_t)(K - 1) * C_in * 2);
            ls.ssm_state  = alloc_zero(dev, (size_t)H * c.ssm_head_dim * N * sizeof(float));
        }

        std::snprintf(nm, sizeof(nm), "blk.%u.ffn_gate.weight", L);
        lw.gate = copy_q8_0(dev, store, nm, (size_t)c.n_int * D);
        std::snprintf(nm, sizeof(nm), "blk.%u.ffn_up.weight", L);
        lw.up = copy_q8_0(dev, store, nm, (size_t)c.n_int * D);
        std::snprintf(nm, sizeof(nm), "blk.%u.ffn_down.weight", L);
        lw.down = copy_q8_0(dev, store, nm, (size_t)D * c.n_int);
        if (!lw.gate || !lw.up || !lw.down) return -48;
    }

    h->loaded = true;
    return 0;
}

namespace meow { namespace granite {

static int run_step(Handle* h, MTL::CommandQueue* q, uint32_t seq) {
    Params p;
    p.seq         = seq;
    p.n_layers    = h->cfg.n_layers;
    p.d_model     = h->cfg.d_model;
    p.n_heads     = h->cfg.n_heads;
    p.n_kv_heads  = h->cfg.n_kv_heads;
    p.head_dim    = h->cfg.head_dim;
    p.n_int       = h->cfg.n_int;
    p.d_inner     = h->cfg.d_inner;
    p.ssm_heads   = h->cfg.ssm_n_heads;
    p.ssm_pdim    = h->cfg.ssm_head_dim;
    p.ssm_state   = h->cfg.ssm_state;
    p.ssm_groups  = h->cfg.ssm_n_groups;
    p.ssm_conv    = h->cfg.ssm_conv;
    p.vocab_size  = h->cfg.vocab_size;
    p.cache_max   = h->cfg.cache_max;
    p.current_pos = h->current_pos;
    p.eps         = h->cfg.eps;
    p.embedding_scale = h->cfg.embedding_scale;
    p.residual_scale  = h->cfg.residual_scale;
    p.attention_scale = h->cfg.attention_scale;
    p.logit_scale     = h->cfg.logit_scale;
    h->last_seq = seq;

    auto* cmd = q->commandBuffer();
    dispatch_model(cmd, h->psos, h->weights, h->states, h->bufs, p);
    cmd->commit();
    cmd->waitUntilCompleted();
    if (getenv("SK_GRANITE_GPUPROF"))
        std::fprintf(stderr, "[gpuprof] gpu_busy_us=%.1f\n",
                     (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1e6);
    cmd->release();
    h->current_pos += seq;
    return 0;
}

}}  // namespace meow::granite

extern "C" int sk_granite_forward(sk_granite_handle* hp,
                                  const int* input_ids, uint32_t seq, int* output_id) {
    if (!hp || !input_ids || !output_id) return -1;
    auto* h = reinterpret_cast<meow::granite::Handle*>(hp);
    if (!h->loaded) return -7;
    if (seq == 0 || seq > h->cfg.seq_max) return -2;
    // Chunked mamba prefill is unsupported (each chunk would re-zero-pad the
    // conv left edge); a prompt must fit one prefill forward.
    if (h->current_pos > 0 && seq > 1) return -6;
    if (h->current_pos + seq > h->cfg.cache_max) return -4;
    auto* q = sk::bindings_queue();
    if (!q) return -3;

    std::memcpy(h->bufs.input_ids->contents(), input_ids, (size_t)seq * sizeof(int32_t));
    if (int rc = meow::granite::run_step(h, q, seq)) return rc;
    std::memcpy(output_id, h->bufs.output_id->contents(), sizeof(int32_t));
    return 0;
}

extern "C" int sk_granite_generate_n(sk_granite_handle* hp,
                                     const int* prompt_ids, uint32_t prompt_seq,
                                     int* out_tokens, uint32_t n_tokens, int32_t eos_id) {
    if (!hp || !prompt_ids || !out_tokens) return -1;
    auto* h = reinterpret_cast<meow::granite::Handle*>(hp);
    if (!h->loaded) return -7;
    if (prompt_seq == 0 || prompt_seq > h->cfg.seq_max) return -2;
    if (n_tokens == 0) return 0;
    if (h->current_pos != 0) return -6;  // mamba state demands a fresh sequence
    auto* q = sk::bindings_queue();
    if (!q) return -3;

    std::memcpy(h->bufs.input_ids->contents(), prompt_ids,
                (size_t)prompt_seq * sizeof(int32_t));
    if (int rc = meow::granite::run_step(h, q, prompt_seq)) return rc;

    int32_t* in_ids = (int32_t*)h->bufs.input_ids->contents();
    const int32_t* out_id = (const int32_t*)h->bufs.output_id->contents();

    int32_t last = out_id[0];
    out_tokens[0] = last;
    if (eos_id >= 0 && last == eos_id) return 1;
    uint32_t written = 1;
    while (written < n_tokens) {
        if (h->current_pos + 1 > h->cfg.cache_max) break;
        in_ids[0] = last;
        if (int rc = meow::granite::run_step(h, q, 1)) return rc;
        last = out_id[0];
        out_tokens[written++] = last;
        if (eos_id >= 0 && last == eos_id) break;
    }
    return (int)written;
}

extern "C" int sk_granite_get_last_logits(sk_granite_handle* hp, void* out_fp16) {
    if (!hp || !out_fp16) return -1;
    auto* h = reinterpret_cast<meow::granite::Handle*>(hp);
    if (h->last_seq == 0) return -2;
    std::memcpy(out_fp16, h->bufs.logits->contents(),
                (size_t)h->cfg.vocab_size * 2);
    return 0;
}

extern "C" uint32_t sk_granite_get_pos(sk_granite_handle* hp) {
    if (!hp) return 0;
    return reinterpret_cast<meow::granite::Handle*>(hp)->current_pos;
}

extern "C" void sk_granite_reset(sk_granite_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::granite::Handle*>(hp);
    h->current_pos = 0;
    for (auto& s : h->states) {
        if (s.conv_state) std::memset(s.conv_state->contents(), 0, s.conv_state->length());
        if (s.ssm_state)  std::memset(s.ssm_state->contents(),  0, s.ssm_state->length());
    }
}

extern "C" void sk_granite_destroy(sk_granite_handle* hp) {
    if (!hp) return;
    auto* h = reinterpret_cast<meow::granite::Handle*>(hp);
    auto rel = [](MTL::Buffer* b) { if (b) b->release(); };
    rel(h->weights.embed); rel(h->weights.head_q8);
    rel(h->weights.attn_norm); rel(h->weights.ffn_norm); rel(h->weights.final_norm);
    for (auto& lw : h->weights.layers) {
        rel(lw.ssm_in); rel(lw.ssm_out); rel(lw.conv_w); rel(lw.conv_b);
        rel(lw.dt_bias); rel(lw.A_log); rel(lw.ssm_D); rel(lw.ssm_norm);
        rel(lw.wq); rel(lw.wk); rel(lw.wv); rel(lw.wo);
        rel(lw.gate); rel(lw.up); rel(lw.down);
    }
    for (auto& s : h->states) {
        rel(s.k_cache); rel(s.v_cache); rel(s.conv_state); rel(s.ssm_state);
    }
    auto& b = h->bufs;
    rel(b.input_ids); rel(b.output_id); rel(b.x_a); rel(b.x_b); rel(b.x_norm);
    rel(b.ffn_inp); rel(b.mixer_out); rel(b.logits); rel(b.in_proj_out);
    rel(b.z); rel(b.xBC); rel(b.dt_raw); rel(b.xBC_post); rel(b.ssd_out);
    rel(b.gated); rel(b.q); rel(b.k_tmp); rel(b.v_tmp); rel(b.attn_out);
    rel(b.q_th); rel(b.k_th); rel(b.v_th); rel(b.attn_out_seq);
    rel(b.gate_buf); rel(b.up_buf); rel(b.mlp_out);
    delete h;
}
