#include "weights.h"
#include "qwen_model.h"
#include "../../inference/weight_store.h"
#include "../../inference/silicon/mmap_buffer.h"
#include "../../kernels/runtime_bindings.h"
#include "../load/gguf/gguf.h"
#include "../load/safetensor/safetensor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fcntl.h>
#include <unistd.h>

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
    uint32_t       layers_run;
    int32_t        capture_layer;
    uint32_t       last_seq;
    sk::silicon::MmapBuffer* gguf_mmap;
    std::vector<sk::silicon::MmapBuffer*> tensor_mmaps;
};
}}

namespace {

inline std::string layer_key(uint32_t L, const char* suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "model.layers.%u.%s", L, suffix);
    return std::string(buf);
}

inline uint16_t fp32_bits_to_fp16(uint32_t f) {
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

inline void bf16_to_fp16(uint16_t* dst, const uint16_t* src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t f = ((uint32_t)src[i]) << 16;
        dst[i] = fp32_bits_to_fp16(f);
    }
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
    char* out = (char*)dst->contents() + dst_off;
    if (v->dtype == sk::Dtype::BF16) {
        bf16_to_fp16((uint16_t*)out, (const uint16_t*)v->data, expect_bytes / 2);
    } else {
        std::memcpy(out, v->data, expect_bytes);
    }
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
    const bool is_bf16 = (v->dtype == sk::Dtype::BF16);
    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            uint16_t s = src[j * out_rows + i];
            if (is_bf16) {
                uint32_t f = ((uint32_t)s) << 16;
                d[i * out_cols + j] = fp32_bits_to_fp16(f);
            } else {
                d[i * out_cols + j] = s;
            }
        }
    }
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
    if (!c.tie_word_embeddings && h->weights.w_lm_head) {
        if (!copy_into(h->weights.w_lm_head, 0, store,
                       "lm_head.weight", (size_t)c.vocab_size * dm * fp16)) return -12;
    }

    // In the safetensors path all per-layer vector entries share one big
    // MTL::Buffer with strided offsets — walk it via the [0] entry.
    auto* qkv_base = (char*)h->weights.w_qkv[0]->contents();

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
        if (!copy_transpose_fp16(h->weights.w_o[0], o_off, store,
                       layer_key(L, "self_attn.o_proj.weight"), Nq, dm)) return -24;
        if (!copy_transpose_fp16(h->weights.w_gate[0], gate_off, store,
                       layer_key(L, "mlp.gate_proj.weight"), dm, ni)) return -25;
        if (!copy_transpose_fp16(h->weights.w_up[0], gate_off, store,
                       layer_key(L, "mlp.up_proj.weight"), dm, ni)) return -26;
        if (!copy_transpose_fp16(h->weights.w_down[0], down_off, store,
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
        auto tx_into = [&](const void* src_v, sk::Dtype dt, size_t out_rows, size_t out_cols, size_t col_off) {
            const uint16_t* s = (const uint16_t*)src_v;
            const bool is_bf16 = (dt == sk::Dtype::BF16);
            for (size_t i = 0; i < out_rows; ++i) {
                for (size_t j = 0; j < out_cols; ++j) {
                    uint16_t x = s[j * out_rows + i];
                    if (is_bf16) {
                        uint32_t f = ((uint32_t)x) << 16;
                        layer[i * qkvN + col_off + j] = fp32_bits_to_fp16(f);
                    } else {
                        layer[i * qkvN + col_off + j] = x;
                    }
                }
            }
        };
        tx_into(q_v->data, q_v->dtype, dm, Nq,  0);
        tx_into(k_v->data, k_v->dtype, dm, Nkv, Nq);
        tx_into(v_v->data, v_v->dtype, dm, Nkv, Nq + Nkv);
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

// ─────────────────────────────────────────────────────────────────────
// Debug: run q8_0_matvec on raw user-supplied buffers, for unit testing.
// ─────────────────────────────────────────────────────────────────────

extern "C" int sk_qwen_debug_q8_matvec(
    const void* x_fp16,   // [K] fp16
    const void* w_q8_0,   // [N, K] q8_0 row-major (N*(K/32)*34 bytes)
    void*       y_fp16,   // [N] fp16
    uint32_t    N, uint32_t K)
{
    auto* dev = sk::bindings_device();
    auto* qq  = sk::bindings_queue();
    auto* pso = sk::bindings_pso("q8_0_matvec");
    if (!dev || !qq || !pso) return -1;
    if (K % 32) return -2;
    auto* B = dev->newBuffer(K * 2, MTL::ResourceStorageModeShared);
    std::memcpy(B->contents(), x_fp16, K * 2);
    size_t wbytes = (size_t)N * (K/32) * 34;
    auto* A = dev->newBuffer(wbytes, MTL::ResourceStorageModeShared);
    std::memcpy(A->contents(), w_q8_0, wbytes);
    auto* C = dev->newBuffer(N * 2, MTL::ResourceStorageModeShared);
    std::memset(C->contents(), 0, N * 2);
    auto* cmd = qq->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(B, 0, 0);
    enc->setBuffer(A, 0, 1);
    enc->setBuffer(C, 0, 2);
    enc->setBytes(&K, 4, 3);
    enc->setBytes(&N, 4, 4);
    enc->dispatchThreadgroups(MTL::Size((N + 1) / 2, 1, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    std::memcpy(y_fp16, C->contents(), N * 2);
    A->release(); B->release(); C->release();
    return 0;
}

// ─────────────────────────────────────────────────────────────────────
// GGUF (Q8_0) loader for Qwen3
// ─────────────────────────────────────────────────────────────────────

namespace {

// Q8_0 dequantize: 32 elems / block, header is fp16 scale, then int8 qs[32].
void dequant_q8_0_to_fp16(uint16_t* dst, const uint8_t* src, size_t n_elems) {
    const size_t n_blocks = n_elems / 32;
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* p = src + b * 34;
        uint16_t scale_h;
        std::memcpy(&scale_h, p, 2);
        // half → float
        uint32_t s = scale_h;
        uint32_t sign = (s & 0x8000u) << 16;
        uint32_t exp  = (s >> 10) & 0x1fu;
        uint32_t mant = s & 0x3ffu;
        float scale;
        if (exp == 0) {
            if (mant == 0) { uint32_t v = sign; std::memcpy(&scale, &v, 4); }
            else {
                exp = 1;
                while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
                mant &= 0x3ffu;
                uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
                std::memcpy(&scale, &v, 4);
            }
        } else if (exp == 31) {
            uint32_t v = sign | 0x7f800000u | (mant << 13);
            std::memcpy(&scale, &v, 4);
        } else {
            uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
            std::memcpy(&scale, &v, 4);
        }
        const int8_t* qs = (const int8_t*)(p + 2);
        for (int i = 0; i < 32; ++i) {
            float f = (float)qs[i] * scale;
            // round-to-nearest fp32 → fp16 via fp32_bits_to_fp16
            uint32_t fb; std::memcpy(&fb, &f, 4);
            dst[b * 32 + i] = fp32_bits_to_fp16(fb);
        }
    }
}

// Decode an fp16 (bits) to float. Mirrors the inline path in dequant_q8_0.
inline float fp16_bits_to_f32(uint16_t s) {
    uint32_t sign = (s & 0x8000u) << 16;
    uint32_t exp  = (s >> 10) & 0x1fu;
    uint32_t mant = s & 0x3ffu;
    float out;
    if (exp == 0) {
        if (mant == 0) { uint32_t v = sign; std::memcpy(&out, &v, 4); }
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
            mant &= 0x3ffu;
            uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
            std::memcpy(&out, &v, 4);
        }
    } else if (exp == 31) {
        uint32_t v = sign | 0x7f800000u | (mant << 13);
        std::memcpy(&out, &v, 4);
    } else {
        uint32_t v = sign | ((exp + 112) << 23) | (mant << 13);
        std::memcpy(&out, &v, 4);
    }
    return out;
}

inline uint16_t f32_to_fp16_bits(float f) {
    uint32_t fb; std::memcpy(&fb, &f, 4);
    return fp32_bits_to_fp16(fb);
}

// Q4_K dequant (144 B / 256 elems). Layout: half d, half dmin, uint8 scales[12],
// uint8 qs[128]. The 8 sub-blocks of 32 use 6-bit (scale, min) pairs packed in
// scales[12] (matches llama.cpp get_scale_min_k4 / dequantize_row_q4_K).
void dequant_q4_k_to_fp16(uint16_t* dst, const uint8_t* src, size_t n_elems) {
    const size_t nb = n_elems / 256;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 144;
        uint16_t dh, dminh;
        std::memcpy(&dh, p, 2); std::memcpy(&dminh, p + 2, 2);
        const float d    = fp16_bits_to_f32(dh);
        const float dmin = fp16_bits_to_f32(dminh);
        const uint8_t* sc = p + 4;
        const uint8_t* qs = p + 4 + 12;
        uint16_t* out = dst + b * 256;
        for (int j = 0; j < 8; ++j) {
            uint8_t scl, mn;
            if (j < 4) {
                scl = sc[j] & 63;
                mn  = sc[j + 4] & 63;
            } else {
                scl = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
                mn  = (sc[j + 4] >>   4) | ((sc[j    ] >> 6) << 4);
            }
            const float d1 = d * (float)scl;
            const float m1 = dmin * (float)mn;
            // Each sub-block j covers 32 elems; low nibble for even j-half,
            // high nibble for odd. qs is 128 bytes = two 32-elem nibble-planes.
            const uint8_t* qbase = qs + (j / 2) * 32;
            const int shift = (j & 1) ? 4 : 0;
            for (int i = 0; i < 32; ++i) {
                const int q = (qbase[i] >> shift) & 0x0F;
                out[j * 32 + i] = f32_to_fp16_bits(d1 * (float)q - m1);
            }
        }
    }
}

// Q6_K dequant (210 B / 256 elems). Layout: uint8 ql[128], uint8 qh[64],
// int8 scales[16], half d. Matches llama.cpp dequantize_row_q6_K.
void dequant_q6_k_to_fp16(uint16_t* dst, const uint8_t* src, size_t n_elems) {
    const size_t nb = n_elems / 256;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * 210;
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t*  sc = (const int8_t*)(p + 128 + 64);
        uint16_t dh; std::memcpy(&dh, p + 128 + 64 + 16, 2);
        const float d = fp16_bits_to_f32(dh);
        uint16_t* out = dst + b * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[n + l +  0] = f32_to_fp16_bits(d * (float)sc[is + 0] * (float)q1);
                out[n + l + 32] = f32_to_fp16_bits(d * (float)sc[is + 2] * (float)q2);
                out[n + l + 64] = f32_to_fp16_bits(d * (float)sc[is + 4] * (float)q3);
                out[n + l + 96] = f32_to_fp16_bits(d * (float)sc[is + 6] * (float)q4);
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
}

// F32 → FP16.
void f32_to_fp16(uint16_t* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t fb; std::memcpy(&fb, &src[i], 4);
        dst[i] = fp32_bits_to_fp16(fb);
    }
}

// Read a norm/embed tensor as fp16 into `out` (n_elems fp16 entries).
// Accepts F32, F16, BF16, Q8_0.
bool read_to_fp16(uint16_t* out, const sk::TensorView* v, size_t n_elems) {
    if (!v) return false;
    if (v->dtype == sk::Dtype::F16) {
        std::memcpy(out, v->data, n_elems * 2);
        return true;
    }
    if (v->dtype == sk::Dtype::F32) {
        f32_to_fp16(out, (const float*)v->data, n_elems);
        return true;
    }
    if (v->dtype == sk::Dtype::BF16) {
        const uint16_t* s = (const uint16_t*)v->data;
        for (size_t i = 0; i < n_elems; ++i) {
            uint32_t f = ((uint32_t)s[i]) << 16;
            out[i] = fp32_bits_to_fp16(f);
        }
        return true;
    }
    if (v->dtype == sk::Dtype::Q8_0) {
        dequant_q8_0_to_fp16(out, (const uint8_t*)v->data, n_elems);
        return true;
    }
    if (v->dtype == sk::Dtype::Q4_K) {
        dequant_q4_k_to_fp16(out, (const uint8_t*)v->data, n_elems);
        return true;
    }
    if (v->dtype == sk::Dtype::Q6_K) {
        dequant_q6_k_to_fp16(out, (const uint8_t*)v->data, n_elems);
        return true;
    }
    return false;
}

}  // namespace

extern "C" int sk_qwen_load_gguf(sk_qwen_handle* hp, const char* path) {
    if (!hp || !path) return -1;
    auto* h = reinterpret_cast<meow::qwen::Handle*>(hp);
    const auto& c = h->cfg;

    auto* dev = sk::bindings_device();
    if (!dev) return -2;

    sk::WeightStore store;
    sk::gguf::Model gmodel;
    int rc = sk::gguf::load_gguf(path, store, &gmodel);
    if (rc != 0) {
        std::fprintf(stderr, "sk_qwen_load_gguf: parse failed rc=%d\n", rc);
        return rc;
    }

    // We use per-tensor range mmaps below for the bulk Q8_0 weights
    // (lm_head + per-layer qkv/o/gate/up/down). Open the GGUF file ONCE and
    // hand the fd to every mmap call — opening per tensor pushes 250+ fds for
    // a 36-layer 4B model and trips the default `ulimit -n 256` on lexie,
    // silently falling back to memcpy and tanking decode tok/s. The mmaps
    // survive `close(fd)` (Darwin/POSIX retains a vnode reference per mapping),
    // so the fd is closed at end of load.
    int gguf_fd = ::open(path, O_RDONLY);
    if (gguf_fd < 0) {
        std::fprintf(stderr, "sk_qwen_load_gguf: open('%s') failed\n", path);
        return -3;
    }

    const size_t dm   = c.d_model;
    const size_t hd   = c.head_dim;
    const size_t Nq   = (size_t)c.n_heads    * hd;
    const size_t Nkv  = (size_t)c.n_kv_heads * hd;
    const size_t qkvN = Nq + 2 * Nkv;
    const size_t ni   = c.n_int;

    // ── Norms + embedding (fp16 buffers, dequant from F32/Q8_0 as needed) ──
    {
        auto* v = store.get("token_embd.weight");
        if (!v) { std::fprintf(stderr, "gguf: missing token_embd.weight\n"); return -10; }
        if (!read_to_fp16((uint16_t*)h->weights.w_embed->contents(), v,
                          (size_t)c.vocab_size * dm)) return -11;
    }
    {
        auto* v = store.get("output_norm.weight");
        if (!v) { std::fprintf(stderr, "gguf: missing output_norm.weight\n"); return -12; }
        if (!read_to_fp16((uint16_t*)h->weights.w_final_norm->contents(), v, dm)) return -13;
    }
    {
        // LM head keeps Q8_0 layout (V × D row-major) to enable
        // the q8_0_matvec fast path. Tied models reuse token_embd; untied use
        // output.weight. Either way, allocate a Q8_0-sized w_lm_head buffer.
        const char* tname = c.tie_word_embeddings ? "token_embd.weight" : "output.weight";
        auto* v = store.get(tname);
        if (!v) { std::fprintf(stderr, "gguf: missing %s\n", tname); return -14; }
        if (v->dtype == sk::Dtype::Q8_0) {
            const size_t bytes = sk::dtype_bytes(sk::Dtype::Q8_0, (size_t)c.vocab_size * dm);
            if (v->nbytes != bytes) {
                std::fprintf(stderr, "gguf: %s size %zu != expected %zu\n",
                             tname, v->nbytes, bytes);
                return -15;
            }
            // Zero-copy: alias the LM-head tensor in the mmap'd GGUF directly
            // as the Metal buffer for w_lm_head. Saves the ~600 MB memcpy on
            // Qwen3-8B Q8_0. We use a per-tensor range mmap rather than a
            // whole-file mmap: on memory-constrained systems (16 GB lexie),
            // page-faulting an 8 GB+ region through MTL's resource registry
            // can trigger jetsam SIGKILL during load.
            //
            // Use the absolute byte offset GGUF records on disk — cannot
            // pointer-subtract v->data because WeightStore mmaps the file
            // independently of MmapBuffer.
            size_t tensor_off = (size_t)-1;
            for (const auto& ti : gmodel.tensors) {
                if (ti.name == tname) { tensor_off = (size_t)ti.abs_offset; break; }
            }
            if (tensor_off == (size_t)-1) {
                std::fprintf(stderr, "gguf: %s tensor missing from gmodel\n", tname);
                return -17;
            }
            // SK_NO_MMAP_LMHEAD=1 forces the legacy memcpy path for A/B testing.
            const bool disable_mmap = std::getenv("SK_NO_MMAP_LMHEAD") != nullptr;
            sk::silicon::MmapBuffer* lmh_map = nullptr;
            size_t inner_off = 0;
            if (!disable_mmap) {
                lmh_map = sk::silicon::MmapBuffer::from_fd_range(
                    dev, gguf_fd, tensor_off, bytes, &inner_off);
            }
            if (!lmh_map) {
                if (!disable_mmap) {
                    std::fprintf(stderr, "gguf: %s mmap range failed; falling back to memcpy\n", tname);
                }
                if (h->weights.w_lm_head) h->weights.w_lm_head->release();
                h->weights.w_lm_head = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
                if (!h->weights.w_lm_head) return -16;
                std::memcpy(h->weights.w_lm_head->contents(), v->data, bytes);
                h->weights.off_w_lm_head = 0;
            } else {
                if (h->weights.w_lm_head) h->weights.w_lm_head->release();
                if (h->gguf_mmap) { delete h->gguf_mmap; }
                h->gguf_mmap             = lmh_map;
                h->weights.w_lm_head     = lmh_map->buffer();
                h->weights.off_w_lm_head = inner_off;
            }
            h->weights.dt_lm_head = sk::Dtype::Q8_0;
        } else if (v->dtype == sk::Dtype::Q6_K) {
            // Keep the LM head in Q6_K and dispatch q6k_matvec at decode: the
            // weight is the single largest per-token read (vocab·d_model), so
            // host-dequantizing it to fp16 would stream ~2.3× the bytes
            // (fp16 vs Q6_K's 210 B / 256 weights) every step. Copy the raw
            // Q6_K bytes into a right-sized buffer; allocate one (the launcher
            // leaves w_lm_head null for tied models, and pre-allocates an
            // fp16-sized one for untied). q6k_matvec reads token_embd as a
            // [vocab, d_model] row-major matrix — the same orientation the
            // tied transB gemm head used — so the dedicated Q6_K head is
            // numerically identical to the prior fp16-tied head, just dequanted
            // in-kernel. w_embed stays fp16 for the per-token embedding gather.
            const size_t bytes = sk::dtype_bytes(sk::Dtype::Q6_K, (size_t)c.vocab_size * dm);
            if (v->nbytes != bytes) {
                std::fprintf(stderr, "gguf: %s Q6_K size %zu != expected %zu\n",
                             tname, v->nbytes, bytes);
                return -15;
            }
            if (h->weights.w_lm_head) h->weights.w_lm_head->release();
            h->weights.w_lm_head = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
            if (!h->weights.w_lm_head) return -16;
            std::memcpy(h->weights.w_lm_head->contents(), v->data, bytes);
            h->weights.off_w_lm_head = 0;
            h->weights.dt_lm_head = sk::Dtype::Q6_K;
        } else if (!c.tie_word_embeddings && h->weights.w_lm_head) {
            if (!read_to_fp16((uint16_t*)h->weights.w_lm_head->contents(), v,
                              (size_t)c.vocab_size * dm)) return -15;
        }
    }

    // ── Per-projection dtype detection. q/k/o/gate/up are uniform across layers
    // (Q4_K in Q4_K_M); attn_v + ffn_down alternate Q4_K/Q6_K PER LAYER (the
    // "_M" variant bumps every other layer to Q6_K), so those are read per L. ──
    auto supported_proj_dt = [](sk::Dtype d) {
        return d == sk::Dtype::Q8_0 || d == sk::Dtype::Q4_K || d == sk::Dtype::Q6_K
            || d == sk::Dtype::F16  || d == sk::Dtype::BF16;
    };
    auto proj_dtype = [&](uint32_t L, const char* suffix, sk::Dtype* out) -> bool {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "blk.%u.%s", L, suffix);
        auto* v = store.get(nm);
        if (!v) { std::fprintf(stderr, "gguf: missing %s\n", nm); return false; }
        if (!supported_proj_dt(v->dtype)) {
            std::fprintf(stderr, "gguf: unsupported dtype %d for %s\n", (int)v->dtype, nm);
            return false;
        }
        *out = v->dtype;
        return true;
    };
    sk::Dtype dt_q, dt_k, dt_o, dt_gate, dt_up;
    if (!proj_dtype(0, "attn_q.weight",      &dt_q)   ||
        !proj_dtype(0, "attn_k.weight",      &dt_k)   ||
        !proj_dtype(0, "attn_output.weight", &dt_o)   ||
        !proj_dtype(0, "ffn_gate.weight",    &dt_gate)||
        !proj_dtype(0, "ffn_up.weight",      &dt_up)) return -20;
    if (dt_q != dt_k) {
        std::fprintf(stderr, "gguf: q/k dtype mismatch (%d vs %d) — unsupported\n",
                     (int)dt_q, (int)dt_k);
        return -22;
    }
    const sk::Dtype dt_qk = dt_q;

    // Per-layer dtype for V and down. v_split = any layer needs V in its own
    // buffer (always true once a Q6_K V appears; split unconditionally so
    // mixed models so the dispatch reads V from w_v uniformly).
    std::vector<sk::Dtype> dt_v_layer(c.n_layers), dt_down_layer(c.n_layers);
    bool v_quant_mixed = false;
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        if (!proj_dtype(L, "attn_v.weight",  &dt_v_layer[L]))  return -23;
        if (!proj_dtype(L, "ffn_down.weight",&dt_down_layer[L]))return -24;
        if (dt_v_layer[L] != dt_qk) v_quant_mixed = true;
    }
    // Split V whenever any layer's V differs from Q/K, OR all V are Q6_K.
    const bool v_split = v_quant_mixed || (dt_v_layer[0] != dt_qk);

    const size_t o_layer_bytes    = sk::dtype_bytes(dt_o, Nq * dm);
    const size_t ffn_layer_bytes_gate = sk::dtype_bytes(dt_gate, dm * ni);
    const size_t ffn_layer_bytes_up   = sk::dtype_bytes(dt_up,   dm * ni);

    // Release the default fan-out buffers allocated in sk_qwen_create — we
    // are about to replace every per-layer entry with its own (mmap-backed
    // or memcpy-backed) MTL::Buffer. Each default-path vector currently
    // holds n_layers copies of the same pointer; dedupe before release.
    auto release_default_fanout = [](std::vector<MTL::Buffer*>& v) {
        if (v.empty()) return;
        MTL::Buffer* shared = v[0];
        bool all_same = true;
        for (auto* b : v) if (b != shared) { all_same = false; break; }
        if (all_same && shared) shared->release();
        else for (auto* b : v) if (b) b->release();
        v.clear();
    };
    release_default_fanout(h->weights.w_qkv);   h->weights.w_qkv_off.clear();
    release_default_fanout(h->weights.w_o);     h->weights.w_o_off.clear();
    release_default_fanout(h->weights.w_gate);  h->weights.w_gate_off.clear();
    release_default_fanout(h->weights.w_up);    h->weights.w_up_off.clear();
    release_default_fanout(h->weights.w_down);  h->weights.w_down_off.clear();
    release_default_fanout(h->weights.w_v);     h->weights.w_v_off.clear();

    h->weights.w_qkv .resize(c.n_layers, nullptr); h->weights.w_qkv_off .resize(c.n_layers, 0);
    h->weights.w_o   .resize(c.n_layers, nullptr); h->weights.w_o_off   .resize(c.n_layers, 0);
    h->weights.w_gate.resize(c.n_layers, nullptr); h->weights.w_gate_off.resize(c.n_layers, 0);
    h->weights.w_up  .resize(c.n_layers, nullptr); h->weights.w_up_off  .resize(c.n_layers, 0);
    h->weights.w_down.resize(c.n_layers, nullptr); h->weights.w_down_off.resize(c.n_layers, 0);
    if (v_split) {
        h->weights.w_v.resize(c.n_layers, nullptr); h->weights.w_v_off.resize(c.n_layers, 0);
    }

    h->weights.dt_qkv  = dt_qk;
    h->weights.dt_o    = dt_o;
    h->weights.dt_gate = dt_gate;
    h->weights.dt_up   = dt_up;
    h->weights.dt_down_layer = dt_down_layer;
    if (v_split) h->weights.dt_v_layer = dt_v_layer;

    // Look up a tensor's absolute byte offset in the GGUF file.
    auto find_abs_off = [&](const char* name, uint64_t* out_off, uint64_t* out_nbytes) -> bool {
        for (const auto& ti : gmodel.tensors) {
            if (ti.name == name) {
                if (out_off)    *out_off    = ti.abs_offset;
                if (out_nbytes) *out_nbytes = ti.nbytes;
                return true;
            }
        }
        return false;
    };

    const bool disable_mmap = std::getenv("SK_NO_MMAP_WEIGHTS") != nullptr;

    // Try to mmap a single-GGUF-tensor span into a per-layer MTL::Buffer.
    // Falls back to a fresh MTL::Buffer + memcpy on failure or when disabled.
    auto load_single_tensor = [&](const char* name, sk::Dtype want_dt, size_t expect_bytes,
                                  MTL::Buffer** out_buf, size_t* out_off) -> bool {
        auto* v = store.get(name);
        if (!v) { std::fprintf(stderr, "gguf: missing %s\n", name); return false; }
        if (v->dtype != want_dt) {
            std::fprintf(stderr, "gguf: dtype mismatch %s (got %d want %d)\n",
                         name, (int)v->dtype, (int)want_dt);
            return false;
        }
        if (v->nbytes != expect_bytes) {
            std::fprintf(stderr, "gguf: size mismatch %s got %zu want %zu\n",
                         name, v->nbytes, expect_bytes);
            return false;
        }
        uint64_t abs_off = 0, nb = 0;
        if (!find_abs_off(name, &abs_off, &nb)) {
            std::fprintf(stderr, "gguf: %s not in gmodel.tensors\n", name);
            return false;
        }
        if (!disable_mmap) {
            size_t inner = 0;
            auto* mb = sk::silicon::MmapBuffer::from_fd_range(
                dev, gguf_fd, (size_t)abs_off, (size_t)nb, &inner);
            if (mb) {
                h->tensor_mmaps.push_back(mb);
                *out_buf = mb->buffer();
                *out_off = inner;
                return true;
            }
            std::fprintf(stderr, "gguf: %s mmap failed; falling back to memcpy\n", name);
        }
        // Memcpy fallback (or SK_NO_MMAP_WEIGHTS=1).
        auto* b = dev->newBuffer(expect_bytes, MTL::ResourceStorageModeShared);
        if (!b) return false;
        std::memcpy(b->contents(), v->data, expect_bytes);
        *out_buf = b;
        *out_off = 0;
        return true;
    };

    // QKV slab. Uniform-dtype models: one per-layer [Q|K|V] row-concat buffer
    // (mmap'd as a single range when Q/K/V are file-adjacent). Mixed-dtype
    // (Q4_K_M: Q/K=Q4_K, V=Q6_K): w_qkv holds [Q|K] (dt_qk), V loads separately
    // into w_v (dt_v) — the dispatch issues two matvecs writing one packed out.
    auto load_qkv_layer = [&](uint32_t L) -> bool {
        char qn[128], kn[128], vn[128];
        std::snprintf(qn, sizeof(qn), "blk.%u.attn_q.weight", L);
        std::snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", L);
        std::snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", L);
        const sk::Dtype dt_vL = dt_v_layer[L];
        const size_t qb = sk::dtype_bytes(dt_qk, Nq  * dm);
        const size_t kb = sk::dtype_bytes(dt_qk, Nkv * dm);
        const size_t vb = sk::dtype_bytes(dt_vL, Nkv * dm);

        auto* qv = store.get(qn); auto* kv = store.get(kn); auto* vv = store.get(vn);
        if (!qv || !kv || !vv) { std::fprintf(stderr, "gguf: qkv miss L=%u\n", L); return false; }
        if (qv->nbytes != qb || kv->nbytes != kb || vv->nbytes != vb ||
            qv->dtype != dt_qk || kv->dtype != dt_qk || vv->dtype != dt_vL) {
            std::fprintf(stderr, "gguf: qkv shape/dtype mismatch L=%u\n", L);
            return false;
        }

        uint64_t qoff = 0, koff = 0, voff_ = 0, qn_ = 0, kn_ = 0, vn_ = 0;
        bool have_offs =
            find_abs_off(qn, &qoff, &qn_) &&
            find_abs_off(kn, &koff, &kn_) &&
            find_abs_off(vn, &voff_, &vn_);

        if (v_split) {
            // V lives in its own buffer; pack only Q|K into w_qkv.
            const size_t qk_total = qb + kb;
            bool qk_done = false;
            if (!disable_mmap && have_offs && koff == qoff + qn_) {
                size_t inner = 0;
                auto* mb = sk::silicon::MmapBuffer::from_fd_range(
                    dev, gguf_fd, (size_t)qoff, qk_total, &inner);
                if (mb) {
                    h->tensor_mmaps.push_back(mb);
                    h->weights.w_qkv[L]     = mb->buffer();
                    h->weights.w_qkv_off[L] = inner;
                    qk_done = true;
                }
            }
            if (!qk_done) {
                auto* b = dev->newBuffer(qk_total, MTL::ResourceStorageModeShared);
                if (!b) return false;
                char* dst = (char*)b->contents();
                std::memcpy(dst,      qv->data, qb);
                std::memcpy(dst + qb, kv->data, kb);
                h->weights.w_qkv[L]     = b;
                h->weights.w_qkv_off[L] = 0;
            }
            // V: own buffer (mmap range or memcpy), per-layer dtype.
            if (!load_single_tensor(vn, dt_vL, vb,
                                    &h->weights.w_v[L], &h->weights.w_v_off[L])) return false;
            return true;
        }

        const size_t total = qb + kb + vb;
        if (!disable_mmap && have_offs &&
            koff == qoff + qn_ && voff_ == koff + kn_) {
            // Q, K, V are file-adjacent → mmap one combined range.
            size_t inner = 0;
            auto* mb = sk::silicon::MmapBuffer::from_fd_range(
                dev, gguf_fd, (size_t)qoff, total, &inner);
            if (mb) {
                h->tensor_mmaps.push_back(mb);
                h->weights.w_qkv[L]     = mb->buffer();
                h->weights.w_qkv_off[L] = inner;
                return true;
            }
            std::fprintf(stderr, "gguf: qkv L=%u mmap failed; falling back to memcpy\n", L);
        }
        // Memcpy fallback: pack Q|K|V into a fresh Buffer.
        auto* b = dev->newBuffer(total, MTL::ResourceStorageModeShared);
        if (!b) return false;
        char* dst = (char*)b->contents();
        std::memcpy(dst,           qv->data, qb);
        std::memcpy(dst + qb,      kv->data, kb);
        std::memcpy(dst + qb + kb, vv->data, vb);
        h->weights.w_qkv[L]     = b;
        h->weights.w_qkv_off[L] = 0;
        return true;
    };

    char nbuf[128];
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const size_t pre_off   = (size_t)L * dm * 2;
        const size_t qnorm_off = (size_t)L * hd * 2;

        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_norm.weight", L);
        if (!read_to_fp16((uint16_t*)((char*)h->weights.w_pre_attn_norm->contents() + pre_off),
                          store.get(nbuf), dm)) return -40;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_norm.weight", L);
        if (!read_to_fp16((uint16_t*)((char*)h->weights.w_pre_mlp_norm->contents() + pre_off),
                          store.get(nbuf), dm)) return -41;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_q_norm.weight", L);
        if (!read_to_fp16((uint16_t*)((char*)h->weights.w_q_norm->contents() + qnorm_off),
                          store.get(nbuf), hd)) return -42;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_k_norm.weight", L);
        if (!read_to_fp16((uint16_t*)((char*)h->weights.w_k_norm->contents() + qnorm_off),
                          store.get(nbuf), hd)) return -43;

        if (!load_qkv_layer(L)) return -50;

        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_output.weight", L);
        if (!load_single_tensor(nbuf, dt_o, o_layer_bytes,
                                &h->weights.w_o[L], &h->weights.w_o_off[L])) return -53;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_gate.weight", L);
        if (!load_single_tensor(nbuf, dt_gate, ffn_layer_bytes_gate,
                                &h->weights.w_gate[L], &h->weights.w_gate_off[L])) return -54;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_up.weight", L);
        if (!load_single_tensor(nbuf, dt_up, ffn_layer_bytes_up,
                                &h->weights.w_up[L], &h->weights.w_up_off[L])) return -55;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_down.weight", L);
        const size_t down_bytes_L = sk::dtype_bytes(dt_down_layer[L], ni * dm);
        if (!load_single_tensor(nbuf, dt_down_layer[L], down_bytes_L,
                                &h->weights.w_down[L], &h->weights.w_down_off[L])) return -56;
    }

    ::close(gguf_fd);
    return 0;
}
