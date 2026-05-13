#include "weights.h"
#include "qwen_model.h"
#include "../../inference/weight_store.h"
#include "../../kernels/runtime_bindings.h"
#include "../load/gguf/gguf.h"
#include "../load/safetensor/safetensor.h"

#include <cstdio>
#include <cstring>
#include <string>

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

    auto* qkv_base = (char*)h->weights.w_qkv->contents();

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
        if (!copy_transpose_fp16(h->weights.w_o, o_off, store,
                       layer_key(L, "self_attn.o_proj.weight"), Nq, dm)) return -24;
        if (!copy_transpose_fp16(h->weights.w_gate, gate_off, store,
                       layer_key(L, "mlp.gate_proj.weight"), dm, ni)) return -25;
        if (!copy_transpose_fp16(h->weights.w_up, gate_off, store,
                       layer_key(L, "mlp.up_proj.weight"), dm, ni)) return -26;
        if (!copy_transpose_fp16(h->weights.w_down, down_off, store,
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
        // For LM head we prefer to keep Q8_0 layout (V × D row-major) to enable
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
            // Release previously-allocated fp16 lm_head (if any) and create Q8 buffer.
            if (h->weights.w_lm_head) h->weights.w_lm_head->release();
            h->weights.w_lm_head = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
            if (!h->weights.w_lm_head) return -16;
            std::memcpy(h->weights.w_lm_head->contents(), v->data, bytes);
            h->weights.dt_lm_head = sk::Dtype::Q8_0;
        } else if (!c.tie_word_embeddings && h->weights.w_lm_head) {
            if (!read_to_fp16((uint16_t*)h->weights.w_lm_head->contents(), v,
                              (size_t)c.vocab_size * dm)) return -15;
        }
    }

    // Helper: reallocate a buffer to nbytes (releases the old buffer first).
    auto realloc_buf = [&](MTL::Buffer*& b, size_t nbytes) -> bool {
        if (b) b->release();
        b = dev->newBuffer(nbytes, MTL::ResourceStorageModeShared);
        if (b) std::memset(b->contents(), 0, nbytes);
        return b != nullptr;
    };

    // ── Inspect dtype of layer-0 projections to decide allocation sizes. ──
    auto* v0_q = store.get("blk.0.attn_q.weight");
    if (!v0_q) { std::fprintf(stderr, "gguf: missing blk.0.attn_q.weight\n"); return -20; }
    const sk::Dtype dt_proj = v0_q->dtype;
    if (dt_proj != sk::Dtype::Q8_0 && dt_proj != sk::Dtype::F16 && dt_proj != sk::Dtype::BF16) {
        std::fprintf(stderr, "gguf: unsupported projection dtype %d\n", (int)dt_proj);
        return -21;
    }

    const size_t qkv_layer_bytes  = sk::dtype_bytes(dt_proj, dm * qkvN);
    const size_t o_layer_bytes    = sk::dtype_bytes(dt_proj, Nq * dm);
    const size_t ffn_layer_bytes  = sk::dtype_bytes(dt_proj, dm * ni);
    const size_t down_layer_bytes = sk::dtype_bytes(dt_proj, ni * dm);

    if (!realloc_buf(h->weights.w_qkv,  c.n_layers * qkv_layer_bytes))  return -30;
    if (!realloc_buf(h->weights.w_o,    c.n_layers * o_layer_bytes))    return -31;
    if (!realloc_buf(h->weights.w_gate, c.n_layers * ffn_layer_bytes))  return -32;
    if (!realloc_buf(h->weights.w_up,   c.n_layers * ffn_layer_bytes))  return -33;
    if (!realloc_buf(h->weights.w_down, c.n_layers * down_layer_bytes)) return -34;

    h->weights.dt_qkv  = dt_proj;
    h->weights.dt_o    = dt_proj;
    h->weights.dt_gate = dt_proj;
    h->weights.dt_up   = dt_proj;
    h->weights.dt_down = dt_proj;

    auto copy_layer_proj = [&](MTL::Buffer* dst, size_t dst_off, const char* gname,
                               size_t expect_bytes) -> bool {
        auto* v = store.get(gname);
        if (!v) { std::fprintf(stderr, "gguf: missing %s\n", gname); return false; }
        if (v->dtype != dt_proj) {
            std::fprintf(stderr, "gguf: dtype mismatch %s (got %d)\n", gname, (int)v->dtype);
            return false;
        }
        if (v->nbytes != expect_bytes) {
            std::fprintf(stderr, "gguf: size mismatch %s got %zu want %zu\n",
                         gname, v->nbytes, expect_bytes);
            return false;
        }
        std::memcpy((char*)dst->contents() + dst_off, v->data, expect_bytes);
        return true;
    };

    char nbuf[128];
    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const size_t pre_off   = (size_t)L * dm * 2;
        const size_t qnorm_off = (size_t)L * hd * 2;
        const size_t qkv_off   = (size_t)L * qkv_layer_bytes;
        const size_t o_off_b   = (size_t)L * o_layer_bytes;
        const size_t ffn_off   = (size_t)L * ffn_layer_bytes;
        const size_t down_off  = (size_t)L * down_layer_bytes;

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

        // QKV concat: rows are [Q (Nq), K (Nkv), V (Nkv)] with K columns = dm.
        // GGUF stores each separately row-major (N, K). We memcpy them sequentially.
        const size_t qb_bytes  = sk::dtype_bytes(dt_proj, Nq  * dm);
        const size_t kb_bytes  = sk::dtype_bytes(dt_proj, Nkv * dm);
        const size_t vb_bytes  = sk::dtype_bytes(dt_proj, Nkv * dm);
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_q.weight", L);
        if (!copy_layer_proj(h->weights.w_qkv, qkv_off,             nbuf, qb_bytes)) return -50;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_k.weight", L);
        if (!copy_layer_proj(h->weights.w_qkv, qkv_off + qb_bytes,  nbuf, kb_bytes)) return -51;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_v.weight", L);
        if (!copy_layer_proj(h->weights.w_qkv, qkv_off + qb_bytes + kb_bytes,
                             nbuf, vb_bytes)) return -52;

        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.attn_output.weight", L);
        if (!copy_layer_proj(h->weights.w_o,    o_off_b,  nbuf, o_layer_bytes))    return -53;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_gate.weight", L);
        if (!copy_layer_proj(h->weights.w_gate, ffn_off,  nbuf, ffn_layer_bytes))  return -54;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_up.weight", L);
        if (!copy_layer_proj(h->weights.w_up,   ffn_off,  nbuf, ffn_layer_bytes))  return -55;
        std::snprintf(nbuf, sizeof(nbuf), "blk.%u.ffn_down.weight", L);
        if (!copy_layer_proj(h->weights.w_down, down_off, nbuf, down_layer_bytes)) return -56;
    }

    return 0;
}
