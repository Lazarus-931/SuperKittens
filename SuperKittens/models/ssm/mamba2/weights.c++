// weights.c++ — load HF mamba2-130m-hf safetensors into fused SK buffers.
//
// Notes on layout conventions (vs. HF):
//   * HF in_proj.weight has shape (IN_OUT, d_model). SK stores transposed
//     (d_model, IN_OUT) so the GEMM op is (T,D) x (D,IN_OUT).
//   * HF conv1d.weight has shape (C_in, 1, K). SK stores (K, C_in) for the
//     depthwise causal convolution kernel ordering used in conv1d_silu.metal.
//   * HF out_proj.weight has shape (d_model, E). SK stores transposed (E, D).
//   * HF A_log is stored as log(A); the kernel takes -exp(A_log). We pass
//     A_log raw; the SSD kernel applies the negation.
//   * tied lm_head — no separate weight; uses w_embed transposed at decode.

#include "weights.h"
#include "mamba2_model.h"
#include "../../../inference/weight_store.h"
#include "../../load/safetensor/safetensor.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace meow { namespace mamba2 {
struct Handle {
    sk_mamba2_config cfg;
    uint32_t       current_pos;
    ModelPSOs      psos;
    ModelWeights   weights;
    ModelBuffers   bufs;
    std::vector<LayerState> layer_states;
};
}}

namespace {

inline std::string lkey(uint32_t L, const char* suffix) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "backbone.layers.%u.%s", L, suffix);
    return std::string(buf);
}

inline uint16_t fp32_to_fp16(float fv) {
    uint32_t f; std::memcpy(&f, &fv, 4);
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

inline uint16_t bf16_to_fp16(uint16_t b) {
    uint32_t f = ((uint32_t)b) << 16;
    float fv; std::memcpy(&fv, &f, 4);
    return fp32_to_fp16(fv);
}

bool copy_into(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
               const std::string& name, size_t expect_bytes)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "mamba2 weights: missing '%s'\n", name.c_str()); return false; }
    const size_t nelem = expect_bytes / 2;
    size_t src_elem = (v->dtype == sk::Dtype::F32) ? 4
                    : (v->dtype == sk::Dtype::BF16 || v->dtype == sk::Dtype::F16) ? 2 : 0;
    if (src_elem == 0 || v->nbytes != nelem * src_elem) {
        std::fprintf(stderr, "mamba2 weights: size mismatch '%s' got %zu expect %zu (dtype=%d)\n",
                     name.c_str(), v->nbytes, nelem * src_elem, (int)v->dtype);
        return false;
    }
    char* out = (char*)dst->contents() + dst_off;
    if (v->dtype == sk::Dtype::BF16) {
        const uint16_t* s = (const uint16_t*)v->data;
        uint16_t* d = (uint16_t*)out;
        for (size_t i = 0; i < nelem; ++i) d[i] = bf16_to_fp16(s[i]);
    } else if (v->dtype == sk::Dtype::F32) {
        const float* s = (const float*)v->data;
        uint16_t* d = (uint16_t*)out;
        for (size_t i = 0; i < nelem; ++i) d[i] = fp32_to_fp16(s[i]);
    } else {
        std::memcpy(out, v->data, expect_bytes);
    }
    return true;
}

// HF stores in_proj.weight as (OUT, IN). We need (IN, OUT) for our GEMM.
bool copy_transpose_fp16(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
                         const std::string& name, size_t out_rows, size_t out_cols)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "mamba2 weights: missing '%s'\n", name.c_str()); return false; }
    const size_t nelem = out_rows * out_cols;
    size_t src_elem = (v->dtype == sk::Dtype::F32) ? 4
                    : (v->dtype == sk::Dtype::BF16 || v->dtype == sk::Dtype::F16) ? 2 : 0;
    if (src_elem == 0 || v->nbytes != nelem * src_elem) {
        std::fprintf(stderr, "mamba2 weights: tx size mismatch '%s' got %zu expect %zu (dtype=%d)\n",
                     name.c_str(), v->nbytes, nelem * src_elem, (int)v->dtype);
        return false;
    }
    uint16_t* d = (uint16_t*)((char*)dst->contents() + dst_off);
    auto at = [&](size_t idx) -> uint16_t {
        if (v->dtype == sk::Dtype::F32)  return fp32_to_fp16(((const float*)v->data)[idx]);
        if (v->dtype == sk::Dtype::BF16) return bf16_to_fp16(((const uint16_t*)v->data)[idx]);
        return ((const uint16_t*)v->data)[idx];
    };
    for (size_t i = 0; i < out_rows; ++i)
        for (size_t j = 0; j < out_cols; ++j)
            d[i * out_cols + j] = at(j * out_rows + i);
    return true;
}

// HF conv1d.weight is (C_in, 1, K). Strip the inner-1 dim → (C_in, K), then
// transpose to (K, C_in) which is the depthwise layout the SK conv1d_silu
// kernel expects (each thread reads K weights for one channel column).
bool copy_conv1d(MTL::Buffer* dst, size_t dst_off, sk::WeightStore* store,
                 const std::string& name, size_t C_in, size_t K)
{
    auto* v = store->get(name);
    if (!v) { std::fprintf(stderr, "mamba2 weights: missing '%s'\n", name.c_str()); return false; }
    const size_t nelem = C_in * K;
    size_t src_elem = (v->dtype == sk::Dtype::F32) ? 4
                    : (v->dtype == sk::Dtype::BF16 || v->dtype == sk::Dtype::F16) ? 2 : 0;
    if (src_elem == 0 || v->nbytes != nelem * src_elem) {
        std::fprintf(stderr, "mamba2 weights: conv size mismatch '%s' got %zu expect %zu (dtype=%d)\n",
                     name.c_str(), v->nbytes, nelem * src_elem, (int)v->dtype);
        return false;
    }
    uint16_t* d = (uint16_t*)((char*)dst->contents() + dst_off);
    auto at = [&](size_t idx) -> uint16_t {
        if (v->dtype == sk::Dtype::F32)  return fp32_to_fp16(((const float*)v->data)[idx]);
        if (v->dtype == sk::Dtype::BF16) return bf16_to_fp16(((const uint16_t*)v->data)[idx]);
        return ((const uint16_t*)v->data)[idx];
    };
    for (size_t c = 0; c < C_in; ++c)
        for (size_t k = 0; k < K; ++k)
            d[c * K + k] = at(c * K + k);
    return true;
}

}  // anon

extern "C" int sk_mamba2_load_from_store(sk_mamba2_handle* hp, sk::WeightStore* store) {
    if (!hp || !store) return -1;
    auto* h = reinterpret_cast<meow::mamba2::Handle*>(hp);
    const auto& c = h->cfg;
    const size_t fp16 = 2;
    const size_t D = c.d_model;
    const size_t E = c.intermediate;
    const size_t H = c.n_heads;
    const size_t G = c.n_groups;
    const size_t N = c.state_size;
    const size_t K = c.conv_kernel;
    const size_t IN_OUT = 2 * E + 2 * G * N + H;
    const size_t C_in   = E + 2 * G * N;

    if (!copy_into(h->weights.w_embed, 0, store,
                   "backbone.embeddings.weight", (size_t)c.vocab_size * D * fp16)) return -10;
    if (!copy_into(h->weights.w_final_norm, 0, store,
                   "backbone.norm_f.weight", D * fp16)) return -11;

    for (uint32_t L = 0; L < c.n_layers; ++L) {
        const size_t pre_off    = (size_t)L * D * fp16;
        const size_t in_off     = (size_t)L * D * IN_OUT * fp16;
        const size_t conv_off   = (size_t)L * K * C_in * fp16;
        const size_t conv_b_off = (size_t)L * C_in * fp16;
        const size_t dt_off     = (size_t)L * H * fp16;
        const size_t A_off      = (size_t)L * H * fp16;
        const size_t D_off      = (size_t)L * H * fp16;
        const size_t norm_off   = (size_t)L * E * fp16;
        const size_t out_off    = (size_t)L * E * D * fp16;

        if (!copy_into(h->weights.w_pre_norm, pre_off, store,
                       lkey(L, "norm.weight"), D * fp16)) return -20;

        // HF in_proj.weight shape (IN_OUT, D) → store (D, IN_OUT)
        if (!copy_transpose_fp16(h->weights.w_in_proj, in_off, store,
                                 lkey(L, "mixer.in_proj.weight"), D, IN_OUT)) return -21;

        if (!copy_conv1d(h->weights.w_conv, conv_off, store,
                         lkey(L, "mixer.conv1d.weight"), C_in, K)) return -22;
        if (!copy_into(h->weights.w_conv_b, conv_b_off, store,
                       lkey(L, "mixer.conv1d.bias"), C_in * fp16)) return -23;

        if (!copy_into(h->weights.w_dt_bias, dt_off, store,
                       lkey(L, "mixer.dt_bias"), H * fp16)) return -24;
        if (!copy_into(h->weights.w_A_log, A_off, store,
                       lkey(L, "mixer.A_log"), H * fp16)) return -25;
        if (!copy_into(h->weights.w_D, D_off, store,
                       lkey(L, "mixer.D"), H * fp16)) return -26;

        if (!copy_into(h->weights.w_norm, norm_off, store,
                       lkey(L, "mixer.norm.weight"), E * fp16)) return -27;

        // HF out_proj.weight shape (D, E) → store (E, D)
        if (!copy_transpose_fp16(h->weights.w_out_proj, out_off, store,
                                 lkey(L, "mixer.out_proj.weight"), E, D)) return -28;
    }
    return 0;
}

extern "C" int sk_mamba2_load_safetensors(sk_mamba2_handle* h, const char* path) {
    if (!h || !path) return -1;
    auto store = new sk::WeightStore();
    int rc = sk::load_safetensors(path, *store);
    if (rc != 0) { delete store; return rc; }
    rc = sk_mamba2_load_from_store(h, store);
    delete store;
    return rc;
}

extern "C" int sk_mamba2_load_safetensors_index(sk_mamba2_handle* h, const char* index_json_path) {
    if (!h || !index_json_path) return -1;
    auto store = new sk::WeightStore();
    int rc = sk::load_safetensors_index(index_json_path, *store);
    if (rc != 0) { delete store; return rc; }
    rc = sk_mamba2_load_from_store(h, store);
    delete store;
    return rc;
}
