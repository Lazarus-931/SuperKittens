//
//  dispatch.c++ — GPU kernel dispatch with PSO caching
//

#include "dispatch.h"
#include <unordered_map>
#include <string>
#include <cstring>
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

// ── PSO cache ──────────────────────────────────────────────────────

static MTL::Library* g_metallib = nullptr;
static const char*   g_metallib_path = nullptr;
static std::unordered_map<std::string, MTL::ComputePipelineState*> g_psos;

static MTL::ComputePipelineState* get_pso(const char* kernel_name, const char* path) {
    // Load metallib once
    if (!g_metallib || (path && (!g_metallib_path || std::strcmp(g_metallib_path, path) != 0))) {
        const char* p = path ? path : (getenv("SK_METALLIB") ? getenv("SK_METALLIB") : "build/libsk.metallib");
        g_metallib_path = p;
        NS::Error* err = nullptr;
        auto* url = NS::URL::fileURLWithPath(NS::String::string(p, NS::UTF8StringEncoding));
        g_metallib = sk_device()->newLibrary(url, &err);
        if (!g_metallib) return nullptr;
    }

    // Check cache
    std::string key(kernel_name);
    auto it = g_psos.find(key);
    if (it != g_psos.end()) return it->second;

    // Create and cache
    auto* fn = g_metallib->newFunction(NS::String::string(kernel_name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    NS::Error* err = nullptr;
    auto* pso = sk_device()->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) { if (err) err->release(); return nullptr; }
    g_psos[key] = pso;
    return pso;
}

// ── batch infrastructure ───────────────────────────────────────────

struct sk_batch_t {
    MTL::CommandBuffer*          cmd = nullptr;
    MTL::ComputeCommandEncoder*  enc = nullptr;
    const char*                  metallib_path = nullptr;
    bool                         committed = false;
};

static MTL::ComputeCommandEncoder* ensure_encoder(sk_batch_t* b) {
    if (!b->enc) b->enc = b->cmd->computeCommandEncoder();
    return b->enc;
}

extern "C"
sk_batch_t* sk_batch_create(const char* metallib_path) {
    auto* b = new sk_batch_t;
    b->cmd = sk_queue()->commandBuffer();
    b->metallib_path = metallib_path;
    return b;
}

extern "C"
void sk_batch_free(sk_batch_t* batch) {
    if (!batch) return;
    if (!batch->committed && batch->cmd) batch->cmd->release();
    delete batch;
}

extern "C"
int sk_batch_commit(sk_batch_t* b) {
    if (!b || b->committed) return -1;
    if (b->enc) { b->enc->endEncoding(); b->enc = nullptr; }
    b->cmd->commit();
    b->cmd->waitUntilCompleted();
    b->cmd->release();
    b->committed = true;
    return 0;
}

// ── elementwise dispatch ───────────────────────────────────────────

extern "C"
int sk_dispatch_elementwise(const char* kernel_name, void* x_handle, void* y_handle,
                             const char* metallib_path) {
    auto* x = static_cast<sk::runtime::SKTensor*>(x_handle);
    auto* y = static_cast<sk::runtime::SKTensor*>(y_handle);

    // Treat as 2D: flatten leading dims into "rows"
    uint32_t rows = 1;
    for (int i = 0; i < x->ndim - 1; i++) rows *= x->shape[i];
    uint32_t cols = x->shape[x->ndim - 1];

    auto* pso = get_pso(kernel_name, metallib_path);
    if (!pso) return -1;

    uint32_t grid_y = (rows + 3) / 4;  // 4 rows per SIMD group

    auto* cmd = sk_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(x->buf, 0, 0);
    enc->setBuffer(y->buf, 0, 1);
    enc->setBytes(&rows, sizeof(uint32_t), 2);
    enc->setBytes(&cols, sizeof(uint32_t), 3);
    enc->dispatchThreadgroups(MTL::Size(1, grid_y, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();
    return 0;
}

// ── GEMM dispatch ──────────────────────────────────────────────────

extern "C"
int sk_dispatch_gemm(const char* kernel_name, void* a_handle, void* b_handle,
                      void* c_handle, const char* metallib_path) {
    auto* a = static_cast<sk::runtime::SKTensor*>(a_handle);
    auto* b = static_cast<sk::runtime::SKTensor*>(b_handle);
    auto* c = static_cast<sk::runtime::SKTensor*>(c_handle);

    const uint32_t M = a->shape[0], K = a->shape[1], N = b->shape[1];

    auto* pso = get_pso(kernel_name, metallib_path);
    if (!pso) return -1;

    uint32_t grid_x = (N + 63) / 64;
    uint32_t grid_y = (M + 63) / 64;

    auto* cmd = sk_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(a->buf, 0, 0);
    enc->setBuffer(b->buf, 0, 1);
    enc->setBuffer(c->buf, 0, 2);
    enc->setBytes(&M, sizeof(uint32_t), 3);
    enc->setBytes(&N, sizeof(uint32_t), 4);
    enc->setBytes(&K, sizeof(uint32_t), 5);
    enc->dispatchThreadgroups(MTL::Size(grid_x, grid_y, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();
    return 0;
}

// ── attention dispatch ─────────────────────────────────────────────

extern "C"
int sk_dispatch_attention(void* q_handle, void* kv_handle, void* kv2_handle,
                           void* o_handle, int causal, const char* metallib_path) {
    auto* qt = static_cast<sk::runtime::SKTensor*>(q_handle);
    auto* kt = static_cast<sk::runtime::SKTensor*>(kv_handle);
    auto* vt = static_cast<sk::runtime::SKTensor*>(kv2_handle);
    auto* ot = static_cast<sk::runtime::SKTensor*>(o_handle);

    const uint32_t n_heads = qt->shape[0];
    const uint32_t seq     = qt->shape[1];
    const uint32_t d       = qt->shape[2];

    const char* kname = (d == 64)
        ? (causal ? "fa_causal_64" : "fa_noncausal_64")
        : (causal ? "mha_causal"   : "mha_noncausal");

    auto* pso = get_pso(kname, metallib_path);
    if (!pso) return -1;

    const uint32_t grid_x = n_heads;
    const uint32_t grid_y = (d == 64) ? (seq + 31) / 32 : (seq + 3) / 4;
    const uint32_t threads = (d == 64) ? 1024 : 128;

    auto* cmd = sk_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(qt->buf, 0, 0);
    enc->setBuffer(kt->buf, 0, 1);
    enc->setBuffer(vt->buf, 0, 2);
    enc->setBuffer(ot->buf, 0, 3);
    enc->setBytes(&seq,      sizeof(uint32_t), 4);
    enc->setBytes(&d,        sizeof(uint32_t), 5);
    enc->setBytes(&n_heads,  sizeof(uint32_t), 6);
    enc->dispatchThreadgroups(MTL::Size(grid_x, grid_y, 1), MTL::Size(threads, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();
    return 0;
}
