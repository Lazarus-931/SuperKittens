//
//  dispatch.h — GPU kernel dispatch over SKTensor handles
//
//  Single-kernel functions for one-off use.
//  Batch API chains multiple kernels in one MTLCommandBuffer — intermediates
//  stay in GPU memory, one commit instead of N.
//

#ifndef SK_DISPATCH_H
#define SK_DISPATCH_H

#include "tensor.h"
#include <Metal/Metal.hpp>

MTL::Device*      sk_device();
MTL::CommandQueue* sk_queue();

// ── single-kernel dispatch ────────────────────────────────────────

extern "C" int sk_dispatch_elementwise(
    const char* kernel_name, void* x_handle, void* y_handle,
    const char* metallib_path);

extern "C" int sk_dispatch_gemm(
    const char* kernel_name, void* a_handle, void* b_handle, void* c_handle,
    const char* metallib_path);

extern "C" int sk_dispatch_attention(
    void* q_handle, void* k_handle, void* v_handle, void* o_handle,
    int causal, const char* metallib_path);

// ── batched dispatch (one MTLCommandBuffer, many kernels) ─────────

/// Opaque handle to an in-progress command batch.
typedef struct sk_batch_t sk_batch_t;

/// Create a new batch.  All add_* calls go into it.  commit submits.
extern "C" sk_batch_t* sk_batch_create(const char* metallib_path);

/// Add an elementwise kernel to the batch.  x → y.
extern "C" int sk_batch_add_elementwise(
    sk_batch_t* batch, const char* kernel_name, void* x_handle, void* y_handle);

/// Add a GEMM to the batch.  a @ b → c.
extern "C" int sk_batch_add_gemm(
    sk_batch_t* batch, const char* kernel_name,
    void* a_handle, void* b_handle, void* c_handle);

/// Add attention to the batch.  q, k, v → o.
extern "C" int sk_batch_add_attention(
    sk_batch_t* batch, void* q_handle, void* k_handle, void* v_handle, void* o_handle,
    int causal);

/// Submit all queued kernels and wait for completion.
extern "C" int sk_batch_commit(sk_batch_t* batch);

/// Free batch resources.
extern "C" void sk_batch_free(sk_batch_t* batch);

#endif
