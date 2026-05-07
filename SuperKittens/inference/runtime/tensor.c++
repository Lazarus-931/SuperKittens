//
//  tensor.c++ — GPU tensor allocation + device singleton
//

#include "tensor.h"
#include <cstdlib>
#include <cstring>

// ── shared device singleton ────────────────────────────────────────
// All kernel dispatchers (sk_activation, sk_attn, ...) share this.
// Created lazily on first use.

static MTL::Device*      g_device = nullptr;
static MTL::CommandQueue* g_queue  = nullptr;

MTL::Device* sk_device() {
    if (!g_device) {
        g_device = MTL::CreateSystemDefaultDevice();
        g_queue  = g_device->newCommandQueue();
    }
    return g_device;
}

MTL::CommandQueue* sk_queue() {
    sk_device();  // ensure initialized
    return g_queue;
}

namespace sk {
namespace runtime {

SKTensor tensor_alloc_impl(MTL::Device* dev, uint32_t d0, DType dt,
                            uint32_t d1, uint32_t d2, uint32_t d3) {
    SKTensor t;
    t.dtype = dt;
    t.owns  = true;
    t.shape[0] = d0; t.shape[1] = d1; t.shape[2] = d2; t.shape[3] = d3;

    if (d3 > 1) t.ndim = 4;
    else if (d2 > 1) t.ndim = 3;
    else if (d1 > 1) t.ndim = 2;
    else t.ndim = 1;

    t.stride[3] = 1;
    t.stride[2] = t.shape[3];
    t.stride[1] = t.shape[2] * t.shape[3];
    t.stride[0] = t.shape[1] * t.shape[2] * t.shape[3];

    t.buf = dev->newBuffer(t.nbytes(), MTL::ResourceStorageModeShared);
    return t;
}

} // namespace runtime
} // namespace sk

// ── extern "C" implementations ─────────────────────────────────────

void* sk_tensor_alloc(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, int dtype) {
    auto dt = static_cast<sk::runtime::DType>(dtype);
    auto* t = new sk::runtime::SKTensor(
        sk::runtime::tensor_alloc_impl(sk_device(), d0, dt, d1, d2, d3));
    return static_cast<void*>(t);
}

void sk_tensor_free(void* handle) {
    if (!handle) return;
    auto* t = static_cast<sk::runtime::SKTensor*>(handle);
    if (t->owns && t->buf) t->buf->release();
    delete t;
}

void sk_tensor_to_cpu(void* handle, void* dst) {
    auto* t = static_cast<sk::runtime::SKTensor*>(handle);
    std::memcpy(dst, t->buf->contents(), t->nbytes());
}

void sk_tensor_from_cpu(void* handle, const void* src) {
    auto* t = static_cast<sk::runtime::SKTensor*>(handle);
    std::memcpy(t->buf->contents(), src, t->nbytes());
}

int sk_tensor_shape(void* handle, uint32_t* shape_out) {
    auto* t = static_cast<sk::runtime::SKTensor*>(handle);
    for (int i = 0; i < t->ndim; i++) shape_out[i] = t->shape[i];
    return t->ndim;
}

int sk_tensor_dtype(void* handle) {
    return static_cast<int>(
        static_cast<sk::runtime::SKTensor*>(handle)->dtype);
}

uint32_t sk_tensor_numel(void* handle) {
    return static_cast<sk::runtime::SKTensor*>(handle)->numel();
}
