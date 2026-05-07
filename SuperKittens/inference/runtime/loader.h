//
//  loader.h — weight loading: raw bytes → MTLBuffer
//
//  Model weights are loaded once into GPU memory, then reused across
//  inference calls with zero CPU round-trips.
//
//  Usage:
//    auto W = sk_load_fp16(device, data_ptr, rows, cols);
//    // W is an SKTensor backed by MTL::Buffer via Shared memory
//    // dispatch_gemm("gemm_fp16", x_handle, W_handle, out_handle);
//

#ifndef SK_LOADER_H
#define SK_LOADER_H

#include "tensor.h"
#include <cstring>

// Forward declaration — defined in tensor.c++
MTL::Device* sk_device();

namespace sk {
namespace runtime {

/// Load fp16 weight matrix [rows][cols] from CPU into GPU buffer.
/// Returns tensor that OWNS its buffer — caller must tensor_free.
inline SKTensor load_fp16(MTL::Device* dev, const void* data,
                           uint32_t rows, uint32_t cols) {
    auto t = tensor_alloc_impl(dev, rows, DType::f16, cols);
    std::memcpy(t.buf->contents(), data, t.nbytes());
    return t;
}

/// Load fp32 weight, converting to fp16 on upload.
inline SKTensor load_fp32_as_fp16(MTL::Device* dev, const float* data,
                                   uint32_t rows, uint32_t cols) {
    auto t = tensor_alloc_impl(dev, rows, DType::f16, cols);
    auto* dst = static_cast<__fp16*>(t.buf->contents());
    for (uint32_t i = 0; i < t.numel(); i++) dst[i] = __fp16(data[i]);
    return t;
}

} // namespace runtime
} // namespace sk

// ── extern C (Python ctypes) ────────────────────────────────────────

extern "C" {

/// Load fp16 weights from CPU pointer into GPU tensor.
/// Returns handle (caller must sk_tensor_free).
void* sk_load_fp16(const void* data, uint32_t rows, uint32_t cols);

/// Load fp32 weights, auto-convert to fp16 GPU tensor.
void* sk_load_fp32_as_fp16(const float* data, uint32_t rows, uint32_t cols);

} // extern "C"

#endif
