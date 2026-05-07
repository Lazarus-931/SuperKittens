//
//  tensor.h — SKTensor: runtime GPU tensor + extern C API for Python
//

#ifndef SK_TENSOR_H
#define SK_TENSOR_H

#include <cstdint>
#include <Metal/Metal.hpp>

namespace sk {
namespace runtime {

enum class DType : uint8_t { f16=0, f32=1, i32=2, i8=3, bf16=4 };

inline uint32_t dtype_size(DType dt) {
    switch (dt) { case DType::f16:return 2; case DType::bf16:return 2; case DType::f32:return 4; case DType::i32:return 4; case DType::i8:return 1; }
    return 2;
}

struct SKTensor {
    MTL::Buffer* buf = nullptr;
    bool owns = false;
    DType dtype = DType::f16;
    uint8_t ndim = 0;
    uint32_t shape[4] = {0,0,0,0};
    uint32_t stride[4] = {0,0,0,0};

    uint32_t numel() const { uint32_t n=1; for(int i=0;i<ndim;i++)n*=shape[i]; return n; }
    uint32_t nbytes() const { return numel()*dtype_size(dtype); }
    void* data() { return buf?buf->contents():nullptr; }
};

// Internal factory (called from extern C wrappers)
SKTensor tensor_alloc_impl(MTL::Device* d, uint32_t d0, DType dt, uint32_t d1=1, uint32_t d2=1, uint32_t d3=1);

} // namespace runtime
} // namespace sk

// ── extern "C" API (called from Python via ctypes) ──────────────────

extern "C" {

/// Allocate a GPU tensor. Returns opaque handle (heap pointer).
/// Caller must free with sk_tensor_free.
void* sk_tensor_alloc(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, int dtype);

/// Free a tensor allocated by sk_tensor_alloc.
void sk_tensor_free(void* handle);

/// Copy GPU → CPU.  dst must be pre-allocated with at least tensor.nbytes().
void sk_tensor_to_cpu(void* handle, void* dst);

/// Copy CPU → GPU.  src must have at least tensor.nbytes().
void sk_tensor_from_cpu(void* handle, const void* src);

/// Query tensor shape.  Returns ndim, fills shape_out[0..ndim-1].
int sk_tensor_shape(void* handle, uint32_t* shape_out);

/// Query tensor dtype.
int sk_tensor_dtype(void* handle);

/// Total element count.
uint32_t sk_tensor_numel(void* handle);
}

#endif
