//
//  runtime/buffer.h — Typed Metal buffer with shape + dtype
//  Handles the "allocate → memcpy in → dispatch → memcpy out" pattern.
//

#ifndef SK_RUNTIME_BUFFER_H
#define SK_RUNTIME_BUFFER_H

#include <Metal/Metal.hpp>
#include <cstdint>

namespace sk::runtime {

enum class DType : uint8_t { f16 = 0, f32 = 1, u8 = 2, i32 = 3 };

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::f16: return 2;
        case DType::f32: return 4;
        case DType::u8:  return 1;
        case DType::i32: return 4;
    }
    return 2;
}

struct Buffer {
    MTL::Buffer* mtl = nullptr;
    uint32_t     shape[4] = {0,0,0,0};
    uint32_t     ndim    = 0;
    DType        dtype   = DType::f16;
    size_t       bytes   = 0;

    // Allocate uninitialized shared buffer
    static Buffer alloc(uint32_t d0, uint32_t d1=1, uint32_t d2=1, uint32_t d3=1,
                         DType dt = DType::f16);

    // Allocate + copy host data in
    static Buffer from(const void* data, uint32_t d0, uint32_t d1=1, uint32_t d2=1, uint32_t d3=1,
                        DType dt = DType::f16);

    // Copy from Metal buffer back to host
    void read(void* dst) const;

    // Number of elements
    size_t elements() const { return (size_t)shape[0] * shape[1] * shape[2] * shape[3]; }

    void release() { if (mtl) mtl->release(); mtl = nullptr; }
};

// ── utility — allocate constant buffer (uint32, float, bool) ──
Buffer constant_buffer(const void* value, size_t size);

} // namespace sk::runtime

#endif
