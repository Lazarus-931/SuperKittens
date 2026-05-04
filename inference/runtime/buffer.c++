//
//  runtime/buffer.c++ — Buffer implementation
//

#include "buffer.h"
#include "device.h"
#include <cstring>

namespace sk::runtime {

Buffer Buffer::alloc(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, DType dt) {
    Buffer b;
    b.shape[0] = d0; b.shape[1] = d1; b.shape[2] = d2; b.shape[3] = d3;
    b.ndim  = (d3 > 1) ? 4 : (d2 > 1) ? 3 : (d1 > 1) ? 2 : 1;
    b.dtype = dt;
    b.bytes = b.elements() * dtype_size(dt);
    b.mtl   = device()->newBuffer(b.bytes, MTL::ResourceStorageModeShared);
    return b;
}

Buffer Buffer::from(const void* data, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, DType dt) {
    Buffer b = alloc(d0, d1, d2, d3, dt);
    memcpy(b.mtl->contents(), data, b.bytes);
    return b;
}

void Buffer::read(void* dst) const {
    memcpy(dst, mtl->contents(), bytes);
}

Buffer constant_buffer(const void* value, size_t size) {
    Buffer b;
    b.shape[0] = 1; b.ndim = 1; b.dtype = DType::u8; b.bytes = size;
    b.mtl = device()->newBuffer(value, size, MTL::ResourceStorageModeShared);
    return b;
}

} // namespace sk::runtime
