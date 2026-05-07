//
//  loader.c++ — weight loading implementations
//

#include "loader.h"
#include "tensor.h"

void* sk_load_fp16(const void* data, uint32_t rows, uint32_t cols) {
    auto t = new sk::runtime::SKTensor(
        sk::runtime::load_fp16(sk_device(), data, rows, cols));
    return static_cast<void*>(t);
}

void* sk_load_fp32_as_fp16(const float* data, uint32_t rows, uint32_t cols) {
    auto t = new sk::runtime::SKTensor(
        sk::runtime::load_fp32_as_fp16(sk_device(), data, rows, cols));
    return static_cast<void*>(t);
}
