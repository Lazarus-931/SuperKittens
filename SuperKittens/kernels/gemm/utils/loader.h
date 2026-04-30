//
//  loader.h
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/5/26.
//

#ifndef SUPERKITTENS_GEMM_LOADER_METAL
#define SUPERKITTENS_GEMM_LOADER_METAL

#include <metal_stdlib>
using namespace metal;

namespace meow {
namespace gemm {

METAL_FUNC half load_matrix_element(
    const device half* src,
    uint row,
    uint col,
    stride_t ld,
    uint32_t op)
{
    if (op == GEMM_OP_N) {
        return src[row * ld + col];
    }
    return src[col * ld + row];
}

METAL_FUNC half load_output_element(
    const device half* src,
    uint row,
    uint col,
    stride_t ld)
{
    return src[row * ld + col];
}

METAL_FUNC float load_bias_element(
    const device half* src,
    uint col)
{
    return float(src[col]);
}

METAL_FUNC void store_output_element(
    device half* dst,
    uint row,
    uint col,
    stride_t ld,
    half value)
{
    dst[row * ld + col] = value;
}

} // namespace gemm
} // namespace meow

#endif // SUPERKITTENS_GEMM_LOADER_METAL
