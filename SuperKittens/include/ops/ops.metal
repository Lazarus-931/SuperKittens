//
//  ops.metal
//  SuperKittens
//
//  Metal-side ops aggregator and namespace layout.
//

#pragma once

#include <metal_stdlib>
using namespace metal;

namespace superkittens {
namespace ops {


namespace simdgroup {}
namespace convert {}
namespace threadgroup {}
namespace memory {}
namespace math {}
namespace tiles {}

} // namespace ops
} // namespace superkittens

#include "convert.h"
#include "simdgroup.h"
#include "math.h"
#include "memory.h"
