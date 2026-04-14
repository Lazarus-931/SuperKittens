//
//  meow.metal
//  SuperKittens — single Metal include point
//
//  #include "meow.metal" from any .metal kernel to get everything.
//

#ifndef SUPERKITTENS_MEOW_METAL
#define SUPERKITTENS_MEOW_METAL

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── MMA primitives ───────────────────────────────────────────
#include "include/group/mma/base.h"
#include "include/group/mma/ops.h"

// ── Tile / Frag (register-resident 8×8) ─────────────────────
#include "kernels/tools/tile.h"

// ── Group ops ────────────────────────────────────────────────
#include "include/ops/convert.h"
#include "include/ops/memory.h"
#include "include/ops/math.h"
#include "include/ops/simdgroup.h"

#endif // SUPERKITTENS_MEOW_METAL
