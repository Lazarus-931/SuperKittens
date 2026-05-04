//
//  meow.h
//  SuperKittens — unified master include
//
//  #include "meow.h" from .metal or .cpp — the right stuff gets pulled in.
//
//  Created by Alazar Manakelew on 4/1/26.
//

#ifndef SUPERKITTENS_MEOW_H
#define SUPERKITTENS_MEOW_H

#ifdef __METAL_VERSION__
// ═══════════════════════════════════════════════════════════════
//  GPU side (Metal Shading Language)
// ═══════════════════════════════════════════════════════════════

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── MMA primitives ───────────────────────────────────────────
#include "include/group/mma/base.h"
#include "include/group/mma/ops.h"

// ── Tiles ────────────────────────────────────────────────────
#include "kernels/tools/tile.h"

	// ── Ops ──────────────────────────────────────────────────────
	#include "include/ops/convert.h"
	#include "include/ops/memory.h"
	#include "include/ops/math.h"
	#include "include/ops/simdgroup.h"
	#include "include/ops/tools.h"

	// ── GEMM ─────────────────────────────────────────────────────
	#include "kernels/gemm/gemm_impl.h"
	
#else
// ═══════════════════════════════════════════════════════════════
//  Host side (C++)
// ═══════════════════════════════════════════════════════════════

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
	#include <QuartzCore/QuartzCore.hpp>

	#include "kernels/gemm/gemm_impl.h"
		#include "kernels/gemm/gemm_host.h"
	#include "kernels/mamba/mamba_impl.h"
	#include "kernels/mamba/mamba_host.h"

#endif // __METAL_VERSION__

using namespace meow;

#endif // SUPERKITTENS_MEOW_H
