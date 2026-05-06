//
//  test_layout.metal — verify layout.h decomposition + budget validation
//

#include "../../../sk/compiler/layout.h"
#include "../../../sk/compiler/budget.h"

using namespace sk::compiler;

// ── Test: Validate GEMM decomposition against M1 budget ──
// GemmDecomp_128: 2×2 SIMD grid, MR=2, MC=8
//   n_simds = 4, total_rows = 2*16 = 32, total_cols = 2*64 = 128
//   Register pressure: MR*MC*8 = 128 per lane (M1 has 96, compiler spills)
//   The spill enum = 1 documents this.  Not an error — just slower on M1.

using V1 = ValidateDecomposition<GemmDecomp_128, GpuFamily::Apple7>;
static_assert(V1::simds == 4, "128 threads = 4 SIMD groups");
// V1::spill == 1 → compiler spills on M1. On M3+ (128 regs), spill == 0.

// ── Kernel: minimal kernel using layout params ──
using Params = GemmParams<GemmDecomp_128, 32>;

[[host_name("test_layout_params")]]
[[kernel, max_total_threads_per_threadgroup(Params::n_threads)]]
void test_layout_params(device half* dst [[buffer(0)]], uint lid [[thread_index_in_threadgroup]])
{
    // Just verify the compile-time constants are accessible in kernel code
    if (lid == 0) {
        dst[0] = half(float(Params::BM));
        dst[1] = half(float(Params::BN));
        dst[2] = half(float(Params::BK));
        dst[3] = half(float(Params::n_threads));
    }
}
