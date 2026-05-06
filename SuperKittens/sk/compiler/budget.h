//
//  budget.h — per-family hardware budgets (enums, Metal-safe)
//

#pragma once

namespace sk {
namespace compiler {

enum class GpuFamily : int {
    Apple7  = 7,   // M1
    Apple8  = 8,   // M2 — adds bfloat
    Apple9  = 9,   // M3/M4 — adds dynamic cache
    Apple10 = 10   // M5 — adds fp8
};

template<GpuFamily F> struct FamilyBudget;

template<> struct FamilyBudget<GpuFamily::Apple7> {
    enum : int { regs_per_lane = 96, tg_mem = 32768, max_simds = 32 };
    enum : int { has_bfloat = 0, has_fp8 = 0, has_dynamic_cache = 0 };
};
template<> struct FamilyBudget<GpuFamily::Apple8> {
    enum : int { regs_per_lane = 112, tg_mem = 32768, max_simds = 32 };
    enum : int { has_bfloat = 1, has_fp8 = 0, has_dynamic_cache = 0 };
};
template<> struct FamilyBudget<GpuFamily::Apple9> {
    enum : int { regs_per_lane = 128, tg_mem = 32768, max_simds = 32 };
    enum : int { has_bfloat = 1, has_fp8 = 0, has_dynamic_cache = 1 };
};
template<> struct FamilyBudget<GpuFamily::Apple10> {
    enum : int { regs_per_lane = 128, tg_mem = 32768, max_simds = 32 };
    enum : int { has_bfloat = 1, has_fp8 = 1, has_dynamic_cache = 1 };
};

template<typename TgDecomp, GpuFamily Family>
struct ValidateDecomposition {
    using Tg = TgDecomp;
    using Budget = FamilyBudget<Family>;
    static_assert(Tg::n_simds <= Budget::max_simds, "too many SIMD groups");
    // Register budget is a soft limit — Metal compiler spills transparently.
    // We document the pressure but don't assert. High pressure = slower.
    enum : int {
        simds  = Tg::n_simds,
        regs   = Tg::Simd::template regs_used<float>(),
        budget = Budget::regs_per_lane,
        spill  = (regs > budget) ? 1 : 0  // 1 = Metal compiler will spill
    };
};

} // namespace compiler
} // namespace sk
