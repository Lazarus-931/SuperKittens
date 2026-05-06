//
//  feature.h — compile-time Apple GPU family detection
//
//  Boolean feature flags only.  Hardware budgets live in sk/compiler/budget.h.
//

#pragma once

namespace sk {
namespace feature {

#if defined(__METAL_FAMILY_APPLE_8__) || defined(__METAL_FAMILY_APPLE_9__) || defined(__METAL_FAMILY_APPLE_10__)
enum : int { has_bfloat = 1 };
#else
enum : int { has_bfloat = 0 };
#endif

#if defined(__METAL_FAMILY_APPLE_9__) || defined(__METAL_FAMILY_APPLE_10__)
enum : int { has_dynamic_cache = 1 };
#else
enum : int { has_dynamic_cache = 0 };
#endif

#if defined(__METAL_FAMILY_APPLE_10__)
enum : int { has_fp8 = 1 };
#else
enum : int { has_fp8 = 0 };
#endif

}  // namespace feature
}  // namespace sk
