//
//  paws/dtype.c++ — single-table dtype system
//  To add a dtype: add one row to kDtypeTable + one row to kGemmCompat (if needed).
//

#include "dtype.h"
#include <Metal/Metal.hpp>
#include <cstdio>

// ═══════════════════════════════════════════════════════════════
//  Master dtype table — the ONLY place dtype properties are defined
// ═══════════════════════════════════════════════════════════════

static const paws_dtype_info_t kDtypeTable[] = {
    { PAWS_F16,      "f16",     PAWS_CHIP_M1, 512, 0,   0, 0 },
    { PAWS_F32,      "f32",     PAWS_CHIP_M1, 1024,0,   0, 0 },
    { PAWS_I8,       "i8",      PAWS_CHIP_M1, 256, 0,   1, 0 },
    { PAWS_I4,       "i4",      PAWS_CHIP_M1, 128, 0,   1, 0 },
    { PAWS_F8,       "f8",      PAWS_CHIP_M3, 256, 0,   0, 0 },  // M3+
    { PAWS_BF16,     "bf16",    PAWS_CHIP_M3, 512, 0,   0, 0 },  // M3+
    { PAWS_I4_GS64,  "i4_gs64", PAWS_CHIP_M1, 128, 64,  1, 0 },
    { PAWS_I4_GS128, "i4_gs128",PAWS_CHIP_M1, 128, 128, 1, 0 },
    { PAWS_I8_GS64,  "i8_gs64", PAWS_CHIP_M1, 256, 64,  1, 0 },
    // ── add new dtypes here ──
};

static_assert(sizeof(kDtypeTable)/sizeof(kDtypeTable[0]) == PAWS_COUNT,
              "dtype table count must match PAWS_COUNT");

const paws_dtype_info_t* paws_dtype_info(paws_dtype_t dt) {
    if (dt < PAWS_COUNT) return &kDtypeTable[dt];
    static const paws_dtype_info_t unknown = { PAWS_COUNT, "?", PAWS_CHIP_UNKNOWN, 0, 0, 0, 0 };
    return &unknown;
}

// ═══════════════════════════════════════════════════════════════
//  Chip detection — ordered check, newest first
// ═══════════════════════════════════════════════════════════════

paws_chip_t paws_detect_chip(void) {
    auto* dev = MTL::CreateSystemDefaultDevice();
    if (!dev) return PAWS_CHIP_UNKNOWN;
    if (dev->supportsFamily(MTL::GPUFamilyApple10)) return PAWS_CHIP_M4;
    if (dev->supportsFamily(MTL::GPUFamilyApple9))  return PAWS_CHIP_M3;
    if (dev->supportsFamily(MTL::GPUFamilyApple8))  return PAWS_CHIP_M2;
    if (dev->supportsFamily(MTL::GPUFamilyApple7))  return PAWS_CHIP_M1;
    return PAWS_CHIP_UNKNOWN;
}

// ═══════════════════════════════════════════════════════════════
//  Convenience
// ═══════════════════════════════════════════════════════════════

int paws_dtype_assert(paws_dtype_t dt) {
    paws_chip_t chip = paws_detect_chip();
    if (paws_dtype_check(dt, chip) != 0) {
        fprintf(stderr, "paws: dtype %s requires M%d+ (running on M%d)\n",
                paws_dtype_name(dt),
                (int)paws_dtype_info(dt)->min_chip,
                (int)chip);
        return -10;
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════
//  GEMM mixed-dtype dispatch table
//  Add rows here for new dtype combinations.
// ═══════════════════════════════════════════════════════════════

typedef struct {
    paws_dtype_t a, b;
    paws_chip_t  min_chip;   // chip must be >= this for this entry to match
    const char*  kernel;
} gemm_compat_entry_t;

// Table is priority-ordered: first match wins.
// Put hardware-accelerated entries FIRST (higher min_chip), software fallbacks AFTER.
static const gemm_compat_entry_t kGemmCompat[] = {
    // ═══ same-dtype — hardware paths ═══
    { PAWS_F16, PAWS_F16, PAWS_CHIP_M1, "gemm_fp16" },
    { PAWS_F8,  PAWS_F8,  PAWS_CHIP_M3, "gemm_fp8"  },   // M3+ hardware fp8

    // ═══ weight-only int quantization ═══
    { PAWS_I4,       PAWS_F16, PAWS_CHIP_M1, "gemm_i4_f16"  },
    { PAWS_I4_GS64,  PAWS_F16, PAWS_CHIP_M1, "gemm_i4_f16"  },
    { PAWS_I4_GS128, PAWS_F16, PAWS_CHIP_M1, "gemm_i4_f16"  },
    { PAWS_I8,       PAWS_F16, PAWS_CHIP_M1, "gemm_i8_f16"  },
    { PAWS_I8_GS64,  PAWS_F16, PAWS_CHIP_M1, "gemm_i8_f16"  },
    { PAWS_F16, PAWS_I4_GS64,  PAWS_CHIP_M1, "gemm_f16_i4"  },
    { PAWS_F16, PAWS_I8_GS64,  PAWS_CHIP_M1, "gemm_f16_i8"  },

    // ═══ fp32 — hardware ═══
    { PAWS_F32, PAWS_F32, PAWS_CHIP_M1, "gemm_fp32" },
    { PAWS_F32, PAWS_F16, PAWS_CHIP_M1, "gemm_fp32" },
    { PAWS_F16, PAWS_F32, PAWS_CHIP_M1, "gemm_fp32" },

    // ═══ upcast/fallback paths — lower priority ═══
    // fp8 on M2 (no hardware fp8) — upcast to fp16
    { PAWS_F8,  PAWS_F8,  PAWS_CHIP_M1, "gemm_fp16" },  // software: upcast both to fp16
    { PAWS_F8,  PAWS_F16, PAWS_CHIP_M1, "gemm_fp16" },  // upcast A
    { PAWS_F16, PAWS_F8,  PAWS_CHIP_M1, "gemm_fp16" },  // upcast B

    // bf16 — always upcast to fp16 (no hardware bf16 matmul on any current chip)
    { PAWS_BF16, PAWS_F16,  PAWS_CHIP_M1, "gemm_fp16" },
    { PAWS_F16,  PAWS_BF16, PAWS_CHIP_M1, "gemm_fp16" },

    // ── add new combos here (hardware paths FIRST, then fallbacks) ──
};
static const uint32_t kGemmCompatCount = sizeof(kGemmCompat)/sizeof(kGemmCompat[0]);

// Chip-aware dispatch: returns the BEST kernel for this chip.
// First match where chip >= entry.min_chip wins.
const char* paws_gemm_dtype_name(paws_dtype_t a, paws_dtype_t b) {
    for (uint32_t i = 0; i < kGemmCompatCount; i++) {
        if (kGemmCompat[i].a == a && kGemmCompat[i].b == b) {
            paws_chip_t chip = paws_detect_chip();
            if (chip >= kGemmCompat[i].min_chip)
                return kGemmCompat[i].kernel;
            // else: this entry requires a newer chip. Keep scanning for fallback.
        }
    }
    return nullptr;
}

int paws_gemm_dtype_validate(paws_dtype_t a, paws_dtype_t b, paws_chip_t chip) {
    if (paws_dtype_check(a, chip) != 0) return -10;
    if (paws_dtype_check(b, chip) != 0) return -10;
    if (!paws_gemm_dtype_name(a, b))     return -10;
    return 0;
}
