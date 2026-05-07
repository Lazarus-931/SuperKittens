//
//  paws/dtype.h — dtype system with chip-forward-compatible design
//
//  Adding a new dtype (e.g. PAWS_F4 for M5) requires:
//    1. Add one value to the enum
//    2. Add one row to the dtype_info_table in dtype.c++
//    3. Update MAX_CHIP if a new chip generation exists
//  That's it. No other code changes needed.
//

#ifndef PAWS_DTYPE_H
#define PAWS_DTYPE_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ────────────────────────────────────────────────────────────────
//  DType enum — add new values BEFORE PAWS_COUNT
// ────────────────────────────────────────────────────────────────

typedef enum {
    PAWS_F16 = 0,     // M1+
    PAWS_F32 = 1,     // M1+
    PAWS_I8  = 2,     // M1+ (software dequant)
    PAWS_I4  = 3,     // M1+ (software dequant, 2 per byte packed)
    PAWS_F8  = 4,     // M3+ (hardware E4M3/E5M2)
    PAWS_BF16= 5,     // M3+
    PAWS_I4_GS64  = 6,  // M1+  int4 group_size=64
    PAWS_I4_GS128 = 7,  // M1+
    PAWS_I8_GS64  = 8,  // M1+
    // ── add new dtypes above this line ──
    PAWS_COUNT
} paws_dtype_t;

// ────────────────────────────────────────────────────────────────
//  Chip families — ordered: higher = newer = more features
// ────────────────────────────────────────────────────────────────

typedef enum {
    PAWS_CHIP_UNKNOWN = 0,
    PAWS_CHIP_M1 = 1,
    PAWS_CHIP_M2 = 2,
    PAWS_CHIP_M3 = 3,   // adds fp8, bf16
    PAWS_CHIP_M4 = 4,
    // ── add new chips above this line, increment MAX_CHIP ──
    PAWS_CHIP_MAX = PAWS_CHIP_M4
} paws_chip_t;

paws_chip_t paws_detect_chip(void);

// ────────────────────────────────────────────────────────────────
//  Dtype descriptor — one row per dtype, table-driven
// ────────────────────────────────────────────────────────────────

typedef struct {
    paws_dtype_t dtype;
    const char*  name;
    paws_chip_t  min_chip;          // minimum chip required (chip >= this → supported)
    uint32_t     bytes_x256;        // element size * 256 (0.5 → 128, 2.0 → 512)
    uint16_t     group_size;        // 0 = not quantized, 64/128 = group size
    uint8_t      is_quantized;      // has per-group or per-channel scales
    uint8_t      _pad;
} paws_dtype_info_t;

// All dtype metadata is in this table.  Everything else looks up from it.
const paws_dtype_info_t* paws_dtype_info(paws_dtype_t dt);

// ── queries (all table-driven, never need to change when adding dtype) ──

static inline uint32_t  paws_dtype_bytes_x256(paws_dtype_t dt) { return paws_dtype_info(dt)->bytes_x256; }
static inline uint16_t  paws_dtype_group_size(paws_dtype_t dt) { return paws_dtype_info(dt)->group_size; }
static inline int       paws_dtype_is_quantized(paws_dtype_t dt) { return paws_dtype_info(dt)->is_quantized; }
static inline const char* paws_dtype_name(paws_dtype_t dt)    { return paws_dtype_info(dt)->name; }

// ── compatibility (chip >= min_chip → supported) ──

static inline int paws_dtype_check(paws_dtype_t dt, paws_chip_t chip) {
    return (chip >= paws_dtype_info(dt)->min_chip && chip != PAWS_CHIP_UNKNOWN)
           ? 0 : -10 /* PAWS_ERR_UNSUPPORTED */;
}
int paws_dtype_assert(paws_dtype_t dt);

// ── mixed-dtype GEMM dispatch ──

// Returns kernel name suffix, or nullptr if unsupported.
// Built from the gemm_compat_table in dtype.c++
const char* paws_gemm_dtype_name(paws_dtype_t a, paws_dtype_t b);

// Combined chip + combo check.  0 = valid, negative = error.
int paws_gemm_dtype_validate(paws_dtype_t a, paws_dtype_t b, paws_chip_t chip);

#ifdef __cplusplus
}
#endif
#endif
