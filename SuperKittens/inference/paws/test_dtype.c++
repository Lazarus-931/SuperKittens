#include "dtype.h"
#include <cstdio>
#include <cassert>
#include <cstring>

int main() {
    paws_chip_t chip = paws_detect_chip();
    printf("chip: M%d\n", (int)chip);

    // fp8×fp8 on M2 → should get "gemm_fp16" (software fallback), not "gemm_fp8"
    const char* k = paws_gemm_dtype_name(PAWS_F8, PAWS_F8);
    printf("fp8×fp8 on M%d → %s\n", (int)chip, k ? k : "nullptr");
    if (chip <= PAWS_CHIP_M2)
        assert(k && strcmp(k, "gemm_fp16") == 0);  // software upcast on M2
    else
        assert(k && strcmp(k, "gemm_fp8") == 0);   // hardware on M3+

    // f16×f16 always works
    k = paws_gemm_dtype_name(PAWS_F16, PAWS_F16);
    printf("f16×f16 → %s\n", k);
    assert(k && strcmp(k, "gemm_fp16") == 0);

    // bf16×f16 always upcasts to fp16 (no hardware bf16 on any current chip)
    k = paws_gemm_dtype_name(PAWS_BF16, PAWS_F16);
    printf("bf16×f16 → %s\n", k);
    assert(k && strcmp(k, "gemm_fp16") == 0);

    // i4×f16 on any chip
    k = paws_gemm_dtype_name(PAWS_I4_GS64, PAWS_F16);
    printf("i4_gs64×f16 → %s\n", k);
    assert(k && strcmp(k, "gemm_i4_f16") == 0);

    printf("PASS\n");
    return 0;
}
