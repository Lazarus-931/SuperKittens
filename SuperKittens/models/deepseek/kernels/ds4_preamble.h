//  ds4_preamble.h — preamble for the ds4-sourced kernels under deepseek/kernels/.

#ifndef SK_DS4_PREAMBLE_H
#define SK_DS4_PREAMBLE_H

#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

#define MAX(x, y)   ((x) > (y) ? (x) : (y))
#define MIN(x, y)   ((x) < (y) ? (x) : (y))
#define SWAP(x, y)  { auto tmp = (x); (x) = (y); (y) = tmp; }

#define QK8_0           32
#define QK5_0           32
#define N_SIMDWIDTH     32
#define N_R0_Q8_0        2
#define N_R0_Q5_0        2
#define N_SG_Q8_0        4
#define FC_MUL_MV      600
#define FC_MUL_MM      700
#define FC_BIN        1300

#define FOR_UNROLL(x)  _Pragma("clang loop unroll(full)") for (x)
#ifndef M_PI_F
#define M_PI_F         3.14159265358979323846f
#endif

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

// GGML Q5_0 block: 22 bytes / 32 weights (5.5 bit). qh is the packed 5th-bit
// plane (bit i = high bit of weight i); qs packs two 4-bit lows per byte.
// q[i]      = ((qs[i] & 0x0F) | ((qh >> i)        & 1) << 4) - 16   (i in 0..15)
// q[i+16]   = ((qs[i] >>   4) | ((qh >> (i+16))   & 1) << 4) - 16   (i in 0..15)
struct block_q5_0 {
    half    d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
};

struct ds4_metal_args_mul_mv {
    int   ne00;
    int   ne01;
    int   ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int   ne10;
    int   ne11;
    int   ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int   ne0;
    int   ne1;
    int   nr0;
    short r2;
    short r3;
};

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

#endif
