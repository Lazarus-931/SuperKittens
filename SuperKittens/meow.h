//
//  meow.h
//  SuperKittens — master include header
//
//  Created by Alazar Manakelew on 4/1/26.
//

#ifndef SUPERKITTENS_MEOW_H
#define SUPERKITTENS_MEOW_H

// Kernel host-side headers
#include "kernels/gemm/fp16_m2/fp16_m2_gemm.cc"
#include "kernel/gemm/fp32_m2/fp32_m2_gemm.cc"
#include "kernels/attn/attn.h"


#endif // SUPERKITTENS_MEOW_H


#if defined(M1)
#include <>

#if defined(M2)
#include <>

#if defined(M3)
#include <>