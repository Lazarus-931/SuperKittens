//
//  fp16.m
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/2/26.
//

#include <metal_stdlib>
#include "types.h"

using namespace metal;


/////////
/// GEMM for <32, 32, 16, 2, 2>
////////

kernel
