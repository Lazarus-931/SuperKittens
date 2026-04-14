//
//  mimo_fwd.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/13/26.
//


#include <metal_stdlib>


using namespace superkittens::mamba;


kernel static void Mamba_MIMO_Fwd(device const float a [[buffer(0)]])
