//
//  attn.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/1/26.
//

#include <metal_stdlib>

using namespace metal;

kernel void attention_forward(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device float*       O       [[buffer(3)]],
    constant uint&      N       [[buffer(4)]],
    constant uint&      d       [[buffer(5)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint row = gid.y;  // sequence position
    uint col = gid.x;  // head dimension

    if (row >= N || col >= d) return;

    float scale = 1.0 / sqrt(float(d));

    // Compute attention scores for this row
    float max_score = -1e9;
    for (uint j = 0; j < N; j++) {
        float dot = 0.0;
        for (uint k = 0; k < d; k++) {
            dot += Q[row * d + k] * K[j * d + k];
        }
        dot *= scale;
        max_score = max(max_score, dot);
    }

    // Softmax numerator and denominator
    float sum_exp = 0.0;
    for (uint j = 0; j < N; j++) {
        float dot = 0.0;
        for (uint k = 0; k < d; k++) {
            dot += Q[row * d + k] * K[j * d + k];
        }
        sum_exp += exp(dot * scale - max_score);
    }

    // Weighted sum of V for this output element
    float result = 0.0;
    for (uint j = 0; j < N; j++) {
        float dot = 0.0;
        for (uint k = 0; k < d; k++) {
            dot += Q[row * d + k] * K[j * d + k];
        }
        float weight = exp(dot * scale - max_score) / sum_exp;
        result += weight * V[j * d + col];
    }

    O[row * d + col] = result;
}
