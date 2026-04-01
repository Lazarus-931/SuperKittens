//
//  Shader.Metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 3/31/26.
//

#include <metal_stdlib>
using namespace metal;

// Attention forward: softmax(Q * K^T / sqrt(d)) * V
kernel void attention_forward(
    device const float* Q    [[ buffer(0) ]],
    device const float* K    [[ buffer(1) ]],
    device const float* V    [[ buffer(2) ]],
    device float*       O    [[ buffer(3) ]],
    constant uint&      N    [[ buffer(4) ]],   // sequence length
    constant uint&      d    [[ buffer(5) ]],   // head dimension
    uint2 gid                [[ thread_position_in_grid ]]
) {
    uint row = gid.y;  // query index
    uint col = gid.x;  // output dim index

    if (row >= N || col >= d) return;

    float scale = 1.0f / sqrt((float)d);

    // Compute QK^T scores for this query row, all key rows
    // scores[j] = dot(Q[row], K[j]) * scale
    float scores[1024]; // max seq len — adjust if needed
    float max_score = -INFINITY;

    for (uint j = 0; j < N; j++) {
        float dot = 0.0f;
        for (uint k = 0; k < d; k++) {
            dot += Q[row * d + k] * K[j * d + k];
        }
        scores[j] = dot * scale;
        max_score = max(max_score, scores[j]);
    }

    // Softmax (numerically stable)
    float sum_exp = 0.0f;
    for (uint j = 0; j < N; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }
    for (uint j = 0; j < N; j++) {
        scores[j] /= sum_exp;
    }

    // Output: weighted sum of V rows
    float out = 0.0f;
    for (uint j = 0; j < N; j++) {
        out += scores[j] * V[j * d + col];
    }

    O[row * d + col] = out;
}
