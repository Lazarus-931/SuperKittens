//
//  attn_causal.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/3/26.
//
//  Naive attention baseline — three separate kernels, not fused.
//  Step 1: scores = Q × K^T  (matmul kernel)
//  Step 2: scores = softmax(scores / sqrt(d))  (softmax kernel)
//  Step 3: O = scores × V  (matmul kernel)
//
//  Each step is a separate dispatch. Intermediates go through DRAM.
//  This is what we're trying to beat with the fused version.

#include <metal_stdlib>
using namespace metal;

// Step 1: Q(seq, d) × K^T(d, seq) → scores(seq, seq)
// K is (seq, d) stored row-major, read transposed
kernel void naive_qk(
    device const half* Q      [[buffer(0)]],
    device const half* K      [[buffer(1)]],
    device float* scores      [[buffer(2)]],
    constant uint& seq        [[buffer(3)]],
    constant uint& head_dim   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= seq || col >= seq) return;

    float acc = 0.0f;
    for (uint k = 0; k < head_dim; k++)
        acc += float(Q[row * head_dim + k]) * float(K[col * head_dim + k]);

    scores[row * seq + col] = acc;
}

// Step 2: row-wise softmax with scale
// scores(seq, seq) → probs(seq, seq)
kernel void naive_softmax(
    device float* scores      [[buffer(0)]],
    device float* probs       [[buffer(1)]],
    constant uint& seq        [[buffer(2)]],
    constant uint& head_dim   [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= seq) return;

    float rsqrt_d = 1.0f / sqrt(float(head_dim));

    // find max for numerical stability
    float row_max = -INFINITY;
    for (uint j = 0; j < seq; j++)
        row_max = max(row_max, scores[row * seq + j] * rsqrt_d);

    // exp and sum
    float sum = 0.0f;
    for (uint j = 0; j < seq; j++) {
        float val = exp(scores[row * seq + j] * rsqrt_d - row_max);
        probs[row * seq + j] = val;
        sum += val;
    }

    // normalize
    float inv_sum = 1.0f / sum;
    for (uint j = 0; j < seq; j++)
        probs[row * seq + j] *= inv_sum;
}

// Step 3: probs(seq, seq) × V(seq, d) → O(seq, d)
kernel void naive_pv(
    device const float* probs [[buffer(0)]],
    device const half* V      [[buffer(1)]],
    device half* O            [[buffer(2)]],
    constant uint& seq        [[buffer(3)]],
    constant uint& head_dim   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= seq || col >= head_dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < seq; k++)
        acc += probs[row * seq + k] * float(V[k * head_dim + col]);

    O[row * head_dim + col] = half(acc);
}
