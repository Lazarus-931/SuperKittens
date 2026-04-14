//
//  Untitled.swift
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/13/26.
//

#include 


kernel void rotary_qk_best_kernel(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *angleState [[buffer(2)]],
    device const float *angleProj [[buffer(3)]],
    device const float *dt [[buffer(4)]],
    device const float *biasQ [[buffer(5)]],
    device const float *biasK [[buffer(6)]],
    device float *outQ [[buffer(7)]],
    device float *outK [[buffer(8)]],
    device float *outAngleState [[buffer(9)]],
    constant RotaryQKArgs &args [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint halfDim = args.headdim >> 1;
    uint totalPairs = args.mimo_dim * halfDim;
    uint item = gid.x;
    uint b = gid.y;
    uint h = gid.z;
    if (item >= totalPairs || b >= args.batch || h >= args.nheads) return;

    uint m = item / halfDim;
    uint p = item - m * halfDim;
    uint rotaryPairs = args.rotary_dim >> 1;
    uint angleBase = ((b * args.nheads + h) * rotaryPairs);
    float signedSin = (args.conjugate != 0) ? -1.0f : 1.0f;
    float c = 1.0f;
    float s = 0.0f;
    float angle = 0.0f;
    if (p < rotaryPairs) {
        float proj = angleProj[angleBase + p];
        proj = 2.0f / (1.0f + exp(-2.0f * proj)) - 1.0f;
        angle = angleState[angleBase + p] + proj * dt[b * args.nheads + h] * 3.141592653589793f;
        c = cos(angle);
        s = sin(angle) * signedSin;
        if (m == 0) outAngleState[angleBase + p] = angle;
    }

    uint base = (((b * args.mimo_dim + m) * args.nheads + h) * args.headdim);
    uint biasBase = ((m * args.nheads + h) * args.headdim);
    uint d0 = (args.rotate_pairwise != 0) ? (p << 1) : p;
    uint d1 = (args.rotate_pairwise != 0) ? (d0 + 1) : (p + halfDim);
    float q0 = q[base + d0];
    float q1 = q[base + d1];
    float k0 = k[base + d0];
    float k1 = k[base + d1];
    if (args.has_bias_q != 0) {
        q0 += biasQ[biasBase + d0];
        q1 += biasQ[biasBase + d1];
    }
    if (args.has_bias_k != 0) {
        k0 += biasK[biasBase + d0];
        k1 += biasK[biasBase + d1];
    }
    outQ[base + d0] = q0 * c - q1 * s;
    outQ[base + d1] = q0 * s + q1 * c;
    outK[base + d0] = k0 * c - k1 * s;
    outK[base + d1] = k0 * s + k1 * c;
}
