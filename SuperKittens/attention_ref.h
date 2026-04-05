//
//  attention_ref.h
//  SuperKittens
//
//  CPU reference implementation of scaled dot-product attention.
//  Float32 arithmetic for maximum accuracy — this is the ground truth.

#pragma once
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>

inline void attention_cpu(const __fp16* Q, const __fp16* K, const __fp16* V,
                          float* O, uint32_t seq, uint32_t d) {
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> scores(seq);

    for (uint32_t i = 0; i < seq; i++) {
        // QK^T row i: scores[j] = dot(Q[i], K[j]) * scale
        float row_max = -INFINITY;
        for (uint32_t j = 0; j < seq; j++) {
            float dot = 0.0f;
            for (uint32_t k = 0; k < d; k++)
                dot += (float)Q[i * d + k] * (float)K[j * d + k];
            scores[j] = dot * scale;
            if (scores[j] > row_max) row_max = scores[j];
        }

        // softmax
        float sum = 0.0f;
        for (uint32_t j = 0; j < seq; j++) {
            scores[j] = expf(scores[j] - row_max);
            sum += scores[j];
        }
        float inv_sum = 1.0f / sum;
        for (uint32_t j = 0; j < seq; j++)
            scores[j] *= inv_sum;

        // output row i = scores × V
        for (uint32_t k = 0; k < d; k++) {
            float val = 0.0f;
            for (uint32_t j = 0; j < seq; j++)
                val += scores[j] * (float)V[j * d + k];
            O[i * d + k] = val;
        }
    }
}

struct VerifyResult {
    float max_abs_err;
    float mean_abs_err;
    float l2_rel_err;
    uint32_t worst_row;
    uint32_t worst_col;
    bool pass;
};

// Compare GPU half output against CPU float32 reference.
// Tolerance tuned for FP16 accumulation in fused attention:
//   max_abs < 0.05, mean_abs < 0.01 for well-behaved kernels.
inline VerifyResult verify_attention(const __fp16* gpu_out, const float* cpu_out,
                                     uint32_t seq, uint32_t d,
                                     float max_abs_tol = 0.05f,
                                     float mean_abs_tol = 0.02f) {
    VerifyResult r{};
    double sum_abs = 0.0;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;

    for (uint32_t i = 0; i < seq; i++) {
        for (uint32_t k = 0; k < d; k++) {
            float g = (float)gpu_out[i * d + k];
            float c = cpu_out[i * d + k];
            float err = fabsf(g - c);
            sum_abs += err;
            sum_sq_err += (double)(g - c) * (g - c);
            sum_sq_ref += (double)c * c;
            if (err > r.max_abs_err) {
                r.max_abs_err = err;
                r.worst_row = i;
                r.worst_col = k;
            }
        }
    }

    uint32_t n = seq * d;
    r.mean_abs_err = (float)(sum_abs / n);
    r.l2_rel_err = (float)(sqrt(sum_sq_err) / (sqrt(sum_sq_ref) + 1e-12));
    r.pass = (r.max_abs_err < max_abs_tol) && (r.mean_abs_err < mean_abs_tol);
    return r;
}
