//
//  mamba3_siso_bwd_smoke_test.cpp
//  SuperKittens
//

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

struct Mamba3FwdArgsHost {
    uint32_t batch;
    uint32_t nheads;
    uint32_t seq_len;
    uint32_t n_chunks;
};

struct Mamba3BwdArgsHost {
    uint32_t batch;
    uint32_t nheads;
    uint32_t seq_len;
    uint32_t n_chunks;
};

struct Mamba3BwdSplitArgsHost {
    uint32_t batch;
    uint32_t nheads;
    uint32_t nheads_qk;
    uint32_t seq_len;
    uint32_t n_chunks;
};

struct ErrStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float l2_rel = 0.0f;
};

struct ForwardCache {
    std::vector<float> out;
    std::vector<float> states;  // [B,H,n_chunks,DQ,DV]
};

struct BackwardGrads {
    std::vector<float> dQ;
    std::vector<float> dK;
    std::vector<float> dV;
    std::vector<float> dA;
    std::vector<float> dB;
    std::vector<float> dAngle;
};

static size_t qk_index(int b, int h, int t, int d, int H, int L, int D) {
    return (((size_t)b * H + h) * L + t) * D + d;
}

static size_t v_index(int b, int h, int t, int d, int H, int L, int D) {
    return (((size_t)b * H + h) * L + t) * D + d;
}

static size_t scalar_index(int b, int h, int t, int H, int L) {
    return ((size_t)b * H + h) * L + t;
}

static size_t angle_index(int b, int h, int t, int d, int H, int L, int halfD) {
    return ((((size_t)b * H + h) * L) + t) * halfD + d;
}

static size_t state_index(int b, int h, int c, int i, int j,
                          int H, int n_chunks, int DQ, int DV) {
    return (((((size_t)b * H + h) * n_chunks + c) * DQ) + i) * DV + j;
}

static void fill_half(std::vector<__fp16>& v, std::mt19937& rng, float scale) {
    std::normal_distribution<float> n(0.0f, 1.0f);
    for (auto& x : v) x = __fp16(n(rng) * scale);
}

static void fill_float(std::vector<float>& v, std::mt19937& rng, float mean, float stddev) {
    std::normal_distribution<float> n(mean, stddev);
    for (auto& x : v) x = n(rng);
}

static ErrStats compare_half_float(const std::vector<__fp16>& gpu,
                                   const std::vector<float>& ref) {
    ErrStats s;
    double sum_abs = 0.0;
    double sum_num = 0.0;
    double sum_den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = float(gpu[i]);
        const float r = ref[i];
        const float d = std::fabs(g - r);
        s.max_abs = std::max(s.max_abs, d);
        sum_abs += d;
        sum_num += double(d) * double(d);
        sum_den += double(r) * double(r);
    }
    s.mean_abs = float(sum_abs / ref.size());
    s.l2_rel = float(std::sqrt(sum_num) / (std::sqrt(sum_den) + 1e-12));
    return s;
}

static ErrStats compare_float(const std::vector<float>& gpu,
                              const std::vector<float>& ref) {
    ErrStats s;
    double sum_abs = 0.0;
    double sum_num = 0.0;
    double sum_den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float d = std::fabs(gpu[i] - ref[i]);
        s.max_abs = std::max(s.max_abs, d);
        sum_abs += d;
        sum_num += double(d) * double(d);
        sum_den += double(ref[i]) * double(ref[i]);
    }
    s.mean_abs = float(sum_abs / ref.size());
    s.l2_rel = float(std::sqrt(sum_num) / (std::sqrt(sum_den) + 1e-12));
    return s;
}

static inline void rotate_forward(
    float q0, float q1, float k0, float k1, float theta,
    float& q0_rot, float& q1_rot, float& k0_rot, float& k1_rot) {
    const float c = std::cos(theta);
    const float s = std::sin(theta);
    q0_rot = q0 * c - q1 * s;
    q1_rot = q0 * s + q1 * c;
    k0_rot = k0 * c - k1 * s;
    k1_rot = k0 * s + k1 * c;
}

static constexpr float kPi = 3.14159265358979323846f;

static inline void rotate_backward(
    float q0, float q1, float k0, float k1, float theta,
    float dq0_rot, float dq1_rot, float dk0_rot, float dk1_rot,
    float& dq0, float& dq1, float& dk0, float& dk1, float& dtheta) {
    const float c = std::cos(theta);
    const float s = std::sin(theta);
    dq0 = dq0_rot * c + dq1_rot * s;
    dq1 = -dq0_rot * s + dq1_rot * c;
    dk0 = dk0_rot * c + dk1_rot * s;
    dk1 = -dk0_rot * s + dk1_rot * c;
    dtheta =
        dq0_rot * (-q0 * s - q1 * c) +
        dq1_rot * ( q0 * c - q1 * s) +
        dk0_rot * (-k0 * s - k1 * c) +
        dk1_rot * ( k0 * c - k1 * s);
}

static ForwardCache cpu_forward(
    const std::vector<__fp16>& Q,
    const std::vector<__fp16>& K,
    const std::vector<__fp16>& V,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<__fp16>& angle,
    int batch, int heads, int length, int DQ, int DV, int chunk) {
    const int halfD = DQ / 2;
    const int n_chunks = (length + chunk - 1) / chunk;
    ForwardCache cache;
    cache.out.assign((size_t)batch * heads * length * DV, 0.0f);
    cache.states.assign((size_t)batch * heads * n_chunks * DQ * DV, 0.0f);

    std::vector<float> state((size_t)DQ * DV, 0.0f);
    std::vector<float> q_rot((size_t)chunk * DQ, 0.0f);
    std::vector<float> k_rot((size_t)chunk * DQ, 0.0f);
    std::vector<float> a_cs(chunk, 0.0f);
    std::vector<float> b_scale(chunk, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            std::fill(state.begin(), state.end(), 0.0f);
            for (int c = 0; c < n_chunks; ++c) {
                const int chunk_start = c * chunk;
                const int chunk_len = std::min(chunk, length - chunk_start);
                float running = 0.0f;
                for (int t = 0; t < chunk_len; ++t) {
                    const int seq = chunk_start + t;
                    running += A[scalar_index(b, h, seq, heads, length)];
                    a_cs[t] = running;
                    b_scale[t] = 1.0f + B[scalar_index(b, h, seq, heads, length)] * std::exp(-a_cs[t]);
                }

                for (int t = 0; t < chunk_len; ++t) {
                    const int seq = chunk_start + t;
                    for (int i = 0; i < halfD; ++i) {
                        const float theta = a_cs[t] * float(angle[angle_index(b, h, seq, i, heads, length, halfD)]) * kPi;
                        float q0r, q1r, k0r, k1r;
                        rotate_forward(
                            float(Q[qk_index(b, h, seq, i, heads, length, DQ)]),
                            float(Q[qk_index(b, h, seq, i + halfD, heads, length, DQ)]),
                            float(K[qk_index(b, h, seq, i, heads, length, DQ)]),
                            float(K[qk_index(b, h, seq, i + halfD, heads, length, DQ)]),
                            theta, q0r, q1r, k0r, k1r);
                        q_rot[(size_t)t * DQ + i] = q0r;
                        q_rot[(size_t)t * DQ + i + halfD] = q1r;
                        k_rot[(size_t)t * DQ + i] = k0r;
                        k_rot[(size_t)t * DQ + i + halfD] = k1r;
                    }
                }

                const float chunk_decay = std::exp(a_cs[chunk_len - 1]) * b_scale[chunk_len - 1];
                for (size_t i = 0; i < state.size(); ++i) state[i] *= chunk_decay;
                for (int t = 0; t < chunk_len; ++t) {
                    const int seq = chunk_start + t;
                    for (int i = 0; i < DQ; ++i) {
                        const float kval = k_rot[(size_t)t * DQ + i];
                        for (int j = 0; j < DV; ++j) {
                            state[(size_t)i * DV + j] += kval * float(V[v_index(b, h, seq, j, heads, length, DV)]);
                        }
                    }
                }

                for (int i = 0; i < DQ; ++i) {
                    for (int j = 0; j < DV; ++j) {
                        cache.states[state_index(b, h, c, i, j, heads, n_chunks, DQ, DV)] = state[(size_t)i * DV + j];
                    }
                }

                for (int r = 0; r < chunk_len; ++r) {
                    const int seq_r = chunk_start + r;
                    for (int j = 0; j < DV; ++j) {
                        float y = 0.0f;
                        for (int cc = 0; cc <= r; ++cc) {
                            float score = 0.0f;
                            for (int i = 0; i < DQ; ++i) {
                                score += q_rot[(size_t)r * DQ + i] * k_rot[(size_t)cc * DQ + i];
                            }
                            y += score * std::exp(a_cs[r] - a_cs[cc]) * float(V[v_index(b, h, chunk_start + cc, j, heads, length, DV)]);
                        }
                        const float q_decay = std::exp(a_cs[r]) * b_scale[r];
                        float inter = 0.0f;
                        for (int i = 0; i < DQ; ++i) {
                            inter += q_rot[(size_t)r * DQ + i] * state[(size_t)i * DV + j];
                        }
                        cache.out[v_index(b, h, seq_r, j, heads, length, DV)] = y + q_decay * inter;
                    }
                }
            }
        }
    }

    return cache;
}

static BackwardGrads cpu_backward(
    const std::vector<__fp16>& Q,
    const std::vector<__fp16>& K,
    const std::vector<__fp16>& V,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<__fp16>& angle,
    const std::vector<float>& saved_states,
    const std::vector<__fp16>& dO,
    int batch, int heads, int length, int DQ, int DV, int chunk) {
    const int halfD = DQ / 2;
    const int n_chunks = (length + chunk - 1) / chunk;
    BackwardGrads g;
    g.dQ.assign((size_t)batch * heads * length * DQ, 0.0f);
    g.dK.assign((size_t)batch * heads * length * DQ, 0.0f);
    g.dV.assign((size_t)batch * heads * length * DV, 0.0f);
    g.dA.assign((size_t)batch * heads * length, 0.0f);
    g.dB.assign((size_t)batch * heads * length, 0.0f);
    g.dAngle.assign((size_t)batch * heads * length * halfD, 0.0f);

    std::vector<float> state_grad((size_t)DQ * DV, 0.0f);
    std::vector<float> q_rot((size_t)chunk * DQ, 0.0f);
    std::vector<float> k_rot((size_t)chunk * DQ, 0.0f);
    std::vector<float> a_cs(chunk, 0.0f);
    std::vector<float> b_scale(chunk, 0.0f);
    std::vector<float> dq_rot((size_t)chunk * DQ, 0.0f);
    std::vector<float> dk_rot((size_t)chunk * DQ, 0.0f);
    std::vector<float> dv_acc((size_t)chunk * DV, 0.0f);
    std::vector<float> d_a_cs(chunk, 0.0f);
    std::vector<float> d_b_scale(chunk, 0.0f);
    std::vector<float> d_q_decay(chunk, 0.0f);
    std::vector<float> next_state_grad((size_t)DQ * DV, 0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            std::fill(state_grad.begin(), state_grad.end(), 0.0f);
            for (int c = n_chunks - 1; c >= 0; --c) {
                const int chunk_start = c * chunk;
                const int chunk_len = std::min(chunk, length - chunk_start);

                float running = 0.0f;
                for (int t = 0; t < chunk_len; ++t) {
                    const int seq = chunk_start + t;
                    running += A[scalar_index(b, h, seq, heads, length)];
                    a_cs[t] = running;
                    b_scale[t] = 1.0f + B[scalar_index(b, h, seq, heads, length)] * std::exp(-a_cs[t]);
                    d_a_cs[t] = 0.0f;
                    d_b_scale[t] = 0.0f;
                    d_q_decay[t] = 0.0f;
                }
                std::fill(dq_rot.begin(), dq_rot.begin() + (size_t)chunk_len * DQ, 0.0f);
                std::fill(dk_rot.begin(), dk_rot.begin() + (size_t)chunk_len * DQ, 0.0f);
                std::fill(dv_acc.begin(), dv_acc.begin() + (size_t)chunk_len * DV, 0.0f);

                for (int t = 0; t < chunk_len; ++t) {
                    const int seq = chunk_start + t;
                    for (int i = 0; i < halfD; ++i) {
                        const float theta = a_cs[t] * float(angle[angle_index(b, h, seq, i, heads, length, halfD)]) * kPi;
                        float q0r, q1r, k0r, k1r;
                        rotate_forward(
                            float(Q[qk_index(b, h, seq, i, heads, length, DQ)]),
                            float(Q[qk_index(b, h, seq, i + halfD, heads, length, DQ)]),
                            float(K[qk_index(b, h, seq, i, heads, length, DQ)]),
                            float(K[qk_index(b, h, seq, i + halfD, heads, length, DQ)]),
                            theta, q0r, q1r, k0r, k1r);
                        q_rot[(size_t)t * DQ + i] = q0r;
                        q_rot[(size_t)t * DQ + i + halfD] = q1r;
                        k_rot[(size_t)t * DQ + i] = k0r;
                        k_rot[(size_t)t * DQ + i + halfD] = k1r;
                    }
                }

                const float* s_cur = &saved_states[state_index(b, h, c, 0, 0, heads, n_chunks, DQ, DV)];
                const float* s_prev = (c > 0) ? &saved_states[state_index(b, h, c - 1, 0, 0, heads, n_chunks, DQ, DV)] : nullptr;

                for (int r = 0; r < chunk_len; ++r) {
                    const int seq_r = chunk_start + r;
                    const float q_decay = std::exp(a_cs[r]) * b_scale[r];
                    for (int j = 0; j < DV; ++j) {
                        float inter = 0.0f;
                        for (int i = 0; i < DQ; ++i) inter += q_rot[(size_t)r * DQ + i] * s_cur[(size_t)i * DV + j];
                        d_q_decay[r] += float(dO[v_index(b, h, seq_r, j, heads, length, DV)]) * inter;
                    }
                    for (int i = 0; i < DQ; ++i) {
                        float s_grad = 0.0f;
                        for (int j = 0; j < DV; ++j) {
                            const float do_val = float(dO[v_index(b, h, seq_r, j, heads, length, DV)]);
                            s_grad += do_val * s_cur[(size_t)i * DV + j];
                            state_grad[(size_t)i * DV + j] += q_decay * q_rot[(size_t)r * DQ + i] * do_val;
                        }
                        dq_rot[(size_t)r * DQ + i] += q_decay * s_grad;
                    }
                }

                for (int r = 0; r < chunk_len; ++r) {
                    const int seq_r = chunk_start + r;
                    for (int cc = 0; cc <= r; ++cc) {
                        const int seq_c = chunk_start + cc;
                        float score = 0.0f;
                        for (int i = 0; i < DQ; ++i) score += q_rot[(size_t)r * DQ + i] * k_rot[(size_t)cc * DQ + i];
                        const float decay = std::exp(a_cs[r] - a_cs[cc]);
                        float alpha = 0.0f;
                        for (int j = 0; j < DV; ++j) {
                            const float do_val = float(dO[v_index(b, h, seq_r, j, heads, length, DV)]);
                            const float v_val = float(V[v_index(b, h, seq_c, j, heads, length, DV)]);
                            alpha += do_val * v_val;
                            dv_acc[(size_t)cc * DV + j] += do_val * (decay * score);
                        }
                        alpha *= decay;
                        for (int i = 0; i < DQ; ++i) {
                            dq_rot[(size_t)r * DQ + i] += alpha * k_rot[(size_t)cc * DQ + i];
                            dk_rot[(size_t)cc * DQ + i] += alpha * q_rot[(size_t)r * DQ + i];
                        }
                        const float pair_term = alpha * score;
                        d_a_cs[r] += pair_term;
                        d_a_cs[cc] -= pair_term;
                    }
                }

                float d_decay = 0.0f;
                for (int i = 0; i < DQ; ++i) {
                    for (int j = 0; j < DV; ++j) {
                        const float prev = s_prev ? s_prev[(size_t)i * DV + j] : 0.0f;
                        d_decay += state_grad[(size_t)i * DV + j] * prev;
                    }
                }
                const float chunk_decay = std::exp(a_cs[chunk_len - 1]) * b_scale[chunk_len - 1];
                for (int i = 0; i < DQ * DV; ++i) next_state_grad[i] = state_grad[i] * chunk_decay;

                for (int t = 0; t < chunk_len; ++t) {
                    const int seq = chunk_start + t;
                    for (int i = 0; i < DQ; ++i) {
                        float acc = 0.0f;
                        for (int j = 0; j < DV; ++j) {
                            acc += state_grad[(size_t)i * DV + j] * float(V[v_index(b, h, seq, j, heads, length, DV)]);
                            dv_acc[(size_t)t * DV + j] += k_rot[(size_t)t * DQ + i] * state_grad[(size_t)i * DV + j];
                        }
                        dk_rot[(size_t)t * DQ + i] += acc;
                    }
                }

                d_a_cs[chunk_len - 1] += d_decay * chunk_decay;
                d_b_scale[chunk_len - 1] += d_decay * std::exp(a_cs[chunk_len - 1]);
                for (int t = 0; t < chunk_len; ++t) {
                    const float q_decay = std::exp(a_cs[t]) * b_scale[t];
                    d_a_cs[t] += d_q_decay[t] * q_decay;
                    d_b_scale[t] += d_q_decay[t] * std::exp(a_cs[t]);
                }
                for (int t = 0; t < chunk_len; ++t) {
                    const int seq = chunk_start + t;
                    const float b_raw = B[scalar_index(b, h, seq, heads, length)];
                    const float exp_neg = std::exp(-a_cs[t]);
                    g.dB[scalar_index(b, h, seq, heads, length)] += d_b_scale[t] * exp_neg;
                    d_a_cs[t] += d_b_scale[t] * (-b_raw * exp_neg);
                }

                for (int t = 0; t < chunk_len; ++t) {
                    const int seq = chunk_start + t;
                    for (int i = 0; i < halfD; ++i) {
                        const float q0 = float(Q[qk_index(b, h, seq, i, heads, length, DQ)]);
                        const float q1 = float(Q[qk_index(b, h, seq, i + halfD, heads, length, DQ)]);
                        const float k0 = float(K[qk_index(b, h, seq, i, heads, length, DQ)]);
                        const float k1 = float(K[qk_index(b, h, seq, i + halfD, heads, length, DQ)]);
                        const float raw_ang = float(angle[angle_index(b, h, seq, i, heads, length, halfD)]);
                        const float theta = a_cs[t] * raw_ang * kPi;
                        float dq0, dq1, dk0, dk1, dtheta;
                        rotate_backward(
                            q0, q1, k0, k1, theta,
                            dq_rot[(size_t)t * DQ + i],
                            dq_rot[(size_t)t * DQ + i + halfD],
                            dk_rot[(size_t)t * DQ + i],
                            dk_rot[(size_t)t * DQ + i + halfD],
                            dq0, dq1, dk0, dk1, dtheta);
                        g.dQ[qk_index(b, h, seq, i, heads, length, DQ)] += dq0;
                        g.dQ[qk_index(b, h, seq, i + halfD, heads, length, DQ)] += dq1;
                        g.dK[qk_index(b, h, seq, i, heads, length, DQ)] += dk0;
                        g.dK[qk_index(b, h, seq, i + halfD, heads, length, DQ)] += dk1;
                        g.dAngle[angle_index(b, h, seq, i, heads, length, halfD)] += dtheta * a_cs[t] * kPi;
                        d_a_cs[t] += dtheta * raw_ang * kPi;
                    }
                    for (int j = 0; j < DV; ++j) {
                        g.dV[v_index(b, h, seq, j, heads, length, DV)] += dv_acc[(size_t)t * DV + j];
                    }
                }

                float suffix = 0.0f;
                for (int t = chunk_len - 1; t >= 0; --t) {
                    suffix += d_a_cs[t];
                    g.dA[scalar_index(b, h, chunk_start + t, heads, length)] += suffix;
                }

                std::copy(next_state_grad.begin(), next_state_grad.end(), state_grad.begin());
            }
        }
    }

    return g;
}

static double run_encoder(MTL::CommandQueue* queue,
                          const std::function<void(MTL::ComputeCommandEncoder*)>& encode) {
    auto* cmd = queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    encode(enc);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    return cmd->GPUEndTime() - cmd->GPUStartTime();
}

static void clear_bwd_outputs(MTL::Buffer* bufDQ,
                              MTL::Buffer* bufDK,
                              MTL::Buffer* bufDV,
                              MTL::Buffer* bufDA,
                              MTL::Buffer* bufDB,
                              MTL::Buffer* bufDAngle,
                              size_t qk_elems,
                              size_t v_elems,
                              size_t scalar_elems,
                              size_t angle_elems) {
    std::memset(bufDQ->contents(), 0, qk_elems * sizeof(float));
    std::memset(bufDK->contents(), 0, qk_elems * sizeof(float));
    std::memset(bufDV->contents(), 0, v_elems * sizeof(float));
    std::memset(bufDA->contents(), 0, scalar_elems * sizeof(float));
    std::memset(bufDB->contents(), 0, scalar_elems * sizeof(float));
    std::memset(bufDAngle->contents(), 0, angle_elems * sizeof(float));
}

static double median_ms(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    return xs[xs.size() / 2] * 1e3;
}

}  // namespace

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <metallib> [L] [B] [H]\n", argv[0]);
        return 1;
    }

    const char* metallib_path = argv[1];
    const int L = (argc > 2) ? std::atoi(argv[2]) : 256;
    const int B = (argc > 3) ? std::atoi(argv[3]) : 1;
    const int H = (argc > 4) ? std::atoi(argv[4]) : 2;
    const int CHUNK = 32;
    const int DQ = 64;
    const int DV = 64;
    const int HALF_D = DQ / 2;
    const int N_CHUNKS = (L + CHUNK - 1) / CHUNK;

    if (L % CHUNK != 0) {
        std::fprintf(stderr, "seq_len must be a multiple of %d\n", CHUNK);
        return 1;
    }

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "No Metal device\n");
        return 1;
    }
    MTL::CommandQueue* queue = device->newCommandQueue();

    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(metallib_path, NS::UTF8StringEncoding));
    MTL::Library* lib = device->newLibrary(url, &err);
    if (!lib) {
        std::fprintf(stderr, "Failed to load metallib at %s: %s\n", metallib_path,
                     err ? err->localizedDescription()->utf8String() : "(unknown)");
        return 1;
    }

    auto* fwd_fn = lib->newFunction(NS::String::string("mamba3_siso_fwd_32_64_64", NS::UTF8StringEncoding));
    auto* dzdo_fn = lib->newFunction(NS::String::string("mamba3_siso_bwd_dzdo_32_64_64", NS::UTF8StringEncoding));
    auto* bwd_fn = lib->newFunction(NS::String::string("mamba3_siso_bwd_main_32_64_64", NS::UTF8StringEncoding));
    if (!fwd_fn || !dzdo_fn || !bwd_fn) {
        std::fprintf(stderr, "Required kernels not found in metallib\n");
        return 1;
    }
    auto* fwd_pso = device->newComputePipelineState(fwd_fn, &err);
    auto* dzdo_pso = device->newComputePipelineState(dzdo_fn, &err);
    auto* bwd_pso = device->newComputePipelineState(bwd_fn, &err);
    fwd_fn->release();
    dzdo_fn->release();
    bwd_fn->release();
    if (!fwd_pso || !dzdo_pso || !bwd_pso) {
        std::fprintf(stderr, "PSO creation failed: %s\n",
                     err ? err->localizedDescription()->utf8String() : "(no error)");
        return 1;
    }

    std::mt19937 rng(42);
    const size_t qk_elems = (size_t)B * H * L * DQ;
    const size_t v_elems = (size_t)B * H * L * DV;
    const size_t scalar_elems = (size_t)B * H * L;
    const size_t angle_elems = scalar_elems * HALF_D;
    const size_t state_elems = (size_t)B * H * N_CHUNKS * DQ * DV;
    const uint32_t fwd_threads = uint32_t(CHUNK * 4);
    const uint32_t bwd_threads = 64;

    std::vector<__fp16> Q(qk_elems), K(qk_elems), V(v_elems), angle(angle_elems), dO(v_elems);
    std::vector<float> A(scalar_elems), Btrap(scalar_elems);
    fill_half(Q, rng, 0.5f);
    fill_half(K, rng, 0.5f);
    fill_half(V, rng, 0.5f);
    fill_half(angle, rng, 0.1f);
    fill_half(dO, rng, 0.5f);
    fill_float(A, rng, -0.05f, 0.2f);
    fill_float(Btrap, rng, 0.0f, 0.2f);

    const auto t0 = std::chrono::steady_clock::now();
    ForwardCache ref_fwd = cpu_forward(Q, K, V, A, Btrap, angle, B, H, L, DQ, DV, CHUNK);
    BackwardGrads ref_bwd = cpu_backward(Q, K, V, A, Btrap, angle, ref_fwd.states, dO, B, H, L, DQ, DV, CHUNK);
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    auto mode = MTL::ResourceStorageModeShared;
    auto* bufQ = device->newBuffer(Q.data(), Q.size() * sizeof(__fp16), mode);
    auto* bufK = device->newBuffer(K.data(), K.size() * sizeof(__fp16), mode);
    auto* bufV = device->newBuffer(V.data(), V.size() * sizeof(__fp16), mode);
    auto* bufA = device->newBuffer(A.data(), A.size() * sizeof(float), mode);
    auto* bufB = device->newBuffer(Btrap.data(), Btrap.size() * sizeof(float), mode);
    auto* bufAngle = device->newBuffer(angle.data(), angle.size() * sizeof(__fp16), mode);
    auto* bufO = device->newBuffer(v_elems * sizeof(__fp16), mode);
    auto* bufDO = device->newBuffer(dO.data(), dO.size() * sizeof(__fp16), mode);
    auto* bufStates = device->newBuffer(ref_fwd.states.data(), ref_fwd.states.size() * sizeof(float), mode);
    auto* bufDQ = device->newBuffer(qk_elems * sizeof(float), mode);
    auto* bufDK = device->newBuffer(qk_elems * sizeof(float), mode);
    auto* bufDV = device->newBuffer(v_elems * sizeof(float), mode);
    auto* bufDA = device->newBuffer(scalar_elems * sizeof(float), mode);
    auto* bufDB = device->newBuffer(scalar_elems * sizeof(float), mode);
    auto* bufDAngle = device->newBuffer(angle_elems * sizeof(float), mode);
    auto* bufDZ = device->newBuffer(v_elems * sizeof(__fp16), mode);
    auto* bufDOScaled = device->newBuffer(v_elems * sizeof(__fp16), mode);

    Mamba3FwdArgsHost fwd_args{(uint32_t)B, (uint32_t)H, (uint32_t)L, (uint32_t)N_CHUNKS};
    Mamba3BwdArgsHost bwd_args{(uint32_t)B, (uint32_t)H, (uint32_t)L, (uint32_t)N_CHUNKS};
    Mamba3BwdSplitArgsHost bwd_split_args{(uint32_t)B, (uint32_t)H, (uint32_t)H, (uint32_t)L, (uint32_t)N_CHUNKS};
    auto* bufFwdArgs = device->newBuffer(&fwd_args, sizeof(fwd_args), mode);
    auto* bufBwdArgs = device->newBuffer(&bwd_args, sizeof(bwd_args), mode);
    auto* bufBwdSplitArgs = device->newBuffer(&bwd_split_args, sizeof(bwd_split_args), mode);

    auto run_fwd_once = [&]() {
        return run_encoder(queue, [&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(fwd_pso);
            enc->setBuffer(bufQ, 0, 0);
            enc->setBuffer(bufK, 0, 1);
            enc->setBuffer(bufV, 0, 2);
            enc->setBuffer(bufA, 0, 3);
            enc->setBuffer(bufB, 0, 4);
            enc->setBuffer(bufAngle, 0, 5);
            enc->setBuffer(bufO, 0, 6);
            enc->setBuffer(bufFwdArgs, 0, 7);
            enc->dispatchThreadgroups(MTL::Size(B, H, 1), MTL::Size(fwd_threads, 1, 1));
        });
    };

    auto run_bwd_once = [&]() {
        clear_bwd_outputs(bufDQ, bufDK, bufDV, bufDA, bufDB, bufDAngle,
                          qk_elems, v_elems, scalar_elems, angle_elems);
        return run_encoder(queue, [&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(bwd_pso);
            enc->setBuffer(bufQ, 0, 0);
            enc->setBuffer(bufK, 0, 1);
            enc->setBuffer(bufV, 0, 2);
            enc->setBuffer(bufA, 0, 3);
            enc->setBuffer(bufB, 0, 4);
            enc->setBuffer(bufAngle, 0, 5);
            enc->setBuffer(bufDO, 0, 6);
            enc->setBuffer(bufStates, 0, 7);
            enc->setBuffer(bufDQ, 0, 8);
            enc->setBuffer(bufDK, 0, 9);
            enc->setBuffer(bufDV, 0, 10);
            enc->setBuffer(bufDA, 0, 11);
            enc->setBuffer(bufDB, 0, 12);
            enc->setBuffer(bufDAngle, 0, 13);
            enc->setBuffer(bufBwdArgs, 0, 14);
            enc->dispatchThreadgroups(MTL::Size(H, B, 1), MTL::Size(bwd_threads, 1, 1));
        });
    };

    auto run_dzdo_once = [&]() {
        return run_encoder(queue, [&](MTL::ComputeCommandEncoder* enc) {
            enc->setComputePipelineState(dzdo_pso);
            enc->setBuffer(bufDO, 0, 0);
            enc->setBuffer(bufO, 0, 1);
            enc->setBuffer(bufO, 0, 2);
            enc->setBuffer(bufDZ, 0, 3);
            enc->setBuffer(bufDOScaled, 0, 4);
            enc->setBuffer(bufBwdSplitArgs, 0, 5);
            enc->dispatchThreadgroups(MTL::Size(N_CHUNKS, H, B), MTL::Size(64, 1, 1));
        });
    };

    run_fwd_once();
    run_dzdo_once();
    run_bwd_once();

    std::vector<double> fwd_times, dzdo_times, bwd_times;
    for (int i = 0; i < 10; ++i) fwd_times.push_back(run_fwd_once());
    for (int i = 0; i < 10; ++i) dzdo_times.push_back(run_dzdo_once());
    for (int i = 0; i < 10; ++i) bwd_times.push_back(run_bwd_once());

    std::vector<__fp16> gpuO(v_elems);
    std::copy_n(static_cast<__fp16*>(bufO->contents()), v_elems, gpuO.begin());
    std::vector<float> gpuDQ(qk_elems), gpuDK(qk_elems), gpuDV(v_elems), gpuDA(scalar_elems), gpuDB(scalar_elems), gpuDAngle(angle_elems);
    std::copy_n(static_cast<float*>(bufDQ->contents()), qk_elems, gpuDQ.begin());
    std::copy_n(static_cast<float*>(bufDK->contents()), qk_elems, gpuDK.begin());
    std::copy_n(static_cast<float*>(bufDV->contents()), v_elems, gpuDV.begin());
    std::copy_n(static_cast<float*>(bufDA->contents()), scalar_elems, gpuDA.begin());
    std::copy_n(static_cast<float*>(bufDB->contents()), scalar_elems, gpuDB.begin());
    std::copy_n(static_cast<float*>(bufDAngle->contents()), angle_elems, gpuDAngle.begin());

    ErrStats fwd_err = compare_half_float(gpuO, ref_fwd.out);
    ErrStats dq_err = compare_float(gpuDQ, ref_bwd.dQ);
    ErrStats dk_err = compare_float(gpuDK, ref_bwd.dK);
    ErrStats dv_err = compare_float(gpuDV, ref_bwd.dV);
    ErrStats da_err = compare_float(gpuDA, ref_bwd.dA);
    ErrStats db_err = compare_float(gpuDB, ref_bwd.dB);
    ErrStats dangle_err = compare_float(gpuDAngle, ref_bwd.dAngle);

    std::printf("== Mamba3 SISO Fwd+Bwd ==\n");
    std::printf("B=%d H=%d L=%d DQ=%d DV=%d CHUNK=%d\n", B, H, L, DQ, DV, CHUNK);
    std::printf("CPU reference total: %.3f ms\n", cpu_ms);
    std::printf("Metal fwd median: %.3f ms\n", median_ms(fwd_times));
    std::printf("Metal dzdo median: %.3f ms\n", median_ms(dzdo_times));
    std::printf("Metal bwd median: %.3f ms\n", median_ms(bwd_times));
    std::printf("fwd l2_rel=%.6f max_abs=%.6f\n", fwd_err.l2_rel, fwd_err.max_abs);
    std::printf("dQ l2_rel=%.6f max_abs=%.6f\n", dq_err.l2_rel, dq_err.max_abs);
    std::printf("dK l2_rel=%.6f max_abs=%.6f\n", dk_err.l2_rel, dk_err.max_abs);
    std::printf("dV l2_rel=%.6f max_abs=%.6f\n", dv_err.l2_rel, dv_err.max_abs);
    std::printf("dA l2_rel=%.6f max_abs=%.6f\n", da_err.l2_rel, da_err.max_abs);
    std::printf("dB l2_rel=%.6f max_abs=%.6f\n", db_err.l2_rel, db_err.max_abs);
    std::printf("dAngle l2_rel=%.6f max_abs=%.6f\n", dangle_err.l2_rel, dangle_err.max_abs);
    {
        std::vector<std::pair<float, size_t>> worst;
        worst.reserve(gpuDA.size());
        for (size_t i = 0; i < gpuDA.size(); ++i) {
            worst.push_back({std::fabs(gpuDA[i] - ref_bwd.dA[i]), i});
        }
        const size_t topk = std::min<size_t>(8, worst.size());
        std::partial_sort(worst.begin(), worst.begin() + topk, worst.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t k = 0; k < topk; ++k) {
            size_t idx = worst[k].second;
            std::printf("dA_diff[%zu] ref=%.6f got=%.6f abs=%.6f\n",
                        idx, ref_bwd.dA[idx], gpuDA[idx], worst[k].first);
        }
    }

    return 0;
}
