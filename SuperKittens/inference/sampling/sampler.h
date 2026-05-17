// Sampler: composable post-logits sampling for SK model launchers.
//
// Why: every model currently hard-codes argmax inline; production chat needs
// temperature / top-p / min-p / multinomial. The Sampler lowers a sampling
// policy to a sequence of dispatches against kernels in sample.metal
// (softmax_temp, top_p_mask, min_p_mask, multinomial, argmax) recorded into a
// caller-owned ComputeCommandEncoder. The caller owns lifetime of the encoder,
// device, queue, and logits/out buffers; the Sampler owns only its PSO cache
// and a small per-row uniform buffer used for multinomial draws.

#ifndef SK_INFERENCE_SAMPLING_SAMPLER_H
#define SK_INFERENCE_SAMPLING_SAMPLER_H

#include "Metal/Metal.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>

namespace sk { namespace sampling {

enum class Mode : std::uint8_t {
    Greedy      = 0,
    TopP        = 1,
    MinP        = 2,
    Multinomial = 3,
};

class Sampler {
public:
    static Sampler* create(MTL::Device* dev, MTL::CommandQueue* q);
    ~Sampler();

    static Sampler greedy(MTL::Device* dev, MTL::CommandQueue* q);
    static Sampler top_p(MTL::Device* dev, MTL::CommandQueue* q, float p, float temp);
    static Sampler min_p(MTL::Device* dev, MTL::CommandQueue* q, float p, float temp);
    static Sampler multinomial(MTL::Device* dev, MTL::CommandQueue* q, float temp);

    void set_greedy();
    void set_top_p(float p, float temp);
    void set_min_p(float p, float temp);
    void set_multinomial(float temp);
    void set_seed(std::uint64_t s);

    // Record dispatches that sample 1 token (rows=1) from `logits_buf` into
    // `out_buf` using the policy set on this Sampler. `out_buf` must be
    // sizeof(int32) × rows; logits_buf must be half × rows × vocab_size.
    // Caller owns `enc` (must be active, ends after this call returns).
    void sample(MTL::Buffer* logits_buf, MTL::Buffer* out_buf,
                std::uint32_t vocab_size, MTL::ComputeCommandEncoder* enc);

private:
    Sampler() = default;
    MTL::ComputePipelineState* pso(const char* host_name);
    void ensure_scratch(std::uint32_t vocab_size);
    void refresh_uniform();

    MTL::Device*       dev_   = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    MTL::Library*      lib_   = nullptr;
    std::unordered_map<std::string, MTL::ComputePipelineState*> psos_;

    // Per-row uniform draws for the multinomial kernel. rows=1 in this build.
    MTL::Buffer* u_buf_      = nullptr;
    MTL::Buffer* probs_buf_  = nullptr;
    std::uint32_t probs_cap_ = 0;

    Mode  mode_ = Mode::Greedy;
    float temp_ = 1.0f;
    float p_    = 1.0f;
    std::mt19937_64 rng_{0xC0FFEEu};
};

}}  // namespace sk::sampling

#endif  // SK_INFERENCE_SAMPLING_SAMPLER_H
