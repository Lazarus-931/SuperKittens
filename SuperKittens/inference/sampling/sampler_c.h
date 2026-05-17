// C ABI for the sampling pipeline. Consumed by ctypes in inference/sampler.py.
// Why extern "C": stable symbol names across compilers, no name mangling.

#ifndef SK_INFERENCE_SAMPLING_SAMPLER_C_H
#define SK_INFERENCE_SAMPLING_SAMPLER_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_sampler_t sk_sampler_t;

sk_sampler_t* sk_sampler_create(void* device, void* queue);
void          sk_sampler_set_greedy(sk_sampler_t*);
void          sk_sampler_set_top_p(sk_sampler_t*, float p, float temp);
void          sk_sampler_set_min_p(sk_sampler_t*, float p, float temp);
void          sk_sampler_set_multinomial(sk_sampler_t*, float temp);
void          sk_sampler_set_seed(sk_sampler_t*, uint64_t seed);
void          sk_sampler_sample(sk_sampler_t*,
                                void* logits_buf,
                                void* out_buf,
                                uint32_t vocab_size,
                                void* command_encoder);
void          sk_sampler_destroy(sk_sampler_t*);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SK_INFERENCE_SAMPLING_SAMPLER_C_H
