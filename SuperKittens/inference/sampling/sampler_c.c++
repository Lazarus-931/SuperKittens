// C ABI shim around sk::sampling::Sampler.

#include "sampler.h"
#include "sampler_c.h"

using sk::sampling::Sampler;

extern "C" {

sk_sampler_t* sk_sampler_create(void* device, void* queue) {
    return reinterpret_cast<sk_sampler_t*>(
        Sampler::create(reinterpret_cast<MTL::Device*>(device),
                        reinterpret_cast<MTL::CommandQueue*>(queue)));
}
void sk_sampler_set_greedy(sk_sampler_t* s) {
    if (s) reinterpret_cast<Sampler*>(s)->set_greedy();
}
void sk_sampler_set_top_p(sk_sampler_t* s, float p, float temp) {
    if (s) reinterpret_cast<Sampler*>(s)->set_top_p(p, temp);
}
void sk_sampler_set_min_p(sk_sampler_t* s, float p, float temp) {
    if (s) reinterpret_cast<Sampler*>(s)->set_min_p(p, temp);
}
void sk_sampler_set_multinomial(sk_sampler_t* s, float temp) {
    if (s) reinterpret_cast<Sampler*>(s)->set_multinomial(temp);
}
void sk_sampler_set_seed(sk_sampler_t* s, uint64_t seed) {
    if (s) reinterpret_cast<Sampler*>(s)->set_seed(seed);
}
void sk_sampler_sample(sk_sampler_t* s,
                       void* logits_buf, void* out_buf,
                       uint32_t vocab_size, void* command_encoder) {
    if (!s) return;
    reinterpret_cast<Sampler*>(s)->sample(
        reinterpret_cast<MTL::Buffer*>(logits_buf),
        reinterpret_cast<MTL::Buffer*>(out_buf),
        vocab_size,
        reinterpret_cast<MTL::ComputeCommandEncoder*>(command_encoder));
}
void sk_sampler_destroy(sk_sampler_t* s) {
    if (s) delete reinterpret_cast<Sampler*>(s);
}

}  // extern "C"
