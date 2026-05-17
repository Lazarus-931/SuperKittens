// C ABI shim around sk::Executor.

#include "executor.h"
#include "executor_c.h"

using sk::Executor;

namespace {
inline MTL::Size to_mtl(sk_mtl_size_t s) {
    return MTL::Size((NS::UInteger)s.width, (NS::UInteger)s.height, (NS::UInteger)s.depth);
}
}

extern "C" {

void* sk_executor_default_device(void) {
    // Lazily create + leak (process-lifetime singleton). Tests close their
    // Executor — but the device itself is shared with the launcher's
    // bindings layer if it gets initialized later.
    static MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    return reinterpret_cast<void*>(dev);
}

sk_executor_t* sk_executor_create(void* device) {
    return reinterpret_cast<sk_executor_t*>(
        new Executor(reinterpret_cast<MTL::Device*>(device)));
}

void sk_executor_destroy(sk_executor_t* e) {
    if (e) delete reinterpret_cast<Executor*>(e);
}

void sk_executor_dispatch(sk_executor_t* e, const char* host_name,
                          void** buffers, int n_buffers,
                          sk_mtl_size_t grid, sk_mtl_size_t tg) {
    if (!e || !host_name) return;
    reinterpret_cast<Executor*>(e)->dispatch_v(
        host_name,
        reinterpret_cast<MTL::Buffer**>(buffers),
        n_buffers,
        to_mtl(grid), to_mtl(tg));
}

void sk_executor_record_token_icb_begin(sk_executor_t* e) {
    if (e) reinterpret_cast<Executor*>(e)->record_token_icb_begin();
}

void sk_executor_record_token_icb_end(sk_executor_t* e) {
    if (e) reinterpret_cast<Executor*>(e)->record_token_icb_end();
}

void sk_executor_replay_token(sk_executor_t* e, void* cmd_buffer) {
    if (!e) return;
    reinterpret_cast<Executor*>(e)->replay_token(
        reinterpret_cast<MTL::CommandBuffer*>(cmd_buffer));
}

void sk_executor_reset(sk_executor_t* e) {
    if (e) reinterpret_cast<Executor*>(e)->reset();
}

void* sk_executor_pso(sk_executor_t* e, const char* host_name) {
    if (!e || !host_name) return nullptr;
    return reinterpret_cast<void*>(
        reinterpret_cast<Executor*>(e)->pso(host_name));
}

void* sk_executor_scratch(sk_executor_t* e, uint64_t nbytes) {
    if (!e) return nullptr;
    return reinterpret_cast<void*>(
        reinterpret_cast<Executor*>(e)->scratch((std::size_t)nbytes));
}

}  // extern "C"
