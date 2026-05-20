// C ABI for sk::Executor. Consumed by ctypes in inference/executor.py.
// WHY extern "C": stable symbol names, no mangling.

#ifndef SK_INFERENCE_EXECUTOR_EXECUTOR_C_H
#define SK_INFERENCE_EXECUTOR_EXECUTOR_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sk_executor_t sk_executor_t;

// Flat MTLSize triple. ctypes pushes structs by value via a Structure mirror.
typedef struct sk_mtl_size_t {
    uint64_t width;
    uint64_t height;
    uint64_t depth;
} sk_mtl_size_t;

// Returns the default MTLDevice (same one the bindings layer uses) as an
// opaque void*. Lets Python tests construct an Executor against a real
// device without pulling in PyObjC.
void*          sk_executor_default_device(void);

sk_executor_t* sk_executor_create(void* device);
void           sk_executor_destroy(sk_executor_t*);

// Dispatch a kernel by host_name. buffers is a contiguous array of MTL::Buffer*
// pointers; ctypes passes (void**, int). Grid/tg are flat triples to avoid
// dragging metal-cpp into the C surface.
void sk_executor_dispatch(sk_executor_t*, const char* host_name,
                          void** buffers, int n_buffers,
                          sk_mtl_size_t grid, sk_mtl_size_t tg);

void sk_executor_record_token_icb_begin(sk_executor_t*);
void sk_executor_record_token_icb_end(sk_executor_t*);
void sk_executor_replay_token(sk_executor_t*, void* cmd_buffer);
void sk_executor_reset(sk_executor_t*);

// Test helpers — let Python verify the PSO cache + scratch pool without
// shelling all the way back into the dispatch path. Returning void* keeps
// the ABI minimal.
void* sk_executor_pso(sk_executor_t*, const char* host_name);
void* sk_executor_scratch(sk_executor_t*, uint64_t nbytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SK_INFERENCE_EXECUTOR_EXECUTOR_C_H
