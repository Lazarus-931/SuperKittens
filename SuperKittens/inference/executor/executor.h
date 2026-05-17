// Executor — shared per-model dispatch object. See PHASE4_DESIGN.md.
//
// WHY: every model launcher today re-rolls the same command queue, PSO cache,
// scratch pool, residency set, and (post-Phase 2) ICB lifecycle. Pulling them
// into one object lets per-model launchers shrink ~50% LOC while keeping
// kernel-level dispatch sites identical bit-for-bit.

#ifndef SK_INFERENCE_EXECUTOR_EXECUTOR_H
#define SK_INFERENCE_EXECUTOR_EXECUTOR_H

#include <Metal/Metal.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../sampling/sampler_c.h"

namespace sk {

class Executor {
public:
    explicit Executor(MTL::Device* dev);
    ~Executor();

    // PSO cache. host_name is a Metal kernel function name in libsk.metallib.
    // Idempotent; first call compiles + caches, repeats return the same handle.
    MTL::ComputePipelineState* pso(const char* host_name);

    // Scratch pool. Returns a buffer with capacity >= nbytes; size is rounded
    // up to next power-of-two so repeated calls for similar sizes hit a slot.
    // Buffers persist until reset(); never freed mid-decode (decode is hot).
    MTL::Buffer* scratch(std::size_t nbytes);

    // Residency. Caller hands in long-lived buffers (weights, KV cache,
    // token_args). When MTLResidencySet is available the buffer joins the
    // set; otherwise we fall back to encoder-time useResource:usage:.
    // Either way, the buffer pointer is retained in _resident for the
    // useResource fallback path during dispatch().
    void make_resident(MTL::Buffer* buf);

    // Convenience dispatch. Direct-encodes against the currently-open
    // ComputeCommandEncoder (set via dispatch_with_encoder). If no encoder
    // is set, allocates a transient one off the executor's queue.
    void dispatch(const char* host_name,
                  std::initializer_list<MTL::Buffer*> buffers,
                  MTL::Size grid, MTL::Size tg);

    // Flat-arg variant for the C ABI (ctypes can't pass initializer_list).
    void dispatch_v(const char* host_name,
                    MTL::Buffer** buffers, int n_buffers,
                    MTL::Size grid, MTL::Size tg);

    // Token-graph lifecycle. begin opens an ICB-recording window; the next
    // dispatch() calls record into the ICB instead of executing live; end
    // closes it. replay_token encodes one executeCommandsInBuffer for the
    // recorded slots into the supplied command buffer.
    //
    // WHY split begin/end rather than the design-doc closure form: ctypes
    // can't pass a C++ lambda. The internal C++ surface keeps both styles —
    // record_token_icb(closure) below wraps begin/end for native callers.
    void record_token_icb_begin();
    void record_token_icb_end();
    void record_token_icb(std::function<void(MTL::ComputeCommandEncoder*)> graph_fn);

    void replay_token(MTL::CommandBuffer* cmd);

    // Sampler integration (Phase 1). Pure handoff: Executor does not own the
    // sampler's lifetime — callers destroy it.
    void attach_sampler(sk_sampler_t* s) { _sampler = s; }
    sk_sampler_t* sampler() const { return _sampler; }

    // Accessors for callers that still drive their own encoders (qwen3 today
    // is mid-migration; full record-replay lands later).
    MTL::Device*       device() const { return _dev; }
    MTL::CommandQueue* queue()  const { return _queue; }

    // Drop ICB + scratch pool. Keeps PSO cache and residency list (those are
    // tied to weights, which outlive a reset).
    void reset();

private:
    MTL::ComputePipelineState* _compile(const char* host_name);

    MTL::Device*       _dev   = nullptr;
    MTL::CommandQueue* _queue = nullptr;
    MTL::Library*      _lib   = nullptr;

    std::unordered_map<std::string, MTL::ComputePipelineState*> _psos;

    // Scratch pool keyed by rounded-up capacity. We keep at most one entry
    // per pow2 bucket; decode shapes are stable so the pool warms once.
    std::unordered_map<std::size_t, MTL::Buffer*> _scratch_by_cap;

    // Resident buffer list. When MTLResidencySet is unavailable we replay
    // these via useResource:usage: on each encoder we open.
    std::vector<MTL::Buffer*> _resident;

    // ICB recording state. _recording flips on between begin/end; while on,
    // dispatch() does NOT touch the GPU — it stores a deferred call.
    bool _recording = false;
    struct Recorded {
        MTL::ComputePipelineState* pso;
        std::vector<MTL::Buffer*>  buffers;
        MTL::Size grid;
        MTL::Size tg;
    };
    std::vector<Recorded> _recorded;

    sk_sampler_t* _sampler = nullptr;
};

}  // namespace sk

#endif  // SK_INFERENCE_EXECUTOR_EXECUTOR_H
