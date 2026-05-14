// IcbRecorder — record GPU compute dispatches once, replay every token.
//
// Motivation: qwen3-8B emits ~473 compute dispatches per decoded token. The
// per-dispatch CPU cost on M4 (computeCommandEncoder() + setComputePipelineState
// + setBuffer×N + dispatchThreadgroups + endEncoding) is ~35 µs. That's ~12 ms
// of pure CPU encode work per token, ~14% of an 8B decode budget.
//
// MTLIndirectCommandBuffer (ICB) lets us pre-record the dispatch sequence and
// replay it with a single `executeCommandsInBuffer` call on a regular compute
// encoder. CPU encode cost drops to ~1 ms/token.
//
// What ICBs DO NOT support:
//  - `setBytes` (inline constants). Every per-slot scalar/struct argument that
//    the kernel reads as `constant T& [[buffer(N)]]` has to come from a real
//    MTL::Buffer. Callers can either pre-allocate small "args" buffers or
//    re-record affected slots between tokens (cheap).
//
// Usage:
//   auto* rec = sk::silicon::IcbRecorder::create(dev, max_slots, max_bind);
//   rec->record(slot, pso, bufs, offsets, n_bufs, grid, tg);
//   ...
//   rec->mark_resource(buf);  // anything the ICB will touch via setKernelBuffer
//   ...
//   auto* enc = cmd->computeCommandEncoder();
//   rec->execute(enc, 0, n_recorded);
//   enc->endEncoding();
//
// Threading: the recorder is single-producer. Don't call record/execute
// concurrently on the same instance.

#ifndef SK_INFERENCE_SILICON_ICB_RECORDER_H
#define SK_INFERENCE_SILICON_ICB_RECORDER_H

#include "Metal/Metal.hpp"

#include <cstdint>
#include <vector>

namespace sk { namespace silicon {

class IcbRecorder {
public:
    // Build an ICB sized for `max_slots` concurrent-dispatch commands, where
    // any single slot binds up to `max_buffer_bindings` kernel buffers.
    // Returns nullptr on Metal allocation failure.
    static IcbRecorder* create(MTL::Device* dev,
                               std::uint32_t max_slots,
                               std::uint32_t max_buffer_bindings);

    ~IcbRecorder();

    // Record one concurrent-dispatch command at `slot_idx`. Overwrites the
    // prior contents of that slot.
    //   pso       : pipeline state to bind for this slot
    //   buffers   : kernel buffers in setKernelBuffer-index order
    //   offsets   : per-buffer byte offsets
    //   n_buffers : length of `buffers`/`offsets` (must be ≤ max_buffer_bindings)
    //   grid      : threadgroups per grid
    //   tg        : threads per threadgroup
    //   barrier_before
    //             : if true (default) this slot waits for all earlier slots to
    //               finish before launching. ICB commands of type
    //               ConcurrentDispatch run concurrently by default; barriers
    //               are the only way to express producer/consumer ordering
    //               *within* a recorded ICB sequence.
    void record(std::uint32_t slot_idx,
                MTL::ComputePipelineState* pso,
                const MTL::Buffer* const* buffers,
                const NS::UInteger*      offsets,
                std::uint32_t            n_buffers,
                MTL::Size                grid,
                MTL::Size                tg,
                bool                     barrier_before = true);

    // Track a buffer that will be referenced by any recorded slot. The
    // execute() call will hand the tracked list to the parent encoder via
    // `useResources(...)` so the GPU residency tracker sees them.
    // Safe to call repeatedly with the same buffer (dedup'd).
    void mark_resource(MTL::Buffer* b);

    // Execute slots [first, first+count). Caller owns the encoder. Caller
    // must `endEncoding()` afterwards.
    void execute(MTL::ComputeCommandEncoder* enc,
                 std::uint32_t first,
                 std::uint32_t count);

    // Reset slot — useful when shrinking the recorded range. Cheap.
    void reset_slot(std::uint32_t slot_idx);

    std::uint32_t capacity() const { return cap_; }

    MTL::IndirectCommandBuffer* raw() const { return icb_; }

private:
    IcbRecorder() = default;
    IcbRecorder(const IcbRecorder&) = delete;
    IcbRecorder& operator=(const IcbRecorder&) = delete;

    MTL::IndirectCommandBuffer* icb_ = nullptr;
    std::vector<MTL::Resource*> tracked_;
    std::uint32_t               cap_           = 0;
    std::uint32_t               max_bindings_  = 0;
};

}}  // namespace sk::silicon

#endif  // SK_INFERENCE_SILICON_ICB_RECORDER_H
