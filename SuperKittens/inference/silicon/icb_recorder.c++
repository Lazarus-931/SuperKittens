// IcbRecorder implementation. See icb_recorder.h for rationale.

#include "icb_recorder.h"

#include <cassert>
#include <cstdio>

namespace sk { namespace silicon {

IcbRecorder* IcbRecorder::create(MTL::Device* dev,
                                 std::uint32_t max_slots,
                                 std::uint32_t max_buffer_bindings) {
    if (!dev || max_slots == 0) return nullptr;

    auto* desc = MTL::IndirectCommandBufferDescriptor::alloc()->init();
    if (!desc) return nullptr;

    desc->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatch);
    desc->setInheritPipelineState(false);
    desc->setMaxKernelBufferBindCount(max_buffer_bindings);

    MTL::IndirectCommandBuffer* icb = dev->newIndirectCommandBuffer(
        desc, max_slots, MTL::ResourceStorageModeShared);
    desc->release();

    if (!icb) {
        std::fprintf(stderr,
                     "[icb_recorder] newIndirectCommandBuffer failed "
                     "(max_slots=%u, max_bindings=%u)\n",
                     max_slots, max_buffer_bindings);
        return nullptr;
    }

    auto* r = new IcbRecorder();
    r->icb_          = icb;
    r->cap_          = max_slots;
    r->max_bindings_ = max_buffer_bindings;
    return r;
}

IcbRecorder::~IcbRecorder() {
    if (icb_) icb_->release();
    icb_ = nullptr;
}

void IcbRecorder::record(std::uint32_t slot_idx,
                         MTL::ComputePipelineState* pso,
                         const MTL::Buffer* const* buffers,
                         const NS::UInteger*       offsets,
                         std::uint32_t             n_buffers,
                         MTL::Size                 grid,
                         MTL::Size                 tg,
                         bool                      barrier_before) {
    assert(slot_idx < cap_);
    assert(n_buffers <= max_bindings_);
    assert(pso != nullptr);

    auto* cmd = icb_->indirectComputeCommand(slot_idx);
    cmd->reset();
    cmd->setComputePipelineState(pso);
    if (barrier_before) cmd->setBarrier();
    for (std::uint32_t i = 0; i < n_buffers; ++i) {
        cmd->setKernelBuffer(buffers[i], offsets[i], i);
    }
    cmd->concurrentDispatchThreadgroups(grid, tg);
}

void IcbRecorder::reset_slot(std::uint32_t slot_idx) {
    assert(slot_idx < cap_);
    icb_->indirectComputeCommand(slot_idx)->reset();
}

void IcbRecorder::mark_resource(MTL::Buffer* b) {
    if (!b) return;
    // Linear dedup. Tracked count is typically small (~few dozen) for a
    // full layer/model graph, so this is cheaper than a hash set.
    for (auto* r : tracked_) {
        if (r == static_cast<MTL::Resource*>(b)) return;
    }
    tracked_.push_back(static_cast<MTL::Resource*>(b));
}

void IcbRecorder::execute(MTL::ComputeCommandEncoder* enc,
                          std::uint32_t first,
                          std::uint32_t count) {
    assert(enc != nullptr);
    assert(first + count <= cap_);
    if (count == 0) return;

    if (!tracked_.empty()) {
        enc->useResources(
            reinterpret_cast<const MTL::Resource* const*>(tracked_.data()),
            tracked_.size(),
            MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
    enc->executeCommandsInBuffer(icb_, NS::Range::Make(first, count));
}

}}  // namespace sk::silicon
