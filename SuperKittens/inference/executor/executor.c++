// Executor implementation. WHY-only commentary; see executor.h for surface.

#include "executor.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace sk {

namespace {

// Reuse the sampler library-loader rationale: avoid a hard build-time path so
// tests + dev-tree binaries can both find libsk.metallib via env or build/.
MTL::Library* load_default_library(MTL::Device* dev) {
    auto try_path = [&](const char* path) -> MTL::Library* {
        if (!path || !*path) return nullptr;
        if (!std::filesystem::exists(path)) return nullptr;
        auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
        NS::Error* err = nullptr;
        return dev->newLibrary(url, &err);
    };
    if (auto* L = try_path(std::getenv("SK_METALLIB"))) return L;
    if (auto* L = try_path("build/libsk.metallib"))    return L;
    return nullptr;
}

// Round up to next power-of-two (>= 64 to avoid pathological tiny buffers).
std::size_t pow2_ceil(std::size_t n) {
    if (n < 64) n = 64;
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

}  // namespace

Executor::Executor(MTL::Device* dev) : _dev(dev) {
    // Caller-supplied device may be null in unit tests on non-Metal hosts;
    // we degrade to inert state rather than crashing — tests still exercise
    // PSO-cache and scratch-pool data structures.
    if (_dev) {
        _queue = _dev->newCommandQueue();
        _lib   = load_default_library(_dev);
    }
}

Executor::~Executor() {
    for (auto& kv : _psos) if (kv.second) kv.second->release();
    _psos.clear();
    for (auto& kv : _scratch_by_cap) if (kv.second) kv.second->release();
    _scratch_by_cap.clear();
    // _resident pointers are NOT owned; weights/KV outlive the executor.
    if (_lib)   _lib->release();
    if (_queue) _queue->release();
}

MTL::ComputePipelineState* Executor::_compile(const char* host_name) {
    if (!_dev || !_lib) return nullptr;
    auto* fn = _lib->newFunction(NS::String::string(host_name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    NS::Error* err = nullptr;
    auto* p = _dev->newComputePipelineState(fn, &err);
    fn->release();
    if (!p) {
        std::fprintf(stderr, "[executor] PSO compile failed for %s\n", host_name);
        return nullptr;
    }
    return p;
}

MTL::ComputePipelineState* Executor::pso(const char* host_name) {
    if (!host_name) return nullptr;
    auto it = _psos.find(host_name);
    if (it != _psos.end()) return it->second;
    auto* p = _compile(host_name);
    if (p) _psos.emplace(host_name, p);
    return p;
}

MTL::Buffer* Executor::scratch(std::size_t nbytes) {
    if (!_dev || nbytes == 0) return nullptr;
    const std::size_t cap = pow2_ceil(nbytes);
    auto it = _scratch_by_cap.find(cap);
    if (it != _scratch_by_cap.end()) return it->second;
    auto* b = _dev->newBuffer(cap, MTL::ResourceStorageModeShared);
    _scratch_by_cap.emplace(cap, b);
    return b;
}

void Executor::make_resident(MTL::Buffer* buf) {
    if (!buf) return;
    _resident.push_back(buf);
}

void Executor::dispatch(const char* host_name,
                        std::initializer_list<MTL::Buffer*> buffers,
                        MTL::Size grid, MTL::Size tg) {
    auto* p = pso(host_name);
    if (!p) return;
    if (_recording) {
        Recorded r{p, std::vector<MTL::Buffer*>(buffers.begin(), buffers.end()), grid, tg};
        _recorded.push_back(std::move(r));
        return;
    }
    // Direct-encode against a one-shot command buffer. We commit + wait here
    // so the executor's dispatch() is a synchronous primitive; callers that
    // need batching should drive their own command buffer via queue().
    if (!_queue) return;
    auto* cmd = _queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    // Replay residency list (useResource fallback path; cheap if list is small).
    for (auto* r : _resident) {
        enc->useResource(r, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
    enc->setComputePipelineState(p);
    int slot = 0;
    for (auto* b : buffers) enc->setBuffer(b, 0, slot++);
    enc->dispatchThreadgroups(grid, tg);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

void Executor::dispatch_v(const char* host_name,
                          MTL::Buffer** buffers, int n_buffers,
                          MTL::Size grid, MTL::Size tg) {
    auto* p = pso(host_name);
    if (!p) return;
    if (_recording) {
        Recorded r;
        r.pso = p;
        r.buffers.assign(buffers, buffers + n_buffers);
        r.grid = grid; r.tg = tg;
        _recorded.push_back(std::move(r));
        return;
    }
    if (!_queue) return;
    auto* cmd = _queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    for (auto* r : _resident) {
        enc->useResource(r, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
    enc->setComputePipelineState(p);
    for (int i = 0; i < n_buffers; ++i) enc->setBuffer(buffers[i], 0, i);
    enc->dispatchThreadgroups(grid, tg);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
}

void Executor::record_token_icb_begin() {
    _recording = true;
    _recorded.clear();
}

void Executor::record_token_icb_end() {
    _recording = false;
}

void Executor::record_token_icb(std::function<void(MTL::ComputeCommandEncoder*)> graph_fn) {
    // We invoke graph_fn with nullptr because under recording, dispatch()
    // does not touch a live encoder. Callers that need an encoder during
    // record (e.g. legacy graph_fn closures) should migrate to dispatch().
    record_token_icb_begin();
    if (graph_fn) graph_fn(nullptr);
    record_token_icb_end();
}

void Executor::replay_token(MTL::CommandBuffer* cmd) {
    if (!cmd) return;
    // WHY a plain encoder rather than a true MTLIndirectCommandBuffer here:
    // the SDK-level MTLIndirectCommandBuffer path requires every PSO to be
    // built with setSupportIndirectCommandBuffers(true) plus a recorder of
    // matching shape (see inference/silicon/icb_recorder). That migration is
    // tracked in Phase 2 — until per-token scalars are read from TokenArgs in
    // each kernel, we cannot record the full decode graph. Replaying through
    // a normal encoder is functionally equivalent for the unit-test scope and
    // keeps the proof-point on real launchers honest.
    auto* enc = cmd->computeCommandEncoder();
    for (auto* r : _resident) {
        enc->useResource(r, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
    for (auto& rec : _recorded) {
        enc->setComputePipelineState(rec.pso);
        for (std::size_t i = 0; i < rec.buffers.size(); ++i) {
            enc->setBuffer(rec.buffers[i], 0, (NS::UInteger)i);
        }
        enc->dispatchThreadgroups(rec.grid, rec.tg);
    }
    enc->endEncoding();
}

void Executor::reset() {
    _recording = false;
    _recorded.clear();
    for (auto& kv : _scratch_by_cap) if (kv.second) kv.second->release();
    _scratch_by_cap.clear();
}

}  // namespace sk
