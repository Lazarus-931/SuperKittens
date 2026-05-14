//
//  runtime_bindings.h — shared device/queue/library/PSO for kernel bindings
//  Include from every binding .c++ to eliminate ~20 lines of duplicate boilerplate.
//

#ifndef SK_RUNTIME_BINDINGS_H
#define SK_RUNTIME_BINDINGS_H

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace sk {

static MTL::Device*      _dev = nullptr;
static MTL::CommandQueue* _q   = nullptr;
static MTL::Library*     _lib = nullptr;

inline void bindings_init() {
    if (_dev) return;
    _dev = MTL::CreateSystemDefaultDevice();
    _q   = _dev->newCommandQueue();
    const char* path = getenv("SK_METALLIB") ? getenv("SK_METALLIB") : "build/libsk.metallib";
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
    _lib = _dev->newLibrary(url, &err);
}

inline MTL::ComputePipelineState* bindings_pso(const char* name) {
    static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    bindings_init();
    if (!_lib) return nullptr;
    auto* fn = _lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    NS::Error* err = nullptr;
    auto* pso = _dev->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) { if (err) err->release(); return nullptr; }
    cache[name] = pso;
    return pso;
}

// ICB-compatible PSO factory.
//
// MTLIndirectCommandBuffer rejects any PSO that was not built with
// `setSupportIndirectCommandBuffers(true)` — calling
// `indirectComputeCommand::setComputePipelineState(pso)` on a PSO built via
// `newComputePipelineState(MTL::Function*, &err)` (the default route in
// bindings_pso) faults at record time on Apple silicon. This factory builds
// the PSO via the descriptor route with the flag enabled, so the resulting
// PSO is safe to bind into both a regular ComputeCommandEncoder AND an ICB
// slot. Cached separately from bindings_pso so callers don't accidentally
// share a non-ICB PSO.
inline MTL::ComputePipelineState* bindings_pso_icb(const char* name) {
    static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    bindings_init();
    if (!_lib) return nullptr;
    auto* fn = _lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    auto* desc = MTL::ComputePipelineDescriptor::alloc()->init();
    desc->setComputeFunction(fn);
    desc->setSupportIndirectCommandBuffers(true);
    NS::Error* err = nullptr;
    auto* pso = _dev->newComputePipelineState(
        desc, MTL::PipelineOptionNone, nullptr, &err);
    desc->release();
    fn->release();
    if (!pso) { if (err) err->release(); return nullptr; }
    cache[name] = pso;
    return pso;
}

inline MTL::Device*       bindings_device()  { bindings_init(); return _dev; }
inline MTL::CommandQueue* bindings_queue()   { bindings_init(); return _q;   }
inline MTL::Library*      bindings_library() { bindings_init(); return _lib;  }

} // namespace sk
#endif
