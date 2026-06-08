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
#include <cstdio>
#include <string>
#include <unordered_map>

namespace sk {

// Single shared state across all TUs that include this header. A plain
// `static` at namespace scope gives each TU its own copy (internal linkage),
// which lets one TU init the device while another reads a still-null pointer.
// An inline function with a function-local static is merged to one instance.
struct _BindState {
    MTL::Device*       dev = nullptr;
    MTL::CommandQueue* q   = nullptr;
    MTL::Library*      lib = nullptr;
    MTL::Library*      lib_src = nullptr;
};
inline _BindState& _bs() { static _BindState s; return s; }

inline bool _slurp_file(const char* path, std::string& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0) { std::fclose(f); return false; }
    out.resize((size_t)n);
    size_t rd = std::fread(out.data(), 1, (size_t)n, f);
    std::fclose(f);
    out.resize(rd);
    return true;
}

inline void bindings_init() {
    auto& bs = _bs();
    if (bs.dev) return;
    bs.dev = MTL::CreateSystemDefaultDevice();
    bs.q   = bs.dev->newCommandQueue();
    const char* path = getenv("SK_METALLIB") ? getenv("SK_METALLIB") : "build/libsk.metallib";
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(NS::String::string(path, NS::UTF8StringEncoding));
    bs.lib = bs.dev->newLibrary(url, &err);
    if (getenv("SK_BINDINGS_DEBUG")) {
        std::fprintf(stderr, "[sk] metallib path=%s lib=%p err=%s\n", path, (void*)bs.lib,
                     (!bs.lib && err) ? err->localizedDescription()->utf8String() : "");
    }

    // newLibraryWithSource fallback (SK_METAL_SRC_FALLBACK = .metal path): the
    // only PSO route on a CommandLineTools-only host (no offline metal/metallib).
    // Supplies functions missing from the prebuilt metallib. No-op when unset.
    const char* src_path = getenv("SK_METAL_SRC_FALLBACK");
    if (src_path && src_path[0]) {
        std::string src;
        if (_slurp_file(src_path, src)) {
            auto* opt = MTL::CompileOptions::alloc()->init();
            NS::Error* serr = nullptr;
            bs.lib_src = bs.dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding),
                                            opt, &serr);
            opt->release();
            if (!bs.lib_src && serr) {
                std::fprintf(stderr, "[sk] SK_METAL_SRC_FALLBACK compile failed: %s\n",
                             serr->localizedDescription()->utf8String());
            }
        } else {
            std::fprintf(stderr, "[sk] SK_METAL_SRC_FALLBACK unreadable: %s\n", src_path);
        }
    }
}

inline MTL::Function* _resolve_fn(const char* name) {
    auto& bs = _bs();
    auto* nm = NS::String::string(name, NS::UTF8StringEncoding);
    if (bs.lib) { if (auto* fn = bs.lib->newFunction(nm)) return fn; }
    if (bs.lib_src) { if (auto* fn = bs.lib_src->newFunction(nm)) return fn; }
    if (getenv("SK_BINDINGS_DEBUG")) std::fprintf(stderr, "[sk] UNRESOLVED fn: %s\n", name);
    return nullptr;
}

inline MTL::ComputePipelineState* bindings_pso(const char* name) {
    static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    bindings_init();
    auto* fn = _resolve_fn(name);
    if (!fn) return nullptr;
    NS::Error* err = nullptr;
    auto* pso = _bs().dev->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) {
        if (getenv("SK_BINDINGS_DEBUG"))
            std::fprintf(stderr, "[sk] PSO BUILD FAIL %s: %s\n", name,
                         err ? err->localizedDescription()->utf8String() : "?");
        if (err) err->release();
        return nullptr;
    }
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
    auto* fn = _resolve_fn(name);
    if (!fn) return nullptr;
    auto* desc = MTL::ComputePipelineDescriptor::alloc()->init();
    desc->setComputeFunction(fn);
    desc->setSupportIndirectCommandBuffers(true);
    NS::Error* err = nullptr;
    auto* pso = _bs().dev->newComputePipelineState(
        desc, MTL::PipelineOptionNone, nullptr, &err);
    desc->release();
    fn->release();
    if (!pso) { if (err) err->release(); return nullptr; }
    cache[name] = pso;
    return pso;
}

inline MTL::Device*       bindings_device()      { bindings_init(); return _bs().dev; }
inline MTL::CommandQueue* bindings_queue()       { bindings_init(); return _bs().q;   }
inline MTL::Library*      bindings_library()     { bindings_init(); return _bs().lib; }
inline MTL::Library*      bindings_library_src() { bindings_init(); return _bs().lib_src; }

// Resolve a function by name across the prebuilt metallib AND the optional
// SK_METAL_SRC_FALLBACK source library (the only PSO route on a CLT-only host).
// Callers that build PSOs with function-constant specialization use this so a
// kernel that lives only in the source-fallback lib still resolves.
inline MTL::Function* bindings_function(const char* name,
                                        MTL::FunctionConstantValues* fcv = nullptr,
                                        NS::Error** out_err = nullptr) {
    bindings_init();
    auto& bs = _bs();
    auto* nm = NS::String::string(name, NS::UTF8StringEncoding);
    NS::Error* err = nullptr;
    if (bs.lib) {
        auto* fn = fcv ? bs.lib->newFunction(nm, fcv, &err) : bs.lib->newFunction(nm);
        if (fn) return fn;
    }
    if (bs.lib_src) {
        NS::Error* serr = nullptr;
        auto* fn = fcv ? bs.lib_src->newFunction(nm, fcv, &serr) : bs.lib_src->newFunction(nm);
        if (fn) return fn;
        if (out_err) *out_err = serr;
        return nullptr;
    }
    if (out_err) *out_err = err;
    return nullptr;
}

} // namespace sk
#endif
