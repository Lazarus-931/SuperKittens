//
//  runtime/device.h — Device, Queue, Library singleton
//  Replaces the scattered ensure_device() in every binding .c++
//

#ifndef SK_RUNTIME_DEVICE_H
#define SK_RUNTIME_DEVICE_H

#include <Metal/Metal.hpp>

namespace sk::runtime {

// ── one-time init ──
// Call once at startup.  All kernel bindings share these.
void init(MTL::Device* device = nullptr, const char* metallib_path = nullptr);
void shutdown();

// ── accessors (never null after init) ──
MTL::Device*        device();
MTL::CommandQueue*  queue();
MTL::Library*       library();

} // namespace sk::runtime

#endif
