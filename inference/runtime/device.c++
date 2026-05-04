//
//  runtime/device.c++ — Device singleton implementation
//

#include "device.h"
#include <Foundation/Foundation.hpp>
#include <cstdio>
#include <cstdlib>

namespace sk::runtime {

static MTL::Device*       g_device = nullptr;
static MTL::CommandQueue* g_queue  = nullptr;
static MTL::Library*      g_lib    = nullptr;

void init(MTL::Device* dev, const char* metallib_path) {
    if (g_device) return;

    g_device = dev ? dev : MTL::CreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "runtime: no Metal device\n");
        return;
    }
    g_queue = g_device->newCommandQueue();

    const char* path = metallib_path
        ? metallib_path
        : (getenv("SK_METALLIB") ? getenv("SK_METALLIB") : "build/libsk.metallib");

    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(
        NS::String::string(path, NS::UTF8StringEncoding));
    g_lib = g_device->newLibrary(url, &err);
    if (!g_lib) {
        fprintf(stderr, "runtime: failed to load %s\n", path);
    }
}

void shutdown() {
    if (g_lib)    { g_lib->release();    g_lib    = nullptr; }
    if (g_queue)  { g_queue->release();  g_queue  = nullptr; }
    if (g_device) { g_device->release(); g_device = nullptr; }
}

MTL::Device*       device() { return g_device; }
MTL::CommandQueue* queue()  { return g_queue;  }
MTL::Library*      library(){ return g_lib;    }

} // namespace sk::runtime
