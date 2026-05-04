//
//  activation.c++ — C binding implementation for activation kernels
//

#include "activation.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstring>

static MTL::Device*      g_device = nullptr;
static MTL::CommandQueue* g_queue  = nullptr;
static MTL::Library*     g_lib    = nullptr;

static void ensure_device(const char* metallib_path) {
    if (g_device) return;
    g_device = MTL::CreateSystemDefaultDevice();
    g_queue  = g_device->newCommandQueue();
    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(
        NS::String::string(metallib_path, NS::UTF8StringEncoding));
    g_lib = g_device->newLibrary(url, &err);
    if (!g_lib) {
        fprintf(stderr, "activation: failed to load %s\n", metallib_path);
    }
}

int sk_activation(const char* kernel_name,
                  void* x, void* y,
                  uint32_t rows, uint32_t cols) {

    // Default metallib path — caller can override via env or we ship fixed
    const char* lib_path = getenv("SK_METALLIB")
        ? getenv("SK_METALLIB") : "build/libsk.metallib";

    ensure_device(lib_path);
    if (!g_lib) return -1;

    auto* fn = g_lib->newFunction(
        NS::String::string(kernel_name, NS::UTF8StringEncoding));
    if (!fn) return -2;

    NS::Error* err = nullptr;
    auto* pso = g_device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso) { if (err) err->release(); return -3; }

    size_t bytes = (size_t)rows * cols * sizeof(__fp16);
    auto* bx = g_device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    auto* by = g_device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    memcpy(bx->contents(), x, bytes);  // copy input in

    auto* bRows = g_device->newBuffer(&rows, sizeof(uint32_t), MTL::ResourceStorageModeShared);
    auto* bCols = g_device->newBuffer(&cols, sizeof(uint32_t), MTL::ResourceStorageModeShared);

    uint32_t grid_y = (rows + 3) / 4;

    auto* cmd = g_queue->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bx,    0, 0);
    enc->setBuffer(by,    0, 1);
    enc->setBuffer(bRows, 0, 2);
    enc->setBuffer(bCols, 0, 3);
    enc->dispatchThreadgroups(MTL::Size(1, grid_y, 1), MTL::Size(128, 1, 1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    memcpy(y, by->contents(), bytes);  // copy output back

    cmd->release();
    pso->release();
    bx->release(); by->release(); bRows->release(); bCols->release();
    return 0;
}
