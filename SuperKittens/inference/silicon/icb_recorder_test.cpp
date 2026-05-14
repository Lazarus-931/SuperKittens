// icb_recorder_test — standalone smoke test for sk::silicon::IcbRecorder.
//
// What it tests:
//   * Build a recorder with 3 slots, max 4 buffer bindings.
//   * Record three `add_f16` dispatches:
//       slot 0:  y = a + b
//       slot 1:  z = y + a       (reuses slot-0 output)
//       slot 2:  w = z + b       (reuses slot-1 output)
//   * Run them via a single executeCommandsInBuffer call.
//   * Validate w[i] = (a[i] + b[i]) + a[i] + b[i] = 2*(a[i]+b[i]).
//
// Build (separate-compile + link; PRIVATE_IMPLEMENTATION macros must only land
// in metal_impl.cpp — defining them in any other TU produces ~1900 duplicate
// Metal selector symbols at link time):
//
//   clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//       -DNS_PRIVATE_IMPLEMENTATION -DMTL_PRIVATE_IMPLEMENTATION \
//       -DCA_PRIVATE_IMPLEMENTATION \
//       -c SuperKittens/kernels/metal_impl.cpp -o /tmp/metal_impl.o
//   clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//       -c SuperKittens/inference/silicon/icb_recorder.c++ -o /tmp/icb_recorder.o
//   clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//       -c SuperKittens/inference/silicon/icb_recorder_test.cpp \
//       -o /tmp/icb_recorder_test.o
//   clang++ -arch arm64 -framework Metal -framework Foundation \
//       -framework QuartzCore \
//       /tmp/metal_impl.o /tmp/icb_recorder.o /tmp/icb_recorder_test.o \
//       -o /tmp/icb_recorder_test
//   /tmp/icb_recorder_test [path/to/libsk.metallib]
//
// File extension is `.cpp` (not `.c++`) so build.sh's dispatcher glob doesn't
// pick it up — we don't want a main() linked into libsk.dylib.

// NOTE: do NOT define {NS,MTL,CA}_PRIVATE_IMPLEMENTATION here — those macros
// MUST be defined in exactly one TU on the link line (metal_impl.cpp). Even a
// "#define ... 0" form is enough to make headers expand the bodies, which then
// collide with metal_impl's copies at link time.
#include "Metal/Metal.hpp"

#include "icb_recorder.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// Half conversion (IEEE 754 binary16, round-to-nearest-even, finite-only).
// Adequate for the test inputs (small values).
inline std::uint16_t f32_to_f16(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, 4);
    std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t  exp  = ((bits >> 23) & 0xff) - 127 + 15;
    std::uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) return static_cast<std::uint16_t>(sign);
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
}
inline float f16_to_f32(std::uint16_t h) {
    std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h >> 10) & 0x1f;
    std::uint32_t mant =  h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}

}  // namespace

int main(int argc, char** argv) {
    const char* metallib = (argc > 1) ? argv[1] : "build/libsk.metallib";

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    if (!dev) { std::fprintf(stderr, "no metal device\n"); return 1; }
    std::fprintf(stderr, "device: %s  metallib: %s\n",
                 dev->name()->utf8String(), metallib);

    NS::Error* err = nullptr;
    auto* lib_path = NS::String::string(metallib, NS::UTF8StringEncoding);
    auto* url      = NS::URL::fileURLWithPath(lib_path);
    MTL::Library*  lib = dev->newLibrary(url, &err);
    if (!lib) {
        std::fprintf(stderr, "load metallib failed: %s\n",
                     err ? err->localizedDescription()->utf8String() : "?");
        return 2;
    }

    auto*          fn_name = NS::String::string("add_f16", NS::UTF8StringEncoding);
    MTL::Function* fn      = lib->newFunction(fn_name);
    if (!fn) { std::fprintf(stderr, "missing add_f16 in metallib\n"); return 3; }

    // ICB requires PSOs to opt in via supportIndirectCommandBuffers; without
    // this, indirectComputeCommand::setComputePipelineState faults at record.
    auto* pso_desc = MTL::ComputePipelineDescriptor::alloc()->init();
    pso_desc->setComputeFunction(fn);
    pso_desc->setSupportIndirectCommandBuffers(true);
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(
        pso_desc, MTL::PipelineOptionNone, nullptr, &err);
    pso_desc->release();
    if (!pso) { std::fprintf(stderr, "pso failed\n"); return 4; }

    constexpr std::uint32_t N = 1024;
    const std::size_t bytes = N * sizeof(std::uint16_t);

    MTL::Buffer* ba = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* bb = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* by = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* bz = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* bw = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    // ICB cannot consume setBytes; the kernel reads `n` from buffer(3) so we
    // park it in a tiny dedicated buffer.
    MTL::Buffer* bn = dev->newBuffer(sizeof(std::uint32_t),
                                     MTL::ResourceStorageModeShared);

    auto* ha = static_cast<std::uint16_t*>(ba->contents());
    auto* hb = static_cast<std::uint16_t*>(bb->contents());
    auto* hn = static_cast<std::uint32_t*>(bn->contents());
    for (std::uint32_t i = 0; i < N; ++i) {
        ha[i] = f32_to_f16(static_cast<float>(i) * 0.01f);
        hb[i] = f32_to_f16(static_cast<float>(i) * 0.02f);
    }
    *hn = N;

    auto* rec = sk::silicon::IcbRecorder::create(dev, /*max_slots=*/3,
                                                 /*max_buffer_bindings=*/4);
    if (!rec) { std::fprintf(stderr, "recorder create failed\n"); return 5; }

    auto record = [&](std::uint32_t slot,
                      MTL::Buffer* in0, MTL::Buffer* in1, MTL::Buffer* out) {
        const MTL::Buffer* bufs[4] = { in0, in1, out, bn };
        NS::UInteger offs[4]       = { 0, 0, 0, 0 };
        MTL::Size grid(std::max<std::uint32_t>(1u, (N + 3) / 4), 1, 1);
        MTL::Size tg(64, 1, 1);
        // Round grid up to threadgroup multiple.
        grid.width = (grid.width + tg.width - 1) / tg.width;
        rec->record(slot, pso, bufs, offs, 4, grid, tg);
        rec->mark_resource(in0);
        rec->mark_resource(in1);
        rec->mark_resource(out);
        rec->mark_resource(bn);
    };

    record(0, ba, bb, by);  // y = a + b
    record(1, by, ba, bz);  // z = y + a
    record(2, bz, bb, bw);  // w = z + b

    MTL::CommandQueue* q = dev->newCommandQueue();
    MTL::CommandBuffer* cmd = q->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    rec->execute(enc, 0, 3);
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    if (cmd->status() == MTL::CommandBufferStatusError) {
        std::fprintf(stderr, "GPU error: %s\n",
                     cmd->error() ? cmd->error()->localizedDescription()->utf8String() : "?");
        return 6;
    }

    auto* hw = static_cast<std::uint16_t*>(bw->contents());
    int bad = 0;
    float max_abs = 0.f;
    for (std::uint32_t i = 0; i < N; ++i) {
        float a   = f16_to_f32(ha[i]);
        float b   = f16_to_f32(hb[i]);
        float ref = 2.f * (a + b);
        float got = f16_to_f32(hw[i]);
        float d   = std::fabs(ref - got);
        if (d > max_abs) max_abs = d;
        // fp16 of ~60 has ulp ~0.06; relative-tolerance.
        if (d > 0.1f * std::max(1.f, std::fabs(ref))) {
            if (bad < 4) {
                std::fprintf(stderr, "  mismatch [%u]: got=%f ref=%f\n",
                             i, got, ref);
            }
            ++bad;
        }
    }
    std::fprintf(stderr, "max_abs=%g  mismatches=%d/%u\n",
                 max_abs, bad, N);

    delete rec;
    pso->release(); fn->release(); lib->release();
    ba->release(); bb->release(); by->release();
    bz->release(); bw->release(); bn->release();
    q->release(); dev->release();
    pool->release();

    if (bad > 0) { std::fprintf(stderr, "FAIL\n"); return 7; }
    std::fprintf(stderr, "OK\n");
    return 0;
}
