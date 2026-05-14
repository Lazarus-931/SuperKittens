// icb_argmax_tail_test — validates that the qwen3 ICB-tail path
// (argmax_partial + argmax_reduce, pre-recorded into an MTLIndirectCommandBuffer
// and replayed via a single executeCommandsInBuffer call) produces a token id
// identical to the direct (non-ICB) 2-pass argmax path on synthetic logits.
//
// This is the local-mac correctness witness for commit 2 of dev-icb-wire-qwen3.
// It exercises the exact pattern the qwen launcher records at create() time:
//   slot 0: argmax_partial(logits, val_buf, idx_buf, args[vocab_size])
//   slot 1: argmax_reduce (val_buf, idx_buf, output_id, args[n_blocks])
//
// Build (same recipe as icb_recorder_test):
//   clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//       -DNS_PRIVATE_IMPLEMENTATION -DMTL_PRIVATE_IMPLEMENTATION \
//       -DCA_PRIVATE_IMPLEMENTATION \
//       -c SuperKittens/kernels/metal_impl.cpp -o /tmp/metal_impl.o
//   clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//       -c SuperKittens/inference/silicon/icb_recorder.c++ -o /tmp/icb_recorder.o
//   clang++ -std=gnu++20 -O2 -arch arm64 -I metal-cpp \
//       -c SuperKittens/inference/silicon/icb_argmax_tail_test.cpp \
//       -o /tmp/icb_argmax_tail_test.o
//   clang++ -arch arm64 -framework Metal -framework Foundation \
//       -framework QuartzCore \
//       /tmp/metal_impl.o /tmp/icb_recorder.o /tmp/icb_argmax_tail_test.o \
//       -o /tmp/icb_argmax_tail_test
//   /tmp/icb_argmax_tail_test [path/to/libsk.metallib]
//
// Extension is .cpp so build.sh's *.c++ glob does not link a main() into the dylib.

#include "Metal/Metal.hpp"

#include "icb_recorder.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
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

// Build a PSO with setSupportIndirectCommandBuffers(true).
MTL::ComputePipelineState* pso_icb(MTL::Device* dev, MTL::Library* lib,
                                   const char* name) {
    auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    auto* desc = MTL::ComputePipelineDescriptor::alloc()->init();
    desc->setComputeFunction(fn);
    desc->setSupportIndirectCommandBuffers(true);
    NS::Error* err = nullptr;
    auto* pso = dev->newComputePipelineState(
        desc, MTL::PipelineOptionNone, nullptr, &err);
    desc->release(); fn->release();
    return pso;
}
MTL::ComputePipelineState* pso_plain(MTL::Device* dev, MTL::Library* lib,
                                     const char* name) {
    auto* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) return nullptr;
    NS::Error* err = nullptr;
    auto* pso = dev->newComputePipelineState(fn, &err);
    fn->release();
    return pso;
}
}  // namespace

int main(int argc, char** argv) {
    const char* metallib = (argc > 1) ? argv[1] : "build/libsk.metallib";
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    if (!dev) { std::fprintf(stderr, "no metal device\n"); return 1; }
    std::fprintf(stderr, "device: %s\n", dev->name()->utf8String());

    NS::Error* err = nullptr;
    auto* url = NS::URL::fileURLWithPath(
        NS::String::string(metallib, NS::UTF8StringEncoding));
    MTL::Library* lib = dev->newLibrary(url, &err);
    if (!lib) { std::fprintf(stderr, "metallib load failed\n"); return 2; }

    auto* part_icb = pso_icb  (dev, lib, "argmax_partial");
    auto* red_icb  = pso_icb  (dev, lib, "argmax_reduce");
    auto* part     = pso_plain(dev, lib, "argmax_partial");
    auto* red      = pso_plain(dev, lib, "argmax_reduce");
    if (!part_icb || !red_icb || !part || !red) {
        std::fprintf(stderr, "missing argmax PSOs\n"); return 3;
    }

    // Qwen3-0.6B vocab size.
    constexpr std::uint32_t V = 151936;
    constexpr std::uint32_t ELTS_PER_TG = 16384u;
    const std::uint32_t n_blocks = (V + ELTS_PER_TG - 1u) / ELTS_PER_TG;

    auto* b_logits = dev->newBuffer(V * sizeof(std::uint16_t),
                                    MTL::ResourceStorageModeShared);
    auto* b_val    = dev->newBuffer(n_blocks * sizeof(float),
                                    MTL::ResourceStorageModeShared);
    auto* b_idx    = dev->newBuffer(n_blocks * sizeof(int),
                                    MTL::ResourceStorageModeShared);
    auto* b_out_icb = dev->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    auto* b_out_ref = dev->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    auto* b_args   = dev->newBuffer(2 * sizeof(std::uint32_t),
                                    MTL::ResourceStorageModeShared);

    auto* logits = static_cast<std::uint16_t*>(b_logits->contents());
    // Generic distribution + a single salient peak we know the answer to.
    const std::uint32_t peak = 13048;  // matches the task brief's smoke-test id
    for (std::uint32_t i = 0; i < V; ++i) {
        // tiny pseudo-random pattern; not important — just deterministic.
        float v = static_cast<float>((i * 1103515245u + 12345u) & 0xffffu) / 65535.f - 0.5f;
        logits[i] = f32_to_f16(v);
    }
    logits[peak] = f32_to_f16(10.f);

    auto* args = static_cast<std::uint32_t*>(b_args->contents());
    args[0] = V;
    args[1] = n_blocks;

    // ── ICB tail path ────────────────────────────────────────────────────
    auto* rec = sk::silicon::IcbRecorder::create(dev, /*slots=*/2, /*max_bind=*/4);
    if (!rec) { std::fprintf(stderr, "recorder create failed\n"); return 4; }
    {
        const MTL::Buffer* bufs[4] = { b_logits, b_val, b_idx, b_args };
        NS::UInteger offs[4] = { 0, 0, 0, 0 };
        rec->record(0, part_icb, bufs, offs, 4,
                    MTL::Size(n_blocks, 1, 1), MTL::Size(1024, 1, 1),
                    /*barrier_before=*/false);
    }
    {
        const MTL::Buffer* bufs[4] = { b_val, b_idx, b_out_icb, b_args };
        NS::UInteger offs[4] = { 0, 0, 0, sizeof(std::uint32_t) };  // n_blocks at byte 4
        rec->record(1, red_icb, bufs, offs, 4,
                    MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1),
                    /*barrier_before=*/true);
    }
    rec->mark_resource(b_logits);
    rec->mark_resource(b_val);
    rec->mark_resource(b_idx);
    rec->mark_resource(b_out_icb);
    rec->mark_resource(b_args);

    auto* q = dev->newCommandQueue();
    {
        auto* cmd = q->commandBuffer();
        auto* enc = cmd->computeCommandEncoder();
        rec->execute(enc, 0, 2);
        enc->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();
        if (cmd->status() == MTL::CommandBufferStatusError) {
            std::fprintf(stderr, "ICB cmd error: %s\n",
                cmd->error() ? cmd->error()->localizedDescription()->utf8String() : "?");
            return 5;
        }
    }
    int id_icb = *static_cast<int*>(b_out_icb->contents());

    // ── Reference (non-ICB) 2-pass path ───────────────────────────────────
    {
        auto* cmd = q->commandBuffer();
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(part);
            enc->setBuffer(b_logits, 0, 0);
            enc->setBuffer(b_val,    0, 1);
            enc->setBuffer(b_idx,    0, 2);
            enc->setBytes(&V, 4, 3);
            enc->dispatchThreadgroups(MTL::Size(n_blocks, 1, 1),
                                      MTL::Size(1024, 1, 1));
            enc->endEncoding();
        }
        {
            auto* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(red);
            enc->setBuffer(b_val,     0, 0);
            enc->setBuffer(b_idx,     0, 1);
            enc->setBuffer(b_out_ref, 0, 2);
            enc->setBytes(&n_blocks, 4, 3);
            enc->dispatchThreadgroups(MTL::Size(1, 1, 1),
                                      MTL::Size(1024, 1, 1));
            enc->endEncoding();
        }
        cmd->commit();
        cmd->waitUntilCompleted();
        if (cmd->status() == MTL::CommandBufferStatusError) {
            std::fprintf(stderr, "ref cmd error\n"); return 6;
        }
    }
    int id_ref = *static_cast<int*>(b_out_ref->contents());

    std::fprintf(stderr, "expected_peak=%u  icb_id=%d  ref_id=%d\n",
                 peak, id_icb, id_ref);

    delete rec;
    part_icb->release(); red_icb->release();
    part->release(); red->release();
    b_logits->release(); b_val->release(); b_idx->release();
    b_out_icb->release(); b_out_ref->release(); b_args->release();
    q->release(); lib->release(); dev->release();
    pool->release();

    if (id_icb != id_ref || id_icb != static_cast<int>(peak)) {
        std::fprintf(stderr, "FAIL\n");
        return 7;
    }
    std::fprintf(stderr, "OK\n");
    return 0;
}
