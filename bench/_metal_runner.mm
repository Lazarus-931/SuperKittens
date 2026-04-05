// Metal runner for attention benchmarks
// Compiled and invoked by test.py
// Usage: ./runner <fused|naive|both> <seq> <d> <iters> [trace_dir]
// Output: kernel_name,median_time_us (one per line)
// If trace_dir is provided, saves .gputrace files there

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;

static id<MTLLibrary> compileFile(const char* filename) {
    NSError* e;
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString* path = [cwd stringByAppendingPathComponent:[NSString stringWithUTF8String:filename]];
    NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&e];
    if (!src) { fprintf(stderr, "Can't read %s\n", filename); exit(1); }

    // strip the #include "types.h" line since we're compiling standalone
    src = [src stringByReplacingOccurrencesOfString:@"#include \"types.h\"" withString:@""];

    auto* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;
    auto* lib = [dev newLibraryWithSource:src options:opts error:&e];
    if (!lib) { fprintf(stderr, "Compile %s: %s\n", filename, e.localizedDescription.UTF8String); exit(1); }
    return lib;
}

static id<MTLComputePipelineState> pso(id<MTLLibrary> lib, const char* name) {
    NSError* e;
    auto* fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "Missing kernel: %s\n", name); exit(1); }
    auto* p = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { fprintf(stderr, "PSO fail: %s\n", e.localizedDescription.UTF8String); exit(1); }
    return p;
}

static double dispatch(id<MTLComputePipelineState> p,
                       void(^enc)(id<MTLComputeCommandEncoder>),
                       MTLSize grid, MTLSize tg, bool threadgroups) {
    auto cb = [q commandBuffer];
    auto e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    enc(e);
    if (threadgroups) [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    else [e dispatchThreads:grid threadsPerThreadgroup:tg];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.GPUEndTime - cb.GPUStartTime;
}

static double median_time(int n, double* times) {
    std::sort(times, times + n);
    return times[n / 2] * 1e6; // seconds -> us
}

static bool startTrace(const char* tracePath) {
    auto* mgr = [MTLCaptureManager sharedCaptureManager];
    if (![mgr supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
        fprintf(stderr, "GPU trace not supported\n");
        return false;
    }
    auto* desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = dev;
    desc.destination = MTLCaptureDestinationGPUTraceDocument;
    desc.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:tracePath]];
    NSError* e = nil;
    bool ok = [mgr startCaptureWithDescriptor:desc error:&e];
    if (!ok) fprintf(stderr, "Trace start failed: %s\n", e.localizedDescription.UTF8String);
    return ok;
}

static void stopTrace() {
    [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc < 5) { fprintf(stderr, "Usage: %s <fused|naive|both> <seq> <d> <iters> [trace_dir]\n", argv[0]); return 1; }

        const char* mode = argv[1];
        uint32_t seq = atoi(argv[2]);
        uint32_t d = atoi(argv[3]);
        int iters = atoi(argv[4]);
        const char* traceDir = (argc > 5) ? argv[5] : nullptr;

        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];

        size_t qkv_bytes = seq * d * sizeof(__fp16);
        auto* bufQ = [dev newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
        auto* bufK = [dev newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
        auto* bufV = [dev newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];
        auto* bufO = [dev newBufferWithLength:qkv_bytes options:MTLResourceStorageModeShared];

        __fp16* pQ = (__fp16*)bufQ.contents;
        __fp16* pK = (__fp16*)bufK.contents;
        __fp16* pV = (__fp16*)bufV.contents;
        srand(42);
        for (size_t i = 0; i < seq * d; i++) {
            pQ[i] = (__fp16)((float)rand() / RAND_MAX * 0.5f);
            pK[i] = (__fp16)((float)rand() / RAND_MAX * 0.5f);
            pV[i] = (__fp16)((float)rand() / RAND_MAX * 0.5f);
        }

        auto* bufSeq = [dev newBufferWithBytes:&seq length:4 options:MTLResourceStorageModeShared];
        auto* bufD = [dev newBufferWithBytes:&d length:4 options:MTLResourceStorageModeShared];

        bool do_fused = (strcmp(mode, "fused") == 0 || strcmp(mode, "both") == 0);
        bool do_naive = (strcmp(mode, "naive") == 0 || strcmp(mode, "both") == 0);

        // ── Fused attention ──
        if (do_fused) {
            auto* lib = compileFile("attention.metal");
            auto* p = pso(lib, "fused_attention");

            uint32_t gx = 1;
            uint32_t gy = (seq + 15) / 16;

            auto enc = ^(id<MTLComputeCommandEncoder> e) {
                [e setBuffer:bufQ offset:0 atIndex:0];
                [e setBuffer:bufK offset:0 atIndex:1];
                [e setBuffer:bufV offset:0 atIndex:2];
                [e setBuffer:bufO offset:0 atIndex:3];
                [e setBuffer:bufSeq offset:0 atIndex:4];
                [e setBuffer:bufD offset:0 atIndex:5];
            };

            for (int i = 0; i < 3; i++)
                dispatch(p, enc, MTLSizeMake(gx, gy, 1), MTLSizeMake(128, 1, 1), true);

            // trace one run
            bool tracing = false;
            if (traceDir) {
                std::string path = std::string(traceDir) + "/fused.gputrace";
                tracing = startTrace(path.c_str());
            }

            double times[100];
            for (int i = 0; i < iters && i < 100; i++)
                times[i] = dispatch(p, enc, MTLSizeMake(gx, gy, 1), MTLSizeMake(128, 1, 1), true);

            if (tracing) stopTrace();
            printf("fused,%.1f\n", median_time(iters, times));
        }

        // ── Naive attention (3 separate kernels) ──
        if (do_naive) {
            auto* lib = compileFile("attn_causal.metal");
            auto* pQK = pso(lib, "naive_qk");
            auto* pSM = pso(lib, "naive_softmax");
            auto* pPV = pso(lib, "naive_pv");

            size_t scores_bytes = seq * seq * sizeof(float);
            auto* bufScores = [dev newBufferWithLength:scores_bytes options:MTLResourceStorageModeShared];
            auto* bufProbs = [dev newBufferWithLength:scores_bytes options:MTLResourceStorageModeShared];

            auto run_naive = ^double() {
                double t = dispatch(pQK, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:bufQ offset:0 atIndex:0];
                    [e setBuffer:bufK offset:0 atIndex:1];
                    [e setBuffer:bufScores offset:0 atIndex:2];
                    [e setBuffer:bufSeq offset:0 atIndex:3];
                    [e setBuffer:bufD offset:0 atIndex:4];
                }, MTLSizeMake(seq, seq, 1), MTLSizeMake(16, 16, 1), false);

                t += dispatch(pSM, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:bufScores offset:0 atIndex:0];
                    [e setBuffer:bufProbs offset:0 atIndex:1];
                    [e setBuffer:bufSeq offset:0 atIndex:2];
                    [e setBuffer:bufD offset:0 atIndex:3];
                }, MTLSizeMake(seq, 1, 1), MTLSizeMake(256, 1, 1), false);

                t += dispatch(pPV, ^(id<MTLComputeCommandEncoder> e) {
                    [e setBuffer:bufProbs offset:0 atIndex:0];
                    [e setBuffer:bufV offset:0 atIndex:1];
                    [e setBuffer:bufO offset:0 atIndex:2];
                    [e setBuffer:bufSeq offset:0 atIndex:3];
                    [e setBuffer:bufD offset:0 atIndex:4];
                }, MTLSizeMake(d, seq, 1), MTLSizeMake(16, 16, 1), false);

                return t;
            };

            for (int i = 0; i < 3; i++) run_naive();

            bool tracing = false;
            if (traceDir) {
                std::string path = std::string(traceDir) + "/naive.gputrace";
                tracing = startTrace(path.c_str());
            }

            double times[100];
            for (int i = 0; i < iters && i < 100; i++)
                times[i] = run_naive();

            if (tracing) stopTrace();
            printf("naive,%.1f\n", median_time(iters, times));
        }
    }
    return 0;
}
