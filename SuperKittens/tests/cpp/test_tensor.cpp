//
//  test_tensor.cpp — verify SKTensor alloc/free/copy
//
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include "../../inference/runtime/tensor.h"
#include "../../inference/runtime/dispatch.h"

using namespace sk::runtime;

int main() {
    int passed = 0, failed = 0;
    auto check = [&](bool ok, const char* name) { ok ? passed++ : failed++; printf("  %s: %s\n", ok?"PASS":"FAIL", name); };

    printf("=== tensor tests ===\n");
    auto* dev = MTL::CreateSystemDefaultDevice();

    // 1. alloc
    auto t = tensor_alloc_impl(dev, 512, DType::f16, 1024);
    check(t.numel() == 512*1024, "numel 512×1024");
    check(t.nbytes() == 512*1024*2, "nbytes fp16");
    check(t.ndim == 2, "ndim=2");
    check(t.shape[0] == 512 && t.shape[1] == 1024, "shape");
    t.buf->release();

    // 2. 3D tensor
    auto t3 = tensor_alloc_impl(dev, 8, DType::f16, 2048, 64);
    check(t3.ndim == 3, "ndim=3");
    check(t3.numel() == 8*2048*64, "numel 3D");
    t3.buf->release();

    // 3. CPU ↔ GPU round-trip
    auto t4 = tensor_alloc_impl(dev, 4, DType::f16, 16);
    std::vector<__fp16> src(64);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    for (auto& x : src) x = __fp16(nd(rng));
    std::memcpy(t4.buf->contents(), src.data(), 64*2);

    std::vector<__fp16> dst(64);
    std::memcpy(dst.data(), t4.buf->contents(), 64*2);
    bool match = true;
    for (int i = 0; i < 64; i++) if (float(src[i]) != float(dst[i])) match = false;
    check(match, "CPU→GPU→CPU roundtrip");
    t4.buf->release();

    // 4. extern C API
    void* h = sk_tensor_alloc(8, 16, 1, 1, (int)DType::f16);
    check(h != nullptr, "sk_tensor_alloc");
    auto numel = sk_tensor_numel(h);
    check(numel == 128, "sk_tensor_numel=128");
    sk_tensor_free(h);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
