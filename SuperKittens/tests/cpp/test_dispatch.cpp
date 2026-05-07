//
//  test_dispatch.cpp — verify GPU dispatch with tensor handles
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

static float l2_rel(const std::vector<__fp16>& g, const std::vector<float>& r) {
    double n=0,d=0;
    for(size_t i=0;i<r.size();i++){float e=float(g[i])-r[i];n+=e*e;d+=r[i]*r[i];}
    return float(sqrt(n)/(sqrt(d)+1e-12));
}

int main() {
    int passed=0,failed=0;
    auto chk=[&](bool ok,const char* s){ok?passed++:failed++;printf("  %s: %s\n",ok?"PASS":"FAIL",s);};

    printf("=== dispatch tests ===\n");

    // Test: gelu dispatch on GPU tensors
    const uint32_t rows=8, cols=64, n=rows*cols;

    // CPU reference
    std::mt19937 rng(42);
    std::vector<__fp16> x_cpu(n);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    for(auto& v:x_cpu) v=__fp16(nd(rng));

    std::vector<float> ref(n);
    for(size_t i=0;i<n;i++){
        float v=float(x_cpu[i]);
        float a=0.044715f*v*v*v;
        ref[i]=0.5f*v*(1.0f+tanhf(0.79788456f*(v+a)));
    }

    // Allocate GPU tensors
    void* xh = sk_tensor_alloc(rows, cols, 1, 1, 0); // f16
    void* yh = sk_tensor_alloc(rows, cols, 1, 1, 0);
    sk_tensor_from_cpu(xh, x_cpu.data());

    // Dispatch gelu
    int ret = sk_dispatch_elementwise("gelu", xh, yh, nullptr);
    chk(ret == 0, "dispatch gelu");

    // Read back
    std::vector<__fp16> y_cpu(n);
    sk_tensor_to_cpu(yh, y_cpu.data());

    float l2 = l2_rel(y_cpu, ref);
    chk(l2 < 0.01, "accuracy l2_rel < 0.01");

    // SiLU test
    for(size_t i=0;i<n;i++){float v=float(x_cpu[i]);ref[i]=v/(1.0f+expf(-v));}
    sk_tensor_from_cpu(xh, x_cpu.data());
    ret = sk_dispatch_elementwise("silu", xh, yh, nullptr);
    chk(ret == 0, "dispatch silu");
    sk_tensor_to_cpu(yh, y_cpu.data());
    l2 = l2_rel(y_cpu, ref);
    chk(l2 < 0.01, "silu accuracy");

    // ReLU test
    for(size_t i=0;i<n;i++) ref[i]=fmaxf(0.0f, float(x_cpu[i]));
    sk_tensor_from_cpu(xh, x_cpu.data());
    ret = sk_dispatch_elementwise("relu", xh, yh, nullptr);
    chk(ret == 0, "dispatch relu");
    sk_tensor_to_cpu(yh, y_cpu.data());
    l2 = l2_rel(y_cpu, ref);
    chk(l2 < 1e-6, "relu accuracy");

    sk_tensor_free(xh);
    sk_tensor_free(yh);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
