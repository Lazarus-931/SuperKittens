//
//  kv_up_pair.c++ — host launcher for the dual K-up/V-up matmul fusion.
//

#include "kv_up_pair.h"
#include "../runtime_bindings.h"
#include <cstring>

extern "C" int sk_kv_up_pair(void* c_kv, void* w_k_up, void* w_v_up,
                             void* k_no_pe, void* v_out,
                             uint32_t T, uint32_t R, uint32_t K_OUT, uint32_t V_OUT) {
    auto* pso = sk::bindings_pso("kv_up_pair");
    if (!pso) return -1;
    auto* dev = sk::bindings_device();

    const size_t cb_sz = (size_t)T * R * 2;
    const size_t wk_sz = (size_t)R * K_OUT * 2;
    const size_t wv_sz = (size_t)R * V_OUT * 2;
    const size_t ko_sz = (size_t)T * K_OUT * 2;
    const size_t vo_sz = (size_t)T * V_OUT * 2;

    auto* bC  = dev->newBuffer(cb_sz, MTL::ResourceStorageModeShared);
    auto* bWk = dev->newBuffer(wk_sz, MTL::ResourceStorageModeShared);
    auto* bWv = dev->newBuffer(wv_sz, MTL::ResourceStorageModeShared);
    auto* bK  = dev->newBuffer(ko_sz, MTL::ResourceStorageModeShared);
    auto* bV  = dev->newBuffer(vo_sz, MTL::ResourceStorageModeShared);
    std::memcpy(bC->contents(),  c_kv,   cb_sz);
    std::memcpy(bWk->contents(), w_k_up, wk_sz);
    std::memcpy(bWv->contents(), w_v_up, wv_sz);

    auto* cmd = sk::bindings_queue()->commandBuffer();
    auto* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(bC,  0, 0); enc->setBuffer(bWk, 0, 1); enc->setBuffer(bWv, 0, 2);
    enc->setBuffer(bK,  0, 3); enc->setBuffer(bV,  0, 4);
    enc->setBytes(&T,     4, 5); enc->setBytes(&R,     4, 6);
    enc->setBytes(&K_OUT, 4, 7); enc->setBytes(&V_OUT, 4, 8);

    // v2 tile-MMA: BM=32, BN=64, 128 threads. Grid: (ceil(max(K,V)/BN), ceil(T/BM)).
    const uint32_t BM = 32, BN = 64;
    const uint32_t max_out = (K_OUT > V_OUT) ? K_OUT : V_OUT;
    enc->dispatchThreadgroups(
        MTL::Size((max_out + BN - 1) / BN, (T + BM - 1) / BM, 1),
        MTL::Size(128, 1, 1));
    enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted();
    std::memcpy(k_no_pe, bK->contents(), ko_sz);
    std::memcpy(v_out,   bV->contents(), vo_sz);

    cmd->release();
    bC->release(); bWk->release(); bWv->release(); bK->release(); bV->release();
    return 0;
}
