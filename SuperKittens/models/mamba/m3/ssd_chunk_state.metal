//
//  ssd_chunk_state.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/11/26.
//

#include <metal_stdlib>
#include "../../../meow_metal.h"
#include "../../../kernels/mamba/mamba_impl.h"

using namespace meow::mamba;


// will be replaced by centralized struct
struct ChunkCumsumArgs {
    uint batch;
    uint nChunks;
    uint heads;
    uint chunkSize;
    uint hdim;
    uint dstate;
    uint hasD;
    uint hasZ;
    uint seqlen;
    uint nheads;
    uint hasBias;
    uint useSoftplus;
    uint dtMin;
    uint dtMax;
};

kernel void chunk_cumsum_fwd_kernel(
    device const float *dt [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *dtBias [[buffer(2)]],
    device float *dtOut [[buffer(3)]],
    device float *dACumsum [[buffer(4)]],
    constant ChunkCumsumArgs &args [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint BLOCK_H = 8;
    if (tid >= 32 * BLOCK_H) return;

    uint pidHC = group.x;
    uint pidY = group.y;
    uint pidC = pidY / args.batch;
    uint pidB = pidY % args.batch;
    if (pidC >= args.nChunks) return;

    uint localH = tid >> 5;
    uint lane = tid & 31;
    uint h = pidHC * BLOCK_H + localH;
    if (h >= args.nheads) return;

    uint chunkBase = pidC * args.chunkSize;
    uint chunkLimit = min(args.chunkSize, args.seqlen - chunkBase);
    uint c0 = lane * 2;
    uint c1 = c0 + 1;
    uint dtBase = ((pidB * args.seqlen + chunkBase) * args.nheads + h);
    uint outBase = (((pidB * args.nChunks + pidC) * args.nheads + h) * args.chunkSize);
    float bias = (args.hasBias != 0) ? dtBias[h] : 0.0f;
    float a = A[h];
    float dt0 = 0.0f;
    float dt1 = 0.0f;

    if (c0 < chunkLimit) {
        dt0 = dt[dtBase + c0 * args.nheads] + bias;
        if (args.useSoftplus != 0) dt0 = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
        dt0 = metal::min(metal::max(dt0, args.dtMin), args.dtMax);
        
        
    if (c1 < chunkLimit) {
        dt1 = dt[dtBase + c1 * args.nheads] + bias;
        if (args.useSoftplus != 0) dt1 = dt1 <= 20.0f ? log(1.0f + exp(dt1)) : dt1;
        dt1 = metal::min(metal::max(dt1, args.dtMin), args.dtMax);
    }

    dtOut[outBase + c0] = (c0 < chunkLimit) ? dt0 : 0.0f;
    dtOut[outBase + c1] = (c1 < chunkLimit) ? dt1 : 0.0f;

    float dA0 = dt0 * a;
    float dA1 = fma(dt1, a, dA0);
    float total = dA1;

    {
        float n = simd_shuffle_up(total, 1);
        if (lane >= 1) total += n;
        n = simd_shuffle_up(total, 2);
        if (lane >= 2) total += n;
        n = simd_shuffle_up(total, 4);
        if (lane >= 4) total += n;
        n = simd_shuffle_up(total, 8);
        if (lane >= 8) total += n;
        n = simd_shuffle_up(total, 16);
        if (lane >= 16) total += n;
    }

    float carry = (lane == 0) ? 0.0f : simd_shuffle_up(total, 1);
    dACumsum[outBase + c0] = (c0 < chunkLimit) ? (dA0 + carry) : 0.0f;
    dACumsum[outBase + c1] = (c1 < chunkLimit) ? (dA1 + carry) : 0.0f;
}
