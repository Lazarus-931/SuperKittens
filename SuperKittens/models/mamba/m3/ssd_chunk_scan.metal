//
//  ssd_chunk_scan.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/11/26.
//

/*
 SSD chunk scan forward
 */

#include "../../../meow_metal.h"
#include "../../../kernels/mamba/mamba_impl.h"

using namespace meow::mamba;

///////// SSD Chunk Scan Arg
struct SSDChunkScanArgs {
    uint batch;
    uint nChunks;
    uint heads;
    uint chunkSize;
    uint hdim;
    uint dstate;
    uint hasD;
    uint hasZ;
};

#include "../../../kernels/mamba/tools.h"
#include "../../../kernels/mamba/index.h"


kernel void ssd_chunk_scan_fwd_kernel(
    device const float *cb [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *z [[buffer(2)]],
    device float *out [[buffer(3)]],
    device float *outX [[buffer(4)]],
    device const float *dt [[buffer(5)]],
    device const float *dA [[buffer(6)]],
    device const float *C [[buffer(7)]],
    device const float *prev [[buffer(8)]],
    device const float *D [[buffer(9)]],
    constant SSDChunkScanArgs &args [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint TILE = 64;
    constexpr uint THREADS = 256;
    constexpr uint ROWS_PER_THREAD = 4;
    constexpr uint COLS_PER_THREAD = 4;
    static_assert(ROWS_PER_THREAD * COLS_PER_THREAD * THREADS == TILE * TILE);

    threadgroup half cbTile[TILE * TILE];
    threadgroup half xTile[TILE * TILE];

    uint pidC = group.y / args.batch;
    uint pidB = group.y % args.batch;
    uint pidH = group.z;
    if (pidC >= args.nChunks || pidH >= args.heads || tid >= THREADS) return;

    for (uint idx = tid; idx < TILE * TILE; idx += THREADS) {
        uint m = idx / TILE;
        uint k = idx % TILE;
        cbTile[idx] = half(cb[meow::mamba::index::ssd_best_cb_index(args, pidB, pidC, pidH, m, k)]);
        xTile[idx] = half(x[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, m, k)]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint blockRow = tid / 16;
    uint blockCol = tid % 16;
    uint rowBase = blockRow * ROWS_PER_THREAD;
    uint colBase = blockCol * COLS_PER_THREAD;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    for (uint ds = 0; ds < TILE; ds += 4) {
        for (uint lane = 0; lane < 4; ++lane) {
            float c0 = C[meow::mamba::index::ssd_c_index(args, pidB, pidC, pidH, rowBase + 0, ds + lane)];
            float c1 = C[meow::mamba::index::ssd_c_index(args, pidB, pidC, pidH, rowBase + 1, ds + lane)];
            float c2 = C[meow::mamba::index::ssd_c_index(args, pidB, pidC, pidH, rowBase + 2, ds + lane)];
            float c3 = C[meow::mamba::index::ssd_c_index(args, pidB, pidC, pidH, rowBase + 3, ds + lane)];
            float4 p = float4(
                prev[meow::mamba::index::ssd_prev_index(args, pidB, pidC, pidH, colBase + 0, ds + lane)],
                prev[meow::mamba::index::ssd_prev_index(args, pidB, pidC, pidH, colBase + 1, ds + lane)],
                prev[meow::mamba::index::ssd_prev_index(args, pidB, pidC, pidH, colBase + 2, ds + lane)],
                prev[meow::mamba::index::ssd_prev_index(args, pidB, pidC, pidH, colBase + 3, ds + lane)]
            );
            acc0 += c0 * p;
            acc1 += c1 * p;
            acc2 += c2 * p;
            acc3 += c3 * p;
        }
    }

    float dA0 = dA[meow::mamba::index::ssd_vec_index(args, pidB, pidC, pidH, rowBase + 0)];
    float dA1 = dA[meow::mamba::index::ssd_vec_index(args, pidB, pidC, pidH, rowBase + 1)];
    float dA2 = dA[meow::mamba::index::ssd_vec_index(args, pidB, pidC, pidH, rowBase + 2)];
    float dA3 = dA[meow::mamba::index::ssd_vec_index(args, pidB, pidC, pidH, rowBase + 3)];
    float scales[4] = {
        fast::exp(dA0),
        fast::exp(dA1),
        fast::exp(dA2),
        fast::exp(dA3),
    };
    acc0 *= scales[0];
    acc1 *= scales[1];
    acc2 *= scales[2];
    acc3 *= scales[3];

    for (uint k = 0; k < TILE; ++k) {
        float4 xVec = float4(
            float(xTile[k * TILE + colBase + 0]),
            float(xTile[k * TILE + colBase + 1]),
            float(xTile[k * TILE + colBase + 2]),
            float(xTile[k * TILE + colBase + 3])
        );
        float dAk = dA[meow::mamba::index::ssd_vec_index(args, pidB, pidC, pidH, k)];
        float dtk = dt[meow::mamba::index::ssd_vec_index(args, pidB, pidC, pidH, k)];
        float decay0 = fast::exp(min(dA0 - dAk, 0.0f)) * dtk;
        float decay1 = fast::exp(min(dA1 - dAk, 0.0f)) * dtk;
        float decay2 = fast::exp(min(dA2 - dAk, 0.0f)) * dtk;
        float decay3 = fast::exp(min(dA3 - dAk, 0.0f)) * dtk;
        acc0 += float(cbTile[(rowBase + 0) * TILE + k]) * decay0 * xVec;
        acc1 += float(cbTile[(rowBase + 1) * TILE + k]) * decay1 * xVec;
        acc2 += float(cbTile[(rowBase + 2) * TILE + k]) * decay2 * xVec;
        acc3 += float(cbTile[(rowBase + 3) * TILE + k]) * decay3 * xVec;
    }

    if (args.hasD != 0) {
        float4 dVec = float4(
            D[pidH * TILE + colBase + 0],
            D[pidH * TILE + colBase + 1],
            D[pidH * TILE + colBase + 2],
            D[pidH * TILE + colBase + 3]
        );
        acc0 += float4(
            float(xTile[(rowBase + 0) * TILE + colBase + 0]),
            float(xTile[(rowBase + 0) * TILE + colBase + 1]),
            float(xTile[(rowBase + 0) * TILE + colBase + 2]),
            float(xTile[(rowBase + 0) * TILE + colBase + 3])
        ) * dVec;
        acc1 += float4(
            float(xTile[(rowBase + 1) * TILE + colBase + 0]),
            float(xTile[(rowBase + 1) * TILE + colBase + 1]),
            float(xTile[(rowBase + 1) * TILE + colBase + 2]),
            float(xTile[(rowBase + 1) * TILE + colBase + 3])
        ) * dVec;
        acc2 += float4(
            float(xTile[(rowBase + 2) * TILE + colBase + 0]),
            float(xTile[(rowBase + 2) * TILE + colBase + 1]),
            float(xTile[(rowBase + 2) * TILE + colBase + 2]),
            float(xTile[(rowBase + 2) * TILE + colBase + 3])
        ) * dVec;
        acc3 += float4(
            float(xTile[(rowBase + 3) * TILE + colBase + 0]),
            float(xTile[(rowBase + 3) * TILE + colBase + 1]),
            float(xTile[(rowBase + 3) * TILE + colBase + 2]),
            float(xTile[(rowBase + 3) * TILE + colBase + 3])
        ) * dVec;
    }

    if (args.hasZ != 0) {
        for (uint j = 0; j < 4; ++j) {
            outX[meow::mamba::index::ssd_out_index(args, pidB, pidC, pidH, rowBase + 0, colBase + j)] = acc0[j];
            outX[meow::mamba::index::ssd_out_index(args, pidB, pidC, pidH, rowBase + 1, colBase + j)] = acc1[j];
            outX[meow::mamba::index::ssd_out_index(args, pidB, pidC, pidH, rowBase + 2, colBase + j)] = acc2[j];
            outX[meow::mamba::index::ssd_out_index(args, pidB, pidC, pidH, rowBase + 3, colBase + j)] = acc3[j];
        }
        float4 z0 = float4(
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 0, colBase + 0)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 0, colBase + 1)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 0, colBase + 2)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 0, colBase + 3)]
        );
        float4 z1 = float4(
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 1, colBase + 0)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 1, colBase + 1)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 1, colBase + 2)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 1, colBase + 3)]
        );
        float4 z2 = float4(
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 2, colBase + 0)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 2, colBase + 1)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 2, colBase + 2)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 2, colBase + 3)]
        );
        float4 z3 = float4(
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 3, colBase + 0)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 3, colBase + 1)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 3, colBase + 2)],
            z[meow::mamba::index::ssd_x_index(args, pidB, pidC, pidH, rowBase + 3, colBase + 3)]
        );
        acc0 *= z0 / (1.0f + fast::exp(-z0));
        acc1 *= z1 / (1.0f + fast::exp(-z1));
        acc2 *= z2 / (1.0f + fast::exp(-z2));
        acc3 *= z3 / (1.0f + fast::exp(-z3));
    }

    for (uint j = 0; j < 4; ++j) {
        out[meow::mamba::index::ssd_out_index(args, pidB, pidC, pidH, rowBase + 0, colBase + j)] = acc0[j];
        out[meow::mamba::index::ssd_out_index(args, pidB, pidC, pidH, rowBase + 1, colBase + j)] = acc1[j];
        out[meow::mamba::index::ssd_out_index(args, pidB, pidC, pidH, rowBase + 2, colBase + j)] = acc2[j];
        out[meow::mamba::index::ssd_out_index(args, pidB, pidC, pidH, rowBase + 3, colBase + j)] = acc3[j];
    }
}


kernel void ssd_chunk_scan_bwd_kernel() {
    
}
