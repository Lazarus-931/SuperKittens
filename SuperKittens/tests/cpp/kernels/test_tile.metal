//
//  test_tile.metal — verify tile.h load/store
//

#include "../../../sk/src/cpp/tile.h"

using namespace sk::dsl;

[[host_name("test_tile_identity")]]
[[kernel, max_total_threads_per_threadgroup(64)]]
void test_tile_identity(
    device const half* src [[buffer(0)]],
    device half* dst       [[buffer(1)]],
    constant uint& rows    [[buffer(2)]],
    constant uint& cols    [[buffer(3)]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup Fp16Tile<4, 8> tile;
    load_tile<4, 8, 64>(src, cols, 0, 0, rows, cols, &tile, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    store_tile<4, 8, 64>(dst, cols, 0, 0, rows, cols, &tile, lid);
}

// Budget check: 4KB + 8KB + 8KB = 20KB → under 32KB
[[host_name("test_tile_budget")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void test_tile_budget(device half* dst [[buffer(0)]], uint lid [[thread_index_in_threadgroup]])
{
    threadgroup Fp16Tile<32, 64> as;
    threadgroup Fp16Tile<64, 64> bs;
    threadgroup Fp16Tile<64, 64> cs;
    constexpr int b = tg_budget<as.SizeBytes, bs.SizeBytes, cs.SizeBytes>();
    if (lid == 0) dst[0] = half(float(b) / 1024.0f);
}
