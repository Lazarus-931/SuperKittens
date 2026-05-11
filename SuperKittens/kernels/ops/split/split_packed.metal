//  split_packed.metal — generic packed-tensor split.

#include <metal_stdlib>
using namespace metal;

[[host_name("split_packed")]]
[[kernel, max_total_threads_per_threadgroup(128)]]
void split_packed(
    device const half*  src     [[buffer(0)]],
    device half*        outA    [[buffer(1)]],
    device half*        outB    [[buffer(2)]],
    constant uint&      T       [[buffer(3)]],
    constant uint&      A       [[buffer(4)]],
    constant uint&      B       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint t = gid.y;
    const uint c = gid.x;
    if (t >= T) return;
    const uint tot = A + B;
    if (c >= tot) return;
    half v = src[t * tot + c];
    if (c < A) outA[t * A + c] = v;
    else       outB[t * B + (c - A)] = v;
}
