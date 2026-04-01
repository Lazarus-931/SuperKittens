








kernel void fp16_gemm(
    device const half*  A       [[buffer(0)]],
    device const half*  B       [[buffer(1)]],
    device half*        C       [[buffer(2)]],
    constant uint&      M       [[buffer(3)]],
    constant uint&      N       [[buffer(4)]],
    constant uint&      K       [[buffer(5)]],
    uint2 gid                   [[thread_position_in_grid]],
    uint2 tid                   [[thread_position_in_threadgroup]]
) {

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    half acc = 0.0h;
    for (uint k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}