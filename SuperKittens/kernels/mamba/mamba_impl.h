//
//  mamba_impl.h
//  SuperKittens
//
//  Shared param structs for Mamba kernels.
//  Included from both .metal (GPU) and .cpp (host) files.
//

#ifndef SUPERKITTENS_MAMBA_IMPL_H
#define SUPERKITTENS_MAMBA_IMPL_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using stride_t = ulong;    // Metal: ulong = 64-bit unsigned
#else
#include <cstdint>
using stride_t = uint64_t; // C++: uint64_t
#endif

namespace meow {
namespace mamba {

// ── Mamba-1 (sequential selective scan) ──────────────────────

struct FwdParams {
    stride_t A_d, A_d_state;
    stride_t B_batch, B_d, B_b_state, B_group;
    stride_t C_d, C_dstate, C_group;
    stride_t u_batch, u_d;
    stride_t delta_batch, delta_d;
    stride_t z_batch, z_d;
    stride_t out_batch, out_d;
    stride_t out_z_batch, out_z_d;

    int batch, dim, seq, d_state, n_groups, n_chunks;
    bool has_z;
    int dim_ngroups_ratio;
};

struct BwdParams {
    stride_t A_d, A_d_state;
    stride_t B_batch, B_d, B_b_state, B_group;
    stride_t C_d, C_dstate, C_group;
    stride_t u_batch, u_d;
    stride_t delta_batch, delta_d;
    stride_t z_batch, z_d;
    stride_t out_batch, out_d;
    stride_t out_z_batch, out_z_d;

    int batch, dim, seq, d_state, n_groups, n_chunks;
    bool has_z;
};

// ── Mamba-2 (SSD / chunked matmul) ──────────────────────────
// Q, K: [B, 1, N, D]   V, O: [B, H, N, D]   A: [B, H, N]

#ifdef __METAL_VERSION__
constant constexpr int CHUNK_SIZE = 64;
constant constexpr int HEAD_DIM   = 64;
constant constexpr int N_SIMD     = 4;
constant constexpr int N_THREADS  = N_SIMD * 32;
#else
static constexpr int CHUNK_SIZE = 64;
static constexpr int HEAD_DIM   = 64;
static constexpr int N_SIMD     = 4;
static constexpr int N_THREADS  = N_SIMD * 32;
#endif

struct Mamba2Params {
    int batch, heads, seq, head_dim, n_chunks;

    // Strides (element offsets)
    stride_t qk_batch, qk_seq;         // Q, K
    stride_t v_batch, v_head, v_seq;    // V, O
    stride_t a_batch, a_head;           // A
    stride_t o_batch, o_head;
};

} // namespace mamba
} // namespace meow

#endif // SUPERKITTENS_MAMBA_IMPL_H
