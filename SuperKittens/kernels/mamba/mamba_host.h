//
//  mamba_host.h
//  SuperKittens
//
//  Host-side param setup for Mamba selective scan kernels.
//  Include from .cpp files only (not .metal).
//

#ifndef SUPERKITTENS_MAMBA_HOST_H
#define SUPERKITTENS_MAMBA_HOST_H

#include "mamba_impl.h"

namespace superkittens {
namespace mamba {

// Tensor layout: [batch, dim, seq] for u, delta, and out
//                [dim, d_state] for A
//                [batch, n_groups, seq, d_state] for B, C (variable)
//                [dim, d_state] for B, C (fixed)

inline FwdParams set_fwd_params(
    int batch, int dim, int seq, int d_state, int n_groups,
    int n_chunks, bool has_z)
{
    FwdParams p = {};

    p.batch    = batch;
    p.dim      = dim;
    p.seq      = seq;
    p.d_state  = d_state;
    p.n_groups = n_groups;
    p.n_chunks = n_chunks;
    p.has_z    = has_z;
    p.dim_ngroups_ratio = dim / n_groups;

    // u, delta, out: [batch, dim, seq]
    p.u_d          = seq;
    p.u_batch      = dim * seq;
    p.delta_d      = seq;
    p.delta_batch  = dim * seq;
    p.out_d        = seq;
    p.out_batch    = dim * seq;

    // A: [dim, d_state]
    p.A_d          = d_state;
    p.A_d_state    = 1;

    // B, C (variable): [batch, n_groups, seq, d_state]
    p.B_b_state    = 1;
    p.B_d          = d_state;
    p.B_group      = seq * d_state;
    p.B_batch      = n_groups * seq * d_state;
    p.C_dstate     = 1;
    p.C_d          = d_state;
    p.C_group      = seq * d_state;

    // z, out_z: [batch, dim, seq]
    p.z_d          = seq;
    p.z_batch      = dim * seq;
    p.out_z_d      = seq;
    p.out_z_batch  = dim * seq;

    return p;
}

inline BwdParams set_bwd_params(
    int batch, int dim, int seq, int d_state, int n_groups,
    int n_chunks, bool has_z)
{
    BwdParams p = {};

    p.batch    = batch;
    p.dim      = dim;
    p.seq      = seq;
    p.d_state  = d_state;
    p.n_groups = n_groups;
    p.n_chunks = n_chunks;
    p.has_z    = has_z;

    // Same strides as fwd
    p.u_d          = seq;
    p.u_batch      = dim * seq;
    p.delta_d      = seq;
    p.delta_batch  = dim * seq;
    p.out_d        = seq;
    p.out_batch    = dim * seq;

    p.A_d          = d_state;
    p.A_d_state    = 1;

    p.B_b_state    = 1;
    p.B_d          = d_state;
    p.B_group      = seq * d_state;
    p.B_batch      = n_groups * seq * d_state;
    p.C_dstate     = 1;
    p.C_d          = d_state;
    p.C_group      = seq * d_state;

    p.z_d          = seq;
    p.z_batch      = dim * seq;
    p.out_z_d      = seq;
    p.out_z_batch  = dim * seq;

    return p;
}

} // namespace mamba
} // namespace superkittens

#endif // SUPERKITTENS_MAMBA_HOST_H
