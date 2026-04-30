//
//  params.h
//  SuperKittens — attention kernel parameters
//
//  By Alazar Manakelew
//
//  Host-side parameter struct. Maps to the 7-buffer Metal kernel API:
//    Q, K, V, O, seq, head_dim, num_heads
//  For GQA, add num_kv_heads and compute head mapping in the kernel.

#ifndef SUPERKITTENS_ATTN_PARAMS_H
#define SUPERKITTENS_ATTN_PARAMS_H

namespace meow {
namespace attn {

struct MHA_Params {
    unsigned int seq;
    unsigned int head_dim;
    unsigned int num_heads;
    bool causal;
};

} // namespace attn
} // namespace meow

#endif // SUPERKITTENS_ATTN_PARAMS_H
