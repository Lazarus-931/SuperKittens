//
//  params.h
//  SuperKittens — attention kernel parameters
//
//  By Alazar Manakelew

#ifndef SUPERKITTENS_ATTN_PARAMS_H
#define SUPERKITTENS_ATTN_PARAMS_H

namespace superkittens {
namespace attn {

struct Params {
    unsigned int seq;
    unsigned int head_dim;
    unsigned int num_heads;
    bool causal;
};


} // namespace attn
} // namespace superkittens

#endif // SUPERKITTENS_ATTN_PARAMS_H
