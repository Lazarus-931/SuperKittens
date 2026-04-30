//
//  m2_mimo_bwd.metal
//  SuperKittens
//
//


#include "../../../../meow.h"


namespace meow::mamba::mamba3 {
    
    struct M3MimoBwdArgs {
        uint batch;
        uint nheads;
        uint nheads_qk;
        uint seq_len;
        uint n_chunks;
    };
    
    struct Mamba3SisoBwdMainArgs {
        uint batch;
        uint nheads;
        uint seq_len;
        uint n_chunks;
    };
    
    // TODO: to be done, someday..
    
}
