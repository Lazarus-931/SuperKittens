//
//  mamba.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/6/26.
//

#include "mamba_impl.h"

using namespace superkittens::mamba;


struct fwd_kernel_traits {
    const int k_threads;
    const int k_items;
    
    
};

template<typename T, int N_THREADS, int N_ITEMS, int N_ROWS>
[[kernel, max_total_threads_per_threadgroup(N_THREADS)]]
struct bwd_kernel_traits {
    
};



template< typename T>
void set_fwd_params(FwdParams param,
                    /// constants
                    const size_t b,
                    const size_t d,
                    const size_t seq,
                    const size_t dstate,
                    const size_t n_groups,
                    const size_t n_chunks,
                    /// ptr's
                    device const T* U,
                    device const T* delta,
                    device const T* A,
                    device const T* B,
                    device const T* C,
                    device const T* out,
                    device const T* z,
                    device const T* out_z,
                    ) {
    
}
                    



