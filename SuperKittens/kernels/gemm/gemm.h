
#include "../../../../metal-cpp/Foundation/Foundation.hpp"
#include "../../../../metal-cpp/Metal/Metal.hpp"
#include <cstdint>


namespace superkittens;

template < int BM, int BN, int BK, int Z, int Y, bool transpose>
kernel void gemm(
                 const device T* A [[buffer(0)]],
                 const device T* B [[buffer(1)]],
                 const device T* C [[buffer(2)]],
                 const GEMMParam* gemm_param [[buffer(4)]]) {
    
    
}

