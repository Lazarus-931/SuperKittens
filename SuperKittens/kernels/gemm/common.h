
#include "../../../../metal-cpp/Foundation/Foundation.hpp"
#include "../../../../metal-cpp/Metal/Metal.hpp"
#include <cstdint>


struct GEMM_params {
    const int MB;
    const int NB;
    const int DB;
    bool tranpose_a;
    bool tranpose_b;
};