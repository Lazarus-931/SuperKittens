#ifndef SK_INFERENCE_WEIGHT_STORE_H
#define SK_INFERENCE_WEIGHT_STORE_H

#include <Metal/Metal.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace sk {

enum class Dtype : uint8_t {
    F32, F16, BF16,
    Q2_K, Q4_K, IQ2_XXS,
    Q8_0,
    Q6_K,
    Q5_0,
};

// Bytes required to hold `n_elems` elements of dtype `d`.
// For Q8_0 caller must ensure n_elems % 32 == 0 (one block = 32 elems = 34 bytes).
inline size_t dtype_bytes(Dtype d, size_t n_elems) {
    switch (d) {
        case Dtype::F32:  return n_elems * 4;
        case Dtype::F16:  return n_elems * 2;
        case Dtype::BF16: return n_elems * 2;
        case Dtype::Q8_0: return (n_elems / 32) * 34;
        case Dtype::Q5_0: return (n_elems / 32) * 22;
        case Dtype::Q4_K: return (n_elems / 256) * 144;
        case Dtype::Q6_K: return (n_elems / 256) * 210;
        default:          return n_elems * 2;  // best-effort fallback
    }
}

struct TensorView {
    const void*          data;
    size_t               nbytes;
    Dtype                dtype;
    std::vector<int64_t> shape;
    MTL::Buffer*         buffer;
};

class WeightStore {
public:
    WeightStore();
    ~WeightStore();

    int   map_file(const char* path, const void** out_base = nullptr,
                   size_t* out_size = nullptr);

    MTL::Buffer* add(const std::string& name,
                     const void* src, size_t nbytes,
                     Dtype dtype, std::vector<int64_t> shape,
                     bool zero_copy);

    const TensorView* get(const std::string& name) const;
    MTL::Buffer*      buffer(const std::string& name) const;

    size_t bytes_resident() const { return bytes_resident_; }
    size_t bytes_mapped()   const { return bytes_mapped_;  }

private:
    struct Mapping { int fd; void* base; size_t size; };
    std::vector<Mapping> maps_;

    size_t bytes_resident_ = 0;
    size_t bytes_mapped_   = 0;

    std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace sk

#endif
