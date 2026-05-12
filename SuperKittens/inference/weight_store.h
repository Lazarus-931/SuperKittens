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
};

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
