#ifndef SK_MODELS_LOAD_SAFETENSOR_H
#define SK_MODELS_LOAD_SAFETENSOR_H

#include "../../../inference/weight_store.h"
#include <cstdint>
#include <string>
#include <vector>

namespace sk {

struct SafetensorEntry {
    std::string          name;
    Dtype                dtype;
    std::vector<int64_t> shape;
    size_t               nbytes;
};

int load_safetensors(const char* path, WeightStore& store);

int load_safetensors_index(const char* index_json_path, WeightStore& store);

int enumerate_safetensors(const char* path, std::vector<SafetensorEntry>& out);

}  // namespace sk

#endif
