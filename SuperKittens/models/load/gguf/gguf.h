#ifndef SK_MODELS_LOAD_GGUF_H
#define SK_MODELS_LOAD_GGUF_H

#include "../../../inference/weight_store.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sk::gguf {

enum ValueType : uint32_t {
    V_U8 = 0, V_I8 = 1, V_U16 = 2, V_I16 = 3,
    V_U32 = 4, V_I32 = 5, V_F32 = 6, V_BOOL = 7,
    V_STRING = 8, V_ARRAY = 9,
    V_U64 = 10, V_I64 = 11, V_F64 = 12,
};

struct MetaValue {
    uint32_t    type   = 0;
    uint64_t    arr_type = 0;
    uint64_t    arr_len  = 0;
    uint64_t    raw_pos  = 0;
    std::string str;
    union {
        uint64_t u64;
        int64_t  i64;
        double   f64;
    } scalar = { 0 };
};

struct TensorInfo {
    std::string          name;
    uint32_t             ggml_type   = 0;
    std::vector<int64_t> shape;
    uint64_t             nbytes      = 0;
    uint64_t             abs_offset  = 0;
    uint64_t             elements    = 0;
    bool                 supported   = false;
    Dtype                dtype       = Dtype::F32;
};

struct Model {
    uint32_t   version       = 0;
    uint64_t   n_kv          = 0;
    uint64_t   n_tensors     = 0;
    uint64_t   alignment     = 32;
    uint64_t   tensor_data_pos = 0;
    const uint8_t* map_base  = nullptr;
    size_t     map_size      = 0;

    std::unordered_map<std::string, MetaValue> meta;
    std::vector<TensorInfo>                    tensors;
};

int load_gguf(const char* path, WeightStore& store, Model* out_model = nullptr);

int parse_gguf(const char* path, Model& out);

const MetaValue* meta_find(const Model& m, const std::string& key);
bool meta_u32(const Model& m, const std::string& key, uint32_t* out);
bool meta_u64(const Model& m, const std::string& key, uint64_t* out);
bool meta_f32(const Model& m, const std::string& key, float* out);
bool meta_string(const Model& m, const std::string& key, std::string* out);

}  // namespace sk::gguf

#endif
