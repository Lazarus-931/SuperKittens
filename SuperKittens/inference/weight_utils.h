#pragma once

#include "weight_store.h"
#include "Metal/Metal.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace sk {

uint16_t fp32_bits_to_fp16(uint32_t f);

void bf16_to_fp16(uint16_t* dst, const uint16_t* src, size_t n);

bool copy_into(MTL::Buffer* dst, size_t dst_off, WeightStore* store,
               const std::string& name, size_t expect_bytes,
               const char* family_tag);

bool copy_transpose_fp16(MTL::Buffer* dst, size_t dst_off, WeightStore* store,
                         const std::string& name, size_t out_rows, size_t out_cols,
                         const char* family_tag);

}
