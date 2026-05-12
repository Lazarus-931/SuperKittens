#ifndef SK_QWEN_WEIGHTS_H
#define SK_QWEN_WEIGHTS_H

#include "launcher.h"

namespace sk { class WeightStore; }

#ifdef __cplusplus
extern "C" {
#endif

int sk_qwen_load_from_store(sk_qwen_handle* h, sk::WeightStore* store);

int sk_qwen_load_safetensors(sk_qwen_handle* h, const char* path);
int sk_qwen_load_safetensors_index(sk_qwen_handle* h, const char* index_json_path);

#ifdef __cplusplus
}
#endif
#endif
