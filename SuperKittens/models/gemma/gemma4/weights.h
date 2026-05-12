#ifndef SK_GEMMA4_WEIGHTS_H
#define SK_GEMMA4_WEIGHTS_H

#include "launcher.h"

namespace sk { class WeightStore; }

#ifdef __cplusplus
extern "C" {
#endif

int sk_gemma4_load_from_store(sk_gemma4_handle* h, sk::WeightStore* store);

int sk_gemma4_load_safetensors(sk_gemma4_handle* h, const char* path);
int sk_gemma4_load_safetensors_index(sk_gemma4_handle* h, const char* index_json_path);

int sk_gemma4_set_rope_tables(sk_gemma4_handle* h,
                              const void* cos_local, const void* sin_local,
                              const void* cos_global, const void* sin_global);

#ifdef __cplusplus
}
#endif
#endif
