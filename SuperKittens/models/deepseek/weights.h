#ifndef SK_DEEPSEEK_WEIGHTS_H
#define SK_DEEPSEEK_WEIGHTS_H

#include "launcher.h"

namespace sk { class WeightStore; }

#ifdef __cplusplus
extern "C" {
#endif

int sk_deepseek_load_from_store(sk_deepseek_handle* h, sk::WeightStore* store);
int sk_deepseek_load_gguf(sk_deepseek_handle* h, const char* path);

#ifdef __cplusplus
}
#endif
#endif
