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

// GGUF loader (Q8_0 + F16/BF16/F32 norms supported). Sets per-projection dtype
// fields on the handle so the dispatch path routes M=1 GEMMs to q8_0_matvec.
int sk_qwen_load_gguf(sk_qwen_handle* h, const char* path);

// Pipeline-parallel resident-slice GGUF loader (additive; the full loader above
// delegates here with the full range). Resident-allocates + loads ONLY layers
// [start_layer, end_layer); loads the embedding when with_embed (stage 0) and
// the final norm + LM head when with_head (last stage). Layers outside the range
// keep null weight buffers, so a model that OOMs one device fits across two:
// each stage holds only its layer slice. Tied heads (Qwen3-0.6B) reuse the
// embedding, so with_head implies loading the embedding even when with_embed=0.
int sk_qwen_load_gguf_range(sk_qwen_handle* h, const char* path,
                            uint32_t start_layer, uint32_t end_layer,
                            int with_embed, int with_head);

#ifdef __cplusplus
}
#endif
#endif
