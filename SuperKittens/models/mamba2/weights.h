#ifndef SK_MAMBA2_WEIGHTS_H
#define SK_MAMBA2_WEIGHTS_H

#include "launcher.h"

namespace sk { class WeightStore; }

#ifdef __cplusplus
extern "C" {
#endif

// Load fused buffers from a populated WeightStore.
int sk_mamba2_load_from_store(sk_mamba2_handle* h, sk::WeightStore* store);

// Load directly from a .safetensors file or sharded index.json.
int sk_mamba2_load_safetensors(sk_mamba2_handle* h, const char* path);
int sk_mamba2_load_safetensors_index(sk_mamba2_handle* h, const char* index_json_path);

#ifdef __cplusplus
}
#endif

// HF tensor names (single file model.safetensors for mamba2-130m-hf):
//   backbone.embeddings.weight                                    (V, D)
//   backbone.norm_f.weight                                        (D,)
//   backbone.layers.{L}.norm.weight                               (D,)
//   backbone.layers.{L}.mixer.in_proj.weight                      (IN_OUT, D)  -> transpose
//   backbone.layers.{L}.mixer.conv1d.weight                       (C_in, 1, K) -> reshape to (K, C_in)
//   backbone.layers.{L}.mixer.conv1d.bias                         (C_in,)
//   backbone.layers.{L}.mixer.dt_bias                             (H,)
//   backbone.layers.{L}.mixer.A_log                               (H,)
//   backbone.layers.{L}.mixer.D                                   (H,)
//   backbone.layers.{L}.mixer.norm.weight                         (E,)
//   backbone.layers.{L}.mixer.out_proj.weight                     (D, E) -> transpose to (E, D)
// IN_OUT = 2*E + 2*G*N + H = 2*1536 + 2*1*128 + 24 = 3328
// lm_head is tied (uses embedding weight transposed).

#endif
