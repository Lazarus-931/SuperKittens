# V4 Flash deltas — deferred

Reference: `~/ds4/ds4.c` lines 1250-1262 (metadata keys), 1459 (E4M3
dequant), 1489 (FP8 KV quantize), 1509 (Q8_K activation quant for int2 dot).

## 1. Sliding window (per-layer)
- Per-layer flag `sliding_window_radius` in GGUF metadata
  (`attention.sliding_window`).
- Implementation: extend `causal_mask_fill` kernel to also zero out positions
  with `q_pos - k_pos > radius`. Add a second arg `radius` (0 = global).
- Plumbing: per-layer field on `LayerParams`; loader reads from GGUF.

## 2. Sparse-attention indexer
- New kernel `dsv4_indexer.metal`: computes
  `topk(q_idx @ K_idx.T, k=indexer.top_k)` over the cached K projected by an
  `indexer_q/k` head (head_count, key_length in GGUF metadata
  `attention.indexer.*`).
- Output: per-query a bitmask of allowed K positions. Feed to FA as an
  additive mask alongside causal mask.
- Sized at: head_count ≈ 4, key_length ≈ 128, top_k ≈ 1024 per ds4.c metadata.

## 3. FP8 E4M3 KV cache
- Replace `c_kv_cache` / `k_pe_cache` fp16 buffers with packed uint8 (E4M3) +
  per-(layer, kv_head) fp16 scale.
- New kernels:
  - `kv_quantize_e4m3.metal` (write path; ds4.c:1489 reference).
  - `kv_dequantize_e4m3.metal` (read inside FA, or a separate decode step
    before flash_attn_vec).
- Cleaner: extend flash_attn_vec to read E4M3 K/V directly with the scale
  buffer; same loop, just decode-on-load. Requires FA kernel API change —
  flag in PROMOTE.md.

## 4. MTP head at layer 61
- Last layer in V3 (`num_hidden_layers=61`) has the MTP slot ignored
  (modeling_deepseek_v3.py:547). In V4 it carries a multi-token-prediction
  head: a small transformer block + LM-head-shaped projection that predicts
  token t+2.
- Inference impact: optional. v0 can skip it without affecting greedy decode.

## 5. IQ2_XXS / Q2_K expert weights
- `MoeQuant::IQ2_XXS` and `Q2_K` enum values exist; need to verify
  `swiglu_pair_iq2xxs.metal` and `down_scatter_q2k.metal` actually decode
  the per-block sub-byte layout from ds4.c (`block_iq2_xxs`, `block_q2_K`).
- Activation must be Q8_K quantized for the int2 dot (ds4.c:1509). New kernel
  `quantize_q8k.metal` needed if not already present.
