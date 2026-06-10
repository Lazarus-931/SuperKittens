# Phi-4-reasoning (14B) port — STATUS

**Verdict: PORTED-COHERENT** (config-only adapter over the shared dense core +
a one-time offline GGUF repack; ZERO kernel / launcher / loader edits).

Branch: `dev-sk-phi4` (based on local `main` @ bf05198, which includes the
fp16-GEMM row-grid fix — gate 3 below doubles as its regression check).
Bench host: amelia (M4 mini 16 GB, CLT-only, colima resident — NOT the
canonical lexie protocol).

## Config (verified from GGUF header + HF config.json, not memory)

| field | value | source |
|---|---|---|
| arch | `phi3` (Phi3ForCausalLM) | `general.architecture` |
| n_layers | 40 | `phi3.block_count` |
| d_model | 5120 | `phi3.embedding_length` |
| n_heads / n_kv_heads | 40 / 10 (GQA 4:1) | `phi3.attention.head_count{,_kv}` |
| head_dim | 128 (= core's only supported dim) | 5120/40; `phi3.rope.dimension_count` |
| partial rotary | **NONE** — `rope.dimension_count=128` = full | GGUF header |
| rope | plain NeoX (type-2), theta=500000, `rope_scaling=null` | GGUF + config.json |
| rope_interleaved | 0 (phi3 GGUFs are NOT q/k-permuted; only llama-arch gets the type-0 permute) | llama.cpp convert behavior |
| n_int (FFN) | 17920 | `phi3.feed_forward_length` |
| vocab | 100352 (tiktoken-style GPT-lineage) | token_embd dims |
| eps | 1e-5 | `phi3.attention.layer_norm_rms_epsilon` |
| qkv bias / qk-norm / sliding window | none / none / 0 | config.json + GGUF |
| LM head | UNTIED `output.weight`, Q6_K → native q6k_matvec | GGUF |
| BOS / EOS / pad | 100257 `<|endoftext|>` / 100265 `<|im_end|>` (+100257 stop) / 100349 `<|dummy_85|>` | tokenizer_config.json |
| chat format | `<|im_start|>role<|im_sep|>…<|im_end|>` + hardcoded reasoning system preamble | tokenizer_config chat_template |
| BOS-prepend | not needed (GPT-lineage; template starts at `<|im_start|>`) | — |

Quant mix (bartowski Q4_K_M, imatrix): attn_qkv Q5_K (uniform), attn_output
Q4_K, ffn gate/up Q4_K, ffn_down Q4_K/Q6_K per-layer mix, embed Q4_K
(host-dequant fp16), head Q6_K. All dtypes already supported
(Q5_K is an in-tree fit-enabler).

## The one structural wrinkle: fused GGUF tensors → offline repack

phi3-arch GGUFs fuse `blk.N.attn_qkv.weight` ([Q;K;V] row-concat) and
`blk.N.ffn_up.weight` ([gate;up], 2*n_ff rows). The shared loader
(`sk_qwen_load_gguf`) wants separate `attn_q/k/v` + `ffn_gate/ffn_up`.
Rather than teach the core a fused path (out of scope for a breadth port),
`SuperKittens/models/phi4/repack_phi3_gguf.py` rewrites the GGUF once:
pure row-range byte split (K-quant rows are independent → bit-exact, no
dequant/requant; HF order is q,k,v and gate,up). 243 → 363 tensors,
+7 KB file size. Validated: md5 byte-identity of every split range for
layers 0/20/39 + embed/head/down spot checks, and `gguf` pkg re-parse
(363 tensors, all expected names present) — PASSED.

Artifact on amelia: `~/phi4-gguf/microsoft_Phi-4-reasoning-Q4_K_M-sk.gguf`
(9.05 GB). The fused original was deleted for disk headroom; regenerate via
`curl -L` of `bartowski/microsoft_Phi-4-reasoning-GGUF` Q4_K_M + one repack run
(~3 min total). Sidecar files in the same dir: config.json, tokenizer.json,
tokenizer_config.json (from `microsoft/Phi-4-reasoning`).

## What was added (all Python, zero core edits)

- `SuperKittens/models/phi4/phi4.py` — `Phi4(DenseDecoder)`, the Mistral/Yi
  pattern: `use_qk_norm=0`, `rope_interleaved=0`; base RoPE bake correct as-is.
- `SuperKittens/models/phi4/repack_phi3_gguf.py` — one-time GGUF repack tool.
- `inference/registry.py` — `"phi4-reasoning"` ModelSpec row.
- `models/load/tokenizer/tokenizer.py` — `"phi4"` family specials row.
- `models/load/tokenizer/chat_templates.py` — `phi4_template`
  (`<|im_sep|>` ChatML variant + official reasoning system preamble).

## Gates

1. **Load + config print — PASS.** Loads the repacked GGUF in 13.4s; printed
   config matches the table above; tokenizer resolves
   bos=100257 / eos=100265 / eos_ids={100257,100265} / pad=100349.
2. **Coherence (greedy) — PASS.** Finite logits on all runs.
   - raw "Generate a poem about pizza dough:" (64 tok): fluent constraint-style
     continuation (instruct model completing without template — expected style).
   - chat-templated poem (96 tok): fluent `<think>` planning scaffold.
   - chat-templated reasoning ("train 60 miles in 40 min, how far in 2h?",
     96 tok): correct reasoning — converts units, derives 1.5 mi/min ✓.
3. **No-regression A/B (Qwen3-1.7B-Q8, 32-tok greedy, same prompt) — PASS.**
   dev-sk-phi4 build (bf05198+adapter) vs pristine origin/main (3491001)
   build: token-identical (see `lab_qwen_{A,B}.log`). Also covers the
   row-grid fix regression check.
4. **Decode tok/s — RECORDED** (2 warmup + 5 reps, median, 0.3 s gaps,
   caffeinate -is; amelia-with-colima, NOT the canonical lexie protocol):
   - pure decode (per-token forward after prefill): **10.49 tok/s**
     (reps 10.46-10.51 — thermally tight)
   - generate-loop (bench.py convention, ~8-tok prompt + 64 new): 9.79 tok/s
   - first cold run pages in the 9 GB mmap: 4.58 tok/s (warmup only)
   - context: qwen3-14B Q4_K_M is 11.28 tok/s on lexie; phi4 has a slightly
     larger FFN (17920 vs 17408) and this host carries colima.
5. **Batched N=4 lockstep smoke — PASS** (bonus; `batch=4, cache_max=256`,
   `prefill_batched` + 8 `forward_batched` steps, distinct prompts): all ids
   in vocab range, no NaN, all four lanes coherent and prompt-relevant
   ("…100°C at sea level", "…concise way to create lists", …). Minor word
   echo at lane starts is the lockstep equal-length prompt trim, not the model.

## Memory notes (16 GB + colima)

- Pinned `cache_max=512, seq_max=512` (single-stream gates); ONE handle per
  process, no sequential reloads.
- Swap during the 14B runs: 1.18 G → 2.18 G used (total grew to 3 G); below the
  3.5 G back-off line, stable across bench reps and the batched run.

## How to run (amelia)

```sh
cd ~/sk-phi4-port && source skenv.sh   # SK_DYLIB + SK_METAL_SRC_FALLBACK + PYTHONPATH
python3 lab_phi4_gates12.py            # load + coherence
python3 lab_phi4_bench.py              # decode tok/s
# A/B: SK_ROOT=$HOME/sk-phi4-port/base_main SK_DYLIB=$PWD/build/libsk_base.dylib \
#      source skenv.sh && python3 lab_qwen_identity.py
```
