# quantize-to-gguf

Single-file C quantizer: HuggingFace safetensors -> GGUF Q8_0 (and F16
passthrough). Uses NEON BF16->Q8_0 fast path on Apple Silicon and
`pthread`-parallel work distribution across tensors.

The target architecture is auto-detected from `config.json:architectures[0]`
and dispatched to a per-family tensor-name table and metadata emitter.

## Supported architectures

| HF `architectures[0]`           | GGUF arch key | Status                              |
|---------------------------------|---------------|-------------------------------------|
| `Qwen3ForCausalLM`              | `qwen3`       | validated end-to-end                |
| `Gemma3ForCausalLM` / `Gemma4*` | `gemma3`      | validated on small fixtures         |
| `Mamba2ForCausalLM`             | `mamba2`      | validated on small fixtures         |
| `DeepseekV2ForCausalLM` / `V3*` | `deepseek2`   | code-only (weights too big to test) |

Gemma3 multimodal checkpoints (with the text tower nested under
`language_model.model.*`) are accepted; dims are read from the nested
`text_config` block.

DeepSeek per-expert tensors are emitted as separate GGUF tensors
(`blk.%d.ffn_{gate,up,down}_exps.%d.weight`). The canonical llama.cpp
loader expects experts pre-stacked into one tensor per layer of shape
`[n_experts, d_in, d_out]`; that repacking is **not** performed here and
must be done by a downstream pass if needed.

## Build

From the repo root:

```sh
make tools/quantize-to-gguf
```

This produces `tools/quantize-to-gguf` (a binary, ignored by git).

## Usage

```sh
tools/quantize-to-gguf <safetensors_dir> <out.gguf> --quant q8_0   # default
tools/quantize-to-gguf <safetensors_dir> <out.gguf> --quant f16    # F16 passthrough
```

`<safetensors_dir>` must contain `config.json`, `tokenizer.json`, and
`model.safetensors`. The architecture is detected from the config.

### Examples

```sh
# Qwen3 family
tools/quantize-to-gguf ~/models/Qwen3-0.6B    out/qwen3-0.6B-Q8_0.gguf
# Gemma3 family (text-only or multimodal text tower)
tools/quantize-to-gguf ~/models/gemma-3-1b-pt out/gemma-3-1b-Q8_0.gguf
# Mamba2 family
tools/quantize-to-gguf ~/models/mamba2-130m   out/mamba2-130m-Q8_0.gguf
# DeepSeek family
tools/quantize-to-gguf ~/models/DeepSeek-V2-Lite out/deepseek-v2-lite-Q8_0.gguf
```

## Quantization policy

Per-tensor type selection is conservative and family-aware:

- **All families.** 1-D tensors (RMSNorm/LayerNorm weights, biases) are
  written as F32 to preserve normalization precision.
- **Q8_0 fallback.** If a 2-D tensor's innermost dim is not a multiple
  of 32 (Q8_0 block size), it is written as F32 instead.
- **Mamba2.** SSM scalars and conv kernels (`ssm_a`, `ssm_d`,
  `ssm_conv1d`, `ssm_dt`) stay F32 unconditionally - quantizing the
  state-space dynamics to Q8_0 destroys numerical behavior.
- **Gemma3.** No special carve-outs in v0; the PLE (per-layer
  embeddings) table, if present, follows the standard 2-D Q8_0 path.
  May want to keep that F32 in a future revision if dequant noise
  visibly biases the residual stream.
- **DeepSeek.** Per-expert weights follow the standard 2-D Q8_0 path.
  Router (`mlp.gate.weight`) is 2-D and quantizes too; the router bias
  (`e_score_correction_bias`) is 1-D and stays F32.

## Performance

On an M-series laptop, Qwen3-0.6B (~0.6B params) quantizes in **~3.9 s**
end-to-end with the multi-arch dispatcher. Throughput is dominated by
quantize + write phases; arch detection and dispatch add no measurable
overhead.

## Numerical equivalence vs llama.cpp

The output is numerically **equivalent** (not byte-exact) to llama.cpp's
`llama-quantize`: mean absolute Δ on dequantized weights is ~1.5e-4,
well below the Q8_0 round-trip noise floor of ~1.5e-3. The non-bit-exact
delta comes from a minor difference in block-scale rounding
(round-half-to-even vs llama.cpp's truncate-then-bias). Both schemes are
within Q8_0 spec. This caveat applies to all supported families.

## Validation

A `test_all_archs.sh` smoke script ships alongside the source. It runs
the quantizer against locally-available fixtures and skips families
without weights, printing the command that would have validated them.
Tensor count and GGUF arch key are verified for each produced file via
the Python `gguf` reader.
