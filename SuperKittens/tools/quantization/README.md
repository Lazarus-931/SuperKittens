# quantize-to-gguf

Single-file C quantizer: safetensors → GGUF Q8_0 (and F16 passthrough).
Uses NEON BF16→Q8_0 fast path on Apple Silicon and `pthread`-parallel
work distribution across tensors.

Currently supports the **Qwen3** safetensors layout only.

## Build

From the repo root:

```sh
make tools/quantize-to-gguf
```

This produces `tools/quantize-to-gguf` (a binary, ignored by git).

## Usage

```sh
tools/quantize-to-gguf <safetensors_dir> <out.gguf> --quant q8_0
```

`<safetensors_dir>` must contain Qwen3 weight shards plus `config.json`
and `tokenizer.json`. Pass `--quant f16` to skip Q8_0 packing and emit
a half-precision GGUF.

## Performance

On an M-series laptop, Qwen3-0.6B (~0.6B params) quantizes in **~3.5 s**.
Larger Qwen3 sizes scale near-linearly with parameter count.

## Numerical equivalence vs llama.cpp

The output is numerically **equivalent** (not byte-exact) to llama.cpp's
`llama-quantize`: mean absolute Δ on dequantized weights is **~1.5e-4**,
well below the Q8_0 round-trip noise floor of ~1.5e-3. The non-bit-exact
delta comes from a minor difference in block-scale rounding (round-half-to-even
vs llama.cpp's truncate-then-bias). Both schemes are within Q8_0 spec.
