# Models

Per-family directories under `SuperKittens/models/<family>/`. Each family follows the same layout:

```
<family>/
  SPEC.md            # architecture + SK port notes (required)
  <family>_model.h   # Metal-side dispatch orchestrator
  launcher.{h,c++}   # C ABI: create / load / forward / destroy
  weights.{h,c++}    # HF tensor-name → fused-buffer mapping
  <family>.py        # ctypes wrapper, inherits inference.generation.Model
  __init__.py        # calls SuperKittens.register(spec, Class)
  *.metal            # family-specific kernels (shared kernels live in kernels/)
```

See `dev_env.md` "Adding a new model" for the full template.

## Inference status

| Family | Spec covered | Argmax vs HF | Decode tok/s | llama.cpp baseline |
|---|---|---|---|---|
| Qwen 3 | 0.6B, 1.7B, 8B, 32B | 0.6B: yes | 71.0 (lexie M4) | 118.8 Q8_0 |
| Gemma 4 | E2B, E4B, 26B, 31B | E2B: yes | 0.10 (derek M4) | 54.9 |
| Mamba 2 | 130m | yes | 90 (lexie M4) | 14.5 HF fp32 |
| Mamba 3 | scaffold | — | — | — |
| Mamba 1 | scaffold | — | — | — |
| DeepSeek V3 | spec | — | — | — |

Specs that exist but lack working end-to-end inference are marked `scaffold` / `spec`.

## Shared kernels

Anything reusable across families lives in `SuperKittens/kernels/` (gemm, attention, rope, rmsnorm, fusion, ops, …). Per-family directories should only hold kernels that are family-specific (e.g. `gemma4_ple_inject.metal`, `mamba2_ssd.metal`).
