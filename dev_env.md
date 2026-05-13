# Contributing to SuperKittens

SuperKittens is a Metal kernel library for transformer (and SSM) inference on Apple Silicon. This document covers how to bring up a dev environment, the repository layout, conventions, and how to contribute a new model or kernel.

## Prerequisites

- **Apple Silicon Mac** (M1/M2/M3/M4). x86 not supported.
- **macOS 14+** with **full Xcode** installed (not just Command Line Tools — `metallib` ships only with the full app). Activate the toolchain:
  ```bash
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  sudo xcodebuild -license accept
  ```
- **Homebrew** and **Python 3.12+** (Homebrew Python is fine; venv strongly recommended).

## First-time setup

```bash
git clone https://github.com/Lazarus-931/SuperKittens.git
cd SuperKittens
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[hf,tokenizer]'
./build.sh
# pull weights (any HF-hosted repo works; example):
huggingface-cli download google/gemma-4-E2B-it --local-dir SuperKittens/model_weights/gemma-4-E2B-it
```

A successful setup produces `build/libsk.dylib` and `build/libsk.metallib`. Sanity check:
```python
import SuperKittens as sk
import SuperKittens.models.gemma.gemma4    # auto-registers gemma4 specs
m = sk.load("gemma4-e2b")
print(m.chat("Hi", max_new_tokens=8))
```

## Repository layout

```
SuperKittens/
├── build.sh                        # compile .metal → .metallib, .c++ → .dylib
├── dev_env.md                      # this file
├── SuperKittens/                   # python package + C++ sources
│   ├── __init__.py                 # exposes sk.load / sk.register / sk.Model
│   ├── api.py                      # MODEL_REGISTRY + load()
│   ├── inference/                  # WeightStore, generation loop, Model base
│   ├── kernels/                    # shared Metal kernels (gemm, attn, rope, norm, …)
│   ├── models/                     # per-family code
│   │   ├── load/                   # gguf, safetensor, tokenizer
│   │   └── <family>/               # model-specific orchestrator + kernels
│   ├── model_weights/              # downloaded weights (gitignored)
│   └── temp/                       # scratch / experiments (gitignored)
└── metal-cpp/                      # vendored Apple metal-cpp headers
```

## Build flow

`./build.sh` does two things:
1. Compiles every `.metal` under `SuperKittens/kernels/` and `SuperKittens/models/` to `.air` then links them into `build/libsk.metallib`.
2. Compiles every `.c++` under those same trees plus `SuperKittens/inference/` into `build/libsk.dylib`, linked against `metal-cpp`.

The build fails loud on any error — fix the first one before re-running. Python wrappers ctypes-load the dylib at runtime and look up kernel PSOs from the metallib (path read from `SK_METALLIB` env var, defaulting to `build/libsk.metallib` relative to cwd).

## Adding a new model

1. Create `SuperKittens/models/<family>/` with:
   - `<family>_model.h` — Metal-side dispatch orchestrator. Mirrors HF's `forward()` step order.
   - `launcher.{h,c++}` — C ABI: `sk_<family>_create / load_safetensors / forward / dump_layer / destroy`.
   - `weights.{h,c++}` — HF/GGUF tensor-name → fused-buffer mapping. Handles dtype casts.
   - `<family>.py` — ctypes wrapper class. Inherits `SuperKittens.inference.generation.Model` for chat/generate.
   - `<family>/__init__.py` — calls `SuperKittens.register("<spec>", <Class>)`.
   - Any family-specific Metal kernels alongside the orchestrator.
2. Write a validation harness under `SuperKittens/temp/<family>_validate/`:
   - `dump_hf_<family>.py` — load HF reference via `transformers`, register forward hooks on every interesting module, save activations to `hf_ref.npz`.
   - `layer_diff.py` — compare SK forward against HF reference layer by layer.
4. Iterate until `argmax(SK_logits) == argmax(HF_logits)` and rel_err is in the bf16/fp16 noise floor.

## Conventions

- **Default to no comments**: comments explain WHY only, not WHAT. Identifiers are self-explanatory.
- **No emojis in code or files** unless explicitly requested.
- **Build must stay clean**: `./build.sh` exits non-zero on compile failure. No silent skips.
- **Validate against HF**: synthetic smoke tests are not sufficient. Every model port needs `rel_err < 0.1` + argmax match on a real prompt against the official `transformers` reference.
- **Throttle GPU benches**: ≤5 iters, ≥1 warmup, `time.sleep(0.3)` between iters, serialize. Shared validation hardware is fragile.
- **Default dtype per model**: match what the upstream releases ship in. Gemma 4 → bf16 native. Qwen 3 → fp16. DeepSeek V3 → bf16 with int2/int4 quant for routed experts.
- **Fused kernels**: SK rewards fusion (fewer kernel dispatches, less bandwidth). If you can fuse `add + rmsnorm` or `gate + silu + up` into one kernel, do it — Apple's command-buffer encoding cost is real.
- **Don't touch other models' code**: when porting a new family, treat existing model code (qwen, deepseek, gemma, mamba) as untouchable unless you're fixing a shared kernel that's verifiably backward-compatible.

## Distributed development

SuperKittens is designed so multiple contributors can work on different model families in parallel without stepping on each other. The structural rule: **per-family directories are isolated**; only `kernels/` is shared.

For long-running validation against real models, run on whatever Apple Silicon hardware you have. The build output (`libsk.dylib`, `libsk.metallib`) is small enough (~1MB) to sync over the network; the heavy weights live locally.

A typical workflow:
1. Iterate locally on your laptop (M-series). Build with `./build.sh`.
2. If the model is too big for your laptop, sync the repo + binaries to a beefier Mac (rsync; exclude `model_weights/`, `__pycache__/`, `.git/`, `.venv/`).
3. Download weights on the validation host.
4. Run forward / layer-diff there.
5. Push fixes back via git. **GitHub (`main` branch) is the single source of truth.** Pull on your laptop, the validation host pulls in turn.

## Quantization

SK already has matvec kernels for Q2_K, Q4_K, IQ2_XXS (`SuperKittens/kernels/gemm/`). To consume a quantized GGUF, the loader path is `models/load/gguf/` → `WeightStore::add(name, ptr, nbytes, Dtype::Q4_K, shape, zero_copy=true)`. The orchestrator picks the appropriate matvec PSO based on the buffer's `Dtype`.

## Reference perf

`llama.cpp` is the de-facto Apple Silicon inference baseline. Install via `brew install llama.cpp`, then `llama-bench -m <path-to-gguf> -p 64 -n 64 -ngl 999` for prefill+decode numbers. SK's goal is to match (at equivalent quant) and beat (with native bf16 + fused decode paths).

## Where to ask

- **Code questions / new-model port**: open a GitHub issue with the model name + HF link.
- **Kernel perf**: include the `llama-bench` baseline and your SK number on the same hardware.
- **Bugs in a specific family's forward**: include the prompt, the rel_err per layer (via `layer_diff.py`), and the first divergent step.
