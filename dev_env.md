# SuperKittens — dev environment

Working notes for contributors / agents. Anything that's "how do I get the project running on a fresh Mac mini" lives here.

## Hardware in the tailnet (validation hosts)

| Host (tailnet) | IP | User | Chip | RAM | Owner | Notes |
|---|---|---|---|---|---|---|
| `dereks-mac-mini` | 100.64.169.42 | `derek` | Apple M4 | 16 GB | GrowlyX | Primary Gemma 4 validation host. SSH authorized. |
| `amelias-mac-mini` | 100.102.119.75 | TBD | (verify) | (verify) | GrowlyX | Available for benchmarking. |
| `github-lexies-mac-mini` | 100.77.36.51 | TBD | M4 (16 GB earlier session) | 16 GB | tagged-devices | Mamba 2 inference host. |
| `alazars-macbook-air` | 100.84.102.71 | `alazarmanakelew` | (laptop) | (laptop) | — | Dev box. Source of all rsync. |

SSH pattern: `ssh <user>@<ip>`. If user is unknown try `derek`, `lexie`, `amelia`, or fall back to `whoami` from the laptop login.

## Repo layout

```
SuperKittens/                       # repo root (this folder)
├── build.sh                        # compile .metal → .metallib, .c++ → .dylib
├── setup.sh                        # one-shot: chip detect, brew, venv, deps, build
├── download.sh                     # unified model downloader (registry)
├── dev_env.md                      # this file
├── SuperKittens/                   # python package + C++ sources
│   ├── __init__.py                 # exposes sk.load / sk.register / sk.Model
│   ├── api.py                      # MODEL_REGISTRY + load()
│   ├── inference/                  # WeightStore, generation loop
│   ├── kernels/                    # shared Metal kernels (fp16 path; Qwen/DS reuse)
│   ├── models/                     # per-family code
│   │   ├── load/                   # gguf, safetensor, tokenizer
│   │   ├── gemma/gemma4/           # Gemma 4 (bf16 native)
│   │   ├── qwen/                   # Qwen3 (fp16)
│   │   ├── deepseek/               # DeepSeek V3/V4 Flash (mixed precision)
│   │   ├── mamba2/                 # Mamba 2 SSD
│   │   └── mamba3/                 # Mamba 3
│   ├── model_weights/              # downloaded weights (gitignored)
│   └── temp/                       # agent scratch / experiments (gitignored)
└── metal-cpp/                      # vendored Apple metal-cpp headers
```

## Local first-time setup (your laptop)

```bash
git clone https://github.com/Lazarus-931/SuperKittens.git
cd SuperKittens
./setup.sh
```

`setup.sh` will:
1. Verify Apple Silicon
2. Verify `xcrun metal` + `xcrun metallib` (needs full Xcode — Command Line Tools are NOT enough)
3. Install Homebrew if missing
4. Install `python@3.12` + `expat` (workaround for known brew pyexpat ABI mismatch)
5. Create `.venv/` and install `huggingface_hub[cli] numpy sentencepiece tokenizers`
6. Run `./build.sh` → produces `build/libsk.dylib`, `build/libsk.metallib`

Common gotchas:
- `xcrun metallib` not found → Xcode not installed (CLT alone insufficient). Install Xcode from App Store; `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.
- pip `pyexpat` ImportError → `export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib` before pip.

## Bringing up a remote mini (after first-time on laptop)

```bash
# 1. rsync the repo (exclude weights + build cache to not blow tailscale bandwidth)
rsync -avz --delete \
  --exclude='SuperKittens/model_weights/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='*.xcuserstate' \
  --exclude='temp/' \
  ~/SuperKittens/ <user>@<ip>:~/SuperKittens/

# 2. run setup on the mini (one-time)
ssh <user>@<ip> 'cd ~/SuperKittens && ./setup.sh'

# 3. download a model
ssh <user>@<ip> 'cd ~/SuperKittens && ./download.sh gemma4-e2b'

# 4. ship a fresh build from the laptop (skips remote toolchain)
scp ~/SuperKittens/build/libsk.dylib ~/SuperKittens/build/libsk.metallib <user>@<ip>:~/SuperKittens/build/
```

## Running anything on a remote mini

Always prepend the env knobs:
```bash
ssh <user>@<ip> '
  export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
  export SK_METALLIB=/Users/<user>/SuperKittens/build/libsk.metallib
  source ~/sk-venv/bin/activate
  cd ~/SuperKittens
  python3 -u <your-script>
'
```

`SK_METALLIB` is needed because `bindings_init` reads `build/libsk.metallib` relative to CWD by default; explicitly setting the env var lets you run from any directory.

## Reference perf (M4 mini, 16 GB, derek)

llama.cpp `llama-bench` on Gemma 4 E2B Q4_K_M:
- prefill (pp64): **680 tok/s**
- decode  (tg64): **54.9 tok/s**

This is the bar SK should match (at equivalent quant) and beat (with bf16 native + fused decode).

## Validation harness

HF reference activations dump (per model):
- `SuperKittens/temp/gemma4_validate/dump_hf_e2b.py` → produces `hf_ref.npz` (484 tensors for Gemma 4).
- `SuperKittens/temp/gemma4_validate/layer_diff.py` → compares SK dumps against HF ref, reports first-divergent layer.

SK-side dumps via `m.set_dump_enabled(True)` then `m.dump(name)` returning a numpy array. See `gemma4.py`.

## Adding a new model

1. Add an entry to `download.sh`'s `resolve()` case statement (one line: `spec → hf_repo  local_dir`).
2. Create `SuperKittens/models/<family>/` with:
   - `<family>_model.h` — dispatch orchestrator
   - `launcher.{h,c++}` — C ABI
   - `weights.{h,c++}` — HF/GGUF tensor-name → fused-buffer mapping
   - `<family>.py` — Python wrapper class
3. In `models/<family>/__init__.py`, register with `SuperKittens.register("<spec>", <Class>, variant="<v>")`.
4. Document in this file under "validation hosts" or in the family folder if relevant.

## Conventions

- **Default to no comments**: comments explain WHY only, not WHAT. Identifiers are self-explanatory.
- **Build must stay clean**: `build.sh` exits non-zero on compile failure (no silent skip). Fix the first error before moving on.
- **No worktree clutter**: agent worktrees under `.claude/worktrees/` are ephemeral. Use `git worktree remove -f -f` to clean.
- **Throttle GPU benches**: ≤5 iters, ≥1 warmup, `time.sleep(0.3)` between iters, serialize. The validation Macs are shared.
- **Real-checkpoint validation**: every model port must be checked against HF reference logits (rel_err < 0.1 + argmax match). Synthetic smoke tests are not sufficient.
- **Default dtype per model**: Gemma 4 → bf16 native. Qwen 3 → fp16. DeepSeek V3 → bf16 with int2/int4 quant for routed experts.
