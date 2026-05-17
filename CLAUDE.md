# SK agent bootstrap

Single-source context for agents working in this repo. Read this BEFORE planning or coding.

## What SK is
SuperKittens is a hand-tuned Metal kernel library for transformer + SSM inference on Apple Silicon. Three layers: **BOTTOM** (silicon primitives — `silicon/`, `kernels/`), **MIDDLE** (scheduler, op-graph, ICB recorder — `inference/`), **TOP** (per-family adapters — `models/`).

## Hard rules
- **NEVER push to `main`.** Every change goes through a PR.
- **DO NOT touch other open PR branches** when starting new work.
- **Lab work lives in `temp/`** (gitignored). One lab dir per experiment.
- **Co-Authored-By trailers** are blocked by the classifier — do not add them.
- **No mass-delete remote branches** without explicit per-target user authorization.
- **On shared bench hosts**, do not run `sudo`, `ssh-keygen`, or destructive cross-host commands (mass `rm -rf`, broad `find -delete`, etc.) without explicit user authorization.

## How to invoke agents
- Lab work in `temp/`. One lab dir per experiment.
- **ZERO edits to `SuperKittens/kernels/` or `SuperKittens/models/` from lab agents.** Promotion to those trees happens via a normal PR after a bench result is on the table.
- One agent per kernel. Don't fan out across unrelated kernels in a single agent.
- Don't assume `xcrun metal` is available on every bench host (CommandLineTools-only Macs don't have it). See the runtime section for the two options.

## Hardware / runtime
- Targets Apple Silicon. `.metallib` builds require a Mac with full Xcode + the Metal toolchain.
- If the host you're benching on doesn't have Xcode (CommandLineTools-only), two options:
  1. **Runtime compile on the bench host**: load `.metal` source as a string and call `MTLDevice.newLibraryWithSource:options:error:`. No Xcode needed. Use this for lab kernels.
  2. **Compile elsewhere + copy**: build `.metallib` on a Mac with Xcode (serialize the compile — concurrent `xcrun metal` invocations can crash the build host), then transfer to the bench host and load via `newLibraryWithURL:`.
- Bandwidth ceilings (rough): M4 base ≈ 120 GB/s, M4 Pro ≈ 273 GB/s, M-series Max/Ultra higher. Pick a bench host whose bandwidth class matches the kernel — most decode kernels here are bandwidth-bound and regressions look different on a Pro vs. a base.

## Bench harness
Use `SuperKittens/benchmark/harness/bench_harness.py` (`from SuperKittens.benchmark.harness import BenchHarness, FunctionConstants`).
- Owns device, queue, library, PSO cache.
- Type-safe function constants. Buffer creation with explicit zeroing.
- GPU timing via `cmd.GPUEndTime() - cmd.GPUStartTime()`.
- Standard protocol: 1 warmup + 5 reps × 200 dispatches per command buffer, 0.3s gap, min-of-reps.

Numerical refs and roofline are sibling modules (`numeric_ref.py`, `roofline.py`).

**Stop rewriting pyobjc boilerplate in labs.** The harness suppresses the Pyright NSURL/MTLSizeMake noise.

## Where things live
- `kernels/` — shared Metal kernel library, organized by op (dense, flash_attn, moe, norm, rope, sample, fusion, …).
- `models/qwen/`, `models/gemma/gemma4/`, `models/deepseek/`, `models/ssm/mamba2/`, `models/ssm/mamba3/` — per-family adapters (Python + C++ launcher + family-specific kernels).
- `inference/` — sampling, ICB args, weight loader, executor, generation entrypoint, tokenizer plumbing.
- `benchmark/harness/` — shared bench infra.
- `temp/` — gitignored experiment dirs.

## Open architectural state
- **Plumbing phases 1-4** landed under `dev-plumbing` (sampler, ICB args, WeightLoader, Executor) but are foundation-only. ICB perf win is expected but unmeasured — the bench gates a follow-up PR.
- **`mamba2_ssd.metal`** is numerically wrong (missing softplus / dt*B*x / D*x / n_groups). Launcher falls back to `mamba2_ssd_ref`. See `models/ssm/mamba2/STATUS.md`.
- **`down_scatter` n_int tail-drop bug** was fixed recently; the "fast" pre-fix numbers were a kernel skipping 27-33% of work. Check `git log` on `kernels/moe/down_scatter.metal` if perf benches feel off.
- **`mha_causal`** in `kernels/attn/attn.metal` IS the production attention with the right GQA design for Apple Silicon. Lab attempts to replace it with a uzu-style direct-K kernel won 5/6 shapes but regressed qwen3-8B kv=128 (0.79×) due to losing GQA amortization. Don't blindly try to "make it faster" — TG-staged Br=2 design is structurally right.
- **Q4_K MoE port** (~4× over fp16 swiglu_pair) is proven in an out-of-tree lab. Ready for in-tree promotion when DSv2-Lite / Qwen3-MoE-A3B end-to-end wire up.

## Don't write narrative comments
WHY-only. If removing the comment wouldn't confuse a future reader, don't write it. Never re-explain the task or write "this fixes the bug" in code.

## Quick links
- Best scoreboard: `best.md`
- Architecture index: `SuperKittens/docs/kernels.md`
- Mamba2 KNOWN BROKEN doc: `SuperKittens/models/ssm/mamba2/STATUS.md`
