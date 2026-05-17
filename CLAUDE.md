# SK agent bootstrap

Single-source context for agents working in this repo. Read this BEFORE planning or coding.

## What SK is
SuperKittens is a hand-tuned Metal kernel library for transformer + SSM inference on Apple Silicon. Three layers: **BOTTOM** (silicon primitives — `silicon/`, `kernels/`), **MIDDLE** (scheduler, op-graph, ICB recorder — `inference/`), **TOP** (per-family adapters — `models/`).

## Hard rules
- **NEVER push to `main`.** Every change goes through a PR.
- **DO NOT touch other open PR branches** when starting new work.
- **Lab work lives in `temp/`** (gitignored). One lab dir per experiment.
- **Co-Authored-By trailers** are blocked by the classifier — do not add them.
- **`sudo` on the minis (lexie/derek/amelia)** is blocked. So is `ssh-keygen` on shared hosts and mass `rm -rf` across hosts.
- **No mass-delete remote branches** without explicit per-target user authorization.

## Hardware / runtime
- Three minis on Tailscale: `lexie` (M4), `derek` (M4), `amelia` (M4). All accessible as ssh aliases.
- **Only the user's local Mac has Xcode + the Metal compiler.** Minis only have CommandLineTools — no `xcrun metal`. Two ways to bench remotely:
  1. **Runtime compile**: load `.metal` source as a string and call `MTLDevice.newLibraryWithSource:options:error:` on the mini. Works without Xcode. Use this for lab kernels.
  2. **Compile locally + scp**: build `.metallib` on the local Mac (serialized — concurrent compiles crash the Mac), then `scp` to the mini and bench.
- LPDDR5 peaks: M4 base ≈ 120 GB/s. M4 Pro ≈ 273 GB/s.

## Bench harness
Use `SuperKittens/benchmark/harness/bench_harness.py` (lives at `from SuperKittens.benchmark.harness import BenchHarness, FunctionConstants`).
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
- **Plumbing phases 1-4** landed under `dev-plumbing` (sampler, ICB args, WeightLoader, Executor) but are foundation-only. The actual perf wins (10-15% from ICB) gate a follow-up PR that requires bench-on-mini.
- **`mamba2_ssd.metal`** is numerically wrong (missing softplus / dt*B*x / D*x / n_groups). Launcher falls back to `mamba2_ssd_ref`. See `models/ssm/mamba2/STATUS.md`.
- **`down_scatter` n_int tail-drop bug** was fixed recently; the "fast" pre-fix numbers were a kernel skipping 27-33% of work. Check `git log` on `kernels/moe/down_scatter.metal` if perf benches feel off.
- **`mha_causal`** in `kernels/attn/attn.metal` IS the production attention with the right GQA design for Apple Silicon. Lab attempts to replace it with a uzu-style direct-K kernel won 5/6 shapes but regressed qwen3-8B kv=128 (0.79×) due to losing GQA amortization. Don't blindly try to "make it faster" — TG-staged Br=2 design is structurally right.
- **Q4_K MoE port** (4.0× over fp16 swiglu_pair) is proven in `temp/sk_q4k_moe_port/`. Ready for in-tree promotion when DSv2-Lite / Qwen3-MoE-A3B end-to-end wire up.

## How to invoke agents
- Subagents should ssh-bench on derek or amelia (per task per mini). One agent per kernel. Lab work in `temp/`. ZERO edits to `SuperKittens/kernels/` from lab agents.
- DO NOT run agents that try `xcrun metal` on the minis (it's not installed). Either runtime-compile on the mini OR compile-local + scp.

## Don't write narrative comments
WHY-only. If removing the comment wouldn't confuse a future reader, don't write it. Never re-explain the task or write "this fixes the bug" in code.

## Quick links
- Best scoreboard: `best.md`
- Architecture index: `SuperKittens/docs/kernels.md`
- Mamba2 KNOWN BROKEN doc: `SuperKittens/models/ssm/mamba2/STATUS.md`
