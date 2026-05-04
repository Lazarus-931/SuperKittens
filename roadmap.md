# SuperKittens Roadmap

## Goal
Build SuperKittens as an Apple-native kernel stack that is:

- fast on M-series GPUs
- reusable across models
- structured around small primitive layers instead of one-off kernels

The order of work is:

1. primitive utilities
2. training kernels
3. inference kernels

Training and inference are not fully separate systems, but they are different optimization targets.
They should share the same low-level primitives while keeping kernel families specialized for their own workloads.

## Core Principles

- Prefer reusable primitives over copying logic between kernels.
- Store less when recompute is cheap.
- Keep threadgroup memory pressure low.
- Specialize kernel variants per shape when it helps Apple GPUs.
- Let host-side dispatch choose the right variant for a device and workload.
- Keep correctness-first reference paths available while optimizing.

## Phase 1: Primitive Layer

These utilities should exist before expanding the kernel surface.

### 1. SIMD / Threadgroup Collectives

Status:
- `mma` exists
- cross-SIMD reduction support exists and should keep expanding

Needed:
- `reduce_sum`
- `reduce_max`
- `reduce_min`
- `reduce_dot`
- broadcast helpers
- allgather helpers
- threadgroup reduction wrappers built on top of simdgroup collectives

Why:
- required for softmax, norms, losses, optimizer stats, attention, decode

### 2. Scan Primitives

Status:
- cumsum utilities exist

Needed:
- inclusive / exclusive scan
- reverse scan
- segmented scan
- scan wrappers for both simdgroup and threadgroup scope

Why:
- essential for Mamba recurrence, state passing, varlen sequence work

### 3. Vectorized I/O

Needed:
- aligned `half2`, `half4`, `float2`, `float4` load/store helpers
- masked loads/stores
- tile copy helpers
- transpose-friendly movement helpers

Why:
- Apple kernels are often limited by movement and layout, not raw math

### 4. Mask / Sequence Utilities

Needed:
- causal mask helpers
- sequence-validity mask helpers
- varlen segment mask helpers

Why:
- shared across attention, Mamba, decode, packed sequence kernels

### 5. Math Policy Helpers

Needed:
- fast vs exact `exp`
- `sigmoid`
- `softplus`
- `gelu` / `silu`
- clamp / min / max
- rsqrt helpers

Why:
- keeps kernel math consistent and lets SK expose deliberate speed/accuracy tradeoffs

## Phase 2: Training Kernels

Training comes first because it forces the stack to support the full graph:

- forward
- backward
- reductions
- saved-state handling
- recompute decisions

### Immediate Priorities

1. Mamba-3 MIMO forward - DONE
2. Mamba-3 MIMO backward - ALMOST DONE
3. stabilize Mamba training paths across SISO and MIMO
4. add norm kernels
5. add loss kernels
6. add optimizer/update kernels

### Training Kernel Families to Add

- RMSNorm
- LayerNorm
- residual + norm fused kernels
- SwiGLU / GeGLU / MLP kernels
- cross-entropy / label-smoothed CE
- optimizer kernels:
  - AdamW
  - SGD / momentum
  - Muon-style updates if kept in scope

### Training Success Criteria

- exact or near-exact gradient parity with reference implementations
- forward/backward timing tracked together
- stage-level timing available for complex kernels
- reference path retained for debugging

## Phase 3: Inference Kernels

Inference comes after training primitives are stable.
It shares many kernels with training, but the optimization target changes:

- minimize latency
- minimize memory traffic
- optimize for cache/state reuse
- specialize for small-batch decode

### Inference Priorities

1. Mamba prefill kernels
2. Mamba decode / recurrent-state kernels
3. KV/state cache layout helpers
4. varlen and packed-batch inference
5. low-latency device-specialized variants

### Inference Kernel Families to Add

- prefill kernels
- decode kernels
- grouped-query / multi-query attention
- paged or compact state-cache utilities
- quantized or mixed-precision inference paths if needed

## Host-Side Dispatch Plan

The host should eventually choose among specialized variants based on:

- chip family
- threadgroup memory limit
- sequence length
- batch size
- head dimension
- rank / mode (SISO vs MIMO)
- training vs inference

This should not become one giant dynamic kernel.
Instead:

- compile a small number of good specializations
- select among them at dispatch time

## Near-Term Execution Order

1. keep expanding the primitive layer
2. finish Mamba-3 MIMO forward
3. finish Mamba-3 MIMO backward
4. harden training benchmark and accuracy harnesses
5. add norm + loss primitives
6. move to inference-specific kernels

## Current Utility Focus

The next utilities should be chosen for maximum reuse on Apple GPUs.

Priority order:

1. generic simdgroup/threadgroup reductions
2. better scan helpers
3. vectorized I/O helpers
4. mask helpers
5. math policy helpers

This is the layer that should carry the "SuperKittens spirit":

- compact
- reusable
- Apple-aware
- strong on recurrence as well as attention
