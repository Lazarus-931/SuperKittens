# add_rmsnorm — fused residual-add + RMSNorm

Replaces (`add_f16` then `rmsnorm`). Saves one full d_model HBM round-trip per
call, plus a kernel-launch boundary.

## Design

One TG (128 threads = 4 simdgroups) per output row. **Recompute-not-cache**:
- Pass 1: all 128 threads stride over D in half4 chunks, summing `(x+δ)²`. Each
  simdgroup `simd_sum`s its partial; the four simdgroup sums are reduced via
  a 4-slot threadgroup-memory write-then-broadcast.
- Pass 2: re-read x and δ (hot in L2 from pass 1), compute `x+δ` again, write
  both `y` and `y_norm` in one walk.

The earlier register-hold variant spilled on Apple Metal (dynamic-indexed locals
go to memory) and tg-staged `y` choked occupancy at large D. The recompute
design has 16 bytes of tg memory → max occupancy, and 4× the threads per row
vs a 32-thread simdgroup-only design.

## Bench (fp16, GPU-timestamp median of 20)

| Shape       | v2 µs | unfused µs | v1 µs | v2/unfused |
|-------------|------:|-----------:|------:|-----------:|
| T=1 D=4096  | 19.9  | 39.9       | 44.8  | **2.00×**  |
| T=1 D=7168  | 28.1  | 62.0       | 73.7  | **2.21×**  |
| T=8 D=7168  | 27.1  | 63.4       | 71.0  | **2.34×**  |
| T=64 D=4096 | 32.4  | 50.9       | 46.1  | **1.57×**  |

The 1.25× bandwidth ceiling is exceeded because we also save kernel-launch
overhead — the unfused path does two dispatches with an inter-kernel barrier
between them. Effective speedup at small T is BW + launch latency combined.

## Public API

C ABI: `sk_add_rmsnorm(x, delta, gamma, y, y_norm, T, D, eps)`. PSO `add_rmsnorm`
in libsk.metallib.

Verified bit-exact via SK launcher (max-err 0.0000 on y, ≤ 0.001 on y_norm)
at (T=4, D=128), (T=16, D=4096), (T=2, D=7168).

## Use in models

This fusion fires at every transformer layer boundary:
- Post-attn:  `y_attn = x + attn_out;  m_in = RMSNorm(y_attn)`  → one call
- Post-mlp:   `y_out  = y_attn + mlp_out;  next_x_norm = RMSNorm(y_out)` → one call

Wire-in is one substitution in dispatch_layer (replace the residual-add +
rmsnorm pair with this fused kernel). See `models/deepseek/deepseek_model.h`
and `models/gemma/gemma4/gemma4_model.h` for candidate sites.
