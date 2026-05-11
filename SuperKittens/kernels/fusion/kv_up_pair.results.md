# kv_up_pair — fused MLA K-up + V-up

Two matmuls sharing the same left operand `c_kv`. Tile-MMA design (BM=32, BN=64,
BK=32, MR=1, MC=8, 128 threads / 4 simdgroups). One TG loads `c_kv` into shmem
ONCE per K-step, then runs the MMA against both `W_k_up` and `W_v_up`, reusing
a single `Bs` shmem tile across passes.

Grid: `(ceil(max(K_OUT, V_OUT) / BN), ceil(T / BM))`.

## Bench (fp16, R=512, K=V=2048, GPU-timestamp median of 20)

| T  | v2 (fused) ms | v1 (matvec) ms | 2× gemm_fp16 ms | v2/2× gemm |
|---:|--------------:|---------------:|----------------:|-----------:|
|  1 | 0.166 | 0.212 | 0.241 | **1.45×** |
|  2 | 0.215 | 0.605 | 0.360 | **1.67×** |
|  4 | 0.263 | 1.039 | 0.243 | 0.92× |
| 16 | 0.174 | 2.780 | 0.203 | **1.16×** |
| 64 | 0.173 | 8.532 | 0.354 | **2.04×** |

v2 beats v1 at every T. v2 beats 2× separate GEMMs at T=1, 2, 16, 64 (near-tie
at T=4). BW: 25–33 GB/s (12–17% of unified-memory peak). Compute-light kernel
at small R; saturating BW is the right target.

**Critical fix from naive v2**: initial design with two separate `Bs_k` + `Bs_v`
tiles (18 KB shmem) tanked occupancy. Collapsing to one 4 KB Bs reused across
the two MMA passes was the win.

## Public API

C ABI: `sk_kv_up_pair(c_kv, w_k_up, w_v_up, k_no_pe, v_out, T, R, K_OUT, V_OUT)`.
Wired inline in `models/deepseek/deepseek_model.h::dispatch_attn` (steps 10/11)
via PSO `kv_up_pair`.
