# Qwen Q2_K decode path — STATUS

Branch: `dev-qwen-q2k` (off `origin/main` @ 907a0fe). Bench host: **derek** (Apple
M4 base, 16 GiB — same bandwidth class as lexie, which was unreachable: port 22
timed out / 100% packet loss even though Tailscale listed it active).

## What landed

1. **`kernels/gemm/q2k_matvec.metal`** — production Q2_K dense matvec, host_name
   `q2k_matvec`, mirroring the q4k/q6k binding + dispatch convention
   (`B=0,A=1,C=2,K=3,N=4`; NR0=2 rows/TG, 4 simdgroups, 2 SGs cooperate per row
   tiling K). Reuses the validated ds4 / `down_scatter_q2k` per-block dequant.
   The pre-existing self-contained `gemm_q2k_mv` (quant_mv.c++ lab launcher) is
   kept intact in the same file.
   - numpy validation vs gguf-py Q2_K dequant: worst **max_rel 4.79e-4**
     (fp16-output rounding) over N/K ∈ {256,512,1024,130}×{256,1024,2560,768}.

2. **Per-projection dispatch** — `quant_matvec_pso` now routes Q2_K →
   `q2k_matvec` (alongside q8_0/q4k/q6k); launcher binds the PSO;
   `dtype_bytes(Q2_K) = (n/256)·84` (was hitting the n·2 fallback).

3. **Q3_K host-dequant fallback** — real Q2_K_M GGUFs (bartowski / unsloth) use
   **Q3_K** for `attn_output`, `ffn_down`, and `output.weight` per the K-quant
   recipe. SK has no Q3_K matvec, so the loader host-dequants any
   no-native-kernel projection to fp16 (transposed to K-major for the
   `gemv_fp16_m1` fallback). Added Dtype::Q3_K (gguf code 11),
   `dtype_bytes(Q3_K)=110/256`, `dequant_q3_k_to_fp16` (validated **bit-exact**
   vs gguf-py), and Python `weight_loader` registration. Q2_K/Q4_K/Q6_K stay on
   their native kernels. **V must stay native** (the split-QKV dispatch has no
   fp16 path) — the loader errors clearly if a future GGUF puts Q3_K on V.

## Model

`qwen3-4b-q2k` → `bartowski/Qwen_Qwen3-4B-GGUF` / `Qwen_Qwen3-4B-Q2_K.gguf`
(1.6 GB vs 2.3 GB Q4_K_M). Qwen's official 4B-GGUF repo ships no Q2_K.
Dtype mix: Q2_K (q/k/gate/up), Q4_K (v), Q3_K (o/down/output), Q6_K (token_embd,
tied → LM head uses q6k_matvec). The bartowski GGUF has no sidecar tokenizer;
the official `Qwen/Qwen3-4B` `tokenizer.json` is dropped into the weight dir.

## Bench (derek M4, cache_max=2048, 64-tok decode, 5-rep median, prefill excluded)

| model | tok/s (median) | rel. |
|---|---|---|
| qwen3-4b-q4km | **39.31** (39.16–39.38) | 1.00× (same host/build baseline; lexie ref was 38.48) |
| qwen3-4b-q2k  | **26.57** (26.52–26.64) | **0.68× — SLOWER** |

## Verdict: Q2_K is a NET LOSS at 4B here, both speed and quality.

**Speed (-32%):** the win is real on the Q2_K projections (q/k/gate/up: ~2.6 bpw
vs Q4_K's ~4.5), but this GGUF puts `attn_output` and `ffn_down` in **Q3_K**,
which SK has no matvec for, so they are host-dequanted to **fp16** (16 bpw). The
fp16 reads stream ~3.6× the bytes of Q4_K for those two projections — and
`ffn_down` (n_int×d_model) is the single largest per-layer matvec — so the o+down
bandwidth regression swamps the q/k/gate/up savings. The only way Q2_K beats
Q4_K_M end-to-end is a native Q3_K matvec (so o/down stay quantized), or a GGUF
recipe that keeps every projection on a dtype SK has a kernel for.

**Coherence (degraded):** prompt "Generate a poem about pizza dough", greedy.
- Q2_K: `推动, with the following structure: 1. Introduction: Describe the
  origin of pizza dough. 2. Body: Discuss the process of making pizza dough…`
  — leads with a stray Chinese token, then describes how to structure a poem
  instead of writing one. On-topic and grammatical but instruction-ignoring +
  a non-English artifact. The 2-bit quant noise is visible at 4B.
- Q4_K_M: `which is a metaphor for life. The poem should have 4 stanzas, each
  with 4 lines, and a rhyme scheme of ABAB…` — also meta (small 4B base model,
  no chat template here), but clean English, no glitch tokens.

**Recommendation:** Q2_K at 4B is too lossy AND slower as-shipped — not worth it.
Q2_K's value (smaller footprint, quality-for-fit trade) only materializes at
14B+ AND requires a native Q3_K matvec so the K-quant recipe's o/down don't
fall back to fp16. The q2k_matvec kernel itself is correct and ready; the
end-to-end win is gated on a Q3_K kernel, not on this kernel.
