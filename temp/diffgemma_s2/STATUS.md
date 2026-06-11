# DiffusionGemma Stage 2 — sampler + self-conditioning + e2e generation

Continues temp/diffgemma_s1/STATUS.md (forward parity). Goal: close the
EntropyBound sampler on real reference logits, verify the self-conditioning
(SC) subgraph, and produce coherent end-to-end generations on the SK stack.

Reference: llama.cpp PR #24423 @ c84e85af (amelia `~/llamacpp-diffg`, CPU
Release, GGML_CPU_REPACK=OFF — the build cache had drifted to REPACK=ON and
was reconfigured back before any Stage-2 runs). Model:
diffusiongemma-26B-A4B-it-Q4_K_M (read-only, amelia `~/diffgemma-gguf`).
Lab: amelia `~/sk-diffg-s2b`.

## Gate 1 — SC subgraph verification: GREEN

### Op-for-op review (oracle graph_ref.sc_signal/embed_tokens vs PR
`dg_canvas_embed`, src/models/diffusion-gemma.cpp)

| PR op | SK oracle | note |
|---|---|---|
| `soft_max(scale(sc_logits, sc_temp_inv))` | `softmax(sc_logits * sc_temp_inv)` | f32 |
| `mul_mat(sc_embT, probs)` (embed dequant+T, f16) | chunked `probs @ embed_rows` f32 | ggml side accumulates the 262144-long dot in f16 lanes |
| `scale(soft, sqrt(n_embd))` | `* sqrt(2816)` | |
| `build_norm(soft, sc_pre_norm, RMS)` | `rms_norm(soft, eps, self_cond_pre_norm)` | scaled rms |
| `gelu(mul_mat(sc_gate, normed))` | `gelu_tanh(normed @ gate.T)` | ggml_gelu = tanh approx |
| `mul_mat(sc_up, normed)` | `normed @ up.T` | |
| `mul_mat(sc_down, g*u)` | `(g*u) @ down.T` | |
| `scale(sc_sig, sc_use)` | `* sc_use` | {0,1} runtime gate |
| `add(canvas, sc_sig)` then `rms_norm` (no scale) | `embed_tokens(..., sc_sig)` | add BEFORE the canvas rms, both sides |

SC temp contract (PR EB sampler): step k's forward conditions on
softmax(L_{k-1} / t_{k-1}) — `prev_temp_inv`, gated off (sc_use=0) at k=0.
sampler.py exposes exactly this (`prev_temp_inv` updated at end of step);
generate.py skips the SC compute entirely at step 0 (sig*0 == 0 bit-exactly).

### Empirical (instrumented reference: cb(sc_sig) + cb_eval tensor dump in
the server — tools/instrument_sc_dump.py; driver tools/gate1_sc.py)

Inputs: p1 prompt (P=20), canvas0 = sampler(seed 1234) random init, L0 =
reference zero-SC logits on canvas0, canvas1 = sampler step-0 renoise,
S=16 ⇒ t0=0.8 (sc_temp_inv=1.25). Reference forwards ~30 s each (CPU).

| check | result |
|---|---|
| ref sc_sig at use_sc=0 | all-zero exactly (gate semantics confirmed) |
| sc_sig oracle-vs-ref [256,2816] | rel_rms 4.4e-4, cos 0.9999999, max_abs 0.014 (signal rms 5.37) |
| inp_region canvas rows oracle-vs-ref | rel_rms 4.2e-4, cos 0.99999991 |
| inp_region prompt rows oracle-vs-ref | bit-exact (0.0) |
| ref SC effect on step-1 logits (L1_sc vs L1_nosc) | rel_rms 1.35, argmax agree 99.6% — SC is a first-order input, the check has teeth |

(GPU leg + envelope below: gate1b)

GPU leg (forward_metal._sc_signal: probs f16 on the wire, embT [d,V] f16
streamed in 8x352-row chunks through dg_gemm_qkt_f32, MLP on quant GEMM):

| check | result |
|---|---|
| sc_sig GPU-vs-ref | rel_rms 6.8e-4, cos 0.99999977 |
| step-1 logits GPU-vs-ref, SC ACTIVE | argmax-canvas **256/256 = 100%**, rel_rms 0.133 |
| step-1 logits GPU-vs-ref, zero-SC control | argmax-canvas 100%, rel_rms 0.256 |
| logits finite | yes, |max| = 29.80/29.71 (softcap 30) |

SC-active agreement is BETTER than the zero-SC envelope on the same canvas
(SC sharpens the distributions: canvas1 carries one denoise step of signal).
GPU forwards 47.5 s (zero-SC) / 52.2 s (SC) warm — the SC stream costs ~5 s.

Benign noise note: numpy-on-Accelerate raises FP flags
(divide-by-zero/overflow/invalid "in matmul") in the host router matmul and
in f64 comparison dots; outputs verified finite + softcap-bounded, and the
SC-active forward still lands argmax-100% vs the reference. Flag noise, not
data corruption (subnormal-class inputs to BLAS kernels).

## Gate 2 — sampler parity on REAL reference logits

**GREEN — TOKEN-IDENTICAL, 0/70 fields mismatched.**

Setup: instrumented `llama-diffusion-cli` (DG_EB_DUMP, throttle ≤ 2 logits
files on disk), prompt "What is the capital of France?" through the cli's own
chat template (P=23: `<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n…`),
S=10, seed=1234, C=256, CPU reference w/ prompt-KV cache. Consumer
(tools/gate2_real_logits.py) replays the SK sampler on the exact per-step
logits the reference consumed and diffs EVERY decision field.

| step | t | max\|ΔH\| | accepted | held | Hbar | fin |
|---|---|---|---|---|---|---|
| 0 | 0.80 | 2.4e-7 | 122 | 0 | 0.468 | no |
| 1 | 0.76 | 2.4e-7 | 189 | 0 | 0.311 | no |
| 2 | 0.72 | 2.4e-7 | 205 | 0 | 0.265 | no |
| 3 | 0.68 | 1.2e-7 | 214 | 0 | 0.123 | no |
| 4 | 0.64 | 1.2e-7 | 230 | 0 | 0.068 | no |
| 5 | 0.60 | 6.0e-8 | 243 | 0 | 0.0085 | no |
| 6 | 0.56 | 1.5e-8 | 255 | 1 | 0.0016 | **yes** |

All of: working canvas, RNG draws (u, renoise), entropy (≤2.4e-7), argmax,
multinomial picks, accept-sets, renoised canvas, held counter, and the
adaptive stop — identical on real logits, every step. The documented
cum-plateau flip risk did not materialize on this trajectory (it remains
possible in principle on exact f32 ties; the synthetic gate quantified it at
2/1792 discarded picks).

Reference cli's own final answer (same run): thought channel reasoning +
"The capital of France is Paris." — 7 steps, 30.5 s/step CPU.

## Gate 3 — e2e coherent generation

Full SK loop (Metal forward + SC + sampler) via generate.py under
tools/run_gate3.sh (caffeinate -is, detached, watchdog swap>3.5G /
disk<4G / 3600s). All prompts use the reference cli's own chat template
(`<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n{msg}<turn|>\n
<|turn>model\n`); every id file was detokenized against
tokenizer.ggml.tokens and matched the intended text exactly. S=16,
seed=1234, C=256, SC on. Lab tree verified == repo tip by md5 before runs.

### p1 "What is the capital of France?" (P=23) — run gen/p1_s16

12 steps (adaptive stop), wall 604.8 s, fw 593.8 s, sampler 7.7 s,
0 mask tokens in canvas, logits finite every step (generate.py raises
otherwise), trim 45. H_mean 0.464 -> 0.0017, accepted 126 -> 252,
swap flat ~1.1 G, disk drift ~170 MB over the run — the bcb5b61 scratch
fix holds over a full 12-forward generation. Output (verbatim):

> `<|channel>thought`
> `The user is asking for the capital of France.`
> `    *   Identify country: France.`
> `    *   Identify city: Paris.`
> `State the answer clearly.<channel|>The capital of France is Paris.`

GATE3_MORE_PLACEHOLDER

## Gate 4 — cross-check vs llama-diffusion-cli

GATE4_PLACEHOLDER

## Stage-3 baseline numbers

GATE5_PLACEHOLDER

## Tools (this dir, all run on amelia from ~/sk-diffg-s2b)

- `instrument_diffusion.py` — per-step logits+decision dumps in the reference
  EB sampler (DG_EB_DUMP / DG_EB_DUMP_THROTTLE keeps ≤ 2×268 MB on disk).
- `instrument_sc_dump.py` — names sc_sig in the model graph + cb_eval named-
  tensor dumps in diffusion-gemma-server (DG_DUMP_TENSORS/DG_DUMP_DIR).
- `gate1_sc.py` / `gate1b_gpu.sh` — Gate-1 drivers (server + oracle + GPU).
- `gate2_real_logits.py` — spawns the instrumented cli, streams its dumps
  through the SK sampler, diffs every decision field, deletes consumed logits.
- `make_embt.py` — one-time [d_model, vocab] f16 transposed embed (1.48 GB,
  amelia ~/sk-diffg-s2b/dg_embT_f16.bin) for the GPU SC soft-embed stream.
- `run_gate3.sh` — watchdogged e2e generation (swap>5.5G / disk<400M / 3600s).
- `compare_eb.py`, `eb_ref_harness.cpp`, `rng_dump.cpp` — Stage-2a synthetic
  sampler gate (committed earlier, still pass).

## Memory war story, Stage-2 edition (multi-forward generation)

Stage-1 stabilized ONE forward per process; generation runs many. First two
e2e attempts died during step 3 (first: external SIGKILL — kernel got there
first; second: the disk watchdog, correctly): system swap grew ~800 MB/30 s
during forwards, macOS swapfiles consumed the ~2 GB-free root volume, and the
box headed for the Stage-1 disk-exhaustion failure mode. Telemetry
(gen/*/mem.log): swap 1.57→2.35 GB in 30 s, disk free 2.1 GB→264 MB in 60 s.

Fix (commit bcb5b61), forward verified bit-identical pre/post on both the
zero-SC and SC legs:
- persistent grow-only activation scratch in MetalCtx (per-call-site tag):
  every _gemm_f32 A/C, attention q/k/vt/mask/s/p/o, and a packed-MoE rewrite
  (all hit experts share three row-packed scratch buffers + offsets instead
  of ~3 GB/forward of per-expert MTLBuffer alloc/free churn);
- generation loop converts each step's logits to the next step's SC probs
  (f16) immediately and frees the raw 268 MB f32 block (`probs16_of`);
- in-place final softcap (the expression form held ~3 extra 268 MB
  transients at the highest-pressure moment).
Also freed: Gate-1 logit artifacts (~1.3 GB) after the bit-regression diff.
Scratch reuse is also slightly faster: 43.8 s zero-SC / 50.8 s SC (was
47.5/52.2).

## Host notes

- amelia root volume runs ~2.5-4.8 GB free with the embT + dumps in place;
  every logits file is 268 MB — delete as consumed (gate2 streams + deletes).
- A GGUF→derek transfer (`cat ~/diffgemma-gguf/...gguf`) was running through
  amelia during the correctness runs; timing-sensitive numbers (Gate 5) were
  taken TRANSFER_NOTE_PLACEHOLDER.
- Stale `/Users/amelia/SuperKittens` partial copy exists; the lab runs pin
  PYTHONPATH=~/sk-diffg-s2b so the rsynced tree wins. Don't import without it.
