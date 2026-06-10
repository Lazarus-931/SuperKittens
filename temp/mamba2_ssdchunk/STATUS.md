# mamba2-130m SSD prefill: chunked/parallel scan — WIN (1.27–1.41× e2e TTFT, gates pass)

Host derek (M4 base mini, CLT-only, runtime metal-compile via SK_METAL_SRC_FALLBACK).
Tree ~/sk-mamba-ssdchunk = this worktree (branch dev-sk-mamba-ssdchunk, base origin/main @6931da9).
Model AntonV/mamba2-130m-hf (fp16, weights APFS-cloned from sk-mamba-prefill).
Flag-gated + additive: `SK_MAMBA2_SSD_CHUNKED=1` selects the new path at dispatch
(read fresh per forward → in-process A/B); default OFF leaves main byte-identical.
`SK_MAMBA2_SSD_CHUNK=<Q>` sets chunk size (default 1024); `SK_MAMBA2_SSD_PB=1`
forces the un-blocked variant (attribution knob).

## Design (mamba2_ssd_chunked.metal, 3 passes + p-blocking)
s[t] = dA[t]·s[t-1] + dt[t]·B[t]·x[t] with scalar dA per (t,h) decomposes per chunk:
1. `mamba2_ssd_chunk_scan[_pb4]` (grid B*H × P/PB × NC): per-chunk local scan from
   zero state → y_local into y_out, S_local[c] into chunk_states (B,NC,H,P,N) fp32,
   per-token inclusive prefix decay cumdecay (B,L,H) fp32 (p==0 TG writes).
2. `mamba2_ssd_chunk_prop` (grid B*H × P): serial over NC: S_in[c] =
   prodA[c-1]·S_in[c-1] + S_local[c-1], seeded from incoming ssm_state; rewrites
   chunk_states slot c to S_in[c] in place; writes final state to ssm_state
   (decode continuation).
3. `mamba2_ssd_chunk_fix[_pb4]` (grid B*H × P/PB × NC): y[t] +=
   cumdecay[t]·(C[t]·S_in[chunk(t)]); early-out when S_in == 0 (chunk 0 after reset).
fp32 state + fp32 decay products throughout (fp16 cumdecay underflows at 6e-5 within
a few dozen tokens for fast-decaying heads). NC=1 is valid and dispatched (pass 3
early-outs); decode (seq==1) never enters the path — mamba2_step untouched.

**PB=4 p-blocking is what wins.** One simdgroup scans 4 p-rows, sharing each B/C
token read (and the dt softplus/exp) 4×. The serial production kernel re-reads
B/C once per p-row: 1536 TGs × 512B/token ≈ 400 MB/layer at T=512, ≈3.1 ms/layer
at ~110 GB/s — exactly the measured SSD stage. The GPU is already saturated at
B*H*P TGs, so traffic (not occupancy/latency) is the limiter.

## Numerics gates — ALL PASS
Greedy decode token-for-token vs baseline serial SSD, finite logits, final
ssm_state (all 24 layers, fp32) rel err vs baseline (gate < 1e-2):
- T=128 prefill + 48 decode: Q=32/64/128 → identical tokens, max_state_rel 1.6–2.7e-3
- T=512 prefill + 32 decode: Q=64/256/512 → identical tokens, max_state_rel 1.6–3.6e-3
- default-Q (1024) confirm at T=128/512/1024 + 32 decode: identical tokens (final.log)
- states NOT bitwise == serial (proves the chunked path actually ran), logits finite
- decode rate flag-off vs flag-on, same process: 249.7 vs 249.9 tok/s (1.001×);
  decode dispatch is literally the same code path (gated on seq>1).

## e2e TTFT A/B (median of 7, 2 warmups, 0.3 s gaps, SAME process via env flip;
base re-measured after each sweep — drift < 1%)
PB=4 chunked vs baseline serial mamba2_ssd:

| T    | base (ms) | Q=64  | Q=128 | Q=256 | Q=512 | Q=T (NC=1) | best    |
|------|-----------|-------|-------|-------|-------|------------|---------|
| 128  | 45.4      | 1.16× | 1.27× | —     | —     | 35.8 ms    | **1.27×** (−21%) |
| 512  | 148.1     | 1.21× | 1.22× | 1.28× | 1.38× | 107.1 ms   | **1.38×** (−28%) |
| 1024 | 286.1     | 1.21× | 1.24× | 1.25× | 1.29× | 203.0 ms   | **1.41×** (−29%) |

SSD-stage-only (FULL − SKIP, SK_MAMBA_SKIP=ssd ablation; SKIP = 25.9/72.1/130.2 ms):
| T    | base SSD | chunked SSD (best Q) | SSD speedup |
|------|----------|----------------------|-------------|
| 128  | 19.5 ms  | 9.9 ms               | 1.97×       |
| 512  | 76.0 ms  | 35.0 ms              | 2.17×       |
| 1024 | 155.9 ms | 72.8 ms              | 2.14×       |

## Chunk-size sweep verdict: bigger is better, NC=1 optimal
TTFT improves monotonically with Q at every T. So the **chunk-parallelism itself is
a (mild) net negative on M4 base** — first measured plain (PB=1, NC≥2): 0.76–0.88×
REGRESSION at every (T,Q), worse with more chunks (ab_pb1.log) — and the whole win
is the p-blocked scan's 4×-shared B/C reads (+ pass-3's extra C traffic shrinking
as NC→1). Default Q=1024 (= NC=1 up to T=1024, NC=2 at seq_max=2048).

## Files
- `SuperKittens/models/ssm/mamba2/mamba2_ssd_chunked.metal` — 5 kernels (scan,
  scan_pb4, prop, fix, fix_pb4); pb4 used when P%4==0 && N<=128 (130m: P=64, N=128).
- `mamba2_model.h` — env helpers, LayerPSOs/ModelBuffers/DispatchBufs fields,
  flag-gated 3-encoder dispatch in dispatch_layer (else-branch = untouched baseline).
- `launcher.c++` — PSO resolution (optional, warn-free), scratch alloc
  (chunk_states ~50 MB @ seq_max 2048 b=1, cumdecay 200 KB; only when PSOs exist), release.
- Bench/gate scripts + raw logs on derek: ~/sk-mamba-ssdchunk/temp/mamba2_ssdchunk/
  (ab.log = pb4 sweep, ab_pb1.log = plain chunked NEG, ab_pb4_nc2.log, final.log).

## Verdict
**WIN, gate met with margin: −21% / −28% / −29% e2e TTFT at T=128/512/1024 (≥8%
required), all numeric gates pass, decode byte-identical-path and rate-unchanged.**
Caveat for promotion honesty: the literal "parallel chunked scan" hypothesis was
falsified (occupancy was never the limiter; plain chunking regresses) — the shipped
kernel wins via shared-B/C p-blocking, with the chunk machinery kept correct and
useful for NC>1 (seq>1024) and future nonzero-initial-state prefill.

## Future work
- PB=8 (s[8][4] registers) — SSD stage is at 2.1×, bandwidth model says ~4× headroom.
- p-block the serial production kernel itself (same lever, no chunk machinery) if
  the flag ever graduates to default-on.
