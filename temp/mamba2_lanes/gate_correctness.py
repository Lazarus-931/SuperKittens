"""Gate 1: batched-lane prefill correctness vs sequential prefill_lane.

N=8 DISTINCT equal-length prompts (T=128). For each path (A = sequential
prefill_lane, B = one prefill_batched):
  - prefill, capture per-lane first token + per-lane conv/ssm state (all layers)
  - greedy decode 32 steps via sk_mamba2_decode_batched
B must match A token-for-token per lane; states rel < 1e-2; logits finite.
"""
import ctypes as C
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from SuperKittens.inference import registry

SNAP = os.environ["SNAP"]
N, T, STEPS = 8, 128, 32

m = registry.load("mamba2-130m", snapshot=SNAP, batch=N, seq_max=512)
cfg = m.cfg
print(f"loaded: batch={cfg.batch} seq_max={cfg.seq_max}", flush=True)

rng = np.random.default_rng(7)
prompts = [list(rng.integers(1000, 40000, size=T).astype(np.int32)) for _ in range(N)]
assert len({tuple(p) for p in prompts}) == N, "prompts must be distinct"


def reset_all():
    for lane in range(N):
        m._lib.sk_mamba2_reset_lane(m._h, lane)


def capture_states():
    convs, ssms = [], []
    for L in range(cfg.n_layers):
        convs.append(m.dump(f"conv_state.L{L}").reshape(cfg.batch, -1).copy())
        ssms.append(m.dump(f"ssm_state.L{L}").reshape(cfg.batch, -1).copy())
    return convs, ssms


def decode_steps(first):
    cur = list(first)
    toks = [list(first)]
    inb = (C.c_int * N)()
    outb = (C.c_int * N)()
    for _ in range(STEPS):
        for i in range(N):
            inb[i] = cur[i]
        rc = m._lib.sk_mamba2_decode_batched(m._h, inb, N, outb)
        assert rc == 0, f"decode rc={rc}"
        cur = [int(outb[i]) for i in range(N)]
        toks.append(list(cur))
    return [[toks[s][i] for s in range(STEPS + 1)] for i in range(N)]


# ── A: sequential prefill_lane ──
reset_all()
firstA = []
for lane in range(N):
    arr = (C.c_int * T)(*prompts[lane])
    o = C.c_int(0)
    rc = m._lib.sk_mamba2_prefill_lane(m._h, lane, arr, T, C.byref(o))
    assert rc == 0, f"prefill_lane rc={rc}"
    firstA.append(int(o.value))
convA, ssmA = capture_states()
toksA = decode_steps(firstA)

# ── B: one batched prefill (wrapper resets lanes itself) ──
firstB = m.prefill_batched(prompts)
logits = m.dump("logits").reshape(-1, cfg.vocab_size)[:N].astype(np.float32)
convB, ssmB = capture_states()
toksB = decode_steps(firstB)

# ── compare ──
fails = []
if firstA != firstB:
    fails.append(f"first tokens differ: A={firstA} B={firstB}")
print(f"first tokens A={firstA}")
print(f"first tokens B={firstB}")

if not np.all(np.isfinite(logits)):
    fails.append("non-finite logits in batched prefill")
print(f"logits finite={bool(np.all(np.isfinite(logits)))} "
      f"absmax={np.abs(logits).max():.3f}")

worst_conv, worst_ssm = 0.0, 0.0
for L in range(cfg.n_layers):
    for lane in range(N):
        a, b = convA[L][lane].astype(np.float32), convB[L][lane].astype(np.float32)
        rel = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)
        worst_conv = max(worst_conv, rel)
        a, b = ssmA[L][lane], ssmB[L][lane]
        rel = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)
        worst_ssm = max(worst_ssm, rel)
print(f"worst per-lane state rel-L2: conv={worst_conv:.3e} ssm={worst_ssm:.3e}")
if worst_conv >= 1e-2 or worst_ssm >= 1e-2:
    fails.append(f"state mismatch conv={worst_conv:.3e} ssm={worst_ssm:.3e}")

mismatch = 0
for lane in range(N):
    if toksA[lane] != toksB[lane]:
        mismatch += 1
        da = next(i for i in range(STEPS + 1) if toksA[lane][i] != toksB[lane][i])
        print(f"lane {lane} DIVERGES at step {da}: A={toksA[lane]} B={toksB[lane]}")
print(f"token-for-token ({STEPS + 1} toks/lane): {N - mismatch}/{N} lanes match")
if mismatch:
    fails.append(f"{mismatch} lanes diverge")

for lane in range(N):
    print(f"lane {lane} continuation: {toksA[lane][:12]}...")

print("GATE1_FAIL: " + "; ".join(fails) if fails else "GATE1_PASS", flush=True)
