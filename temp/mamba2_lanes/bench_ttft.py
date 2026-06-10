"""Gate 3/4: serving-TTFT A/B + decode tok/s.

A = N sequential sk_mamba2_prefill_lane calls; B = one sk_mamba2_prefill_batched.
Timed region: lane-states already reset -> all N first tokens on the host.
2 warmups + 7 reps, 0.3s gaps, A/B alternated within each rep (same process,
same handle), median reported. Decode tok/s measured after each prefill style.
If the loaded dylib lacks the batched ABI (base build), B is skipped.
"""
import ctypes as C
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from SuperKittens.inference import registry

SNAP = os.environ["SNAP"]
REPS = int(os.environ.get("REPS", "7"))
WARMUP = 2
DECODE_STEPS = 64
BATCH = 8

m = registry.load("mamba2-130m", snapshot=SNAP, batch=BATCH, seq_max=512)
cfg = m.cfg
has_b = hasattr(m._lib, "sk_mamba2_prefill_batched")
print(f"loaded: batch={cfg.batch} seq_max={cfg.seq_max} batched_abi={has_b}", flush=True)

rng = np.random.default_rng(11)


def reset_lanes(n):
    for lane in range(n):
        m._lib.sk_mamba2_reset_lane(m._h, lane)


def run_A(prompt_arrs, n, T):
    o = C.c_int(0)
    t0 = time.perf_counter()
    for lane in range(n):
        rc = m._lib.sk_mamba2_prefill_lane(m._h, lane, prompt_arrs[lane], T, C.byref(o))
        assert rc == 0
    return time.perf_counter() - t0


def run_B(flat_arr, n, T):
    outb = (C.c_int * n)()
    t0 = time.perf_counter()
    rc = m._lib.sk_mamba2_prefill_batched(m._h, flat_arr, T, n, outb)
    assert rc == 0
    return time.perf_counter() - t0


def decode_rate(n):
    inb = (C.c_int * n)(*([1] * n))
    outb = (C.c_int * n)()
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        rc = m._lib.sk_mamba2_decode_batched(m._h, inb, n, outb)
        assert rc == 0
        for i in range(n):
            inb[i] = outb[i]
    dt = time.perf_counter() - t0
    return n * DECODE_STEPS / dt


results = []
for T in (128, 256):
    for n in (2, 4, 8):
        prompts = [rng.integers(1000, 40000, size=T).astype(np.int32) for _ in range(n)]
        arrs = [(C.c_int * T)(*[int(x) for x in p]) for p in prompts]
        flat = (C.c_int * (n * T))(*[int(x) for p in prompts for x in p])

        for _ in range(WARMUP):
            reset_lanes(n); run_A(arrs, n, T)
            if has_b:
                reset_lanes(n); run_B(flat, n, T)

        tA, tB, dA, dB = [], [], [], []
        for _ in range(REPS):
            time.sleep(0.3)
            reset_lanes(n)
            tA.append(run_A(arrs, n, T))
            dA.append(decode_rate(n))
            if has_b:
                time.sleep(0.3)
                reset_lanes(n)
                tB.append(run_B(flat, n, T))
                dB.append(decode_rate(n))
        medA = sorted(tA)[len(tA) // 2] * 1000
        med_dA = sorted(dA)[len(dA) // 2]
        if has_b:
            medB = sorted(tB)[len(tB) // 2] * 1000
            med_dB = sorted(dB)[len(dB) // 2]
            print(f"T={T} N={n}: A_seq={medA:.2f}ms B_batched={medB:.2f}ms "
                  f"speedup={medA / medB:.2f}x | decode_after_A={med_dA:.1f} "
                  f"decode_after_B={med_dB:.1f} tok/s (agg)", flush=True)
        else:
            print(f"T={T} N={n}: A_seq={medA:.2f}ms (base dylib, no batched ABI) "
                  f"| decode_after_A={med_dA:.1f} tok/s (agg)", flush=True)

print("BENCH_DONE", flush=True)
