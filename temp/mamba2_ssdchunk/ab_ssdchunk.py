# Chunked-SSD prefill A/B: numerics gate + TTFT + decode check, ONE process,
# flag flipped via env (launcher reads SK_MAMBA2_SSD_CHUNKED fresh per dispatch).
import os, time
import numpy as np
from SuperKittens.inference import registry

SNAP = os.environ["SNAP"]

def flag(on, q=None):
    if on:
        os.environ["SK_MAMBA2_SSD_CHUNKED"] = "1"
    else:
        os.environ.pop("SK_MAMBA2_SSD_CHUNKED", None)
    if q is not None:
        os.environ["SK_MAMBA2_SSD_CHUNK"] = str(q)
    else:
        os.environ.pop("SK_MAMBA2_SSD_CHUNK", None)

flag(False)
m = registry.load("mamba2-130m", snapshot=SNAP)
NL = m.cfg.n_layers
base_ids = [510, 5347, 273, 6181, 310, 247, 2846, 273, 253, 5112]
def make_ids(T): return [base_ids[i % len(base_ids)] for i in range(T)]

def prefill_decode(T, ndec):
    m.reset()
    toks = [m.forward(make_ids(T))]
    states = [m.dump(f"ssm_state.L{i}").copy() for i in range(NL)]
    logits = m.get_last_logits().copy()
    for _ in range(ndec):
        toks.append(m.forward([toks[-1]]))
    return toks, states, logits

def gate(T, ndec, qs):
    print(f"=== GATE T={T} prefill + {ndec} greedy decode ===", flush=True)
    flag(False)
    bt, bs, bl = prefill_decode(T, ndec)
    print(f"  baseline toks[:12]={bt[:12]}", flush=True)
    ok = True
    for q in qs:
        if q > T:
            continue
        flag(True, q)
        ct, cs, cl = prefill_decode(T, ndec)
        ident = ct == bt
        rels = [float(np.abs(c - b).max() / (np.abs(b).max() + 1e-30))
                for c, b in zip(cs, bs)]
        bitident = all((c == b).all() for c, b in zip(cs, bs))
        finite = bool(np.isfinite(cl.astype(np.float32)).all())
        print(f"  Q={q}: tokens_identical={ident} max_state_rel={max(rels):.3e} "
              f"state_bitwise==serial={bitident} logits_finite={finite}", flush=True)
        if not ident:
            for i, (a, b) in enumerate(zip(ct, bt)):
                if a != b:
                    print(f"    first divergence tok {i}: {a} vs {b}", flush=True)
                    break
        ok &= ident and max(rels) < 1e-2 and finite
    return ok

def ttft(T, reps=7, warm=2):
    ids = make_ids(T)
    for _ in range(warm):
        m.reset(); m.forward(ids)
    ts = []
    for _ in range(reps):
        time.sleep(0.3); m.reset()
        t = time.time(); m.forward(ids)
        ts.append(time.time() - t)
    ts.sort()
    return ts[len(ts) // 2]

def decode_rate(nsteps=40, reps=5):
    m.reset()
    nxt = m.forward(make_ids(128))
    rates = []
    for _ in range(reps):
        t = time.time()
        for _ in range(nsteps):
            nxt = m.forward([nxt])
        rates.append(nsteps / (time.time() - t))
    rates.sort()
    return rates[len(rates) // 2]

g1 = gate(128, 48, (32, 64, 128))
g2 = gate(512, 32, (64, 256, 512))
print(f"GATE_RESULT pass={g1 and g2}", flush=True)

print("=== TTFT A/B (median of 7, 2 warm, 0.3s gaps, same process) ===", flush=True)
for T in (128, 512, 1024):
    flag(False)
    base = ttft(T)
    parts = [f"T={T} base={base*1000:.3f}ms"]
    for q in (64, 128, 256, 512, 1024):
        if q > T:
            continue
        flag(True, q)
        c = ttft(T)
        parts.append(f"Q={q} {c*1000:.3f}ms ({base/c:.3f}x)")
    flag(False)
    base2 = ttft(T)
    parts.append(f"base2={base2*1000:.3f}ms")
    print(" | ".join(parts), flush=True)

print("=== decode tok/s (flag off vs on; decode path must be untouched) ===", flush=True)
flag(False); r0 = decode_rate()
flag(True, 64); r1 = decode_rate()
print(f"decode base={r0:.1f} tok/s flag_on={r1:.1f} tok/s ratio={r1/r0:.3f}", flush=True)
print("AB_DONE", flush=True)
