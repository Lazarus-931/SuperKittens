# Final confirmation: default-Q chunked path (env flag only, no SK_MAMBA2_SSD_CHUNK)
# gate at T=1024 + TTFT spot-check.
import os, time
import numpy as np
from SuperKittens.inference import registry

SNAP = os.environ["SNAP"]
os.environ.pop("SK_MAMBA2_SSD_CHUNKED", None)
os.environ.pop("SK_MAMBA2_SSD_CHUNK", None)
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

for T in (128, 512, 1024):
    bt, bs, _ = prefill_decode(T, 32)
    b_ttft = ttft(T)
    os.environ["SK_MAMBA2_SSD_CHUNKED"] = "1"
    ct, cs, cl = prefill_decode(T, 32)
    c_ttft = ttft(T)
    del os.environ["SK_MAMBA2_SSD_CHUNKED"]
    rels = max(float(np.abs(c - b).max() / (np.abs(b).max() + 1e-30))
               for c, b in zip(cs, bs))
    print(f"DEFAULTQ T={T}: tokens_identical={ct == bt} max_state_rel={rels:.3e} "
          f"logits_finite={bool(np.isfinite(cl.astype(np.float32)).all())} "
          f"base={b_ttft*1000:.3f}ms chunked={c_ttft*1000:.3f}ms ({b_ttft/c_ttft:.3f}x)",
          flush=True)
print("FINAL_DONE", flush=True)
