# SSD-stage ablation companion: run with SK_MAMBA_SKIP=ssd (static-read env, so
# its own process). FULL - SKIP = SSD stage GPU time.
import os, time
from SuperKittens.inference import registry

SNAP = os.environ["SNAP"]
m = registry.load("mamba2-130m", snapshot=SNAP)
base_ids = [510, 5347, 273, 6181, 310, 247, 2846, 273, 253, 5112]
def make_ids(T): return [base_ids[i % len(base_ids)] for i in range(T)]

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
    print(f"SKIP T={T} TTFT_median={ttft(T)*1000:.3f} ms", flush=True)
print("SKIP_DONE", flush=True)
