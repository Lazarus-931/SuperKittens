# pyright: reportMissingImports=false
"""compare_logits.py — SK vs llama.cpp canvas-logit parity report.

Both files are raw f32 [C, n_vocab] (canvas rows). Reports per-position rel
err stats + argmax identity (the meaningful bar at Q4) + top-5 overlap.

  python3 compare_logits.py ref.bin sk.bin [n_vocab]
"""
import sys

import numpy as np

V = int(sys.argv[3]) if len(sys.argv) > 3 else 262144
ref = np.fromfile(sys.argv[1], np.float32).reshape(-1, V)
got = np.fromfile(sys.argv[2], np.float32).reshape(-1, V)
assert ref.shape == got.shape, (ref.shape, got.shape)
C = ref.shape[0]

diff = np.abs(got - ref)
denom = np.maximum(np.abs(ref), 1e-3)
rel = diff / denom
rel_pos_max = rel.max(axis=1)
rel_pos_mean = rel.mean(axis=1)
rms_rel = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)

am_ref = ref.argmax(axis=1)
am_got = got.argmax(axis=1)
am_match = (am_ref == am_got)

# argmax with the mask token suppressed (the sampler's view; discriminative
# when the canvas is all-mask and <mask> trivially dominates)
MASK = 4
ref_nm = ref.copy(); ref_nm[:, MASK] = -np.inf
got_nm = got.copy(); got_nm[:, MASK] = -np.inf
amn_ref = ref_nm.argmax(axis=1)
amn_got = got_nm.argmax(axis=1)
amn_match = (amn_ref == amn_got)

top5_ref = np.argsort(-ref, axis=1)[:, :5]
top5_got = np.argsort(-got, axis=1)[:, :5]
t5 = np.array([len(np.intersect1d(top5_ref[i], top5_got[i])) for i in range(C)])

print(f"positions          : {C}")
print(f"rel err  max/pos   : mean {rel_pos_max.mean():.4f}  median {np.median(rel_pos_max):.4f}  worst {rel_pos_max.max():.4f}")
print(f"rel err  mean/pos  : mean {rel_pos_mean.mean():.5f}  worst {rel_pos_mean.max():.5f}")
print(f"rms rel per pos    : mean {rms_rel.mean():.5f}  worst {rms_rel.max():.5f}")
print(f"max abs diff       : {diff.max():.4f} (logit scale, softcap 30)")
print(f"ARGMAX match       : {am_match.sum()}/{C} = {100.0*am_match.mean():.2f}%")
print(f"ARGMAX match nomask: {amn_match.sum()}/{C} = {100.0*amn_match.mean():.2f}%")
print(f"top5 overlap       : mean {t5.mean():.2f}/5  min {t5.min()}")
if not am_match.all():
    bad = np.nonzero(~am_match)[0][:10]
    for i in bad:
        print(f"  pos {i}: ref argmax {am_ref[i]} ({ref[i, am_ref[i]]:.3f}) vs "
              f"got {am_got[i]} ({got[i, am_got[i]]:.3f}); "
              f"got[ref_am]={got[i, am_ref[i]]:.3f} ref[got_am]={ref[i, am_got[i]]:.3f}")
