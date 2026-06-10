# pyright: reportMissingImports=false
"""compare_dumps.py — first-divergence bisect between two dump dirs
(runner --dump-dir for gpu vs cpu modes). Prints rms-rel per tap in layer
order so the first bad layer/op is obvious.

  python3 compare_dumps.py dir_a dir_b
"""
import sys
from pathlib import Path

import numpy as np

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
names = sorted(set(p.name for p in a_dir.glob("*.npy")) & set(p.name for p in b_dir.glob("*.npy")),
               key=lambda n: (int(n.split(".")[-2]), n))
for n in names:
    a = np.load(a_dir / n).astype(np.float64)
    b = np.load(b_dir / n).astype(np.float64)
    if a.shape != b.shape:
        print(f"{n}: SHAPE {a.shape} vs {b.shape}")
        continue
    if a.dtype.kind == "i" or n.startswith("moe_sel"):
        mism = (a != b).mean()
        print(f"{n}: sel mismatch {100*mism:.3f}%")
        continue
    rms = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    mx = np.abs(a - b).max()
    print(f"{n}: rms_rel {rms:.3e}  max_abs {mx:.3e}")
