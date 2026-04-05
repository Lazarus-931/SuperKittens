#!/usr/bin/env python3
import subprocess, sys, os

seq = sys.argv[1] if len(sys.argv) > 1 else "2048"
d = sys.argv[2] if len(sys.argv) > 2 else "128"
python = "/opt/miniconda3/bin/python3"
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Baselines: seq={seq} d={d}")
print(f"{'method':<15} {'time(us)':>10} {'GFLOPS':>10}")
print("-" * 37)

for name, script in [("torch", "torch/bench.py"), ("mlx", "mlx/bench.py")]:
    path = os.path.join(base, script)
    ret = subprocess.run([python, path, seq, d], capture_output=True, text=True)
    if ret.returncode == 0:
        t, gf = ret.stdout.strip().split(",")
        print(f"{name:<15} {t:>10} {gf:>10}")
    else:
        print(f"{name:<15} {'FAILED':>10}")
        if ret.stderr:
            print(f"  {ret.stderr[:100]}")
