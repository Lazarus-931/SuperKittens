#!/usr/bin/env python3
import torch, torch.nn.functional as F, time, sys

seq = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
d = int(sys.argv[2]) if len(sys.argv) > 2 else 128
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20

Q = torch.randn(1, 1, seq, d, dtype=torch.float16, device='mps')
K = torch.randn(1, 1, seq, d, dtype=torch.float16, device='mps')
V = torch.randn(1, 1, seq, d, dtype=torch.float16, device='mps')

for _ in range(5):
    F.scaled_dot_product_attention(Q, K, V)
torch.mps.synchronize()

times = []
for _ in range(iters):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    F.scaled_dot_product_attention(Q, K, V)
    torch.mps.synchronize()
    times.append((time.perf_counter() - t0) * 1e6)

times.sort()
t = times[len(times) // 2]
flops = 4 * seq * seq * d + 5 * seq * seq
print(f"{t:.0f},{flops / (t * 1e3):.1f}")
