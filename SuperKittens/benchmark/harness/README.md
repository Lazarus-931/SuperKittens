# SK bench harness

Drop the boilerplate. One `BenchHarness` owns device, queue, metallib, PSO
cache, and the warmup + reps + dispatches-per-cmd GPU-time loop. Function
constants go through `FunctionConstants` (type-safe wrapper around
`MTLFunctionConstantValues` — no more passing Python ints where a ctypes
pointer is required). Roofline + numpy refs live next door.

```python
import numpy as np
from benchmark.harness import BenchHarness, FunctionConstants
from benchmark.harness.numeric_ref import bf16_from_f32, rmsnorm
from benchmark.harness.roofline import roofline_us

h = BenchHarness(metallib_path="build/libsk.metallib")
pso = h.pso("rmsnorm_bf16")  # or fc=FunctionConstants().set_int(420, 256)
x  = np.random.randn(1, 4096).astype(np.float32) * 0.1
g  = np.ones(4096, np.float32)
bX, bG = h.make_buf(bf16_from_f32(x)), h.make_buf(bf16_from_f32(g))
bY     = h.make_zero_buf(1 * 4096 * 2)
bR, bD, bE = (h.make_buf(np.array([v], dt)) for v, dt in
              [(1, np.uint32), (4096, np.uint32), (1e-6, np.float32)])
r = h.run(pso, bufs=[bX, bG, bY, bR, bD, bE],
          grid=(1, 1, 1), tg=(128, 1, 1),
          bytes_per_dispatch=4096 * 4)
h.print_table([("rmsnorm_bf16", "D=4096", r)])
print("roof:", roofline_us(4096 * 4, 4096 * 4, "m4"))
```
