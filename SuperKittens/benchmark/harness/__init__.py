"""SK Metal bench harness — reusable pyobjc-Metal benchmark scaffolding.

Public API:
    BenchHarness    — device + library + buffers + GPU timing loop
    Buf             — opaque MTLBuffer alias
    FunctionConstants — type-safe MTLFunctionConstantValues builder
    BenchResult     — min/mean/p10 microseconds (+ optional GB/s)

Helpers (sibling modules):
    roofline.DEVICE_SPECS / roofline_us
    numeric_ref.{rmsnorm, softmax_online, silu_mul, geglu, matvec_fp32, ...}
"""

from .bench_harness import (
    BenchHarness,
    BenchResult,
    Buf,
    FunctionConstants,
)

__all__ = ["BenchHarness", "BenchResult", "Buf", "FunctionConstants"]
