"""roofline.py — simple roofline-bound estimator for Apple Silicon GPUs.

Numbers are nominal peak GPU FP16 throughput and DRAM bandwidth. They give
you the "is my kernel compute- or memory-bound, and how close to roof?" gut
check that every bench needs.

Sources: Apple device pages / public reverse-engineered numbers. Adjust
per-machine if you have measured peaks.
"""

from __future__ import annotations

DEVICE_SPECS: dict[str, dict[str, float]] = {
    # base M-series (8-10 GPU cores, LPDDR)
    "m1":    {"tflops_fp16": 2.6,  "bw_gbs": 68.0},
    "m2":    {"tflops_fp16": 3.6,  "bw_gbs": 100.0},
    "m3":    {"tflops_fp16": 4.0,  "bw_gbs": 100.0},
    "m4":    {"tflops_fp16": 4.6,  "bw_gbs": 120.0},
    # Pro variants
    "m1pro": {"tflops_fp16": 5.2,  "bw_gbs": 200.0},
    "m2pro": {"tflops_fp16": 6.8,  "bw_gbs": 200.0},
    "m3pro": {"tflops_fp16": 7.4,  "bw_gbs": 150.0},
    "m4pro": {"tflops_fp16": 9.0,  "bw_gbs": 273.0},
    # Max
    "m1max": {"tflops_fp16": 10.4, "bw_gbs": 400.0},
    "m2max": {"tflops_fp16": 13.6, "bw_gbs": 400.0},
    "m3max": {"tflops_fp16": 14.0, "bw_gbs": 400.0},
    "m4max": {"tflops_fp16": 18.0, "bw_gbs": 546.0},
}


def roofline_us(
    bytes_per_dispatch: int,
    flops_per_dispatch: int,
    device: str,
) -> dict[str, float]:
    """Return the roofline-bound microseconds per dispatch.

    Output keys:
      - bw_bound_us:    µs to move `bytes_per_dispatch` at peak DRAM BW
      - compute_bound_us: µs to do `flops_per_dispatch` at peak FP16 TFLOPS
      - bound_us:       max(bw_bound_us, compute_bound_us) (the roof)
      - which:          "bandwidth" or "compute"
      - arithmetic_intensity: flops / bytes
    """
    if device not in DEVICE_SPECS:
        raise KeyError(
            f"unknown device {device!r}; known: {sorted(DEVICE_SPECS)}"
        )
    spec = DEVICE_SPECS[device]
    bw_gbs = spec["bw_gbs"]
    tflops = spec["tflops_fp16"]
    # us = bytes / (GB/s) / 1e3   (since 1 GB/s = 1 byte/ns = 1e3 byte/us)
    bw_us = bytes_per_dispatch / bw_gbs / 1e3 if bw_gbs > 0 else float("inf")
    # us = flops / (TFLOPS) / 1e6 (since 1 TFLOP/s = 1 flop/ps = 1e6 flop/us)
    cmp_us = flops_per_dispatch / tflops / 1e6 if tflops > 0 else float("inf")
    bound = max(bw_us, cmp_us)
    return {
        "bw_bound_us": bw_us,
        "compute_bound_us": cmp_us,
        "bound_us": bound,
        "which": "compute" if cmp_us >= bw_us else "bandwidth",
        "arithmetic_intensity": (
            flops_per_dispatch / bytes_per_dispatch if bytes_per_dispatch else float("inf")
        ),
    }


def roof_efficiency(measured_us: float, bytes_per_dispatch: int,
                    flops_per_dispatch: int, device: str) -> float:
    """Return fraction of roof achieved (0..1+). >1 means we beat the model."""
    r = roofline_us(bytes_per_dispatch, flops_per_dispatch, device)
    if measured_us <= 0:
        return 0.0
    return r["bound_us"] / measured_us
