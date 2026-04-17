"""
harness.py — Benchmark harness for MLX baselines across SK
"""

import time
import statistics
import csv
import io
from dataclasses import dataclass, field
from typing import Callable, Any
from itertools import product

try:
    import mlx.core as mx
except ImportError:
    mx = None


@dataclass
class BenchResult:
    name: str
    config: dict
    median_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    gflops: float = 0.0
    gbps: float = 0.0
    iters: int = 0


@dataclass
class BenchHarness:
    warmup: int = 5
    iters: int = 20
    results: list = field(default_factory=list)

    def run(self, fn, **cfg) -> BenchResult:
        """Run a single benchmark config."""
        if mx is not None:
            mx.eval(mx.zeros(1))  # warm up mlx

        # warmup
        for _ in range(self.warmup):
            out = fn(cfg)
            if mx is not None:
                mx.eval(out)

        # timed runs
        times = []
        for _ in range(self.iters):
            if mx is not None:
                mx.synchronize()
            t0 = time.perf_counter()
            out = fn(cfg)
            if mx is not None:
                mx.eval(out)
                mx.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        r = BenchResult(
            name=getattr(fn, "_bench_name", fn.__name__),
            config=cfg,
            median_ms=statistics.median(times),
            min_ms=min(times),
            max_ms=max(times),
            stddev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            iters=self.iters,
        )

        # compute metrics if fn has flops/bytes annotations
        if hasattr(fn, "_bench_flops"):
            flops = fn._bench_flops(cfg)
            r.gflops = flops / (r.median_ms * 1e6)
        if hasattr(fn, "_bench_bytes"):
            nbytes = fn._bench_bytes(cfg)
            r.gbps = nbytes / (r.median_ms * 1e6)

        self.results.append(r)
        return r

    def sweep(self, fn, **param_lists) -> list:
        """Run benchmark across all combos of param lists."""
        keys = list(param_lists.keys())
        vals = [v if isinstance(v, list) else [v] for v in param_lists.values()]
        results = []
        for combo in product(*vals):
            cfg = dict(zip(keys, combo))
            r = self.run(fn, **cfg)
            self.print_result(r)
            results.append(r)
        return results

    def validate(self, out, ref, atol=1e-2, rtol=1e-2) -> bool:
        """Compare output against reference within tolerance."""
        if mx is not None:
            diff = mx.abs(out - ref)
            threshold = atol + rtol * mx.abs(ref)
            return bool(mx.all(diff < threshold))
        return True

    @staticmethod
    def print_result(r: BenchResult):
        cfg_str = " ".join(f"{k}={v}" for k, v in r.config.items())
        line = f"{r.name:30s}  {cfg_str:40s}  median={r.median_ms:.3f}ms  min={r.min_ms:.3f}ms  max={r.max_ms:.3f}ms"
        if r.gflops > 0:
            line += f"  {r.gflops:.1f} GFLOPS"
        if r.gbps > 0:
            line += f"  {r.gbps:.1f} GB/s"
        print(line)

    def report(self):
        """Print all collected results."""
        print(f"\n{'='*100}")
        print(f"{'Name':30s}  {'Config':40s}  {'Median':>10s}  {'Min':>10s}  {'Max':>10s}  {'GFLOPS':>8s}  {'GB/s':>8s}")
        print(f"{'='*100}")
        for r in self.results:
            self.print_result(r)
        print()

    def to_csv(self) -> str:
        """Export results as CSV string."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["name", "config", "median_ms", "min_ms", "max_ms", "stddev_ms", "gflops", "gbps"])
        for r in self.results:
            cfg_str = str(r.config)
            w.writerow([r.name, cfg_str, f"{r.median_ms:.4f}", f"{r.min_ms:.4f}",
                        f"{r.max_ms:.4f}", f"{r.stddev_ms:.4f}", f"{r.gflops:.2f}", f"{r.gbps:.2f}"])
        return buf.getvalue()


def bench(name: str, flops: Callable = None, bytes: Callable = None):
    """Decorator to annotate a benchmark function."""
    def decorator(fn):
        fn._bench_name = name
        if flops is not None:
            fn._bench_flops = flops
        if bytes is not None:
            fn._bench_bytes = bytes
        return fn
    return decorator
