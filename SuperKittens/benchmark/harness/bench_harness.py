# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportMissingTypeStubs=false, reportGeneralTypeIssues=false
"""bench_harness.py — Reusable Metal benchmark harness for SuperKittens.

The goal: stop every lab agent from rewriting the same pyobjc boilerplate
(NSURL loading, MTLBuffer plumbing, function-constant dtype dispatch, GPU
timing loop). One small, well-typed API that handles the gotchas.

Canonical 10-line usage:

    from benchmark.harness.bench_harness import BenchHarness, FunctionConstants

    h = BenchHarness(metallib_path="build/libsk.metallib")
    pso = h.pso("rmsnorm_bf16")
    bX  = h.make_buf(x_bf16); bG = h.make_buf(g_bf16)
    bY  = h.make_zero_buf(rows * D * 2)
    bR  = h.make_buf(np.array([rows], np.uint32))
    bD  = h.make_buf(np.array([D],    np.uint32))
    bE  = h.make_buf(np.array([1e-6], np.float32))
    r = h.run(pso, bufs=[bX, bG, bY, bR, bD, bE],
              grid=(1, (rows+3)//4, 1), tg=(128, 1, 1),
              reps=5, dispatches_per_cmd=200, warmup=1, gap_s=0.3,
              bytes_per_dispatch=rows*D*4)
    h.print_table([("rmsnorm_bf16", f"D={D}", r)])
"""

from __future__ import annotations

import os
import time
import ctypes
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# pyobjc imports. The bundle-load for Foundation is what stops Pyright from
# flagging `NSURL` as an unknown attribute on the Foundation module — every
# lab that skipped this paid the tax with 10+ "reportAttributeAccessIssue"
# warnings in their editor.
import objc  # noqa: F401
import Metal  # type: ignore[import-not-found]
import Foundation  # type: ignore[import-not-found]

# Force-load NSURL / NSString into the Foundation module namespace. Without
# this, on a fresh Python process `Foundation.NSURL` may not be resolvable
# until something else triggers the bundle, and static analyzers
# (pyright/pylance) flag the access.
try:
    objc.loadBundle(
        "Foundation",
        globals=Foundation.__dict__,
        bundle_identifier="com.apple.Foundation",
    )
except Exception:
    # Already loaded — that's fine.
    pass

NSURL = Foundation.NSURL


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Public alias: anything with `setBuffer_offset_atIndex_` semantics — an
# MTLBuffer in practice. We keep it as `Any` because pyobjc gives us a
# dynamic proxy, but the name documents intent.
Buf = Any


@dataclass
class FunctionConstants:
    """Type-safe wrapper around MTLFunctionConstantValues.

    Maps Python int/bool/float to the correct MTLDataType. Today every agent
    miscalls `setConstantValue_type_atIndex_` by passing a Python int where a
    ctypes-pointer-to-int32 is required, so we centralize the dispatch here.

    Usage:
        fc = FunctionConstants()
        fc.set_bool(400, False)
        fc.set_int(420, 256)   # MTLDataTypeInt   (int32)
        fc.set_short(422, 4)   # MTLDataTypeShort (int16)
        fc.set_float(430, 1.0) # MTLDataTypeFloat
    """

    # We keep the ctypes scalars alive for the lifetime of this object
    # (otherwise pyobjc reads from freed memory — a classic gotcha).
    _entries: list[tuple[int, int, Any]] = field(default_factory=list)
    _keepalive: list[Any] = field(default_factory=list)

    def set_bool(self, idx: int, v: bool) -> "FunctionConstants":
        cv = ctypes.c_bool(bool(v))
        self._keepalive.append(cv)
        self._entries.append((idx, Metal.MTLDataTypeBool, cv))
        return self

    def set_short(self, idx: int, v: int) -> "FunctionConstants":
        cv = ctypes.c_int16(int(v))
        self._keepalive.append(cv)
        self._entries.append((idx, Metal.MTLDataTypeShort, cv))
        return self

    def set_ushort(self, idx: int, v: int) -> "FunctionConstants":
        cv = ctypes.c_uint16(int(v))
        self._keepalive.append(cv)
        self._entries.append((idx, Metal.MTLDataTypeUShort, cv))
        return self

    def set_int(self, idx: int, v: int) -> "FunctionConstants":
        cv = ctypes.c_int32(int(v))
        self._keepalive.append(cv)
        self._entries.append((idx, Metal.MTLDataTypeInt, cv))
        return self

    def set_uint(self, idx: int, v: int) -> "FunctionConstants":
        cv = ctypes.c_uint32(int(v))
        self._keepalive.append(cv)
        self._entries.append((idx, Metal.MTLDataTypeUInt, cv))
        return self

    def set_float(self, idx: int, v: float) -> "FunctionConstants":
        cv = ctypes.c_float(float(v))
        self._keepalive.append(cv)
        self._entries.append((idx, Metal.MTLDataTypeFloat, cv))
        return self

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def build(self) -> Any:
        fcv = Metal.MTLFunctionConstantValues.alloc().init()
        for idx, dtype, cv in self._entries:
            fcv.setConstantValue_type_atIndex_(ctypes.addressof(cv), dtype, idx)
        return fcv


@dataclass
class BenchResult:
    """One bench result, in microseconds per dispatch."""

    min_us: float
    mean_us: float
    p10_us: float
    samples_us: list[float]
    gpu_time_method: str = "MTLCommandBuffer.GPUEndTime - GPUStartTime"
    gbps: float | None = None
    bytes_per_dispatch: int | None = None

    def to_row(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "min_us": self.min_us,
            "mean_us": self.mean_us,
            "p10_us": self.p10_us,
            "gpu_time_method": self.gpu_time_method,
        }
        if self.gbps is not None:
            d["gbps"] = self.gbps
        return d


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class BenchHarness:
    """Reusable Metal bench harness.

    One instance owns: device, command queue, and the loaded MTLLibrary.
    PSOs are cached by (kernel_name, function-constants-fingerprint).
    """

    def __init__(self, metallib_path: str):
        self.metallib_path = os.path.abspath(metallib_path)
        if not os.path.exists(self.metallib_path):
            raise FileNotFoundError(f"metallib not found: {self.metallib_path}")
        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is None:
            raise RuntimeError("MTLCreateSystemDefaultDevice() returned None")
        self.device = dev
        self.queue = dev.newCommandQueue()
        url = NSURL.fileURLWithPath_(self.metallib_path)
        lib, err = dev.newLibraryWithURL_error_(url, None)
        if lib is None:
            raise RuntimeError(f"failed to load metallib {self.metallib_path}: {err}")
        self.library = lib
        self._pso_cache: dict[tuple[str, int], Any] = {}

    # -- PSO -----------------------------------------------------------------

    def function_names(self) -> list[str]:
        return list(self.library.functionNames())

    def pso(self, kernel_name: str, fc: FunctionConstants | None = None) -> Any:
        """Build (or return cached) compute pipeline state for `kernel_name`."""
        key = (kernel_name, id(fc) if fc is not None else 0)
        cached = self._pso_cache.get(key)
        if cached is not None:
            return cached

        if fc is not None and not fc.is_empty():
            fcv = fc.build()
            fn, err = self.library.newFunctionWithName_constantValues_error_(
                kernel_name, fcv, None
            )
            if fn is None:
                raise RuntimeError(
                    f"newFunctionWithName_constantValues_ failed for "
                    f"{kernel_name!r}: {err}"
                )
        else:
            fn = self.library.newFunctionWithName_(kernel_name)
            if fn is None:
                raise RuntimeError(
                    f"kernel not found in metallib: {kernel_name!r}"
                )
        p, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if p is None:
            raise RuntimeError(f"PSO build failed for {kernel_name!r}: {err}")
        self._pso_cache[key] = p
        return p

    # -- Buffers -------------------------------------------------------------

    def make_buf(self, arr: np.ndarray) -> Buf:
        """Make an MTLBuffer (shared storage) from a numpy array.

        Copies via newBufferWithBytes_length_options_. Caller can keep the
        numpy array alive separately; the buffer owns its own copy.
        """
        arr = np.ascontiguousarray(arr)
        nbytes = int(arr.nbytes)
        if nbytes == 0:
            raise ValueError("make_buf: array has zero bytes")
        b = self.device.newBufferWithBytes_length_options_(
            arr.tobytes(), nbytes, Metal.MTLResourceStorageModeShared
        )
        if b is None:
            raise RuntimeError(f"newBufferWithBytes_length_options_ returned None ({nbytes} bytes)")
        return b

    def make_zero_buf(self, nbytes: int) -> Buf:
        """Make a zero-filled MTLBuffer (shared storage)."""
        if nbytes <= 0:
            raise ValueError(f"make_zero_buf: nbytes must be > 0, got {nbytes}")
        b = self.device.newBufferWithLength_options_(
            int(nbytes), Metal.MTLResourceStorageModeShared
        )
        if b is None:
            raise RuntimeError(f"newBufferWithLength_options_ returned None ({nbytes} bytes)")
        # newBufferWithLength is not guaranteed to zero; explicitly zero it.
        mv = b.contents().as_buffer(nbytes)
        ctypes.memset(
            ctypes.addressof(ctypes.c_char.from_buffer(mv)), 0, nbytes
        )
        return b

    def read_buf(self, buf: Buf, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        """Read an MTLBuffer back as a numpy array."""
        nbytes = int(np.dtype(dtype).itemsize * int(np.prod(shape)))
        mv = buf.contents().as_buffer(nbytes)
        return np.frombuffer(mv, dtype=dtype).copy().reshape(shape)

    # -- Dispatch ------------------------------------------------------------

    def run(
        self,
        pso: Any,
        bufs: Sequence[Buf],
        grid: tuple[int, int, int],
        tg: tuple[int, int, int],
        reps: int = 5,
        dispatches_per_cmd: int = 200,
        warmup: int = 1,
        gap_s: float = 0.3,
        bytes_per_dispatch: int | None = None,
        buf_offsets: Sequence[int] | None = None,
    ) -> BenchResult:
        """Run the PSO `reps` times × `dispatches_per_cmd` per cmd buffer.

        Returns a BenchResult with min/mean/p10 microseconds per dispatch.
        GPU time comes from MTLCommandBuffer GPUStart/GPUEnd, which is more
        accurate than wall-clock and skips encoding overhead.
        """
        if buf_offsets is None:
            buf_offsets = [0] * len(bufs)
        if len(buf_offsets) != len(bufs):
            raise ValueError("buf_offsets length must match bufs length")
        gx, gy, gz = grid
        tx, ty, tz = tg
        mgrid = Metal.MTLSizeMake(int(gx), int(gy), int(gz))
        mtg = Metal.MTLSizeMake(int(tx), int(ty), int(tz))

        def _encode_one(enc: Any) -> None:
            enc.setComputePipelineState_(pso)
            for i, (b, off) in enumerate(zip(bufs, buf_offsets)):
                enc.setBuffer_offset_atIndex_(b, int(off), i)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(mgrid, mtg)

        # Warmup
        for _ in range(max(0, warmup)):
            cb = self.queue.commandBuffer()
            enc = cb.computeCommandEncoder()
            _encode_one(enc)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

        samples_us: list[float] = []
        for r in range(reps):
            cb = self.queue.commandBuffer()
            enc = cb.computeCommandEncoder()
            enc.setComputePipelineState_(pso)
            for _ in range(dispatches_per_cmd):
                for i, (b, off) in enumerate(zip(bufs, buf_offsets)):
                    enc.setBuffer_offset_atIndex_(b, int(off), i)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(mgrid, mtg)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            gpu_us = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6 / dispatches_per_cmd
            samples_us.append(gpu_us)
            if r != reps - 1 and gap_s > 0:
                time.sleep(gap_s)

        min_us = min(samples_us)
        mean_us = statistics.fmean(samples_us)
        p10_us = float(np.percentile(samples_us, 10))
        result = BenchResult(
            min_us=min_us,
            mean_us=mean_us,
            p10_us=p10_us,
            samples_us=samples_us,
            bytes_per_dispatch=bytes_per_dispatch,
        )
        if bytes_per_dispatch is not None and min_us > 0:
            result.gbps = bytes_per_dispatch / (min_us * 1e3)  # bytes/(us*1e3)=GB/s
        return result

    # -- Printing ------------------------------------------------------------

    @staticmethod
    def print_table(rows: Iterable[tuple]) -> None:
        """Pretty-print a list of (name, shape_label, BenchResult) tuples.

        Columns: kernel | shape | min µs | mean µs | p10 µs | GB/s
        """
        rows = list(rows)
        if not rows:
            print("(no rows)")
            return
        # normalize
        norm: list[tuple[str, str, BenchResult]] = []
        for row in rows:
            if len(row) == 3:
                name, label, r = row
            elif len(row) == 2:
                name, r = row
                label = ""
            else:
                raise ValueError(f"unexpected row shape: {row!r}")
            norm.append((str(name), str(label), r))
        name_w = max(8, max(len(n) for n, _, _ in norm))
        lbl_w = max(6, max(len(l) for _, l, _ in norm))
        header = (
            f"{'kernel':<{name_w}}  {'shape':<{lbl_w}}  "
            f"{'min µs':>10}  {'mean µs':>10}  {'p10 µs':>10}  {'GB/s':>10}"
        )
        print(header)
        print("-" * len(header))
        for name, label, r in norm:
            gbps_str = f"{r.gbps:10.2f}" if r.gbps is not None else f"{'-':>10}"
            print(
                f"{name:<{name_w}}  {label:<{lbl_w}}  "
                f"{r.min_us:10.2f}  {r.mean_us:10.2f}  {r.p10_us:10.2f}  {gbps_str}"
            )
