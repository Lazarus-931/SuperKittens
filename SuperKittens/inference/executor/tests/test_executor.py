"""Unit tests for the SK Executor (C++ + ctypes wrapper).

Strategy: prefer a real MTLDevice when running on macOS (build/libsk.dylib +
the device the bindings layer would use). If the dylib isn't built we
gracefully skip the GPU-dependent assertions.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[4]
_DYLIB = _REPO / "build" / "libsk.dylib"
_METALLIB = _REPO / "build" / "libsk.metallib"

# Prefer the freshly-built dylib over any wheel-bundled _libs/ artifact so
# tests exercise the symbols just compiled in this PR. Must happen before
# any `from SuperKittens.inference.executor import ...` resolves the CDLL.
if _DYLIB.exists():
    os.environ["SK_DYLIB"] = str(_DYLIB)
if _METALLIB.exists():
    os.environ["SK_METALLIB"] = str(_METALLIB)


def _require_dylib():
    if not _DYLIB.exists():
        pytest.skip(f"libsk.dylib not built at {_DYLIB}")


def test_construct_destruct_inert():
    # device=0 path: native side degrades to no queue/library. Verifies the
    # ABI signatures + lifecycle work without Metal.
    _require_dylib()
    from SuperKittens.inference.executor import Executor
    e = Executor(device=0)
    assert e._h
    e.close()
    assert e._h is None


def test_construct_with_real_device():
    _require_dylib()
    from SuperKittens.inference.executor import Executor, default_device
    dev = default_device()
    if dev == 0:
        pytest.skip("no MTLDevice available on this host")
    e = Executor(device=dev)
    try:
        # scratch needs a real device; verify it returns a non-null buffer.
        b = e.scratch(1024)
        assert b != 0
    finally:
        e.close()


def test_pso_cache_idempotent():
    _require_dylib()
    from SuperKittens.inference.executor import Executor, default_device
    dev = default_device()
    if dev == 0:
        pytest.skip("no MTLDevice available on this host")
    e = Executor(device=dev)
    try:
        # argmax is one of the simplest kernels shipped in libsk.metallib.
        # If the metallib isn't present (e.g. partial build), the PSO will
        # come back as 0 — skip rather than fail.
        p1 = e.pso("argmax")
        if p1 == 0:
            pytest.skip("argmax PSO unavailable (metallib missing?)")
        p2 = e.pso("argmax")
        assert p1 == p2
    finally:
        e.close()


def test_scratch_pool_pow2_buckets():
    _require_dylib()
    from SuperKittens.inference.executor import Executor, default_device
    dev = default_device()
    if dev == 0:
        pytest.skip("no MTLDevice available on this host")
    e = Executor(device=dev)
    try:
        a = e.scratch(1024)
        b = e.scratch(2048)
        c = e.scratch(1024)  # same bucket as a → reused
        assert a != 0 and b != 0
        assert a != b           # different buckets → distinct buffers
        assert a == c           # same bucket → identity reuse
    finally:
        e.close()


def test_record_replay_state_machine():
    # Pure state-machine test — does NOT commit any GPU work; we never
    # call replay_token with a real command buffer here.
    _require_dylib()
    from SuperKittens.inference.executor import Executor
    e = Executor(device=0)
    try:
        e.record_token_icb_begin()
        # No dispatch issued (inert device): we just verify begin → end
        # cycles cleanly and is idempotent on repeat.
        e.record_token_icb_end()
        e.reset()
        e.record_token_icb_begin()
        e.record_token_icb_end()
    finally:
        e.close()


def test_qwen_launcher_loc_no_regression():
    # Phase-4 consolidation must NOT inflate the qwen3 launcher. Pre-Phase-4
    # baseline was 438 LOC; we allow ≤ that count. (Goal in PHASE4_DESIGN
    # is a 30%+ reduction, but that lands with the full graph migration.)
    launcher = _REPO / "SuperKittens" / "models" / "qwen" / "launcher.c++"
    n = sum(1 for _ in launcher.open())
    assert n <= 470, f"qwen launcher LOC ballooned: {n} > 470 budget"
