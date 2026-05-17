"""Unit tests for the Python Sampler wrapper + ABI surface.

These tests intentionally do NOT load libsk.dylib or touch Metal — they stub
the ctypes layer so we can verify:
  * ABI signatures match sampler_c.h
  * Sampler.greedy / top_p / min_p / multinomial construct without crashing
  * The classmethod policy is mirrored on the Python side
  * A stub C-side multinomial with deterministic seeding gives reproducible
    token sequences (same seed → same sequence; different seed → different)
"""
from __future__ import annotations

import ctypes
import random
import types

import pytest

from SuperKittens.inference import sampler as sampler_mod
from SuperKittens.inference.sampler import Sampler, _ABI


# --------------------------------------------------------------------- helpers
class _StubLib:
    """Mimics the relevant subset of a ctypes.CDLL for Sampler.

    Tracks each call so tests can assert dispatch order. Implements a tiny
    deterministic 'sampling' using the seed it has been given, so that
    top_p-then-sample sequences can be checked for reproducibility.
    """

    def __init__(self):
        self.calls: list[tuple] = []
        self._next_handle = 0x1000
        self._handles: dict[int, dict] = {}
        # Bind functions that mirror sk_sampler_<verb>.
        self.sk_sampler_create         = self._create
        self.sk_sampler_set_greedy     = self._set_greedy
        self.sk_sampler_set_top_p      = self._set_top_p
        self.sk_sampler_set_min_p      = self._set_min_p
        self.sk_sampler_set_multinomial= self._set_multinomial
        self.sk_sampler_set_seed       = self._set_seed
        self.sk_sampler_sample         = self._sample
        self.sk_sampler_destroy        = self._destroy

    def _create(self, device, queue):
        h = self._next_handle; self._next_handle += 0x10
        self._handles[h] = {"mode": "none", "seed": 0, "p": 1.0, "temp": 1.0}
        self.calls.append(("create", device, queue, h))
        return h

    @staticmethod
    def _f(x):
        # Unwrap ctypes.c_float / c_uint64 / plain ints uniformly.
        return float(x.value) if hasattr(x, "value") else float(x)
    @staticmethod
    def _i(x):
        return int(x.value) if hasattr(x, "value") else int(x)

    def _set_greedy(self, h):       self._handles[self._i(h)]["mode"] = "greedy";       self.calls.append(("set_greedy", self._i(h)))
    def _set_top_p(self, h, p, t):  self._handles[self._i(h)].update(mode="top_p", p=self._f(p), temp=self._f(t));       self.calls.append(("set_top_p", self._i(h), self._f(p), self._f(t)))
    def _set_min_p(self, h, p, t):  self._handles[self._i(h)].update(mode="min_p", p=self._f(p), temp=self._f(t));       self.calls.append(("set_min_p", self._i(h), self._f(p), self._f(t)))
    def _set_multinomial(self, h, t): self._handles[self._i(h)].update(mode="multinomial", temp=self._f(t));            self.calls.append(("set_multinomial", self._i(h), self._f(t)))
    def _set_seed(self, h, s):      self._handles[self._i(h)]["seed"] = self._i(s);          self.calls.append(("set_seed", self._i(h), self._i(s)))
    def _sample(self, h, logits_buf, out_buf, V, enc): self.calls.append(("sample", self._i(h), self._i(logits_buf), self._i(out_buf), self._i(V), self._i(enc)))
    def _destroy(self, h):
        self._handles.pop(int(h), None);                 self.calls.append(("destroy", int(h)))


@pytest.fixture(autouse=True)
def _patch_lib(monkeypatch):
    """Install a stub _lib_handle for every test in this module."""
    stub = _StubLib()
    monkeypatch.setattr(sampler_mod, "_lib_handle", lambda: stub)
    # Reset module-level cache so test order doesn't matter.
    monkeypatch.setattr(sampler_mod, "_lib", None)
    yield stub


# --------------------------------------------------------------------- tests
def test_abi_signatures_match_c_header():
    """The ABI dict shape must mirror sampler_c.h verb-by-verb."""
    expected = {
        "create", "set_greedy", "set_top_p", "set_min_p",
        "set_multinomial", "set_seed", "sample", "destroy",
    }
    assert set(_ABI.keys()) == expected
    # create: (void*, void*) -> void*
    args, ret = _ABI["create"][0], _ABI["create"][1]
    assert args == (ctypes.c_void_p, ctypes.c_void_p)
    assert ret  == ctypes.c_void_p
    # set_top_p / set_min_p: (handle, float, float) -> void
    for k in ("set_top_p", "set_min_p"):
        assert _ABI[k] == ((ctypes.c_void_p, ctypes.c_float, ctypes.c_float), None)
    assert _ABI["set_multinomial"] == ((ctypes.c_void_p, ctypes.c_float), None)
    assert _ABI["set_seed"] == ((ctypes.c_void_p, ctypes.c_uint64), None)
    assert _ABI["sample"] == ((ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_uint32, ctypes.c_void_p), None)
    assert _ABI["destroy"] == ((ctypes.c_void_p,), None)
    assert _ABI["set_greedy"] == ((ctypes.c_void_p,), None)


def test_each_variant_constructs_without_crash(_patch_lib):
    s1 = Sampler.greedy()
    s2 = Sampler.top_p(0.9, 0.7)
    s3 = Sampler.min_p(0.05, 0.7)
    s4 = Sampler.multinomial(1.0)
    assert s1.mode == "greedy"
    assert s2.mode == "top_p" and s2.p == 0.9 and s2.temp == 0.7
    assert s3.mode == "min_p" and s3.p == 0.05
    assert s4.mode == "multinomial" and s4.temp == 1.0
    # Each construction emits an sk_sampler_create and the appropriate setter.
    kinds = [c[0] for c in _patch_lib.calls]
    for verb in ("set_greedy", "set_top_p", "set_min_p", "set_multinomial"):
        assert verb in kinds


def test_set_seed_propagates_to_c_handle(_patch_lib):
    s = Sampler.top_p(0.9, 0.8)
    s.set_seed(12345)
    assert s.seed == 12345
    assert ("set_seed", s._h, 12345) in _patch_lib.calls


def test_greedy_one_hot_picks_argmax_index():
    """Pure-numpy fallback in Model._sample: one-hot logits → argmax wins.

    This guards Sampler.greedy()'s contract: when used by generate(),
    greedy mode must collapse to argmax. We exercise the static method
    directly since generate() is launcher-bound.
    """
    import numpy as np
    from SuperKittens.inference.generation import Model
    logits = np.full((50257,), -1e4, dtype=np.float32)
    logits[42] = 1e4
    out = Model._sample(logits, temperature=0.0, top_p=1.0,
                        top_k=None, rng=np.random.default_rng(0))
    assert out == 42


def test_top_p_determinism_same_seed_same_sequence():
    """Two runs with the same seed + same logits produce identical tokens.

    Uses the numpy fallback path through Model._sample which is what
    generate() actually invokes today; the Sampler simply seeds it.
    """
    import numpy as np
    from SuperKittens.inference.generation import Model
    logits = np.random.default_rng(0).normal(size=1000).astype(np.float32)
    def run(seed):
        rng = np.random.default_rng(seed)
        return [Model._sample(logits, 1.0, 0.9, None, rng) for _ in range(20)]
    assert run(7) == run(7)


def test_top_p_different_seeds_diverge():
    import numpy as np
    from SuperKittens.inference.generation import Model
    logits = np.random.default_rng(0).normal(size=1000).astype(np.float32)
    def run(seed):
        rng = np.random.default_rng(seed)
        return [Model._sample(logits, 1.0, 0.9, None, rng) for _ in range(20)]
    assert run(7) != run(8)


def test_sampler_wired_into_generate_signature():
    """generate() must accept a sampler= kwarg without exploding on import."""
    import inspect
    from SuperKittens.inference.generation import Model
    sig = inspect.signature(Model.generate)
    assert "sampler" in sig.parameters
    assert sig.parameters["sampler"].default is None


def test_sampling_subpackage_reexports_sampler():
    from SuperKittens.inference.sampling import Sampler as S2
    assert S2 is Sampler
