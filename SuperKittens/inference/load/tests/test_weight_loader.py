"""Structural tests for WeightLoader. No real model files.

Builds synthetic GGUF + safetensors on disk, exercises iter_tensors(), and
verifies dtype dispatch round-trips. Migration smoke test asserts qwen.load_gguf
delegates to WeightLoader.
"""
from __future__ import annotations

import ctypes
import json
import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from SuperKittens.inference.load.weight_loader import (
    DTYPE_REGISTRY,
    WeightLoader,
    _GGUF_CODE_TO_SK,
)


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    # safetensors on-disk: <u64 header_len><JSON header><raw tensor bytes>.
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    dtype_map = {np.dtype("float32"): "F32", np.dtype("float16"): "F16"}
    for name, arr in tensors.items():
        b = arr.tobytes()
        header[name] = {
            "dtype": dtype_map[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(b)],
        }
        blobs.append(b)
        offset += len(b)
    hdr = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)


def test_safetensors_roundtrip(tmp_path: Path):
    p = tmp_path / "model.safetensors"
    arrs = {
        "embed":     np.arange(8, dtype=np.float32).reshape(2, 4),
        "norm":      np.ones(4, dtype=np.float16),
    }
    _write_safetensors(p, arrs)
    seen: dict[str, tuple] = {}
    with WeightLoader(p) as wl:
        assert wl.backend_name == "safetensors"
        for name, t in wl.iter_tensors():
            # Read bytes back from data_ptr to confirm zero-copy view is valid.
            buf = (ctypes.c_ubyte * t.nbytes).from_address(t.data_ptr)
            raw = bytes(buf)
            seen[name] = (t.dtype, t.shape, raw)
    assert set(seen) == {"embed", "norm"}
    assert seen["embed"][0] == "FP32" and seen["embed"][1] == (2, 4)
    assert seen["norm"][0]  == "FP16" and seen["norm"][1]  == (4,)
    assert seen["embed"][2] == arrs["embed"].tobytes()
    assert seen["norm"][2]  == arrs["norm"].tobytes()


def test_safetensors_sharded(tmp_path: Path):
    s1 = tmp_path / "model-00001-of-00002.safetensors"
    s2 = tmp_path / "model-00002-of-00002.safetensors"
    _write_safetensors(s1, {"a": np.zeros(2, dtype=np.float32)})
    _write_safetensors(s2, {"b": np.zeros(3, dtype=np.float16)})
    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(json.dumps({
        "metadata": {"total_size": 8 + 6},
        "weight_map": {"a": s1.name, "b": s2.name},
    }))
    with WeightLoader(idx) as wl:
        names = {n for n, _ in wl.iter_tensors()}
    assert names == {"a", "b"}


def test_gguf_roundtrip(tmp_path: Path):
    import gguf
    p = tmp_path / "tiny.gguf"
    w = gguf.GGUFWriter(str(p), arch="qwen3")
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    b = np.ones((2, 2), dtype=np.float16)
    w.add_tensor("alpha", a)
    w.add_tensor("beta",  b)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    with WeightLoader(p) as wl:
        assert wl.backend_name == "gguf"
        d = {n: t for n, t in wl.iter_tensors()}
    assert set(d) == {"alpha", "beta"}
    assert d["alpha"].dtype == "FP32"
    assert d["beta"].dtype  == "FP16"
    # Synthetic GGUF only writes F32/F16; shape semantics: GGUF stores shape
    # reversed vs numpy. We assert the element count instead of strict order.
    assert int(np.prod(d["alpha"].shape)) == 16
    assert int(np.prod(d["beta"].shape))  == 4


def test_dtype_registry_quant_entries():
    # Quant entries must report block_size & bytes_per_block so callers can
    # compute nbytes = (n_elements / block_size) * bytes_per_block.
    for name in ("Q8_0", "Q4_K", "Q2_K", "IQ2_XXS"):
        e = DTYPE_REGISTRY[name]
        assert e["itemsize"] is None
        assert e["block_size"] >= 32
        assert e["bytes_per_block"] > 0
    assert DTYPE_REGISTRY["Q8_0"]["block_size"] == 32
    assert DTYPE_REGISTRY["Q8_0"]["bytes_per_block"] == 34
    assert DTYPE_REGISTRY["Q4_K"]["bytes_per_block"] == 144
    assert DTYPE_REGISTRY["IQ2_XXS"]["bytes_per_block"] == 66


def test_gguf_code_dispatch():
    # Codes per ggml.h. Spot-check the ones SK actually consumes.
    assert _GGUF_CODE_TO_SK[0]  == "FP32"
    assert _GGUF_CODE_TO_SK[1]  == "FP16"
    assert _GGUF_CODE_TO_SK[8]  == "Q8_0"
    assert _GGUF_CODE_TO_SK[10] == "Q2_K"
    assert _GGUF_CODE_TO_SK[12] == "Q4_K"
    assert _GGUF_CODE_TO_SK[16] == "IQ2_XXS"


def test_qwen_load_gguf_uses_weight_loader(tmp_path: Path):
    # Build a 1-tensor GGUF so WeightLoader has something real to enumerate.
    import gguf
    p = tmp_path / "stub.gguf"
    w = gguf.GGUFWriter(str(p), arch="qwen3")
    w.add_tensor("x", np.zeros(4, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    from SuperKittens.models.qwen import qwen as qmod

    fake_self = mock.MagicMock()
    fake_lib = mock.MagicMock()
    fake_lib.sk_qwen_load_gguf.return_value = 0
    with mock.patch.object(qmod, "_load", return_value=fake_lib), \
         mock.patch("SuperKittens.inference.load.weight_loader.WeightLoader",
                    wraps=__import__(
                        "SuperKittens.inference.load.weight_loader",
                        fromlist=["WeightLoader"]).WeightLoader) as WL:
        qmod.Qwen.load_gguf(fake_self, str(p))
    WL.assert_called_once_with(str(p))
    fake_lib.sk_qwen_load_gguf.assert_called_once()
