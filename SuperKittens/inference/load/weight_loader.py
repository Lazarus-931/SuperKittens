"""Unified weight loader.

Yields (name, WeightTensor) from a GGUF file or one-or-more safetensors files
without copying. Tensor payloads remain in the underlying mmap; consumers read
through ``data_ptr`` / ``nbytes``. Lifetime of every WeightTensor is tied to the
parent WeightLoader (close it and the mmaps go away).
"""
from __future__ import annotations

import ctypes
import json
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# Canonical SK dtype names. The byte-layout comments describe ONE element of the
# logical tensor; for block-quant types the entry is "per super-block of K
# weights" (block_size, bytes_per_block) — consumers compute total bytes from
# tensor shape / block_size.
#
# FP32      4 bytes/element, IEEE 754 binary32.
# FP16      2 bytes/element, IEEE 754 binary16.
# BF16      2 bytes/element, truncated binary32 (sign+exp8+mant7).
# Q8_0      32 weights packed in 34 bytes: fp16 scale + 32 int8 quants.
# Q4_K      256 weights packed in 144 bytes: 12-byte scales/mins header + 128 nibbles.
# Q2_K      256 weights packed in  84 bytes: super-block scales + 2-bit quants.
# IQ2_XXS   256 weights packed in  66 bytes: fp16 scale + 8x int32 codebook indices.
DTYPE_REGISTRY: dict[str, dict] = {
    "FP32":    {"itemsize": 4,  "block_size": 1,   "bytes_per_block": 4},
    "FP16":    {"itemsize": 2,  "block_size": 1,   "bytes_per_block": 2},
    "BF16":    {"itemsize": 2,  "block_size": 1,   "bytes_per_block": 2},
    "Q8_0":    {"itemsize": None, "block_size": 32,  "bytes_per_block": 34},
    "Q4_K":    {"itemsize": None, "block_size": 256, "bytes_per_block": 144},
    "Q2_K":    {"itemsize": None, "block_size": 256, "bytes_per_block": 84},
    "IQ2_XXS": {"itemsize": None, "block_size": 256, "bytes_per_block": 66},
}

# GGUF type code -> SK canonical name. Codes mirror gguf.GGMLQuantizationType.
_GGUF_CODE_TO_SK: dict[int, str] = {
    0:  "FP32",     # F32
    1:  "FP16",     # F16
    8:  "Q8_0",
    10: "Q2_K",
    12: "Q4_K",
    16: "IQ2_XXS",
    30: "BF16",     # GGMLQuantizationType.BF16 in modern gguf packages
}

# safetensors dtype string -> SK canonical name.
_ST_DTYPE_TO_SK: dict[str, str] = {
    "F32":  "FP32",
    "F16":  "FP16",
    "BF16": "BF16",
}


def _sk_dtype_from_gguf(code: int) -> str:
    if code not in _GGUF_CODE_TO_SK:
        raise ValueError(f"unsupported GGUF dtype code {code}")
    return _GGUF_CODE_TO_SK[code]


def _sk_dtype_from_safetensors(s: str) -> str:
    if s not in _ST_DTYPE_TO_SK:
        raise ValueError(f"unsupported safetensors dtype {s!r}")
    return _ST_DTYPE_TO_SK[s]


@dataclass
class WeightTensor:
    """Zero-copy view of one tensor inside a mmap-backed weight file."""
    name: str
    dtype: str        # one of DTYPE_REGISTRY keys
    shape: tuple
    data_ptr: int     # absolute address of element 0 inside the mmap
    nbytes: int


class _GGUFBackend:
    def __init__(self, path: Path):
        # Defer import: gguf is heavy and only needed when actually loading.
        import gguf
        self._reader = gguf.GGUFReader(str(path), mode="r")
        self._path = path

    def iter_tensors(self) -> Iterator[WeightTensor]:
        for t in self._reader.tensors:
            sk_dtype = _sk_dtype_from_gguf(int(t.tensor_type))
            arr = t.data  # np.ndarray view into the underlying mmap
            data_ptr = arr.ctypes.data
            nbytes = int(t.n_bytes)
            shape = tuple(int(x) for x in t.shape)
            yield WeightTensor(
                name=str(t.name), dtype=sk_dtype, shape=shape,
                data_ptr=data_ptr, nbytes=nbytes,
            )

    def close(self) -> None:
        # gguf.GGUFReader owns its mmap; drop the reference and let GC unmap.
        self._reader = None


class _SafetensorsBackend:
    def __init__(self, path: Path):
        self._mmaps: list[mmap.mmap] = []
        self._files: list[Path] = []
        # Either a single .safetensors file, a sharded index JSON, or a dir.
        if path.is_dir():
            idx = path / "model.safetensors.index.json"
            single = path / "model.safetensors"
            if idx.exists():
                self._init_sharded(idx)
            elif single.exists():
                self._init_single(single)
            else:
                shards = sorted(path.glob("*.safetensors"))
                if not shards:
                    raise FileNotFoundError(f"no safetensors files in {path}")
                self._files = shards
        elif path.suffix == ".safetensors":
            self._init_single(path)
        elif path.name.endswith(".index.json"):
            self._init_sharded(path)
        else:
            raise ValueError(f"not a safetensors path: {path}")

    def _init_single(self, p: Path):
        self._files = [p]

    def _init_sharded(self, index_json: Path):
        meta = json.loads(index_json.read_text())
        wm = meta.get("weight_map", {})
        # Preserve insertion order; dedupe via dict.fromkeys.
        names = list(dict.fromkeys(wm.values()))
        self._files = [index_json.parent / n for n in names]

    def iter_tensors(self) -> Iterator[WeightTensor]:
        # We parse the safetensors header ourselves so we can hand callers a
        # raw pointer into the mmap (safetensors.safe_open exposes tensors as
        # framework tensors, which would force a copy).
        import struct
        for fpath in self._files:
            fd = os.open(str(fpath), os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                # ACCESS_COPY = private writable mapping (COW). We need a
                # writable handle for ctypes.from_buffer to compute an address,
                # but we never write — no disk writeback occurs.
                mm = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
            finally:
                os.close(fd)
            self._mmaps.append(mm)
            base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            (hdr_len,) = struct.unpack("<Q", mm[:8])
            header = json.loads(mm[8:8 + hdr_len].decode("utf-8"))
            data_start = 8 + hdr_len
            for tname, tinfo in header.items():
                if tname == "__metadata__":
                    continue
                sk_dtype = _sk_dtype_from_safetensors(tinfo["dtype"])
                off_lo, off_hi = tinfo["data_offsets"]
                yield WeightTensor(
                    name=tname,
                    dtype=sk_dtype,
                    shape=tuple(int(x) for x in tinfo["shape"]),
                    data_ptr=base + data_start + int(off_lo),
                    nbytes=int(off_hi) - int(off_lo),
                )

    def close(self) -> None:
        for mm in self._mmaps:
            try:
                mm.close()
            except Exception:
                pass
        self._mmaps.clear()


class WeightLoader:
    """Backend-agnostic, mmap-based tensor iterator.

    Detects GGUF vs safetensors by extension / directory shape. Holds the
    backing mmap until ``.close()`` (or context-manager exit); WeightTensor
    pointers become invalid after that.
    """

    def __init__(self, path: str | os.PathLike):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        self.path = p
        if p.is_dir() or p.suffix == ".safetensors" or p.name.endswith(".index.json"):
            self._backend = _SafetensorsBackend(p)
            self.backend_name = "safetensors"
        elif p.suffix == ".gguf":
            self._backend = _GGUFBackend(p)
            self.backend_name = "gguf"
        else:
            raise ValueError(f"cannot detect backend for {p}")

    def iter_tensors(self) -> Iterator[tuple[str, WeightTensor]]:
        for t in self._backend.iter_tensors():
            yield t.name, t

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "WeightLoader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
