# pyright: reportMissingImports=false
"""gguf_io.py — minimal, dependency-free GGUF reader + numpy K-quant dequant.

DiffusionGemma is GGUF-canonical (Q4_K_M, 15.65 GiB) and cannot take the
gemma4 dequant-to-bf16 load path on a 16 GB host, so the family reads tensors
straight off the mmap'd file (native quant blocks feed the SK quant GEMM
kernels; dequant here is only for F32 sidecars, embedding rows, and the CPU
reference forward). Dequant formulas mirror ggml's dequantize_row_* exactly.
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass

import numpy as np

GGUF_MAGIC = 0x46554747

# ggml type ids -> (name, block_elems, block_bytes)
GGML_TYPES = {
    0:  ("F32",  1,   4),
    1:  ("F16",  1,   2),
    2:  ("Q4_0", 32,  18),
    8:  ("Q8_0", 32,  34),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    6:  ("Q5_0", 32,  22),
    30: ("BF16", 1,   2),
}

_KV_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           10: "<Q", 11: "<q", 12: "<d"}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]   # ggml ne order: ne0 (fastest/K) first
    ggml_type: int
    offset: int              # absolute byte offset into the file

    @property
    def type_name(self) -> str:
        return GGML_TYPES[self.ggml_type][0]

    @property
    def nbytes(self) -> int:
        _, be, bb = GGML_TYPES[self.ggml_type]
        n = 1
        for d in self.shape:
            n *= d
        assert n % be == 0, (self.name, self.shape, self.type_name)
        return n // be * bb

    @property
    def row_bytes(self) -> int:
        """bytes per ne0-row (the K dimension all quant kernels stride by)."""
        _, be, bb = GGML_TYPES[self.ggml_type]
        assert self.shape[0] % be == 0
        return self.shape[0] // be * bb


class GGUFFile:
    """mmap-backed GGUF: metadata dict + tensor table + raw/dequant access."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self.mm = mmap.mmap(self._f.fileno(), 0, prot=mmap.PROT_READ)
        self.meta: dict[str, object] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self._parse()

    # -- parsing ------------------------------------------------------------

    def _read(self, fmt: str) -> object:
        v = struct.unpack_from(fmt, self.mm, self._pos)[0]
        self._pos += struct.calcsize(fmt)
        return v

    def _read_str(self) -> str:
        n = self._read("<Q")
        s = self.mm[self._pos:self._pos + n].decode("utf-8", "replace")
        self._pos += n
        return s

    def _read_kv_value(self, t: int):
        if t in _KV_FMT:
            return self._read(_KV_FMT[t])
        if t == 7:  # bool
            return bool(self._read("<B"))
        if t == 8:  # string
            return self._read_str()
        if t == 9:  # array
            et = self._read("<I")
            n = self._read("<Q")
            return [self._read_kv_value(et) for _ in range(n)]
        raise ValueError(f"unknown gguf kv type {t}")

    def _parse(self) -> None:
        self._pos = 0
        magic = self._read("<I")
        assert magic == GGUF_MAGIC, f"not a GGUF file: {self.path}"
        version = self._read("<I")
        assert version >= 2, f"gguf v{version} unsupported"
        n_tensors = self._read("<Q")
        n_kv = self._read("<Q")
        for _ in range(n_kv):
            key = self._read_str()
            t = self._read("<I")
            # the tokenizer arrays are huge; we keep them (cheap lists) anyway
            self.meta[key] = self._read_kv_value(t)
        infos = []
        for _ in range(n_tensors):
            name = self._read_str()
            n_dims = self._read("<I")
            shape = tuple(self._read("<Q") for _ in range(n_dims))
            ggml_type = self._read("<I")
            off = self._read("<Q")
            infos.append((name, shape, ggml_type, off))
        align = int(self.meta.get("general.alignment", 32))
        data_start = (self._pos + align - 1) // align * align
        for name, shape, ggml_type, off in infos:
            self.tensors[name] = TensorInfo(name, shape, ggml_type, data_start + off)

    # -- access ---------------------------------------------------------------

    def raw(self, name: str) -> memoryview:
        ti = self.tensors[name]
        return memoryview(self.mm)[ti.offset:ti.offset + ti.nbytes]

    def dequant(self, name: str, rows: slice | np.ndarray | None = None) -> np.ndarray:
        """Dequantize to f32, shaped [shape[::-1]] (numpy row-major: last ggml
        dim first). `rows` selects along the OUTER (row) dimension, where a row
        is one ne0-slice — exactly ggml's get_rows granularity."""
        ti = self.tensors[name]
        k = ti.shape[0]
        n_rows = 1
        for d in ti.shape[1:]:
            n_rows *= d
        rb = ti.row_bytes
        buf = np.frombuffer(self.mm, dtype=np.uint8, count=n_rows * rb,
                            offset=ti.offset).reshape(n_rows, rb)
        if rows is not None:
            buf = buf[rows]
        out = dequant_rows(buf, ti.type_name, k)
        if rows is None and len(ti.shape) > 1:
            out = out.reshape(tuple(ti.shape[::-1]))
        return out


# -- numpy dequant (mirrors ggml dequantize_row_*) ---------------------------

def _f16(u16: np.ndarray) -> np.ndarray:
    return u16.view(np.float16).astype(np.float32)


def dequant_rows(buf: np.ndarray, type_name: str, k: int) -> np.ndarray:
    """buf: uint8 [R, row_bytes] -> f32 [R, k]."""
    r = buf.shape[0]
    if type_name == "F32":
        return buf.reshape(-1).view(np.float32).reshape(r, k).copy()
    if type_name == "F16":
        return buf.reshape(-1).view(np.float16).astype(np.float32).reshape(r, k)
    if type_name == "Q8_0":
        nb = k // 32
        b = buf.reshape(r * nb, 34)
        d = _f16(b[:, :2].copy().view(np.uint16)[:, 0])
        q = b[:, 2:].view(np.int8).astype(np.float32)
        return (q * d[:, None]).reshape(r, k)
    if type_name == "Q4_K":
        return _dequant_q4k(buf, k)
    if type_name == "Q6_K":
        return _dequant_q6k(buf, k)
    raise ValueError(f"dequant for {type_name} not implemented")


def _dequant_q4k(buf: np.ndarray, k: int) -> np.ndarray:
    r = buf.shape[0]
    nb = k // 256
    b = buf.reshape(r * nb, 144)
    d = _f16(b[:, 0:2].copy().view(np.uint16)[:, 0])      # [B]
    dmin = _f16(b[:, 2:4].copy().view(np.uint16)[:, 0])
    sc_raw = b[:, 4:16].astype(np.uint16)                  # [B, 12]
    qs = b[:, 16:144]                                      # [B, 128]
    # get_scale_min_k4 for j = 0..7
    sc = np.empty((b.shape[0], 8), np.float32)
    mn = np.empty((b.shape[0], 8), np.float32)
    for j in range(4):
        sc[:, j] = (sc_raw[:, j] & 63).astype(np.float32)
        mn[:, j] = (sc_raw[:, j + 4] & 63).astype(np.float32)
    for j in range(4, 8):
        sc[:, j] = ((sc_raw[:, j + 4] & 0x0F) | ((sc_raw[:, j - 4] >> 6) << 4)).astype(np.float32)
        mn[:, j] = ((sc_raw[:, j + 4] >> 4) | ((sc_raw[:, j] >> 6) << 4)).astype(np.float32)
    lo = (qs & 0x0F).astype(np.float32).reshape(-1, 4, 32)  # byte group g -> sub-block 2g
    hi = (qs >> 4).astype(np.float32).reshape(-1, 4, 32)    # -> sub-block 2g+1
    y = np.empty((b.shape[0], 8, 32), np.float32)
    y[:, 0::2, :] = lo
    y[:, 1::2, :] = hi
    y = y * (d[:, None] * sc)[:, :, None] - (dmin[:, None] * mn)[:, :, None]
    return y.reshape(r, k)


def _dequant_q6k(buf: np.ndarray, k: int) -> np.ndarray:
    r = buf.shape[0]
    nb = k // 256
    b = buf.reshape(r * nb, 210)
    ql = b[:, 0:128]
    qh = b[:, 128:192]
    scales = b[:, 192:208].view(np.int8).astype(np.float32)  # [B, 16]
    d = _f16(b[:, 208:210].copy().view(np.uint16)[:, 0])
    y = np.empty((b.shape[0], 256), np.float32)
    for half in range(2):
        qlh = ql[:, 64 * half:64 * half + 64]
        qhh = qh[:, 32 * half:32 * half + 32]
        sch = scales[:, 8 * half:8 * half + 8]
        l = np.arange(32)
        is_ = l >> 4                                          # [32] in {0,1}
        q1 = ((qlh[:, :32] & 0x0F) | (((qhh >> 0) & 3) << 4)).astype(np.int32) - 32
        q2 = ((qlh[:, 32:] & 0x0F) | (((qhh >> 2) & 3) << 4)).astype(np.int32) - 32
        q3 = ((qlh[:, :32] >> 4) | (((qhh >> 4) & 3) << 4)).astype(np.int32) - 32
        q4 = ((qlh[:, 32:] >> 4) | (((qhh >> 6) & 3) << 4)).astype(np.int32) - 32
        base = 128 * half
        y[:, base + 0:base + 32]  = sch[:, is_ + 0] * q1
        y[:, base + 32:base + 64] = sch[:, is_ + 2] * q2
        y[:, base + 64:base + 96] = sch[:, is_ + 4] * q3
        y[:, base + 96:base + 128] = sch[:, is_ + 6] * q4
    y *= d[:, None]
    return y.reshape(r, k)
