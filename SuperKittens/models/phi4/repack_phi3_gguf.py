"""repack_phi3_gguf.py — split phi3-arch fused GGUF tensors for the SK loader.

phi3-arch GGUFs (Phi-3/Phi-4 family) carry two fused per-layer projections:

  * ``blk.N.attn_qkv.weight`` — [Q;K;V] row-concatenated
    (rows = n_head*head_dim + 2*n_kv_head*head_dim, ne0 = d_model)
  * ``blk.N.ffn_up.weight``   — [gate;up] row-concatenated (rows = 2*n_ff)

The shared dense loader (``sk_qwen_load_gguf``) expects the separate
qwen/llama-style tensors (attn_q/attn_k/attn_v, ffn_gate/ffn_up). GGUF tensor
data is row-major with each row quantized independently (K-quant superblocks
run along ne0), so splitting along the row dimension is a pure contiguous byte
copy — bit-exact, no dequant/requant. This tool rewrites the GGUF once,
offline; metadata KVs are copied verbatim (the SK loader takes dims from the
Python-side Config, not from GGUF KVs).

Usage:
    python3 repack_phi3_gguf.py in.gguf out.gguf

Reads fused-split geometry (head counts, head_dim, n_ff) from the GGUF header.
Non-fused tensors are passed through untouched.
"""
from __future__ import annotations

import struct
import sys

ALIGN = 32  # GGUF default; general.alignment override is honored below.

# GGML dtype code -> (block_size, bytes_per_block) for row-size math.
GGML_BLOCK = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    30: (1, 2),      # BF16
}

_SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "?", 10: "Q", 11: "q", 12: "d"}


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _skip_value(f, vtype, kvs, key):
    if vtype in _SCALAR_FMT:
        fmt = _SCALAR_FMT[vtype]
        (v,) = struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))
        kvs[key] = v
    elif vtype == 8:
        kvs[key] = _read_str(f)
    elif vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype == 8:
            for _ in range(count):
                _read_str(f)
        elif etype in _SCALAR_FMT:
            f.seek(count * struct.calcsize(_SCALAR_FMT[etype]), 1)
        else:
            raise ValueError(f"bad array elem type {etype} for {key}")
        kvs[key] = f"<array[{count}]>"
    else:
        raise ValueError(f"bad kv type {vtype} for {key}")


def row_bytes(dtype: int, ne0: int) -> int:
    bs, bb = GGML_BLOCK[dtype]
    if ne0 % bs:
        raise ValueError(f"ne0 {ne0} not divisible by block {bs} (dtype {dtype})")
    return ne0 // bs * bb


def parse_header(path: str):
    """Returns (kvs, tensors, kv_raw, data_start). tensors: [name, dims, dtype, off]."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {magic!r}")
        (version,) = struct.unpack("<I", f.read(4))
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        kvs: dict = {}
        kv_start = f.tell()
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            _skip_value(f, vtype, kvs, key)
        kv_end = f.tell()
        f.seek(kv_start)
        kv_raw = f.read(kv_end - kv_start)

        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            dtype, off = struct.unpack("<IQ", f.read(12))
            tensors.append([name, dims, dtype, off])
        info_end = f.tell()

    align = int(kvs.get("general.alignment", ALIGN))
    data_start = (info_end + align - 1) // align * align
    return kvs, tensors, kv_raw, data_start, align, n_kv


def repack(src: str, dst: str) -> None:
    kvs, tensors, kv_raw, data_start, align, n_kv = parse_header(src)

    arch = kvs.get("general.architecture")
    if arch != "phi3":
        raise SystemExit(f"expected phi3 arch, got {arch!r}")
    n_head = int(kvs["phi3.attention.head_count"])
    n_kv_head = int(kvs["phi3.attention.head_count_kv"])
    d_model = int(kvs["phi3.embedding_length"])
    n_ff = int(kvs["phi3.feed_forward_length"])
    head_dim = d_model // n_head
    nq, nkv = n_head * head_dim, n_kv_head * head_dim

    out_tensors = []  # (name, dims, dtype, src_byte_off, nbytes)
    for name, dims, dtype, off in tensors:
        rb = row_bytes(dtype, dims[0])
        nrows = dims[1] if len(dims) > 1 else 1
        base = data_start + off
        if name.endswith(".attn_qkv.weight"):
            if nrows != nq + 2 * nkv:
                raise SystemExit(f"{name}: rows {nrows} != q+2kv {nq + 2 * nkv}")
            blk = name[: -len("attn_qkv.weight")]
            out_tensors.append((blk + "attn_q.weight", [dims[0], nq], dtype, base, nq * rb))
            out_tensors.append((blk + "attn_k.weight", [dims[0], nkv], dtype, base + nq * rb, nkv * rb))
            out_tensors.append((blk + "attn_v.weight", [dims[0], nkv], dtype, base + (nq + nkv) * rb, nkv * rb))
        elif name.endswith(".ffn_up.weight") and nrows == 2 * n_ff:
            blk = name[: -len("ffn_up.weight")]
            out_tensors.append((blk + "ffn_gate.weight", [dims[0], n_ff], dtype, base, n_ff * rb))
            out_tensors.append((blk + "ffn_up.weight", [dims[0], n_ff], dtype, base + n_ff * rb, n_ff * rb))
        else:
            out_tensors.append((name, dims, dtype, base, nrows * rb))

    # New tensor-info block with recomputed (aligned) data offsets.
    infos = bytearray()
    data_off = 0
    placed = []  # (src_off, nbytes, dst_rel_off)
    for name, dims, dtype, src_off, nbytes in out_tensors:
        data_off = (data_off + align - 1) // align * align
        nb = name.encode("utf-8")
        infos += struct.pack("<Q", len(nb)) + nb
        infos += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", dtype, data_off)
        placed.append((src_off, nbytes, data_off))
        data_off += nbytes

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(b"GGUF" + struct.pack("<IQQ", 3, len(out_tensors), n_kv))
        fout.write(kv_raw)
        fout.write(infos)
        new_data_start = (fout.tell() + align - 1) // align * align
        fout.write(b"\x00" * (new_data_start - fout.tell()))
        for src_off, nbytes, rel in placed:
            fout.seek(new_data_start + rel)
            fin.seek(src_off)
            left = nbytes
            while left:
                chunk = fin.read(min(left, 64 << 20))
                if not chunk:
                    raise SystemExit(f"short read at src_off={src_off}")
                fout.write(chunk)
                left -= len(chunk)

    print(f"repacked {len(tensors)} -> {len(out_tensors)} tensors; "
          f"geometry: heads={n_head} kv={n_kv_head} head_dim={head_dim} n_ff={n_ff}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    repack(sys.argv[1], sys.argv[2])
