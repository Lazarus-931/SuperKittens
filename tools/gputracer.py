#!/usr/bin/env python3
"""gputracer — single-file Apple ``.gputrace`` bundle parser + reporter for SuperKittens.

Ported and consolidated from MetalBench (``agent_steel/profiler/gputrace`` —
bundle/capture/metadata), kept faithful to the original reverse-engineering.

A ``.gputrace`` is a *command-intent recording*, NOT a profile log: it carries
WHAT was dispatched and HOW — the compute pipeline (kernel function name), the
dispatch grid + threadgroup, and every bound MTLBuffer (index, offset, label,
byte length). It has NO per-dispatch GPU timestamps, NO counter samples, NO
occupancy. Xcode reconstructs timing by *replaying* the captured commands.

So: use this for "what ran and how it was launched" (great for spotting a wrong
grid/threadgroup, a missing/oversized buffer binding, or an unexpected extra
dispatch), and pair it with the SK bench harness
(``SuperKittens/benchmark/harness`` — GPU-timed reps + roofline) for "how fast".

Usage
-----
Parse an existing capture (pure stdlib, runs anywhere)::

    python tools/gputracer.py path/to/kernel.gputrace          # findings report
    python tools/gputracer.py path/to/kernel.gputrace --json   # machine-readable
    python tools/gputracer.py path/to/kernel.gputrace --raw     # + every dispatch

Capture a SK kernel's trace then report (needs a GPU host + MTL_CAPTURE_ENABLED=1;
use ``tools/gputracer.sh`` which sets that up)::

    python tools/gputracer.py capture out.gputrace \\
        --src SuperKittens/kernels/norm/rmsnorm.metal --fn rmsnorm \\
        --grid 4096,1,1 --tg 256,1,1 --buf 16384 --buf 16384 --buf 64

``parse(path) -> dict`` is also importable for programmatic use.
"""
# pyobjc's ``Metal`` module resolves its symbols dynamically at runtime; Pyright
# can't see them statically (same noise the bench harness suppresses).
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import json
import os
import plistlib
import struct
import sys
from dataclasses import dataclass
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# metadata: the ``metadata`` binary plist (capture session info)
# ─────────────────────────────────────────────────────────────────────────────


def parse_metadata(path: str) -> dict:
    """Load the bundle's ``metadata`` bplist. Returns {} if missing."""
    mpath = os.path.join(path, "metadata")
    if not os.path.exists(mpath):
        return {}
    with open(mpath, "rb") as f:
        try:
            return plistlib.load(f)
        except Exception as e:  # noqa: BLE001
            return {"_parse_error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# capture: the ``capture`` / ``unsorted-capture`` MTSP record stream
#
# Format (reverse-engineered, macOS 15/26 + Xcode 16/17 single-frame compute):
#     magic: 4 bytes b"MTSP"; version: u32 (observed 0x4)
#     records: stream of [size:u32][type:u32][payload of size-8 bytes]
# Each record payload starts with 24 reserved bytes, then a length-tag + a
# NUL-terminated Obj-C type-encoding string padded to a type-dependent
# alignment, then the typed field area.
# ─────────────────────────────────────────────────────────────────────────────

RT_PIPELINE     = 0xffffc05e   # newComputePipelineState — carries function name
RT_RESOURCE     = 0xffffc00c   # labelObject — sets MTLBuffer label
RT_NEW_BUFFER   = 0xffffc046   # newBufferWithLength:options: (older single-stream captures)
RT_BUFFER_DESC  = 0xffffd803   # buffer descriptor (macOS 26 / Xcode 17 device-resources): handle+length+label in one record
RT_CB           = 0xffffc013   # command buffer
RT_ENCODER      = 0xffffc02d   # compute command encoder
RT_SET_BUFFER   = 0xffffc030   # setBuffer:offset:atIndex:
RT_DISPATCH     = 0xffffc132   # dispatchThreads:threadsPerThreadgroup:
RT_COMMIT       = 0xffffc017   # commit
RT_WAIT         = 0xffffc01d   # waitUntilCompleted
RT_END_ENCODING = 0xffffc03b   # endEncoding

RECORD_NAMES = {
    RT_PIPELINE: "computePipelineState",
    RT_RESOURCE: "labelObject",
    RT_NEW_BUFFER: "newBufferWithLength",
    RT_BUFFER_DESC: "bufferDescriptor",
    RT_CB: "commandBuffer",
    RT_ENCODER: "computeCommandEncoder",
    RT_SET_BUFFER: "setBuffer",
    RT_DISPATCH: "dispatchThreads",
    RT_COMMIT: "commit",
    RT_WAIT: "waitUntilCompleted",
    RT_END_ENCODING: "endEncoding",
}


@dataclass
class Record:
    offset: int
    size: int
    type: int
    payload: bytes


def iter_records(data: bytes) -> list[Record]:
    """Walk the MTSP record stream, skipping the 8-byte file header."""
    if data[:4] != b"MTSP":
        raise ValueError("not an MTSP capture stream")
    off = 8
    recs: list[Record] = []
    while off + 8 <= len(data):
        size, typ = struct.unpack_from("<II", data, off)
        if size < 8 or size > len(data) - off:
            break
        recs.append(Record(off, size, typ, data[off + 8:off + size]))
        off += size
    return recs


def _read_cstring(buf: bytes, offset: int, maxlen: int = 256) -> str:
    end = buf.find(b"\x00", offset, offset + maxlen)
    if end < 0:
        end = min(offset + maxlen, len(buf))
    return buf[offset:end].decode("utf-8", errors="replace")


def _parse_type_encoded(payload: bytes, align: int = 4) -> tuple[str, int]:
    """Return (type-encoding string, byte-offset where the typed-field area starts).

    Preamble shared by typed records: [24 reserved=0][4-byte tag][type-encoding
    ASCII, NUL-terminated, padded to ``align``][typed fields...]. ``align`` is
    record-type-dependent (4 for setBuffer, 8 for dispatchThreads).
    """
    if len(payload) < 28:
        return "", 28
    te_off = 28
    te_end = payload.find(b"\x00", te_off)
    if te_end < 0:
        return "", 28
    te = payload[te_off:te_end].decode("ascii", errors="replace")
    field_start = ((te_end + 1) + align - 1) & ~(align - 1)
    return te, field_start


def parse_pipeline(rec: Record) -> dict:
    # function name is a NUL-terminated C-string at payload offset 40
    return {"_kind": "pipeline", "function": _read_cstring(rec.payload, 40)}


def parse_new_buffer(rec: Record) -> dict | None:
    """newBufferWithLength:options: — te='Cul...'; fields: device(u64), length(u64), options(u64)."""
    te, fs = _parse_type_encoded(rec.payload, align=4)
    if not te.startswith("Cul"):
        return None
    pl = rec.payload
    if fs + 24 > len(pl):
        return None
    device = struct.unpack_from("<Q", pl, fs)[0]
    length = struct.unpack_from("<Q", pl, fs + 8)[0]
    options = struct.unpack_from("<Q", pl, fs + 16)[0]
    return {"_kind": "newBuffer", "device": device, "length": length, "options": options}


def parse_resource(rec: Record) -> dict:
    # label C-string at offset 40, object handle (u64) at offset 32
    label = _read_cstring(rec.payload, 40)
    handle = struct.unpack_from("<Q", rec.payload, 32)[0] if len(rec.payload) >= 40 else 0
    return {"_kind": "resource", "label": label, "handle": handle}


def parse_buffer_desc(rec: Record) -> dict | None:
    """macOS 26 / Xcode 17 buffer descriptor (in device-resources) — handle, length, label in one record.

    te='CU<b>ulul' (align=4 -> fields at offset 40): device(u64), label(C-string),
    length(u64 @ fs+24), options(u64 @ fs+32), then a 4-byte tag + the buffer
    handle (u64 @ fs+44). The setBuffer records in the capture stream reference
    that same handle, so this is what links a binding index to its byte length.
    """
    te, fs = _parse_type_encoded(rec.payload, align=4)
    if "ulul" not in te:
        return None
    pl = rec.payload
    if fs + 52 > len(pl):
        return None
    label = _read_cstring(pl, fs + 8)
    length = struct.unpack_from("<Q", pl, fs + 24)[0]
    options = struct.unpack_from("<Q", pl, fs + 32)[0]
    handle = struct.unpack_from("<Q", pl, fs + 44)[0]
    return {"_kind": "bufferDesc", "handle": handle, "length": length,
            "options": options, "label": label or None}


def parse_cb(rec: Record) -> dict:
    return {"_kind": "commandBuffer", "label": _read_cstring(rec.payload, 40)}


def parse_encoder(rec: Record) -> dict:
    return {"_kind": "encoder", "label": _read_cstring(rec.payload, 40)}


def parse_set_buffer(rec: Record) -> dict | None:
    # te = "Ctulul" -> char, ptr(encoder), ulong(buffer-handle), ulong(offset), ulong(index)
    te, fs = _parse_type_encoded(rec.payload, align=4)
    if not te.startswith("Ctul"):
        return None
    pl = rec.payload
    if fs + 32 > len(pl):
        return None
    return {
        "_kind": "setBuffer",
        "encoder": struct.unpack_from("<Q", pl, fs)[0],
        "buffer": struct.unpack_from("<Q", pl, fs + 8)[0],
        "offset": struct.unpack_from("<Q", pl, fs + 16)[0],
        "index": struct.unpack_from("<Q", pl, fs + 24)[0],
    }


def parse_dispatch(rec: Record) -> dict | None:
    # te = "C@3ul@3ul" -> char, MTLSize grid (3 u64), MTLSize tg (3 u64)
    # layout after te-padding: encoder(u64), grid(3*u64), tg(3*u64)
    te, fs = _parse_type_encoded(rec.payload, align=8)
    if "3ul" not in te:
        return None
    pl = rec.payload
    if fs + 8 + 48 > len(pl):
        return None
    encoder = struct.unpack_from("<Q", pl, fs)[0]
    grid = list(struct.unpack_from("<3Q", pl, fs + 8))
    tg = list(struct.unpack_from("<3Q", pl, fs + 8 + 24))
    return {"_kind": "dispatchThreads", "encoder": encoder, "grid": grid, "threadgroup": tg}


def parse_capture(path: str) -> dict:
    """Parse capture + unsorted-capture into a logical command-buffer view.

    Heuristic: stream records in order, accumulate handle->label, and bind each
    dispatch to the most-recent encoder + pipeline. Holds for the small-encoder
    single-frame captures this tool targets.
    """
    out: dict[str, Any] = {"command_buffers": [], "_record_counts": {}, "_unknown_record_types": {}}

    # device-resources-* streams carry the buffer descriptors + (on macOS 26 /
    # Xcode 17) the compute pipeline, so they must be walked BEFORE the capture
    # stream's dispatch is bound — otherwise the function name and bound-buffer
    # byte lengths are not yet known when the dispatch record is reached.
    sources: list[tuple[str, bytes]] = []
    for name in sorted(os.listdir(path)) if os.path.isdir(path) else []:
        if name.startswith("device-resources-") or name.startswith("delta-device-resources-"):
            fpath = os.path.join(path, name)
            try:
                with open(fpath, "rb") as f:
                    if f.read(4) == b"MTSP":
                        f.seek(0)
                        sources.append((name, f.read()))
            except OSError:
                pass

    cpath = os.path.join(path, "capture")
    upath = os.path.join(path, "unsorted-capture")
    if os.path.exists(cpath):
        with open(cpath, "rb") as f:
            sources.append(("capture", f.read()))
    elif os.path.exists(upath):
        with open(upath, "rb") as f:
            sources.append(("unsorted-capture", f.read()))

    if not sources:
        return out

    resources: dict[int, dict] = {}
    last_pipeline: str | None = None
    current_cb: dict | None = None
    current_encoder: dict | None = None
    pending_dispatch_bufs: list[dict] = []
    pending_new_buffer_length: int | None = None
    pending_new_buffer_options: int | None = None
    counts: dict[str, int] = {}
    unknown: dict[int, int] = {}

    def bump(s: str) -> None:
        counts[s] = counts.get(s, 0) + 1

    for _, data in sources:
        try:
            recs = iter_records(data)
        except ValueError:
            continue

        for rec in recs:
            name = RECORD_NAMES.get(rec.type)
            if name is None:
                unknown[rec.type] = unknown.get(rec.type, 0) + 1
                continue
            bump(name)

            if rec.type == RT_PIPELINE:
                last_pipeline = parse_pipeline(rec)["function"]

            elif rec.type == RT_BUFFER_DESC:
                info = parse_buffer_desc(rec)
                if info is not None:
                    resources[info["handle"]] = info

            elif rec.type == RT_NEW_BUFFER:
                info = parse_new_buffer(rec)
                if info is not None:
                    pending_new_buffer_length = info["length"]
                    pending_new_buffer_options = info["options"]

            elif rec.type == RT_RESOURCE:
                info = parse_resource(rec)
                if pending_new_buffer_length is not None:
                    info["length"] = pending_new_buffer_length
                    info["options"] = pending_new_buffer_options
                    pending_new_buffer_length = None
                    pending_new_buffer_options = None
                # macOS 26: a bufferDescriptor already carries length+label for this
                # handle, and labelObject arrives after with an empty label — merge
                # so we never clobber a known length or a real label with the blank.
                existing = resources.get(info["handle"])
                if existing is not None:
                    if not info.get("label"):
                        info["label"] = existing.get("label")
                    if info.get("length") is None:
                        info["length"] = existing.get("length")
                        info["options"] = existing.get("options")
                resources[info["handle"]] = info

            elif rec.type == RT_CB:
                current_cb = {"label": parse_cb(rec)["label"], "dispatches": []}
                out["command_buffers"].append(current_cb)

            elif rec.type == RT_ENCODER:
                current_encoder = {"label": parse_encoder(rec)["label"]}
                pending_dispatch_bufs = []

            elif rec.type == RT_SET_BUFFER:
                info = parse_set_buffer(rec)
                if info is not None:
                    bufrec = resources.get(info["buffer"], {})
                    pending_dispatch_bufs.append({
                        "index": info["index"],
                        "offset": info["offset"],
                        "label": bufrec.get("label"),
                        "length": bufrec.get("length"),
                        "options": bufrec.get("options"),
                        "handle": info["buffer"],
                    })

            elif rec.type == RT_DISPATCH:
                info = parse_dispatch(rec)
                if info is None:
                    continue
                disp = {
                    "function": last_pipeline,
                    "grid": info["grid"],
                    "threadgroup": info["threadgroup"],
                    "encoder_label": current_encoder["label"] if current_encoder else None,
                    "buffers": list(pending_dispatch_bufs),
                }
                if current_cb is None:
                    current_cb = {"label": None, "dispatches": []}
                    out["command_buffers"].append(current_cb)
                current_cb["dispatches"].append(disp)
                pending_dispatch_bufs = []

            elif rec.type == RT_END_ENCODING:
                current_encoder = None
                pending_dispatch_bufs = []
            # RT_COMMIT, RT_WAIT: nothing we need

    out["_record_counts"] = counts
    out["_unknown_record_types"] = {hex(k): v for k, v in unknown.items()}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# bundle: file enumeration + the parse(path) entry point
# ─────────────────────────────────────────────────────────────────────────────


def _classify(name: str, full_path: str) -> str:
    if name == "metadata":
        return "bplist"
    if name in ("capture", "unsorted-capture"):
        return "mtsp_capture"
    if name.startswith(("device-resources-", "delta-device-resources-", "unused-device-resources-")):
        return "mtsp_resources"
    if name == "index":
        return "index"
    if name == "store0":
        return "zlib_store"
    if name.startswith("startup-"):
        return "platform_info"
    try:
        with open(full_path, "rb") as f:
            head = f.read(8)
    except OSError:
        return "unknown"
    if head[:4] == b"MTLB":
        return "metallib"
    if head[:4] == b"MTSP":
        return "mtsp_capture"
    return "unknown"


def list_files(path: str) -> dict[str, dict]:
    """Return {filename: {size, kind}} for each file in the bundle dir."""
    out: dict[str, dict] = {}
    if not os.path.isdir(path):
        return out
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        out[name] = {"size": os.path.getsize(full), "kind": _classify(name, full)}
    return out


def parse(path: str) -> dict[str, Any]:
    """Parse a ``.gputrace`` bundle into a profile-oriented dict."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"not a .gputrace bundle dir: {path}")

    files = list_files(path)
    md = parse_metadata(path)
    cap = parse_capture(path)

    link_versions = md.get("DYCaptureSession.library_link_time_versions")
    device = {
        "device_id": md.get("DYCaptureSession.deviceId"),
        "graphics_api": md.get("DYCaptureSession.graphics_api"),
        "capture_version": md.get("DYCaptureSession.capture_version"),
        "metal_link_version": link_versions.get("Metal") if isinstance(link_versions, dict) else None,
        "captured_frames": md.get("DYCaptureEngine.captured_frames_count"),
    }

    metallib = None
    for fname, info in files.items():
        if info["kind"] == "metallib":
            metallib = {"name": fname, "size": info["size"]}
            break

    return {
        "bundle_path": os.path.abspath(path),
        "device": device,
        "metallib": metallib,
        "files": files,
        "command_buffers": cap.get("command_buffers", []),
        "_diagnostics": {
            "record_counts": cap.get("_record_counts", {}),
            "unknown_record_types": cap.get("_unknown_record_types", {}),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# findings report — MetalBench-style, with an SK per-kernel rollup
# ─────────────────────────────────────────────────────────────────────────────


def _prod(xs) -> int:
    n = 1
    for x in xs or []:
        n *= int(x)
    return n


def per_kernel_rollup(parsed: dict) -> list[dict]:
    """Aggregate dispatches by kernel function: count, launch config, bytes bound.

    ``bytes_bound`` (sum of unique bound-buffer lengths per dispatch) is the hook
    into the roofline: compare it against ``benchmark/harness/roofline.py`` to
    flag a kernel that moves more memory than its math warrants.
    """
    agg: dict[str, dict] = {}
    for cb in parsed.get("command_buffers", []):
        for d in cb.get("dispatches", []):
            fn = d.get("function") or "<unknown>"
            row = agg.setdefault(fn, {
                "function": fn, "dispatches": 0, "grids": set(), "tgs": set(),
                "total_threads": 0, "bytes_bound": 0, "max_buffers": 0,
            })
            row["dispatches"] += 1
            g, tg = tuple(d.get("grid") or []), tuple(d.get("threadgroup") or [])
            row["grids"].add(g)
            row["tgs"].add(tg)
            row["total_threads"] += _prod(g)
            seen: set[int] = set()
            for b in d.get("buffers", []):
                h = b.get("handle")
                if h in seen:
                    continue
                seen.add(h)
                row["bytes_bound"] += int(b.get("length") or 0)
            row["max_buffers"] = max(row["max_buffers"], len(d.get("buffers", [])))
    rows = []
    for r in agg.values():
        r["grids"] = sorted(r["grids"])
        r["tgs"] = sorted(r["tgs"])
        rows.append(r)
    rows.sort(key=lambda r: (-r["dispatches"], r["function"]))
    return rows


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "-"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{int(f)}{unit}" if unit == "B" else f"{f:.1f}{unit}"
        f /= 1024
    return f"{n}B"


def _xyz(t) -> str:
    return "x".join(str(v) for v in t) if t else "-"


def report(parsed: dict, *, raw: bool = False) -> str:
    L: list[str] = []
    L.append(f"gputrace findings — {parsed.get('bundle_path')}")
    dev = parsed.get("device", {})
    L.append("")
    L.append("device:")
    for k in ("device_id", "graphics_api", "capture_version", "metal_link_version", "captured_frames"):
        if dev.get(k) is not None:
            L.append(f"    {k:<20} {dev[k]}")
    mlib = parsed.get("metallib")
    if mlib:
        L.append(f"    {'metallib':<20} {mlib['name']} ({_fmt_bytes(mlib['size'])})")

    cbs = parsed.get("command_buffers", [])
    n_disp = sum(len(cb.get("dispatches", [])) for cb in cbs)
    L.append("")
    L.append(f"command buffers: {len(cbs)}   dispatches: {n_disp}")

    rollup = per_kernel_rollup(parsed)
    if rollup:
        L.append("")
        L.append("per-kernel (command-intent — pair with the bench harness for timing):")
        L.append(f"    {'kernel':<34} {'disp':>4}  {'grid':<16} {'threadgroup':<12} {'threads':>13}  {'bound':>8}")
        for r in rollup:
            grid = _xyz(r["grids"][0]) if len(r["grids"]) == 1 else f"{len(r['grids'])} shapes"
            tg = _xyz(r["tgs"][0]) if len(r["tgs"]) == 1 else f"{len(r['tgs'])} shapes"
            L.append(f"    {r['function'][:34]:<34} {r['dispatches']:>4}  {grid:<16} {tg:<12} "
                     f"{r['total_threads']:>13,}  {_fmt_bytes(r['bytes_bound']):>8}")

    if raw:
        L.append("")
        L.append("dispatches (in capture order):")
        for ci, cb in enumerate(cbs):
            L.append(f"  command buffer #{ci}  label={cb.get('label')!r}  ({len(cb.get('dispatches', []))} dispatch)")
            for di, d in enumerate(cb.get("dispatches", [])):
                L.append(f"    [{di}] {d.get('function')}  grid={_xyz(d.get('grid'))}  tg={_xyz(d.get('threadgroup'))}")
                for b in sorted(d.get("buffers", []), key=lambda x: x.get("index", 0)):
                    L.append(f"          buf[{b.get('index')}] {_fmt_bytes(b.get('length'))}"
                             f"  off={b.get('offset')}  label={b.get('label')!r}")

    diag = parsed.get("_diagnostics", {})
    if diag.get("record_counts"):
        L.append("")
        L.append("record counts: " + ", ".join(f"{k}={v}" for k, v in sorted(diag["record_counts"].items())))
    if diag.get("unknown_record_types"):
        L.append("unknown record types: " + ", ".join(f"{k}={v}" for k, v in diag["unknown_record_types"].items()))
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# capture — run one SK kernel under Metal frame capture, write a .gputrace
# (pyobjc + a GPU host; needs MTL_CAPTURE_ENABLED=1 — tools/gputracer.sh sets it)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_xyz(s: str, default=(1, 1, 1)) -> tuple[int, int, int]:
    if not s:
        return default
    parts = [int(x) for x in s.replace("x", ",").split(",") if x != ""]
    parts = (parts + [1, 1, 1])[:3]
    return parts[0], parts[1], parts[2]


def _set_fc(Metal, cv, idx: int, typ: str, val: str) -> None:
    import ctypes
    t = typ.lower()
    if t in ("bool", "b"):
        v = ctypes.c_bool(val not in ("0", "false", "False")); dt = Metal.MTLDataTypeBool
    elif t in ("int", "i"):
        v = ctypes.c_int32(int(val)); dt = Metal.MTLDataTypeInt
    elif t in ("uint", "u"):
        v = ctypes.c_uint32(int(val)); dt = Metal.MTLDataTypeUInt
    elif t in ("short", "s"):
        v = ctypes.c_int16(int(val)); dt = Metal.MTLDataTypeShort
    elif t in ("ushort", "us"):
        v = ctypes.c_uint16(int(val)); dt = Metal.MTLDataTypeUShort
    elif t in ("float", "f"):
        v = ctypes.c_float(float(val)); dt = Metal.MTLDataTypeFloat
    else:
        raise ValueError(f"unknown function-constant type {typ!r}")
    cv.setConstantValue_type_atIndex_(memoryview(v).tobytes(), dt, idx)


def capture(out_path: str, *, fn: str, src: str | None = None, metallib: str | None = None,
            grid=(1, 1, 1), tg=(1, 1, 1), bufs: list[int] | None = None,
            fc: list[str] | None = None) -> str:
    """Dispatch ``fn`` once inside a Metal capture scope and write ``out_path``.

    Library comes from a runtime-compiled ``--src`` (newLibraryWithSource — no
    Xcode needed, matches SK's SK_METAL_SRC_FALLBACK) or a prebuilt ``--metallib``.
    Buffers are zero-filled device buffers sized by ``--buf`` (one per binding
    index, in order). Structural capture (what gets dispatched), not a
    correctness or timing run.
    """
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("set MTL_CAPTURE_ENABLED=1 before capturing (use tools/gputracer.sh)")
    try:
        import Metal  # type: ignore
        from Foundation import NSURL  # type: ignore
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError("capture needs pyobjc (pip install pyobjc-framework-Metal) on a GPU host") from e

    dev = Metal.MTLCreateSystemDefaultDevice()
    if dev is None:
        raise RuntimeError("no Metal device (capture must run on a GPU host, e.g. a mini)")

    if src:
        with open(src, "r") as f:
            source = f.read()
        opts = Metal.MTLCompileOptions.alloc().init()
        lib, err = dev.newLibraryWithSource_options_error_(source, opts, None)
        if lib is None:
            raise RuntimeError(f"newLibraryWithSource failed: {err}")
    elif metallib:
        lib, err = dev.newLibraryWithURL_error_(NSURL.fileURLWithPath_(metallib), None)
        if lib is None:
            raise RuntimeError(f"newLibraryWithURL failed: {err}")
    else:
        raise RuntimeError("capture needs --src KERNEL.metal or --metallib LIB.metallib")

    if fc:
        cv = Metal.MTLFunctionConstantValues.alloc().init()
        for spec in fc:  # "index:type:value"
            idx_s, typ, val_s = spec.split(":")
            _set_fc(Metal, cv, int(idx_s), typ, val_s)
        func, err = lib.newFunctionWithName_constantValues_error_(fn, cv, None)
        if func is None:
            raise RuntimeError(f"specialised function {fn!r} failed: {err}")
    else:
        func = lib.newFunctionWithName_(fn)
        if func is None:
            raise RuntimeError(f"function {fn!r} not in library; have: {list(lib.functionNames())}")

    pso, err = dev.newComputePipelineStateWithFunction_error_(func, None)
    if pso is None:
        raise RuntimeError(f"PSO build failed: {err}")

    queue = dev.newCommandQueue()
    buffers = [dev.newBufferWithLength_options_(max(1, n), Metal.MTLResourceStorageModeShared)
               for n in (bufs or [])]

    mgr = Metal.MTLCaptureManager.sharedCaptureManager()
    desc = Metal.MTLCaptureDescriptor.alloc().init()
    desc.setCaptureObject_(dev)
    desc.setDestination_(Metal.MTLCaptureDestinationGPUTraceDocument)
    if os.path.exists(out_path):
        import shutil
        shutil.rmtree(out_path, ignore_errors=True)
    desc.setOutputURL_(NSURL.fileURLWithPath_(os.path.abspath(out_path)))
    ok, err = mgr.startCaptureWithDescriptor_error_(desc, None)
    if not ok:
        raise RuntimeError(f"startCapture failed (MTL_CAPTURE_ENABLED=1?): {err}")

    cb = queue.commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pso)
    for i, b in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(b, 0, i)
    enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(*tg))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    mgr.stopCapture()
    return os.path.abspath(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _emit(parsed: dict, as_json: bool, raw: bool) -> None:
    print(json.dumps(parsed, indent=2, default=list) if as_json else report(parsed, raw=raw))


def _cmd_parse(args) -> int:
    _emit(parse(args.bundle), args.json, args.raw)
    return 0


def _cmd_capture(args) -> int:
    out = capture(
        args.out, fn=args.fn, src=args.src, metallib=args.metallib,
        grid=_parse_xyz(args.grid), tg=_parse_xyz(args.tg),
        bufs=[int(x) for x in (args.buf or [])], fc=args.fc or [],
    )
    print(f"captured -> {out}\n", file=sys.stderr)
    _emit(parse(out), args.json, args.raw)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `capture` subcommand — kept separate from the default parse parser so a
    # bundle path is never mistaken for a subcommand choice.
    if argv and argv[0] == "capture":
        cap = argparse.ArgumentParser(prog="gputracer capture",
                                      description="run a kernel once under Metal capture, then report")
        cap.add_argument("out", help="output .gputrace path")
        grp = cap.add_mutually_exclusive_group(required=True)
        grp.add_argument("--src", help="kernel .metal source (runtime-compiled; no Xcode needed)")
        grp.add_argument("--metallib", help="prebuilt .metallib")
        cap.add_argument("--fn", required=True, help="kernel function name")
        cap.add_argument("--grid", default="1,1,1", help="dispatch grid X,Y,Z (threads)")
        cap.add_argument("--tg", default="1,1,1", help="threadgroup X,Y,Z")
        cap.add_argument("--buf", action="append", help="zero buffer size in bytes (repeat, in binding order)")
        cap.add_argument("--fc", action="append", help="function constant index:type:value (repeat)")
        cap.add_argument("--json", action="store_true")
        cap.add_argument("--raw", action="store_true")
        return _cmd_capture(cap.parse_args(argv[1:]))

    # default: parse `gputracer <bundle> [--json] [--raw]`
    ap = argparse.ArgumentParser(prog="gputracer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="capture mode: gputracer capture <out.gputrace> --src K.metal --fn F ...")
    ap.add_argument("bundle", nargs="?", help="path to a .gputrace bundle to parse")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of the findings report")
    ap.add_argument("--raw", action="store_true", help="also print every dispatch + its buffer bindings")
    args = ap.parse_args(argv)
    if not args.bundle:
        ap.print_help()
        return 2
    return _cmd_parse(args)


if __name__ == "__main__":
    raise SystemExit(main())
