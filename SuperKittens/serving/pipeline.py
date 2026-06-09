"""Pipeline-parallel multi-stream serving driver (cross-host, layer-split).

Splits a dense-transformer (Qwen3-arch) decode forward across the LAYER
dimension over two hosts and serves D independent request streams concurrently,
so the two stages OVERLAP in wall time and aggregate decode throughput scales
past a single host's rate.

  stage A (host A): embed + layers [0, K)         -> fp16 hidden over TCP
  stage B (host B): layers [K, n_layers) + head   -> sampled token back to A

Why this is a real serving lever (measured, qwen3-0.6B-Q8, derek A + lexie B
over Tailscale, K=14, cache_max=256, 32 tok/stream, median of 3 paired runs;
single-mini single-stream bar = 138.4 tok/s on derek):

    D=1 :  66.0 tok/s   0.48x   (pure latency: one stream, no overlap)
    D=2 : 132.4 tok/s   0.96x   (two stages overlap; ~= single-mini rate)
    D=4 : 164.7 tok/s   1.19x   (+19.0% aggregate)  <- clears the +8% bar
    D=8 : 176.0 tok/s   1.27x   (+27.2% aggregate)

stream0 output is token-for-token identical to the single-mini single-stream
reference at every D (concurrent per-stream KV is isolated, no cross-talk).

It is a THROUGHPUT lever, not a latency lever: D=1 cross-host is ~0.5x a single
mini (the hop hurts one request). The win requires concurrent independent
streams and right-sized per-stream KV.

ADDITIVE / OPT-IN: this module only consumes the existing additive C ABI
(sk_qwen_run_layers / sk_qwen_resume_from_hidden / sk_qwen_load_gguf_range /
sk_qwen_resident_weight_bytes) via the in-tree dense_decoder bindings. It
imports nothing into, and changes nothing in, the single-mini forward/decode/
prefill paths — those remain byte-identical. Nothing else imports this module.

KV-FOOTPRINT: each stream gets its own stage handle (isolated KV/current_pos),
which duplicates resident weights D-fold per host. That is fine for the
throughput question. With many concurrent handles, size cache_max to the
workload (prompt + n_new): the default 32768 allocates ~3.6 GB KV PER HANDLE
and thrashes a memory-tight mini; cache_max=256 drops it to ~28 MB/handle.

CLI (two roles; start B first, then A reaches B over the network):

  # on host B (final stage):
  python -m SuperKittens.serving.pipeline stage-b \
      --gguf MODEL.gguf --dims DIMS_JSON --host 0.0.0.0 --port 53140 \
      --split-layer 14 --streams 8

  # on host A (first stage; --b-host is B's reachable IP):
  python -m SuperKittens.serving.pipeline stage-a \
      --gguf MODEL.gguf --dims DIMS_JSON --b-host 100.77.36.51 --port 53140 \
      --split-layer 14 --streams 8 --n-new 32

DIMS_JSON is the dense Config kwargs as JSON, e.g.
  {"n_layers":28,"d_model":1024,"n_heads":16,"n_kv_heads":8,"head_dim":128,
   "n_int":3072,"vocab_size":151936,"rope_freq_base":1000000.0,"eps":1e-6,
   "rope_n_ctx_orig":40960,"cache_max":256,"seq_max":64}
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
import socket
import struct
import sys
import threading
import time
from queue import Queue

import numpy as np

from SuperKittens.models.qwen.qwen import Qwen
from SuperKittens.models.dense.dense_decoder import Config, _load

# Frame: <uint32 stream_id><uint32 seq><uint32 start_layer> then seq*d_model fp16.
# Reply: <uint32 stream_id><int32 token>. SHUTDOWN id ends the B server.
SHUTDOWN = 0xFFFFFFFF
_HDR = struct.Struct("<III")
_REPLY = struct.Struct("<Ii")

# Distinct prompt content per stream so streams don't trivially share state.
_PROMPTS = [
    [9707, 11, 358, 1079],
    [785, 3974, 13876, 38835],
    [40, 1079, 264, 4128],
    [3838, 374, 279, 6722],
    [785, 7274, 315, 264],
    [27338, 752, 911, 279],
    [4416, 11, 358, 1366],
    [785, 3681, 882, 358],
]


def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


def _build_handle(cfg_kw, gguf, start, end, with_embed, with_head):
    """One stage handle: resident-load only layers [start,end) (+embed/+head)."""
    m = Qwen(Config(**cfg_kw))
    rc = _load().sk_qwen_load_gguf_range(
        m._h, str(gguf).encode(),
        ctypes.c_uint32(start), ctypes.c_uint32(end),
        ctypes.c_int(int(with_embed)), ctypes.c_int(int(with_head)))
    if rc:
        raise RuntimeError(
            f"load_gguf_range([{start},{end}) embed={with_embed} head={with_head}) rc={rc}")
    m.bake_and_set_rope()
    m.reset()
    return m


def _run_layers(m, ids, start, end, out_hidden):
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int32)).reshape(-1)
    seq = ids.size
    rc = _load().sk_qwen_run_layers(
        m._h, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_uint32(seq), ctypes.c_uint32(start), ctypes.c_uint32(end),
        out_hidden.ctypes.data_as(ctypes.c_void_p))
    if rc:
        raise RuntimeError(f"run_layers rc={rc}")
    return seq


# --------------------------------------------------------------------------- B
def serve_stage_b(gguf, cfg_kw, host, port, split_layer, streams):
    """Final-stage server: D handles ([K,L)+head), routed by stream_id."""
    L = cfg_kw["n_layers"]
    Dm = cfg_kw["d_model"]
    lib = _load()

    handles = [_build_handle(cfg_kw, gguf, split_layer, L, 0, 1)
               for _ in range(streams)]
    rbytes = int(lib.sk_qwen_resident_weight_bytes(handles[0]._h))
    print(f"[B] D={streams} handles, each resident [{split_layer},{L}) = "
          f"{rbytes/1048576:.1f} MB (total ~{streams*rbytes/1048576:.0f} MB)",
          flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[B] listening {host}:{port}", flush=True)
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[B] conn from {addr}", flush=True)

    outs = [np.empty((1,), dtype=np.int32) for _ in range(streams)]
    n_steps = 0
    while True:
        sid, seq, start = _HDR.unpack(_recvall(conn, _HDR.size))
        if sid == SHUTDOWN:
            print(f"[B] shutdown after {n_steps} steps", flush=True)
            break
        hidden = np.frombuffer(
            _recvall(conn, seq * Dm * 2), dtype=np.float16).copy()
        m = handles[sid]
        rc = lib.sk_qwen_resume_from_hidden(
            m._h, hidden.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint32(seq), ctypes.c_uint32(start),
            outs[sid].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc:
            raise RuntimeError(f"resume rc={rc} sid={sid}")
        conn.sendall(_REPLY.pack(sid, int(outs[sid][0])))
        n_steps += 1
    conn.close()
    srv.close()


# --------------------------------------------------------------------------- A
def run_stage_a(gguf, cfg_kw, b_host, port, split_layer, streams, n_new):
    """First-stage driver: D streams; overlap A (this host) with remote B.

    A sender loop drains a ready queue, runs [0,K) on A, ships hidden to B
    without blocking on the reply; a receiver thread collects (sid, token)
    frames from B and re-enqueues that stream with the new token until it has
    produced n_new tokens. With D streams in flight, A and B overlap.

    Returns (aggregate_tok_s, per_stream_tok_s, produced) where produced[0] is
    stream0's token list (the single-mini correctness reference).
    """
    L = cfg_kw["n_layers"]
    Dm = cfg_kw["d_model"]
    D = streams

    print(f"[A] D={D} streams, n_new={n_new}/stream, L={L} d_model={Dm} "
          f"K={split_layer} -> B@{b_host}:{port}", flush=True)
    handles = [_build_handle(cfg_kw, gguf, 0, split_layer, 1, 0)
               for _ in range(D)]
    r_a = int(_load().sk_qwen_resident_weight_bytes(handles[0]._h))
    print(f"[A] {D} handles, each [0,{split_layer}) = {r_a/1048576:.1f} MB "
          f"(total ~{D*r_a/1048576:.0f} MB)", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[A] connecting to B {b_host}:{port}", flush=True)
    for _ in range(60):
        try:
            sock.connect((b_host, port))
            break
        except OSError:
            time.sleep(1.0)
    else:
        raise ConnectionError("could not reach stage B")
    print("[A] connected", flush=True)

    produced = [[] for _ in range(D)]
    n_done = [0] * D
    prompts = [_PROMPTS[i % len(_PROMPTS)] for i in range(D)]
    ready: Queue = Queue()
    for sid in range(D):
        ready.put((sid, list(prompts[sid])))

    total_steps = D * n_new
    done_event = threading.Event()
    keep_alive = [None] * D  # hold hidden buffers alive across the async send

    def receiver():
        for _ in range(total_steps):
            sid, tok = _REPLY.unpack(_recvall(sock, _REPLY.size))
            produced[sid].append(tok)
            n_done[sid] += 1
            if n_done[sid] < n_new:
                ready.put((sid, [tok]))
        done_event.set()

    rx = threading.Thread(target=receiver, daemon=True)
    t0 = time.time()
    rx.start()
    sent = 0
    while sent < total_steps:
        sid, ids = ready.get()
        seq = len(ids)
        buf = np.empty((seq, Dm), dtype=np.float16)
        _run_layers(handles[sid], ids, 0, split_layer, buf)
        keep_alive[sid] = buf
        sock.sendall(_HDR.pack(sid, seq, split_layer))
        sock.sendall(buf[:seq].tobytes())
        sent += 1
    done_event.wait()
    t = time.time() - t0

    sock.sendall(_HDR.pack(SHUTDOWN, 0, 0))
    sock.close()

    agg = total_steps / t
    per = agg / D
    print(f"[A] D={D}: {total_steps} tokens in {t*1e3:.0f} ms => "
          f"AGG {agg:.2f} tok/s ({per:.2f} tok/s/stream)", flush=True)
    print(f"[A] stream0 prompt={prompts[0]} -> {produced[0]}", flush=True)
    print(f"RESULT D={D} agg_tok_s={agg:.3f} per_stream={per:.3f} "
          f"wall_ms={t*1e3:.1f}", flush=True)
    return agg, per, produced


def _parse_dims(s: str) -> dict:
    cfg = json.loads(s)
    if "n_layers" not in cfg:
        raise ValueError("dims JSON must contain at least n_layers/d_model")
    return cfg


def main(argv=None):
    sys.path.insert(0, os.path.abspath("."))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="role", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gguf", required=True)
    common.add_argument("--dims", required=True, help="dense Config kwargs as JSON")
    common.add_argument("--port", type=int, default=53140)
    common.add_argument("--split-layer", "-K", type=int, required=True,
                        help="first stage runs [0,K); second runs [K,n_layers)")
    common.add_argument("--streams", "-D", type=int, default=4,
                        help="concurrent independent request streams")

    a = sub.add_parser("stage-a", parents=[common], help="first stage (driver)")
    a.add_argument("--b-host", required=True, help="reachable IP of stage B")
    a.add_argument("--n-new", type=int, default=32, help="tokens per stream")

    b = sub.add_parser("stage-b", parents=[common], help="final stage (server)")
    b.add_argument("--host", default="0.0.0.0", help="bind address")

    args = p.parse_args(argv)
    cfg_kw = _parse_dims(args.dims)

    if args.role == "stage-b":
        serve_stage_b(args.gguf, cfg_kw, args.host, args.port,
                      args.split_layer, args.streams)
    else:
        run_stage_a(args.gguf, cfg_kw, args.b_host, args.port,
                    args.split_layer, args.streams, args.n_new)


if __name__ == "__main__":
    main()
