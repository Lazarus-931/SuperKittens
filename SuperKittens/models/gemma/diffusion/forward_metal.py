# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
"""forward_metal.py — DiffusionGemma unified [prompt|canvas] forward on Metal.

Stage-1 correctness driver. Weight matmuls run on SK quant GEMM kernels
(kernels/gemm/gemm_mma.metal, native Q4_K/Q6_K/Q8_0 off the mmap'd GGUF —
no-copy MTLBuffers, the OS pager is the streaming layer). Attention is
GEMM-composed: QK^T (gemm_mma_f16) -> dg_softmax_mask (additive region mask)
-> @V (gemm_mma_f16). Host glue (norms, rope, router, geglu, residuals,
region scalars) is f32 numpy via graph_ref — shared with the CPU oracle so a
layer-dump diff bisects GPU vs host exactly.

Known Stage-1 perf debt (deliberate): per-op CPU round trips, per-expert GEMM
loop, fp16 activation casts at each hop. Stage 3 moves the glue on-device.
"""
from __future__ import annotations

import contextlib
import ctypes
import fcntl
import gc
import mmap
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import objc  # noqa: F401
import Metal  # type: ignore[import-not-found]

from .config import DiffusionGemmaConfig
from .gguf_io import GGUFFile
from .graph_ref import (F32, Weights, build_mask, embed_tokens, gelu_tanh,
                        moe_route, rms_norm, rope_neox, softmax)

_SK_ROOT = Path(__file__).resolve().parents[3]
_KERNEL_SOURCES = [
    _SK_ROOT / "kernels" / "gemm" / "gemm_mma.metal",
    Path(__file__).parent / "dg_kernels.metal",
]
_MMA_BY_TYPE = {"F16": "gemm_mma_f16", "BF16": "gemm_mma_bf16",
                "Q8_0": "gemm_mma_q8_0", "Q4_K": "gemm_mma_q4k",
                "Q6_K": "gemm_mma_q6k", "Q5_0": "dg_gemm_mma_q5_0"}


def _pad32(n: int) -> int:
    return (n + 31) // 32 * 32


def _dontneed(mm, offset: int, nbytes: int):
    """Drop the file-cache pages behind a consumed mmap range. Clean pages are
    evictable anyway, but letting them accumulate (15.6 GB/forward when
    streaming) is what kept shoving the 16 GB host into swap."""
    try:
        pg = mmap.PAGESIZE
        a = offset - (offset % pg)
        ln = (offset + nbytes + pg - 1) // pg * pg - a
        mm.madvise(mmap.MADV_DONTNEED, a, ln)
    except (AttributeError, ValueError, OSError):
        pass


class Timing:
    """Per-forward stage accounting (wall + GPU + bytes). Collection is a few
    thousand clock reads per 50 s forward — always on; printing is opt-in
    (DG_TIMING=1) so A/B runs and profiling read the same code path."""

    def __init__(self):
        self.wall = defaultdict(float)
        self.gpu = defaultdict(float)
        self.bytes = defaultdict(int)
        self.n = defaultdict(int)

    def reset(self):
        self.wall.clear(); self.gpu.clear(); self.bytes.clear(); self.n.clear()

    @contextlib.contextmanager
    def t(self, key: str, nbytes: int = 0):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.wall[key] += time.perf_counter() - t0
            self.bytes[key] += nbytes
            self.n[key] += 1

    def report(self, total_s: float) -> str:
        rows = ["%-22s %8s %8s %8s %6s" % ("stage", "wall_s", "gpu_s", "GB", "n")]
        acc = 0.0
        for k in sorted(self.wall, key=lambda k: -self.wall[k]):
            w = self.wall[k]
            if "(nested)" not in k:     # envelope keys overlap their components
                acc += w
            g = self.gpu.get(k, 0.0)
            gb = self.bytes[k] / 1e9
            rows.append("%-22s %8.2f %8.2f %8.2f %6d" % (k, w, g, gb, self.n[k]))
        rows.append("%-22s %8.2f  (forward %.2f, untracked %.2f)"
                    % ("TOTAL tracked", acc, total_s, total_s - acc))
        return "\n".join(rows)


class MetalCtx:
    """Device + queue + runtime-compiled PSOs (CLT-only hosts: no metallib)."""

    def __init__(self):
        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is None:
            raise RuntimeError("no Metal device")
        self.device = dev
        self.queue = dev.newCommandQueue()
        src = "\n".join(p.read_text() for p in _KERNEL_SOURCES)
        opts = Metal.MTLCompileOptions.alloc().init()
        # bfloat (gemm_mma_bf16) needs Metal >= 3.1 on runtime compile
        opts.setLanguageVersion_(getattr(Metal, "MTLLanguageVersion3_1", (3 << 16) + 1))
        lib, err = dev.newLibraryWithSource_options_error_(src, opts, None)
        if lib is None:
            raise RuntimeError(f"metal compile failed: {err}")
        self.lib = lib
        self._pso = {}
        self._scratch = {}
        self.tim = Timing()
        self._mmaps: list = []   # ctypes views: no-copy mappings live forever
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
        self._libc = libc

    def map_nocopy(self, fd: int, offset: int, length: int):
        """MAP_SHARED + PROT_READ file region as a no-copy MTLBuffer. Python's
        mmap module can't hand pyobjc a writable view of a PROT_READ mapping,
        so the map is made via libc and wrapped with ctypes."""
        addr = self._libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED,
                               fd, offset)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f"mmap({offset}, {length})")
        view = (ctypes.c_char * length).from_address(addr)
        buf = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            view, length, Metal.MTLResourceStorageModeShared, None)
        if buf is None:
            raise RuntimeError(f"no-copy buffer failed ({length} B)")
        self._mmaps.append(view)
        return buf

    def scratch(self, tag: str, nbytes: int):
        """Persistent grow-only buffer per call-site tag. Per-call Metal
        alloc/free churn (activations, ~3 GB/forward) swap-stormed the 16 GB
        host once generation ran many forwards in one process — same failure
        class Stage 1 hit with weights, same fix."""
        b = self._scratch.get(tag)
        if b is None or b.length() < nbytes:
            b = self.buf_empty(nbytes)
            self._scratch[tag] = b
        return b

    def scratch_fill(self, tag: str, arr: np.ndarray):
        with self.tim.t("h2d:acts", arr.nbytes):
            arr = np.ascontiguousarray(arr)
            b = self.scratch(tag, arr.nbytes)
            b.contents().as_buffer(arr.nbytes)[:] = arr.tobytes()
        return b

    def pso(self, name: str):
        p = self._pso.get(name)
        if p is None:
            fn = self.lib.newFunctionWithName_(name)
            if fn is None:
                raise RuntimeError(f"kernel not found: {name}")
            p, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if p is None:
                raise RuntimeError(f"PSO failed for {name}: {err}")
            self._pso[name] = p
        return p

    def buf_from(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        b = self.device.newBufferWithBytes_length_options_(
            arr, arr.nbytes, Metal.MTLResourceStorageModeShared)
        if b is None:
            raise RuntimeError(f"buffer alloc failed ({arr.nbytes} B)")
        return b

    def buf_empty(self, nbytes: int):
        b = self.device.newBufferWithLength_options_(
            nbytes, Metal.MTLResourceStorageModeShared)
        if b is None:
            raise RuntimeError(f"buffer alloc failed ({nbytes} B)")
        return b

    def read(self, buf, dtype, shape, offset: int = 0) -> np.ndarray:
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        with self.tim.t("d2h:acts", n):
            mv = buf.contents().as_buffer(offset + n)[offset:offset + n]
            return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()


class MappedWeights:
    """No-copy weight binding: MAP_SHARED + PROT_READ regions of the GGUF,
    wrapped as MTLBuffers (llama.cpp's Metal mmap pattern; regions split at
    tensor boundaries to fit device.maxBufferLength).

    This is NOT the Stage-1 MAP_PRIVATE disaster: PRIVATE pages turn anonymous
    (un-evictable) on GPU touch, while SHARED read-only pages stay clean file
    cache the pager can drop freely — the GPU faults in exactly what it reads
    and the per-step disk traffic is only the evicted share, not the whole
    15.6 GB the memcpy+MADV_DONTNEED streaming path re-read every forward."""

    def __init__(self, ctx: MetalCtx, gg: GGUFFile):
        self.ctx = ctx
        self.gg = gg
        self._bind: dict[str, tuple] = {}    # name -> (buf, offset_in_buf)
        fd = gg._f.fileno()
        pg = mmap.PAGESIZE
        maxlen = int(ctx.device.maxBufferLength())
        infos = sorted(gg.tensors.values(), key=lambda ti: ti.offset)
        regions: list[list] = []             # [page_start, end, [tis]]
        for ti in infos:
            end = ti.offset + ti.nbytes
            if regions and (end + pg - 1) // pg * pg - regions[-1][0] <= maxlen:
                regions[-1][1] = max(regions[-1][1], end)
                regions[-1][2].append(ti)
            else:
                regions.append([ti.offset // pg * pg, end, [ti]])
        for start, end, tis in regions:
            buf = ctx.map_nocopy(fd, start, (end + pg - 1) // pg * pg - start)
            for ti in tis:
                self._bind[ti.name] = (buf, ti.offset - start)

    def get(self, name: str) -> tuple:
        ti = self.gg.tensors[name]
        buf, delta = self._bind[name]
        return (buf, delta, ti, None)

    def evict_prefix(self, prefix: str):
        return  # the pager owns residency


class WeightBufs:
    """Persistent per-ROLE scratch MTLBuffers, refilled by memcpy per layer.

    Allocation history on the 16 GB host: (1) MAP_PRIVATE no-copy windows —
    GPU access turned touched pages into un-evictable anonymous memory, box
    down twice; (2) per-tensor copied buffers, even with per-layer eviction +
    autorelease pools — Metal-side allocation churn (~0.85 GB/layer, outside
    process rss) still swap-stormed the host. Persistent slots make Metal
    allocation a one-time ~1.5 GB and the steady state pure memcpy."""

    def __init__(self, ctx: MetalCtx, gg: GGUFFile):
        self.ctx = ctx
        self.gg = gg
        self._slots: dict[str, tuple] = {}      # role -> (buf, capacity)
        self._occupant: dict[str, str] = {}     # role -> tensor name in slot
        self._cap: dict[str, int] = {}
        for name, ti in gg.tensors.items():
            r = self._role(name)
            self._cap[r] = max(self._cap.get(r, 0), ti.nbytes)
        # streamed tensors are re-read fully every step: F_NOCACHE pread
        # straight into the slot runs at SSD sequential rate, where the
        # mmap-slice memcpy paid single-threaded page-fault servicing
        # (~0.36 GB/s measured) and needed DONTNEED to avoid filling the
        # page cache with bytes that are about to be re-read anyway
        self._fd = None
        if os.environ.get("DG_STREAM_IO", "pread") == "pread" and hasattr(os, "preadv"):
            self._fd = os.open(gg.path, os.O_RDONLY)
            with contextlib.suppress(AttributeError, OSError):
                fcntl.fcntl(self._fd, fcntl.F_NOCACHE, 1)

    @staticmethod
    def _role(name: str) -> str:
        parts = name.split(".")
        return parts[2] if parts[0] == "blk" else parts[0]

    def get(self, name: str) -> tuple:
        """-> (MTLBuffer, byte_offset, TensorInfo, None); fills the role slot."""
        ti = self.gg.tensors[name]
        role = self._role(name)
        slot = self._slots.get(role)
        if slot is None:
            buf = self.ctx.buf_empty(self._cap[role])
            self._slots[role] = slot = (buf, self._cap[role])
        buf, cap = slot
        if self._occupant.get(role) != name:
            with self.ctx.tim.t(f"wb:{role}", ti.nbytes):
                mv = buf.contents().as_buffer(cap)
                if self._fd is not None:
                    done = 0
                    while done < ti.nbytes:
                        got = os.preadv(self._fd, [mv[done:ti.nbytes]],
                                        ti.offset + done)
                        if got <= 0:
                            raise OSError(f"pread short at {name}+{done}")
                        done += got
                else:
                    mv[:ti.nbytes] = self.gg.mm[ti.offset:ti.offset + ti.nbytes]
                    _dontneed(self.gg.mm, ti.offset, ti.nbytes)
            self._occupant[role] = name
        return (buf, 0, ti, None)

    def evict_prefix(self, prefix: str):
        return  # slots are persistent; kept for driver-loop compatibility


class ResidentWeights:
    """Budgeted resident weight cache. The dense backbone (embed/head, attn,
    dense ffn, SC projections — ~1.6 GB) is copied once into permanent
    MTLBuffers; expert layers are pinned in layer order while total resident
    stays under SK_DG_RESIDENT_GB; the remainder streams through the
    WeightBufs memcpy+DONTNEED slots.

    Why not the pager (MappedWeights): on the 16 GB host the 15.6 GB working
    set cannot stay file-cache-resident, so every forward re-faults nearly
    the whole model from SSD at ~0.4 GB/s — measured break-even with
    streaming. Pinned anonymous copies are not evictable as clean cache, so
    the per-step disk traffic is capped at (model - budget) bytes; the budget
    must leave room for the colima VM + OS + activations or the host swaps
    (watchdog territory)."""

    def __init__(self, ctx: MetalCtx, gg: GGUFFile):
        self.ctx = ctx
        self.gg = gg
        self.stream = WeightBufs(ctx, gg)
        budget = int(float(os.environ.get("SK_DG_RESIDENT_GB", "6.0")) * 1e9)
        self._res: dict[str, tuple] = {}

        def is_gemm(ti):
            return ti.name.endswith(".weight") and ti.type_name != "F32"

        backbone = [ti for ti in gg.tensors.values()
                    if is_gemm(ti) and "_exps" not in ti.name]
        experts = sorted((ti for ti in gg.tensors.values()
                          if is_gemm(ti) and "_exps" in ti.name),
                         key=lambda ti: (int(ti.name.split(".")[1]), ti.name))
        used = 0
        for ti in backbone:                       # mandatory, even over budget
            self._pin(ti)
            used += ti.nbytes
        n_exp = 0
        for ti in experts:                        # fixed prefix in layer order
            if used + ti.nbytes > budget:
                break
            self._pin(ti)
            used += ti.nbytes
            n_exp += 1
        print(f"[resident] {used / 1e9:.2f} GB pinned ({len(backbone)} backbone"
              f" + {n_exp}/{len(experts)} expert tensors; budget"
              f" {budget / 1e9:.2f} GB)", flush=True)

    def _pin(self, ti):
        with objc.autorelease_pool():
            buf = self.ctx.buf_empty(ti.nbytes)
            buf.contents().as_buffer(ti.nbytes)[:] = \
                self.gg.mm[ti.offset:ti.offset + ti.nbytes]
            self._res[ti.name] = (buf, 0, ti, None)
        _dontneed(self.gg.mm, ti.offset, ti.nbytes)

    def get(self, name: str) -> tuple:
        r = self._res.get(name)
        return r if r is not None else self.stream.get(name)

    def evict_prefix(self, prefix: str):
        return


class GemmBatch:
    """Record gemm_mma / dg_softmax_mask dispatches into one command buffer;
    memory barriers split dependent stages. run() commits + waits."""

    def __init__(self, ctx: MetalCtx, label: str = "gemm"):
        self.ctx = ctx
        self.label = label
        self.cmd = ctx.queue.commandBuffer()
        self.enc = self.cmd.computeCommandEncoder()
        self._keep: list = []   # encoder retains resources, but keep pyobjc refs too
        self._u32 = lambda v: np.uint32(v).tobytes()

    def gemm(self, kernel: str, a_buf, a_off: int, w_buf, w_off: int,
             c_buf, c_off: int, M: int, N: int, K: int, ldc: int | None = None):
        ldc = N if ldc is None else ldc
        enc = self.enc
        self._keep += [a_buf, w_buf, c_buf]
        enc.setComputePipelineState_(self.ctx.pso(kernel))
        enc.setBuffer_offset_atIndex_(a_buf, a_off, 0)
        enc.setBuffer_offset_atIndex_(w_buf, w_off, 1)
        enc.setBuffer_offset_atIndex_(c_buf, c_off, 2)
        enc.setBytes_length_atIndex_(self._u32(M), 4, 3)
        enc.setBytes_length_atIndex_(self._u32(N), 4, 4)
        enc.setBytes_length_atIndex_(self._u32(K), 4, 5)
        enc.setBytes_length_atIndex_(self._u32(ldc), 4, 6)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((N + 31) // 32, (M + 31) // 32, 1),
            Metal.MTLSizeMake(64, 1, 1))

    def softmax_mask(self, s_buf, p_buf, mask_buf, rows: int, ncols: int,
                     ntok: int, scale: float = 1.0):
        enc = self.enc
        self._keep += [s_buf, p_buf, mask_buf]
        enc.setComputePipelineState_(self.ctx.pso("dg_softmax_mask"))
        enc.setBuffer_offset_atIndex_(s_buf, 0, 0)
        enc.setBuffer_offset_atIndex_(p_buf, 0, 1)
        enc.setBuffer_offset_atIndex_(mask_buf, 0, 2)
        enc.setBytes_length_atIndex_(self._u32(ncols), 4, 3)
        enc.setBytes_length_atIndex_(self._u32(ntok), 4, 4)
        enc.setBytes_length_atIndex_(np.float32(scale).tobytes(), 4, 5)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(rows, 1, 1), Metal.MTLSizeMake(256, 1, 1))

    def barrier(self):
        self.enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)

    def run(self):
        self.enc.endEncoding()
        key = f"gpu:{self.label}"
        with self.ctx.tim.t(key):
            self.cmd.commit()
            self.cmd.waitUntilCompleted()
        self.ctx.tim.gpu[key] += max(0.0, self.cmd.GPUEndTime() - self.cmd.GPUStartTime())
        if self.cmd.error() is not None:
            raise RuntimeError(f"command buffer failed: {self.cmd.error()}")


class DiffusionGemmaMetal:
    """Stage-1 unified zero-SC forward. forward(ids, P) -> canvas logits f32."""

    def __init__(self, gguf_path: str, cfg: DiffusionGemmaConfig,
                 sc_embt_path: str | None = None):
        self.gg = GGUFFile(gguf_path)
        self.cfg = cfg
        self.ctx = MetalCtx()
        # DG_WEIGHTS: resident (default) | stream (Stage-2 memcpy) | mapped
        # (no-copy pager binding; measured break-even with stream on 16 GB)
        mode = os.environ.get("DG_WEIGHTS", "resident")
        self.wb = {"stream": WeightBufs, "mapped": MappedWeights,
                   "resident": ResidentWeights}[mode](self.ctx, self.gg)
        self.w = Weights(self.gg)   # F32 sidecars (norms, scales, router)
        self.dump = None            # optional (name, il, arr) tap
        # self-conditioning soft-embed: transposed dequantized embed
        # [d_model, vocab] f16 on disk (make_embt.py), no-copy mapped
        self.sc_embt_path = sc_embt_path
        self._sc_bufs = None        # (probs_buf, embt_buf, out_buf, file)

    # -- self-conditioning -------------------------------------------------------

    def _sc_soft_embed(self, probs16: np.ndarray) -> np.ndarray:
        """probs16 fp16 [C, V] @ embed [V, d] -> f32 [C, d] via the no-copy
        transposed-embed GEMM (dg_gemm_qkt_f32: f16 inputs, f32 C — matches
        the reference's f16 sc_embT matmul, f32-accumulated)."""
        cfg = self.cfg
        C, V = probs16.shape
        d = cfg.d_model
        if self._sc_bufs is None:
            f = open(self.sc_embt_path, "rb")
            assert os.fstat(f.fileno()).st_size == d * V * 2, \
                "embT size mismatch (expect [d, V] f16)"
            w_buf = self.ctx.map_nocopy(f.fileno(), 0, d * V * 2)
            self._sc_bufs = (self.ctx.buf_empty(C * V * 2), w_buf,
                             self.ctx.buf_empty(C * d * 4), f)
        p_buf, w_buf, o_buf, _ = self._sc_bufs
        with self.ctx.tim.t("h2d:acts", C * V * 2):
            p_buf.contents().as_buffer(C * V * 2)[:] = probs16.tobytes()
        b = GemmBatch(self.ctx, "sc_embed")
        b.gemm("dg_gemm_qkt_f32", p_buf, 0, w_buf, 0, o_buf, 0,
               M=C, N=d, K=V, ldc=d)
        b.run()
        return self.ctx.read(o_buf, np.float32, (C, d))

    def _sc_signal(self, sc_logits: np.ndarray | None, sc_temp_inv: float,
                   sc_use: float, probs16: np.ndarray | None = None) -> np.ndarray:
        """PR dg_canvas_embed SC subgraph -> sc_sig f32 [C, d_model].
        probs16 (precomputed softmax(prev/t) f16) lets the generation loop
        free the 268 MB raw-logit block between forwards."""
        cfg, w = self.cfg, self.w
        if probs16 is None:
            assert sc_logits is not None
            C, V = sc_logits.shape
            # softmax(prev raw logits / prev t), fp16 on the wire like the
            # reference (ggml converts the f32 probs to the f16 vec_dot type)
            probs16 = np.empty((C, V), np.float16)
            for c0 in range(0, C, 32):
                c1 = min(c0 + 32, C)
                probs16[c0:c1] = softmax(sc_logits[c0:c1] * F32(sc_temp_inv)).astype(np.float16)
        if self.sc_embt_path is not None:
            soft = self._sc_soft_embed(probs16)
        else:  # oracle-style host fallback (slow: full embed dequant per step)
            soft = np.zeros((C, cfg.d_model), F32)
            pf = probs16.astype(F32)
            for v0 in range(0, V, 16384):
                v1 = min(v0 + 16384, V)
                soft += pf[:, v0:v1] @ self.w.dq("token_embd.weight", rows=slice(v0, v1))
        soft = soft * F32(np.sqrt(F32(cfg.d_model)))
        normed = rms_norm(soft, cfg.eps, w.f32("self_cond_pre_norm.weight"))
        g = gelu_tanh(self._gemm_f32("self_cond_gate.weight", normed, cfg.n_ff))
        u = self._gemm_f32("self_cond_up.weight", normed, cfg.n_ff)
        sig = self._gemm_f32("self_cond_down.weight", g * u, cfg.d_model)
        return (sig * F32(sc_use)).astype(F32)

    # -- helpers ---------------------------------------------------------------

    def _wgemm(self, batch: GemmBatch, wname: str, a_buf, a_off: int,
               c_buf, c_off: int, M: int, N: int, K: int,
               row0: int = 0, ldc: int | None = None):
        """C[M,N] = A[M,K] @ W[row0:row0+N, :K]^T for GGUF weight `wname`."""
        buf, delta, ti, _ = self.wb.get(wname)
        kern = _MMA_BY_TYPE[ti.type_name]
        assert ti.shape[0] == K, (wname, ti.shape, K)
        w_off = delta + row0 * ti.row_bytes
        batch.gemm(kern, a_buf, a_off, buf, w_off, c_buf, c_off, M, N, K, ldc)

    def _gemm_f32(self, wname: str, a: np.ndarray, N: int, row0: int = 0) -> np.ndarray:
        """One-shot weight GEMM with f32 host I/O (fp16 on the wire)."""
        M, K = a.shape
        amax = float(np.abs(a).max(initial=0.0))
        if amax > 3.0e4:
            print(f"[warn] fp16 activation near overflow ({amax:.1f}) into {wname}")
        a_buf = self.ctx.scratch_fill(f"A:{wname}", a.astype(np.float16))
        c_buf = self.ctx.scratch(f"C:{wname}", M * N * 2)
        b = GemmBatch(self.ctx, WeightBufs._role(wname))
        self._wgemm(b, wname, a_buf, 0, c_buf, 0, M, N, K, row0=row0)
        b.run()
        return self.ctx.read(c_buf, np.float16, (M, N)).astype(F32)

    # -- attention -------------------------------------------------------------

    def _attention(self, il: int, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                   mask: np.ndarray) -> np.ndarray:
        """q [N,H,hd], k/v [N,Kv,hd] (post norm+rope, f32) -> [N, H*hd] f32."""
        cfg = self.cfg
        N, H, hd = q.shape
        Kv = k.shape[1]
        gqa = H // Kv
        Np = _pad32(N)

        with self.ctx.tim.t("host:attn_pack"):
            qp = np.ascontiguousarray(q.transpose(1, 0, 2)).astype(np.float16)   # [H,N,hd]
            kp = np.zeros((Kv, Np, hd), np.float16)
            kp[:, :N] = k.transpose(1, 0, 2)
            vtp = np.zeros((Kv, hd, Np), np.float16)
            vtp[:, :, :N] = v.transpose(1, 2, 0)

        q_buf = self.ctx.scratch_fill("attn_q", qp)
        k_buf = self.ctx.scratch_fill("attn_k", kp)
        vt_buf = self.ctx.scratch_fill("attn_vt", vtp)
        m_buf = self.ctx.scratch_fill("attn_mask", np.ascontiguousarray(mask[:, :Np]))
        s_buf = self.ctx.scratch("attn_s", H * N * Np * 4)   # f32 scores (kq needs range)
        p_buf = self.ctx.scratch("attn_p", H * N * Np * 2)   # f16 probs
        o_buf = self.ctx.scratch("attn_o", H * N * hd * 2)

        b = GemmBatch(self.ctx, "attn")
        for h in range(H):
            g = h // gqa
            b.gemm("dg_gemm_qkt_f32", q_buf, h * N * hd * 2, k_buf, g * Np * hd * 2,
                   s_buf, h * N * Np * 4, M=N, N=Np, K=hd, ldc=Np)
        b.barrier()
        b.softmax_mask(s_buf, p_buf, m_buf, rows=H * N, ncols=Np, ntok=N,
                       scale=cfg.attn_scale)
        b.barrier()
        for h in range(H):
            g = h // gqa
            b.gemm("gemm_mma_f16", p_buf, h * N * Np * 2, vt_buf, g * hd * Np * 2,
                   o_buf, h * N * hd * 2, M=N, N=hd, K=Np, ldc=hd)
        b.run()

        o = self.ctx.read(o_buf, np.float16, (H, N, hd)).astype(F32)
        return np.ascontiguousarray(o.transpose(1, 0, 2)).reshape(N, H * hd)

    # -- MoE -------------------------------------------------------------------

    def _moe(self, il: int, attn_out: np.ndarray, e_in: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        N = attn_out.shape[0]
        with self.ctx.tim.t("host:moe_route"):
            sel, wts = moe_route(self.w, cfg, attn_out, il)
        if self.dump:
            self.dump("moe_sel", il, sel.astype(np.int32))
            self.dump("moe_wts", il, wts)
        down_s = self.w.f32(f"blk.{il}.ffn_down_exps.scale")
        gu_name = f"blk.{il}.ffn_gate_up_exps.weight"
        dn_name = f"blk.{il}.ffn_down_exps.weight"

        d, nff = cfg.d_model, cfg.n_ff_exp
        experts = []
        r0 = 0
        for e in np.unique(sel):
            tok, slot = np.nonzero(sel == e)
            experts.append((int(e), tok, slot, r0))
            r0 += len(tok)
        m_total = r0

        # All hit experts share three persistent row-packed scratch buffers
        # (per-expert alloc/free churned ~GBs of Metal allocations per
        # forward; fine for one Stage-1 forward, deadly across a generation)
        cap_rows = N * cfg.n_expert_used
        a_buf = self.ctx.scratch("moe_a", cap_rows * d * 2)
        c_buf = self.ctx.scratch("moe_c", cap_rows * 2 * nff * 2)
        d_buf = self.ctx.scratch("moe_d", cap_rows * d * 2)
        with self.ctx.tim.t("host:moe_pack"):
            packed = np.concatenate([e_in[tok] for _, tok, _, _ in experts]).astype(np.float16)
            a_buf.contents().as_buffer(packed.nbytes)[:] = packed.tobytes()

        # stage A: gate_up for every hit expert in one command buffer
        b = GemmBatch(self.ctx, "moe_gu")
        for e, tok, _, r0 in experts:
            self._wgemm(b, gu_name, a_buf, r0 * d * 2, c_buf, r0 * 2 * nff * 2,
                        M=len(tok), N=2 * nff, K=d, row0=e * 2 * nff)
        b.run()

        # host geglu (whole packed block), then stage B: down per expert
        gu = self.ctx.read(c_buf, np.float16, (m_total, 2 * nff)).astype(F32)
        with self.ctx.tim.t("host:moe_geglu"):
            act = (gelu_tanh(gu[:, :nff]) * gu[:, nff:]).astype(np.float16)
            a_buf.contents().as_buffer(act.nbytes)[:] = act.tobytes()
        b = GemmBatch(self.ctx, "moe_down")
        for e, tok, _, r0 in experts:
            self._wgemm(b, dn_name, a_buf, r0 * nff * 2, d_buf, r0 * d * 2,
                        M=len(tok), N=d, K=nff, row0=e * d)
        b.run()

        dn = self.ctx.read(d_buf, np.float16, (m_total, d)).astype(F32)
        with self.ctx.tim.t("host:moe_scatter"):
            moe = np.zeros((N, d), dtype=F32)
            for e, tok, slot, r0 in experts:
                moe[tok] += dn[r0:r0 + len(tok)] * down_s[e] * wts[tok, slot][:, None]
        return moe

    # -- forward ---------------------------------------------------------------

    def forward(self, ids: np.ndarray, P: int, sc_logits: np.ndarray | None = None,
                sc_temp_inv: float = 1.0, sc_use: float = 1.0,
                sc_probs16: np.ndarray | None = None) -> np.ndarray:
        """sc_logits=None and sc_probs16=None -> the Stage-1-validated zero-SC
        unified forward."""
        cfg, w = self.cfg, self.w
        ids = np.asarray(ids)
        N = len(ids)
        C = N - P
        pos = np.arange(N, dtype=np.int64)
        dmp = self.dump or (lambda name, il, arr: None)
        tim = self.ctx.tim
        tim.reset()
        t_fw0 = time.perf_counter()

        sc_sig = None
        if sc_logits is not None or sc_probs16 is not None:
            with tim.t("sc(nested)"), objc.autorelease_pool():
                sc_sig = self._sc_signal(sc_logits, sc_temp_inv, sc_use,
                                         probs16=sc_probs16)
            dmp("sc_sig", -1, sc_sig)
        with tim.t("host:embed"):
            x = embed_tokens(w, cfg, ids, P, sc_sig=sc_sig)
        dmp("inp_region", -1, x)
        rope_ff = w.f32("rope_freqs.weight")

        for il in range(cfg.n_layers):
            # drain ObjC autoreleases per layer: without a pool the bridge
            # pins every MTLBuffer proxy until process exit (~0.85 GB/layer
            # leak that out-grew the box even with cache eviction)
            with objc.autorelease_pool():
                swa = cfg.is_swa[il]
                hd = cfg.head_dim(il)
                n_kv = cfg.n_kv_heads[il]
                base, use_ff = cfg.rope_params(il)
                ff = rope_ff if use_ff else None

                with tim.t("host:norms"):
                    h = rms_norm(x, cfg.eps, w.f32(f"blk.{il}.attn_norm.weight"))

                q = self._gemm_f32(f"blk.{il}.attn_q.weight", h, cfg.n_heads * hd)
                k_raw = self._gemm_f32(f"blk.{il}.attn_k.weight", h, n_kv * hd)
                vname = f"blk.{il}.attn_v.weight"
                v_raw = self._gemm_f32(vname, h, n_kv * hd) if vname in self.gg.tensors else k_raw

                with tim.t("host:qkv_norm_rope"):
                    q = rms_norm(q.reshape(N, cfg.n_heads, hd), cfg.eps,
                                 w.f32(f"blk.{il}.attn_q_norm.weight"))
                    k = rms_norm(k_raw.reshape(N, n_kv, hd), cfg.eps,
                                 w.f32(f"blk.{il}.attn_k_norm.weight"))
                    v = rms_norm(v_raw.reshape(N, n_kv, hd), cfg.eps)
                    q = rope_neox(q, pos, base, ff)
                    k = rope_neox(k, pos, base, ff)
                dmp("q_pos", il, q); dmp("k_pos", il, k); dmp("v_normed", il, v)

                with tim.t("host:mask"):
                    mask = build_mask(P, C, swa, cfg.window, n_cols=_pad32(N))
                o = self._attention(il, q, k, v, mask)
                attn = self._gemm_f32(f"blk.{il}.attn_output.weight", o, cfg.d_model)
                with tim.t("host:norms"):
                    attn = rms_norm(attn, cfg.eps, w.f32(f"blk.{il}.post_attention_norm.weight"))
                    attn_out = (attn + x).astype(F32)
                dmp("attn_out", il, attn_out)

                with tim.t("host:norms"):
                    m = rms_norm(attn_out, cfg.eps, w.f32(f"blk.{il}.ffn_norm.weight"))
                g_ = gelu_tanh(self._gemm_f32(f"blk.{il}.ffn_gate.weight", m, cfg.n_ff))
                u_ = self._gemm_f32(f"blk.{il}.ffn_up.weight", m, cfg.n_ff)
                mlp = self._gemm_f32(f"blk.{il}.ffn_down.weight", g_ * u_, cfg.d_model)
                with tim.t("host:norms"):
                    mlp = rms_norm(mlp, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm_1.weight"))
                dmp("ffn_mlp", il, mlp)

                with tim.t("host:norms"):
                    e_in = rms_norm(attn_out, cfg.eps, w.f32(f"blk.{il}.pre_ffw_norm_2.weight"))
                moe = self._moe(il, attn_out, e_in)
                with tim.t("host:norms"):
                    moe = rms_norm(moe, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm_2.weight"))
                dmp("ffn_moe", il, moe)

                with tim.t("host:norms"):
                    f = rms_norm(mlp + moe, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm.weight"))
                    cur = (f + attn_out).astype(F32)
                    cur[:P] *= w.f32(f"blk.{il}.enc_layer_output_scale.weight")[0]
                    cur[P:] *= w.f32(f"blk.{il}.layer_output_scale.weight")[0]
                dmp("l_out", il, cur)
                x = cur
                self.wb.evict_prefix(f"blk.{il}.")
            with tim.t("host:gc"):
                gc.collect()

        x = rms_norm(x, cfg.eps, w.f32("output_norm.weight"))
        dmp("result_norm", -1, x)

        with objc.autorelease_pool():
            logits = self._gemm_f32("token_embd.weight", x[P:], cfg.vocab_size)
        with tim.t("host:softcap"):
            cap = F32(cfg.final_logit_softcap)
            # in-place softcap: the expression form held ~3 extra 268 MB transients
            np.divide(logits, cap, out=logits)
            np.tanh(logits, out=logits)
            np.multiply(logits, cap, out=logits)
        dmp("result_output", -1, logits)
        if os.environ.get("DG_TIMING"):
            print(tim.report(time.perf_counter() - t_fw0), flush=True)
        return logits
