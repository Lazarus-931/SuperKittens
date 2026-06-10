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

import gc
import mmap
import os
from pathlib import Path

import numpy as np
import objc  # noqa: F401
import Metal  # type: ignore[import-not-found]

from .config import DiffusionGemmaConfig
from .gguf_io import GGUFFile, TensorInfo
from .graph_ref import (F32, Weights, build_mask, embed_tokens, gelu_tanh,
                        moe_route, rms_norm, rope_neox)

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
        mv = buf.contents().as_buffer(offset + n)[offset:offset + n]
        return np.frombuffer(mv, dtype=dtype).reshape(shape).copy()


class WeightBufs:
    """Per-tensor MTLBuffers over the GGUF. No-copy mmap windows when the
    bridge cooperates (verified per process at init), else lazy copies."""

    def __init__(self, ctx: MetalCtx, gg: GGUFFile):
        self.ctx = ctx
        self.gg = gg
        self._cache: dict[str, tuple] = {}
        self._f = open(gg.path, "rb")
        # No-copy is opt-in: the bridge only takes WRITABLE buffers, and GPU
        # access to MAP_PRIVATE+PROT_WRITE windows turned every touched weight
        # byte into anonymous memory (system swap storm took the host down
        # twice; process rss stayed flat at ~0.4 GB). The copy path + per-layer
        # eviction bounds anonymous memory at ~1 layer of weights.
        self.nocopy_ok = os.environ.get("SK_DG_NOCOPY") == "1" and self._probe_nocopy()

    def _map(self, ti: TensorInfo):
        page = mmap.ALLOCATIONGRANULARITY
        base = ti.offset // page * page
        delta = ti.offset - base
        length = (delta + ti.nbytes + page - 1) // page * page
        if base + length > os.path.getsize(self.gg.path):
            raise RuntimeError("page-rounded window past EOF (last tensor)")
        # MAP_PRIVATE + writable prot: the bridge requires a writable buffer
        # object for the void* arg; pages stay clean (we never write).
        mm = mmap.mmap(self._f.fileno(), length, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        buf = self.ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            mm, length, Metal.MTLResourceStorageModeShared, None)
        if buf is None:
            raise RuntimeError("newBufferWithBytesNoCopy returned None")
        return buf, delta, mm

    def _probe_nocopy(self) -> bool:
        try:
            ti = min(self.gg.tensors.values(), key=lambda t: t.nbytes)
            buf, delta, mm = self._map(ti)
            got = bytes(buf.contents().as_buffer(delta + 16)[delta:delta + 16])
            want = bytes(self.gg.mm[ti.offset:ti.offset + 16])
            return got == want
        except Exception as e:  # noqa: BLE001
            print(f"[weights] no-copy probe failed ({e}); falling back to copies")
            return False

    def get(self, name: str) -> tuple:
        """-> (MTLBuffer, byte_offset, TensorInfo)"""
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        ti = self.gg.tensors[name]
        ent = None
        if self.nocopy_ok:
            try:
                buf, delta, mm = self._map(ti)
                ent = (buf, delta, ti, mm)
            except RuntimeError:
                pass  # e.g. page-rounded window past EOF: copy just this one
        if ent is None:
            arr = np.frombuffer(self.gg.mm, dtype=np.uint8, count=ti.nbytes,
                                offset=ti.offset)
            ent = (self.ctx.buf_from(arr), 0, ti, None)
        self._cache[name] = ent
        return ent

    def evict_prefix(self, prefix: str):
        """Drop cached buffers/mmaps for one layer once it has run: a 16 GB
        model's windows can't all stay resident on a 16 GB box (the late-layer
        kill was the process croaking under memory pressure, not a kernel)."""
        for k in [k for k in self._cache if k.startswith(prefix)]:
            del self._cache[k]


class GemmBatch:
    """Record gemm_mma / dg_softmax_mask dispatches into one command buffer;
    memory barriers split dependent stages. run() commits + waits."""

    def __init__(self, ctx: MetalCtx):
        self.ctx = ctx
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
        self.cmd.commit()
        self.cmd.waitUntilCompleted()
        if self.cmd.error() is not None:
            raise RuntimeError(f"command buffer failed: {self.cmd.error()}")


class DiffusionGemmaMetal:
    """Stage-1 unified zero-SC forward. forward(ids, P) -> canvas logits f32."""

    def __init__(self, gguf_path: str, cfg: DiffusionGemmaConfig):
        self.gg = GGUFFile(gguf_path)
        self.cfg = cfg
        self.ctx = MetalCtx()
        self.wb = WeightBufs(self.ctx, self.gg)
        self.w = Weights(self.gg)   # F32 sidecars (norms, scales, router)
        self.dump = None            # optional (name, il, arr) tap

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
        a_buf = self.ctx.buf_from(a.astype(np.float16))
        c_buf = self.ctx.buf_empty(M * N * 2)
        b = GemmBatch(self.ctx)
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

        qp = np.ascontiguousarray(q.transpose(1, 0, 2)).astype(np.float16)   # [H,N,hd]
        kp = np.zeros((Kv, Np, hd), np.float16)
        kp[:, :N] = k.transpose(1, 0, 2)
        vtp = np.zeros((Kv, hd, Np), np.float16)
        vtp[:, :, :N] = v.transpose(1, 2, 0)

        q_buf = self.ctx.buf_from(qp)
        k_buf = self.ctx.buf_from(kp)
        vt_buf = self.ctx.buf_from(vtp)
        m_buf = self.ctx.buf_from(np.ascontiguousarray(mask[:, :Np]))
        s_buf = self.ctx.buf_empty(H * N * Np * 4)   # f32 scores (kq needs range)
        p_buf = self.ctx.buf_empty(H * N * Np * 2)   # f16 probs
        o_buf = self.ctx.buf_empty(H * N * hd * 2)

        b = GemmBatch(self.ctx)
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
        sel, wts = moe_route(self.w, cfg, attn_out, il)
        if self.dump:
            self.dump("moe_sel", il, sel.astype(np.int32))
            self.dump("moe_wts", il, wts)
        down_s = self.w.f32(f"blk.{il}.ffn_down_exps.scale")
        gu_name = f"blk.{il}.ffn_gate_up_exps.weight"
        dn_name = f"blk.{il}.ffn_down_exps.weight"

        experts = []
        for e in np.unique(sel):
            tok, slot = np.nonzero(sel == e)
            experts.append((int(e), tok, slot))

        # stage A: gate_up for every hit expert in one command buffer
        b = GemmBatch(self.ctx)
        stage = []
        for e, tok, slot in experts:
            m = len(tok)
            a_buf = self.ctx.buf_from(e_in[tok].astype(np.float16))
            c_buf = self.ctx.buf_empty(m * 2 * cfg.n_ff_exp * 2)
            self._wgemm(b, gu_name, a_buf, 0, c_buf, 0, M=m, N=2 * cfg.n_ff_exp,
                        K=cfg.d_model, row0=e * 2 * cfg.n_ff_exp)
            stage.append((e, tok, slot, c_buf, m))
        b.run()

        # host geglu, then stage B: down for every hit expert
        b = GemmBatch(self.ctx)
        stage2 = []
        for e, tok, slot, c_buf, m in stage:
            gu = self.ctx.read(c_buf, np.float16, (m, 2 * cfg.n_ff_exp)).astype(F32)
            act = gelu_tanh(gu[:, :cfg.n_ff_exp]) * gu[:, cfg.n_ff_exp:]
            a_buf = self.ctx.buf_from(act.astype(np.float16))
            d_buf = self.ctx.buf_empty(m * cfg.d_model * 2)
            self._wgemm(b, dn_name, a_buf, 0, d_buf, 0, M=m, N=cfg.d_model,
                        K=cfg.n_ff_exp, row0=e * cfg.d_model)
            stage2.append((e, tok, slot, d_buf, m))
        b.run()

        moe = np.zeros((N, cfg.d_model), dtype=F32)
        for e, tok, slot, d_buf, m in stage2:
            d_ = self.ctx.read(d_buf, np.float16, (m, cfg.d_model)).astype(F32)
            moe[tok] += d_ * down_s[e] * wts[tok, slot][:, None]
        return moe

    # -- forward ---------------------------------------------------------------

    def forward(self, ids: np.ndarray, P: int) -> np.ndarray:
        cfg, w = self.cfg, self.w
        ids = np.asarray(ids)
        N = len(ids)
        C = N - P
        pos = np.arange(N, dtype=np.int64)
        dmp = self.dump or (lambda name, il, arr: None)

        x = embed_tokens(w, cfg, ids, P)
        dmp("inp_region", -1, x)
        rope_ff = w.f32("rope_freqs.weight")

        for il in range(cfg.n_layers):
            swa = cfg.is_swa[il]
            hd = cfg.head_dim(il)
            n_kv = cfg.n_kv_heads[il]
            base, use_ff = cfg.rope_params(il)
            ff = rope_ff if use_ff else None

            h = rms_norm(x, cfg.eps, w.f32(f"blk.{il}.attn_norm.weight"))

            q = self._gemm_f32(f"blk.{il}.attn_q.weight", h, cfg.n_heads * hd)
            k_raw = self._gemm_f32(f"blk.{il}.attn_k.weight", h, n_kv * hd)
            vname = f"blk.{il}.attn_v.weight"
            v_raw = self._gemm_f32(vname, h, n_kv * hd) if vname in self.gg.tensors else k_raw

            q = rms_norm(q.reshape(N, cfg.n_heads, hd), cfg.eps,
                         w.f32(f"blk.{il}.attn_q_norm.weight"))
            k = rms_norm(k_raw.reshape(N, n_kv, hd), cfg.eps,
                         w.f32(f"blk.{il}.attn_k_norm.weight"))
            v = rms_norm(v_raw.reshape(N, n_kv, hd), cfg.eps)
            q = rope_neox(q, pos, base, ff)
            k = rope_neox(k, pos, base, ff)
            dmp("q_pos", il, q); dmp("k_pos", il, k); dmp("v_normed", il, v)

            mask = build_mask(P, C, swa, cfg.window, n_cols=_pad32(N))
            o = self._attention(il, q, k, v, mask)
            attn = self._gemm_f32(f"blk.{il}.attn_output.weight", o, cfg.d_model)
            attn = rms_norm(attn, cfg.eps, w.f32(f"blk.{il}.post_attention_norm.weight"))
            attn_out = (attn + x).astype(F32)
            dmp("attn_out", il, attn_out)

            m = rms_norm(attn_out, cfg.eps, w.f32(f"blk.{il}.ffn_norm.weight"))
            g_ = gelu_tanh(self._gemm_f32(f"blk.{il}.ffn_gate.weight", m, cfg.n_ff))
            u_ = self._gemm_f32(f"blk.{il}.ffn_up.weight", m, cfg.n_ff)
            mlp = self._gemm_f32(f"blk.{il}.ffn_down.weight", g_ * u_, cfg.d_model)
            mlp = rms_norm(mlp, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm_1.weight"))
            dmp("ffn_mlp", il, mlp)

            e_in = rms_norm(attn_out, cfg.eps, w.f32(f"blk.{il}.pre_ffw_norm_2.weight"))
            moe = self._moe(il, attn_out, e_in)
            moe = rms_norm(moe, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm_2.weight"))
            dmp("ffn_moe", il, moe)

            f = rms_norm(mlp + moe, cfg.eps, w.f32(f"blk.{il}.post_ffw_norm.weight"))
            cur = (f + attn_out).astype(F32)
            cur[:P] *= w.f32(f"blk.{il}.enc_layer_output_scale.weight")[0]
            cur[P:] *= w.f32(f"blk.{il}.layer_output_scale.weight")[0]
            dmp("l_out", il, cur)
            x = cur
            self.wb.evict_prefix(f"blk.{il}.")
            gc.collect()  # drop Metal buffers + mmap windows deterministically

        x = rms_norm(x, cfg.eps, w.f32("output_norm.weight"))
        dmp("result_norm", -1, x)

        logits = self._gemm_f32("token_embd.weight", x[P:], cfg.vocab_size)
        cap = F32(cfg.final_logit_softcap)
        logits = (np.tanh(logits / cap) * cap).astype(F32)
        dmp("result_output", -1, logits)
        return logits
