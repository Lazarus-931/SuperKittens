"""sampler.py — EntropyBound denoiser for DiffusionGemma, mirroring llama.cpp
PR #24423 `diffusion_generate_entropy_bound` (examples/diffusion/diffusion.cpp
@ c84e85af) decision-for-decision.

RNG is bit-exact with the reference on macOS: std::mt19937(seed)'s raw 32-bit
stream equals numpy RandomState(seed)'s full-range draws (same init_genrand
seeding + genrand_int32), and libc++'s distributions reduce to
  uniform_int_distribution(0, 2^w ranges)  -> low-w-bit mask (+ rejection when
                                              the range isn't a power of two)
  uniform_real_distribution<float>(0,1)    -> float32(raw) / float32(2^32)
verified bit-for-bit against a compiled libc++ dump (temp/diffgemma_s2).
Draw ORDER matters and is mirrored: C canvas-init draws at construction, then
per step C interleaved (u, renoise) pairs — drawn after the forward, exactly
where the reference pre-draws "single-threaded for seed-reproducibility".

Float mirroring vs the reference (clang -O2/-O3, default -ffp-contract=on):
  exp arg     : expf(row[v]*temp_inv - m) is CONTRACTED to expf(fmaf(...)) —
                emulated here as f32(f64(row)*f64(ti) - f64(m)) (the f64
                product is exact for f32 inputs; double-rounding mismatch
                odds ~2^-28/element)
  Z / cum     : sequential f32 adds == np.cumsum(f32); the multinomial pick
                uses the same scan values bit-for-bit
  exp / log   : numpy's f32 routines bit-match Apple libm expf/logf (verified)
  entropy H   : the C++ `H -= p*logf(p)` chain may itself contract into fused
                accumulates — not replicable vectorized; numpy uses a
                sequential-equivalent cumsum. Residual |dH| ~1e-4..1e-3 can
                flip an accept-set boundary only on near-tied entropies.
  sort        : std::sort is unstable; np.argsort(stable) — differs only on
                exact f32 entropy ties (identical logit rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32


class LibcxxMT19937:
    """std::mt19937 + libc++ uniform_int/uniform_real, draw-compatible."""

    def __init__(self, seed: int):
        self._rs = np.random.RandomState(seed & 0xFFFFFFFF)

    def raw(self, n: int) -> np.ndarray:
        return self._rs.randint(0, 2 ** 32, size=n, dtype=np.uint64).astype(np.uint32)

    @staticmethod
    def _width(rp: int) -> int:
        w = rp.bit_length() - 1
        if rp & ((1 << w) - 1):
            w += 1
        return w

    def uniform_int_vec(self, n: int, lo: int, hi: int) -> np.ndarray:
        """n draws of uniform_int_distribution(lo, hi); fixed one-raw-per-draw
        only when the range is a power of two (DiffusionGemma vocab is 2^18)."""
        rp = hi - lo + 1
        w = self._width(rp)
        mask = np.uint32((1 << w) - 1) if w < 32 else np.uint32(0xFFFFFFFF)
        if rp & (rp - 1) == 0:
            return (self.raw(n) & mask).astype(np.int64) + lo
        out = np.empty(n, np.int64)
        for i in range(n):
            while True:
                u = int(self.raw(1)[0]) & int(mask)
                if u < rp:
                    out[i] = u + lo
                    break
        return out

    def step_draws(self, C: int, n_vocab: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-step (u[C], renoise[C]) with the reference's per-position
        interleave: u[pos] = uni01(rng); renoise[pos] = vocab_dist(rng)."""
        if n_vocab & (n_vocab - 1) == 0:
            raw = self.raw(2 * C).reshape(C, 2)
            u = raw[:, 0].astype(F32) / F32(2 ** 32)
            renoise = (raw[:, 1] & np.uint32(n_vocab - 1)).astype(np.int64)
            return u, renoise
        u = np.empty(C, F32)
        renoise = np.empty(C, np.int64)
        for pos in range(C):
            u[pos] = self.raw(1)[0].astype(F32) / F32(2 ** 32)
            renoise[pos] = self.uniform_int_vec(1, 0, n_vocab - 1)[0]
        return u, renoise


@dataclass
class EBParams:
    """diffusion_eb_params reference defaults (diffusion.h); the GGUF carries
    no diffusion.eb_* overrides."""
    max_steps: int = 48
    t_min: float = 0.4
    t_max: float = 0.8
    entropy_bound: float = 0.1
    stability_threshold: int = 1
    confidence_threshold: float = 0.005
    seed: int = 0
    # NOT in the reference EB sampler (only the masked-diffusion path
    # suppresses the mask token); off by default to stay decision-identical.
    suppress_mask_token: bool = False
    mask_token_id: int = 4


@dataclass
class StepResult:
    step_idx: int
    cur_step: int
    t: float
    entropy: np.ndarray         # [C] f32
    argmax: np.ndarray          # [C] i32 — the output canvas this step
    sampled: np.ndarray         # [C] i32 — per-position multinomial draw
    accepted: np.ndarray        # [C] bool
    canvas_next: np.ndarray     # [C] i32 — renoised working canvas (next input)
    u: np.ndarray               # [C] f32 pre-drawn multinomial uniforms
    renoise: np.ndarray         # [C] i32 pre-drawn renoise tokens
    held: int
    finished: bool
    entropy_mean: float


class EntropyBoundSampler:
    """One denoising block. Construction random-inits the working canvas
    (consuming the reference's C init draws); step(logits) consumes one
    forward's canvas logits and returns every decision the reference makes.

    Self-conditioning contract for the NEXT forward (caller's job):
      sc_logits   = this step's raw logits (keep the array passed to step)
      sc_temp_inv = self.prev_temp_inv
      sc_use      = 0.0 before the first step, else 1.0
    """

    def __init__(self, params: EBParams, n_vocab: int, C: int):
        self.p = params
        self.n_vocab = n_vocab
        self.C = C
        self.S = max(1, params.max_steps)
        self.rng = LibcxxMT19937(params.seed)
        self.canvas = self.rng.uniform_int_vec(C, 0, n_vocab - 1).astype(np.int32)
        self.argmax_canvas = np.zeros(C, np.int32)
        self._prev_argmax = np.full(C, -1, np.int32)
        self.prev_temp_inv = F32(1.0)
        self.held = 0
        self.finished = False
        self.step_idx = 0          # 0-based; cur_step = S - step_idx

    def temperature(self, step_idx: int) -> F32:
        cur_step = self.S - step_idx
        return F32(self.p.t_min) + (F32(self.p.t_max) - F32(self.p.t_min)) * (
            F32(cur_step) / F32(self.S))

    def step(self, logits: np.ndarray, _chunk: int = 32) -> StepResult:
        """logits: f32 [C, n_vocab] canvas rows for the CURRENT working canvas."""
        assert not self.finished and self.step_idx < self.S
        assert logits.shape == (self.C, self.n_vocab) and logits.dtype == F32
        p, C, V = self.p, self.C, self.n_vocab
        step_idx = self.step_idx
        cur_step = self.S - step_idx
        t = self.temperature(step_idx)
        temp_inv = F32(1.0) / t

        u, renoise = self.rng.step_draws(C, V)

        if p.suppress_mask_token:
            logits = logits.copy()
            logits[:, p.mask_token_id] = -np.inf

        entropy = np.empty(C, F32)
        amax = np.empty(C, np.int64)
        sampled = np.empty(C, np.int64)
        # position-chunked: z/e/cum at full [C, V] f32 would be 3 x 268 MB
        for c0 in range(0, C, _chunk):
            c1 = min(c0 + _chunk, C)
            rows = logits[c0:c1]
            z = rows * temp_inv                               # plain f32 mult (C++ loop 1)
            m = z.max(axis=1)
            amax[c0:c1] = z.argmax(axis=1)
            # expf(fmaf(row, temp_inv, -m)) per the contracted C++ loops 2/3
            x = (rows.astype(np.float64) * np.float64(temp_inv)
                 - m.astype(np.float64)[:, None]).astype(F32)
            e = np.exp(x)
            cum = np.cumsum(e, axis=1, dtype=F32)             # sequential, like the C++ scan
            Z = cum[:, -1]
            target = (u[c0:c1] * Z).astype(F32)
            hit = cum >= target[:, None]
            idx = np.argmax(hit, axis=1)
            idx[~hit.any(axis=1)] = V - 1                     # reference fallback
            sampled[c0:c1] = idx
            prob = e / Z[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                h = np.where(prob > 0, prob * np.log(prob), F32(0))
            entropy[c0:c1] = -np.cumsum(h, axis=1, dtype=F32)[:, -1]

        order = np.argsort(entropy, kind="stable")            # std::sort: ties unspecified
        cumE = np.cumsum(entropy[order].astype(np.float64))
        ok = (cumE - entropy[order]) <= np.float64(F32(p.entropy_bound))
        accepted = np.zeros(C, bool)
        accepted[order[ok]] = True

        canvas_next = np.where(accepted, sampled, renoise).astype(np.int32)
        argmax_i32 = amax.astype(np.int32)
        entropy_sum = np.cumsum(entropy, dtype=F32)[-1]       # sequential f32 like the C++

        self.held = self.held + 1 if np.array_equal(self._prev_argmax, argmax_i32) else 0
        confident = (entropy_sum / F32(C)) < F32(p.confidence_threshold)
        if self.held >= p.stability_threshold and confident:
            self.finished = True
        self._prev_argmax = argmax_i32
        self.prev_temp_inv = temp_inv
        self.argmax_canvas = argmax_i32
        self.canvas = canvas_next
        self.step_idx += 1
        if self.step_idx >= self.S:
            self.finished = True

        return StepResult(step_idx, cur_step, float(t), entropy, argmax_i32,
                          sampled.astype(np.int32), accepted, canvas_next,
                          u, renoise.astype(np.int32), self.held, self.finished,
                          float(entropy_sum / F32(C)))
