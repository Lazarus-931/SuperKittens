from __future__ import annotations

import argparse
import time
from typing import Tuple

import mlx.core as mx
import numpy as np


PI = 3.14159265358979323846


def rotate_qk(q: mx.array, k: mx.array, theta: mx.array) -> Tuple[mx.array, mx.array]:
    half = q.shape[-1] // 2
    q0, q1 = q[..., :half], q[..., half:]
    k0, k1 = k[..., :half], k[..., half:]
    c = mx.cos(theta)
    s = mx.sin(theta)
    q_rot = mx.concatenate([q0 * c - q1 * s, q0 * s + q1 * c], axis=-1)
    k_rot = mx.concatenate([k0 * c - k1 * s, k0 * s + k1 * c], axis=-1)
    return q_rot, k_rot


def mamba3_siso_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    angle: mx.array,
    chunk_size: int = 32,
) -> mx.array:
    batch, heads, length, dq = q.shape
    dv = v.shape[-1]
    n_chunks = (length + chunk_size - 1) // chunk_size
    outs = []

    for batch_idx in range(batch):
        per_head = []
        for head_idx in range(heads):
            state = mx.zeros((dq, dv), dtype=mx.float32)
            head_chunks = []
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = min(length, start + chunk_size)
                q_chunk = q[batch_idx, head_idx, start:end].astype(mx.float32)
                k_chunk = k[batch_idx, head_idx, start:end].astype(mx.float32)
                v_chunk = v[batch_idx, head_idx, start:end].astype(mx.float32)
                a_chunk = a[batch_idx, head_idx, start:end].astype(mx.float32)
                b_chunk = b[batch_idx, head_idx, start:end].astype(mx.float32)
                ang_chunk = angle[batch_idx, head_idx, start:end].astype(mx.float32)

                a_cs = mx.cumsum(a_chunk, axis=0)
                b_scale = 1.0 + b_chunk * mx.exp(-a_cs)
                theta = a_cs[:, None] * ang_chunk * PI
                q_rot, k_rot = rotate_qk(q_chunk, k_chunk, theta)

                state = state * (mx.exp(a_cs[-1]) * b_scale[-1]) + mx.matmul(k_rot.T, v_chunk)

                scores = mx.matmul(q_rot, k_rot.T)
                decay = mx.exp(a_cs[:, None] - a_cs[None, :])
                causal = mx.tril(mx.ones((end - start, end - start), dtype=mx.float32))
                intra = mx.matmul(scores * decay * causal, v_chunk)
                q_decay = (mx.exp(a_cs) * b_scale)[:, None]
                inter = q_decay * mx.matmul(q_rot, state)
                head_chunks.append(intra + inter)
            per_head.append(mx.concatenate(head_chunks, axis=0))
        outs.append(mx.stack(per_head, axis=0))
    return mx.stack(outs, axis=0)


def rotate_forward_np(q0: float, q1: float, k0: float, k1: float, theta: float):
    c = np.cos(theta)
    s = np.sin(theta)
    return q0 * c - q1 * s, q0 * s + q1 * c, k0 * c - k1 * s, k0 * s + k1 * c


def rotate_backward_np(q0, q1, k0, k1, theta, dq0r, dq1r, dk0r, dk1r):
    c = np.cos(theta)
    s = np.sin(theta)
    dq0 = dq0r * c + dq1r * s
    dq1 = -dq0r * s + dq1r * c
    dk0 = dk0r * c + dk1r * s
    dk1 = -dk0r * s + dk1r * c
    dtheta = (
        dq0r * (-q0 * s - q1 * c)
        + dq1r * (q0 * c - q1 * s)
        + dk0r * (-k0 * s - k1 * c)
        + dk1r * (k0 * c - k1 * s)
    )
    return dq0, dq1, dk0, dk1, dtheta


def cpu_reference(q, k, v, a, b, angle, do, chunk):
    B, H, L, DQ = q.shape
    DV = v.shape[-1]
    half = DQ // 2
    n_chunks = (L + chunk - 1) // chunk
    out = np.zeros((B, H, L, DV), dtype=np.float32)
    states = np.zeros((B, H, n_chunks, DQ, DV), dtype=np.float32)
    dQ = np.zeros_like(q, dtype=np.float32)
    dK = np.zeros_like(k, dtype=np.float32)
    dV = np.zeros_like(v, dtype=np.float32)
    dA = np.zeros_like(a, dtype=np.float32)
    dB = np.zeros_like(b, dtype=np.float32)
    dAngle = np.zeros_like(angle, dtype=np.float32)

    for bi in range(B):
        for hi in range(H):
            state = np.zeros((DQ, DV), dtype=np.float32)
            q_rot_chunks = []
            k_rot_chunks = []
            a_cs_chunks = []
            b_scale_chunks = []
            chunk_lens = []

            for ci in range(n_chunks):
                start = ci * chunk
                end = min(L, start + chunk)
                chunk_lens.append(end - start)
                a_cs = np.cumsum(a[bi, hi, start:end], axis=0)
                b_scale = 1.0 + b[bi, hi, start:end] * np.exp(-a_cs)
                q_rot = np.zeros((end - start, DQ), dtype=np.float32)
                k_rot = np.zeros((end - start, DQ), dtype=np.float32)
                for t in range(end - start):
                    seq = start + t
                    for i in range(half):
                        theta = a_cs[t] * angle[bi, hi, seq, i] * PI
                        q0r, q1r, k0r, k1r = rotate_forward_np(
                            q[bi, hi, seq, i], q[bi, hi, seq, i + half],
                            k[bi, hi, seq, i], k[bi, hi, seq, i + half], theta)
                        q_rot[t, i] = q0r
                        q_rot[t, i + half] = q1r
                        k_rot[t, i] = k0r
                        k_rot[t, i + half] = k1r
                state = state * (np.exp(a_cs[-1]) * b_scale[-1]) + k_rot.T @ v[bi, hi, start:end]
                states[bi, hi, ci] = state
                q_rot_chunks.append(q_rot)
                k_rot_chunks.append(k_rot)
                a_cs_chunks.append(a_cs)
                b_scale_chunks.append(b_scale)

                scores = q_rot @ k_rot.T
                decay = np.exp(a_cs[:, None] - a_cs[None, :])
                causal = np.tril(np.ones((end - start, end - start), dtype=np.float32))
                intra = (scores * decay * causal) @ v[bi, hi, start:end]
                inter = (np.exp(a_cs) * b_scale)[:, None] * (q_rot @ state)
                out[bi, hi, start:end] = intra + inter

            state_grad = np.zeros((DQ, DV), dtype=np.float32)
            for ci in range(n_chunks - 1, -1, -1):
                start = ci * chunk
                clen = chunk_lens[ci]
                a_cs = a_cs_chunks[ci]
                b_scale = b_scale_chunks[ci]
                q_rot = q_rot_chunks[ci]
                k_rot = k_rot_chunks[ci]
                s_cur = states[bi, hi, ci]
                s_prev = states[bi, hi, ci - 1] if ci > 0 else np.zeros_like(s_cur)

                dq_rot = np.zeros((clen, DQ), dtype=np.float32)
                dk_rot = np.zeros((clen, DQ), dtype=np.float32)
                dv_acc = np.zeros((clen, DV), dtype=np.float32)
                d_a_cs = np.zeros(clen, dtype=np.float32)
                d_b_scale = np.zeros(clen, dtype=np.float32)
                d_q_decay = np.zeros(clen, dtype=np.float32)

                for r in range(clen):
                    seq = start + r
                    q_decay = np.exp(a_cs[r]) * b_scale[r]
                    inter = q_rot[r] @ s_cur
                    d_q_decay[r] += np.sum(do[bi, hi, seq] * inter)
                    dq_rot[r] += q_decay * (do[bi, hi, seq] @ s_cur.T)
                    state_grad += np.outer(q_decay * q_rot[r], do[bi, hi, seq])

                for r in range(clen):
                    seq_r = start + r
                    for c2 in range(r + 1):
                        seq_c = start + c2
                        score = np.dot(q_rot[r], k_rot[c2])
                        decay = np.exp(a_cs[r] - a_cs[c2])
                        alpha = decay * np.dot(do[bi, hi, seq_r], v[bi, hi, seq_c])
                        dq_rot[r] += alpha * k_rot[c2]
                        dk_rot[c2] += alpha * q_rot[r]
                        dv_acc[c2] += do[bi, hi, seq_r] * (decay * score)
                        pair_term = alpha * score
                        d_a_cs[r] += pair_term
                        d_a_cs[c2] -= pair_term

                d_decay = np.sum(state_grad * s_prev)
                next_state_grad = state_grad * (np.exp(a_cs[-1]) * b_scale[-1])
                for t in range(clen):
                    seq = start + t
                    dk_rot[t] += state_grad @ v[bi, hi, seq]
                    dv_acc[t] += k_rot[t] @ state_grad

                d_a_cs[-1] += d_decay * np.exp(a_cs[-1]) * b_scale[-1]
                d_b_scale[-1] += d_decay * np.exp(a_cs[-1])
                for t in range(clen):
                    q_decay = np.exp(a_cs[t]) * b_scale[t]
                    d_a_cs[t] += d_q_decay[t] * q_decay
                    d_b_scale[t] += d_q_decay[t] * np.exp(a_cs[t])
                for t in range(clen):
                    seq = start + t
                    exp_neg = np.exp(-a_cs[t])
                    dB[bi, hi, seq] += d_b_scale[t] * exp_neg
                    d_a_cs[t] += d_b_scale[t] * (-b[bi, hi, seq] * exp_neg)
                for t in range(clen):
                    seq = start + t
                    for i in range(half):
                        theta = a_cs[t] * angle[bi, hi, seq, i] * PI
                        dq0, dq1, dk0, dk1, dtheta = rotate_backward_np(
                            q[bi, hi, seq, i], q[bi, hi, seq, i + half],
                            k[bi, hi, seq, i], k[bi, hi, seq, i + half],
                            theta,
                            dq_rot[t, i], dq_rot[t, i + half],
                            dk_rot[t, i], dk_rot[t, i + half],
                        )
                        dQ[bi, hi, seq, i] += dq0
                        dQ[bi, hi, seq, i + half] += dq1
                        dK[bi, hi, seq, i] += dk0
                        dK[bi, hi, seq, i + half] += dk1
                        dAngle[bi, hi, seq, i] += dtheta * a_cs[t] * PI
                        d_a_cs[t] += dtheta * angle[bi, hi, seq, i] * PI
                    dV[bi, hi, seq] += dv_acc[t]
                suffix = 0.0
                for t in range(clen - 1, -1, -1):
                    suffix += d_a_cs[t]
                    dA[bi, hi, start + t] += suffix
                state_grad = next_state_grad

    return out, (dQ, dK, dV, dA, dB, dAngle)


def err_stats(got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    diff = np.abs(got - ref)
    rel = np.linalg.norm((got - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12)
    return float(rel), float(diff.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dq", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    B = args.batch
    H = args.heads
    L = args.length
    DQ = args.dq
    DV = args.dv
    CHUNK = args.chunk
    seed = args.seed

    mx.random.seed(seed)
    q = mx.random.normal((B, H, L, DQ), dtype=mx.float32) * 0.5
    k = mx.random.normal((B, H, L, DQ), dtype=mx.float32) * 0.5
    v = mx.random.normal((B, H, L, DV), dtype=mx.float32) * 0.5
    angle = mx.random.normal((B, H, L, DQ // 2), dtype=mx.float32) * 0.1
    a = mx.random.normal((B, H, L), dtype=mx.float32) * 0.2 - 0.05
    b = mx.random.normal((B, H, L), dtype=mx.float32) * 0.2
    do = mx.random.normal((B, H, L, DV), dtype=mx.float32) * 0.5

    def loss_fn(q_, k_, v_, a_, b_, angle_):
        out = mamba3_siso_forward(q_, k_, v_, a_, b_, angle_, CHUNK).astype(mx.float32)
        return mx.sum(out * do)

    grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))

    q_np = np.array(q)
    k_np = np.array(k)
    v_np = np.array(v)
    a_np = np.array(a)
    b_np = np.array(b)
    angle_np = np.array(angle)
    do_np = np.array(do)
    ref_out, ref_grads = cpu_reference(q_np, k_np, v_np, a_np, b_np, angle_np, do_np, CHUNK)

    t0 = time.perf_counter()
    out = mamba3_siso_forward(q, k, v, a, b, angle, CHUNK)
    mx.eval(out)
    fwd_ms = (time.perf_counter() - t0) * 1e3

    t1 = time.perf_counter()
    grads = grad_fn(q, k, v, a, b, angle)
    mx.eval(*grads)
    bwd_ms = (time.perf_counter() - t1) * 1e3

    out_np = np.array(out)
    grads_np = [np.array(g) for g in grads]

    print("== MLX Mamba3 SISO Fwd+Bwd ==")
    print(f"B={B} H={H} L={L} DQ={DQ} DV={DV} CHUNK={CHUNK}")
    print(f"forward_ms={fwd_ms:.3f}")
    print(f"backward_ms={bwd_ms:.3f}")
    f_rel, f_max = err_stats(out_np, ref_out)
    print(f"fwd l2_rel={f_rel:.6f} max_abs={f_max:.6f}")
    for name, grad, ref in zip(["dQ", "dK", "dV", "dA", "dB", "dAngle"], grads_np, ref_grads):
        rel, mx_abs = err_stats(grad, ref)
        print(f"{name} l2_rel={rel:.6f} max_abs={mx_abs:.6f}")


if __name__ == "__main__":
    main()
