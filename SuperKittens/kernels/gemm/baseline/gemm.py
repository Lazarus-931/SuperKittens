from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def gemm_nn(a: mx.array, b: mx.array, c_in: mx.array | None = None,
            alpha: float = 1.0, beta: float = 0.0) -> mx.array:
    out = alpha * mx.matmul(a, b)
    if c_in is not None and beta != 0.0:
        out = out + beta * c_in
    return out


def gemm_bias_silu(a: mx.array, b: mx.array, bias: mx.array,
                   c_in: mx.array | None = None,
                   alpha: float = 1.0, beta: float = 0.0) -> mx.array:
    out = alpha * mx.matmul(a, b)
    if c_in is not None and beta != 0.0:
        out = out + beta * c_in
    out = out + bias
    return out * mx.sigmoid(out)


def load_half(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float16).reshape(shape)


def load_float(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def parse_meta(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        if " " in line:
            parts = line.split()
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k] = v
            continue
        k, v = line.split("=", 1)
        meta[k] = v
    return meta


def err_stats(got: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    diff = np.abs(got - ref)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    l2_rel = float(np.linalg.norm((got - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12))
    return max_abs, mean_abs, l2_rel
