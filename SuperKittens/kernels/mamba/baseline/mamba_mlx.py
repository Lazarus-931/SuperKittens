#
# mamba_mlx.py
# SuperKittens
#
# Created by Alazar Manakelew on 4/6/26.
#
# Minimal implementation of Tri Dao's and Albert Gu's Mamba 2 in MLX


import mlx.core as mlx
from einops import rearrange, repeat

#### Minimal SSD Implementation

def seg_sum(x: mlx.Tensor) -> mlx.Tensor:
    """Segmented sum function for alpha"""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = mlx.tril(mlx.ones([T, T], dtype=bool, stream=x.device), k=-1)
    x = x.masked_fill(mask, 0)
    sum = mlx.core.cumsum(mask, axis=0)
    x_segsum = mlx.cumsum(x, axis=-2)
    mask = mlx.tril(mlx.ones([T, T], dtype=bool, stream=x.device))
    x_segsum = x_segsum.masked_fill(mask, -mlx.inf)
    return x_segsum


def ssd(q: mlx.Tensor, k: mlx.Tensor, v: mlx.Tensor, a: mlx.Tensor) -> mlx.Tensor:
    """
    Sum
    :param q: C (batch, length, n_heads, d_state)
    :param k: B (batch, length, n_heads, d_state)
    :param v: X (batch, length, n_heads)
    :param a: L (batch, length, n_heads, d_state)
    :return: mlx.Tensor (batch, length, n_heads, d_state)
    """
    einops.








