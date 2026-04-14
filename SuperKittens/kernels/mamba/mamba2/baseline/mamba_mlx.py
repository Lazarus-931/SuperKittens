#
# mamba_mlx.py
# SuperKittens
#
# Created by Alazar Manakelew on 4/6/26.
#
# Minimal implementation of Tri Dao's and Albert Gu's Mamba 2 in MLX


import mlx.core as mlx


def seg_sum(x: mlx.array) -> mlx.array:
    """Inclusive lower-triangular segment sums over the last dimension."""
    t = x.shape[-1]
    csum = mlx.cumsum(x, axis=-1)
    prefix = mlx.concatenate(
        [mlx.zeros((*x.shape[:-1], 1), dtype=x.dtype), csum[..., :-1]],
        axis=-1,
    )
    seg = csum[..., :, None] - prefix[..., None, :]
    mask = mlx.tril(mlx.ones((t, t), dtype=mlx.bool_))
    return mlx.where(mask, seg, mlx.full(seg.shape, -mlx.inf, dtype=seg.dtype))


def ssd(
    q: mlx.array,
    k: mlx.array,
    v: mlx.array,
    x: mlx.array,
    block_len: int,
    initial_states: mlx.array | None = None,
) -> tuple[mlx.array, mlx.array]:
    """
    Minimal Mamba-2 SSD recurrence.

    Args:
        q: C, shape (batch, length, n_heads, d_state)
        k: B, shape (batch, length, n_heads, d_state)
        v: X, shape (batch, length, n_heads, d_value)
        x: A, shape (batch, length, n_heads)
        block_len: Number of time steps to process per chunk.
        initial_states: Optional initial state of shape
            (batch, n_heads, d_state, d_value)

    Returns:
        y: shape (batch, length, n_heads, d_value)
        final_state: shape (batch, n_heads, d_state, d_value)
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")

    if not (q.dtype == k.dtype == v.dtype == x.dtype):
        raise TypeError("q, k, v, and x must have the same dtype")

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or x.ndim != 3:
        raise ValueError("expected q/k/v to be 4D and x to be 3D")

    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")

    batch, length, n_heads, d_state = q.shape
    if v.shape[:3] != (batch, length, n_heads):
        raise ValueError("v must have shape (batch, length, n_heads, d_value)")

    if x.shape != (batch, length, n_heads):
        raise ValueError("x must have shape (batch, length, n_heads)")

    d_value = v.shape[-1]

    if initial_states is None:
        state = mlx.zeros((batch, n_heads, d_state, d_value), dtype=q.dtype)
    else:
        expected = (batch, n_heads, d_state, d_value)
        if initial_states.shape != expected:
            raise ValueError(
                f"initial_states must have shape {expected}, got {initial_states.shape}"
            )
        state = initial_states

    outputs = []
    for chunk_start in range(0, length, block_len):
        chunk_end = min(chunk_start + block_len, length)

        q_chunk = q[:, chunk_start:chunk_end]
        k_chunk = k[:, chunk_start:chunk_end]
        v_chunk = v[:, chunk_start:chunk_end]
        a_chunk = x[:, chunk_start:chunk_end]

        for i in range(q_chunk.shape[1]):
            decay = mlx.exp(a_chunk[:, i])[..., None, None]
            kv = mlx.einsum("bhn,bhp->bhnp", k_chunk[:, i], v_chunk[:, i])
            state = decay * state + kv
            y_t = mlx.einsum("bhn,bhnp->bhp", q_chunk[:, i], state)
            outputs.append(y_t)

    y = mlx.stack(outputs, axis=1)
    return y, state








