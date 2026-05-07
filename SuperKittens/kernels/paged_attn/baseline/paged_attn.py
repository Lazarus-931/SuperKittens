"""Paged attention — MLX reference."""
import mlx.core as mx
import numpy as np


def paged_attention(Q: mx.array, K_cache: mx.array, V_cache: mx.array,
                    block_table: mx.array, seq_lens: mx.array,
                    block_size: int = 16) -> mx.array:
    """Q: (num_seqs, num_heads, head_dim). K/V_cache: (num_blocks, block_size, num_kv_heads, head_dim).
    block_table: (num_seqs, max_blocks) int32. seq_lens: (num_seqs,) int32."""
    num_seqs, num_heads, head_dim = Q.shape
    _, _, num_kv_heads, _ = K_cache.shape
    scale = 1.0 / np.sqrt(float(head_dim))

    # Convert to numpy for indexing ease
    Q_np  = np.array(Q)
    K_np  = np.array(K_cache)
    V_np  = np.array(V_cache)
    bt_np = np.array(block_table)
    sl_np = np.array(seq_lens)

    O_np = np.zeros((num_seqs, num_heads, head_dim), dtype=np.float16)

    for s in range(num_seqs):
        sl = int(sl_np[s])
        nblocks = (sl + block_size - 1) // block_size
        K_rows, V_rows = [], []
        for b in range(nblocks):
            bid = int(bt_np[s, b])
            tk = min(block_size, sl - b * block_size)
            K_rows.append(K_np[bid, :tk, :, :].reshape(-1, head_dim))
            V_rows.append(V_np[bid, :tk, :, :].reshape(-1, head_dim))
        K_all = np.concatenate(K_rows, axis=0)
        V_all = np.concatenate(V_rows, axis=0)

        for h in range(num_heads):
            kv_h = (h * num_kv_heads) // num_heads
            qs = Q_np[s, h].astype(np.float32) * scale
            ks = K_all[kv_h::num_kv_heads, :].astype(np.float32)
            vs = V_all[kv_h::num_kv_heads, :].astype(np.float32)
            scores = qs @ ks.T
            probs = np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()
            O_np[s, h] = (probs @ vs).astype(np.float16)

    return mx.array(O_np)
