"""Gate 1: a multi-token forward at current_pos>0 must return CORRECT per-row
logits (the spike's mha_causal Br=2 / kv_len>seq bug, fixed by PR #54).

Method: build an autoregressive ground-truth token sequence with seq=1 decode
(each forward at current_pos>0, one token). Then replay the SAME token stream as
ONE multi-token forward at a non-empty KV (the verify scenario) and require that
every per-position argmax matches the autoregressive next-token. Repeated at
several prompt lengths and chunk widths K so the Br=2 / parity pattern (which
broke rows 1,3 in the spike) is exercised.
"""
import _env  # noqa: F401
import numpy as np
import SuperKittens as sk
import SuperKittens.models.qwen.qwen  # register adapter

VOCAB = 151936


def argmax_row(r):
    return int(np.argmax(r.astype(np.float32)))


def autoregressive_next(m, prompt_ids, n):
    """seq=1 ground truth: returns the list of next-tokens predicted after each
    position of [prompt + generated], one forward per step (current_pos>0)."""
    m.reset()
    m.forward(np.asarray(prompt_ids, dtype=np.int32))
    cur = argmax_row(m.get_logits_rows(len(prompt_ids))[-1])
    seq = [cur]
    for _ in range(n - 1):
        m.forward(np.asarray([cur], dtype=np.int32))
        cur = argmax_row(m.get_logits_rows(1)[-1])
        seq.append(cur)
    return seq  # seq[i] = token after (prompt + seq[:i])


def verify_chunk(m, prompt_ids, chunk):
    """Multi-token forward of `chunk` AFTER prefilling prompt (current_pos>0).
    Returns the per-row argmaxes (row i predicts the token after chunk position i)."""
    m.reset()
    m.forward(np.asarray(prompt_ids, dtype=np.int32))
    m.forward(np.asarray(chunk, dtype=np.int32))
    rows = m.get_logits_rows(len(chunk))
    return [argmax_row(rows[i]) for i in range(len(chunk))]


def main():
    m = sk.load("qwen3-0.6b", cache_max=512)
    fails = 0
    total = 0
    for prompt_len in (1, 7, 16, 17, 31, 33):
        prompt = list(range(10, 10 + prompt_len))
        # Ground-truth autoregressive continuation of length 8.
        gt = autoregressive_next(m, prompt, 8)
        # Verify scenario: feed [first_token] + draft proposals as ONE chunk.
        # chunk = [gt0, gt1, ..., gt_{K-1}]; row i should predict gt[i+1] (and the
        # last row predicts the bonus token = gt continuation step K).
        for K in (2, 4, 6):
            chunk = gt[:K]
            tgt = verify_chunk(m, prompt, chunk)
            # row i predicts token after chunk[i]; ground truth for that is gt[i+1]
            # (the autoregressive token following the same prefix).
            gt_after = gt[1:K] + [autoregressive_next(m, prompt + chunk, 1)[0]]
            mism = [i for i in range(K) if tgt[i] != gt_after[i]]
            total += K
            fails += len(mism)
            status = "OK" if not mism else f"MISMATCH rows {mism}"
            print(f"prompt_len={prompt_len:2d} K={K}: tgt={tgt} gt={gt_after}  {status}")
    m.close()
    print(f"\nTOTAL rows checked={total}  mismatches={fails}")
    print("GATE 1", "PASS — verify logits correct at current_pos>0" if fails == 0
          else "FAIL — interior logits still wrong")


if __name__ == "__main__":
    main()
