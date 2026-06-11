"""Single-stream references only (runs on ANY dylib incl. pristine baseline):
per-prompt token-by-token prefill+decode vs production single-forward
prefill+decode. Proves whether their near-tie disagreement pre-exists the
batched-serving patch. Writes OUT_JSON for cross-dylib byte comparison."""
import json
import os
import sys

import numpy as np

REPO = os.environ.get("SK_REPO", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, REPO)
os.environ.setdefault("SK_DYLIB", os.path.join(REPO, "build", "libsk.dylib"))
os.environ.setdefault("SK_METALLIB", os.path.join(REPO, "build", "libsk.metallib"))

from SuperKittens.models.granite.granite import Granite, Config  # noqa: E402

GGUF = os.environ["SK_GRANITE_GGUF"]
N    = int(os.environ.get("N", "8"))
T    = int(os.environ.get("T", "32"))
DEC  = int(os.environ.get("DEC", "32"))


def main():
    cfg = Config(batch=1, seq_max=max(T, 64), cache_max=max(2 * T + DEC + 8, 256))
    m = Granite(cfg)
    m.load_gguf(GGUF)
    rng = np.random.default_rng(7)
    prompts = [rng.integers(100, 50000, size=T).astype(np.int32) for _ in range(N)]

    refs_tbt, refs_prod, logits_tail = [], [], []
    for p in prompts:
        m.reset()
        for t in p:
            cur = m.forward(np.array([t], dtype=np.int32))
        toks = [int(cur)]
        for _ in range(DEC):
            cur = m.forward(np.array([cur], dtype=np.int32))
            toks.append(int(cur))
        refs_tbt.append(toks)

        m.reset()
        cur = m.forward(p)
        toks = [int(cur)]
        for _ in range(DEC):
            cur = m.forward(np.array([cur], dtype=np.int32))
            toks.append(int(cur))
        refs_prod.append(toks)
        import hashlib
        logits_tail.append(hashlib.sha256(m._last_logits().tobytes()).hexdigest())

    agree = sum(1 for a, b in zip(refs_tbt, refs_prod) if a == b)
    print(f"[refs] dylib={os.environ['SK_DYLIB']}")
    print(f"[refs] tbt-vs-production agree {agree}/{N}")
    for i, (a, b) in enumerate(zip(refs_tbt, refs_prod)):
        if a != b:
            d = next(j for j, (x, y) in enumerate(zip(a, b)) if x != y)
            print(f"  lane {i}: diverge at tok {d}: tbt={a[d]} prod={b[d]}")
    out = os.environ.get("OUT_JSON",
                         os.path.join(os.path.dirname(__file__), "refs.json"))
    with open(out, "w") as f:
        json.dump(dict(refs_tbt=refs_tbt, refs_prod=refs_prod,
                       logits_sha256=logits_tail), f)
    print(f"[refs] wrote {out}")


if __name__ == "__main__":
    main()
