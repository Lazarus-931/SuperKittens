from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Any


class Model:
    """Base class for SK model wrappers.

    Subclasses provide:
        _handle, cfg, vocab_size                 — handle to native code + config
        _forward(input_ids: np.ndarray) -> np.ndarray   # returns argmax per-batch int32
        _last_logits() -> np.ndarray             # (vocab,) fp16 of last position
        reset()                                  # reset per-sequence cursor
        tokenizer                                # optional, attached by from_pretrained

    Lifecycle (close/__enter__/__exit__/__del__/__repr__) is provided here.
    Subclasses set:
        _destroy_fn — callable(handle) destroying the native handle (required for close)
        _handle_attr — name of the instance attribute holding the handle (default "_h")
        _repr_fields — optional tuple of (label, attr) for __repr__ (else uses cfg fields)
    """

    cfg: Any
    tokenizer: Optional[Any] = None
    _destroy_fn = None
    _handle_attr: str = "_h"
    _repr_fields: tuple = ()

    def close(self) -> None:
        h = getattr(self, self._handle_attr, None)
        if h and self._destroy_fn is not None:
            self._destroy_fn(h)
            setattr(self, self._handle_attr, None)
            if hasattr(self, "_w_keep"):
                self._w_keep = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        cls = type(self).__name__
        if self._repr_fields:
            parts = [f"{label}={getattr(self.cfg, attr, '?')}" for label, attr in self._repr_fields]
            return f"{cls}({', '.join(parts)})"
        return f"{cls}(cfg={type(self.cfg).__name__})"

    def _forward(self, input_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _last_logits(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float, top_p: float,
                top_k: Optional[int], rng: np.random.Generator) -> int:
        if temperature <= 0.0:
            return int(np.argmax(logits))
        x = logits.astype(np.float32) / temperature
        x -= x.max()
        p = np.exp(x); p /= p.sum()
        if top_k and 0 < top_k < p.size:
            idx = np.argpartition(p, -top_k)[-top_k:]
            mask = np.zeros_like(p); mask[idx] = p[idx]
            p = mask / mask.sum()
        if 0.0 < top_p < 1.0:
            order = np.argsort(p)[::-1]
            ps = p[order]
            cum = np.cumsum(ps)
            cutoff = int(np.searchsorted(cum, top_p)) + 1
            keep = order[:cutoff]
            mask = np.zeros_like(p); mask[keep] = p[keep]
            p = mask / mask.sum()
        return int(rng.choice(p.size, p=p))

    def generate(self, input_ids, *, max_new_tokens: int = 64,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k: Optional[int] = None, eos_id: Optional[int] = None,
                 eos_ids: Optional[set] = None,
                 seed: int = 0) -> list[int]:
        rng = np.random.default_rng(seed)
        ids = np.asarray(input_ids, dtype=np.int32).reshape(-1)
        # Resolve the full stop-set: explicit eos_ids wins, else fall back to
        # tokenizer.eos_ids (multi-stop), else single eos_id.
        stops: set = set()
        if eos_ids:
            stops |= {int(x) for x in eos_ids}
        if eos_id is not None:
            stops.add(int(eos_id))
        if not stops and self.tokenizer is not None:
            t_eos = getattr(self.tokenizer, "eos_ids", None)
            if t_eos:
                stops |= {int(x) for x in t_eos}
        self.reset()
        argmax_first = self._forward(ids)
        greedy = temperature <= 0.0 and (top_p >= 1.0 or top_p <= 0.0) and not top_k
        first = int(argmax_first[0]) if greedy else self._sample(
            self._last_logits(), temperature, top_p, top_k, rng)
        out = [first]
        if first in stops:
            return out
        last = first
        for _ in range(max_new_tokens - 1):
            arg = self._forward(np.array([last], dtype=np.int32))
            last = int(arg[0]) if greedy else self._sample(
                self._last_logits(), temperature, top_p, top_k, rng)
            out.append(last)
            if last in stops:
                break
        return out

    def chat(self, prompt, **gen_kwargs) -> str:
        """Apply the tokenizer's chat template (with BOS) and decode the response.

        `prompt` may be a string (treated as a single user message) or a list of
        chat-format dicts ([{"role":..., "content":...}, ...]).
        """
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer attached")
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        ids = self.tokenizer.chat(messages, bos=True)
        prompt_len = len(ids)
        eos_ids = gen_kwargs.pop("eos_ids", getattr(self.tokenizer, "eos_ids", None))
        eos = gen_kwargs.pop("eos_id", getattr(self.tokenizer, "eos_id", None))
        out_ids = self.generate(np.array(ids, dtype=np.int32),
                                eos_id=eos, eos_ids=eos_ids, **gen_kwargs)
        # Decode only the newly generated tail, skipping specials so trailing
        # <turn|>/<eos> never leak into the user-visible string.
        return self.tokenizer.decode(out_ids, skip_special=True)


def resolve_weights_dir(name: str, variant_to_dir: dict[str, str]) -> Path:
    if name in variant_to_dir:
        dir_name = variant_to_dir[name]
    else:
        dir_name = name
    root = Path(__file__).resolve().parents[2] / "SuperKittens" / "model_weights" / dir_name
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")
    return root


def attach_tokenizer(model: Model, root: Path) -> None:
    sp = root / "tokenizer.model"
    jh = root / "tokenizer.json"
    try:
        from SuperKittens.models.load.tokenizer import Tokenizer
    except Exception as e:
        print(f"[sk] tokenizer module unavailable: {e}")
        return
    if sp.exists():
        try:
            model.tokenizer = Tokenizer.from_sentencepiece(str(sp))
        except Exception as e:
            print(f"[sk] sentencepiece attach failed: {e}")
    elif jh.exists():
        try:
            model.tokenizer = Tokenizer.from_hf_json(str(jh))
        except Exception as e:
            print(f"[sk] hf-json attach failed: {e}")
