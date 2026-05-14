from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from .chat_templates import CHAT_TEMPLATES, gemma_template, qwen_template, deepseek_template


_BACKEND_SP = "sentencepiece"
_BACKEND_HF = "hf"
_BACKEND_GGUF = "gguf"


@dataclass
class Tokenizer:
    backend: str
    impl: Any = field(repr=False)
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    pad_id: Optional[int] = None
    unk_id: Optional[int] = None
    vocab_size: int = 0
    chat_template: Callable[[Sequence[dict], bool], str] = field(default=qwen_template, repr=False)
    family: str = "qwen"
    _special_ids: set = field(default_factory=set, repr=False)
    _gguf_vocab: Optional[List[str]] = field(default=None, repr=False)
    _gguf_lookup: Optional[dict] = field(default=None, repr=False)

    # ---- constructors ----

    @classmethod
    def from_sentencepiece(cls, model_path: str, *, family: str = "gemma") -> "Tokenizer":
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError("sentencepiece not installed; `pip install sentencepiece`") from e
        sp = spm.SentencePieceProcessor()
        sp.Load(str(Path(model_path).expanduser()))
        bos = sp.bos_id() if sp.bos_id() >= 0 else None
        eos = sp.eos_id() if sp.eos_id() >= 0 else None
        pad = sp.pad_id() if sp.pad_id() >= 0 else None
        unk = sp.unk_id() if sp.unk_id() >= 0 else None
        return cls(
            backend=_BACKEND_SP,
            impl=sp,
            bos_id=bos,
            eos_id=eos,
            pad_id=pad,
            unk_id=unk,
            vocab_size=sp.GetPieceSize(),
            chat_template=CHAT_TEMPLATES.get(family, gemma_template),
            family=family,
            _special_ids={i for i in (bos, eos, pad) if i is not None},
        )

    @classmethod
    def from_hf_json(cls, json_path: str, *, family: str = "qwen3") -> "Tokenizer":
        try:
            from tokenizers import Tokenizer as HFTok
        except ImportError as e:
            raise ImportError("`tokenizers` not installed; `pip install tokenizers`") from e
        hf = HFTok.from_file(str(Path(json_path).expanduser()))

        bos_id = eos_id = pad_id = unk_id = None
        specials = set()
        eos_candidates = ("<|im_end|>", "<end_of_turn>", "<turn|>", "<eos>", "<|endoftext|>")
        bos_candidates = ("<|im_start|>", "<start_of_turn>", "<bos>", "<|turn>", "<|startoftext|>", "<s>")
        pad_candidates = ("<pad>", "<|pad|>", "<|endoftext|>")
        for name in eos_candidates:
            tid = hf.token_to_id(name)
            if tid is not None:
                eos_id = tid; specials.add(tid); break
        for name in bos_candidates:
            tid = hf.token_to_id(name)
            if tid is not None:
                bos_id = tid; specials.add(tid); break
        for name in pad_candidates:
            tid = hf.token_to_id(name)
            if tid is not None:
                pad_id = tid; break
        tid = hf.token_to_id("<unk>")
        if tid is not None:
            unk_id = tid

        return cls(
            backend=_BACKEND_HF,
            impl=hf,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            unk_id=unk_id,
            vocab_size=hf.get_vocab_size(),
            chat_template=CHAT_TEMPLATES.get(family, qwen_template),
            family=family,
            _special_ids=specials,
        )

    @classmethod
    def from_gguf(cls, gguf_or_meta, *, family: str = "deepseek") -> "Tokenizer":
        """Build from GGUF metadata.

        `gguf_or_meta` is either:
          - a path-like to a .gguf file (requires `gguf` python package), or
          - a pre-parsed dict with keys:
              tokens: list[str]
              scores: list[float]   (optional)
              token_type: list[int] (optional; 3 = control/special)
              merges: list[str]     (optional)
              bos_id, eos_id, pad_id, unk_id (optional ints)
        """
        meta = _coerce_gguf_meta(gguf_or_meta)
        tokens: List[str] = list(meta["tokens"])
        token_type = meta.get("token_type") or []
        bos_id = meta.get("bos_id")
        eos_id = meta.get("eos_id")
        pad_id = meta.get("pad_id")
        unk_id = meta.get("unk_id")
        specials = set()
        for i, t in enumerate(token_type):
            if int(t) in (2, 3, 4, 5, 6):
                specials.add(i)
        for x in (bos_id, eos_id, pad_id):
            if x is not None:
                specials.add(int(x))
        lookup = {tok: i for i, tok in enumerate(tokens)}
        return cls(
            backend=_BACKEND_GGUF,
            impl=None,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            unk_id=unk_id,
            vocab_size=len(tokens),
            chat_template=CHAT_TEMPLATES.get(family, deepseek_template),
            family=family,
            _special_ids=specials,
            _gguf_vocab=tokens,
            _gguf_lookup=lookup,
        )

    # ---- core API ----

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> List[int]:
        if self.backend == _BACKEND_SP:
            ids = self.impl.EncodeAsIds(text)
        elif self.backend == _BACKEND_HF:
            ids = self.impl.encode(text, add_special_tokens=False).ids
        elif self.backend == _BACKEND_GGUF:
            ids = _gguf_bpe_encode(text, self._gguf_lookup)
        else:
            raise RuntimeError(f"unknown backend {self.backend}")
        if bos and self.bos_id is not None:
            ids = [self.bos_id] + list(ids)
        if eos and self.eos_id is not None:
            ids = list(ids) + [self.eos_id]
        return list(ids)

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        ids = [int(i) for i in ids]
        if skip_special:
            ids = [i for i in ids if i not in self._special_ids]
        if self.backend == _BACKEND_SP:
            return self.impl.DecodeIds(ids)
        if self.backend == _BACKEND_HF:
            return self.impl.decode(ids, skip_special_tokens=skip_special)
        if self.backend == _BACKEND_GGUF:
            pieces = [self._gguf_vocab[i] for i in ids if 0 <= i < len(self._gguf_vocab)]
            return "".join(pieces).replace("▁", " ").lstrip(" ")
        raise RuntimeError(f"unknown backend {self.backend}")

    def chat(self, messages: Sequence[dict], *, add_generation_prompt: bool = True,
             bos: bool = True) -> List[int]:
        prompt = self.chat_template(messages, add_generation_prompt)
        return self.encode(prompt, bos=bos, eos=False)

    def is_eos(self, token_id: int) -> bool:
        return self.eos_id is not None and int(token_id) == self.eos_id


# ---- helpers ----

def _coerce_gguf_meta(src) -> dict:
    if isinstance(src, dict):
        if "tokens" not in src:
            raise ValueError("gguf meta dict requires `tokens`")
        return src
    try:
        import gguf  # type: ignore
    except ImportError as e:
        raise ImportError(
            "from_gguf with a path requires the `gguf` package, or pass a pre-parsed dict "
            "with keys: tokens, [scores], [token_type], [merges], [bos_id], [eos_id], [pad_id]."
        ) from e
    reader = gguf.GGUFReader(str(Path(src).expanduser()))
    fields = reader.fields

    def _get(name, default=None):
        f = fields.get(name)
        if f is None:
            return default
        return f.parts[f.data[-1]].tolist() if f.data else default

    def _scalar(name, default=None):
        f = fields.get(name)
        if f is None or not f.data:
            return default
        return int(f.parts[f.data[-1]][0])

    tokens_field = fields.get("tokenizer.ggml.tokens")
    tokens = []
    if tokens_field is not None:
        for idx in tokens_field.data:
            raw = tokens_field.parts[idx].tobytes()
            tokens.append(raw.decode("utf-8", errors="replace"))
    return {
        "tokens": tokens,
        "scores": _get("tokenizer.ggml.scores"),
        "token_type": _get("tokenizer.ggml.token_type"),
        "merges": _get("tokenizer.ggml.merges"),
        "bos_id": _scalar("tokenizer.ggml.bos_token_id"),
        "eos_id": _scalar("tokenizer.ggml.eos_token_id"),
        "pad_id": _scalar("tokenizer.ggml.padding_token_id"),
        "unk_id": _scalar("tokenizer.ggml.unknown_token_id"),
    }


def _gguf_bpe_encode(text: str, lookup: dict) -> List[int]:
    """Fallback longest-prefix encode over a GGUF token table.

    This is intentionally simple: real DS4 tokenization should run through
    the merges table. Use this only as a placeholder until a proper
    BPE merger lands. SentencePiece-style space → U+2581 substitution is
    applied so the table lookups match.
    """
    s = text.replace(" ", "▁")
    out: List[int] = []
    i = 0
    n = len(s)
    while i < n:
        matched = None
        for j in range(min(n, i + 32), i, -1):
            tid = lookup.get(s[i:j])
            if tid is not None:
                matched = (tid, j)
                break
        if matched is None:
            byte_tok = lookup.get(f"<0x{ord(s[i]):02X}>")
            if byte_tok is not None:
                out.append(byte_tok)
            i += 1
        else:
            out.append(matched[0])
            i = matched[1]
    return out
