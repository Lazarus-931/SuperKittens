"""tokenizer.py — DeepSeek V4 Flash tokenizer glue.

DS4 uses a SentencePiece BPE tokenizer with a ~129,280-token vocabulary plus
DSML-extension special tokens for tool calling and chat-turn framing. Wraps
the standard `sentencepiece` Python package + special-token handling +
chat-template rendering.

Usage:
    from models.deepseek.tokenizer import DeepSeekTokenizer
    tok = DeepSeekTokenizer.from_file("/path/to/ds4_tokenizer.model")
    ids = tok.encode("hi")
    text = tok.decode(ids)
    chat_ids = tok.encode_chat([
        {"role": "user", "content": "hi"},
    ])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


# Special-token IDs for DS4. These match the values in the official
# DeepSeek V4 tokenizer config; verify against tokenizer_config.json in
# the model release before shipping with real weights.
BOS_ID = 0
EOS_ID = 1
PAD_ID = 2
UNK_ID = 3

# Chat / DSML framing
START_OF_TURN_ID  = 100000   # <｜start_of_turn｜> placeholder; verify
END_OF_TURN_ID    = 100001   # <｜end_of_turn｜>
USER_ROLE_ID      = 100002   # <｜user｜>
ASSISTANT_ROLE_ID = 100003   # <｜assistant｜>
SYSTEM_ROLE_ID    = 100004   # <｜system｜>

# DSML tool-call markers (ds4 streams these at temp=0 during sampling)
DSML_OPEN_ID  = 100100
DSML_CLOSE_ID = 100101


@dataclass
class DeepSeekTokenizer:
    """Thin wrapper around sentencepiece.SentencePieceProcessor + chat templates."""
    sp: object = field(repr=False)
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    pad_id: int = PAD_ID
    vocab_size: int = 129280

    @classmethod
    def from_file(cls, model_path: str) -> "DeepSeekTokenizer":
        """Load a `tokenizer.model` (SentencePiece protobuf)."""
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise RuntimeError(
                "DeepSeekTokenizer needs `sentencepiece`. Install with "
                "`pip install sentencepiece`."
            ) from e
        sp = spm.SentencePieceProcessor()
        sp.Load(str(Path(model_path).expanduser()))
        return cls(sp=sp, vocab_size=sp.vocab_size())

    # ── encoding / decoding ──────────────────────────────────────────
    def encode(self, text: str, *, bos: bool = True, eos: bool = False) -> List[int]:
        """Text → token IDs. Prepend BOS / append EOS optionally."""
        ids = self.sp.EncodeAsIds(text)
        if bos: ids = [self.bos_id] + ids
        if eos: ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Token IDs → text. Strips control / framing tokens."""
        special = {self.bos_id, self.eos_id, self.pad_id,
                   START_OF_TURN_ID, END_OF_TURN_ID,
                   USER_ROLE_ID, ASSISTANT_ROLE_ID, SYSTEM_ROLE_ID,
                   DSML_OPEN_ID, DSML_CLOSE_ID}
        filtered = [int(i) for i in ids if int(i) not in special]
        return self.sp.DecodeIds(filtered)

    # ── chat-template framing ────────────────────────────────────────
    def encode_chat(self, messages: list, *, add_generation_prompt: bool = True) -> List[int]:
        """Render a list of `{role, content}` messages into DS4's chat-turn format.

        Final framing (add_generation_prompt=True): caller is preparing the
        model to emit the assistant's reply, so we close with the assistant-
        role + start-of-turn marker but don't include any content yet.
        """
        ids: List[int] = [self.bos_id]
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            role_tok = {
                "user": USER_ROLE_ID,
                "assistant": ASSISTANT_ROLE_ID,
                "system": SYSTEM_ROLE_ID,
            }.get(role, USER_ROLE_ID)
            ids.append(role_tok)
            ids.append(START_OF_TURN_ID)
            ids.extend(self.sp.EncodeAsIds(content))
            ids.append(END_OF_TURN_ID)
        if add_generation_prompt:
            ids.append(ASSISTANT_ROLE_ID)
            ids.append(START_OF_TURN_ID)
        return ids

    # ── helpers ──────────────────────────────────────────────────────
    def is_eos(self, token_id: int) -> bool:
        return int(token_id) in (self.eos_id, END_OF_TURN_ID)


# ── Stub fallback for testing without a real tokenizer.model ─────────
class _StubTokenizer:
    """Identity-map tokenizer for smoke tests when no model file is available.
    Maps bytes to IDs 1:1 (vocab_size=256). Useful for end-to-end API testing
    when you don't want to ship a real DS4 tokenizer.model."""
    vocab_size = 256
    bos_id, eos_id, pad_id = 0, 1, 2

    def encode(self, text: str, *, bos: bool = True, eos: bool = False) -> List[int]:
        ids = [int(b) for b in text.encode("utf-8")]
        if bos: ids = [self.bos_id] + ids
        if eos: ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        bs = bytes(int(i) for i in ids if int(i) >= 32 and int(i) < 256)
        return bs.decode("utf-8", errors="replace")

    def encode_chat(self, messages, **_) -> List[int]:
        text = "\n".join(m.get("content", "") for m in messages)
        return self.encode(text)

    def is_eos(self, token_id: int) -> bool:
        return int(token_id) == self.eos_id


def stub_tokenizer() -> _StubTokenizer:
    """Identity-byte tokenizer for smoke testing without a model file."""
    return _StubTokenizer()
