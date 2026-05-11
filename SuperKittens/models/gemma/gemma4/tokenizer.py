"""tokenizer.py — Gemma 4 tokenizer glue (all variants share this).

The Gemma family uses a SentencePiece BPE tokenizer with a 262144-token
vocabulary. The same tokenizer file (`tokenizer.model`) ships with every
Gemma 4 variant — E2B, E4B, 26B-A4B, 31B all decode/encode identically.

This wrapper is intentionally thin. The actual tokenizer.model file comes
from the .sh weight-prep scripts that download model artifacts. We just
wrap the standard `sentencepiece` Python package with the special-token
IDs Gemma uses, plus the chat-template formatting for the instruction-tuned
variants.

Usage:
    from SuperKittens.models.gemma.gemma4.tokenizer import GemmaTokenizer

    tok = GemmaTokenizer.from_file("/path/to/tokenizer.model")
    ids = tok.encode("Why is the sky blue?")        # → list[int]
    text = tok.decode(ids)                          # → str
    chat_ids = tok.encode_chat([
        {"role": "user", "content": "Why is the sky blue?"},
    ])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


# Special token IDs — fixed across all Gemma 4 variants.
# Cross-checked against https://huggingface.co/google/gemma-4-E4B/blob/main/tokenizer_config.json
PAD_ID  = 0
EOS_ID  = 1
BOS_ID  = 2
UNK_ID  = 3

# Gemma chat template tokens
START_OF_TURN_ID = 105   # <start_of_turn>
END_OF_TURN_ID   = 106   # <end_of_turn>


@dataclass
class _ChatTurn:
    role: str         # "user" | "model" | "system"
    content: str


class GemmaTokenizer:
    """Thin wrapper over `sentencepiece` for the Gemma 262K vocab.

    Loads lazily so importing this module doesn't require sentencepiece
    until the user actually calls `from_file`. Lets you import the class
    in environments where the .model file isn't present yet.
    """

    def __init__(self, sp):
        self._sp = sp                          # SentencePieceProcessor
        self.vocab_size = sp.GetPieceSize()
        if self.vocab_size != 262144:
            # Soft warning — Gemma 4 ships 262144 but we don't crash on
            # mismatches in case someone wants to use a custom slice.
            import warnings
            warnings.warn(
                f"Gemma tokenizer expected vocab=262144, got {self.vocab_size}. "
                f"Make sure this is the right model file.",
                stacklevel=2,
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "GemmaTokenizer":
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "Gemma tokenizer requires `sentencepiece`. "
                "Install with: pip install sentencepiece"
            ) from e
        sp = spm.SentencePieceProcessor()
        sp.Load(str(path))
        return cls(sp)

    # ── Encoding / decoding ───────────────────────────────────────────

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode raw text. By default prepends BOS; pass add_bos=False to skip."""
        ids = self._sp.EncodeAsIds(text)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids back to text. Strips BOS/EOS automatically."""
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self._sp.DecodeIds(list(ids))

    # ── Chat template (instruction-tuned variants) ────────────────────

    def encode_chat(
        self,
        turns: Sequence[dict],
        *,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Format a chat history for an instruction-tuned Gemma 4 model.

        Gemma chat template:
            <start_of_turn>user
            {content}<end_of_turn>
            <start_of_turn>model
            {content}<end_of_turn>
            ...

        Each turn dict has "role" ∈ {"user","model","system"} and "content".
        If add_generation_prompt=True, ends with `<start_of_turn>model\n`
        for the model to continue from.
        """
        lines: List[str] = []
        for turn in turns:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
        if add_generation_prompt:
            lines.append("<start_of_turn>model\n")
        prompt = "\n".join(lines)
        return self.encode(prompt, add_bos=True, add_eos=False)

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def eos_id(self) -> int: return EOS_ID
    @property
    def bos_id(self) -> int: return BOS_ID
    @property
    def pad_id(self) -> int: return PAD_ID
