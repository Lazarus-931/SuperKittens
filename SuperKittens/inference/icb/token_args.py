"""Python mirror of `sk::TokenArgs` (see token_args.h).

WHY: host patches a 32-byte block in a shared MTL::Buffer per token (no
encoder, ICB-safe). This module provides the ctypes layout + a state-machine
helper used by the ICB record/replay tests, kept dependency-light so it
imports without libsk loaded.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Callable, List, Optional


class TokenArgs(ctypes.Structure):
    """Must stay bit-identical to `sk::TokenArgs` (32 bytes, 4-byte aligned)."""

    _fields_ = [
        ("current_pos", ctypes.c_uint32),
        ("kv_idx_base", ctypes.c_uint32),
        ("token_id",    ctypes.c_int32),
        ("layer_idx",   ctypes.c_uint32),
        ("reserved",    ctypes.c_uint32 * 4),
    ]


# Slot byte offsets used by `mark_resource` callers that bind sub-fields
# (e.g. argmax_args pattern).
OFFSET_CURRENT_POS = 0
OFFSET_KV_IDX_BASE = 4
OFFSET_TOKEN_ID    = 8
OFFSET_LAYER_IDX   = 12

TOKEN_ARGS_SIZE = 32


def patch(dst: bytearray, args: TokenArgs) -> None:
    """memcpy-equivalent: write `args` into a bytearray (mock of MTL::Buffer.contents)."""
    if len(dst) < TOKEN_ARGS_SIZE:
        raise ValueError(f"dst too small: {len(dst)} < {TOKEN_ARGS_SIZE}")
    ctypes.memmove((ctypes.c_char * TOKEN_ARGS_SIZE).from_buffer(dst), ctypes.byref(args), TOKEN_ARGS_SIZE)


# ----------------------------------------------------------------------
# Record / replay state machine — pure-Python mock of the IcbRecorder
# usage protocol. Lets us unit-test the host-side sequencing logic
# without any GPU. The real recorder lives in inference/silicon/icb_recorder.{h,c++}.
# ----------------------------------------------------------------------

@dataclass
class IcbStateMachine:
    """Models the lifecycle: ALLOC → RECORD → (PATCH → REPLAY)+ → DESTROY.

    The contract being tested:
      * record() may only be called between ALLOC and the first REPLAY.
      * patch() may only be called after at least one record() and never during a replay.
      * replay() observes the *latest* patched TokenArgs.
    """

    _state: str = "ALLOC"
    _slots: List[dict] = field(default_factory=list)
    _args_buf: bytearray = field(default_factory=lambda: bytearray(TOKEN_ARGS_SIZE))
    _replays: int = 0
    on_dispatch: Optional[Callable[[dict, TokenArgs], None]] = None

    def record(self, slot_idx: int, pso: str, buffers: list, offsets: list) -> None:
        if self._state not in ("ALLOC", "RECORDED", "PATCHED", "REPLAYED"):
            raise RuntimeError(f"record() forbidden in state {self._state}")
        while len(self._slots) <= slot_idx:
            self._slots.append({})
        self._slots[slot_idx] = dict(pso=pso, buffers=list(buffers), offsets=list(offsets))
        self._state = "RECORDED"

    def patch(self, args: TokenArgs) -> None:
        if self._state not in ("RECORDED", "REPLAYED"):
            raise RuntimeError(f"patch() forbidden in state {self._state}")
        patch(self._args_buf, args)
        self._state = "PATCHED"

    def replay(self) -> TokenArgs:
        if self._state not in ("PATCHED", "REPLAYED"):
            raise RuntimeError(f"replay() forbidden in state {self._state}")
        current = TokenArgs.from_buffer_copy(bytes(self._args_buf))
        self._replays += 1
        if self.on_dispatch is not None:
            for slot in self._slots:
                if slot:
                    self.on_dispatch(slot, current)
        self._state = "REPLAYED"
        return current

    @property
    def replays(self) -> int:
        return self._replays

    @property
    def state(self) -> str:
        return self._state
