"""Phase-2 plumbing tests: TokenArgs struct + ICB record/patch/replay state machine.

No GPU, no model forward. Validates host-side patch protocol invariants.
"""
from __future__ import annotations

import ctypes
import os
import re

import pytest

from SuperKittens.inference.icb.token_args import (
    IcbStateMachine,
    TOKEN_ARGS_SIZE,
    TokenArgs,
    patch,
)


# ---- struct shape -----------------------------------------------------

def test_token_args_size_is_32_bytes():
    assert ctypes.sizeof(TokenArgs) == 32 == TOKEN_ARGS_SIZE


def test_token_args_field_offsets_match_cpp_layout():
    # Mirrors sk::TokenArgs in inference/icb/token_args.h.
    assert TokenArgs.current_pos.offset == 0
    assert TokenArgs.kv_idx_base.offset == 4
    assert TokenArgs.token_id.offset    == 8
    assert TokenArgs.layer_idx.offset   == 12
    assert TokenArgs.reserved.offset    == 16
    assert TokenArgs.reserved.size      == 16  # 4 * uint32


def test_cpp_header_static_assert_size():
    """Belt-and-suspenders: the C++ header asserts 32 B. Make sure nobody
    edited it to a different size without updating both ends."""
    here = os.path.dirname(__file__)
    header = os.path.abspath(os.path.join(here, "..", "token_args.h"))
    with open(header) as f:
        src = f.read()
    assert re.search(r"static_assert\(sizeof\(TokenArgs\)\s*==\s*32", src), \
        "C++ TokenArgs size static_assert missing or wrong"


# ---- host patch API ---------------------------------------------------

def test_patch_writes_bit_exact_bytes():
    buf = bytearray(TOKEN_ARGS_SIZE)
    a = TokenArgs(current_pos=42, kv_idx_base=7, token_id=-3,
                  layer_idx=11, reserved=(0, 0, 0, 0))
    patch(buf, a)
    decoded = TokenArgs.from_buffer_copy(bytes(buf))
    assert decoded.current_pos == 42
    assert decoded.kv_idx_base == 7
    assert decoded.token_id    == -3
    assert decoded.layer_idx   == 11


def test_patch_rejects_undersized_dst():
    with pytest.raises(ValueError):
        patch(bytearray(8), TokenArgs())


# ---- ICB state machine ------------------------------------------------

def test_must_record_before_patch():
    m = IcbStateMachine()
    with pytest.raises(RuntimeError):
        m.patch(TokenArgs(current_pos=1))


def test_must_patch_before_replay():
    m = IcbStateMachine()
    m.record(0, "rmsnorm", buffers=["x", "w", "out"], offsets=[0, 0, 0])
    with pytest.raises(RuntimeError):
        m.replay()


def test_record_then_patch_then_replay_basic():
    m = IcbStateMachine()
    m.record(0, "rmsnorm", buffers=["x", "w", "out", "token_args"],
             offsets=[0, 0, 0, 0])
    m.patch(TokenArgs(current_pos=1, token_id=100))
    out = m.replay()
    assert out.current_pos == 1
    assert out.token_id == 100
    assert m.replays == 1


def test_multi_token_replay_observes_each_patch():
    """Record-once, patch-replay-N: the kernel must see each token's args."""
    seen: list[tuple[int, int]] = []
    m = IcbStateMachine(on_dispatch=lambda _slot, a: seen.append((a.current_pos, a.token_id)))
    m.record(0, "kv_cache_write", buffers=["k", "v", "kc", "vc", "token_args"],
             offsets=[0, 0, 0, 0, 0])
    for pos, tok in [(0, 31), (1, 42), (2, 53), (3, 64)]:
        m.patch(TokenArgs(current_pos=pos, token_id=tok))
        m.replay()
    assert seen == [(0, 31), (1, 42), (2, 53), (3, 64)]
    assert m.replays == 4


def test_record_after_replay_is_allowed_for_reconfigure():
    """Recording is idempotent until destroy; useful for re-binding when
    e.g. a new layer's PSO needs to be installed in a slot."""
    m = IcbStateMachine()
    m.record(0, "kernel_v1", buffers=["a"], offsets=[0])
    m.patch(TokenArgs())
    m.replay()
    # Re-record same slot — allowed.
    m.record(0, "kernel_v2", buffers=["a", "b"], offsets=[0, 0])
    m.patch(TokenArgs(current_pos=99))
    out = m.replay()
    assert out.current_pos == 99


def test_two_slot_record_and_replay():
    """Mirrors the existing argmax_partial → argmax_reduce ICB pair."""
    dispatched: list[str] = []
    m = IcbStateMachine(on_dispatch=lambda slot, _a: dispatched.append(slot["pso"]))
    m.record(0, "argmax_partial", buffers=["logits", "val", "idx", "args"],
             offsets=[0, 0, 0, 0])
    m.record(1, "argmax_reduce",  buffers=["val", "idx", "out", "args"],
             offsets=[0, 0, 0, 4])
    m.patch(TokenArgs(current_pos=5))
    m.replay()
    assert dispatched == ["argmax_partial", "argmax_reduce"]
