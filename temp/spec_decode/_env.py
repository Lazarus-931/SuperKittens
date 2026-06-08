"""Local env shim: point SK at the freshly-built dylib/metallib in build/.

WHY: the worktree has no installed wheel; SK_DYLIB/SK_METALLIB override the
bundled _libs/ lookup so the spec-decode ABI symbols (get_logits_rows, get_pos,
set_pos) and the gemm_mma PSOs from this branch's build are used.
"""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # worktree root
os.environ.setdefault("SK_DYLIB", str(_ROOT / "build" / "libsk.dylib"))
os.environ.setdefault("SK_METALLIB", str(_ROOT / "build" / "libsk.metallib"))
