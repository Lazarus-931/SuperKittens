from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"
MAMBA_BUILD = BUILD / "mamba"

M3_BIN = BUILD / "mamba3_smoke_test"
M3_LIB = MAMBA_BUILD / "mamba3.metallib"

LENGTHS = [64, 128, 256, 512, 1024]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=False)
    print()


def main() -> None:
    for length in LENGTHS:
        run([str(M3_BIN), str(M3_LIB), "siso", str(length), "1", "2"])
    for length in LENGTHS:
        run([str(M3_BIN), str(M3_LIB), "mimo", str(length), "1", "2"])


if __name__ == "__main__":
    main()
