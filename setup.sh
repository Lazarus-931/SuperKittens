#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

color() { printf "\033[1;36m%s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m%s\033[0m\n" "$*"; }
fail()  { printf "\033[1;31m%s\033[0m\n" "$*" >&2; exit 1; }

color "[1/6] system check"
[[ "$(uname)" == "Darwin" ]] || fail "SuperKittens requires macOS."
[[ "$(uname -m)" == "arm64" ]] || fail "Apple Silicon required (got $(uname -m))."

CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")"
MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
MEM_GB=$((MEM_BYTES / 1024 / 1024 / 1024))
CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 0)"
MACOS_VER="$(sw_vers -productVersion)"
echo "  chip:    $CHIP"
echo "  memory:  ${MEM_GB} GB"
echo "  cores:   $CORES"
echo "  macOS:   $MACOS_VER"

color "[2/6] xcode toolchain"
if ! xcrun -f metal >/dev/null 2>&1; then
    fail "'metal' compiler not found. Install Xcode from the App Store, then:
    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
    sudo xcodebuild -license accept"
fi
if ! xcrun -f metallib >/dev/null 2>&1; then
    fail "'metallib' not found (Command Line Tools alone aren't enough).
    Install full Xcode from the App Store, then:
    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
    sudo xcodebuild -license accept
  Or, if Xcode is already installed:
    xcodebuild -downloadComponent MetalToolchain"
fi
echo "  metal:    $(xcrun -f metal)"
echo "  metallib: $(xcrun -f metallib)"

color "[3/6] homebrew + python3.12"
if ! command -v brew >/dev/null 2>&1; then
    warn "  installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi
if ! command -v python3.12 >/dev/null 2>&1; then
    brew install python@3.12
fi
brew list expat >/dev/null 2>&1 || brew install expat

color "[4/6] python venv at .venv"
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:${DYLD_LIBRARY_PATH:-}"
if [[ ! -d "$ROOT/.venv" ]]; then
    python3.12 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install -q --upgrade pip
pip install -q -U "huggingface_hub[cli]" numpy sentencepiece tokenizers

color "[5/6] build libsk"
./build.sh

color "[6/6] summary"
echo "  built: $ROOT/build/libsk.dylib + libsk.metallib"
echo
echo "next steps:"
echo "  1. authenticate with HF (for gated models like gemma):"
echo "     hf auth login"
echo "  2. download a model:"
echo "     ./SuperKittens/models/gemma/gemma4/download.sh e2b"
echo "  3. run a forward pass:"
echo "     source .venv/bin/activate"
echo "     python -c 'from SuperKittens.models.gemma.gemma4.gemma4 import Gemma4;"
echo "                m=Gemma4.from_pretrained(\"e2b\"); print(m.chat(\"Hi\"))'"
