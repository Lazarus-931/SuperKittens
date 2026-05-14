#!/usr/bin/env bash
# Smoke test for quantize-to-gguf across all four supported architectures.
#
# For each family, point Q2G_FIXTURE_<FAMILY> at a local safetensors dir
# (containing config.json + tokenizer.json + model.safetensors). Missing
# fixtures are reported as "skip" along with the command that would
# validate them.
#
# Env:
#   Q2G_BIN              path to the built binary (default: ./quantize-to-gguf)
#   Q2G_FIXTURE_QWEN3    path to a Qwen3 safetensors dir
#   Q2G_FIXTURE_GEMMA    path to a Gemma3 safetensors dir
#   Q2G_FIXTURE_MAMBA    path to a Mamba2 safetensors dir
#   Q2G_FIXTURE_DEEPSEEK path to a DeepSeek V2/V3 safetensors dir
#   Q2G_OUT_DIR          where to place output .gguf files (default: $TMPDIR)

set -u
BIN="${Q2G_BIN:-./quantize-to-gguf}"
OUT="${Q2G_OUT_DIR:-${TMPDIR:-/tmp}}"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: quantizer binary not found at $BIN (set Q2G_BIN or build first)" >&2
    exit 1
fi

PASS=0
FAIL=0
SKIP=0

run_one() {
    local family="$1" fixture="$2" expect_arch="$3"
    local out="$OUT/q2g-test-$family.gguf"
    if [[ -z "$fixture" || ! -d "$fixture" ]]; then
        echo "[skip] $family : no fixture (would run: $BIN <dir> $out --quant q8_0)"
        SKIP=$((SKIP+1))
        return
    fi
    echo "[run]  $family : $BIN $fixture $out --quant q8_0"
    if ! "$BIN" "$fixture" "$out" --quant q8_0 >/tmp/q2g-test-$family.log 2>&1; then
        echo "[FAIL] $family : quantizer exited non-zero"
        tail -20 /tmp/q2g-test-$family.log
        FAIL=$((FAIL+1))
        return
    fi
    # Validate via the gguf python reader: arch key + tensor count.
    python3 - "$out" "$expect_arch" <<'PY'
import sys, gguf
path, expect = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(path)
arch_field = r.fields["general.architecture"]
arch = bytes(arch_field.parts[arch_field.data[0]]).decode()
n = len(r.tensors)
ok = (arch == expect) and (n > 0)
print(f"   -> arch={arch} tensors={n} {'OK' if ok else 'BAD'}")
sys.exit(0 if ok else 1)
PY
    if [[ $? -eq 0 ]]; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
    fi
}

run_one qwen3    "${Q2G_FIXTURE_QWEN3:-}"    qwen3
run_one gemma    "${Q2G_FIXTURE_GEMMA:-}"    gemma3
run_one mamba    "${Q2G_FIXTURE_MAMBA:-}"    mamba2
run_one deepseek "${Q2G_FIXTURE_DEEPSEEK:-}" deepseek2

echo
echo "summary: pass=$PASS fail=$FAIL skip=$SKIP"
[[ $FAIL -eq 0 ]]
