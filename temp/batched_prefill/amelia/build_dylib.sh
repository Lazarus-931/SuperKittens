#!/bin/bash
# Builds two dylibs on a CLT-only host (no xcrun metal; kernels runtime-compile
# via SK_METAL_SRC_FALLBACK at run time):
#   build/libsk.dylib       — patched tree (sk_qwen_prefill_batched)
#   build/libsk_base.dylib  — origin/main copies of the 3 changed files
set -e
cd ~/sk-batched-prefill
SK_DIR="SuperKittens"

build_into() {
  local out="$1" bdir="$2"
  rm -rf "$bdir" && mkdir -p "$bdir"
  clang++ -std=gnu++20 -O3 -arch arm64 \
    -DNS_PRIVATE_IMPLEMENTATION -DMTL_PRIVATE_IMPLEMENTATION -DCA_PRIVATE_IMPLEMENTATION \
    -I metal-cpp -c "$SK_DIR/kernels/metal_impl.cpp" -o "$bdir/metal_impl.o"
  local OBJ=("$bdir/metal_impl.o")
  local FAIL=0
  while IFS= read -r -d "" f; do
    rel="${f#$SK_DIR/}"
    obj="$bdir/$(echo "$rel" | tr "/" "_" | sed "s/\.c++$//").o"
    if clang++ -std=gnu++20 -O3 -arch arm64 -I metal-cpp -c "$f" -o "$obj" 2>"$bdir/err_$(basename "$obj").log"; then
      OBJ+=("$obj")
    else
      echo "COMPILE FAIL: $f"; cat "$bdir/err_$(basename "$obj").log"; FAIL=1
    fi
  done < <(find "$SK_DIR/kernels" "$SK_DIR/models" "$SK_DIR/inference" -name "*.c++" -not -path "*/paged_attn/*" -not -path "*/utils/rmsnorm/*" -print0)
  [ "$FAIL" = 1 ] && { echo "BUILD ABORTED ($out)"; exit 1; }
  mkdir -p build
  clang++ -std=gnu++20 -arch arm64 -framework Metal -framework Foundation \
    -framework QuartzCore -framework Accelerate -dynamiclib "${OBJ[@]}" -o "$out"
  echo "BUILT: $(ls -l "$out")"
}

build_into build/libsk.dylib build/obj_new

CHANGED="models/qwen/launcher.c++ models/qwen/launcher.h models/qwen/qwen_model.h"
for f in $CHANGED; do
  cp "$SK_DIR/$f" "$SK_DIR/$f.patched"
  cp "base_files/$(basename "$f")" "$SK_DIR/$f"
done
build_into build/libsk_base.dylib build/obj_base
for f in $CHANGED; do
  mv "$SK_DIR/$f.patched" "$SK_DIR/$f"
done
echo "ALL DONE"
