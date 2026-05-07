#!/bin/bash
set -e
# SuperKittens build script — compiles .metal → .metallib, .c++ → .dylib
# Prerequisites: Xcode 16+ (or Command Line Tools with Metal compiler)
# Usage: ./build.sh

BUILD_DIR="build"
SK_DIR="SuperKittens"
KERNELS_DIR="$SK_DIR/kernels"

mkdir -p "$BUILD_DIR"

echo "=== compiling Metal kernels ==="
AIR_FILES=()
while IFS= read -r -d '' metal_file; do
    rel="${metal_file#$KERNELS_DIR/}"
    air="$BUILD_DIR/$(echo "$rel" | tr '/' '_' | sed 's/\.metal$//').air"
    echo "  $rel"
    if xcrun -sdk macosx metal -std=metal3.1 -O3 \
        -I "$SK_DIR/sk/src/cpp" -I "$SK_DIR/sk/compiler" -I "$SK_DIR/include" \
        -c "$metal_file" -o "$air" 2>/dev/null; then
        AIR_FILES+=("$air")
    else
        echo "    ⚠ compile failed — skipping"
    fi
done < <(find "$KERNELS_DIR" -name "*.metal" \
    -not -path "*/paged_attn/*" \
    -not -path "*/utils/rmsnorm/*" \
    -print0)

echo "=== linking metallib ==="
xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$BUILD_DIR/libsk.metallib"
echo "  → build/libsk.metallib"

echo "=== compiling C dispatchers ==="
# metal_impl.cpp defines metal-cpp private symbols ONCE
IMPL_STUB="$KERNELS_DIR/metal_impl.cpp"
echo "  $IMPL_STUB (metal-cpp impls)"
clang++ -std=gnu++20 -O3 -arch arm64 \
    -DNS_PRIVATE_IMPLEMENTATION -DMTL_PRIVATE_IMPLEMENTATION -DCA_PRIVATE_IMPLEMENTATION \
    -I metal-cpp -c "$IMPL_STUB" -o "$BUILD_DIR/metal_impl.o"

OBJ_FILES=("$BUILD_DIR/metal_impl.o")
while IFS= read -r -d '' cpp_file; do
    obj="$BUILD_DIR/$(basename "$cpp_file" .c++).o"
    echo "  $cpp_file"
    if clang++ -std=gnu++20 -O3 -arch arm64 -I metal-cpp \
        -c "$cpp_file" -o "$obj" 2>/dev/null; then
        OBJ_FILES+=("$obj")
    else
        echo "    ⚠ failed — skipping"
    fi
done < <(find "$KERNELS_DIR" "SuperKittens/inference" -name "*.c++" \
    -not -path "*/paged_attn/*" -not -path "*/utils/rmsnorm/*" -not -name "metal_impl.cpp" \
    -print0)

echo "=== linking dylib ==="
clang++ -std=gnu++20 -arch arm64 \
    -framework Metal -framework Foundation -framework QuartzCore \
    -dynamiclib "${OBJ_FILES[@]}" -o "$BUILD_DIR/libsk.dylib" 2>/dev/null \
    && echo "  → build/libsk.dylib" \
    || echo "  ⚠ dylib link failed — use Xcode project for full build"

echo ""
echo "Build complete.  Files in build/:"
ls -lh "$BUILD_DIR"/*.{metallib,dylib}
