#!/bin/bash
set -e
# Run all SuperKittens tests

BUILD="build"
TESTS="SuperKittens/tests"
SK_INC="-I SuperKittens/sk/src/cpp -I SuperKittens/sk/compiler -I SuperKittens/inference/runtime"
CPP_FLAGS="-std=gnu++20 -O2 -arch arm64 -I metal-cpp $SK_INC -framework Metal -framework Foundation -framework QuartzCore -Lbuild -lsk"

mkdir -p "$BUILD"

echo "=== compiling Metal test kernels ==="
for metal in $TESTS/cpp/kernels/test_*.metal; do
    name=$(basename "$metal" .metal)
    air="$BUILD/${name}.air"
    echo "  $name"
    xcrun -sdk macosx metal -std=metal3.1 -O3 \
        -I SuperKittens/sk/src/cpp -I SuperKittens/sk/compiler \
        -c "$metal" -o "$air"
done

echo "=== linking test metallib ==="
xcrun -sdk macosx metallib $BUILD/test_*.air -o "$BUILD/test_dsl.metallib"

echo "=== compiling C++ host tests ==="
for cpp in $TESTS/cpp/test_*.cpp; do
    name=$(basename "$cpp" .cpp)
    bin="$BUILD/$name"
    echo "  $name"
    clang++ $CPP_FLAGS "$cpp" -o "$bin"
done

echo ""
echo "=== running tests ==="
export DYLD_LIBRARY_PATH="$PWD/build:$DYLD_LIBRARY_PATH"
failed=0
for bin in $BUILD/test_dsl $BUILD/test_tensor $BUILD/test_dispatch; do
    if [ -x "$bin" ]; then
        echo ""
        ./"$bin" || failed=1
    fi
done

echo ""
if [ $failed -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $failed
