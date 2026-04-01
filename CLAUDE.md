# SuperKittens

ThunderKittens-inspired GPU kernel library for Apple Metal Shading Language (MSL).
Targets Apple Silicon (M2+) with optimized ML kernels.

## Build

- **IDE**: Xcode 26+
- **Target**: macOS 26.0, command-line tool
- **Language**: C++20 (GNU++20), Metal Shading Language
- **Dependencies**: metal-cpp (vendored in `metal-cpp/`), Metal.framework, Foundation.framework, QuartzCore.framework

Open `SuperKittens.xcodeproj` and build (Cmd+B). No external package managers.

## Project Structure

```
SuperKittens/
├── SuperKittens/               # Xcode source group
│   ├── main.cpp                # Entry point, Metal device setup, kernel dispatch
│   ├── MetalImpl.cpp           # metal-cpp private implementation (compile once)
│   ├── meow.h                  # Master include header (like TK's kittens.cuh)
│   ├── Shaders.metal           # Shared Metal utilities
│   └── kernels/                # Kernel implementations
│       ├── gemm/               # Matrix multiplication kernels
│       │   └── fp16_m2/        # FP16 GEMM optimized for M2
│       │       ├── fp16_m2_gemm.metal  # Metal kernel
│       │       └── fp16_m2_gemm.c++    # Host-side dispatch
│       └── attn/               # Attention kernels
│           ├── attn.metal      # Metal kernel
│           └── attn.h          # Host-side config (excluded from compile)
├── metal-cpp/                  # Apple's C++ Metal bindings (do not edit)
└── SuperKittens.xcodeproj/     # Xcode project
```

## Architecture

Follows ThunderKittens' organizational pattern adapted for Metal:

- **Kernels by type**: Each kernel family (gemm, attn) gets its own directory under `kernels/`
- **Per-kernel pair**: Each kernel has a `.metal` file (GPU shader) and a `.c++` file (host-side dispatch/config)
- **Chip variants**: Subdirectories per chip target (e.g., `fp16_m2/` for M2-optimized FP16 GEMM)
- **Template configs**: Metal kernels use template structs with `static_assert` for tile-size validation
- **meow.h**: Master header analogous to ThunderKittens' `kittens.cuh`

## Conventions

- Metal kernels use `kernel void` functions with explicit buffer bindings
- Host code uses metal-cpp (C++ wrappers around Metal API), not Objective-C
- MetalImpl.cpp must be the only file defining `NS_PRIVATE_IMPLEMENTATION` / `MTL_PRIVATE_IMPLEMENTATION`
- Tile sizes follow Metal SIMD group conventions (multiples of 16)
- Header search path: `$(PROJECT_DIR)/metal-cpp`

## Current Focus

GEMM kernels — specifically FP16 on M2.
