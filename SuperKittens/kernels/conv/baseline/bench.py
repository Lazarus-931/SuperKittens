#!/usr/bin/env python3
"""conv MLX baseline benchmark — conv1d, conv2d, conv3d."""
from conv import conv1d, conv2d, conv3d

def main():
    print("=" * 55)
    print("Conv — MLX Baseline")
    print("=" * 55)

    print("\n── conv1d ──")
    conv1d()

    print("\n── conv2d ──")
    conv2d()

    print("\n── conv3d ──")
    conv3d()

if __name__ == "__main__":
    main()
