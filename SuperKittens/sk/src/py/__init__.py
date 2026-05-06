"""
SuperKittens Python Bindings

SuperKittens is a tiling DSL for Metal Shading Language.
It provides a compiler that decomposes GPU compute problems
into 8x8 tiles distributed across SIMD groups.

The Python layer connects to compiled Metal kernels via ctypes.
It auto-selects the fastest kernel variant for the target GPU
family and input dimensions.

Usage:
    import superkittens as sk
    y = sk.gelu(x)           # activation
    y = sk.rmsnorm(x, w)     # normalization
    y = sk.attention(q,k,v)  # attention (auto FA vs MHA)
    c = sk.gemm(a, b)        # matrix multiply
"""

from . import _core
from . import activation
from . import norm
from . import gemm

# Top-level convenience imports
from .activation import gelu, silu, relu
from .norm import rmsnorm, layernorm
