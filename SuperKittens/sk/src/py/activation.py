"""Activation bindings."""

def gelu(x, out=None):
    return _dispatch("gelu", x, out)

def silu(x, out=None):
    return _dispatch("silu", x, out)

def relu(x, out=None):
    return _dispatch("relu", x, out)

def _dispatch(name, x, out):
    raise NotImplementedError("bindings not yet wired — kernel library in progress")
