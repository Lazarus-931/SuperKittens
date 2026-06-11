"""DiffusionGemma (block text-diffusion MoE on a Gemma-4 backbone) family.

Stage 1 (logits parity) ships: GGUF-native loader (`gguf_io`, `config`), the
ggml-mirror CPU oracle (`graph_ref`), and the Metal unified forward
(`forward_metal`). Sampler / cached decode are Stage 2+.
"""
from .config import DiffusionGemmaConfig, config_from_gguf
from .gguf_io import GGUFFile

__all__ = ["DiffusionGemmaConfig", "config_from_gguf", "GGUFFile"]
