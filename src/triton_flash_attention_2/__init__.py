import importlib.metadata

from triton_flash_attention_2.flash_attention_triton import FlashAttentionTriton

__all__ = ["FlashAttentionTriton"]

try:
    __version__ = importlib.metadata.version("triton-flash-attention-2")
except importlib.metadata.PackageNotFoundError:
    pass
