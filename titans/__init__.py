"""
Titans: Learning to Memorize at Test Time

A PyTorch implementation of the Titans architecture from the paper:
"Titans: Learning to Memorize at Test Time" by Behrouz, Zhong, and Mirrokni (Google Research)

This package provides:
- Neural Long-term Memory (LMM) module
- Three architectural variants: MAC, MAG, MAL
- Persistent memory components
- Full Titans models for sequence modeling

References:
    https://arxiv.org/abs/2501.00663
"""

from .memory import NeuralMemory, NeuralMemoryMLP
from .attention import MultiHeadAttention, SlidingWindowAttention
from .delta_attention import DeltaAttention, FlashDeltaHybrid, ChunkedFlashDelta, DeltaState
from .persistent import PersistentMemory
from .titans_mac import TitansMAC
from .titans_mag import TitansMAG
from .titans_mal import TitansMAL
from .model import TitansLM, TitansConfig

__version__ = "0.1.0"
__all__ = [
    "NeuralMemory",
    "NeuralMemoryMLP",
    "MultiHeadAttention", 
    "SlidingWindowAttention",
    "PersistentMemory",
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLM",
    "TitansConfig",
]
