"""
Titans MAL (Memory as Layer) Architecture

Memory used as a sequential layer before attention.
Similar to H3/Mamba hybrid architectures.
"""

import torch
import torch.nn as nn  
from typing import Optional, Tuple, List

from .memory import NeuralMemory, MemoryState
from .attention import SlidingWindowAttention
from .persistent import PersistentMemory


class TitansMALBlock(nn.Module):
    """Block with memory layer followed by attention."""
    
    def __init__(self, d_model: int, n_heads: int = 8, n_persistent: int = 8,
                 window_size: int = 256, memory_depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.long_term_memory = NeuralMemory(d_model, memory_depth, chunk_size=64)
        self.persistent_memory = PersistentMemory(n_persistent, d_model)
        self.attention = SlidingWindowAttention(d_model, n_heads, window_size, dropout)
        
        self.norm_memory = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, memory_state: Optional[MemoryState] = None):
        batch_size = x.shape[0]
        if memory_state is None:
            memory_state = self.long_term_memory.memory_mlp.get_initial_state(batch_size, x.device)
        
        # Memory layer first
        memory_out, new_state = self.long_term_memory(x, memory_state, return_state=True)
        x = self.norm_memory(x + memory_out)
        
        # Then attention
        persistent = self.persistent_memory(batch_size)
        attn_out = self.attention(x, prefix=persistent)
        x = self.norm_attn(x + attn_out)
        
        # FFN
        x = self.norm_ffn(x + self.ffn(x))
        return x, new_state


class TitansMAL(nn.Module):
    """Full Titans MAL Model."""
    
    def __init__(self, d_model: int, n_layers: int = 12, n_heads: int = 8,
                 n_persistent: int = 8, window_size: int = 256,
                 memory_depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TitansMALBlock(d_model, n_heads, n_persistent, window_size, memory_depth, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, memory_states: Optional[List[MemoryState]] = None):
        if memory_states is None:
            memory_states = [None] * len(self.layers)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, memory_states[i])
            new_states.append(state)
        return self.final_norm(x), new_states
