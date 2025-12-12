"""
Persistent Memory Module for Titans

Persistent memory consists of learnable, input-independent parameters
that encode task-related knowledge. These are prepended to the sequence
before attention and serve several purposes:

1. Memory Perspective: Store task knowledge that shouldn't change with input
2. FFN Perspective: Act like data-independent attention weights
3. Technical Perspective: Mitigate attention sink issues with initial tokens
"""

import torch
import torch.nn as nn
from typing import Optional


class PersistentMemory(nn.Module):
    """
    Persistent (Meta) Memory Module.
    
    Learnable parameters that are prepended to sequences to provide
    stable, input-independent memory tokens.
    
    Args:
        n_tokens: Number of persistent memory tokens
        d_model: Model dimension
        init_std: Standard deviation for initialization
    """
    
    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        init_std: float = 0.02
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        
        # Learnable persistent memory parameters
        self.memory = nn.Parameter(
            torch.randn(1, n_tokens, d_model) * init_std
        )
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get persistent memory tokens expanded for batch.
        
        Args:
            batch_size: Current batch size
            
        Returns:
            Persistent memory of shape (batch_size, n_tokens, d_model)
        """
        return self.memory.expand(batch_size, -1, -1)
    
    def prepend_to_sequence(
        self,
        x: torch.Tensor,
        additional_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prepend persistent memory to a sequence.
        
        Args:
            x: Input sequence of shape (batch, seq_len, d_model)
            additional_context: Optional additional context to insert
                               (e.g., long-term memory output)
        
        Returns:
            Sequence with persistent memory prepended:
            [persistent_memory || additional_context || x]
        """
        batch_size = x.size(0)
        persistent = self.forward(batch_size)
        
        if additional_context is not None:
            return torch.cat([persistent, additional_context, x], dim=1)
        return torch.cat([persistent, x], dim=1)
    
    def extract_from_output(
        self,
        output: torch.Tensor,
        original_seq_len: int
    ) -> torch.Tensor:
        """
        Extract the relevant output portion after attention with persistent memory.
        
        Args:
            output: Full output including persistent memory positions
            original_seq_len: Length of original sequence (without persistent memory)
            
        Returns:
            Output corresponding to original sequence positions
        """
        return output[:, -original_seq_len:, :]


class SegmentedPersistentMemory(nn.Module):
    """
    Persistent memory for segment-based processing (MAC variant).
    
    Provides persistent memory tokens that are shared across all segments
    but processed independently in each segment's attention.
    
    Args:
        n_tokens: Number of persistent memory tokens
        d_model: Model dimension
        n_segments: Expected number of segments (for optional segment-specific params)
    """
    
    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        n_segments: Optional[int] = None
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        
        # Shared persistent memory
        self.memory = nn.Parameter(
            torch.randn(1, n_tokens, d_model) * 0.02
        )
        
        # Optional segment-specific scaling
        self.n_segments = n_segments
        if n_segments is not None:
            self.segment_scale = nn.Parameter(torch.ones(n_segments, 1, 1))
    
    def forward(
        self,
        batch_size: int,
        segment_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get persistent memory for a segment.
        
        Args:
            batch_size: Current batch size
            segment_idx: Optional segment index for segment-specific scaling
            
        Returns:
            Persistent memory of shape (batch, n_tokens, d_model)
        """
        memory = self.memory.expand(batch_size, -1, -1)
        
        if segment_idx is not None and self.n_segments is not None:
            memory = memory * self.segment_scale[segment_idx]
        
        return memory
