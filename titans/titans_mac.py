"""
Titans MAC (Memory as Context) Architecture

In this variant, memory output is treated as additional context for attention.
The sequence is segmented and for each segment:
1. Query long-term memory with current segment to get historical context
2. Concatenate: [persistent_memory || long_term_memory_output || segment]
3. Apply full causal attention over the concatenated sequence
4. Update long-term memory with attention output

This design allows attention to decide what information from memory is useful,
and helps memory store only the most relevant information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .memory import NeuralMemory, MemoryState
from .attention import FullAttention
from .persistent import SegmentedPersistentMemory


class TitansMACBlock(nn.Module):
    """
    Single block of Titans MAC architecture.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_persistent: Number of persistent memory tokens
        segment_size: Size of each segment
        memory_depth: Depth of neural memory MLP
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_persistent: int = 8,
        segment_size: int = 512,
        memory_depth: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.segment_size = segment_size
        self.n_persistent = n_persistent
        
        # Long-term memory module
        self.long_term_memory = NeuralMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            chunk_size=64
        )
        
        # Persistent memory
        self.persistent_memory = SegmentedPersistentMemory(
            n_tokens=n_persistent,
            d_model=d_model
        )
        
        # Full attention for segment processing
        self.attention = FullAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        
        # Gating for memory-attention combination
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
    def process_segment(
        self,
        segment: torch.Tensor,
        memory_state: MemoryState,
        segment_idx: int
    ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Process a single segment with memory as context.
        
        Args:
            segment: Input segment of shape (batch, seg_len, d_model)
            memory_state: Current long-term memory state
            segment_idx: Index of current segment
            
        Returns:
            - Processed segment output
            - Updated memory state
        """
        batch_size, seg_len, _ = segment.shape
        
        # Get persistent memory
        persistent = self.persistent_memory(batch_size, segment_idx)
        
        # Query long-term memory with segment (retrieve historical context)
        # Use segment as query to get corresponding memories
        memory_output, _ = self.long_term_memory(
            segment,
            memory_state,
            return_state=False
        )
        memory_output = self.norm_memory(memory_output)
        
        # Attention over: [persistent || memory_output || segment]
        attn_output = self.attention(
            segment,
            persistent_memory=persistent,
            long_term_memory=memory_output
        )
        
        # Residual connection
        segment = segment + attn_output
        segment = self.norm1(segment)
        
        # Gated combination with memory for output
        gate_input = torch.cat([segment, memory_output], dim=-1)
        gate = self.memory_gate(gate_input)
        output = segment + gate * memory_output
        
        # Feedforward
        output = output + self.ffn(output)
        output = self.norm2(output)
        
        # Update memory with processed output
        _, new_memory_state = self.long_term_memory(
            output,
            memory_state,
            return_state=True
        )
        
        return output, new_memory_state
    
    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[MemoryState] = None
    ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Forward pass through MAC block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            memory_state: Optional initial memory state
            
        Returns:
            - Output tensor of shape (batch, seq_len, d_model)
            - Final memory state
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize memory if needed
        if memory_state is None:
            memory_state = self.long_term_memory.memory_mlp.get_initial_state(
                batch_size, device
            )
        
        # Segment the sequence
        n_segments = (seq_len + self.segment_size - 1) // self.segment_size
        outputs = []
        
        for i in range(n_segments):
            start = i * self.segment_size
            end = min(start + self.segment_size, seq_len)
            segment = x[:, start:end, :]
            
            # Process segment
            segment_output, memory_state = self.process_segment(
                segment, memory_state, i
            )
            outputs.append(segment_output)
        
        # Concatenate outputs
        output = torch.cat(outputs, dim=1)
        
        return output, memory_state


class TitansMAC(nn.Module):
    """
    Full Titans MAC (Memory as Context) Model.
    
    This stacks multiple MAC blocks to form a deep network.
    
    Args:
        d_model: Model dimension
        n_layers: Number of MAC blocks
        n_heads: Number of attention heads
        n_persistent: Number of persistent memory tokens
        segment_size: Size of each segment
        memory_depth: Depth of neural memory MLP
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 12,
        n_heads: int = 8,
        n_persistent: int = 8,
        segment_size: int = 512,
        memory_depth: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Stack of MAC blocks
        self.layers = nn.ModuleList([
            TitansMACBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_persistent=n_persistent,
                segment_size=segment_size,
                memory_depth=memory_depth,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        memory_states: Optional[List[MemoryState]] = None
    ) -> Tuple[torch.Tensor, List[MemoryState]]:
        """
        Forward pass through full MAC model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            memory_states: Optional list of memory states for each layer
            
        Returns:
            - Output tensor of shape (batch, seq_len, d_model)
            - List of final memory states for each layer
        """
        if memory_states is None:
            memory_states = [None] * self.n_layers
        
        new_memory_states = []
        
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, memory_states[i])
            new_memory_states.append(new_state)
        
        x = self.final_norm(x)
        
        return x, new_memory_states
