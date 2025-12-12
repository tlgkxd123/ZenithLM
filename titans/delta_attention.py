"""
Flash-Delta Hybrid Attention

Combines:
- Flash Attention: IO-efficient softmax attention for local context
- Delta Attention: Linear recurrent attention with delta rule for long-range memory

Delta rule: Before adding a new KV association, remove the old value first.
This prevents memory overflow and allows updating associations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DeltaState:
    """State for Delta Attention recurrence."""
    S: torch.Tensor  # (B, H, D, D) - matrix-valued memory
    z: torch.Tensor  # (B, H, D) - normalization term
    
    def detach(self) -> 'DeltaState':
        return DeltaState(S=self.S.detach(), z=self.z.detach())


class DeltaAttention(nn.Module):
    """
    Delta Attention with delta rule memory update.
    
    Update rule:
        S_t = S_{t-1} - β_t * (S_{t-1} @ k_t) @ k_t^T + β_t * v_t @ k_t^T
        
    This removes the old value associated with k_t before adding new v_t.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Data-dependent beta gate
        self.beta_proj = nn.Linear(d_model, n_heads)
        
        # 1D convolutions for local context
        self.conv_q = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model)
        self.conv_k = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
        
    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
        nn.init.zeros_(self.beta_proj.bias)
    
    def get_initial_state(self, batch_size: int, device: torch.device) -> DeltaState:
        """Initialize delta state."""
        S = torch.zeros(batch_size, self.n_heads, self.head_dim, self.head_dim, device=device)
        z = torch.zeros(batch_size, self.n_heads, self.head_dim, device=device)
        return DeltaState(S=S, z=z)
    
    def _apply_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = conv(x)[:, :, :x.size(2)]
        return x.transpose(1, 2)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[DeltaState] = None,
        return_state: bool = True
    ) -> Tuple[torch.Tensor, Optional[DeltaState]]:
        """
        Forward pass with delta rule recurrence.
        
        Args:
            x: (B, L, D)
            state: Optional initial state
            return_state: Whether to return final state
        """
        B, L, D = x.shape
        H, HD = self.n_heads, self.head_dim
        
        if state is None:
            state = self.get_initial_state(B, x.device)
        
        # Projections with conv
        q = self._apply_conv(self.q_proj(x), self.conv_q)
        k = self._apply_conv(self.k_proj(x), self.conv_k)
        v = self.v_proj(x)
        
        # Reshape to heads: (B, L, H, HD)
        q = q.view(B, L, H, HD)
        k = k.view(B, L, H, HD)
        v = v.view(B, L, H, HD)
        
        # Normalize keys
        k = F.normalize(k, p=2, dim=-1)
        
        # Beta gate: (B, L, H)
        beta = torch.sigmoid(self.beta_proj(x))
        
        # Recurrence over sequence
        S = state.S  # (B, H, HD, HD)
        z = state.z  # (B, H, HD)
        
        outputs = []
        
        for t in range(L):
            q_t = q[:, t]  # (B, H, HD)
            k_t = k[:, t]  # (B, H, HD)
            v_t = v[:, t]  # (B, H, HD)
            beta_t = beta[:, t, :, None]  # (B, H, 1)
            
            # Retrieve: y_t = S @ q_t / (z @ q_t + eps)
            y_t = torch.einsum('bhij,bhj->bhi', S, q_t)
            denom = torch.einsum('bhi,bhi->bh', z, q_t).unsqueeze(-1) + 1e-6
            y_t = y_t / denom
            
            outputs.append(y_t)
            
            # Delta update: remove old, add new
            # old_val = S @ k_t
            old_val = torch.einsum('bhij,bhj->bhi', S, k_t)  # (B, H, HD)
            
            # S = S - beta * old_val @ k_t^T + beta * v_t @ k_t^T
            delta = v_t - old_val  # (B, H, HD)
            S = S + beta_t.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', delta, k_t)
            
            # Update normalization
            z = z + beta_t * k_t
        
        # Stack outputs: (B, L, H, HD)
        output = torch.stack(outputs, dim=1)
        output = output.reshape(B, L, D)
        output = self.out_proj(output)
        output = self.layer_norm(output)
        
        new_state = DeltaState(S=S, z=z) if return_state else None
        return output, new_state


class FlashDeltaHybrid(nn.Module):
    """
    Hybrid attention combining Flash Attention (local) and Delta Attention (global).
    
    - Flash branch: Precise local attention within window
    - Delta branch: Efficient long-range memory with delta rule
    - Learned gating combines both outputs
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.0,
        gate_bias: float = 0.0  # Positive = favor flash, negative = favor delta
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # Shared QKV for flash (efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Delta branch
        self.delta_attn = DeltaAttention(d_model, n_heads, dropout)
        
        # Learned gate
        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate_proj.bias, gate_bias)
        
        # Norms
        self.flash_norm = nn.LayerNorm(d_model)
        self.delta_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def _sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal sliding window mask."""
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)  # Causal
        mask = mask | torch.tril(torch.ones_like(mask), diagonal=-self.window_size)  # Window
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        delta_state: Optional[DeltaState] = None,
        return_state: bool = True
    ) -> Tuple[torch.Tensor, Optional[DeltaState]]:
        """
        Forward pass through hybrid attention.
        
        Args:
            x: (B, L, D)
            delta_state: Optional delta state for recurrence
            return_state: Whether to return delta state
        """
        B, L, D = x.shape
        H, HD = self.n_heads, self.head_dim
        
        # === Flash Attention Branch ===
        qkv = self.qkv_proj(x).reshape(B, L, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, HD)
        
        # Try Flash Attention first
        if hasattr(F, 'scaled_dot_product_attention') and L <= 2048:
            # Use flash attention with causal mask
            flash_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0
            )
        else:
            # Fallback to manual with sliding window
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = self._sliding_window_mask(L, x.device)
            attn.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)
            flash_out = torch.matmul(attn, v)
        
        flash_out = flash_out.transpose(1, 2).reshape(B, L, D)
        flash_out = self.out_proj(flash_out)
        flash_out = self.flash_norm(flash_out)
        
        # === Delta Attention Branch ===
        delta_out, new_state = self.delta_attn(x, delta_state, return_state)
        delta_out = self.delta_norm(delta_out)
        
        # === Gated Combination ===
        gate = torch.sigmoid(self.gate_proj(x))
        output = gate * flash_out + (1 - gate) * delta_out
        
        return output, new_state


class ChunkedFlashDelta(nn.Module):
    """
    Segment-based Flash-Delta for very long sequences.
    
    Within each chunk: Flash attention (full local)
    Across chunks: Delta state carries memory
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        chunk_size: int = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.hybrid = FlashDeltaHybrid(d_model, n_heads, chunk_size, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        delta_state: Optional[DeltaState] = None
    ) -> Tuple[torch.Tensor, DeltaState]:
        """Process sequence in chunks."""
        B, L, D = x.shape
        
        outputs = []
        n_chunks = (L + self.chunk_size - 1) // self.chunk_size
        
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, L)
            chunk = x[:, start:end]
            
            chunk_out, delta_state = self.hybrid(chunk, delta_state, return_state=True)
            outputs.append(chunk_out)
        
        return torch.cat(outputs, dim=1), delta_state
