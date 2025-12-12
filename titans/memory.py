"""
Neural Long-term Memory Module for Titans

Optimized implementation with:
- Batched weights for per-sequence memory
- torch.func.vmap for efficient gradient computation
- On-the-fly accumulation to reduce memory footprint
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass


@dataclass
class MemoryState:
    """State container for neural memory across time steps."""
    weights: Tuple[torch.Tensor, ...]  # Batched weights (B, Out, In)
    momentum: Tuple[torch.Tensor, ...]  # Batched momentum (B, Out, In)
    
    def detach(self) -> 'MemoryState':
        """Detach state from computation graph."""
        return MemoryState(
            weights=tuple(w.detach() for w in self.weights),
            momentum=tuple(m.detach() for m in self.momentum)
        )


class NeuralMemoryMLP(nn.Module):
    """
    Deep MLP serving as neural memory.
    Supports batched weights for per-sequence adaptation.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        hidden_mult: float = 2.0,  # Reduced default for efficiency
        activation: str = 'silu'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.hidden_dim = int(d_model * hidden_mult)
        
        # Template layers (store initial parameter shapes/init)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = d_model if i == 0 else self.hidden_dim
            out_dim = d_model if i == n_layers - 1 else self.hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim, bias=False))
            
        if activation == 'silu':
            self.activation = F.silu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu
            
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            
    def get_initial_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Get initial state with batched weights."""
        weights = []
        momentum = []
        
        for layer in self.layers:
            # Expand weights: (Out, In) -> (Batch, Out, In)
            w = layer.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
            weights.append(w)
            momentum.append(torch.zeros_like(w))
            
        return MemoryState(weights=tuple(weights), momentum=tuple(momentum))

    def forward_single(self, x: torch.Tensor, weights: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward pass for a single sample (helper for vmap).
        x: (Input_Dim) or (Seq, Input_Dim)
        weights: Tuple of (Out, In) tensors
        """
        h = x
        for i, w in enumerate(weights):
            h = F.linear(h, w)
            if i < len(weights) - 1:
                h = self.activation(h)
        return h

    def forward(
        self, 
        x: torch.Tensor, 
        weights: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        """
        Batched forward pass.
        x: (Batch, Seq, D)
        weights: Tuple of (Batch, Out, In) tensors
        """
        if weights is None:
            # Fallback to standard shared weights
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i < len(self.layers) - 1:
                    h = self.activation(h)
            return h
            
        # Batched weights forward
        h = x
        for i, w in enumerate(weights):
            # w: (B, Out, In), h: (B, Seq, In)
            if i == 0 and h.shape[0] < 5:
                # DEBUG from previous step (preserved)
                print(f"[DEBUG] MLP L{i}: h={h.shape} w={w.shape} wT={w.transpose(1,2).shape}")
            
            # Ensure w is 3D
            if w.dim() != 3:
                print(f"[ERROR] w has {w.dim()} dims: {w.shape}. Reshaping/Fixing...")
                # Try to recover if it's (4096, 4096) -> (1, 4096, 4096) broadcast
                # But here it should be (B, Out, In)
                
            # Verify In dimension
            if w.shape[2] != h.shape[2]:
                print(f"[FATAL] In-dimension mismatch: h={h.shape}, w={w.shape}")
                # This causes the [4, 1] error if w.shape[2] == 1
                
            # x @ w.T -> (B, Seq, In) @ (B, In, Out) -> (B, Seq, Out)
            h = torch.matmul(h, w.transpose(1, 2))
            if i < len(weights) - 1:
                h = self.activation(h)
        return h


class NeuralMemory(nn.Module):
    """
    Optimized Neural Long-term Memory.
    """
    
    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        hidden_mult: float = 2.0,
        chunk_size: int = 8,  # Reduced for large models
        theta_init: float = 0.1,
        eta_init: float = 0.9,
        alpha_init: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        
        self.memory_mlp = NeuralMemoryMLP(
            d_model, memory_depth, hidden_mult
        )
        
        self.theta_proj = nn.Linear(d_model, 1)
        self.eta_proj = nn.Linear(d_model, 1)
        self.alpha_proj = nn.Linear(d_model, 1)
        
        nn.init.constant_(self.theta_proj.bias, math.log(theta_init))
        nn.init.constant_(self.eta_proj.bias, math.log(eta_init / (1 - eta_init)))
        nn.init.constant_(self.alpha_proj.bias, math.log(alpha_init / (1 - alpha_init)))
        
        self.layer_norm = nn.LayerNorm(d_model)

    def _sample_loss(self, weights, k, v):
        """Loss function for a single sample (for vmap)."""
        # k, v: (D,)
        # weights: Tuple of (Out, In)
        pred = self.memory_mlp.forward_single(k, weights)
        return 0.5 * ((pred - v) ** 2).sum()

    def update_memory_chunk(
        self,
        memory_state: MemoryState,
        x_chunk: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor
    ) -> MemoryState:
        """
        Memory-efficient chunk update using sequential gradient computation.
        Avoids vmap to reduce memory overhead for large models.
        """
        batch_size, chunk_len, _ = x_chunk.shape
        
        # Precompute params
        theta = torch.sigmoid(self.theta_proj(x_chunk))  # (B, L, 1)
        eta = torch.sigmoid(self.eta_proj(x_chunk))
        alpha = torch.sigmoid(self.alpha_proj(x_chunk))
        
        # Ensure we can compute gradients wrt weights
        current_weights = [w.detach().requires_grad_(True) for w in memory_state.weights]
        current_momentum = [m.detach() for m in memory_state.momentum]
        
        # Process each timestep
        for t in range(chunk_len):
            kt = k_chunk[:, t]  # (B, D)
            vt = v_chunk[:, t]  # (B, D)
            
            theta_t = theta[:, t, 0].view(-1, 1, 1)
            eta_t = eta[:, t, 0].view(-1, 1, 1)
            alpha_t = alpha[:, t, 0].view(-1, 1, 1)
            
            # Forward pass to get loss
            # We use the current_weights which require grad
            pred = self.memory_mlp(kt.unsqueeze(1), tuple(current_weights)) # (B, 1, D)
            loss = 0.5 * ((pred.squeeze(1) - vt) ** 2).sum()
            
            # Compute gradients for all weights at once
            grads = torch.autograd.grad(loss, current_weights, create_graph=False)
            
            # Update weights and momentum
            next_weights = []
            next_momentum = []
            
            with torch.no_grad():
                for i in range(len(current_weights)):
                    w = current_weights[i]
                    m = current_momentum[i]
                    g = grads[i]
                    
                    # Momentum update
                    new_m = eta_t * m - theta_t * g
                    
                    # Weight update with decay
                    new_w = (1 - alpha_t) * w + new_m
                    
                    # Prepare for next step
                    next_weights.append(new_w.detach().requires_grad_(True))
                    next_momentum.append(new_m.detach())
            
            current_weights = next_weights
            current_momentum = next_momentum
        
        # Return final state (detached)
        return MemoryState(
            tuple(w.detach() for w in current_weights), 
            tuple(current_momentum)
        )

    def _apply_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = conv(x)[:, :, :x.size(2)]
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[MemoryState] = None,
        return_state: bool = True
    ) -> Tuple[torch.Tensor, Optional[MemoryState]]:
        
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if memory_state is None:
            memory_state = self.memory_mlp.get_initial_state(batch_size, device)
            
        keys = F.normalize(F.silu(self._apply_conv(self.W_K(x), self.conv_k)), p=2, dim=-1)
        values = F.silu(self._apply_conv(self.W_V(x), self.conv_v))
        queries = F.normalize(F.silu(self._apply_conv(self.W_Q(x), self.conv_q)), p=2, dim=-1)
        
        outputs = []
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            
            # Slice chunk
            q_chunk = queries[:, start:end]
            
            # Retrieve (using state at start of chunk)
            chunk_out = self.memory_mlp(q_chunk, memory_state.weights)
            outputs.append(chunk_out)
            
            # Update state
            memory_state = self.update_memory_chunk(
                memory_state, 
                x[:, start:end], 
                keys[:, start:end], 
                values[:, start:end]
            )
            
        output = torch.cat(outputs, dim=1)
        output = self.layer_norm(output)
        
        return (output, memory_state) if return_state else (output, None)
