"""
Complete Titans Language Model

Full language model with embedding, backbone, and output layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .titans_mac import TitansMAC
from .titans_mag import TitansMAG
from .titans_mal import TitansMAL
from .memory import MemoryState


@dataclass
class TitansConfig:
    """Configuration for Titans models."""
    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 11
    n_heads: int = 16
    n_persistent: int = 8
    segment_size: int = 512  # For MAC
    window_size: int = 256   # For MAG/MAL
    memory_depth: int = 2
    dropout: float = 0.1
    max_seq_len: int = 8192
    variant: str = "mac"  # "mac", "mag", or "mal"
    tie_embeddings: bool = True


class TitansLM(nn.Module):
    """
    Titans Language Model.
    
    Complete LM with token embeddings, positional encoding,
    one of the Titans backbones, and output projection.
    """
    
    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding (RoPE-style or learned)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Select backbone variant
        if config.variant == "mac":
            self.backbone = TitansMAC(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                n_persistent=config.n_persistent,
                segment_size=config.segment_size,
                memory_depth=config.memory_depth,
                dropout=config.dropout
            )
        elif config.variant == "mag":
            self.backbone = TitansMAG(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                n_persistent=config.n_persistent,
                window_size=config.window_size,
                memory_depth=config.memory_depth,
                dropout=config.dropout
            )
        elif config.variant == "mal":
            self.backbone = TitansMAL(
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                n_persistent=config.n_persistent,
                window_size=config.window_size,
                memory_depth=config.memory_depth,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown variant: {config.variant}")
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings
        if config.tie_embeddings:
            self.output_proj.weight = self.token_embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: Optional[List[MemoryState]] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Forward pass for language modeling."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        
        # Backbone
        x, new_memory_states = self.backbone(x, memory_states)
        
        # Output logits
        logits = self.output_proj(x)
        
        result = {"logits": logits, "memory_states": new_memory_states}
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            result["loss"] = loss
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        memory_states = None
        
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids, memory_states)
            logits = outputs["logits"][:, -1, :] / temperature
            memory_states = outputs["memory_states"]
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def create_titans_model(variant: str = "mac", size: str = "300M") -> TitansLM:
    """Factory function to create Titans models."""
    sizes = {
        "tiny": dict(d_model=256, n_layers=6, n_heads=4),
        "small": dict(d_model=512, n_layers=8, n_heads=8),
        "medium": dict(d_model=768, n_layers=12, n_heads=12),
        "300M": dict(d_model=1024, n_layers=11, n_heads=16),
        "large": dict(d_model=1024, n_layers=24, n_heads=16),
    }
    
    config = TitansConfig(variant=variant, **sizes.get(size, sizes["300M"]))
    return TitansLM(config)
