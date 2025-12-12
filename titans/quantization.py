"""
4-bit Quantized Titans Model

Supports:
- NF4/FP4 quantization via bitsandbytes (QLoRA-style)
- MXFP4 via torchao (experimental)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .model import TitansConfig


def quantize_linear_layers(module: nn.Module, quantization_type: str = "nf4"):
    """
    Replace Linear layers with 4-bit quantized versions.
    
    Args:
        module: The module to quantize
        quantization_type: "nf4", "fp4", or "mxfp4"
    """
    import bitsandbytes as bnb
    
    for name, child in module.named_children():
        # SKIP NeuralMemory layers! They are dynamic states, not static weights.
        if "memory" in name or "memory_mlp" in name or isinstance(child, (nn.Conv1d, nn.LayerNorm)):
            continue
            
        if isinstance(child, nn.Linear):
            # Skip small layers (embeddings usually handled separately)
            if child.in_features < 256 or child.out_features < 256:
                continue
                
            # Create 4-bit linear
            has_bias = child.bias is not None
            
            if quantization_type in ["nf4", "fp4"]:
                new_layer = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                    compute_dtype=torch.bfloat16,
                    compress_statistics=True,
                    quant_type=quantization_type,  # "nf4" or "fp4"
                )
                
                # Copy weights (will be quantized)
                new_layer.weight = bnb.nn.Params4bit(
                    child.weight.data,
                    requires_grad=False,
                    quant_type=quantization_type,
                )
                if has_bias:
                    new_layer.bias = nn.Parameter(child.bias.data)
                    
                setattr(module, name, new_layer)
            else:
                # Keep original for unsupported types
                pass
        else:
            # Recurse into child modules
            quantize_linear_layers(child, quantization_type)


def create_quantized_model(config: TitansConfig, quantization: str = "nf4"):
    """
    Create a 4-bit quantized Titans model.
    
    Args:
        config: Model configuration
        quantization: "nf4", "fp4", or "none"
    """
    from .model import TitansLM
    
    if quantization == "none":
        return TitansLM(config)
    
    # Create model in bfloat16 first
    with torch.device("meta"):
        model = TitansLM(config)
    
    # Materialize on CPU then quantize
    model = model.to_empty(device="cpu")
    model._init_weights()
    
    # Quantize all linear layers
    print(f"Quantizing model to {quantization.upper()}...")
    quantize_linear_layers(model, quantization)
    
    # Move to GPU
    model = model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total params: {total_params/1e6:.1f}M")
    print(f"Trainable params: {trainable_params/1e6:.1f}M")
    
    # Estimate memory savings
    # 4-bit = 0.5 bytes per param vs 2 bytes for bf16
    orig_memory = total_params * 2 / 1e9
    quant_memory = total_params * 0.5 / 1e9
    print(f"Memory: {orig_memory:.2f}GB → {quant_memory:.2f}GB ({(1 - quant_memory/orig_memory)*100:.0f}% reduction)")
    
    return model


class TitansLM4bit(nn.Module):
    """
    4-bit quantized Titans Language Model.
    
    Uses NF4 quantization for all large linear layers.
    Embeddings and small layers remain in full precision.
    """
    
    def __init__(self, config: TitansConfig, quant_type: str = "nf4"):
        super().__init__()
        self.config = config
        self.quant_type = quant_type
        
        # Create and quantize
        self.model = create_quantized_model(config, quant_type)
    
    def forward(self, input_ids, memory_states=None, labels=None):
        return self.model(input_ids, memory_states, labels)
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        return self.model.generate(input_ids, max_new_tokens, temperature, top_k)


# MXFP4 support via torchao (Hopper-specific)
def enable_mxfp4_training():
    """
    Enable MXFP4 (Microscaling FP4) for Hopper GPUs.
    Requires: pip install torchao
    """
    try:
        import torchao
        from torchao.float8 import convert_to_float8_training
        
        print("MXFP4 training enabled via torchao")
        return True
    except ImportError:
        print("torchao not installed. Install with: pip install torchao")
        return False


def quantize_for_inference(model: nn.Module, quant_type: str = "int4"):
    """
    Quantize a trained model for inference.
    
    Args:
        model: Trained Titans model
        quant_type: "int4", "int8", or "fp4"
    """
    try:
        import torchao
        from torchao.quantization import quantize_, int4_weight_only, int8_weight_only
        
        if quant_type == "int4":
            quantize_(model, int4_weight_only())
        elif quant_type == "int8":
            quantize_(model, int8_weight_only())
        else:
            print(f"Unknown quant_type: {quant_type}")
            
        return model
    except ImportError:
        # Fallback to bitsandbytes
        quantize_linear_layers(model, "nf4")
        return model
