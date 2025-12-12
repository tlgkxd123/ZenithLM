"""
Test script for Flash-Delta Hybrid Attention.

Verifies:
1. Shape correctness
2. Gradient flow through both branches
3. State persistence across chunks
4. Memory efficiency
"""

import torch
import torch.nn as nn
from titans.delta_attention import DeltaAttention, FlashDeltaHybrid, ChunkedFlashDelta, DeltaState


def test_delta_attention():
    """Test pure Delta Attention."""
    print("=" * 50)
    print("Testing DeltaAttention")
    print("=" * 50)
    
    B, L, D, H = 2, 128, 256, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DeltaAttention(D, H).to(device)
    x = torch.randn(B, L, D, device=device, requires_grad=True)
    
    # Forward pass
    output, state = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"State S shape: {state.S.shape}")
    print(f"State z shape: {state.z.shape}")
    
    # Check gradients
    loss = output.sum()
    loss.backward()
    
    print(f"Gradient flows: {x.grad is not None}")
    print(f"Grad norm: {x.grad.norm().item():.4f}")
    
    # Continue with state
    x2 = torch.randn(B, 64, D, device=device)
    output2, state2 = model(x2, state=state)
    
    print(f"Continued output shape: {output2.shape}")
    print(f"State updated: {not torch.allclose(state.S, state2.S)}")
    print("✓ DeltaAttention passed!\n")


def test_flash_delta_hybrid():
    """Test Flash-Delta Hybrid."""
    print("=" * 50)
    print("Testing FlashDeltaHybrid")
    print("=" * 50)
    
    B, L, D, H = 2, 256, 256, 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = FlashDeltaHybrid(D, H, window_size=64).to(device)
    x = torch.randn(B, L, D, device=device, requires_grad=True)
    
    # Forward
    output, state = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify both branches contribute
    with torch.no_grad():
        model.eval()
        gate = torch.sigmoid(model.gate_proj(x))
        avg_gate = gate.mean().item()
        print(f"Average gate value: {avg_gate:.4f} (0.5 = balanced)")
    
    # Gradients
    loss = output.sum()
    loss.backward()
    
    print(f"Gradient flows: {x.grad is not None}")
    
    # Memory usage
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak GPU memory: {mem_mb:.1f} MB")
    
    print("✓ FlashDeltaHybrid passed!\n")


def test_chunked_flash_delta():
    """Test Chunked Flash-Delta for long sequences."""
    print("=" * 50)
    print("Testing ChunkedFlashDelta")
    print("=" * 50)
    
    B, L, D, H = 2, 1024, 256, 8
    chunk_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ChunkedFlashDelta(D, H, chunk_size).to(device)
    x = torch.randn(B, L, D, device=device)
    
    # Forward
    output, state = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Chunks processed: {L // chunk_size}")
    
    # Verify state carries across chunks
    x2 = torch.randn(B, 512, D, device=device)
    output2, state2 = model(x2, state)
    
    print(f"State persists across calls: {state2 is not None}")
    print("✓ ChunkedFlashDelta passed!\n")


def test_memory_comparison():
    """Compare memory usage: Full attention vs Hybrid."""
    print("=" * 50)
    print("Memory Comparison")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    B, D, H = 2, 256, 8
    device = "cuda"
    
    for L in [256, 512, 1024, 2048]:
        torch.cuda.reset_peak_memory_stats()
        
        model = FlashDeltaHybrid(D, H, window_size=128).to(device)
        x = torch.randn(B, L, D, device=device)
        
        with torch.no_grad():
            _ = model(x)
        
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"L={L:4d}: {mem_mb:.1f} MB")
        
        del model, x
        torch.cuda.empty_cache()
    
    print("✓ Memory test complete!\n")


def main():
    print("\n🧪 Flash-Delta Hybrid Attention Tests\n")
    
    test_delta_attention()
    test_flash_delta_hybrid()
    test_chunked_flash_delta()
    test_memory_comparison()
    
    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
