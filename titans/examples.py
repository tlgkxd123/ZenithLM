"""
Example usage of Titans models.

Demonstrates:
- Creating models with different variants
- Forward pass and memory state handling  
- Text generation
- Memory retrieval patterns
"""

import torch
from titans import TitansLM, TitansConfig, NeuralMemory
from titans.model import create_titans_model


def demo_model_variants():
    """Show different Titans variants."""
    print("=" * 60)
    print("Titans Model Variants Demo")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    seq_len = 256
    
    for variant in ["mac", "mag", "mal"]:
        print(f"\n--- Titans {variant.upper()} ---")
        
        config = TitansConfig(
            vocab_size=1000,
            d_model=256,
            n_layers=4,
            n_heads=4,
            variant=variant,
            segment_size=64,  # For MAC
            window_size=64,   # For MAG/MAL
        )
        
        model = TitansLM(config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params/1e6:.2f}M")
        
        # Forward pass
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        outputs = model(input_ids)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Memory states: {len(outputs['memory_states'])} layers")


def demo_neural_memory():
    """Demonstrate the neural memory module."""
    print("\n" + "=" * 60)
    print("Neural Memory Demo")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create neural memory
    memory = NeuralMemory(
        d_model=128,
        memory_depth=2,
        chunk_size=32,
    ).to(device)
    
    print(f"\nMemory MLP layers: {memory.memory_mlp.n_layers}")
    print(f"Hidden dimension: {memory.memory_mlp.hidden_dim}")
    
    # Process a sequence
    x = torch.randn(2, 64, 128, device=device)
    output, state = memory(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Memory weights: {[w.shape for w in state.weights]}")
    print(f"Memory momentum: {[m.shape for m in state.momentum]}")
    
    # Continue with more data (memory persists)
    x2 = torch.randn(2, 32, 128, device=device)
    output2, state2 = memory(x2, memory_state=state)
    print(f"\nContinued with new input: {x2.shape}")
    print(f"Output: {output2.shape}")


def demo_generation():
    """Demonstrate text generation."""
    print("\n" + "=" * 60)
    print("Text Generation Demo")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a small model
    model = create_titans_model(variant="mag", size="tiny").to(device)
    model.eval()
    
    # Generate from a prompt
    prompt = torch.randint(0, 100, (1, 10), device=device)
    print(f"Prompt tokens: {prompt.tolist()[0]}")
    
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50
    )
    
    print(f"Generated tokens: {generated.tolist()[0]}")


def demo_memory_comparison():
    """Compare shallow vs deep memory."""
    print("\n" + "=" * 60)
    print("Memory Depth Comparison")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for depth in [1, 2, 3]:
        memory = NeuralMemory(
            d_model=128,
            memory_depth=depth,
            chunk_size=32,
        ).to(device)
        
        n_params = sum(p.numel() for p in memory.parameters())
        x = torch.randn(1, 128, 128, device=device)
        output, _ = memory(x)
        
        print(f"Depth {depth}: {n_params:,} params, output norm: {output.norm():.4f}")


if __name__ == "__main__":
    demo_model_variants()
    demo_neural_memory()
    demo_generation()
    demo_memory_comparison()
