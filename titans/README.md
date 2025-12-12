# Titans: Learning to Memorize at Test Time

A PyTorch implementation of the **Titans** architecture from the paper:
> "Titans: Learning to Memorize at Test Time" by Behrouz, Zhong, and Mirrokni (Google Research, 2024)

## Overview

Titans introduces a **neural long-term memory module** that learns to memorize at test time, combined with attention mechanisms for powerful sequence modeling. The architecture features:

- **🧠 Neural Memory**: Deep MLP that learns to store key-value associations through gradient descent
- **⚡ Surprise Metric**: Momentum-based gradient updates capture both momentary and past surprise
- **🔄 Forgetting Mechanism**: Weight decay allows dynamic memory management
- **📦 Three Variants**: MAC, MAG, and MAL architectures for different use cases

## Architecture Variants

### MAC (Memory as Context)
Memory output serves as additional context for attention. Good for tasks requiring long-range dependencies with segment-based processing.

### MAG (Memory as Gate) 
Parallel branches of sliding window attention and neural memory combined via gating. Balanced approach for many tasks.

### MAL (Memory as Layer)
Memory used as a sequential layer before attention. Similar to hybrid architectures like H3/Mamba.

## Installation

```bash
pip install torch>=2.0
cd titans
pip install -e .
```

## Quick Start

```python
from titans import TitansLM, TitansConfig

# Create a Titans model
config = TitansConfig(
    vocab_size=32000,
    d_model=768,
    n_layers=12,
    n_heads=12,
    variant="mac",  # or "mag", "mal"
    memory_depth=2,
)

model = TitansLM(config).cuda()

# Forward pass
input_ids = torch.randint(0, 32000, (2, 1024)).cuda()
outputs = model(input_ids)
logits = outputs["logits"]
memory_states = outputs["memory_states"]

# Generation
generated = model.generate(input_ids[:, :10], max_new_tokens=100)
```

## Neural Memory Module

The core innovation - a learnable memory that updates through gradient descent:

```python
from titans import NeuralMemory

memory = NeuralMemory(
    d_model=512,
    memory_depth=2,  # MLP depth for expressiveness
    chunk_size=64,   # Chunk size for parallel training
)

# Process sequence (memory learns online)
output, state = memory(x)

# Continue with persistent memory state
output2, state2 = memory(x2, memory_state=state)
```

### Key Concepts

1. **Associative Memory Loss**: `ℓ(M; x_t) = ||M(k_t) - v_t||²`
2. **Surprise Metric**: Gradient of loss measures how "surprising" new data is
3. **Momentum**: Captures past surprise for smoother updates
4. **Weight Decay**: Implements forgetting for memory management

## Training

```bash
python -m titans.train \
    --variant mac \
    --d_model 512 \
    --n_layers 8 \
    --batch_size 8 \
    --max_steps 100000
```

## Comparison with Related Work

| Model | Memory Type | Forgetting | Deep Memory | Momentum |
|-------|-------------|------------|-------------|----------|
| Transformer | KV Cache | ✗ | ✗ | ✗ |
| Mamba | Linear State | ✓ (gate) | ✗ | ✗ |
| DeltaNet | Linear | ✗ | ✗ | ✗ |
| TTT | Gradient | ✗ | ✓ | ✗ |
| **Titans** | **Gradient** | **✓** | **✓** | **✓** |

## File Structure

```
titans/
├── __init__.py         # Package init
├── memory.py           # Neural long-term memory
├── attention.py        # Multi-head & sliding window attention
├── persistent.py       # Persistent memory module
├── titans_mac.py       # Memory as Context variant
├── titans_mag.py       # Memory as Gate variant  
├── titans_mal.py       # Memory as Layer variant
├── model.py            # Full language model
├── train.py            # Training script
└── examples.py         # Usage examples
```

## Citation

```bibtex
@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}
```

## License

MIT License
