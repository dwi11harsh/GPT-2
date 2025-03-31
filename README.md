# GPT-2 124M Replication Project

This repository contains implementations progressing from a simple bigram model to a transformer-based architecture similar to GPT-2. The current implementation approximates a smaller version of GPT-2 (~124M parameters when scaled up) using PyTorch.

## File Structure

### `bigram.py`

- Basic bigram language model
- Simple embedding layer implementation
- Training loop with evaluation
- Character-level text generation

### `gpt.py`

- Transformer-based language model
- Implements key GPT components:
  - Multi-head self-attention
  - Positional embeddings
  - Feed-forward networks
  - Layer normalization
  - Residual connections
- More sophisticated training setup

## Key Components

### Shared Infrastructure

1. **Data Handling**

   - Character-level tokenization
   - Train/validation split (90/10)
   - Batch generation with context windows

2. **Training Loop**
   - AdamW optimizer
   - Periodic evaluation
   - Loss estimation with multiple batches

### GPT-specific Components

1. **Transformer Block**

   - Multi-head attention (`MultiHeadAttention`)
   - Position-wise feed forward network (`FeedForward`)
   - Residual connections with layer norm

2. **Model Architecture**

   - Token and positional embeddings
   - Stacked transformer blocks
   - Final layer normalization
   - Language model head

3. **Optimizations**
   - Dropout regularization
   - Weight initialization (normal distribution)
   - Gradient clipping (implicit via AdamW)

## Hyperparameters

| Parameter           | `bigram.py` | `gpt.py` | GPT-2 124M Equivalent |
| ------------------- | ----------- | -------- | --------------------- |
| Batch size          | 32          | 64       | 512                   |
| Block size          | 8           | 256      | 1024                  |
| Embedding dimension | -           | 384      | 768                   |
| Number of heads     | -           | 6        | 12                    |
| Number of layers    | -           | 6        | 12                    |
| Dropout             | -           | 0.2      | 0.1                   |
| Learning rate       | 1e-2        | 3e-4     | 1.5e-4                |

## Usage

1. **Training**

```bash
python bigram.py  # Simple baseline
python gpt.py     # Transformer model
```
