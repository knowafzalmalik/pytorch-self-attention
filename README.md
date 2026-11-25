# PyTorch Self-Attention

A clean, humanized implementation of **Single-Head** and **Multi-Head Self-Attention** in PyTorch. Every step is commented with both **what** it does (including tensor shapes) and **why** that design choice is needed.

## What's Inside

- `self_attention.py`
  - `SelfAttention`: Single-head scaled dot-product self-attention with optional masking and dropout.
  - `MultiHeadSelfAttention`: Parallel multi-head self-attention supporting both causal (autoregressive) masking and padding masks.
- `test_self_attention.py`: Unit tests covering output shapes, probability normalization, causal masking, padding masks, and gradient flow.
- `requirements.txt`: Minimal dependency list (`torch>=2.0.0`).

## Quick Start

```bash
pip install -r requirements.txt

# Run the demo
python self_attention.py

# Run unit tests
python -m unittest test_self_attention.py -v
```
