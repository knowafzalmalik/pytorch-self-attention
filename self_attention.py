"""
Humanized PyTorch implementation of Single-Head and Multi-Head Self-Attention.

Each section explains both *what* the code is doing (including tensor shapes)
and *why* that design choice exists in the Transformer architecture.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention.

    What it does:
        Lets every token in a sequence look at every other token (including itself)
        and build a new representation weighted by relevance.

    Why we need it:
        A raw token embedding only knows about itself. To understand context
        (for example, what "it" refers to in a sentence), each token needs a way
        to gather information from the rest of the sequence.
    """

    def __init__(self, embed_dim: int, head_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim or embed_dim

        # What: Three separate linear projections for Query, Key, and Value.
        # Why:  If we compared raw embeddings directly, a token's "search criteria"
        #       would be forced to match its "advertised identity" and its "payload".
        #       Separate Q, K, V matrices let each token ask one question (Q),
        #       advertise another (K), and pass along different content (V).
        #       bias=False is common here because LayerNorm/centering handles offsets.
        self.q_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=False)

        # What: Project the attended representation back to embed_dim.
        # Why:  Keeps input and output dimensions identical so residual connections
        #       (x + attention(x)) work cleanly, especially when head_dim != embed_dim.
        self.out_proj = nn.Linear(self.head_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            mask: Optional boolean mask broadcastable to (batch_size, seq_len, seq_len).
                  Positions with False are masked out (ignored).

        Returns:
            output: Context-aware tensor of shape (batch_size, seq_len, embed_dim).
            attn_weights: Attention probabilities of shape (batch_size, seq_len, seq_len).
        """
        # What: Project x into Queries, Keys, and Values.
        #       Shapes: (B, T, embed_dim) -> (B, T, head_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # What: Compute raw similarity scores between every Query and every Key.
        #       (B, T, head_dim) @ (B, head_dim, T) -> (B, T, T)
        # Why:  A dot product is high when Q and K point in a similar direction,
        #       giving us a fast, differentiable measure of token-to-token relevance.
        scores = torch.matmul(q, k.transpose(-2, -1))

        # What: Scale scores by 1 / sqrt(head_dim).
        # Why:  As head_dim grows, dot products grow in variance (proportional to head_dim).
        #       Large scores push softmax into flat regions where gradients almost vanish.
        #       Dividing by sqrt(head_dim) restores unit variance so training stays stable.
        scale = math.sqrt(self.head_dim)
        scores = scores / scale

        # What: Replace disallowed positions (e.g., future tokens or padding) with -inf.
        # Why:  softmax(-inf) == 0, which cleanly removes those tokens from the weighted
        #       sum while still ensuring the remaining weights sum to 1.0.
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # What: Convert raw scores into a probability distribution over keys.
        #       Shape remains (B, T, T), where each row sums to 1.
        # Why:  Probabilities turn arbitrary similarity scores into safe, interpretable
        #       mixing weights for the Value vectors.
        attn_weights = F.softmax(scores, dim=-1)

        # What: Randomly zero out some attention links during training.
        # Why:  Prevents the model from over-relying on a single dominant token and
        #       encourages it to use broader context.
        attn_weights_dropped = self.dropout(attn_weights)

        # What: Take a weighted sum of Values using the attention probabilities.
        #       (B, T, T) @ (B, T, head_dim) -> (B, T, head_dim)
        context = torch.matmul(attn_weights_dropped, v)

        # What: Mix the attended features back into the model's embedding space.
        #       (B, T, head_dim) -> (B, T, embed_dim)
        output = self.out_proj(context)
        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head scaled dot-product self-attention.

    What it does:
        Splits the embedding dimension into `num_heads` smaller subspaces, runs
        self-attention in parallel on each subspace, and merges the results.

    Why we need it:
        A single attention head has to average all relationships into one attention
        map. Multiple heads let the model track different kinds of relationships
        at the same time (e.g., one head for grammar, one for coreference, one
        for nearby words).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        # What: One fused linear layer that produces Q, K, and V together.
        # Why:  Computing one (embed_dim -> 3 * embed_dim) matrix multiply is faster
        #       on GPUs/CPUs than three separate smaller matrix multiplies.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # What: Final linear projection after concatenating all heads.
        # Why:  Lets information discovered by different heads interact and combine
        #       before passing to the next layer.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        What: Creates a lower-triangular boolean matrix of shape (1, 1, seq_len, seq_len).
        Why:  In autoregressive models (like GPT), token `i` may only attend to
              tokens `0..i`. Masking out `j > i` prevents the model from cheating
              by peeking at future tokens during training.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Optional boolean mask of shape (batch_size, seq_len)
                            where True = valid token, False = padding token.

        Returns:
            output: Tensor of shape (batch_size, seq_len, embed_dim).
            attn_weights: Attention probabilities of shape
                          (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, _ = x.shape

        # What: Project to QKV and split into 3 tensors of shape (B, T, embed_dim).
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # What: Reshape from (B, T, embed_dim) -> (B, num_heads, T, head_dim).
        # Why:  Moving num_heads before seq_len lets torch.matmul run attention
        #       across all heads and batches in a single batched operation.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # What: Scaled dot-product scores of shape (B, num_heads, T, T).
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # What: Combine causal mask (if enabled) and padding mask (if provided).
        combined_mask: Optional[torch.Tensor] = None
        if self.causal:
            combined_mask = self._build_causal_mask(seq_len, x.device)

        if attention_mask is not None:
            # Expand (B, T) -> (B, 1, 1, T) so it masks out padded key columns
            # for every head and every query position.
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            combined_mask = pad_mask if combined_mask is None else (combined_mask & pad_mask)

        if combined_mask is not None:
            scores = scores.masked_fill(~combined_mask, float("-inf"))

        # What: Normalize scores into attention probabilities along the key axis.
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # What: Weighted sum of Values -> (B, num_heads, T, head_dim).
        context = torch.matmul(attn_weights_dropped, v)

        # What: Recombine heads back into (B, T, embed_dim).
        # Why:  contiguous() ensures memory is laid out sequentially before view().
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(context)
        return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, embed_dim = 2, 5, 32
    x = torch.randn(batch_size, seq_len, embed_dim)

    single_head = SelfAttention(embed_dim=embed_dim, head_dim=16, dropout=0.1)
    out_single, weights_single = single_head(x)
    print(f"Single-head output shape:  {tuple(out_single.shape)}")
    print(f"Single-head weights shape: {tuple(weights_single.shape)}")

    multi_head = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=4, dropout=0.1, causal=True)
    out_multi, weights_multi = multi_head(x)
    print(f"Multi-head output shape:   {tuple(out_multi.shape)}")
    print(f"Multi-head weights shape:  {tuple(weights_multi.shape)}")
    print("Causal mask check (upper triangle max weight):", weights_multi[0, 0].triu(diagonal=1).max().item())
