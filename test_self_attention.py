"""Unit tests for SelfAttention and MultiHeadSelfAttention."""

import unittest

import torch

from self_attention import MultiHeadSelfAttention, SelfAttention


class TestSelfAttention(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_single_head_shapes_and_probabilities(self) -> None:
        x = torch.randn(2, 6, 16)
        attn = SelfAttention(embed_dim=16, head_dim=8)
        out, weights = attn(x)

        self.assertEqual(out.shape, (2, 6, 16))
        self.assertEqual(weights.shape, (2, 6, 6))
        # Every query's attention distribution must sum to 1.0
        row_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_multi_head_causal_mask_blocks_future_tokens(self) -> None:
        x = torch.randn(2, 5, 32)
        attn = MultiHeadSelfAttention(embed_dim=32, num_heads=4, causal=True)
        out, weights = attn(x)

        self.assertEqual(out.shape, (2, 5, 32))
        self.assertEqual(weights.shape, (2, 4, 5, 5))

        # Strictly upper-triangular entries (future tokens) must have zero weight
        upper_triangle = weights.triu(diagonal=1)
        self.assertTrue(torch.allclose(upper_triangle, torch.zeros_like(upper_triangle)))

    def test_multi_head_padding_mask_ignores_padded_keys(self) -> None:
        x = torch.randn(2, 4, 16)
        # First sample has 1 padded token at the end; second sample has 2 padded tokens
        padding_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, False, False],
            ]
        )
        attn = MultiHeadSelfAttention(embed_dim=16, num_heads=2, causal=False)
        _, weights = attn(x, attention_mask=padding_mask)

        # Padded key positions should receive exactly 0 attention weight
        self.assertTrue(torch.allclose(weights[0, :, :, 3], torch.zeros(2, 4)))
        self.assertTrue(torch.allclose(weights[1, :, :, 2:], torch.zeros(2, 4, 2)))

    def test_gradients_flow_through_all_parameters(self) -> None:
        x = torch.randn(2, 4, 16, requires_grad=True)
        attn = MultiHeadSelfAttention(embed_dim=16, num_heads=4, causal=True)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for name, param in attn.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
            self.assertGreater(param.grad.abs().sum().item(), 0.0)

    def test_invalid_num_heads_raises(self) -> None:
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(embed_dim=30, num_heads=4)


if __name__ == "__main__":
    unittest.main()
