# -*- coding: utf-8 -*-
import unittest
import torch
from fishy.models.deep.nextformer import (
    NextFormer,
    RMSNorm,
    SwiGLU,
    GroupedQueryAttention,
)


class TestNextFormer(unittest.TestCase):
    def test_rmsnorm_forward(self):
        dim = 64
        norm = RMSNorm(dim)
        x = torch.randn(2, 10, dim)
        y = norm(x)
        self.assertEqual(y.shape, (2, 10, dim))
        # Check if it approximately normalizes (RMS should be ~1)
        rms = torch.sqrt(y.pow(2).mean(-1))
        # Since there is a learnable weight initialized to 1,
        # it should be close to 1 before training
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_swiglu_forward(self):
        dim = 64
        hidden_dim = 128
        ffn = SwiGLU(dim, hidden_dim)
        x = torch.randn(2, 10, dim)
        y = ffn(x)
        self.assertEqual(y.shape, (2, 10, dim))
        # Check bias-free
        self.assertIsNone(ffn.w1.bias)
        self.assertIsNone(ffn.w2.bias)
        self.assertIsNone(ffn.w3.bias)

    def test_gqa_forward(self):
        dim = 64
        n_heads = 8
        n_kv_heads = 2
        gqa = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        x = torch.randn(2, 10, dim)
        y = gqa(x)
        self.assertEqual(y.shape, (2, 10, dim))
        # Check bias-free
        self.assertIsNone(gqa.wq.bias)
        self.assertIsNone(gqa.wk.bias)
        self.assertIsNone(gqa.wv.bias)
        self.assertIsNone(gqa.wo.bias)

    def test_nextformer_forward(self):
        model = NextFormer(
            input_dim=100, output_dim=5, num_heads=4, num_kv_heads=2, hidden_dim=64
        )
        x = torch.randn(2, 100)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))
        # Check bias-free output head
        self.assertIsNone(model.fc_out.bias)

    def test_nextformer_return_attention(self):
        num_layers = 2
        model = NextFormer(
            input_dim=100,
            output_dim=5,
            num_layers=num_layers,
            num_heads=4,
            num_kv_heads=2,
        )
        x = torch.randn(2, 100)
        y, attentions = model(x, return_attention=True)
        self.assertEqual(y.shape, (2, 5))
        self.assertEqual(len(attentions), num_layers)
        # Attention shape: (batch, n_heads, seq_len, seq_len)
        # ensure_conv_input makes x (2, 1, 100)
        self.assertEqual(attentions[0].shape, (2, 4, 1, 1))

    def test_create_model_nextformer(self):
        from fishy._core.factory import create_model
        from fishy._core.config import TrainingConfig

        config = TrainingConfig(
            model="nextformer", hidden_dim=64, num_layers=2, num_heads=4, num_kv_heads=2
        )
        model = create_model(config, input_dim=100, output_dim=5)
        self.assertIsInstance(model, NextFormer)
        self.assertEqual(len(model.blocks), 2)

        x = torch.randn(2, 100)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
