# -*- coding: utf-8 -*-
import unittest
import torch
from fishy.models.deep.cnn import CNN
from fishy.models.deep.transformer import Transformer
from fishy.models.deep.dense import Dense


class TestDeepModels(unittest.TestCase):
    def test_cnn_forward(self):
        model = CNN(input_dim=100, output_dim=5)
        x = torch.randn(2, 100)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_transformer_forward(self):
        model = Transformer(input_dim=100, output_dim=5, num_heads=4, hidden_dim=64)
        x = torch.randn(2, 100)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_dense_forward(self):
        model = Dense(input_dim=100, output_dim=5)
        x = torch.randn(2, 100)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
