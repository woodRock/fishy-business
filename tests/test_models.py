# -*- coding: utf-8 -*-
import unittest
import torch
import numpy as np
from fishy.models.utils import ensure_conv_input, ensure_seq_input
from fishy.models.deep.cnn import CNN
from fishy.models.deep.transformer import Transformer
from fishy.models.deep.dense import Dense
from fishy.models.deep.lstm import LSTM
from fishy.models.evolutionary.pso import PSO
from fishy.models.evolutionary.ga import GA


class TestTensorUtils(unittest.TestCase):
    def test_ensure_conv_input_2d(self):
        x = torch.randn(4, 100)
        out = ensure_conv_input(x)
        self.assertEqual(out.shape, (4, 1, 100))

    def test_ensure_conv_input_already_3d(self):
        x = torch.randn(4, 1, 100)
        out = ensure_conv_input(x)
        self.assertEqual(out.shape, (4, 1, 100))

    def test_ensure_seq_input_2d(self):
        x = torch.randn(4, 100)
        out = ensure_seq_input(x)
        self.assertEqual(out.shape, (4, 100, 1))

    def test_ensure_seq_input_already_3d(self):
        x = torch.randn(4, 100, 1)
        out = ensure_seq_input(x)
        self.assertEqual(out.shape, (4, 100, 1))


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

    def test_lstm_forward(self):
        model = LSTM(input_dim=100, output_dim=5)
        x = torch.randn(2, 100)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))


class TestEvolutionaryModels(unittest.TestCase):
    def test_pso_predict_proba(self):
        X = np.random.rand(20, 10)
        y = np.random.randint(0, 2, 20)
        model = PSO(iterations=2, population_size=5)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        self.assertEqual(probs.shape, (5, 2))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))

    def test_ga_predict_proba(self):
        X = np.random.rand(20, 10)
        y = np.random.randint(0, 2, 20)
        model = GA(generations=2, population_size=5)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        self.assertEqual(probs.shape, (5, 2))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
