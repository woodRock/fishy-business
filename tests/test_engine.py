# -*- coding: utf-8 -*-
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fishy.engine.losses import coral_loss, levels_from_labelbatch, cumulative_link_loss
from fishy.engine.trainer import Trainer, DeepEngine

class TestLosses(unittest.TestCase):
    def test_levels_from_labelbatch(self):
        labels = torch.tensor([0, 1, 2])
        levels = levels_from_labelbatch(labels, num_classes=4)
        expected = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0]
        ])
        self.assertTrue(torch.equal(levels.long(), expected))

    def test_coral_loss(self):
        logits = torch.tensor([[10.0, 5.0], [-5.0, -10.0]])
        levels = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        loss = coral_loss(logits, levels)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertLess(loss.item(), 0.1)

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(10, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.train_data = TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
        self.train_loader = DataLoader(self.train_data, batch_size=2)

    def test_trainer_init(self):
        trainer = Trainer(self.model, self.criterion, self.optimizer, torch.device('cpu'), num_epochs=1)
        self.assertEqual(trainer.num_epochs, 1)

    def test_trainer_train_smoke(self):
        # Smoke test: runs one epoch without crashing
        trainer = Trainer(self.model, self.criterion, self.optimizer, torch.device('cpu'), num_epochs=1)
        results = trainer.train(self.train_loader)
        self.assertIn("best_accuracy", results)
        self.assertIn("epoch_metrics", results)

    def test_deep_engine_evaluate(self):
        results = DeepEngine.evaluate_model(self.model, self.train_loader, self.criterion, device='cpu')
        self.assertIn("loss", results)
        self.assertIn("metrics", results)

if __name__ == "__main__":
    unittest.main()
