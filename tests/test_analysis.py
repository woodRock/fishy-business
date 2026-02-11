# -*- coding: utf-8 -*-
import unittest
import torch
import torch.nn as nn
from fishy.analysis.statistical import perform_significance_test
from fishy.analysis.benchmark import measure_model_size


class TestAnalysis(unittest.TestCase):
    def test_significance_test(self):
        model = [0.9, 0.95, 0.92]
        baseline = [0.8, 0.82, 0.81]
        res = perform_significance_test(model, baseline)
        self.assertTrue(bool(res["significant"]))
        self.assertEqual(res["symbol"], "+")

    def test_measure_model_size(self):
        model = nn.Linear(100, 100)
        size = measure_model_size(model)
        self.assertGreater(size, 0)


if __name__ == "__main__":
    unittest.main()
