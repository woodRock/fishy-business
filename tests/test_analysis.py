# -*- coding: utf-8 -*-
import unittest
import torch
import torch.nn as nn
import numpy as np
from fishy.analysis.statistical import perform_significance_test, summarize_results
from fishy.analysis.benchmark import measure_model_size
from fishy.analysis.xai import ModelWrapper


class TestAnalysis(unittest.TestCase):
    def test_significance_test(self):
        model = [0.9, 0.95, 0.92]
        baseline = [0.8, 0.82, 0.81]
        res = perform_significance_test(model, baseline)
        self.assertTrue(bool(res["significant"]))
        self.assertEqual(res["symbol"], "+")

    def test_significance_test_mismatched_lengths(self):
        # Should pad and still work
        model = [0.9, 0.9, 0.9]
        baseline = [0.8]
        res = perform_significance_test(model, baseline)
        self.assertEqual(res["symbol"], "+")

    def test_summarize_results_with_baseline(self):
        results_map = {
            "ds|||m1": [{"stats": {"val_balanced_accuracy": 0.9}}],
            "ds|||opls-da": [{"stats": {"val_balanced_accuracy": 0.8}}],
        }
        df = summarize_results(results_map)
        self.assertIn("Baseline", df.columns)
        self.assertEqual(df[df["Method"] == "m1"]["Baseline"].iloc[0], "opls-da")

    def test_measure_model_size(self):
        model = nn.Linear(100, 100)
        size = measure_model_size(model)
        self.assertGreater(size, 0)

    def test_model_wrapper_torch(self):
        model = nn.Sequential(nn.Linear(10, 2), nn.Softmax(dim=1))
        wrapper = ModelWrapper(model, device="cpu")
        x = np.random.rand(5, 10).astype(np.float32)
        probs = wrapper.predict_proba(x)
        self.assertEqual(probs.shape, (5, 2))
        self.assertAlmostEqual(np.sum(probs[0]), 1.0, places=5)

    def test_model_wrapper_sklearn(self):
        class DummySklearn:
            def predict_proba(self, x):
                return np.zeros((len(x), 3)) + 0.33
        
        wrapper = ModelWrapper(DummySklearn(), device="cpu")
        x = np.random.rand(5, 10)
        probs = wrapper.predict_proba(x)
        self.assertEqual(probs.shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
