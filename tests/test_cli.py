# -*- coding: utf-8 -*-
import unittest
from fishy.cli.main import detect_method, get_all_models, setup_parser


class TestCLI(unittest.TestCase):
    def test_detect_method(self):
        self.assertEqual(detect_method("transformer"), "deep")
        self.assertEqual(detect_method("opls-da"), "classic")
        self.assertEqual(detect_method("ga"), "evolutionary")
        self.assertEqual(detect_method("simclr"), "contrastive")

    def test_get_all_models(self):
        models = get_all_models()
        self.assertIn("transformer", models)
        self.assertIn("opls-da", models)

    def test_setup_parser(self):
        parser = setup_parser()
        # Test parsing some basic args
        args = parser.parse_args(["train", "-m", "cnn", "-d", "species"])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.model, "cnn")
        self.assertEqual(args.dataset, "species")


if __name__ == "__main__":
    unittest.main()
