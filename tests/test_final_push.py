# -*- coding: utf-8 -*-
import pytest
import os
import argparse
from fishy.cli.main import setup_parser, detect_method
from fishy._core.utils import RunContext, set_seed, get_device
from fishy.experiments.transfer import run_sequential_transfer_learning


def test_cli_parsing():
    parser = setup_parser()
    # Test a few commands
    args = parser.parse_args(
        ["train", "--model", "transformer", "--dataset", "species"]
    )
    assert args.command == "train"
    assert args.model == "transformer"

    args_all = parser.parse_args(["run_all", "--quick"])
    assert args_all.command == "run_all"
    assert args_all.quick is True


def test_method_detection():
    assert detect_method("transformer") == "deep"
    assert detect_method("rf") == "classic"
    assert detect_method("ga") == "evolutionary"
    assert detect_method("simclr") == "contrastive"


def test_utils_context(tmp_path):
    set_seed(42)
    device = get_device()
    ctx = RunContext("test_ds", "test_meth", "test_mdl")
    ctx.save_results({"test": 1})
    assert (ctx.run_dir / "results" / "metrics.json").exists()

    # Test dataframe saving
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2]})
    ctx.save_dataframe(df, "test.csv")
    assert (ctx.run_dir / "results" / "test.csv").exists()


def test_transfer_learning_flow(tmp_path):
    data_file = "data/REIMS.xlsx"
    if not os.path.exists(data_file):
        pytest.skip("Data file not found")

    # Very quick transfer test
    model, results = run_sequential_transfer_learning(
        model_name="dense",
        transfer_datasets=["oil"],
        target_dataset="species",
        num_epochs_transfer=1,
        num_epochs_finetune=1,
        file_path=data_file,
        wandb_log=False,
    )
    # The results structure is nested: results['finetune'][target_dataset]
    assert "finetune" in results
    assert "species" in results["finetune"]
    assert (
        "val_balanced_acc" in results["finetune"]["species"]
        or "val_balanced_accuracy" in results["finetune"]["species"]
    )
