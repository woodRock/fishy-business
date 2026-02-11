#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the fishy business project.
"""

import warnings
import os

# 1. Global warning suppression for cleaner CLI experience
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress noisy library logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLBACKEND"] = "Agg"

from fishy.cli.main import main

if __name__ == "__main__":
    main()
