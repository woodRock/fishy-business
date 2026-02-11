#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the fishy business project.

Examples:
    >>> import main
    >>> hasattr(main, 'main')
    True
"""

import warnings

# Suppress the urllib3 NotOpenSSLWarning on macOS
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

from fishy.cli.main import main

if __name__ == "__main__":
    main()
