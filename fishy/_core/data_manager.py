# -*- coding: utf-8 -*-
"""
Utility for managing and downloading the private REIMS dataset.
"""

import os
import requests
import logging
from pathlib import Path
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

def download_dataset(token: str = None, destination: str = None):
    """
    Downloads the REIMS dataset from a secure remote source.
    Expects a GitHub Personal Access Token (PAT) for private repo access.
    """
    # 1. Configuration
    DATA_URL = os.environ.get("FISHY_DATA_URL", "https://raw.githubusercontent.com/woodRock/fishy-data/main/REIMS.xlsx")
    
    if not token:
        token = os.environ.get("FISHY_DATA_TOKEN")
    
    if not token:
        import sys
        # Only prompt if we are in an interactive terminal and not in CI
        if sys.stdin.isatty() and not os.environ.get("CI"):
            import getpass
            console.print("[bold yellow]This dataset is private.[/]")
            token = getpass.getpass("Enter your GitHub Personal Access Token: ")
    
    if not token or len(token.strip()) == 0:
        raise ValueError("No authentication token provided. Please set FISHY_DATA_TOKEN.")

    if not destination:
        from fishy import get_data_path
        # Use get_data_path to find where the package expects the data
        destination = get_data_path()
    
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"token {token}"}
    
    console.print(f"[bold blue]Downloading dataset from:[/] {DATA_URL}")
    
    try:
        response = requests.get(DATA_URL, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        console.print(f"[bold green]Successfully downloaded dataset to:[/] {dest_path}")
        return True
    except Exception as e:
        console.print(f"[bold red]Download failed:[/] {e}")
        return False

def check_data_exists() -> bool:
    """Checks if the dataset is present."""
    from fishy import get_data_path
    path = Path(get_data_path())
    return path.exists()
