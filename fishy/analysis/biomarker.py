# -*- coding: utf-8 -*-
"""
Automated Biomarker Discovery module.

This module maps high-importance m/z values (from XAI) to chemical entities
using public databases (LipidMaps).
"""

import logging
import requests
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from .xai import GradCAM

logger = logging.getLogger(__name__)


class AutomatedBiomarkerDiscovery:
    """
    Connects XAI results to chemical databases for automated biomarker identification.
    """

    def __init__(self, m_z_values: List[float]) -> None:
        """
        Args:
            m_z_values (List[float]): List of m/z values corresponding to the input features.
        """
        self.m_z_values = m_z_values
        self.cache: Dict[float, List[Dict[str, Any]]] = {}

    def query_lipidmaps(
        self, mz: float, tolerance: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Queries LipidMaps REST API for potential matches for a given m/z.
        Uses M-H ion type for negative mode REIMS.
        """
        mz_rounded = round(mz, 4)
        if mz_rounded in self.cache:
            return self.cache[mz_rounded]

        # REST API endpoint for m/z search
        # Format: /rest/moverz/{database}/{m/z}/{ion-type}/{mass-tolerance}/{output-format}
        url = f"https://www.lipidmaps.org/rest/moverz/LMSD/{mz_rounded}/M-H/{tolerance}/txt"

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                text = response.text.strip()
                if not text or "No records found" in text:
                    self.cache[mz_rounded] = []
                    return []

                results = []
                # LipidMaps TXT output is tab-separated with a header
                lines = text.split("\n")
                if len(lines) > 1:
                    header = lines[0].split("\t")
                    for line in lines[1:4]:  # Take top 3
                        parts = line.split("\t")
                        if len(parts) == len(header):
                            row = dict(zip(header, parts))
                            results.append(
                                {
                                    "name": row.get(
                                        "name", row.get("common_name", "Unknown")
                                    ),
                                    "formula": row.get("formula"),
                                    "id": row.get("lm_id"),
                                    "mz": row.get("exact_mass"),
                                    "source": "LipidMaps",
                                }
                            )

                self.cache[mz_rounded] = results
                return results
        except Exception as e:
            logger.warning(f"LipidMaps query failed for m/z {mz}: {e}")

        return []

    def identify_top_peaks(
        self, importance_scores: np.ndarray, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Finds the top N m/z values and queries databases.
        """
        top_indices = np.argsort(importance_scores)[-top_n:][::-1]
        results = []

        for idx in top_indices:
            mz = self.m_z_values[idx]
            matches = self.query_lipidmaps(mz)
            results.append(
                {
                    "feature_index": int(idx),
                    "m_z": float(mz),
                    "importance": float(importance_scores[idx]),
                    "matches": matches,
                }
            )

        return results

    def generate_report(self, biomarker_results: List[Dict[str, Any]]) -> str:
        """
        Formats the discovery results into a readable string.
        """
        lines = ["--- Automated Biomarker Discovery Report (LipidMaps) ---"]
        for res in biomarker_results:
            match_str = "[dim]No database matches[/]"
            if res["matches"]:
                best_match = res["matches"][0]
                match_str = (
                    f"[bold green]{best_match['name']}[/] ({best_match['source']})"
                )

            lines.append(
                f"Peak m/z {res['m_z']:.4f} (Imp: {res['importance']:.4f}) -> {match_str}"
            )

        return "\n".join(lines)


def run_biomarker_pipeline(model, data_loader, feature_names, device, top_n=5):
    """
    Complete flow: Model -> Grad-CAM -> Top Peaks -> Chemical Identity
    """
    target_layer = None
    # 1. Look for Gating layer if GMOE
    if hasattr(model, "gate"):
        target_layer = model.gate

    # 2. Otherwise look for Convolutional layers
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv1d):
                target_layer = module
                break

    # 3. Fallback to Linear
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Linear):
                target_layer = module
                break

    if target_layer is None:
        return "Could not identify target layer for biomarker discovery."

    gc = GradCAM(model, target_layer)
    features, _ = next(iter(data_loader))
    # Run CAM on first batch
    cam = gc.generate_cam(features.to(device)).mean(dim=0).cpu().numpy()
    gc.remove_hooks()

    # 2. Map to chemical names
    try:
        mz_floats = [float(f) for f in feature_names]
    except ValueError:
        mz_floats = list(range(len(feature_names)))

    discovery = AutomatedBiomarkerDiscovery(mz_floats)
    biomarkers = discovery.identify_top_peaks(cam, top_n=top_n)

    return discovery.generate_report(biomarkers)
