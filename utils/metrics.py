"""Metrics helpers — imported by train_federated.py"""
import numpy as np
from typing import List, Dict


def compute_metrics(rewards: List[float], window: int = 10) -> Dict:
    arr = np.array(rewards) if rewards else np.array([0.0])
    return {
        "mean":    float(np.mean(arr)),
        "std":     float(np.std(arr)),
        "max":     float(np.max(arr)),
        "min":     float(np.min(arr)),
        "last_n":  float(np.mean(arr[-window:])) if len(arr) >= window else float(np.mean(arr)),
    }
