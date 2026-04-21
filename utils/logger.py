"""
Training logger and metrics utilities
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class TrainingLogger:
    """Logs training metrics to JSON. Can be read by TensorBoard or plotted."""

    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self.entries: List[Dict] = []

    def log_round(
        self,
        round_num: int,
        avg_reward: float,
        avg_loss: float,
        epsilons: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        entry = {
            "round": round_num,
            "avg_reward": round(avg_reward, 4),
            "avg_loss": round(avg_loss, 6),
            "epsilons": epsilons or {},
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.entries.append(entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_rewards(self) -> List[float]:
        return [e["avg_reward"] for e in self.entries]

    def get_losses(self) -> List[float]:
        return [e["avg_loss"] for e in self.entries]


def compute_metrics(rewards: List[float], window: int = 10) -> Dict:
    """Compute summary statistics over a reward history."""
    arr = np.array(rewards)
    return {
        "mean":    float(np.mean(arr)),
        "std":     float(np.std(arr)),
        "max":     float(np.max(arr)),
        "min":     float(np.min(arr)),
        "last_10": float(np.mean(arr[-window:])) if len(arr) >= window else float(np.mean(arr)),
    }
