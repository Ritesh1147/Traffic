"""
Federated Averaging (FedAvg) Aggregator
=========================================
Implements McMahan et al. (2017) — "Communication-Efficient Learning of
Deep Networks from Decentralized Data"

In this traffic system:
  - Each edge agent trains locally for E steps (one local round)
  - Agent sends its Q-network weights + sample count to this aggregator
  - Aggregator computes weighted average → new global model
  - Global model is broadcast back to all agents
  - Repeat for R communication rounds

This file can run as:
  1. In-process aggregator (for simulation/testing — all in one Python process)
  2. Flower server strategy (for real distributed deployment)
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# ─── Data structures ─────────────────────────────────────────────────────────
@dataclass
class AgentUpdate:
    """Encapsulates one agent's contribution to a FL round."""
    agent_id: str
    weights: List[np.ndarray]      # Q-network parameters
    num_samples: int               # local training samples (for weighting)
    metrics: Dict = field(default_factory=dict)  # loss, reward, epsilon, etc.


@dataclass
class RoundResult:
    """Summary of one completed FL round."""
    round_num: int
    num_agents: int
    global_weights: List[np.ndarray]
    avg_reward: float
    avg_loss: float
    participating_agents: List[str]


# ─── FedAvg Aggregator ───────────────────────────────────────────────────────
class FedAvgAggregator:
    """
    Central FedAvg aggregator.

    Weighted average formula:
        W_global = Σᵢ (nᵢ / N) * Wᵢ
    where nᵢ = samples used by agent i, N = total samples across all agents.

    Agents with more training experience contribute proportionally more
    to the global model — appropriate because busy intersections see more
    diverse traffic patterns.
    """

    def __init__(
        self,
        num_agents: int,
        min_agents_per_round: int = 2,
        save_dir: str = "results/checkpoints",
    ):
        self.num_agents = num_agents
        self.min_agents = min_agents_per_round
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.global_weights: Optional[List[np.ndarray]] = None
        self.round_history: List[RoundResult] = []
        self.current_round = 0

    # ── Core aggregation ────────────────────────────────────────────────────
    def aggregate(self, updates: List[AgentUpdate]) -> Optional[List[np.ndarray]]:
        """
        Compute the weighted average of agent weights.

        Args:
            updates: list of AgentUpdate from participating agents

        Returns:
            Aggregated global weights, or None if too few agents.
        """
        if len(updates) < self.min_agents:
            print(
                f"[Aggregator] Round {self.current_round}: only {len(updates)} agents "
                f"(need {self.min_agents}). Skipping aggregation."
            )
            return None

        # Total samples across all participating agents
        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            # Fallback: equal weighting
            total_samples = len(updates)
            weights_counts = [(u.weights, 1) for u in updates]
        else:
            weights_counts = [(u.weights, u.num_samples) for u in updates]

        # Weighted average layer by layer
        num_layers = len(updates[0].weights)
        aggregated = []
        for layer_idx in range(num_layers):
            layer_avg = np.zeros_like(updates[0].weights[layer_idx], dtype=np.float64)
            for agent_weights, count in weights_counts:
                proportion = count / total_samples
                layer_avg += proportion * agent_weights[layer_idx].astype(np.float64)
            aggregated.append(layer_avg.astype(np.float32))

        self.global_weights = aggregated

        # Compute round metrics
        avg_reward = np.mean([
            u.metrics.get("avg_reward", 0.0) for u in updates
        ])
        avg_loss = np.mean([
            u.metrics.get("avg_loss", 0.0) for u in updates
            if u.metrics.get("avg_loss") is not None
        ])

        result = RoundResult(
            round_num=self.current_round,
            num_agents=len(updates),
            global_weights=aggregated,
            avg_reward=float(avg_reward),
            avg_loss=float(avg_loss),
            participating_agents=[u.agent_id for u in updates],
        )
        self.round_history.append(result)
        self.current_round += 1

        print(
            f"[Aggregator] Round {result.round_num} complete | "
            f"agents={result.num_agents} | "
            f"total_samples={total_samples} | "
            f"avg_reward={result.avg_reward:.3f} | "
            f"avg_loss={result.avg_loss:.4f}"
        )

        # Print per-agent contribution breakdown
        for u in updates:
            pct = 100 * u.num_samples / total_samples
            print(f"  [{u.agent_id}] samples={u.num_samples} ({pct:.1f}%) | "
                  f"reward={u.metrics.get('avg_reward', 0):.3f} | "
                  f"ε={u.metrics.get('epsilon', '?'):.3f}")

        return aggregated

    # ── History and persistence ──────────────────────────────────────────────
    def save_round_history(self, path: Optional[str] = None) -> str:
        """Save round-by-round metrics to JSON for plotting."""
        path = path or os.path.join(self.save_dir, "round_history.json")
        history = [
            {
                "round": r.round_num,
                "num_agents": r.num_agents,
                "avg_reward": r.avg_reward,
                "avg_loss": r.avg_loss,
                "agents": r.participating_agents,
            }
            for r in self.round_history
        ]
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        return path

    def save_global_model(self, round_num: Optional[int] = None) -> Optional[str]:
        """Save the current global weights as a .npz file."""
        if self.global_weights is None:
            return None
        rnd = round_num if round_num is not None else self.current_round
        path = os.path.join(self.save_dir, f"global_model_round_{rnd:04d}.npz")
        np.savez(path, *self.global_weights)
        return path

    def load_global_model(self, path: str) -> List[np.ndarray]:
        """Load global weights from a .npz file."""
        data = np.load(path)
        self.global_weights = [data[k] for k in sorted(data.files)]
        print(f"[Aggregator] Loaded global model from {path}")
        return self.global_weights

    def get_best_round(self) -> Optional[RoundResult]:
        """Return the round with highest average reward."""
        if not self.round_history:
            return None
        return max(self.round_history, key=lambda r: r.avg_reward)


# ─── Flower Strategy (for real distributed deployment) ──────────────────────
def make_flower_strategy(num_agents: int, save_dir: str = "results/flower"):
    """
    Create a Flower FedAvg strategy for use with flwr.server.start_server().

    This enables real distributed deployment where each edge device runs
    a separate Python process (or even a separate machine).

    Usage:
        strategy = make_flower_strategy(num_agents=3)
        flwr.server.start_server(
            server_address="0.0.0.0:8080",
            config=flwr.server.ServerConfig(num_rounds=50),
            strategy=strategy,
        )
    """
    try:
        import flwr as fl
        from flwr.common import Parameters, FitRes, parameters_to_ndarrays

        class TrafficFedAvg(fl.server.strategy.FedAvg):
            """
            Extends Flower's built-in FedAvg with:
              - Logging per round
              - Model checkpoint saving
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.round_rewards = []
                os.makedirs(save_dir, exist_ok=True)

            def aggregate_fit(self, server_round, results, failures):
                """Called after each round — aggregate and log."""
                aggregated_params, metrics = super().aggregate_fit(
                    server_round, results, failures
                )
                if aggregated_params is not None:
                    # Save checkpoint every 10 rounds
                    if server_round % 10 == 0:
                        arrays = parameters_to_ndarrays(aggregated_params)
                        path = os.path.join(
                            save_dir, f"global_round_{server_round:04d}.npz"
                        )
                        np.savez(path, *arrays)
                        print(f"[FlowerServer] Saved checkpoint: {path}")

                return aggregated_params, metrics

        strategy = TrafficFedAvg(
            fraction_fit=1.0,           # use all available clients each round
            fraction_evaluate=1.0,
            min_fit_clients=2,          # need at least 2 agents
            min_evaluate_clients=2,
            min_available_clients=2,
        )
        return strategy

    except ImportError:
        print("Flower not installed. Run: pip install flwr")
        return None
