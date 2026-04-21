"""
Flower Client — bridges DQN agent with the Flower FL framework
================================================================
Each edge device runs one instance of TrafficFlowerClient.
It wraps a DQNAgent and implements the Flower client interface:
  fit()      — local training, returns updated weights + metrics
  evaluate() — local evaluation, returns loss + metrics

For simulation (all agents in one process), use LocalFlowerClient
which skips the network layer entirely.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from agents.dqn_agent import DQNAgent
from envs.traffic_env import TrafficSignalEnv


# ─── Local client (simulation mode — no network) ────────────────────────────
class LocalFlowerClient:
    """
    Simulates a Flower client locally.
    Used when running all agents in a single Python process (no sockets).
    Ideal for Windows 11 laptop development and testing.
    """

    def __init__(
        self,
        agent: DQNAgent,
        env: TrafficSignalEnv,
        local_steps: int = 200,
        learn_every: int = 4,
    ):
        self.agent = agent
        self.env = env
        self.local_steps = local_steps
        self.learn_every = learn_every
        self._obs, _ = env.reset()

    def get_parameters(self) -> List[np.ndarray]:
        """Return current model weights."""
        return self.agent.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load global weights from aggregator."""
        self.agent.set_weights(parameters)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Load global weights, run local_steps of DQN training, return results.

        Returns:
            (updated_weights, num_samples, metrics_dict)
        """
        # Load global model
        self.set_parameters(parameters)

        rewards = []
        losses = []
        episode_reward = 0.0
        samples_this_round = 0

        for step in range(self.local_steps):
            # Select action and step environment
            action = self.agent.select_action(self._obs, training=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store experience
            self.agent.store(self._obs, action, reward, next_obs, done)
            self._obs = next_obs
            episode_reward += reward
            samples_this_round += 1

            # Learn every N steps
            if step % self.learn_every == 0:
                loss = self.agent.learn()
                if loss is not None:
                    losses.append(loss)

            # Reset on episode end
            if done:
                rewards.append(episode_reward)
                episode_reward = 0.0
                self._obs, _ = self.env.reset()

        # Decay epsilon after each local round
        self.agent.decay_epsilon()

        metrics = {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon": float(self.agent.epsilon),
            "agent_id": self.agent.agent_id,
            "num_episodes": len(rewards),
        }

        return self.agent.get_weights(), samples_this_round, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate global model on local environment (no exploration).

        Returns:
            (loss, num_samples, metrics_dict)
        """
        self.set_parameters(parameters)

        obs, _ = self.env.reset()
        total_reward = 0.0
        total_steps = 0
        done = False

        while not done and total_steps < 500:
            action = self.agent.select_action(obs, training=False)  # greedy
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_steps += 1

        return (
            -total_reward,   # loss = negative reward (lower is better)
            total_steps,
            {"eval_reward": float(total_reward), "agent_id": self.agent.agent_id},
        )


# ─── Real Flower client (distributed mode) ──────────────────────────────────
def make_flower_client(
    agent: DQNAgent,
    env: TrafficSignalEnv,
    local_steps: int = 200,
    server_address: str = "127.0.0.1:8080",
):
    """
    Create and start a real Flower client that connects to a Flower server.
    Each edge device calls this to participate in federated training.

    Usage on edge device:
        agent = DQNAgent("intersection_A")
        env   = TrafficSignalEnv("A")
        make_flower_client(agent, env, server_address="192.168.1.10:8080")
    """
    try:
        import flwr as fl
        from flwr.common import NDArrays, Scalar

        local_client = LocalFlowerClient(agent, env, local_steps)

        class FlowerClientWrapper(fl.client.NumPyClient):
            def get_parameters(self, config) -> NDArrays:
                return local_client.get_parameters()

            def fit(self, parameters: NDArrays, config) -> Tuple[NDArrays, int, Dict]:
                return local_client.fit(parameters, config)

            def evaluate(self, parameters: NDArrays, config) -> Tuple[float, int, Dict]:
                return local_client.evaluate(parameters, config)

        fl.client.start_numpy_client(
            server_address=server_address,
            client=FlowerClientWrapper(),
        )

    except ImportError:
        print("Flower not installed. Run: pip install flwr")
