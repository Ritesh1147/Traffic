"""
DQN Agent (CPU-optimised)
==========================
Deep Q-Network for traffic signal control.
Designed to run efficiently on CPU — no GPU required.

Architecture:
  Input(8) → Dense(64) → ReLU → Dense(64) → ReLU → Output(4)

Small network is intentional:
  - Fits in CPU cache
  - Fast inference on edge devices (Raspberry Pi, Jetson Nano)
  - Sufficient capacity for the 8-dim state / 4-action problem
"""

import os
import copy
import random
from collections import deque
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─── Q-Network ──────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Small feed-forward network.
    Two hidden layers of 64 units each — suitable for edge deployment.
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # Initialise weights with Xavier uniform for stable early training
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Replay Buffer ───────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Experience replay buffer.
    Stores (state, action, reward, next_state, done) tuples.
    Samples random mini-batches for training — breaks temporal correlation.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ─── DQN Agent ──────────────────────────────────────────────────────────────
class DQNAgent:
    """
    DQN agent for traffic signal control.

    Key design decisions for CPU / federated use:
      - Small network (fast forward pass on CPU)
      - Soft target update (tau) instead of hard copy every N steps
        (more stable with fewer samples per round in FL)
      - Epsilon decay via multiplicative factor (simple, predictable)

    Federated Learning interface:
      get_weights()  → list of numpy arrays (to send to aggregator)
      set_weights()  → load numpy arrays back (received from aggregator)
    """

    def __init__(
        self,
        agent_id: str,
        state_dim: int = 8,
        action_dim: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_tau: float = 0.005,
        hidden_size: int = 64,
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = target_update_tau

        # Force CPU — no GPU needed
        self.device = torch.device("cpu")

        # Online and target networks
        self.q_net = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.train_steps = 0
        self.episode_count = 0
        self.total_samples = 0  # tracked for FedAvg weighting

    # ── Action selection ────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    # ── Experience storage ───────────────────────────────────────────────────
    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)
        self.total_samples += 1

    # ── Learning step ────────────────────────────────────────────────────────
    def learn(self) -> Optional[float]:
        """
        One gradient update step.
        Returns the loss value (float) or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for chosen actions
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN style)
        with torch.no_grad():
            # Use online net to select action, target net to evaluate
            next_actions = self.q_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # Huber loss (less sensitive to outliers than MSE)
        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients on CPU
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft update target network: θ_target = τ·θ_online + (1-τ)·θ_target
        for target_param, online_param in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

        self.train_steps += 1
        return loss.item()

    # ── Epsilon decay ────────────────────────────────────────────────────────
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── Federated Learning interface ─────────────────────────────────────────
    def get_weights(self) -> List[np.ndarray]:
        """
        Extract Q-network weights as a list of numpy arrays.
        This is what gets sent to the FL aggregator.
        """
        return [
            param.data.cpu().numpy().copy()
            for param in self.q_net.parameters()
        ]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Load weights received from the FL aggregator back into the Q-network.
        Also syncs the target network to match.
        """
        with torch.no_grad():
            for param, w in zip(self.q_net.parameters(), weights):
                param.data.copy_(torch.FloatTensor(w))
        # Sync target network after receiving global weights
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── Checkpoint save/load ─────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "total_samples": self.total_samples,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_steps = checkpoint["train_steps"]
        self.total_samples = checkpoint.get("total_samples", 0)
        print(f"[{self.agent_id}] Loaded checkpoint from {path}")

    def __repr__(self) -> str:
        return (
            f"DQNAgent({self.agent_id}, "
            f"ε={self.epsilon:.3f}, "
            f"steps={self.train_steps}, "
            f"buffer={len(self.buffer)})"
        )
