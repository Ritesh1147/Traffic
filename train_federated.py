"""
Federated Traffic Signal Control — Main Training Script
=========================================================
Run this file to start training:
    python train_federated.py

What happens:
  1. Creates N intersections (agents + environments)
  2. Runs R communication rounds
  3. Each round: agents train locally for E steps, then FedAvg aggregates
  4. Saves metrics, checkpoints, and plots to results/

Configuration: edit configs/config.yaml or pass args (see bottom of file)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed on Windows without a window
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from agents.dqn_agent import DQNAgent
from envs.traffic_env import TrafficSignalEnv
from federated.aggregator import FedAvgAggregator, AgentUpdate
from federated.flower_client import LocalFlowerClient
from utils.logger import TrainingLogger
from utils.metrics import compute_metrics


# ─── Training configuration ──────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Topology
    "num_agents": 3,
    "agent_ids": ["A", "B", "C"],

    # FL hyperparameters
    "num_rounds": 100,               # enough rounds for epsilon to decay meaningfully
    "local_steps_per_round": 600,    # ~3 full episodes per round (200 steps each)
    "learn_every": 2,                # learn more frequently
    "min_agents_per_round": 2,

    # DQN hyperparameters
    "state_dim": 8,
    "action_dim": 4,
    "lr": 5e-4,                      # slightly lower lr for stability
    "gamma": 0.95,                   # lower gamma — reward is dense, no long horizons
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.990,          # faster decay: after 100 rounds ε ≈ 0.36
    "buffer_capacity": 20_000,       # larger buffer for better sample diversity
    "batch_size": 128,               # larger batches on CPU are fine
    "hidden_size": 128,              # bigger network — more capacity
    "target_update_tau": 0.01,       # slightly faster target sync

    # Environment
    "use_sumo": False,

    # Saving
    "save_dir": "results",
    "checkpoint_every": 10,
    "eval_every": 5,
    "seed": 42,
}


# ─── Build agents and environments ───────────────────────────────────────────
def build_system(config: dict):
    agents = []
    envs = []
    clients = []

    for i, aid in enumerate(config["agent_ids"]):
        seed = config["seed"] + i

        # Environment
        env = TrafficSignalEnv(
            intersection_id=f"intersection_{aid}",
            use_sumo=config["use_sumo"],
            seed=seed,
        )

        # Agent
        agent = DQNAgent(
            agent_id=aid,
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            lr=config["lr"],
            gamma=config["gamma"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay=config["epsilon_decay"],
            buffer_capacity=config["buffer_capacity"],
            batch_size=config["batch_size"],
            hidden_size=config["hidden_size"],
            target_update_tau=config["target_update_tau"],
        )

        # Local FL client
        client = LocalFlowerClient(
            agent=agent,
            env=env,
            local_steps=config["local_steps_per_round"],
            learn_every=config["learn_every"],
        )

        agents.append(agent)
        envs.append(env)
        clients.append(client)

    # Aggregator
    aggregator = FedAvgAggregator(
        num_agents=config["num_agents"],
        min_agents_per_round=config["min_agents_per_round"],
        save_dir=os.path.join(config["save_dir"], "checkpoints"),
    )

    return agents, envs, clients, aggregator


# ─── Evaluation (greedy policy, no exploration) ──────────────────────────────
def evaluate_agents(clients, global_weights, num_episodes=3):
    """Run greedy evaluation on all agents and return mean reward."""
    all_rewards = []
    for client in clients:
        for _ in range(num_episodes):
            _, _, metrics = client.evaluate(global_weights, {})
            all_rewards.append(metrics["eval_reward"])
    return float(np.mean(all_rewards))


# ─── Main training loop ──────────────────────────────────────────────────────
def train(config: dict):
    print("\n" + "="*60)
    print("  FEDERATED TRAFFIC SIGNAL CONTROL TRAINING")
    print("="*60)
    print(f"  Agents       : {config['num_agents']} intersections")
    print(f"  FL Rounds    : {config['num_rounds']}")
    print(f"  Local steps  : {config['local_steps_per_round']} per round")
    print(f"  Device       : CPU only")
    print(f"  SUMO         : {'enabled' if config['use_sumo'] else 'mock simulator'}")
    print("="*60 + "\n")

    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["save_dir"], "plots"), exist_ok=True)

    agents, envs, clients, aggregator = build_system(config)
    logger = TrainingLogger(log_dir=os.path.join(config["save_dir"], "logs"))

    # Initialise global weights from agent 0's randomly-initialised network
    global_weights = clients[0].get_parameters()

    # Track metrics across rounds
    round_rewards = []
    eval_rewards  = []
    round_losses  = []

    start_time = time.time()

    # ── Communication rounds ─────────────────────────────────────────────
    for rnd in tqdm(range(config["num_rounds"]), desc="FL Rounds", unit="round"):
        round_updates = []
        round_agent_rewards = []
        round_agent_losses  = []

        # ── Local training on each agent ────────────────────────────────
        for i, (client, agent) in enumerate(zip(clients, agents)):
            updated_weights, num_samples, metrics = client.fit(global_weights, {})

            update = AgentUpdate(
                agent_id=agent.agent_id,
                weights=updated_weights,
                num_samples=num_samples,
                metrics=metrics,
            )
            round_updates.append(update)
            round_agent_rewards.append(metrics["avg_reward"])
            if metrics["avg_loss"] is not None:
                round_agent_losses.append(metrics["avg_loss"])

        # ── FedAvg aggregation ────────────────────────────────────────
        new_weights = aggregator.aggregate(round_updates)
        if new_weights is not None:
            global_weights = new_weights

        # ── Metrics ───────────────────────────────────────────────────
        avg_reward = float(np.mean(round_agent_rewards))
        avg_loss   = float(np.mean(round_agent_losses)) if round_agent_losses else 0.0
        round_rewards.append(avg_reward)
        round_losses.append(avg_loss)

        logger.log_round(
            round_num=rnd,
            avg_reward=avg_reward,
            avg_loss=avg_loss,
            epsilons={a.agent_id: a.epsilon for a in agents},
        )

        # ── Periodic evaluation ────────────────────────────────────────
        if rnd % config["eval_every"] == 0:
            eval_r = evaluate_agents(clients, global_weights)
            eval_rewards.append((rnd, eval_r))
            tqdm.write(
                f"Round {rnd:3d} | train_reward={avg_reward:.3f} | "
                f"eval_reward={eval_r:.3f} | loss={avg_loss:.4f} | "
                f"ε={agents[0].epsilon:.3f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────
        if rnd % config["checkpoint_every"] == 0:
            aggregator.save_global_model(round_num=rnd)
            for agent in agents:
                ckpt_path = os.path.join(
                    config["save_dir"], "checkpoints",
                    f"agent_{agent.agent_id}_round_{rnd:04d}.pth"
                )
                agent.save(ckpt_path)

    # ── Training complete ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Save final model and history
    aggregator.save_global_model(round_num=config["num_rounds"])
    aggregator.save_round_history()

    # ── Generate plots ────────────────────────────────────────────────────
    _plot_training_curves(
        round_rewards, round_losses, eval_rewards,
        save_dir=os.path.join(config["save_dir"], "plots"),
    )

    # ── Final summary ─────────────────────────────────────────────────────
    best_round = aggregator.get_best_round()
    print("\n" + "="*60)
    print("  TRAINING SUMMARY")
    print("="*60)
    print(f"  Total rounds      : {config['num_rounds']}")
    if best_round:
        print(f"  Best round        : {best_round.round_num}")
        print(f"  Best avg reward   : {best_round.avg_reward:.4f}")
    else:
        print("  Best round        : N/A")
        print("  Best avg reward   : N/A")
    print(f"  Final epsilon     : {agents[0].epsilon:.4f}")
    print(f"  Results saved to  : {config['save_dir']}/")
    print("="*60)

    return aggregator


# ─── Plotting ────────────────────────────────────────────────────────────────
def _plot_training_curves(round_rewards, round_losses, eval_rewards, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Federated Traffic Signal Control — Training Results", fontsize=13, fontweight="bold")

    def smooth(data, window=8):
        if len(data) < window:
            return np.array(data)
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode="valid")

    rounds = list(range(len(round_rewards)))

    ax = axes[0, 0]
    ax.plot(rounds, round_rewards, alpha=0.25, color="#1D9E75", linewidth=0.8, label="per round")
    sm = smooth(round_rewards)
    ax.plot(range(len(sm)), sm, color="#1D9E75", linewidth=2.5, label="smoothed")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("FL Round"); ax.set_ylabel("Avg reward")
    ax.set_title("Training reward — rises toward 0 as agent improves")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(rounds, round_losses, alpha=0.25, color="#534AB7", linewidth=0.8)
    sl = smooth(round_losses)
    ax.plot(range(len(sl)), sl, color="#534AB7", linewidth=2.5)
    ax.set_xlabel("FL Round"); ax.set_ylabel("Huber loss")
    ax.set_title("TD loss — should stabilise after initial rise")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    if eval_rewards:
        e_rounds, e_vals = zip(*eval_rewards)
        ax.plot(e_rounds, e_vals, "o-", color="#D85A30", linewidth=2, markersize=5)
        trend = "improving" if len(e_vals) > 1 and e_vals[-1] > e_vals[0] else "still warming up"
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("FL Round"); ax.set_ylabel("Eval reward")
        ax.set_title(f"Eval reward (greedy policy) — {trend}")
        ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    eps_decay, eps_start, eps_end = 0.990, 1.0, 0.05
    eps_curve = [max(eps_end, eps_start * (eps_decay ** r)) for r in rounds]
    ax.plot(rounds, eps_curve, color="#BA7517", linewidth=2.5)
    ax.fill_between(rounds, eps_end, eps_curve, alpha=0.12, color="#BA7517")
    ax.axhline(y=eps_end, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label=f"floor ε={eps_end}")
    ax.set_xlabel("FL Round"); ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Exploration decay — less random over time")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {path}")


# ─── Comparison baseline (fixed-time controller) ─────────────────────────────
def run_fixed_time_baseline(config: dict, num_episodes: int = 5):
    """
    Run a fixed-time controller as a comparison baseline.
    Fixed-time: cycles through phases 0→1→2→3 every 10 steps.
    """
    print("\nRunning fixed-time baseline...")
    rewards = []
    for i, aid in enumerate(config["agent_ids"]):
        env = TrafficSignalEnv(
            intersection_id=f"baseline_{aid}",
            seed=config["seed"] + i,
        )
        for _ in range(num_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            phase = 0
            for step in range(500):
                if step % 10 == 0:
                    phase = (phase + 1) % 4
                obs, reward, term, trunc, _ = env.step(phase)
                ep_reward += reward
                if term or trunc:
                    break
            rewards.append(ep_reward)

    mean_r = float(np.mean(rewards))
    print(f"Fixed-time baseline mean reward: {mean_r:.3f}")
    return mean_r


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Traffic Signal Control")
    parser.add_argument("--rounds",       type=int,   default=DEFAULT_CONFIG["num_rounds"])
    parser.add_argument("--agents",       type=int,   default=DEFAULT_CONFIG["num_agents"])
    parser.add_argument("--local-steps",  type=int,   default=DEFAULT_CONFIG["local_steps_per_round"])
    parser.add_argument("--use-sumo",     action="store_true")
    parser.add_argument("--baseline",     action="store_true", help="run fixed-time baseline first")
    parser.add_argument("--seed",         type=int,   default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["num_rounds"]              = args.rounds
    config["num_agents"]              = args.agents
    config["agent_ids"]               = [chr(65 + i) for i in range(args.agents)]  # A, B, C...
    config["local_steps_per_round"]   = args.local_steps
    config["use_sumo"]                = args.use_sumo
    config["seed"]                    = args.seed

    # Save config
    os.makedirs(config["save_dir"], exist_ok=True)
    with open(os.path.join(config["save_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if args.baseline:
        baseline_reward = run_fixed_time_baseline(config)

    train(config)
