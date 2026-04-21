"""
Quick smoke test — run this FIRST to verify your environment is working.
Takes about 30 seconds on CPU.

    python test_setup.py

Expected output: all tests pass with [OK] markers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    print("Testing imports...")
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"  [FAIL] PyTorch: {e}")
        return False

    try:
        import flwr
        print(f"  [OK] Flower {flwr.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Flower: {e}")
        return False

    try:
        import gymnasium
        print(f"  [OK] Gymnasium {gymnasium.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Gymnasium: {e}")
        return False

    try:
        import numpy, matplotlib, tqdm
        print(f"  [OK] NumPy {numpy.__version__}, Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Scientific stack: {e}")
        return False

    return True


def test_environment():
    print("\nTesting traffic environment...")
    from envs.traffic_env import TrafficSignalEnv
    env = TrafficSignalEnv("test_intersection", seed=0)
    obs, info = env.reset()
    assert obs.shape == (8,), f"Expected shape (8,), got {obs.shape}"
    assert all(0 <= v <= 1 for v in obs), "Observations should be in [0, 1]"

    total_reward = 0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (8,)
        assert reward <= 0, "Reward should be <= 0 (minimising congestion)"
    print(f"  [OK] 20 steps complete | avg_reward={total_reward/20:.3f}")
    return True


def test_agent():
    print("\nTesting DQN agent...")
    import numpy as np
    from agents.dqn_agent import DQNAgent

    agent = DQNAgent("test_agent")

    # Test action selection
    obs = np.random.rand(8).astype(np.float32)
    action = agent.select_action(obs)
    assert 0 <= action <= 3, f"Action {action} out of range"

    # Fill buffer and test learning
    for _ in range(100):
        s  = np.random.rand(8).astype(np.float32)
        ns = np.random.rand(8).astype(np.float32)
        agent.store(s, action, -0.5, ns, False)

    loss = agent.learn()
    assert loss is not None, "Learning returned None with 100 samples"
    assert loss >= 0, f"Loss should be non-negative, got {loss}"
    print(f"  [OK] Action={action}, Loss={loss:.4f}")

    # Test weight export/import for FL
    weights = agent.get_weights()
    assert len(weights) > 0
    agent.set_weights(weights)
    print(f"  [OK] FL weight export/import ({len(weights)} layers)")
    return True


def test_aggregator():
    print("\nTesting FedAvg aggregator...")
    import numpy as np
    from agents.dqn_agent import DQNAgent
    from federated.aggregator import FedAvgAggregator, AgentUpdate

    aggregator = FedAvgAggregator(num_agents=3, min_agents_per_round=2)

    # Create fake updates from 3 agents
    updates = []
    for i in range(3):
        agent = DQNAgent(f"agent_{i}")
        updates.append(AgentUpdate(
            agent_id=f"agent_{i}",
            weights=agent.get_weights(),
            num_samples=100 + i * 50,
            metrics={"avg_reward": -2.0 + i * 0.5, "avg_loss": 0.1, "epsilon": 0.5},
        ))

    global_weights = aggregator.aggregate(updates)
    assert global_weights is not None
    assert len(global_weights) == len(updates[0].weights)
    print(f"  [OK] Aggregated {len(updates)} agents → {len(global_weights)} weight layers")
    return True


def test_full_mini_run():
    print("\nRunning mini federated training (3 rounds)...")
    import numpy as np
    from agents.dqn_agent import DQNAgent
    from envs.traffic_env import TrafficSignalEnv
    from federated.aggregator import FedAvgAggregator, AgentUpdate
    from federated.flower_client import LocalFlowerClient

    clients = []
    agents = []
    for i in range(2):
        env   = TrafficSignalEnv(f"mini_{i}", seed=i)
        agent = DQNAgent(f"mini_{i}")
        client = LocalFlowerClient(agent, env, local_steps=50)
        clients.append(client)
        agents.append(agent)

    aggregator = FedAvgAggregator(num_agents=2, min_agents_per_round=2)
    global_weights = clients[0].get_parameters()

    for rnd in range(3):
        updates = []
        for client, agent in zip(clients, agents):
            w, n, metrics = client.fit(global_weights, {})
            updates.append(AgentUpdate(agent.agent_id, w, n, metrics))
        global_weights = aggregator.aggregate(updates)

    print(f"  [OK] 3 rounds complete | best_reward={aggregator.get_best_round().avg_reward:.3f}")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  FEDERATED TRAFFIC CONTROL — SETUP TEST")
    print("=" * 50)

    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_environment()
    all_passed &= test_agent()
    all_passed &= test_aggregator()
    all_passed &= test_full_mini_run()

    print("\n" + "=" * 50)
    if all_passed:
        print("  ALL TESTS PASSED — ready to train!")
        print("  Run: python train_federated.py")
    else:
        print("  SOME TESTS FAILED — check output above")
        sys.exit(1)
    print("=" * 50)
