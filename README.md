# Decentralized Traffic Signal Control via Federated MARL
## Complete Setup & Usage Guide — Windows 11 / CPU-only

---

## Project Structure

```
federated_traffic/
├── setup.bat                   ← Run this FIRST to install everything
├── test_setup.py               ← Run SECOND to verify installation
├── train_federated.py          ← Main training script
│
├── envs/
│   └── traffic_env.py          ← Gymnasium environment (mock + SUMO)
│
├── agents/
│   └── dqn_agent.py            ← DQN agent with replay buffer
│
├── federated/
│   ├── aggregator.py           ← FedAvg aggregator
│   └── flower_client.py        ← Flower client wrapper
│
├── utils/
│   ├── logger.py               ← Training logger
│   └── metrics.py              ← Metric helpers
│
├── configs/
│   └── config.yaml             ← Hyperparameter config
│
└── results/                    ← Created automatically during training
    ├── checkpoints/            ← Saved model weights
    ├── plots/                  ← Training curves (PNG)
    └── logs/                   ← Per-round metrics (JSONL)
```

---

## Step 1 — Install Python 3.10

1. Go to: https://www.python.org/downloads/release/python-31011/
2. Download **Windows installer (64-bit)**
3. Run installer — **check "Add Python to PATH"** before clicking Install
4. Open Command Prompt and verify:
   ```
   python --version
   ```
   Should print: `Python 3.10.x`

> Why 3.10? It has the best compatibility with SUMO TraCI bindings
> and the PyTorch/Flower versions used in this project.

---

## Step 2 — Run the Setup Script

Open Command Prompt (`Win + R` → type `cmd` → Enter):

```batch
cd path\to\federated_traffic
setup.bat
```

This will:
- Verify Python 3.10+
- Create a virtual environment (`venv\`)
- Install PyTorch (CPU), Flower, Gymnasium, NumPy, Matplotlib, TensorBoard
- Install SUMO Python bindings

**Expected time:** 3–7 minutes depending on internet speed.

---

## Step 3 — Install SUMO (Optional but Recommended)

For real traffic simulation (not required for the mock simulator):

1. Download SUMO 1.19: https://sumo.dlr.de/docs/Downloads.php
2. Run `sumo-win64-1.19.0.msi`
3. The installer sets `SUMO_HOME` environment variable automatically
4. Restart your Command Prompt after installation
5. Verify: `sumo --version`

> If you skip SUMO, the project uses the built-in mock simulator.
> The mock simulator is sufficient for developing and testing the
> federated learning logic. SUMO gives more realistic traffic patterns.

---

## Step 4 — Verify Installation

```batch
venv\Scripts\activate
python test_setup.py
```

Expected output:
```
ALL TESTS PASSED — ready to train!
```

---

## Step 5 — Run Training

### Quick run (fast, for testing)
```batch
venv\Scripts\activate
python train_federated.py --rounds 20 --local-steps 100
```

### Full run (recommended for your project)
```batch
python train_federated.py --rounds 50 --agents 3 --local-steps 200
```

### With fixed-time baseline comparison
```batch
python train_federated.py --rounds 50 --baseline
```

### With SUMO (if installed)
```batch
python train_federated.py --rounds 50 --use-sumo
```

---

## Step 6 — View Training Curves

### Option A: PNG plots (simplest)
After training, open: `results\plots\training_curves.png`

### Option B: TensorBoard (interactive)
```batch
venv\Scripts\activate
tensorboard --logdir results\logs
```
Then open: http://localhost:6006 in your browser

---

## Key Hyperparameters (edit in train_federated.py)

| Parameter | Default | Description |
|---|---|---|
| `num_rounds` | 50 | Total FL communication rounds |
| `local_steps_per_round` | 200 | Local training steps per round |
| `num_agents` | 3 | Number of intersections |
| `lr` | 0.001 | DQN learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_decay` | 0.995 | Exploration decay per round |
| `batch_size` | 64 | Replay buffer mini-batch size |
| `hidden_size` | 64 | Network hidden layer size |

---

## Understanding the Output

```
Round  0 | train_reward=-3.241 | eval_reward=-3.180 | loss=0.0421 | ε=0.990
Round  5 | train_reward=-2.891 | eval_reward=-2.750 | loss=0.0318 | ε=0.951
Round 10 | train_reward=-2.412 | eval_reward=-2.380 | loss=0.0247 | ε=0.905
```

- `train_reward`: mean reward during exploration (negative = congestion)
- `eval_reward`: mean reward with greedy policy (should improve over time)
- `loss`: TD error — expect it to decrease and stabilise
- `ε`: epsilon (exploration rate) — starts at 1.0, decays toward 0.05

A good run shows eval_reward improving (becoming less negative) over rounds.

---

## Troubleshooting

**`python` not found:**
Re-install Python 3.10 and make sure "Add to PATH" is checked.

**`ModuleNotFoundError: No module named 'torch'`:**
Run `venv\Scripts\activate` first, then try again.

**SUMO not found / TraCI error:**
The mock simulator will be used automatically. SUMO is optional.

**Very slow training:**
Normal on CPU. 50 rounds with 3 agents takes ~5–15 minutes.
Reduce `--local-steps 50` for faster iteration.

**Out of memory:**
Reduce `buffer_capacity` to 5000 in `train_federated.py`.

---

## Extending the Project

1. **More agents:** `--agents 5` (add more intersections)
2. **Real SUMO maps:** edit `sumo_config` in `train_federated.py`
3. **Distributed deployment:** use `make_flower_client()` in `flower_client.py`
4. **Different algorithms:** swap `DQNAgent` for PPO or A2C
5. **FedProx:** replace `FedAvgAggregator` with FedProx variant for non-IID data
