"""
Traffic Signal Control Environment
===================================
A Gymnasium-compatible environment for a single traffic intersection.
Works in two modes:
  - SUMO mode: uses real SUMO simulator (requires SUMO installation)
  - Mock mode:  built-in simulator for testing without SUMO

State space  (8 values per intersection):
  [queue_N, queue_S, queue_E, queue_W,
   wait_N,  wait_S,  wait_E,  wait_W]

Action space (4 discrete phases):
  0: North-South green
  1: North-South + left turns green
  2: East-West green
  3: East-West + left turns green

Reward: negative sum of all waiting vehicles (minimise congestion)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# ─── SUMO availability check ────────────────────────────────────────────────
def _sumo_available() -> bool:
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        return False
    tools = os.path.join(sumo_home, "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    try:
        import traci
        return True
    except ImportError:
        return False


SUMO_AVAILABLE = _sumo_available()


# ─── Mock Traffic Simulator (no SUMO needed) ────────────────────────────────
class MockTrafficSimulator:
    """
    Simple stochastic traffic simulator.
    Approximates intersection dynamics without SUMO.
    Good enough for development and CPU-only training.
    """

    def __init__(self, intersection_id: str, seed: Optional[int] = None):
        self.intersection_id = intersection_id
        self.rng = np.random.default_rng(seed)

        # Arrival rates (vehicles per step) — realistic busy intersection
        # Different intersections get different rates to simulate heterogeneity
        base_rate = 2.0 + self.rng.uniform(-0.5, 0.5)
        self.arrival_rates = {
            "N": base_rate * self.rng.uniform(0.6, 1.4),
            "S": base_rate * self.rng.uniform(0.6, 1.4),
            "E": base_rate * self.rng.uniform(0.6, 1.4),
            "W": base_rate * self.rng.uniform(0.6, 1.4),
        }

        # Phase definitions: (green directions, departure capacity per step)
        self.phases = {
            0: (["N", "S"], 3.0),   # N-S through — high capacity
            1: (["N", "S"], 1.5),   # N-S + left turns — slower
            2: (["E", "W"], 3.0),   # E-W through — high capacity
            3: (["E", "W"], 1.5),   # E-W + left turns — slower
        }

        # Episode length: longer so queues build up and agent has time to learn
        self.max_steps = 200

        self.reset()

    def reset(self) -> np.ndarray:
        self.queues = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
        self.waiting = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
        self.current_phase = 0
        self.phase_duration = 0
        self.step_count = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_phase = action
        self.phase_duration += 1
        self.step_count += 1

        green_dirs, depart_capacity = self.phases[action]

        # Arrivals: Poisson process — vehicles arrive each step
        for direction in ["N", "S", "E", "W"]:
            arrivals = self.rng.poisson(self.arrival_rates[direction])
            self.queues[direction] = min(self.queues[direction] + arrivals, 50)

        # Departures: green directions discharge vehicles up to capacity
        for direction in green_dirs:
            max_depart = self.rng.poisson(depart_capacity)
            departures = min(self.queues[direction], max_depart)
            self.queues[direction] = max(0.0, self.queues[direction] - departures)

        # Waiting time: accumulates for red directions, resets on green
        for direction in ["N", "S", "E", "W"]:
            if direction not in green_dirs:
                self.waiting[direction] += self.queues[direction] * 0.1
            else:
                self.waiting[direction] = max(0.0, self.waiting[direction] - 2.0)

        # Reward: negative total queue — normalised to roughly [-1, 0] range
        total_queue = sum(self.queues.values())
        reward = -(total_queue / 200.0)   # 200 = max possible (50×4)

        # Episode ends after max_steps
        done = self.step_count >= self.max_steps

        info = {
            "queues": dict(self.queues),
            "waiting": dict(self.waiting),
            "total_vehicles": total_queue,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        state = np.array([
            self.queues["N"] / 50.0,
            self.queues["S"] / 50.0,
            self.queues["E"] / 50.0,
            self.queues["W"] / 50.0,
            min(self.waiting["N"] / 100.0, 1.0),
            min(self.waiting["S"] / 100.0, 1.0),
            min(self.waiting["E"] / 100.0, 1.0),
            min(self.waiting["W"] / 100.0, 1.0),
        ], dtype=np.float32)
        return state


# ─── SUMO-backed simulator ───────────────────────────────────────────────────
class SUMOTrafficSimulator:
    """
    Wraps SUMO via TraCI for real traffic simulation.
    Requires SUMO to be installed and SUMO_HOME set.
    """

    def __init__(self, intersection_id: str, net_file: str, route_file: str,
                 sumo_binary: str = "sumo", seed: Optional[int] = None):
        import traci
        self.traci = traci
        self.intersection_id = intersection_id
        self.net_file = net_file
        self.route_file = route_file
        self.sumo_binary = sumo_binary
        self.seed = seed or 42
        self.step_count = 0
        self._started = False

    def reset(self) -> np.ndarray:
        if self._started:
            self.traci.close()
        sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log",
            "--no-warnings",
            "--seed", str(self.seed),
        ]
        self.traci.start(sumo_cmd)
        self._started = True
        self.step_count = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Set traffic light phase
        self.traci.trafficlight.setPhase(self.intersection_id, action)
        self.traci.simulationStep()
        self.step_count += 1

        state = self._get_state()
        reward = self._compute_reward()
        done = self.step_count >= 500

        info = {"step": self.step_count}
        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        lanes = self.traci.trafficlight.getControlledLanes(self.intersection_id)
        queues = []
        waits = []
        for lane in lanes[:4]:  # use first 4 lanes (N,S,E,W)
            queues.append(
                min(self.traci.lane.getLastStepHaltingNumber(lane) / 50.0, 1.0)
            )
            waits.append(
                min(self.traci.lane.getWaitingTime(lane) / 100.0, 1.0)
            )
        # Pad if fewer than 4 lanes
        while len(queues) < 4:
            queues.append(0.0)
            waits.append(0.0)
        return np.array(queues + waits, dtype=np.float32)

    def _compute_reward(self) -> float:
        lanes = self.traci.trafficlight.getControlledLanes(self.intersection_id)
        total_halt = sum(
            self.traci.lane.getLastStepHaltingNumber(l) for l in lanes
        )
        return -total_halt / 50.0


# ─── Main Gymnasium Environment ─────────────────────────────────────────────
class TrafficSignalEnv(gym.Env):
    """
    Gymnasium environment wrapping either MockTrafficSimulator or
    SUMOTrafficSimulator depending on what is available.

    Usage:
        env = TrafficSignalEnv(intersection_id="A")
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        intersection_id: str = "intersection_0",
        use_sumo: bool = False,
        sumo_config: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.intersection_id = intersection_id
        self.use_sumo = use_sumo and SUMO_AVAILABLE

        # 8-dimensional observation: [queue×4, wait×4] all in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # 4 traffic phases
        self.action_space = spaces.Discrete(4)

        # Initialise simulator backend
        if self.use_sumo and sumo_config:
            self.simulator = SUMOTrafficSimulator(
                intersection_id=intersection_id,
                seed=seed,
                **sumo_config,
            )
            print(f"[{intersection_id}] Using SUMO simulator")
        else:
            self.simulator = MockTrafficSimulator(
                intersection_id=intersection_id,
                seed=seed,
            )
            if use_sumo and not SUMO_AVAILABLE:
                print(f"[{intersection_id}] SUMO not found — using mock simulator")
            else:
                print(f"[{intersection_id}] Using mock simulator")

        self._episode_reward = 0.0
        self._episode_steps = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        obs = self.simulator.reset()
        self._episode_reward = 0.0
        self._episode_steps = 0
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, done, info = self.simulator.step(int(action))
        self._episode_reward += reward
        self._episode_steps += 1
        info["episode_reward"] = self._episode_reward
        info["episode_steps"] = self._episode_steps
        return obs, reward, done, False, info  # (obs, reward, terminated, truncated, info)

    def render(self):
        """Print a simple text render of the current state."""
        sim = self.simulator
        if hasattr(sim, "queues"):
            print(
                f"[{self.intersection_id}] "
                f"N:{sim.queues['N']:.0f} S:{sim.queues['S']:.0f} "
                f"E:{sim.queues['E']:.0f} W:{sim.queues['W']:.0f} | "
                f"Phase:{sim.current_phase}"
            )

    def close(self):
        if self.use_sumo and hasattr(self.simulator, "_started"):
            if self.simulator._started:
                self.simulator.traci.close()
