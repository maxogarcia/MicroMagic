"""Environment speed benchmark: steps per second on one core, plus clone/determinize cost.

Three loops are timed, each with both seats choosing uniformly random legal actions:
- engine: legal_actions() + step(), the cost of the rules alone
- agent_loop: legal_actions() + observe() + step(), what an RL agent's environment step costs
- gym_wrapper: MicroMagicGymEnv.step() including the random opponent (only if gymnasium is installed)
Decks and the first player alternate between games so the numbers are not tied to one matchup.
"""

import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from micromagic.env.spec import DECK_NAMES
from micromagic.env.state import GameState


def pin_to_one_core() -> int | None:
    """Pin this process to a single CPU where the OS allows it. Returns the CPU id or None."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    return cpu


def _game(index: int, seed: int) -> GameState:
    decks = DECK_NAMES if index % 2 == 0 else DECK_NAMES[::-1]
    return GameState(seed=seed + index, decks=decks, first_player=(index // 2) % 2)


def _time_engine(games: int, seed: int, observe: bool) -> tuple[int, float]:
    rng = np.random.default_rng(seed)
    steps = 0
    start = time.perf_counter()
    for index in range(games):
        state = _game(index, seed)
        while not state.is_terminal():
            legal = np.flatnonzero(state.legal_actions())
            if observe:
                state.observe(state.current_player())
            state.step(int(legal[rng.integers(legal.size)]))
            steps += 1
    return steps, time.perf_counter() - start


def _time_gym(games: int, seed: int) -> tuple[int, float] | None:
    try:
        from micromagic.env.gym_env import MicroMagicGymEnv
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    steps = 0
    start = time.perf_counter()
    for index in range(games):
        env = MicroMagicGymEnv(agent_player=index % 2)
        env.reset(seed=seed + index)
        done = False
        while not done:
            legal = np.flatnonzero(env.action_masks())
            _, _, terminated, truncated, _ = env.step(int(legal[rng.integers(legal.size)]))
            done = terminated or truncated
            steps += 1
    return steps, time.perf_counter() - start


def _mid_game_states(samples: int, seed: int) -> list[GameState]:
    """Snapshots taken at random points of random games, so clone cost reflects real boards."""
    rng = np.random.default_rng(seed)
    states = []
    for index in range(samples):
        state = _game(index, seed)
        stop = int(rng.integers(10, 120))
        for _ in range(stop):
            if state.is_terminal():
                break
            legal = np.flatnonzero(state.legal_actions())
            state.step(int(legal[rng.integers(legal.size)]))
        if not state.is_terminal():
            states.append(state)
    return states


def _microseconds_per_call(states: list[GameState], calls: int, fn: Callable[[GameState], object]) -> float:
    start = time.perf_counter()
    for state in states:
        for _ in range(calls):
            fn(state)
    return (time.perf_counter() - start) / (len(states) * calls) * 1e6


def _summarize(runs: list[tuple[int, float]]) -> dict:
    rates = [steps / seconds for steps, seconds in runs]
    return {
        "steps_per_sec_median": round(statistics.median(rates), 1),
        "steps_per_sec_min": round(min(rates), 1),
        "steps_per_sec_max": round(max(rates), 1),
        "steps_per_repeat": [steps for steps, _ in runs],
    }


def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": commit,
    }


def run_benchmark(config: dict) -> dict:
    seed = int(config["seed"])
    games = int(config["games_per_repeat"])
    repeats = int(config["repeats"])
    pinned = pin_to_one_core() if config.get("pin_to_one_core", True) else None

    # One untimed game warms caches and lazy imports.
    _time_engine(1, seed + 10_000, observe=True)

    loops = {
        "engine": [_time_engine(games, seed, observe=False) for _ in range(repeats)],
        "agent_loop": [_time_engine(games, seed, observe=True) for _ in range(repeats)],
    }
    gym_runs = [_time_gym(games, seed) for _ in range(repeats)]
    if all(run is not None for run in gym_runs):
        loops["gym_wrapper"] = gym_runs

    states = _mid_game_states(int(config["state_samples"]), seed)
    calls = int(config["calls_per_state"])
    rng = np.random.default_rng(seed)
    target = float(config["target_steps_per_sec"])
    results = {name: _summarize(runs) for name, runs in loops.items()}
    headline = results["agent_loop"]["steps_per_sec_median"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "machine": {**machine_info(), "pinned_cpu": pinned},
        "loops": results,
        "clone_us": round(_microseconds_per_call(states, calls, lambda s: s.clone()), 2),
        "determinize_us": round(_microseconds_per_call(states, calls, lambda s: s.determinize(s.current_player(), rng)), 2),
        "headline_steps_per_sec": headline,
        "target_steps_per_sec": target,
        "meets_target": headline >= target,
    }
