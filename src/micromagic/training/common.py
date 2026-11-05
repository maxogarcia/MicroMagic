"""Pieces shared by the PPO and DQN trainers: matchups, opponents, self-play pool, evaluation.

GameState gives seat 0 the aggro deck and the first turn by default, and
midrange beats aggro 82% of the time under random play. So training rotates
the learner through all four (deck, seat) matchups, and evaluation reports
every result per deck and seat.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from micromagic.agents.base import Agent
from micromagic.agents.heuristic import HeuristicAgent
from micromagic.agents.random_agent import RandomAgent
from micromagic.env.gym_env import MicroMagicGymEnv
from micromagic.env.spec import DECK_NAMES
from micromagic.env.state import GameState
from micromagic.evaluation.wilson import compute_agent_win_rates, wilson_score_interval
from micromagic.tournaments.runner import TournamentResult, TournamentRunner


# Seat 0 always moves first, so (deck pair, seat) covers deck x play/draw.
DECK_PAIRS: tuple[tuple[str, str], ...] = (DECK_NAMES, DECK_NAMES[::-1])


def make_opponent(kind: str, seed: int) -> Agent:
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent()
    raise ValueError(f"Unknown opponent {kind!r}; supported: 'random', 'heuristic'")


class PolicyAgent(Agent):
    """Acts with an SB3 policy whose predict() takes action_masks (MaskablePPO's
    policy and MaskableDQNPolicy both do). Used for frozen self-play snapshots."""

    def __init__(self, policy: Any, deterministic: bool = False, name: str = "snapshot") -> None:
        super().__init__(name)
        self.policy = policy
        self.deterministic = deterministic

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        action, _ = self.policy.predict(
            obs, action_masks=np.asarray(mask, dtype=bool), deterministic=self.deterministic
        )
        return int(action)


@dataclass
class OpponentPool:
    """Samples one opponent per game.

    weights maps "random", "heuristic" and "self" to sampling weights. "self"
    draws uniformly from the frozen snapshots; until the first snapshot exists
    its weight is shared out among the other kinds.
    """

    weights: dict[str, float]
    seed: int = 0
    max_snapshots: int = 10
    snapshots: list[Agent] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = set(self.weights) - {"random", "heuristic", "self"}
        if unknown:
            raise ValueError(f"Unknown opponent kinds {sorted(unknown)}")
        self.rng = np.random.default_rng(self.seed)
        self.random = RandomAgent(seed=self.seed)

    def add_snapshot(self, agent: Agent) -> None:
        self.snapshots.append(agent)
        # Keep the newest snapshots; the oldest are far weaker than the learner.
        del self.snapshots[: -self.max_snapshots]

    def sample(self) -> Agent:
        kinds = [k for k, w in self.weights.items() if w > 0 and (k != "self" or self.snapshots)]
        if not kinds:
            raise ValueError("OpponentPool has no opponent with positive weight")
        p = np.array([self.weights[k] for k in kinds], dtype=float)
        kind = kinds[self.rng.choice(len(kinds), p=p / p.sum())]
        if kind == "self":
            return self.snapshots[self.rng.integers(len(self.snapshots))]
        # A fresh heuristic per game: its block memory must not leak between envs.
        return self.random if kind == "random" else HeuristicAgent()


class MatchupWrapper(gym.Wrapper):
    """Before every game, picks a random deck pair, learner seat and opponent."""

    def __init__(self, env: MicroMagicGymEnv, pool: OpponentPool, seed: int) -> None:
        super().__init__(env)
        self.pool = pool
        self.matchup_rng = np.random.default_rng(seed)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        base: MicroMagicGymEnv = self.env.unwrapped
        base.decks = DECK_PAIRS[self.matchup_rng.integers(len(DECK_PAIRS))]
        base.agent_player = int(self.matchup_rng.integers(2))
        base.opponent_policy = self.pool.sample()
        return self.env.reset(**kwargs)


def make_vec_env(n_envs: int, pool: OpponentPool, seed: int) -> DummyVecEnv:
    """All envs share one pool, so a snapshot added by the callback reaches every env."""

    def factory(index: int):
        def _init() -> Monitor:
            env = MatchupWrapper(MicroMagicGymEnv(), pool, seed=seed + 1000 + index)
            return Monitor(env)

        return _init

    venv = DummyVecEnv([factory(i) for i in range(n_envs)])
    venv.seed(seed)
    return venv


class SnapshotCallback(BaseCallback):
    """Every `every` env steps, freezes a copy of the current policy into the pool."""

    def __init__(self, pool: OpponentPool, every: int) -> None:
        super().__init__()
        self.pool = pool
        self.every = every
        self._next = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            policy = copy.deepcopy(self.model.policy)
            policy.set_training_mode(False)
            self.pool.add_snapshot(PolicyAgent(policy, name=f"snapshot_{self.num_timesteps}"))
        return True


# Evaluation


def matchup_env_factory(base_seed: int):
    """Env factory for TournamentRunner that, together with its seat swap on odd
    games, cycles every 4 games through all four (deck, seat) matchups."""

    def factory(seed: int) -> GameState:
        game_id = seed - base_seed
        return GameState(seed=seed, decks=DECK_PAIRS[(game_id // 2) % 2])

    return factory


def play_matchups(agent: Agent, opponent: Agent, games: int, base_seed: int) -> TournamentResult:
    runner = TournamentRunner(matchup_env_factory(base_seed))
    return runner.run_match(agent, opponent, games, base_seed=base_seed)


def by_deck_and_seat(result: TournamentResult, agent_name: str, base_seed: int) -> pd.DataFrame:
    """Win rate of `agent_name` per (deck, seat), plus an "all" row, with 95% Wilson CIs.
    Draws count as half a win, as in compute_agent_win_rates."""
    df = result.to_dataframe()
    seat = np.where(df["player_0_name"] == agent_name, 0, 1)
    pair = ((df["seed"] - base_seed) // 2) % 2
    df = df.assign(
        seat=seat,
        deck=[DECK_PAIRS[p][s] for p, s in zip(pair, seat)],
        win=(df["winner_name"] == agent_name).astype(int),
        draw=df["is_draw"].astype(int),
    )

    def row(group: pd.DataFrame, deck: str, seat_label: str) -> dict[str, Any]:
        n, wins, draws = len(group), int(group["win"].sum()), int(group["draw"].sum())
        lo, hi = wilson_score_interval(wins, n, draws=draws)
        return {
            "deck": deck,
            "seat": seat_label,
            "games": n,
            "wins": wins,
            "draws": draws,
            "win_rate": (wins + 0.5 * draws) / n if n else 0.0,
            "ci_lower": lo,
            "ci_upper": hi,
        }

    rows = [
        row(g, deck, "first" if s == 0 else "second")
        for (deck, s), g in df.groupby(["deck", "seat"], sort=True)
    ]
    rows += [row(g, deck, "both") for deck, g in df.groupby("deck", sort=True)]
    rows.append(row(df, "all", "both"))
    return pd.DataFrame(rows)


def evaluate(agent: Agent, config: dict[str, Any], out: Path) -> pd.DataFrame:
    """Held-out games against each eval opponent; writes per-game CSVs and the summaries."""
    eval_cfg = config["eval"]
    seed, games = int(eval_cfg["seed"]), int(eval_cfg["games"])
    opponents = eval_cfg.get("opponents") or [eval_cfg.get("opponent", "random")]
    tables, summaries = [], []
    for kind in opponents:
        result = play_matchups(agent, make_opponent(kind, seed), games, seed)
        result.to_dataframe().to_csv(out / f"eval_matches_{kind}.csv", index=False)
        summaries.append(compute_agent_win_rates(result).assign(opponent=kind))
        tables.append(by_deck_and_seat(result, agent.name, seed).assign(opponent=kind))
    pd.concat(summaries).to_csv(out / "eval_summary.csv", index=False)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "eval_by_deck_seat.csv", index=False)
    return table


class LearningCurveCallback(BaseCallback):
    """Every `every` env steps, plays `games` greedy games against each opponent
    and appends the win rates to learning_curve.csv."""

    def __init__(self, make_agent, opponents: list[str], every: int, games: int, seed: int, path: Path) -> None:
        super().__init__()
        self.make_agent = make_agent
        self.opponents = opponents
        self.every = every
        self.games = games
        self.seed = seed
        self.path = path
        self._next = every
        self.rows: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.record()
        return True

    def _on_training_end(self) -> None:
        if not self.rows or self.rows[-1]["timesteps"] != self.num_timesteps:
            self.record()

    def record(self) -> None:
        agent = self.make_agent(self.model)
        for kind in self.opponents:
            result = play_matchups(agent, make_opponent(kind, self.seed), self.games, self.seed)
            overall = by_deck_and_seat(result, agent.name, self.seed).iloc[-1]
            self.rows.append(
                {"timesteps": self.num_timesteps, "opponent": kind}
                | overall[["win_rate", "ci_lower", "ci_upper"]].to_dict()
            )
            self.logger.record(f"eval/win_rate_vs_{kind}", overall["win_rate"])
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


def run_dir(config: dict[str, Any]) -> Path:
    out = Path(config["out_dir"]) / config["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def opponent_weights(config: dict[str, Any]) -> dict[str, float]:
    """`opponents: {random: 0.1, heuristic: 0.2, self: 0.7}`, or the older `opponent: random`."""
    if "opponents" in config:
        return {k: float(v) for k, v in config["opponents"].items()}
    return {config.get("opponent", "random"): 1.0}


def callbacks(config: dict[str, Any], pool: OpponentPool, make_agent, out: Path) -> list[BaseCallback]:
    found: list[BaseCallback] = []
    if pool.weights.get("self", 0) > 0:
        found.append(SnapshotCallback(pool, int(config.get("snapshot_every", 50_000))))
    curve = config.get("curve")
    if curve:
        found.append(
            LearningCurveCallback(
                make_agent,
                list(curve.get("opponents", ["random", "heuristic"])),
                every=int(curve["every"]),
                games=int(curve["games"]),
                seed=int(curve.get("seed", 200_000)),
                path=out / "learning_curve.csv",
            )
        )
    return found


def load_config(argv: list[str] | None, description: str) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--total-timesteps", type=int, help="override config total_timesteps")
    parser.add_argument("--seed", type=int, help="override config seed")
    parser.add_argument("--run-name", help="override config run_name")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.run_name is not None:
        config["run_name"] = args.run_name
    config.setdefault("run_name", f"{args.config.stem}_seed{config['seed']}")
    return config


def save_config(config: dict[str, Any], out: Path) -> None:
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
